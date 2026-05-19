"""
ONNX Runtime backed inference engine (runs on the worker's CPU).

This engine is the default fallback when no Hailo accelerator is detected.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import onnxruntime as ort

from shared.models import (
    InferenceMode,
    InferenceRequest,
    payloads_to_tensorfeed,
)
from shared.util import load_adapter
from worker.inference.inference_engine import InferenceModelEngine

logger = logging.getLogger(__name__)


# Map ONNX type strings (as reported by onnxruntime) to numpy dtypes
_ONNX_TYPE_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(bool)": np.bool_,
}


class OnnxEngine(InferenceModelEngine):
    def __init__(self, model_path: str, adapter_path: Optional[str] = None) -> None:
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()
        self.input_names = [i.name for i in self.inputs]
        self.output_names = [o.name for o in self.outputs]

        self._validated_signature: Optional[dict[str, tuple[str, int]]] = None
        self.adapter = load_adapter(adapter_path) if adapter_path else None

        logger.info("OnnxEngine loaded: %s | inputs=%s outputs=%s",
                    model_path, self.input_names, self.output_names)
        if self.adapter is not None:
            logger.info("Adapter: %s", type(self.adapter).__name__)

    def get_engine_input_info(self) -> dict[str, Any]:
        """Surface the model's first input shape / name / dtype so the
        adapter can resize to whatever tile size this ONNX export was
        compiled with (avoids hardcoded INPUT_SIZE drift)."""
        if not self.inputs:
            return {}
        inp = self.inputs[0]
        # ONNX dynamic axes show up as None or string names; replace with
        # 0 so adapter's "find the largest non-1, non-3 dim" heuristic
        # ignores them.
        shape = [int(d) if isinstance(d, int) else 0 for d in inp.shape]
        # Heuristic: ONNX exports for vision models in this codebase are
        # NCHW (channels at index 1). If shape[1] in (1, 3, 4) we surface
        # NCHW so adapters that consume engine_layout don't need to
        # re-derive it. This is a hint, not a contract — adapters should
        # still accept their own meta['layout'] override.
        engine_layout = "NCHW" if (len(shape) == 4 and shape[1] in (1, 3, 4)) else "UNKNOWN"
        return {
            "engine_input_name": inp.name,
            "engine_input_shape": shape,
            "engine_input_dtype": inp.type,
            "engine_layout": engine_layout,
        }

    # =========================================================================
    # Core inference
    # =========================================================================
    def infer_tensors(self, input_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        input_data = self._remap_single_input(input_data)
        self._validate_or_lock_signature(input_data)
        outputs = self.session.run(self.output_names, input_data)
        return dict(zip(self.output_names, outputs))

    def _remap_single_input(
        self, input_data: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """If the adapter produced a single tensor under a generic key
        (e.g. ``"input"``) but the model expects a model-prefixed name
        like ``"real_esrgan_x2/input_layer1"``, transparently remap.

        Common case: the user exports a TF / Keras model to ONNX and the
        export keeps the network-name-prefixed input names. The adapter
        doesn't know that exact string but there's only one input, so
        any sensible mapping is unambiguous. Skip remapping for
        multi-input models — those need to keep their explicit keys.
        """
        if len(input_data) == 1 and len(self.input_names) == 1:
            adapter_key = next(iter(input_data))
            model_key = self.input_names[0]
            if adapter_key != model_key:
                logger.info("Remapping adapter input %r -> model input %r",
                            adapter_key, model_key)
                return {model_key: input_data[adapter_key]}
        return input_data

    def _validate_or_lock_signature(self, input_data: dict[str, np.ndarray]) -> None:
        for i in self.input_names:
            if i not in input_data:
                raise ValueError(
                    f"Missing inputs: {i}. Required={self.input_names}"
                )
        if self._validated_signature is None:
            sig: dict[str, tuple[str, int]] = {}
            for i in self.input_names:
                arr = input_data[i]
                sig[i] = (str(arr.dtype), arr.ndim)
            self._validated_signature = sig
            logger.info("Locked input signature: %s", sig)
            return
        for name, (dtype_str, ndim) in self._validated_signature.items():
            arr = input_data[name]
            if str(arr.dtype) != dtype_str or arr.ndim != ndim:
                raise ValueError(
                    f"Input signature mismatch for {name}: "
                    f"expect ({dtype_str}, ndim={ndim}), "
                    f"got ({arr.dtype}, ndim={arr.ndim})"
                )

    # =========================================================================
    # No-adapter dummy feed (handy for ONNX models without an adapter)
    # =========================================================================
    def dummy_feed_from_signature(self, batch_size: int = 1,
                                  seed: int = 42) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        feed: dict[str, np.ndarray] = {}
        for inp in self.inputs:
            shape: list[int] = [
                batch_size if (s is None or isinstance(s, str) or s < 1) else int(s)
                for s in inp.shape
            ]
            np_dtype = _ONNX_TYPE_TO_NUMPY.get(inp.type, np.float32)
            if np.issubdtype(np_dtype, np.floating):
                feed[inp.name] = rng.standard_normal(shape).astype(np_dtype)
            elif np.issubdtype(np_dtype, np.integer):
                feed[inp.name] = rng.integers(0, 256, size=shape, dtype=np_dtype)
            else:
                feed[inp.name] = np.zeros(shape, dtype=np_dtype)
        return feed

    # =========================================================================
    # Request dispatch (kept for symmetry with HailoEngine)
    # =========================================================================
    def handle_request(self, req: InferenceRequest) -> Any:
        outputs, result, _ = self.handle_request_with_timing(req)
        if req.run_postprocess and self.adapter is not None:
            return result
        return outputs
