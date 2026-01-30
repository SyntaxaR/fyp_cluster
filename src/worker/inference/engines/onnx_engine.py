"""
Support for ONNX models running on CPU
"""
import logging
from typing import Optional, Any

from common.model import RawItem, InferenceRequest, payloads_to_tensorfeed
from common.util import load_adapter
from worker.inference.inference_engine import InferenceModelEngine
import onnxruntime as ort
import numpy as np
import importlib.util

logger = logging.getLogger(__name__)

class OnnxEngine(InferenceModelEngine):
    """To load and run ONNX models."""

    def __init__(self, model_path, adapter_path: Optional[str] = None):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()

        self.input_names = [i.name for i in self.inputs]
        self.output_names = [o.name for o in self.outputs]

        self._validated_signature: Optional[dict[str, tuple[str, int]]] = None # name -> (dtype_str, ndim)
        self.adapter = load_adapter(adapter_path) if adapter_path else None

        logger.info(f"Loaded ONNX model. inputs={self.input_names} outputs={self.output_names}")
        if self.adapter:
            logger.info("Loaded user ModelAdapter: %s", type(self.adapter).__name__)


    # Core inference
    def infer_tensors(self, input_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self._validate_or_lock_signature(input_data)
        output = self.session.run(self.output_names, input_data)
        return dict(zip(self.output_names, output))

    # Tensor validation
    def _validate_or_lock_signature(self, input_data: dict[str, np.ndarray]) -> None:
        # 1) name check
        for i in self.input_names:
            if i not in input_data:
                print(input_data)
                raise ValueError(f"Missing inputs: {i}. Required={self.input_names}")

        # 2) lock signature on first successful call (cheap checks later)
        if self._validated_signature is None:
            sig = {}
            for i in self.input_names:
                arr = input_data[i]
                sig[i] = (str(arr.dtype), arr.ndim)
            self._validated_signature = sig
            logger.info("Locked inputs signature: %s", self._validated_signature)
            return

        # 3) subsequent calls: quick check dtype/ndim
        for name, (dtype_str, ndim) in self._validated_signature.items():
            arr = input_data[name]
            if str(arr.dtype) != dtype_str or arr.ndim != ndim:
                raise ValueError(
                    f"Input signature mismatch for {name}: expect ({dtype_str}, ndim={ndim}), "
                    f"got ({arr.dtype}, ndim={arr.ndim})"
                )

    # --- Raw Item Mode pipeline
    def infer_raw_items(self, items: list[RawItem], *, meta: Optional[dict[str, Any]] = None):
        if self.adapter is None:
            raise ValueError("Raw Item mode requires a custom ModelAdapter!")
        feed = self.adapter.preprocess(items if isinstance(items, list) else [items], meta=meta)
        outputs = self.infer_tensors(feed)
        return outputs

    # --- Dummy Input Mode pipeline
    def infer_dummy_inputs(self, batch_size: int = 1, seed: int = 42):
        if self.adapter is None:
            raise ValueError("Dummy Input mode requires a custom ModelAdapter!")
        inputs = self.adapter.generate_dummy_inputs(batch_size=batch_size, seed=seed)
        outputs = self.infer_tensors(inputs)
        return outputs


    # --- Entrance
    def handle_request(self, req: InferenceRequest):
        outputs: Optional[dict[str, np.ndarray]] = None
        meta = req.meta or {}
        if req.mode == "tensor":
            if not req.inputs:
                raise ValueError("Tensor mode requires `inputs` payload!")
            feed = payloads_to_tensorfeed(req.inputs)
            outputs = self.infer_tensors(feed)

        elif req.mode == "raw":
            if not req.items:
                raise ValueError("Raw Item mode requires `items` payload!")
            if self.adapter is None:
                raise ValueError("Raw Item mode requires a custom ModelAdapter!")
            meta = {**meta, "items": req.items}
            outputs = self.infer_raw_items(req.items, meta=meta)

        elif req.mode == "dummy":
            if self.adapter is None:
                raise ValueError("Random mode requires a custom ModelAdapter!")
            dummy_batch_size = req.dummy_batch_size or 10
            dummy_seed = req.dummy_seed or 42
            outputs = self.infer_dummy_inputs(batch_size=dummy_batch_size, seed=dummy_seed)

        else:
            raise ValueError(f"Unsupported inference mode: {req.mode}")

        if outputs is None:
            raise RuntimeError("Inference failed!")

        if req.run_postprocess:
            if self.adapter is None:
                return outputs
            return self.adapter.postprocess(outputs, meta=meta)
        return outputs