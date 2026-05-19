"""
Abstract base class for all worker inference engines.

Subclasses (ONNX-CPU, Hailo-NPU) implement these methods:
    infer_tensors        — pure tensor->tensor
    infer_raw_items      — adapter.preprocess(items) -> infer_tensors
    infer_dummy_inputs   — adapter.generate_dummy_inputs -> infer_tensors
    handle_request       — InferenceRequest dispatch (tensor / raw / dummy)
"""
from __future__ import annotations

import abc
import time
from typing import Any, Optional

import numpy as np

from shared.models import InferenceRequest, RawItem


class InferenceModelEngine(abc.ABC):
    """Pluggable inference backend, owned by the worker."""

    adapter: Optional[Any] = None

    @abc.abstractmethod
    def __init__(self, model_path: str, adapter_path: Optional[str] = None) -> None:
        ...

    @abc.abstractmethod
    def infer_tensors(self, input_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        ...

    # ------------------------------------------------------------------------
    # Optional hook — subclasses surface model input metadata so adapters
    # can resize / dequantize correctly without hardcoding a tile size that
    # may not match how the operator compiled their HEF / exported their
    # ONNX. Default returns an empty dict; OnnxEngine and HailoEngine
    # override to fill in shape + name + dtype.
    # ------------------------------------------------------------------------
    def get_engine_input_info(self) -> dict[str, Any]:
        return {}

    @abc.abstractmethod
    def handle_request(self, req: InferenceRequest) -> Any:
        """
        Run an InferenceRequest end-to-end.

        Returns either:
            * a dict[str, np.ndarray] of raw outputs (when no adapter or
              run_postprocess=False), or
            * the value produced by `adapter.postprocess(...)` if
              run_postprocess=True and an adapter is registered.
        """

    # ---- shared concrete helpers ---------------------------------------------
    def infer_raw_items(self, items: list[RawItem],
                        meta: Optional[dict[str, Any]] = None) -> dict[str, np.ndarray]:
        if self.adapter is None:
            raise ValueError("Raw item mode requires a ModelAdapter")
        feed = self.adapter.preprocess(
            items if isinstance(items, list) else [items], meta=meta
        )
        return self.infer_tensors(feed)

    def infer_dummy_inputs(self, batch_size: int = 1, seed: int = 42) -> dict[str, np.ndarray]:
        if self.adapter is None:
            raise ValueError("Dummy mode requires a ModelAdapter")
        # Surface engine input metadata so the adapter can size the dummy
        # tensor to the engine's actual input shape (see comment in
        # handle_request_with_timing for why this matters).
        meta = self.get_engine_input_info()
        try:
            feed = self.adapter.generate_dummy_inputs(
                batch_size=batch_size, seed=seed, meta=meta,
            )
        except TypeError as e:
            if "meta" not in str(e):
                raise
            feed = self.adapter.generate_dummy_inputs(
                batch_size=batch_size, seed=seed,
            )
        return self.infer_tensors(feed)

    def handle_request_with_timing(self, req: InferenceRequest):
        """
        Wrapper around handle_request that returns (outputs, result, timing).
        The data API uses this so it can attach inference_s to its response.

        Returns:
            outputs (dict[str, np.ndarray] | None) — raw outputs if available
            result  (Any | None)                   — postprocess result if any
            timing  (dict)                         — {"inference_s", ...}
        """
        from shared.models import InferenceMode

        # Merge engine-side input metadata (shape, name, dtype/format) into
        # meta so the adapter can resize to the model's expected tile size
        # without hardcoding a constant. Adapters may still ignore this and
        # use a hardcoded INPUT_SIZE if they know the model never changes.
        meta = {**self.get_engine_input_info(), **(req.meta or {})}
        timing: dict[str, float] = {}

        # ---- Resolve feed
        t0 = time.monotonic()
        if req.mode == InferenceMode.TENSOR.value or req.mode == InferenceMode.TENSOR:
            if not req.inputs:
                raise ValueError("tensor mode requires inputs payload")
            from shared.models import payloads_to_tensorfeed
            feed = payloads_to_tensorfeed(req.inputs)
        elif req.mode == InferenceMode.RAW.value or req.mode == InferenceMode.RAW:
            if self.adapter is None:
                raise ValueError(
                    "raw mode requires an adapter, but the engine was "
                    "loaded without one. The most common cause is that "
                    "the controller's `adapters/` directory is missing "
                    "the adapter file referenced when the model was "
                    "distributed — re-upload the adapter (or copy it "
                    "from demo/adapters/ to the runtime adapters/ dir) "
                    "and re-Distribute the model."
                )
            if not req.items:
                raise ValueError("raw mode requires items payload")
            feed = self.adapter.preprocess(req.items, meta=meta)
            meta = {**meta, "items": req.items}
        elif req.mode == InferenceMode.DUMMY.value or req.mode == InferenceMode.DUMMY:
            if self.adapter is None:
                # Allow engines to provide a default dummy feed via the model
                # signature (see OnnxEngine.dummy_feed_from_signature below).
                if hasattr(self, "dummy_feed_from_signature"):
                    feed = self.dummy_feed_from_signature(
                        batch_size=req.dummy_batch_size or 1,
                        seed=req.dummy_seed or 42,
                    )
                else:
                    raise ValueError("dummy mode requires adapter")
            else:
                # Pass meta to the adapter so it can size + layout the
                # dummy tensor to match the engine's actual input shape
                # (avoids the "256² NCHW vs 512² NHWC, 4× buffer
                # mismatch" trap that bites Hailo when the adapter falls
                # back to its INPUT_SIZE constant). Older adapters whose
                # signature is `(batch_size, seed)` raise TypeError on the
                # meta kwarg — catch it and retry without.
                try:
                    feed = self.adapter.generate_dummy_inputs(
                        batch_size=req.dummy_batch_size or 1,
                        seed=req.dummy_seed or 42,
                        meta=meta,
                    )
                except TypeError as e:
                    if "meta" not in str(e):
                        raise
                    feed = self.adapter.generate_dummy_inputs(
                        batch_size=req.dummy_batch_size or 1,
                        seed=req.dummy_seed or 42,
                    )
        else:
            raise ValueError(f"Unsupported mode: {req.mode}")
        timing["preprocess_s"] = time.monotonic() - t0

        # ---- Run model
        t1 = time.monotonic()
        outputs = self.infer_tensors(feed)
        timing["inference_s"] = time.monotonic() - t1

        # ---- Optional postprocess
        result = None
        if req.run_postprocess and self.adapter is not None:
            t2 = time.monotonic()
            result = self.adapter.postprocess(outputs, meta=meta)
            timing["postprocess_s"] = time.monotonic() - t2

        return outputs, result, timing
