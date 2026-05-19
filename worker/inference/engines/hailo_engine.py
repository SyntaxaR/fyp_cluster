"""
Hailo-8/8L NPU inference engine.

This is a *complete* implementation of the engine that was only stubbed out in
the previous codebase. It supports:

    - Tensor mode  (caller-supplied numpy feed)
    - Raw item mode (adapter-driven preprocess)
    - Dummy mode    (adapter-driven random feed, or signature-derived random feed)
    - Sync inference via configured_model.run([bindings])
    - Schedule priority and group sharing (multiple engines on one device)

The hailo_platform Python package is provided by the system-wide `hailo-all`
apt package (installed via dkms), NOT pip. This module imports it lazily so
the rest of the worker can still start when no Hailo card is present.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from shared.models import InferenceRequest, payloads_to_tensorfeed
from shared.util import load_adapter
from worker.inference.inference_engine import InferenceModelEngine

logger = logging.getLogger(__name__)

# Lazy imports — these will fail on workers without hailo-all installed.
try:
    from hailo_platform import (  # type: ignore
        HEF,
        FormatType,
        HailoSchedulingAlgorithm,
        VDevice,
    )
    HAILO_AVAILABLE = True
except Exception as e:  # pragma: no cover - import path varies per platform
    logger.info("hailo_platform unavailable: %s", e)
    HAILO_AVAILABLE = False


class HailoEngine(InferenceModelEngine):
    """Wraps a configured Hailo InferModel for synchronous inference."""

    def __init__(self, model_path: str, adapter_path: Optional[str] = None,
                 batch_size: int = 1, scheduler_priority: int = 0,
                 group_id: str = "SHARED") -> None:
        if not HAILO_AVAILABLE:
            raise ImportError(
                "hailo_platform is not installed. Install hailo-all (apt) "
                "and ensure a Hailo accelerator is present."
            )

        # Configure VDevice with the round-robin scheduler so multiple workers
        # / engines can co-exist on the same physical device.
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = group_id
        self.target = VDevice(params)

        self.hef = HEF(model_path)
        self.infer_model = self.target.create_infer_model(model_path)
        self.infer_model.set_batch_size(batch_size)
        self._batch_size = batch_size

        # Force every input + output stream to FLOAT32 BEFORE we configure
        # the model. Without this the binding silently uses whatever native
        # format the HEF was compiled with (often UINT8 quantized for size
        # / speed) — and our np.empty(shape, dtype=float32) buffers no
        # longer match the binding's expected byte count, giving the
        # cryptic "Output buffer size X different than expected Y" error.
        # FLOAT32 makes the binding handle dequant internally and the
        # buffer dtype unambiguous.
        #
        # Caveat: some HEFs (compiled with explicit input-format settings)
        # don't accept FLOAT32 forcing — set_format_type returns without
        # raising but the binding keeps the native format. We READ BACK
        # the resulting format after the call so `_actual_input_dtype` and
        # `_actual_output_dtype` reflect reality, not what we hoped for.
        # The adapter then sees the truthful dtype via
        # ``get_engine_input_info()`` and picks the matching branch.
        def _force_and_verify(stream, label, default_name):
            name = getattr(stream, "name", default_name)
            try:
                stream.set_format_type(FormatType.FLOAT32)
            except Exception as e:
                logger.warning("FLOAT32 force on %s %s raised: %s",
                               label, name, e)
            # Read back; treat any failure as "we don't know, assume uint8"
            # which is Hailo's most common native format and the safer
            # default (smaller buffer → at worst we feed extra bytes, but
            # we control the adapter's output so we can avoid that).
            try:
                actual = stream.format.type
            except Exception:
                actual = None
            ok = (actual == FormatType.FLOAT32)
            logger.info("  %s %s: forced FLOAT32 -> %s (%s)",
                        label, name, actual,
                        "OK" if ok else "REVERTED to native")
            return actual

        self._input_actual_formats = []
        for inp in self.infer_model.inputs:
            self._input_actual_formats.append(
                _force_and_verify(inp, "input", "?")
            )
        self._output_actual_formats = []
        for out in self.infer_model.outputs:
            self._output_actual_formats.append(
                _force_and_verify(out, "output", "?")
            )

        # Hailo's API is context-managed; we keep the configured model alive
        # for the engine's lifetime by entering the context manually.
        self._config_ctx = self.infer_model.configure()
        self.configured_model = self._config_ctx.__enter__()
        self.configured_model.set_scheduler_priority(scheduler_priority)

        # Cache stream metadata for dummy feed generation and validation
        self._input_streams = self.hef.get_input_vstream_infos()
        self._output_streams = self.hef.get_output_vstream_infos()
        self.input_names = [s.name for s in self._input_streams]
        self.output_names = [s.name for s in self._output_streams]

        self.adapter = load_adapter(adapter_path) if adapter_path else None

        logger.info(
            "HailoEngine loaded: %s | inputs=%s outputs=%s | batch=%d",
            model_path, self.input_names, self.output_names, batch_size,
        )
        # Log the actual per-stream shapes so a future buffer-size mismatch
        # is immediately diagnosable.
        for s in self._input_streams:
            logger.info("  in  %s: shape=%s format=%s",
                        s.name, tuple(s.shape), getattr(s, "format", None))
        for s in self._output_streams:
            logger.info("  out %s: shape=%s format=%s",
                        s.name, tuple(s.shape), getattr(s, "format", None))

    # =========================================================================
    # Engine input metadata — surface real per-stream shape so adapters can
    # resize to whatever tile size this HEF was actually compiled with,
    # instead of hardcoding 256/512/etc. and silently feeding the wrong
    # buffer size. The Hailo vstream shape is (H, W, C) without batch; we
    # prepend a 1 so adapters can treat the result the same way as the
    # ONNX path (4-D, "find the largest non-1, non-3 dim").
    # =========================================================================
    def get_engine_input_info(self) -> dict[str, Any]:
        if not self._input_streams:
            return {}
        s = self._input_streams[0]
        try:
            shape_tuple = tuple(int(d) for d in s.shape)
        except Exception:
            shape_tuple = tuple(s.shape)
        # Normalize to 4-D NHWC by prepending the batch dim. This matches
        # the OnnxEngine convention so adapters don't need a per-engine
        # branch to interpret engine_input_shape.
        if len(shape_tuple) == 3:
            shape = [1, *shape_tuple]
        else:
            shape = list(shape_tuple)
        # Advertise the ACTUAL post-configure format, not the one we
        # wished we'd set. If FLOAT32 force took effect for the first
        # input, adapters that want float32 NHWC are correct; if it
        # reverted to UINT8, adapters MUST send uint8 or they'll trip
        # the "input buffer size N is different than expected M" check.
        actual = (self._input_actual_formats[0]
                  if self._input_actual_formats else None)
        if actual == FormatType.FLOAT32:
            dtype_str = "tensor(float)"
            elem_size = 4
        elif actual == FormatType.UINT8:
            dtype_str = "tensor(uint8)"
            elem_size = 1
        elif actual == FormatType.UINT16:
            dtype_str = "tensor(uint16)"
            elem_size = 2
        else:
            dtype_str = "tensor(float)"  # safest unknown fallback
            elem_size = 4
        return {
            "engine_input_name": s.name,
            "engine_input_shape": shape,
            "engine_input_dtype": dtype_str,
            "engine_input_element_size": elem_size,
            "engine_layout": "NHWC",
        }

    # =========================================================================
    # Lifecycle
    # =========================================================================
    def close(self) -> None:
        try:
            self._config_ctx.__exit__(None, None, None)
        except Exception as e:
            logger.warning("HailoEngine close: %s", e)
        try:
            self.target.release()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # =========================================================================
    # Core inference
    # =========================================================================
    def infer_tensors(self, input_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run a single synchronous inference and return the output feed."""
        # Single-input convenience: if the adapter produced exactly one
        # tensor under a generic key like "input", map it onto whatever
        # name the HEF actually exposes (e.g. "real_esrgan_x2/input_layer1").
        # Multi-input HEFs keep their explicit naming.
        if len(input_data) == 1 and len(self.input_names) == 1:
            adapter_key = next(iter(input_data))
            model_key = self.input_names[0]
            if adapter_key != model_key:
                logger.info("Remapping adapter input %r -> HEF input %r",
                            adapter_key, model_key)
                input_data = {model_key: input_data[adapter_key]}

        # Build bindings for this call
        bindings = self.configured_model.create_bindings()

        # Pre-compute the target numpy dtype per input based on the
        # binding's ACTUAL post-configure format. We convert the
        # adapter's tensor to this dtype regardless of what it gave
        # us — that way an out-of-date adapter (or one that simply
        # doesn't know the engine's preferred dtype) can't trip the
        # "input buffer size X is different than expected Y" check.
        _FMT_TO_NP_IN = {
            FormatType.FLOAT32: np.float32,
            FormatType.UINT8:   np.uint8,
            FormatType.UINT16:  np.uint16,
        }

        for idx, name in enumerate(self.input_names):
            if name not in input_data:
                raise ValueError(
                    f"Missing input '{name}'. Required={self.input_names}"
                )
            arr = input_data[name]

            # Resolve the binding's expected dtype. If we don't know
            # (force verification didn't yield a known FormatType),
            # fall back to whatever the adapter gave us.
            actual_fmt = (self._input_actual_formats[idx]
                          if idx < len(self._input_actual_formats) else None)
            target_dtype = _FMT_TO_NP_IN.get(actual_fmt)

            if target_dtype is not None and arr.dtype != target_dtype:
                logger.info(
                    "Converting input %r: adapter gave dtype=%s "
                    "(nbytes=%d), binding wants %s — engine reshapes "
                    "transparently.",
                    name, arr.dtype, int(arr.nbytes), np.dtype(target_dtype).name,
                )
                # uint8 [0,255] → float32 keeps the [0,255] range, which
                # is exactly what FormatType.FLOAT32 on a uint8-calibrated
                # HEF wants (the binding internally re-quantizes).
                # float32 [0,255] → uint8 needs clipping to avoid wraparound.
                if (target_dtype == np.uint8
                        and np.issubdtype(arr.dtype, np.floating)):
                    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
                else:
                    arr = arr.astype(target_dtype)

            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            bindings.input(name).set_buffer(arr)

        # Allocate output buffers matching the binding's ACTUAL format.
        # Earlier code unconditionally allocated float32, assuming our
        # set_format_type(FLOAT32) call always took. For HEFs that
        # silently reject the force, the binding still wants the native
        # uint8 buffer — and a float32 alloc trips the validate_bindings
        # size check before any inference happens.
        _FMT_TO_NP = {}
        try:
            _FMT_TO_NP = {
                FormatType.FLOAT32: np.float32,
                FormatType.UINT8:   np.uint8,
                FormatType.UINT16:  np.uint16,
            }
        except Exception:
            pass

        output_buffers: dict[str, np.ndarray] = {}
        for idx, out_info in enumerate(self._output_streams):
            shape = tuple(out_info.shape)
            actual = (self._output_actual_formats[idx]
                      if idx < len(self._output_actual_formats) else None)
            dtype = _FMT_TO_NP.get(actual, np.float32)
            buf = np.empty(shape, dtype=dtype)
            output_buffers[out_info.name] = buf
            bindings.output(out_info.name).set_buffer(buf)

        # Run synchronous inference. timeout in milliseconds.
        job = self.configured_model.run_async([bindings])
        try:
            job.wait(timeout_ms=10_000)
        except Exception as e:
            logger.error("Hailo inference job failed: %s", e)
            raise

        return output_buffers

    # =========================================================================
    # Dummy feed for compute-only benchmarking when no adapter is provided
    # =========================================================================
    def dummy_feed_from_signature(self, batch_size: int = 1,
                                  seed: int = 42) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        feed: dict[str, np.ndarray] = {}
        for stream in self._input_streams:
            shape = tuple(stream.shape)
            # Most vision models use uint8 in HWC layout for Hailo input streams.
            # We probe the format type and fall back to float32.
            try:
                fmt_type = stream.format.type  # type: ignore[attr-defined]
            except Exception:
                fmt_type = None
            if fmt_type == FormatType.UINT8:
                arr = rng.integers(0, 256, size=shape, dtype=np.uint8)
            elif fmt_type == FormatType.UINT16:
                arr = rng.integers(0, 65536, size=shape, dtype=np.uint16)
            else:
                arr = rng.standard_normal(shape).astype(np.float32)
            feed[stream.name] = np.ascontiguousarray(arr)
        return feed

    # =========================================================================
    # Request dispatch
    # =========================================================================
    def handle_request(self, req: InferenceRequest) -> Any:
        outputs, result, _ = self.handle_request_with_timing(req)
        if req.run_postprocess and self.adapter is not None:
            return result
        return outputs
