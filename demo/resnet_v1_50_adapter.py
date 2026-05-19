"""
ModelAdapter for ResNet-50 (resnet_v1_50) — DUMMY MODE ONLY.

Purpose: pure NPU / CPU compute throughput benchmarking. We don't run
real-image classification on this adapter; we only need a known-MACs
model that the inference engine can crunch random tensors through, so
we can compute observed TOPS via:

    TOPS_observed = 2 × MACs × FPS / 10^12
                  = 2 × 4.10 G × FPS / 10^12
                  = FPS × 8.2 / 1000

Hailo Model Zoo reference (resnet_v1_50.hef, INT8, Hailo-8):
    Input: 224 × 224 × 3
    OPS:   ~6.98 G   (MACs ≈ 3.49 G — see note below)
    FPS:   ~1372 @ batch=1  →  ~9.6 TOPS observed
    (Note: Hailo's "OPS" already counts mul+add as 2 ops, so the
    common "FPS × 8.2 / 1000" formula uses ~4.1 G MACs from the
    standard ResNet-50 paper instead. Both are within 10-20% of
    each other; pick one source and cite it consistently in the
    report.)

Layout selection mirrors real_esrgan_x2_adapter:
    * Hailo  + FormatType.FLOAT32  → NHWC, float32, [0, 255]
    * ONNX                         → NCHW, float32, [0, 1]
The actual numeric range doesn't matter for compute benchmarking
(the chip does the same number of MACs regardless of input values),
but matching the engine's expected range avoids quantizer warnings.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Log only the first N dummy calls so we have evidence the adapter is
# producing what we think it is, without spamming the journal on long
# benchmark runs. Reset on engine reload.
_DEBUG_FIRST_N = 3
_call_counter = [0]


# Mirror shared.models.RawItem so this file stands alone — controller
# distributes adapters as single .py files.
@dataclass
class RawItem:
    type: str
    data: Any
    mime: Optional[str] = None


# ResNet-50 ImageNet input — both ONNX and the Hailo Model Zoo
# resnet_v1_50.hef use 224 × 224 × 3.
INPUT_SIZE = 224


def _resolve_input_size(meta) -> int:
    """If the engine surfaced its true input shape, use that — covers
    HEFs compiled with non-default input dims. Falls back to the
    ImageNet 224 default."""
    if not meta:
        return INPUT_SIZE
    shape = meta.get("engine_input_shape")
    if not shape:
        return INPUT_SIZE
    candidates = [int(d) for d in shape if isinstance(d, int) and d not in (0, 1, 3, 4)]
    return max(candidates) if candidates else INPUT_SIZE


class ModelAdapter:
    """ResNet-50 — benchmark-only adapter.

    The class-level ``SUPPORTED_MODES`` declares which inference modes
    this adapter implements. The controller's experiment UI reads this
    attribute to grey out unsupported modes in the dropdown. Adapters
    that don't declare the attribute are assumed to support all three
    modes (backward-compatible with older adapters in the codebase).
    """

    # ResNet-50 here is dummy-only. Real classification (preprocess +
    # postprocess into top-K labels) isn't implemented because the FYP
    # only uses this adapter for compute throughput benchmarking.
    SUPPORTED_MODES = frozenset({"dummy"})

    # =========================================================================
    # preprocess / postprocess intentionally NOT implemented.
    # The controller-side experiment UI gates `raw` and `tensor` modes off
    # via SUPPORTED_MODES so these should never be called; but we still
    # raise a clear error in case someone bypasses the UI.
    # =========================================================================
    def preprocess(self, items, meta=None):
        raise NotImplementedError(
            "resnet50 adapter is dummy-only (compute benchmarking). "
            "Real-image classification is out of scope for this adapter — "
            "use mode='dummy' in the experiment config."
        )

    def postprocess(self, outputs, meta=None):
        raise NotImplementedError(
            "resnet50 adapter is dummy-only — postprocess not implemented."
        )

    # =========================================================================
    # Dummy mode — generates random tensors that match the engine's
    # expected shape + layout so neither the chip nor the binding
    # complains, and the inference time reflects pure compute throughput.
    # =========================================================================
    def generate_dummy_inputs(self, batch_size: int = 1, seed: int = 42,
                              meta=None):
        rng = np.random.default_rng(seed)
        input_size = _resolve_input_size(meta)

        # Layout selection — same priority order as real_esrgan_x2_adapter.
        # NEW: honour the engine's truthful post-configure dtype. Some
        # HEFs silently reject FormatType.FLOAT32 forcing and keep the
        # native uint8 format; the engine now reports the actual dtype
        # via engine_input_dtype, and we pick uint8 vs float32 here so
        # the adapter's buffer matches the binding's expected byte count.
        layout = (meta or {}).get("layout")
        if layout is None:
            engine_layout = (meta or {}).get("engine_layout")
            engine_dtype = (meta or {}).get("engine_input_dtype") or ""
            if engine_layout == "NHWC":
                if "uint8" in engine_dtype:
                    layout = "NHWC_uint8"
                else:
                    layout = "NHWC_float32"
            else:
                layout = "NCHW_float32"

        if layout == "NHWC_uint8":
            shape = (batch_size, input_size, input_size, 3)
            arr = rng.integers(0, 256, size=shape, dtype=np.uint8)
        elif layout == "NHWC_float32":
            # Hailo with FormatType.FLOAT32 forced — feed [0, 255] float32
            # to match the chip's uint8 calibration (same convention as
            # real_esrgan_x2_adapter).
            shape = (batch_size, input_size, input_size, 3)
            arr = rng.uniform(0.0, 255.0, size=shape).astype(np.float32)
        else:
            # ONNX NCHW float32 [0, 1]. Real ResNet-50 ONNX inputs are
            # ImageNet-mean-normalized but for compute benchmarking the
            # numeric range is irrelevant — the chip does the same MACs.
            shape = (batch_size, 3, input_size, input_size)
            arr = rng.uniform(0.0, 1.0, size=shape).astype(np.float32)

        # First-N diagnostic log so a buffer-size mismatch is easy to
        # diagnose: the journal will show exactly which branch fired,
        # the resolved input_size, and the produced tensor's dtype +
        # shape + nbytes. If you see `nbytes=150528` (uint8) when you
        # expected `nbytes=602112` (float32), the adapter version
        # deployed to the worker is stale — re-distribute.
        if _call_counter[0] < _DEBUG_FIRST_N:
            _call_counter[0] += 1
            logger.warning(
                "[resnet50] dummy: layout=%s input_size=%d "
                "engine_layout=%r meta_layout=%r "
                "tensor: dtype=%s shape=%s nbytes=%d",
                layout, input_size,
                (meta or {}).get("engine_layout"),
                (meta or {}).get("layout"),
                arr.dtype, tuple(arr.shape), int(arr.nbytes),
            )

        return {"input": np.ascontiguousarray(arr)}
