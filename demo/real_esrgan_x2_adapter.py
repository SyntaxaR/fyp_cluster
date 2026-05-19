"""
ModelAdapter for Real-ESRGAN **x2** super-resolution
(reference: https://github.com/ai-forever/Real-ESRGAN).

Wire format (worker side):
    preprocess(items=[RawItem(image_bytes)])
        -> {"input": ndarray (1, 3, H, H) float32 [0,1]}        # ONNX path
        -> {"input": ndarray (1, H, H, 3) float32 [0,255]}      # Hailo path
                ↑ layout selected automatically from
                   meta['engine_layout'] surfaced by the engine.

    postprocess(outputs=...)
        -> {"image_b64": "...JPEG base64...",
            "shape": [out_h, out_w, 3], "scale": 2}

Two pieces of magic that this adapter handles transparently:

1. **Auto input size.** ``meta['engine_input_shape']`` is filled in by the
   worker engine via ``get_engine_input_info()``. We pick the largest
   non-batch / non-channel dim as the model's expected tile size — works
   for both NCHW (B,C,H,W) and NHWC (B,H,W,C). So if your HEF was
   compiled with 256×256 OR 512×512, this adapter feeds the right size
   without you tweaking a constant.

2. **Aspect-ratio-preserving letterbox.** Source frames are usually 16:9
   (1280×720 etc.) but the model's input is square. Naive resize to
   square stretches the image and the displayed SR pane looks like
   "square content centered in a 16:9 viewport." Instead, we
   letterbox-pad to square before inference and crop the SR'd black
   bars out after, so the output preserves the source aspect ratio.

Working defaults (no env vars needed):
    Hailo + FormatType.FLOAT32 → feed [0, 255] float32 NHWC
    ONNX                       → feed [0,   1] float32 NCHW

Knobs (env vars, only for triage if a different HEF was calibrated
differently):
    FYP_SR_DEBUG=1            log every call (very noisy)
    FYP_SR_DEBUG_FIRST_N=3    auto-log the first N calls per process
    FYP_SR_INPUT_SCALE=K      override input feed max from default
                              to K (e.g. 1.0 if your HEF wants [0,1])
    FYP_SR_OUTPUT_SCALE=K     multiply output by K before clipping
                              to [0, 255]; default 255 (output is [0,1])
    FYP_SR_DISABLE_LETTERBOX=1  skip letterbox; squish-to-square (legacy)
"""
from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# Mirror shared.models.RawItem so the file stands alone (controller dispatches
# this adapter to workers as a single .py — it can't import shared/).
@dataclass
class RawItem:
    type: str
    data: Any
    mime: Optional[str] = None


# =============================================================================
# Model-specific constants — Real-ESRGAN x2
# =============================================================================
# Used as a fallback when the engine didn't surface a shape via
# meta['engine_input_shape'] (rare — only direct tensor-mode calls). The
# auto-detected shape from the engine takes priority.
INPUT_SIZE = 256


# =============================================================================
# Diagnostics — kept around so future model swaps can re-run the
# range/layout triage without code changes.
# =============================================================================
_DEBUG_ALWAYS = os.environ.get("FYP_SR_DEBUG", "0") not in ("0", "", "false", "False")
try:
    _DEBUG_FIRST_N = int(os.environ.get("FYP_SR_DEBUG_FIRST_N", "3"))
except ValueError:
    _DEBUG_FIRST_N = 3
try:
    _INPUT_SCALE_OVERRIDE = float(os.environ.get("FYP_SR_INPUT_SCALE", "0")) or None
except ValueError:
    _INPUT_SCALE_OVERRIDE = None
try:
    _OUTPUT_SCALE_OVERRIDE = float(os.environ.get("FYP_SR_OUTPUT_SCALE", "0")) or None
except ValueError:
    _OUTPUT_SCALE_OVERRIDE = None
_DISABLE_LETTERBOX = os.environ.get("FYP_SR_DISABLE_LETTERBOX", "0") not in (
    "0", "", "false", "False",
)
_call_counter = {"pre": 0, "post": 0}


def _arr_stats(arr) -> str:
    """One-line min/max/mean/std summary for a numpy array."""
    try:
        return (f"shape={tuple(arr.shape)} dtype={arr.dtype} "
                f"min={float(arr.min()):.4f} max={float(arr.max()):.4f} "
                f"mean={float(arr.mean()):.4f} std={float(arr.std()):.4f}")
    except Exception as e:
        return f"<stats failed: {e}>"


def _should_log(stage: str) -> bool:
    if _DEBUG_ALWAYS:
        return True
    n = _call_counter[stage]
    _call_counter[stage] = n + 1
    return n < _DEBUG_FIRST_N


def _resolve_input_size(meta) -> int:
    """Pick the model's expected spatial tile size from engine metadata."""
    if meta is None:
        return INPUT_SIZE
    shape = meta.get("engine_input_shape")
    if not shape:
        return INPUT_SIZE
    candidates = [int(d) for d in shape if isinstance(d, int) and d not in (0, 1, 3, 4)]
    if not candidates:
        return INPUT_SIZE
    return max(candidates)


# =============================================================================
# Image helpers — keep imports lazy so adapter file imports cleanly even on
# a controller that doesn't have cv2 installed.
# =============================================================================
def _decode_jpeg(item) -> np.ndarray:
    """RawItem with image_bytes -> HWC BGR uint8 ndarray."""
    import cv2
    data = item.data if not isinstance(item.data, str) else base64.b64decode(item.data)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode failed for image_bytes")
    return img


def _encode_jpeg(arr_bgr_uint8: np.ndarray, quality: int = 88) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".jpg", arr_bgr_uint8, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed in postprocess")
    return buf.tobytes()


def _letterbox_to_square(img_bgr: np.ndarray, target: int) -> tuple[np.ndarray, dict]:
    """Resize ``img_bgr`` to fit inside ``(target, target)`` preserving its
    aspect ratio, padded with black on whichever sides are short.

    Returns ``(square_image, pad_info)`` where ``pad_info`` records the
    inner content box so postprocess can crop the SR'd black bars away.
    """
    import cv2
    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"bad image shape {img_bgr.shape}")
    s = target / max(h, w)
    new_h = max(1, min(target, int(round(h * s))))
    new_w = max(1, min(target, int(round(w * s))))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_top = (target - new_h) // 2
    pad_bottom = target - new_h - pad_top
    pad_left = (target - new_w) // 2
    pad_right = target - new_w - pad_left

    if pad_top or pad_bottom or pad_left or pad_right:
        square = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
    else:
        square = resized

    return square, {
        "content_h": new_h, "content_w": new_w,
        "pad_top": pad_top, "pad_left": pad_left,
    }


# =============================================================================
# ModelAdapter
# =============================================================================
class ModelAdapter:
    """Real-ESRGAN x2.

    Works with both ONNX (NCHW float32 [0,1]) and Hailo (NHWC float32
    [0,255]) backends. The layout is auto-picked from the engine's
    ``engine_layout`` hint; ``meta['layout']`` overrides it.
    """

    # =========================================================================
    # preprocess
    # =========================================================================
    def preprocess(self, items, meta=None):
        import cv2
        if not isinstance(items, list):
            items = [items]
        input_size = _resolve_input_size(meta)

        feed_imgs = []
        orig_shapes = []
        pad_infos = []
        for item in items:
            img_bgr = _decode_jpeg(item)
            orig_shapes.append(img_bgr.shape[:2])  # (H, W) of source

            if _DISABLE_LETTERBOX:
                # Legacy squish-to-square path.
                tile_bgr = cv2.resize(
                    img_bgr, (input_size, input_size),
                    interpolation=cv2.INTER_AREA,
                )
                pad_info = {
                    "content_h": input_size, "content_w": input_size,
                    "pad_top": 0, "pad_left": 0,
                }
            else:
                tile_bgr, pad_info = _letterbox_to_square(img_bgr, input_size)

            tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
            feed_imgs.append(tile_rgb)
            pad_infos.append(pad_info)
        batch = np.stack(feed_imgs, axis=0)  # (N, H, W, 3) uint8 RGB

        # ----- Layout selection ------------------------------------------------
        # Priority:
        #   1. meta['layout']        — explicit caller override (legacy)
        #   2. engine_layout         — Hailo => NHWC, ONNX => NCHW
        #   3. NCHW_float32          — fallback
        layout = (meta or {}).get("layout")
        if layout is None:
            engine_layout = (meta or {}).get("engine_layout")
            engine_dtype = (meta or {}).get("engine_input_dtype") or ""
            if engine_layout == "NHWC":
                # Honour the engine's truthful post-configure dtype: some
                # HEFs silently reject FormatType.FLOAT32 forcing.
                if "uint8" in engine_dtype:
                    layout = "NHWC_uint8"
                else:
                    layout = "NHWC_float32"
            else:
                layout = "NCHW_float32"

        # ----- Build feed tensor with the right range / dtype / axis order ----
        if layout == "NHWC_uint8":
            # Native Hailo uint8 path (rare — only when the HEF was
            # compiled WITHOUT FormatType.FLOAT32 forced on the input).
            tensor = np.ascontiguousarray(batch)
            applied_scale = 255.0  # for logging
        elif layout == "NHWC_float32":
            # Hailo with FormatType.FLOAT32 forced on the input vstream.
            # The chip was calibrated against uint8 [0, 255], so feed
            # the *same* numeric range as float32, NOT the [0, 1] PyTorch
            # convention. (Validated empirically — feeding [0, 1] makes
            # the chip see ~zero everywhere and emit a black image.)
            scale = (_INPUT_SCALE_OVERRIDE
                     if _INPUT_SCALE_OVERRIDE is not None else 255.0)
            tensor = batch.astype(np.float32) * (scale / 255.0)
            tensor = np.ascontiguousarray(tensor)
            applied_scale = scale
        else:
            # ONNX NCHW float32 [0, 1]
            scale = (_INPUT_SCALE_OVERRIDE
                     if _INPUT_SCALE_OVERRIDE is not None else 1.0)
            tensor = batch.astype(np.float32) * (scale / 255.0)
            tensor = np.transpose(tensor, (0, 3, 1, 2))
            tensor = np.ascontiguousarray(tensor)
            applied_scale = scale

        if meta is not None:
            meta["sr_orig_shapes"] = orig_shapes
            meta["sr_layout"] = layout
            meta["sr_input_size"] = input_size
            meta["sr_pad_infos"] = pad_infos
            meta["sr_letterbox"] = not _DISABLE_LETTERBOX

        if _should_log("pre"):
            logger.warning(
                "[real_esrgan_x2] preprocess: layout=%s input_size=%d "
                "letterbox=%s applied_input_scale=%.3f feed_stats: %s",
                layout, input_size, not _DISABLE_LETTERBOX,
                applied_scale, _arr_stats(tensor),
            )

        return {"input": tensor}

    # =========================================================================
    # postprocess
    # =========================================================================
    def postprocess(self, outputs, meta=None):
        import cv2
        meta = meta or {}
        layout = meta.get("sr_layout", "NCHW_float32")

        # Pick the first output tensor regardless of name.
        out = next(iter(outputs.values()))

        log_now = _should_log("post")
        if log_now:
            logger.warning(
                "[real_esrgan_x2] postprocess: sr_layout=%s raw_output: %s",
                layout, _arr_stats(out),
            )

        # ----- Bring into HWC uint8 RGB regardless of engine layout -----
        if layout == "NHWC_uint8":
            # (1, OUT_H, OUT_W, 3) uint8 RGB.
            if out.ndim == 4:
                out = out[0]
            arr_rgb = np.clip(out, 0, 255).astype(np.uint8)
        elif layout == "NHWC_float32":
            # Hailo with FormatType.FLOAT32 forced on output stream.
            # Output range is typically [0, 1] (PyTorch convention preserved
            # through the HEF). Multiply by 255 to get pixel values.
            if out.ndim == 4:
                out = out[0]
            scale = (_OUTPUT_SCALE_OVERRIDE
                     if _OUTPUT_SCALE_OVERRIDE is not None else 255.0)
            arr_rgb = np.clip(out * scale, 0.0, 255.0).astype(np.uint8)
        else:
            # ONNX NCHW float32 in [0, 1] (Real-ESRGAN occasionally
            # over/undershoots — clip is mandatory).
            if out.ndim == 4:
                out = out[0]
            arr = np.transpose(out, (1, 2, 0))           # CHW -> HWC
            scale = (_OUTPUT_SCALE_OVERRIDE
                     if _OUTPUT_SCALE_OVERRIDE is not None else 255.0)
            arr = np.clip(arr * scale, 0.0, 255.0).astype(np.uint8)
            arr_rgb = arr

        # ----- Crop the letterbox bars away to recover source aspect -----
        # We computed sr_factor BEFORE cropping so it reflects the model's
        # actual scale (2× for x2), not the post-crop dims.
        in_size = int(meta.get("sr_input_size") or INPUT_SIZE)
        out_h_full = int(arr_rgb.shape[0])
        out_w_full = int(arr_rgb.shape[1])
        sr_factor = (out_h_full / in_size) if in_size else 1.0

        pad_infos = meta.get("sr_pad_infos") or []
        did_crop = False
        if pad_infos and meta.get("sr_letterbox", True):
            pi = pad_infos[0]  # batch=1 in the live demo
            crop_top = int(round(pi["pad_top"] * sr_factor))
            crop_left = int(round(pi["pad_left"] * sr_factor))
            crop_h = int(round(pi["content_h"] * sr_factor))
            crop_w = int(round(pi["content_w"] * sr_factor))
            # Bounds-clamp so a rounding wobble can't index out of range.
            crop_top = max(0, min(out_h_full, crop_top))
            crop_left = max(0, min(out_w_full, crop_left))
            crop_h = max(1, min(crop_h, out_h_full - crop_top))
            crop_w = max(1, min(crop_w, out_w_full - crop_left))
            if (crop_top, crop_left, crop_h, crop_w) != (0, 0, out_h_full, out_w_full):
                arr_rgb = arr_rgb[crop_top:crop_top + crop_h,
                                  crop_left:crop_left + crop_w]
                did_crop = True

        # JPEG encoder wants BGR.
        arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
        jpeg = _encode_jpeg(arr_bgr, quality=88)

        h_final = int(arr_bgr.shape[0])
        w_final = int(arr_bgr.shape[1])
        derived_scale = max(1, int(round(sr_factor)))

        if log_now:
            logger.warning(
                "[real_esrgan_x2] postprocess: final_rgb (cropped=%s): %s "
                "(if mean≈0 → black; ≈255 → white; 30-200 → real content)",
                did_crop, _arr_stats(arr_rgb),
            )

        return {
            "image_b64": base64.b64encode(jpeg).decode("ascii"),
            "shape": [h_final, w_final, 3],
            "scale": derived_scale,
        }

    # =========================================================================
    # Dummy mode — used by the benchmark / "throughput-only" inference path.
    # Generates random tensors that match the engine's expected shape AND
    # layout so we don't waste seconds on JPEG decode + cv2 resize when
    # we're benchmarking raw NPU throughput.
    # =========================================================================
    def generate_dummy_inputs(self, batch_size: int = 1, seed: int = 42,
                              meta=None):
        """Produce a dummy feed sized + laid out for whichever engine is
        loaded.

        ``meta`` is filled in by the engine layer (see
        ``InferenceModelEngine.handle_request_with_timing``) — keys we
        consume:

            engine_input_shape : list[int]   — auto-detected tile size
            engine_layout      : "NHWC"/"NCHW"

        Without ``meta`` we fall back to the legacy NCHW float32 [0, 1]
        @ ``INPUT_SIZE`` shape, which matches the ONNX path. That keeps
        the old engine-less unit tests working.
        """
        rng = np.random.default_rng(seed)

        input_size = _resolve_input_size(meta) if meta else INPUT_SIZE

        # Determine layout exactly the same way preprocess() does so the
        # dummy path produces a buffer Hailo's binding will accept.
        layout = (meta or {}).get("layout")
        if layout is None:
            engine_layout = (meta or {}).get("engine_layout")
            engine_dtype = (meta or {}).get("engine_input_dtype") or ""
            if engine_layout == "NHWC":
                # Honour the engine's truthful post-configure dtype: some
                # HEFs silently reject FormatType.FLOAT32 forcing.
                if "uint8" in engine_dtype:
                    layout = "NHWC_uint8"
                else:
                    layout = "NHWC_float32"
            else:
                layout = "NCHW_float32"

        if layout == "NHWC_uint8":
            # Native Hailo uint8 — values in [0, 255] integer.
            shape = (batch_size, input_size, input_size, 3)
            arr = rng.integers(0, 256, size=shape, dtype=np.uint8)
        elif layout == "NHWC_float32":
            # Hailo with FormatType.FLOAT32 forced — float32 NHWC, [0, 255]
            # range so the chip's quantizer sees realistic pixel values.
            shape = (batch_size, input_size, input_size, 3)
            arr = rng.uniform(0.0, 255.0, size=shape).astype(np.float32)
        else:
            # ONNX NCHW float32, [0, 1] — same range preprocess() uses.
            shape = (batch_size, 3, input_size, input_size)
            arr = rng.uniform(0.0, 1.0, size=shape).astype(np.float32)

        return {"input": np.ascontiguousarray(arr)}
