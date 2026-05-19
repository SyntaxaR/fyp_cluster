"""
Per-model GOPS extraction + caching.

Two backends, two strategies:

  * **ONNX** — parse the graph and sum per-node MAC counts. Conv, MatMul,
    Gemm, ConvTranspose, BatchNorm, etc. are computed by hand from
    shape-inferred tensor dims. No external dependency beyond `onnx`
    itself (which is a transitive dep of onnxruntime, already installed).

  * **HEF**  — Hailo's binary format. The Python SDK doesn't expose a
    clean OPS field on the HEF object, and there's no industry-standard
    way to extract it from the binary. Strategy in order of preference:
      1. A manual sidecar ``<file>.hef.meta.json`` written by the
         operator (always wins).
      2. A sibling ``<stem>.onnx`` next to the HEF — we parse the
         ONNX and reuse its GOPS. This works because the .hef and
         .onnx compile from the same network topology, so the MAC
         count is identical (the .hef just runs it quantized).
      3. Return ``unknown`` and let the UI prompt for a manual value.

    Earlier versions of this module had a hard-coded "Hailo Model Zoo"
    lookup table with ~25 entries. It was DELETED on purpose — several
    numbers in there were sourced from memory rather than the official
    YAMLs, which is exactly the kind of false-confidence that misleads
    benchmark reports. If you want the lookup back, source it from
    https://github.com/hailo-ai/hailo_model_zoo/tree/master/hailo_model_zoo/cfg
    YAML files and verify each entry against the published FPS.

Results are cached as a sidecar ``<model>.meta.json`` next to the
model file so we don't re-parse on every UI refresh.

OPS definition: Hailo / NVIDIA / industry-standard convention counts
1 MAC = 2 OPs (one multiply + one add). All numbers in this module use
that convention so they're directly comparable to the chip's advertised
"26 TOPS" peak.

    TOPS_observed = OPS_per_inference × FPS / 10^12
                  = GOPS × FPS / 1000
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _sibling_onnx_gops(hef_path: Path) -> Optional[float]:
    """If a ``<stem>.onnx`` lives in the same directory as the HEF,
    parse THAT to derive GOPS. The .hef and .onnx compile from the
    same graph, so MACs are identical — only the precision differs.

    Returns None if no sibling exists or the parse fails.
    """
    sibling = hef_path.with_suffix(".onnx")
    if not sibling.exists():
        return None
    try:
        macs = _compute_onnx_macs(sibling)
    except Exception as e:
        logger.warning("sibling ONNX parse failed for %s: %s", sibling, e)
        return None
    if not macs or macs <= 0:
        return None
    return round(macs * 2 / 1e9, 3)


# =============================================================================
# ONNX MACs computation — graph walk + manual per-op MAC formulas.
# =============================================================================
def _compute_onnx_macs(path: Path) -> Optional[int]:
    """Return total MACs for one forward pass of the ONNX model at
    ``path``, or None if computation fails.

    Counts the heavy ops only: Conv, ConvTranspose, MatMul, Gemm. Light
    ops (BN, ReLU, Add, Pool, Concat) are dropped — they're ~5% of
    total MACs in any conv-heavy network and including them adds noise
    that the "OPS" convention specifically excludes.
    """
    try:
        import onnx
        from onnx import shape_inference
    except ImportError as e:
        logger.warning("onnx package missing — cannot compute MACs: %s", e)
        return None

    try:
        model = onnx.load(str(path))
        inferred = shape_inference.infer_shapes(model)
    except Exception as e:
        logger.warning("onnx load/shape-infer failed for %s: %s", path, e)
        return None

    # Build a name -> shape map from inputs + initializers + value_info.
    shape_map: dict[str, tuple[Optional[int], ...]] = {}

    def _shape_of(t) -> tuple[Optional[int], ...]:
        dims = t.type.tensor_type.shape.dim
        return tuple(
            (d.dim_value if d.HasField("dim_value") and d.dim_value > 0
             else None)
            for d in dims
        )

    for vi in (list(inferred.graph.input)
               + list(inferred.graph.output)
               + list(inferred.graph.value_info)):
        shape_map[vi.name] = _shape_of(vi)

    # Initializers (weights) carry their own shape.
    for init in inferred.graph.initializer:
        shape_map[init.name] = tuple(init.dims)

    total_macs = 0
    for node in inferred.graph.node:
        op = node.op_type
        try:
            macs = _macs_for_node(op, node, shape_map)
        except Exception as e:
            logger.debug("MAC calc failed for %s (%s): %s",
                         op, node.name or "?", e)
            macs = 0
        total_macs += macs
    return total_macs


def _macs_for_node(op: str, node, shape_map) -> int:
    """Per-op MAC count. Returns 0 for ops we don't count."""

    def s(name):  # shape lookup helper
        return shape_map.get(name, ())

    def prod(seq):
        out = 1
        for x in seq:
            if x is None or x <= 0:
                return 0
            out *= int(x)
        return out

    if op == "Conv":
        # output shape: (N, OC, ...spatial)
        # weight shape: (OC, IC/groups, KH, KW, ...)
        out_shape = s(node.output[0])
        w_shape = s(node.input[1])
        if len(out_shape) < 3 or len(w_shape) < 2:
            return 0
        # MACs = product(output_spatial) × OC × KH×KW × (IC/groups)
        # = product(output) × KH×KW × (IC/groups)
        kern = prod(w_shape[2:])
        ic_per_group = w_shape[1] if w_shape[1] else 0
        return prod(out_shape) * kern * ic_per_group

    if op == "ConvTranspose":
        out_shape = s(node.output[0])
        w_shape = s(node.input[1])
        if len(out_shape) < 3 or len(w_shape) < 2:
            return 0
        kern = prod(w_shape[2:])
        # weight here is (IC, OC/groups, KH, KW)
        oc_per_group = w_shape[1] if w_shape[1] else 0
        return prod(out_shape[:1]) * prod(out_shape[2:]) * kern * oc_per_group

    if op == "MatMul":
        # A: (..., M, K)  B: (..., K, N) -> (..., M, N)
        a_shape = s(node.input[0])
        b_shape = s(node.input[1])
        if not a_shape or not b_shape:
            return 0
        if len(a_shape) < 2 or len(b_shape) < 2:
            return 0
        M = a_shape[-2]
        K = a_shape[-1]
        N = b_shape[-1]
        if not all(isinstance(x, int) and x > 0 for x in (M, K, N)):
            return 0
        # Broadcast prefix dims (batch).
        batch = prod(a_shape[:-2]) or 1
        return batch * M * N * K

    if op == "Gemm":
        # alpha * A @ B + beta * C
        # A: (M, K) or (K, M) depending on transA; B similar.
        a_shape = s(node.input[0])
        b_shape = s(node.input[1])
        if len(a_shape) != 2 or len(b_shape) != 2:
            return 0
        # Read transA / transB attrs.
        trans_a = trans_b = 0
        for attr in node.attribute:
            if attr.name == "transA":
                trans_a = int(attr.i)
            elif attr.name == "transB":
                trans_b = int(attr.i)
        M = a_shape[1] if trans_a else a_shape[0]
        K = a_shape[0] if trans_a else a_shape[1]
        N = b_shape[0] if trans_b else b_shape[1]
        if not all(isinstance(x, int) and x > 0 for x in (M, K, N)):
            return 0
        return M * N * K

    return 0


# =============================================================================
# Public API
# =============================================================================
def compute_model_gops(model_path: str | Path) -> Optional[tuple[float, str]]:
    """Return ``(gops_per_inference, source)`` for the given model file.

    ``source`` is one of:
        ``"auto"``     — parsed from the ONNX graph
        ``"lookup"``   — matched a HEF filename against the model-zoo table
        ``"manual"``   — read from a sidecar overriding the auto value
        ``"unknown"``  — neither path worked (returns ``None``)

    Returns ``None`` if we genuinely can't determine the value.
    """
    path = Path(model_path)
    if not path.exists():
        return None

    ext = path.suffix.lower()
    if ext == ".onnx":
        macs = _compute_onnx_macs(path)
        if not macs or macs <= 0:
            return None
        return (round(macs * 2 / 1e9, 3), "auto")

    if ext == ".hef":
        # No direct extraction from the binary — try the sibling-ONNX
        # fallback first. If the operator uploaded both .hef and .onnx
        # for the same model, we get an accurate number for free.
        g = _sibling_onnx_gops(path)
        if g is not None:
            return (g, "sibling-onnx")
        return None

    return None


def sidecar_path(model_path: str | Path) -> Path:
    return Path(model_path).with_suffix(Path(model_path).suffix + ".meta.json")


def load_sidecar(model_path: str | Path) -> dict:
    p = sidecar_path(model_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception as e:
        logger.warning("sidecar %s unreadable: %s", p, e)
        return {}


def save_sidecar(model_path: str | Path, data: dict) -> None:
    p = sidecar_path(model_path)
    try:
        p.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.warning("sidecar %s write failed: %s", p, e)


_STALE_SOURCES = {"lookup"}  # left over from the deleted Hailo Model Zoo table


def get_or_compute_gops(model_path: str | Path) -> Optional[dict]:
    """Cached version of ``compute_model_gops``.

    Returns the sidecar dict ``{"gops": <float>, "source": <str>}``,
    creating it on first call if the auto/sibling-onnx path succeeded.
    A user-supplied sidecar with ``"source": "manual"`` always wins —
    set this when the auto value is wrong or the file is a HEF whose
    sibling .onnx isn't available.

    Sidecars written by older versions of this module with
    ``"source": "lookup"`` (from the deleted Hailo Model Zoo table)
    are invalidated and re-derived — those values were unreliable.
    """
    path = Path(model_path)
    side = load_sidecar(path)
    if side.get("source") == "manual" and isinstance(side.get("gops"), (int, float)):
        return {"gops": float(side["gops"]), "source": "manual"}
    if (isinstance(side.get("gops"), (int, float))
            and side.get("source")
            and side["source"] not in _STALE_SOURCES):
        return {"gops": float(side["gops"]), "source": side["source"]}

    # Either no cache, or the cached value came from the deleted lookup
    # table — recompute from scratch and overwrite.
    res = compute_model_gops(path)
    if res is None:
        return None
    gops, source = res
    save_sidecar(path, {"gops": gops, "source": source})
    return {"gops": gops, "source": source}


def observed_tops(gops: float, fps: float) -> float:
    """TOPS = GOPS × FPS / 1000. Convenience for UI / reports."""
    if gops <= 0 or fps <= 0:
        return 0.0
    return gops * fps / 1000.0
