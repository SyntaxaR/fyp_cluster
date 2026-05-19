"""
ModelAdapter template.

Copy this file, rename it `<model>_adapter.py`, and implement the three methods
to fit your model. The worker's inference engine loads adapters dynamically via
`shared.util.load_adapter()`, so the file can live anywhere on disk.

Lifecycle (per InferenceRequest mode):
    tensor  : caller-supplied feed -> infer_tensors -> postprocess?
    raw     : items -> preprocess -> infer_tensors -> postprocess?
    dummy   : generate_dummy_inputs -> infer_tensors -> postprocess?

Function signatures MUST remain unchanged so that ONNX and Hailo engines can
call them interchangeably. See `worker/inference/adapters/yolov11_adapter.py`
for a worked example.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class RawItem:
    """Mirror of shared.models.RawItem (kept here so adapter files can be
    distributed without importing the rest of the project)."""
    type: str               # "image_bytes" | "image_path" | "image_url" | "text"
    data: Any
    mime: Optional[str] = None


class ModelAdapter:
    """Base class — subclass this in your `<model>_adapter.py`."""

    def __init__(self) -> None:
        pass

    def preprocess(self, items: list[RawItem],
                   meta: Optional[dict[str, Any]] = None) -> dict[str, np.ndarray]:
        """Convert raw items (images / text / ...) into the model's input feed."""
        raise NotImplementedError

    def postprocess(self, outputs: dict[str, np.ndarray],
                    meta: Optional[dict[str, Any]] = None) -> Any:
        """Turn raw model outputs into a useful representation (boxes,
        labels, JSON-serializable structures, paths to saved files, etc.)."""
        raise NotImplementedError

    def generate_dummy_inputs(self, batch_size: int = 1,
                              seed: int = 42) -> dict[str, np.ndarray]:
        """Produce a deterministic dummy feed for compute-only benchmarking."""
        raise NotImplementedError
