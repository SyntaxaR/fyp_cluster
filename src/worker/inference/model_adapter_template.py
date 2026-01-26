"""
All model adapters used should edit from ModelAdapter
ModelAdapter class contains the preprocess, postprocess, and dummy inputs generation of a specific model
Copy and edit this file for adapting a new custom model, the function signatures MUST remain unchanged
For example, see implementations under src/worker/inference/adapters
"""

import numpy as np
from typing import Any, Optional
from dataclasses import dataclass

@dataclass
class RawItem:
    type: str
    data: Any
    mime: Optional[str] = None

class ModelAdapter:
    """
    Implement these methods to fit a custom model.
    - Required for Raw Item mode inference
        preprocess: Raw items -> tensor feed
    - Required for Dummy Input mode inference
        generate_dummy_inputs: for compute-only benchmarking
    - Required if any postprocess is needed (e.g. draw boxes and save for object detection models)
        postprocess: tensor outputs -> result
    """
    def __init__(self):
        pass

    def preprocess(self, items: list[RawItem], meta: Optional[dict[str, Any]] = None) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def postprocess(self, outputs: dict[str, np.ndarray], meta: Optional[dict[str, Any]] = None) -> Any:
        raise NotImplementedError

    def generate_dummy_inputs(self, batch_size: int = 1, seed: int = 42) -> dict[str, np.ndarray]:
        raise NotImplementedError