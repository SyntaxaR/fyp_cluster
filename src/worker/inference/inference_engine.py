"""
worker/inference_engine.py
Inference Engine for worker nodes. (ONNX format)
"""

from abc import ABC, abstractmethod
import numpy as np
from common.model import InferenceRequest


class InferenceModelEngine(ABC):
    @abstractmethod
    def __init__(self, model_path):
        pass

    @abstractmethod
    def infer_tensors(self, input_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        pass

    @abstractmethod
    def handle_request(self, req: InferenceRequest):
        pass