from typing import Optional

logger = logging.getLogger(__name__)
from worker.inference.inference_engine import InferenceModelEngine
from hailo_platform import (HEF, VDevice,FormatType, HailoSchedulingAlgorithm)
from common.util import load_adapter

class HailoEngine(InferenceModelEngine):
    def __init__(self, model_path, adapter_path: Optional[str] = None):
        params = VDevice.create_params()
        # Set the scheduling algorithm to round-robin to activate the scheduler
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = "SHARED"
        vDevice = VDevice(params)

        self.target = vDevice
        self.hef = HEF(model_path)

        self.infer_model = self.target.create_infer_model(model_path)
        self.infer_model.set_batch_size(1)

        self.adapter = load_adapter(adapter_path) if adapter_path else None

        self.config_ctx = self.infer_model.configure()
        self.configured_model = self.config_ctx.__enter__()
        self.configured_model.set_scheduler_priority(0)
        self.last_infer_job = None
