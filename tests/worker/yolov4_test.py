from common.model import InferenceRequest, RawItem
from worker.inference.engines.onnx_engine import OnnxEngine

engine = OnnxEngine("src/worker/inference/models/yolov4/yolov4.onnx", "src/worker/inference/models/yolov4/yolov4_adapter.py")

if __name__ == "__main__":
    engine.handle_request(
        InferenceRequest(
            model = "yolov4",
            mode = "raw",
            items = [
                RawItem(type="image_path", data="src/worker/inference/models/yolov4/inputs/input1.jpg"),
                RawItem(type="image_path", data="src/worker/inference/models/yolov4/inputs/input2.jpg")
            ],
            run_postprocess = True,
            meta = {
                "save_images": True,
                "output_dir": "src/worker/inference/models/yolov4/outputs"
            }
        )
    )