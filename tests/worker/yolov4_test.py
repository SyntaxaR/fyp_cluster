from common.model import InferenceRequest, RawItem
from worker.inference.engines.onnx_engine import OnnxEngine
import os
import gc
import time

if __name__ == "__main__":
    print("Make sure the model is under src/worker/inference/models/yolov4")
    print("0: Exit\n1: Inference all .jpg/.jpeg/.png files under src/worker/inference/models/yolov4/inputs/\n2: Inference a single image\n3: Inference baseline test with random dummy input")
    user_input = input("Your choice:")
    if user_input == "0":
        exit()
    engine = OnnxEngine("src/worker/inference/models/yolov4/yolov4.onnx",                 "src/worker/inference/models/yolov4/yolov4_adapter.py")
    if user_input == "1":
        # Walk through all .jpg/.jpeg/png files under src/worker/inference/models/yolov4/inputs/ and run inference on them
        items = []
        for (root, dir, files) in os.walk('src/worker/inference/models/yolov4/inputs'):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                    items.append(RawItem(type="image_path", data=os.path.join(root, file)))
        engine.handle_request(
            InferenceRequest(
                model = "yolov4",
                mode = "raw",
                items = items,
                run_postprocess = True,
                meta = {
                    "save_images": True,
                    "output_dir": "src/worker/inference/models/yolov4/outputs"
                }
            )
        )
    elif user_input == "2":
        user_path = input("Please input image path (absolute directory, ending extension with .jpg/.jpeg/.png):")
        # Split folder path and filename
        folder, filename = os.path.split(user_path)
        # Save the output to user_path/output.jpg
        engine.handle_request(
            InferenceRequest(model="yolov4", mode="raw", items=[RawItem(type="image_path", data=user_path)],meta={
                "save_images": True,
                "output_path_template": str(filename.split(".")[:-1])+"_output.jpg",
                "output_dir": folder
            })
        )
        print(f"Output saved to {user_path}/{str(filename.split('.')[:-1])+'_output.jpg'}")
    elif user_input == "3":
        try:
            batch_size = int(input("Batch size? (default=10):"))
        except ValueError:
            batch_size = 10
        try:
            seed = int(input("Seed? (default=42):"))
        except ValueError:
            seed = 42
        while True:
            run_subprocess = input("Run subprocess?\nIt is recommended to choose no as dummy data is usually used for baseline tests, and postprocess (draw the squares & save image data to disk) consumes extra CPU and may slow down the process, affecting evaluation of the AI inference performance.\nYour choice (y/n):")
            if run_subprocess == "y" or run_subprocess == "Y":
                run_subprocess = True
                break
            elif run_subprocess == "n":
                run_subprocess = False
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
        req = InferenceRequest(model="yolov4", mode="dummy", dummy_batch_size=batch_size, dummy_seed=seed, run_postprocess=False)

        # Disable garbage collection to prevent jitter
        gc_was_enabled = gc.isenabled()
        gc.disable()

        # Warm up (not timed)
        print("Warming up...")
        warmup_iters = 5
        report_every = 1
        for i in range(warmup_iters):
            if i % report_every == 0:
                print(f"Warmup iteration {i+1}/{warmup_iters}...")
            engine.handle_request(req)


        # Timed
        iters = 20
        report_every = 5
        t0 = time.perf_counter()
        for i in range(iters):
            if i % report_every == 0:
                print(f"Iteration {i+1}/{iters}...")
            engine.handle_request(req)
        t1 = time.perf_counter()

        # Restart gc
        if gc_was_enabled:
            gc.enable()

        elapsed = t1 - t0
        avg_ms = (elapsed / iters) * 1000.0
        fps = (iters * batch_size) / elapsed

        print("\n=== Dummy Benchmark Results ===")
        print(f"batch_size      : {batch_size}")
        print(f"warmup iters    : {warmup_iters}")
        print(f"measured iters  : {iters}")
        print(f"total time (s)  : {elapsed:.4f}")
        print(f"avg latency (ms): {avg_ms:.3f} per iter")
        print(f"throughput (FPS): {fps:.2f} images/sec")