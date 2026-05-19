"""
Worker DATA plane FastAPI app.

Endpoints:
    POST /api/load_model           — Fetch + verify a model from controller, swap engine
    POST /api/inference            — Run an InferenceRequest, return outputs/result
    POST /api/calibration_burst    — Run a CPU stress loop for N seconds (no model needed)
    GET  /api/connectivity_test
    GET  /api/health
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

import requests
from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel

from shared.models import (
    ConnectionType,
    ConnectivityTestResponse,
    InferenceRequest,
    InferenceResponse,
    LoadModelRequest,
    LoadModelResponse,
    ResponseStatus,
    tensorfeed_to_payloads,
)
from shared.util import md5_of_file

logger = logging.getLogger(__name__)

LOCAL_MODELS_DIR = Path("worker_models")
LOCAL_ADAPTERS_DIR = Path("worker_adapters")


def make_data_app(deps: "WorkerDataDeps") -> FastAPI:
    app = FastAPI(title="Worker Data API", version="1.0.0")
    router = APIRouter()

    @router.get("/api/health")
    async def health() -> dict[str, Any]:
        # Surface adapter presence + class name so the dashboard can
        # verify what's ACTUALLY in the engine — not just what the
        # controller's distribution table claims is there. The two
        # can drift if a previous distribution succeeded under buggy
        # silent-fallback code.
        eng = deps.get_engine()
        adapter_obj = getattr(eng, "adapter", None) if eng is not None else None
        return {
            "status": "ok",
            "engine": deps.get_engine_name(),
            "loaded_model": deps.get_loaded_model(),
            "adapter_loaded": adapter_obj is not None,
            "adapter_class": (type(adapter_obj).__module__ + "."
                              + type(adapter_obj).__name__
                              if adapter_obj is not None else None),
        }

    @router.get("/api/connectivity_test")
    async def connectivity_test(request: Request) -> ConnectivityTestResponse:
        host = request.client.host if request.client else ""
        eth_subnet = deps.config["network"]["ethernet_subnet"]
        wifi_subnet = deps.config["network"]["wifi_subnet"]
        if host.startswith(eth_subnet):
            plane = ConnectionType.ETHERNET
        elif host.startswith(wifi_subnet):
            plane = ConnectionType.WIFI
        else:
            plane = ConnectionType.INVALID
        return ConnectivityTestResponse(
            from_identifier=deps.identifier, message="ok", plane=plane,
        )

    @router.post("/api/load_model")
    async def load_model(req: LoadModelRequest) -> LoadModelResponse:
        descriptor = req.descriptor
        # Resolve controller download host. In mock mode the cluster's "real"
        # ethernet subnet (e.g. 192.168.10.1) doesn't exist on the loopback
        # host, so honour FYP_CONTROLLER_HOST / [mock].controller_host instead.
        mock = deps.config.get("mock", {}) or {}
        if mock.get("enabled"):
            controller_ip = (os.environ.get("FYP_CONTROLLER_HOST", "").strip()
                             or mock.get("controller_host")
                             or "127.0.0.1")
        else:
            controller_ip = f"{deps.config['network']['ethernet_subnet']}1"
        controller_data_port = deps.config["controller"]["data_port"]

        LOCAL_MODELS_DIR.mkdir(exist_ok=True)
        LOCAL_ADAPTERS_DIR.mkdir(exist_ok=True)
        model_target = LOCAL_MODELS_DIR / descriptor.filename

        # Skip download if a matching file already exists
        if not (model_target.exists() and md5_of_file(model_target) == descriptor.md5):
            url = f"http://{controller_ip}:{controller_data_port}/api/models/{descriptor.filename}"
            try:
                r = await asyncio.to_thread(requests.get, url, stream=True, timeout=120)
                if r.status_code != 200:
                    return LoadModelResponse(
                        status=ResponseStatus.FAILURE,
                        worker_id=deps.get_worker_id(),
                        error=f"GET {url} -> {r.status_code}",
                    )
                with open(model_target, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
            except Exception as e:
                return LoadModelResponse(
                    status=ResponseStatus.FAILURE,
                    worker_id=deps.get_worker_id(),
                    error=f"download error: {e}",
                )
            actual = md5_of_file(model_target)
            if actual != descriptor.md5:
                return LoadModelResponse(
                    status=ResponseStatus.FAILURE,
                    worker_id=deps.get_worker_id(),
                    error=f"md5 mismatch (expected {descriptor.md5}, got {actual})",
                )

        # Adapter handling:
        #   * If the descriptor names an adapter, we MUST end up with that
        #     file on disk before swapping the engine. Silently falling
        #     back to "no adapter" was the cause of the cryptic "raw mode
        #     requires adapter" error downstream — the controller's
        #     adapters/ dir was missing the file, the GET 404'd, and the
        #     worker quietly built an adapter-less engine that then
        #     refused every raw-mode inference. Now: hard-fail with a
        #     clear message naming which file the controller doesn't have.
        #   * If the descriptor has no adapter_filename at all, that's
        #     a benchmark / tensor-only model and adapter_target stays
        #     None as before.
        adapter_target: Path | None = None
        if descriptor.adapter_filename:
            adapter_target = LOCAL_ADAPTERS_DIR / descriptor.adapter_filename
            if not (adapter_target.exists()
                    and (descriptor.adapter_md5 is None
                         or md5_of_file(adapter_target) == descriptor.adapter_md5)):
                url = (f"http://{controller_ip}:{controller_data_port}"
                       f"/api/adapters/{descriptor.adapter_filename}")
                try:
                    r = await asyncio.to_thread(requests.get, url, timeout=30)
                except Exception as e:
                    return LoadModelResponse(
                        status=ResponseStatus.FAILURE,
                        worker_id=deps.get_worker_id(),
                        error=(f"adapter download error: {type(e).__name__}: {e} "
                               f"(GET {url})"),
                    )
                if r.status_code != 200:
                    return LoadModelResponse(
                        status=ResponseStatus.FAILURE,
                        worker_id=deps.get_worker_id(),
                        error=(f"adapter '{descriptor.adapter_filename}' is not "
                               f"available from the controller (HTTP "
                               f"{r.status_code}). Make sure "
                               f"adapters/{descriptor.adapter_filename} "
                               f"exists on the controller and re-Distribute."),
                    )
                adapter_target.write_bytes(r.content)
                if descriptor.adapter_md5 is not None:
                    actual_a = md5_of_file(adapter_target)
                    if actual_a != descriptor.adapter_md5:
                        return LoadModelResponse(
                            status=ResponseStatus.FAILURE,
                            worker_id=deps.get_worker_id(),
                            error=(f"adapter md5 mismatch (expected "
                                   f"{descriptor.adapter_md5}, got {actual_a})"),
                        )

        try:
            await asyncio.to_thread(
                deps.swap_engine,
                str(model_target),
                str(adapter_target) if adapter_target else None,
                descriptor.backend,
                descriptor.name,
            )
        except Exception as e:
            return LoadModelResponse(
                status=ResponseStatus.FAILURE,
                worker_id=deps.get_worker_id(),
                error=f"engine swap failed: {e}",
            )

        return LoadModelResponse(
            status=ResponseStatus.SUCCESS,
            worker_id=deps.get_worker_id(),
            loaded_model=descriptor.name,
        )

    @router.post("/api/inference")
    async def inference(req: InferenceRequest) -> InferenceResponse:
        engine = deps.get_engine()
        if engine is None:
            raise HTTPException(status_code=400, detail="No engine loaded")
        if deps.get_loaded_model() != req.model:
            raise HTTPException(
                status_code=400,
                detail=f"Loaded model is '{deps.get_loaded_model()}', "
                       f"not '{req.model}'",
            )

        t0 = time.monotonic()
        try:
            outputs, result, timing = await asyncio.to_thread(
                _run_inference, engine, req
            )
        except Exception as e:
            logger.error("inference failed: %s", e)
            return InferenceResponse(
                status=ResponseStatus.FAILURE,
                request_id=req.request_id,
                worker_id=deps.get_worker_id(),
                error=str(e),
            )

        # Convert numpy outputs to TensorPayload only if a postprocess wasn't run
        ten_outputs = None
        if outputs is not None and not req.run_postprocess:
            try:
                ten_outputs = tensorfeed_to_payloads(outputs)
            except Exception as e:
                logger.warning("Failed to serialize outputs: %s", e)

        return InferenceResponse(
            status=ResponseStatus.SUCCESS,
            request_id=req.request_id,
            worker_id=deps.get_worker_id(),
            outputs=ten_outputs,
            result=result if req.run_postprocess else None,
            inference_s=timing.get("inference_s", time.monotonic() - t0),
            preprocess_s=timing.get("preprocess_s", 0.0),
            postprocess_s=timing.get("postprocess_s", 0.0),
        )

    # ---- Worker-driven benchmark progress -----------------------------
    # The benchmark loop publishes its current (successful, failed,
    # elapsed_s) snapshot to this dict every ~0.3 s. The controller
    # polls `GET /api/benchmark_progress` at ~2 Hz so the dashboard's
    # throughput chart and per-worker stats table can render live
    # numbers during a long-running benchmark instead of jumping
    # from 0 to the final total at the end.
    #
    # Updated from the synchronous benchmark thread, read from FastAPI's
    # asyncio loop. Python's GIL gives us atomic dict updates for these
    # simple types so no lock is needed.
    _bench_progress: dict[str, Any] = {
        "running": False, "model": "", "started_ms": 0,
        "successful": 0, "failed": 0,
        "duration_s": 0.0, "elapsed_s": 0.0,
        "batch_size": 1,
    }

    @router.get("/api/benchmark_progress")
    async def benchmark_progress() -> dict:
        return dict(_bench_progress)

    @router.post("/api/benchmark")
    async def benchmark(req: dict) -> dict:
        """Worker-driven dummy-mode benchmark.

        Why this exists: the experiment manager's per-request dispatch
        loop fronts every inference with an HTTP POST. That's correct
        for `raw` / `tensor` modes (the inputs come from the controller)
        but **wrong for `dummy`**: the worker generates its own random
        tensor, so wrapping each inference in a network round-trip
        caps the throughput at HTTP's RPS (~20-60 per worker) rather
        than the chip's actual TOPS. Result: a Hailo-8 that does
        1372 FPS locally appears to do ~10 FPS over HTTP.

        This endpoint runs the inference loop **inside the worker** for
        ``duration_s`` seconds, locally regenerating the dummy tensor
        each call (or reusing if ``reuse_input=True``). Returns the
        aggregated count and timing so the controller can update its
        per-worker stats with one HTTP call total instead of N.

        Request body:
            model: str
            duration_s: float
            batch_size: int (default 1)
            seed: int (default 42)
            reuse_input: bool (default True — keep the same dummy
                tensor for every call so we measure compute, not RNG)
        """
        import time as _time
        engine = deps.get_engine()
        if engine is None:
            raise HTTPException(400, "No engine loaded")
        model = req.get("model")
        if model and deps.get_loaded_model() != model:
            raise HTTPException(
                400, f"Loaded model is '{deps.get_loaded_model()}', "
                     f"not '{model}'",
            )
        duration_s = float(req.get("duration_s", 30.0))
        batch_size = int(req.get("batch_size", 1) or 1)
        seed = int(req.get("seed", 42) or 42)
        reuse_input = bool(req.get("reuse_input", True))

        if not hasattr(engine, "dummy_feed_from_signature") and engine.adapter is None:
            raise HTTPException(
                400, "Benchmark requires either an engine with "
                     "dummy_feed_from_signature or an adapter with "
                     "generate_dummy_inputs.",
            )

        # Compose the meta dict in case we fall through to the adapter.
        meta = dict(engine.get_engine_input_info())

        # PREFERENCE ORDER for dummy mode benchmarking:
        #   1. engine.dummy_feed_from_signature(...) — uses the model's
        #      REAL input names, shapes, and dtypes straight from the
        #      ONNX / HEF metadata. Deterministic, no heuristics, no
        #      remap step, no chance of adapter-vs-binding mismatch.
        #   2. adapter.generate_dummy_inputs(...) — fallback for engines
        #      that don't expose a signature-based generator.
        #
        # The earlier reverse-order (adapter first) caused mass
        # inference failures whenever the adapter's heuristic produced
        # an input name / shape / dtype that didn't quite match what
        # the binding expected — even when each individual piece looked
        # right in isolation. The engine's own generator avoids the
        # whole class of problems by reading directly from the model
        # metadata. Raw / item modes still use the adapter for
        # preprocess, so the adapter is exercised through its primary
        # code path.
        def _gen_feed():
            if hasattr(engine, "dummy_feed_from_signature"):
                return engine.dummy_feed_from_signature(
                    batch_size=batch_size, seed=seed,
                )
            try:
                return engine.adapter.generate_dummy_inputs(
                    batch_size=batch_size, seed=seed, meta=meta,
                )
            except TypeError as e:
                if "meta" not in str(e):
                    raise
                return engine.adapter.generate_dummy_inputs(
                    batch_size=batch_size, seed=seed,
                )

        # Publish-cadence for live progress updates. 0.3 s gives the
        # controller's 2 Hz poller enough freshness without taking
        # measurable cycles from the inference loop.
        PUBLISH_INTERVAL_S = 0.3

        def _run() -> dict:
            # Generated outside the loop when reuse_input is True so we
            # only measure compute, not RNG / preprocess. The chip does
            # the same MACs whatever bytes go in.
            #
            # Adapter failures here (e.g. adapter raises during
            # generate_dummy_inputs because the meta shape was weird)
            # should bubble up as a clean HTTP 500 instead of being
            # silently swallowed inside the loop's try/except — that's
            # what caused the "418200 of 418200 fail with no logs"
            # scenario we just hit. With reuse_input=True the inference
            # gets the SAME feed every call, so a failure here means
            # every subsequent inference will also fail; bail early.
            import traceback as _tb
            try:
                feed = _gen_feed() if reuse_input else None
            except Exception as e:
                tb = _tb.format_exc()
                logger.error("benchmark _gen_feed failed:\n%s", tb)
                raise HTTPException(
                    500,
                    f"adapter.generate_dummy_inputs raised "
                    f"{type(e).__name__}: {e}\n{tb}",
                )

            successful = 0
            failed = 0
            sum_latency_ms = 0.0
            min_latency_ms = float("inf")
            max_latency_ms = 0.0
            # Capture the first inference exception with full traceback
            # so the controller's report can show WHY everything failed
            # instead of a bare count.
            first_error: Optional[str] = None
            first_error_type: Optional[str] = None
            t_start = _time.monotonic()
            end_at = t_start + duration_s

            _bench_progress.update({
                "running": True,
                "model": model or "",
                "started_ms": int(_time.time() * 1000),
                "successful": 0, "failed": 0,
                "duration_s": duration_s, "elapsed_s": 0.0,
                "batch_size": batch_size,
            })
            next_publish_at = t_start + PUBLISH_INTERVAL_S

            while _time.monotonic() < end_at:
                cur_feed = feed if reuse_input else _gen_feed()
                t0 = _time.monotonic()
                try:
                    engine.infer_tensors(cur_feed)
                    dt_ms = (_time.monotonic() - t0) * 1000.0
                    successful += batch_size
                    sum_latency_ms += dt_ms
                    if dt_ms < min_latency_ms:
                        min_latency_ms = dt_ms
                    if dt_ms > max_latency_ms:
                        max_latency_ms = dt_ms
                except Exception as e:
                    failed += 1
                    if first_error is None:
                        # Capture first error verbatim with traceback so
                        # the controller-side response carries the cause
                        # back to the dashboard. Subsequent failures are
                        # counted but not re-logged — at 13k fails/sec
                        # the journal would otherwise drown in repeats.
                        first_error = f"{type(e).__name__}: {e}"
                        first_error_type = type(e).__name__
                        import traceback as _tb
                        logger.error(
                            "benchmark FIRST inference error — every "
                            "subsequent inference in this run will be "
                            "the same:\n%s", _tb.format_exc(),
                        )

                # Publish progress periodically. We snapshot under the
                # GIL — atomic dict updates of simple types — so the
                # asyncio reader gets a consistent view.
                now = _time.monotonic()
                if now >= next_publish_at:
                    _bench_progress.update({
                        "successful": successful,
                        "failed": failed,
                        "elapsed_s": now - t_start,
                    })
                    next_publish_at = now + PUBLISH_INTERVAL_S

            actual_duration = _time.monotonic() - t_start
            # Final flush so the poller's last tick sees the totals.
            _bench_progress.update({
                "running": False,
                "successful": successful,
                "failed": failed,
                "elapsed_s": actual_duration,
            })

            calls = max(1, successful // max(1, batch_size))
            return {
                "status": ResponseStatus.SUCCESS.value,
                "successful": successful,
                "failed": failed,
                "duration_s": actual_duration,
                "fps": successful / actual_duration if actual_duration > 0 else 0.0,
                "avg_latency_ms": sum_latency_ms / calls,
                "min_latency_ms": (0.0 if min_latency_ms == float("inf")
                                   else min_latency_ms),
                "max_latency_ms": max_latency_ms,
                "batch_size": batch_size,
                "reuse_input": reuse_input,
                # First inference error (if any), so the controller can
                # surface the actual cause back to the dashboard without
                # the user having to SSH each worker and read journalctl.
                "first_error": first_error,
                "first_error_type": first_error_type,
            }

        try:
            result = await asyncio.to_thread(_run)
        except Exception as e:
            logger.exception("benchmark loop crashed")
            raise HTTPException(500, f"benchmark failed: {e}")
        return result

    @router.post("/api/calibration_burst")
    async def calibration_burst(req: CalibrationBurstRequest) -> dict[str, Any]:
        """Pin all CPU cores to ~100 % for ``duration_s`` seconds.

        Used by the controller's CalibrationManager to look for an INA226 chip
        whose measured power spikes in lockstep with this worker's load. No
        model needs to be loaded — this is a pure busy-loop.
        """
        duration = max(0.5, min(float(req.duration_s), 30.0))
        n_threads = max(1, (os.cpu_count() or 4))
        logger.info("calibration_burst: %d threads for %.1fs", n_threads, duration)
        await asyncio.to_thread(_busy_burst, duration, n_threads)
        return {"status": "ok",
                "worker_id": deps.get_worker_id(),
                "duration_s": duration,
                "threads": n_threads}

    app.include_router(router)
    return app


class CalibrationBurstRequest(BaseModel):
    duration_s: float = 4.0


def _busy_burst(duration_s: float, n_threads: int) -> None:
    """Run ``n_threads`` CPU-bound loops for ``duration_s`` seconds."""
    stop_at = time.monotonic() + duration_s
    stop_evt = threading.Event()

    def worker() -> None:
        # Pure-Python integer churn — releases the GIL infrequently enough that
        # one thread per core actually heats the cores.
        x = 0
        while not stop_evt.is_set():
            for _ in range(50_000):
                x = (x * 1103515245 + 12345) & 0x7FFFFFFF
            if time.monotonic() >= stop_at:
                break

    threads = [threading.Thread(target=worker, daemon=True, name=f"burst-{i}")
               for i in range(n_threads)]
    for t in threads:
        t.start()
    # Sleep on the main thread; rely on stop_at for exit
    remaining = stop_at - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)
    stop_evt.set()
    for t in threads:
        t.join(timeout=1.0)


def _run_inference(engine, req: InferenceRequest):
    """Synchronous inference shim run in a worker thread."""
    return engine.handle_request_with_timing(req)


class WorkerDataDeps:
    def __init__(self, config: dict[str, Any], identifier: str,
                 get_worker_id: Callable[[], int],
                 get_engine: Callable[[], Any],
                 get_engine_name: Callable[[], str | None],
                 get_loaded_model: Callable[[], str | None],
                 swap_engine: Callable[[str, str | None, str, str], None]):
        self.config = config
        self.identifier = identifier
        self.get_worker_id = get_worker_id
        self.get_engine = get_engine
        self.get_engine_name = get_engine_name
        self.get_loaded_model = get_loaded_model
        self.swap_engine = swap_engine
