"""
Experiment lifecycle:
    PREPARE      -> validate config, choose dispatcher, resolve enrollment
    DISTRIBUTE   -> push the right model + adapter to each enrolled worker
                    (engine_overrides may pick .onnx or .hef per worker)
    RUN          -> dispatch inference requests until duration elapses
    AGGREGATE    -> compute latency / throughput / energy
    REPORT       -> persist to SQLite (experiments + experiment_workers)

Rather than blocking the controller's main loop, run() spawns its own asyncio
tasks. The Web UI reads result snapshots from cluster_state.last_experiment_result
and the dispatcher's per-request telemetry as it accumulates.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import statistics
import time
import uuid
import zipfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

import requests

from controller.cluster_state import (
    ClusterState,
    ModelDistributionState,
    WorkerStats,
)
from controller.dispatcher import make_dispatcher
from controller.dispatcher.base import BaseDispatcher
from shared.models import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    InferenceMode,
    LoadModelRequest,
    ModelDescriptor,
    RawItem,
    ResponseStatus,
    TensorPayload,
    WorkerStatus,
    ndarray_to_payload,
    tensorfeed_to_payloads,
)
from shared.util import load_adapter, md5_of_file

logger = logging.getLogger(__name__)


# Demo-content directories — see controller/_paths.py for the canonical map.
from controller._paths import MODELS_DIR, ADAPTERS_DIR, DATASETS_DIR  # noqa: E402

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class ExperimentManager:
    def __init__(self, config: dict[str, Any], state: ClusterState, database):
        self.config = config
        self.state = state
        self.db = database

        self._task: Optional[asyncio.Task] = None
        self._dispatcher: BaseDispatcher = make_dispatcher(
            config["dispatcher"].get("algorithm", "round_robin")
        )
        self._latencies_ms: list[float] = []
        self._success: int = 0
        self._fail: int = 0
        self._started_ms: int = 0
        self._finished_ms: int = 0

        # Workers participating in the current experiment (resolved at start).
        # Worker IDs are kept stable across runs by ClusterState.
        self._enrolled: list[int] = []
        # Per-worker engine assignment (worker_id -> "onnx"|"hailo") used when
        # picking which model file to push.
        self._engine_per_worker: dict[int, str] = {}
        # Snapshot of per-worker stats at experiment start so the report only
        # reflects this experiment's traffic.
        self._stats_baseline: dict[int, dict[str, float]] = {}
        # Per-worker timing windows (worker_id -> (started_ms, finished_ms))
        self._worker_windows: dict[int, tuple[int, int]] = {}
        # Pre-built workload for tensor / raw modes:
        #   raw    -> list of [RawItem]            (one batch == one request)
        #   tensor -> list of {input_name: TensorPayload}
        # The dispatch loop cycles through this list. None for dummy mode.
        self._workload_items: Optional[list[list[RawItem]]] = None
        self._workload_tensors: Optional[list[dict[str, TensorPayload]]] = None

    # =========================================================================
    # Public API
    # =========================================================================
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self, exp: ExperimentConfig) -> None:
        if self.is_running():
            raise RuntimeError("An experiment is already running")
        self.state.current_experiment = exp
        self.state.current_experiment_status = ExperimentStatus.PREPARING
        self._dispatcher = make_dispatcher(exp.dispatcher)
        if exp.weights:
            self._dispatcher.set_weights(exp.weights)
        self._latencies_ms.clear()
        self._success = 0
        self._fail = 0
        self._started_ms = 0
        self._finished_ms = 0
        self._enrolled = self._resolve_enrollment(exp)
        self._engine_per_worker = self._resolve_engines(exp, self._enrolled)
        # ---- Distribution gate ----
        # Refuse to launch unless every enrolled worker's distribution status
        # for this model is 'ok' AND the backend / adapter match what the
        # experiment is asking for. The user must click Distribute first.
        adapter_for_check = self._resolve_adapter_filename(exp)
        not_ready = []
        for wid in self._enrolled:
            backend = self._engine_per_worker[wid]
            if not self.state.is_distributed(exp.model_name, wid,
                                             backend=backend,
                                             adapter_filename=adapter_for_check):
                not_ready.append(wid)
        if not_ready:
            self.state.current_experiment_status = ExperimentStatus.FAILED
            raise RuntimeError(
                f"Model '{exp.model_name}' has not been distributed (or is "
                f"out-of-date) on workers {not_ready}. "
                "Click 'Distribute' on the experiment page first."
            )
        self._stats_baseline = {
            wid: self._snapshot_stats(wid) for wid in self._enrolled
        }
        self._worker_windows = {}
        self._workload_items = None
        self._workload_tensors = None
        # Build raw / tensor workload up-front so the dispatch loop is just a
        # tight cycle. Raises if dataset is missing or adapter cannot be
        # imported — the user sees the failure before tasks start firing.
        self._prepare_workload(exp)
        logger.info(
            "Experiment '%s': %d enrolled workers, engines=%s, mode=%s",
            exp.name, len(self._enrolled), self._engine_per_worker,
            exp.mode.value if isinstance(exp.mode, InferenceMode) else exp.mode,
        )
        self._task = asyncio.create_task(self._run(exp))

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    def record_result(self, payload: dict[str, Any]) -> None:
        """Called by data API when a worker POSTs telemetry."""
        ok = bool(payload.get("ok", True))
        latency_ms = float(payload.get("latency_ms", 0.0))
        if ok:
            self._success += 1
            self._latencies_ms.append(latency_ms)
        else:
            self._fail += 1

    # =========================================================================
    # Enrollment / engine resolution
    # =========================================================================
    def _resolve_enrollment(self, exp: ExperimentConfig) -> list[int]:
        active = self.state.active_workers()
        if exp.enrolled_workers:
            chosen = [wid for wid in exp.enrolled_workers if wid in active]
            missing = [wid for wid in exp.enrolled_workers if wid not in active]
            if missing:
                logger.warning("Enrolled workers not active and skipped: %s", missing)
            if not chosen:
                raise RuntimeError(
                    f"None of the enrolled workers {exp.enrolled_workers} "
                    f"are currently active (active={active})"
                )
            return chosen
        if not active:
            raise RuntimeError("No active workers to dispatch to")
        return list(active)

    # =========================================================================
    # Raw / tensor workload preparation
    # =========================================================================
    def _resolve_dataset_paths(self, dataset_filename: str) -> list[Path]:
        """Resolve a dataset_filename to a list of image paths on disk.

        Accepts either:
          * A single image (.jpg/.png/.bmp/.webp) — used as a 1-image dataset.
          * A .zip archive — extracted to ``datasets/_extracted/<stem>/`` once,
            then every image inside is returned.
        """
        ds_path = DATASETS_DIR / dataset_filename
        if not ds_path.exists():
            raise FileNotFoundError(f"Dataset not found: {ds_path}")

        if ds_path.suffix.lower() in _IMAGE_SUFFIXES:
            return [ds_path]

        if ds_path.suffix.lower() == ".zip":
            extracted_root = DATASETS_DIR / "_extracted" / ds_path.stem
            if not extracted_root.exists():
                extracted_root.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(ds_path, "r") as zf:
                    zf.extractall(extracted_root)
            images: list[Path] = []
            for p in sorted(extracted_root.rglob("*")):
                if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES:
                    images.append(p)
            if not images:
                raise ValueError(
                    f"No images found inside extracted {ds_path.name}"
                )
            return images

        raise ValueError(
            f"Unsupported dataset format: {ds_path.suffix} "
            f"(only single-image or .zip is supported by the dispatch loop)"
        )

    def _prepare_workload(self, exp: ExperimentConfig) -> None:
        """Build the per-request payload list for tensor / raw modes.

        Dummy mode is built request-by-request in ``_send_one`` because each
        request just needs a batch_size + seed.
        """
        mode = exp.mode if isinstance(exp.mode, InferenceMode) else InferenceMode(exp.mode)
        if mode == InferenceMode.DUMMY:
            return

        if mode == InferenceMode.RAW:
            if not exp.dataset_filename:
                raise ValueError("raw mode requires a dataset_filename")
            paths = self._resolve_dataset_paths(exp.dataset_filename)
            items_list: list[list[RawItem]] = []
            for p in paths:
                with open(p, "rb") as f:
                    raw = f.read()
                items_list.append([RawItem(
                    type="image_bytes",
                    data=base64.b64encode(raw).decode("ascii"),
                    mime=f"image/{p.suffix.lower().lstrip('.')}",
                )])
            self._workload_items = items_list
            logger.info("Raw workload prepared: %d image(s) from %s",
                        len(items_list), exp.dataset_filename)
            return

        if mode == InferenceMode.TENSOR:
            # Run the adapter's preprocess on the controller, then ship the
            # resulting tensor as TensorPayload. Worker skips preprocessing.
            adapter_path = self._resolve_adapter_path(exp)
            if adapter_path is None:
                raise ValueError(
                    "tensor mode requires an adapter on the controller "
                    "to preprocess inputs (no matching adapter found)"
                )
            adapter = load_adapter(adapter_path)
            if not exp.dataset_filename:
                raise ValueError("tensor mode requires a dataset_filename "
                                 "to feed the adapter's preprocess")
            paths = self._resolve_dataset_paths(exp.dataset_filename)
            tensors: list[dict[str, TensorPayload]] = []
            for p in paths:
                with open(p, "rb") as f:
                    raw = f.read()
                # Build a per-adapter RawItem; adapter may not import shared
                # modules so we duck-type the dataclass it expects.
                item = RawItem(
                    type="image_bytes",
                    data=raw,                       # bytes — adapter handles
                    mime=f"image/{p.suffix.lower().lstrip('.')}",
                )
                feed = adapter.preprocess([item], meta={})
                tensors.append(tensorfeed_to_payloads(feed))
            self._workload_tensors = tensors
            logger.info("Tensor workload prepared: %d batch(es) from %s",
                        len(tensors), exp.dataset_filename)

    def _resolve_adapter_path(self, exp: ExperimentConfig) -> Optional[Path]:
        if exp.adapter_filename:
            cand = ADAPTERS_DIR / exp.adapter_filename
            if cand.exists():
                return cand
        cand = ADAPTERS_DIR / f"{exp.model_name}_adapter.py"
        if cand.exists():
            return cand
        return None

    def _resolve_adapter_filename(self, exp: ExperimentConfig) -> Optional[str]:
        p = self._resolve_adapter_path(exp)
        return p.name if p is not None else None

    def _resolve_engines(self, exp: ExperimentConfig,
                         enrolled: list[int]) -> dict[int, str]:
        """Return worker_id -> "onnx"|"hailo" used to pick the model file."""
        out: dict[int, str] = {}
        for wid in enrolled:
            override = exp.engine_overrides.get(wid)
            if override in ("onnx", "hailo"):
                out[wid] = override
                continue
            # Fall back to whatever the worker self-reported via its registration
            reg = self.state.get_registered(wid)
            self_reported = (reg.engine if reg else None) or "onnx"
            if self_reported not in ("onnx", "hailo"):
                self_reported = "onnx"
            out[wid] = self_reported
        return out

    # =========================================================================
    # Internal phases
    # =========================================================================
    async def _run(self, exp: ExperimentConfig) -> None:
        try:
            self.state.current_experiment_status = ExperimentStatus.DISTRIBUTING
            await self._distribute_model(exp)

            self.state.current_experiment_status = ExperimentStatus.RUNNING
            self._started_ms = int(time.time() * 1000)
            await self._dispatch_loop(exp)
            self._finished_ms = int(time.time() * 1000)

            self.state.current_experiment_status = ExperimentStatus.AGGREGATING
            result = self._aggregate(exp)
            self.state.last_experiment_result = result

            # Append any first-failure messages captured from
            # /api/benchmark to the notes column so the user sees the
            # actual exception in the report instead of "OK 0 / fail N"
            # with no explanation.
            notes_out = exp.notes or ""
            bench_errors = getattr(self, "_bench_first_errors", None) or {}
            if bench_errors:
                err_lines = "\n".join(
                    f"  worker {wid}: {msg}"
                    for wid, msg in sorted(bench_errors.items())
                )
                notes_out = (notes_out + ("\n\n" if notes_out else "")
                             + "Benchmark first failures:\n" + err_lines)
            try:
                exp_id = self.db.insert_experiment(
                    result, model=exp.model_name,
                    dispatcher=exp.dispatcher,
                    mode=exp.mode.value if isinstance(exp.mode, InferenceMode) else str(exp.mode),
                    notes=notes_out,
                )
                self._persist_per_worker(exp_id, exp, result)
            except Exception as e:
                logger.error("Failed to persist experiment: %s", e)

            self.state.current_experiment_status = ExperimentStatus.COMPLETED
            logger.info("Experiment '%s' completed: %s", exp.name, result)
        except asyncio.CancelledError:
            self.state.current_experiment_status = ExperimentStatus.FAILED
            raise
        except Exception as e:
            logger.error("Experiment failed: %s", e)
            self.state.current_experiment_status = ExperimentStatus.FAILED

    # -------------------------------------------------------------------------
    # Distribution
    # -------------------------------------------------------------------------
    def _build_descriptor(self,
                          model_name: str,
                          backend: str,
                          adapter_filename: Optional[str] = None) -> ModelDescriptor:
        """Locate {model_name}.{onnx|hef} and bundle adapter info."""
        if backend not in ("onnx", "hailo"):
            raise ValueError(f"unsupported backend {backend!r}")
        ext = ".onnx" if backend == "onnx" else ".hef"
        model_path = MODELS_DIR / f"{model_name}{ext}"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No {backend} model file found for '{model_name}' "
                f"(expected {model_path})"
            )

        # Adapter resolution priority:
        #   1. explicit adapter_filename
        #   2. {model_name}_adapter.py
        #   3. None (worker uses its built-in default)
        adapter_path: Optional[Path] = None
        if adapter_filename:
            cand = ADAPTERS_DIR / adapter_filename
            if cand.exists():
                adapter_path = cand
            else:
                logger.warning("adapter_filename %s not found on disk", cand)
        if adapter_path is None:
            cand = ADAPTERS_DIR / f"{model_name}_adapter.py"
            if cand.exists():
                adapter_path = cand
        # Loud warning at distribution time when no adapter resolves —
        # the only way the worker side fails in this state is at the
        # first raw-mode call, which is far downstream and confusing.
        # Print here so the operator sees the issue when distributing.
        if adapter_path is None:
            logger.warning(
                "MODEL '%s': no adapter resolved (neither explicit "
                "adapter_filename=%r nor by-name '%s_adapter.py' is "
                "in %s). The worker will load the engine without an "
                "adapter, which works ONLY for tensor / dummy modes — "
                "raw-mode requests (e.g. /live) will fail.",
                model_name, adapter_filename, model_name, ADAPTERS_DIR,
            )

        return ModelDescriptor(
            name=model_name,
            filename=model_path.name,
            md5=md5_of_file(model_path),
            size_bytes=model_path.stat().st_size,
            backend=backend,
            adapter_filename=adapter_path.name if adapter_path else None,
            adapter_md5=md5_of_file(adapter_path) if adapter_path else None,
        )

    async def distribute_model(
        self,
        model_name: str,
        adapter_filename: Optional[str] = None,
        engine_overrides: Optional[dict[int, str]] = None,
        target_workers: Optional[list[int]] = None,
    ) -> dict[int, dict[str, Any]]:
        """Push a model + adapter to one or more workers and update
        ``state.model_distribution``. Returns per-worker outcome dicts.

        * ``target_workers`` defaults to all currently-ACTIVE workers.
        * ``engine_overrides`` forces a per-worker backend; missing entries
          fall back to the worker's self-reported engine, then to ``onnx``.
        * Each worker that already has the same (md5, backend, adapter)
          OK-record is short-circuited so re-clicking Distribute is cheap.
        """
        engine_overrides = engine_overrides or {}
        active = set(self.state.active_workers())
        if target_workers is None:
            targets = list(active)
        else:
            targets = [wid for wid in target_workers if wid in active]
            missing = [wid for wid in target_workers if wid not in active]
            if missing:
                logger.warning("Skipping inactive workers in distribute: %s",
                               missing)
        if not targets:
            return {}

        # Resolve (worker -> backend) once and build one descriptor per backend
        per_worker_backend: dict[int, str] = {}
        for wid in targets:
            override = engine_overrides.get(wid)
            if override in ("onnx", "hailo"):
                per_worker_backend[wid] = override
                continue
            reg = self.state.get_registered(wid)
            self_reported = (reg.engine if reg else None) or "onnx"
            if self_reported not in ("onnx", "hailo"):
                self_reported = "onnx"
            per_worker_backend[wid] = self_reported

        descriptors: dict[str, ModelDescriptor] = {}
        for backend in set(per_worker_backend.values()):
            descriptors[backend] = self._build_descriptor(
                model_name, backend, adapter_filename,
            )

        # Mark every target as 'pending' before firing so the UI updates fast.
        for wid in targets:
            backend = per_worker_backend[wid]
            desc = descriptors[backend]
            self.state.set_distribution(model_name, wid, ModelDistributionState(
                status="pending", backend=backend,
                adapter_filename=desc.adapter_filename, md5=desc.md5,
            ))

        default_data_port = self.config["worker"]["data_port"]
        results: dict[int, dict[str, Any]] = {}
        results_lock = asyncio.Lock()

        async def _send(wid: int) -> None:
            backend = per_worker_backend[wid]
            desc = descriptors[backend]
            reg = self.state.get_registered(wid)
            if reg is None:
                async with results_lock:
                    results[wid] = {"status": "failed",
                                    "error": "worker disappeared"}
                self.state.set_distribution(model_name, wid, ModelDistributionState(
                    status="failed", backend=backend,
                    adapter_filename=desc.adapter_filename, md5=desc.md5,
                    error="worker disappeared",
                ))
                return
            data_port = getattr(reg, "data_port", 0) or default_data_port
            url = f"http://{reg.data_ip}:{data_port}/api/load_model"
            err: Optional[str] = None
            try:
                resp = await asyncio.to_thread(
                    requests.post, url,
                    json=LoadModelRequest(descriptor=desc).model_dump(),
                    timeout=120,
                )
                if resp.status_code == 200:
                    body = resp.json()
                    if body.get("status") == ResponseStatus.SUCCESS.value:
                        reg.loaded_model = model_name
                        self.state.set_distribution(
                            model_name, wid,
                            ModelDistributionState(
                                status="ok", backend=backend,
                                adapter_filename=desc.adapter_filename,
                                md5=desc.md5,
                                distributed_ms=int(time.time() * 1000),
                            ),
                        )
                        async with results_lock:
                            results[wid] = {"status": "ok", "backend": backend,
                                            "md5": desc.md5}
                        logger.info("Distributed %s (%s) to worker %d",
                                    desc.filename, backend, wid)
                        return
                    err = body.get("error") or "worker rejected load"
                else:
                    err = f"HTTP {resp.status_code}"
            except Exception as e:
                err = str(e)

            self.state.set_distribution(model_name, wid, ModelDistributionState(
                status="failed", backend=backend,
                adapter_filename=desc.adapter_filename, md5=desc.md5,
                error=err,
            ))
            async with results_lock:
                results[wid] = {"status": "failed", "error": err}
            logger.error("distribute_model: worker %d failed: %s", wid, err)

        await asyncio.gather(*(asyncio.create_task(_send(wid)) for wid in targets),
                             return_exceptions=True)
        return results

    # =========================================================================
    # Single-shot inference (used by the /live page)
    # =========================================================================
    async def single_shot_inference(
        self,
        worker_id: int,
        model_name: str,
        image_bytes: bytes,
        run_postprocess: bool = True,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        """Send one image to one worker and return the parsed response.

        Skips the experiment lifecycle entirely — no DB write, no metrics
        baseline, no dispatcher. The /live page calls this directly per frame.

        ``timeout_s`` defaults to 60 s as a generous safety net — callers
        should still pass an explicit value matched to the worker's engine
        type (Hailo ~15 s, ONNX/CPU ~120 s for Real-ESRGAN). The
        SuperResPipeline uses ``_timeout_for_worker`` to do this.
        """
        reg = self.state.get_registered(worker_id)
        if reg is None:
            return {"ok": False, "error": f"worker {worker_id} not registered"}
        data_port = (getattr(reg, "data_port", 0)
                     or self.config["worker"]["data_port"])
        url = f"http://{reg.data_ip}:{data_port}/api/inference"

        item = {
            "type": "image_bytes",
            "data": base64.b64encode(image_bytes).decode("ascii"),
            "mime": "image/jpeg",
        }
        payload = {
            "model": model_name,
            "mode": InferenceMode.RAW.value,
            "items": [item],
            "run_postprocess": bool(run_postprocess),
            "request_id": f"live-{uuid.uuid4().hex[:8]}",
        }
        t0 = time.monotonic()
        try:
            resp = await asyncio.to_thread(
                requests.post, url, json=payload, timeout=timeout_s,
            )
            latency_ms = (time.monotonic() - t0) * 1000.0
            if resp.status_code != 200:
                return {"ok": False, "error": f"HTTP {resp.status_code}",
                        "latency_ms": latency_ms, "worker_id": worker_id}
            body = resp.json()
            return {
                "ok": body.get("status") == "success",
                "worker_id": worker_id,
                "latency_ms": latency_ms,
                "inference_s": float(body.get("inference_s", 0.0)),
                "preprocess_s": float(body.get("preprocess_s", 0.0)),
                "postprocess_s": float(body.get("postprocess_s", 0.0)),
                "result": body.get("result"),
                "error": body.get("error"),
            }
        except Exception as e:
            return {"ok": False, "error": str(e),
                    "latency_ms": (time.monotonic() - t0) * 1000.0,
                    "worker_id": worker_id}

    async def _distribute_model(self, exp: ExperimentConfig) -> None:
        """Backward-compat wrapper for the in-process experiment flow.

        With the manual-Distribute gate in place this should be a no-op (every
        enrolled worker is already 'ok'); we still call the public method to
        cover edge cases where a worker reconnected between the gate check and
        the dispatch loop start.
        """
        await self.distribute_model(
            model_name=exp.model_name,
            adapter_filename=self._resolve_adapter_filename(exp),
            engine_overrides=exp.engine_overrides,
            target_workers=self._enrolled,
        )

    # -------------------------------------------------------------------------
    # Worker-driven benchmark — for dummy mode without QPS cap
    # -------------------------------------------------------------------------
    async def _dispatch_loop_worker_benchmark(self, exp: ExperimentConfig) -> None:
        """Kick each active worker into a local `duration_s` benchmark
        loop and aggregate the results.

        One HTTP call per worker (instead of per-inference), so total
        cluster overhead is ``N_workers`` requests over the whole run.
        Each worker runs ``engine.infer_tensors(feed)`` in a tight Python
        loop using a worker-generated dummy tensor, eliminating both the
        controller-side dispatch overhead and any input-payload network
        cost. This is the path that lets a Hailo-8 actually report its
        ~1372 FPS / ~9.6 TOPS on ResNet-50 instead of being capped at
        HTTP RPS.
        """
        worker_data_port = self.config["worker"]["data_port"]
        active_initial = [
            wid for wid in self._enrolled
            if (reg := self.state.get_registered(wid)) is not None
            and (reg.status == WorkerStatus.ACTIVE.value
                 or reg.status == WorkerStatus.ACTIVE)
        ]
        if not active_initial:
            logger.warning("No active workers; benchmark dispatch skipped.")
            return

        logger.info(
            "Dummy-mode benchmark: %d worker(s) × %.1f s — using "
            "worker-driven /api/benchmark loop.",
            len(active_initial), exp.duration_s,
        )

        async def _one(wid: int):
            reg = self.state.get_registered(wid)
            if reg is None:
                return wid, None
            port = getattr(reg, "data_port", 0) or worker_data_port
            url = f"http://{reg.data_ip}:{port}/api/benchmark"
            payload = {
                "model": exp.model_name,
                "duration_s": float(exp.duration_s),
                "batch_size": int(exp.dummy_batch_size or 1),
                "seed": 42,
                "reuse_input": True,
            }
            # HTTP timeout = duration + 30 s grace for the response trip.
            t0 = time.monotonic()
            try:
                resp = await asyncio.to_thread(
                    requests.post, url, json=payload,
                    timeout=float(exp.duration_s) + 30.0,
                )
                wall_ms = (time.monotonic() - t0) * 1000.0
                if resp.status_code != 200:
                    logger.error(
                        "worker %d benchmark HTTP %d: %s",
                        wid, resp.status_code, resp.text[:200],
                    )
                    return wid, None
                body = resp.json()
                return wid, {**body, "wall_ms": wall_ms}
            except Exception as e:
                logger.error("worker %d benchmark RPC failed: %s", wid, e)
                return wid, None

        # ---- Live progress poller ----
        # While the per-worker /api/benchmark calls run their loops,
        # poll each worker's /api/benchmark_progress at ~2 Hz and feed
        # the snapshots into worker_stats. The Monitor page's
        # throughput chart reads those stats every 1 s, so the chart
        # ramps up smoothly during the run instead of jumping from
        # zero to the final total at the end.
        progress_done = asyncio.Event()
        # Track the count each worker had at the last poll tick so we
        # can compute deltas — worker_stats.requests_dispatched is a
        # monotonic counter across an entire experiment.
        last_snap: dict[int, dict[str, int]] = {
            w: {"successful": 0, "failed": 0} for w in active_initial
        }

        async def _poll_one_progress(wid: int) -> None:
            reg = self.state.get_registered(wid)
            if reg is None:
                return
            port = getattr(reg, "data_port", 0) or worker_data_port
            url = f"http://{reg.data_ip}:{port}/api/benchmark_progress"
            try:
                resp = await asyncio.to_thread(
                    requests.get, url, timeout=2.0,
                )
                if resp.status_code != 200:
                    return
                body = resp.json()
            except Exception:
                return
            cur_ok = int(body.get("successful", 0))
            cur_fail = int(body.get("failed", 0))
            prev = last_snap[wid]
            d_ok = max(0, cur_ok - prev["successful"])
            d_fail = max(0, cur_fail - prev["failed"])
            prev["successful"] = cur_ok
            prev["failed"] = cur_fail
            if d_ok == 0 and d_fail == 0:
                return
            stats = self.state.worker_stats.setdefault(wid, WorkerStats())
            stats.requests_dispatched += d_ok + d_fail
            stats.requests_succeeded += d_ok
            stats.requests_failed += d_fail

        async def _progress_loop():
            try:
                while not progress_done.is_set():
                    await asyncio.gather(
                        *[_poll_one_progress(w) for w in active_initial],
                        return_exceptions=True,
                    )
                    try:
                        await asyncio.wait_for(progress_done.wait(),
                                               timeout=0.5)
                    except asyncio.TimeoutError:
                        pass
            except asyncio.CancelledError:
                pass

        async def _run_benchmarks():
            try:
                return await asyncio.gather(*[_one(w) for w in active_initial])
            finally:
                progress_done.set()

        progress_task = asyncio.create_task(_progress_loop())
        results = await _run_benchmarks()
        await progress_task

        # The poller has already incremented dispatched / succeeded /
        # failed for the live window. The aggregation below adds the
        # TIMING stats (latency percentiles, per-worker windows) on
        # top, and reconciles any rounding drift between the poller's
        # last snapshot and the final totals returned by /api/benchmark.

        # Aggregate per-worker stats so the report-builder + UI see the
        # same shape they'd see from the per-request dispatch path.
        #
        # Reconcile with the live poller: the progress loop above
        # already incremented requests_dispatched / _succeeded / _failed
        # in lock-step with the worker's internal counters. Here we
        # only need to add the **delta** between the poller's last
        # snapshot and the worker's final totals (a few hundred
        # inferences usually fire after the last poll tick), plus the
        # TIMING stats that the progress endpoint doesn't carry.
        now_ms = int(time.time() * 1000)
        run_start_ms = self._started_ms or (now_ms - int(exp.duration_s * 1000))
        for wid, body in results:
            if body is None:
                continue
            stats = self.state.worker_stats.setdefault(wid, WorkerStats())
            ok_n = int(body.get("successful", 0))
            fail_n = int(body.get("failed", 0))
            # Top up to the final totals — anything still pending from
            # the last poller snapshot.
            snap = last_snap.get(wid, {"successful": 0, "failed": 0})
            d_ok = max(0, ok_n - snap["successful"])
            d_fail = max(0, fail_n - snap["failed"])
            stats.requests_dispatched += d_ok + d_fail
            stats.requests_succeeded += d_ok
            stats.requests_failed += d_fail

            stats.last_latency_ms = float(body.get("avg_latency_ms", 0.0))
            # The "latency" we record here is per-call (compute-only) —
            # there's no network round-trip on the worker side. Multiply
            # by ok_n so the eventual avg-latency calculation in
            # `_aggregate` divides out correctly.
            stats.sum_latency_ms += float(body.get("avg_latency_ms", 0.0)) * max(1, ok_n)
            self._success += ok_n
            self._fail += fail_n
            # Per-call latency goes into the global percentile bucket
            # too. We approximate each call's latency as the reported
            # average — the report's p95/p99 will therefore equal the
            # avg in dummy mode (acceptable for a synthetic benchmark).
            avg_ms = float(body.get("avg_latency_ms", 0.0))
            self._latencies_ms.extend([avg_ms] * ok_n)
            actual_dur_ms = int(float(body.get("duration_s",
                                                exp.duration_s)) * 1000.0)
            self._worker_windows[wid] = (run_start_ms,
                                         run_start_ms + actual_dur_ms)
            logger.info(
                "worker %d benchmark: %d OK / %d fail in %.2fs → "
                "%.1f FPS  (avg compute %.2f ms, min %.2f, max %.2f)",
                wid, ok_n, fail_n,
                float(body.get("duration_s", 0)),
                float(body.get("fps", 0)),
                float(body.get("avg_latency_ms", 0)),
                float(body.get("min_latency_ms", 0)),
                float(body.get("max_latency_ms", 0)),
            )
            # If this worker had a failure, log the first exception
            # verbatim so the controller's journal carries the cause
            # and the run report shows something the user can act on.
            first_err = body.get("first_error")
            if first_err:
                logger.error(
                    "worker %d benchmark first failure: %s "
                    "(this error repeats for every failed inference)",
                    wid, first_err,
                )
                # Stash on the experiment instance so _aggregate can
                # include it in the result's notes.
                if not hasattr(self, "_bench_first_errors"):
                    self._bench_first_errors = {}
                self._bench_first_errors[wid] = first_err

    # -------------------------------------------------------------------------
    # Dispatch loop
    # -------------------------------------------------------------------------
    async def _dispatch_loop(self, exp: ExperimentConfig) -> None:
        """Send inference requests at the configured rate for `duration_s`."""
        self._dispatcher.reset()
        end_at = time.monotonic() + exp.duration_s
        worker_data_port = self.config["worker"]["data_port"]
        target_qps = exp.target_qps

        # ---------------------------------------------------------------
        # SPECIAL CASE: dummy mode with no QPS cap → use the worker's
        # /api/benchmark endpoint instead of looping HTTP requests from
        # the controller. Rationale: the whole point of dummy mode is to
        # measure pure NPU/CPU compute throughput. Wrapping each inference
        # in a network round-trip caps the throughput at ~20-60 RPS per
        # worker (HTTP overhead) instead of the chip's actual TOPS. The
        # /api/benchmark endpoint runs the inference loop locally on the
        # worker for `duration_s` and returns the aggregate count, so
        # each worker is bounded only by its own compute.
        #
        # `target_qps > 0` disables this — when the user explicitly
        # asks for a rate cap, they want per-request scheduling. Same
        # for raw / tensor modes where the controller has to supply
        # the inputs.
        if (exp.mode == InferenceMode.DUMMY.value
                or exp.mode == InferenceMode.DUMMY) and not (target_qps and target_qps > 0):
            await self._dispatch_loop_worker_benchmark(exp)
            return

        # If no rate cap, fire as fast as the workers will let us; we use a
        # bounded semaphore to limit the in-flight count.
        max_inflight = 64
        sem = asyncio.Semaphore(max_inflight)
        next_send_at = time.monotonic()
        request_idx = 0

        async def _send_one(worker_id: int, request_id: str, idx: int):
            reg = self.state.get_registered(worker_id)
            if reg is None:
                return
            data_port = getattr(reg, "data_port", 0) or worker_data_port
            url = f"http://{reg.data_ip}:{data_port}/api/inference"
            mode = exp.mode.value if isinstance(exp.mode, InferenceMode) else str(exp.mode)

            payload: dict[str, Any] = {
                "model": exp.model_name,
                "mode": mode,
                "run_postprocess": bool(exp.run_postprocess),
                "request_id": request_id,
            }
            if mode == InferenceMode.DUMMY.value:
                payload["dummy_batch_size"] = exp.dummy_batch_size
            elif mode == InferenceMode.RAW.value:
                if not self._workload_items:
                    return
                items = self._workload_items[idx % len(self._workload_items)]
                payload["items"] = [
                    asdict(it) if is_dataclass(it) else it for it in items
                ]
            elif mode == InferenceMode.TENSOR.value:
                if not self._workload_tensors:
                    return
                tensors = self._workload_tensors[idx % len(self._workload_tensors)]
                payload["inputs"] = {
                    k: (asdict(v) if is_dataclass(v) else v)
                    for k, v in tensors.items()
                }
            stats = self.state.worker_stats.setdefault(worker_id, WorkerStats())
            stats.requests_dispatched += 1
            stats.inflight += 1

            # Track per-worker first/last timestamps for the drill-down view.
            now_ms = int(time.time() * 1000)
            window = self._worker_windows.get(worker_id)
            if window is None:
                self._worker_windows[worker_id] = (now_ms, now_ms)
            else:
                self._worker_windows[worker_id] = (window[0], now_ms)

            t0 = time.monotonic()
            try:
                resp = await asyncio.to_thread(
                    requests.post, url, json=payload, timeout=30
                )
                latency_ms = (time.monotonic() - t0) * 1000.0
                stats.last_latency_ms = latency_ms
                stats.sum_latency_ms += latency_ms
                if resp.status_code == 200:
                    body = resp.json()
                    if body.get("status") == ResponseStatus.SUCCESS.value:
                        self._success += 1
                        self._latencies_ms.append(latency_ms)
                        stats.requests_succeeded += 1
                    else:
                        self._fail += 1
                        stats.requests_failed += 1
                else:
                    self._fail += 1
                    stats.requests_failed += 1
            except Exception as e:
                logger.debug("dispatch error to %d: %s", worker_id, e)
                self._fail += 1
                stats.requests_failed += 1
            finally:
                stats.inflight -= 1
                sem.release()

        while time.monotonic() < end_at:
            # Only dispatch to workers that are both enrolled AND still active.
            active = [
                wid for wid in self._enrolled
                if (reg := self.state.get_registered(wid)) is not None
                and (reg.status == WorkerStatus.ACTIVE.value
                     or reg.status == WorkerStatus.ACTIVE)
            ]
            if not active:
                await asyncio.sleep(0.1)
                continue

            await sem.acquire()
            chosen = self._dispatcher.next(active)
            if chosen is None:
                sem.release()
                await asyncio.sleep(0.05)
                continue

            request_idx += 1
            request_id = f"req-{request_idx:08d}-{uuid.uuid4().hex[:6]}"
            asyncio.create_task(_send_one(chosen, request_id, request_idx - 1))

            if target_qps and target_qps > 0:
                next_send_at += 1.0 / target_qps
                sleep_for = next_send_at - time.monotonic()
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)

        # Drain in-flight requests with a bounded grace period so the
        # experiment doesn't run forever waiting on slow CPU workers
        # (e.g. Real-ESRGAN on ONNX/CPU = 30-90 s per frame, and we may
        # have dozens of in-flight requests when the deadline hits).
        # Anything still pending after the grace window counts as failed
        # — the worker will keep computing it but its result is discarded.
        drain_grace_s = max(5.0, exp.duration_s * 0.1)
        drain_end = time.monotonic() + drain_grace_s
        acquired = 0
        while acquired < max_inflight and time.monotonic() < drain_end:
            try:
                remaining = drain_end - time.monotonic()
                if remaining <= 0:
                    break
                await asyncio.wait_for(sem.acquire(), timeout=remaining)
                acquired += 1
            except asyncio.TimeoutError:
                break
        unfinished = max_inflight - acquired
        if unfinished > 0:
            logger.warning(
                "Dispatch drain timeout — %d request(s) abandoned after "
                "%.1fs grace. Counting as failed for this experiment.",
                unfinished, drain_grace_s,
            )
            self._fail += unfinished

    # -------------------------------------------------------------------------
    # Aggregate / persist
    # -------------------------------------------------------------------------
    def _snapshot_stats(self, wid: int) -> dict[str, float]:
        s = self.state.worker_stats.setdefault(wid, WorkerStats())
        return {
            "dispatched": float(s.requests_dispatched),
            "succeeded": float(s.requests_succeeded),
            "failed":    float(s.requests_failed),
            "sum_latency_ms": float(s.sum_latency_ms),
        }

    def _aggregate(self, exp: ExperimentConfig) -> ExperimentResult:
        latencies = sorted(self._latencies_ms)
        avg = statistics.fmean(latencies) if latencies else 0.0
        p95 = latencies[int(0.95 * (len(latencies) - 1))] if latencies else 0.0
        p99 = latencies[int(0.99 * (len(latencies) - 1))] if latencies else 0.0
        duration_s = max((self._finished_ms - self._started_ms) / 1000.0, 1e-9)
        throughput = (self._success / duration_s) if duration_s > 0 else 0.0

        avg_power = self.db.average_power_in_range(self._started_ms, self._finished_ms)
        energy_j = avg_power * duration_s
        energy_per_req = (energy_j / self._success) if self._success > 0 else 0.0

        # Per-worker stats relative to baseline taken at start().
        per_worker: dict[int, dict[str, float]] = {}
        for wid in self._enrolled:
            now = self._snapshot_stats(wid)
            base = self._stats_baseline.get(wid, {k: 0.0 for k in now})
            disp = now["dispatched"] - base["dispatched"]
            ok   = now["succeeded"]  - base["succeeded"]
            bad  = now["failed"]     - base["failed"]
            sumlat = now["sum_latency_ms"] - base["sum_latency_ms"]
            per_worker[wid] = {
                "dispatched": disp,
                "succeeded":  ok,
                "failed":     bad,
                "avg_latency_ms": (sumlat / disp) if disp > 0 else 0.0,
            }

        return ExperimentResult(
            name=exp.name,
            started_ms=self._started_ms,
            finished_ms=self._finished_ms,
            status=ExperimentStatus.COMPLETED,
            total_requests=self._success + self._fail,
            successful_requests=self._success,
            failed_requests=self._fail,
            avg_latency_ms=avg,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            avg_throughput_qps=throughput,
            avg_cluster_power_w=avg_power,
            energy_per_request_j=energy_per_req,
            per_worker=per_worker,
        )

    def _persist_per_worker(self, exp_id: int, exp: ExperimentConfig,
                            result: ExperimentResult) -> None:
        """Record one row per enrolled worker for drill-down reports."""
        rows = []
        for wid in self._enrolled:
            reg = self.state.get_registered(wid)
            stats = result.per_worker.get(wid, {})
            window = self._worker_windows.get(wid, (self._started_ms, self._finished_ms))
            chip = self.state.get_chip_for_worker(wid)
            rows.append({
                "experiment_id": exp_id,
                "worker_id": wid,
                "serial": reg.serial if reg else "",
                "identifier": reg.hardware_identifier if reg else "",
                "engine": self._engine_per_worker.get(wid, ""),
                "i2c_address": chip,
                "started_ms": window[0],
                "finished_ms": window[1],
                "dispatched": int(stats.get("dispatched", 0)),
                "succeeded":  int(stats.get("succeeded",  0)),
                "failed":     int(stats.get("failed",     0)),
                "avg_latency_ms": float(stats.get("avg_latency_ms", 0.0)),
            })
        try:
            self.db.insert_experiment_workers(rows)
        except Exception as e:
            logger.error("insert_experiment_workers failed: %s", e)
