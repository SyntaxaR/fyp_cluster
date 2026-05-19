"""
Super-resolution video streaming pipeline (live demo).

Reads frames from a video file, dispatches each frame to one of the active
workers via ExperimentManager.single_shot_inference, and pushes the (orig,
upscaled) pair into a small queue that the Web UI polls at display time.

Demo semantics (result-paced playback):
    The pipeline keeps `max_inflight = len(active_workers)` frames in flight.
    As each result arrives, the next source frame is dispatched. So the
    display rate equals the cluster's throughput. The user adjusts the active
    worker count via a slider — the visible playback speed scales linearly.

This module is independent of the batch ExperimentManager; it talks to
workers directly via ExperimentManager.single_shot_inference, but maintains
its own dispatch loop and stats. That keeps the /live demo cleanly separated
from the report-generating experiment flow.
"""
from __future__ import annotations

import asyncio
import base64
import collections
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FramePair:
    """Display payload — original (input) + upscaled (output) JPEG bytes."""
    frame_id: int
    orig_jpeg: bytes
    sr_jpeg: bytes
    worker_id: int
    inference_ms: float
    e2e_ms: float


@dataclass
class PipelineStatus:
    running: bool = False
    video_path: Optional[str] = None
    model_name: Optional[str] = None
    active_workers: list[int] = field(default_factory=list)
    frames_processed: int = 0
    frames_failed: int = 0
    fps: float = 0.0
    avg_inference_ms: float = 0.0
    last_error: Optional[str] = None
    # Source video dimensions (populated when the pipeline opens the
    # mp4). Display dims of the cluster's SR'd output (populated from
    # the first arriving FramePair, since the SR factor depends on the
    # model and we don't know it until inference returns).
    source_width: Optional[int] = None
    source_height: Optional[int] = None
    output_width: Optional[int] = None
    output_height: Optional[int] = None
    # Record-to-file state — populated only when the user enabled "save SR
    # video" before clicking Start. ``record_path`` is relative to the
    # controller's working directory.
    record_active: bool = False
    record_path: Optional[str] = None
    record_frames_written: int = 0
    record_total_frames: Optional[int] = None    # None = looping live source


class SuperResPipeline:
    """Drives the /live super-resolution demo."""

    def __init__(self, experiment_manager, state):
        self.experiment = experiment_manager
        self.state = state

        self._video_path: Optional[Path] = None
        self._model_name: Optional[str] = None
        # Subset of worker IDs the pipeline is allowed to dispatch to. The
        # /live page mutates this set live via set_active_workers().
        self._active_workers: list[int] = []

        # Result deque — UI polls .latest() at display time. cap is short so
        # the UI shows fresh frames; older frames just get dropped.
        self._latest_lock = asyncio.Lock()
        self._latest: Optional[FramePair] = None

        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # Run generation: incremented on every start(). Each dispatched frame
        # carries the run_id it was issued under; results from earlier runs
        # are dropped on arrival. Without this, in-flight HTTP requests
        # outliving stop() can clobber _latest with stale frames after the
        # next start(), and (worse) frame_id resets to 0 each run so the
        # ``pair.frame_id > self._latest.frame_id`` ordering check
        # incorrectly rejects every fresh frame until the new run reaches
        # the previous run's last id.
        self._run_id: int = 0

        # Stats
        self._frames_processed = 0
        self._frames_failed = 0
        self._processed_ts: collections.deque[float] = collections.deque(maxlen=120)
        self._inf_ms_window: collections.deque[float] = collections.deque(maxlen=60)
        self._last_error: Optional[str] = None

        # ---- Record-to-file mode --------------------------------------------
        # When ``set_record_path()`` is called BEFORE start(), each upscaled
        # frame is also muxed into an mp4. Frames arrive out of order from
        # the parallel worker pool, so we hold them in a reorder buffer
        # keyed by frame_id and flush in sequence to keep the output video
        # monotonic (otherwise the result is garbled motion).
        self._record_enabled: bool = False
        self._record_path: Optional[Path] = None
        self._record_writer = None              # type: Any  (cv2.VideoWriter)
        self._record_buffer: dict[int, FramePair] = {}
        self._record_next_id: int = 0
        self._record_frames_written: int = 0
        self._record_total_frames: Optional[int] = None
        self._record_fps: float = 30.0
        self._record_lock = asyncio.Lock()

        # Resolution metadata — populated lazily so the /live page can
        # display the actual source / output dimensions instead of
        # hardcoded placeholders. Source dims come from cv2 when we
        # open the file; output dims come from the first SR'd frame
        # because they depend on the model's SR factor (x2 / x4 …).
        self._source_w: Optional[int] = None
        self._source_h: Optional[int] = None
        self._output_w: Optional[int] = None
        self._output_h: Optional[int] = None

        # ---- Run persistence — completed runs are dropped into the
        # experiments DB so they show up on /reports next to regular
        # benchmark runs. We track all per-frame latencies (not just
        # the rolling 60-sample UI window) so the report's p95/p99 are
        # meaningful, plus per-worker counters for the drill-down view.
        self._run_started_ms: Optional[int] = None
        self._run_finished_ms: Optional[int] = None
        self._run_name: Optional[str] = None
        self._all_inf_ms: list[float] = []
        self._per_worker_processed: dict[int, int] = {}
        self._per_worker_failed: dict[int, int] = {}
        self._per_worker_inf_sum_ms: dict[int, float] = {}
        self._per_worker_window: dict[int, tuple[int, int]] = {}
        self._persisted_run_id: Optional[int] = None

        # Per-engine inference timeout (seconds). Hailo NPU finishes a tile
        # in ~30-100 ms, but Real-ESRGAN on a CPU-only worker can take
        # 30-90 s per frame — well past the previous hardcoded 15 s, which
        # was killing every frame with `Read timed out`. Pick the timeout
        # at dispatch time based on the worker's reported engine so we
        # don't apply the slow-CPU budget to fast Hailo workers (otherwise
        # a genuinely-stuck Hailo worker would be invisible for 90 s).
        self._timeout_hailo_s: float = 15.0
        self._timeout_onnx_s: float = 120.0
        self._timeout_default_s: float = 60.0

    # =========================================================================
    # Configuration
    # =========================================================================
    def set_video(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        self._video_path = path

    def set_model(self, model_name: str) -> None:
        self._model_name = model_name

    def set_active_workers(self, wids: list[int]) -> None:
        """Update the worker subset live. The dispatcher sees the new set on
        its next iteration; in-flight frames complete on whichever worker
        they were sent to."""
        self._active_workers = list(dict.fromkeys(int(x) for x in wids))

    def set_record_path(self, path: Optional[Path]) -> None:
        """If ``path`` is set, the next start() will mux upscaled frames into
        an mp4 at that path. Pass ``None`` to disable recording — live
        streaming continues normally."""
        if path is None:
            self._record_enabled = False
            self._record_path = None
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._record_enabled = True
            self._record_path = path

    # =========================================================================
    # Status
    # =========================================================================
    def status(self) -> PipelineStatus:
        return PipelineStatus(
            running=self.is_running(),
            video_path=str(self._video_path) if self._video_path else None,
            model_name=self._model_name,
            active_workers=list(self._active_workers),
            frames_processed=self._frames_processed,
            frames_failed=self._frames_failed,
            fps=self._current_fps(),
            avg_inference_ms=(sum(self._inf_ms_window) / len(self._inf_ms_window)
                              if self._inf_ms_window else 0.0),
            last_error=self._last_error,
            source_width=self._source_w,
            source_height=self._source_h,
            output_width=self._output_w,
            output_height=self._output_h,
            record_active=self._record_enabled and self._record_writer is not None,
            record_path=str(self._record_path) if self._record_path else None,
            record_frames_written=self._record_frames_written,
            record_total_frames=self._record_total_frames,
        )

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def latest_frame(self) -> Optional[FramePair]:
        async with self._latest_lock:
            return self._latest

    def _current_fps(self) -> float:
        if len(self._processed_ts) < 2:
            return 0.0
        span = self._processed_ts[-1] - self._processed_ts[0]
        if span <= 0:
            return 0.0
        return (len(self._processed_ts) - 1) / span

    # =========================================================================
    # Lifecycle
    # =========================================================================
    async def start(self) -> None:
        if self.is_running():
            return
        if self._video_path is None:
            raise RuntimeError("Pipeline: video not set")
        if not self._model_name:
            raise RuntimeError("Pipeline: model not set")
        if not self._active_workers:
            raise RuntimeError("Pipeline: no active workers selected")

        self._stop_event.clear()
        self._frames_processed = 0
        self._frames_failed = 0
        self._processed_ts.clear()
        self._inf_ms_window.clear()
        self._last_error = None
        # Reset resolution snapshot so a previous run's dims don't linger
        # on the dashboard while the new run is opening the new file.
        self._source_w = None
        self._source_h = None
        self._output_w = None
        self._output_h = None
        # ---- Run persistence — fresh state for the new run.
        self._run_started_ms = int(time.time() * 1000)
        self._run_finished_ms = None
        self._run_name = (
            f"live-sr-{time.strftime('%H%M%S', time.localtime())}"
        )
        self._all_inf_ms.clear()
        self._per_worker_processed.clear()
        self._per_worker_failed.clear()
        self._per_worker_inf_sum_ms.clear()
        self._per_worker_window.clear()
        self._persisted_run_id = None
        # Bump generation BEFORE clearing _latest so any stale _process_frame
        # task that wakes up between these two lines sees the new run_id and
        # bails out without writing to _latest.
        self._run_id += 1
        async with self._latest_lock:
            self._latest = None
        run_id = self._run_id

        # ---- Record-mode setup (must happen before dispatch starts) ----
        # We don't open the cv2.VideoWriter here because we need to know the
        # output frame size, which we only learn from the first SR'd frame.
        # _process_frame opens it on the first arriving result. Reset the
        # buffer + counters so a previous run's state can't leak in.
        self._record_buffer.clear()
        self._record_next_id = 0
        self._record_frames_written = 0
        self._record_total_frames = None
        self._record_writer = None    # opened lazily on first frame
        self._record_codec_verified = False  # set True after first byte-check

        self._task = asyncio.create_task(self._run(run_id))

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        self._task = None
        # Flush any frames still in the reorder buffer that we've already
        # received in-order, then close the video file. Anything still
        # dangling out-of-order is dropped (worker crashed mid-pipeline,
        # nothing we can do).
        await self._close_record_writer()
        # Drop the run into the experiments DB so it shows up alongside
        # benchmarks on /reports. Idempotent — only persists once per run.
        self._persist_run()

    def _persist_run(self) -> None:
        """Insert the just-finished SR run into the experiments DB so it
        shows up on /reports. Idempotent: subsequent calls in the same
        run are no-ops.

        We treat the live run as a `raw`-mode round-robin experiment for
        the purposes of the reports table — that's exactly what it is,
        even though it didn't go through ExperimentManager.
        """
        from shared.models import ExperimentResult, ExperimentStatus

        if self._persisted_run_id is not None:
            return  # already persisted this run
        if self._run_started_ms is None:
            return  # never actually started
        # If no frames at all completed and no failures recorded, the
        # pipeline was started but stopped immediately — skip the empty
        # report row to avoid clutter.
        if (self._frames_processed + self._frames_failed) == 0:
            return

        finished_ms = self._run_finished_ms or int(time.time() * 1000)
        self._run_finished_ms = finished_ms
        duration_s = max((finished_ms - self._run_started_ms) / 1000.0, 1e-9)
        total = self._frames_processed + self._frames_failed

        # Latency aggregates from the full per-frame log.
        latencies = sorted(self._all_inf_ms)
        if latencies:
            avg_lat = sum(latencies) / len(latencies)
            p95 = latencies[int(0.95 * (len(latencies) - 1))]
            p99 = latencies[int(0.99 * (len(latencies) - 1))]
        else:
            avg_lat = p95 = p99 = 0.0

        throughput = (self._frames_processed / duration_s) if duration_s > 0 else 0.0

        # Power: pull the same DB the experiment manager uses, average
        # over the run window. Returns 0 when no INA226 samples exist
        # (mock mode / disabled monitor).
        try:
            avg_power = self.experiment.db.average_power_in_range(
                self._run_started_ms, finished_ms,
            )
        except Exception:
            avg_power = 0.0
        energy_per_req = ((avg_power * duration_s) / self._frames_processed
                          if self._frames_processed > 0 else 0.0)

        per_worker: dict[int, dict[str, float]] = {}
        for wid in (set(self._per_worker_processed) | set(self._per_worker_failed)):
            disp = (self._per_worker_processed.get(wid, 0)
                    + self._per_worker_failed.get(wid, 0))
            ok_ = self._per_worker_processed.get(wid, 0)
            avg_w_lat = (self._per_worker_inf_sum_ms.get(wid, 0.0) / ok_
                         if ok_ > 0 else 0.0)
            per_worker[wid] = {
                "dispatched": disp,
                "succeeded":  ok_,
                "failed":     self._per_worker_failed.get(wid, 0),
                "avg_latency_ms": avg_w_lat,
            }

        result = ExperimentResult(
            name=self._run_name or "live-sr",
            started_ms=self._run_started_ms,
            finished_ms=finished_ms,
            status=ExperimentStatus.COMPLETED,
            total_requests=total,
            successful_requests=self._frames_processed,
            failed_requests=self._frames_failed,
            avg_latency_ms=avg_lat,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            avg_throughput_qps=throughput,
            avg_cluster_power_w=avg_power,
            energy_per_request_j=energy_per_req,
            per_worker=per_worker,
        )

        notes_bits = ["Live SR run"]
        if self._video_path is not None:
            notes_bits.append(f"video={self._video_path.name}")
        if self._record_path is not None and self._record_frames_written > 0:
            notes_bits.append(
                f"record={self._record_path.name} "
                f"({self._record_frames_written} frames)"
            )
        if self._active_workers:
            notes_bits.append(f"workers={self._active_workers}")
        notes = " · ".join(notes_bits)

        try:
            exp_id = self.experiment.db.insert_experiment(
                result,
                model=self._model_name or "(unknown)",
                dispatcher="round_robin",  # SR pipeline does its own RR
                mode="raw",
                notes=notes,
            )
            self._persisted_run_id = exp_id
            logger.info("Live SR run persisted as experiment id=%d (%s)",
                        exp_id, self._run_name)
        except Exception as e:
            logger.error("Failed to persist live SR run: %s", e)
            return

        # Per-worker rows so the /reports drill-down has its row table.
        rows = []
        for wid in sorted(per_worker.keys()):
            stats = per_worker[wid]
            reg = self.state.get_registered(wid) if self.state else None
            window = self._per_worker_window.get(
                wid, (self._run_started_ms, finished_ms)
            )
            chip = (self.state.get_chip_for_worker(wid)
                    if self.state else None)
            rows.append({
                "experiment_id": exp_id,
                "worker_id": wid,
                "serial":     reg.serial if reg else "",
                "identifier": reg.hardware_identifier if reg else "",
                "engine":     reg.engine or "" if reg else "",
                "i2c_address": chip,
                "started_ms":  int(window[0]),
                "finished_ms": int(window[1]),
                "dispatched":  int(stats["dispatched"]),
                "succeeded":   int(stats["succeeded"]),
                "failed":      int(stats["failed"]),
                "avg_latency_ms": float(stats["avg_latency_ms"]),
            })
        try:
            self.experiment.db.insert_experiment_workers(rows)
        except Exception as e:
            logger.error("Failed to persist live SR per-worker rows: %s", e)

    async def _close_record_writer(self) -> None:
        async with self._record_lock:
            if self._record_writer is not None:
                try:
                    self._record_writer.release()
                    logger.info(
                        "Recording closed: %s (%d frames)",
                        self._record_path, self._record_frames_written,
                    )
                except Exception as e:
                    logger.warning("Error closing video writer: %s", e)
                self._record_writer = None
            self._record_buffer.clear()

    # =========================================================================
    # Core loop
    # =========================================================================
    async def _run(self, run_id: int) -> None:
        """Pull frames from the video, dispatch in parallel, update _latest.

        ``run_id`` is the generation counter snapshot taken at start() time —
        propagates into every _process_frame task so stale callbacks from a
        previous run can be safely discarded.
        """
        try:
            import cv2  # heavy import — only when the live page actually starts
        except ImportError as e:
            self._last_error = f"OpenCV missing: {e}"
            return

        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            self._last_error = f"Could not open video {self._video_path}"
            return

        # Pull the source video's fps + frame count for record mode. We use
        # the source fps as the output mp4's playback rate, so the recorded
        # video has the same nominal duration as the input even if the
        # cluster processes faster/slower than 1× realtime.
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
        # Source resolution — surfaced on the /live page so the labels
        # show the real dims of THIS video (not a hardcoded sample size).
        self._source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
        self._source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
        self._record_fps = float(src_fps)
        self._record_total_frames = src_total
        if self._record_enabled:
            logger.info("Pipeline starting in RECORD mode → %s "
                        "(src=%dfps, %s frames)",
                        self._record_path, src_fps,
                        src_total or "unknown")

        # In record mode we don't want to loop the source — we want to
        # produce exactly one output mp4 of the same length and stop.
        loop_video = not self._record_enabled

        # Each in-flight task increments this; release on completion.
        sem: Optional[asyncio.Semaphore] = None

        def _resize_inflight(target: int) -> asyncio.Semaphore:
            # We can't hot-resize a Semaphore; recreate when the active subset
            # changes size. Tasks already holding permits drain naturally.
            return asyncio.Semaphore(max(1, target))

        last_subset_size = len(self._active_workers)
        sem = _resize_inflight(last_subset_size)

        worker_cursor = 0
        frame_id = 0

        # Pre-encode the source frames lazily; the demo videos are small so
        # decoding is cheap, no need for a pre-buffer thread.
        try:
            while not self._stop_event.is_set():
                # Adapt the inflight cap whenever the user changes worker count.
                cur_size = len(self._active_workers)
                if cur_size == 0:
                    await asyncio.sleep(0.1)
                    continue
                if cur_size != last_subset_size:
                    sem = _resize_inflight(cur_size)
                    last_subset_size = cur_size

                ok, frame = cap.read()
                if not ok:
                    if loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                # Encode original frame to JPEG (this is what we send to the
                # worker AND show on the left side of the UI).
                ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok2:
                    continue
                orig_jpeg = bytes(buf)

                # Pick next worker (round-robin within the active subset).
                subset = list(self._active_workers)
                if not subset:
                    await asyncio.sleep(0.05)
                    continue
                wid = subset[worker_cursor % len(subset)]
                worker_cursor += 1

                await sem.acquire()
                asyncio.create_task(
                    self._process_frame(run_id, frame_id, wid, orig_jpeg, sem)
                )
                frame_id += 1

                # Yield to the loop so dispatched tasks can actually run.
                await asyncio.sleep(0)
        finally:
            cap.release()
            # Close the cv2.VideoWriter HERE, not just in stop(). The
            # mp4 container's `moov` atom (codec table, frame timing) is
            # written ONLY by VideoWriter.release(); without it the file
            # has bytes in `mdat` but no player can decode it (the user
            # downloads "sr_output.mp4" and gets ftyp+free+mdat with no
            # moov => unplayable). When the video ends naturally in
            # record mode the user often doesn't click Stop, so we must
            # finalise the writer ourselves the moment the dispatch
            # loop exits.
            try:
                await self._close_record_writer()
            except Exception as e:
                logger.error("VideoWriter close on EOF failed: %s", e)
            # Stamp finish time + persist NOW. If the user clicks Stop
            # later, _persist_run() is idempotent (no-op if already
            # persisted). This handles the natural-EOF case (record mode
            # ran the video to completion, nobody clicks Stop).
            if self._run_finished_ms is None:
                self._run_finished_ms = int(time.time() * 1000)
            try:
                self._persist_run()
            except Exception as e:
                logger.error("auto-persist of live SR run failed: %s", e)

    def _timeout_for_worker(self, wid: int) -> float:
        """Pick an HTTP timeout based on what's ACTUALLY running on the
        worker, not on the worker's hardware capability.

        Why this matters: ``reg.engine`` is set from the worker's heartbeat
        and reports its hardware ("hailo" if the Pi has a Hailo card).
        But you can run an ``.onnx`` model on a Hailo-equipped Pi (CPU
        fallback), in which case ``reg.engine == "hailo"`` is misleading
        — the actual inference is on CPU, slow, and a 15 s Hailo timeout
        would kill every frame.

        The cluster state's distribution table records which file was
        pushed for this (model, worker) pair — ``.onnx`` => CPU, ``.hef``
        => Hailo NPU. That's the source of truth for "how fast should
        I expect this to be".
        """
        # 1) Prefer the actual distributed backend.
        if self._model_name and self.state is not None:
            dist = self.state.get_distribution(self._model_name, wid)
            backend = (getattr(dist, "backend", None) or "").lower() if dist else ""
            if backend == "hailo":
                return self._timeout_hailo_s
            if backend == "onnx":
                return self._timeout_onnx_s

        # 2) Fall back to the worker's reported hardware engine.
        reg = self.state.get_registered(wid) if self.state else None
        engine = (getattr(reg, "engine", None) or "").lower() if reg else ""
        if engine == "hailo":
            return self._timeout_hailo_s
        if engine == "onnx":
            return self._timeout_onnx_s
        return self._timeout_default_s

    async def _process_frame(self, run_id: int, frame_id: int, wid: int,
                             orig_jpeg: bytes, sem: asyncio.Semaphore) -> None:
        t0 = time.monotonic()
        try:
            resp = await self.experiment.single_shot_inference(
                worker_id=wid,
                model_name=self._model_name or "",
                image_bytes=orig_jpeg,
                run_postprocess=True,
                timeout_s=self._timeout_for_worker(wid),
            )
            # Generation guard — discard any callback whose run was already
            # superseded. Stops a stale HTTP completion from polluting the
            # next run's _latest, stats, or last_error.
            if run_id != self._run_id:
                return
            if not resp.get("ok"):
                self._frames_failed += 1
                self._per_worker_failed[wid] = (
                    self._per_worker_failed.get(wid, 0) + 1
                )
                self._last_error = resp.get("error") or "unknown failure"
                return
            result = resp.get("result") or {}
            sr_b64 = result.get("image_b64") if isinstance(result, dict) else None
            if not sr_b64:
                self._frames_failed += 1
                self._per_worker_failed[wid] = (
                    self._per_worker_failed.get(wid, 0) + 1
                )
                self._last_error = "worker returned no image_b64"
                return
            sr_jpeg = base64.b64decode(sr_b64)
            inference_ms = float(resp.get("inference_s", 0.0)) * 1000.0
            e2e_ms = (time.monotonic() - t0) * 1000.0

            # Stash the SR'd dims from the first arriving frame so the
            # /live page can show actual numbers. The adapter reports
            # ``shape = [H, W, 3]``. We only set this once per run; later
            # frames have the same dims (model output size is static).
            if self._output_w is None or self._output_h is None:
                shape = result.get("shape") if isinstance(result, dict) else None
                if isinstance(shape, list) and len(shape) >= 2:
                    self._output_h = int(shape[0]) or None
                    self._output_w = int(shape[1]) or None

            pair = FramePair(
                frame_id=frame_id, orig_jpeg=orig_jpeg, sr_jpeg=sr_jpeg,
                worker_id=wid, inference_ms=inference_ms, e2e_ms=e2e_ms,
            )
            async with self._latest_lock:
                # Re-check generation under the lock — start() also clears
                # _latest under this lock, so this ordering is the only race
                # window we need to close.
                if run_id != self._run_id:
                    return
                # Drop older frames — result-paced playback always shows the
                # most recently completed pair.
                if (self._latest is None
                        or pair.frame_id > self._latest.frame_id):
                    self._latest = pair
            self._frames_processed += 1
            self._processed_ts.append(time.monotonic())
            self._inf_ms_window.append(inference_ms)
            # Persistence-grade stats: full latency history + per-worker
            # rollup so the eventual /reports entry has real percentiles.
            self._all_inf_ms.append(inference_ms)
            self._per_worker_processed[wid] = (
                self._per_worker_processed.get(wid, 0) + 1
            )
            self._per_worker_inf_sum_ms[wid] = (
                self._per_worker_inf_sum_ms.get(wid, 0.0) + inference_ms
            )
            now_ms = int(time.time() * 1000)
            window = self._per_worker_window.get(wid)
            if window is None:
                self._per_worker_window[wid] = (now_ms, now_ms)
            else:
                self._per_worker_window[wid] = (window[0], now_ms)

            # ---- record mode: push into reorder buffer, flush in order ----
            if self._record_enabled:
                await self._record_drain(pair, run_id)
        except Exception as e:
            if run_id != self._run_id:
                return  # stale failure; don't surface as the next run's error
            self._frames_failed += 1
            self._per_worker_failed[wid] = (
                self._per_worker_failed.get(wid, 0) + 1
            )
            self._last_error = str(e)
        finally:
            sem.release()

    async def _record_drain(self, pair: FramePair, run_id: int) -> None:
        """Stash this frame in the reorder buffer, then flush every contiguous
        prefix starting at ``self._record_next_id`` to the cv2.VideoWriter.

        Frames arrive out of order (parallel workers), but a video file must
        be written sequentially. We hold them in a dict keyed by frame_id,
        and on every arrival drain the longest in-order prefix. Memory is
        bounded by the inflight depth (≈ len(active_workers)).
        """
        import cv2
        async with self._record_lock:
            if run_id != self._run_id:
                return
            self._record_buffer[pair.frame_id] = pair

            # Lazily open the writer on the first sequential frame so we
            # know the actual output dimensions.
            while self._record_next_id in self._record_buffer:
                p = self._record_buffer.pop(self._record_next_id)
                try:
                    arr = cv2.imdecode(
                        np.frombuffer(p.sr_jpeg, dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if arr is None:
                        logger.warning("record: frame %d decode failed", p.frame_id)
                        self._record_next_id += 1
                        continue
                    if self._record_writer is None:
                        h, w = arr.shape[:2]
                        # Codec selection — try in order:
                        #
                        #   1. mp4v (MPEG-4 Part 2) — universally
                        #      supported by opencv-python from PyPI;
                        #      always actually writes frames. Plays in
                        #      VLC / mpv / ffplay / Windows Media Player.
                        #      Browsers (HTML5 <video>) won't play it
                        #      but operator downloads + plays locally.
                        #   2. MJPG inside .avi — last-resort universal
                        #      fallback. Each frame is a JPEG so no
                        #      codec-library dependency at all. Changes
                        #      the file extension so the container
                        #      matches the codec.
                        #
                        # avc1 (H.264) is intentionally NOT on the list:
                        # opencv-python from PyPI doesn't ship libx264
                        # (licensing). `cv2.VideoWriter` reports
                        # `isOpened() == True` for avc1 anyway because
                        # cv2 sees ffmpeg present, but every subsequent
                        # `write()` silently no-ops. The result is a
                        # 44-byte mp4 (just `ftyp + free + mdat header`
                        # with no payload, no `moov`) — exactly the
                        # symptom we hit. Skip avc1 and pick a codec
                        # that actually writes data.
                        target = Path(self._record_path)
                        codecs = [
                            ("mp4v", target),
                            ("MJPG", target.with_suffix(".avi")),
                        ]
                        chosen = None
                        for cc, path in codecs:
                            fourcc = cv2.VideoWriter_fourcc(*cc)
                            vw = cv2.VideoWriter(
                                str(path), fourcc, self._record_fps, (w, h),
                            )
                            if vw.isOpened():
                                chosen = (cc, path, vw)
                                break
                            # Release the failed writer so it doesn't leave
                            # a zero-byte file lying around.
                            try:
                                vw.release()
                            except Exception:
                                pass
                            try:
                                if path.exists() and path.stat().st_size == 0:
                                    path.unlink()
                            except Exception:
                                pass

                        if chosen is None:
                            self._last_error = (
                                f"Could not open VideoWriter at "
                                f"{self._record_path} with any codec "
                                f"(mp4v/MJPG both failed — is "
                                f"opencv-python installed?)"
                            )
                            self._record_writer = None
                            self._record_enabled = False
                            return

                        cc, actual_path, vw = chosen
                        self._record_writer = vw
                        self._record_path = actual_path
                        self._record_codec_verified = False
                        logger.info(
                            "record: opened %s @ %dx%d %sfps codec=%s",
                            actual_path, w, h, self._record_fps, cc,
                        )
                    self._record_writer.write(arr)
                    self._record_frames_written += 1

                    # Sanity check on the FIRST frame: did write() actually
                    # put bytes on disk? If after one write the file is
                    # still header-sized (≤ 64 B), the codec is silently
                    # no-op'ing — abandon and fall back. Catches future
                    # cv2 bug-of-the-day codec issues without us having
                    # to chase them in code.
                    if not getattr(self, "_record_codec_verified", False):
                        try:
                            sz = self._record_path.stat().st_size
                        except OSError:
                            sz = 0
                        if sz <= 64:
                            logger.error(
                                "record: codec wrote 0 bytes after first "
                                "frame (file size=%d). Closing this writer "
                                "and disabling record-mode for the run.",
                                sz,
                            )
                            try:
                                self._record_writer.release()
                            except Exception:
                                pass
                            try:
                                self._record_path.unlink()
                            except FileNotFoundError:
                                pass
                            self._record_writer = None
                            self._record_enabled = False
                            self._last_error = (
                                "Recording disabled: codec opened but "
                                "wrote 0 bytes. Reinstall opencv-python "
                                "with proper codec support, or check "
                                "/tmp for permissions."
                            )
                            return
                        self._record_codec_verified = True
                except Exception as e:
                    logger.warning("record: write failed for frame %d: %s",
                                   p.frame_id, e)
                self._record_next_id += 1

            # If source has a known total and we've written it all, signal
            # the dispatcher to stop the run cleanly.
            if (self._record_total_frames is not None
                    and self._record_frames_written >= self._record_total_frames):
                logger.info("record: reached EOF (%d frames written) — stopping",
                            self._record_frames_written)
                self._stop_event.set()
