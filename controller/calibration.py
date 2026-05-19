"""
INA226 <-> worker calibration.

Algorithm
---------
1. Sample baseline power on every detected INA226 for ``baseline_s`` seconds
   (median per chip — ignores brief noise spikes).
2. For each ACTIVE worker, in series:
       a. Tell the worker to run a CPU calibration burst for ``burst_s`` seconds.
       b. While the burst is running, sample every chip's median power.
       c. ΔP[chip] = median_burst - median_baseline.
       d. Cool down ``cool_s`` seconds before moving on.
3. Greedy assignment: sort all (worker, chip) pairs by ΔP descending; bind the
   top pair if ΔP exceeds ``min_delta_w``, then strike that worker AND that chip
   from the candidate set, and continue. This naturally handles
   |chips| ≠ |workers|: extra chips end up unbound, extra workers end up
   unbound (and we log a warning).
4. Persist accepted bindings to SQLite keyed by worker serial so they survive
   restarts.

The whole thing runs as an asyncio task on the controller's main loop so the
Web UI stays responsive.
"""
from __future__ import annotations

import asyncio
import logging
import statistics
import time
from typing import Any, Optional

import requests

from controller.cluster_state import ClusterState
from controller.database import Database
from shared.models import WorkerStatus

logger = logging.getLogger(__name__)


class CalibrationManager:
    def __init__(self, config: dict[str, Any], state: ClusterState, db: Database,
                 power_monitor):
        self.config = config
        self.state = state
        self.db = db
        self.power_monitor = power_monitor
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        cal_cfg = config.setdefault("calibration", {})
        self.baseline_s = float(cal_cfg.get("baseline_s", 2.0))
        self.burst_s = float(cal_cfg.get("burst_s", 4.0))
        self.cool_s = float(cal_cfg.get("cool_s", 1.5))
        self.min_delta_w = float(cal_cfg.get("min_delta_w", 0.3))

    # =========================================================================
    # Public API
    # =========================================================================
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def trigger(self) -> None:
        """Start a calibration run. No-op if one is already running."""
        if self.is_running():
            logger.info("Calibration already running, ignoring trigger.")
            return
        self._task = asyncio.create_task(self._run())

    def load_persisted(self) -> None:
        """Restore previously-saved bindings keyed by serial. Called once on
        controller startup AFTER ClusterState is populated, so workers that
        haven't reattached yet just stay unbound until they show up."""
        try:
            persisted = self.db.load_bindings()
        except Exception as e:
            logger.error("Failed to load persisted bindings: %s", e)
            return
        if not persisted:
            return

        restored = 0
        for serial, meta in persisted.items():
            wid = self.state.find_registered_by_serial(serial)
            if wid is None:
                continue
            self.state.set_power_binding(
                worker_id=wid,
                i2c_address=int(meta["i2c_address"]),
                delta_w=float(meta.get("delta_w", 0.0)),
                serial=serial,
                calibrated_ms=int(meta.get("calibrated_ms", 0)),
            )
            restored += 1
        if restored:
            logger.info("Restored %d worker<->chip bindings from DB", restored)

    # =========================================================================
    # Internals
    # =========================================================================
    async def _run(self) -> None:
        async with self._lock:
            self.state.calibration_in_progress = True
            try:
                await self._do_calibration()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("calibration failed: %s", e)
            finally:
                self.state.calibration_in_progress = False
                self.state.calibration_active_worker_id = None
                self.state.touch()

    async def _do_calibration(self) -> None:
        chip_addrs = list(self.power_monitor.chip_addresses())
        self.state.known_chip_addresses = list(chip_addrs)
        if not chip_addrs:
            logger.warning("No INA226 chips detected; calibration aborted.")
            return

        active_wids = [
            wid for wid, reg in self.state.registered_workers.items()
            if reg.status == WorkerStatus.ACTIVE.value
            or reg.status == WorkerStatus.ACTIVE
        ]
        if not active_wids:
            logger.warning("No ACTIVE workers; calibration aborted.")
            return

        logger.info("Calibration: %d chips %s vs %d workers %s "
                    "(baseline=%.1fs, burst=%.1fs, threshold=%.2fW)",
                    len(chip_addrs), [hex(a) for a in chip_addrs],
                    len(active_wids), active_wids,
                    self.baseline_s, self.burst_s, self.min_delta_w)

        # ---- 1. baseline ----
        baseline = await self._collect_phase("baseline", chip_addrs,
                                             self.baseline_s, None)
        for addr in chip_addrs:
            logger.info("  baseline 0x%02X: %.3f W", addr, baseline.get(addr, 0.0))

        # ---- 2. per-worker burst ----
        deltas: dict[int, dict[int, float]] = {}  # worker_id -> {addr: delta}
        for wid in active_wids:
            burst_medians = await self._burst_one_worker(wid, chip_addrs)
            deltas[wid] = {
                addr: burst_medians.get(addr, baseline.get(addr, 0.0))
                      - baseline.get(addr, 0.0)
                for addr in chip_addrs
            }
            for addr in chip_addrs:
                logger.info("  worker %d  chip 0x%02X  Δ=%+.3f W",
                            wid, addr, deltas[wid][addr])
            await asyncio.sleep(self.cool_s)

        # ---- 3. greedy assignment ----
        bindings = self._assign(deltas)

        # ---- 4. persist + publish ----
        self.state.clear_all_power_bindings()
        try:
            self.db.clear_bindings()
        except Exception as e:
            logger.error("clear_bindings failed: %s", e)

        now_ms = int(time.time() * 1000)
        for wid, (addr, delta) in bindings.items():
            reg = self.state.get_registered(wid)
            if reg is None:
                continue
            self.state.set_power_binding(
                worker_id=wid, i2c_address=addr, delta_w=delta,
                serial=reg.serial, calibrated_ms=now_ms,
            )
            try:
                self.db.upsert_binding(reg.serial, addr,
                                       reg.hardware_identifier, delta, now_ms)
            except Exception as e:
                logger.error("upsert_binding failed: %s", e)

        # Diagnostics: which workers/chips ended up unbound?
        unbound_workers = [w for w in active_wids if w not in bindings]
        bound_chips = {addr for addr, _ in bindings.values()}
        unbound_chips = [a for a in chip_addrs if a not in bound_chips]
        if unbound_workers:
            logger.warning("Workers without an INA226 binding: %s "
                           "(no chip's ΔP exceeded %.2f W)",
                           unbound_workers, self.min_delta_w)
        if unbound_chips:
            logger.info("INA226 chips not bound to any worker: %s",
                        [hex(a) for a in unbound_chips])
        logger.info("Calibration complete: %d/%d workers bound.",
                    len(bindings), len(active_wids))

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------
    async def _burst_one_worker(self, wid: int,
                                chip_addrs: list[int]) -> dict[int, float]:
        reg = self.state.get_registered(wid)
        if reg is None:
            return {}
        worker_data_port = int(self.config["worker"]["data_port"])
        url = f"http://{reg.data_ip}:{worker_data_port}/api/calibration_burst"

        # Mock-mode hint: tell MockPowerMonitor which worker is bursting so its
        # synthetic chip can spike accordingly. Real hardware ignores this.
        self.state.calibration_active_worker_id = wid
        burst_started = False
        burst_task: Optional[asyncio.Task] = None
        try:
            # Fire the burst HTTP call; don't await yet — sample chips while it runs.
            burst_task = asyncio.create_task(asyncio.to_thread(
                requests.post, url,
                json={"duration_s": self.burst_s},
                timeout=self.burst_s + 15.0,
            ))
            burst_started = True

            # Give the worker ~250 ms to spin up, then sample for slightly
            # less than the burst window to stay clear of the trailing edge.
            await asyncio.sleep(0.25)
            sample_window = max(0.5, self.burst_s - 0.5)
            medians = await self._collect_phase(
                f"burst worker={wid}", chip_addrs, sample_window, wid
            )
        finally:
            if burst_task is not None:
                try:
                    await burst_task
                except Exception as e:
                    logger.warning("calibration_burst HTTP failed for wid=%d: %s",
                                   wid, e)
            self.state.calibration_active_worker_id = None

        if not burst_started:
            return {}
        return medians

    async def _collect_phase(self, label: str, chip_addrs: list[int],
                             duration_s: float,
                             active_wid: Optional[int]) -> dict[int, float]:
        """Sleep ``duration_s`` then read DB samples for that window, return
        per-chip median power. ``active_wid`` is just for logging."""
        start_ms = int(time.time() * 1000)
        await asyncio.sleep(duration_s)
        end_ms = int(time.time() * 1000)

        out: dict[int, float] = {}
        for addr in chip_addrs:
            try:
                samples = self.db.power_samples_in_range(start_ms, end_ms, addr)
            except Exception as e:
                logger.error("phase %s addr=0x%02X DB read failed: %s",
                             label, addr, e)
                continue
            if not samples:
                logger.debug("phase %s addr=0x%02X: no samples in window",
                             label, addr)
                continue
            out[addr] = statistics.median(s.power_w for s in samples)
        return out

    def _assign(self, deltas: dict[int, dict[int, float]]
                ) -> dict[int, tuple[int, float]]:
        """Greedy: pick the (worker, chip) pair with the largest ΔP exceeding
        the threshold, claim both, repeat. Returns worker_id -> (addr, delta)."""
        flat: list[tuple[float, int, int]] = [
            (delta, wid, addr)
            for wid, by_addr in deltas.items()
            for addr, delta in by_addr.items()
        ]
        flat.sort(key=lambda t: -t[0])

        bound: dict[int, tuple[int, float]] = {}
        used_chips: set[int] = set()
        for delta, wid, addr in flat:
            if delta < self.min_delta_w:
                break  # all remaining deltas are below the noise floor
            if wid in bound or addr in used_chips:
                continue
            bound[wid] = (addr, delta)
            used_chips.add(addr)
        return bound
