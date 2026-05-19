"""
Synthetic power monitor for mock / local-debug mode.

Generates plausible per-chip power samples (baseline + sinusoidal load + small
noise) at the configured poll interval, and writes them to the same SQLite
table as the real INA226 path. Useful for exercising the live power chart on
a development laptop without I2C hardware.
"""
from __future__ import annotations

import logging
import math
import random
import threading
import time
from typing import Any, Optional

from controller.cluster_state import ClusterState
from controller.database import Database
from shared.models import PowerSample

logger = logging.getLogger(__name__)


class MockPowerMonitor:
    """Drop-in stand-in for PowerMonitor that fabricates samples."""

    def __init__(self, config: dict[str, Any], db: Database, state: ClusterState):
        pm_cfg = config["power_monitor"]
        mock_cfg = config.get("mock", {})
        self.db = db
        self.state = state
        self.poll_interval_s = float(pm_cfg["poll_interval_ms"]) / 1000.0

        self.chip_count = int(mock_cfg.get("power_chip_count", 4))
        self.baseline_w = float(mock_cfg.get("power_baseline_w", 3.0))
        self.load_w = float(mock_cfg.get("power_load_w", 2.5))
        self.actual_vbus = float(pm_cfg.get("actual_vbus", 5.07))
        self.start_addr = int(pm_cfg.get("i2c_address_start", 0x40))
        self._addresses = [self.start_addr + i for i in range(self.chip_count)]

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._batch: list[PowerSample] = []
        self._batch_lock = threading.Lock()
        self._t0 = time.monotonic()

    # ------------------------------------------------------------------
    # Public surface (matches PowerMonitor)
    # ------------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="mock-power-monitor", daemon=True
        )
        self._thread.start()
        logger.info("MockPowerMonitor started with %d synthetic chip(s) at %s",
                    self.chip_count, [hex(a) for a in self._addresses])

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._flush_batch(force=True)

    def chip_addresses(self) -> list[int]:
        return list(self._addresses)

    # ------------------------------------------------------------------
    # Synthetic sample generator
    # ------------------------------------------------------------------
    def _synthesize(self, addr: int, t: float) -> tuple[float, float, float, float]:
        # Different phase per chip so the chart isn't flat-lined identically.
        phase = (addr - self.start_addr) * 0.6
        load = max(0.0, self.load_w * (0.5 + 0.5 * math.sin(0.3 * t + phase)))
        noise = random.uniform(-0.15, 0.15)

        # Mock-mode "ground truth" mapping: chip i  -> worker_id i. When
        # CalibrationManager bursts a worker, add a clear power spike on that
        # worker's chip so the calibration algorithm has a strong signal to
        # latch onto. Real hardware doesn't need this hook.
        truth_wid = addr - self.start_addr
        burst_active = (
            self.state.calibration_in_progress
            and self.state.calibration_active_worker_id == truth_wid
        )
        if burst_active:
            load += 3.0  # +3 W spike — well above the 0.15 W noise floor

        power_w = max(0.0, self.baseline_w + load + noise)
        voltage = self.actual_vbus
        current = power_w / voltage
        shunt_mv = current * 10.0  # arbitrary, equivalent to a 10 mΩ shunt
        return shunt_mv, current, voltage, power_w

    def _run(self) -> None:
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            ts_ms = int(time.time() * 1000)
            t = time.monotonic() - self._t0
            for addr in self._addresses:
                shunt_mv, current_a, voltage_v, power_w = self._synthesize(addr, t)
                # Bindings populated by CalibrationManager; pre-calibration
                # samples are tagged worker_id=None.
                worker_id = self.state.get_worker_for_chip(addr)
                sample = PowerSample(
                    timestamp_ms=ts_ms,
                    i2c_address=addr,
                    worker_id=worker_id,
                    shunt_mv=shunt_mv,
                    current_a=current_a,
                    voltage_v=voltage_v,
                    power_w=power_w,
                )
                with self._batch_lock:
                    self._batch.append(sample)

            self._flush_batch()
            next_tick += self.poll_interval_s
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                self._stop_event.wait(timeout=sleep_for)
            else:
                next_tick = time.monotonic()

    def _flush_batch(self, force: bool = False, max_pending: int = 50) -> None:
        with self._batch_lock:
            if not self._batch:
                return
            if not force and len(self._batch) < max_pending:
                return
            to_write = self._batch
            self._batch = []
        try:
            self.db.insert_power_samples(to_write)
        except Exception as e:
            logger.error("Failed to flush mock power samples: %s", e)
            with self._batch_lock:
                self._batch[:0] = to_write
