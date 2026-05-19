"""
Single source of truth for the controller's view of the cluster.

Holds:
* pending_workers           — Heartbeats from unassigned workers
* registered_workers        — worker_id -> WorkerRegistration
* worker_id_counter         — monotonically increasing assignment counter
* last_heartbeat_ms         — for staleness detection
* current_experiment        — None or active ExperimentConfig/Result
* per_worker_request_stats  — telemetry for the dispatcher / Web UI

Designed to be mutated only from the controller's asyncio event loop. All
HTTP request handlers and the WebSocket manager run on that loop, so no
locks are necessary; the few cross-thread touchpoints (uvicorn workers,
the I2C polling thread) marshal updates back via asyncio queues.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from shared.models import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    WorkerHeartbeat,
    WorkerRegistration,
    WorkerStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class WorkerStats:
    """Per-worker counters maintained by the dispatcher / inference path."""
    requests_dispatched: int = 0
    requests_succeeded: int = 0
    requests_failed: int = 0
    last_latency_ms: float = 0.0
    sum_latency_ms: float = 0.0
    inflight: int = 0


@dataclass
class ModelDistributionState:
    """Per (model, worker) record of the model file's distribution status.

    Lifecycle: pending -> ok | failed. A status of `ok` means the worker has
    downloaded the file, verified MD5, and successfully swapped its engine —
    i.e. the worker is ready to receive inference requests for this model.

    `backend` records which file variant was pushed (`onnx` / `hailo`); if the
    user later flips the engine_override for this worker, the entry is reset
    so the dashboard prompts another Distribute.
    """
    status: str = "pending"     # pending | ok | failed
    backend: Optional[str] = None
    adapter_filename: Optional[str] = None
    md5: Optional[str] = None
    error: Optional[str] = None
    distributed_ms: int = 0


@dataclass
class ClusterState:
    pending_workers: dict[str, WorkerHeartbeat] = field(default_factory=dict)
    registered_workers: dict[int, WorkerRegistration] = field(default_factory=dict)
    worker_id_counter: int = 0

    # Aggregated experiment state
    current_experiment: Optional[ExperimentConfig] = None
    current_experiment_status: ExperimentStatus = ExperimentStatus.IDLE
    last_experiment_result: Optional[ExperimentResult] = None

    # Telemetry for dispatcher / Web UI
    worker_stats: dict[int, WorkerStats] = field(default_factory=dict)

    # INA226 power-monitor bindings, populated by CalibrationManager.
    # power_bindings:        worker_id  -> i2c_address
    # binding_meta:          worker_id  -> {"delta_w", "calibrated_ms", "serial"}
    # known_chip_addresses:  i2c addresses observed in last scan (for the UI)
    power_bindings: dict[int, int] = field(default_factory=dict)
    binding_meta: dict[int, dict] = field(default_factory=dict)
    known_chip_addresses: list[int] = field(default_factory=list)
    calibration_in_progress: bool = False
    calibration_active_worker_id: Optional[int] = None  # mock-mode hint

    # Model distribution status (model_name -> worker_id -> state). Populated
    # by ExperimentManager.distribute_model and read by the experiment-launch
    # gate so we can refuse to start until every enrolled worker's entry is OK.
    model_distribution: dict[str, dict[int, ModelDistributionState]] = field(
        default_factory=dict
    )

    # Async signal so the Web UI / dispatcher can wait for state changes
    change_event: asyncio.Event = field(default_factory=asyncio.Event)

    # =========================================================================
    # Worker bookkeeping
    # =========================================================================
    def upsert_pending(self, hb: WorkerHeartbeat) -> None:
        hb.timestamp = int(time.time())
        self.pending_workers[hb.serial] = hb
        self.touch()

    def remove_pending(self, serial: str) -> None:
        self.pending_workers.pop(serial, None)
        self.touch()

    def assign_worker_id(self) -> int:
        wid = self.worker_id_counter
        self.worker_id_counter += 1
        return wid

    def add_registered(self, wid: int, registration: WorkerRegistration) -> None:
        self.registered_workers[wid] = registration
        self.worker_stats.setdefault(wid, WorkerStats())
        self.touch()

    def get_registered(self, wid: int) -> Optional[WorkerRegistration]:
        return self.registered_workers.get(wid)

    def find_registered_by_serial(self, serial: str) -> Optional[int]:
        for wid, reg in self.registered_workers.items():
            if reg.serial == serial:
                return wid
        return None

    def touch_heartbeat(self, wid: int,
                        heartbeat: Optional[WorkerHeartbeat] = None) -> None:
        """Update the worker's last-seen timestamp.

        If a fresh heartbeat is supplied, also propagate the worker's
        latest self-reported state (data plane / IPs / Hailo presence)
        into the registration so the dispatcher sees the right address
        after a plane switch.
        """
        reg = self.registered_workers.get(wid)
        if reg is None:
            return
        reg.timestamp = int(time.time())
        if heartbeat is not None:
            new_plane = (heartbeat.data_plane
                         if isinstance(heartbeat.data_plane, str)
                         else heartbeat.data_plane.value)
            reg.data_plane = new_plane
            if heartbeat.data_ip_address:
                reg.data_ip = heartbeat.data_ip_address
            if heartbeat.control_ip_address:
                reg.control_ip = heartbeat.control_ip_address
            if heartbeat.has_hailo is not None:
                reg.has_hailo = bool(heartbeat.has_hailo)
                # Pre-fill engine if the worker hasn't explicitly self-
                # reported one yet — saves the user one click on the
                # Distribute page.
                if reg.engine is None:
                    reg.engine = "hailo" if heartbeat.has_hailo else "onnx"
            # Propagate host-stats so the Monitor page sees fresh values.
            # We accept None too (means "this worker version doesn't
            # report it") and just leave the field as-is in that case to
            # avoid blanking out a previously-reported value.
            if getattr(heartbeat, "cpu_temp_c", None) is not None:
                reg.cpu_temp_c = float(heartbeat.cpu_temp_c)
            if getattr(heartbeat, "cpu_usage_pct", None) is not None:
                reg.cpu_usage_pct = float(heartbeat.cpu_usage_pct)
            if getattr(heartbeat, "npu_temp_c", None) is not None:
                reg.npu_temp_c = float(heartbeat.npu_temp_c)
        self.touch()

    def set_status(self, wid: int, status: WorkerStatus) -> None:
        reg = self.registered_workers.get(wid)
        if reg is not None:
            reg.status = status
            self.touch()

    def active_workers(self) -> list[int]:
        return [
            wid for wid, reg in self.registered_workers.items()
            if reg.status == WorkerStatus.ACTIVE.value or reg.status == WorkerStatus.ACTIVE
        ]

    # =========================================================================
    # Power-monitor bindings
    # =========================================================================
    def set_power_binding(self, worker_id: int, i2c_address: int,
                          delta_w: float = 0.0, serial: str = "",
                          calibrated_ms: int = 0) -> None:
        """Bind worker_id <-> chip i2c_address. Existing claims on the chip are
        cleared (a chip can only belong to one worker)."""
        # Drop any other worker that previously owned this chip
        for wid, addr in list(self.power_bindings.items()):
            if addr == i2c_address and wid != worker_id:
                self.power_bindings.pop(wid, None)
                self.binding_meta.pop(wid, None)
        self.power_bindings[worker_id] = i2c_address
        self.binding_meta[worker_id] = {
            "delta_w": float(delta_w),
            "calibrated_ms": int(calibrated_ms),
            "serial": serial,
        }
        self.touch()

    def clear_power_binding(self, worker_id: int) -> None:
        self.power_bindings.pop(worker_id, None)
        self.binding_meta.pop(worker_id, None)
        self.touch()

    def clear_all_power_bindings(self) -> None:
        self.power_bindings.clear()
        self.binding_meta.clear()
        self.touch()

    def get_chip_for_worker(self, worker_id: int) -> Optional[int]:
        return self.power_bindings.get(worker_id)

    def get_worker_for_chip(self, i2c_address: int) -> Optional[int]:
        for wid, addr in self.power_bindings.items():
            if addr == i2c_address:
                return wid
        return None

    # =========================================================================
    # Model distribution tracking
    # =========================================================================
    def set_distribution(self, model_name: str, worker_id: int,
                         state: ModelDistributionState) -> None:
        self.model_distribution.setdefault(model_name, {})[worker_id] = state
        self.touch()

    def get_distribution(self, model_name: str,
                         worker_id: int) -> Optional[ModelDistributionState]:
        return self.model_distribution.get(model_name, {}).get(worker_id)

    def clear_distribution(self, model_name: str) -> None:
        """Drop tracking for a model — call after the file is deleted from disk
        or whenever the controller knows worker-side state is no longer valid."""
        self.model_distribution.pop(model_name, None)
        self.touch()

    def clear_distribution_for_worker(self, worker_id: int) -> None:
        """Forget every model's distribution record for a single worker.
        Called when a worker disconnects so it's required to redistribute."""
        for entries in self.model_distribution.values():
            entries.pop(worker_id, None)
        self.touch()

    def is_distributed(self, model_name: str, worker_id: int,
                       backend: Optional[str] = None,
                       adapter_filename: Optional[str] = None) -> bool:
        """True iff the (model, worker) entry is OK and matches backend/adapter
        (when those are supplied — None means 'don't care')."""
        entry = self.get_distribution(model_name, worker_id)
        if entry is None or entry.status != "ok":
            return False
        if backend is not None and entry.backend != backend:
            return False
        if adapter_filename is not None and entry.adapter_filename != adapter_filename:
            return False
        return True

    def touch(self) -> None:
        self.change_event.set()
        self.change_event.clear()
