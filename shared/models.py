"""
Pydantic and dataclass models, plus shared enums.

Used by both the controller and the worker over HTTP / WebSocket.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from shared.util import generate_identifier


# =============================================================================
# Enums
# =============================================================================
class ResponseStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


@unique
class WorkerClusterNetworkInterface(str, Enum):
    ETHERNET = "eth0"
    WIFI = "wlan0"


@unique
class ConnectionType(str, Enum):
    ETHERNET = "ethernet"
    WIFI = "wifi"
    INVALID = "invalid"


@unique
class InterfaceStatus(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    UNAVAILABLE = "unavailable"
    CONNECTING = "connecting"


@unique
class WorkerStatus(str, Enum):
    PENDING_REGISTRATION = "pending_registration"
    REGISTERED = "registered"
    ACTIVE = "active"
    RECONNECTING = "reconnecting"
    INACTIVE = "inactive"


@unique
class ExperimentStatus(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    DISTRIBUTING = "distributing"
    RUNNING = "running"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"


@unique
class InferenceMode(str, Enum):
    TENSOR = "tensor"
    RAW = "raw"
    DUMMY = "dummy"


# =============================================================================
# Worker control / lifecycle
# =============================================================================
class WorkerHeartbeat(BaseModel):
    """Sent by worker -> controller every `heartbeat_interval` seconds."""
    worker_id: int                  # -1 = unassigned
    serial: str                     # Hardware serial (CPU)
    hardware_identifier: str        # MD5(serial) -> "Adjective-Animal"
    control_ip_address: str
    data_connectivity: bool
    data_plane: ConnectionType
    data_ip_address: str
    timestamp: int
    # Worker tells the controller which ports it actually listens on. In a
    # real cluster these match the cluster-wide config (8001 / 8002), but
    # mock-mode workers may pick unique loopback ports. Optional for backward
    # compatibility — controller falls back to the config defaults if absent.
    control_port: Optional[int] = None
    data_port: Optional[int] = None
    # True if this worker has a Hailo NPU on its PCIe bus (lspci | grep -i
    # Hailo). The controller uses this to pre-pick the engine backend
    # (`onnx` vs `hailo`) when the user clicks Distribute without an
    # explicit override.
    has_hailo: Optional[bool] = None
    # Optional host-stats snapshot (shared.host_stats.collect()). Workers
    # populate this each heartbeat so the Monitor page can display CPU
    # usage / temperatures without polling each worker separately.
    # ``None`` on workers that don't ship host_stats (older deploys).
    cpu_temp_c: Optional[float] = None
    cpu_usage_pct: Optional[float] = None
    npu_temp_c: Optional[float] = None

    model_config = ConfigDict(use_enum_values=True)


class WorkerRegistration(BaseModel):
    serial: str
    hardware_identifier: str
    control_ip: str
    data_ip: str
    data_plane: ConnectionType
    timestamp: int
    status: WorkerStatus
    # Worker's actual listen ports (reported via heartbeat). Defaults match
    # the standard cluster ports for back-compat with older workers.
    control_port: int = 8001
    data_port: int = 8002
    # Optional capacity hint for weighted dispatch (set by experiment manager)
    capacity_weight: float = 1.0
    # Worker self-reported inference engine ("onnx" | "hailo")
    engine: Optional[str] = None
    # Currently loaded model name (None if no model loaded)
    loaded_model: Optional[str] = None
    # Mirrors WorkerHeartbeat.has_hailo — populated from the latest heartbeat
    # so the rest of the controller (Distribute gate, /experiment UI engine
    # auto-pick) can decide based on registration alone.
    has_hailo: Optional[bool] = None
    # Latest host-stats snapshot from this worker (mirrors WorkerHeartbeat).
    # Surfaced on the Monitor page; updated each heartbeat.
    cpu_temp_c: Optional[float] = None
    cpu_usage_pct: Optional[float] = None
    npu_temp_c: Optional[float] = None

    model_config = ConfigDict(use_enum_values=True)


class WorkerIdAssignmentRequest(BaseModel):
    worker_id: int
    hardware_serial: str


class WorkerNetworkModeRequest(BaseModel):
    mode: Literal["ethernet", "wifi"]
    ssid: Optional[str] = None
    password: Optional[str] = None


class WorkerControlInfo:
    """In-memory handle for a worker (controller side, not serialized)."""

    def __init__(self, worker_id: int, control_ip: str, serial: str,
                 identifier: str = "", control_port: int = 8001):
        self.worker_id = worker_id
        self.control_ip = control_ip
        self.serial = serial
        self.identifier = identifier or generate_identifier(serial)
        self.control_port = int(control_port)

    def __eq__(self, value: object) -> bool:
        return str(self) == str(value)

    def __hash__(self) -> int:
        return hash((self.worker_id, self.serial))

    def __str__(self) -> str:
        return (f'Worker{self.worker_id} "{self.identifier}" '
                f'(Serial: {self.serial}, Control IP: {self.control_ip})')

    def __int__(self) -> int:
        return self.worker_id


class ConnectivityTestResponse(BaseModel):
    from_identifier: str
    message: str
    plane: ConnectionType

    model_config = ConfigDict(use_enum_values=True)


class WorkerConfigResponse(BaseModel):
    """Response body of GET /api/get_config (worker bootstrap)."""
    config: dict[str, Any]
    controller_identifier: str
    controller_serial: str


# =============================================================================
# Tensor payload helpers
# =============================================================================
@dataclass
class TensorPayload:
    """Wire-format binary tensor (base64-encoded for JSON transit)."""
    dtype: str          # e.g. "float32", "int64"
    shape: list[int]    # e.g. [1, 3, 416, 416]
    data: str           # base64-encoded raw bytes


def ndarray_to_payload(arr: np.ndarray) -> TensorPayload:
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return TensorPayload(
        dtype=str(arr.dtype),
        shape=list(arr.shape),
        data=base64.b64encode(arr.tobytes()).decode("ascii"),
    )


def payload_to_ndarray(p: TensorPayload | dict) -> np.ndarray:
    if isinstance(p, dict):
        p = TensorPayload(**p)
    raw = base64.b64decode(p.data)
    arr = np.frombuffer(raw, dtype=np.dtype(p.dtype))
    return arr.reshape(p.shape)


def tensorfeed_to_payloads(feed: dict[str, np.ndarray]) -> dict[str, TensorPayload]:
    return {k: ndarray_to_payload(v) for k, v in feed.items()}


def payloads_to_tensorfeed(payloads: dict[str, TensorPayload | dict]) -> dict[str, np.ndarray]:
    return {k: payload_to_ndarray(v) for k, v in payloads.items()}


# =============================================================================
# Inference request / response
# =============================================================================
RawItemType = Literal["image_bytes", "image_path", "image_url", "text"]


@dataclass
class RawItem:
    type: RawItemType
    data: Any                       # bytes (base64 str over the wire), str path, etc.
    mime: Optional[str] = None


class InferenceRequest(BaseModel):
    model: str                      # logical model name on the worker
    mode: InferenceMode

    # Tensor mode
    inputs: Optional[dict[str, TensorPayload]] = None
    # Raw item mode
    items: Optional[list[RawItem]] = None
    # Dummy mode
    dummy_batch_size: Optional[int] = None
    dummy_seed: Optional[int] = None

    run_postprocess: bool = True
    meta: Optional[dict[str, Any]] = None

    # Telemetry: dispatcher's request id
    request_id: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, use_enum_values=True)


class InferenceResponse(BaseModel):
    status: ResponseStatus
    request_id: Optional[str] = None
    worker_id: Optional[int] = None
    # Either tensor outputs (always) or postprocessed result (any JSON value)
    outputs: Optional[dict[str, TensorPayload]] = None
    result: Optional[Any] = None
    # Timing breakdown (seconds)
    preprocess_s: float = 0.0
    inference_s: float = 0.0
    postprocess_s: float = 0.0
    error: Optional[str] = None

    model_config = ConfigDict(use_enum_values=True)


# =============================================================================
# Model distribution
# =============================================================================
class ModelDescriptor(BaseModel):
    """Describes a model file the controller wants to push to a worker."""
    name: str
    filename: str               # basename to save as on worker
    md5: str                    # integrity check
    size_bytes: int
    backend: Literal["onnx", "hailo"]
    adapter_filename: Optional[str] = None   # optional ModelAdapter .py
    adapter_md5: Optional[str] = None


class LoadModelRequest(BaseModel):
    descriptor: ModelDescriptor


class LoadModelResponse(BaseModel):
    status: ResponseStatus
    worker_id: int
    loaded_model: Optional[str] = None
    error: Optional[str] = None

    model_config = ConfigDict(use_enum_values=True)


# =============================================================================
# Power telemetry
# =============================================================================
@dataclass
class PowerSample:
    timestamp_ms: int           # epoch ms
    i2c_address: int
    worker_id: Optional[int]    # None if unassigned to a worker
    shunt_mv: float
    current_a: float
    voltage_v: float
    power_w: float


class PowerStatsResponse(BaseModel):
    address_to_worker_id: dict[int, Optional[int]]
    samples_per_second: float
    last_sample_ms: int


# =============================================================================
# Experiment / dispatcher
# =============================================================================
@dataclass
class ExperimentConfig:
    name: str
    model_name: str
    dispatcher: str = "round_robin"     # "round_robin" | "weighted_round_robin" | "<plugin_stem>"
    mode: InferenceMode = InferenceMode.DUMMY
    duration_s: float = 30.0
    target_qps: Optional[float] = None  # None => as fast as possible
    dummy_batch_size: int = 1
    weights: dict[int, float] = field(default_factory=dict)
    notes: str = ""
    # Empty list => use all currently-ACTIVE workers.
    # Populated => only these worker_ids participate.
    enrolled_workers: list[int] = field(default_factory=list)
    # worker_id -> "onnx" | "hailo". Missing entries => use worker's auto/default.
    engine_overrides: dict[int, str] = field(default_factory=dict)
    # Optional dataset filename (under datasets/ on controller) for raw / tensor mode.
    dataset_filename: Optional[str] = None
    # Optional adapter override; if None, uses descriptor.adapter_filename
    # found at model-load time. Must match a file under adapters/ on controller.
    adapter_filename: Optional[str] = None
    # If True, the worker runs adapter.postprocess() and returns the structured
    # result (e.g. YOLO bounding boxes) instead of raw output tensors.
    run_postprocess: bool = False


@dataclass
class ExperimentResult:
    name: str
    started_ms: int
    finished_ms: int
    status: ExperimentStatus
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_throughput_qps: float
    avg_cluster_power_w: float
    energy_per_request_j: float
    per_worker: dict[int, dict[str, float]] = field(default_factory=dict)
