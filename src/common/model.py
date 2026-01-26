from enum import Enum, unique
from pydantic import BaseModel
import numpy as np

from common.util import generate_identifier
from typing import Literal, Optional, Any
from dataclasses import dataclass

class ResponseStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"

@unique
class WorkerClusterNetworkInterface(str, Enum):
    ETHERNET: str = "eth0"
    WIFI: str = "wlan0"

class WorkerIdAssignmentRequest(BaseModel):
    worker_id: int # Worker ID to be assigned
    hardware_serial: str # Hardware serial number of the worker

class WorkerClusterNetworkConfig(BaseModel):
    interface: WorkerClusterNetworkInterface
    subnet: str
    gateway: str

@unique
class ConnectionType(str, Enum):
    ETHERNET: str = "ethernet"
    WIFI: str = "wifi"
    INVALID: str = "invalid"

@unique
class InterfaceStatus(str, Enum):
    CONNECTED: str = "connected"
    DISCONNECTED: str = "disconnected"
    UNAVAILABLE: str = "unavailable"
    CONNECTING: str = "connecting"

@unique
class WorkerStatus(str, Enum):
    PENDING_REGISTRATION: str = "pending_registration"
    REGISTERED: str = "registered"
    ACTIVE: str = "active"
    RECONNECTING: str = "reconnecting"
    INACTIVE: str = "inactive"

# WorkerControlInfo class is supposed to be used in controller only
class WorkerControlInfo():
    def __init__(self, worker_id: int, control_ip: str, serial: str, identifier: str = ""):
        self.worker_id = worker_id
        self.control_ip = control_ip
        self.serial = serial
        self.identifier = identifier
        if self.identifier == "":
            self.identifier = generate_identifier(serial)
        else:
            self.identifier = identifier
    
    def __eq__(self, value):
        return str(self) == str(value)
        
    def __str__(self):
        return f'Worker{self.worker_id} "{self.identifier}" (Serial: {self.serial}, Control IP: {self.control_ip})'

    def __int__(self):
        return self.worker_id

class WorkerNetworkModeRequest(BaseModel):
    mode: str # "ethernet" or "wifi"

class WorkerHeartbeat(BaseModel):
    worker_id: int # -1: Unassigned, 0-99: Assigned Worker ID
    serial: str # Hardware serial number
    hardware_identifier: str # Generated hardware identifier
    control_ip_address: str # Current IP address of the worker
    data_connectivity: bool # Whether data plane connectivity to controller is verified
    data_plane: ConnectionType # Data interface used by the worker
    data_ip_address: str # Current IP address of the worker data interface
    timestamp: int # Timestamp of the heartbeat

class WorkerRegistration(BaseModel):
    serial: str
    hardware_identifier: str
    control_ip: str
    data_ip: str
    data_plane: ConnectionType
    timestamp: int
    status: WorkerStatus

class ConnectivityTestResponse(BaseModel):
    from_identifier: str
    message: str
    plane: ConnectionType

"""
Raw Item Schema for Worker Inference
Tensor mode: Controller sends a list of tensors to the worker
Raw Item mode: Controller sends a list of raw items to the worker
Dummy mode: Worker generates random data for inference
"""

# Tensor mode payload
@dataclass
class TensorPayload:
    dtype: str          # e.g. "float32" "int64"...
    shape: list[int]    # e.g. [1, 416, 416, 3]
    data: bytes         # np.ndarray raw bytes

def ndarray_to_payload(arr: np.ndarray) -> TensorPayload:
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return TensorPayload(dtype=str(arr.dtype), shape=list(arr.shape), data=arr.tobytes())


def payload_to_ndarray(p: TensorPayload) -> np.ndarray:
    arr = np.frombuffer(p.data, dtype=np.dtype(p.dtype))
    return arr.reshape(p.shape)


def tensorfeed_to_payloads(feed: dict[str, np.ndarray]) -> dict[str, TensorPayload]:
    return {k: ndarray_to_payload(v) for k, v in feed.items()}


def payloads_to_tensorfeed(payloads: dict[str, TensorPayload]) -> dict[str, np.ndarray]:
    return {k: payload_to_ndarray(v) for k, v in payloads.items()}

# Raw Item mode schema
RawItemType = Literal["image_bytes", "image_path", "text"]

@dataclass
class RawItem:
    type: RawItemType
    data: Any
    mime: Optional[str] = None

@dataclass
class InferenceRequest:
    model: str
    mode: Literal["tensor", "raw", "dummy"]
    # Tensor mode
    inputs: Optional[dict[str, TensorPayload]] = None
    # Raw Item mode
    items: Optional[list[RawItem]] = None
    # Random mode
    dummy_batch_size: Optional[int] = None
    dummy_seed: Optional[int] = None

    run_postprocess: bool = True
    meta: Optional[dict[str, Any]] = None
