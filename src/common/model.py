from enum import Enum
from pydantic import BaseModel

class ResponseStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"

class RequestResponse:
    def __init__(self, request_id: str, message: str):
        self.request_id = request_id
        self.message = message
    def __bool__(self):
        return self.status == ResponseStatus.SUCCESS

class WorkerClusterNetworkInterface(Enum):
    ETHERNET = "eth0"
    WIFI = "wlan0"

class WorkerIdAssignmentRequest(BaseModel):
    worker_id: int # Worker ID to be assigned
    hardware_serial: str # Hardware serial number of the worker

class WorkerClusterNetworkConfig(BaseModel):
    interface: WorkerClusterNetworkInterface
    subnet: str
    gateway: str

class WorkerNetworkMode(Enum):
    ETHERNET = "ethernet"
    WIFI = "wifi"
    UNASSIGNED = "unassigned"

class InterfaceStatus(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    UNAVAILABLE = "unavailable"
    CONNECTING = "connecting"

class WorkerNetworkModeRequest(BaseModel):
    mode: str # "ethernet" or "wifi"

class WorkerHeartbeat(BaseModel):
    worker_id: int # -1: Unassigned, 0-99: Assigned Worker ID
    serial: str # Hardware serial number
    hardware_identifier: str # Generated hardware identifier
    control_ip_address: str # Current IP address of the worker
    data_connectivity: bool # Whether data plane connectivity to controller is verified
    data_plane: str # Data interface used by the worker
    data_ip_address: str # Current IP address of the worker data interface

class WorkerRegistration(BaseModel):
    serial: str
    hardware_identifier: str
    control_ip: str
    data_ip: str
    data_plane: WorkerNetworkMode
    timestamp: int
    status: str
