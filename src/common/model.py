from enum import Enum
from pydantic import BaseModel
from common.util import generate_identifier

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

class ConnectionType(Enum):
    ETHERNET = "ethernet"
    WIFI = "wifi"
    INVALID = "invalid"

class InterfaceStatus(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    UNAVAILABLE = "unavailable"
    CONNECTING = "connecting"

class WorkerStatus(Enum):
    PENDING_REGISTRATION = "pending_registration"
    REGISTERED = "registered"
    ACTIVE = "active"
    RECONNECTING = "reconnecting"
    INACTIVE = "inactive"

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