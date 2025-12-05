from enum import Enum
from pydantic import BaseModel
import hashlib

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

class WorkerIdAssignmentRequest:
    def __init__(self, worker_id: int):
        self.worker_id = worker_id

class WorkerClusterNetworkConfig(BaseModel):
    interface: WorkerClusterNetworkInterface
    subnet: str
    gateway: str

class WorkerNetworkMode(Enum):
    ETHERNET = "ethernet"
    WIFI = "wifi"

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
    data_plane: WorkerNetworkMode # Data interface used by the worker
    data_ip_address: str # Current IP address of the worker data interface

class WorkerRegistration(BaseModel):
    serial: str
    hardware_identifier: str
    control_ip: str
    data_ip: str
    data_plane: WorkerNetworkMode
    timestamp: int
    status: str

def generate_identifier(serial: str) -> str:
    # Generate user-friendly identifier from the hardware serial
    hash_obj = hashlib.md5(serial.encode())
    hash_bytes = hash_obj.digest()
    adj_index = int.from_bytes(hash_bytes[:4]) % len(ADJECTIVES)
    animal_index = int.from_bytes(hash_bytes[4:8]) % len(ANIMALS)
    return f"{ADJECTIVES[adj_index]}-{ANIMALS[animal_index]}"

ANIMALS = [
    "Panda", "Tiger", "Eagle", "Whale", "Bear", "Wolf", "Fox", "Hawk",
    "Deer", "Seal", "Otter", "Lynx", "Owl", "Swan", "Dove", "Wren",
    "Lark", "Robin", "Crane", "Heron", "Raven", "Finch", "Sparrow", "Falcon",
    "Koala", "Lemur", "Bison", "Moose", "Zebra", "Giraffe", "Rhino", "Hippo",
    "Puma", "Jaguar", "Cheetah", "Leopard", "Panther", "Cougar", "Bobcat", "Ocelot",
    "Rabbit", "Hare", "Mouse", "Squirrel", "Beaver", "Badger", "Marten", "Ferret",
    "Dolphin", "Orca", "Shark", "Ray", "Salmon", "Trout", "Bass", "Pike"
]

ADJECTIVES = [
    "Swift", "Brave", "Calm", "Wise", "Quick", "Bright", "Keen", "Bold",
    "Cool", "Warm", "Fast", "Slow", "Kind", "Neat", "Safe", "Pure",
    "Rare", "Vast", "Wild", "Young", "Agile", "Clear", "Crisp", "Dense",
    "Eager", "Fancy", "Fleet", "Fresh", "Giant", "Grand", "Happy", "Jolly",
    "Light", "Lively", "Lucky", "Merry", "Noble", "Proud", "Quiet", "Rapid",
    "Royal", "Sharp", "Smart", "Snowy", "Solid", "Spry", "Stark", "Stout",
    "Sturdy", "Sunny", "Super", "Tidy", "Tiny", "Vivid", "Witty", "Zesty"
]