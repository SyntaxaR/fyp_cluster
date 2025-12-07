import logging
import hashlib

logger = logging.getLogger(__name__)

def get_cpu_serial() -> str:
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('Serial'):
                    return line.split(':')[1].strip()
    except Exception as e:
        logger.warning(f"Failed to read CPU serial number: {e}")
        logger.warning("Use Ethernet MAC address instead as fallback")
        raise e

def generate_identifier(serial: str) -> str:
    # Generate user-friendly identifier from the hardware serial
    hash_obj = hashlib.md5(serial.encode())
    hash_bytes = hash_obj.digest()
    adj_index = int.from_bytes(hash_bytes[:4]) % len(ADJECTIVES)
    animal_index = int.from_bytes(hash_bytes[4:8]) % len(ANIMALS)
    return f"{ADJECTIVES[adj_index]}-{ANIMALS[animal_index]}"

ANIMALS = [
    "Panda", "Tiger", "Eagle", "Whale", "Bear", "Wolf", "Fox", "Hawk",
    "Deer", "Seal", "Otter", "Lynx", "Owl", "Swan", "Crane", "Falcon", 
    "Koala", "Zebra", "Giraffe", "Rhino", "Hippo", "Puma", "Jaguar", "Cheetah", 
    "Leopard", "Rabbit", "Mouse", "Squirrel", "Dolphin", "Shark", "Cat", "Fish",
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