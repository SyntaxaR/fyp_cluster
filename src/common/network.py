import logging
import subprocess
from time import sleep
from dataclasses import dataclass
from model import InterfaceStatus

logger = logging.getLogger(__name__)

class NetworkManager:
    def __init__(self):
        return

    def run_command(self, cmd: list[str], check: bool = True, capture_output: bool = True, timeout: int = 30) -> str | None:
        try:
            logger.debug(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=check, capture_output=capture_output, timeout=timeout)
            logger.debug(f"Command output: {result.stdout.decode().strip() if result.stdout else 'No Output'}")
            return result.stdout.decode().strip() if capture_output else None
        except subprocess.CalledProcessError as e:
            logger.error(f"Command '{' '.join(cmd)}' failed\nWith error: {e}")
            raise e
        except subprocess.TimeoutExpired as e:
            logger.error(f"Command '{' '.join(cmd)}' timed out")
            raise e

    def ping_test(self, target: str, count: int = 3, timeout: int = 5) -> bool:
        try:        
            logger.info(f"Pinging {target} with {count} packets")
            result = self.run_command(['ping', '-c', str(count), '-W', str(timeout), target], check=False)
            if result and "0%" in result:
                logger.info(f"Ping to {target} successful")
                return True
            else:
                logger.warning(f"Ping to {target} failed")
                return False
        except subprocess.CalledProcessError:
            logger.error(f"Ping command failed")
            return False

    def _check_interface_status(self, interface: str) -> InterfaceStatus:
        logger.info(f"Checking status of interface {interface}...")
        status = self.run_command(['nmcli', '-t', '-f', 'DEVICE,STATE', 'device', 'status'])
        for line in status.split('\n'):
            device, state = line.split(':')
            if device == interface:
                logger.info(f"Found interface status: {interface}:{state}")
                sstate, _ = state.split(' ') if ' ' in state else (state, '')
                if sstate == "connected":
                    return InterfaceStatus.CONNECTED
                elif sstate == "disconnected":
                    return InterfaceStatus.DISCONNECTED
                elif sstate == "unavailable":
                    return InterfaceStatus.UNAVAILABLE
                elif sstate == "connecting":
                    return InterfaceStatus.CONNECTING
                else:
                    raise ValueError(f"Unknown interface state: {state}")
        # Interface not found
        logger.error(f"Interface {interface} not found in the device list!")
        raise ValueError(f"Interface {interface} not found in the device list!")