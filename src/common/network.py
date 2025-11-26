import logging
import subprocess
from time import sleep
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class InterfaceConfig:
    interface: str
    use_dhcp: bool = True
    ip_address_v4: str = ""
    gateway_address_v4: str = ""
    metric: int = 100

class NetworkManager:
    def __init__(self):
        return

    def run_command(self, cmd: list[str], check: bool = True, capture_output: bool = True):
        try:
            result = subprocess.run(cmd, check=check, capture_output=capture_output, timeout=30)
            return result.stdout.decode().strip() if capture_output else None
        except subprocess.CalledProcessError as e:
            logger.error(f"Command '{' '.join(cmd)}' failed\nWith error: {e}")
            raise e
        except subprocess.TimeoutExpired as e:
            logger.error(f"Command '{' '.join(cmd)}' timed out")
            raise e

    def get_interface_status(self, interface: str) -> dict:
        try:
            output = self.run_command(['ip', 'link', 'show', interface])
            interface_up = 'STATE UP' in output.upper()
            ip_output = self.run_command(['ip', '-4', 'addr', 'show', interface])
            ip_address = None
            for line in ip_output.splitlines():
                if 'inet ' in line:
                    ip_address = line.strip().split(' ')[1].split('/')[0]
                    break
            return {'interface': interface, 'exists': True, 'is_up': interface_up, 'ip_address': ip_address}
        except subprocess.CalledProcessError:
            return {'interface': interface, 'exists': False, 'is_up': False, 'ip_address': None}
    
    def bring_interface_up(self, interface: str):
        self.run_command(['sudo', 'ip', 'link', 'set', interface, 'up'])
        sleep(10) # Wait for starting up

    def bring_interface_down(self, interface: str):
        self.run_command(['sudo', 'ip', 'link', 'set', interface, 'down'])
        sleep(10) # Wait for shutting down

    def set_ip(self, config: InterfaceConfig):
        if config.use_dhcp:
            if config.interface.find('wlan') != -1:
                logger.error(f"Cannot use DHCP on WiFi interface {config.interface} as the DHCP is solely intended for the control plane over Ethernet.")
                raise ValueError("DHCP can only be used on Ethernet interfaces for control plane.")
            logger.info(f"Setting {config.interface} to use DHCP")
            self._enable_dhcp(config.interface)

        if not config.ip_address_v4:
            logger.error("IP address must be provided for static IP configuration")
            return
        logger.info(f"Setting static IP {config.ip_address_v4} on {config.interface}")
        self.run_command(['sudo', 'nmcli', 'con', 'mod', f'"{config.interface}"', 'ipv4.method', 'manual', 'ipv4.addresses', f'{config.ip_address_v4}/24', 'ipv4.gateway', config.gateway_address_v4])
        if config.gateway_address_v4 != "":
            self.set_default_route(config.gateway_address_v4, config.interface, config.metric)

    def _enable_dhcp(self, interface: str):
        logger.info(f"Enabling DHCP on {interface}")
        self.run_command(['sudo', 'nmcli', 'con', 'mod', f'"{interface}"', 'ipv4.method', 'auto', 'ipv4.address' '""', 'ipv4.gateway', '""'])
        self.run_command(['sudo', 'nmcli', 'con', 'down', f'"{interface}"'])
        sleep(1)
        self.run_command(['sudo', 'nmcli', 'con', 'up', f'"{interface}"'])
        sleep(4)  # Wait for DHCP to assign IP

    def set_default_route(self, gateway: str, interface: str, metric: int = 100):
        logger.info(f"Adding default route via {gateway} on {interface} with metric {metric}")
        logger.info("1. Clearing existing default routes")
        self.clear_default_routes(interface)
        logger.info("2. Setting up a new default route")
        self.run_command(['sudo', 'ip', 'route', 'add', 'default', 'via', gateway, 'dev', f'"{interface}"', 'metric', str(metric)], check=False)
    
    def clear_default_routes(self, interface: str):
        logger.info(f"Clearing default routes on {interface}")
        self.run_command(['sudo', 'ip', 'route', 'del', 'default', 'dev', f'"{interface}"'], check=False)

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
