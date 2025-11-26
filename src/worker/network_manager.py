from enum import Enum
from common.network import NetworkManager, InterfaceConfig
from common.model import WorkerHeartbeat, WorkerNetworkMode
import logging
import re
from time import sleep
import requests

logger = logging.getLogger(__name__)



class WorkerNetworkController(NetworkManager):
    def __init__(self, worker_id: int, config: dict[str, any]):
        super().__init__()
        self.worker_id = worker_id
        self.config = config
        # x.x.x.1: Controller/Gateway
        # x.x.x.2-99: DHCP Pool
        # x.x.x.100-199: Workers
        
        if worker_id < -1 or worker_id > 99:
            raise ValueError("worker_id must be between 0 and 99 (IP range: x.x.x.100 - x.x.x.199) or -1 (Unassigned)")

        self.wifi_ssid = self.config['project']['wifi_ssid']
        self.wifi_password = self.config['project']['wifi_password']
        self.ethernet_interface = self.config['worker']['ethernet_interface']
        self.wifi_interface = self.config['worker']['wifi_interface']
        self.ethernet_gateway = f"{self.config['worker']['ethernet_subnet']}1"
        self.wifi_gateway = f"{self.config['worker']['wifi_subnet']}1"
        self.eth_ipv4 = f"{self.config['worker']['ethernet_subnet']}1{"0" if worker_id < 10 else ""}{worker_id}"
        self.wifi_ipv4 = f"{self.config['worker']['wifi_subnet']}1{"0" if worker_id < 10 else ""}{worker_id}"
        self.eth_controller_ipv4 = f"{self.config['worker']['ethernet_subnet']}1"
        self.wifi_controller_ipv4 = f"{self.config['worker']['wifi_subnet']}1"
        self.control_port = self.config['worker']['control_port']
        self.data_port = self.config['worker']['data_port']

    def initialize(self):
        logger.info("Initializing worker network...")

        # Setup control plane & (ethernet-based) data plane
        eth_config = InterfaceConfig(
            use_dhcp=False,
            interface=self.ethernet_interface,
            ip_address_v4=self.eth_ipv4,
            gateway_address_v4=self.ethernet_gateway,
            metric=50
        )

        if self.worker_id == -1:
            eth_config.use_dhcp = True

        self.set_ip(eth_config)
        self.current_mode = WorkerNetworkMode.ETHERNET
        logger.info(f"Worker network initialized on {self.ethernet_interface}: {self.eth_ipv4}/24 via {self.ethernet_gateway}")

    def destroy(self):
        logger.info("Destroying worker network configuration...")
        # Bring down wifi & delete wifi routing rules
        self._clear_wifi_routing()
        self.bring_interface_down(self.wifi_interface)
        # Enable DHCP on ethernet interface
        dhcp_config = InterfaceConfig(
            use_dhcp=True,
            interface=self.ethernet_interface
        )
        self.set_ip(dhcp_config)
        logger.info("Worker network configuration destroyed")

    def use_wifi_dataplane(self, use_wifi_gateway: bool = False):
        logger.info("Switching worker data plane to WiFi...")

        self._connect_wifi(
            ssid=self.wifi_ssid,
            password=self.wifi_password
        )

        wifi_config = InterfaceConfig(
            use_dhcp=False,
            interface=self.wifi_interface,
            ip_address_v4=self.wifi_ipv4,
            gateway_address_v4=self.wifi_gateway if use_wifi_gateway else "",
            metric=100
        )
        self.set_ip(wifi_config)
        self._set_wifi_source_routing()
        self.current_mode = WorkerNetworkMode.WIFI

        logger.info(f"Worker data plane switched to WiFi on {self.wifi_interface}: {self.wifi_ipv4}/24 via {self.wifi_gateway}")

    def use_ethernet_dataplane(self):
        logger.info("Switching worker data plane to Ethernet...")

        self._clear_wifi_routing()
        self.bring_interface_down(self.wifi_interface)

        self.current_mode = WorkerNetworkMode.ETHERNET
        logger.info(f"Worker data plane switched to Ethernet on {self.ethernet_interface}: {self.eth_ipv4}/24 via {self.ethernet_gateway}")

    def _set_wifi_source_routing(self, default_routing: bool = False):
        logger.info("Setting source-based routing for WiFi interface...")
        
        self._ensure_routing_table("wifi_dataplane", 200)
        self.run_command(['sudo', 'ip', 'route', 'add', f'{self.config['worker']['wifi_subnet']}0/24', 'dev', self.wifi_interface, 'src', self.wifi_ipv4, 'table', 'wifi_dataplane'])
        self.run_command(['sudo', 'ip', 'rule', 'add', 'from', self.wifi_ipv4, 'table', 'wifi_dataplane', 'priority', '101'])

        self.current_mode = WorkerNetworkMode.WIFI
        logger.info("Source-based routing for WiFi interface set")
        logger.info(f"  - WiFi IP: {self.wifi_ipv4}")
        logger.info(f"  - WiFi Subnet: {self.config['worker']['wifi_subnet']}0/24")
        logger.info("  - Routing Table: wifi_dataplane (200)")
        self._verify_wifi_routing()

    def _clear_wifi_routing(self):
        logger.info("Clearing WiFi routing table and rules...")
        try:
            self.run_command(['sudo', 'ip', 'route', 'flush', 'table', 'wifi_dataplane'], check=False)
            self.run_command(['sudo', 'ip', 'rule', 'del', 'from', self.wifi_ipv4, 'table', 'wifi_dataplane'], check=False)
            self.set_default_route(self.ethernet_gateway, self.ethernet_interface, metric=50)
            logger.info("WiFi routing table and rules cleared")
        except Exception as e:
            logger.error(f"Failed to clear WiFi routing table and rules: {e}")
            raise e

    def _verify_wifi_routing(self) -> bool:
        try:
            r = requests.get(f"http://{self.wifi_controller_ipv4}:{self.control_port}/test")
            if r.status_code == 200:
                logger.info("WiFi routing verified successfully")
                return True
            else:
                logger.error(f"WiFi routing verification failed, unexpected response code {r.status_code}")
                return False
        except Exception as e:
            logger.error(f"Failed to verify WiFi routing: {e}")
        

    def _ensure_routing_table(self, name: str, table_id: int):
        # Ensure routing table exists
        table_path = "/etc/iproute2/rt_tables"

        try:
            with open(table_path, 'r') as f:
                if f"{table_id} {name}" in f.read():
                    logger.info(f"Routing table {name} already exists")
                    return
        except Exception:
            pass
        logger.info(f"Creating routing table {name} with ID {table_id}")
        self.run_command(['sudo', 'sh', '-c', f'echo "{table_id} {name}" >> {table_path}'], check=True, capture_output=False)
        logger.info(f"Routing table {name} created")

    def _connect_wifi(self, ssid: str, password: str):
        logger.info(f"Connecting to WiFi SSID '{ssid}' on interface {self.wifi_interface}...")
        self.run_command(['sudo', 'nmcli', 'dev', 'wifi', 'connect', ssid, 'password', password, 'ifname', self.wifi_interface], capture_output=True)
        sleep(10)  # Wait for connection to establish
        # Check connection status
        status_output = self.run_command(['nmcli', '-t', '-f', 'ACTIVE,SSID', 'dev', 'wifi'], capture_output=True)
        for line in status_output.splitlines():
            active, connected_ssid = line.split(':')
            if active == 'yes' and connected_ssid == ssid:
                logger.info(f"Successfully connected to WiFi SSID '{ssid}'")
                return
        logger.info(f"Connected to WiFi SSID '{ssid}' on interface {self.wifi_interface}")

    def _verify_data_connectivity(self) -> bool:
        if self.current_mode != WorkerNetworkMode.WIFI and self.current_mode != WorkerNetworkMode.ETHERNET:
            logger.error("Unknown network mode for verifying data connectivity")
            return False
        logger.info("Verifying connectivity to controller...")
        target_ip = self.wifi_controller_ipv4 if self.current_mode == WorkerNetworkMode.WIFI else self.eth_controller_ipv4
        try:
            r = requests.get(f"http://{target_ip}:{self.control_port}/test")
            if r.status_code == 200:
                logger.info("Connectivity to controller verified successfully")
                return True
            else:
                logger.error(f"Connectivity verification failed, unexpected response code {r.status_code}")
                return False
        except Exception as e:
            logger.error(f"Failed to verify connectivity to controller: {e}")
            return False

    def _send_control_heartbeat(self, serial: str, hardware_identifier: str):
        logger.info("Sending heartbeat to controller...")
        heartbeat = WorkerHeartbeat(
            worker_id=self.worker_id,
            serial=serial,
            hardware_identifier=hardware_identifier,
            control_ip_address=self.wifi_ipv4 if self.current_mode == WorkerNetworkMode.WIFI else self.eth_ipv4,
            data_connectivity=self._verify_data_connectivity(),
            data_ip_address=self.wifi_ipv4 if self.current_mode == WorkerNetworkMode.WIFI else self.eth_ipv4,
            data_plane=WorkerNetworkMode.WIFI if self.current_mode == WorkerNetworkMode.WIFI else WorkerNetworkMode.ETHERNET,
        )
        requests.post(f"http://{self.eth_controller_ipv4}:{self.control_port}/heartbeat", json=heartbeat.__dict__, timeout=5)
