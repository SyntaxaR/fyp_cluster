from enum import Enum
from common.network import NetworkManager
from common.model import WorkerHeartbeat, WorkerNetworkMode, InterfaceStatus
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
        
        if worker_id < -1 or worker_id > 254:
            raise ValueError("worker_id must be between 0 and 254 or -1 (Unassigned)")

        self.ethernet_interface = self.config['worker']['ethernet_interface']
        self.wifi_interface = self.config['controller']['wifi_interface']
        self.ethernet_gateway = f"{self.config['controller']['ethernet_subnet']}1"
        self.wifi_gateway = f"{self.config['network']['wifi_subnet']}1"
        self.eth_controller_ipv4 = f"{self.config['network']['ethernet_subnet']}1"
        self.wifi_controller_ipv4 = f"{self.config['network']['wifi_subnet']}1"
        self.control_port = self.config['controller']['control_port']
        self.data_port = self.config['controller']['data_port']

    def initialize(self):
        logger.info("Initializing worker network...")

        # Check if ethernet interface is connected
        count = 1
        while self._check_interface_status(self.ethernet_interface) == InterfaceStatus.UNAVAILABLE:
            logger.warning(f"Ethernet interface {self.ethernet_interface} is unavailable, please check your ethernet cable. Retrying in 5 seconds... (Retry: {count}/5)")
            sleep(5)
            count += 1
            if count > 5:
                logger.error("Ethernet interface failed to connect after multiple attempts. Worker initialization failed, aborting...")
                raise ConnectionError("Ethernet interface connection failed")   
        # Configure Ethernet interface to use DHCP
        self._ethernet_use_dhcp(self.ethernet_interface)

        # Configure Ethernet Interface to use DHCP
        

    def _ethernet_use_dhcp(self, interface: str):
        logger.info(f"Configuring {interface} to use DHCP...")
        # Get all NetworkManager connections with the interface
        connection = self.run_command(['nmcli', '-g', 'GENERAL.CONNECTION', 'device', 'show', interface])
        # Delete any existing connections
        for conn in connection.split('\n'): 
            self.run_command(['nmcli', 'connection', 'delete', conn], check=False)
        sleep(1)
        # Verify if interface is disconnected
        if self._check_interface_status(interface) != InterfaceStatus.DISCONNECTED:
            sleep(2)
            if self._check_interface_status(interface) != InterfaceStatus.DISCONNECTED:
                logger.error(f"Interface {interface} is not in 'disconnected' state, cannot proceed to set DHCP")
                raise OSError(f"Interface {interface} is not 'disconnected' after deleting all NetworkManager connections")
        # Create new DHCP connection
        self.run_command(['nmcli', 'connection', 'add', 'type', 'ethernet', 'ifname', interface, 'con-name', f'{interface}-worker-dhcp', 'ipv4.method', 'auto', 'ipv6.method', 'disable'])
        self.run_command(['nmcli', 'connection', 'up', f'{interface}-worker-dhcp'])


    def _verify_data_connectivity(self) -> bool:
        if self.current_mode != WorkerNetworkMode.WIFI and self.current_mode != WorkerNetworkMode.ETHERNET:
            logger.error("Unknown network mode for verifying data connectivity")
            return False
        logger.info(f"Verifying {self.current_mode.value} data plane connectivity to controller...")
        target_ip = self.wifi_controller_ipv4 if self.current_mode == WorkerNetworkMode.WIFI else self.eth_controller_ipv4
        try:
            r = requests.get(f"http://{target_ip}:{self.data_port}/datatest")
            if r.status_code == 200:
                logger.info("Connectivity to controller verified successfully")
                return True
            else:
                logger.error(f"Connectivity verification failed, unexpected response code {r.status_code}")
                return False
        except Exception as e:
            logger.error(f"Failed to verify connectivity to controller: {e}")
            return False
        
    def _verify_control_connectivity(self) -> bool:
        logger.info("Verifying control plane connectivity to controller...")
        try:
            r = requests.get(f"http://{self.eth_controller_ipv4}:{self.control_port}/test")
            if r.status_code == 200:
                logger.info("Control plane connectivity to controller verified successfully")
                return True
            else:
                logger.error(f"Control plane connectivity verification failed, unexpected response code {r.status_code}")
                return False
        except Exception as e:
            logger.error(f"Failed to verify control plane connectivity to controller: {e}")
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
