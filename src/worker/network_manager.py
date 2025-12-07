from common.network import NetworkManager
from common.model import WorkerHeartbeat, ConnectionType, InterfaceStatus, ConnectivityTestResponse
import logging
from time import sleep, time
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
        self.wifi_interface = self.config['worker']['wifi_interface']
        self.eth_controller_ipv4 = f"{self.config['network']['ethernet_subnet']}1"
        self.wifi_controller_ipv4 = f"{self.config['network']['wifi_subnet']}1"
        self.control_port = self.config['controller']['control_port']
        self.data_port = self.config['controller']['data_port']

        self.eth_ipv4 = None
        self.wifi_ipv4 = None
        self.current_mode: ConnectionType = ConnectionType.INVALID

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
        self.current_mode = ConnectionType.ETHERNET
        print(f"Worker network initialized in {self.current_mode.value} mode")

    def switch_to_ethernet(self):
        if self.current_mode == ConnectionType.ETHERNET:
            logger.info("Already in Ethernet mode, no switch needed")
            print("Requested switch to Ethernet mode, but already in Ethernet mode")
            return
        elif self.current_mode == ConnectionType.WIFI:
            logger.info("Switching to Ethernet connection mode...")
            self.disable_wifi_interface(self.wifi_interface)
        else:
            raise RuntimeError("Cannot switch to Ethernet mode from invalid current network mode")
        # Ethernet should already be configured via DHCP during initialization, just disable wifi
        self.current_mode = ConnectionType.ETHERNET
        self.wifi_ipv4 = None
        logger.info("Switched to Ethernet connection mode")
        print("Switched to Ethernet connection mode")
    
    def switch_to_wifi(self, ssid: str, password: str):
        if self.current_mode == ConnectionType.WIFI:
            logger.info("Already in WiFi mode, no switch needed")
            print("Requested switch to WiFi mode, but already in WiFi mode")
            return
        logger.info("Switching to WiFi connection mode...")
        # Enable and connect to WiFi
        self.enable_wifi_interface(self.wifi_interface, ssid, password)
        logger.info("Switched to WiFi connection mode")
        print("Switched to WiFi connection mode")
    
    def enable_wifi_interface(self, interface: str, ssid: str, password: str):
        logger.info(f"Enabling WiFi interface {interface} and connecting to SSID '{ssid}'...")
        self.run_command(['nmcli', 'radio', 'wifi', 'on'])
        sleep(3) # Wait for wifi scan to complete
        # Connect to specified SSID
        self.run_command(['nmcli', 'device', 'wifi', 'connect', ssid, 'password', password, 'ifname', interface])
        sleep(3)
        # Verify connection
        status = self._check_interface_status(interface)
        if status != InterfaceStatus.CONNECTED:
            logger.error(f"Failed to connect WiFi interface {interface} to SSID '{ssid}'")
            print(f"Failed to connect WiFi interface {interface} to SSID '{ssid}'")
            raise ConnectionError(f"WiFi interface {interface} failed to connect to SSID '{ssid}'")
        # Get assigned IP address
        result = self.run_command(['ip', '-4', 'addr', 'show', self.wifi_interface])
        ip_address = None
        for line in result.splitlines():
            if 'inet ' in line:
                self.wifi_ipv4 = line.strip().split(' ')[1].split('/')[0]
                break
        if not ip_address:
            logger.error(f"Failed to obtain IP address for WiFi interface {interface} after connection")
            print(f"Failed to obtain IP address for WiFi interface {interface} after connection")
            raise ConnectionError(f"WiFi interface {interface} has no assigned IP address after connection")
        self.wifi_ipv4 = ip_address
        self.current_mode = ConnectionType.WIFI
        logger.info(f"WiFi interface {interface} connected with IP address {self.wifi_ipv4}")
        print(f"WiFi interface {interface} connected with IP address {self.wifi_ipv4}")

    def disable_wifi_interface(self, interface: str):
        logger.info(f"Disabling WiFi interface {interface}...")
        self.run_command(['nmcli', 'radio', 'wifi', 'off'])
        sleep(2)
        if self._check_interface_status(interface) != InterfaceStatus.DISCONNECTED:
            logger.warning(f"WiFi interface {interface} is not disconnected after disabling WiFi radio")
        logger.info("Switched to Ethernet connection mode")
        print("Wifi interface disabled")

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
        self.eth_ipv4 = self._wait_for_eth_dhcp_ip()
    
    def _wait_for_eth_dhcp_ip(self) -> str:
        logger.info("Waiting for DHCP to assign Ethernet IP address...")
        print("Waiting for DHCP to assign Ethernet IP address...")
        start_time = time()
        timeout = 30 # Seconds waiting for DHCP assignment
        while time() - start_time < timeout:
            result = self.run_command(['ip', '-4', 'addr', 'show', self.ethernet_interface])
            ip_address = None
            for line in result.splitlines():
                if 'inet ' in line:
                    ip_address = line.strip().split(' ')[1].split('/')[0]
                    break
            if ip_address:
                logger.info(f"Assigned IP address: {ip_address}")
                print(f"Assigned Control Plane Ethernet IP address: {ip_address}")
                if ip_address.find(self.config['network']['ethernet_subnet']) == 0:
                    logger.info("DHCP assigned IP is within expected subnet")
                    return ip_address
                logger.warning(f"DHCP assigned IP {ip_address} is outside expected subnet {self.config['network']['ethernet_subnet']}")
            sleep(3)
        raise TimeoutError("Timed out waiting for DHCP to assign IP address")


    def _verify_data_connectivity(self) -> bool:
        if self.current_mode != ConnectionType.WIFI and self.current_mode != ConnectionType.ETHERNET:
            logger.error("Unknown network mode for verifying data connectivity")
            return False
        logger.info(f"Verifying {self.current_mode.value} data plane connectivity to controller...")
        target_ip = self.wifi_controller_ipv4 if self.current_mode == ConnectionType.WIFI else self.eth_controller_ipv4
        try:
            r = requests.get(f"http://{target_ip}:{self.data_port}/api/connectivity_test")
            if r.status_code == 200:
                logger.info("Control connectivity got status 200, parsing response...")
                response = ConnectivityTestResponse(**r.json())
                logger.info(f'Data plane connectivity to controller "{response.from_identifier}" verified successfully on {response.plane.value} plane')
                return True
            else:
                logger.error(f"Connectivity verification failed, unexpected response code {r.status_code}")
                return False
        except Exception as e:
            logger.error(f"Failed to verify connectivity to controller: {e}")
            return False
    
    # async def connectivity_test(request: Request) -> ConnectivityTestResponse
    def _verify_control_connectivity(self) -> bool:
        logger.info("Verifying control plane connectivity to controller...")
        try:
            r = requests.get(f"http://{self.eth_controller_ipv4}:{self.control_port}/api/connectivity_test")
            # Load response to ConnectivityTestResponse
            if r.status_code == 200:
                logger.info("Control connectivity got status 200, parsing response...")
                response = ConnectivityTestResponse(**r.json())
                logger.info(f'Control plane connectivity to controller "{response.from_identifier}" verified successfully on {response.plane.value} plane')
                return True
            else:
                logger.error(f"Control connectivity verification failed, unexpected response code {r.status_code}")
                return False
        except Exception as e:
            logger.error(f"Failed to verify control plane connectivity to controller: {e}")
            return False

    def _send_control_heartbeat(self, serial: str, hardware_identifier: str) -> bool:
        try:
            logger.info("Sending heartbeat to controller...")
            heartbeat = WorkerHeartbeat(
                worker_id=self.worker_id,
                serial=serial,
                hardware_identifier=hardware_identifier,
                control_ip_address=self.eth_ipv4,
                data_connectivity=self._verify_data_connectivity(),
                data_ip_address=self.wifi_ipv4 if self.current_mode == ConnectionType.WIFI else self.eth_ipv4,
                data_plane=self.current_mode,
                timestamp=int(time())
            )
            r = requests.post(f"http://{self.eth_controller_ipv4}:{self.control_port}/api/heartbeat", json=heartbeat.__dict__, timeout=5)
            if r.status_code == 200:
                logger.info("Heartbeat sent successfully")
                return True
            else:
                logger.error(f"Failed to send heartbeat, status code: {r.status_code}")
                return False
        except Exception as e:
            logger.error(f"Exception occurred while sending heartbeat: {e}")
            return False
    
    def destroy(self):
        print("TODO: Implement worker network cleanup")
