from common.network import NetworkManager
from common.model import WorkerHeartbeat, WorkerNetworkMode, InterfaceStatus
import logging
from time import sleep
from pathlib import Path
import subprocess
import threading

logger = logging.getLogger(__name__)

class ControllerNetworkManager(NetworkManager):
    def __init__(self, config: dict[str, any]):
        super().__init__()
        self.config = config

        self.ethernet_interface = self.config['worker']['ethernet_interface']
        self.wifi_interface = self.config['worker']['wifi_interface']
        self.wifi_ssid = self.config['network']['wifi_ssid']
        self.wifi_password = self.config['network']['wifi_password']
        self.ethernet_gateway = f"{self.config['network']['ethernet_subnet']}1"
        self.wifi_gateway = f"{self.config['network']['wifi_subnet']}1"
        self.eth_ipv4 = f"{self.config['network']['ethernet_subnet']}1"
        self.wifi_ipv4 = f"{self.config['network']['wifi_subnet']}1"
        self.control_port = self.config['worker']['control_port']
        self.data_port = self.config['worker']['data_port']

        self.dnsmasq_process = None
        self.hostapd_process = None

        self.dnsmasq_conf_file = Path('/tmp/dnsmasq-controller.conf')
        self.hostapd_conf_file = Path('/tmp/hostapd-controller.conf')
    
    def initialize_test_wifi(self):
        logger.info("#TESTING ONLY# Initializing controller network with only wifi AP...")
        # Disable DNSMASQ & Hostapd if running
        self.run_command(['sudo', 'systemctl', 'stop', 'dnsmasq'], check=False)
        self.run_command(['sudo', 'systemctl', 'stop', 'hostapd'], check=False)

        # Kill any existing dnsmasq/hostapd processes
        self.run_command(['sudo', 'pkill', 'dnsmasq'], check=False)
        self.run_command(['sudo', 'pkill', 'hostapd'], check=False)

        logger.info("Configuring DNSMASQ for DHCP server on wifi interface only...")
        logger.info(f"Writing DNSMASQ configuration to /tmp/dnsmasq-controller.conf, wifi: True, eth: False")
        self.dnsmasq_conf_file.write_text(self._generate_dnsmasq_dhcp_config(include_wifi=True, include_eth=False))
        self._configure_wifi_ap()

        print("Launching DNSMASQ...")
        # Launch DNSMASQ
        try:
            self._start_dnsmasq()
        except Exception as e:
            logger.error(f"Failed to start DNSMASQ service: {e}")
            raise RuntimeError(f"Failed to start DNSMASQ service: {e}")
        
        print("Initialization complete (TESTING CONFIGURATION!!!).")

    def initialize(self, initialize_wifi: bool = False):
        logger.info("Initializing controller network...")
        
        print("Initializing controller network...")
        print("Clearing existing network services...")
        # Disable DNSMASQ & Hostapd if running
        self.run_command(['sudo', 'systemctl', 'stop', 'dnsmasq'], check=False)
        self.run_command(['sudo', 'systemctl', 'stop', 'hostapd'], check=False)

        # Kill any existing dnsmasq/hostapd processes
        self.run_command(['sudo', 'pkill', 'dnsmasq'], check=False)
        self.run_command(['sudo', 'pkill', 'hostapd'], check=False)

        # Check if ethernet interface is connected
        count = 1
        while self._check_interface_status(self.ethernet_interface) == InterfaceStatus.UNAVAILABLE:
            logger.warning(f"Ethernet interface {self.ethernet_interface} is unavailable, please check the ethernet cable/interface. Retrying in 5 seconds... (Retry: {count}/5)")
            sleep(5)
            count += 1
            if count > 5:
                logger.error("Ethernet interface failed to connect after multiple attempts. Worker initialization failed, aborting...")
                raise ConnectionError("Ethernet interface connection failed")   
        
        # Get ready to setup DHCP server on ethernet interface
        # Set ethernet interface to static IP
        self._configure_ethernet_static_ip()
        
        # Configure and start DNSMASQ for DHCP server on ethernet interface
        logger.info("Configuring DNSMASQ for DHCP server on ethernet interface...")
        logger.info(f"Writing DNSMASQ configuration to /tmp/dnsmasq-controller.conf, wifi: {initialize_wifi}")
        self.dnsmasq_conf_file.write_text(self._generate_dnsmasq_dhcp_config(initialize_wifi))

        if initialize_wifi:
            self._configure_wifi_ap()            

        # Launch DNSMASQ
        print("Launching DNSMASQ...")
        try:
            self._start_dnsmasq()
        except Exception as e:
            logger.error(f"Failed to start DNSMASQ service: {e}")
            raise RuntimeError(f"Failed to start DNSMASQ service: {e}")
        
        print("Initialization complete.")
        
    def _start_hostapd(self):
        print("Launching Hostapd...")
        logger.info("Starting Hostapd service...")
        if self.hostapd_process:
            logger.info("Hostapd service is already running when attempting to start it again!")
            raise RuntimeError("Hostapd service is already running when attempting to start it again!")
        self.hostapd_process = subprocess.Popen(['sudo', 'hostapd', '/tmp/hostapd-controller.conf'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        sleep(2) # Wait for service to start
        if self.hostapd_process.poll() is not None:
            stderr = self.hostapd_process.stderr.read()
            logger.error(f"Hostapd service failed to start:\n{stderr}")
            raise RuntimeError(f"Hostapd service failed to start: {stderr}")
        logger.info(f"Hostapd service started successfully, pid: {self.hostapd_process.pid}")
        self._monitor_process(self.hostapd_process, "hostapd")

    def _start_dnsmasq(self):
        logger.info("Starting DNSMASQ service...")
        if self.dnsmasq_process:
            logger.info("DNSMASQ service is already running when attempting to start it again!")
            raise RuntimeError("DNSMASQ service is already running when attempting to start it again!")
        self.dnsmasq_process = subprocess.Popen(['sudo', 'dnsmasq', '--no-daemon', '--conf-file=/tmp/dnsmasq-controller.conf', '--log-facility=-'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        sleep(2) # Wait for service to start
        if self.dnsmasq_process.poll() is not None:
            stderr = self.dnsmasq_process.stderr.read()
            logger.error(f"DNSMASQ service failed to start:\n{stderr}")
            raise ConnectionError(f"DNSMASQ service failed to start: {stderr}")
        logger.info(f"DNSMASQ service started successfully, pid: {self.dnsmasq_process.pid}")
        self._monitor_process(self.dnsmasq_process, "dnsmasq")

    def _monitor_process(self, process: subprocess.Popen, name: str):
        def read_output(pipe, prefix):
            try:
                for line in iter(pipe.readline, ''):
                    if line:
                        logger.debug(f"[{prefix}] {line.strip()}")
            except Exception as e:
                logger.warning(f"Error reading {prefix} output: {e}")
        if process.stdout:
            stdout_thread = threading.Thread(target=read_output, args=(process.stdout, f"{name}--stdout"), daemon=True)
            stdout_thread.start()
        if process.stderr:
            stderr_thread = threading.Thread(target=read_output, args=(process.stderr, f"{name}--stderr"), daemon=True)
            stderr_thread.start()

    def _check_subprocess_health(self) -> bool:
        if self.dnsmasq_process and self.dnsmasq_process.poll() is not None:
            logger.error("DNSMASQ process has terminated unexpectedly")
            return False
        if self.hostapd_process and self.hostapd_process.poll() is not None:
            logger.error("Hostapd process has terminated unexpectedly")
            return False
        return True

    def _configure_ethernet_static_ip(self):
        print(f"Configuring Ethernet interface {self.ethernet_interface} to static IP {self.eth_ipv4}...")
        logger.info(f"Configuring {self.ethernet_interface} to static IP {self.eth_ipv4}...")
        # Get all NetworkManager connections with the interface
        connection = self.run_command(['nmcli', '-g', 'GENERAL.CONNECTION', 'device', 'show', self.ethernet_interface])
        # Delete any existing connections
        for conn in connection.split('\n'): 
            self.run_command(['nmcli', 'connection', 'delete', conn], check=False)
        sleep(1)
        # Verify if interface is disconnected
        if self._check_interface_status(self.ethernet_interface) != InterfaceStatus.DISCONNECTED:
            sleep(2)
            check_status = self._check_interface_status(self.ethernet_interface)
            if check_status != InterfaceStatus.DISCONNECTED:
                logger.error(f"Interface {self.ethernet_interface} is not in 'disconnected' ({check_status} instead) state, cannot proceed to set DHCP")
                raise OSError(f"Interface {self.ethernet_interface} is not 'disconnected' ({check_status} instead) after deleting all NetworkManager connections")
        # Create new static IP connection
        self.run_command(['nmcli', 'connection', 'add', 'type', 'ethernet', 'ifname', self.ethernet_interface, 'con-name', f'{self.ethernet_interface}-controller-static', 'ipv4.method', 'manual', 'ipv4.addresses', f'{self.eth_ipv4}/24', 'ipv4.gateway', "", 'ipv4.dns', "", 'ipv6.method', 'disable'])
        self.run_command(['nmcli', 'connection', 'up', f'{self.ethernet_interface}-controller-static'])
    
    def _configure_wifi_ap(self):
        logger.info(f"Configuring {self.wifi_interface} to static IP {self.wifi_ipv4}...")

        # # Get all NetworkManager connections with the interface
        # connection = self.run_command(['nmcli', '-g', 'GENERAL.CONNECTION', 'device', 'show', self.wifi_interface])
        # # Delete any existing connections
        # for conn in connection.split('\n'): 
        #     self.run_command(['nmcli', 'connection', 'delete', conn], check=False)
        # sleep(1)
        # # Verify if interface is disconnected
        # if self._check_interface_status(self.wifi_interface) != InterfaceStatus.DISCONNECTED:
        #     sleep(2)
        #     if self._check_interface_status(self.wifi_interface) != InterfaceStatus.DISCONNECTED:
        #         logger.error(f"Interface {self.wifi_interface} is not in 'disconnected' state, cannot proceed to set DHCP")
        #         raise OSError(f"Interface {self.wifi_interface} is not 'disconnected' after deleting all NetworkManager connections")
        
        # Disable NetworkManager control over wifi interface
        nm_conf_dir = Path('/etc/NetworkManager/conf.d')
        if not nm_conf_dir.exists():
            raise FileNotFoundError(f"NetworkManager configuration directory not found: {nm_conf_dir}! The system network may not be managed by NetworkManager and thus incompatible!")
        for conf_file in nm_conf_dir.glob('*-controller-unmanaged.conf'):
            logger.info(f"Deleting existing NetworkManager unmanaged configuration file: {conf_file}")
            conf_file.unlink()
        nm_conf_file = nm_conf_dir / f'{self.wifi_interface}-controller-unmanaged.conf'
        logger.info(f"Writing NetworkManager configuration to unmanaged wifi interface: {nm_conf_file}")
        nm_conf_file.write_text(f"[keyfile]\nunmanaged-devices=interface-name:{self.wifi_interface}\n")
        logger.info("Reloading NetworkManager...")
        self.run_command(['sudo', 'systemctl', 'restart', 'NetworkManager'])
        sleep(1)

        # Set static IP for wifi interface
        logger.info(f"Setting static IP {self.wifi_ipv4} for wifi interface {self.wifi_interface}...")
        self.run_command(['ip', 'addr', 'flush', 'dev', self.wifi_interface])
        self.run_command(['sudo', 'ip', 'addr', 'add', f'{self.wifi_ipv4}/24', 'dev', self.wifi_interface])
        self.run_command(['sudo', 'ip', 'link', 'set', self.wifi_interface, 'up'])
        sleep(1)

        # Use hostapd to create wifi AP
        logger.info(f"Setting up Hostapd to create wifi AP on interface {self.wifi_interface}...")
        self.hostapd_conf_file.write_text(self._generate_hostapd_config())
        logger.info("Starting Hostapd service for wifi AP...")
        self._start_hostapd()

    def _generate_dnsmasq_dhcp_config(self, include_wifi: bool, include_eth: bool=True) -> str:
        if not include_eth and not include_wifi:
            raise ValueError("At least one of include_eth or include_wifi must be True to generate DNSMASQ configuration")
        config_text = f"""
# AUTO-GENERATED TEMPORARY CONFIGURATION FILE

# BASIC SETTINGS
domain-needed
bogus-priv
no-resolv
no-poll
bind-interfaces

# Lease file location
dhcp-leasefile=/tmp/dnsmasq-controller.leases

# Logging
log-dhcp
log-queries
"""
        if include_eth:
            config_text += f"""
# {self.ethernet_interface}: Ethernet DHCP Configuration, static IP managed by NetworkManager
interface={self.ethernet_interface}
dhcp-range=interface:{self.ethernet_interface},{self.config['network']['ethernet_subnet']}5,{self.config['network']['ethernet_subnet']}254,24h
dhcp-option=interface:{self.ethernet_interface},1,255.255.255.0
dhcp-option=interface:{self.ethernet_interface},3,{self.ethernet_gateway}
dhcp-option=interface:{self.ethernet_interface},6,{self.ethernet_gateway}
"""
        if include_wifi:
            config_text += f"""
# {self.wifi_interface}: WiFi DHCP Configuration, static IP managed by ip command
interface={self.wifi_interface}
dhcp-range=interface:{self.wifi_interface},{self.config['network']['wifi_subnet']}5,{self.config['network']['wifi_subnet']}254,24h
dhcp-option=interface:{self.wifi_interface},1,255.255.255.0
dhcp-option=interface:{self.wifi_interface},3,{self.wifi_gateway}
dhcp-option=interface:{self.wifi_interface},6,{self.wifi_gateway}
"""
        return config_text

    def _generate_hostapd_config(self) -> str:
        return f"""
interface={self.wifi_interface}
driver=nl80211
ssid={self.wifi_ssid}
wpa_passphrase={self.wifi_password}
wpa=2
wpa_key_mgmt=WPA-PSK
auth_algs=1
ignore_broadcast_ssid=0
macaddr_acl=0
rsn_pairwise=CCMP
hw_mode=a
channel=40
ieee80211n=0
ieee80211ac=1
wmm_enabled=1
"""