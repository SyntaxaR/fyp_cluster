from common.network import NetworkManager
from common.model import WorkerHeartbeat, WorkerNetworkMode, InterfaceStatus
import logging
from time import sleep
from pathlib import Path
import subprocess

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

        self.dnsmasq_conf_file = Path('/tmp/dnsmasq-controller.conf')
        self.hostapd_conf_file = Path('/tmp/hostapd-controller.conf')
    
    def initialize(self, initialize_wifi: bool = False):
        logger.info("Initializing controller network...")

        # Disable DNSMASQ & Hostapd if running
        self.run_command(['sudo', 'systemctl', 'stop', 'dnsmasq'], check=False)
        self.run_command(['sudo', 'systemctl', 'stop', 'hostapd'], check=False)

        # Check if ethernet interface is connected
        count = 1
        while self._check_interface_status(self.ethernet_interface) == InterfaceStatus.UNAVAILABLE:
            logger.warning(f"Ethernet interface {self.ethernet_interface} is unavailable, please check your ethernet cable. Retrying in 5 seconds... (Retry: {count}/5)")
            sleep(5)
            count += 1
            if count > 5:
                logger.error("Ethernet interface failed to connect after multiple attempts. Worker initialization failed, aborting...")
                raise ConnectionError("Ethernet interface connection failed")   
        
        # Get ready to setup DHCP server on ethernet interface
        # Set ethernet interface to static IP
        self._configure_ethernet_static_ip()

        # Delete existing DNSMASQ configuration files
        dnsmasq_conf_dir = Path('/etc/dnsmasq.d')
        dnsmasq_conf_dir.mkdir(parents=True, exist_ok=True)
        for conf_file in dnsmasq_conf_dir.glob('*.conf'):
            logger.info(f"Deleting existing DNSMASQ configuration file: {conf_file}")
            conf_file.unlink()
        
        # Configure and start DNSMASQ for DHCP server on ethernet interface
        logger.info("Configuring DNSMASQ for DHCP server on ethernet interface...")
        logger.info(f"Writing DNSMASQ configuration for ethernet interface: {dnsmasq_conf_dir}/controller-ethernet-dhcp.conf")
        dnsmasq_eth_conf = dnsmasq_conf_dir / 'controller-ethernet-dhcp.conf'
        dnsmasq_eth_conf.write_text(self._generate_dnsmasq_ethernet_dhcp_config())

        if initialize_wifi:
            # Configure DNSMASQ for DHCP server on wifi interface
            logger.info("Configuring DNSMASQ for DHCP server on wifi interface...")
            dnsmasq_wifi_conf = dnsmasq_conf_dir / 'controller-wifi-dhcp.conf'
            dnsmasq_wifi_conf.write_text(self._generate_dnsmasq_wifi_dhcp_config())
        
        # Start DNSMASQ service


        logger.info("Starting DNSMASQ service...")
        

    def _configure_ethernet_static_ip(self):
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
            if self._check_interface_status(self.ethernet_interface) != InterfaceStatus.DISCONNECTED:
                logger.error(f"Interface {self.ethernet_interface} is not in 'disconnected' state, cannot proceed to set DHCP")
                raise OSError(f"Interface {self.ethernet_interface} is not 'disconnected' after deleting all NetworkManager connections")
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
        hostapd_conf_dir = Path('/etc/hostapd')
        logger.info(f"Writing Hostapd configuration for wifi interface: {hostapd_conf_dir}/hostapd.conf")
        hostapd_conf_dir.mkdir(parents=True, exist_ok=True)
        hostapd_conf = hostapd_conf_dir / 'hostapd.conf'
        hostapd_conf.write_text(self._generate_hostapd_config())
        self.run_command(['sudo', 'systemctl', 'start', 'hostapd'])

        

    def _generate_dnsmasq_ethernet_dhcp_config(self) -> str:
        return f"""
interface={self.ethernet_interface}
bind-interfaces
dhcp-range={self.config['network']['ethernet_subnet']}2,{self.config['network']['ethernet_subnet']}254,24h
# Subnet Mask
dhcp-option=1,255.255.255.0
# Gateway
dhcp-option=3,{self.ethernet_gateway}
"""
    
    def _generate_dnsmasq_wifi_dhcp_config(self) -> str:
        return f"""
interface={self.wifi_interface}
bind-interfaces
dhcp-range={self.config['network']['wifi_subnet']}2,{self.config['network']['wifi_subnet']}254,24h
# Subnet Mask
dhcp-option=1,255.255.255.0
# Gateway
dhcp-option=3,{self.wifi_gateway}
"""
    
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
channel=0
ieee80211d=1
ieee80211n=1
ieee80211ac=1
wmm_enabled=1
"""