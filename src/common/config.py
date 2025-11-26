import tomllib
import os
import re
import logging

logger = logging.getLogger(__name__)

def load_config() -> dict[str, any]:
    # Load configuration file
    with open(os.path.join(os.path.dirname(__file__), '../..', 'config.toml'), 'rb') as f:
        config = tomllib.load(f)
        
    # Validate and set defaults for config
    if not config['worker'].get('ethernet_subnet') or type(config['worker']['ethernet_subnet']) is not str or not re.match(r"^(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]\d?)\.(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]\d?|0)\.(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]\d?|0)\.$", str(config['worker']['ethernet_subnet'])):
        logger.warning("Ethernet subnet is not defined or invalid in configuration, defaulting to 10.0.100.worker_id")
        logger.warning('Define ethernet_subnet in config.toml for customization (e.g. ethernet_subnet = "10.0.100.")')
        config['worker']['ethernet_subnet'] = "10.0.100."
            
    if not config['worker'].get('wifi_subnet') or type(config['worker']['wifi_subnet']) is not str or not re.match(r"^(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]\d?)\.(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]\d?|0)\.(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]\d?|0)\.$", str(config['worker']['wifi_subnet'])):
        logger.warning("WiFi subnet is not defined or invalid in configuration, defaulting to 10.0.200.worker_id")
        logger.warning('Define wifi_subnet in config.toml for customization (e.g. wifi_subnet = "10.0.200.")')
        config['worker']['wifi_subnet'] = "10.0.200."
            
    if not config['worker'].get('control_port') or type(config['worker']['control_port']) is not int or config['worker']['control_port'] < 1 or config['worker']['control_port'] > 65535:
        logger.warning("Control port is not defined or invalid in configuration, defaulting to 8001")
        config['worker']['control_port'] = 8001
                
    if not config['worker'].get('data_port') or type(config['worker']['data_port']) is not int or config['worker']['data_port'] < 1 or config['worker']['data_port'] > 65535:
        logger.warning("Data port is not defined or invalid in configuration, defaulting to 8002")
        config['worker']['data_port'] = 8002

    if not config['worker'].get('ethernet_interface') or type(config['worker']['ethernet_interface']) is not str:
        logger.warning("Ethernet interface is not defined in configuration, defaulting to eth0")
        config['worker']['ethernet_interface'] = "eth0"

    if not config['worker'].get('wifi_interface') or type(config['worker']['wifi_interface']) is not str:
        logger.warning("WiFi interface is not defined in configuration, defaulting to wlan0")
        config['worker']['wifi_interface'] = "wlan0"

    if not config['project'].get('wifi_ssid') or type(config['project']['wifi_ssid']) is not str:
        logger.warning("WiFi SSID is not defined or invalid in configuration, defaulting to 'ClusterNet'")
        config['project']['wifi_ssid'] = "ClusterNet"

    if not config['project'].get('wifi_password') or type(config['project']['wifi_password']) is not str or len(config['project']['wifi_password']) < 8:
        logger.warning("WiFi password is not defined or invalid (min. 8 characters) in configuration, defaulting to 'Password'")
        config['project']['wifi_password'] = "Password"