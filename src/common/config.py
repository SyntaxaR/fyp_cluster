import tomllib
import os
import re
import logging

logger = logging.getLogger(__name__)

def load_config() -> dict[str, any]:
    # Load configuration file
    with open(os.path.join(os.path.dirname(__file__), '../..', 'config.toml'), 'rb') as f:
        config = tomllib.load(f)
        
    if not config['worker'].get('control_port') or type(config['worker']['control_port']) is not int or config['worker']['control_port'] < 1 or config['worker']['control_port'] > 65535:
        logger.warning("Worker Control port is not defined or invalid in configuration, defaulting to 8001")
        config['worker']['control_port'] = 8001
                
    if not config['worker'].get('data_port') or type(config['worker']['data_port']) is not int or config['worker']['data_port'] < 1 or config['worker']['data_port'] > 65535:
        logger.warning("Worker Data port is not defined or invalid in configuration, defaulting to 8002")
        config['worker']['data_port'] = 8002

    if not config['controller'].get('control_port') or type(config['controller']['control_port']) is not int or config['controller']['control_port'] < 1 or config['controller']['control_port'] > 65535:
        logger.warning("Controller Control port is not defined or invalid in configuration, defaulting to 8001")
        config['controller']['control_port'] = 8001
    
    if not config['controller'].get('data_port') or type(config['controller']['data_port']) is not int or config['controller']['data_port'] < 1 or config['controller']['data_port'] > 65535:
        logger.warning("Controller Data port is not defined or invalid in configuration, defaulting to 8002")
        config['controller']['data_port'] = 8002

    if not config['network'].get('ethernet_subnet') or type(config['network']['ethernet_subnet']) is not str or not re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.$', config['network']['ethernet_subnet']):
        logger.warning("Ethernet subnet is not defined or invalid in configuration, defaulting to 192.168.10.")
        config['network']['ethernet_subnet'] = "192.168.10."
    
    if not config['network'].get('wifi_subnet') or type(config['network']['wifi_subnet']) is not str or not re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.$', config['network']['wifi_subnet']):
        logger.warning("WiFi subnet is not defined or invalid in configuration, defaulting to 192.168.20.")
        config['network']['wifi_subnet'] = "192.168.20."

    if not config['worker'].get('ethernet_interface') or type(config['worker']['ethernet_interface']) is not str:
        logger.warning("Worker Ethernet interface is not defined in configuration, defaulting to eth0")
        config['worker']['ethernet_interface'] = "eth0"

    if not config['worker'].get('wifi_interface') or type(config['worker']['wifi_interface']) is not str:
        logger.warning("Worker WiFi interface is not defined in configuration, defaulting to wlan0")
        config['worker']['wifi_interface'] = "wlan0"
    
    if not config['controller'].get('ethernet_interface') or type(config['controller']['ethernet_interface']) is not str:
        logger.warning("Controller Ethernet interface is not defined in configuration, defaulting to eth0")
        config['controller']['ethernet_interface'] = "eth0"

    if not config['controller'].get('wifi_interface') or type(config['controller']['wifi_interface']) is not str:
        logger.warning("Controller WiFi interface is not defined in configuration, defaulting to wlan0")
        config['controller']['wifi_interface'] = "wlan0"
    
    # [Network]
    # wifi_ssid = "FYP_Cluster_AP"
    # wifi_password = "fyp_cluster_pass"

    if not config['network'].get('wifi_ssid') or type(config['network']['wifi_ssid']) is not str:
        logger.warning("WiFi SSID is not defined or invalid in configuration, defaulting to FYP_Cluster_AP")
        config['network']['wifi_ssid'] = "FYP_Cluster_AP"
    
    if not config['network'].get('wifi_password') or type(config['network']['wifi_password']) is not str or len(config['network']['wifi_password']) < 8:
        logger.warning("WiFi Password is not defined or invalid in configuration, defaulting to fyp_cluster_pass")
        config['network']['wifi_password'] = "fyp_cluster_pass"
    
    return config