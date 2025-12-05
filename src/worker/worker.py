import logging
import os
from common.config import load_config
from network_manager import WorkerNetworkController, WorkerNetworkMode
from common.network import WorkerNetworkMode
from common.model import generate_identifier, WorkerIdAssignmentRequest, WorkerNetworkModeRequest
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import requests
import threading

logger = logging.getLogger(__name__)

class Worker:
    def __init__(self, config: dict[str, any]):
        self.config = config
        self.worker_id = -1
        self.network_controller = None
        self.hardware_serial = None
        self.hardware_identifier = None
        self.stop_heartbeat = threading.Event()
        self.initialized = False

        self.app = FastAPI()
        self._setup_fastapi_routes()

    def _setup_fastapi_routes(self):
        @self.app.post('/worker/assign_id')
        async def assign_worker_id(request: WorkerIdAssignmentRequest):
            logger.info(f"Assigning new Worker ID: {self.worker_id}")
            try:
                if self.worker_id < 0 or self.worker_id > 99:
                    raise HTTPException(status_code=400, detail="worker_id must be between 0 and 99")
                self.stop_heartbeat.set()
                self.worker_id = request.worker_id
                if self.network_controller:
                    self.network_controller.destroy()
                self.network_controller = WorkerNetworkController(self.worker_id, self.config)
                self.network_controller.initialize()
                self._cache_worker_id(self.worker_id)
                return {"status": "success", "worker_id": self.worker_id, "serial": self.hardware_serial, "identifier": self.hardware_identifier}
            except Exception as e:
                logger.error(f"Failed to assign Worker ID: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post('/worker/network_config')
        async def update_network_config(config: WorkerNetworkModeRequest):
            try:
                if not self.network_controller:
                    raise HTTPException(status_code=400, detail="Network controller not initialized")
                if config.mode == WorkerNetworkMode.ETHERNET.value:
                    self.network_controller.use_ethernet_dataplane()
                elif config.mode == WorkerNetworkMode.WIFI.value:
                    self.network_controller.use_wifi_dataplane()
                else:
                    raise HTTPException(status_code=400, detail="Invalid network mode")
                return {"status": "success", "mode": config.mode}
            except Exception as e:
                logger.error(f"Failed to update network configuration: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            
        @self.app.get('/worker/status')
        async def get_worker_status():
            try:
                status = {
                    "worker_id": self.worker_id,
                    "hardware_serial": self.hardware_serial,
                    "hardware_identifier": self.hardware_identifier,
                    "network_mode": self.network_controller.current_mode.value if self.network_controller else "unknown",
                }
                return {"status": "success", "worker_status": status}
            except Exception as e:
                logger.error(f"Failed to get worker status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def start_api_server(self):
        def run():
            uvicorn.run(self.app, host="0.0.0.0", port=self.config['worker']['control_port'], log_level="info")
        api_thread = threading.Thread(target=run, daemon=True)
        logger.info("Starting FastAPI server for worker control API...")
        api_thread.start()
    
    def intitialize(self):
        if self.initialized:
            logger.error("Worker.initialize() is called while Worker is already initialized!")
            raise RuntimeError("Worker.initialize() is called while Worker is already initialized!")
        else:
            self.initialized = True
    
        # Get hardware serial and generate identifier
        self.hardware_serial = self._get_cpu_serial()
        logger.info(f"Worker Hardware Serial: {self.hardware_serial}")
        self.hardware_identifier = generate_identifier(self.hardware_serial)
        logger.info(f"Worker Hardware Identifier: {self.hardware_identifier}")

        # Start FastAPI server
        self.start_api_server()

        # Get intial network setup via DHCP for cached Worker ID conflict checking / new Worker ID assignment
        logger.info("Using DHCP for initial network setup...")
        eth_interface = self.config['worker']['ethernet_interface']
        self.network_controller = WorkerNetworkController(-1, self.config)
        self.network_controller.initialize() # DHCP on ethernet

        # Wait for DHCP to assign IP
        logger.info("Waiting for DHCP to assign IP address...")
        dhcp_ip = self._wait_for_dhcp_ip()
        logger.info(f"DHCP assigned IP address: {dhcp_ip}")

        # Check for cached worker ID
        self.worker_id = self._load_cached_worker_id()
        if self.worker_id != -1:
            logger.info(f"Loaded cached Worker ID: {self.worker_id}")
            self._handle_startup_with_cached_id()
        else:
            logger.info("No valid cached Worker ID found")
            self._handle_startup_initial()
        
    def _handle_startup_with_cached_id(self):
        logger.info(f"Starting worker with cached ID {self.worker_id}...")
        # Check static IP conflict with cached Worker ID
        static_ip = f"{self.config['worker']['ethernet_subnet']}1{"0" if self.worker_id < 10 else ""}{self.worker_id}"
        logger.info(f"Checking for IP conflict on static IP {static_ip}...")
        if self._is_ip_conflict(static_ip):
            logger.error(f"IP conflict detected for static IP {static_ip}. Cannot start worker with cached ID {self.worker_id}.")
            self.worker_id = -1
            self._handle_startup_initial()
            return
        else:
            logger.info(f"No IP conflict detected for static IP {static_ip}. Proceeding with static configuration.")
            assert self.network_controller is not None
            self.network_controller.destroy()  # Clear DHCP config
            self.stop_heartbeat.set()
            self.network_controller = WorkerNetworkController(self.worker_id, self.config)
            self.network_controller.initialize()
            self._start_heartbeat_loop()
            return
    
    def _handle_startup_initial(self):
        logger.info("Starting worker without cached Worker ID...")
        
        # Start sending heartbeats to controller, waiting for Worker ID assignment
        self._start_heartbeat_loop()

    def _wait_for_dhcp_ip(self) -> str:
        start_time = time.time()
        timeout = 60 # Seconds waiting for DHCP assignment
        while time.time() - start_time < timeout:
            result = self.network_controller.run_command(['sudo', 'ip', '-4', 'addr', 'show', self.network_controller.ethernet_interface])
            ip_address = None
            for line in result.splitlines():
                if 'inet ' in line:
                    ip_address = line.strip().split(' ')[1].split('/')[0]
                    break
            if ip_address:
                logger.info(f"Assigned IP address: {ip_address}")
                if ip_address.find(self.config['worker']['ethernet_subnet']) == 0:
                    logger.info("DHCP assigned IP is within expected subnet")
                    return ip_address
                logger.warning(f"DHCP assigned IP {ip_address} is outside expected subnet {self.config['worker']['ethernet_subnet']}")
            time.sleep(3)
        raise TimeoutError("Timed out waiting for DHCP to assign IP address")

    def _start_heartbeat_loop(self):
        self.stop_heartbeat.clear()

        def heartbeat_task():
            while not self.stop_heartbeat.is_set():
                try:
                    logger.info(f"Sending heartbeat to controller {time.time()}")
                    self.network_controller._send_control_heartbeat(self.hardware_serial, self.hardware_identifier)
                    time.sleep(15)  # Send heartbeat every 30 seconds
                except requests.exceptions.ConnectionError:
                    logger.warning("Failed to connect to controller for heartbeat")
                except Exception as e:
                    logger.error(f"Error sending heartbeat: {e}")
                self.stop_heartbeat.wait(15)
            logger.info("Heartbeat loop stopped.")
        
        self.heartbeat_thread = threading.Thread(target=heartbeat_task, daemon=True)
        self.heartbeat_thread.start()
        logger.info("Control heartbeat loop started with interval 15 seconds.")

    def _send_control_heartbeat(self):
        self.network_controller._send_control_heartbeat(self.hardware_serial, self.hardware_identifier)

    def _is_ip_conflict(self, ip_address: str) -> bool:
        import subprocess
        try:
            # Find "100% packet loss" in ping output to determine no conflict
            result = subprocess.run(['ping', '-c', '3', '-W', '3', ip_address], check=True, capture_output=True, text=True)
            if "100% packet loss" in result.stdout:
                logger.info(f"No conflict detected for IP address {ip_address}")
                return False
            else:
                logger.error(f"IP address {ip_address} is already in use!")
                return True
        except subprocess.CalledProcessError:
            logger.info(f"Error pinging IP address: {ip_address}, assuming conflict")
            return True

    def _get_cpu_serial(self) -> str:
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('Serial'):
                        return line.split(':')[1].strip()
        except Exception as e:
            logger.warning(f"Failed to read CPU serial number: {e}")
            logger.warning("Use Ethernet MAC address instead as fallback")
            return self._get_mac_address() 
    
    def _get_mac_address(self, interface: str) -> str:
        try:
            interface = self.config.get('worker', {}).get('ethernet_interface')
            if not interface:
                logger.warning("Ethernet interface not specified in config, defaulting to 'eth0'")
                interface = 'eth0'
            with open(f'/sys/class/net/{interface}/address', 'r') as f:
                mac = f.read().strip().replace(":", "")
                return mac
        except Exception as e:
            logger.error(f"Failed to read MAC address for interface {interface}: {e}")
            raise e
    
    def _load_cached_worker_id(self) -> int:
        cache_file = os.path.join(os.path.dirname(__file__), 'worker_id')
        if not os.path.exists(cache_file):
            return -1
        try:
            with open(cache_file, 'r') as f:
                if f"{self.hardware_serial} " in f.read():
                    f.seek(0)
                    line = f.readline().strip()
                    _, worker_id_str = line.split(' ')
                    if int(worker_id_str) < 0 or int(worker_id_str) > 99:
                        raise ValueError("Cached worker_id is out of valid range (0-99)")
                    return int(worker_id_str)
                else:
                    return -1
        except Exception as e:
            logger.error(f"Failed to read cached worker ID: {e}")
            return -1

    def _cache_worker_id(self, worker_id: int):
        cache_file = os.path.join(os.path.dirname(__file__), 'worker_id')
        # Clear existing & write new cache
        try:
            with open(cache_file, 'w') as f:
                f.write(f"{self.hardware_serial} {worker_id}\n")
            logger.info(f"Cached Worker ID {worker_id} to {cache_file}")
        except Exception as e:
            logger.error(f"Failed to cache Worker ID: {e}")

    def _clear_cached_worker_id(self):
        cache_file = os.path.join(os.path.dirname(__file__), 'worker_id')
        try:
            if os.path.exists(cache_file):
                os.remove(cache_file)
                logger.info("Cleared cached Worker ID")
        except Exception as e:
            logger.error(f"Failed to clear cached Worker ID: {e}")
    

if __name__ == "__main__":
    config = load_config()
    worker = Worker(config)
    worker.intitialize()

    try:
        while True:
            time.sleep(60) # Hold main thread
    except KeyboardInterrupt:
        logger.info("Shutting down worker...")
        worker.stop_heartbeat.set()
        worker.network_controller.destroy()
        logger.info("Worker shut down successfully.")