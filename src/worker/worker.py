import logging
import os
from common.config import load_config
from common.util import get_cpu_serial, generate_identifier
from worker.network_manager import WorkerNetworkController
from common.model import WorkerIdAssignmentRequest, WorkerNetworkModeRequest, ConnectionType
from worker.websocket_server import WorkerWebSocketServer
import time
from fastapi import FastAPI, WebSocket
import uvicorn
import requests
import threading

logger = logging.getLogger(__name__)
logging.basicConfig(filename='worker.log', level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

heartbeat_interval = 5  # seconds

# Control plane:
# Controller -> Worker uses WebSocket for real-time commands
# Worker -> Controller uses HTTP REST API for heartbeats and status updates

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
        self.ws_server = WorkerWebSocketServer(config)
        self._setup_fastapi_routes()

    def _setup_fastapi_routes(self):
        # WebSocket endpoint for real-time controller -> worker communication
        @self.app.websocket("/worker_ws")
        async def worker_handle_websocket(websocket: WebSocket):
            await self.ws_server.handle_connection(websocket)
        
        async def handle_switch_to_ethernet(data: dict[str, any]):
            logger.info("Received command to switch to Ethernet connection")
            self.network_controller.switch_to_ethernet()
        
        async def handle_switch_to_wifi(data: dict[str, any]):
            logger.info("Received command to switch to WiFi connection")
            self.network_controller.switch_to_wifi(ssid=data.get('ssid'), password=data.get('password'))
        
        self.ws_server.register_handler('switch_to_ethernet', handle_switch_to_ethernet)
        self.ws_server.register_handler('switch_to_wifi', handle_switch_to_wifi)

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
        self.hardware_serial = get_cpu_serial()
        logger.info(f"Worker Hardware Serial: {self.hardware_serial}")
        self.hardware_identifier = generate_identifier(self.hardware_serial)
        logger.info(f"Worker Hardware Identifier: {self.hardware_identifier}")

        # Get intial network setup via DHCP for cached Worker ID conflict checking / new Worker ID assignment
        logger.info("Using DHCP for initial network setup...")
        self.network_controller = WorkerNetworkController(-1, self.config)
        self.network_controller.initialize() # DHCP on ethernet

        # Depreciated: Check for cached worker ID
        # self.worker_id = self._load_cached_worker_id()
        # logger.info("No valid cached Worker ID found")

        self._handle_startup()

        # Start FastAPI server
        self.start_api_server()
        
    def _handle_startup(self):
        logger.info("Starting worker without cached Worker ID...")
        # Start sending heartbeats to controller, waiting for Worker ID assignment
        self._start_heartbeat_loop(heartbeat_interval)


    def _start_heartbeat_loop(self, interval: int = 5):
        self.stop_heartbeat.clear()

        def heartbeat_task():
            count = 0
            while not self.stop_heartbeat.is_set():
                try:
                    logger.info(f"Sending heartbeat to controller {time.time()}")
                    success = self.network_controller._send_control_heartbeat(self.hardware_serial, self.hardware_identifier)
                    if not success:
                        logger.warning("Controller did not acknowledge heartbeat")
                        count += 1
                        if count % 10 == 1:
                            print(f"Controller did not acknowledge heartbeat (consecutive attempt #{count})")
                    else:
                        count = 0
                    time.sleep(interval)
                except requests.exceptions.ConnectionError:
                    logger.warning("Failed to connect to controller for heartbeat")
                except Exception as e:
                    logger.error(f"Error sending heartbeat: {e}")
                self.stop_heartbeat.wait(interval)
            logger.info("Heartbeat loop stopped.")
        
        self.heartbeat_thread = threading.Thread(target=heartbeat_task, daemon=True)
        self.heartbeat_thread.start()
        logger.info(f"Control heartbeat loop started with interval {interval} seconds.")

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
    
    # def _load_cached_worker_id(self) -> int:
    #     cache_file = os.path.join(os.path.dirname(__file__), 'worker_id')
    #     if not os.path.exists(cache_file):
    #         return -1
    #     try:
    #         with open(cache_file, 'r') as f:
    #             if f"{self.hardware_serial} " in f.read():
    #                 f.seek(0)
    #                 line = f.readline().strip()
    #                 _, worker_id_str = line.split(' ')
    #                 if int(worker_id_str) < 0 or int(worker_id_str) > 99:
    #                     raise ValueError("Cached worker_id is out of valid range (0-99)")
    #                 return int(worker_id_str)
    #             else:
    #                 return -1
    #     except Exception as e:
    #         logger.error(f"Failed to read cached worker ID: {e}")
    #         return -1
    # 
    # def _cache_worker_id(self, worker_id: int):
    #     cache_file = os.path.join(os.path.dirname(__file__), 'worker_id')
    #     # Clear existing & write new cache
    #     try:
    #         with open(cache_file, 'w') as f:
    #             f.write(f"{self.hardware_serial} {worker_id}\n")
    #         logger.info(f"Cached Worker ID {worker_id} to {cache_file}")
    #     except Exception as e:
    #         logger.error(f"Failed to cache Worker ID: {e}")

    # def _clear_cached_worker_id(self):
    #     cache_file = os.path.join(os.path.dirname(__file__), 'worker_id')
    #     try:
    #         if os.path.exists(cache_file):
    #             os.remove(cache_file)
    #             logger.info("Cleared cached Worker ID")
    #     except Exception as e:
    #         logger.error(f"Failed to clear cached Worker ID: {e}")
    

if __name__ == "__main__":
    config = load_config()
    worker = Worker(config)
    worker.intitialize()
    print(f'Worker "{worker.hardware_identifier}" is running!\nHardware Serial: {worker.hardware_serial}')

    try:
        while True:
            time.sleep(60) # Hold main thread
    except KeyboardInterrupt:
        logger.info("Shutting down worker...")
        worker.stop_heartbeat.set()
        worker.network_controller.destroy()
        logger.info("Worker shut down successfully.")