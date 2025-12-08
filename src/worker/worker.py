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

        self.control_app = FastAPI()
        self.data_app = FastAPI()
        self.ws_server = WorkerWebSocketServer(config)
        self._setup_fastapi_routes()

        # TODO: FOR DEMO ONLY
        from worker.tmp.demo251208 import InferenceServer
        self.inference_server: InferenceServer = None

    def _setup_fastapi_routes(self):
        # WebSocket endpoint for real-time controller -> worker communication
        @self.control_app.websocket("/worker_ws")
        async def worker_handle_websocket(websocket: WebSocket):
            await self.ws_server.handle_connection(websocket)
        
        async def handle_switch_to_ethernet(data: dict[str, any]):
            logger.info("Received command to switch to Ethernet connection")
            self.network_controller.switch_to_ethernet()
        
        async def handle_switch_to_wifi(data: dict[str, any]):
            logger.info("Received command to switch to WiFi connection")
            self.network_controller.switch_to_wifi(ssid=data.get('ssid'), password=data.get('password'))
        
        async def _handle_update_worker_id(data: dict[str, any]):
            new_worker_id = data.get('worker_id', -1)
            if isinstance(new_worker_id, int) and -1 < new_worker_id:
                logger.info(f"Updating Worker ID from {self.worker_id} to {new_worker_id}")
                self.worker_id = new_worker_id
                self.network_controller.worker_id = new_worker_id
            else:
                logger.error(f"Received invalid Worker ID update: {new_worker_id}")

        self.ws_server.register_handler('switch_to_ethernet', handle_switch_to_ethernet)
        self.ws_server.register_handler('switch_to_wifi', handle_switch_to_wifi)
        self.ws_server.register_handler('update_worker_id', _handle_update_worker_id)

    def start_control_api_server(self):
        def run():
            uvicorn.run(self.control_app, host="0.0.0.0", port=self.config['worker']['control_port'], log_level="info")
        api_thread = threading.Thread(target=run, daemon=True)
        logger.info("Starting FastAPI server for worker control API...")
        api_thread.start()
    
    def start_data_api_server(self):
        def run():
            uvicorn.run(self.data_app, host="0.0.0.0", port=self.config['worker']['data_port'], log_level="info")
        api_thread = threading.Thread(target=run, daemon=True)
        logger.info("Starting FastAPI server for worker data API...")
        api_thread.start()

    def initialize(self):
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
        self.start_control_api_server()
        self.start_data_api_server()
        
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


    # START 251208 DEMO

    def _setup_demo_routes(self):
        @self.data_app.get("/demo/start_inference")
        async def demo_start_inference():
            if self.inference_server is None:
                from worker.tmp.demo251208 import InferenceServer
                self.inference_server = InferenceServer()
                logger.info("Demo Inference Server initialized")
                # Run inference in a separate thread
                threading.Thread(target=self.run_inference, daemon=True).start()
                return {"status": "Inference server started"}
            else:
                logger.warning("Demo Inference Server is already running")
                return {"status": "Inference server already running"}
        
    inputs = []
    # Report to controller every 10 inferences
    report_interval = 10
    # When inputs is empty, request a new batch from controller
    # Else, keep running inference on existing inputs until empty
    def run_inference(self):
        if self.inference_server is None:
            logger.error("Inference server is not initialized!")
            return
        correct_count = 0
        total_count = 0
        while True:
            if len(self.inputs) == 0:
                # Request new batch from controller
                #     @data_app.get("/demo/get_batch")
                #     async def demo_get_batch():
                #     batch_size = 1024
                #     global dataset
                #     if dataset is None:
                #         return {"success": False, "message": "No dataset loaded"}
                #     # Get a batch of data
                #     batch = dataset.shuffle().select(range(batch_size))
                #     # Mix fulltext and overall labels into a list of tuples
                #     data = list(zip(batch['fulltext'], batch['overall']))
                #     return {"success": True, "data": data}
                try:
                    response = requests.get(f"{self.network_controller.eth_controller_ipv4 if self.network_controller.current_mode == ConnectionType.ETHERNET else self.network_controller.wifi_controller_ipv4}:{self.config['controller']['data_port']}/demo/get_batch")
                    if response.status_code == 200:
                        data = response.json().get('data', [])
                        self.inputs = data
                        logger.info(f"Received new inference batch of size {len(data)} from controller")
                    else:
                        logger.error(f"Failed to get inference batch from controller: {response.status_code}")
                        time.sleep(2)
                        continue
                except Exception as e:
                    logger.error(f"Error requesting inference batch from controller: {e}")
                    time.sleep(2)
                    continue

            text, true_id = self.inputs.pop(0)
            is_correct = self.inference_server.run_inference(text, true_id)
            total_count += 1
            if is_correct:
                correct_count += 1

            if total_count % self.report_interval == 0:
                try:
                    #@data_app.get("/demo/submit_inference/{serial}/{correct}/{total}")
                    response = requests.get(f"{self.network_controller.eth_controller_ipv4 if self.network_controller.current_mode == ConnectionType.ETHERNET else self.network_controller.wifi_controller_ipv4}:{self.config['controller']['data_port']}/demo/submit_inference/{self.hardware_serial}/{correct_count}/{self.report_interval}")
                except Exception as e:
                    logger.error(f"Error reporting inference results to controller: {e}")
                    continue
                total_count = 0
                correct_count = 0


    # END 251208 DEMO


if __name__ == "__main__":
    config = load_config()
    worker = Worker(config)
    worker._setup_demo_routes()
    worker.initialize()
    print(f'Worker "{worker.hardware_identifier}" is running!\nHardware Serial: {worker.hardware_serial}')

    try:
        while True:
            time.sleep(60) # Hold main thread
    except KeyboardInterrupt:
        logger.info("Shutting down worker...")
        worker.stop_heartbeat.set()
        worker.network_controller.destroy()
        logger.info("Worker shut down successfully.")