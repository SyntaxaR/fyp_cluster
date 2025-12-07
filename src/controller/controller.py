from fastapi import FastAPI, Request
from common.model import WorkerHeartbeat, ConnectionType, WorkerRegistration, WorkerStatus, ConnectivityTestResponse
from common.util import generate_identifier, get_cpu_serial
from common.config import load_config
import logging
import time
from controller.network_manager import ControllerNetworkManager
import uvicorn
import threading
import websockets

logger = logging.getLogger(__name__)
logging.basicConfig(filename='controller.log', level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = FastAPI(title="Controller API")
serial = get_cpu_serial()
identifier = generate_identifier(serial)
config = load_config()
eth_subnet = config['network']['ethernet_subnet']
wifi_subnet = config['network']['wifi_subnet']

pending_workers: dict[str, WorkerHeartbeat] = {}
registered_workers: dict[int, WorkerRegistration] = {}
worker_id_counter = 0

@app.post('/api/heartbeat')
async def receive_heartbeat(heartbeat: WorkerHeartbeat):
    logger.info(f"Received heartbeat from Worker ID {heartbeat.worker_id} (Serial: {heartbeat.serial})")
    print(f"Received heartbeat from Worker ID {heartbeat.worker_id} {heartbeat.hardware_identifier} (Serial: {heartbeat.serial}, Data Connectivity: {heartbeat.data_connectivity}, Data Plane: {heartbeat.data_plane}, Control IP: {heartbeat.control_ip_address}, Data IP: {heartbeat.data_ip_address}, Timestamp: {int(time.time())})")
    if heartbeat.worker_id == -1:
        # Unassigned worker, add to pending list if not already present
        if heartbeat.serial in pending_workers:
            pending_workers[heartbeat.serial].timestamp = int(time.time())
            logger.info(f"Updated timestamp for pending worker (Serial: {heartbeat.serial})")
            # Check if the same serial is in registered workers, if so assign worker ID back
            for wid, reg in registered_workers.items():
                if reg.serial == heartbeat.serial:
                    logger.info(f'Re-assigning Worker ID {wid} to pending worker "{heartbeat.hardware_identifier}"')
                    # TODO: Reassign worker ID
        else:
            heartbeat.timestamp = int(time.time()) # Ensure timestamp is current based on controller's local time
            pending_workers[heartbeat.serial] = heartbeat
            logger.info(f"Worker (Serial: {heartbeat.serial}) added to pending registration list")
    else:
        # Registered worker, update status
        registered_workers[heartbeat.worker_id].status = WorkerStatus.ACTIVE
        registered_workers[heartbeat.worker_id].timestamp = int(time.time())
        logger.info(f'Worker ID {heartbeat.worker_id} "{registered_workers[heartbeat.worker_id].hardware_identifier}" status updated (active)')

# plane depends on the incoming request interface
@app.post('/api/connectivity_test')
async def connectivity_test(request: Request) -> ConnectivityTestResponse:
    if request.client.host.startswith(eth_subnet):
        plane = ConnectionType.ETHERNET
    elif request.client.host.startswith(wifi_subnet):
        plane = ConnectionType.WIFI
    else:
        plane = ConnectionType.INVALID
    logger.info(f"Received connectivity test from {request.client.host} on {plane.value} plane")
    return ConnectivityTestResponse(from_identifier=identifier, message="Connectivity test successful", plane=plane)

def register_worker(heartbeat: WorkerHeartbeat, worker_id: int=-1) -> bool:
    if heartbeat.serial not in pending_workers:
        logger.error(f"Attempted to register unknown worker (Serial: {heartbeat.serial})")
        return False
    if worker_id < 0:
        global worker_id_counter
        worker_id = worker_id_counter
        worker_id_counter += 1
        logger.info(f'Assigning new Worker ID {worker_id} to worker "{heartbeat.hardware_identifier}" (Serial: {serial})')
    else:
        logger.info(f'Re-assigning Worker ID {worker_id} to worker "{heartbeat.hardware_identifier}" (Serial: {serial})')
    # TODO: FINISH LOGIC

def start_api_server(app):
    def run():
        uvicorn.run(app, host="0.0.0.0", port=config['controller']['control_port'], log_level="info")
    api_thread = threading.Thread(target=run, daemon=True)
    logger.info("Starting FastAPI server for controller API...")
    api_thread.start()

if __name__ == "__main__":
    network_manager = ControllerNetworkManager(config)
    # FOR TESTING ONLY
    print("Initializing controller network interfaces...")
    network_manager.initialize(initialize_wifi=False)
    # Start API server
    print("Starting Controller API server...")
    start_api_server(app)
    print("Controller is running.")
    # Hold the process
    count = 0
    while True:
        time.sleep(30)
        count += 1
        if count % 4 == 0:
            print(f"Controller running... {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
            count = 0
        if not network_manager._check_subprocess_health():
            logger.error("One or more network subprocesses have terminated unexpectedly. Exiting controller...")
            break