from fastapi import FastAPI
from common.model import WorkerHeartbeat, WorkerRegistration
from common.config import load_config
import logging
import time
from controller.network_manager import ControllerNetworkManager
import uvicorn
import threading

logger = logging.getLogger(__name__)
logging.basicConfig(filename='controller.log', level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = FastAPI(title="Controller API")

pending_workers = {}
registered_workers = {}
worker_id_counter = 0

@app.post('/heartbeat')
def receive_heartbeat(heartbeat: WorkerHeartbeat):
    logger.info(f"Received heartbeat from Worker ID {heartbeat.worker_id} (Serial: {heartbeat.serial})")
    print(f"Received heartbeat from Worker ID {heartbeat.worker_id} {heartbeat.hardware_identifier} (Serial: {heartbeat.serial}, Data Connectivity: {heartbeat.data_connectivity}, Data Plane: {heartbeat.data_plane}, Control IP: {heartbeat.control_ip_address}, Data IP: {heartbeat.data_ip_address}, Timestamp: {int(time.time())})")
    if heartbeat.worker_id == -1:
        # Unassigned worker, add to pending list
        pending_workers[heartbeat.serial] = heartbeat
        logger.info(f"Worker (Serial: {heartbeat.serial}) added to pending registration list")
    else:
        # Registered worker, update status
        registered_workers[heartbeat.worker_id] = WorkerRegistration(
            serial=heartbeat.serial,
            hardware_identifier=heartbeat.hardware_identifier,
            data_ip=heartbeat.data_ip_address,
            control_ip=heartbeat.control_ip_address,
            data_plane=heartbeat.data_plane,
            timestamp=int(time.time()),
            status="active"
        )
        logger.info(f"Worker ID {heartbeat.worker_id} status updated")

def start_api_server(app):
    def run():
        uvicorn.run(app, host="0.0.0.0", port=config['controller']['control_port'], log_level="info")
    api_thread = threading.Thread(target=run, daemon=True)
    logger.info("Starting FastAPI server for controller API...")
    api_thread.start()

if __name__ == "__main__":
    config = load_config()
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