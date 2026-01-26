from fastapi import FastAPI, Request
from common.model import WorkerHeartbeat, ConnectionType, WorkerRegistration, WorkerStatus, ConnectivityTestResponse, \
    WorkerControlInfo
from common.util import generate_identifier, get_cpu_serial
from common.config import load_config
import logging
import time
from controller.network_manager import ControllerNetworkManager
from controller.workers_websocket_manager import WorkersWebSocketManager
import uvicorn
import threading
import asyncio

logger = logging.getLogger(__name__)
logging.basicConfig(filename='controller.log', level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

monitor_interval = 10  # seconds
timeout_threshold = 15 # seconds (15s -> 3x worker's heartbeat interval)

control_app = FastAPI(title="Controller Control API")
data_app = FastAPI(title="Controller Data API")
serial = get_cpu_serial()
identifier = generate_identifier(serial)
config = load_config()
eth_subnet = config['network']['ethernet_subnet']
wifi_subnet = config['network']['wifi_subnet']

pending_workers: dict[str, WorkerHeartbeat] = {}
registered_workers: dict[int, WorkerRegistration] = {}
worker_id_counter = 0
workers_ws_manager: WorkersWebSocketManager

@control_app.post('/api/heartbeat')
async def receive_heartbeat(heartbeat: WorkerHeartbeat):
    logger.info(f"Received heartbeat from Worker ID {heartbeat.worker_id} (Serial: {heartbeat.serial})")
    if heartbeat.worker_id == -1:
        # Unassigned worker, add to the pending list if not already present
        if heartbeat.serial in pending_workers:
            # Update timestamp for existing pending worker
            pending_workers[heartbeat.serial].timestamp = int(time.time())
            logger.info(f"Updated timestamp for pending worker (Serial: {heartbeat.serial})")
            # Check if the same serial is in registered workers, if so, assign worker ID back
            for wid, reg in registered_workers.items():
                if reg.serial == heartbeat.serial:
                    logger.info(f'Re-assigning Worker ID {wid} to pending worker "{heartbeat.hardware_identifier}"')
                    # TODO: Reassign worker ID
        else:
            # New pending worker
            heartbeat.timestamp = int(time.time()) # Ensure the timestamp is current based on the controller's local time
            pending_workers[heartbeat.serial] = heartbeat
            logger.info(f"Worker (Serial: {heartbeat.serial}) added to pending registration list")
    else:
        # Registered worker, update timestamp
        registered_workers[heartbeat.worker_id].timestamp = int(time.time())
        logger.info(f'Worker ID {heartbeat.worker_id} "{registered_workers[heartbeat.worker_id].hardware_identifier}" heartbeat timestamp updated (active)')

# plane depends on the incoming request interface
@control_app.get('/api/connectivity_test')
@data_app.get('/api/connectivity_test')
async def connectivity_test(request: Request) -> ConnectivityTestResponse:
    if request.client.host.startswith(eth_subnet):
        plane = ConnectionType.ETHERNET
    elif request.client.host.startswith(wifi_subnet):
        plane = ConnectionType.WIFI
    else:
        plane = ConnectionType.INVALID
    logger.info(f"Received connectivity test from {request.client.host} on {plane} plane")
    return ConnectivityTestResponse(from_identifier=identifier, message="Connectivity test successful", plane=plane)

async def register_worker(heartbeat: WorkerHeartbeat, worker_id: int=-1) -> bool:
    global worker_id_counter
    if heartbeat.serial not in pending_workers:
        logger.error(f"Attempted to register unknown worker (Serial: {heartbeat.serial})")
        return False
    
    if worker_id < 0:
        worker_id = worker_id_counter
        worker_id_counter += 1
        logger.info(f'Assigning new Worker ID {worker_id} to worker "{heartbeat.hardware_identifier}" (Serial: {serial})')
    else:
        logger.info(f'Re-assigning Worker ID {worker_id} to worker "{heartbeat.hardware_identifier}" (Serial: {serial})')
    
    registration = WorkerRegistration(
        serial=heartbeat.serial,
        hardware_identifier=heartbeat.hardware_identifier,
        control_ip=heartbeat.control_ip_address,
        data_ip=heartbeat.data_ip_address,
        data_plane=heartbeat.data_plane,
        timestamp=int(time.time()),
        status=WorkerStatus.REGISTERED
    )

    registered_workers[worker_id] = registration
    del pending_workers[heartbeat.serial]

    logger.info(f'Establishing WebSocket connection to newly registered worker ID {worker_id}...')
    w_control_info = WorkerControlInfo(
        control_ip=registration.control_ip,
        worker_id=worker_id,
        identifier=registration.hardware_identifier,
        serial=registration.serial,
    )
    ws_status = await workers_ws_manager.connect_to_worker(w_control_info)
    if ws_status:
        print(f'Worker {worker_id} "{registration.hardware_identifier}" registered successfully and WebSocket connection established.')
        logger.info(f'Worker {worker_id} "{registration.hardware_identifier}" registered successfully and WebSocket connection established.')
        return True
    else:
        registration.status = WorkerStatus.RECONNECTING
        print(f'Worker {worker_id} "{registration.hardware_identifier}" registered successfully but failed to establish WebSocket connection.')
        logger.error(f'Worker {worker_id} "{registration.hardware_identifier}" registered successfully but failed to establish WebSocket connection.')
        return False

def get_worker_control_info(worker_id: int) -> WorkerControlInfo:
    return WorkerControlInfo(
        control_ip=registered_workers[worker_id].control_ip,
        worker_id=worker_id,
        identifier=registered_workers[worker_id].hardware_identifier,
        serial=registered_workers[worker_id].serial,
    )

async def on_worker_status_change(worker_id: int, status: WorkerStatus):
    if worker_id in registered_workers:
        registered_workers[worker_id].status = status
        logger.info(f'Worker {worker_id} "{registered_workers[worker_id].hardware_identifier}" status updated to {status.value}')
    else:
        logger.warning(f'Received status update for unknown Worker ID {worker_id}')

def start_api_server(app, port):
    def run():
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    api_thread = threading.Thread(target=run, daemon=True)
    logger.info("Starting FastAPI server for controller...")
    api_thread.start()

async def monitor_worker_timestamp():
    while True:
        await asyncio.sleep(monitor_interval)
        current_time = int(time.time())
        threshold_time = current_time - timeout_threshold
        for worker_id, registration in registered_workers.items():
            if registration.timestamp < threshold_time:
                logger.warning(f"Worker ID {worker_id} \"{registration.hardware_identifier}\" heartbeat timeout detected (last timestamp: {registration.timestamp}), disconnecting...")
                # Handle timeout: set status to INACTIVE and close WebSocket connection
                # Will try to reconnect to the worker during the next monitor cycle if received new heartbeats
                registration.status = WorkerStatus.INACTIVE
                try:
                    await workers_ws_manager.disconnect_worker(worker_id)
                except Exception as e:
                    logger.error(f"Error handling disconnection for Worker ID {worker_id}: {e}, might already be disconnected")
            elif registration.status == WorkerStatus.INACTIVE:
                # Reconnect
                logger.info(f"New heartbeat received! Attempting to reconnect to inactive Worker ID {worker_id} \"{registration.hardware_identifier}\"...")
                try:
                    await workers_ws_manager.connect_to_worker(get_worker_control_info(worker_id))
                except Exception as e:
                    logger.error(f"Error reconnecting to Worker ID {worker_id}: {e}")


async def async_main():
    global workers_ws_manager
    workers_ws_manager = WorkersWebSocketManager(config)
    workers_ws_manager.register_status_change_callback(on_worker_status_change)
    asyncio.create_task(monitor_worker_timestamp())
    try:
        while True:
            await asyncio.sleep(30)
            # Print all kinds of workers (pending registration, registered, active, reconnecting, inactive...)
            logger.info(f"\n----- Worker Status Summary {int(time.time())} -----")
            print(f"\n----- Worker Status Summary {int(time.time())} -----")
            for worker_serial, heartbeat in pending_workers.items():
                print(f'Pending Registration: Worker "{heartbeat.hardware_identifier}" (Serial: {worker_serial}, Last Heartbeat: {int(time.time()) - heartbeat.timestamp}s before)')
                logger.info(f'Pending Registration: Worker "{heartbeat.hardware_identifier}" (Serial: {worker_serial}, Last Heartbeat: {int(time.time()) - heartbeat.timestamp}s before)')
            for worker_id, registration in registered_workers.items():
                print(f'Registered: Worker {worker_id} "{registration.hardware_identifier}" : {registration.status.value} (Serial: {registration.serial}, Last Heartbeat: {int(time.time()) - registration.timestamp}s before)')
                logger.info(f'Registered: Worker {worker_id} "{registration.hardware_identifier}" : {registration.status.value} (Serial: {registration.serial}, Last Heartbeat: {int(time.time()) - registration.timestamp}s before)')
    except KeyboardInterrupt:
        logger.info("Controller shutting down...")
    finally:
        await workers_ws_manager.disconnect_all()

if __name__ == "__main__":
    network_manager = ControllerNetworkManager(config)
    # FOR TESTING ONLY
    print("Initializing controller network interfaces...")
    network_manager.initialize(initialize_wifi=False)
    # Start API server
    print("Starting Controller API server...")
    start_api_server(control_app, config['controller']['control_port'])
    start_api_server(data_app, config['controller']['data_port'])
    print("Controller is running.")
    asyncio.run(async_main())