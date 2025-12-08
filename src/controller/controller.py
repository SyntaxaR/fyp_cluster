from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from common.model import WorkerHeartbeat, ConnectionType, WorkerRegistration, WorkerStatus, ConnectivityTestResponse, WorkerControlInfo
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
workers_ws_manager = None

templates = Jinja2Templates(directory="src/controller/templates")

# HTML, for pc management to show all workers, register & switch data plane
@control_app.get('/')
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "pending_workers": pending_workers, 
        "registered_workers": registered_workers
    })
    
@control_app.post('/api/heartbeat')
async def receive_heartbeat(heartbeat: WorkerHeartbeat):
    logger.info(f"Received heartbeat from Worker ID {heartbeat.worker_id} (Serial: {heartbeat.serial})")
    if heartbeat.worker_id == -1:
        # Unassigned worker, add to pending list if not already present
        if heartbeat.serial in pending_workers:
            # Update timestamp for existing pending worker
            pending_workers[heartbeat.serial].timestamp = int(time.time())
            logger.info(f"Updated timestamp for pending worker (Serial: {heartbeat.serial})")
            # Check if the same serial is in registered workers, if so assign worker ID back
        else:
            # New pending worker
            heartbeat.timestamp = int(time.time()) # Ensure timestamp is current based on controller's local time
            pending_workers[heartbeat.serial] = heartbeat
            logger.info(f"Worker (Serial: {heartbeat.serial}) added to pending registration list")
            for wid, reg in registered_workers.items():
                if reg.serial == heartbeat.serial:
                    logger.info(f'Re-assigning Worker ID {wid} to pending worker "{heartbeat.hardware_identifier}"')
                    # TODO: Reassign worker ID
                    # For now just delete from registered and let the user register again
                    del registered_workers[wid]
                    break
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

@control_app.get('/op/register_worker/{worker_serial}')
async def register_worker_endpoint(worker_serial: str):
    print(f'Registering worker "{generate_identifier(worker_serial)}" {worker_serial}...')
    # return {"success": False, "message": "Registration endpoint is disabled for testing"}
    if worker_serial not in pending_workers:
        logger.error(f"Attempted to register unknown worker (Serial: {worker_serial}) via API")
        return {"success": False, "message": f"Unknown worker serial {worker_serial}"}
    success = await register_worker(pending_workers[worker_serial])
    if success:
        return {"success": True, "message": f"Worker {worker_serial} registered successfully"}
    else:
        return {"success": False, "message": f"Failed to register worker {worker_serial}"}

@control_app.get('/op/switch_data_plane/{worker_id}/{plane}')
async def switch_data_plane(worker_id: int, plane: str):
    print(f"Switching data plane for worker ID {worker_id} to {plane}...")
    if worker_id not in registered_workers:
        logger.error(f"Attempted to switch data plane for unknown worker ID {worker_id}")
        return {"success": False, "message": f"Unknown worker ID {worker_id}"}
    if plane.lower() not in ['ethernet', 'wifi']:
        logger.error(f"Invalid data plane '{plane}' requested for worker ID {worker_id}")
        return {"success": False, "message": f"Invalid data plane '{plane}'"}
    
    if plane.lower() == 'ethernet':
        command = 'switch_to_ethernet'
        data = {}
    else:
        command = 'switch_to_wifi'
        data = {"ssid": config['network']['wifi_ssid'], "password": config['network']['wifi_password']}

    success = await workers_ws_manager.send_command(registered_workers[worker_id], command, data)
    if success:
        logger.info(f"Sent command to switch Worker ID {worker_id} data plane to {plane}")
        registered_workers[worker_id].data_plane = ConnectionType.ETHERNET if plane.lower() == 'ethernet' else ConnectionType.WIFI
        return {"success": True, "message": f"Switch data plane command sent to Worker ID {worker_id}"}
    else:
        logger.error(f"Failed to send switch data plane command to Worker ID {worker_id}")
        return {"success": False, "message": f"Failed to send switch data plane command to Worker ID {worker_id}"}

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
        worker_id=worker_id,
        serial=heartbeat.serial,
        hardware_identifier=heartbeat.hardware_identifier,
        control_ip=heartbeat.control_ip_address,
        data_ip=heartbeat.data_ip_address,
        data_plane=heartbeat.data_plane, # TODO: I
        timestamp=int(time.time()),
        status=WorkerStatus.ACTIVE
    )

    logger.info(f'Establishing WebSocket connection to newly registered worker ID {worker_id}...')
    worker = WorkerControlInfo(
        worker_id=worker_id,
        control_ip=registration.control_ip,
        serial=registration.serial,
        identifier=registration.hardware_identifier
    )
    ws_status = await workers_ws_manager.connect_to_worker(worker)
    
    if ws_status:
        print(f'Worker {worker_id} "{registration.hardware_identifier}" registered successfully and WebSocket connection established.')
        logger.info(f'Worker {worker_id} "{registration.hardware_identifier}" registered successfully and WebSocket connection established.')
        registered_workers[worker_id] = registration
        del pending_workers[heartbeat.serial]
        return True
    else:
        registration.status = WorkerStatus.RECONNECTING
        print(f'Worker {worker_id} "{registration.hardware_identifier}" registered successfully but failed to establish WebSocket connection.')
        logger.error(f'Worker {worker_id} "{registration.hardware_identifier}" registered successfully but failed to establish WebSocket connection.')
        return False

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
        for worker_serial, heartbeat in pending_workers.items():
            if heartbeat.timestamp < threshold_time:
                logger.warning(f"Pending worker (Serial: {worker_serial}, Identifier: {heartbeat.hardware_identifier}) heartbeat timeout detected (last timestamp: {heartbeat.timestamp}), removing from pending list...")
                del pending_workers[worker_serial]
        for worker_id, registration in registered_workers.items():
            if registration.timestamp < threshold_time:
                logger.warning(f"Worker ID {worker_id} \"{registration.hardware_identifier}\" heartbeat timeout detected (last timestamp: {registration.timestamp}), disconnecting...")
                # Handle timeout: set status to INACTIVE and close WebSocket connection
                # Will try to reconnect to the worker during the next monitor cycle if received new heartbeats
                registration.status = WorkerStatus.INACTIVE
                try:
                    await workers_ws_manager._handle_disconnection(worker_id, reconnect=False)
                except Exception as e:
                    logger.error(f"Error handling disconnection for Worker ID {worker_id}: {e}, might already be disconnected")
            elif registration.status == WorkerStatus.INACTIVE:
                # Reconnect
                logger.info(f"New heartbeat received! Attempting to reconnect to inactive Worker ID {worker_id} \"{registration.hardware_identifier}\"...")
                worker = WorkerControlInfo(
                    worker_id=worker_id,
                    control_ip=registration.control_ip,
                    serial=registration.serial,
                    identifier=registration.hardware_identifier
                )
                try:
                    await workers_ws_manager.connect_to_worker(worker)
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
            for serial, heartbeat in pending_workers.items():
                print(f'Pending Registration: Worker "{heartbeat.hardware_identifier}" (Serial: {serial}, Last Heartbeat: {int(time.time()) - heartbeat.timestamp}s before)')
                logger.info(f'Pending Registration: Worker "{heartbeat.hardware_identifier}" (Serial: {serial}, Last Heartbeat: {int(time.time()) - heartbeat.timestamp}s before)')
            for worker_id, registration in registered_workers.items():
                print(f'Registered: Worker {worker_id} "{registration.hardware_identifier}" : {registration.status.value} (Serial: {registration.serial}, Last Heartbeat: {int(time.time()) - registration.timestamp}s before)')
                logger.info(f'Registered: Worker {worker_id} "{registration.hardware_identifier}" : {registration.status.value} (Serial: {registration.serial}, Last Heartbeat: {int(time.time()) - registration.timestamp}s before)')
    except KeyboardInterrupt:
        logger.info("Controller shutting down...")
    finally:
        await workers_ws_manager.disconnect_all()

# START 251208 DEMO
from controller.tmp.demo251208 import *
from fastapi import File, UploadFile
from datasets import load_dataset

dataset = None
batch_size = 1024

@control_app.get("/demo")
async def demo_index_251208(request: Request):
    return templates.TemplateResponse("demo251208.html", {
        "request": request,
        "registered_workers": registered_workers,
    })

# Load .csv
# fulltext,overall
# Some Title Some Text,1-5
@control_app.post("/demo/csv_upload")
async def csv_upload(file: UploadFile):
    global dataset
    import tempfile
    import os
    
    # 保存上传的文件到临时文件
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # 使用文件路径加载数据集
        dataset = load_dataset('csv', data_files={'data': temp_path}, delimiter=',', split='data')
        print(f"Dataset loaded with {len(dataset)} rows")
        return {"success": True, "message": f"Dataset loaded with {len(dataset)} rows"}
    finally:
        # 清理临时文件
        os.unlink(temp_path)

# Upload & load csv file, and upload model folder to all active registered workers
@control_app.post("/demo/upload_and_deploy")
async def model_upload_and_deploy(files: list[UploadFile]):
    # Save uploaded files to local temp directory
    local_model_dir = "/tmp/uploaded_model/"
    import os
    import shutil
    if os.path.exists(local_model_dir):
        shutil.rmtree(local_model_dir)
    os.makedirs(local_model_dir, exist_ok=True)
    for file in files:
        file_location = os.path.join(local_model_dir, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())
    # Deploy to all active registered workers
    for worker_id, registration in registered_workers.items():
        if registration.status == WorkerStatus.ACTIVE:
            worker = WorkerControlInfo(
                worker_id=worker_id,
                control_ip=registration.control_ip,
                serial=registration.serial,
                identifier=registration.hardware_identifier
            )
            deploy_model_to_worker(worker, local_model_dir)

def deploy_model_to_worker(worker: WorkerControlInfo, local_dir: str):
    send_directory(worker_ip=worker.control_ip, local_dir=local_dir)

@control_app.get("/demo/start_inference")
async def demo_start_inference():
    for worker_id, registration in registered_workers.items():
        if registration.status == WorkerStatus.ACTIVE:
            worker = WorkerControlInfo(
                worker_id=worker_id,
                control_ip=registration.control_ip,
                serial=registration.serial,
                identifier=registration.hardware_identifier
            )
            try:
                await workers_ws_manager.send_command(worker, "start_demo_inference", {})
                print(f'Sent start demo inference command to Worker {worker_id} "{registration.hardware_identifier}"')
            except Exception as e:
                logger.error(f"Error sending start demo inference command to Worker ID {worker_id}: {e}")
    return {"success": True, "message": "Start inference commands sent to all active workers"}

@data_app.get("/demo/get_batch")
async def demo_get_batch():
    batch_size = 1024
    global dataset
    if demo_start_time is None:
        demo_start_time = int(time.time())
    if dataset is None:
        return {"success": False, "message": "No dataset loaded"}
    # Get a batch of data
    batch = dataset.shuffle().select(range(batch_size))
    # Mix fulltext and overall labels into a list of tuples
    data = list(zip(batch['fulltext'], batch['overall']))
    return {"success": True, "data": data}

@control_app.get("/demo/set_batch_size/{size}")
async def demo_set_batch_size(size: int):
    global batch_size
    batch_size = size
    return {"success": True, "message": f"Batch size set to {size}"}

demo_start_time = None
demo_submissions = {}

# Only need to include correct & incorrect counts
@data_app.get("/demo/submit_inference/{serial}/{correct}/{total}")
async def demo_submission(request: Request, serial: str, correct: int, total: int):
    logger.info(f"Received demo inference submission: {correct} correct, {total} total")
    print(f"Received demo inference submission: {correct} correct, {total} total")
    global demo_submissions
    if serial not in demo_submissions:
        demo_submissions[serial] = {"correct": correct, "incorrect": total - correct}
    else:
        demo_submissions[serial]['correct'] += correct
        demo_submissions[serial]['incorrect'] += total - correct
    return {"success": True, "message": "Submission received"}

@control_app.get("/demo/get_stats")
async def demo_get_stats():
    global demo_submissions, demo_start_time, registered_workers
    
    # Build worker stats with identifiers
    worker_stats = {}
    for serial, stats in demo_submissions.items():
        # Find the worker with this serial
        worker_info = None
        for worker_id, registration in registered_workers.items():
            if registration.serial == serial:
                worker_info = {
                    "id": worker_id,
                    "identifier": registration.hardware_identifier,
                    "status": registration.status.value
                }
                break
        
        if worker_info:
            worker_stats[serial] = {
                **worker_info,
                "correct": stats["correct"],
                "incorrect": stats["incorrect"],
                "total": stats["correct"] + stats["incorrect"]
            }
    
    return {
        "success": True,
        "start_time": demo_start_time,
        "current_time": time.time(),
        "worker_stats": worker_stats
    }

# END 251208 DEMO


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