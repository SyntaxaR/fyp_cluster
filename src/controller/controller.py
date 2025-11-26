from fastapi import FastAPI
from pydantic import BaseModel
from common.model import WorkerHeartbeat, WorkerRegistration
import logging
import time

logger = logging.getLogger(__name__)

app = FastAPI(title="Controller API")

pending_workers = {}
registered_workers = {}
worker_id_counter = 0

@app.post('/heartbeat')
def receive_heartbeat(heartbeat: WorkerHeartbeat):
    logger.info(f"Received heartbeat from Worker ID {heartbeat.worker_id} (Serial: {heartbeat.serial})")
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