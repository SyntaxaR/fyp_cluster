from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from worker.network_manager import WorkerNetworkController, ConnectionType
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import os
import logging
import tomllib

logger = logging.getLogger(__name__)

# Load configuration file
with open(os.path.join(os.path.dirname(__file__), '../..', 'config.toml'), 'rb') as f:
    config = tomllib.load(f)

app = FastAPI()

# Get worker ID from environment variable


worker_id = int(os.environ.get('WORKER_ID', '0'))
network_controller = WorkerNetworkController(worker_id=worker_id, config=config)

class NetworkConfigRequest(BaseModel):
    mode: ConnectionType
    worker_id: int

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup
    logger.info("Starting up Worker Control API...")
    try:
        # Set ethernet as control plane & default data plane
        network_controller.initialize()
    except Exception as e:
        logger.error(f"Failed to start Worker Control API: {e}")
        raise e
    
    yield
    
    # Shutdown
    logger.info("Shutting down Worker Control API...")


@app.post('/control/network/configure')
async def configure_network(request: NetworkConfigRequest):
    try:
        match request.mode:
            case ConnectionType.ETHERNET:
                network_controller.use_ethernet_dataplane()
            case ConnectionType.WIFI:
                network_controller.use_wifi_dataplane()
            case _:
                raise HTTPException(status_code=400, detail="Invalid network mode specified")
        return {"status": "success", "message": f"Network mode switched to {request.mode.value}"}
    except Exception as e:
        logger.error(f"Error configuring network: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get('/control/network/status')
async def get_network_status():
    results = {
        "connectivity": network_controller._verify_connectivity(),
        "current_mode": network_controller.current_mode.value,
        "ipv4_address": network_controller.eth_ipv4 if network_controller.current_mode == ConnectionType.ETHERNET else network_controller.wifi_ipv4
    }
    return results