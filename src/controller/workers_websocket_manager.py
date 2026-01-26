import asyncio
import logging
from typing import Callable, Any, Coroutine

import websockets
from websockets.asyncio.client import connect, ClientConnection
from websockets.exceptions import ConnectionClosed, WebSocketException
from common.model import WorkerStatus, WorkerControlInfo
import json

logger = logging.getLogger(__name__)

class WorkersWebSocketManager:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.connections: dict[int, ClientConnection] = {}
        self.connection_tasks: dict[int, asyncio.Task] = {}
        self.worker_status_change_callbacks: list[Callable[[int, WorkerStatus], Coroutine[Any, Any, Any]]] = []

        self.ws_port = self.config['worker']['control_port']

        # In seconds
        self.reconnect_interval = 5.0
        self.max_reconnect_attempts = 5 # maximum number of reconnection attempts before marking a worker as disconnected
        self.connection_timeout = 5.0

    async def connect_to_worker(self, worker: WorkerControlInfo) -> bool:
        ws_uri = f"ws://{worker.control_ip}:{self.ws_port}/worker_ws"
        logger.info(f"Attempting to establish WebSocket connection to {worker} at {ws_uri}...")
        try:
            ws = await asyncio.wait_for(connect(ws_uri), timeout=self.connection_timeout)
            self.connections[worker.worker_id] = ws
            task = asyncio.create_task(self._receive_loop(worker, ws))
            self.connection_tasks[worker.worker_id] = task
            logger.info(f"Successfully established WebSocket connection to {worker} at {ws_uri}")
            await self._notify_status_change(worker, WorkerStatus.ACTIVE)
            return True
        except asyncio.TimeoutError:
            logger.error(f"WebSocket connection to {worker} at {ws_uri} timed out")
            return False
        except WebSocketException as e:
            logger.error(f"WebSocket connection to {worker} at {ws_uri} failed: {e}")
            return False
    
    async def _receive_loop(self, worker: WorkerControlInfo, ws: ClientConnection):
        try:
            while True:
                message = await ws.recv()
                logger.debug(f"Received WebSocket message from {worker}: {message}")
        except ConnectionClosed:
            logger.warning(f"WebSocket connection to {worker} lost!")
        finally:
            await self._handle_disconnection(worker)

    async def _handle_disconnection(self, worker: WorkerControlInfo | int, reconnect: bool = True):
        await self._notify_status_change(worker, WorkerStatus.INACTIVE)
        worker_id = worker.worker_id if isinstance(worker, WorkerControlInfo) else worker
        if worker_id in self.connections:
            try:
                await self.connections[worker_id].close()
            except Exception as e:
                logger.error(f"Error closing WebSocket connection to {worker}: {e}")
            del self.connections[worker_id]

        if worker_id in self.connection_tasks:
            self.connection_tasks[worker_id].cancel()
            del self.connection_tasks[worker_id]

        if not reconnect:
            await self._notify_status_change(worker, WorkerStatus.INACTIVE)

        if self.max_reconnect_attempts > 0:
            logger.info(f"Setting {worker} status to RECONNECTING and attempting reconnection...")
            await self._notify_status_change(worker, WorkerStatus.RECONNECTING)
            asyncio.create_task(self._reconnect_worker(worker))
        else:
            logger.info(f"Not attempting reconnection to {worker} (max_reconnect_attempts set to 0)")
            await self._notify_status_change(worker, WorkerStatus.INACTIVE)
    
    async def reconnect_worker(self, worker: WorkerControlInfo):
        asyncio.create_task(self._reconnect_worker(worker))

    async def _reconnect_worker(self, worker: WorkerControlInfo):
        attempts = 0
        while attempts < self.max_reconnect_attempts:
            logger.info(f"Reconnection attempt {attempts + 1} for {worker}...")
            success = await self.connect_to_worker(worker)
            if success:
                logger.info(f"Reconnected to {worker} successfully")
                return
            attempts += 1
            await asyncio.sleep(self.reconnect_interval)
        logger.error(f"Failed to reconnect to {worker} after {self.max_reconnect_attempts} attempts (max reached)")
        await self._notify_status_change(worker, WorkerStatus.INACTIVE)
    
    async def _notify_status_change(self, worker: WorkerControlInfo | int, status: WorkerStatus):
        worker_id = worker.worker_id if isinstance(worker, WorkerControlInfo) else worker
        for callback in self.worker_status_change_callbacks:
            try:
                await callback(worker_id, status)
            except Exception as e:
                logger.error(f"Error in worker status change callback for {worker}: {e}")
        
    async def send_command(self, worker: WorkerControlInfo, command: str, data: dict[str, Any]=None) -> bool:
        if data is None:
            data = {}
        if worker.worker_id not in self.connections:
            logger.error(f"No active WebSocket connection to {worker} for sending command")
            return False
        ws = self.connections[worker.worker_id]
        try:
            payload = json.dumps({"command": command, "data": data})
            await ws.send(payload)
            logger.debug(f"Sent command to {worker}: {payload}")
            return True
        except ConnectionClosed:
            logger.error(f"WebSocket connection to {worker} is closed, cannot send command")
            await self._handle_disconnection(worker)
            return False
        except Exception as e:
            logger.error(f"Failed to send command to {worker}: {e}")
            return False
    
    def is_connected(self, worker: WorkerControlInfo | int) -> bool:
        worker_id = worker.worker_id if isinstance(worker, WorkerControlInfo) else worker
        return worker_id in self.connections and self.connections[worker_id].state != websockets.protocol.State.CLOSING and self.connections[worker_id].state != websockets.protocol.State.CLOSED
    
    async def disconnect_worker(self, worker: WorkerControlInfo | int):
        worker_id = worker.worker_id if isinstance(worker, WorkerControlInfo) else worker
        if worker_id in self.connections:
            await self.connections[worker_id].close()
            await self._handle_disconnection(worker, reconnect=False)
            logger.info(f"Disconnected WebSocket connection to {worker}")
    
    async def disconnect_all(self):
        for worker_id in list(self.connections.keys()):
            await self.disconnect_worker(worker_id)
        logger.info("Disconnected all WebSocket connections to workers")

    def register_status_change_callback(self, callback: Callable[[int, WorkerStatus], Coroutine[Any, Any, Any]]):
        # Callback signature: async def callback(worker: int, status: WorkerStatus)
        self.worker_status_change_callbacks.append(callback)
    
