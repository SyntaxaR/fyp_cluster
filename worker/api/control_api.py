"""
Worker CONTROL plane FastAPI app.

Exposes:
    /worker_ws       — Inbound WebSocket from controller (JSON commands)
    /api/health      — Lightweight liveness probe (curl-friendly)
    /api/connectivity_test  — Symmetric to controller's endpoint
"""
from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, FastAPI, Request, WebSocket, WebSocketDisconnect

from shared.models import ConnectionType, ConnectivityTestResponse

logger = logging.getLogger(__name__)


class WorkerWebSocketServer:
    """Routes JSON commands sent by the controller to async handler functions."""

    def __init__(self):
        self.current_websocket: WebSocket | None = None
        self.command_handlers: dict[str, Callable[[dict], Awaitable[Any]]] = {}

    def register_handler(self, command: str,
                         handler: Callable[[dict], Awaitable[Any]]) -> None:
        self.command_handlers[command] = handler

    async def _send_event(self, event: str, data: dict[str, Any] | None = None) -> None:
        if self.current_websocket is None:
            return
        try:
            await self.current_websocket.send_text(
                json.dumps({"event": event, "data": data or {}})
            )
        except Exception as e:
            logger.debug("event send failed: %s", e)

    async def handle_connection(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.current_websocket = websocket
        logger.info("Controller WS connected")
        try:
            while True:
                message = await websocket.receive_text()
                logger.debug("WS recv: %s", message)
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning("Bad JSON from controller WS: %r", message[:200])
                    continue
                command = payload.get("command", "")
                data = payload.get("data") or {}
                if not command:
                    continue
                handler = self.command_handlers.get(command)
                if handler is None:
                    logger.warning("Unknown command '%s'", command)
                    continue
                try:
                    result = await handler(data)
                    if result is not None:
                        await self._send_event(f"{command}_ack",
                                               {"command": command, "result": result})
                except Exception as e:
                    logger.error("handler '%s' failed: %s", command, e)
                    await self._send_event(f"{command}_error",
                                           {"command": command, "error": str(e)})
        except WebSocketDisconnect:
            logger.info("Controller WS disconnected")
        except Exception as e:
            logger.error("WS error: %s", e)
        finally:
            self.current_websocket = None
            try:
                await websocket.close()
            except Exception:
                pass


def make_control_app(deps: "WorkerControlDeps") -> FastAPI:
    app = FastAPI(title="Worker Control API", version="1.0.0")
    router = APIRouter()
    ws_server = deps.ws_server

    @app.websocket("/worker_ws")
    async def _ws_endpoint(websocket: WebSocket):
        await ws_server.handle_connection(websocket)

    @router.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "worker_id": deps.get_worker_id(),
            "identifier": deps.identifier,
            "engine": deps.get_engine_name(),
            "loaded_model": deps.get_loaded_model(),
        }

    @router.get("/api/connectivity_test")
    async def connectivity_test(request: Request) -> ConnectivityTestResponse:
        host = request.client.host if request.client else ""
        eth_subnet = deps.config["network"]["ethernet_subnet"]
        wifi_subnet = deps.config["network"]["wifi_subnet"]
        if host.startswith(eth_subnet):
            plane = ConnectionType.ETHERNET
        elif host.startswith(wifi_subnet):
            plane = ConnectionType.WIFI
        else:
            plane = ConnectionType.INVALID
        return ConnectivityTestResponse(
            from_identifier=deps.identifier, message="ok", plane=plane,
        )

    app.include_router(router)
    return app


class WorkerControlDeps:
    def __init__(self, config: dict[str, Any], identifier: str,
                 ws_server: WorkerWebSocketServer,
                 get_worker_id: Callable[[], int],
                 get_engine_name: Callable[[], str | None],
                 get_loaded_model: Callable[[], str | None]):
        self.config = config
        self.identifier = identifier
        self.ws_server = ws_server
        self.get_worker_id = get_worker_id
        self.get_engine_name = get_engine_name
        self.get_loaded_model = get_loaded_model
