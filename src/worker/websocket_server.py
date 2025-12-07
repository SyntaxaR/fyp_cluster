import asyncio
import logging
from fastapi import WebSocket, WebSocketDisconnect
import json

logger = logging.getLogger(__name__)

class WorkerWebSocketServer:
    def __init__(self, config: dict[str, any]):
        self.config = config
        self.current_websocket: WebSocket | None = None
        self.command_handlers: dict[str, callable] = {}

    def register_handler(self, command: str, handler: callable):
        self.command_handlers[command] = handler

    async def handle_connection(self, websocket: WebSocket):
        await websocket.accept()
        self.current_websocket = websocket
        logger.info("New WebSocket connection established with controller")
        print("New WebSocket connection established with controller")
        try:
            while True:
                message = await websocket.receive_text()
                logger.debug(f"Received WebSocket message: {message}")
                try:
                    payload: dict[str, any] = json.loads(message)
                    command: str = payload.get("command", "")
                    data: dict[str, any] = payload.get("data", {})
                    if not command:
                        logger.warning("Received WebSocket message without 'command' field")
                        continue
                    if command in self.command_handlers:
                        print(f"Handling command: {command} with data: {data}")
                        await self.command_handlers[command](data)
                    else:
                        logger.warning(f"Received unknown command '{command}' via WebSocket")
                except json.JSONDecodeError:
                    logger.error("Failed to decode WebSocket message as JSON")
        except WebSocketDisconnect:
            logger.warning("WebSocket connection to controller disconnected")
            print("WebSocket connection to controller disconnected")
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            print(f"WebSocket connection error: {e}")
        finally:
            self.current_websocket = None
            await websocket.close()
            logger.info("WebSocket connection closed")
            print("WebSocket connection closed")