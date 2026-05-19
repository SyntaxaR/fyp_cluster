"""
Manages outbound WebSocket connections from controller to each worker.

Lifecycle:
  connect_to_worker()  -> ws://<worker_ip>:<control_port>/worker_ws
  _receive_loop()      -> pull async events from the worker
  _handle_disconnection() -> reconnect_worker (bounded retries)
  send_command()       -> JSON {"command": ..., "data": ...}
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Coroutine

import websockets
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, WebSocketException

from shared.models import WorkerControlInfo, WorkerStatus

logger = logging.getLogger(__name__)


class WorkersWebSocketManager:
    def __init__(self, config: dict[str, Any]):
        self.config = config

        cluster = config.get("cluster", {})
        self.reconnect_interval: float = float(cluster.get("ws_reconnect_interval", 5))
        self.max_reconnect_attempts: int = int(cluster.get("ws_max_reconnect", 5))
        self.connection_timeout: float = 5.0
        # Heartbeat freshness threshold (mirrors monitor's heartbeat_timeout).
        # connect_to_worker uses this to refuse to flip a stale-heartbeat
        # worker back to ACTIVE on a TCP-only reconnect.
        self._hb_fresh_threshold_s: float = float(
            cluster.get("heartbeat_timeout", 15)
        )
        # Set by Controller after construction so we can read worker
        # heartbeat timestamps without taking a circular import on
        # cluster_state at module load.
        self._state: Any = None
        # Default port (used as a fallback if a WorkerControlInfo arrives
        # without an explicit control_port — shouldn't normally happen now
        # that heartbeats carry the worker's actual port).
        self.default_ws_port: int = config["worker"]["control_port"]

        self.connections: dict[int, ClientConnection] = {}
        self.connection_tasks: dict[int, asyncio.Task] = {}
        # Reconnect tasks spawned by `_handle_disconnection`. Tracked so a
        # heartbeat-driven `request_reconnect_now` can cancel a slow
        # retry-with-sleep loop and re-try immediately.
        self.reconnect_tasks: dict[int, asyncio.Task] = {}
        self.worker_status_change_callbacks: list[
            Callable[[int, WorkerStatus], Coroutine[Any, Any, Any]]
        ] = []
        self.event_callbacks: list[
            Callable[[int, str, dict[str, Any]], Coroutine[Any, Any, Any]]
        ] = []

    def set_state(self, state) -> None:
        """Inject a ClusterState reference. Used to gate the WS-reconnect
        ACTIVE notification on heartbeat freshness."""
        self._state = state

    def _heartbeat_is_fresh(self, worker_id: int) -> bool:
        """True iff the worker has heartbeated within the freshness window.

        Falls back to True when no state has been wired (preserves the
        old "TCP reconnect = ACTIVE" semantics), so the change is opt-in
        from controller.py.
        """
        if self._state is None:
            return True
        reg = self._state.get_registered(worker_id)
        if reg is None:
            return False
        ts = int(getattr(reg, "timestamp", 0) or 0)
        if ts <= 0:
            return False
        import time as _time
        return (_time.time() - ts) < self._hb_fresh_threshold_s

    # =========================================================================
    # Connection management
    # =========================================================================
    async def connect_to_worker(self, worker: WorkerControlInfo) -> bool:
        port = getattr(worker, "control_port", 0) or self.default_ws_port
        ws_uri = f"ws://{worker.control_ip}:{port}/worker_ws"
        logger.info("Connecting WS to %s at %s", worker, ws_uri)
        try:
            ws = await asyncio.wait_for(connect(ws_uri), timeout=self.connection_timeout)
            self.connections[worker.worker_id] = ws
            self.connection_tasks[worker.worker_id] = asyncio.create_task(
                self._receive_loop(worker, ws)
            )
            # ACTIVE is gated on the worker also having a fresh heartbeat
            # — a stale-heartbeat worker whose WS server still answers
            # TCP must NOT be flipped back to ACTIVE just because we
            # reconnected. Otherwise the dashboard shows "active" for a
            # worker that hasn't reported anything in minutes (the
            # python service is wedged but the OS network stack is fine).
            # The watchdog sets ACTIVE on its own when heartbeats arrive
            # again via the elif INACTIVE -> reconnect branch.
            if self._heartbeat_is_fresh(worker.worker_id):
                await self._notify_status_change(worker, WorkerStatus.ACTIVE)
            else:
                logger.info(
                    "WS reconnected to %s but heartbeat is stale — "
                    "leaving status as INACTIVE until a fresh heartbeat "
                    "arrives.", worker,
                )
            return True
        except asyncio.TimeoutError:
            logger.error("WS connect to %s timed out", worker)
            return False
        except WebSocketException as e:
            logger.error("WS connect to %s failed: %s", worker, e)
            return False

    async def _receive_loop(self, worker: WorkerControlInfo, ws: ClientConnection) -> None:
        try:
            async for message in ws:
                await self._dispatch_event(worker, message)
        except ConnectionClosed:
            logger.warning("WS to %s closed", worker)
        except Exception as e:
            logger.error("WS receive error from %s: %s", worker, e)
        finally:
            await self._handle_disconnection(worker)

    async def _dispatch_event(self, worker: WorkerControlInfo, message: str | bytes) -> None:
        if isinstance(message, bytes):
            try:
                message = message.decode("utf-8")
            except UnicodeDecodeError:
                logger.warning("Discarding binary WS message from %s", worker)
                return
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            logger.warning("Bad JSON from %s: %r", worker, message[:200])
            return

        event = payload.get("event") or payload.get("command")
        data = payload.get("data") or {}
        if not event:
            return
        for cb in self.event_callbacks:
            try:
                await cb(worker.worker_id, event, data)
            except Exception as e:
                logger.error("event callback error: %s", e)

    async def _handle_disconnection(self, worker: WorkerControlInfo | int,
                                    reconnect: bool = True) -> None:
        await self._notify_status_change(worker, WorkerStatus.INACTIVE)
        worker_id = worker.worker_id if isinstance(worker, WorkerControlInfo) else worker

        ws = self.connections.pop(worker_id, None)
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
        task = self.connection_tasks.pop(worker_id, None)
        if task is not None:
            task.cancel()

        if reconnect and self.max_reconnect_attempts > 0 and isinstance(worker, WorkerControlInfo):
            await self._notify_status_change(worker, WorkerStatus.RECONNECTING)
            self.reconnect_tasks[worker.worker_id] = asyncio.create_task(
                self._reconnect_worker(worker)
            )

    async def reconnect_worker(self, worker: WorkerControlInfo) -> None:
        asyncio.create_task(self._reconnect_worker(worker))

    async def _reconnect_worker(self, worker: WorkerControlInfo) -> None:
        try:
            for attempt in range(1, self.max_reconnect_attempts + 1):
                logger.info("Reconnect attempt %d/%d for %s",
                            attempt, self.max_reconnect_attempts, worker)
                if await self.connect_to_worker(worker):
                    return
                await asyncio.sleep(self.reconnect_interval)
            logger.error("Reconnect to %s failed after %d attempts",
                         worker, self.max_reconnect_attempts)
            await self._notify_status_change(worker, WorkerStatus.INACTIVE)
        except asyncio.CancelledError:
            # Cancelled by `request_reconnect_now` — caller is taking over
            # the connect attempt. Don't notify anything; caller does.
            logger.info("Reconnect loop for %s cancelled (heartbeat-driven "
                        "fast reconnect taking over)", worker)
            raise
        finally:
            # Drop our slot so heartbeat-driven reconnects don't try to
            # cancel a finished task.
            self.reconnect_tasks.pop(worker.worker_id, None)

    async def request_reconnect_now(self, worker: WorkerControlInfo) -> bool:
        """Heartbeat-driven fast reconnect.

        The slow `_reconnect_worker` loop sleeps `reconnect_interval`
        seconds between attempts, which means a worker that came back
        within those gaps stays in RECONNECTING until the next attempt
        happens to fire. Heartbeats are a much stronger "alive right now"
        signal — when one arrives, the worker's HTTP server is definitely
        bound, so its WS server is too. Cancel the slow retry loop and
        try once immediately.

        Returns True if the connect succeeded, False otherwise. A False
        return is fine — `_handle_disconnection` will respawn the slow
        loop on the next disconnect, and another heartbeat will fire
        another fast attempt.
        """
        # Already connected? nothing to do.
        if self.is_connected(worker.worker_id):
            return True

        # Cancel any in-flight slow-retry loop so we don't double-connect.
        old = self.reconnect_tasks.pop(worker.worker_id, None)
        if old is not None and not old.done():
            old.cancel()
            try:
                await old
            except (asyncio.CancelledError, Exception):
                pass

        ok = await self.connect_to_worker(worker)
        if not ok:
            # Re-arm the slow retry loop so we still get periodic
            # attempts if heartbeats stop arriving.
            await self._notify_status_change(worker, WorkerStatus.RECONNECTING)
            self.reconnect_tasks[worker.worker_id] = asyncio.create_task(
                self._reconnect_worker(worker)
            )
        return ok

    async def _notify_status_change(self, worker: WorkerControlInfo | int,
                                    status: WorkerStatus) -> None:
        worker_id = worker.worker_id if isinstance(worker, WorkerControlInfo) else worker
        for cb in self.worker_status_change_callbacks:
            try:
                await cb(worker_id, status)
            except Exception as e:
                logger.error("status callback error: %s", e)

    # =========================================================================
    # Public API
    # =========================================================================
    def register_status_change_callback(
        self, cb: Callable[[int, WorkerStatus], Coroutine[Any, Any, Any]]
    ) -> None:
        self.worker_status_change_callbacks.append(cb)

    def register_event_callback(
        self, cb: Callable[[int, str, dict[str, Any]], Coroutine[Any, Any, Any]]
    ) -> None:
        self.event_callbacks.append(cb)

    async def send_command(
        self, worker_id: int, command: str, data: dict[str, Any] | None = None
    ) -> bool:
        ws = self.connections.get(worker_id)
        if ws is None:
            logger.error("No WS connection for worker %d", worker_id)
            return False
        try:
            await ws.send(json.dumps({"command": command, "data": data or {}}))
            return True
        except ConnectionClosed:
            logger.error("WS to worker %d closed; cannot send '%s'", worker_id, command)
            return False
        except Exception as e:
            logger.error("Failed to send '%s' to worker %d: %s", command, worker_id, e)
            return False

    def is_connected(self, worker_id: int) -> bool:
        ws = self.connections.get(worker_id)
        if ws is None:
            return False
        return ws.state not in (
            websockets.protocol.State.CLOSING,
            websockets.protocol.State.CLOSED,
        )

    async def disconnect_worker(self, worker_id: int) -> None:
        ws = self.connections.get(worker_id)
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass
            await self._handle_disconnection(worker_id, reconnect=False)

    async def disconnect_all(self) -> None:
        for wid in list(self.connections.keys()):
            await self.disconnect_worker(wid)
