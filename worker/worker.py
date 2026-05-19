"""
Worker main entry point.

Bootstrap order:
    1.  Load local config.toml.
    2.  Bring up ethernet via DHCP.
    3.  Fetch /api/get_config from controller (subnet host .1) and merge it.
    4.  Start two FastAPI servers (control + data).
    5.  Start the heartbeat coroutine; the controller will assign worker_id
        in the heartbeat response.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
import threading
import time
from typing import Any, Optional

import uvicorn

from shared.config import load_config
from shared.util import generate_identifier, get_cpu_serial
from worker.api.control_api import (
    WorkerControlDeps,
    WorkerWebSocketServer,
    make_control_app,
)
from worker.api.data_api import WorkerDataDeps, make_data_app
from worker.bootstrap import fetch_controller_config, merge_remote_config
from worker.network_manager import WorkerNetworkController
from worker.network_manager_mock import MockWorkerNetworkController

logger = logging.getLogger(__name__)


# =============================================================================
# Engine factory
# =============================================================================
def _build_engine(backend: str, model_path: str, adapter_path: Optional[str]):
    """Construct an inference engine instance for the requested backend."""
    if backend == "onnx":
        from worker.inference.engines.onnx_engine import OnnxEngine
        return OnnxEngine(model_path, adapter_path)
    if backend == "hailo":
        try:
            from worker.inference.engines.hailo_engine import HailoEngine
        except ImportError as e:
            raise RuntimeError(
                "Hailo backend requested but hailo_platform is not available. "
                "Install hailo-all (apt) and ensure the Hailo accelerator is "
                "present on this worker."
            ) from e
        return HailoEngine(model_path, adapter_path)
    raise ValueError(f"Unknown backend '{backend}'")


# =============================================================================
# Worker
# =============================================================================
class Worker:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.mock = bool(config.get("mock", {}).get("enabled", False))
        self.serial = get_cpu_serial()
        self.identifier = generate_identifier(self.serial)
        self.worker_id = -1

        # Allow per-process port overrides (so several mock workers can run
        # side-by-side on the same host).
        env_ctrl = os.environ.get("FYP_WORKER_CONTROL_PORT", "").strip()
        env_data = os.environ.get("FYP_WORKER_DATA_PORT", "").strip()
        if env_ctrl.isdigit():
            self.config["worker"]["control_port"] = int(env_ctrl)
        if env_data.isdigit():
            self.config["worker"]["data_port"] = int(env_data)

        if self.mock:
            logger.warning("MOCK mode ON — worker bound to loopback")
            self.network = MockWorkerNetworkController(self.worker_id, config)
        else:
            self.network = WorkerNetworkController(self.worker_id, config)
        self.ws_server = WorkerWebSocketServer()

        self.engine = None
        self.engine_name: Optional[str] = None
        self.loaded_model: Optional[str] = None

        self._control_server: uvicorn.Server | None = None
        self._data_server: uvicorn.Server | None = None
        self._stop_event = asyncio.Event()
        self._heartbeat_task: Optional[asyncio.Task] = None

    # =========================================================================
    # Engine swapping
    # =========================================================================
    def swap_engine(self, model_path: str, adapter_path: Optional[str],
                    backend: str, model_name: str) -> None:
        engine = _build_engine(backend, model_path, adapter_path)
        self.engine = engine
        self.engine_name = backend
        self.loaded_model = model_name
        logger.info("Engine swapped: backend=%s model=%s adapter=%s",
                    backend, model_name, adapter_path)

    # =========================================================================
    # Bootstrap network + remote config
    # =========================================================================
    def initialize_network(self) -> None:
        self.network.initialize()

    def fetch_remote_config(self) -> None:
        prefix = self.config["network"]["ethernet_subnet"]
        port = self.config["controller"]["control_port"]
        host_override = (self.config.get("mock", {}).get("controller_host")
                         if self.mock else None)
        remote = fetch_controller_config(prefix, port, timeout=10,
                                         controller_host=host_override)
        if remote:
            logger.info("Fetched controller config; merging.")
            # Preserve worker-side port overrides (set from env above) so the
            # remote controller config doesn't clobber them when several mock
            # workers share a host.
            ctrl_port = self.config["worker"]["control_port"]
            data_port = self.config["worker"]["data_port"]
            self.config = merge_remote_config(self.config, remote)
            self.config["worker"]["control_port"] = ctrl_port
            self.config["worker"]["data_port"] = data_port
        else:
            logger.warning("Could not fetch controller config; using local copy.")

    # =========================================================================
    # WebSocket command handlers
    # =========================================================================
    def _register_ws_handlers(self) -> None:
        async def switch_to_ethernet(data: dict[str, Any]):
            await asyncio.to_thread(self.network.switch_to_ethernet)
            return {"current_mode": self.network.current_mode.value}

        async def switch_to_wifi(data: dict[str, Any]):
            ssid = data.get("ssid")
            password = data.get("password") or ""   # empty => open AP
            if not ssid:
                raise ValueError("switch_to_wifi requires ssid")
            await asyncio.to_thread(self.network.switch_to_wifi, ssid, password)
            return {"current_mode": self.network.current_mode.value}

        async def shutdown(data: dict[str, Any]):
            self._stop_event.set()
            return {"status": "shutting down"}

        async def reload_engine(data: dict[str, Any]):
            backend = data.get("backend", "onnx")
            model_path = data["model_path"]
            adapter_path = data.get("adapter_path")
            model_name = data.get("model_name", "manual")
            await asyncio.to_thread(
                self.swap_engine, model_path, adapter_path, backend, model_name
            )
            return {"loaded_model": self.loaded_model}

        self.ws_server.register_handler("switch_to_ethernet", switch_to_ethernet)
        self.ws_server.register_handler("switch_to_wifi", switch_to_wifi)
        self.ws_server.register_handler("shutdown", shutdown)
        self.ws_server.register_handler("reload_engine", reload_engine)

    # =========================================================================
    # Heartbeat loop
    # =========================================================================
    async def _heartbeat_loop(self) -> None:
        interval = float(self.config["worker"].get("heartbeat_interval", 5))
        consec_fail = 0
        while not self._stop_event.is_set():
            try:
                ok, assigned = await asyncio.to_thread(
                    self.network.send_control_heartbeat,
                    self.serial, self.identifier, self.worker_id,
                )
                if ok:
                    consec_fail = 0
                    if assigned >= 0 and assigned != self.worker_id:
                        logger.info("Controller assigned worker_id=%d", assigned)
                        self.worker_id = assigned
                        self.network.worker_id = assigned
                else:
                    consec_fail += 1
                    if consec_fail % 5 == 1:
                        logger.warning("Controller did not ack heartbeat (#%d)",
                                       consec_fail)
            except Exception as e:
                logger.error("heartbeat error: %s", e)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    # =========================================================================
    # Servers
    # =========================================================================
    def _build_apps(self):
        ctl = WorkerControlDeps(
            config=self.config, identifier=self.identifier,
            ws_server=self.ws_server,
            get_worker_id=lambda: self.worker_id,
            get_engine_name=lambda: self.engine_name,
            get_loaded_model=lambda: self.loaded_model,
        )
        data = WorkerDataDeps(
            config=self.config, identifier=self.identifier,
            get_worker_id=lambda: self.worker_id,
            get_engine=lambda: self.engine,
            get_engine_name=lambda: self.engine_name,
            get_loaded_model=lambda: self.loaded_model,
            swap_engine=self.swap_engine,
        )
        return make_control_app(ctl), make_data_app(data)

    async def _serve_uvicorn(self, app, port: int) -> uvicorn.Server:
        cfg = uvicorn.Config(app, host="0.0.0.0", port=port,
                             log_level=self.config["worker"]["log_level"].lower(),
                             lifespan="on")
        server = uvicorn.Server(cfg)
        asyncio.create_task(server.serve())
        return server

    # =========================================================================
    # Run loop
    # =========================================================================
    async def run(self) -> None:
        self._register_ws_handlers()
        control_app, data_app = self._build_apps()
        self._control_server = await self._serve_uvicorn(
            control_app, self.config["worker"]["control_port"]
        )
        self._data_server = await self._serve_uvicorn(
            data_app, self.config["worker"]["data_port"]
        )
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info("Worker '%s' (serial=%s) running.", self.identifier, self.serial)
        print(f"[worker] running. identifier={self.identifier} "
              f"control={self.config['worker']['control_port']} "
              f"data={self.config['worker']['data_port']}")

        try:
            await self._stop_event.wait()
        finally:
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
            for srv in (self._control_server, self._data_server):
                if srv is not None:
                    srv.should_exit = True
            self.network.destroy()

    def stop(self) -> None:
        self._stop_event.set()


# =============================================================================
# CLI
# =============================================================================
def _setup_logging(config: dict[str, Any]) -> None:
    level = getattr(logging, config["worker"]["log_level"].upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    log_file = (os.environ.get("FYP_WORKER_LOG_FILE", "").strip()
                or config["worker"]["log_file"])
    logging.basicConfig(
        level=level, format=fmt,
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(log_file)],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-network", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    _setup_logging(config)

    worker = Worker(config)

    if not args.skip_network:
        worker.initialize_network()
        worker.fetch_remote_config()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _sig(_s, _f):
        logger.info("Shutting down worker...")
        loop.call_soon_threadsafe(worker.stop)

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    try:
        loop.run_until_complete(worker.run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
