"""
Controller main entry point.

Brings up:
* Network plane (dnsmasq DHCP + optional hostapd AP)
* SQLite database
* INA226 power-monitor poller
* Two FastAPI servers — control (8001) and data (8002)
* WorkersWebSocketManager (controller-side WS client to each worker)
* ExperimentManager
* NiceGUI Web UI on :8080
* Heartbeat watchdog coroutine
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

import uvicorn

from controller._paths import ensure_runtime_dirs
from controller.api.control_api import ControlDeps, make_control_app
from controller.api.data_api import DataDeps, make_data_app
from controller.auto_onboard import AutoOnboarder
from controller.calibration import CalibrationManager
from controller.cluster_state import ClusterState, WorkerStats
from controller.database import Database
from controller.experiment_manager import ExperimentManager
from controller.network_manager import ControllerNetworkManager
from controller.network_manager_mock import MockControllerNetworkManager
from controller.power_monitor import PowerMonitor
from controller.power_monitor_mock import MockPowerMonitor
from controller.streaming import SuperResPipeline
from controller.workers_websocket_manager import WorkersWebSocketManager
from shared.config import load_config
from shared.models import (
    WorkerControlInfo,
    WorkerHeartbeat,
    WorkerRegistration,
    WorkerStatus,
)
from shared.util import generate_identifier, get_cpu_serial

logger = logging.getLogger(__name__)


# =============================================================================
# Top-level controller
# =============================================================================
class Controller:
    def __init__(self, config: dict[str, Any], skip_network: bool = False):
        self.config = config
        self.skip_network = skip_network
        self.mock = bool(config.get("mock", {}).get("enabled", False))
        self.serial = get_cpu_serial()
        self.identifier = generate_identifier(self.serial)
        self.started_at = time.time()

        self.state = ClusterState()
        self.db = Database(self.config["database"]["path"])
        if self.mock:
            logger.warning("MOCK mode ON — using loopback network + synthetic power")
            self.network = MockControllerNetworkManager(config)
        else:
            self.network = ControllerNetworkManager(config)
        self.workers_ws = WorkersWebSocketManager(config)
        # Wire state into the WS manager so it can refuse to flip a
        # stale-heartbeat worker back to ACTIVE on a TCP-only reconnect.
        self.workers_ws.set_state(self.state)
        self.experiment = ExperimentManager(config, self.state, self.db)
        if config["power_monitor"]["enabled"]:
            self.power_monitor = (
                MockPowerMonitor(config, self.db, self.state) if self.mock
                else PowerMonitor(config, self.db, self.state)
            )
        else:
            self.power_monitor = None

        self.calibration: CalibrationManager | None = (
            CalibrationManager(config, self.state, self.db, self.power_monitor)
            if self.power_monitor is not None else None
        )

        # Live demo pipelines — driven by the /live page, independent of the
        # batch experiment flow.
        self.sr_pipeline = SuperResPipeline(self.experiment, self.state)

        # Auto-onboard: SSH-deploy fresh Pis to the cluster on first boot.
        # Started in run() if `[auto_onboard].enabled` is true.
        project_root = Path(__file__).resolve().parent.parent
        self.auto_onboarder = AutoOnboarder(self.config, self.state, project_root)

        # Three uvicorn servers driven from the same asyncio loop:
        # control plane, data plane, and the NiceGUI dashboard.
        self._control_server: uvicorn.Server | None = None
        self._data_server: uvicorn.Server | None = None
        self._web_server: uvicorn.Server | None = None
        self._stop_event = asyncio.Event()

        # Soft-restart flow (triggered by /api/restart):
        # _skip_shutdown    — skip network.shutdown() in the finally block
        #                     so hostapd/dnsmasq keep running across the
        #                     restart, AP never drops
        # _reexec_pending   — tells main() to os.execv into a new copy of
        #                     ourselves with --skip-network, instead of
        #                     just exiting. The old hostapd/dnsmasq become
        #                     orphans (PPID=1) that the new controller
        #                     adopts via pgrep on next real shutdown.
        self._skip_shutdown: bool = False
        self._reexec_pending: bool = False

    # =========================================================================
    # Bootstrapping
    # =========================================================================
    def initialize_network(self) -> None:
        if self.skip_network:
            logger.warning("Skipping network initialization (--skip-network)")
            # Try to adopt any dnsmasq/hostapd that the previous python
            # instance left running across a soft-restart. If nothing's
            # there (developer started by hand, no prior controller),
            # adopt_existing is a no-op and shutdown later won't try to
            # kill phantoms.
            try:
                self.network.adopt_existing()
            except AttributeError:
                # Mock network manager in older code paths might not
                # implement it. Harmless.
                pass
            return
        wifi = bool(self.config["controller"].get("ap_enabled", True))
        self.network.initialize(initialize_wifi=wifi)

    async def register_worker(self, heartbeat: WorkerHeartbeat,
                              worker_id: int = -1) -> int:
        """Promote a pending worker to registered + open a control WS to it."""
        if heartbeat.serial not in self.state.pending_workers:
            logger.error("Cannot register unknown serial=%s", heartbeat.serial)
            return -1

        if worker_id < 0:
            worker_id = self.state.assign_worker_id()

        registration = WorkerRegistration(
            serial=heartbeat.serial,
            hardware_identifier=heartbeat.hardware_identifier,
            control_ip=heartbeat.control_ip_address,
            data_ip=heartbeat.data_ip_address,
            data_plane=heartbeat.data_plane,
            timestamp=int(time.time()),
            status=WorkerStatus.REGISTERED,
            control_port=(heartbeat.control_port
                          or self.config["worker"]["control_port"]),
            data_port=(heartbeat.data_port
                       or self.config["worker"]["data_port"]),
            has_hailo=heartbeat.has_hailo,
            engine=("hailo" if heartbeat.has_hailo else "onnx"
                    if heartbeat.has_hailo is not None else None),
        )
        self.state.add_registered(worker_id, registration)
        self.state.remove_pending(heartbeat.serial)

        info = WorkerControlInfo(
            worker_id=worker_id,
            control_ip=registration.control_ip,
            serial=registration.serial,
            identifier=registration.hardware_identifier,
            control_port=registration.control_port,
        )
        if await self.workers_ws.connect_to_worker(info):
            self.state.set_status(worker_id, WorkerStatus.ACTIVE)
            logger.info("Registered & WS-connected: %s", info)
        else:
            self.state.set_status(worker_id, WorkerStatus.RECONNECTING)
            logger.warning("Registered but WS failed: %s", info)

        # If this worker was calibrated in a previous session, restore its
        # INA226 binding now that we know its current worker_id.
        if self.calibration is not None:
            try:
                self.calibration.load_persisted()
            except Exception as e:
                logger.error("load_persisted bindings failed: %s", e)

        return worker_id

    # =========================================================================
    # Watchdog & callbacks
    # =========================================================================
    async def _on_status_change(self, worker_id: int, status: WorkerStatus) -> None:
        self.state.set_status(worker_id, status)

    async def _monitor_heartbeats(self) -> None:
        cluster = self.config["cluster"]
        interval = float(cluster.get("monitor_interval", 10))
        timeout = float(cluster.get("heartbeat_timeout", 15))
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(interval)
                now = int(time.time())
                threshold = now - timeout
                for wid, reg in list(self.state.registered_workers.items()):
                    if reg.timestamp < threshold:
                        cur_status = (reg.status if isinstance(reg.status, str)
                                      else reg.status.value)
                        if cur_status != WorkerStatus.INACTIVE.value:
                            logger.warning(
                                "Worker %d '%s' heartbeat timeout (last=%ds ago)",
                                wid, reg.hardware_identifier, now - reg.timestamp)
                            self.state.set_status(wid, WorkerStatus.INACTIVE)
                            # Drop distribution tracking — when this worker
                            # comes back the user must re-distribute. The
                            # alternative (keep entries) would silently let
                            # experiments target a worker whose engine state
                            # may have been lost across the reconnect.
                            self.state.clear_distribution_for_worker(wid)
                            try:
                                await self.workers_ws.disconnect_worker(wid)
                            except Exception:
                                pass
                    elif (reg.status == WorkerStatus.INACTIVE.value
                          or reg.status == WorkerStatus.INACTIVE):
                        info = WorkerControlInfo(
                            worker_id=wid, control_ip=reg.control_ip,
                            serial=reg.serial,
                            identifier=reg.hardware_identifier,
                            control_port=reg.control_port,
                        )
                        logger.info("Reattempting WS to %s", info)
                        try:
                            await self.workers_ws.connect_to_worker(info)
                        except Exception as e:
                            logger.error("WS reconnect to %d failed: %s", wid, e)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("monitor_heartbeats error: %s", e)

    async def _periodic_summary(self) -> None:
        # 5 minutes — used to be 30s but the steady stream cluttered the
        # journal during long demos / overnight runs.
        SUMMARY_INTERVAL_S = 300
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(SUMMARY_INTERVAL_S)
                now = int(time.time())
                logger.info("--- Worker status summary @ %d ---", now)
                for serial, hb in self.state.pending_workers.items():
                    logger.info("  pending: %s (%s) %ds ago",
                                hb.hardware_identifier, serial, now - hb.timestamp)
                for wid, reg in self.state.registered_workers.items():
                    status_str = (reg.status if isinstance(reg.status, str)
                                  else reg.status.value)
                    logger.info("  worker %d '%s' status=%s last=%ds ago",
                                wid, reg.hardware_identifier, status_str,
                                now - reg.timestamp)
            except asyncio.CancelledError:
                break

    # =========================================================================
    # Servers
    # =========================================================================
    def _build_apps(self):
        control_deps = ControlDeps(
            config=self.config,
            state=self.state,
            identifier=self.identifier,
            serial=self.serial,
            started_at=self.started_at,
            register_worker_coro=self.register_worker,
            calibration=self.calibration,
            power_monitor=self.power_monitor,
            workers_ws=self.workers_ws,
            auto_onboarder=self.auto_onboarder,
            controller_obj=self,
        )
        data_deps = DataDeps(
            config=self.config,
            identifier=self.identifier,
            serial=self.serial,
            experiment_manager=self.experiment,
            state=self.state,
        )
        control_app = make_control_app(control_deps)
        data_app = make_data_app(data_deps)
        return control_app, data_app

    async def _serve_uvicorn(self, app, port: int) -> uvicorn.Server:
        config = uvicorn.Config(app, host="0.0.0.0", port=port,
                                log_level=self.config["controller"]["log_level"].lower(),
                                lifespan="on")
        server = uvicorn.Server(config)
        # Run server.serve() concurrently
        asyncio.create_task(server.serve())
        return server

    async def _start_web_ui(self) -> None:
        """Mount NiceGUI on its own ASGI app and serve it via uvicorn.

        NiceGUI on a daemon thread used to silently hang because ui.run()
        assumes main-thread signal-handler install. Mounting via ui.run_with
        lets uvicorn run it on the controller's main asyncio loop.
        """
        try:
            from controller.web_ui import build_web_app
        except Exception as e:
            logger.error("Web UI import failed: %s", e)
            return
        web_app = build_web_app(self)
        if web_app is None:
            logger.warning("NiceGUI unavailable — dashboard disabled.")
            return
        port = int(self.config["controller"]["web_port"])
        self._web_server = await self._serve_uvicorn(web_app, port)
        logger.info("Web UI mounted at :%d (NiceGUI)", port)

    # =========================================================================
    # Lifecycle
    # =========================================================================
    async def run(self) -> None:
        # Hook WS event/status callbacks into state
        self.workers_ws.register_status_change_callback(self._on_status_change)

        # Build & launch FastAPI servers
        control_app, data_app = self._build_apps()
        self._control_server = await self._serve_uvicorn(
            control_app, self.config["controller"]["control_port"]
        )
        self._data_server = await self._serve_uvicorn(
            data_app, self.config["controller"]["data_port"]
        )

        # Power monitor
        if self.power_monitor is not None:
            self.power_monitor.start()

        # Web UI (uvicorn-served on the same asyncio loop)
        await self._start_web_ui()

        # Background coroutines
        watchdog = asyncio.create_task(self._monitor_heartbeats())
        summary = asyncio.create_task(self._periodic_summary())

        # Auto-onboard: only starts if `[auto_onboard].enabled = true`. Sits
        # on its own asyncio task and polls dnsmasq leases.
        self.auto_onboarder.start()

        logger.info("Controller '%s' (serial=%s) running.", self.identifier, self.serial)
        print(f"[controller] running. identifier={self.identifier} "
              f"control={self.config['controller']['control_port']} "
              f"data={self.config['controller']['data_port']} "
              f"web={self.config['controller']['web_port']}")

        try:
            await self._stop_event.wait()
        finally:
            watchdog.cancel()
            summary.cancel()
            await self.auto_onboarder.stop()
            await self.experiment.stop()
            if self.power_monitor is not None:
                self.power_monitor.stop()
            await self.workers_ws.disconnect_all()
            for srv in (self._control_server, self._data_server, self._web_server):
                if srv is not None:
                    srv.should_exit = True
            if self._skip_shutdown:
                # Soft-restart path — keep hostapd/dnsmasq alive across the
                # exec so the AP doesn't blink. The new python instance will
                # start with --skip-network and adopt the orphaned helpers.
                logger.warning(
                    "Skipping network.shutdown() (soft-restart in progress)."
                )
            else:
                self.network.shutdown()
            self.db.close()

    def stop(self) -> None:
        self._stop_event.set()

    def request_soft_restart(self, delay_s: float = 1.5) -> None:
        """Trigger an in-process restart.

        Sets two flags + schedules the asyncio loop to break:
          * _skip_shutdown   — finally block won't kill hostapd/dnsmasq
          * _reexec_pending  — main() will os.execv into a fresh python after
                               run() returns (instead of plain exit)

        ``delay_s`` lets the HTTP response that triggered this return cleanly
        before we tear the asyncio loop down. 1.5 s is plenty.
        """
        self._skip_shutdown = True
        self._reexec_pending = True

        loop = asyncio.get_event_loop()
        loop.call_later(delay_s, self._stop_event.set)
        logger.warning(
            "Soft-restart requested. python will re-exec in %.1fs; "
            "hostapd/dnsmasq stay alive (AP not dropped).", delay_s,
        )


# =============================================================================
# Module entrypoint
# =============================================================================
def _setup_logging(config: dict[str, Any]) -> None:
    level_name = config["controller"]["log_level"].upper()
    level = getattr(logging, level_name, logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=level, format=fmt,
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(config["controller"]["log_file"])],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FYP cluster controller. By default brings up the WiFi "
                    "AP on wlan0 so the experiment PC can join the cluster "
                    "WiFi without an external router.",
    )
    parser.add_argument("--config", default=None,
                        help="Path to config.toml (default: <repo>/config.toml).")
    parser.add_argument("--skip-network", action="store_true",
                        help="Don't bring up dnsmasq / hostapd / nmcli at all "
                             "(useful when iterating on the controller).")

    # Mutually-exclusive WiFi mode overrides. Without any of these the
    # controller honours config.toml — which defaults to wifi_mode = 'ap',
    # so the AP comes up automatically on a fresh deploy.
    wifi_grp = parser.add_mutually_exclusive_group()
    wifi_grp.add_argument(
        "--ap", action="store_true",
        help="Force WiFi mode = 'ap' (default behaviour; explicit override).",
    )
    wifi_grp.add_argument(
        "--no-ap", "--no-wifi", action="store_true", dest="no_ap",
        help="Disable the WiFi AP at startup. wlan0 is left untouched. The "
             "experiment PC must join the cluster via the wired switch.",
    )
    wifi_grp.add_argument(
        "--client", nargs=2, metavar=("SSID", "PASSWORD"), default=None,
        help="Force WiFi client mode — controller joins the given existing "
             "WiFi network instead of hosting an AP.",
    )

    onboard_grp = parser.add_mutually_exclusive_group()
    onboard_grp.add_argument(
        "--auto-onboard", action="store_true",
        help="Enable auto-onboard: SSH to every new DHCP client with default "
             "Pi credentials (pi/raspberry), deploy the worker payload, and "
             "start fyp-worker.service. Default OFF.",
    )
    onboard_grp.add_argument(
        "--no-auto-onboard", action="store_true",
        help="Force auto-onboard off, even if config.toml enables it.",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    # CLI mode overrides take precedence over config.toml.
    if args.ap:
        config["controller"]["wifi_mode"] = "ap"
        config["controller"]["ap_enabled"] = True
    elif args.no_ap:
        config["controller"]["wifi_mode"] = "off"
        config["controller"]["ap_enabled"] = False
    elif args.client is not None:
        ssid, password = args.client
        config["controller"]["wifi_mode"] = "client"
        config["controller"]["ap_enabled"] = False
        config.setdefault("network", {})["wifi_client_ssid"] = ssid
        config["network"]["wifi_client_password"] = password

    # Auto-onboard CLI overrides take precedence over config.toml.
    if args.auto_onboard:
        config.setdefault("auto_onboard", {})["enabled"] = True
    elif args.no_auto_onboard:
        config.setdefault("auto_onboard", {})["enabled"] = False

    _setup_logging(config)
    # Bracketing banner — easy to grep for in `journalctl -b 0`. Every
    # restart prints exactly one BEGIN line containing all the user-visible
    # decisions made on this boot, so you don't have to chase logger.info
    # lines across the journal to confirm what mode the controller booted in.
    wifi_mode = config["controller"].get("wifi_mode", "(unset)")
    auto_on = bool(config.get("auto_onboard", {}).get("enabled"))
    print("============================================================")
    print(f"  fyp-controller starting")
    print(f"  WiFi mode    : {wifi_mode}")
    print(f"  Auto-onboard : {'ON' if auto_on else 'off'}")
    print(f"  Config file  : {args.config or '(default config.toml)'}")
    print("============================================================")
    logger.warning("=== fyp-controller boot: wifi=%s auto_onboard=%s ===",
                   wifi_mode, "ON" if auto_on else "off")

    # Make sure demo/ + recordings/ exist before any upload / record happens.
    ensure_runtime_dirs()

    ctrl = Controller(config, skip_network=args.skip_network)
    ctrl.initialize_network()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _signal(_sig, _frame):
        logger.info("Received signal, shutting down...")
        loop.call_soon_threadsafe(ctrl.stop)

    signal.signal(signal.SIGINT, _signal)
    signal.signal(signal.SIGTERM, _signal)

    try:
        loop.run_until_complete(ctrl.run())
    finally:
        loop.close()

    # Soft-restart path: replace the current python image with a fresh one,
    # passing --skip-network so the new controller doesn't try to re-init
    # the dnsmasq / hostapd we deliberately left running. PID stays the
    # same — systemd has no idea we restarted, AP doesn't blink, browser
    # WebSocket reconnects within ~2 s.
    #
    # We pass the PIDs of the surviving subprocesses through the env so the
    # new controller's adopt_existing() takes them over without scanning.
    # Falls back to pgrep if the env vars are missing for any reason.
    if ctrl._reexec_pending:
        import os as _os
        argv = [sys.executable, "-m", "controller.controller", "--skip-network"]
        for a in sys.argv[1:]:
            if a in ("--skip-network",):
                continue
            argv.append(a)

        env = _os.environ.copy()
        for proc, env_var in (
            (getattr(ctrl.network, "dnsmasq_process", None), "FYP_DNSMASQ_PID"),
            (getattr(ctrl.network, "hostapd_process", None), "FYP_HOSTAPD_PID"),
        ):
            try:
                if proc is not None and proc.poll() is None:
                    env[env_var] = str(proc.pid)
            except Exception:
                pass

        logger.warning(
            "Soft-restart: execvp -> %s   adopting=[%s]",
            " ".join(argv),
            ", ".join(f"{k}={v}" for k, v in env.items()
                      if k.startswith("FYP_")) or "(none)",
        )
        _os.execvpe(sys.executable, argv, env)


if __name__ == "__main__":
    main()
