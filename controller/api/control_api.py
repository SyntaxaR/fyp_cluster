"""
Controller's CONTROL plane FastAPI app.

Endpoints:
    POST /api/heartbeat                    — Worker -> controller liveness ping
    GET  /api/get_config                   — Worker bootstrapping
    GET  /api/connectivity_test            — Round-trip sanity check
    GET  /api/cluster_status               — Web UI / external observers
    POST /api/workers/{wid}/switch_plane   — Tell a worker to flip its data plane
"""
from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request

from shared.models import (
    ConnectionType,
    ConnectivityTestResponse,
    WorkerConfigResponse,
    WorkerControlInfo,
    WorkerHeartbeat,
    WorkerNetworkModeRequest,
    WorkerRegistration,
    WorkerStatus,
)

logger = logging.getLogger(__name__)


def make_control_app(deps: "ControlDeps") -> FastAPI:
    """Build the control FastAPI app with the given dependency object."""
    app = FastAPI(title="Controller Control API", version="1.0.0")
    router = APIRouter()

    # =========================================================================
    # Worker -> controller heartbeats
    # =========================================================================
    async def _kick_ws_if_idle(wid: int) -> None:
        """If the worker's WebSocket isn't currently live, fire a fast
        reconnect attempt. Heartbeats are the strongest "alive right now"
        signal — same process serves /heartbeat and /worker_ws, so seeing
        a heartbeat means the WS port is bound. This collapses the
        worst-case 25-50 s "RECONNECTING" stall after `systemctl restart
        fyp-worker` down to one heartbeat interval (~3 s).
        """
        if deps.workers_ws is None:
            return
        if deps.workers_ws.is_connected(wid):
            return
        reg = deps.state.get_registered(wid)
        if reg is None or not reg.control_ip:
            return
        info = WorkerControlInfo(
            worker_id=wid,
            control_ip=reg.control_ip,
            serial=reg.serial,
            identifier=reg.hardware_identifier,
            control_port=reg.control_port,
        )
        try:
            await deps.workers_ws.request_reconnect_now(info)
        except Exception as e:
            # Best-effort — don't fail the heartbeat over a reconnect hiccup.
            logger.warning("Fast reconnect for worker %d failed: %s", wid, e)

    @router.post("/api/heartbeat")
    async def receive_heartbeat(heartbeat: WorkerHeartbeat) -> dict[str, Any]:
        state = deps.state
        logger.info("Heartbeat from worker_id=%d serial=%s",
                    heartbeat.worker_id, heartbeat.serial)
        if heartbeat.worker_id == -1:
            # Unassigned — pending or returning after reboot
            existing_wid = state.find_registered_by_serial(heartbeat.serial)
            if existing_wid is not None:
                logger.info("Re-attaching serial=%s to existing worker_id=%d",
                            heartbeat.serial, existing_wid)
                state.touch_heartbeat(existing_wid, heartbeat)
                # Worker just came back from a restart; kick the WS so we
                # don't sit in RECONNECTING for the full retry window.
                await _kick_ws_if_idle(existing_wid)
                return {"assigned_worker_id": existing_wid}
            state.upsert_pending(heartbeat)
            # Auto-register pending workers immediately for now.
            try:
                wid = await deps.register_worker(heartbeat)
                return {"assigned_worker_id": wid}
            except Exception as e:
                logger.error("auto-register failed: %s", e)
                return {"assigned_worker_id": -1}
        else:
            reg = state.get_registered(heartbeat.worker_id)
            if reg is None:
                logger.warning("Heartbeat for unknown worker_id=%d", heartbeat.worker_id)
                return {"assigned_worker_id": -1}
            state.touch_heartbeat(heartbeat.worker_id, heartbeat)
            # Same fast-reconnect kicker for workers that kept their assigned
            # ID across the restart — covers the common case where the
            # python process restarts but the worker_id was passed back via
            # the previous /api/heartbeat response.
            await _kick_ws_if_idle(heartbeat.worker_id)
            return {"assigned_worker_id": heartbeat.worker_id}

    # =========================================================================
    # Worker bootstrap: fetch the cluster-wide config
    # =========================================================================
    @router.get("/api/get_config")
    async def get_config() -> WorkerConfigResponse:
        return WorkerConfigResponse(
            config=deps.config,
            controller_identifier=deps.identifier,
            controller_serial=deps.serial,
        )

    # =========================================================================
    # Connectivity probe
    # =========================================================================
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
            from_identifier=deps.identifier,
            message="connectivity test ok",
            plane=plane,
        )

    # =========================================================================
    # Power-monitor bindings & calibration
    # =========================================================================
    @router.get("/api/bindings")
    async def bindings() -> dict[str, Any]:
        s = deps.state
        chips = (deps.power_monitor.chip_addresses()
                 if deps.power_monitor is not None else [])
        s.known_chip_addresses = list(chips)

        rows = []
        for wid, reg in s.registered_workers.items():
            chip = s.power_bindings.get(wid)
            meta = s.binding_meta.get(wid, {})
            rows.append({
                "worker_id": wid,
                "identifier": reg.hardware_identifier,
                "serial": reg.serial,
                "i2c_address": chip,                  # may be None
                "i2c_address_hex": (f"0x{chip:02X}" if chip is not None else None),
                "delta_w": meta.get("delta_w"),
                "calibrated_ms": meta.get("calibrated_ms"),
            })
        bound_chips = {chip for chip in s.power_bindings.values()}
        unbound_chips = [a for a in chips if a not in bound_chips]
        return {
            "in_progress": s.calibration_in_progress,
            "active_worker_id": s.calibration_active_worker_id,
            "chips": [{"i2c_address": a, "i2c_address_hex": f"0x{a:02X}"}
                      for a in chips],
            "unbound_chips": [{"i2c_address": a, "i2c_address_hex": f"0x{a:02X}"}
                              for a in unbound_chips],
            "workers": rows,
        }

    @router.post("/api/recalibrate")
    async def recalibrate() -> dict[str, Any]:
        if deps.calibration is None:
            return {"status": "unavailable",
                    "message": "Power monitor disabled — nothing to calibrate."}
        if deps.calibration.is_running():
            return {"status": "in_progress"}
        await deps.calibration.trigger()
        return {"status": "started"}

    # =========================================================================
    # Self-restart — in-process os.execv, NOT systemctl
    # =========================================================================
    # We deliberately don't go through `systemctl restart fyp-controller`:
    # systemd's KillMode=mixed sends SIGTERM to the python parent only, but
    # our finally block then calls network.shutdown() which kills hostapd
    # and dnsmasq. That drops the AP, kicks the operator's laptop offline,
    # and forces them to rejoin the cluster every time they want to redeploy
    # the controller code.
    #
    # Instead we set a flag, let the asyncio loop unwind cleanly while
    # SKIPPING network teardown, then os.execvp into a fresh python
    # invocation. hostapd + dnsmasq are subprocess children of the python
    # we're replacing — they survive the exec (Linux doesn't signal
    # subprocesses on parent execve), so the AP stays up the whole time.
    @router.post("/api/restart")
    async def self_restart() -> dict[str, Any]:
        if deps.controller_obj is None:
            raise HTTPException(
                500,
                "controller object not wired into ControlDeps — soft "
                "restart unavailable.",
            )
        deps.controller_obj.request_soft_restart(delay_s=1.5)
        return {
            "status": "scheduled",
            "mode": "soft",
            "delay_s": 1.5,
            "note": "python re-execs into itself with --skip-network. "
                    "hostapd + dnsmasq stay alive — the AP will NOT drop. "
                    "Browser WebSocket should reconnect within ~3 s.",
        }

    # =========================================================================
    # Force redeploy: push latest worker payload + restart fyp-worker on
    # every ACTIVE registered worker. Companion to /api/restart for the
    # "I edited code on the controller, sync everywhere" flow.
    # =========================================================================
    @router.post("/api/redeploy_workers")
    async def redeploy_workers() -> dict[str, Any]:
        if deps.auto_onboarder is None:
            raise HTTPException(503, "AutoOnboarder unavailable")
        try:
            results = await deps.auto_onboarder.redeploy_to_registered()
        except Exception as e:
            logger.exception("redeploy_workers failed")
            raise HTTPException(500, f"redeploy failed: {e}")
        ok = sum(1 for r in results.values() if r.get("status") == "ok")
        return {
            "count": len(results),
            "ok": ok,
            "failed": len(results) - ok,
            "results": results,
        }

    # =========================================================================
    # Auto-onboard status — flat snapshot, useful for `curl | jq` checks
    # =========================================================================
    @router.get("/api/onboarding")
    async def onboarding() -> dict[str, Any]:
        if deps.auto_onboarder is None:
            return {"enabled": False, "running": False, "results": {}}
        return {
            "enabled": deps.auto_onboarder.is_enabled(),
            "running": deps.auto_onboarder.is_running(),
            "results": deps.auto_onboarder.results_snapshot(),
        }

    # =========================================================================
    # Manual SSH-restart of a worker — for when the worker is alive enough
    # for SSH but its python service is wedged (heartbeats stopped, WS dead,
    # etc.). The Overview page wires a per-row button to this endpoint so
    # the operator can recover without walking to the Pi.
    # =========================================================================
    @router.post("/api/workers/{wid}/restart_via_ssh")
    async def restart_worker_via_ssh(wid: int) -> dict[str, Any]:
        if deps.auto_onboarder is None:
            raise HTTPException(503, "AutoOnboarder unavailable — SSH "
                                     "machinery isn't initialised.")
        reg = deps.state.get_registered(wid)
        if reg is None:
            raise HTTPException(404, f"No registered worker with id={wid}")
        if not reg.control_ip:
            raise HTTPException(409, f"Worker {wid} has no recorded IP")
        try:
            result = await deps.auto_onboarder.restart_worker_via_ssh(
                reg.control_ip
            )
        except Exception as e:
            logger.exception("restart_worker_via_ssh failed")
            raise HTTPException(500, f"restart failed: {e}")
        return {"worker_id": wid, "ip": reg.control_ip, **result}

    # =========================================================================
    # Worker data-plane switching
    # =========================================================================
    @router.post("/api/workers/{wid}/switch_plane")
    async def switch_plane(wid: int, req: WorkerNetworkModeRequest) -> dict[str, Any]:
        """Send the worker a `switch_to_ethernet` / `switch_to_wifi` command
        over its WebSocket. The worker reports back its new data plane in the
        next heartbeat.
        """
        if deps.workers_ws is None:
            raise HTTPException(503, "Worker WebSocket manager unavailable")
        reg = deps.state.get_registered(wid)
        if reg is None:
            raise HTTPException(404, f"No registered worker with id={wid}")
        if not deps.workers_ws.is_connected(wid):
            raise HTTPException(409, f"Worker {wid} has no active WebSocket")

        if req.mode == "ethernet":
            ok = await deps.workers_ws.send_command(wid, "switch_to_ethernet", {})
        elif req.mode == "wifi":
            # Fall back to controller-side cluster credentials when the
            # caller omits them — so the UI can switch workers onto the
            # cluster's own AP without re-typing the SSID. An empty
            # password is OK (open AP); we only require an SSID.
            net = deps.config.get("network", {})
            ssid = req.ssid or net.get("wifi_ssid")
            password = req.password if req.password is not None \
                else (net.get("wifi_password") or "")
            if not ssid:
                raise HTTPException(400, "wifi mode requires an SSID")
            ok = await deps.workers_ws.send_command(
                wid, "switch_to_wifi", {"ssid": ssid, "password": password}
            )
        else:
            raise HTTPException(400, f"Unsupported mode '{req.mode}'")

        if not ok:
            raise HTTPException(500, f"Failed to send switch command to worker {wid}")
        return {"status": "ok", "worker_id": wid, "requested_mode": req.mode}

    # =========================================================================
    # Cluster status (used by Web UI / curl)
    # =========================================================================
    @router.get("/api/cluster_status")
    async def cluster_status() -> dict[str, Any]:
        s = deps.state
        return {
            "controller_identifier": deps.identifier,
            "controller_serial": deps.serial,
            "uptime_s": int(time.time() - deps.started_at),
            "pending_workers": [
                {"serial": serial, "identifier": hb.hardware_identifier,
                 "last_heartbeat_s": int(time.time()) - hb.timestamp}
                for serial, hb in s.pending_workers.items()
            ],
            "workers": [
                {
                    "worker_id": wid,
                    "identifier": reg.hardware_identifier,
                    "serial": reg.serial,
                    "control_ip": reg.control_ip,
                    "data_ip": reg.data_ip,
                    "status": reg.status if isinstance(reg.status, str) else reg.status.value,
                    "data_plane": reg.data_plane if isinstance(reg.data_plane, str) else reg.data_plane.value,
                    "last_heartbeat_s": int(time.time()) - reg.timestamp,
                    "loaded_model": reg.loaded_model,
                    "engine": reg.engine,
                }
                for wid, reg in s.registered_workers.items()
            ],
            "experiment": {
                "status": s.current_experiment_status.value
                if hasattr(s.current_experiment_status, "value")
                else str(s.current_experiment_status),
                "config": (s.current_experiment.__dict__ if s.current_experiment else None),
            },
        }

    app.include_router(router)
    return app


# =============================================================================
# Dependency container (avoids cyclic imports with controller.controller)
# =============================================================================
class ControlDeps:
    """Container injected into make_control_app."""

    def __init__(
        self,
        config: dict[str, Any],
        state,
        identifier: str,
        serial: str,
        started_at: float,
        register_worker_coro,
        calibration=None,
        power_monitor=None,
        workers_ws=None,
        auto_onboarder=None,
        controller_obj=None,
    ):
        self.config = config
        self.state = state
        self.identifier = identifier
        self.serial = serial
        self.started_at = started_at
        self.register_worker = register_worker_coro
        self.calibration = calibration
        self.power_monitor = power_monitor
        self.workers_ws = workers_ws
        self.auto_onboarder = auto_onboarder
        # Reference back to the top-level Controller — needed by /api/restart
        # to call request_soft_restart() on the live instance.
        self.controller_obj = controller_obj
