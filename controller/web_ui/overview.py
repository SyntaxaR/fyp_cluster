"""
Cluster overview page — list of pending + registered workers, status badges,
quick actions (refresh, switch data plane, kick off experiment).
"""
from __future__ import annotations

import asyncio
import time

from controller.web_ui._helpers import (
    page_header, section, status_chip, worker_label,
)
from shared.models import WorkerStatus  # noqa: F401


def register(controller) -> None:
    from nicegui import ui

    async def _confirm_dialog(title: str, body: str) -> bool:
        """Modal yes/no — used as `if await _confirm_dialog(...): ...`."""
        with ui.dialog() as dialog, ui.card():
            ui.label(title).classes("text-lg font-semibold")
            ui.label(body).classes("text-sm text-gray-600 max-w-md")
            with ui.row().classes("justify-end gap-2 mt-2"):
                ui.button("Cancel",
                          on_click=lambda: dialog.submit(False)).props("flat")
                ui.button("Confirm",
                          on_click=lambda: dialog.submit(True)).props("color=warning")
        return bool(await dialog)

    @ui.page("/")
    def index():
        page_header(
            "FYP Cluster Dashboard",
            subtitle=(f"Controller: {controller.identifier} · "
                      f"serial={controller.serial}"),
            active="overview",
        )
        # Action bar — two split actions side-by-side under the nav.
        #   * "Update worker code" — pushes latest payload via SSH/SFTP
        #     to every ACTIVE worker and restarts fyp-worker on each.
        #   * "Restart controller" — soft-restart only (os.execvpe).
        with ui.row().classes("w-full gap-2 justify-end"):

            # ----- Update worker code -------------------------------------
            async def _show_redeploy_results(body: dict) -> None:
                """Dialog with a per-worker breakdown of the redeploy run."""
                results = body.get("results", {}) or {}
                ok = body.get("ok", 0)
                failed = body.get("failed", 0)
                count = body.get("count", 0)

                with ui.dialog() as dialog, ui.card().classes("min-w-[640px]"):
                    ui.label(
                        f"Worker code update: {ok}/{count} ok"
                        + (f" — {failed} failed" if failed else "")
                    ).classes("text-lg font-semibold")
                    if not results:
                        ui.label(
                            "No active workers were targeted. Make sure "
                            "at least one worker is registered and on the "
                            "ethernet subnet."
                        ).classes("text-sm text-gray-600")
                    else:
                        rows = []
                        for ip, r in sorted(
                            results.items(),
                            key=lambda kv: kv[1].get("worker_id") or 0,
                        ):
                            rows.append({
                                "worker_id": r.get("worker_id"),
                                "ip": ip,
                                "status": r.get("status") or "?",
                                "step": r.get("step") or "",
                                "error": (r.get("error") or "")[:120],
                            })
                        columns = [
                            {"name": "worker_id", "label": "Worker",
                             "field": "worker_id", "align": "left"},
                            {"name": "ip", "label": "IP",
                             "field": "ip", "align": "left"},
                            {"name": "status", "label": "Status",
                             "field": "status", "align": "left"},
                            {"name": "step", "label": "Last step",
                             "field": "step", "align": "left"},
                            {"name": "error", "label": "Error",
                             "field": "error", "align": "left"},
                        ]
                        ui.table(columns=columns, rows=rows,
                                 row_key="ip").classes("w-full text-sm")
                        ui.label(
                            "Step legend: ssh (auth) → sftp (upload) → "
                            "untar (extract) → systemd (restart) → complete."
                        ).classes("text-xs text-gray-500 mt-1")
                    with ui.row().classes("justify-end mt-2"):
                        ui.button("Close",
                                  on_click=lambda: dialog.submit(True)
                                  ).props("flat")
                await dialog

            async def _do_redeploy():
                confirmed = await _confirm_dialog(
                    "Push code to workers?",
                    "SSH + SFTP latest payload to every ACTIVE worker "
                    "on the ethernet plane, then `systemctl restart "
                    "fyp-worker` on each. Typically 10-30 s. The "
                    "controller is NOT restarted by this action.",
                )
                if not confirmed:
                    return
                try:
                    import requests
                    port = controller.config["controller"]["control_port"]
                    ui.notify(
                        "Pushing payload to workers — this can take "
                        "10-30 s. Dashboard stays responsive.",
                        type="info", timeout=8000,
                    )
                    r = await asyncio.to_thread(
                        requests.post,
                        f"http://127.0.0.1:{port}/api/redeploy_workers",
                        timeout=120,
                    )
                    if r.status_code != 200:
                        ui.notify(
                            f"Redeploy failed ({r.status_code}): "
                            f"{r.text[:200]}",
                            type="negative", timeout=10000,
                        )
                        return
                    body = r.json()
                    ok = body.get("ok", 0)
                    failed = body.get("failed", 0)
                    count = body.get("count", 0)
                    ui.notify(
                        f"Worker code update: {ok}/{count} ok"
                        + (f", {failed} failed" if failed else ""),
                        type=("positive" if failed == 0 and count > 0
                              else "warning" if count > 0
                              else "info"),
                        timeout=6000,
                    )
                    await _show_redeploy_results(body)
                except Exception as e:
                    ui.notify(f"Could not call /api/redeploy_workers: {e}",
                              type="negative", timeout=10000)

            ui.button("Update worker code",
                      icon="cloud_upload",
                      on_click=_do_redeploy,
                      ).props("color=primary").tooltip(
                "SSH the latest payload to every ACTIVE worker and "
                "restart fyp-worker. Does NOT touch the controller."
            )

            # ----- Restart controller -----------------------------------
            async def _do_restart():
                confirmed = await _confirm_dialog(
                    "Restart controller?",
                    "Soft-restart only: python re-execs itself in-place. "
                    "hostapd + dnsmasq are kept alive across the exec, "
                    "so the AP does NOT drop and clients stay associated. "
                    "The web UI will auto-reconnect in ~3-5 s.",
                )
                if not confirmed:
                    return
                try:
                    import requests
                    port = controller.config["controller"]["control_port"]
                    r = await asyncio.to_thread(
                        requests.post,
                        f"http://127.0.0.1:{port}/api/restart", timeout=3,
                    )
                    body = r.json() if r.status_code == 200 else {}
                    if r.status_code == 200:
                        ui.notify(
                            f"Controller restart in "
                            f"{body.get('delay_s',1.5)} s. "
                            "Reconnecting in ~5 s …",
                            type="warning", timeout=10000,
                        )
                    else:
                        ui.notify(
                            f"Restart failed ({r.status_code}): "
                            f"{r.text[:200]}",
                            type="negative", timeout=10000,
                        )
                except Exception as e:
                    ui.notify(f"Could not call /api/restart: {e}",
                              type="negative", timeout=10000)

            ui.button("Restart controller",
                      icon="restart_alt",
                      on_click=_do_restart,
                      ).props("color=warning").tooltip(
                "Soft-restart the controller in place. AP stays up, "
                "workers stay connected, web UI auto-reconnects."
            )

        worker_table = ui.table(
            columns=[
                {"name": "worker", "label": "Worker", "field": "worker"},
                {"name": "status", "label": "Status", "field": "status"},
                {"name": "data_plane", "label": "Plane", "field": "data_plane"},
                {"name": "control_ip", "label": "Control IP", "field": "control_ip"},
                {"name": "data_ip", "label": "Data IP", "field": "data_ip"},
                {"name": "loaded_model", "label": "Model", "field": "loaded_model"},
                {"name": "last_hb", "label": "Last HB (s)", "field": "last_hb"},
            ],
            rows=[],
            row_key="worker",
        ).classes("w-full")

        # =====================================================================
        # Per-worker data-plane switch
        # =====================================================================
        ui.label("Switch data plane").classes("mt-3 text-lg font-semibold")
        ui.label("Sends a `switch_to_*` command over the worker's WebSocket. "
                 "On Wi-Fi, leave SSID/password blank to reuse the cluster AP."
                 ).classes("text-sm text-gray-500")
        with ui.row().classes("items-end gap-2 mt-1"):
            switch_worker = ui.select([], label="Worker").classes("w-32")
            switch_mode = ui.select(["ethernet", "wifi"],
                                    value="wifi",
                                    label="Plane").classes("w-32")
            switch_ssid = ui.input("SSID (wifi only, blank = cluster AP)") \
                .classes("w-64")
            switch_pass = ui.input("Password (blank = cluster AP)",
                                   password=True).classes("w-64")

            async def _do_switch():
                wid = switch_worker.value
                if wid is None:
                    ui.notify("Pick a worker first.", type="warning")
                    return
                mode = switch_mode.value
                payload = {"mode": mode}
                if mode == "wifi":
                    if switch_ssid.value:
                        payload["ssid"] = switch_ssid.value
                    if switch_pass.value:
                        payload["password"] = switch_pass.value
                if controller.workers_ws is None or not controller.workers_ws.is_connected(int(wid)):
                    ui.notify(f"Worker {wid} has no active WS connection.",
                              type="warning")
                    return
                if mode == "ethernet":
                    ok = await controller.workers_ws.send_command(
                        int(wid), "switch_to_ethernet", {}
                    )
                else:
                    ssid = payload.get("ssid") or controller.config["network"]["wifi_ssid"]
                    password = payload.get("password") or controller.config["network"]["wifi_password"]
                    ok = await controller.workers_ws.send_command(
                        int(wid), "switch_to_wifi",
                        {"ssid": ssid, "password": password},
                    )
                if ok:
                    ui.notify(f"Switch '{mode}' sent to worker {wid}.",
                              type="positive")
                else:
                    ui.notify(f"Failed to send switch to worker {wid}.",
                              type="negative")

            ui.button("Send switch", on_click=_do_switch).props("color=primary")

        # =====================================================================
        # Manual SSH restart — for the "worker still shows ACTIVE but Last HB
        # is forever ago" case where the WS layer thinks it's connected but
        # the worker python has wedged. Attempts ``systemctl restart fyp-worker``
        # via SSH using the same credentials as auto-onboard.
        # =====================================================================
        ui.label("Restart wedged worker (via SSH)").classes("mt-3 text-lg font-semibold")
        ui.label(
            "Use when a worker's heartbeat has stopped but its IP is still "
            "reachable. Triggers `systemctl restart fyp-worker` over SSH. "
            "Requires auto-onboard credentials."
        ).classes("text-sm text-gray-500")
        with ui.row().classes("items-end gap-2 mt-1"):
            restart_worker_select = ui.select([], label="Worker").classes("w-48")

            async def _do_ssh_restart():
                wid = restart_worker_select.value
                if wid is None:
                    ui.notify("Pick a worker first.", type="warning")
                    return
                try:
                    import asyncio as _asyncio
                    import requests
                    port = controller.config["controller"]["control_port"]
                    ui.notify(f"Restarting worker {wid} via SSH …",
                              type="info", timeout=4000)
                    r = await _asyncio.to_thread(
                        requests.post,
                        f"http://127.0.0.1:{port}/api/workers/{int(wid)}/restart_via_ssh",
                        timeout=45,
                    )
                    if r.status_code != 200:
                        ui.notify(f"SSH restart HTTP {r.status_code}: "
                                  f"{r.text[:200]}",
                                  type="negative", timeout=10000)
                        return
                    body = r.json()
                    if body.get("status") == "ok":
                        ui.notify(f"Worker {wid} restart issued. The worker "
                                  f"should heartbeat again in 5-15 s.",
                                  type="positive", timeout=8000)
                    else:
                        ui.notify(f"SSH restart failed: "
                                  f"{body.get('error','unknown')}",
                                  type="negative", timeout=10000)
                except Exception as e:
                    ui.notify(f"SSH restart call failed: {e}",
                              type="negative", timeout=10000)

            ui.button("Restart via SSH",
                      on_click=_do_ssh_restart).props("color=warning")

        pending_table = ui.table(
            columns=[
                {"name": "serial", "label": "Serial", "field": "serial"},
                {"name": "identifier", "label": "Identifier", "field": "identifier"},
                {"name": "last_hb", "label": "Last HB (s)", "field": "last_hb"},
            ],
            rows=[],
            row_key="serial",
        ).classes("w-full")

        ui.label("Pending workers").classes("mt-4 text-lg font-semibold")
        pending_count = ui.label("0 pending")

        # =====================================================================
        # Power-monitor bindings
        # =====================================================================
        ui.separator()
        with ui.row().classes("items-center mt-4"):
            ui.label("Power-monitor bindings (INA226 ↔ worker)") \
                .classes("text-lg font-semibold")
            calib_status = ui.label("idle").classes("text-sm text-gray-500 ml-4")

        bindings_table = ui.table(
            columns=[
                {"name": "worker", "label": "Worker", "field": "worker"},
                {"name": "i2c", "label": "INA226 addr", "field": "i2c"},
                {"name": "delta_w", "label": "Calib ΔP (W)", "field": "delta_w"},
                {"name": "calibrated", "label": "Calibrated", "field": "calibrated"},
            ],
            rows=[],
            row_key="worker",
        ).classes("w-full")

        unbound_chips_label = ui.label("").classes("text-sm text-gray-500 mt-1")

        async def _trigger_recalibrate():
            if controller.calibration is None:
                ui.notify("Power monitor disabled — nothing to calibrate.",
                          type="warning")
                return
            if controller.calibration.is_running():
                ui.notify("Calibration already in progress.", type="info")
                return
            await controller.calibration.trigger()
            ui.notify("Calibration started — burst per worker, ~"
                      f"{int((controller.calibration.baseline_s + len(controller.state.active_workers()) * (controller.calibration.burst_s + controller.calibration.cool_s)))}s total.",
                      type="positive")

        with ui.row().classes("mt-2"):
            ui.button("Recalibrate", on_click=_trigger_recalibrate).props("color=primary")
            ui.button("Clear bindings",
                      on_click=lambda: (controller.state.clear_all_power_bindings(),
                                        controller.db.clear_bindings(),
                                        ui.notify("Bindings cleared.",
                                                  type="warning"))) \
                .props("color=warning flat")

        def refresh():
            now = int(time.time())
            switch_worker.options = [
                wid for wid in controller.state.registered_workers.keys()
            ]
            if (switch_worker.value is not None
                    and switch_worker.value not in switch_worker.options):
                switch_worker.value = None
            switch_worker.update()

            # Same population for the SSH-restart picker.
            restart_worker_select.options = [
                wid for wid in controller.state.registered_workers.keys()
            ]
            if (restart_worker_select.value is not None
                    and restart_worker_select.value
                        not in restart_worker_select.options):
                restart_worker_select.value = None
            restart_worker_select.update()
            worker_table.rows = [
                {
                    "worker": worker_label(wid, controller),
                    "status": (reg.status if isinstance(reg.status, str)
                               else reg.status.value),
                    "data_plane": (reg.data_plane if isinstance(reg.data_plane, str)
                                   else reg.data_plane.value),
                    "control_ip": reg.control_ip,
                    "data_ip": reg.data_ip,
                    "loaded_model": reg.loaded_model or "—",
                    "last_hb": now - reg.timestamp,
                }
                for wid, reg in controller.state.registered_workers.items()
            ]
            pending_table.rows = [
                {
                    "serial": serial,
                    "identifier": hb.hardware_identifier,
                    "last_hb": now - hb.timestamp,
                }
                for serial, hb in controller.state.pending_workers.items()
            ]
            pending_count.text = f"{len(controller.state.pending_workers)} pending"

            # ---- bindings section ----
            chips = (controller.power_monitor.chip_addresses()
                     if controller.power_monitor is not None else [])
            controller.state.known_chip_addresses = list(chips)

            rows = []
            for wid, reg in controller.state.registered_workers.items():
                addr = controller.state.power_bindings.get(wid)
                meta = controller.state.binding_meta.get(wid, {})
                cal_ms = meta.get("calibrated_ms", 0) or 0
                cal_str = (time.strftime("%Y-%m-%d %H:%M:%S",
                                         time.localtime(cal_ms / 1000.0))
                           if cal_ms else "—")
                rows.append({
                    "worker": worker_label(wid, controller),
                    "i2c": (f"0x{addr:02X}" if addr is not None else "— (unbound)"),
                    "delta_w": (round(meta.get("delta_w", 0.0), 3)
                                if addr is not None else "—"),
                    "calibrated": cal_str,
                })
            bindings_table.rows = rows

            bound_chips = set(controller.state.power_bindings.values())
            unbound = [a for a in chips if a not in bound_chips]
            # Format each unbound chip as ``unbounded(0xXX)`` so the address
            # is right next to the label rather than separated by a colon.
            _unbound_str = ", ".join(f"unbounded(0x{a:02X})" for a in unbound)
            unbound_chips_label.text = (
                f"INA226 chips detected: {len(chips)}  •  "
                f"workers bound: {len(controller.state.power_bindings)}"
                + (f"  •  {_unbound_str}" if unbound else "")
            )

            if controller.calibration is None:
                calib_status.text = "disabled (no power monitor)"
            elif controller.state.calibration_in_progress:
                wid = controller.state.calibration_active_worker_id
                calib_status.text = (f"running… (bursting {worker_label(wid, controller)})"
                                     if wid is not None else "running… (baseline)")
            else:
                calib_status.text = "idle"

            worker_table.update()
            pending_table.update()
            bindings_table.update()

        ui.timer(2.0, refresh)
        refresh()
