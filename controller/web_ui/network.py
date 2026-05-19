"""Network configuration page.

Lets the user inspect and edit the controller's WiFi mode (AP / Client / Off)
and the SSID / password used by each. Edits are persisted to ``config.toml``;
the change takes effect after the next controller restart because reconfiguring
hostapd / nmcli at runtime is intrusive enough that we'd rather the operator
explicitly bounce the service.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


CONFIG_PATH = Path("config.toml")


def _persist_network_settings(wifi_mode: str,
                              ap_ssid: str, ap_password: str,
                              client_ssid: str, client_password: str) -> None:
    """Update [controller].wifi_mode and [network].wifi_*_ssid/password
    in-place via simple line rewriting. Skips comments / blanks.
    Falls back to a load+rewrite if tomli_w is available.
    """
    try:
        # Prefer atomic round-trip via tomli/tomli_w if available
        import tomli, tomli_w  # type: ignore
        with CONFIG_PATH.open("rb") as f:
            data = tomli.load(f)
        data.setdefault("controller", {})["wifi_mode"] = wifi_mode
        net = data.setdefault("network", {})
        net["wifi_ssid"] = ap_ssid
        net["wifi_password"] = ap_password
        net["wifi_client_ssid"] = client_ssid
        net["wifi_client_password"] = client_password
        with CONFIG_PATH.open("wb") as f:
            tomli_w.dump(data, f)
        return
    except Exception as e:
        logger.info("tomli_w not available, falling back to line edit: %s", e)

    # Line-based fallback — preserve comments / formatting when round-trip libs
    # aren't installed.
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"{CONFIG_PATH} not found")
    text = CONFIG_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()
    section = ""

    def _replace(line: str, key: str, value: str, quoted: bool = True) -> str:
        # match `<key> = ...` (possibly with surrounding spaces / inline comment)
        stripped = line.lstrip()
        if not stripped.startswith(key):
            return line
        head = line[: len(line) - len(stripped)]
        # split off comment
        no_comment = stripped.split("#", 1)
        comment = ("  #" + no_comment[1]) if len(no_comment) == 2 else ""
        rendered = f'"{value}"' if quoted else value
        return f"{head}{key} = {rendered}{comment}"

    out: list[str] = []
    for line in lines:
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            section = s[1:-1].strip()
            out.append(line)
            continue
        if section == "controller":
            line = _replace(line, "wifi_mode", wifi_mode)
            # also keep ap_enabled in sync as a legacy hint
            if line.lstrip().startswith("ap_enabled"):
                head = line[: len(line) - len(line.lstrip())]
                comment = ("  #" + line.split("#", 1)[1]) if "#" in line else ""
                line = f"{head}ap_enabled = {'true' if wifi_mode == 'ap' else 'false'}{comment}"
        elif section == "network":
            line = _replace(line, "wifi_ssid", ap_ssid)
            line = _replace(line, "wifi_password", ap_password)
            line = _replace(line, "wifi_client_ssid", client_ssid)
            line = _replace(line, "wifi_client_password", client_password)
        out.append(line)
    CONFIG_PATH.write_text("\n".join(out) + ("\n" if text.endswith("\n") else ""),
                           encoding="utf-8")


def register(controller) -> None:
    from nicegui import ui

    @ui.page("/network")
    def network_page():
        from controller.web_ui._helpers import page_header
        page_header(
            "Network configuration",
            subtitle="Changes apply on the next controller restart.",
            active="network",
        )

        cfg = controller.config
        ctrl_cfg = cfg.get("controller", {})
        net_cfg = cfg.get("network", {})

        # Resolve current effective mode (mirror NetworkManager logic)
        explicit = (ctrl_cfg.get("wifi_mode") or "").strip().lower()
        if explicit in ("ap", "client", "off"):
            current_mode = explicit
        else:
            current_mode = "ap" if ctrl_cfg.get("ap_enabled", True) else "off"

        ui.label(f"Active mode (running): {current_mode}").classes("mt-2 font-semibold")

        with ui.card().classes("mt-4 max-w-2xl"):
            ui.label("WiFi mode").classes("font-semibold")
            wifi_mode_select = ui.select(
                ["ap", "client", "off"],
                value=current_mode,
                label="Mode",
            ).classes("w-48")

            ui.label("AP mode (controller hosts WiFi)").classes("mt-3 font-semibold")
            ap_ssid = ui.input("AP SSID",
                               value=net_cfg.get("wifi_ssid", "")).classes("w-72")
            ap_password = ui.input("AP password",
                                   value=net_cfg.get("wifi_password", ""),
                                   password=True,
                                   password_toggle_button=True).classes("w-72")

            ui.label("Client mode (controller joins existing WiFi)") \
                .classes("mt-3 font-semibold")
            client_ssid = ui.input("Client SSID",
                                   value=net_cfg.get("wifi_client_ssid", "")) \
                .classes("w-72")
            client_password = ui.input(
                "Client password",
                value=net_cfg.get("wifi_client_password", ""),
                password=True,
                password_toggle_button=True,
            ).classes("w-72")

            status_label = ui.label("").classes("mt-2 text-sm")

            def _save():
                try:
                    _persist_network_settings(
                        wifi_mode=wifi_mode_select.value,
                        ap_ssid=ap_ssid.value,
                        ap_password=ap_password.value,
                        client_ssid=client_ssid.value,
                        client_password=client_password.value,
                    )
                    # Mutate the in-memory copy so other pages see the new
                    # values immediately (the running NetworkManager keeps its
                    # current configuration until the controller restarts).
                    controller.config.setdefault("controller", {})["wifi_mode"] = \
                        wifi_mode_select.value
                    controller.config.setdefault("controller", {})["ap_enabled"] = \
                        (wifi_mode_select.value == "ap")
                    n = controller.config.setdefault("network", {})
                    n["wifi_ssid"] = ap_ssid.value
                    n["wifi_password"] = ap_password.value
                    n["wifi_client_ssid"] = client_ssid.value
                    n["wifi_client_password"] = client_password.value

                    status_label.text = (
                        f"Saved. Restart the controller to apply "
                        f"(mode → {wifi_mode_select.value})."
                    )
                    ui.notify("config.toml updated.", type="positive")
                except Exception as e:
                    logger.error("Failed to save network settings: %s", e)
                    status_label.text = f"Save failed: {e}"
                    ui.notify(f"Save failed: {e}", type="negative")

            ui.button("Save", on_click=_save).props("color=primary").classes("mt-3")
