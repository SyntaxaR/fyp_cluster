"""
TOML configuration loader with defaults / validation.
"""
from __future__ import annotations

import logging
import os
import re
import tomllib
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Anchor relative loads at the project root (parent of `shared/`)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"


def _validate_port(d: dict, key: str, default: int) -> None:
    v = d.get(key)
    if not isinstance(v, int) or v < 1 or v > 65535:
        logger.warning(f"Port '{key}' is missing/invalid in config, defaulting to {default}")
        d[key] = default


def _validate_subnet(d: dict, key: str, default: str) -> None:
    v = d.get(key)
    if not isinstance(v, str) or not re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.$", v):
        logger.warning(f"Subnet '{key}' missing/invalid, defaulting to {default}")
        d[key] = default


def _validate_string(d: dict, key: str, default: str, min_len: int = 1) -> None:
    v = d.get(key)
    if not isinstance(v, str) or len(v) < min_len:
        logger.warning(f"String '{key}' missing/invalid, defaulting to {default!r}")
        d[key] = default


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load and validate config.toml. Path defaults to <project>/config.toml."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    # Ensure all top-level sections exist
    for section in ("controller", "worker", "network", "cluster",
                    "power_monitor", "database", "dispatcher"):
        config.setdefault(section, {})

    # ---- controller / worker ports ----
    _validate_port(config["controller"], "control_port", 8001)
    _validate_port(config["controller"], "data_port", 8002)
    _validate_port(config["controller"], "web_port", 8080)
    _validate_port(config["worker"], "control_port", 8001)
    _validate_port(config["worker"], "data_port", 8002)

    # ---- interfaces ----
    _validate_string(config["controller"], "ethernet_interface", "eth0")
    _validate_string(config["controller"], "wifi_interface", "wlan0")
    _validate_string(config["worker"], "ethernet_interface", "eth0")
    _validate_string(config["worker"], "wifi_interface", "wlan0")
    _validate_string(config["worker"], "control_interface", "eth0")

    # ---- network plane ----
    _validate_subnet(config["network"], "ethernet_subnet", "192.168.10.")
    _validate_subnet(config["network"], "wifi_subnet", "192.168.20.")
    _validate_string(config["network"], "wifi_ssid", "FYPClusterAP")
    # wifi_password: empty string is LEGAL — it means "open AP, no WPA".
    # Anything else must be ≥8 chars (WPA-PSK minimum). The old min_len=8
    # silently rewrote "" to the hardcoded default, which made it
    # impossible to actually run an open AP no matter what config said.
    _wpw = config["network"].get("wifi_password", "")
    if not isinstance(_wpw, str):
        logger.warning("wifi_password not a string, defaulting to ''")
        config["network"]["wifi_password"] = ""
    elif _wpw and len(_wpw) < 8:
        logger.warning(
            "wifi_password is shorter than 8 chars (WPA-PSK minimum); "
            "treating as empty (open AP)."
        )
        config["network"]["wifi_password"] = ""
    config["network"].setdefault("wifi_client_ssid", "")
    config["network"].setdefault("wifi_client_password", "")
    config["network"].setdefault("dhcp_range_start", 5)
    config["network"].setdefault("dhcp_range_end", 254)
    config["network"].setdefault("lease_time", "5m")

    # ---- cluster orchestration ----
    config["cluster"].setdefault("monitor_interval", 10)
    config["cluster"].setdefault("heartbeat_timeout", 15)
    config["cluster"].setdefault("ws_reconnect_interval", 5)
    config["cluster"].setdefault("ws_max_reconnect", 5)

    # ---- power monitor ----
    pm = config["power_monitor"]
    pm.setdefault("enabled", True)
    pm.setdefault("i2c_bus", 1)
    pm.setdefault("i2c_address_start", 0x40)
    pm.setdefault("i2c_address_end", 0x4F)
    pm.setdefault("shunt_resistance", 0.01)
    pm.setdefault("actual_vbus", 5.07)
    pm.setdefault("conversion_time", 4)
    pm.setdefault("averaging", 5)
    pm.setdefault("poll_interval_ms", 100)

    # ---- database ----
    config["database"].setdefault("path", "cluster.db")

    # ---- dispatcher ----
    config["dispatcher"].setdefault("algorithm", "round_robin")

    # ---- auto-onboard (default ON) ----
    config.setdefault("auto_onboard", {})
    ao = config["auto_onboard"]
    ao.setdefault("enabled", True)
    ao.setdefault("ssh_port", 22)
    ao.setdefault("ssh_user", "pi")
    ao.setdefault("ssh_password", "raspberry")
    ao.setdefault("deploy_path", "/home/pi/fyp_cluster/latest")
    ao.setdefault("poll_interval_s", 10)
    ao.setdefault("ssh_timeout_s", 3)
    ao.setdefault("install_systemd_unit", True)

    # ---- mock / local-debug ----
    config.setdefault("mock", {})
    mock = config["mock"]
    mock.setdefault("enabled", False)
    mock.setdefault("controller_host", "127.0.0.1")
    mock.setdefault("power_chip_count", 4)
    mock.setdefault("power_baseline_w", 3.0)
    mock.setdefault("power_load_w", 2.5)
    # Allow env override (1 / true / yes => on)
    env_mock = os.environ.get("FYP_MOCK", "").strip().lower()
    if env_mock in ("1", "true", "yes", "on"):
        mock["enabled"] = True
    elif env_mock in ("0", "false", "no", "off"):
        mock["enabled"] = False
    env_host = os.environ.get("FYP_CONTROLLER_HOST", "").strip()
    if env_host:
        mock["controller_host"] = env_host

    # ---- worker extras ----
    config["worker"].setdefault("heartbeat_interval", 5)
    config["worker"].setdefault("inference_engine", "auto")

    # ---- controller extras ----
    config["controller"].setdefault("ap_enabled", True)
    # Resolve wifi_mode: explicit > legacy ap_enabled.
    explicit_mode = (config["controller"].get("wifi_mode") or "").strip().lower()
    if explicit_mode in ("ap", "client", "off"):
        config["controller"]["wifi_mode"] = explicit_mode
    else:
        config["controller"]["wifi_mode"] = (
            "ap" if config["controller"].get("ap_enabled", True) else "off"
        )
    config["controller"].setdefault("log_file", "controller.log")
    config["controller"].setdefault("log_level", "INFO")
    config["worker"].setdefault("log_file", "worker.log")
    config["worker"].setdefault("log_level", "INFO")

    return config
