"""
Mock controller network manager — no dnsmasq, no hostapd, no nmcli.

Used in local-debug mode where 1 controller + N workers run on the same host
over loopback. The real ControllerNetworkManager shells out to system services
that aren't available (and aren't useful) on a developer laptop.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MockControllerNetworkManager:
    """No-op stand-in matching the public surface of ControllerNetworkManager."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        ctrl = config["controller"]
        net = config["network"]
        # Pretend we own the canonical controller IP — workers hit 127.0.0.1
        # in mock mode, so this is just for log/telemetry purposes.
        self.ethernet_interface = ctrl.get("ethernet_interface", "lo")
        self.wifi_interface = ctrl.get("wifi_interface", "lo")
        self.wifi_ssid = net["wifi_ssid"]
        self.wifi_password = net["wifi_password"]
        self.eth_ipv4 = "127.0.0.1"
        self.wifi_ipv4 = "127.0.0.1"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self, initialize_wifi: bool = True) -> None:
        logger.info("[mock-net] Skipping dnsmasq/hostapd; controller bound to 127.0.0.1")
        print("[mock-net] Controller network mocked — no system services started.")

    def adopt_existing(self) -> None:
        # Mock has no real subprocesses to adopt — soft-restart in mock mode
        # is purely an HTTP/uvicorn rebind exercise, no AP to preserve.
        logger.info("[mock-net] adopt_existing (no-op)")

    def shutdown(self) -> None:
        logger.info("[mock-net] shutdown (no-op)")

    # ------------------------------------------------------------------
    # Health / status hooks expected by Controller
    # ------------------------------------------------------------------
    def check_subprocess_health(self) -> bool:
        return True

    def get_interface_ipv4(self, interface: str) -> str | None:
        return "127.0.0.1"

    def ping_test(self, target: str, timeout: int = 1) -> bool:  # noqa: D401
        # Loopback is always "reachable" in mock; no need to actually ping.
        return target in ("127.0.0.1", "localhost") or True
