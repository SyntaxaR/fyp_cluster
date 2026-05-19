"""
Mock worker network manager — no nmcli, no DHCP, no Wi-Fi switching.

The "control plane" and "data plane" both collapse onto loopback. This makes
it possible to run multiple worker processes on the same laptop, all dialling
the controller at 127.0.0.1.
"""
from __future__ import annotations

import logging
from time import time
from typing import Any

import requests

from shared.models import (
    ConnectionType,
    ConnectivityTestResponse,
    WorkerHeartbeat,
)

logger = logging.getLogger(__name__)


class MockWorkerNetworkController:
    """Loopback-only stand-in for WorkerNetworkController."""

    def __init__(self, worker_id: int, config: dict[str, Any]):
        self.worker_id = worker_id
        self.config = config
        wcfg = config["worker"]
        self.ethernet_interface = wcfg.get("ethernet_interface", "lo")
        self.wifi_interface = wcfg.get("wifi_interface", "lo")

        # In mock mode the controller lives at controller_host (default 127.0.0.1).
        host = config.get("mock", {}).get("controller_host", "127.0.0.1")
        self.eth_controller_ipv4 = host
        self.wifi_controller_ipv4 = host
        self.control_port = config["controller"]["control_port"]
        self.data_port = config["controller"]["data_port"]

        self.eth_ipv4: str = "127.0.0.1"
        self.wifi_ipv4: str | None = None
        self.current_mode: ConnectionType = ConnectionType.ETHERNET

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        logger.info("[mock-net] worker network mocked — controller at %s",
                    self.eth_controller_ipv4)
        print(f"[mock-net] worker bound to loopback; controller={self.eth_controller_ipv4}")

    def destroy(self) -> None:
        logger.info("[mock-net] destroy (no-op)")

    # ------------------------------------------------------------------
    # Plane switching — no-ops in mock
    # ------------------------------------------------------------------
    def switch_to_ethernet(self) -> None:
        self.current_mode = ConnectionType.ETHERNET
        self.wifi_ipv4 = None
        logger.info("[mock-net] switch_to_ethernet (no-op)")

    def switch_to_wifi(self, ssid: str, password: str) -> None:
        # Pretend Wi-Fi succeeded; reuse loopback so subsequent probes still pass.
        self.current_mode = ConnectionType.WIFI
        self.wifi_ipv4 = "127.0.0.1"
        logger.info("[mock-net] switch_to_wifi(ssid=%s) — pretending success", ssid)

    # ------------------------------------------------------------------
    # Connectivity probes — real HTTP, but to loopback
    # ------------------------------------------------------------------
    def verify_data_connectivity(self) -> bool:
        try:
            r = requests.get(
                f"http://{self.eth_controller_ipv4}:{self.data_port}/api/connectivity_test",
                timeout=3,
            )
            if r.status_code == 200:
                ConnectivityTestResponse(**r.json())
                return True
        except Exception as e:
            logger.debug("[mock-net] data probe failed: %s", e)
        return False

    def verify_control_connectivity(self) -> bool:
        try:
            r = requests.get(
                f"http://{self.eth_controller_ipv4}:{self.control_port}/api/connectivity_test",
                timeout=3,
            )
            if r.status_code == 200:
                ConnectivityTestResponse(**r.json())
                return True
        except Exception as e:
            logger.debug("[mock-net] control probe failed: %s", e)
        return False

    # ------------------------------------------------------------------
    # Heartbeat — same wire format as the real worker
    # ------------------------------------------------------------------
    def send_control_heartbeat(self, serial: str, hardware_identifier: str,
                               worker_id: int) -> tuple[bool, int]:
        try:
            heartbeat = WorkerHeartbeat(
                worker_id=worker_id,
                serial=serial,
                hardware_identifier=hardware_identifier,
                control_ip_address=self.eth_ipv4 or "",
                data_connectivity=self.verify_data_connectivity(),
                data_ip_address=(self.wifi_ipv4
                                 if self.current_mode == ConnectionType.WIFI
                                 else self.eth_ipv4) or "",
                data_plane=self.current_mode,
                timestamp=int(time()),
                control_port=int(self.config["worker"]["control_port"]),
                data_port=int(self.config["worker"]["data_port"]),
                # Mock workers don't have a Hailo NPU.
                has_hailo=False,
            )
            url = f"http://{self.eth_controller_ipv4}:{self.control_port}/api/heartbeat"
            r = requests.post(url, json=heartbeat.model_dump(), timeout=5)
            if r.status_code == 200:
                body = r.json() or {}
                return True, int(body.get("assigned_worker_id", -1))
            logger.error("[mock-net] heartbeat HTTP %d", r.status_code)
            return False, -1
        except Exception as e:
            logger.debug("[mock-net] heartbeat error: %s", e)
            return False, -1
