"""
Worker-side network manager: DHCP on ethernet, optional WiFi switching.
"""
from __future__ import annotations

import logging
import subprocess
from time import sleep, time
from typing import Any, Optional

import requests

from shared.models import (
    ConnectionType,
    ConnectivityTestResponse,
    InterfaceStatus,
    WorkerHeartbeat,
)
from shared.network import NetworkManager

logger = logging.getLogger(__name__)


class WorkerNetworkController(NetworkManager):
    def __init__(self, worker_id: int, config: dict[str, Any]):
        super().__init__()
        self.worker_id = worker_id
        self.config = config

        if worker_id < -1 or worker_id > 254:
            raise ValueError("worker_id must be -1..254")

        wcfg = config["worker"]
        net = config["network"]

        self.ethernet_interface = wcfg["ethernet_interface"]
        self.wifi_interface = wcfg["wifi_interface"]
        self.eth_controller_ipv4 = f"{net['ethernet_subnet']}1"
        self.wifi_controller_ipv4 = f"{net['wifi_subnet']}1"
        self.control_port = config["controller"]["control_port"]
        self.data_port = config["controller"]["data_port"]

        self.eth_ipv4: Optional[str] = None
        self.wifi_ipv4: Optional[str] = None
        self.current_mode: ConnectionType = ConnectionType.INVALID

    # =========================================================================
    # Lifecycle
    # =========================================================================
    def initialize(self) -> None:
        logger.info("Worker network init: bringing up %s via DHCP",
                    self.ethernet_interface)

        count = 1
        while self._check_interface_status(self.ethernet_interface) == InterfaceStatus.UNAVAILABLE:
            logger.warning("Ethernet %s unavailable, retry %d/5",
                           self.ethernet_interface, count)
            sleep(5)
            count += 1
            if count > 5:
                raise ConnectionError("Ethernet interface unavailable after 5 retries")

        self._ethernet_use_dhcp(self.ethernet_interface)
        self.current_mode = ConnectionType.ETHERNET
        logger.info("Worker network up on %s, IP=%s",
                    self.ethernet_interface, self.eth_ipv4)

    def destroy(self) -> None:
        logger.info("Worker network shutdown (no destructive ops performed)")

    # =========================================================================
    # Ethernet DHCP setup
    # =========================================================================
    def _ethernet_use_dhcp(self, interface: str) -> None:
        existing = self.run_command(
            ["nmcli", "-g", "GENERAL.CONNECTION", "device", "show", interface]
        ) or ""
        for conn in existing.split("\n"):
            if conn.strip():
                self.run_command(
                    ["nmcli", "connection", "delete", conn.strip()], check=False
                )
        sleep(1)

        if self._check_interface_status(interface) != InterfaceStatus.DISCONNECTED:
            sleep(2)
            if self._check_interface_status(interface) != InterfaceStatus.DISCONNECTED:
                raise OSError(f"Interface {interface} is not disconnected after cleanup")

        self.run_command([
            "nmcli", "connection", "add",
            "type", "ethernet", "ifname", interface,
            "con-name", f"{interface}-worker-dhcp",
            "ipv4.method", "auto",
            "ipv6.method", "disable",
        ])
        self.run_command(["nmcli", "connection", "up", f"{interface}-worker-dhcp"])
        self.eth_ipv4 = self._wait_for_eth_dhcp_ip()

    def _wait_for_eth_dhcp_ip(self, timeout_s: int = 30) -> str:
        logger.info("Waiting for DHCP on %s...", self.ethernet_interface)
        start = time()
        expected = self.config["network"]["ethernet_subnet"]
        while time() - start < timeout_s:
            ip = self.get_interface_ipv4(self.ethernet_interface)
            if ip:
                if ip.startswith(expected):
                    logger.info("DHCP IP %s matches subnet %s", ip, expected)
                    return ip
                logger.warning("DHCP IP %s outside subnet %s, retrying", ip, expected)
            sleep(3)
        raise TimeoutError("Timed out waiting for DHCP")

    # =========================================================================
    # WiFi switching (data plane)
    # =========================================================================
    def switch_to_ethernet(self) -> None:
        if self.current_mode == ConnectionType.ETHERNET:
            return
        if self.current_mode == ConnectionType.WIFI:
            self.disable_wifi_interface(self.wifi_interface)
        self.current_mode = ConnectionType.ETHERNET
        self.wifi_ipv4 = None
        logger.info("Switched data plane to Ethernet")

    def switch_to_wifi(self, ssid: str, password: str) -> None:
        if self.current_mode == ConnectionType.WIFI:
            return
        self.enable_wifi_interface(self.wifi_interface, ssid, password)
        self.current_mode = ConnectionType.WIFI
        logger.info("Switched data plane to Wi-Fi (SSID=%s)", ssid)

    def enable_wifi_interface(self, interface: str, ssid: str, password: str) -> None:
        """Bring ``interface`` up on ``ssid``.

        Minimal flow:
          1. Clear soft-rfkill + turn the radio on.
          2. Delete any cached profile named after the SSID — that's
             the one `nmcli device wifi connect` auto-creates and the
             one stale-credential bugs hide in.
          3. ``nmcli device wifi connect <SSID> [password X] ifname Y``.
          4. Verify link is CONNECTED and got a DHCP IP.

        Empty ``password`` => open AP (no `password` arg passed at all).
        """
        # 1) Radio on.
        self.run_command(["rfkill", "unblock", "wifi"], check=False)
        self.run_command(["nmcli", "radio", "wifi", "on"], check=False)
        sleep(1)

        # 2) Wipe the stale profile NM auto-created last time.
        self.run_command(
            ["nmcli", "connection", "delete", "id", ssid], check=False,
        )

        # 3) Connect.
        cmd = ["nmcli", "device", "wifi", "connect", ssid,
               "ifname", interface]
        if password:
            cmd += ["password", password]
        try:
            self.run_command(cmd, timeout=30)
        except subprocess.CalledProcessError as e:
            stderr = (e.stderr.decode() if e.stderr else "").strip()
            stdout = (e.stdout.decode() if e.stdout else "").strip()
            raise ConnectionError(
                f"nmcli wifi connect '{ssid}' failed (rc={e.returncode}). "
                f"stderr: {stderr or '<empty>'}  "
                f"stdout: {stdout or '<empty>'}"
            ) from e

        # 4) Verify.
        sleep(3)
        if self._check_interface_status(interface) != InterfaceStatus.CONNECTED:
            raise ConnectionError(
                f"WiFi {interface} not CONNECTED after connect to '{ssid}' "
                f"— check `journalctl -u NetworkManager -n 50`."
            )
        ip = self.get_interface_ipv4(interface)
        if not ip:
            raise ConnectionError(
                f"WiFi {interface} got no DHCP lease on '{ssid}'."
            )
        self.wifi_ipv4 = ip

    def disable_wifi_interface(self, interface: str) -> None:
        self.run_command(["nmcli", "radio", "wifi", "off"])
        sleep(2)

    # =========================================================================
    # Connectivity verification
    # =========================================================================
    def verify_data_connectivity(self) -> bool:
        target_ip = (self.wifi_controller_ipv4
                     if self.current_mode == ConnectionType.WIFI
                     else self.eth_controller_ipv4)
        try:
            r = requests.get(
                f"http://{target_ip}:{self.data_port}/api/connectivity_test",
                timeout=3,
            )
            if r.status_code == 200:
                ConnectivityTestResponse(**r.json())
                return True
        except Exception as e:
            logger.debug("Data connectivity probe failed: %s", e)
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
            logger.debug("Control connectivity probe failed: %s", e)
        return False

    def send_control_heartbeat(self, serial: str, hardware_identifier: str,
                               worker_id: int) -> tuple[bool, int]:
        """Returns (ok, assigned_worker_id_from_response)."""
        try:
            # Best-effort host telemetry — never let a stats probe failure
            # break the heartbeat path. Module is at the top level under
            # `shared/` so it ships with both controller and worker tarballs.
            try:
                from shared.host_stats import collect as _collect_stats
                _stats = _collect_stats()
            except Exception:
                _stats = {}
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
                has_hailo=detect_hailo_present(),
                cpu_temp_c=_stats.get("cpu_temp_c"),
                cpu_usage_pct=_stats.get("cpu_usage_pct"),
                npu_temp_c=_stats.get("npu_temp_c"),
            )
            url = f"http://{self.eth_controller_ipv4}:{self.control_port}/api/heartbeat"
            r = requests.post(url, json=heartbeat.model_dump(), timeout=5)
            if r.status_code == 200:
                body = r.json() or {}
                return True, int(body.get("assigned_worker_id", -1))
            logger.error("heartbeat HTTP %d", r.status_code)
            return False, -1
        except Exception as e:
            logger.debug("heartbeat error: %s", e)
            return False, -1


# Module-level cache so we don't fork lspci on every heartbeat (it's stable
# across the worker's lifetime — the NPU is or isn't physically there).
_HAILO_PRESENT_CACHE: Optional[bool] = None


def detect_hailo_present() -> bool:
    """Look for a Hailo NPU on the PCIe bus.

    Caches the result for the lifetime of the worker process — `lspci` is
    cheap but heartbeat fires every ~5s and the NPU isn't going anywhere.
    """
    global _HAILO_PRESENT_CACHE
    if _HAILO_PRESENT_CACHE is not None:
        return _HAILO_PRESENT_CACHE
    try:
        import subprocess
        out = subprocess.run(
            ["lspci"], capture_output=True, text=True, timeout=3, check=False,
        ).stdout
        _HAILO_PRESENT_CACHE = "hailo" in out.lower()
    except Exception:
        _HAILO_PRESENT_CACHE = False
    return _HAILO_PRESENT_CACHE
