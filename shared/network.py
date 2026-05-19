"""
Base NetworkManager: subprocess command runner + interface state queries.
"""
from __future__ import annotations

import logging
import subprocess

from shared.models import InterfaceStatus

logger = logging.getLogger(__name__)


class NetworkManager:
    """Common helpers shared by the controller and worker network managers."""

    def run_command(
        self,
        cmd: list[str],
        check: bool = True,
        capture_output: bool = True,
        timeout: int = 30,
    ) -> str | None:
        try:
            logger.debug("Running command: %s", " ".join(cmd))
            result = subprocess.run(
                cmd, check=check, capture_output=capture_output, timeout=timeout
            )
            out = result.stdout.decode().strip() if (capture_output and result.stdout) else ""
            if capture_output:
                logger.debug("Command output: %s", out or "<empty>")
            return out if capture_output else None
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode().strip() if e.stderr else ""
            logger.error("Command '%s' failed (rc=%d): %s",
                         " ".join(cmd), e.returncode, stderr)
            raise
        except subprocess.TimeoutExpired:
            logger.error("Command '%s' timed out after %ds", " ".join(cmd), timeout)
            raise

    def ping_test(self, target: str, count: int = 3, timeout: int = 5) -> bool:
        try:
            logger.info("Pinging %s (%d packets)", target, count)
            result = self.run_command(
                ["ping", "-c", str(count), "-W", str(timeout), target],
                check=False,
            )
            if result and "0% packet loss" in result:
                return True
            return False
        except subprocess.CalledProcessError:
            return False

    def _check_interface_status(self, interface: str) -> InterfaceStatus:
        logger.debug("Checking nmcli status of interface %s", interface)
        status = self.run_command(
            ["nmcli", "-t", "-f", "DEVICE,STATE", "device", "status"]
        ) or ""
        for line in status.split("\n"):
            if not line:
                continue
            parts = line.split(":")
            if len(parts) < 2:
                continue
            device, state = parts[0], parts[1]
            if device == interface:
                if state.startswith("connected"):
                    return InterfaceStatus.CONNECTED
                if state.startswith("disconnected"):
                    return InterfaceStatus.DISCONNECTED
                if state.startswith("unavailable"):
                    return InterfaceStatus.UNAVAILABLE
                if state.startswith("connecting"):
                    return InterfaceStatus.CONNECTING
                raise ValueError(f"Unknown nmcli state for {interface}: {state}")
        logger.error("Interface %s not found in nmcli device list", interface)
        raise ValueError(f"Interface {interface} not found in nmcli device list")

    def get_interface_ipv4(self, interface: str) -> str | None:
        try:
            result = self.run_command(
                ["ip", "-4", "addr", "show", interface], check=False
            ) or ""
            for line in result.splitlines():
                line = line.strip()
                if line.startswith("inet "):
                    return line.split(" ")[1].split("/")[0]
        except Exception as e:
            logger.warning("Failed to read ipv4 for %s: %s", interface, e)
        return None
