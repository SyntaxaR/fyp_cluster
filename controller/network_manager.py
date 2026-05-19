"""
Controller-side network manager: static IP on eth0, dnsmasq DHCP, optional
hostapd AP *or* nmcli WiFi client.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from time import sleep
from typing import Any, Optional

from shared.models import InterfaceStatus
from shared.network import NetworkManager


# =============================================================================
# Adopted-process wrapper
# =============================================================================
# When the controller soft-restarts via os.execvp, the python process image
# is replaced but its child processes (dnsmasq, hostapd) survive. The new
# python instance has no Popen handles for them. _AdoptedProcess fakes just
# enough of the Popen interface (poll / terminate / kill / wait / .pid) for
# shutdown() to clean them up later via os.kill.
class _AdoptedProcess:
    def __init__(self, pid: int, name: str):
        self.pid = int(pid)
        self.name = name

    def poll(self) -> Optional[int]:
        """Return None if alive, -1 if gone (mirrors Popen.poll semantics).

        Linux: ``os.kill(pid, 0)`` raises ProcessLookupError if gone.
        Windows: signal 0 is unsupported; it raises a generic OSError
        regardless of liveness. We only target Linux (Pi 5) in production,
        but tolerating Windows lets the unit tests / mock-mode smoke runs
        on the dev laptop import cleanly.
        """
        try:
            os.kill(self.pid, 0)
            return None
        except ProcessLookupError:
            return -1
        except PermissionError:
            # Exists but not ours — treat as alive so we don't try to kill it
            return None
        except OSError:
            # Windows fallback: assume alive. Real production check happens
            # on Pi where ProcessLookupError is the canonical "gone" signal.
            return None

    def terminate(self) -> None:
        try:
            os.kill(self.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def kill(self) -> None:
        try:
            os.kill(self.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def wait(self, timeout: Optional[float] = None) -> int:
        """Poll-based wait — adopted procs aren't our children so waitpid()
        won't work. Returns 0 on graceful exit, raises TimeoutExpired if
        still alive past timeout (callers handle either path)."""
        deadline = time.monotonic() + (timeout if timeout is not None else 5.0)
        while time.monotonic() < deadline:
            if self.poll() is not None:
                return 0
            time.sleep(0.1)
        raise subprocess.TimeoutExpired(cmd=self.name, timeout=timeout or 5.0)

    def __repr__(self) -> str:
        return f"<_AdoptedProcess {self.name} pid={self.pid}>"

logger = logging.getLogger(__name__)


class ControllerNetworkManager(NetworkManager):
    """Bring up the cluster's control plane (Ethernet) and optional Wi-Fi AP."""

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config

        ctrl = config["controller"]
        net = config["network"]

        self.ethernet_interface = ctrl["ethernet_interface"]
        self.wifi_interface = ctrl["wifi_interface"]
        self.wifi_ssid = net["wifi_ssid"]
        self.wifi_password = net["wifi_password"]
        self.wifi_client_ssid = net.get("wifi_client_ssid", "") or ""
        self.wifi_client_password = net.get("wifi_client_password", "") or ""
        self.eth_ipv4 = f"{net['ethernet_subnet']}1"
        self.wifi_ipv4 = f"{net['wifi_subnet']}1"
        self.ethernet_gateway = self.eth_ipv4
        self.wifi_gateway = self.wifi_ipv4
        self.dhcp_start = net["dhcp_range_start"]
        self.dhcp_end = net["dhcp_range_end"]

        # Resolve effective wifi mode: explicit `wifi_mode` wins, else fall
        # back to legacy `ap_enabled` boolean for backward compatibility.
        explicit = (ctrl.get("wifi_mode") or "").strip().lower()
        if explicit in ("ap", "client", "off"):
            self.wifi_mode = explicit
        else:
            self.wifi_mode = "ap" if ctrl.get("ap_enabled", True) else "off"

        self.dnsmasq_process: subprocess.Popen | None = None
        self.hostapd_process: subprocess.Popen | None = None

        self.dnsmasq_conf_file = Path("/tmp/dnsmasq-controller.conf")
        self.dnsmasq_lease_file = Path("/tmp/dnsmasq-controller.leases")
        self.hostapd_conf_file = Path("/tmp/hostapd-controller.conf")

        # Set by initialize() once the network is fully up; checked by
        # shutdown() to know whether there's anything to restore.
        self._initialized: bool = False
        self._effective_mode: str = "off"

    # =========================================================================
    # High-level lifecycle
    # =========================================================================
    def initialize(self, initialize_wifi: bool = True) -> None:
        # `initialize_wifi` is the legacy entry-point flag. When False it
        # forces "off" regardless of config (used by `--no-wifi` CLI flag).
        effective_mode = self.wifi_mode if initialize_wifi else "off"
        logger.info("Initializing controller network (wifi_mode=%s)...",
                    effective_mode)
        print(f"[network] Initializing controller network (wifi_mode={effective_mode})...")

        # Stop any system-managed dnsmasq / hostapd
        self.run_command(["sudo", "systemctl", "stop", "dnsmasq"], check=False)
        self.run_command(["sudo", "systemctl", "stop", "hostapd"], check=False)
        self.run_command(["sudo", "pkill", "dnsmasq"], check=False)
        self.run_command(["sudo", "pkill", "hostapd"], check=False)

        # Wait for ethernet interface to be present
        count = 1
        while self._check_interface_status(self.ethernet_interface) == InterfaceStatus.UNAVAILABLE:
            logger.warning("Ethernet %s unavailable. Retrying in 5s (%d/5)",
                           self.ethernet_interface, count)
            sleep(5)
            count += 1
            if count > 5:
                raise ConnectionError("Ethernet interface unavailable after 5 retries")

        self._configure_ethernet_static_ip()
        # In AP mode, dnsmasq serves DHCP on both eth and wlan; in client/off
        # mode it only owns the eth control plane.
        self.dnsmasq_conf_file.write_text(
            self._generate_dnsmasq_dhcp_config(
                include_wifi=(effective_mode == "ap"),
                include_eth=True,
            )
        )
        # The DHCP lease file at self.dnsmasq_lease_file is INTENTIONALLY
        # preserved across restarts. Wiping it would make every still-online
        # client (your laptop on the AP, registered workers, etc.) invisible
        # in the controller's view until they next renew their lease — which
        # for a default 24h lease can be up to 12 hours away. AutoOnboarder's
        # "already attempted" cache is in-memory and naturally resets on
        # restart, so the lease file plays no role in onboarding state.

        if effective_mode == "ap":
            self._configure_wifi_ap()
        elif effective_mode == "client":
            self._configure_wifi_client()
        else:
            logger.info("WiFi disabled (wifi_mode=off); leaving %s untouched",
                        self.wifi_interface)

        self._start_dnsmasq()
        # Mark that we successfully completed initialize() — shutdown uses
        # this to decide whether the network actually needs restoring (a
        # `--skip-network` run never touched anything, so shutdown skips too).
        self._initialized = True
        # Remember the mode so shutdown knows what it has to undo.
        self._effective_mode = effective_mode
        print("[network] Controller network initialized.")

    # ------------------------------------------------------------------------
    # Soft-restart support — adopt dnsmasq / hostapd that the previous python
    # instance left running. Two strategies:
    #   1. Env vars FYP_DNSMASQ_PID / FYP_HOSTAPD_PID (set by main() before
    #      os.execvp on soft-restart). Fast + reliable.
    #   2. pgrep on the config-file path we always use. Fallback for when
    #      a developer runs `controller.controller --skip-network` by hand.
    # Whichever wins, we wrap the PID in _AdoptedProcess and stash it where
    # shutdown() expects to find a Popen-like handle. Also marks the manager
    # "initialized" so a later real shutdown actually does the cleanup.
    # ------------------------------------------------------------------------
    def adopt_existing(self) -> None:
        adopted_any = False
        for name, env_var, conf_path, attr in (
            ("dnsmasq", "FYP_DNSMASQ_PID",
             str(self.dnsmasq_conf_file), "dnsmasq_process"),
            ("hostapd", "FYP_HOSTAPD_PID",
             str(self.hostapd_conf_file), "hostapd_process"),
        ):
            pid = self._resolve_existing_pid(env_var, conf_path)
            if pid is None:
                logger.info("adopt: no running %s found "
                            "(env=%s, pgrep=%s)", name, env_var, conf_path)
                continue
            setattr(self, attr, _AdoptedProcess(pid, name))
            logger.info("adopt: took over existing %s pid=%d (via %s)",
                        name, pid,
                        "env" if os.environ.get(env_var) else "pgrep")
            adopted_any = True

        if adopted_any:
            # Mark as if initialize() had run so shutdown() does its full
            # cleanup (kill subprocs + restore eth/wlan) on the next stop.
            self._initialized = True
            # Best guess — if hostapd was adopted we were in AP mode.
            self._effective_mode = (
                "ap" if isinstance(self.hostapd_process, _AdoptedProcess)
                else "off"
            )

    def _resolve_existing_pid(self, env_var: str,
                              conf_path: str) -> Optional[int]:
        """Return a live PID via env var first, then pgrep -f conf_path."""
        env_val = os.environ.get(env_var, "").strip()
        if env_val:
            try:
                pid = int(env_val)
                os.kill(pid, 0)        # signal 0 = "is it alive?"
                return pid
            except (ValueError, ProcessLookupError):
                logger.info("adopt: %s=%s but PID not alive; "
                            "falling back to pgrep", env_var, env_val)
            except PermissionError:
                # Process exists but isn't ours — still safe to track
                return int(env_val)

        try:
            r = subprocess.run(
                ["pgrep", "-f", conf_path],
                capture_output=True, text=True, timeout=2, check=False,
            )
            for tok in r.stdout.split():
                try:
                    pid = int(tok)
                    os.kill(pid, 0)
                    return pid
                except (ValueError, ProcessLookupError):
                    continue
        except Exception as e:
            logger.debug("adopt: pgrep -f %s failed: %s", conf_path, e)
        return None

    def shutdown(self) -> None:
        """Tear down the cluster network and hand interfaces back to the OS.

        Called from ``Controller.run()`` 's finally clause on every clean
        exit (Ctrl-C, ``systemctl stop``, SIGTERM, etc.). After this returns:
          * ``dnsmasq`` and ``hostapd`` subprocesses are gone
          * ``eth0`` is back on a DHCP nmcli profile (so the box can rejoin
            its normal LAN for SSH / apt / etc.)
          * ``wlan0`` is back under NetworkManager — the unmanaged drop-in
            we wrote during AP-mode bring-up is removed and NM is restarted
            so it re-takes the interface.

        Best-effort: every step is wrapped in a try/except so a single bad
        ``nmcli`` invocation can't leave the box wedged. Idempotent — safe
        to call from a half-initialized state or after a previous shutdown.
        """
        # 1) Stop the helper subprocesses we own.
        for proc, name in ((self.dnsmasq_process, "dnsmasq"),
                           (self.hostapd_process, "hostapd")):
            if proc and proc.poll() is None:
                logger.info("Terminating %s pid=%d", name, proc.pid)
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()
        self.dnsmasq_process = None
        self.hostapd_process = None

        # 2) Skip the rest if initialize() never ran (e.g. --skip-network).
        if not getattr(self, "_initialized", False):
            logger.info("Skipping interface restore — never initialized.")
            return

        # 3) Eth back to DHCP, wlan back to NM. Order matters slightly:
        # restoring NetworkManager first lets it pick up the new eth DHCP
        # profile without racing against our delete + add.
        try:
            self._restore_wifi_managed()
        except Exception as e:
            logger.warning("WiFi restore failed: %s", e)
        try:
            self._restore_ethernet_dhcp()
        except Exception as e:
            logger.warning("Ethernet restore failed: %s", e)

        self._initialized = False

    # ------------------------------------------------------------------------
    # Restore helpers (called from shutdown())
    # ------------------------------------------------------------------------
    def _restore_ethernet_dhcp(self) -> None:
        """Drop our static-IP nmcli connection and add a DHCP one in its place."""
        iface = self.ethernet_interface
        static_conn = f"{iface}-controller-static"
        dhcp_conn = f"{iface}-controller-dhcp"
        logger.info("Restoring %s to DHCP (deleting %s, adding %s)",
                    iface, static_conn, dhcp_conn)

        # Delete our static connection if it exists. check=False so a missing
        # connection on a half-initialized run doesn't blow up shutdown.
        self.run_command(["nmcli", "connection", "delete", static_conn],
                         check=False)
        # Also drop any stale "-controller-dhcp" from a previous shutdown so
        # repeated start/stop cycles don't accumulate duplicate profiles.
        self.run_command(["nmcli", "connection", "delete", dhcp_conn],
                         check=False)

        # Recreate a DHCP profile and bring it up. NetworkManager will then
        # acquire a lease from whatever DHCP server is on the wire (the lab
        # LAN, the home router, etc.) — the same state the system was in
        # before we touched anything.
        self.run_command([
            "nmcli", "connection", "add", "type", "ethernet",
            "ifname", iface, "con-name", dhcp_conn,
            "ipv4.method", "auto",
            "ipv6.method", "auto",
        ], check=False)
        self.run_command(["nmcli", "connection", "up", dhcp_conn],
                         check=False)

    def _restore_wifi_managed(self) -> None:
        """Hand wlan0 back to NetworkManager.

        AP-mode bring-up writes a ``[keyfile]`` drop-in under
        ``/etc/NetworkManager/conf.d`` that tells NM to ignore wlan0 so
        hostapd can drive it directly. Removing that drop-in + restarting
        NetworkManager is enough to put wlan0 back under NM control. Any
        manual static IP we added with ``ip addr add`` during AP mode is
        flushed first so NM doesn't see a phantom address on bring-up.
        """
        nm_conf_dir = Path("/etc/NetworkManager/conf.d")
        drop_in = nm_conf_dir / f"{self.wifi_interface}-controller-unmanaged.conf"
        removed = False
        if drop_in.exists():
            try:
                drop_in.unlink()
                removed = True
                logger.info("Removed NM unmanaged drop-in for %s",
                            self.wifi_interface)
            except Exception as e:
                logger.warning("Failed to remove %s: %s", drop_in, e)

        # Flush the manual static IP we added during AP-mode setup. In
        # client/off mode we never added one and this is a no-op.
        self.run_command(
            ["sudo", "ip", "addr", "flush", "dev", self.wifi_interface],
            check=False,
        )

        # Restart NM only if we actually changed its config — restarts are
        # disruptive (drops every active connection briefly) and we don't
        # want to do them when not needed (e.g. on a --skip-network shutdown
        # that somehow still got here).
        if removed:
            self.run_command(["sudo", "systemctl", "restart", "NetworkManager"],
                             check=False)
            logger.info("NetworkManager restarted; %s back under NM control",
                        self.wifi_interface)

    # =========================================================================
    # Process supervision
    # =========================================================================
    def _start_dnsmasq(self) -> None:
        if self.dnsmasq_process and self.dnsmasq_process.poll() is None:
            raise RuntimeError("dnsmasq already running")
        logger.info("Starting dnsmasq...")
        self.dnsmasq_process = subprocess.Popen(
            ["sudo", "dnsmasq", "--no-daemon",
             "--conf-file=/tmp/dnsmasq-controller.conf", "--log-facility=-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
        )
        sleep(2)
        if self.dnsmasq_process.poll() is not None:
            err = self.dnsmasq_process.stderr.read() if self.dnsmasq_process.stderr else ""
            raise ConnectionError(f"dnsmasq failed to start: {err}")
        logger.info("dnsmasq started, pid=%d", self.dnsmasq_process.pid)
        self._monitor_process(self.dnsmasq_process, "dnsmasq")

    def _start_hostapd(self) -> None:
        if self.hostapd_process and self.hostapd_process.poll() is None:
            raise RuntimeError("hostapd already running")
        logger.info("Starting hostapd...")
        self.hostapd_process = subprocess.Popen(
            ["sudo", "hostapd", "/tmp/hostapd-controller.conf"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
        )
        sleep(2)
        if self.hostapd_process.poll() is not None:
            err = self.hostapd_process.stderr.read() if self.hostapd_process.stderr else ""
            raise RuntimeError(f"hostapd failed to start: {err}")
        logger.info("hostapd started, pid=%d", self.hostapd_process.pid)
        self._monitor_process(self.hostapd_process, "hostapd")

    @staticmethod
    def _monitor_process(process: subprocess.Popen, name: str) -> None:
        def _read(pipe, prefix):
            try:
                for line in iter(pipe.readline, ""):
                    if line:
                        logger.debug("[%s] %s", prefix, line.strip())
            except Exception as e:
                logger.warning("Error reading %s: %s", prefix, e)
        if process.stdout:
            threading.Thread(target=_read,
                             args=(process.stdout, f"{name}-stdout"),
                             daemon=True).start()
        if process.stderr:
            threading.Thread(target=_read,
                             args=(process.stderr, f"{name}-stderr"),
                             daemon=True).start()

    def check_subprocess_health(self) -> bool:
        if self.dnsmasq_process and self.dnsmasq_process.poll() is not None:
            logger.error("dnsmasq has exited!")
            return False
        if self.hostapd_process and self.hostapd_process.poll() is not None:
            logger.error("hostapd has exited!")
            return False
        return True

    # =========================================================================
    # Interface configuration
    # =========================================================================
    def _configure_ethernet_static_ip(self) -> None:
        logger.info("Setting %s to static %s/24", self.ethernet_interface, self.eth_ipv4)
        existing = self.run_command(
            ["nmcli", "-g", "GENERAL.CONNECTION", "device", "show", self.ethernet_interface]
        ) or ""
        for conn in existing.split("\n"):
            if conn.strip():
                self.run_command(["nmcli", "connection", "delete", conn.strip()], check=False)
        sleep(1)

        # Make sure interface ends up disconnected before re-creating connection
        if self._check_interface_status(self.ethernet_interface) != InterfaceStatus.DISCONNECTED:
            sleep(2)
            if self._check_interface_status(self.ethernet_interface) != InterfaceStatus.DISCONNECTED:
                raise OSError(
                    f"Interface {self.ethernet_interface} is not disconnected after "
                    "deleting all NetworkManager connections."
                )

        self.run_command([
            "nmcli", "connection", "add", "type", "ethernet",
            "ifname", self.ethernet_interface,
            "con-name", f"{self.ethernet_interface}-controller-static",
            "ipv4.method", "manual",
            "ipv4.addresses", f"{self.eth_ipv4}/24",
            "ipv4.gateway", "",
            "ipv4.dns", "",
            "ipv6.method", "disable",
        ])
        self.run_command(
            ["nmcli", "connection", "up", f"{self.ethernet_interface}-controller-static"]
        )

    def _configure_wifi_client(self) -> None:
        """Join an existing WiFi network via nmcli.

        Mirrors the Worker-side switch_to_wifi flow but for the controller's
        own wlan0. The cluster's logical data plane uses this WiFi network
        rather than a controller-hosted AP, so dnsmasq does NOT serve DHCP on
        wlan0 in this mode (the upstream router does).
        """
        if not self.wifi_client_ssid or not self.wifi_client_password:
            raise ValueError(
                "wifi_mode='client' requires network.wifi_client_ssid + "
                "network.wifi_client_password to be set in config.toml"
            )
        logger.info("Joining WiFi '%s' on %s (client mode)",
                    self.wifi_client_ssid, self.wifi_interface)

        # Hand wlan0 back to NetworkManager and remove the unmanaged drop-in
        # if it exists from a previous AP run.
        nm_conf_dir = Path("/etc/NetworkManager/conf.d")
        if nm_conf_dir.exists():
            for f in nm_conf_dir.glob("*-controller-unmanaged.conf"):
                f.unlink()
            self.run_command(["sudo", "systemctl", "restart", "NetworkManager"])
            sleep(1)

        self.run_command(["nmcli", "radio", "wifi", "on"], check=False)
        sleep(2)

        # nmcli will create / update a connection profile keyed by SSID.
        self.run_command([
            "nmcli", "device", "wifi", "connect", self.wifi_client_ssid,
            "password", self.wifi_client_password,
            "ifname", self.wifi_interface,
        ])
        sleep(3)
        if self._check_interface_status(self.wifi_interface) != InterfaceStatus.CONNECTED:
            raise ConnectionError(
                f"WiFi {self.wifi_interface} failed to join "
                f"'{self.wifi_client_ssid}'"
            )
        ip = self.get_interface_ipv4(self.wifi_interface)
        if ip:
            logger.info("Joined %s, got IP %s on %s",
                        self.wifi_client_ssid, ip, self.wifi_interface)
            # Update wifi_ipv4 so heartbeat / data-plane callers see the
            # router-assigned address rather than the AP-mode default.
            self.wifi_ipv4 = ip
        else:
            logger.warning("Joined %s but no IPv4 detected on %s",
                           self.wifi_client_ssid, self.wifi_interface)

    def _configure_wifi_ap(self) -> None:
        logger.info("Configuring %s as Wi-Fi AP %s/24 (SSID=%s)",
                    self.wifi_interface, self.wifi_ipv4, self.wifi_ssid)

        # Detach NetworkManager from the wifi interface so hostapd can take over.
        nm_conf_dir = Path("/etc/NetworkManager/conf.d")
        if not nm_conf_dir.exists():
            raise FileNotFoundError(
                f"{nm_conf_dir} not found — system likely doesn't use NetworkManager."
            )
        for conf_file in nm_conf_dir.glob("*-controller-unmanaged.conf"):
            conf_file.unlink()
        nm_conf_file = nm_conf_dir / f"{self.wifi_interface}-controller-unmanaged.conf"
        nm_conf_file.write_text(
            f"[keyfile]\nunmanaged-devices=interface-name:{self.wifi_interface}\n"
        )
        self.run_command(["sudo", "systemctl", "restart", "NetworkManager"])
        sleep(1)

        self.run_command(["ip", "addr", "flush", "dev", self.wifi_interface])
        self.run_command(["sudo", "ip", "addr", "add",
                          f"{self.wifi_ipv4}/24", "dev", self.wifi_interface])
        self.run_command(["sudo", "ip", "link", "set", self.wifi_interface, "up"])
        sleep(1)

        self.hostapd_conf_file.write_text(self._generate_hostapd_config())
        self._start_hostapd()

    # =========================================================================
    # dnsmasq / hostapd config templating
    # =========================================================================
    def _generate_dnsmasq_dhcp_config(self, include_wifi: bool, include_eth: bool = True) -> str:
        if not include_eth and not include_wifi:
            raise ValueError("dnsmasq config must include at least one interface")
        net = self.config["network"]
        # Lease time governs how often clients renew. Short value = lease
        # file always reflects who's actually online (good for the "I want
        # devices to re-DHCP on reconnect" flow), at the cost of a bit
        # more DHCP chatter. 5 min is a good demo default — clients renew
        # at the half-life mark (T1 = 2.5 min), so any cable replug or
        # controller restart shows up in the lease file within ~3 min.
        lease_time = net.get("lease_time", "5m")

        cfg = f"""
# AUTO-GENERATED — DO NOT EDIT BY HAND
domain-needed
bogus-priv
no-resolv
no-poll
bind-interfaces

dhcp-leasefile=/tmp/dnsmasq-controller.leases
# `dhcp-authoritative` makes us NACK any REQUEST for an IP we don't have a
# record of (e.g. after lease file wipe, or after a client moves between
# subnets). NACK forces the client to start a fresh DHCPDISCOVER, which is
# exactly the "re-DHCP on reconnect" semantic the demo wants. Safe because
# we're the only DHCP server on this private subnet.
dhcp-authoritative
log-dhcp
log-queries
"""
        if include_eth:
            cfg += f"""
# {self.ethernet_interface}: control-plane DHCP (lease={lease_time})
interface={self.ethernet_interface}
dhcp-range=interface:{self.ethernet_interface},{net['ethernet_subnet']}{self.dhcp_start},{net['ethernet_subnet']}{self.dhcp_end},{lease_time}
dhcp-option=interface:{self.ethernet_interface},1,255.255.255.0
dhcp-option=interface:{self.ethernet_interface},3,{self.ethernet_gateway}
dhcp-option=interface:{self.ethernet_interface},6,{self.ethernet_gateway}
"""
        if include_wifi:
            cfg += f"""
# {self.wifi_interface}: data-plane (AP) DHCP (lease={lease_time})
interface={self.wifi_interface}
dhcp-range=interface:{self.wifi_interface},{net['wifi_subnet']}{self.dhcp_start},{net['wifi_subnet']}{self.dhcp_end},{lease_time}
dhcp-option=interface:{self.wifi_interface},1,255.255.255.0
dhcp-option=interface:{self.wifi_interface},3,{self.wifi_gateway}
dhcp-option=interface:{self.wifi_interface},6,{self.wifi_gateway}
"""
        return cfg

    def _generate_hostapd_config(self) -> str:
        # Empty `wifi_password` in config.toml => open AP (no WPA). The
        # demo cluster runs on a private switch + has no internet uplink,
        # so the security tradeoff is acceptable and avoids the long
        # tail of nmcli/NetworkManager secret-handling bugs that bit
        # us during the demo.
        if not self.wifi_password:
            return f"""
interface={self.wifi_interface}
driver=nl80211
ssid={self.wifi_ssid}
auth_algs=1
ignore_broadcast_ssid=0
macaddr_acl=0
hw_mode=a
channel=40
ieee80211n=0
ieee80211ac=1
wmm_enabled=1
"""
        return f"""
interface={self.wifi_interface}
driver=nl80211
ssid={self.wifi_ssid}
wpa_passphrase={self.wifi_password}
wpa=2
wpa_key_mgmt=WPA-PSK
auth_algs=1
ignore_broadcast_ssid=0
macaddr_acl=0
rsn_pairwise=CCMP
hw_mode=a
channel=40
ieee80211n=0
ieee80211ac=1
wmm_enabled=1
"""
