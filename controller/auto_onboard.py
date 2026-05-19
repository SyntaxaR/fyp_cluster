"""
Auto-onboard freshly-flashed Raspberry Pi workers.

Workflow (on every poll tick when ``[auto_onboard].enabled = true``):

  1. Read the dnsmasq lease file (``/tmp/dnsmasq-controller.leases``).
     Each leased IP is a candidate worker.
  2. Filter out:
       * the controller's own IP
       * IPs of already-registered workers
       * IPs we've already tried this run (success or fail)
  3. For each candidate, SSH-probe with the configured default credentials
     (``pi`` / ``raspberry`` on a fresh RPi OS image). If auth fails, mark
     the IP as "skipped" and never retry until controller restart.
  4. On a successful SSH:
       * tar the worker payload (``shared/`` + ``worker/`` + ``pyproject.toml``
         + ``worker.sh`` + ``config.toml`` + ``scripts/fyp-worker.service``)
         into an in-memory ``.tar.gz``
       * SFTP-upload it to ``/tmp/fyp_payload.tar.gz`` on the target
       * unpack to ``deploy_path`` (default ``/home/pi/fyp_cluster/latest``)
       * if ``install_systemd_unit`` is true, install + enable + start
         ``fyp-worker.service``; otherwise, just kick ``worker.sh`` in the
         background

Failures are logged loudly but never crash the controller — onboarding is
purely best-effort. Successful or definitively-failed IPs are remembered so
we don't hammer them every poll cycle.

Security note: this module deliberately uses factory-default credentials.
It's intended for closed lab benches where the operator owns every device
on the cluster subnet. Default ``enabled = false`` enforces opt-in.
"""
from __future__ import annotations

import asyncio
import io
import logging
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Files / directories from the project root we ship to the worker. Anything
# not listed here is left out of the tarball — keeps the payload small and
# avoids leaking controller-only bits (cluster.db, models/, datasets/, ...).
#
# Wheels live under res/wheels{,-hailo}/ on the controller and are served
# via /api/wheels HTTP endpoints — worker.sh fetches them directly from the
# controller after deploy. They're NOT bundled into the SFTP tarball, so
# the auto-onboard payload stays under 50 KB even when the wheel cache is
# hundreds of MB.
PAYLOAD_TOP_LEVEL = (
    "worker",
    "shared",
    "pyproject.toml",
    "worker.sh",
    "config.toml",
)
PAYLOAD_SCRIPTS = (
    "scripts/fyp-worker.service",
)
# Glob fragments to exclude when walking PAYLOAD_TOP_LEVEL — keeps caches /
# logs / build artefacts out of the tarball.
EXCLUDE_FRAGMENTS = (
    "__pycache__",
    ".pyc",
    ".log",
    ".db",
    ".db-shm",
    ".db-wal",
    ".part",
)


@dataclass
class OnboardResult:
    """Outcome of a single auto-onboard attempt against one worker IP.

    `step` records the FURTHEST stage reached. On success that's "complete";
    on failure it's the stage that raised. The dashboard / `/api/onboarding`
    surfaces this so the operator can immediately tell *where* the pipeline
    fell over.
    """
    ip: str
    status: str                 # "ok" | "ssh_failed" | "deploy_failed"
    step: str = "unknown"       # "ssh" | "sftp" | "untar" | "systemd" | "complete"
    error: Optional[str] = None
    stderr: Optional[str] = None       # captured stderr (truncated to 500 chars)
    stdout: Optional[str] = None       # captured stdout (truncated to 500 chars)
    duration_s: float = 0.0
    timestamp: int = field(default_factory=lambda: int(time.time()))


class DeployStepError(Exception):
    """Raised by _deploy_payload helpers — carries the failed step name +
    captured stdout/stderr so the caller can stuff them into OnboardResult.
    """
    def __init__(self, step: str, message: str,
                 stdout: str = "", stderr: str = ""):
        super().__init__(message)
        self.step = step
        self.stdout = stdout
        self.stderr = stderr


def _truncate(s: Optional[str], limit: int = 500) -> Optional[str]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return s if len(s) <= limit else s[:limit] + " …(truncated)"


class AutoOnboarder:
    """Polls dnsmasq leases, SSH-probes new IPs, deploys + starts worker."""

    def __init__(self, config: dict[str, Any], state, project_root: Path):
        self.cfg = config["auto_onboard"]
        self.controller_cfg = config["controller"]
        self.network_cfg = config["network"]
        self.state = state
        self.project_root = project_root

        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # IP -> OnboardResult (every IP we've examined, success or fail).
        self._results: dict[str, OnboardResult] = {}

        # Cache the tarball — built lazily on first need, reused per onboard.
        self._payload_cache: Optional[bytes] = None

        # First idle scan logs INFO; subsequent quiet ones drop to DEBUG.
        self._idle_logged: bool = False

    # =========================================================================
    # Lifecycle
    # =========================================================================
    def is_enabled(self) -> bool:
        return bool(self.cfg.get("enabled", False))

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if not self.is_enabled():
            # Promoted to WARNING so it's hard to miss in the journal —
            # the most common "auto-onboard isn't doing anything" cause is
            # simply that it was never enabled.
            logger.warning(
                "AutoOnboarder is DISABLED. Enable via "
                "config.toml [auto_onboard].enabled = true OR "
                "`./controller.sh --auto-onboard`."
            )
            return
        if self.is_running():
            return
        try:
            import paramiko  # noqa: F401  — fail fast if missing
        except ImportError as e:
            logger.error("AutoOnboarder requires paramiko: %s. "
                         "Run `uv add paramiko` (or pip install paramiko).", e)
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run())
        logger.warning(
            "AutoOnboarder ENABLED: polling every %.1fs as %s@* "
            "on subnet %s*",
            float(self.cfg["poll_interval_s"]),
            self.cfg["ssh_user"],
            self.controller_cfg.get("ethernet_interface", "eth?")
            and self.network_cfg.get("ethernet_subnet", "?."),
        )

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        self._task = None

    async def redeploy_to_registered(self) -> dict[str, dict[str, Any]]:
        """Force-push the worker payload to every currently-registered ACTIVE
        worker (regardless of auto-onboard state) and restart fyp-worker.

        Use case: after editing worker code on the controller, click the
        dashboard's "Restart controller + push to workers" button. We rsync
        the tarball, the worker's systemd unit restarts python with the new
        code. We assume the venv + wheels are unchanged (uv sync is fast in
        that case — see worker.sh's smoke-test path).

        Differs from the regular auto-onboard scan in three ways:
          * Iterates ``state.registered_workers`` instead of dnsmasq leases —
            works even if the lease pool is empty (e.g. just after a
            controller restart that wiped /tmp/dnsmasq leases).
          * Bypasses the ``self._results`` "already attempted" cache so a
            previously-onboarded IP is redeployed normally.
          * Skips the "no Hailo? skip wheels-hailo" branch — assumes worker
            already has whatever wheels it needs.

        Returns ``{ip: {"worker_id", "status", "step", "error"}}``.
        """
        # Lazy paramiko import — same as regular onboard. If it's missing
        # the redeploy can't happen so just bail out cleanly.
        try:
            import paramiko  # noqa: F401
        except ImportError as e:
            logger.error("Redeploy requires paramiko: %s", e)
            return {}

        # Collect targets: registered workers in ACTIVE state with a control_ip
        # that lives on the ethernet subnet (mirrors auto-onboard's filter so
        # we don't accidentally SSH a wifi-plane address).
        eth_prefix = self.network_cfg["ethernet_subnet"]
        targets: list[tuple[int, str]] = []
        for wid, reg in list(self.state.registered_workers.items()):
            ip = reg.control_ip
            if not ip or not ip.startswith(eth_prefix):
                continue
            # Be tolerant of both Enum and str status representations.
            status_ok = (
                reg.status == "active"
                or getattr(reg.status, "value", None) == "active"
            )
            if not status_ok:
                continue
            targets.append((wid, ip))

        if not targets:
            logger.warning("Redeploy: no ACTIVE workers to push to.")
            return {}

        logger.warning(
            "Redeploy: pushing payload + restarting fyp-worker on %d "
            "worker(s): %s", len(targets),
            [f"{wid}@{ip}" for wid, ip in targets],
        )

        # Run all onboards concurrently — each is a blocking SSH flow we
        # offload to a thread.
        async def _one(wid: int, ip: str) -> tuple[str, dict[str, Any]]:
            res = await asyncio.to_thread(self._onboard_blocking, ip)
            return ip, {
                "worker_id": wid,
                "status": res.status,
                "step": res.step,
                "error": res.error,
                "stderr": _truncate(res.stderr),
                "stdout": _truncate(res.stdout),
            }

        out_pairs = await asyncio.gather(
            *(_one(wid, ip) for wid, ip in targets),
            return_exceptions=False,
        )
        results = dict(out_pairs)

        ok = sum(1 for r in results.values() if r["status"] == "ok")
        logger.warning("Redeploy finished: %d/%d ok", ok, len(targets))
        return results

    async def restart_worker_via_ssh(self, ip: str) -> dict[str, Any]:
        """SSH into a worker and `systemctl restart fyp-worker`.

        For the manual "this worker is wedged" recovery path on the
        Overview page. Doesn't redeploy code or upload anything — just
        triggers a service restart, which is enough when the worker is
        alive enough for SSH but its python process is stuck.

        Returns ``{"status": "ok"|"failed", "error": str|None,
        "stdout": str, "stderr": str}``.
        """
        try:
            import paramiko  # noqa: F401
        except ImportError as e:
            return {"status": "failed",
                    "error": f"paramiko missing: {e}",
                    "stdout": "", "stderr": ""}

        def _do_it() -> dict[str, Any]:
            import paramiko
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                ssh.connect(
                    hostname=ip,
                    username=self.cfg["ssh_user"],
                    password=self.cfg["ssh_password"],
                    timeout=float(self.cfg.get("ssh_timeout_s", 8.0)),
                    allow_agent=False, look_for_keys=False,
                )
            except Exception as e:
                return {"status": "failed",
                        "error": f"ssh auth failed: {type(e).__name__}: {e}",
                        "stdout": "", "stderr": ""}
            try:
                pwd = self.cfg["ssh_password"]
                cmd = (
                    f"echo {pwd!r} | sudo -S -p '' "
                    f"systemctl restart fyp-worker"
                )
                rc, out, err = self._exec(ssh, cmd, timeout=30)
                if rc != 0:
                    return {"status": "failed",
                            "error": f"systemctl restart returned rc={rc}",
                            "stdout": _truncate(out), "stderr": _truncate(err)}
                return {"status": "ok", "error": None,
                        "stdout": _truncate(out), "stderr": _truncate(err)}
            finally:
                try:
                    ssh.close()
                except Exception:
                    pass

        return await asyncio.to_thread(_do_it)

    def results_snapshot(self) -> dict[str, dict[str, Any]]:
        """For UI / debugging — flatten OnboardResult into JSON-friendly dict.

        ``step`` tells you exactly where each IP's pipeline ended:
            ssh       — couldn't even authenticate
            sftp      — auth OK but tarball upload failed
            untar     — upload OK but remote untar/chmod failed
            systemd   — file deploy OK but `systemctl restart` failed
            complete  — full success
        Plus stdout/stderr from the failed step (truncated to 500 chars).
        """
        return {
            ip: {
                "status": r.status,
                "step": r.step,
                "error": r.error,
                "stdout": r.stdout,
                "stderr": r.stderr,
                "duration_s": r.duration_s,
                "timestamp": r.timestamp,
            }
            for ip, r in self._results.items()
        }

    # =========================================================================
    # Main loop
    # =========================================================================
    async def _run(self) -> None:
        interval = float(self.cfg.get("poll_interval_s", 10))
        while not self._stop_event.is_set():
            try:
                await self._scan_once()
            except Exception as e:
                logger.exception("AutoOnboarder scan failed: %s", e)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    async def _scan_once(self) -> None:
        # Re-read leases so we can log how the filter culled them — without
        # this, "grep auto journalctl" returns nothing in steady state and
        # the user can't tell if the loop is even alive.
        leases = self._read_leases()
        candidates = self._candidate_ips()
        eth_prefix = self.network_cfg["ethernet_subnet"]
        wifi_count = sum(1 for ip in leases if not ip.startswith(eth_prefix))
        eth_count = sum(1 for ip in leases if ip.startswith(eth_prefix))
        registered_count = len(self.state.registered_workers)
        attempted_count = len(self._results)

        # No candidates: log INFO once for the first idle scan after a
        # candidate window, then drop to DEBUG so the journal stays quiet.
        # The poll loop is still running every poll_interval_s — confirm
        # via `journalctl ... -p debug` or `curl /api/onboarding` if you
        # want to verify liveness without opening the log level.
        if not candidates:
            level = logger.debug if self._idle_logged else logger.info
            level(
                "AutoOnboarder scan: %d total leases (%d eth / %d wifi-skip), "
                "%d already registered, %d already attempted, 0 candidates.",
                len(leases), eth_count, wifi_count,
                registered_count, attempted_count,
            )
            self._idle_logged = True
            return

        # Found new candidates — always log loud.
        logger.info(
            "AutoOnboarder scan: %d total leases (%d eth), "
            "%d already registered, %d already attempted, %d candidates: %s",
            len(leases), eth_count, registered_count, attempted_count,
            len(candidates), candidates,
        )
        self._idle_logged = False
        for ip in candidates:
            if self._stop_event.is_set():
                return
            await self._onboard(ip)

    def _candidate_ips(self) -> list[str]:
        """IPs from dnsmasq leases we haven't dealt with yet.

        Filters out:
          * The controller's own IP (always {eth_subnet}.1).
          * Any IP NOT in the ethernet subnet — WiFi-subnet clients are
            laptops / phones / the experiment PC; we don't want to SSH
            into them with factory Pi credentials. Workers always join the
            cluster via wired ethernet first, so the wired DHCP lease is
            the authoritative onboard signal.
          * IPs already registered as workers.
          * IPs we've already tried (success or definitive failure) — avoids
            re-hammering bad-credential devices every 10 s.
        """
        leases = self._read_leases()
        eth_prefix = self.network_cfg["ethernet_subnet"]   # e.g. "192.168.10."
        controller_ip = f"{eth_prefix}1"
        registered = {
            reg.control_ip
            for reg in self.state.registered_workers.values()
            if reg.control_ip
        }
        out = []
        for ip in leases:
            if not ip.startswith(eth_prefix):
                # WiFi-plane lease (192.168.20.x) — never auto-onboard.
                continue
            if ip == controller_ip:
                continue
            if ip in registered:
                continue
            if ip in self._results:
                continue
            out.append(ip)
        return out

    def _read_leases(self) -> list[str]:
        """Parse dnsmasq's lease file. Returns a deduped list of leased IPs."""
        path = Path("/tmp/dnsmasq-controller.leases")
        if not path.exists():
            return []
        out: list[str] = []
        seen: set[str] = set()
        try:
            with path.open() as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 3:
                        ip = parts[2]
                        if ip not in seen:
                            seen.add(ip)
                            out.append(ip)
        except Exception as e:
            logger.warning("Failed to read lease file %s: %s", path, e)
        return out

    # =========================================================================
    # Per-IP onboard flow
    # =========================================================================
    async def _onboard(self, ip: str) -> None:
        logger.info("[onboard %s] >>> starting auto-onboard pipeline", ip)
        t0 = time.monotonic()
        # paramiko is sync — run the whole onboard in a worker thread so we
        # don't block the controller's event loop on slow SSH handshakes.
        result = await asyncio.to_thread(self._onboard_blocking, ip)
        result.duration_s = round(time.monotonic() - t0, 2)
        self._results[ip] = result

        if result.status == "ok":
            logger.info("[onboard %s] <<< SUCCESS in %.2fs (step=complete)",
                        ip, result.duration_s)
        elif result.status == "ssh_failed":
            logger.warning(
                "[onboard %s] <<< SSH PROBE FAILED in %.2fs: %s "
                "(skipping permanently — restart controller to retry)",
                ip, result.duration_s, result.error,
            )
        else:
            logger.error(
                "[onboard %s] <<< DEPLOY FAILED at step=%s in %.2fs: %s",
                ip, result.step, result.duration_s, result.error,
            )
            if result.stderr:
                logger.error("[onboard %s]     stderr: %s", ip, result.stderr)
            if result.stdout:
                logger.error("[onboard %s]     stdout: %s", ip, result.stdout)

    def _onboard_blocking(self, ip: str) -> OnboardResult:
        import paramiko

        # ---- step 1: SSH connect + auth ----
        logger.info("[onboard %s] step 1/4: SSH connect (port=%s, user=%s)…",
                    ip, self.cfg["ssh_port"], self.cfg["ssh_user"])
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        t_ssh = time.monotonic()
        try:
            ssh.connect(
                hostname=ip,
                port=int(self.cfg["ssh_port"]),
                username=self.cfg["ssh_user"],
                password=self.cfg["ssh_password"],
                timeout=float(self.cfg["ssh_timeout_s"]),
                allow_agent=False,
                look_for_keys=False,
            )
        except paramiko.AuthenticationException:
            return OnboardResult(ip, "ssh_failed", step="ssh",
                                 error="auth failed (pi/raspberry rejected)")
        except (paramiko.SSHException, OSError) as e:
            return OnboardResult(ip, "ssh_failed", step="ssh", error=str(e))
        logger.info("[onboard %s] step 1/4: SSH connect OK (%.2fs)",
                    ip, time.monotonic() - t_ssh)

        # ---- steps 2-4: SFTP / untar / systemctl ----
        try:
            self._deploy_payload(ssh, ip)
        except DeployStepError as e:
            return OnboardResult(ip, "deploy_failed", step=e.step,
                                 error=str(e),
                                 stdout=_truncate(e.stdout),
                                 stderr=_truncate(e.stderr))
        except Exception as e:
            return OnboardResult(ip, "deploy_failed", step="unknown",
                                 error=f"{type(e).__name__}: {e}")
        finally:
            try:
                ssh.close()
            except Exception:
                pass
        return OnboardResult(ip, "ok", step="complete")

    # ------------------------------------------------------------------------
    # Payload tarball + SFTP push + remote untar + service install
    # ------------------------------------------------------------------------
    def _build_payload(self) -> bytes:
        """Pack the worker-side payload into a single in-memory .tar.gz.

        Cached after first call — the project tree doesn't change during a
        run so re-tarring on every onboard is wasteful.
        """
        if self._payload_cache is not None:
            return self._payload_cache

        def _filter(tarinfo):
            name = tarinfo.name
            for frag in EXCLUDE_FRAGMENTS:
                if frag in name:
                    return None
            return tarinfo

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for rel in PAYLOAD_TOP_LEVEL:
                src = self.project_root / rel
                if not src.exists():
                    logger.warning("Payload missing %s — skipping.", src)
                    continue
                tf.add(src, arcname=rel, filter=_filter)
            for rel in PAYLOAD_SCRIPTS:
                src = self.project_root / rel
                if src.exists():
                    tf.add(src, arcname=rel, filter=_filter)
        self._payload_cache = buf.getvalue()
        logger.info("AutoOnboarder payload built: %d bytes",
                    len(self._payload_cache))
        return self._payload_cache

    def _deploy_payload(self, ssh, ip: str) -> None:
        """Upload + unpack + start. Raises DeployStepError on any failure
        with the step name + captured stdout/stderr attached."""
        deploy_path = self.cfg["deploy_path"]
        ssh_pwd = self.cfg["ssh_password"]
        # Use sudo -S so we can pipe the password if the user isn't NOPASSWD.
        # On a stock RPi OS image the `pi` user has NOPASSWD anyway, but the
        # -S path covers tightened images too.
        sudo_prefix = f"echo {ssh_pwd!r} | sudo -S -p ''"

        # ---- step 2: SFTP put the tarball ------------------------------------
        payload = self._build_payload()
        logger.info("[onboard %s] step 2/4: SFTP upload (%d bytes)…",
                    ip, len(payload))
        t = time.monotonic()
        try:
            sftp = ssh.open_sftp()
            try:
                with sftp.file("/tmp/fyp_payload.tar.gz", "wb") as f:
                    f.write(payload)
            finally:
                sftp.close()
        except Exception as e:
            raise DeployStepError("sftp",
                                  f"SFTP upload failed: {type(e).__name__}: {e}")
        logger.info("[onboard %s] step 2/4: SFTP upload OK (%.2fs)",
                    ip, time.monotonic() - t)

        # ---- step 3: untar + chmod -------------------------------------------
        # The trailing `chmod -R 777` on the fyp_cluster parent dir works
        # around a permission grab-bag that bites mixed sudo/non-sudo
        # deploys: a previous run executed under root (systemd) leaves
        # bits of .venv / __pycache__ owned by root:root mode 644, then
        # the next deploy as `pi` can't `tar --overwrite` over them and
        # fails halfway. 777 -R is overkill for security but the cluster
        # is on a private network and we want robustness over least-priv.
        # We chmod the PARENT (e.g. /home/pi/fyp_cluster) so previous
        # `latest/` contents AND any sibling deploy slots are reset.
        logger.info("[onboard %s] step 3/4: untar to %s …",
                    ip, deploy_path)
        t = time.monotonic()
        cmd = (
            f"mkdir -p {deploy_path!r} && "
            f"tar -xzf /tmp/fyp_payload.tar.gz -C {deploy_path!r} --overwrite && "
            f"rm -f /tmp/fyp_payload.tar.gz && "
            f"chmod +x {deploy_path!r}/worker.sh && "
            # `dirname` resolves /home/pi/fyp_cluster/latest -> /home/pi/fyp_cluster.
            # `|| true` keeps the deploy alive if e.g. a sibling dir has
            # something we can't chmod (rare — /home/pi/.cache shouldn't
            # be a child of fyp_cluster, but defensive).
            f"{sudo_prefix} chmod -R 777 \"$(dirname {deploy_path!r})\" || true"
        )
        rc, out, err = self._exec(ssh, cmd, timeout=120)
        if rc != 0:
            raise DeployStepError(
                "untar",
                f"untar/chmod failed (rc={rc})",
                stdout=out, stderr=err,
            )
        logger.info("[onboard %s] step 3/4: untar OK (%.2fs)",
                    ip, time.monotonic() - t)

        # ---- step 4: install systemd unit + start service --------------------
        logger.info("[onboard %s] step 4/4: install + start fyp-worker …", ip)
        t = time.monotonic()
        if self.cfg.get("install_systemd_unit", True):
            unit_src = f"{deploy_path}/scripts/fyp-worker.service"
            cmd = (
                f"{sudo_prefix} cp -f {unit_src!r} /etc/systemd/system/ && "
                f"{sudo_prefix} systemctl daemon-reload && "
                f"{sudo_prefix} systemctl enable fyp-worker && "
                f"{sudo_prefix} systemctl restart fyp-worker"
            )
        else:
            cmd = (
                f"cd {deploy_path!r} && "
                f"nohup bash worker.sh > /tmp/fyp_worker.log 2>&1 &"
            )
        rc, out, err = self._exec(ssh, cmd, timeout=60)
        if rc != 0:
            raise DeployStepError(
                "systemd",
                f"worker start failed (rc={rc})",
                stdout=out, stderr=err,
            )
        logger.info("[onboard %s] step 4/4: systemctl restart OK (%.2fs)",
                    ip, time.monotonic() - t)

    @staticmethod
    def _exec(ssh, cmd: str, timeout: float = 30.0) -> tuple[int, str, str]:
        """Run a remote shell command, return (exit_code, stdout, stderr)."""
        # paramiko's exec_command runs the command as the SSH user's login
        # shell, which on Pi is bash. We don't need a tty — sudo -S reads
        # stdin via the echo pipe.
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
        rc = stdout.channel.recv_exit_status()
        try:
            out = stdout.read().decode("utf-8", errors="replace")
        except Exception:
            out = ""
        try:
            err = stderr.read().decode("utf-8", errors="replace")
        except Exception:
            err = ""
        return rc, out, err
