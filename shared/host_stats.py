"""
Lightweight host telemetry — used by both controller and worker to populate
the Monitor page without taking on a psutil dependency.

Returns a single dict:

    {
        "cpu_temp_c":   float | None,
        "cpu_usage_pct": float | None,
        "npu_temp_c":   float | None,   # Hailo only; None on non-NPU hosts
    }

All probes are best-effort: if a kernel file is missing or a Hailo call
raises, the corresponding key returns ``None`` rather than crashing the
caller. The caller decides how to render missing values.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import Optional

logger = logging.getLogger(__name__)

# =============================================================================
# CPU usage — parses /proc/stat between two snapshots so we don't pull in
# psutil. We cache the previous snapshot in module state so that a single
# `collect()` call returns the usage % since the last call.
# =============================================================================
_prev_cpu: tuple[int, int] | None = None     # (idle_jiffies, total_jiffies)


def _read_proc_stat_cpu() -> tuple[int, int] | None:
    """Return (idle, total) jiffies for the aggregate CPU line, or None."""
    try:
        with open("/proc/stat", "r") as f:
            for line in f:
                if not line.startswith("cpu "):
                    continue
                parts = line.split()
                # cpu  user nice system idle iowait irq softirq steal guest guest_nice
                values = [int(x) for x in parts[1:]]
                idle = values[3] + (values[4] if len(values) > 4 else 0)  # idle + iowait
                total = sum(values)
                return idle, total
    except (OSError, ValueError) as e:
        logger.debug("CPU stat read failed: %s", e)
    return None


def cpu_usage_pct() -> Optional[float]:
    """Aggregate CPU usage % since the last call to this function.

    First call returns ``None`` (no prior snapshot to diff against). Reading
    the file is a few microseconds — calling once per UI tick is cheap.
    """
    global _prev_cpu
    cur = _read_proc_stat_cpu()
    if cur is None:
        return None
    if _prev_cpu is None:
        _prev_cpu = cur
        return None
    idle_d = cur[0] - _prev_cpu[0]
    total_d = cur[1] - _prev_cpu[1]
    _prev_cpu = cur
    if total_d <= 0:
        return None
    return max(0.0, min(100.0, (1.0 - idle_d / total_d) * 100.0))


# =============================================================================
# CPU temperature
# =============================================================================
def cpu_temp_c() -> Optional[float]:
    """Pi 5 / Pi 4 expose the SoC temperature at thermal_zone0."""
    # Try sysfs first (fastest, no subprocess).
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            millideg = int(f.read().strip())
            return millideg / 1000.0
    except (OSError, ValueError):
        pass

    # Fall back to vcgencmd if sysfs is missing (rare, but covers Pi-on-Pi
    # OS variants where thermal_zone0 is a different sensor).
    try:
        out = subprocess.run(
            ["vcgencmd", "measure_temp"],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0:
            # Output: "temp=58.4'C"
            txt = out.stdout.strip()
            if "=" in txt:
                num = txt.split("=", 1)[1].rstrip("'C\n ")
                return float(num)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return None


# =============================================================================
# Hailo NPU temperature — best-effort, requires hailo_platform AND a real chip.
# We cache the device handle across calls because creating a VDevice is
# expensive (hundreds of ms) and reuses the running engine's hardware
# anyway via the round-robin scheduler.
# =============================================================================
_hailo_temp_disabled = False


def npu_temp_c() -> Optional[float]:
    """Best-effort Hailo chip temperature in Celsius.

    Returns ``None`` if hailo_platform isn't installed, no Hailo chip is
    present, or the SDK doesn't expose a temperature method on this version.
    Once a probe fails we mark Hailo as disabled and skip future attempts —
    a missing NPU shouldn't pay the import cost on every UI tick.
    """
    global _hailo_temp_disabled
    if _hailo_temp_disabled:
        return None
    try:
        # Lazy import — we don't want a non-Hailo controller to fail to
        # start just because hailo_platform isn't installed.
        from hailo_platform import Device  # type: ignore
    except Exception:
        _hailo_temp_disabled = True
        return None

    # The Hailo Python SDK has shipped at least three different temperature
    # APIs across 4.x versions:
    #   * Device.get_chip_temperature() -> ChipTemperatureInfo(ts0, ts1)
    #   * Device.control.get_chip_temperature()  -> same
    #   * Device.read_chip_temperature() (legacy)
    # We try them in order and return the first successful read.
    try:
        scan = Device.scan()  # list of device IDs; empty on no hardware
    except Exception:
        _hailo_temp_disabled = True
        return None
    if not scan:
        _hailo_temp_disabled = True
        return None

    dev = None
    try:
        dev = Device(scan[0])
    except Exception:
        _hailo_temp_disabled = True
        return None

    temp_c: Optional[float] = None
    for attr in ("get_chip_temperature", "read_chip_temperature"):
        if not hasattr(dev, attr):
            continue
        try:
            info = getattr(dev, attr)()
            # Most variants return an object with .ts0_temperature / .ts1_temperature
            ts0 = getattr(info, "ts0_temperature", None) or getattr(info, "ts0", None)
            ts1 = getattr(info, "ts1_temperature", None) or getattr(info, "ts1", None)
            samples = [float(t) for t in (ts0, ts1) if t is not None]
            if samples:
                temp_c = max(samples)
                break
        except Exception:
            continue

    try:
        dev.release()
    except Exception:
        pass
    return temp_c


# =============================================================================
# Public collect() — what the UI / heartbeat path calls.
# =============================================================================
def collect() -> dict:
    """Single-call snapshot of host telemetry. Safe to call every UI tick."""
    return {
        "cpu_temp_c": cpu_temp_c(),
        "cpu_usage_pct": cpu_usage_pct(),
        "npu_temp_c": npu_temp_c(),
        "ts": time.time(),
    }
