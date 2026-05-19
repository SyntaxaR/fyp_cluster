"""Re-export of shared.host_stats so callers can write
``from controller.host_stats import collect``."""
from __future__ import annotations

from shared.host_stats import collect, cpu_temp_c, cpu_usage_pct, npu_temp_c

__all__ = ["collect", "cpu_temp_c", "cpu_usage_pct", "npu_temp_c"]
