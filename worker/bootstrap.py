"""
Worker bootstrapping:

1. Bring eth0 up via DHCP (handled by WorkerNetworkController).
2. Infer the controller's IP as `{eth_subnet}.1` — or use FYP_CONTROLLER_HOST
   in mock mode.
3. GET /api/get_config — replace local config with the controller's view.
4. Start heartbeating with worker_id=-1 until the controller assigns one.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)


def fetch_controller_config(eth_subnet_prefix: str, control_port: int,
                            timeout: int = 10,
                            controller_host: str | None = None) -> dict[str, Any] | None:
    """Hit the controller's /api/get_config endpoint.

    If ``controller_host`` is provided (or FYP_CONTROLLER_HOST is set), it
    overrides the {subnet}.1 derivation — used by mock mode.
    """
    host = (controller_host
            or os.environ.get("FYP_CONTROLLER_HOST", "").strip()
            or f"{eth_subnet_prefix}1")
    url = f"http://{host}:{control_port}/api/get_config"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            body = r.json()
            return body.get("config")
        logger.error("get_config returned HTTP %d", r.status_code)
    except Exception as e:
        logger.error("get_config failed: %s", e)
    return None


def merge_remote_config(local: dict[str, Any], remote: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge: remote wins on every section, but local keys not present
    in remote are preserved (useful for worker-only sections)."""
    merged = {**local}
    for k, v in remote.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            sub = {**merged[k], **v}
            merged[k] = sub
        else:
            merged[k] = v
    return merged
