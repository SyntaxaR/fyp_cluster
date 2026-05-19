"""Dispatcher plugins. Pick one via config['dispatcher']['algorithm']."""
from __future__ import annotations

import logging
from pathlib import Path

from controller.dispatcher.base import BaseDispatcher
from controller.dispatcher.round_robin import RoundRobinDispatcher
from controller.dispatcher.weighted_round_robin import WeightedRoundRobinDispatcher
from shared.util import load_dispatcher

logger = logging.getLogger(__name__)

ALGORITHMS = {
    "round_robin": RoundRobinDispatcher,
    "weighted_round_robin": WeightedRoundRobinDispatcher,
}

# Where the controller looks for user-uploaded dispatcher plugins (.py).
# Mirrors demo/adapters/ — written by the Web UI's Upload control and served
# by the data API at /api/dispatchers.
from controller._paths import DISPATCHERS_DIR  # noqa: E402


def make_dispatcher(name: str) -> BaseDispatcher:
    """Construct a dispatcher.

    Resolution order:
      1. Built-in algorithm name (round_robin, weighted_round_robin)
      2. Uploaded plugin in ./dispatchers/<name>.py — class `Dispatcher`
      3. Uploaded plugin in ./dispatchers/<name>     (already includes .py)
    """
    cls = ALGORITHMS.get(name)
    if cls is not None:
        return cls()

    # Try as a plugin filename (with or without .py)
    candidates: list[Path] = []
    if name.endswith(".py"):
        candidates.append(DISPATCHERS_DIR / name)
    else:
        candidates.append(DISPATCHERS_DIR / f"{name}.py")
        candidates.append(DISPATCHERS_DIR / name)
    for cand in candidates:
        if cand.exists() and cand.is_file():
            logger.info("Loading dispatcher plugin from %s", cand)
            return load_dispatcher(cand)

    raise ValueError(
        f"Unknown dispatcher '{name}'. "
        f"Built-ins: {list(ALGORITHMS)}; "
        f"plugin must exist under {DISPATCHERS_DIR}/"
    )


def list_dispatcher_choices() -> list[str]:
    """Built-in algorithms + uploaded plugin stems (sans .py)."""
    out = list(ALGORITHMS.keys())
    if DISPATCHERS_DIR.exists():
        for p in sorted(DISPATCHERS_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() == ".py":
                out.append(p.stem)
    return out
