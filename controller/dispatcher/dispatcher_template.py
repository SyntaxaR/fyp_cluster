"""
Template for a user-uploaded dispatcher plugin.

Drop a copy of this file into ``dispatchers/<your_name>.py`` (via the Web UI
upload control or by hand) and the controller will dynamically import it via
``shared.util.load_dispatcher``. The file MUST define a top-level class named
``Dispatcher``.

The class can subclass ``controller.dispatcher.base.BaseDispatcher`` (which
gives it ``set_weights`` / ``reset`` for free) or just duck-type ``next``.
"""
from __future__ import annotations

from typing import Optional


class Dispatcher:
    """Pick a worker from the active pool. Replace this with your own logic."""

    name = "least_loaded"

    def __init__(self) -> None:
        self._weights: dict[int, float] = {}
        self._counts: dict[int, int] = {}

    # Optional — controller calls these if present.
    def set_weights(self, weights: dict[int, float]) -> None:
        self._weights = {int(k): float(v) for k, v in weights.items()}

    def reset(self) -> None:
        self._counts.clear()

    def next(self, active_worker_ids: list[int]) -> Optional[int]:
        if not active_worker_ids:
            return None
        # Toy "least dispatched" strategy — pick the worker we sent the
        # fewest requests to so far. Ties broken by lowest id.
        chosen = min(
            active_worker_ids,
            key=lambda wid: (self._counts.get(wid, 0), wid),
        )
        self._counts[chosen] = self._counts.get(chosen, 0) + 1
        return chosen
