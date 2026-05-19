"""
Smooth weighted round-robin (Nginx-style).

Each worker has an effective weight w_i. We track current_weight_i, and on
each pick:
    current_weight_i += w_i for all i
    pick the worker with the highest current_weight
    decrement that worker's current_weight by sum(w)
This gives a deterministic, balanced sequence even with non-uniform weights.
"""
from __future__ import annotations

from typing import Optional

from controller.dispatcher.base import BaseDispatcher


class WeightedRoundRobinDispatcher(BaseDispatcher):
    name = "weighted_round_robin"

    def __init__(self) -> None:
        super().__init__()
        self._current: dict[int, float] = {}

    def reset(self) -> None:
        self._current.clear()

    def next(self, active_worker_ids: list[int]) -> Optional[int]:
        if not active_worker_ids:
            return None

        # default weight = 1 if unspecified
        weights = {wid: max(0.0, float(self._weights.get(wid, 1.0)))
                   for wid in active_worker_ids}
        total = sum(weights.values())
        if total <= 0:
            # All weights zero -> degrade to first id
            return sorted(active_worker_ids)[0]

        # Update current weights
        for wid in active_worker_ids:
            self._current[wid] = self._current.get(wid, 0.0) + weights[wid]

        # Pick the maximum current_weight; tie-break by id for determinism
        chosen = max(
            active_worker_ids,
            key=lambda wid: (self._current.get(wid, 0.0), -wid),
        )
        self._current[chosen] -= total
        return chosen
