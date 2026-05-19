"""Plain round-robin dispatcher."""
from __future__ import annotations

from typing import Optional

from controller.dispatcher.base import BaseDispatcher


class RoundRobinDispatcher(BaseDispatcher):
    name = "round_robin"

    def __init__(self) -> None:
        super().__init__()
        self._cursor = 0

    def next(self, active_worker_ids: list[int]) -> Optional[int]:
        if not active_worker_ids:
            return None
        ids = sorted(active_worker_ids)
        chosen = ids[self._cursor % len(ids)]
        self._cursor = (self._cursor + 1) % len(ids)
        return chosen

    def reset(self) -> None:
        self._cursor = 0
