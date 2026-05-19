"""Dispatcher base class."""
from __future__ import annotations

import abc
from typing import Optional


class BaseDispatcher(abc.ABC):
    """Pick the next worker_id from the active worker pool.

    Implementations should be cheap and stateless apart from a small cursor /
    accumulator. Locks are unnecessary because dispatch happens on the
    controller's asyncio event loop.
    """

    name: str = "base"

    def __init__(self) -> None:
        self._weights: dict[int, float] = {}

    @abc.abstractmethod
    def next(self, active_worker_ids: list[int]) -> Optional[int]:
        """Return the next worker_id to dispatch to, or None if pool is empty."""

    def set_weights(self, weights: dict[int, float]) -> None:
        """Set explicit weights. Default base impl just stores them."""
        self._weights = {wid: max(0.0, float(w)) for wid, w in weights.items()}

    def reset(self) -> None:
        """Reset internal state at the start of a new experiment."""
