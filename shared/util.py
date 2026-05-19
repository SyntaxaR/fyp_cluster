"""
Hardware helpers, identifier generation and dynamic adapter loading.
"""
from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import socket
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Hardware identification
# =============================================================================
def get_cpu_serial() -> str:
    """Read the Raspberry Pi CPU serial from /proc/cpuinfo. Falls back to MAC.

    Honours env override FYP_CPU_SERIAL — useful in mock mode when running
    several worker processes on the same host that need distinct identities.
    """
    override = os.environ.get("FYP_CPU_SERIAL", "").strip()
    if override:
        return override

    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("Serial"):
                    return line.split(":", 1)[1].strip()
    except FileNotFoundError:
        logger.warning("/proc/cpuinfo not available, falling back to MAC address")

    # MAC fallback (best-effort, not RPi-specific)
    try:
        import uuid
        mac = uuid.getnode()
        return f"mac{mac:012x}"
    except Exception as e:
        logger.error(f"Failed to obtain a hardware serial: {e}")
        return f"host{socket.gethostname()}"


ANIMALS = [
    "Panda", "Tiger", "Eagle", "Whale", "Bear", "Wolf", "Fox", "Hawk",
    "Deer", "Seal", "Otter", "Lynx", "Owl", "Swan", "Crane", "Falcon",
    "Koala", "Zebra", "Giraffe", "Rhino", "Hippo", "Puma", "Jaguar", "Cheetah",
    "Leopard", "Rabbit", "Mouse", "Squirrel", "Dolphin", "Shark", "Cat", "Fish",
]

ADJECTIVES = [
    "Swift", "Brave", "Calm", "Wise", "Quick", "Bright", "Keen", "Bold",
    "Cool", "Warm", "Fast", "Slow", "Kind", "Neat", "Safe", "Pure",
    "Rare", "Vast", "Wild", "Young", "Agile", "Clear", "Crisp", "Dense",
    "Eager", "Fancy", "Fleet", "Fresh", "Giant", "Grand", "Happy", "Jolly",
    "Light", "Lively", "Lucky", "Merry", "Noble", "Proud", "Quiet", "Rapid",
    "Royal", "Sharp", "Smart", "Snowy", "Solid", "Spry", "Stark", "Stout",
    "Sturdy", "Sunny", "Super", "Tidy", "Tiny", "Vivid", "Witty", "Zesty",
]


def generate_identifier(serial: str) -> str:
    """Deterministic 'Adjective-Animal' nickname from the hardware serial."""
    hash_bytes = hashlib.md5(serial.encode()).digest()
    adj_index = int.from_bytes(hash_bytes[:4], "big") % len(ADJECTIVES)
    animal_index = int.from_bytes(hash_bytes[4:8], "big") % len(ANIMALS)
    return f"{ADJECTIVES[adj_index]}-{ANIMALS[animal_index]}"


# =============================================================================
# Integrity hashing
# =============================================================================
def md5_of_file(path: str | os.PathLike, chunk_size: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def md5_of_bytes(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


# =============================================================================
# Dynamic ModelAdapter loading
# =============================================================================
def load_adapter(adapter_path: str | os.PathLike) -> Any:
    """
    Load `ModelAdapter` from a user-supplied .py file via importlib.

    The adapter module must define a top-level class named `ModelAdapter`
    matching the worker.inference.model_adapter_template.ModelAdapter API.
    """
    import sys
    path = Path(adapter_path)
    if not path.exists():
        raise FileNotFoundError(f"Adapter file not found: {path}")

    mod_name = f"user_adapter_{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot create import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec so dataclass / typing introspection
    # (which calls sys.modules[cls.__module__]) finds the module.
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise

    if not hasattr(module, "ModelAdapter"):
        raise ValueError(f"Adapter file {path} must define a top-level `ModelAdapter` class")

    # Instantiate and return the adapter. Earlier versions of this
    # function validated the class exists but forgot the `return`,
    # which made every `self.adapter = load_adapter(path)` evaluate
    # to `None` — silently breaking raw-mode inference everywhere
    # without producing any error at load time. The engine then
    # raised "raw mode requires adapter" on the first request, far
    # from the actual bug.
    adapter_cls = getattr(module, "ModelAdapter")
    try:
        return adapter_cls()
    except Exception as e:
        # Wrap with the file path so the worker's load_model error
        # tells the operator which file blew up during __init__.
        raise RuntimeError(
            f"Adapter file {path} defines ModelAdapter but instantiation "
            f"failed: {type(e).__name__}: {e}"
        ) from e


def adapter_supported_modes(adapter_path: str | os.PathLike) -> set[str]:
    """Return the set of inference modes the adapter at ``adapter_path``
    declares it supports.

    Adapters MAY define a class-level attribute ``SUPPORTED_MODES``
    (a set / frozenset / tuple / list of mode names). The controller
    UI uses this to grey out unsupported modes in the experiment
    dropdown.

    For backwards compatibility with adapters that don't declare it,
    this returns ``{"tensor", "raw", "dummy"}`` — the full set — so
    older adapters keep working unchanged.

    Returns the empty set if the adapter file fails to load (the UI
    treats that as "no constraint" rather than crashing).
    """
    ALL_MODES = {"tensor", "raw", "dummy"}
    try:
        from shared.util import load_adapter as _load  # self-import-safe
        adapter = _load(adapter_path)
    except Exception as e:
        logger.warning("adapter_supported_modes: load failed for %s: %s",
                       adapter_path, e)
        return ALL_MODES

    cls = type(adapter)
    raw = getattr(cls, "SUPPORTED_MODES", None)
    if raw is None:
        # No declaration → assume all modes (backward compatible).
        return ALL_MODES
    try:
        modes = {str(m).lower() for m in raw}
    except TypeError:
        logger.warning(
            "adapter_supported_modes: %s.SUPPORTED_MODES is not iterable "
            "(got %r) — treating as 'all modes'.", adapter_path, raw,
        )
        return ALL_MODES
    # Filter to known modes only — silently drop typos like "Dummy".
    cleaned = modes & ALL_MODES
    if not cleaned:
        logger.warning(
            "adapter_supported_modes: %s declared SUPPORTED_MODES=%r "
            "but none match {tensor, raw, dummy} — treating as 'all modes'.",
            adapter_path, raw,
        )
        return ALL_MODES
    return cleaned

    return module.ModelAdapter()


# =============================================================================
# Dynamic Dispatcher loading (mirrors load_adapter)
# =============================================================================
def load_dispatcher(dispatcher_path: str | os.PathLike) -> Any:
    """
    Load `Dispatcher` from a user-supplied .py file via importlib.

    The dispatcher module must define a top-level class named `Dispatcher`
    that subclasses controller.dispatcher.base.BaseDispatcher (or quacks the
    same: `next(active_worker_ids)` returning Optional[int]).
    """
    import sys
    path = Path(dispatcher_path)
    if not path.exists():
        raise FileNotFoundError(f"Dispatcher file not found: {path}")

    mod_name = f"user_dispatcher_{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot create import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise

    if not hasattr(module, "Dispatcher"):
        raise ValueError(
            f"Dispatcher file {path} must define a top-level `Dispatcher` class"
        )

    return module.Dispatcher()


# =============================================================================
# Misc
# =============================================================================
def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def first_ipv4_in_subnet(addresses: list[str], subnet_prefix: str) -> str | None:
    """Pick the first address whose dotted-quad starts with `subnet_prefix`."""
    for a in addresses:
        if a.startswith(subnet_prefix):
            return a
    return None
