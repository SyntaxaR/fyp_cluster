"""
Controller's DATA plane FastAPI app.

Endpoints:
    GET  /api/connectivity_test       — Plane verification from worker side
    GET  /api/models/{name}           — Stream a model file (.onnx / .hef)
    GET  /api/models                  — List models the controller has on disk
    POST /api/upload_model            — Dashboard upload of model file
    DELETE /api/models/{name}         — Dashboard delete of model file
    GET  /api/adapters/{name}         — Stream a model adapter .py
    GET  /api/adapters                — List adapter .py files
    POST /api/upload_adapter          — Dashboard upload of adapter .py file
    DELETE /api/adapters/{name}       — Dashboard delete of adapter .py
    GET  /api/dispatchers             — List dispatcher .py uploads
    POST /api/upload_dispatcher       — Dashboard upload of custom dispatcher
    DELETE /api/dispatchers/{name}    — Dashboard delete of dispatcher .py
    GET  /api/datasets                — List uploaded datasets (raw mode)
    POST /api/upload_dataset          — Dashboard upload of dataset zip / tarball
    DELETE /api/datasets/{name}       — Dashboard delete of dataset
    GET  /api/datasets/{name}         — Stream a dataset file
    POST /api/distribute_model        — Push a model+adapter to workers (manual gate)
    GET  /api/distribution_status     — Per-(model, worker) distribution status
    POST /api/inference_results       — Workers POST per-request telemetry
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel

from shared.models import ConnectionType, ConnectivityTestResponse
from shared.util import md5_of_file


class DistributeModelRequest(BaseModel):
    model_name: str
    adapter_filename: str | None = None
    # Optional explicit per-worker backend ("onnx"|"hailo"). Missing entries
    # fall back to the worker's self-reported engine.
    engine_overrides: dict[int, str] = {}
    # Optional: restrict push to a subset of currently-ACTIVE workers.
    target_workers: list[int] | None = None

logger = logging.getLogger(__name__)

# Runtime directory layout — see controller/_paths.py for the canonical
# definition. All user-uploaded artefacts live under demo/, infrastructure
# (wheels, binaries) under res/, recording outputs under recordings/.
from controller._paths import (
    MODELS_DIR, ADAPTERS_DIR, DISPATCHERS_DIR, DATASETS_DIR,
    WHEELS_DIR, WHEELS_HAILO_DIR, BIN_DIR, RECORDINGS_DIR,
)


# Allowed dataset upload extensions — used by both the REST endpoint and
# the Web UI upload control so they stay in sync. Covers:
#   * archives (zip / tar / tar.gz)
#   * jsonl manifests
#   * loose images (jpg / png / bmp / webp)
#   * short video clips for the /live super-resolution demo
DATASET_EXTS = (
    ".zip", ".tar", ".gz", ".tgz", ".jsonl",
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
)

# Filename safety: alnum + ._-  (also no leading dot to avoid hidden files)
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*$")


def _safe_filename(name: str) -> str:
    """Validate a filename, returning the basename or raising HTTPException."""
    base = Path(name).name  # strip directories
    if not base or base in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not _SAFE_NAME_RE.match(base):
        raise HTTPException(
            status_code=400,
            detail=f"Filename '{base}' contains illegal characters "
                   "(allowed: letters, digits, dot, underscore, hyphen).",
        )
    return base


def _enforce_extension(name: str, allowed_exts: tuple[str, ...]) -> None:
    if Path(name).suffix.lower() not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Extension {Path(name).suffix!r} not allowed; "
                   f"expected one of {allowed_exts}.",
        )


def _resolve_under(directory: Path, filename: str) -> Path:
    """Resolve `directory/filename` and assert it stays inside directory."""
    base = _safe_filename(filename)
    target = (directory / base).resolve()
    root = directory.resolve()
    if root != target.parent and root not in target.parents:
        raise HTTPException(status_code=400, detail="Invalid path")
    return target


async def _save_upload(upload: UploadFile, dest: Path) -> int:
    """Stream UploadFile to disk; returns final byte count."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    written = 0
    try:
        with tmp.open("wb") as f:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                written += len(chunk)
        # On Windows, Path.replace() can fail with WinError 5 if the dest is
        # locked by an open handle. Drop the existing file first; if that
        # fails too, swap via os.replace which is atomic on POSIX and
        # tolerant on Windows when the target isn't locked.
        if dest.exists():
            try:
                dest.unlink()
            except PermissionError:
                # Best-effort: try anyway via os.replace, may overwrite.
                pass
        os.replace(tmp, dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return written


def _list_dir(directory: Path,
              extra_per_file=None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not directory.exists():
        return out
    for p in sorted(directory.iterdir()):
        if not p.is_file():
            continue
        # Hide dotfiles like .gitkeep / .DS_Store — they're never user
        # uploads and shouldn't clutter the dashboard's dropdowns.
        if p.name.startswith("."):
            continue
        entry: dict[str, Any] = {
            "name": p.stem,
            "filename": p.name,
            "size_bytes": p.stat().st_size,
            "modified_ms": int(p.stat().st_mtime * 1000),
        }
        if extra_per_file is not None:
            entry.update(extra_per_file(p))
        out.append(entry)
    return out


def make_data_app(deps: "DataDeps") -> FastAPI:
    app = FastAPI(title="Controller Data API", version="1.1.0")
    router = APIRouter()

    @router.get("/api/connectivity_test")
    async def connectivity_test(request: Request) -> ConnectivityTestResponse:
        host = request.client.host if request.client else ""
        eth_subnet = deps.config["network"]["ethernet_subnet"]
        wifi_subnet = deps.config["network"]["wifi_subnet"]
        if host.startswith(eth_subnet):
            plane = ConnectionType.ETHERNET
        elif host.startswith(wifi_subnet):
            plane = ConnectionType.WIFI
        else:
            plane = ConnectionType.INVALID
        return ConnectivityTestResponse(
            from_identifier=deps.identifier,
            message="data connectivity ok",
            plane=plane,
        )

    # =========================================================================
    # Models
    # =========================================================================
    @router.get("/api/models")
    async def list_models() -> list[dict[str, Any]]:
        def per_file(p: Path) -> dict[str, Any]:
            return {
                "backend": "hailo" if p.suffix.lower() == ".hef" else "onnx",
                "md5": md5_of_file(p),
            }
        return _list_dir(MODELS_DIR, per_file)

    @router.get("/api/models/{filename}")
    async def get_model(filename: str):
        target = _resolve_under(MODELS_DIR, filename)
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="Model not found")
        return FileResponse(target, filename=target.name,
                            media_type="application/octet-stream")

    @router.post("/api/upload_model")
    async def upload_model(file: UploadFile = File(...)) -> dict[str, Any]:
        name = _safe_filename(file.filename or "")
        _enforce_extension(name, (".onnx", ".hef"))
        target = _resolve_under(MODELS_DIR, name)
        size = await _save_upload(file, target)
        logger.info("Uploaded model %s (%d bytes)", target.name, size)
        return {
            "status": "ok",
            "filename": target.name,
            "size_bytes": size,
            "md5": md5_of_file(target),
            "backend": "hailo" if target.suffix.lower() == ".hef" else "onnx",
        }

    @router.delete("/api/models/{filename}")
    async def delete_model(filename: str) -> dict[str, str]:
        target = _resolve_under(MODELS_DIR, filename)
        if not target.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        target.unlink()
        # Drop distribution tracking for this model — workers may still hold
        # the file but it's no longer the source of truth, so the dashboard
        # should ask the user to re-distribute after re-uploading.
        if deps.state is not None:
            deps.state.clear_distribution(target.stem)
        logger.info("Deleted model %s", target.name)
        return {"status": "ok"}

    # =========================================================================
    # Adapters
    # =========================================================================
    @router.get("/api/adapters")
    async def list_adapters() -> list[dict[str, Any]]:
        return _list_dir(ADAPTERS_DIR)

    @router.get("/api/adapters/{filename}")
    async def get_adapter(filename: str):
        target = _resolve_under(ADAPTERS_DIR, filename)
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="Adapter not found")
        return FileResponse(target, filename=target.name, media_type="text/x-python")

    @router.post("/api/upload_adapter")
    async def upload_adapter(file: UploadFile = File(...)) -> dict[str, Any]:
        name = _safe_filename(file.filename or "")
        _enforce_extension(name, (".py",))
        target = _resolve_under(ADAPTERS_DIR, name)
        size = await _save_upload(file, target)
        logger.info("Uploaded adapter %s (%d bytes)", target.name, size)
        return {
            "status": "ok",
            "filename": target.name,
            "size_bytes": size,
            "md5": md5_of_file(target),
        }

    @router.delete("/api/adapters/{filename}")
    async def delete_adapter(filename: str) -> dict[str, str]:
        target = _resolve_under(ADAPTERS_DIR, filename)
        if not target.exists():
            raise HTTPException(status_code=404, detail="Adapter not found")
        target.unlink()
        # Any model that pinned this adapter is now stale — invalidate the
        # affected distribution entries so the user is forced to re-distribute
        # after picking a new adapter.
        if deps.state is not None:
            for mname, entries in list(deps.state.model_distribution.items()):
                for wid, st in list(entries.items()):
                    if st.adapter_filename == target.name:
                        st.status = "stale"
                        st.error = f"adapter '{target.name}' deleted"
        logger.info("Deleted adapter %s", target.name)
        return {"status": "ok"}

    # =========================================================================
    # Dispatchers (custom Python dispatchers — controller-side only)
    # =========================================================================
    @router.get("/api/dispatchers")
    async def list_dispatchers() -> list[dict[str, Any]]:
        return _list_dir(DISPATCHERS_DIR)

    @router.post("/api/upload_dispatcher")
    async def upload_dispatcher(file: UploadFile = File(...)) -> dict[str, Any]:
        name = _safe_filename(file.filename or "")
        _enforce_extension(name, (".py",))
        target = _resolve_under(DISPATCHERS_DIR, name)
        size = await _save_upload(file, target)
        logger.info("Uploaded dispatcher %s (%d bytes)", target.name, size)
        return {
            "status": "ok",
            "filename": target.name,
            "size_bytes": size,
            "md5": md5_of_file(target),
        }

    @router.delete("/api/dispatchers/{filename}")
    async def delete_dispatcher(filename: str) -> dict[str, str]:
        target = _resolve_under(DISPATCHERS_DIR, filename)
        if not target.exists():
            raise HTTPException(status_code=404, detail="Dispatcher not found")
        target.unlink()
        logger.info("Deleted dispatcher %s", target.name)
        return {"status": "ok"}

    # =========================================================================
    # Datasets (used by raw inference mode)
    # =========================================================================
    @router.get("/api/datasets")
    async def list_datasets() -> list[dict[str, Any]]:
        return _list_dir(DATASETS_DIR)

    @router.get("/api/datasets/{filename}")
    async def get_dataset(filename: str):
        target = _resolve_under(DATASETS_DIR, filename)
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="Dataset not found")
        return FileResponse(target, filename=target.name,
                            media_type="application/octet-stream")

    @router.post("/api/upload_dataset")
    async def upload_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
        name = _safe_filename(file.filename or "")
        # Accept common dataset bundle types (zip/tar/tar.gz), individual
        # images (jpg/png), and short video clips for the SR live demo.
        _enforce_extension(name, DATASET_EXTS)
        target = _resolve_under(DATASETS_DIR, name)
        size = await _save_upload(file, target)
        logger.info("Uploaded dataset %s (%d bytes)", target.name, size)
        return {
            "status": "ok",
            "filename": target.name,
            "size_bytes": size,
        }

    @router.delete("/api/datasets/{filename}")
    async def delete_dataset(filename: str) -> dict[str, str]:
        target = _resolve_under(DATASETS_DIR, filename)
        if not target.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")
        target.unlink()
        logger.info("Deleted dataset %s", target.name)
        return {"status": "ok"}

    # =========================================================================
    # Manual model distribution gate
    # =========================================================================
    @router.post("/api/distribute_model")
    async def distribute_model(req: DistributeModelRequest) -> dict[str, Any]:
        """Trigger a synchronous push of (model + adapter) to the targeted
        workers. Returns per-worker status so the dashboard can render badges.
        """
        if deps.experiment is None:
            raise HTTPException(503, "ExperimentManager not available")
        try:
            results = await deps.experiment.distribute_model(
                model_name=req.model_name,
                adapter_filename=req.adapter_filename,
                engine_overrides=req.engine_overrides,
                target_workers=req.target_workers,
            )
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        all_ok = all(r.get("status") == "ok" for r in results.values())
        return {
            "model_name": req.model_name,
            "adapter_filename": req.adapter_filename,
            "all_ok": all_ok,
            "results": {str(k): v for k, v in results.items()},
        }

    @router.get("/api/distribution_status")
    async def distribution_status(model_name: str | None = None) -> dict[str, Any]:
        """Report distribution status. With ``model_name`` only that model's
        per-worker entries are returned; without, every tracked model is."""
        if deps.state is None:
            return {}
        if model_name is not None:
            entries = deps.state.model_distribution.get(model_name, {})
            return {
                "model_name": model_name,
                "workers": {
                    str(wid): {
                        "status": s.status, "backend": s.backend,
                        "adapter_filename": s.adapter_filename,
                        "md5": s.md5, "error": s.error,
                        "distributed_ms": s.distributed_ms,
                    }
                    for wid, s in entries.items()
                },
            }
        return {
            mname: {
                str(wid): {
                    "status": s.status, "backend": s.backend,
                    "adapter_filename": s.adapter_filename,
                    "md5": s.md5, "error": s.error,
                    "distributed_ms": s.distributed_ms,
                }
                for wid, s in entries.items()
            }
            for mname, entries in deps.state.model_distribution.items()
        }

    # =========================================================================
    # Offline wheel cache — workers pull deps from here when LAN is air-gapped
    # =========================================================================
    # Source distributions (.tar.gz / .zip) are also served — pip falls back
    # to building these on the worker when no aarch64 wheel exists for a
    # transitive dep (e.g. netifaces). Worker installs them via uv pip
    # alongside the .whl files.
    _PIP_ARTEFACT_SUFFIXES = (".whl", ".tar.gz", ".zip")

    def _is_pip_artefact(p: Path) -> bool:
        if not p.is_file():
            return False
        name = p.name.lower()
        return any(name.endswith(s) for s in _PIP_ARTEFACT_SUFFIXES)

    @router.get("/api/wheels")
    async def list_wheels() -> list[str]:
        """Return the list of *common* wheel/sdist filenames the worker
        should install. Worker bulk-fetches then installs via
        `uv pip install --no-index --find-links`."""
        if not WHEELS_DIR.exists():
            return []
        return sorted(p.name for p in WHEELS_DIR.iterdir()
                      if _is_pip_artefact(p))

    @router.get("/api/wheels/{filename}")
    async def get_wheel(filename: str):
        target = _resolve_under(WHEELS_DIR, filename)
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="Wheel not found")
        return FileResponse(target, filename=target.name,
                            media_type="application/octet-stream")

    @router.get("/api/wheels-hailo")
    async def list_wheels_hailo() -> list[str]:
        """Hailo-only wheels (e.g. hailort) — only worker hosts that detected
        a Hailo NPU via lspci pull from this list."""
        if not WHEELS_HAILO_DIR.exists():
            return []
        return sorted(p.name for p in WHEELS_HAILO_DIR.iterdir()
                      if _is_pip_artefact(p))

    @router.get("/api/wheels-hailo/{filename}")
    async def get_wheel_hailo(filename: str):
        target = _resolve_under(WHEELS_HAILO_DIR, filename)
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="Wheel not found")
        return FileResponse(target, filename=target.name,
                            media_type="application/octet-stream")

    @router.get("/api/recordings")
    async def list_recordings() -> list[dict[str, Any]]:
        """List finished SR-pipeline recordings ready for download."""
        if not RECORDINGS_DIR.exists():
            return []
        return _list_dir(RECORDINGS_DIR)

    @router.get("/api/recordings/{filename}")
    async def get_recording(filename: str):
        target = _resolve_under(RECORDINGS_DIR, filename)
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="Recording not found")
        return FileResponse(target, filename=target.name,
                            media_type="video/mp4")

    @router.get("/api/bin/{filename}")
    async def get_bin(filename: str):
        """Serve static binaries (uv) that workers fetch when missing.

        Operator pre-stages these in res/bin/ — see scripts/prepare-wheels.sh
        for the uv download recipe.
        """
        target = _resolve_under(BIN_DIR, filename)
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="Binary not found")
        return FileResponse(target, filename=target.name,
                            media_type="application/octet-stream")

    # =========================================================================
    # Worker -> controller telemetry
    # =========================================================================
    @router.post("/api/inference_results")
    async def post_inference_result(payload: dict[str, Any]) -> dict[str, str]:
        # Workers can POST per-request telemetry summaries here. The
        # ExperimentManager subscribes via deps.experiment.record_result(...)
        try:
            if deps.experiment is not None:
                deps.experiment.record_result(payload)
        except Exception as e:
            logger.error("Failed to record inference result: %s", e)
            return {"status": "error", "detail": str(e)}
        return {"status": "ok"}

    app.include_router(router)
    return app


class DataDeps:
    def __init__(self, config: dict[str, Any], identifier: str, serial: str,
                 experiment_manager=None, state=None):
        self.config = config
        self.identifier = identifier
        self.serial = serial
        self.experiment = experiment_manager
        self.state = state
