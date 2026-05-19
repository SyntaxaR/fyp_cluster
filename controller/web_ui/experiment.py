"""Experiment configuration + launch page.

Lets the user:
* Upload / list / delete model files (.onnx, .hef), adapters (.py), custom
  dispatchers (.py), and datasets (zip / image bundles).
* Pick the model, adapter (or use the built-in default), and dispatcher.
* Toggle which workers participate (checkbox) and override their engine
  ("auto" / "onnx" / "hailo").
* Pick mode + duration + target QPS + dummy batch size and launch.

Uploads are written directly to ``models/``, ``adapters/``, ``dispatchers/``,
and ``datasets/`` under the controller's working directory — the same
directories the data-plane API serves from, so no HTTP round-trip needed.
"""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

from controller.dispatcher import list_dispatcher_choices
from controller.web_ui._helpers import page_header, worker_label
from shared.model_ops import get_or_compute_gops, save_sidecar, load_sidecar
from shared.models import ExperimentConfig, InferenceMode
from shared.util import adapter_supported_modes


def _badge(status: str) -> tuple[str, str]:
    """Return (label, css color class) for a distribution status string."""
    return {
        "ok":          ("OK",       "text-green-600"),
        "pending":     ("pending",  "text-yellow-600"),
        "failed":      ("failed",   "text-red-600"),
        "stale":       ("stale",    "text-orange-600"),
        "not_pushed":  ("—",        "text-gray-400"),
    }.get(status, (status, "text-gray-500"))

logger = logging.getLogger(__name__)


from controller._paths import (
    MODELS_DIR, ADAPTERS_DIR, DISPATCHERS_DIR, DATASETS_DIR,
)

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*$")


def _safe_name(name: str) -> str:
    base = Path(name).name
    if not base or base in {".", ".."} or not _SAFE_NAME_RE.match(base):
        raise ValueError(f"Illegal filename: {name!r}")
    return base


async def _save_upload(directory: Path, e, allowed_exts: tuple[str, ...]) -> Path:
    """Persist a NiceGUI upload event to `directory` after light validation.

    NiceGUI 3.x exposes the uploaded blob as ``e.file`` (a ``FileUpload``
    instance) with ``.name`` / ``.content_type`` and an async ``save()``.
    For back-compat with NiceGUI 2.x — which used ``e.name`` + ``e.content``
    — we fall back to the old shape if ``e.file`` is missing.
    """
    if hasattr(e, "file") and e.file is not None:
        name = _safe_name(e.file.name)
        if Path(name).suffix.lower() not in allowed_exts:
            raise ValueError(f"Bad extension {Path(name).suffix!r} (need {allowed_exts})")
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / name
        # Save via the FileUpload's own save() — handles SmallFileUpload (in-memory
        # bytes) and LargeFileUpload (on-disk spool) transparently.
        await e.file.save(target)
        return target

    # NiceGUI 2.x fallback path
    name = _safe_name(getattr(e, "name", ""))
    if Path(name).suffix.lower() not in allowed_exts:
        raise ValueError(f"Bad extension {Path(name).suffix!r} (need {allowed_exts})")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    content = getattr(e, "content", None)
    if content is None:
        raise ValueError("Upload event has neither .file nor .content")
    content.seek(0)
    with target.open("wb") as f:
        while True:
            chunk = content.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return target


def _upload_display_name(e) -> str:
    """Return the original filename regardless of NiceGUI version."""
    if hasattr(e, "file") and e.file is not None:
        return e.file.name
    return getattr(e, "name", "<unknown>")


def _list_files(directory: Path, suffixes: tuple[str, ...] | None = None) -> list[str]:
    if not directory.exists():
        return []
    out = []
    for p in sorted(directory.iterdir()):
        if not p.is_file():
            continue
        if suffixes is not None and p.suffix.lower() not in suffixes:
            continue
        out.append(p.name)
    return out


def _model_choices() -> list[str]:
    """List unique model stems available (so the same model with both .onnx
    and .hef variants only shows once)."""
    if not MODELS_DIR.exists():
        return []
    stems = set()
    for p in sorted(MODELS_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in (".onnx", ".hef"):
            stems.add(p.stem)
    return sorted(stems)


def register(controller) -> None:
    from nicegui import ui

    @ui.page("/experiment")
    def experiment_page():
        page_header(
            "Configure & launch experiment",
            subtitle="Upload model + adapter, distribute to workers, then run.",
            active="experiment",
        )

        # =====================================================================
        # File upload section
        # =====================================================================
        with ui.row().classes("items-center mt-4"):
            ui.label("Files").classes("text-lg font-semibold")
            ui.button(
                "Refresh",
                on_click=lambda: (refresh_lists(),
                                  ui.notify("Lists refreshed.", type="info")),
            ).props("flat dense color=primary").classes("ml-2")
        with ui.grid(columns=4).classes("gap-4 w-full"):
            with ui.column().classes("border p-2 rounded"):
                ui.label("Models (.onnx / .hef)").classes("font-semibold")

                async def _on_model_upload(e):
                    try:
                        await _save_upload(MODELS_DIR, e, (".onnx", ".hef"))
                        ui.notify(f"Uploaded model {_upload_display_name(e)}",
                                  type="positive")
                        refresh_lists()
                    except Exception as exc:
                        logger.exception("model upload failed")
                        ui.notify(f"Upload failed: {exc}", type="negative")

                ui.upload(on_upload=_on_model_upload,
                          multiple=True, auto_upload=True,
                          label="Upload model").props("accept=.onnx,.hef")
                models_list = ui.column()

            with ui.column().classes("border p-2 rounded"):
                ui.label("Adapters (.py)").classes("font-semibold")

                async def _on_adapter_upload(e):
                    try:
                        await _save_upload(ADAPTERS_DIR, e, (".py",))
                        ui.notify(f"Uploaded adapter {_upload_display_name(e)}",
                                  type="positive")
                        refresh_lists()
                    except Exception as exc:
                        logger.exception("adapter upload failed")
                        ui.notify(f"Upload failed: {exc}", type="negative")

                ui.upload(on_upload=_on_adapter_upload,
                          multiple=True, auto_upload=True,
                          label="Upload adapter").props("accept=.py")
                adapters_list = ui.column()

            with ui.column().classes("border p-2 rounded"):
                ui.label("Dispatchers (.py)").classes("font-semibold")

                async def _on_dispatcher_upload(e):
                    try:
                        await _save_upload(DISPATCHERS_DIR, e, (".py",))
                        ui.notify(f"Uploaded dispatcher {_upload_display_name(e)}",
                                  type="positive")
                        refresh_lists()
                    except Exception as exc:
                        logger.exception("dispatcher upload failed")
                        ui.notify(f"Upload failed: {exc}", type="negative")

                ui.upload(on_upload=_on_dispatcher_upload,
                          multiple=True, auto_upload=True,
                          label="Upload dispatcher").props("accept=.py")
                dispatchers_list = ui.column()

            with ui.column().classes("border p-2 rounded"):
                ui.label("Datasets").classes("font-semibold")

                # Single source of truth for what counts as a dataset —
                # imported from the REST endpoint module so the upload
                # control's `accept` attribute and the server-side guard
                # stay in lockstep.
                from controller.api.data_api import DATASET_EXTS
                _dataset_accept = ",".join(DATASET_EXTS)

                async def _on_dataset_upload(e):
                    try:
                        await _save_upload(DATASETS_DIR, e, DATASET_EXTS)
                        ui.notify(f"Uploaded dataset {_upload_display_name(e)}",
                                  type="positive")
                        refresh_lists()
                    except Exception as exc:
                        logger.exception("dataset upload failed")
                        ui.notify(f"Upload failed: {exc}", type="negative")

                ui.upload(on_upload=_on_dataset_upload,
                          multiple=True, auto_upload=True,
                          label="Upload dataset").props(
                    f"accept={_dataset_accept}")
                datasets_list = ui.column()

        # =====================================================================
        # Worker enrollment table (checkbox + per-row engine override)
        # =====================================================================
        ui.label("Workers").classes("text-lg font-semibold mt-4")
        ui.label("Tick the workers to include. Use 'auto' to let the worker pick "
                 "its native engine.").classes("text-sm text-gray-500")

        worker_rows_container = ui.column().classes("w-full")
        # worker_id -> {"enabled": bool, "engine": "auto"|"onnx"|"hailo"}
        worker_choices: dict[int, dict] = {}

        def _build_worker_rows():
            worker_rows_container.clear()
            with worker_rows_container:
                # Header
                with ui.row().classes("font-semibold border-b py-1"):
                    ui.label("Use").classes("w-12")
                    ui.label("Worker").classes("w-64")
                    ui.label("Status").classes("w-24")
                    ui.label("Self-reported").classes("w-32")
                    ui.label("Engine").classes("w-32")

                for wid, reg in controller.state.registered_workers.items():
                    row_state = worker_choices.setdefault(
                        wid, {"enabled": True, "engine": "auto"}
                    )
                    status = (reg.status if isinstance(reg.status, str)
                              else reg.status.value)
                    self_engine = reg.engine or "—"
                    with ui.row().classes("items-center py-1"):
                        cb = ui.checkbox(value=row_state["enabled"]) \
                            .classes("w-12")
                        cb.bind_value(row_state, "enabled")
                        ui.label(worker_label(wid, controller)).classes("w-64")
                        ui.label(status).classes("w-24")
                        ui.label(self_engine).classes("w-32")
                        eng_sel = ui.select(
                            ["auto", "onnx", "hailo"],
                            value=row_state["engine"],
                        ).classes("w-32")
                        eng_sel.bind_value(row_state, "engine")

        # =====================================================================
        # Model + Adapter selection
        # =====================================================================
        # Pulled out of "Parameters" so the Distribution section below can
        # depend on them. The flow makes more sense bottom-to-top now:
        #   1. Pick a model + adapter   (this block)
        #   2. Distribute that pair to enrolled workers (next block)
        #   3. Tweak run-time knobs     (Parameters, further down)
        #   4. Launch
        ui.label("Model & adapter").classes("text-lg font-semibold mt-4")
        ui.label(
            "Pick which model file and which preprocess adapter the "
            "experiment will use. These two together are what the "
            "Distribute step pushes to the enrolled workers."
        ).classes("text-sm text-gray-500")
        with ui.grid(columns=2).classes("gap-2 w-full max-w-3xl mt-2"):
            # Initial model dropdown — if no models on disk yet show a single
            # disabled placeholder so the control isn't visually empty/broken.
            _initial_models = _model_choices()
            model_select = ui.select(
                _initial_models or ["(no model uploaded yet)"],
                value=(_initial_models[0] if _initial_models
                       else "(no model uploaded yet)"),
                label="Model",
            )
            adapter_select = ui.select(
                ["(built-in / by-name default)"]
                + _list_files(ADAPTERS_DIR, (".py",)),
                value="(built-in / by-name default)",
                label="Adapter",
            )

        # =====================================================================
        # Distribution — push the chosen model + adapter to enrolled workers.
        # Sits ABOVE Parameters because the per-worker distribution status
        # gates Launch; the run-time knobs (mode, dataset, duration, …) only
        # matter once the file is actually on every worker.
        # =====================================================================
        ui.label("Distribution").classes("text-lg font-semibold mt-4")
        ui.label(
            "Click Distribute to push the selected model + adapter to every "
            "enrolled worker. Launch is disabled until every enrolled worker's "
            "status is OK."
        ).classes("text-sm text-gray-500")

        dist_table_container = ui.column().classes("w-full mt-2")
        dist_msg = ui.label("").classes("text-sm mt-1")

        async def _do_distribute():
            # Anything that goes wrong inside this handler must surface to the
            # user — NiceGUI silently swallows exceptions raised by event
            # handlers and the only symptom is "the button does nothing".
            try:
                if (not model_select.value
                        or model_select.value == "(no model uploaded yet)"):
                    ui.notify("Upload a model and click Refresh first.",
                              type="warning")
                    return
                adapter_filename = (
                    adapter_select.value
                    if adapter_select.value != "(built-in / by-name default)"
                    else None
                )
                engine_overrides: dict[int, str] = {
                    wid: st["engine"]
                    for wid, st in worker_choices.items()
                    if st["enabled"] and st["engine"] in ("onnx", "hailo")
                }
                target_workers = [wid for wid, st in worker_choices.items()
                                  if st["enabled"]]
                if not target_workers:
                    ui.notify("Tick at least one worker (Workers section above).",
                              type="warning")
                    return

                ui.notify(
                    f"Distributing '{model_select.value}' to {len(target_workers)} "
                    f"worker(s) …",
                    type="info",
                )
                dist_msg.text = (f"Distributing '{model_select.value}' to "
                                 f"{len(target_workers)} workers …")

                results = await controller.experiment.distribute_model(
                    model_name=model_select.value,
                    adapter_filename=adapter_filename,
                    engine_overrides=engine_overrides,
                    target_workers=target_workers,
                )
                bad = [wid for wid, r in results.items()
                       if r.get("status") != "ok"]
                if bad:
                    dist_msg.text = (f"Distribution finished with errors on "
                                     f"workers {bad}. See badges below.")
                    ui.notify("Distribution had failures — see badges below.",
                              type="negative")
                else:
                    dist_msg.text = (f"All {len(results)} worker(s) report OK "
                                     f"for '{model_select.value}'.")
                    ui.notify(f"Distribution OK on {len(results)} worker(s).",
                              type="positive")
            except FileNotFoundError as e:
                logger.exception("distribute: file not found")
                dist_msg.text = f"Failed: {e}"
                ui.notify(f"Failed: {e}", type="negative")
            except Exception as e:
                logger.exception("distribute: unexpected error")
                dist_msg.text = f"Failed: {type(e).__name__}: {e}"
                ui.notify(f"Distribute failed: {type(e).__name__}: {e}",
                          type="negative")
            finally:
                try:
                    refresh_distribution()
                except Exception:
                    logger.exception("distribute: refresh_distribution crashed")

        with ui.row().classes("mt-2 gap-2"):
            distribute_btn = ui.button("Distribute now",
                                       on_click=_do_distribute) \
                .props("color=primary")

            async def _verify_workers():
                """Ping each enrolled worker's /api/health and show what's
                actually loaded (engine, model, adapter). Surfaces the
                drift that causes 'raw mode requires adapter' errors —
                the controller thinks the model is distributed, but
                the worker's engine has adapter=None."""
                import requests as _requests
                targets = [wid for wid, st in worker_choices.items()
                           if st["enabled"]]
                if not targets:
                    ui.notify("Tick at least one worker first.", type="warning")
                    return
                ports = controller.config["worker"]
                data_port = ports.get("data_port", 8002)
                rows = []
                # Show what the controller has on disk for the chosen
                # model — helps spot "adapter not in adapters/" issues.
                ctrl_adapter = None
                ctrl_adapter_present = False
                if model_select.value and model_select.value != "(no model uploaded yet)":
                    explicit = (adapter_select.value
                                if adapter_select.value != "(built-in / by-name default)"
                                else None)
                    if explicit:
                        ctrl_adapter = explicit
                    else:
                        ctrl_adapter = f"{model_select.value}_adapter.py"
                    ctrl_adapter_present = (ADAPTERS_DIR / ctrl_adapter).exists()

                async def _ping(wid):
                    reg = controller.state.get_registered(wid)
                    if reg is None or not reg.data_ip:
                        return {"worker_id": wid, "error": "no data_ip"}
                    url = f"http://{reg.data_ip}:{data_port}/api/health"
                    try:
                        r = await asyncio.to_thread(
                            _requests.get, url, timeout=4.0,
                        )
                        if r.status_code != 200:
                            return {"worker_id": wid,
                                    "error": f"HTTP {r.status_code}"}
                        return {"worker_id": wid, **r.json()}
                    except Exception as e:
                        return {"worker_id": wid,
                                "error": f"{type(e).__name__}: {e}"}

                results = await asyncio.gather(*[_ping(w) for w in targets])

                with ui.dialog() as dialog, ui.card().classes("min-w-[720px]"):
                    ui.label("Worker engine state").classes("text-lg font-semibold")
                    if ctrl_adapter is not None:
                        ui.label(
                            f"Controller expects adapter: '{ctrl_adapter}' "
                            + ("✓ present in adapters/"
                               if ctrl_adapter_present
                               else "✗ NOT FOUND in adapters/")
                        ).classes("text-sm "
                                  + ("text-green-600" if ctrl_adapter_present
                                     else "text-red-600 font-semibold"))
                    rows_render = []
                    for r in results:
                        if "error" in r:
                            rows_render.append({
                                "worker": worker_label(r["worker_id"], controller),
                                "engine": "—",
                                "model": "—",
                                "adapter": f"⚠ {r['error']}",
                            })
                            continue
                        adapter_str = (
                            r.get("adapter_class", "loaded")
                            if r.get("adapter_loaded")
                            else "✗ NOT LOADED"
                        )
                        rows_render.append({
                            "worker": worker_label(r["worker_id"], controller),
                            "engine": r.get("engine") or "—",
                            "model":  r.get("loaded_model") or "—",
                            "adapter": adapter_str,
                        })
                    ui.table(
                        columns=[
                            {"name": "worker",  "label": "Worker",  "field": "worker", "align": "left"},
                            {"name": "engine",  "label": "Engine",  "field": "engine", "align": "left"},
                            {"name": "model",   "label": "Model",   "field": "model",  "align": "left"},
                            {"name": "adapter", "label": "Adapter", "field": "adapter", "align": "left"},
                        ],
                        rows=rows_render,
                        row_key="worker",
                    ).classes("w-full text-sm mt-1")
                    ui.label(
                        "If 'Adapter' shows 'NOT LOADED' on any worker, the engine "
                        "needs to be rebuilt. Make sure the adapter file is in the "
                        "controller's adapters/ directory, then click Distribute "
                        "again — that always calls swap_engine, even if MD5s match."
                    ).classes("text-xs text-gray-500 mt-2")
                    with ui.row().classes("justify-end mt-2"):
                        ui.button("Close",
                                  on_click=lambda: dialog.submit(True)
                                  ).props("flat")
                await dialog

            ui.button("Verify workers",
                      icon="health_and_safety",
                      on_click=_verify_workers,
                      ).props("color=info outline") \
                .tooltip("Ping each worker's /api/health and show what's "
                         "actually loaded — useful when the controller's "
                         "view and the worker's reality have drifted.")

        def refresh_distribution():
            """Re-render the per-worker badge table for the chosen model."""
            dist_table_container.clear()
            with dist_table_container:
                if not model_select.value:
                    ui.label("(pick a model to see distribution status)") \
                        .classes("text-sm text-gray-500")
                    return
                with ui.row().classes("font-semibold border-b py-1 text-sm"):
                    ui.label("Worker").classes("w-56")
                    ui.label("Status").classes("w-32")
                    ui.label("Backend").classes("w-20")
                    ui.label("Adapter").classes("w-48")
                    ui.label("Last error").classes("flex-1")
                entries = controller.state.model_distribution.get(
                    model_select.value, {}
                )
                # Show every enrolled worker — missing entries display as "—"
                rows_for: list[int] = sorted(
                    wid for wid, st in worker_choices.items() if st["enabled"]
                )
                if not rows_for:
                    ui.label("(no workers enrolled — tick some above)") \
                        .classes("text-sm text-gray-500")
                    return
                for wid in rows_for:
                    s = entries.get(wid)
                    status = s.status if s is not None else "not_pushed"
                    label, color = _badge(status)
                    with ui.row().classes("items-center py-1 text-sm"):
                        ui.label(worker_label(wid, controller)).classes("w-56")
                        ui.label(label).classes(f"w-32 {color}")
                        ui.label((s.backend if s else "") or "—") \
                            .classes("w-20")
                        ui.label((s.adapter_filename if s else "") or "—") \
                            .classes("w-48")
                        ui.label((s.error if s else "") or "") \
                            .classes("flex-1 text-red-500")

        # =====================================================================
        # Run-time parameters — name, dispatcher, mode, dataset, duration etc.
        # Model + adapter were intentionally pulled out (see block above).
        # =====================================================================
        ui.label("Parameters").classes("text-lg font-semibold mt-4")
        with ui.grid(columns=2).classes("gap-2 w-full max-w-3xl"):
            name = ui.input("Experiment name", value="exp-1")
            dispatcher_select = ui.select(
                list_dispatcher_choices(),
                value=controller.config["dispatcher"].get(
                    "algorithm", "round_robin"),
                label="Dispatcher",
            )
            mode_select = ui.select(
                [m.value for m in InferenceMode],
                value=InferenceMode.DUMMY.value,
                label="Inference mode",
            )
            mode_hint = ui.label("").classes("col-span-2 text-xs text-gray-500")
            dataset_select = ui.select(
                ["(none)"] + _list_files(DATASETS_DIR),
                value="(none)",
                label="Dataset (raw mode)",
            )
            duration = ui.number("Duration (s)", value=30.0, min=1.0, step=1.0)
            target_qps = ui.number("Target QPS (0 = unlimited)",
                                   value=0.0, min=0.0)
            dummy_batch = ui.number("Dummy batch size", value=1, min=1)
            run_postprocess = ui.checkbox(
                "Run adapter postprocess on the worker (raw mode)",
                value=False,
            )
        notes = ui.textarea("Notes", value="").classes("w-full max-w-3xl")

        status_label = ui.label("").classes("mt-2 text-sm")

        # =====================================================================
        # Refresh callbacks
        # =====================================================================
        def refresh_lists():
            # Models
            models_list.clear()
            with models_list:
                files = _list_files(MODELS_DIR, (".onnx", ".hef"))
                if not files:
                    ui.label("(no models uploaded)").classes("text-sm text-gray-500")
                for fn in files:
                    with ui.row().classes("items-center"):
                        ui.label(fn).classes("text-sm")
                        # Per-model GOPS — parsed from ONNX or looked up
                        # by HEF filename, cached as a .meta.json sidecar.
                        # Used by the report page to compute observed TOPS
                        # against measured FPS.
                        try:
                            gops_info = get_or_compute_gops(MODELS_DIR / fn)
                        except Exception:
                            gops_info = None
                        if gops_info is not None:
                            tag = {
                                "auto":         "from ONNX",
                                "sibling-onnx": "from sibling .onnx",
                                "manual":       "manual",
                            }.get(gops_info["source"], gops_info["source"])
                            ui.label(
                                f"{gops_info['gops']:.2f} GOPS ({tag})"
                            ).classes("text-xs text-gray-500 ml-2")
                        else:
                            ui.label("GOPS ?").classes(
                                "text-xs text-gray-400 ml-2"
                            )
                        # Manual GOPS override button. Opens a dialog
                        # that lets the operator type the number from
                        # the Model Zoo / paper / hand-computation
                        # without SSHing into the controller to drop
                        # a sidecar by hand.
                        ui.button(icon="edit",
                                  on_click=lambda _, f=fn: _open_gops_dialog(f)) \
                            .props("flat round dense color=primary size=sm") \
                            .tooltip("Edit GOPS override")
                        ui.button(icon="delete",
                                  on_click=lambda _, f=fn: _delete_file(MODELS_DIR / f)) \
                            .props("flat round dense color=negative size=sm")
            # Adapters
            adapters_list.clear()
            with adapters_list:
                files = _list_files(ADAPTERS_DIR, (".py",))
                if not files:
                    ui.label("(no adapters)").classes("text-sm text-gray-500")
                for fn in files:
                    with ui.row().classes("items-center"):
                        ui.label(fn).classes("text-sm")
                        ui.button(icon="delete",
                                  on_click=lambda _, f=fn: _delete_file(ADAPTERS_DIR / f)) \
                            .props("flat round dense color=negative size=sm")
            # Dispatchers
            dispatchers_list.clear()
            with dispatchers_list:
                files = _list_files(DISPATCHERS_DIR, (".py",))
                if not files:
                    ui.label("(no custom dispatchers)").classes("text-sm text-gray-500")
                for fn in files:
                    with ui.row().classes("items-center"):
                        ui.label(fn).classes("text-sm")
                        ui.button(icon="delete",
                                  on_click=lambda _, f=fn: _delete_file(DISPATCHERS_DIR / f)) \
                            .props("flat round dense color=negative size=sm")
            # Datasets
            datasets_list.clear()
            with datasets_list:
                files = _list_files(DATASETS_DIR)
                if not files:
                    ui.label("(no datasets)").classes("text-sm text-gray-500")
                for fn in files:
                    with ui.row().classes("items-center"):
                        ui.label(fn).classes("text-sm")
                        ui.button(icon="delete",
                                  on_click=lambda _, f=fn: _delete_file(DATASETS_DIR / f)) \
                            .props("flat round dense color=negative size=sm")

            # Selectors
            choices = _model_choices()
            if choices:
                model_select.options = choices
                if model_select.value not in choices:
                    model_select.value = choices[0]
            else:
                # No real models — keep a clear placeholder so the dropdown
                # isn't a confusing empty box.
                model_select.options = ["(no model uploaded yet)"]
                model_select.value = "(no model uploaded yet)"
            model_select.update()

            adapter_select.options = (
                ["(built-in / by-name default)"]
                + _list_files(ADAPTERS_DIR, (".py",))
            )
            if adapter_select.value not in adapter_select.options:
                adapter_select.value = "(built-in / by-name default)"
            adapter_select.update()

            dataset_select.options = ["(none)"] + _list_files(DATASETS_DIR)
            if dataset_select.value not in dataset_select.options:
                dataset_select.value = "(none)"
            dataset_select.update()

            dispatcher_select.options = list_dispatcher_choices()
            if dispatcher_select.value not in dispatcher_select.options:
                dispatcher_select.value = (dispatcher_select.options[0]
                                           if dispatcher_select.options
                                           else "round_robin")
            dispatcher_select.update()

        def _delete_file(p: Path):
            try:
                p.unlink()
                # Also remove the sidecar so a re-upload doesn't inherit
                # a stale GOPS override.
                side = p.with_suffix(p.suffix + ".meta.json")
                if side.exists():
                    side.unlink()
                ui.notify(f"Deleted {p.name}", type="warning")
                refresh_lists()
            except Exception as ex:
                ui.notify(f"Delete failed: {ex}", type="negative")

        async def _open_gops_dialog(model_filename: str):
            """Pop a small dialog letting the user set / clear a manual
            GOPS override for ``model_filename``. Persists as a sidecar
            ``<file>.meta.json`` with ``"source": "manual"`` so it always
            wins over auto-detection."""
            path = MODELS_DIR / model_filename
            side = load_sidecar(path)
            cur_gops = float(side.get("gops") or 0.0)
            cur_source = side.get("source") or "(none)"

            with ui.dialog() as dialog, ui.card().classes("min-w-[420px]"):
                ui.label(f"GOPS override — {model_filename}") \
                    .classes("text-lg font-semibold")
                ui.label(
                    f"Current cached value: {cur_gops:.2f} GOPS "
                    f"(source: {cur_source})"
                ).classes("text-sm text-gray-600")
                ui.label(
                    "Set the model's OPS per inference. Hailo's "
                    "convention: 1 MAC = 2 OPs. The number should match "
                    "what's reported in the Hailo Model Zoo YAML or "
                    "what you compute via thop/onnx-tool."
                ).classes("text-xs text-gray-500 mt-1")

                gops_input = ui.number(
                    label="GOPS per inference",
                    value=cur_gops if cur_gops > 0 else None,
                    min=0.0, step=0.01, format="%.3f",
                ).classes("w-full mt-2")

                async def _save():
                    try:
                        v = float(gops_input.value or 0.0)
                    except (TypeError, ValueError):
                        ui.notify("Enter a positive number.", type="warning")
                        return
                    if v <= 0:
                        ui.notify("GOPS must be > 0.", type="warning")
                        return
                    save_sidecar(path, {"gops": v, "source": "manual"})
                    ui.notify(f"Saved {v:.3f} GOPS for {model_filename}",
                              type="positive")
                    dialog.submit(True)
                    refresh_lists()

                async def _clear():
                    # Drop the sidecar so auto / sibling-onnx detection
                    # takes over on the next refresh.
                    side_path = path.with_suffix(path.suffix + ".meta.json")
                    if side_path.exists():
                        try:
                            side_path.unlink()
                            ui.notify("Cleared override — will auto-detect next refresh.",
                                      type="info")
                        except Exception as e:
                            ui.notify(f"Could not clear: {e}",
                                      type="negative")
                    else:
                        ui.notify("No override to clear.", type="info")
                    dialog.submit(True)
                    refresh_lists()

                with ui.row().classes("justify-end gap-2 mt-3"):
                    ui.button("Clear override",
                              on_click=_clear).props("flat color=warning")
                    ui.button("Cancel",
                              on_click=lambda: dialog.submit(False)) \
                        .props("flat")
                    ui.button("Save", on_click=_save) \
                        .props("color=primary")
            await dialog

        # =====================================================================
        # Launch / Stop
        # =====================================================================
        def _launch():
            if (not model_select.value
                    or model_select.value == "(no model uploaded yet)"):
                ui.notify("Upload a model and click Refresh first.",
                          type="warning")
                return
            enrolled = [wid for wid, st in worker_choices.items() if st["enabled"]]
            engine_overrides: dict[int, str] = {}
            for wid, st in worker_choices.items():
                if st["enabled"] and st["engine"] in ("onnx", "hailo"):
                    engine_overrides[wid] = st["engine"]

            adapter_filename = (
                adapter_select.value
                if adapter_select.value != "(built-in / by-name default)"
                else None
            )
            dataset_filename = (
                dataset_select.value
                if dataset_select.value != "(none)"
                else None
            )

            cfg = ExperimentConfig(
                name=name.value,
                model_name=model_select.value,
                dispatcher=dispatcher_select.value,
                mode=InferenceMode(mode_select.value),
                duration_s=float(duration.value),
                target_qps=(float(target_qps.value) if target_qps.value else None),
                dummy_batch_size=int(dummy_batch.value),
                notes=notes.value,
                enrolled_workers=enrolled,
                engine_overrides=engine_overrides,
                dataset_filename=dataset_filename,
                adapter_filename=adapter_filename,
                run_postprocess=bool(run_postprocess.value),
            )
            try:
                asyncio.create_task(controller.experiment.start(cfg))
                summary = (
                    f"Experiment '{cfg.name}' submitted: "
                    f"{cfg.mode.value} mode, dispatcher={cfg.dispatcher}, "
                    f"{len(enrolled)} worker(s), duration={cfg.duration_s:.0f}s"
                    f"{' @ '+str(cfg.target_qps)+' QPS' if cfg.target_qps else ''}."
                )
                status_label.text = summary
                # Loud + persistent toast so the operator can confirm at a
                # glance that the launch was actually accepted (a click that
                # validates and dispatches without a notification feels dead).
                ui.notify(
                    summary + " Watch 'Current status' below for progress.",
                    type="positive", timeout=8000,
                )
            except Exception as e:
                status_label.text = f"Failed to launch: {e}"
                ui.notify(f"Launch failed: {e}", type="negative", timeout=10000)

        def _stop():
            try:
                asyncio.create_task(controller.experiment.stop())
                status_label.text = "Stop requested."
                ui.notify(
                    "Stop requested — pipeline will drain in-flight requests "
                    "before transitioning to STOPPING → IDLE.",
                    type="warning", timeout=6000,
                )
            except Exception as e:
                status_label.text = f"Stop failed: {e}"
                ui.notify(f"Stop failed: {e}", type="negative", timeout=8000)

        with ui.row().classes("mt-4"):
            launch_btn = ui.button("Launch", on_click=_launch).props("color=primary")
            ui.button("Stop", on_click=_stop).props("color=warning")
        launch_hint = ui.label("").classes("text-sm text-gray-500")

        def _enrolled_workers() -> list[int]:
            return [wid for wid, st in worker_choices.items() if st["enabled"]]

        def refresh_mode_options():
            """Filter `mode_select.options` to only the modes the chosen
            adapter declares it supports (via class attr SUPPORTED_MODES).

            We dynamically rewrite the dropdown's options rather than
            using Quasar's per-option `disable` flag — NiceGUI's ui.select
            doesn't expose per-option disabling cleanly, but rewriting
            options is robust across versions and gives the user a clear
            "this mode just isn't there" instead of a greyed item that
            looks broken.

            "(built-in / by-name default)" → no specific adapter file
            picked, so we don't gate anything.
            """
            ALL = [m.value for m in InferenceMode]
            chosen = adapter_select.value or "(built-in / by-name default)"
            if chosen == "(built-in / by-name default)":
                supported = set(ALL)
                hint = ""
            else:
                cand = ADAPTERS_DIR / chosen
                if not cand.exists():
                    supported = set(ALL)
                    hint = ""
                else:
                    try:
                        supported = adapter_supported_modes(cand)
                    except Exception as e:
                        logger.warning("could not introspect %s: %s", cand, e)
                        supported = set(ALL)
                    if supported == set(ALL):
                        hint = ""
                    else:
                        hint = (
                            f"Adapter '{chosen}' declares "
                            f"SUPPORTED_MODES = {sorted(supported)} — "
                            f"other modes are hidden."
                        )

            options_now = [m for m in ALL if m in supported]
            if not options_now:
                # Defensive — shouldn't happen because adapter_supported_modes
                # falls back to all modes when in doubt.
                options_now = ALL
            if list(mode_select.options) != options_now:
                mode_select.options = options_now
                if mode_select.value not in options_now:
                    mode_select.value = options_now[0]
                mode_select.update()
            mode_hint.text = hint

        # Bind so the dropdown filter refreshes the moment the user
        # picks a different adapter (no need to wait for the 2 s timer).
        adapter_select.on_value_change(lambda _e: refresh_mode_options())

        def refresh_launch_gate():
            """Disable Launch unless every enrolled worker has status='ok'
            for the chosen model. Mirrors ExperimentManager.start()'s gate."""
            enrolled = _enrolled_workers()
            no_real_model = (not model_select.value
                             or model_select.value == "(no model uploaded yet)")
            if no_real_model or not enrolled:
                launch_btn.props("disable")
                launch_hint.text = (
                    "Upload a model first (Files → Models)."
                    if no_real_model else "Tick at least one worker."
                )
                return
            entries = controller.state.model_distribution.get(
                model_select.value, {}
            )
            missing = [wid for wid in enrolled
                       if (entries.get(wid) is None
                           or entries.get(wid).status != "ok")]
            if missing:
                launch_btn.props("disable")
                launch_hint.text = (f"Distribute first — workers {missing} "
                                    f"are not OK for '{model_select.value}'.")
            else:
                launch_btn.props(remove="disable")
                launch_hint.text = (f"Ready to launch on workers {enrolled}.")

        ui.label("Current status").classes("mt-4 font-semibold")
        status = ui.label("idle")

        def refresh_status():
            s = controller.state.current_experiment_status
            status.text = s.value if hasattr(s, "value") else str(s)

        def _periodic():
            _build_worker_rows()
            refresh_distribution()
            refresh_launch_gate()
            # Cheap (cached) — handles the case where a brand-new
            # adapter appeared in the dropdown without firing a click.
            refresh_mode_options()

        def _periodic_files():
            # Slower cadence — directory walks are cheap but still wasteful
            # at 0.5 Hz. 5s is fast enough for "I just dropped a file there".
            refresh_lists()

        ui.timer(2.0, _periodic)
        ui.timer(5.0, _periodic_files)
        ui.timer(1.0, refresh_status)
        _build_worker_rows()
        refresh_lists()
        refresh_distribution()
        refresh_launch_gate()
        refresh_mode_options()
        refresh_status()
