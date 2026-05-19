"""
/live — interactive super-resolution demo.

Layout:
  - Top:    [video select]  [model select]  [per-worker checkboxes]  [Start/Stop]
  - Body:   left = original low-res frame ; right = upscaled SR frame (live)
  - Bottom: FPS counter + avg inference latency + last-error banner

Frames are streamed to the browser as MJPEG (``multipart/x-mixed-replace``)
served by the ``/stream/sr/*.mjpeg`` endpoints registered in
``controller.web_ui.__init__._attach_mjpeg_streams``. The ``<img>`` tags here
just hold those URLs and the browser does the rest — no per-tick set_source
push, no flicker. The 30 Hz timer only updates stats labels.
"""
from __future__ import annotations

import logging
from pathlib import Path

from controller.web_ui._helpers import worker_label

logger = logging.getLogger(__name__)


from controller._paths import DATASETS_DIR  # noqa: E402


def _list_videos() -> list[str]:
    if not DATASETS_DIR.exists():
        return []
    out: list[str] = []
    for p in sorted(DATASETS_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
            out.append(p.name)
    return out


def register(controller) -> None:
    from nicegui import ui

    @ui.page("/live")
    def live_page():
        from controller.web_ui._helpers import page_header
        page_header(
            "Live Super-Resolution Demo",
            subtitle=("Pick a video, distribute the SR model, then drag the "
                      "worker slider. 1 worker can't hit 30 fps; 4 can."),
            active="live",
        )

        # =====================================================================
        # Controls
        # =====================================================================
        with ui.row().classes("items-end gap-3 mt-3 w-full"):
            video_select = ui.select(
                _list_videos() or ["(upload a video to datasets/)"],
                value=(_list_videos()[0] if _list_videos()
                       else "(upload a video to datasets/)"),
                label="Video",
            ).classes("w-72")

            model_select = ui.select(
                _model_choices(),
                value=(_model_choices()[0] if _model_choices() else None),
                label="SR model",
            ).classes("w-48")

            ui.label("Active workers:").classes("ml-2")
            # Worker selection is rendered as a row of toggles — one per
            # registered worker. Examiner can flip individual workers in /
            # out of the active subset without restarting the pipeline.
            worker_toggles_row = ui.row().classes("gap-1")
            worker_toggle_state: dict[int, bool] = {}

            start_btn = ui.button(
                "Start", on_click=lambda: _start()
            ).props("color=primary")
            stop_btn = ui.button(
                "Stop", on_click=lambda: _stop()
            ).props("color=warning")

        # =====================================================================
        # Worker subset slider helper
        # =====================================================================
        def _build_worker_toggles():
            worker_toggles_row.clear()
            with worker_toggles_row:
                regs = sorted(controller.state.registered_workers.keys())
                if not regs:
                    ui.label("(no workers connected)").classes("text-gray-500")
                    return
                for wid in regs:
                    if wid not in worker_toggle_state:
                        worker_toggle_state[wid] = True

                    def _make_handler(_wid):
                        def _h(e):
                            worker_toggle_state[_wid] = bool(e.value)
                            _push_subset()
                        return _h

                    chk = ui.checkbox(
                        worker_label(wid, controller),
                        value=worker_toggle_state[wid],
                    ).classes("min-w-[10rem]")
                    chk.on_value_change(_make_handler(wid))

        def _push_subset():
            wids = [wid for wid, on in worker_toggle_state.items() if on]
            controller.sr_pipeline.set_active_workers(wids)

        # =====================================================================
        # Display panes
        #
        # Both <img> tags point at the controller's MJPEG endpoints — the
        # browser holds the connection open and renders successive JPEG
        # parts in place (no flicker, no set_source pump).
        #
        # Layout strategy: a single ``<div>`` with CSS Grid, two equal
        # ``1fr`` columns and a fixed gap. Inline styles on the wrapper
        # avoid relying on Tailwind arbitrary-value classes (``w-[45%]``)
        # which may or may not be JIT-compiled depending on the NiceGUI
        # build. CSS Grid is honoured by every browser shipped after 2017.
        #
        # Each column then nests an ``aspect-ratio: 16 / 9`` <div> with
        # ``object-fit: contain`` on the <img> inside. Result: the two
        # frame boxes are guaranteed to be exactly the same width AND
        # height — the only thing that can vary is the letterbox bands
        # inside each box if a video's native aspect doesn't match 16:9.
        #
        # ``image-rendering: pixelated`` on the source image disables the
        # browser's bilinear smoothing so the 426×240 frame gets honestly
        # painted as chunky pixel blocks when scaled up. The right pane
        # has 16× more pixels (1704×960), so even after scaling down to
        # the same display size it stays sharp — that's the demo's pitch.
        # =====================================================================
        PANE_BOX_STYLE = (
            "width:100%; aspect-ratio: 16 / 9; background:#1c1c1c; "
            "border:1px solid #ccc; overflow:hidden; display:block;"
        )
        IMG_BASE = (
            "width:100%; height:100%; display:block; object-fit: contain;"
        )
        IMG_PIXELATED = (
            IMG_BASE
            + " image-rendering: pixelated; image-rendering: crisp-edges;"
            + " -ms-interpolation-mode: nearest-neighbor;"
        )

        ui.separator().classes("mt-3")
        # Two-column grid. Each column has:
        #   - dynamic resolution heading  (ui.label, updated on tick)
        #   - the MJPEG <img>             (static URL, browser handles streaming)
        #   - subtitle hint               (static text)
        # We use ui.element("div") wrappers + ui.label so the heading is
        # mutable; an ui.html() block would force a full re-render whenever
        # the resolution string changed.
        with ui.element("div").style(
            "display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; "
            "width: 100%; margin-top: 0.5rem; align-items: start;"
        ):
            # ---- Source pane -------------------------------------------------
            with ui.element("div"):
                src_res_label = ui.label("Source · — × — pending …").style(
                    "font-weight: 600; margin-bottom: 0.25rem;"
                )
                ui.html(
                    f'<div style="{PANE_BOX_STYLE}">'
                    f'  <img src="/stream/sr/orig.mjpeg" '
                    f'       style="{IMG_PIXELATED}" alt="source stream" />'
                    f'</div>'
                )
                ui.label("← Browser zoom shows the actual low-res pixel grid"
                         ).style("font-size: 0.75rem; color: #6b7280; "
                                 "margin-top: 0.25rem;")

            # ---- Cluster output pane -----------------------------------------
            with ui.element("div"):
                out_res_label = ui.label(
                    "Cluster output · — × — pending …"
                ).style("font-weight: 600; margin-bottom: 0.25rem;")
                ui.html(
                    f'<div style="{PANE_BOX_STYLE}">'
                    f'  <img src="/stream/sr/upscaled.mjpeg" '
                    f'       style="{IMG_BASE}" alt="upscaled stream" />'
                    f'</div>'
                )
                ui.label(
                    "← Edges & text stay crisp at the same display size"
                ).style("font-size: 0.75rem; color: #6b7280; "
                        "margin-top: 0.25rem;")

        def _fmt_res(w, h, suffix: str = "") -> str:
            """`<W> × <H> (N MP)` or `<W> × <H> (N KP)` depending on size."""
            if not w or not h:
                return "— × — pending …"
            px = w * h
            if px >= 1_000_000:
                amount = f"{px / 1_000_000:.2f} MP"
            else:
                amount = f"{px // 1000} KP"
            return f"{w} × {h}  ({amount}{suffix})"

        # =====================================================================
        # Stats strip
        # =====================================================================
        with ui.row().classes("mt-3 items-center gap-6"):
            fps_label = ui.label("FPS: —").classes("text-xl font-mono")
            inf_label = ui.label("Avg inference: — ms").classes("font-mono text-gray-700")
            workers_label = ui.label("Active: —").classes("font-mono text-gray-700")
            counters_label = ui.label("Processed: 0   Failed: 0").classes("font-mono text-gray-500")
        err_label = ui.label("").classes("text-sm text-red-500 mt-1")

        # =====================================================================
        # Record-to-MP4 toggle + progress + download link
        # =====================================================================
        ui.separator().classes("mt-4")
        ui.label("Save SR'd video to file").classes("text-lg font-semibold mt-2")
        with ui.row().classes("items-center gap-3 mt-1"):
            record_check = ui.checkbox(
                "Record next run to mp4", value=False,
            )
            ui.label("Output:").classes("text-sm text-gray-600")
            record_name = ui.input("Filename",
                                   value="sr_output.mp4").classes("w-72")
        ui.label("File is written under controller's `recordings/` directory. "
                 "Pipeline auto-stops when the source video ends.") \
            .classes("text-xs text-gray-500")

        record_label = ui.label("").classes("text-sm font-mono mt-1")
        record_link = ui.html("")    # populated when a recording finishes

        # =====================================================================
        # Lifecycle handlers
        # =====================================================================
        async def _start():
            if (not model_select.value
                    or model_select.value == "(no model uploaded yet)"):
                ui.notify("Pick a model first.", type="warning")
                return
            v = video_select.value or ""
            if not v or v.startswith("("):
                ui.notify("Pick a video file.", type="warning")
                return
            try:
                controller.sr_pipeline.set_video(DATASETS_DIR / v)
                controller.sr_pipeline.set_model(model_select.value)
                _push_subset()

                # Record-mode wiring — toggled via the checkbox above. The
                # filename is sanitised to a single basename so user input
                # can't escape into ../etc/shadow style paths.
                record_link.set_content("")
                if record_check.value:
                    safe = (Path(record_name.value or "sr_output.mp4")
                            .name or "sr_output.mp4")
                    if not safe.lower().endswith(".mp4"):
                        safe += ".mp4"
                    target = Path("recordings") / safe
                    controller.sr_pipeline.set_record_path(target)
                    ui.notify(f"Recording to {target}", type="info")
                else:
                    controller.sr_pipeline.set_record_path(None)
                # Distribution gate — the SR adapter / model must have been
                # pushed to every active worker first. Tell the user what to
                # do rather than fail silently inside the dispatch loop.
                missing = []
                no_adapter = []
                for wid in controller.sr_pipeline._active_workers:
                    if not controller.state.is_distributed(
                        model_select.value, wid
                    ):
                        missing.append(wid)
                        continue
                    # is_distributed returns True even if the worker
                    # loaded the model WITHOUT an adapter. That's fine
                    # for tensor / dummy mode but fatal for the live
                    # demo, which always uses raw mode. Catch it here
                    # with a clearer message than "raw mode requires
                    # adapter" deep in the worker.
                    entry = controller.state.get_distribution(
                        model_select.value, wid,
                    )
                    if entry is not None and not entry.adapter_filename:
                        no_adapter.append(wid)
                if missing:
                    ui.notify(
                        f"Distribute '{model_select.value}' on the "
                        f"Experiment page first. Missing on workers {missing}.",
                        type="warning",
                    )
                    return
                if no_adapter:
                    ui.notify(
                        f"Workers {no_adapter} have '{model_select.value}' "
                        f"loaded WITHOUT an adapter — the live SR pipeline "
                        f"uses raw mode and requires one. Make sure "
                        f"adapters/{model_select.value}_adapter.py "
                        f"(or another adapter referenced in the Files "
                        f"section) exists on the controller, then "
                        f"re-Distribute on the Experiment page.",
                        type="negative", timeout=15000,
                    )
                    return
                await controller.sr_pipeline.start()
                ui.notify("Pipeline started.", type="positive")
            except Exception as e:
                logger.exception("live start failed")
                ui.notify(f"Start failed: {e}", type="negative")

        async def _stop():
            try:
                await controller.sr_pipeline.stop()
                ui.notify("Pipeline stopped.", type="info")
            except Exception as e:
                ui.notify(f"Stop failed: {e}", type="negative")

        # =====================================================================
        # Stats refresh — frames update via MJPEG, only the numeric labels
        # need a Python-side timer.
        # =====================================================================
        def _tick_stats():
            s = controller.sr_pipeline.status()
            fps_label.text = f"FPS: {s.fps:5.1f}"
            inf_label.text = f"Avg inference: {s.avg_inference_ms:5.1f} ms"
            active_str = ", ".join(worker_label(w, controller)
                                   for w in s.active_workers) or "—"
            workers_label.text = f"Active: {active_str}"
            counters_label.text = (
                f"Processed: {s.frames_processed}   Failed: {s.frames_failed}"
            )
            err_label.text = s.last_error or ""

            # ---- Resolution labels ----
            # Source dims come from cv2 the moment the file opens; output
            # dims arrive with the first SR'd frame. Until then, both
            # show "pending …". The "Nx pixels" multiplier is computed
            # from the actual area ratio so it works for x2 (4x), x4
            # (16x), x8 (64x), or any aspect-preserving crop.
            src_res_label.text = "Source · " + _fmt_res(
                s.source_width, s.source_height,
            )
            out_suffix = ""
            if (s.source_width and s.source_height
                    and s.output_width and s.output_height):
                src_px = s.source_width * s.source_height
                out_px = s.output_width * s.output_height
                if src_px > 0:
                    ratio = out_px / src_px
                    out_suffix = f", {ratio:.1f}× pixels"
            out_res_label.text = "Cluster output · " + _fmt_res(
                s.output_width, s.output_height, out_suffix,
            )

            # ---- Record progress ----
            if s.record_path:
                if s.record_total_frames:
                    pct = (100 * s.record_frames_written
                           / max(s.record_total_frames, 1))
                    record_label.text = (
                        f"Recording → {s.record_path}  "
                        f"({s.record_frames_written}/{s.record_total_frames}, "
                        f"{pct:5.1f}%)"
                    )
                else:
                    record_label.text = (
                        f"Recording → {s.record_path}  "
                        f"({s.record_frames_written} frames)"
                    )
                # Pipeline finishes when source EOF is reached. Once it stops
                # AND we've written ≥1 frame, surface a download link.
                if (not s.running
                        and s.record_frames_written > 0
                        and not record_link.content):
                    fname = Path(s.record_path).name
                    record_link.set_content(
                        f'<a href="/api/recordings/{fname}" '
                        f'download style="color:#2563eb; '
                        f'text-decoration:underline">Download {fname}</a>'
                    )
            else:
                record_label.text = ""

        def _periodic_controls():
            # Refresh worker toggles + video / model option lists.
            current_wids = set(controller.state.registered_workers.keys())
            stale = set(worker_toggle_state.keys()) - current_wids
            for wid in stale:
                worker_toggle_state.pop(wid, None)
            new_wids = current_wids - set(worker_toggle_state.keys())
            if new_wids or stale:
                _build_worker_toggles()
                _push_subset()

            video_select.options = (_list_videos()
                                    or ["(upload a video to datasets/)"])
            if video_select.value not in video_select.options:
                video_select.value = video_select.options[0]
            video_select.update()

            model_select.options = _model_choices() or ["(no model uploaded yet)"]
            if model_select.value not in model_select.options:
                model_select.value = model_select.options[0]
            model_select.update()

        ui.timer(0.25, _tick_stats)       # 4 Hz is plenty for label updates
        ui.timer(2.0, _periodic_controls)
        _build_worker_toggles()
        _push_subset()


def _model_choices() -> list[str]:
    """Mirror experiment.py: list unique model stems available on disk."""
    p = Path("models")
    if not p.exists():
        return []
    stems = set()
    for f in p.iterdir():
        if f.is_file() and f.suffix.lower() in (".onnx", ".hef"):
            stems.add(f.stem)
    return sorted(stems)
