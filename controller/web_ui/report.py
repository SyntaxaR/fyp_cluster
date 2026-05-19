"""Past-experiments report page + per-experiment drill-down.

Routes:
    /reports          — searchable list of all past experiments
    /reports/{exp_id} — drill-down: summary, power-time curve,
                        per-worker bar chart, CSV / JSON downloads
"""
from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path
from typing import Any


_FLOAT_KEYS = (
    "avg_latency_ms", "p95_latency_ms", "p99_latency_ms",
    "avg_throughput_qps", "avg_cluster_power_w", "energy_per_request_j",
)


def _round_floats(row: dict[str, Any], digits: int = 3) -> dict[str, Any]:
    out = dict(row)
    for k in _FLOAT_KEYS:
        v = out.get(k)
        if isinstance(v, float):
            out[k] = round(v, digits)
    return out


def _csv_bytes(headers: list[str], rows: list[list]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(headers)
    for r in rows:
        w.writerow(r)
    return buf.getvalue().encode("utf-8")


def register(controller) -> None:
    from nicegui import ui

    # =========================================================================
    # /reports — list view
    # =========================================================================
    @ui.page("/reports")
    def report_page():
        from controller.web_ui._helpers import page_header
        page_header(
            "Experiment reports",
            subtitle=("Click a row to drill down. CSV / JSON exports are "
                      "available on the detail page."),
            active="reports",
        )

        table = ui.table(
            columns=[
                {"name": "id", "label": "#", "field": "id"},
                {"name": "name", "label": "Name", "field": "name"},
                {"name": "model", "label": "Model", "field": "model"},
                {"name": "dispatcher", "label": "Dispatcher", "field": "dispatcher"},
                {"name": "mode", "label": "Mode", "field": "mode"},
                {"name": "status", "label": "Status", "field": "status"},
                {"name": "total_requests", "label": "Total", "field": "total_requests"},
                {"name": "successful", "label": "OK", "field": "successful"},
                {"name": "failed", "label": "Fail", "field": "failed"},
                {"name": "avg_latency_ms", "label": "Avg lat (ms)",
                 "field": "avg_latency_ms"},
                {"name": "p95_latency_ms", "label": "p95 (ms)", "field": "p95_latency_ms"},
                {"name": "avg_throughput_qps", "label": "QPS",
                 "field": "avg_throughput_qps"},
                {"name": "avg_cluster_power_w", "label": "Avg power (W)",
                 "field": "avg_cluster_power_w"},
                {"name": "energy_per_request_j", "label": "J / req",
                 "field": "energy_per_request_j"},
            ],
            rows=[],
            row_key="id",
            on_select=lambda e: ui.navigate.to(
                f"/reports/{e.selection[0]['id']}"
            ) if e.selection else None,
        ).classes("w-full")
        table.props("selection=single")

        def refresh():
            try:
                rows = [_round_floats(r) for r in controller.db.list_experiments(limit=200)]
                table.rows = rows
                table.update()
            except Exception as e:
                ui.notify(f"Failed to refresh: {e}", type="warning")

        with ui.row().classes("mt-2"):
            ui.button("Refresh", on_click=refresh)
        ui.timer(5.0, refresh)
        refresh()

    # =========================================================================
    # /reports/{id} — drill-down view
    # =========================================================================
    @ui.page("/reports/{exp_id}")
    def report_detail(exp_id: str):
        from controller.web_ui._helpers import page_header
        page_header(
            f"Experiment #{exp_id}",
            subtitle="Per-worker breakdown, power timeline, raw exports.",
            active="reports",
        )

        try:
            eid = int(exp_id)
        except ValueError:
            ui.label(f"Invalid experiment id: {exp_id}").classes("text-red-500")
            return

        exp = controller.db.get_experiment(eid)
        if exp is None:
            ui.label(f"Experiment #{eid} not found.").classes("text-red-500")
            ui.link("← Back to reports", "/reports")
            return

        per_worker_rows = controller.db.list_experiment_workers(eid)

        # ---------- Summary card ----------
        with ui.card().classes("w-full mt-2"):
            ui.label(f"#{eid}: {exp['name']}").classes("text-lg font-semibold")
            ui.label(
                f"model={exp['model']}  •  dispatcher={exp['dispatcher']}  •  "
                f"mode={exp['mode']}  •  status={exp['status']}"
            ).classes("text-sm text-gray-500")
            started = time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime((exp["started_ms"] or 0) / 1000.0))
            duration_s = max(((exp["finished_ms"] or 0)
                              - (exp["started_ms"] or 0)) / 1000.0, 0.0)
            ui.label(
                f"Started {started}  •  duration {duration_s:.1f} s  •  "
                f"requests {exp['total_requests']} "
                f"(OK {exp['successful']} / fail {exp['failed']})"
            ).classes("text-sm")
            with ui.row().classes("mt-2 gap-4"):
                ui.label(f"avg lat: "
                         f"{(exp['avg_latency_ms'] or 0):.1f} ms")
                ui.label(f"p95: {(exp['p95_latency_ms'] or 0):.1f} ms")
                ui.label(f"p99: {(exp['p99_latency_ms'] or 0):.1f} ms")
                ui.label(f"QPS: {(exp['avg_throughput_qps'] or 0):.1f}")
                ui.label(f"Power: "
                         f"{(exp['avg_cluster_power_w'] or 0):.2f} W")
                ui.label(f"Energy/req: "
                         f"{(exp['energy_per_request_j'] or 0):.3f} J")

            # ----- Observed compute throughput (TOPS) -----
            # Resolve the model's GOPS-per-inference (parsed from ONNX
            # graph or looked up by HEF filename in shared.model_ops),
            # then convert per-worker FPS into a comparable TOPS number.
            # Without this the user has to do the math by hand; with it
            # the report instantly shows "this Hailo worker is hitting
            # ~9.6 TOPS, ~37% of the chip's 26 TOPS peak."
            tops_block_gops = None
            try:
                from shared.model_ops import get_or_compute_gops
                from controller._paths import MODELS_DIR
                # Heuristic: take whichever model file matches exp.model
                # — either <model>.hef or <model>.onnx. Per-worker rows
                # below disambiguate further by backend.
                for ext in (".hef", ".onnx"):
                    cand = MODELS_DIR / f"{exp['model']}{ext}"
                    if cand.exists():
                        info = get_or_compute_gops(cand)
                        if info:
                            tops_block_gops = info["gops"]
                            tops_block_source = info["source"]
                            break
            except Exception:
                tops_block_gops = None

            if tops_block_gops:
                cluster_fps = float(exp['avg_throughput_qps'] or 0.0)
                cluster_tops = tops_block_gops * cluster_fps / 1000.0
                with ui.row().classes("mt-2 gap-4 items-center"):
                    ui.label(
                        f"Model OPS: {tops_block_gops:.2f} GOPS/inf"
                    ).classes("text-sm")
                    ui.label(
                        f"Cluster observed: {cluster_tops:.2f} TOPS"
                    ).classes("text-sm font-semibold")
                    ui.label(
                        "(per-worker breakdown below)"
                    ).classes("text-xs text-gray-500")

            if exp.get("notes"):
                ui.label(f"Notes: {exp['notes']}").classes("text-sm italic")

        # ---------- Power-time curve ----------
        ui.label("Power over time").classes("text-lg font-semibold mt-4")
        # Aggregate power_samples per second across all chips, optionally split
        # per chip when there are bindings.
        start_ms = int(exp["started_ms"] or 0)
        end_ms = int(exp["finished_ms"] or 0) or (start_ms + 1000)
        samples = controller.db.power_samples_in_range(start_ms, end_ms)

        # Aggregation: the INA226 driver pushes ~10 samples/sec per chip.
        # Old code SUMMED them per second, which gave a 10× overstatement
        # ("5 W chip" rendered as 50 W). Correct semantics:
        #
        #   per-chip / second   = AVERAGE of that chip's samples in that second
        #   cluster / second    = SUM of every chip's per-second average
        #
        # Net effect: a 1-chip cluster reads its true wattage; a 4-chip
        # cluster's "cluster" line is the actual aggregate draw.
        per_chip_raw: dict[int, dict[int, list[float]]] = {}
        for s in samples:
            sec = s.timestamp_ms // 1000
            per_chip_raw.setdefault(s.i2c_address, {}) \
                        .setdefault(sec, []).append(s.power_w)

        per_chip_buckets: dict[int, dict[int, float]] = {
            addr: {sec: sum(vals) / len(vals) for sec, vals in sec_dict.items()}
            for addr, sec_dict in per_chip_raw.items()
        }
        cluster_buckets: dict[int, float] = {}
        for sec_dict in per_chip_buckets.values():
            for sec, w in sec_dict.items():
                cluster_buckets[sec] = cluster_buckets.get(sec, 0.0) + w

        xs = sorted(cluster_buckets.keys())
        x_labels = [time.strftime("%H:%M:%S", time.localtime(x)) for x in xs]

        chart_series = [{
            "name": "cluster",
            "type": "line",
            "smooth": True,
            "showSymbol": False,
            "data": [round(cluster_buckets[x], 3) for x in xs],
        }]
        chart_legend = ["cluster"]
        for addr, buckets in sorted(per_chip_buckets.items()):
            name = f"chip 0x{addr:02X}"
            chart_legend.append(name)
            chart_series.append({
                "name": name,
                "type": "line",
                "smooth": True,
                "showSymbol": False,
                "data": [round(buckets.get(x, 0.0), 3) for x in xs],
            })

        ui.echart({
            "title": {"text": f"{len(samples)} samples"},
            "legend": {"data": chart_legend},
            "xAxis": {"type": "category", "data": x_labels},
            "yAxis": {"type": "value", "name": "W"},
            "series": chart_series,
            "tooltip": {"trigger": "axis"},
            "animation": False,
        }).classes("w-full h-72")

        # ---------- Per-worker bar chart ----------
        ui.label("Per-worker requests").classes("text-lg font-semibold mt-4")
        if not per_worker_rows:
            ui.label("(no per-worker data — older experiment, "
                     "or no workers participated)").classes("text-sm text-gray-500")
        else:
            wids = [str(r["worker_id"]) for r in per_worker_rows]
            disp = [r["dispatched"] for r in per_worker_rows]
            succ = [r["succeeded"] for r in per_worker_rows]
            fail = [r["failed"] for r in per_worker_rows]
            ui.echart({
                "legend": {"data": ["dispatched", "succeeded", "failed"]},
                "xAxis": {"type": "category", "data": wids,
                          "name": "worker_id"},
                "yAxis": {"type": "value", "name": "requests"},
                "series": [
                    {"name": "dispatched", "type": "bar", "data": disp},
                    {"name": "succeeded",  "type": "bar", "data": succ},
                    {"name": "failed",     "type": "bar", "data": fail},
                ],
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                "animation": False,
            }).classes("w-full h-72")

            # Per-worker latency bar
            avg_lat = [round(r["avg_latency_ms"], 2) for r in per_worker_rows]
            ui.echart({
                "title": {"text": "Avg latency per worker (ms)"},
                "xAxis": {"type": "category", "data": wids,
                          "name": "worker_id"},
                "yAxis": {"type": "value", "name": "ms"},
                "series": [{"type": "bar", "data": avg_lat}],
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                "animation": False,
            }).classes("w-full h-64")

            # Compute observed FPS and TOPS per worker:
            #   FPS  = succeeded / per-worker active window (seconds)
            #   TOPS = GOPS_per_inference × FPS / 1000
            # The active window comes from the per_worker_rows
            # started_ms / finished_ms (the first and last dispatch on
            # that worker). Falls back to the whole-experiment window
            # if those columns are missing on older rows.
            _ex_start = exp["started_ms"] or 0
            _ex_finish = exp["finished_ms"] or _ex_start

            def _fps_of(r) -> float:
                s = r.get("started_ms") or _ex_start
                f = r.get("finished_ms") or _ex_finish
                dt = max((f - s) / 1000.0, 1e-9)
                return float(r.get("succeeded", 0) or 0) / dt

            rows_out = []
            for r in per_worker_rows:
                fps = _fps_of(r)
                # Per-worker GOPS = model GOPS for this worker's backend
                # (a Hailo HEF and an ONNX file can have different OPS
                # counts even for the "same" model name). Pick whichever
                # extension matches the engine column.
                wgops = None
                try:
                    from shared.model_ops import get_or_compute_gops
                    from controller._paths import MODELS_DIR
                    eng = (r.get("engine") or "").lower()
                    ext_order = (".hef", ".onnx") if eng == "hailo" else (".onnx", ".hef")
                    for ext in ext_order:
                        cand = MODELS_DIR / f"{exp['model']}{ext}"
                        if cand.exists():
                            info = get_or_compute_gops(cand)
                            if info:
                                wgops = info["gops"]
                                break
                except Exception:
                    wgops = None
                tops = (wgops * fps / 1000.0) if wgops else 0.0
                rows_out.append({
                    **r,
                    "i2c_address": (f"0x{r['i2c_address']:02X}"
                                    if r["i2c_address"] is not None else "—"),
                    "avg_latency_ms": round(r["avg_latency_ms"], 2),
                    "fps": round(fps, 1),
                    "tops": (f"{tops:.2f}" if tops > 0 else "—"),
                })

            ui.table(
                columns=[
                    {"name": "worker_id", "label": "ID", "field": "worker_id"},
                    {"name": "identifier", "label": "Identifier", "field": "identifier"},
                    {"name": "engine", "label": "Engine", "field": "engine"},
                    {"name": "i2c_address", "label": "INA226", "field": "i2c_address"},
                    {"name": "dispatched", "label": "Dispatched", "field": "dispatched"},
                    {"name": "succeeded", "label": "OK", "field": "succeeded"},
                    {"name": "failed", "label": "Fail", "field": "failed"},
                    {"name": "avg_latency_ms", "label": "Avg lat (ms)",
                     "field": "avg_latency_ms"},
                    {"name": "fps",  "label": "FPS",  "field": "fps"},
                    {"name": "tops", "label": "TOPS observed", "field": "tops"},
                ],
                rows=rows_out,
                row_key="worker_id",
            ).classes("w-full mt-2")

        # ---------- Downloads ----------
        ui.label("Downloads").classes("text-lg font-semibold mt-4")
        with ui.row().classes("gap-2"):
            def _download_summary():
                headers = list(exp.keys())
                ui.download(
                    _csv_bytes(headers, [[exp.get(h, "") for h in headers]]),
                    filename=f"experiment_{eid}_summary.csv",
                )

            def _download_workers():
                if not per_worker_rows:
                    ui.notify("No per-worker rows", type="warning")
                    return
                headers = list(per_worker_rows[0].keys())
                ui.download(
                    _csv_bytes(headers,
                               [[r.get(h, "") for h in headers]
                                for r in per_worker_rows]),
                    filename=f"experiment_{eid}_workers.csv",
                )

            def _download_power():
                rows = [
                    [s.timestamp_ms, f"0x{s.i2c_address:02X}",
                     s.worker_id if s.worker_id is not None else "",
                     round(s.shunt_mv, 4), round(s.current_a, 4),
                     round(s.voltage_v, 4), round(s.power_w, 4)]
                    for s in samples
                ]
                ui.download(
                    _csv_bytes(
                        ["ts_ms", "i2c_address", "worker_id",
                         "shunt_mv", "current_a", "voltage_v", "power_w"],
                        rows,
                    ),
                    filename=f"experiment_{eid}_power.csv",
                )

            def _download_json():
                payload = {
                    "experiment": exp,
                    "workers": per_worker_rows,
                    "power_samples_count": len(samples),
                }
                ui.download(
                    json.dumps(payload, indent=2, default=str).encode("utf-8"),
                    filename=f"experiment_{eid}.json",
                )

            ui.button("Summary CSV", on_click=_download_summary)
            ui.button("Per-worker CSV", on_click=_download_workers)
            ui.button("Power CSV", on_click=_download_power)
            ui.button("Full JSON", on_click=_download_json)

        ui.link("← Back to reports", "/reports").classes("mt-4")
