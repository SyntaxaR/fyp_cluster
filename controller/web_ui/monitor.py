"""
Monitor view — live cluster telemetry.

Live panels:
* System stats: CPU temp, NPU temp (Hailo), CPU usage % for controller and
  every active worker
* Per-worker request stats (dispatched / OK / fail / inflight / latency / QPS)
* Multi-line throughput chart (req/s, last 60 s)
* Per-worker power chart (W, last 60 s) — INA226 readings

Route: /monitor   (legacy aliases /power-monitor and /realtime are kept so
old bookmarks still work).
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

from controller.web_ui._helpers import page_header, worker_label


def register(controller) -> None:
    from nicegui import ui

    def _build_page():
        page_header(
            "Monitor",
            subtitle="Live telemetry — system stats, per-worker requests, power.",
            active="monitor",
        )

        # =====================================================================
        # System stats (controller + every worker) — CPU %, CPU temp,
        # NPU temp. Refreshed every 2 s. Worker values arrive via heartbeat.
        # =====================================================================
        ui.label("System stats").classes("text-lg font-semibold mt-2")
        ui.label(
            "Controller values are sampled locally each tick. Worker values "
            "arrive via heartbeat (~3 s cadence)."
        ).classes("text-xs text-gray-500")
        # NPU temp removed — Hailo-8 SDK doesn't expose a stable
        # temperature read across versions, so the column was always
        # "—" in practice. CPU % and CPU temp give a useful proxy for
        # NPU thermal load anyway since Hailo runs hot-coupled to the SoC.
        sys_table = ui.table(
            columns=[
                {"name": "host",     "label": "Host",        "field": "host"},
                {"name": "role",     "label": "Role",        "field": "role"},
                {"name": "cpu_pct",  "label": "CPU %",       "field": "cpu_pct"},
                {"name": "cpu_temp", "label": "CPU temp °C", "field": "cpu_temp"},
                {"name": "age",      "label": "Sample age",  "field": "age"},
            ],
            rows=[],
            row_key="host",
        ).classes("w-full")

        def _fmt(v, unit: str = "", digits: int = 1) -> str:
            if v is None:
                return "—"
            try:
                return f"{float(v):.{digits}f}{unit}"
            except Exception:
                return str(v)

        def refresh_sys_stats():
            from controller.host_stats import collect as _collect_local
            rows = []
            now = int(time.time())

            # Controller row
            try:
                ctrl = _collect_local()
            except Exception:
                ctrl = {}
            rows.append({
                "host": controller.identifier or "controller",
                "role": "controller",
                "cpu_pct":  _fmt(ctrl.get("cpu_usage_pct"), "%"),
                "cpu_temp": _fmt(ctrl.get("cpu_temp_c"), "°C"),
                "age": "now",
            })

            # Worker rows — values come from the registration object,
            # populated by the heartbeat handler.
            for wid, reg in sorted(controller.state.registered_workers.items()):
                age_s = max(0, now - int(getattr(reg, "timestamp", now) or now))
                rows.append({
                    "host": worker_label(wid, controller),
                    "role": "worker",
                    "cpu_pct":  _fmt(getattr(reg, "cpu_usage_pct", None), "%"),
                    "cpu_temp": _fmt(getattr(reg, "cpu_temp_c", None), "°C"),
                    "age": f"{age_s}s ago",
                })

            sys_table.rows = rows
            sys_table.update()

        # =====================================================================
        # Per-worker stats table
        # =====================================================================
        worker_table = ui.table(
            columns=[
                {"name": "worker", "label": "Worker", "field": "worker"},
                {"name": "dispatched", "label": "Dispatched", "field": "dispatched"},
                {"name": "succeeded", "label": "OK", "field": "succeeded"},
                {"name": "failed", "label": "Fail", "field": "failed"},
                {"name": "inflight", "label": "In-flight", "field": "inflight"},
                {"name": "last_latency_ms", "label": "Last latency (ms)",
                 "field": "last_latency_ms"},
                {"name": "qps", "label": "QPS (1 s)", "field": "qps"},
            ],
            rows=[],
            row_key="worker",
        ).classes("w-full")

        # =====================================================================
        # Per-worker throughput chart (multi-line, last 60 s)
        # =====================================================================
        throughput_chart = ui.echart({
            "title": {"text": "Per-worker throughput (req/s) — last 60 s"},
            "legend": {"data": []},
            "xAxis": {"type": "category", "data": []},
            "yAxis": {"type": "value"},
            "series": [],
            "tooltip": {"trigger": "axis"},
            "animation": False,
        }).classes("w-full h-72 mt-4")

        # =====================================================================
        # CPU usage + CPU temp chart (controller + every worker, last 60 s)
        # Two y-axes: % on the left, °C on the right. Each host gets two
        # series (cpu% solid, temp dashed) coloured the same.
        # =====================================================================
        cpu_chart = ui.echart({
            "title": {"text": "CPU usage & temperature — last 60 s"},
            "legend": {"data": []},
            "xAxis": {"type": "category", "data": []},
            "yAxis": [
                {"type": "value", "name": "CPU %", "min": 0, "max": 100,
                 "position": "left"},
                # Pin temperature to a fixed window so the line doesn't
                # auto-rescale every tick — easier to eyeball at a glance
                # whether the SoC is idling (~40°C) or thermal-throttling
                # (~85°C). Pi 5 typical operating range fits in 20-100.
                {"type": "value", "name": "°C", "min": 20, "max": 100,
                 "position": "right"},
            ],
            "series": [],
            "tooltip": {"trigger": "axis"},
            "animation": False,
        }).classes("w-full h-72 mt-4")

        # Rolling 60-sample history per (host, metric). Keys are stable
        # (controller identifier or "<wid>:<identifier>") so a worker
        # going INACTIVE mid-window still keeps its line until it scrolls
        # off the right edge.
        cpu_history: dict[str, dict[str, deque]] = defaultdict(
            lambda: {"cpu_pct": deque(maxlen=60), "temp": deque(maxlen=60)}
        )
        cpu_time_axis: deque = deque(maxlen=60)

        def refresh_cpu_chart():
            from controller.host_stats import collect as _collect_local
            now_sec = int(time.time())
            if not cpu_time_axis or cpu_time_axis[-1] != now_sec:
                cpu_time_axis.append(now_sec)

            # Controller sample
            try:
                ctrl = _collect_local()
            except Exception:
                ctrl = {}
            ctrl_label = controller.identifier or "controller"
            cpu_history[ctrl_label]["cpu_pct"].append(
                (now_sec, ctrl.get("cpu_usage_pct"))
            )
            cpu_history[ctrl_label]["temp"].append(
                (now_sec, ctrl.get("cpu_temp_c"))
            )

            # Worker samples (from registration; fresh values arrived via
            # heartbeat). Only include workers active in the last 30 s so
            # stale rows don't pollute the chart with flat lines.
            stale_threshold = now_sec - 30
            for wid, reg in controller.state.registered_workers.items():
                if int(getattr(reg, "timestamp", 0) or 0) < stale_threshold:
                    continue
                key = worker_label(wid, controller)
                cpu_history[key]["cpu_pct"].append(
                    (now_sec, getattr(reg, "cpu_usage_pct", None))
                )
                cpu_history[key]["temp"].append(
                    (now_sec, getattr(reg, "cpu_temp_c", None))
                )

            xs = list(cpu_time_axis)
            x_labels = [time.strftime("%H:%M:%S", time.localtime(x)) for x in xs]

            # Stable per-host color palette. Same host always gets the
            # same color across refreshes — even if workers come and
            # go — so the user can mentally track a single line over
            # time. Picked by md5(host)%len(palette) instead of
            # iteration order. CPU% (solid) and °C (dashed) for the
            # SAME host share this color, so the eye pairs them up
            # automatically.
            HOST_PALETTE = (
                "#2563eb",  # blue
                "#16a34a",  # green
                "#dc2626",  # red
                "#9333ea",  # purple
                "#d97706",  # amber
                "#0891b2",  # cyan
                "#db2777",  # pink
                "#65a30d",  # lime
            )
            import hashlib as _hl
            def _color_for(host: str) -> str:
                h = int(_hl.md5(host.encode("utf-8")).hexdigest()[:8], 16)
                return HOST_PALETTE[h % len(HOST_PALETTE)]

            series = []
            legend = []
            for host, metrics in sorted(cpu_history.items()):
                cpu_map = dict(metrics["cpu_pct"])
                temp_map = dict(metrics["temp"])
                cpu_data = [cpu_map.get(x) for x in xs]
                temp_data = [temp_map.get(x) for x in xs]
                # Skip hosts with no readings AT ALL in the window
                if not any(v is not None for v in cpu_data + temp_data):
                    continue
                color = _color_for(host)
                cpu_name = f"{host} CPU%"
                temp_name = f"{host} °C"
                legend.extend([cpu_name, temp_name])
                series.append({
                    "name": cpu_name, "type": "line",
                    "data": [None if v is None else round(v, 1) for v in cpu_data],
                    "yAxisIndex": 0, "smooth": True, "showSymbol": False,
                    "connectNulls": False,
                    # Solid CPU% line in the host's color.
                    "itemStyle": {"color": color},
                    "lineStyle": {"color": color, "width": 2},
                })
                series.append({
                    "name": temp_name, "type": "line",
                    "data": [None if v is None else round(v, 1) for v in temp_data],
                    "yAxisIndex": 1, "smooth": True, "showSymbol": False,
                    "connectNulls": False,
                    # Dashed °C line in the SAME host color so the
                    # eye pairs CPU% and °C for one device together.
                    "itemStyle": {"color": color},
                    "lineStyle": {"color": color, "width": 2, "type": "dashed"},
                })

            cpu_chart.options["legend"]["data"] = legend
            cpu_chart.options["xAxis"]["data"] = x_labels
            cpu_chart.options["series"] = series
            cpu_chart.update()

        # =====================================================================
        # Per-worker power chart (multi-line, last 60 s) — INA226 readings
        # =====================================================================
        power_chart = ui.echart({
            "title": {"text": "Per-worker power (W) — last 60 s"},
            "legend": {"data": []},
            "xAxis": {"type": "category", "data": []},
            "yAxis": {"type": "value", "name": "W"},
            "series": [],
            "tooltip": {"trigger": "axis"},
            "animation": False,
        }).classes("w-full h-72 mt-4")

        # State for derived metrics:
        # worker_id -> {"prev_succeeded": int, "history": deque[(ts, qps)]}
        per_worker_throughput: dict[int, dict] = {}
        time_axis: deque = deque(maxlen=60)

        # ---------------------------------------------------------------------
        # Throughput refresh
        # ---------------------------------------------------------------------
        def refresh_workers():
            now_sec = int(time.time())
            if not time_axis or time_axis[-1] != now_sec:
                time_axis.append(now_sec)

            rows = []
            chart_series = []
            chart_legend = []

            for wid, stats in controller.state.worker_stats.items():
                label = worker_label(wid, controller)

                state = per_worker_throughput.setdefault(wid, {
                    "prev_succeeded": stats.requests_succeeded,
                    "history": deque(maxlen=60),
                    "last_ts": now_sec,
                })
                # Reset on counter rollback (controller restart, etc.)
                if stats.requests_succeeded < state["prev_succeeded"]:
                    state["prev_succeeded"] = 0
                    state["history"].clear()

                if state["last_ts"] != now_sec:
                    delta = stats.requests_succeeded - state["prev_succeeded"]
                    elapsed = max(now_sec - state["last_ts"], 1)
                    qps = delta / elapsed
                    state["history"].append((now_sec, qps))
                    state["prev_succeeded"] = stats.requests_succeeded
                    state["last_ts"] = now_sec
                cur_qps = (state["history"][-1][1]
                           if state["history"] else 0.0)

                rows.append({
                    "worker": label,
                    "dispatched": stats.requests_dispatched,
                    "succeeded": stats.requests_succeeded,
                    "failed": stats.requests_failed,
                    "inflight": stats.inflight,
                    "last_latency_ms": round(stats.last_latency_ms, 2),
                    "qps": round(cur_qps, 2),
                })

                hist = dict(state["history"])
                series_data = [round(hist.get(ts, 0.0), 2) for ts in time_axis]
                chart_legend.append(label)
                chart_series.append({
                    "name": label,
                    "type": "line",
                    "data": series_data,
                    "smooth": True,
                    "showSymbol": False,
                })

            worker_table.rows = rows
            worker_table.update()

            throughput_chart.options["legend"]["data"] = chart_legend
            throughput_chart.options["xAxis"]["data"] = [
                time.strftime("%H:%M:%S", time.localtime(t)) for t in time_axis
            ]
            throughput_chart.options["series"] = chart_series
            throughput_chart.update()

        # ---------------------------------------------------------------------
        # Power refresh — one line per worker_id (samples without binding go
        # under "(unbound)" so the user still sees the raw INA226 traffic).
        # ---------------------------------------------------------------------
        def refresh_power():
            now_ms = int(time.time() * 1000)
            samples = controller.db.power_samples_in_range(now_ms - 60_000, now_ms)

            # Two-tier bucketing:
            #   * For samples with a bound worker_id, group by worker_id so
            #     the legend shows "<id>:<identifier>" (the calibrated reading).
            #   * For samples with worker_id=None (chip not yet bound), group
            #     by i2c_address so the operator can see EACH unbound chip
            #     separately as "unbounded(0xXX)" — critical when more than
            #     one chip is present and only some are calibrated.
            buckets: dict[tuple[str, object], dict[int, list[float]]] = defaultdict(
                lambda: defaultdict(list)
            )
            seconds: set[int] = set()
            for s in samples:
                sec = s.timestamp_ms // 1000
                seconds.add(sec)
                if s.worker_id is None:
                    key = ("chip", int(s.i2c_address))
                else:
                    key = ("worker", int(s.worker_id))
                buckets[key][sec].append(s.power_w)
            xs = sorted(seconds)

            # Sort: bound workers first (by id), then unbound chips (by address).
            def _sort_key(k):
                kind, value = k
                return (0 if kind == "worker" else 1, value)

            series = []
            legend = []
            for key in sorted(buckets.keys(), key=_sort_key):
                kind, value = key
                if kind == "worker":
                    name = worker_label(int(value), controller)
                else:
                    name = f"unbounded(0x{int(value):02X})"
                ys = []
                for x in xs:
                    samples_in_bucket = buckets[key].get(x, [])
                    if samples_in_bucket:
                        ys.append(round(
                            sum(samples_in_bucket) / len(samples_in_bucket), 3
                        ))
                    else:
                        # Insert null so the chart leaves a gap rather than
                        # connecting the line through silence — important for
                        # workers that go INACTIVE mid-window.
                        ys.append(None)
                legend.append(name)
                series.append({
                    "name": name,
                    "type": "line",
                    "data": ys,
                    "smooth": True,
                    "showSymbol": False,
                    "connectNulls": False,
                })

            power_chart.options["legend"]["data"] = legend
            power_chart.options["xAxis"]["data"] = [
                time.strftime("%H:%M:%S", time.localtime(x)) for x in xs
            ]
            power_chart.options["series"] = series
            power_chart.update()

        ui.timer(2.0, refresh_sys_stats)
        ui.timer(1.0, refresh_workers)
        ui.timer(1.0, refresh_power)
        ui.timer(1.0, refresh_cpu_chart)
        refresh_sys_stats()
        refresh_workers()
        refresh_power()
        refresh_cpu_chart()

    # Primary route + two legacy aliases so old bookmarks still work after
    # the rename ("Power monitor" -> "Monitor").
    ui.page("/monitor")(_build_page)
    ui.page("/power-monitor")(_build_page)
    ui.page("/realtime")(_build_page)
