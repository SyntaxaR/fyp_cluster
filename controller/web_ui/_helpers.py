"""Shared display helpers for the web UI.

This module is the source of truth for page chrome — header bar, nav
pills, section headers, status chips — so every page looks identical
without each one duplicating the styling. Importing pages call:

    from controller.web_ui._helpers import page_header, section, status_chip
    page_header("Overview", active="overview")

If you want a global look-and-feel change, edit it here once.
"""
from __future__ import annotations

from typing import Optional


# =============================================================================
# Worker label — kept from the original module so callers don't need to
# rewrite imports.
# =============================================================================
def worker_label(wid, controller) -> str:
    """Render a worker as ``<id>:<identifier>`` (e.g. ``0:Sunny-Panda``).

    Falls back to ``<id>`` if the worker isn't currently registered (still
    pending, or just disconnected). Used everywhere the dashboard shows a
    worker id so ids stay human-readable across pages.
    """
    try:
        wid_int = int(wid)
    except (TypeError, ValueError):
        return str(wid)
    reg = controller.state.get_registered(wid_int)
    if reg is None or not reg.hardware_identifier:
        return str(wid_int)
    return f"{wid_int}:{reg.hardware_identifier}"


# =============================================================================
# Global theme — injected once per page via ``inject_theme()``. NiceGUI
# pages each render their own <head>, so this has to be called inside
# every ``@ui.page`` handler. The bundled stylesheet sets:
#   * a neutral page background
#   * card shadow + rounded corners
#   * pill-style nav bar
#   * status-chip color palette
# Tailwind classes used elsewhere stack on top without conflict.
# =============================================================================
_THEME_CSS = """
:root {
    --fyp-bg:        #f6f7fb;
    --fyp-card:      #ffffff;
    --fyp-border:    #e5e7eb;
    --fyp-text:      #1f2937;
    --fyp-muted:     #6b7280;
    --fyp-accent:    #2563eb;
    --fyp-accent-2:  #1e40af;
    --fyp-good:      #16a34a;
    --fyp-warn:      #d97706;
    --fyp-bad:       #dc2626;
    --fyp-idle:      #6b7280;
}

body, .nicegui-content {
    background: var(--fyp-bg) !important;
    color: var(--fyp-text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                 "Helvetica Neue", Roboto, sans-serif;
}

/* Top-of-page header */
.fyp-header {
    background: var(--fyp-card);
    border-bottom: 1px solid var(--fyp-border);
    padding: 1rem 1.5rem 0.75rem 1.5rem;
    margin: -1rem -1rem 1.25rem -1rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.fyp-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--fyp-text);
    line-height: 1.2;
}
.fyp-subtitle {
    font-size: 0.85rem;
    color: var(--fyp-muted);
    margin-top: 2px;
}

/* Nav pills */
.fyp-nav {
    display: flex;
    gap: 0.25rem;
    margin-top: 0.75rem;
    flex-wrap: wrap;
}
.fyp-nav a {
    padding: 0.35rem 0.85rem;
    border-radius: 999px;
    color: var(--fyp-muted) !important;
    text-decoration: none !important;
    font-size: 0.875rem;
    font-weight: 500;
    transition: background 120ms ease, color 120ms ease;
}
.fyp-nav a:hover {
    background: #eef2ff;
    color: var(--fyp-accent) !important;
}
.fyp-nav a.active {
    background: var(--fyp-accent);
    color: #fff !important;
}

/* Section card */
.fyp-card {
    background: var(--fyp-card);
    border: 1px solid var(--fyp-border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.fyp-section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--fyp-text);
    margin-bottom: 0.25rem;
}
.fyp-section-subtitle {
    font-size: 0.8rem;
    color: var(--fyp-muted);
    margin-bottom: 0.75rem;
}

/* Status chips */
.fyp-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    line-height: 1.4;
    border: 1px solid transparent;
}
.fyp-chip-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: currentColor;
    opacity: 0.85;
}
.fyp-chip-good     { color: var(--fyp-good);   background: #ecfdf5; border-color: #a7f3d0; }
.fyp-chip-warn     { color: var(--fyp-warn);   background: #fffbeb; border-color: #fde68a; }
.fyp-chip-bad      { color: var(--fyp-bad);    background: #fef2f2; border-color: #fecaca; }
.fyp-chip-info     { color: var(--fyp-accent); background: #eff6ff; border-color: #bfdbfe; }
.fyp-chip-idle     { color: var(--fyp-idle);   background: #f3f4f6; border-color: #e5e7eb; }

/* Tables — soften the default NiceGUI/Quasar look */
.q-table tbody tr:nth-child(even) td {
    background: #fafbfc;
}
.q-table thead tr th {
    font-weight: 600;
    color: var(--fyp-muted);
    font-size: 0.8rem;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

/* Buttons — slightly stronger affordance */
.q-btn:not(.q-btn--flat) {
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
"""

# Map (route → label) — single source of truth for the nav.
_NAV_ITEMS = [
    ("overview",    "Overview",   "/"),
    ("experiment",  "Experiment", "/experiment"),
    ("monitor",     "Monitor",    "/monitor"),
    ("reports",     "Reports",    "/reports"),
    ("network",     "Network",    "/network"),
    ("live",        "Live",       "/live"),
]


def inject_theme() -> None:
    """Inject the shared CSS theme into the current page. Call this once
    inside every ``@ui.page`` handler before any other UI calls."""
    from nicegui import ui
    ui.add_head_html(f"<style>{_THEME_CSS}</style>")


def page_header(title: str,
                subtitle: Optional[str] = None,
                active: Optional[str] = None,
                right_slot=None) -> None:
    """Render the standard page header with title, optional subtitle,
    and the shared nav pills. Call as the FIRST thing in every page.

    ``active`` is one of the keys in ``_NAV_ITEMS`` (or ``None`` to leave
    no nav item highlighted). ``right_slot`` is an optional callable that
    receives no arguments and renders extra controls to the right of the
    title (e.g. a "Restart" button on Overview).
    """
    from nicegui import ui

    inject_theme()
    with ui.element("div").classes("fyp-header"):
        with ui.row().classes("w-full items-center justify-between no-wrap"):
            with ui.element("div"):
                ui.html(f'<div class="fyp-title">{_html_escape(title)}</div>')
                if subtitle:
                    ui.html(
                        f'<div class="fyp-subtitle">{_html_escape(subtitle)}</div>'
                    )
            if right_slot is not None:
                with ui.element("div"):
                    right_slot()

        # Nav pills
        with ui.element("div").classes("fyp-nav"):
            for key, label, route in _NAV_ITEMS:
                cls = "active" if key == active else ""
                ui.html(
                    f'<a href="{route}" class="{cls}">{label}</a>'
                )


def section(title: Optional[str] = None,
            subtitle: Optional[str] = None,
            classes: str = ""):
    """Context-manager-style section card. Use as::

        with section("Workers", "Current cluster fleet"):
            ui.table(...)

    Renders a white rounded card with optional title + subtitle. Returns
    the NiceGUI element so the caller can attach extra classes if needed.
    """
    from nicegui import ui

    card = ui.element("div").classes(f"fyp-card w-full {classes}")
    with card:
        if title:
            ui.html(f'<div class="fyp-section-title">{_html_escape(title)}</div>')
        if subtitle:
            ui.html(
                f'<div class="fyp-section-subtitle">{_html_escape(subtitle)}</div>'
            )
    return card


def status_chip(status: str) -> str:
    """Return an HTML chip string for a worker status. Use with
    ``ui.html(status_chip(reg.status))`` when you want a colored badge
    instead of a plain text label.

    Maps the worker FSM states defined in ``shared.models.WorkerStatus``:
        active        → green
        registered    → blue
        reconnecting  → amber
        inactive      → red
        pending       → grey
    Unknown values fall back to grey.
    """
    s = str(status).lower()
    css, label = {
        "active":       ("fyp-chip-good",  "ACTIVE"),
        "registered":   ("fyp-chip-info",  "REGISTERED"),
        "reconnecting": ("fyp-chip-warn",  "RECONNECTING"),
        "inactive":     ("fyp-chip-bad",   "INACTIVE"),
        "pending":      ("fyp-chip-idle",  "PENDING"),
    }.get(s, ("fyp-chip-idle", s.upper() or "—"))
    return (
        f'<span class="fyp-chip {css}">'
        f'<span class="fyp-chip-dot"></span>{label}</span>'
    )


def _html_escape(s: str) -> str:
    """Cheap inline HTML escape — page_header / section titles pass user-
    controlled identifiers (worker names, model files) through to the DOM."""
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
    )
