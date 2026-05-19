"""
NiceGUI Web UI for the cluster controller.

We mount NiceGUI on its own ASGI app and serve it with uvicorn on the
controller's main asyncio loop alongside the control and data planes.

Earlier versions launched NiceGUI on a daemon thread via ``ui.run(...)``,
which silently hung in mock mode because NiceGUI's own loop / signal-handler
bootstrap assumes it runs on the main thread of the main interpreter. Calling
``ui.run_with(app)`` instead lets uvicorn drive it the same way as the rest
of the controller's HTTP services.

Use:
    from controller.web_ui import build_web_app
    web_app = build_web_app(controller)
    # uvicorn-serve `web_app` on controller.config['controller']['web_port']
"""
from __future__ import annotations

import asyncio
import logging

# Module-level imports for the MJPEG streaming endpoints. Defining the
# endpoint handlers inside a function is fine, but their type annotations
# (`request: Request`) must resolve via the function's globals at decoration
# time — FastAPI uses ``inspect.signature`` to discover dependencies, and
# nested ``import`` statements don't always make it into that resolution
# path on every Python/FastAPI version. Importing at module scope avoids
# the rabbit hole.
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

logger = logging.getLogger(__name__)


def build_web_app(controller):
    """Build a FastAPI app with NiceGUI mounted on it.

    Returns the FastAPI app for uvicorn to serve, or ``None`` if NiceGUI /
    FastAPI are not installed. NiceGUI refuses to mount onto its own
    ``nicegui.app`` (it raises ValueError to avoid infinite recursion), so we
    create a fresh FastAPI app here and hand that to ``ui.run_with``.
    """
    try:
        from fastapi import FastAPI
        from nicegui import ui
    except ImportError as e:
        logger.error("nicegui/fastapi not installed: %s", e)
        return None

    # Defer page-builder imports so an nicegui-less env still imports cleanly.
    from controller.web_ui import (
        overview, experiment, monitor, report, network, live,
    )

    overview.register(controller)
    experiment.register(controller)
    monitor.register(controller)
    report.register(controller)
    network.register(controller)
    live.register(controller)

    app = FastAPI(title="FYP Cluster Dashboard")

    # MJPEG streaming endpoints for the /live page — see comments in
    # _attach_mjpeg_streams() for why this is preferred over data: URL push.
    _attach_mjpeg_streams(app, controller)

    # Recordings download endpoint. The same route exists on the data plane
    # (data_api.py), but the dashboard runs on web_port — so the relative
    # `/api/recordings/<file>` link in /live resolves to web_port, not
    # data_port, and 404s. Mounting the same route here fixes the link
    # without forcing the user to know which port serves what.
    _attach_recordings_endpoint(app)

    # ui.run_with mounts NiceGUI's static files + websocket endpoints onto the
    # supplied FastAPI app. Uvicorn then drives `app` on whatever port the
    # caller wires up — no daemon thread, no second event loop.
    ui.run_with(
        app,
        title="FYP Cluster Dashboard",
        storage_secret="fyp-cluster-dev",
    )
    return app


def _attach_mjpeg_streams(app, controller) -> None:
    """Expose two MJPEG endpoints for the /live page.

    The naïve approach — push base64 frames via ``ui.image.set_source`` on a
    30 Hz timer — flickers because each ``src`` reassignment forces the
    browser to re-decode the data URL, with a brief blank state between
    decodes. ``multipart/x-mixed-replace`` is the IP-camera idiom: the
    browser holds one HTTP connection open, the server pushes successive
    JPEG parts, the ``<img>`` element renders them as a continuous stream
    with no flicker.

    We expose two streams so the /live page's left ('orig') and right
    ('upscaled') panes update independently (and stay in sync because both
    derive from the same FramePair).
    """
    BOUNDARY = "fyp-frame"

    async def _stream_generator(request: Request, get_jpeg):
        last_id = -1
        try:
            while True:
                if await request.is_disconnected():
                    return
                pair = await controller.sr_pipeline.latest_frame()
                if pair is not None and pair.frame_id != last_id:
                    last_id = pair.frame_id
                    jpeg = get_jpeg(pair)
                    head = (
                        f"--{BOUNDARY}\r\n"
                        f"Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(jpeg)}\r\n\r\n"
                    ).encode("ascii")
                    yield head + jpeg + b"\r\n"
                await asyncio.sleep(1 / 60)
        except asyncio.CancelledError:
            return

    @app.get("/stream/sr/orig.mjpeg")
    async def stream_orig(request: Request):
        return StreamingResponse(
            _stream_generator(request, lambda p: p.orig_jpeg),
            media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            headers={"Cache-Control": "no-cache, private",
                     "Pragma": "no-cache"},
        )

    @app.get("/stream/sr/upscaled.mjpeg")
    async def stream_upscaled(request: Request):
        return StreamingResponse(
            _stream_generator(request, lambda p: p.sr_jpeg),
            media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            headers={"Cache-Control": "no-cache, private",
                     "Pragma": "no-cache"},
        )


def _attach_recordings_endpoint(app) -> None:
    """Serve recordings/*.mp4 from the same port as the dashboard.

    The data plane already exposes /api/recordings/{filename}, but a
    relative link from /live resolves to the web_port — different
    server, no such route, browser sees 404. Adding the same handler
    here makes the link "just work" regardless of which port the
    dashboard happens to be on.
    """
    from pathlib import Path
    # Defer the import — controller._paths is part of the controller pkg
    # but pyproject's web_ui is loaded from a sibling package, and we
    # don't want a circular import at module-decode time.
    from controller._paths import RECORDINGS_DIR

    @app.get("/api/recordings/{filename}")
    async def get_recording(filename: str):
        # Resolve under RECORDINGS_DIR with no path traversal — same
        # contract as data_api._resolve_under, inlined to avoid the
        # cross-app dep.
        base = RECORDINGS_DIR.resolve()
        candidate = (base / Path(filename).name).resolve()
        if not str(candidate).startswith(str(base)):
            raise HTTPException(status_code=400, detail="bad filename")
        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(status_code=404, detail="Recording not found")
        return FileResponse(candidate, filename=candidate.name,
                            media_type="video/mp4")
