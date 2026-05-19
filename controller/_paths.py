"""
Single source of truth for the controller's runtime directory layout.

Layout::

    <repo-root>/
    ├── adapters/                   <- runtime uploads land here (gitignored)
    ├── models/                     <- runtime uploads land here (gitignored)
    ├── datasets/                   <- runtime uploads land here (gitignored)
    ├── dispatchers/                <- runtime uploads land here (gitignored)
    ├── recordings/                 <- /live record-mode mp4 outputs
    │
    ├── demo/                       <- DEV-TIME staging area (committed)
    │   ├── adapters/               <- source adapter .py the operator
    │   │                              ships and will upload at demo time
    │   ├── models/                 <- (model files too big — usually
    │   │                              .gitignored individually)
    │   ├── datasets/               <- demo videos / images
    │   └── dispatchers/            <- custom dispatcher .py sources
    │
    ├── res/                        <- offline deployment infra
    │   ├── wheels/                 <- worker dep wheels (operator stages)
    │   ├── wheels-hailo/           <- Hailo-only wheels
    │   └── bin/                    <- e.g. uv binary
    │
    └── worker/inference/adapters/  <- BUILT-IN reference adapters (committed)
                                       e.g. yolov11_adapter.py

Two-tier separation between source and runtime:
  * `demo/adapters/real_esrgan_adapter.py`   — committed source, edited
                                                 in-repo, copied/uploaded
                                                 to runtime dir at demo time
  * `adapters/real_esrgan_adapter.py`        — gitignored runtime copy that
                                                 web-UI uploads land into
                                                 and experiment_manager
                                                 reads to distribute
"""
from __future__ import annotations

from pathlib import Path


# ---- runtime, user-uploaded content (gitignored) ---------------------------
MODELS_DIR      = Path("models")
ADAPTERS_DIR    = Path("adapters")
DATASETS_DIR    = Path("datasets")
DISPATCHERS_DIR = Path("dispatchers")

# ---- pipeline outputs (live recording) -------------------------------------
RECORDINGS_DIR  = Path("recordings")

# ---- dev-time staging area (committed to git) ------------------------------
DEMO_DIR              = Path("demo")
DEMO_ADAPTERS_DIR     = DEMO_DIR / "adapters"
DEMO_MODELS_DIR       = DEMO_DIR / "models"
DEMO_DATASETS_DIR     = DEMO_DIR / "datasets"
DEMO_DISPATCHERS_DIR  = DEMO_DIR / "dispatchers"

# ---- offline deployment infrastructure (operator-staged) -------------------
RES_DIR          = Path("res")
WHEELS_DIR       = RES_DIR / "wheels"
WHEELS_HAILO_DIR = RES_DIR / "wheels-hailo"
BIN_DIR          = RES_DIR / "bin"


def ensure_runtime_dirs() -> None:
    """Create every runtime dir if missing — called once at controller start
    so first-upload doesn't need to mkdir defensively. Does NOT touch
    demo/* — those are source-controlled."""
    for d in (MODELS_DIR, ADAPTERS_DIR, DATASETS_DIR,
              DISPATCHERS_DIR, RECORDINGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
