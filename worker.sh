#!/bin/bash
# Worker startup script for FYP Cluster.
#
# Designed for an air-gapped cluster: neither controller nor worker has
# internet. Wheels are pre-staged on the controller (under res/wheels/ and
# res/wheels-hailo/) and served via the data-plane HTTP API. This script:
#   1. Detects whether THIS worker has a Hailo NPU on the PCIe bus
#   2. Smoke-tests whether the local venv already has every required module
#   3. If not, fetches all common wheels from the controller; if Hailo is
#      present, also fetches Hailo-only wheels (hailort)
#   4. Installs everything into .venv via `uv pip install --no-index`
#   5. Launches the worker process
# pipefail so `cmd | tee` propagates cmd's failure rather than tee's success
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# ============================================================================
# Step 1 — Hailo detection
# ============================================================================
HAS_HAILO=0
if lspci 2>/dev/null | grep -qi hailo; then
    HAS_HAILO=1
    echo "[worker.sh] Hailo NPU detected on PCIe bus."
else
    echo "[worker.sh] No Hailo NPU detected — running in ONNX-only mode."
fi

# ============================================================================
# Step 2 — Resolve controller URL (for wheel pulls)
# Read from config.toml's [network] ethernet_subnet so we don't hardcode.
# Fall back to the project default if config.toml is missing/garbled.
# ============================================================================
ETH_SUBNET=$(python3 - <<'PY' 2>/dev/null
try:
    import tomllib, sys
    with open("config.toml", "rb") as f:
        cfg = tomllib.load(f)
    print(cfg["network"]["ethernet_subnet"])
except Exception:
    print("192.168.10.")
PY
)
DATA_PORT=$(python3 - <<'PY' 2>/dev/null
try:
    import tomllib
    with open("config.toml", "rb") as f:
        cfg = tomllib.load(f)
    print(cfg["controller"]["data_port"])
except Exception:
    print(8002)
PY
)
CONTROLLER_URL="http://${ETH_SUBNET}1:${DATA_PORT}"
echo "[worker.sh] Controller wheel cache: $CONTROLLER_URL/api/wheels"

# ============================================================================
# Step 3 — uv presence — fetch from controller if missing locally.
# Air-gapped clusters can't `curl https://astral.sh/uv/install.sh`, so the
# controller serves the same binary at /api/bin/uv. Operator pre-stages it
# (see scripts/prepare-wheels.sh).
# ============================================================================
ensure_uv() {
    if command -v uv &> /dev/null; then
        return 0
    fi
    if [ -x "$HOME/.local/bin/uv" ]; then
        export PATH="$HOME/.local/bin:$PATH"
        return 0
    fi
    echo "[worker.sh] uv not found locally — fetching from controller..."
    mkdir -p "$HOME/.local/bin"
    if curl -fsS --max-time 60 "${CONTROLLER_URL}/api/bin/uv" \
            -o "$HOME/.local/bin/uv"; then
        chmod +x "$HOME/.local/bin/uv"
        export PATH="$HOME/.local/bin:$PATH"
        echo "[worker.sh] Installed uv to $HOME/.local/bin/uv ($(uv --version 2>/dev/null || echo '?'))"
        return 0
    fi
    echo "[worker.sh] FATAL: could not fetch uv from $CONTROLLER_URL/api/bin/uv."
    echo "[worker.sh] Confirm res/bin/uv exists on the controller."
    return 1
}
ensure_uv || exit 1

# ============================================================================
# Step 4 — Smoke-test the venv. If every required import works we skip the
# whole download+install dance and go straight to launch.
# ============================================================================
REQ_MODULES="fastapi uvicorn websockets requests pydantic numpy onnxruntime smbus2"
if [ "$HAS_HAILO" = "1" ]; then
    REQ_MODULES_FULL="$REQ_MODULES hailo_platform"
else
    REQ_MODULES_FULL="$REQ_MODULES"
fi

needs_install=1
if [ -x ".venv/bin/python" ]; then
    if .venv/bin/python -c "import $(echo $REQ_MODULES_FULL | tr ' ' ',')" \
            > /dev/null 2>&1; then
        echo "[worker.sh] All required modules already importable — skipping install."
        needs_install=0
    fi
fi

# ============================================================================
# Step 5 — Pull wheels from controller, install offline
# ============================================================================
if [ "$needs_install" = "1" ]; then
    echo "[worker.sh] Fetching wheel manifest from controller..."

    mkdir -p .wheels

    # Helper: pull a JSON list, then download each filename it contains.
    fetch_wheel_set() {
        local list_endpoint="$1"
        local file_endpoint_prefix="$2"
        local manifest
        manifest=$(curl -fsS --max-time 10 "${CONTROLLER_URL}${list_endpoint}") \
            || { echo "[worker.sh] Could not fetch ${list_endpoint}"; return 1; }
        # Manifest is a JSON list of filenames; one per line via Python.
        local wheels
        wheels=$(echo "$manifest" | python3 -c \
            "import json,sys; print('\n'.join(json.load(sys.stdin)))")
        if [ -z "$wheels" ]; then
            echo "[worker.sh] (no wheels under ${list_endpoint})"
            return 0
        fi
        for w in $wheels; do
            if [ -f ".wheels/$w" ]; then
                continue       # already cached
            fi
            echo "  ↓ $w"
            curl -fsS --max-time 120 \
                "${CONTROLLER_URL}${file_endpoint_prefix}/${w}" \
                -o ".wheels/$w" \
                || { echo "[worker.sh] FAILED to fetch $w"; rm -f ".wheels/$w"; return 1; }
        done
    }

    fetch_wheel_set /api/wheels /api/wheels \
        || { echo "[worker.sh] Common wheels fetch failed; aborting."; exit 1; }

    if [ "$HAS_HAILO" = "1" ]; then
        fetch_wheel_set /api/wheels-hailo /api/wheels-hailo \
            || { echo "[worker.sh] Hailo wheels fetch failed; aborting."; exit 1; }
    fi

    echo "[worker.sh] $(ls .wheels/*.whl 2>/dev/null | wc -l) wheels cached locally."

    # Create the venv if it doesn't exist yet. uv venv uses the system
    # Python that uv was built against; we ask for 3.11 explicitly to match
    # the wheels' cp311 tag.
    if [ ! -x ".venv/bin/python" ]; then
        echo "[worker.sh] Creating .venv with Python 3.11..."
        uv venv --python 3.11 .venv \
            || { echo "[worker.sh] uv venv failed; is python3.11 installed?"; exit 1; }
    fi

    echo "[worker.sh] Installing wheels offline (--no-index --find-links .wheels)..."
    # Collect every .whl AND any sdist tarballs we cached. uv pip will
    # compile sdists locally — needs build-essential + python3-dev on the
    # Pi for C extensions (e.g. netifaces when no aarch64 wheel exists).
    #
    # Use nullglob so an unmatched glob expands to nothing (instead of
    # leaving the literal "*.tar.gz" string, which would crash uv).
    shopt -s nullglob
    artefact_files=( .wheels/*.whl .wheels/*.tar.gz .wheels/*.zip )
    shopt -u nullglob

    if [ "${#artefact_files[@]}" -eq 0 ]; then
        echo "[worker.sh] FATAL: .wheels/ is empty after fetch."
        echo "[worker.sh] Check controller's wheel cache:"
        echo "[worker.sh]   curl ${CONTROLLER_URL}/api/wheels"
        echo "[worker.sh] If it returns [], operator must run scripts/prepare-wheels.sh"
        exit 1
    fi

    echo "[worker.sh] Installing ${#artefact_files[@]} package(s):"
    for f in "${artefact_files[@]}"; do
        echo "    - ${f##*/}"
    done

    # Tee uv's stdout+stderr to a side log so we can debug after the fact
    # even if systemd journal rolls. The 2>&1 collapses streams; sed adds
    # a clear prefix so this command's output is easy to grep for.
    if ! uv pip install \
            --python .venv/bin/python \
            --no-index --find-links .wheels \
            "${artefact_files[@]}" 2>&1 | tee /tmp/fyp_worker_install.log; then
        echo "[worker.sh] uv pip install FAILED."
        echo "[worker.sh] Full output saved to /tmp/fyp_worker_install.log"
        echo "[worker.sh] Last 30 lines of that log:"
        tail -30 /tmp/fyp_worker_install.log | sed 's/^/  | /'
        exit 1
    fi
fi

# ============================================================================
# Step 6 — Skip the interactive "press enter" prompt under systemd
# ============================================================================
if [ -t 0 ]; then
    echo "[worker.sh] Connect this worker to the cluster network. Press Enter..."
    read -r
fi

# ============================================================================
# Step 7 — Launch
# Same sudo handling as controller.sh — systemd launches us as root, in which
# case re-invoking sudo would block on a non-existent tty.
# ============================================================================
if [ "$(id -u)" -eq 0 ] || [ "$FYP_NO_SUDO" = "1" ]; then
    echo "[worker.sh] Already root (or FYP_NO_SUDO=1) — running directly."
    exec env "PATH=$PATH" "PYTHONPATH=$PYTHONPATH" \
         .venv/bin/python -m worker.worker "$@"
else
    exec sudo env "PATH=$PATH" "PYTHONPATH=$PYTHONPATH" \
         .venv/bin/python -m worker.worker "$@"
fi
