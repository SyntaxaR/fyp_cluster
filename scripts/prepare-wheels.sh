#!/bin/bash
# Pre-stage all worker-side artefacts for an air-gapped Pi cluster.
#
# Run this on a machine WITH internet access. It will produce:
#     res/bin/uv                    — aarch64-linux-gnu uv binary
#     res/wheels/*.whl              — all worker-side Python deps
#     res/wheels-hailo/             — empty; drop hailort wheel here manually
#
# Once res/ is populated, copy the entire project tree (including res/) to
# the controller. Workers will pull binaries + wheels from the controller
# via /api/bin and /api/wheels on first boot.
#
# Usage:
#     bash scripts/prepare-wheels.sh                # incremental (skip cached)
#     FORCE=1 bash scripts/prepare-wheels.sh        # re-download everything
#     UV_VERSION=0.5.13 bash scripts/prepare-wheels.sh   # pin uv version
#
# Notes on platform tags:
#  * --python-version 3.11 must match the worker Pi's Python (3.11 on RPi
#    OS Bookworm).
#  * We pass MULTIPLE --platform tags so pip will accept wheels published
#    for any of them. manylinux_2_17_aarch64 is the widest-compat tag that
#    most projects (numpy, watchfiles, pydantic-core, ...) target.
#  * --only-binary=:all: forbids sdists — every dep must be a prebuilt
#    wheel (offline install can't compile).
#  * `uvicorn` is downloaded WITHOUT the `[standard]` extras: those bring
#    in watchfiles + uvloop + httptools + python-dotenv that the worker
#    does not actually use, and watchfiles in particular is a frequent
#    source of "no matching distribution" pain on aarch64.
#  * The hailort wheel is NOT on PyPI — drop it into res/wheels-hailo/.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

PYTHON_VERSION="3.11"
PLATFORMS=(
    "manylinux_2_28_aarch64"
    "manylinux_2_17_aarch64"
    "linux_aarch64"
    "any"
)

# Worker-side runtime deps. Keep in sync with what worker.py actually
# imports — controller-only packages (nicegui, paramiko) are excluded.
# uvicorn is plain (no [standard]) — see top-of-file note.
WORKER_DEPS=(
    "fastapi>=0.121.2,<0.122.0"
    "uvicorn>=0.38.0,<0.39.0"
    "websockets>=15.0.1,<16.0.0"
    "requests>=2.32.5,<3.0.0"
    "pydantic>=2.12.5,<3.0.0"
    "python-multipart>=0.0.20"
    # numpy<2 because hailort==4.20.0 metadata excludes numpy 2.x.
    "numpy>=1.24,<2"
    # opencv-python 4.10+ is built against numpy 2.x ABI; stay on 4.9.x for
    # the numpy<2 constraint to be satisfiable.
    "opencv-python>=4.9.0.80,<4.10"
    "pillow>=10.0,<12"
    "onnxruntime>=1.23.2"
    "smbus2>=0.4.3"
)

mkdir -p res/wheels res/wheels-hailo res/bin

# ----------------------------------------------------------------------------
# uv binary
# ----------------------------------------------------------------------------
UV_VERSION="${UV_VERSION:-latest}"
UV_TARGET="aarch64-unknown-linux-gnu"
echo "================================================================"
echo "uv binary (${UV_TARGET}, version ${UV_VERSION})"
echo "================================================================"
if [ -x "res/bin/uv" ] && [ "$FORCE" != "1" ]; then
    echo "[skip] res/bin/uv already exists. Use FORCE=1 to re-download."
else
    if [ "$UV_VERSION" = "latest" ]; then
        UV_URL="https://github.com/astral-sh/uv/releases/latest/download/uv-${UV_TARGET}.tar.gz"
    else
        UV_URL="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-${UV_TARGET}.tar.gz"
    fi
    TMPD=$(mktemp -d)
    curl -L --fail -o "$TMPD/uv.tar.gz" "$UV_URL"
    tar -xzf "$TMPD/uv.tar.gz" -C "$TMPD"
    cp "$TMPD/uv-${UV_TARGET}/uv" res/bin/uv
    chmod +x res/bin/uv
    rm -rf "$TMPD"
    echo "[ok] res/bin/uv installed."
fi

# ----------------------------------------------------------------------------
# Common worker wheels
# ----------------------------------------------------------------------------
echo
echo "================================================================"
echo "Common wheels (Python ${PYTHON_VERSION}, platforms: ${PLATFORMS[*]})"
echo "================================================================"
existing=$(ls res/wheels/*.whl 2>/dev/null | wc -l)
if [ "$existing" -gt 0 ] && [ "$FORCE" != "1" ]; then
    echo "[skip] res/wheels/ already has $existing wheel(s). Use FORCE=1 to refresh."
else
    PLATFORM_FLAGS=()
    for p in "${PLATFORMS[@]}"; do
        PLATFORM_FLAGS+=("--platform" "$p")
    done
    pip download \
        --dest res/wheels \
        --python-version "${PYTHON_VERSION}" \
        "${PLATFORM_FLAGS[@]}" \
        --only-binary=:all: \
        "${WORKER_DEPS[@]}"
    echo "[ok] $(ls res/wheels/*.whl | wc -l) wheel(s) under res/wheels/"
fi

# ----------------------------------------------------------------------------
# Hailo transitive deps
#
# hailort-*.whl declares Requires-Dist on a handful of helpers. Most are
# pure-Python (`argcomplete`, `contextlib2`, `future`, `netaddr`) and have
# universal `py3-none-any.whl` releases on PyPI — those download cleanly.
#
# `netifaces` is the troublemaker: it's a C extension that PyPI sometimes
# only publishes as sdist (.tar.gz) for aarch64. We try the binary path
# first; if pip can't find a wheel we fall back to allowing sdist so it
# at least gets cached. The worker then either uses a pre-built wheel
# (if one ended up in res/wheels/) or builds the sdist on first install
# (which needs `apt install build-essential python3-dev` on the Pi).
# ----------------------------------------------------------------------------
HAILO_WHL=$(ls res/wheels-hailo/hailort-*.whl 2>/dev/null | head -1)
if [ -n "$HAILO_WHL" ]; then
    echo
    echo "================================================================"
    echo "Resolving transitive deps for $(basename "$HAILO_WHL")"
    echo "================================================================"
    python3 - <<PY "$HAILO_WHL" || true
import sys, zipfile, email
whl = sys.argv[1]
with zipfile.ZipFile(whl) as z:
    meta_name = next(n for n in z.namelist() if n.endswith("METADATA"))
    meta = z.read(meta_name)
print(f"hailort declares the following Requires-Dist:")
for r in email.message_from_bytes(meta).get_all("Requires-Dist", []):
    print(f"  - {r}")
PY

    # Pass 1 — strict wheels only. Picks up argcomplete / contextlib2 /
    # future / netaddr which all have universal wheels.
    if ! pip download \
            --dest res/wheels \
            --python-version "${PYTHON_VERSION}" \
            "${PLATFORM_FLAGS[@]}" \
            --only-binary=:all: \
            --find-links res/wheels-hailo \
            --find-links res/wheels \
            hailort 2>/tmp/prepare_wheels_pass1.log; then
        echo "[info] strict-wheel resolve failed; retrying with --prefer-binary"
        echo "       (allows sdist for deps that don't ship aarch64 wheels)"
        # Pass 2 — accept sdist as fallback. Same packages but lets pip
        # grab netifaces-*.tar.gz when no aarch64 wheel exists.
        pip download \
            --dest res/wheels \
            --python-version "${PYTHON_VERSION}" \
            "${PLATFORM_FLAGS[@]}" \
            --prefer-binary \
            --find-links res/wheels-hailo \
            --find-links res/wheels \
            hailort \
            || { echo "[error] could not resolve hailort transitive deps."; \
                 cat /tmp/prepare_wheels_pass1.log; }
    fi

    # Warn loudly about any sdist that landed — the worker will need build
    # tools to compile it.
    sdists=$(ls res/wheels/*.tar.gz res/wheels/*.zip 2>/dev/null || true)
    if [ -n "$sdists" ]; then
        echo
        echo "[warn] These hailort transitive deps came as sdist, NOT wheel:"
        for s in $sdists; do echo "          $(basename "$s")"; done
        cat <<'EOF'

       Workers must compile them on first install. Make sure each Pi has:
           sudo apt install -y build-essential python3-dev

       To avoid that, build wheels yourself on any aarch64 Linux box and
       drop them into res/wheels/, e.g.:
           ssh pi@some-pi 'pip wheel netifaces -w /tmp/wh && ls /tmp/wh'
           scp pi@some-pi:/tmp/wh/netifaces-*.whl res/wheels/
       Then delete the sdist tarball from res/wheels/ so the wheel wins.
EOF
    fi

    echo "[ok] hailort deps merged into res/wheels/"
fi

echo
echo "================================================================"
echo "uv binary:     $(test -x res/bin/uv && echo present || echo MISSING)"
echo "Common wheels: $(ls res/wheels/*.whl 2>/dev/null | wc -l) files"
echo "Hailo wheels:  $(ls res/wheels-hailo/*.whl 2>/dev/null | wc -l) files"
echo "================================================================"

if [ "$(ls res/wheels-hailo/*.whl 2>/dev/null | wc -l)" -eq 0 ]; then
    cat <<'EOF'

WARNING: res/wheels-hailo/ is empty.

The hailort wheel is not on PyPI; download it from Hailo's developer portal
or your purchase channel and drop it into res/wheels-hailo/. Filename should
look like:

    hailort-4.20.0-cp311-cp311-linux_aarch64.whl

Workers without a Hailo NPU will skip this directory entirely. Workers with
a Hailo NPU will fetch and install everything in res/wheels-hailo/.
EOF
fi
