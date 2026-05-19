#!/bin/bash
# Controller startup script for FYP Cluster.
#
# By default the controller brings up the WiFi AP on wlan0 so the experiment
# PC can join the cluster WiFi without an external router. To skip that, pass
# `--no-ap` (alias `--no-wifi`):
#     ./controller.sh --no-ap
# Or join an existing WiFi network instead:
#     ./controller.sh --client "MyHomeSSID" "MyPassword"
set -e

# Resolve script directory so the script works regardless of cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# --- uv check ---
if ! command -v uv &> /dev/null; then
    echo "[controller.sh] uv is not installed. Install? (y/n)"
    read -r install_uv
    if [ "$install_uv" = "y" ]; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # shellcheck disable=SC1091
        source "$HOME/.local/bin/env"
        if ! command -v uv &> /dev/null; then
            echo "[controller.sh] uv installation failed."
            exit 1
        fi
        echo "[controller.sh] uv installed successfully."
    else
        echo "[controller.sh] uv installation skipped, aborting."
        exit 1
    fi
fi

# --- system tools the controller relies on ---
for bin in dnsmasq hostapd nmcli i2cdetect; do
    if ! command -v "$bin" &> /dev/null; then
        echo "[controller.sh] WARNING: '$bin' not found. Some controller features will not work."
    fi
done

# Offline-friendly run flags.
#
# `uv run` would normally re-check dependency state against PyPI on every
# invocation — fine when there's a lockfile (which we don't ship) or when
# the cluster has internet (which it doesn't, by design). Without those,
# uv tries to refresh metadata, fails to fetch hatchling to build the
# local fyp-cluster project, and bails out with "failed to build
# fyp-cluster" even though the venv is fully populated from your last
# successful run.
#
# Two flags fix this:
#   --no-sync     : trust the existing .venv, skip the reconciliation step
#                   that triggers PyPI lookups
#   UV_OFFLINE=1  : refuse all network operations, so a misconfigured
#                   cache or a stray import resolve can't fall back to
#                   "let me just check the index real quick" and hang
#
# To force a refresh (e.g. after editing pyproject.toml), run online with
# FYP_FORCE_SYNC=1 ./controller.sh — that drops both flags so uv resolves
# normally.
UV_RUN_FLAGS="--no-sync"
if [ "$FYP_FORCE_SYNC" != "1" ]; then
    export UV_OFFLINE=1
else
    UV_RUN_FLAGS=""
    echo "[controller.sh] FYP_FORCE_SYNC=1 — running uv in online mode."
fi

# Detect whether we're already running as root (e.g. under systemd).
# Interactive use needs sudo to gain dnsmasq/hostapd/I2C privileges; systemd
# launches us as root directly, so re-invoking sudo there would block on a
# non-existent tty. Honour FYP_NO_SUDO=1 as an explicit override too.
if [ "$(id -u)" -eq 0 ] || [ "$FYP_NO_SUDO" = "1" ]; then
    echo "[controller.sh] Already root (or FYP_NO_SUDO=1) — running directly."
    exec env "PATH=$PATH" "PYTHONPATH=$PYTHONPATH" "UV_OFFLINE=$UV_OFFLINE" \
        uv run $UV_RUN_FLAGS python -m controller.controller "$@"
else
    echo "[controller.sh] Starting controller (sudo required for dnsmasq/hostapd)..."
    exec sudo env "PATH=$PATH" "PYTHONPATH=$PYTHONPATH" "UV_OFFLINE=$UV_OFFLINE" \
        uv run $UV_RUN_FLAGS python -m controller.controller "$@"
fi
