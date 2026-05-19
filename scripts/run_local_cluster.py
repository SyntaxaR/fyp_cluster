"""
Spawn 1 controller + N workers locally, all in mock mode, for end-to-end debug.

Usage:
    python scripts/run_local_cluster.py            # 3 workers
    python scripts/run_local_cluster.py --workers 5
    python scripts/run_local_cluster.py --workers 4 --base-port 9000

Each worker runs in its own subprocess with:
    FYP_MOCK=1
    FYP_CPU_SERIAL=mock-worker-<i>      → distinct identifier
    FYP_WORKER_CONTROL_PORT=<port>      → unique control port
    FYP_WORKER_DATA_PORT=<port+1>       → unique data port
    FYP_WORKER_LOG_FILE=worker_<i>.log
    FYP_CONTROLLER_HOST=127.0.0.1

The controller listens on its config'd ports (default 8001/8002/8080) and the
NiceGUI dashboard is reachable at http://127.0.0.1:8080/.

Press Ctrl-C to stop everything.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _stream_prefixed(proc: subprocess.Popen, prefix: str) -> None:
    """Forward subprocess stdout/stderr to ours with a [prefix] tag."""
    def _pump(pipe, stream):
        try:
            for raw in iter(pipe.readline, b""):
                if not raw:
                    break
                line = raw.decode(errors="replace").rstrip()
                stream.write(f"[{prefix}] {line}\n")
                stream.flush()
        except Exception:
            pass
    threading.Thread(target=_pump, args=(proc.stdout, sys.stdout),
                     daemon=True).start()
    threading.Thread(target=_pump, args=(proc.stderr, sys.stderr),
                     daemon=True).start()


def _spawn(name: str, module: str, env: dict[str, str]) -> subprocess.Popen:
    cmd = [sys.executable, "-m", module]
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        # Don't share stdin so Ctrl-C reaches the parent only.
        stdin=subprocess.DEVNULL,
    )
    _stream_prefixed(proc, name)
    return proc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=3,
                        help="Number of mock worker processes (default: 3).")
    parser.add_argument("--base-port", type=int, default=9001,
                        help="First worker control port; data port is +1, "
                             "next worker is +10. Default 9001.")
    parser.add_argument("--controller-only", action="store_true",
                        help="Just run the controller (no workers).")
    parser.add_argument("--no-controller", action="store_true",
                        help="Run only the workers (controller assumed running).")
    args = parser.parse_args()

    base_env = os.environ.copy()
    # Make sure the project root is importable inside the subprocesses.
    base_env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}{os.pathsep}{base_env.get('PYTHONPATH', '')}"
    )
    base_env["FYP_MOCK"] = "1"
    base_env["FYP_CONTROLLER_HOST"] = "127.0.0.1"

    procs: list[tuple[str, subprocess.Popen]] = []

    # Controller
    if not args.no_controller:
        env = dict(base_env)
        env["FYP_CPU_SERIAL"] = "mock-controller"
        print(f"[runner] starting controller (mock) at "
              f"http://127.0.0.1:8080/")
        ctrl = _spawn("controller", "controller.controller", env)
        procs.append(("controller", ctrl))
        # Give it a head start so workers' first heartbeat lands on a live port.
        time.sleep(2.0)

    # Workers
    if not args.controller_only:
        for i in range(args.workers):
            env = dict(base_env)
            ctrl_port = args.base_port + i * 10
            data_port = ctrl_port + 1
            env["FYP_CPU_SERIAL"] = f"mock-worker-{i}"
            env["FYP_WORKER_CONTROL_PORT"] = str(ctrl_port)
            env["FYP_WORKER_DATA_PORT"] = str(data_port)
            env["FYP_WORKER_LOG_FILE"] = f"worker_{i}.log"
            name = f"worker{i}"
            print(f"[runner] starting {name} (control={ctrl_port}, data={data_port})")
            procs.append((name, _spawn(name, "worker.worker", env)))
            time.sleep(0.4)

    # Wait. Forward Ctrl-C to children.
    stopping = False

    def _stop(*_):
        nonlocal stopping
        if stopping:
            return
        stopping = True
        print("\n[runner] stopping all processes...")
        for _, p in procs:
            try:
                p.send_signal(signal.SIGINT)
            except Exception:
                pass

    signal.signal(signal.SIGINT, _stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _stop)

    try:
        # Block until any child dies, then stop the rest. This makes
        # crashing-on-startup show up immediately.
        while not stopping:
            for name, p in procs:
                rc = p.poll()
                if rc is not None:
                    print(f"[runner] {name} exited (rc={rc}); shutting down others")
                    _stop()
                    break
            time.sleep(0.3)
    finally:
        deadline = time.monotonic() + 5.0
        for name, p in procs:
            timeout = max(0.1, deadline - time.monotonic())
            try:
                p.wait(timeout=timeout)
            except Exception:
                p.kill()


if __name__ == "__main__":
    main()
