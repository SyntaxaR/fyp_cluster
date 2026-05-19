"""
SQLite persistence for power samples and experiment results.

Schema (created if not present):
    CREATE TABLE power_samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_ms INTEGER NOT NULL,
        i2c_address INTEGER NOT NULL,
        worker_id INTEGER,
        shunt_mv REAL, current_a REAL, voltage_v REAL, power_w REAL
    );
    CREATE INDEX idx_power_samples_ts ON power_samples(ts_ms);

    CREATE TABLE experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, model TEXT, dispatcher TEXT, mode TEXT,
        started_ms INTEGER, finished_ms INTEGER, status TEXT,
        total_requests INTEGER, successful INTEGER, failed INTEGER,
        avg_latency_ms REAL, p95_latency_ms REAL, p99_latency_ms REAL,
        avg_throughput_qps REAL, avg_cluster_power_w REAL,
        energy_per_request_j REAL, notes TEXT
    );
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Optional

from shared.models import ExperimentResult, ExperimentStatus, PowerSample

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: str):
        self.path = Path(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS power_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_ms      INTEGER NOT NULL,
                    i2c_address INTEGER NOT NULL,
                    worker_id  INTEGER,
                    shunt_mv   REAL,
                    current_a  REAL,
                    voltage_v  REAL,
                    power_w    REAL
                );
                CREATE INDEX IF NOT EXISTS idx_power_samples_ts ON power_samples(ts_ms);
                CREATE INDEX IF NOT EXISTS idx_power_samples_addr ON power_samples(i2c_address);

                CREATE TABLE IF NOT EXISTS worker_bindings (
                    serial         TEXT PRIMARY KEY,
                    i2c_address    INTEGER NOT NULL,
                    identifier     TEXT,
                    delta_w        REAL,
                    calibrated_ms  INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    model TEXT,
                    dispatcher TEXT,
                    mode TEXT,
                    started_ms INTEGER,
                    finished_ms INTEGER,
                    status TEXT,
                    total_requests INTEGER,
                    successful INTEGER,
                    failed INTEGER,
                    avg_latency_ms REAL,
                    p95_latency_ms REAL,
                    p99_latency_ms REAL,
                    avg_throughput_qps REAL,
                    avg_cluster_power_w REAL,
                    energy_per_request_j REAL,
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS experiment_workers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id  INTEGER NOT NULL,
                    worker_id      INTEGER NOT NULL,
                    serial         TEXT,
                    identifier     TEXT,
                    engine         TEXT,
                    i2c_address    INTEGER,
                    started_ms     INTEGER,
                    finished_ms    INTEGER,
                    dispatched     INTEGER,
                    succeeded      INTEGER,
                    failed         INTEGER,
                    avg_latency_ms REAL,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                );
                CREATE INDEX IF NOT EXISTS idx_experiment_workers_exp
                    ON experiment_workers(experiment_id);
                """
            )

    # =========================================================================
    # Power
    # =========================================================================
    def insert_power_samples(self, samples: Iterable[PowerSample]) -> None:
        rows = [
            (s.timestamp_ms, s.i2c_address, s.worker_id,
             s.shunt_mv, s.current_a, s.voltage_v, s.power_w)
            for s in samples
        ]
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT INTO power_samples "
                "(ts_ms, i2c_address, worker_id, shunt_mv, current_a, voltage_v, power_w) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def power_samples_in_range(
        self, start_ms: int, end_ms: int, address: Optional[int] = None
    ) -> list[PowerSample]:
        with self._lock:
            cur = self._conn.cursor()
            if address is None:
                cur.execute(
                    "SELECT ts_ms, i2c_address, worker_id, shunt_mv, current_a, voltage_v, power_w "
                    "FROM power_samples WHERE ts_ms BETWEEN ? AND ? ORDER BY ts_ms",
                    (start_ms, end_ms),
                )
            else:
                cur.execute(
                    "SELECT ts_ms, i2c_address, worker_id, shunt_mv, current_a, voltage_v, power_w "
                    "FROM power_samples WHERE ts_ms BETWEEN ? AND ? AND i2c_address = ? "
                    "ORDER BY ts_ms",
                    (start_ms, end_ms, address),
                )
            rows = cur.fetchall()
        return [
            PowerSample(
                timestamp_ms=r[0], i2c_address=r[1], worker_id=r[2],
                shunt_mv=r[3], current_a=r[4], voltage_v=r[5], power_w=r[6],
            )
            for r in rows
        ]

    def average_power_in_range(self, start_ms: int, end_ms: int) -> float:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT AVG(power_w) FROM power_samples WHERE ts_ms BETWEEN ? AND ?",
                (start_ms, end_ms),
            )
            row = cur.fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0

    # =========================================================================
    # Experiments
    # =========================================================================
    def insert_experiment(self, result: ExperimentResult, model: str,
                          dispatcher: str, mode: str, notes: str = "") -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO experiments "
                "(name, model, dispatcher, mode, started_ms, finished_ms, status, "
                " total_requests, successful, failed, avg_latency_ms, p95_latency_ms, "
                " p99_latency_ms, avg_throughput_qps, avg_cluster_power_w, "
                " energy_per_request_j, notes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result.name, model, dispatcher, mode,
                    result.started_ms, result.finished_ms,
                    result.status.value if isinstance(result.status, ExperimentStatus) else str(result.status),
                    result.total_requests, result.successful_requests, result.failed_requests,
                    result.avg_latency_ms, result.p95_latency_ms, result.p99_latency_ms,
                    result.avg_throughput_qps, result.avg_cluster_power_w,
                    result.energy_per_request_j, notes,
                ),
            )
            return int(cur.lastrowid or 0)

    def list_experiments(self, limit: int = 50) -> list[dict]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT id, name, model, dispatcher, mode, started_ms, finished_ms, "
                "status, total_requests, successful, failed, avg_latency_ms, "
                "p95_latency_ms, p99_latency_ms, avg_throughput_qps, "
                "avg_cluster_power_w, energy_per_request_j, notes "
                "FROM experiments ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_experiment(self, experiment_id: int) -> Optional[dict]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT id, name, model, dispatcher, mode, started_ms, finished_ms, "
                "status, total_requests, successful, failed, avg_latency_ms, "
                "p95_latency_ms, p99_latency_ms, avg_throughput_qps, "
                "avg_cluster_power_w, energy_per_request_j, notes "
                "FROM experiments WHERE id = ?",
                (experiment_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            cols = [c[0] for c in cur.description]
            return dict(zip(cols, row))

    def insert_experiment_workers(self, rows: Iterable[dict]) -> None:
        rows = list(rows)
        if not rows:
            return
        tuples = [
            (
                r["experiment_id"], r["worker_id"], r.get("serial", ""),
                r.get("identifier", ""), r.get("engine", ""),
                r.get("i2c_address"),
                int(r.get("started_ms", 0)), int(r.get("finished_ms", 0)),
                int(r.get("dispatched", 0)), int(r.get("succeeded", 0)),
                int(r.get("failed", 0)),
                float(r.get("avg_latency_ms", 0.0)),
            )
            for r in rows
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT INTO experiment_workers "
                "(experiment_id, worker_id, serial, identifier, engine, "
                " i2c_address, started_ms, finished_ms, dispatched, "
                " succeeded, failed, avg_latency_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                tuples,
            )

    def list_experiment_workers(self, experiment_id: int) -> list[dict]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT worker_id, serial, identifier, engine, i2c_address, "
                "started_ms, finished_ms, dispatched, succeeded, failed, "
                "avg_latency_ms "
                "FROM experiment_workers WHERE experiment_id = ? "
                "ORDER BY worker_id",
                (experiment_id,),
            )
            cols = [c[0] for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    # =========================================================================
    # Worker bindings (worker_serial -> INA226 i2c_address, set by calibration)
    # =========================================================================
    def upsert_binding(self, serial: str, i2c_address: int, identifier: str,
                       delta_w: float, calibrated_ms: int) -> None:
        """Insert or replace a worker -> chip binding."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO worker_bindings (serial, i2c_address, identifier, "
                "delta_w, calibrated_ms) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(serial) DO UPDATE SET "
                "  i2c_address=excluded.i2c_address, "
                "  identifier=excluded.identifier, "
                "  delta_w=excluded.delta_w, "
                "  calibrated_ms=excluded.calibrated_ms",
                (serial, i2c_address, identifier, delta_w, calibrated_ms),
            )

    def load_bindings(self) -> dict[str, dict]:
        """Return {serial: {i2c_address, identifier, delta_w, calibrated_ms}}."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT serial, i2c_address, identifier, delta_w, calibrated_ms "
                "FROM worker_bindings"
            )
            rows = cur.fetchall()
        return {
            r[0]: {
                "i2c_address": int(r[1]),
                "identifier": r[2] or "",
                "delta_w": float(r[3]) if r[3] is not None else 0.0,
                "calibrated_ms": int(r[4]),
            }
            for r in rows
        }

    def clear_bindings(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM worker_bindings")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

