"""
INA226 power monitor.

Scans I2C addresses 0x40..0x4F for INA226 chips, polls each at the configured
interval (default 100 ms), and pushes samples to SQLite via Database. Designed
to run on a dedicated daemon thread to avoid jitter from the asyncio loop.

Each chip is mapped to a worker_id via ClusterState.power_bindings, which is
populated by CalibrationManager. Until calibration runs, samples are persisted
with worker_id=None so the live chart still works but they aren't attributed.

Key formulas (per the report and ina226_test.py):
    Current_LSB = max_expected_current / 2^15  (10 A / 32768)
    CAL = 0.00512 / (Current_LSB * R_shunt)
    I_real = V_shunt / R_shunt   (recommended; bypasses unstable INA226 VBUS)
    P_real = V_actual * I_real   (V_actual is multimeter-measured rail voltage)
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

from controller.cluster_state import ClusterState
from controller.database import Database
from shared.models import PowerSample

logger = logging.getLogger(__name__)


# =============================================================================
# Single-chip driver
# =============================================================================
class INA226Driver:
    CONFIG_REG = 0x00
    SHUNT_VOLTAGE_REG = 0x01
    BUS_VOLTAGE_REG = 0x02
    POWER_REG = 0x03
    CURRENT_REG = 0x04
    CALIBRATION_REG = 0x05
    MANUFACTURER_ID_REG = 0xFE
    DIE_ID_REG = 0xFF

    CONVERSION_TIMES_MS = [0.140, 0.204, 0.332, 0.588, 1.1, 2.116, 4.156, 8.244]
    AVERAGING_COUNTS = [1, 4, 16, 64, 128, 256, 512, 1024]

    def __init__(self, bus, address: int, shunt_resistance: float, actual_vbus: float,
                 conversion_time: int, averaging: int):
        self.bus = bus
        self.address = address
        self.shunt_resistance = shunt_resistance
        self.actual_vbus = actual_vbus
        self.conversion_time = conversion_time
        self.averaging = averaging

        self.current_lsb = 10.0 / (2 ** 15)
        self.calibration_value = int(0.00512 / (self.current_lsb * shunt_resistance))
        self.power_lsb = 25 * self.current_lsb

        self._initialize()

    def _write_register(self, register: int, value: int) -> None:
        data = [(value >> 8) & 0xFF, value & 0xFF]
        self.bus.write_i2c_block_data(self.address, register, data)

    def _read_register(self, register: int) -> int:
        data = self.bus.read_i2c_block_data(self.address, register, 2)
        return (data[0] << 8) | data[1]

    @staticmethod
    def _to_signed(value: int, bits: int = 16) -> int:
        if value & (1 << (bits - 1)):
            return value - (1 << bits)
        return value

    def _initialize(self) -> None:
        # Config: AVG[2:0] | VBUSCT[2:0] | VSHCT[2:0] | MODE[2:0]=111 (continuous)
        config = (
            (self.averaging << 9)
            | (self.conversion_time << 6)
            | (self.conversion_time << 3)
            | 0x07
        )
        self._write_register(self.CONFIG_REG, config)
        time.sleep(0.01)
        self._write_register(self.CALIBRATION_REG, self.calibration_value)
        time.sleep(0.01)

    def read_sample(self) -> tuple[float, float, float, float]:
        """Return (shunt_mv, current_a, voltage_v, power_w)."""
        raw_shunt = self._to_signed(self._read_register(self.SHUNT_VOLTAGE_REG))
        v_shunt = raw_shunt * 2.5e-6                    # 2.5 µV per LSB
        current = v_shunt / self.shunt_resistance       # A
        # We use the multimeter-measured rail voltage rather than INA226's VBUS,
        # which the project report flags as unstable on the test rig.
        voltage = self.actual_vbus
        power = voltage * current
        return v_shunt * 1000.0, current, voltage, power


# =============================================================================
# Threaded poller
# =============================================================================
class PowerMonitor:
    def __init__(self, config: dict[str, Any], db: Database, state: ClusterState):
        self.config = config["power_monitor"]
        self.db = db
        self.state = state
        self.poll_interval_s = float(self.config["poll_interval_ms"]) / 1000.0

        self._bus = None
        self._drivers: dict[int, INA226Driver] = {}    # i2c_address -> driver
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._batch: list[PowerSample] = []
        self._batch_lock = threading.Lock()

    # =========================================================================
    # Public API
    # =========================================================================
    def start(self) -> None:
        if not self._open_bus():
            logger.warning("PowerMonitor: I2C bus not available, disabling.")
            return
        self._scan_chips()
        if not self._drivers:
            logger.warning("PowerMonitor: no INA226 chips detected.")
            return
        self._thread = threading.Thread(
            target=self._run, name="power-monitor", daemon=True
        )
        self._thread.start()
        logger.info("PowerMonitor started with %d chip(s) at addresses %s",
                    len(self._drivers),
                    [hex(a) for a in sorted(self._drivers.keys())])

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._flush_batch(force=True)
        if self._bus is not None:
            try:
                self._bus.close()
            except Exception:
                pass

    def chip_addresses(self) -> list[int]:
        return sorted(self._drivers.keys())

    # =========================================================================
    # I2C setup
    # =========================================================================
    def _open_bus(self) -> bool:
        try:
            import smbus2
            self._bus = smbus2.SMBus(int(self.config["i2c_bus"]))
            return True
        except (ImportError, FileNotFoundError, PermissionError, OSError) as e:
            logger.error("Could not open I2C bus %s: %s", self.config["i2c_bus"], e)
            return False

    def _scan_chips(self) -> None:
        start = int(self.config["i2c_address_start"])
        end = int(self.config["i2c_address_end"])
        for addr in range(start, end + 1):
            try:
                # Probe by reading the manufacturer ID register
                data = self._bus.read_i2c_block_data(addr, INA226Driver.MANUFACTURER_ID_REG, 2)
                mfg_id = (data[0] << 8) | data[1]
                # INA226 reports 0x5449 ("TI") for manufacturer
                if mfg_id != 0x5449:
                    logger.debug("Address 0x%02X responded but mfg_id=0x%04X (not INA226)",
                                 addr, mfg_id)
                    continue
                driver = INA226Driver(
                    bus=self._bus,
                    address=addr,
                    shunt_resistance=float(self.config["shunt_resistance"]),
                    actual_vbus=float(self.config["actual_vbus"]),
                    conversion_time=int(self.config["conversion_time"]),
                    averaging=int(self.config["averaging"]),
                )
                self._drivers[addr] = driver
                logger.info("INA226 detected at 0x%02X", addr)
            except OSError:
                # No device at this address — quietly skip
                continue
            except Exception as e:
                logger.warning("Probe at 0x%02X failed: %s", addr, e)

    # =========================================================================
    # Polling loop
    # =========================================================================
    def _run(self) -> None:
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            ts_ms = int(time.time() * 1000)
            for addr, driver in self._drivers.items():
                try:
                    shunt_mv, current_a, voltage_v, power_w = driver.read_sample()
                except Exception as e:
                    logger.debug("read at 0x%02X failed: %s", addr, e)
                    continue
                # Bindings come from CalibrationManager; before calibration
                # runs, samples are persisted with worker_id=None.
                worker_id = self.state.get_worker_for_chip(addr)
                sample = PowerSample(
                    timestamp_ms=ts_ms,
                    i2c_address=addr,
                    worker_id=worker_id,
                    shunt_mv=shunt_mv,
                    current_a=current_a,
                    voltage_v=voltage_v,
                    power_w=power_w,
                )
                with self._batch_lock:
                    self._batch.append(sample)

            self._flush_batch()
            next_tick += self.poll_interval_s
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                self._stop_event.wait(timeout=sleep_for)
            else:
                # Lagging behind — reset the schedule so we don't spiral.
                next_tick = time.monotonic()

    def _flush_batch(self, force: bool = False, max_pending: int = 50) -> None:
        with self._batch_lock:
            if not self._batch:
                return
            if not force and len(self._batch) < max_pending:
                return
            to_write = self._batch
            self._batch = []
        try:
            self.db.insert_power_samples(to_write)
        except Exception as e:
            logger.error("Failed to flush power samples: %s", e)
            with self._batch_lock:
                # Push back so we don't lose them on the next attempt
                self._batch[:0] = to_write
