# telemetry/ac_shared_memory.py
import ctypes as ct
import logging
import mmap
import time
from typing import List, Dict, Any, Optional

from PyQt5 import QtCore

from telemetry.lap_buffer import LapBuffer

logger = logging.getLogger(__name__)


def _sanitize_ac_time_str(raw: str) -> str:
    """Clean AC shared-memory time strings.

    AC stores lap times as c_wchar[15]. Before a lap is completed the
    buffer may contain uninitialised memory, which surfaces as random
    Unicode (often CJK) characters.  Strip anything that isn't a digit,
    colon, period, or minus sign.
    """
    cleaned = "".join(ch for ch in raw if ch in "0123456789:.-")
    return cleaned


# ===================== PHYSICS SHARED MEMORY =====================

class SPageFilePhysics(ct.Structure):
    _fields_ = [
        ("packetId", ct.c_int),
        ("gas", ct.c_float),
        ("brake", ct.c_float),
        ("fuel", ct.c_float),
        ("gear", ct.c_int),
        ("rpms", ct.c_int),
        ("steerAngle", ct.c_float),
        ("speedKmh", ct.c_float),
        ("velocity", ct.c_float * 3),
        ("accG", ct.c_float * 3),
        ("wheelSlip", ct.c_float * 4),
        ("wheelLoad", ct.c_float * 4),
        ("wheelsPressure", ct.c_float * 4),
        ("wheelAngularSpeed", ct.c_float * 4),
        ("tyreWear", ct.c_float * 4),
        ("tyreDirtyLevel", ct.c_float * 4),
        ("tyreCoreTemperature", ct.c_float * 4),
        ("camberRAD", ct.c_float * 4),
        ("suspensionTravel", ct.c_float * 4),
        ("drs", ct.c_float),
        ("tc", ct.c_float),
        ("heading", ct.c_float),
        ("pitch", ct.c_float),
        ("roll", ct.c_float),
        ("cgHeight", ct.c_float),
        ("carDamage", ct.c_float * 5),
        ("numberOfTyresOut", ct.c_int),
        ("pitLimiterOn", ct.c_int),
        ("abs", ct.c_float),
        ("kersCharge", ct.c_float),
        ("kersInput", ct.c_float),
        ("autoShifterOn", ct.c_int),
        ("rideHeight", ct.c_float * 2),
        ("turboBoost", ct.c_float),
        ("ballast", ct.c_float),
        ("airDensity", ct.c_float),
    ]


# ===================== GRAPHICS SHARED MEMORY =====================

class SPageFileGraphics(ct.Structure):
    # AC uses wchar_t (2 bytes on Windows) for all string fields.
    # Using c_wchar ensures correct field offsets for completedLaps, carCoordinates, etc.
    _fields_ = [
        ("packetId", ct.c_int),
        ("status", ct.c_int),
        ("session", ct.c_int),
        ("currentTime", ct.c_wchar * 15),
        ("lastTime", ct.c_wchar * 15),
        ("bestTime", ct.c_wchar * 15),
        ("splitTime", ct.c_wchar * 15),
        ("completedLaps", ct.c_int),
        ("position", ct.c_int),
        ("currentTimeMs", ct.c_int),
        ("lastTimeMs", ct.c_int),
        ("bestTimeMs", ct.c_int),
        ("sessionTimeLeft", ct.c_float),
        ("distanceTraveled", ct.c_float),
        ("isInPit", ct.c_int),
        ("currentSectorIndex", ct.c_int),
        ("lastSectorTime", ct.c_int),
        ("numberOfLaps", ct.c_int),
        ("tyreCompound", ct.c_wchar * 33),
        ("replayTimeMultiplier", ct.c_float),
        ("normalizedCarPosition", ct.c_float),
        ("carCoordinates", ct.c_float * 3),
    ]


# ===================== STATIC INFO SHARED MEMORY =====================

class SPageFileStatic(ct.Structure):
    # AC uses wchar_t (2 bytes on Windows) for all string fields.
    _fields_ = [
        ("_smVersion", ct.c_wchar * 15),
        ("_acVersion", ct.c_wchar * 15),
        ("numberOfSessions", ct.c_int),
        ("numCars", ct.c_int),
        ("carModel", ct.c_wchar * 33),
        ("track", ct.c_wchar * 33),
        ("playerName", ct.c_wchar * 33),
        ("playerSurname", ct.c_wchar * 33),
        ("playerNick", ct.c_wchar * 33),
        ("sectorCount", ct.c_int),
        ("maxTorque", ct.c_float),
        ("maxPower", ct.c_float),
        ("maxRpm", ct.c_int),
        ("maxFuel", ct.c_float),
        ("suspensionMaxTravel", ct.c_float * 4),
        ("tyreRadius", ct.c_float * 4),
        ("maxTurboBoost", ct.c_float),
        ("deprecated_1", ct.c_float),
        ("deprecated_2", ct.c_float),
        ("penaltiesEnabled", ct.c_int),
        ("aidFuelRate", ct.c_float),
        ("aidTireRate", ct.c_float),
        ("aidMechanicalDamage", ct.c_float),
        ("aidAllowTyreBlankets", ct.c_int),
        ("aidStability", ct.c_float),
        ("aidAutoClutch", ct.c_int),
        ("aidAutoBlip", ct.c_int),
        ("hasDRS", ct.c_int),
        ("hasERS", ct.c_int),
        ("hasKERS", ct.c_int),
        ("kersMaxJ", ct.c_float),
        ("engineBrakeSettingsCount", ct.c_int),
        ("ersPowerControllerCount", ct.c_int),
        ("trackSPlineLength", ct.c_float),
        ("trackConfiguration", ct.c_wchar * 33),
        ("ersMaxJ", ct.c_float),
        ("isTimedRace", ct.c_int),
        ("hasExtraLap", ct.c_int),
        ("carSkin", ct.c_wchar * 33),
        ("reversedGridPositions", ct.c_int),
        ("pitWindowStart", ct.c_int),
        ("pitWindowEnd", ct.c_int),
        ("isOnline", ct.c_int),
        ("dryTyresName", ct.c_wchar * 33),
        ("wetTyresName", ct.c_wchar * 33),
    ]


SHM_NAME_PHYSICS = "acpmf_physics"
SHM_NAME_GRAPHICS = "acpmf_graphics"
SHM_NAME_STATIC = "acpmf_static"
PHYSICS_SIZE = ct.sizeof(SPageFilePhysics)
GRAPHICS_SIZE = ct.sizeof(SPageFileGraphics)
STATIC_SIZE = ct.sizeof(SPageFileStatic)


# ===================== SHARED MEMORY HELPERS =====================

def open_shared_memory(name: str, size: int) -> Optional[mmap.mmap]:
    """Open an existing named shared memory region created by Assetto Corsa."""
    try:
        return mmap.mmap(0, size, tagname=name, access=mmap.ACCESS_READ)
    except Exception as e:
        logger.error("Could not open shared memory '%s': %s", name, e)
        logger.error("Make sure Assetto Corsa is running and you're in a session.")
        return None


def read_physics(mm: mmap.mmap) -> SPageFilePhysics:
    mm.seek(0)
    raw = mm.read(PHYSICS_SIZE)
    return SPageFilePhysics.from_buffer_copy(raw)


def read_graphics(mm: mmap.mmap) -> SPageFileGraphics:
    mm.seek(0)
    raw = mm.read(GRAPHICS_SIZE)
    return SPageFileGraphics.from_buffer_copy(raw)


def read_static(mm: mmap.mmap) -> SPageFileStatic:
    mm.seek(0)
    raw = mm.read(STATIC_SIZE)
    return SPageFileStatic.from_buffer_copy(raw)


# ===================== TELEMETRY THREAD =====================

class AcTelemetryWorker(QtCore.QThread):
    """
    Background thread that reads Assetto Corsa telemetry from shared memory
    and emits normalized packets to the GUI.
    """
    lap_completed = QtCore.pyqtSignal(int, list, bool, int)  # (lap_id, samples, valid, last_time_ms)
    status_update = QtCore.pyqtSignal(str)        # status message
    session_info_update = QtCore.pyqtSignal(dict) # session info (track, car, driver)
    session_reset = QtCore.pyqtSignal()           # emitted when AC restarts (new session detected)
    live_data_update = QtCore.pyqtSignal(dict)    # live telemetry updates
    realtime_sample = QtCore.pyqtSignal(dict)     # realtime telemetry sample (every frame)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self._last_session_key = None  # (track, car) tuple to detect AC restart

    def _close_handles(self, mm_phys, mm_graph, mm_static):
        """Safely close all shared memory handles."""
        for mm in (mm_phys, mm_graph, mm_static):
            if mm is not None:
                try:
                    mm.close()
                except Exception:
                    pass

    def _connect(self):
        """Try to open all shared memory handles. Returns (phys, graph, static) or None."""
        mm_phys = open_shared_memory(SHM_NAME_PHYSICS, PHYSICS_SIZE)
        mm_graph = open_shared_memory(SHM_NAME_GRAPHICS, GRAPHICS_SIZE)
        mm_static = open_shared_memory(SHM_NAME_STATIC, STATIC_SIZE)
        if mm_phys is None or mm_graph is None:
            self._close_handles(mm_phys, mm_graph, mm_static)
            return None
        return mm_phys, mm_graph, mm_static

    # AC session type codes
    AC_SESSION_TYPES = {0: "Practice", 1: "Qualify", 2: "Race", 3: "Hotlap"}

    def _read_session_info(self, mm_static, mm_graph=None):
        """Read and emit static session info (track, car, driver, mode).

        Also detects AC session restarts by comparing against the previous
        session identity. Emits session_reset before session_info_update
        when a new AC session is detected.
        """
        if mm_static is None:
            return
        try:
            static_data = read_static(mm_static)

            # Read session type from graphics shared memory
            session_type = ""
            if mm_graph is not None:
                try:
                    gfx = read_graphics(mm_graph)
                    session_type = self.AC_SESSION_TYPES.get(gfx.session, f"Unknown ({gfx.session})")
                except Exception:
                    pass

            session_key = (static_data.track, static_data.carModel, session_type)

            # Detect AC restart: if we had a previous session and the key
            # matches or differs, either way AC was closed and reopened
            if self._last_session_key is not None:
                logger.info("AC reconnected — new session detected, resetting")
                self.session_reset.emit()

            self._last_session_key = session_key

            session_data = {
                "track": static_data.track,
                "track_config": static_data.trackConfiguration,
                "car_model": static_data.carModel,
                "player_name": static_data.playerName,
                "player_surname": static_data.playerSurname,
                "player_nick": static_data.playerNick,
                "max_rpm": static_data.maxRpm,
                "max_fuel": static_data.maxFuel,
                "session_type": session_type,
            }
            logger.info("Session: Track=%s (%s), Car=%s, Driver=%s %s, Mode=%s",
                        session_data['track'], session_data['track_config'],
                        session_data['car_model'], session_data['player_name'],
                        session_data['player_surname'], session_type)
            self.session_info_update.emit(session_data)
        except Exception as e:
            logger.warning("Could not read static info: %s", e)

    def run(self):
        logger.info("AC Telemetry Worker starting")
        self.running = True

        RECONNECT_INTERVAL = 3  # seconds between reconnection attempts

        # Outer reconnection loop — keeps trying until stopped
        while self.running:
            self.status_update.emit("Connecting to Assetto Corsa...")

            handles = self._connect()
            if handles is None:
                logger.warning("AC not available, retrying in %ds...", RECONNECT_INTERVAL)
                self.status_update.emit("Waiting for Assetto Corsa...")
                # Wait with periodic checks so we can stop quickly
                for _ in range(RECONNECT_INTERVAL * 10):
                    if not self.running:
                        return
                    time.sleep(0.1)
                continue

            mm_phys, mm_graph, mm_static = handles

            # Verify AC is actually alive — stale shared memory from a
            # previous session can linger after the game exits.  If status
            # is OFF (0) for several consecutive reads, release the handles
            # so we don't block AC from reinitializing them on next launch.
            try:
                gfx_check = read_graphics(mm_graph)
                if gfx_check.status == 0:
                    logger.info("Shared memory exists but AC status=OFF — stale region, releasing")
                    self.status_update.emit("Waiting for Assetto Corsa...")
                    self._close_handles(mm_phys, mm_graph, mm_static)
                    for _ in range(RECONNECT_INTERVAL * 10):
                        if not self.running:
                            return
                        time.sleep(0.1)
                    continue
            except Exception:
                self._close_handles(mm_phys, mm_graph, mm_static)
                continue

            logger.info("Connected to AC shared memory")
            self.status_update.emit("Connected! Start driving...")

            self._read_session_info(mm_static, mm_graph)

            # LapBuffer with callback that emits a Qt signal
            completion_meta = {"valid": True, "last_time_ms": 0}
            lap_buffer = LapBuffer(
                on_lap_complete=lambda lap_id, samples: self.lap_completed.emit(
                    lap_id,
                    samples,
                    bool(completion_meta["valid"]),
                    int(completion_meta["last_time_ms"]),
                )
            )

            t0 = time.time()
            frame_count = 0
            last_lap_id = -1
            last_valid_raw_lap = None
            stable_count = 0       # frames with a consistent completedLaps value
            STABLE_THRESHOLD = 10  # require 10 consistent frames before trusting
            off_count = 0          # consecutive frames with status=OFF
            OFF_DISCONNECT = 10    # release handles after this many OFF frames
            integrated_x = 0.0
            integrated_z = 0.0
            last_time = t0
            warmup_frames = 0      # frames since entering LIVE status
            WARMUP_THRESHOLD = 60  # skip first ~1 second of data to avoid garbage
            last_pos = None        # track position for settling detection

            logger.info("Starting telemetry loop at ~60Hz")

            try:
                while self.running:
                    gfx = read_graphics(mm_graph)

                    # AC status: 0=OFF, 1=REPLAY, 2=LIVE, 3=PAUSE
                    ac_status = gfx.status
                    if ac_status != 2:
                        frame_count += 1
                        # Reset warmup so we re-settle when returning to LIVE
                        warmup_frames = 0

                        # If AC has been OFF for a while, release handles so
                        # the game can reinitialize shared memory on restart
                        if ac_status == 0:
                            off_count += 1
                            if off_count >= OFF_DISCONNECT:
                                logger.info("AC status=OFF for %d reads, releasing handles", off_count)
                                self.status_update.emit("Waiting for Assetto Corsa...")
                                break  # exit to outer reconnection loop
                        else:
                            off_count = 0

                        if frame_count % 10 == 0:
                            # Keep lap data consistent with the stabilized lap_id used by
                            # realtime samples so late-join UI logic doesn't see raw spikes.
                            completed_laps = (
                                max(0, last_valid_raw_lap)
                                if last_valid_raw_lap is not None
                                else max(0, int(gfx.completedLaps))
                            )
                            self.live_data_update.emit({
                                "current_lap": (last_lap_id + 1) if last_lap_id >= 0 else 1,
                                "speed": 0, "gear": 1, "rpm": 0, "fuel": 0,
                                "position": gfx.position, "is_in_pit": gfx.isInPit,
                                "ac_status": ac_status,
                                "current_time": gfx.currentTime,
                                "last_time": _sanitize_ac_time_str(gfx.lastTime), "best_time": _sanitize_ac_time_str(gfx.bestTime),
                                "last_time_ms": gfx.lastTimeMs, "best_time_ms": gfx.bestTimeMs,
                                "completed_laps": completed_laps,
                            })
                        time.sleep(0.5)
                        continue
                    off_count = 0  # reset when LIVE

                    phys = read_physics(mm_phys)

                    now = time.time()
                    elapsed = now - t0
                    dt = now - last_time
                    last_time = now

                    # Warmup: skip the first ~1 second of LIVE data.
                    # AC shared memory can contain garbage/stale values
                    # immediately after the game enters LIVE status.
                    warmup_frames += 1
                    if warmup_frames <= WARMUP_THRESHOLD:
                        if warmup_frames == 1:
                            logger.info("Warmup: skipping first %d frames to let telemetry settle", WARMUP_THRESHOLD)
                        if warmup_frames == WARMUP_THRESHOLD:
                            logger.info("Warmup complete, starting telemetry capture")
                            # Reset t0 so elapsed time starts from now
                            t0 = now
                        time.sleep(1 / 60.0)
                        continue

                    # Re-read elapsed after warmup reset
                    elapsed = now - t0

                    x = gfx.carCoordinates[0]
                    z = gfx.carCoordinates[2]

                    # Check for obviously garbage position data
                    # (huge values or NaN suggest uninitialized memory)
                    if abs(x) > 100000 or abs(z) > 100000:
                        time.sleep(1 / 60.0)
                        continue

                    if x == 0.0 and z == 0.0 and dt > 0:
                        integrated_x += phys.velocity[0] * dt
                        integrated_z += phys.velocity[2] * dt
                        x = integrated_x
                        z = integrated_z

                    raw_lap_id = gfx.completedLaps

                    # Reject clearly invalid values (negative)
                    if raw_lap_id < 0:
                        raw_lap_id = last_valid_raw_lap if last_valid_raw_lap is not None else 0

                    # Wait for a stable reading before trusting the lap counter.
                    # Shared memory can contain stale/garbage data on first connect.
                    if last_valid_raw_lap is None:
                        # First reading — start counting stability
                        last_valid_raw_lap = raw_lap_id
                        stable_count = 1
                    elif raw_lap_id == last_valid_raw_lap:
                        stable_count = min(stable_count + 1, STABLE_THRESHOLD + 1)
                    elif stable_count >= STABLE_THRESHOLD:
                        # We had a stable baseline and now the value changed —
                        # trust it (this is a real lap completion or reset)
                        logger.info("completedLaps changed: %d -> %d", last_valid_raw_lap, raw_lap_id)
                        last_valid_raw_lap = raw_lap_id
                    else:
                        # Value changed before we had a stable baseline —
                        # restart stability counting with the new value
                        logger.debug("Unstable completedLaps: %d (was %d, stable_count=%d), restarting",
                                     raw_lap_id, last_valid_raw_lap, stable_count)
                        last_valid_raw_lap = raw_lap_id
                        stable_count = 1

                    lap_id = max(0, last_valid_raw_lap)
                    speed = phys.speedKmh

                    # Reject garbage speed values (AC cars don't exceed ~400 km/h)
                    if speed < 0 or speed > 500:
                        speed = 0.0
                    # Deadzone: AC reports tiny speed jitter when stationary
                    elif speed < 1.0:
                        speed = 0.0

                    raw_gear = phys.gear
                    if raw_gear == 0:
                        display_gear = -1
                    elif raw_gear == 1:
                        display_gear = 0
                    else:
                        display_gear = raw_gear - 1

                    clamped_rpms = max(0, phys.rpms)

                    frame_count += 1
                    if frame_count % 60 == 0:
                        logger.debug("Frame %04d | Lap: %d | Speed: %6.1f km/h | Gear: %d | RPM: %5d | Pos: (%.1f, %.1f)",
                                     frame_count, lap_id+1, speed, display_gear, clamped_rpms, x, z)
                        logger.debug("Pressure FL=%.1f FR=%.1f | Temp FL=%.1f FR=%.1f",
                                     phys.wheelsPressure[0], phys.wheelsPressure[1],
                                     phys.tyreCoreTemperature[0], phys.tyreCoreTemperature[1])

                    if frame_count == 300 and x == 0 and z == 0:
                        logger.warning("Car coordinates still (0,0) after 5 seconds — are you on track?")

                    # Detect lap completion and read validity from AC
                    # AC sets lastTimeMs > 0 for valid laps, 0 for invalid
                    lap_valid = gfx.lastTimeMs > 0

                    if lap_id != last_lap_id and last_lap_id != -1:
                        completion_meta["valid"] = lap_valid
                        completion_meta["last_time_ms"] = int(gfx.lastTimeMs)
                        logger.info("Lap completed: %d -> %d (valid=%s, lastTimeMs=%d)",
                                    last_lap_id+1, lap_id+1, lap_valid, gfx.lastTimeMs)
                        integrated_x = 0.0
                        integrated_z = 0.0
                    last_lap_id = lap_id

                    # AC currentSectorIndex is 0-based; convert to 1-3 for AI layer.
                    raw_sector = int(gfx.currentSectorIndex)
                    sector = (raw_sector + 1) if 0 <= raw_sector <= 2 else None

                    sample_data = {
                        "lap_id": lap_id,
                        "t": elapsed,
                        "sector": sector,
                        "x": x,
                        "z": z,
                        "speed": speed,
                        "gear": display_gear,
                        "rpms": clamped_rpms,
                        "brake": phys.brake,
                        "throttle": phys.gas,
                        "fuel": phys.fuel,
                        "steer_angle": phys.steerAngle,
                        "g_force_lat": phys.accG[0],
                        "g_force_lon": phys.accG[2],
                        "tyre_pressure_fl": phys.wheelsPressure[0],
                        "tyre_pressure_fr": phys.wheelsPressure[1],
                        "tyre_pressure_rl": phys.wheelsPressure[2],
                        "tyre_pressure_rr": phys.wheelsPressure[3],
                        "tyre_temp_fl": phys.tyreCoreTemperature[0],
                        "tyre_temp_fr": phys.tyreCoreTemperature[1],
                        "tyre_temp_rl": phys.tyreCoreTemperature[2],
                        "tyre_temp_rr": phys.tyreCoreTemperature[3],
                        "tyre_wear_fl": phys.tyreWear[0],
                        "tyre_wear_fr": phys.tyreWear[1],
                        "tyre_wear_rl": phys.tyreWear[2],
                        "tyre_wear_rr": phys.tyreWear[3],
                        "wheel_slip_fl": phys.wheelSlip[0],
                        "wheel_slip_fr": phys.wheelSlip[1],
                        "wheel_slip_rl": phys.wheelSlip[2],
                        "wheel_slip_rr": phys.wheelSlip[3],
                        "suspension_fl": phys.suspensionTravel[0],
                        "suspension_fr": phys.suspensionTravel[1],
                        "suspension_rl": phys.suspensionTravel[2],
                        "suspension_rr": phys.suspensionTravel[3],
                        "camber_fl": phys.camberRAD[0],
                        "camber_fr": phys.camberRAD[1],
                        "camber_rl": phys.camberRAD[2],
                        "camber_rr": phys.camberRAD[3],
                        "ride_height_front": phys.rideHeight[0],
                        "ride_height_rear": phys.rideHeight[1],
                        "car_damage_front": phys.carDamage[0],
                        "car_damage_rear": phys.carDamage[1],
                        "car_damage_left": phys.carDamage[2],
                        "car_damage_right": phys.carDamage[3],
                        "car_damage_centre": phys.carDamage[4],
                        "tyres_out": phys.numberOfTyresOut,
                        "is_in_pit": gfx.isInPit,
                        "pit_limiter": phys.pitLimiterOn,
                        "lap_valid": lap_valid,
                        "last_time_ms": gfx.lastTimeMs,
                    }

                    lap_buffer.add_sample(
                        lap_id=lap_id, t=elapsed, x=x, z=z,
                        speed_kmh=speed, gear=display_gear, rpms=clamped_rpms,
                        brake=phys.brake, throttle=phys.gas, fuel=phys.fuel,
                        tyre_pressure_fl=phys.wheelsPressure[0],
                        tyre_pressure_fr=phys.wheelsPressure[1],
                        tyre_pressure_rl=phys.wheelsPressure[2],
                        tyre_pressure_rr=phys.wheelsPressure[3],
                        tyre_temp_fl=phys.tyreCoreTemperature[0],
                        tyre_temp_fr=phys.tyreCoreTemperature[1],
                        tyre_temp_rl=phys.tyreCoreTemperature[2],
                        tyre_temp_rr=phys.tyreCoreTemperature[3],
                        lap_valid=lap_valid,
                    )

                    self.realtime_sample.emit(sample_data)

                    if frame_count % 10 == 0:
                        live_data = {
                            "current_lap": lap_id + 1,
                            "sector": sector,
                            "speed": speed,
                            "gear": display_gear,
                            "rpm": phys.rpms,
                            "fuel": phys.fuel,
                            "position": gfx.position,
                            "is_in_pit": gfx.isInPit,
                            "ac_status": gfx.status,
                            "current_time": gfx.currentTime,
                            "last_time": _sanitize_ac_time_str(gfx.lastTime),
                            "best_time": _sanitize_ac_time_str(gfx.bestTime),
                            "last_time_ms": gfx.lastTimeMs,
                            "best_time_ms": gfx.bestTimeMs,
                            "completed_laps": lap_id,
                        }
                        self.live_data_update.emit(live_data)

                    time.sleep(1 / 60.0)

            except Exception as e:
                logger.warning("Shared memory read failed: %s — will reconnect", e)
                self.status_update.emit("Connection lost, reconnecting...")
            finally:
                self._close_handles(mm_phys, mm_graph, mm_static)

        logger.info("Disconnected from Assetto Corsa")
        self.status_update.emit("Disconnected from Assetto Corsa.")

    def stop(self):
        self.running = False
