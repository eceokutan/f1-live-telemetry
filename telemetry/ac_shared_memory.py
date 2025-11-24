# telemetry/ac_shared_memory.py
import ctypes as ct
import mmap
import time
from typing import List, Dict, Any, Optional

from PyQt5 import QtCore

from .lap_buffer import LapBuffer


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
    _fields_ = [
        ("packetId", ct.c_int),
        ("status", ct.c_int),
        ("session", ct.c_int),
        ("currentTime", ct.c_char * 15),
        ("lastTime", ct.c_char * 15),
        ("bestTime", ct.c_char * 15),
        ("splitTime", ct.c_char * 15),
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
        ("tyreCompound", ct.c_char * 33),
        ("replayTimeMultiplier", ct.c_float),
        ("normalizedCarPosition", ct.c_float),
        ("carCoordinates", ct.c_float * 3),
    ]


# ===================== STATIC INFO SHARED MEMORY =====================

class SPageFileStatic(ct.Structure):
    _fields_ = [
        ("_smVersion", ct.c_char * 15),
        ("_acVersion", ct.c_char * 15),
        ("numberOfSessions", ct.c_int),
        ("numCars", ct.c_int),
        ("carModel", ct.c_char * 33),
        ("track", ct.c_char * 33),
        ("playerName", ct.c_char * 33),
        ("playerSurname", ct.c_char * 33),
        ("playerNick", ct.c_char * 33),
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
        ("trackConfiguration", ct.c_char * 33),
        ("ersMaxJ", ct.c_float),
        ("isTimedRace", ct.c_int),
        ("hasExtraLap", ct.c_int),
        ("carSkin", ct.c_char * 33),
        ("reversedGridPositions", ct.c_int),
        ("pitWindowStart", ct.c_int),
        ("pitWindowEnd", ct.c_int),
        ("isOnline", ct.c_int),
        ("dryTyresName", ct.c_char * 33),
        ("wetTyresName", ct.c_char * 33),
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
        print(f"ERROR: Could not open shared memory '{name}': {e}")
        print("Make sure Assetto Corsa is running and you're in a session.")
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
    lap_completed = QtCore.pyqtSignal(int, list)  # (lap_id, samples: List[Dict])
    status_update = QtCore.pyqtSignal(str)        # status message
    session_info_update = QtCore.pyqtSignal(dict) # session info (track, car, driver)
    live_data_update = QtCore.pyqtSignal(dict)    # live telemetry updates

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False

    def run(self):
        self.status_update.emit("Connecting to Assetto Corsa...")

        mm_phys = open_shared_memory(SHM_NAME_PHYSICS, PHYSICS_SIZE)
        mm_graph = open_shared_memory(SHM_NAME_GRAPHICS, GRAPHICS_SIZE)
        mm_static = open_shared_memory(SHM_NAME_STATIC, STATIC_SIZE)

        if mm_phys is None or mm_graph is None:
            self.status_update.emit("ERROR: Could not connect to AC shared memory.")
            return

        self.status_update.emit("Connected! Start driving...")

        # Read static info once and emit it
        if mm_static is not None:
            try:
                static_data = read_static(mm_static)
                session_data = {
                    "track": static_data.track.decode('utf-8', errors='ignore'),
                    "track_config": static_data.trackConfiguration.decode('utf-8', errors='ignore'),
                    "car_model": static_data.carModel.decode('utf-8', errors='ignore'),
                    "player_name": static_data.playerName.decode('utf-8', errors='ignore'),
                    "player_surname": static_data.playerSurname.decode('utf-8', errors='ignore'),
                    "player_nick": static_data.playerNick.decode('utf-8', errors='ignore'),
                    "max_rpm": static_data.maxRpm,
                    "max_fuel": static_data.maxFuel,
                }
                self.session_info_update.emit(session_data)
            except Exception as e:
                print(f"Warning: Could not read static info: {e}")

        # LapBuffer with callback that emits a Qt signal
        lap_buffer = LapBuffer(
            on_lap_complete=lambda lap_id, samples: self.lap_completed.emit(lap_id, samples)
        )

        t0 = time.time()
        self.running = True
        frame_count = 0

        try:
            while self.running:
                phys = read_physics(mm_phys)
                gfx = read_graphics(mm_graph)

                now = time.time()
                elapsed = now - t0

                x = gfx.carCoordinates[0]
                z = gfx.carCoordinates[2]
                lap_id = gfx.completedLaps
                speed = phys.speedKmh

                # Feed sample into lap buffer
                lap_buffer.add_sample(
                    lap_id=lap_id,
                    t=elapsed,
                    x=x,
                    z=z,
                    speed_kmh=speed,
                    gear=phys.gear,
                    rpms=phys.rpms,
                    brake=phys.brake,
                    throttle=phys.gas,
                )

                # Emit live data every 10 frames (~6 times/sec at 60Hz)
                frame_count += 1
                if frame_count % 10 == 0:
                    live_data = {
                        "current_lap": lap_id + 1,
                        "speed": speed,
                        "gear": phys.gear,
                        "rpm": phys.rpms,
                        "fuel": phys.fuel,
                        "position": gfx.position,
                        "is_in_pit": gfx.isInPit,
                        "current_time": gfx.currentTime.decode('utf-8', errors='ignore'),
                        "last_time": gfx.lastTime.decode('utf-8', errors='ignore'),
                        "best_time": gfx.bestTime.decode('utf-8', errors='ignore'),
                    }
                    self.live_data_update.emit(live_data)

                # ~60 Hz loop
                time.sleep(1 / 60.0)

        except Exception as e:
            self.status_update.emit(f"Error in telemetry loop: {str(e)}")
        finally:
            try:
                mm_phys.close()
                mm_graph.close()
                if mm_static is not None:
                    mm_static.close()
            except Exception:
                pass
            self.status_update.emit("Disconnected from Assetto Corsa.")

    def stop(self):
        self.running = False