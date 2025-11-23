import sys
import ctypes as ct
import mmap
import time
from typing import List, Dict, Any
from PyQt5 import QtWidgets, QtCore
import numpy as np

# Import your dashboard
from dashboard import MainWindow

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


SHM_NAME_STATIC = "acpmf_static"
STATIC_SIZE = ct.sizeof(SPageFileStatic)


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


SHM_NAME_PHYSICS = "acpmf_physics"
SHM_NAME_GRAPHICS = "acpmf_graphics"
PHYSICS_SIZE = ct.sizeof(SPageFilePhysics)
GRAPHICS_SIZE = ct.sizeof(SPageFileGraphics)


# ===================== SHARED MEMORY HELPERS =====================

def open_shared_memory(name, size):
    """Open an existing named shared memory region created by Assetto Corsa."""
    try:
        mm = mmap.mmap(0, size, tagname=name, access=mmap.ACCESS_READ)
        return mm
    except Exception as e:
        print(f"ERROR: Could not open shared memory '{name}': {e}")
        print("Make sure Assetto Corsa is running and you're in a session.")
        return None


def read_physics(mm):
    mm.seek(0)
    raw = mm.read(PHYSICS_SIZE)
    return SPageFilePhysics.from_buffer_copy(raw)


def read_graphics(mm):
    mm.seek(0)
    raw = mm.read(GRAPHICS_SIZE)
    return SPageFileGraphics.from_buffer_copy(raw)


# ===================== LAP BUFFER =====================

class LapBuffer:
    """
    Buffers telemetry samples for the current lap.
    When completedLaps increments, it calls a callback with the finished lap.
    """

    def __init__(self, on_lap_complete):
        self.on_lap_complete = on_lap_complete
        self.current_lap_id = None
        self.samples: List[Dict[str, Any]] = []

    def add_sample(self, lap_id: int, t: float, x: float, z: float, speed_kmh: float, **extra):
        # First ever sample
        if self.current_lap_id is None:
            self.current_lap_id = lap_id

        # Same lap as current: just append
        if lap_id == self.current_lap_id:
            self.samples.append({"t": t, "x": x, "z": z, "speed": speed_kmh, **extra})
            return

        # New lap started (lap_id > current_lap_id)
        if lap_id > self.current_lap_id:
            # Finish previous lap
            if self.samples:
                self.on_lap_complete(self.current_lap_id, self.samples)

            # Start collecting a new lap
            self.current_lap_id = lap_id
            self.samples = [{"t": t, "x": x, "z": z, "speed": speed_kmh, **extra}]
            return

        # If lap_id < current_lap_id, reset
        if lap_id < self.current_lap_id:
            print("\n[LapBuffer] Lap counter went backwards, resetting buffer.")
            self.current_lap_id = lap_id
            self.samples = [{"t": t, "x": x, "z": z, "speed": speed_kmh, **extra}]


# ===================== TELEMETRY THREAD =====================

class TelemetryWorker(QtCore.QThread):
    """Background thread that reads telemetry and emits signals to GUI"""
    lap_completed = QtCore.pyqtSignal(int, list)  # (lap_id, samples)
    status_update = QtCore.pyqtSignal(str)  # status message

    def __init__(self):
        super().__init__()
        self.running = False

    def run(self):
        self.status_update.emit("Connecting to Assetto Corsa...")
        
        mm_phys = open_shared_memory(SHM_NAME_PHYSICS, PHYSICS_SIZE)
        mm_graph = open_shared_memory(SHM_NAME_GRAPHICS, GRAPHICS_SIZE)
        
        if mm_phys is None or mm_graph is None:
            self.status_update.emit("ERROR: Could not connect to AC")
            return

        self.status_update.emit("Connected! Start driving...")

        # LapBuffer with callback that emits signal
        lap_buffer = LapBuffer(on_lap_complete=lambda lap_id, samples: 
                              self.lap_completed.emit(lap_id, samples))

        t0 = time.time()
        self.running = True

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

                # ~60 Hz loop
                time.sleep(1 / 60.0)

        except Exception as e:
            self.status_update.emit(f"Error: {str(e)}")
        finally:
            mm_phys.close()
            mm_graph.close()
            self.status_update.emit("Disconnected")

    def stop(self):
        self.running = False


# ===================== MAIN APPLICATION =====================

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    
    # Create telemetry worker
    telemetry_thread = TelemetryWorker()
    
    # Connect signals
    telemetry_thread.lap_completed.connect(window.handle_lap_complete)
    telemetry_thread.status_update.connect(lambda msg: print(f"[Status] {msg}"))
    
    # Start telemetry thread
    telemetry_thread.start()
    
    # Show window
    window.show()
    
    # Run Qt event loop
    result = app.exec_()
    
    # Clean shutdown
    telemetry_thread.stop()
    telemetry_thread.wait()
    
    sys.exit(result)


if __name__ == "__main__":
    main()