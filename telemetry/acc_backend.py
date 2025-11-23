# acc_backend.py

import time
from typing import List, Dict, Any

from PyQt5 import QtCore

from accapi.client import AccClient, Event


class LapBuffer:
    """
    Same idea as your AC LapBuffer, but we use ACC's lap counter.
    When lap_id increments, we flush the previous lap.
    """

    def __init__(self, on_lap_complete):
        self.on_lap_complete = on_lap_complete
        self.current_lap_id = None
        self.samples: List[Dict[str, Any]] = []

    def add_sample(self, lap_id: int, t: float, x: float, z: float, speed_kmh: float, **extra):
        # First sample ever
        if self.current_lap_id is None:
            self.current_lap_id = lap_id

        # Same lap
        if lap_id == self.current_lap_id:
            self.samples.append(
                {"t": t, "x": x, "z": z, "speed": speed_kmh, **extra}
            )
            return

        # New lap started (lap_id > current)
        if lap_id > self.current_lap_id:
            if self.samples:
                self.on_lap_complete(self.current_lap_id, self.samples)

            self.current_lap_id = lap_id
            self.samples = [
                {"t": t, "x": x, "z": z, "speed": speed_kmh, **extra}
            ]
            return

        # If lap_id < current_lap_id (session reset or pits etc.)
        if lap_id < self.current_lap_id:
            print("[ACC LapBuffer] Lap counter went backwards, resetting buffer.")
            self.current_lap_id = lap_id
            self.samples = [
                {"t": t, "x": x, "z": z, "speed": speed_kmh, **extra}
            ]


class AccTelemetryWorker(QtCore.QThread):
    """
    Background thread for Assetto Corsa Competizione telemetry.
    Uses accapi to talk to the ACC broadcast API.
    """

    lap_completed = QtCore.pyqtSignal(int, list)  # (lap_id, samples)
    status_update = QtCore.pyqtSignal(str)

    def __init__(self, host: str = "127.0.0.1", port: int = 9001, password: str = "asd"):
        super().__init__()
        self.host = host
        self.port = port
        self.password = password
        self.running = False
        self._t0 = None

        self._client: AccClient | None = None
        self._lap_buffer = LapBuffer(
            on_lap_complete=lambda lap_id, samples: self.lap_completed.emit(
                lap_id, samples
            )
        )

    # ------------------ QThread entry ------------------ #

    def run(self):
        self.status_update.emit(
            f"[ACC] Connecting to broadcast API at {self.host}:{self.port}..."
        )

        self._client = AccClient()

        # Register callbacks BEFORE start()
        self._client.onConnectionStateChange.subscribe(
            self._on_connection_state_change
        )
        self._client.onRealtimeCarUpdate.subscribe(
            self._on_realtime_car_update
        )

        # Start the client (non-blocking: it runs its own thread)
        self._client.start(self.host, self.port, self.password)

        self._t0 = time.time()
        self.running = True

        try:
            # Just keep the QThread alive until stop() is called
            while self.running:
                time.sleep(0.1)
        finally:
            if self._client is not None:
                self._client.stop()
            self.status_update.emit("[ACC] Disconnected from broadcast API")

    def stop(self):
        self.running = False

    # ------------------ Callbacks from accapi ------------------ #

    def _on_connection_state_change(self, event: Event) -> None:
        """
        Called whenever ACC connection state changes.
        event.content typically is an enum or similar.
        """
        self.status_update.emit(f"[ACC] Connection state: {event.content}")

    def _on_realtime_car_update(self, event: Event) -> None:
        """
        Called VERY frequently with realtime info for a car.
        This is where we normalize ACC data into our standard
        sample dict and feed LapBuffer.
        """
        if self._t0 is None:
            self._t0 = time.time()

        u = event.content  # accapi model instance
        t = time.time() - self._t0

        # ---------- IMPORTANT: mapping section ----------
        # I *cannot* see the accapi model here, so field names below
        # are my best guess and MUST be verified by you:
        #
        # 1. Temporarily add:  print(u) and print(dir(u))
        # 2. Run ACC, start a session.
        # 3. Look at the console and confirm actual attribute names.
        #
        # Then replace the getattr(...) calls with the exact names.

        # Position (world coordinates)
        try:
            # many tools expose position as a 3-float vector like WorldPosition
            world_pos = getattr(u, "WorldPosition")
            x = float(world_pos[0])
            z = float(world_pos[2])
        except Exception:
            # Fallback if name is different – update this once you inspect u
            x = 0.0
            z = 0.0

        # Speed [km/h]
        speed = float(getattr(u, "Kmh", 0.0))

        # Gear, RPM, brake, throttle
        gear = int(getattr(u, "Gear", 0))
        rpms = int(getattr(u, "EngineRpm", 0))
        brake = float(getattr(u, "Brake", 0.0))
        throttle = float(getattr(u, "Throttle", 0.0))

        # Lap counter – check actual name (Lap, CurrentLap, CompletedLaps, etc.)
        lap_id = int(getattr(u, "Lap", 0))

        # TODO while debugging:
        # print("RealtimeCarUpdate:", u)

        self._lap_buffer.add_sample(
            lap_id=lap_id,
            t=t,
            x=x,
            z=z,
            speed_kmh=speed,
            gear=gear,
            rpms=rpms,
            brake=brake,
            throttle=throttle,
        )
