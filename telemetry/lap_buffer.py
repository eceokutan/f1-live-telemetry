# telemetry/lap_buffer.py
from typing import List, Dict, Any, Callable


class LapBuffer:
    """
    Buffers telemetry samples for the current lap.
    When completedLaps increments, it calls a callback with the finished lap.

    The callback receives:
      - lap_id: int
      - samples: List[Dict[str, Any]]
    """

    def __init__(self, on_lap_complete: Callable[[int, List[Dict[str, Any]]], None]):
        self.on_lap_complete = on_lap_complete
        self.current_lap_id: int | None = None
        self.samples: List[Dict[str, Any]] = []

    def add_sample(
        self,
        lap_id: int,
        t: float,
        x: float,
        z: float,
        speed_kmh: float,
        **extra: Any,
    ) -> None:
        """
        Add one sample. If lap_id changes (increments), finish previous lap.

        Each sample is stored as a dict:
          {"t": t, "x": x, "z": z, "speed": speed_kmh, ...extra}
        """

        # First ever sample
        if self.current_lap_id is None:
            self.current_lap_id = lap_id

        # Same lap as current: just append
        if lap_id == self.current_lap_id:
            self.samples.append(
                {"t": t, "x": x, "z": z, "speed": speed_kmh, **extra}
            )
            return

        # New lap started (lap_id > current_lap_id)
        if lap_id > self.current_lap_id:
            # Finish previous lap
            if self.samples:
                self.on_lap_complete(self.current_lap_id, self.samples)

            # Start collecting a new lap
            self.current_lap_id = lap_id
            self.samples = [
                {"t": t, "x": x, "z": z, "speed": speed_kmh, **extra}
            ]
            return

        # If lap_id < current_lap_id, it's probably replay/time-reset; reset buffer
        if lap_id < self.current_lap_id:
            print("[LapBuffer] Lap counter went backwards, resetting buffer.")
            self.current_lap_id = lap_id
            self.samples = [
                {"t": t, "x": x, "z": z, "speed": speed_kmh, **extra}
            ]
