# telemetry/model.py
from dataclasses import dataclass

@dataclass
class Sample:
    t: float           # seconds since session start
    lap_id: int        # completedLaps-style index
    x: float           # world X
    z: float           # world Z
    speed: float       # km/h
    gear: int          # as given by game (we can map N/R later if needed)
    rpms: int
    brake: float       # 0–1
    throttle: float    # 0–1
