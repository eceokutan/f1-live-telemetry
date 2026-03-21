"""
Pydantic data models for telemetry analysis.
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class TireCorners(BaseModel):
    """
    Data for all four tire corners.

    Used for tire temperatures, tire pressures, etc.
    """
    fl: float = Field(..., description="Front-left")
    fr: float = Field(..., description="Front-right")
    rl: float = Field(..., description="Rear-left")
    rr: float = Field(..., description="Rear-right")

    def __repr__(self) -> str:
        return f"TireCorners(FL={self.fl:.1f}, FR={self.fr:.1f}, RL={self.rl:.1f}, RR={self.rr:.1f})"


class SessionMetadata(BaseModel):
    """
    Metadata about a recorded session.

    Contains track, car, player info, and session statistics.
    """
    session_id: int
    game: str
    track_name: str
    car_model: str
    player_name: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_laps: int = 0
    ai_enabled: bool = False
    notes: str = ""

    def __repr__(self) -> str:
        return f"SessionMetadata({self.track_name}, {self.car_model}, {self.total_laps} laps)"


class LapSummary(BaseModel):
    """
    Summary statistics for a single lap.

    Includes lap time, fuel usage, speed stats, etc.
    """
    lap_number: int
    lap_time: float
    fuel_start: float
    fuel_end: float
    avg_speed: float
    max_speed: float
    min_speed: float
    valid: bool = Field(default=True, description="Whether this lap is valid (completed)")

    @property
    def fuel_used(self) -> float:
        """Calculate fuel used during this lap."""
        return self.fuel_start - self.fuel_end

    def __repr__(self) -> str:
        return f"LapSummary(Lap {self.lap_number}, {self.lap_time:.3f}s, {self.avg_speed:.1f} km/h avg)"


class AIComment(BaseModel):
    """
    AI commentary message from the race engineer.
    """
    timestamp: datetime
    message: str
    trigger: str = Field(description="What triggered this comment (e.g., 'driver_query', 'driver_query_timeout')")
    priority: int = Field(description="Message priority level (1=low, 2=medium, 3=high)")
    lap_number: int
    model: Optional[str] = Field(default=None, description="Model role that generated this message (e.g., coach, analyst)")

    def __repr__(self) -> str:
        return f"AIComment(Lap {self.lap_number}, {self.trigger}, '{self.message[:50]}...')"


class BrakePoint(BaseModel):
    """
    A detected brake point in the telemetry.

    Used for corner entry analysis.
    """
    distance: float = Field(description="Distance from lap start (meters)")
    speed_before: float = Field(description="Speed before braking (km/h)")
    speed_after: float = Field(description="Minimum speed reached (km/h)")
    brake_duration: float = Field(description="Duration of braking (seconds)")
    brake_pressure_max: float = Field(description="Maximum brake input (0.0 to 1.0)")

    def __repr__(self) -> str:
        return f"BrakePoint({self.distance:.1f}m, {self.speed_before:.1f} -> {self.speed_after:.1f} km/h)"


class Corner(BaseModel):
    """
    A corner detected in the telemetry.

    Includes entry, apex, and exit points.
    """
    corner_number: int
    entry_distance: float = Field(description="Distance at corner entry (meters)")
    apex_distance: float = Field(description="Distance at apex (meters)")
    exit_distance: float = Field(description="Distance at corner exit (meters)")
    entry_speed: float = Field(description="Speed at entry (km/h)")
    apex_speed: float = Field(description="Minimum speed at apex (km/h)")
    exit_speed: float = Field(description="Speed at exit (km/h)")
    corner_type: str = Field(default="medium", description="Corner type: slow, medium, fast")

    def __repr__(self) -> str:
        return f"Corner {self.corner_number} ({self.corner_type}): {self.entry_speed:.1f} -> {self.apex_speed:.1f} -> {self.exit_speed:.1f} km/h"
