"""
Telemetry data schemas for Jarvis-Granite Live Telemetry.

Defines Pydantic models for:
- TireTemps: Tire temperatures for all four corners
- TireWear: Tire wear percentages
- GForces: Lateral and longitudinal G-forces
- TelemetryData: Complete telemetry snapshot
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class TireTemps(BaseModel):
    """
    Tire temperatures in Celsius for all four corners.

    Attributes:
        fl: Front-left tire temperature
        fr: Front-right tire temperature
        rl: Rear-left tire temperature
        rr: Rear-right tire temperature
    """
    fl: float = Field(..., description="Front-left tire temperature (°C)")
    fr: float = Field(..., description="Front-right tire temperature (°C)")
    rl: float = Field(..., description="Rear-left tire temperature (°C)")
    rr: float = Field(..., description="Rear-right tire temperature (°C)")

    @field_validator('fl', 'fr', 'rl', 'rr')
    @classmethod
    def temperature_must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError('Tire temperature cannot be negative')
        return v


class TireWear(BaseModel):
    """
    Tire wear percentages for all four corners.

    Values range from 0 (fresh) to 100 (completely worn).

    Attributes:
        fl: Front-left tire wear percentage
        fr: Front-right tire wear percentage
        rl: Rear-left tire wear percentage
        rr: Rear-right tire wear percentage
    """
    fl: float = Field(..., ge=0, le=100, description="Front-left tire wear (%)")
    fr: float = Field(..., ge=0, le=100, description="Front-right tire wear (%)")
    rl: float = Field(..., ge=0, le=100, description="Rear-left tire wear (%)")
    rr: float = Field(..., ge=0, le=100, description="Rear-right tire wear (%)")


class TirePressure(BaseModel):
    """
    Tire pressure in PSI for all four corners.

    Attributes:
        fl: Front-left tire pressure
        fr: Front-right tire pressure
        rl: Rear-left tire pressure
        rr: Rear-right tire pressure
    """
    fl: float = Field(..., ge=0, description="Front-left tire pressure (PSI)")
    fr: float = Field(..., ge=0, description="Front-right tire pressure (PSI)")
    rl: float = Field(..., ge=0, description="Rear-left tire pressure (PSI)")
    rr: float = Field(..., ge=0, description="Rear-right tire pressure (PSI)")

    @field_validator('fl', 'fr', 'rl', 'rr')
    @classmethod
    def pressure_must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError('Tire pressure cannot be negative')
        return v


class GForces(BaseModel):
    """
    G-force measurements.

    Attributes:
        lateral: Lateral G-force (positive = right turn)
        longitudinal: Longitudinal G-force (positive = acceleration)
    """
    lateral: float = Field(..., description="Lateral G-force")
    longitudinal: float = Field(..., description="Longitudinal G-force")


class TelemetryData(BaseModel):
    """
    Complete telemetry data snapshot from a racing session.

    Modified to support AC telemetry format.
    """
    # Speed and engine (AC format)
    speed: float = Field(..., ge=0, description="Current speed in km/h")
    rpms: int = Field(..., ge=0, description="Engine RPM")
    gear: int = Field(..., description="Current gear (0=neutral, -1=reverse)")

    # Driver inputs
    throttle: float = Field(..., ge=0, le=1, description="Throttle position (0-1)")
    brake: float = Field(..., ge=0, le=1, description="Brake pressure (0-1)")
    steering_angle: Optional[float] = Field(default=0.0, ge=-1, le=1, description="Steering input (-1 to 1)")

    # Resources
    fuel: Optional[float] = Field(default=None, description="Fuel remaining in liters")

    # Tire data
    tire_temps: TireTemps = Field(..., description="Tire temperatures")
    tire_wear: Optional[TireWear] = Field(default=None, description="Tire wear percentages")
    tire_pressure: Optional['TirePressure'] = Field(default=None, description="Tire pressures (PSI)")

    # Physics
    g_forces: Optional[GForces] = Field(default=None, description="G-force measurements")

    # Position data (AC format)
    x: Optional[float] = Field(default=0.0, description="World position X coordinate")
    z: Optional[float] = Field(default=0.0, description="World position Z coordinate")

    # Lap data (AC format)
    lap_id: int = Field(default=0, ge=0, description="Current lap number")
    t: float = Field(default=0.0, ge=0, description="Time since lap start in seconds")

    # Track position (optional)
    track_position: Optional[float] = Field(default=None, ge=0, le=1, description="Position on track (0-1)")
    lap_number: Optional[int] = Field(default=None, ge=0, description="Current lap number")
    lap_time_current: Optional[float] = Field(default=None, ge=0, description="Current lap time in seconds")
    sector: Optional[int] = Field(default=None, ge=1, le=3, description="Current sector (1-3)")

    # Race position (optional - may not be available in all modes)
    position: Optional[int] = Field(default=None, ge=1, description="Race position")
    gap_ahead: Optional[float] = Field(default=None, description="Gap to car ahead (seconds)")
    gap_behind: Optional[float] = Field(default=None, description="Gap to car behind (seconds)")
