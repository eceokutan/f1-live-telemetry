"""
Configuration models for the local race engineer pipeline.
"""

from pydantic import BaseModel, Field


class ThresholdsConfig(BaseModel):
    """Thresholds used by the rule-based telemetry agent."""

    tire_temp_warning: float = Field(default=100.0)
    tire_temp_critical: float = Field(default=110.0)
    tire_wear_warning: float = Field(default=70.0)
    tire_wear_critical: float = Field(default=85.0)
    fuel_warning_laps: int = Field(default=5)
    fuel_critical_laps: int = Field(default=2)
    gap_change_threshold: float = Field(default=1.0)
    wheel_slip_warning: float = Field(default=50.0)
    wheel_slip_critical: float = Field(default=100.0)
