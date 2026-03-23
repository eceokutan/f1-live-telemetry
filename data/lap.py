"""
Lap class - represents a single lap with telemetry data.
"""
import pandas as pd
import numpy as np
from typing import Optional
from .models import LapSummary, TireCorners


class Lap:
    """
    Represents a single lap with telemetry data and statistics.

    Attributes:
        lap_number: Lap number (0 for outlap, 1+ for racing laps)
        telemetry: DataFrame with all telemetry samples for this lap
        lap_time: Total lap time in seconds
        summary: LapSummary with calculated statistics
    """

    def __init__(
        self,
        lap_number: int,
        telemetry: pd.DataFrame,
        summary: Optional[LapSummary] = None
    ):
        self.lap_number = lap_number
        self.telemetry = telemetry.copy()

        # Normalize elapsed_time to start at 0 for this lap (it's session-relative in the DB)
        if len(self.telemetry) > 0 and 'elapsed_time' in self.telemetry.columns:
            t0 = self.telemetry['elapsed_time'].iloc[0]
            self.telemetry['elapsed_time'] = self.telemetry['elapsed_time'] - t0

        # Calculate lap time from telemetry (now lap-relative)
        self.lap_time = self.telemetry['elapsed_time'].max() if len(self.telemetry) > 0 else 0.0

        # Use provided summary or calculate it
        if summary:
            self.summary = summary
        else:
            self._calculate_summary()

    def _calculate_summary(self) -> None:
        """Calculate lap summary statistics from telemetry."""
        df = self.telemetry

        if len(df) == 0:
            self.summary = LapSummary(
                lap_number=self.lap_number,
                lap_time=0.0,
                fuel_start=0.0,
                fuel_end=0.0,
                avg_speed=0.0,
                max_speed=0.0,
                min_speed=0.0,
                valid=False
            )
            return

        fuel_start = df['fuel'].iloc[0] if 'fuel' in df.columns and len(df) > 0 else 0.0
        fuel_end = df['fuel'].iloc[-1] if 'fuel' in df.columns and len(df) > 0 else 0.0

        self.summary = LapSummary(
            lap_number=self.lap_number,
            lap_time=self.lap_time,
            fuel_start=float(fuel_start),
            fuel_end=float(fuel_end),
            avg_speed=float(df['speed'].mean()) if 'speed' in df.columns else 0.0,
            max_speed=float(df['speed'].max()) if 'speed' in df.columns else 0.0,
            min_speed=float(df['speed'].min()) if 'speed' in df.columns else 0.0,
            valid=bool(self.lap_time > 0.0)
        )

    def get_speed_trace(self) -> np.ndarray:
        """Get speed trace for this lap."""
        return self.telemetry['speed'].values if 'speed' in self.telemetry.columns else np.array([])

    def get_throttle_trace(self) -> np.ndarray:
        """Get throttle input trace."""
        return self.telemetry['throttle'].values if 'throttle' in self.telemetry.columns else np.array([])

    def get_brake_trace(self) -> np.ndarray:
        """Get brake input trace."""
        return self.telemetry['brake'].values if 'brake' in self.telemetry.columns else np.array([])

    def get_gear_trace(self) -> np.ndarray:
        """Get gear trace."""
        return self.telemetry['gear'].values if 'gear' in self.telemetry.columns else np.array([])

    def get_racing_line(self) -> tuple[np.ndarray, np.ndarray]:
        """Get racing line coordinates (X, Z positions)."""
        x = self.telemetry['pos_x'].values if 'pos_x' in self.telemetry.columns else np.array([])
        z = self.telemetry['pos_z'].values if 'pos_z' in self.telemetry.columns else np.array([])
        return x, z

    def get_avg_tire_temps(self) -> TireCorners:
        """Calculate average tire temperatures for all corners."""
        df = self.telemetry
        return TireCorners(
            fl=float(df['tyre_temp_fl'].mean()) if 'tyre_temp_fl' in df.columns else 0.0,
            fr=float(df['tyre_temp_fr'].mean()) if 'tyre_temp_fr' in df.columns else 0.0,
            rl=float(df['tyre_temp_rl'].mean()) if 'tyre_temp_rl' in df.columns else 0.0,
            rr=float(df['tyre_temp_rr'].mean()) if 'tyre_temp_rr' in df.columns else 0.0
        )

    def get_avg_tire_pressures(self) -> TireCorners:
        """Calculate average tire pressures for all corners."""
        df = self.telemetry
        return TireCorners(
            fl=float(df['tyre_pressure_fl'].mean()) if 'tyre_pressure_fl' in df.columns else 0.0,
            fr=float(df['tyre_pressure_fr'].mean()) if 'tyre_pressure_fr' in df.columns else 0.0,
            rl=float(df['tyre_pressure_rl'].mean()) if 'tyre_pressure_rl' in df.columns else 0.0,
            rr=float(df['tyre_pressure_rr'].mean()) if 'tyre_pressure_rr' in df.columns else 0.0
        )

    def __repr__(self) -> str:
        return f"Lap({self.lap_number}, {self.lap_time:.3f}s, {len(self.telemetry)} samples)"

    def __len__(self) -> int:
        return len(self.telemetry)
