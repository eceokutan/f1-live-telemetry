"""
Session class - represents a complete recorded session with all laps.
"""
import pandas as pd
from typing import List, Optional
from .models import SessionMetadata, AIComment
from .lap import Lap


class Session:
    """
    Represents a complete recorded telemetry session.

    Attributes:
        metadata: Session metadata (track, car, player, etc.)
        laps: List of all laps in the session
        telemetry: Full telemetry DataFrame for all laps
        ai_commentary: Optional AI commentary messages
    """

    def __init__(
        self,
        metadata: SessionMetadata,
        laps: List[Lap],
        telemetry: pd.DataFrame,
        ai_commentary: Optional[List[AIComment]] = None
    ):
        self.metadata = metadata
        self.laps = sorted(laps, key=lambda lap: lap.lap_number)
        self.telemetry = telemetry
        self.ai_commentary = ai_commentary or []

    def get_lap(self, lap_number: int) -> Optional[Lap]:
        """Get a specific lap by number."""
        for lap in self.laps:
            if lap.lap_number == lap_number:
                return lap
        return None

    def get_fastest_lap(self) -> Optional[Lap]:
        """Get the fastest valid lap in the session."""
        valid_laps = [lap for lap in self.laps if lap.summary.valid and lap.lap_time > 0]
        if not valid_laps:
            return None
        return min(valid_laps, key=lambda lap: lap.lap_time)

    def get_lap_times(self) -> List[float]:
        """Get lap times for all laps."""
        return [lap.lap_time for lap in self.laps]

    def get_laps_by_range(self, start: int, end: int) -> List[Lap]:
        """Get laps within a range (inclusive)."""
        return [lap for lap in self.laps if start <= lap.lap_number <= end]

    def calculate_consistency(self) -> dict:
        """Calculate lap time consistency metrics."""
        valid_times = [lap.lap_time for lap in self.laps if lap.summary.valid and lap.lap_time > 0]

        if not valid_times:
            return {
                'mean': 0.0,
                'std_dev': 0.0,
                'coefficient_of_variation': 0.0,
                'fastest': 0.0,
                'slowest': 0.0
            }

        import numpy as np
        mean = np.mean(valid_times)
        std_dev = np.std(valid_times)

        return {
            'mean': mean,
            'std_dev': std_dev,
            'coefficient_of_variation': (std_dev / mean * 100) if mean > 0 else 0.0,
            'fastest': min(valid_times),
            'slowest': max(valid_times)
        }

    def __repr__(self) -> str:
        return f"Session({self.metadata.track_name}, {len(self.laps)} laps, {self.metadata.car_model})"

    def __len__(self) -> int:
        return len(self.laps)
