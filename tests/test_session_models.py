from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from data.lap import Lap
from data.models import LapSummary, SessionMetadata
from data.session import Session

pytestmark = [pytest.mark.unit, pytest.mark.component, pytest.mark.regression]


def _lap_df(base_elapsed: float, lap_number: int, speeds: list[float], fuels: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lap_number": [lap_number] * len(speeds),
            "elapsed_time": [base_elapsed + idx * 0.5 for idx in range(len(speeds))],
            "speed": speeds,
            "fuel": fuels,
            "throttle": [0.7] * len(speeds),
            "brake": [0.1] * len(speeds),
            "gear": [4] * len(speeds),
            "pos_x": [1.0] * len(speeds),
            "pos_z": [2.0] * len(speeds),
            "tyre_temp_fl": [90.0] * len(speeds),
            "tyre_temp_fr": [91.0] * len(speeds),
            "tyre_temp_rl": [89.0] * len(speeds),
            "tyre_temp_rr": [90.0] * len(speeds),
            "tyre_pressure_fl": [26.0] * len(speeds),
            "tyre_pressure_fr": [26.1] * len(speeds),
            "tyre_pressure_rl": [25.8] * len(speeds),
            "tyre_pressure_rr": [25.9] * len(speeds),
        }
    )


def test_lap_normalizes_elapsed_time_and_calculates_summary():
    df = _lap_df(base_elapsed=12.0, lap_number=1, speeds=[100, 110, 120], fuels=[20.0, 19.9, 19.8])
    lap = Lap(lap_number=1, telemetry=df)

    assert lap.telemetry["elapsed_time"].iloc[0] == 0.0
    assert lap.lap_time == pytest.approx(1.0)
    assert lap.summary.valid is True
    assert lap.summary.avg_speed == pytest.approx(110.0)
    assert lap.summary.fuel_used == pytest.approx(0.2)


def test_session_fastest_lap_and_consistency_metrics():
    lap1 = Lap(1, _lap_df(base_elapsed=0.0, lap_number=1, speeds=[100, 105, 110], fuels=[20.0, 19.9, 19.8]))
    lap2 = Lap(2, _lap_df(base_elapsed=2.0, lap_number=2, speeds=[99, 103, 108], fuels=[19.8, 19.7, 19.6]))
    lap3_summary = LapSummary(
        lap_number=3,
        lap_time=0.0,
        fuel_start=19.6,
        fuel_end=19.6,
        avg_speed=0.0,
        max_speed=0.0,
        min_speed=0.0,
        valid=False,
    )
    lap3 = Lap(3, _lap_df(base_elapsed=4.0, lap_number=3, speeds=[0, 0, 0], fuels=[19.6, 19.6, 19.6]), summary=lap3_summary)

    metadata = SessionMetadata(
        session_id=1,
        game="ac",
        track_name="Monza",
        car_model="F1",
        player_name="Driver",
        start_time=datetime.now(timezone.utc),
    )
    session = Session(metadata=metadata, laps=[lap2, lap1, lap3], telemetry=pd.concat([lap1.telemetry, lap2.telemetry, lap3.telemetry]))

    fastest = session.get_fastest_lap()
    consistency = session.calculate_consistency()

    assert fastest is not None
    assert fastest.lap_number == 1
    assert consistency["fastest"] == pytest.approx(lap1.lap_time)
    assert consistency["slowest"] == pytest.approx(lap2.lap_time)
    assert consistency["mean"] > 0
