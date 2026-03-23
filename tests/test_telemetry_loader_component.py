from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from data.telemetry_loader import TelemetryLoader

pytestmark = [pytest.mark.component, pytest.mark.regression]


def test_validate_telemetry_raises_for_missing_required_columns():
    df = pd.DataFrame({"lap_number": [1], "elapsed_time": [0.0]})

    with pytest.raises(ValueError, match="missing required columns"):
        TelemetryLoader._validate_telemetry(df)


def test_load_session_from_file_path_parses_metadata_and_ai(tmp_path):
    telemetry = pd.DataFrame(
        {
            "lap_number": [1, 1],
            "elapsed_time": [0.0, 0.5],
            "speed": [100.0, 120.0],
            "fuel": [20.0, 19.8],
            "pos_x": [1.0, 1.1],
            "pos_z": [2.0, 2.1],
        }
    )
    laps = pd.DataFrame(
        {
            "lap_number": [1],
            "lap_time": [90.0],
            "fuel_start": [20.0],
            "fuel_end": [19.8],
            "avg_speed": [110.0],
            "max_speed": [120.0],
            "min_speed": [100.0],
            "valid": [1],
        }
    )
    commentary = pd.DataFrame(
        {
            "timestamp": [1700000000.0],
            "message": ["Manage tires."],
            "trigger": ["lap_complete"],
            "priority": ["high"],
            "lap_number": [1],
            "model": ["coach"],
        }
    )

    base = tmp_path / "session_42"
    telemetry_path = tmp_path / "session_42_telemetry.csv"
    laps_path = tmp_path / "session_42_laps.csv"
    ai_path = tmp_path / "session_42_ai_commentary.csv"
    metadata_path = tmp_path / "session_42_metadata.json"

    telemetry.to_csv(telemetry_path, index=False)
    laps.to_csv(laps_path, index=False)
    commentary.to_csv(ai_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "session_id": 42,
                "game": "ac",
                "track_name": "Silverstone",
                "car_model": "F1",
                "player_name": "Tester",
                "total_laps": 1,
                "ai_enabled": True,
                "notes": "component test",
            }
        ),
        encoding="utf-8",
    )

    session = TelemetryLoader.load_session(str(telemetry_path))

    assert session.metadata.session_id == 42
    assert session.metadata.track_name == "Silverstone"
    assert len(session.laps) == 1
    assert len(session.ai_commentary) == 1
    assert session.ai_commentary[0].priority == 3
    assert session.ai_commentary[0].model == "coach"
