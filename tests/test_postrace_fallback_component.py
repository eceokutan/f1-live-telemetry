from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from analysis.ai_pipeline_bridge import AIPipelineBridge
from data.lap import Lap
from data.models import LapSummary, SessionMetadata
from data.session import Session

pytestmark = [pytest.mark.component, pytest.mark.regression]


def _build_session_for_fallback(df: pd.DataFrame) -> tuple[Session, Lap]:
    summary = LapSummary(
        lap_number=1,
        lap_time=1.0,
        fuel_start=20.0,
        fuel_end=19.8,
        avg_speed=102.5,
        max_speed=105.0,
        min_speed=100.0,
        valid=True,
    )
    lap = Lap(lap_number=1, telemetry=df, summary=summary)
    metadata = SessionMetadata(
        session_id=42,
        game="ac",
        track_name="Spa",
        car_model="F1",
        player_name="Tester",
        start_time=datetime.now(timezone.utc),
        total_laps=1,
    )
    session = Session(
        metadata=metadata,
        laps=[lap],
        telemetry=df,
        ai_commentary=[],
    )
    return session, lap


def test_generate_fallback_reports_received_fields_without_missing_column_errors():
    telemetry = pd.DataFrame(
        {
            "lap_number": [1, 1],
            "elapsed_time": [0.0, 1.0],
            "speed": [100.0, 105.0],
            "g_force_lat": [1.1, 1.3],  # g_force_lon intentionally missing
            "driver_note": ["stable", "push"],
        }
    )
    session, lap = _build_session_for_fallback(telemetry)
    bridge = AIPipelineBridge.__new__(AIPipelineBridge)

    result = bridge._generate_fallback(session, lap)

    assert result.source == "built_in_fallback"
    assert "Telemetry fields" in result.coach
    assert "g_force_lat" in result.coach
    assert "driver_note" in result.coach
    assert "Field summaries:" in result.analyst
    assert "- g_force_lat:" in result.analyst
    assert "- driver_note:" in result.analyst
