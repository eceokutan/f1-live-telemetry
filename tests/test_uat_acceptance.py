from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path

import pytest

from ai.race_engineer_core.llm_client import LLMClient
from data.csv_importer import ensure_tables
from data.session_exporter import SessionExporter
from data.telemetry_loader import TelemetryLoader

pytestmark = [pytest.mark.uat, pytest.mark.regression]


def _seed_db(db_path: Path) -> int:
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    ensure_tables(db)
    cursor = db.cursor()
    now = time.time()

    cursor.execute(
        """
        INSERT INTO sessions (
            start_time, end_time, game, track_name, car_model, player_name,
            total_laps, best_lap_time, ai_enabled, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (now - 75.0, now, "ac", "Spa", "F1", "Driver", 1, 89.5, 1, "uat"),
    )
    session_id = int(cursor.lastrowid)

    cursor.execute(
        """
        INSERT INTO laps (
            session_id, lap_number, lap_time, fuel_start, fuel_end,
            avg_speed, max_speed, min_speed, valid, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (session_id, 1, 89.5, 19.8, 18.1, 158.0, 288.0, 92.0, 1, now),
    )

    cursor.executemany(
        """
        INSERT INTO telemetry (
            session_id, lap_number, elapsed_time, pos_x, pos_z, speed,
            gear, rpm, throttle, brake, fuel,
            tyre_pressure_fl, tyre_pressure_fr, tyre_pressure_rl, tyre_pressure_rr,
            tyre_temp_fl, tyre_temp_fr, tyre_temp_rl, tyre_temp_rr, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (session_id, 1, 0.0, 2.0, 3.0, 118.0, 4, 9100, 0.5, 0.1, 19.8, 26.1, 26.2, 25.9, 26.0, 90.0, 91.0, 89.0, 90.0, now),
            (session_id, 1, 0.5, 2.2, 3.1, 142.0, 5, 9900, 0.8, 0.0, 19.5, 26.1, 26.2, 25.9, 26.0, 92.0, 93.0, 91.0, 92.0, now),
        ],
    )

    cursor.execute(
        """
        INSERT INTO ai_commentary (session_id, timestamp, message, trigger, priority, lap_number)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (session_id, now, "Tyres stable through sector 2.", "lap_complete", "2", 1),
    )
    db.commit()
    db.close()
    return session_id


def test_uat_driver_can_review_lap_performance(tmp_path):
    db_path = tmp_path / "uat_review.db"
    session_id = _seed_db(db_path)
    exporter = SessionExporter(db_path=str(db_path))

    sessions = exporter.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == session_id
    assert sessions[0]["track_name"] == "Spa"
    assert sessions[0]["car_model"] == "F1"

    export_dir = tmp_path / "uat_review_export"
    exporter.export_session(session_id, output_dir=str(export_dir))
    loaded = TelemetryLoader.load_session(str(export_dir))

    assert len(loaded.laps) == 1
    lap = loaded.laps[0]
    assert {"elapsed_time", "speed", "pos_x", "pos_z"}.issubset(set(lap.telemetry.columns))
    assert loaded.get_fastest_lap() is not None


def test_uat_driver_receives_understandable_coaching_output():
    client = LLMClient(force_rule_based_fallback=True)

    response = asyncio.run(client.invoke("fuel critical"))

    assert response.strip()
    normalized = response.lower()
    assert "fuel" in normalized or "box" in normalized or "laps" in normalized
    assert "<|start_of_role|>" not in response


def test_uat_session_sharing_via_bundle_export_import(tmp_path):
    source_db = tmp_path / "uat_source.db"
    source_session_id = _seed_db(source_db)
    source_exporter = SessionExporter(db_path=str(source_db))

    bundle_path = tmp_path / "uat_session.jsession"
    source_exporter.export_session_bundle(source_session_id, str(bundle_path))
    assert bundle_path.exists()

    target_db = tmp_path / "uat_target.db"
    target_exporter = SessionExporter(db_path=str(target_db))
    imported_session_id = target_exporter.import_session_bundle(str(bundle_path))

    assert imported_session_id > 0
    sessions = target_exporter.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["track_name"] == "Spa"
