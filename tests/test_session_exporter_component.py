from __future__ import annotations

import csv
import json
import sqlite3
import time
from pathlib import Path

import pytest

from data.csv_importer import ensure_tables
from data.session_exporter import SessionExporter

pytestmark = [pytest.mark.component, pytest.mark.regression]


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
        (now - 120.0, now, "ac", "Monza", "F1", "Driver", 1, 90.5, 1, "initial notes"),
    )
    session_id = cursor.lastrowid

    cursor.execute(
        """
        INSERT INTO laps (
            session_id, lap_number, lap_time, fuel_start, fuel_end,
            avg_speed, max_speed, min_speed, valid, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (session_id, 1, 90.5, 20.0, 18.5, 150.0, 280.0, 80.0, 1, now),
    )

    telemetry_rows = [
        (session_id, 1, 0.0, 1.0, 2.0, 100.0, 3, 9000, 0.5, 0.1, 20.0, 26.1, 26.2, 25.9, 26.0, 90.0, 91.0, 89.0, 90.0, now),
        (session_id, 1, 0.5, 1.1, 2.1, 120.0, 4, 9500, 0.7, 0.0, 19.8, 26.1, 26.2, 25.9, 26.0, 91.0, 92.0, 90.0, 91.0, now),
    ]
    cursor.executemany(
        """
        INSERT INTO telemetry (
            session_id, lap_number, elapsed_time, pos_x, pos_z, speed,
            gear, rpm, throttle, brake, fuel,
            tyre_pressure_fl, tyre_pressure_fr, tyre_pressure_rl, tyre_pressure_rr,
            tyre_temp_fl, tyre_temp_fr, tyre_temp_rl, tyre_temp_rr, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        telemetry_rows,
    )

    cursor.execute(
        """
        INSERT INTO ai_commentary (session_id, timestamp, message, trigger, priority, lap_number)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (session_id, now, "Fuel looks good.", "driver_query", "2", 1),
    )
    cursor.execute(
        """
        INSERT INTO voice_queries (session_id, timestamp, query_text, response_text, lap_number)
        VALUES (?, ?, ?, ?, ?)
        """,
        (session_id, now, "How is fuel?", "Fuel looks good.", 1),
    )
    db.commit()
    db.close()
    return int(session_id)


def test_list_sessions_returns_seeded_session(tmp_path):
    db_path = tmp_path / "telemetry.db"
    session_id = _seed_db(db_path)
    exporter = SessionExporter(db_path=str(db_path))

    sessions = exporter.list_sessions()

    assert len(sessions) == 1
    assert sessions[0]["session_id"] == session_id
    assert sessions[0]["track_name"] == "Monza"
    assert sessions[0]["ai_enabled"] is True


def test_export_session_writes_expected_files(tmp_path):
    db_path = tmp_path / "telemetry.db"
    session_id = _seed_db(db_path)
    exporter = SessionExporter(db_path=str(db_path))
    output_dir = tmp_path / "export"

    exported_path = Path(exporter.export_session(session_id, output_dir=str(output_dir)))

    telemetry_csv = exported_path / f"session_{session_id}_telemetry.csv"
    laps_csv = exported_path / f"session_{session_id}_laps.csv"
    ai_csv = exported_path / f"session_{session_id}_ai_commentary.csv"
    metadata_json = exported_path / f"session_{session_id}_metadata.json"

    assert telemetry_csv.exists()
    assert laps_csv.exists()
    assert ai_csv.exists()
    assert metadata_json.exists()

    with open(telemetry_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 3  # header + 2 telemetry samples

    with open(metadata_json, encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["session_id"] == session_id
    assert metadata["track_name"] == "Monza"


def test_rename_and_delete_session_updates_database(tmp_path):
    db_path = tmp_path / "telemetry.db"
    session_id = _seed_db(db_path)
    exporter = SessionExporter(db_path=str(db_path))

    exporter.rename_session(session_id, "Race stints run")
    exporter.delete_session(session_id)

    db = sqlite3.connect(str(db_path))
    cursor = db.cursor()
    for table_name in ("sessions", "laps", "telemetry", "ai_commentary", "voice_queries"):
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        assert cursor.fetchone()[0] == 0
    db.close()
