from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from data.csv_importer import ensure_tables
from data.session_exporter import SessionExporter
from data.telemetry_loader import TelemetryLoader

pytestmark = [pytest.mark.system, pytest.mark.regression]


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
        (now - 90.0, now, "ac", "Silverstone", "F1", "Driver", 1, 88.0, 1, "system smoke"),
    )
    session_id = int(cursor.lastrowid)

    cursor.execute(
        """
        INSERT INTO laps (
            session_id, lap_number, lap_time, fuel_start, fuel_end,
            avg_speed, max_speed, min_speed, valid, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (session_id, 1, 88.0, 20.0, 18.3, 160.0, 292.0, 95.0, 1, now),
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
            (session_id, 1, 0.0, 1.0, 1.0, 120.0, 4, 9000, 0.5, 0.0, 20.0, 26.0, 26.1, 25.8, 25.9, 92.0, 93.0, 91.0, 92.0, now),
            (session_id, 1, 0.5, 1.2, 1.1, 145.0, 5, 9800, 0.8, 0.1, 19.7, 26.0, 26.1, 25.8, 25.9, 94.0, 95.0, 93.0, 94.0, now),
        ],
    )

    cursor.execute(
        """
        INSERT INTO ai_commentary (session_id, timestamp, message, trigger, priority, lap_number)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (session_id, now, "Push now.", "lap_complete", "3", 1),
    )
    db.commit()
    db.close()
    return session_id


def test_system_smoke_end_to_end_session_roundtrip(tmp_path):
    source_db = tmp_path / "system_source.db"
    source_session_id = _seed_db(source_db)
    source_exporter = SessionExporter(db_path=str(source_db))

    export_dir = tmp_path / "system_export"
    source_exporter.export_session(source_session_id, output_dir=str(export_dir))
    loaded = TelemetryLoader.load_session(str(export_dir))

    assert loaded.metadata.track_name == "Silverstone"
    assert loaded.metadata.total_laps == 1
    assert loaded.get_fastest_lap() is not None
    assert len(loaded.ai_commentary) == 1

    bundle_path = tmp_path / "system_bundle.jsession"
    source_exporter.export_session_bundle(source_session_id, str(bundle_path))
    assert bundle_path.exists()

    target_db = tmp_path / "system_target.db"
    target_exporter = SessionExporter(db_path=str(target_db))
    imported_session_id = target_exporter.import_session_bundle(str(bundle_path))

    assert imported_session_id > 0
    target_sessions = target_exporter.list_sessions()
    assert len(target_sessions) == 1
    assert target_sessions[0]["track_name"] == "Silverstone"
