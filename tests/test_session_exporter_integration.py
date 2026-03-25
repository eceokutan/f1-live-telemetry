from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from data.csv_importer import ensure_tables
from data.session_exporter import SessionExporter
from data.telemetry_loader import TelemetryLoader

pytestmark = [pytest.mark.integration, pytest.mark.regression]


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
        (now - 60.0, now, "ac", "Spa", "F1", "Driver", 1, 88.0, 1, "integration source"),
    )
    session_id = cursor.lastrowid
    cursor.execute(
        """
        INSERT INTO laps (
            session_id, lap_number, lap_time, fuel_start, fuel_end,
            avg_speed, max_speed, min_speed, valid, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (session_id, 1, 88.0, 18.0, 16.4, 165.0, 300.0, 95.0, 1, now),
    )
    cursor.execute(
        """
        INSERT INTO telemetry (
            session_id, lap_number, elapsed_time, pos_x, pos_z, speed,
            gear, rpm, throttle, brake, fuel,
            tyre_pressure_fl, tyre_pressure_fr, tyre_pressure_rl, tyre_pressure_rr,
            tyre_temp_fl, tyre_temp_fr, tyre_temp_rl, tyre_temp_rr, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (session_id, 1, 0.0, 1.0, 1.0, 120.0, 4, 9500, 0.6, 0.0, 18.0, 26.0, 26.1, 25.8, 25.9, 94.0, 95.0, 93.0, 94.0, now),
    )
    cursor.execute(
        """
        INSERT INTO telemetry (
            session_id, lap_number, elapsed_time, pos_x, pos_z, speed,
            gear, rpm, throttle, brake, fuel,
            tyre_pressure_fl, tyre_pressure_fr, tyre_pressure_rl, tyre_pressure_rr,
            tyre_temp_fl, tyre_temp_fr, tyre_temp_rl, tyre_temp_rr, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (session_id, 1, 0.5, 1.2, 1.1, 140.0, 5, 10000, 0.8, 0.1, 17.8, 26.0, 26.1, 25.8, 25.9, 95.0, 96.0, 94.0, 95.0, now),
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
    return int(session_id)


def test_exported_csv_roundtrip_loads_into_session_object(tmp_path):
    db_path = tmp_path / "source.db"
    session_id = _seed_db(db_path)
    exporter = SessionExporter(db_path=str(db_path))

    export_dir = tmp_path / "csv_export"
    exporter.export_session(session_id, output_dir=str(export_dir))

    session = TelemetryLoader.load_session(str(export_dir))

    assert session.metadata.track_name == "Spa"
    assert session.metadata.total_laps == 1
    assert len(session.laps) == 1
    assert len(session.ai_commentary) == 1
    assert session.get_fastest_lap() is not None


def test_bundle_export_then_import_works_with_fresh_database(tmp_path):
    source_db = tmp_path / "source.db"
    source_session_id = _seed_db(source_db)
    source_exporter = SessionExporter(db_path=str(source_db))

    bundle_path = tmp_path / "session.jsession"
    source_exporter.export_session_bundle(source_session_id, str(bundle_path))
    assert bundle_path.exists()

    target_db = tmp_path / "target.db"
    target_exporter = SessionExporter(db_path=str(target_db))
    imported_session_id = target_exporter.import_session_bundle(str(bundle_path))

    assert imported_session_id > 0
    sessions = target_exporter.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["track_name"] == "Spa"


def test_bundle_import_supports_camber_columns(tmp_path):
    source_db = tmp_path / "source_with_camber.db"
    source_session_id = _seed_db(source_db)

    source_conn = sqlite3.connect(str(source_db))
    source_conn.row_factory = sqlite3.Row
    source_cursor = source_conn.cursor()
    source_cursor.execute("ALTER TABLE telemetry ADD COLUMN camber_fl REAL")
    source_cursor.execute("ALTER TABLE telemetry ADD COLUMN camber_fr REAL")
    source_cursor.execute("ALTER TABLE telemetry ADD COLUMN camber_rl REAL")
    source_cursor.execute("ALTER TABLE telemetry ADD COLUMN camber_rr REAL")
    source_cursor.execute(
        """
        UPDATE telemetry
        SET camber_fl = ?, camber_fr = ?, camber_rl = ?, camber_rr = ?
        WHERE session_id = ?
        """,
        (-3.1, -3.2, -2.4, -2.5, source_session_id),
    )
    source_conn.commit()
    source_conn.close()

    source_exporter = SessionExporter(db_path=str(source_db))
    bundle_path = tmp_path / "session_with_camber.jsession"
    source_exporter.export_session_bundle(source_session_id, str(bundle_path))

    target_db = tmp_path / "target_with_camber.db"
    target_exporter = SessionExporter(db_path=str(target_db))
    imported_session_id = target_exporter.import_session_bundle(str(bundle_path))

    assert imported_session_id > 0

    target_conn = sqlite3.connect(str(target_db))
    target_conn.row_factory = sqlite3.Row
    target_cursor = target_conn.cursor()
    target_cursor.execute("PRAGMA table_info(telemetry)")
    telemetry_cols = {row["name"] for row in target_cursor.fetchall()}
    assert {"camber_fl", "camber_fr", "camber_rl", "camber_rr"}.issubset(telemetry_cols)

    target_cursor.execute(
        """
        SELECT camber_fl, camber_fr, camber_rl, camber_rr
        FROM telemetry
        WHERE session_id = ?
        ORDER BY telemetry_id ASC
        LIMIT 1
        """,
        (imported_session_id,),
    )
    row = target_cursor.fetchone()
    target_conn.close()

    assert row is not None
    assert row["camber_fl"] == pytest.approx(-3.1)
    assert row["camber_fr"] == pytest.approx(-3.2)
    assert row["camber_rl"] == pytest.approx(-2.4)
    assert row["camber_rr"] == pytest.approx(-2.5)
