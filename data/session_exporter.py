"""
Programmatic session exporter - exports SQLite session data to CSV files
for consumption by the post-race telemetry analyzer.
"""
import csv
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List

logger = logging.getLogger(__name__)


class SessionExporter:
    """Export recorded sessions from SQLite to CSV files."""

    def __init__(self, db_path: str = "data/telemetry_sessions.db"):
        self.db_path = db_path

    def list_sessions(self) -> List[dict]:
        """
        List all recorded sessions with metadata.

        Returns:
            List of dicts with session info (session_id, start_time, track, car, etc.)
        """
        if not Path(self.db_path).exists():
            return []

        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

        # Ensure session_type column exists (migration for older DBs)
        try:
            cursor.execute("ALTER TABLE sessions ADD COLUMN session_type TEXT DEFAULT ''")
            db.commit()
        except sqlite3.OperationalError:
            pass  # column already exists

        cursor.execute("""
            SELECT
                session_id,
                start_time,
                end_time,
                game,
                track_name,
                car_model,
                player_name,
                total_laps,
                best_lap_time,
                ai_enabled,
                COALESCE(session_type, '') as session_type,
                COALESCE(notes, '') as notes
            FROM sessions
            WHERE total_laps > 0
            ORDER BY start_time DESC
        """)

        sessions = []
        for row in cursor.fetchall():
            start_dt = None
            if row["start_time"]:
                try:
                    start_dt = datetime.fromtimestamp(row["start_time"])
                except (OSError, ValueError):
                    pass

            end_dt = None
            if row["end_time"]:
                try:
                    end_dt = datetime.fromtimestamp(row["end_time"])
                except (OSError, ValueError):
                    pass

            sessions.append({
                "session_id": row["session_id"],
                "start_time": start_dt,
                "end_time": end_dt,
                "game": row["game"] or "unknown",
                "track_name": row["track_name"] or "Unknown Track",
                "car_model": row["car_model"] or "Unknown Car",
                "player_name": row["player_name"] or "Unknown",
                "total_laps": row["total_laps"] or 0,
                "best_lap_time": row["best_lap_time"],
                "ai_enabled": bool(row["ai_enabled"]),
                "session_type": row["session_type"] or "",
                "notes": row["notes"] or "",
            })

        db.close()
        return sessions

    def get_last_session_id(self) -> Optional[int]:
        """Get the most recent session ID."""
        sessions = self.list_sessions()
        if sessions:
            return sessions[0]["session_id"]
        return None

    def delete_session(self, session_id: int) -> None:
        """Delete a session and all its related data from the database."""
        if not Path(self.db_path).exists():
            return

        db = sqlite3.connect(self.db_path)
        cursor = db.cursor()

        cursor.execute("DELETE FROM telemetry WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM laps WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM ai_commentary WHERE session_id = ?", (session_id,))
        try:
            cursor.execute("DELETE FROM voice_queries WHERE session_id = ?", (session_id,))
        except Exception:
            pass
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

        db.commit()
        db.close()
        logger.info("Session %d deleted", session_id)

    def export_session(self, session_id: int, output_dir: Optional[str] = None) -> str:
        """
        Export a session to CSV files.

        Args:
            session_id: Session ID to export
            output_dir: Output directory (defaults to data/exports)

        Returns:
            Path to the output directory containing CSVs

        Raises:
            FileNotFoundError: If database not found
            ValueError: If session not found
        """
        if not Path(self.db_path).exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        if output_dir is None:
            output_dir = f"data/exports/session_{session_id}"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row

        # Verify session exists
        cursor = db.cursor()
        cursor.execute(
            """
            SELECT session_id, game, track_name, car_model, player_name,
                   start_time, end_time, total_laps, ai_enabled, notes
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,),
        )
        session_row = cursor.fetchone()
        if not session_row:
            db.close()
            raise ValueError(f"Session {session_id} not found in database")

        # Export telemetry
        self._export_telemetry(db, session_id, output_path)

        # Export laps
        self._export_laps(db, session_id, output_path)

        # Export AI commentary (if exists)
        self._export_ai_commentary(db, session_id, output_path)
        self._export_metadata(session_row, output_path)

        db.close()
        logger.info("Session %d exported to %s", session_id, output_path)
        return str(output_path)

    def _export_metadata(self, session_row: sqlite3.Row, output_path: Path) -> None:
        """Export session metadata to JSON sidecar used by TelemetryLoader."""
        session_id = int(session_row["session_id"])
        metadata_file = output_path / f"session_{session_id}_metadata.json"
        payload = {
            "session_id": session_id,
            "game": session_row["game"] or "unknown",
            "track_name": session_row["track_name"] or "Unknown Track",
            "car_model": session_row["car_model"] or "Unknown Car",
            "player_name": session_row["player_name"] or "Unknown Player",
            "start_time": session_row["start_time"],
            "end_time": session_row["end_time"],
            "total_laps": int(session_row["total_laps"] or 0),
            "ai_enabled": bool(session_row["ai_enabled"]),
            "notes": session_row["notes"] or "",
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def _export_telemetry(self, db: sqlite3.Connection, session_id: int, output_path: Path) -> None:
        """Export telemetry samples to CSV."""
        cursor = db.cursor()
        cursor.execute("""
            SELECT * FROM telemetry
            WHERE session_id = ?
            ORDER BY lap_number, elapsed_time
        """, (session_id,))

        telemetry_file = output_path / f"session_{session_id}_telemetry.csv"

        # Get column names from the query result
        rows = cursor.fetchall()
        if not rows:
            return

        # Export all columns except internal ones (telemetry_id, session_id, timestamp)
        skip_cols = {"telemetry_id", "session_id", "timestamp"}
        col_names = [key for key in rows[0].keys() if key not in skip_cols]

        with open(telemetry_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(col_names)
            for row in rows:
                writer.writerow([row[col] for col in col_names])

    def _export_laps(self, db: sqlite3.Connection, session_id: int, output_path: Path) -> None:
        """Export lap summaries to CSV."""
        cursor = db.cursor()
        cursor.execute("""
            SELECT * FROM laps
            WHERE session_id = ?
            ORDER BY lap_number
        """, (session_id,))

        laps_file = output_path / f"session_{session_id}_laps.csv"
        with open(laps_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["lap_number", "lap_time", "fuel_start", "fuel_end",
                             "avg_speed", "max_speed", "valid"])

            for row in cursor:
                writer.writerow([
                    row["lap_number"], row["lap_time"], row["fuel_start"], row["fuel_end"],
                    row["avg_speed"], row["max_speed"], row["valid"]
                ])

    def rename_session(self, session_id: int, new_name: str) -> None:
        """Rename a session by setting its notes field."""
        if not Path(self.db_path).exists():
            return
        db = sqlite3.connect(self.db_path)
        cursor = db.cursor()
        cursor.execute("UPDATE sessions SET notes = ? WHERE session_id = ?", (new_name, session_id))
        db.commit()
        db.close()
        logger.info("Session %d renamed to '%s'", session_id, new_name)

    def export_session_bundle(self, session_id: int, output_file: str) -> None:
        """
        Export a full session (all laps + metadata) to a .jsession file for sharing.

        Args:
            session_id: Session to export
            output_file: Path for the .jsession file
        """
        if not Path(self.db_path).exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

        # Session metadata
        cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
        session_row = cursor.fetchone()
        if not session_row:
            db.close()
            raise ValueError(f"Session {session_id} not found")

        bundle = {
            "version": 1,
            "metadata": {
                "game": session_row["game"],
                "track_name": session_row["track_name"],
                "car_model": session_row["car_model"],
                "player_name": session_row["player_name"],
                "total_laps": session_row["total_laps"],
                "best_lap_time": session_row["best_lap_time"],
                "notes": session_row["notes"] or "",
            },
        }

        # Laps
        cursor.execute("""
            SELECT lap_number, lap_time, fuel_start, fuel_end, avg_speed, max_speed, min_speed, valid
            FROM laps WHERE session_id = ? ORDER BY lap_number
        """, (session_id,))
        bundle["laps"] = [dict(row) for row in cursor.fetchall()]

        # Telemetry
        cursor.execute("SELECT * FROM telemetry WHERE session_id = ? ORDER BY lap_number, elapsed_time", (session_id,))
        rows = cursor.fetchall()
        if rows:
            skip_cols = {"telemetry_id", "session_id", "timestamp"}
            col_names = [k for k in rows[0].keys() if k not in skip_cols]
            bundle["telemetry"] = {col: [row[col] for row in rows] for col in col_names}
        else:
            bundle["telemetry"] = {}

        # AI commentary
        cursor.execute("""
            SELECT timestamp, message, trigger, priority, lap_number
            FROM ai_commentary WHERE session_id = ? ORDER BY timestamp
        """, (session_id,))
        ai_rows = cursor.fetchall()
        bundle["ai_commentary"] = [dict(row) for row in ai_rows]

        db.close()

        with open(output_file, 'w') as f:
            json.dump(bundle, f)

        logger.info("Session %d exported to %s", session_id, output_file)

    def import_session_bundle(self, input_file: str) -> int:
        """
        Import a .jsession file into the database.

        Args:
            input_file: Path to the .jsession file

        Returns:
            The new session_id assigned to the imported session
        """
        with open(input_file, 'r') as f:
            bundle = json.load(f)

        meta = bundle["metadata"]
        import time as _time

        db = sqlite3.connect(self.db_path)
        cursor = db.cursor()

        # Ensure tables exist (in case DB is fresh)
        self._ensure_import_tables(cursor)

        # Insert session
        now = _time.time()
        cursor.execute("""
            INSERT INTO sessions (start_time, game, track_name, car_model, player_name,
                                  total_laps, best_lap_time, ai_enabled, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
        """, (
            now, meta.get("game", "unknown"), meta.get("track_name", ""),
            meta.get("car_model", ""), meta.get("player_name", ""),
            meta.get("total_laps", 0), meta.get("best_lap_time"),
            meta.get("notes", f"Imported from {Path(input_file).name}"),
        ))
        session_id = cursor.lastrowid

        # Insert laps
        for lap in bundle.get("laps", []):
            cursor.execute("""
                INSERT INTO laps (session_id, lap_number, lap_time, fuel_start, fuel_end,
                                  avg_speed, max_speed, min_speed, valid, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, lap["lap_number"], lap.get("lap_time"),
                lap.get("fuel_start"), lap.get("fuel_end"),
                lap.get("avg_speed"), lap.get("max_speed"), lap.get("min_speed", 0),
                lap.get("valid", 1), now,
            ))

        # Insert telemetry
        telem = bundle.get("telemetry", {})
        if telem:
            table_columns = self._get_table_columns(cursor, "telemetry")
            col_names = [name for name in telem.keys() if name in table_columns]
            num_rows = len(telem[col_names[0]]) if col_names else 0
            for i in range(num_rows):
                row_vals = {col: telem[col][i] for col in col_names}
                row_vals["session_id"] = session_id
                row_vals["timestamp"] = now
                cols = list(row_vals.keys())
                placeholders = ", ".join(["?"] * len(cols))
                cursor.execute(
                    f"INSERT INTO telemetry ({', '.join(cols)}) VALUES ({placeholders})",
                    [row_vals[c] for c in cols]
                )

        # Insert AI commentary
        for comment in bundle.get("ai_commentary", []):
            cursor.execute("""
                INSERT INTO ai_commentary (session_id, timestamp, message, trigger, priority, lap_number)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id, comment.get("timestamp", now), comment.get("message", ""),
                comment.get("trigger", ""), comment.get("priority", 2),
                comment.get("lap_number", 0),
            ))

        db.commit()
        db.close()
        logger.info("Imported session from %s as session_id=%d", input_file, session_id)
        return session_id

    def _ensure_import_tables(self, cursor: sqlite3.Cursor) -> None:
        """Create all tables required by import_session_bundle()."""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time REAL NOT NULL,
                end_time REAL,
                game TEXT NOT NULL,
                track_name TEXT,
                car_model TEXT,
                player_name TEXT,
                total_laps INTEGER DEFAULT 0,
                best_lap_time REAL,
                total_distance REAL DEFAULT 0.0,
                ai_enabled INTEGER DEFAULT 0,
                notes TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS laps (
                lap_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                lap_number INTEGER NOT NULL,
                lap_time REAL,
                sector1_time REAL,
                sector2_time REAL,
                sector3_time REAL,
                valid INTEGER DEFAULT 1,
                fuel_start REAL,
                fuel_end REAL,
                avg_speed REAL,
                max_speed REAL,
                min_speed REAL,
                timestamp REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                telemetry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                lap_number INTEGER NOT NULL,
                elapsed_time REAL NOT NULL,
                pos_x REAL NOT NULL,
                pos_z REAL NOT NULL,
                speed REAL NOT NULL,
                gear INTEGER,
                rpm INTEGER,
                throttle REAL,
                brake REAL,
                fuel REAL,
                steer_angle REAL,
                g_force_lat REAL,
                g_force_lon REAL,
                tyre_pressure_fl REAL,
                tyre_pressure_fr REAL,
                tyre_pressure_rl REAL,
                tyre_pressure_rr REAL,
                tyre_temp_fl REAL,
                tyre_temp_fr REAL,
                tyre_temp_rl REAL,
                tyre_temp_rr REAL,
                tyre_wear_fl REAL,
                tyre_wear_fr REAL,
                tyre_wear_rl REAL,
                tyre_wear_rr REAL,
                wheel_slip_fl REAL,
                wheel_slip_fr REAL,
                wheel_slip_rl REAL,
                wheel_slip_rr REAL,
                suspension_fl REAL,
                suspension_fr REAL,
                suspension_rl REAL,
                suspension_rr REAL,
                camber_fl REAL,
                camber_fr REAL,
                camber_rl REAL,
                camber_rr REAL,
                ride_height_front REAL,
                ride_height_rear REAL,
                car_damage_front REAL,
                car_damage_rear REAL,
                car_damage_left REAL,
                car_damage_right REAL,
                car_damage_centre REAL,
                is_in_pit INTEGER,
                pit_limiter INTEGER,
                timestamp REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ai_commentary (
                commentary_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                message TEXT NOT NULL,
                trigger TEXT,
                priority TEXT,
                lap_number INTEGER,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)

        # Ensure newer telemetry columns exist in older target databases.
        required_telemetry_cols = [
            ("steer_angle", "REAL"),
            ("g_force_lat", "REAL"),
            ("g_force_lon", "REAL"),
            ("tyre_wear_fl", "REAL"),
            ("tyre_wear_fr", "REAL"),
            ("tyre_wear_rl", "REAL"),
            ("tyre_wear_rr", "REAL"),
            ("wheel_slip_fl", "REAL"),
            ("wheel_slip_fr", "REAL"),
            ("wheel_slip_rl", "REAL"),
            ("wheel_slip_rr", "REAL"),
            ("suspension_fl", "REAL"),
            ("suspension_fr", "REAL"),
            ("suspension_rl", "REAL"),
            ("suspension_rr", "REAL"),
            ("camber_fl", "REAL"),
            ("camber_fr", "REAL"),
            ("camber_rl", "REAL"),
            ("camber_rr", "REAL"),
            ("ride_height_front", "REAL"),
            ("ride_height_rear", "REAL"),
            ("car_damage_front", "REAL"),
            ("car_damage_rear", "REAL"),
            ("car_damage_left", "REAL"),
            ("car_damage_right", "REAL"),
            ("car_damage_centre", "REAL"),
            ("is_in_pit", "INTEGER"),
            ("pit_limiter", "INTEGER"),
        ]
        existing = self._get_table_columns(cursor, "telemetry")
        for col_name, col_type in required_telemetry_cols:
            if col_name in existing:
                continue
            cursor.execute(f"ALTER TABLE telemetry ADD COLUMN {col_name} {col_type}")
            existing.add(col_name)

    @staticmethod
    def _get_table_columns(cursor: sqlite3.Cursor, table_name: str) -> set[str]:
        """Return column names for an existing SQLite table."""
        cursor.execute(f"PRAGMA table_info({table_name})")
        rows = cursor.fetchall()
        return {row[1] for row in rows}

    def _export_ai_commentary(self, db: sqlite3.Connection, session_id: int, output_path: Path) -> None:
        """Export AI commentary to CSV (if any exists)."""
        cursor = db.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM ai_commentary WHERE session_id = ?", (session_id,))
        if cursor.fetchone()["count"] == 0:
            return

        cursor.execute("""
            SELECT timestamp, message, trigger, priority, lap_number
            FROM ai_commentary
            WHERE session_id = ?
            ORDER BY timestamp
        """, (session_id,))

        ai_file = output_path / f"session_{session_id}_ai_commentary.csv"
        with open(ai_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "message", "trigger", "priority", "lap_number"])

            for row in cursor:
                writer.writerow([row["timestamp"], row["message"], row["trigger"],
                                 row["priority"], row["lap_number"]])
