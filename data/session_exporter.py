"""
Programmatic session exporter - exports SQLite session data to CSV files
for consumption by the post-race telemetry analyzer.
"""
import csv
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
                ai_enabled
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
            })

        db.close()
        return sessions

    def get_last_session_id(self) -> Optional[int]:
        """Get the most recent session ID."""
        sessions = self.list_sessions()
        if sessions:
            return sessions[0]["session_id"]
        return None

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
        cursor.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
        if not cursor.fetchone():
            db.close()
            raise ValueError(f"Session {session_id} not found in database")

        # Export telemetry
        self._export_telemetry(db, session_id, output_path)

        # Export laps
        self._export_laps(db, session_id, output_path)

        # Export AI commentary (if exists)
        self._export_ai_commentary(db, session_id, output_path)

        db.close()
        logger.info("Session %d exported to %s", session_id, output_path)
        return str(output_path)

    def _export_telemetry(self, db: sqlite3.Connection, session_id: int, output_path: Path) -> None:
        """Export telemetry samples to CSV."""
        cursor = db.cursor()
        cursor.execute("""
            SELECT * FROM telemetry
            WHERE session_id = ?
            ORDER BY lap_number, elapsed_time
        """, (session_id,))

        telemetry_file = output_path / f"session_{session_id}_telemetry.csv"
        with open(telemetry_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "lap_number", "elapsed_time", "pos_x", "pos_z", "speed", "gear", "rpm",
                "throttle", "brake", "fuel",
                "tyre_pressure_fl", "tyre_pressure_fr", "tyre_pressure_rl", "tyre_pressure_rr",
                "tyre_temp_fl", "tyre_temp_fr", "tyre_temp_rl", "tyre_temp_rr"
            ])

            for row in cursor:
                writer.writerow([
                    row["lap_number"], row["elapsed_time"], row["pos_x"], row["pos_z"],
                    row["speed"], row["gear"], row["rpm"], row["throttle"], row["brake"],
                    row["fuel"],
                    row["tyre_pressure_fl"], row["tyre_pressure_fr"],
                    row["tyre_pressure_rl"], row["tyre_pressure_rr"],
                    row["tyre_temp_fl"], row["tyre_temp_fr"],
                    row["tyre_temp_rl"], row["tyre_temp_rr"]
                ])

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
