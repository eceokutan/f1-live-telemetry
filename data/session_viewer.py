"""
Session Viewer for F1 Telemetry Dashboard.

Command-line utility to query and view recorded sessions.
Useful for post-race analysis and data export.

Usage:
    python -m data.session_viewer list                    # List all sessions
    python -m data.session_viewer info <session_id>       # View session details
    python -m data.session_viewer laps <session_id>       # View lap times
    python -m data.session_viewer export <session_id>     # Export session to CSV
    python -m data.session_viewer replay <session_id>     # Show telemetry summary
"""

import sqlite3
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class SessionViewer:
    """Query and display recorded telemetry sessions."""

    def __init__(self, db_path: str = "data/telemetry_sessions.db"):
        """
        Initialize session viewer.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path

        if not Path(db_path).exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row

    def list_sessions(self):
        """List all recorded sessions."""
        cursor = self.db.cursor()
        cursor.execute("""
            SELECT
                session_id,
                datetime(start_time, 'unixepoch', 'localtime') as start_time,
                game,
                track_name,
                car_model,
                total_laps,
                best_lap_time
            FROM sessions
            ORDER BY start_time DESC
        """)

        rows = cursor.fetchall()

        if not rows:
            print("No sessions found in database.")
            return

        print("\n" + "="*100)
        print("RECORDED SESSIONS")
        print("="*100)

        for row in rows:
            session_id = row["session_id"]
            start_time = row["start_time"]
            game = row["game"].upper()
            track = row["track_name"] or "Unknown Track"
            car = row["car_model"] or "Unknown Car"
            laps = row["total_laps"]
            best_lap = row["best_lap_time"]

            best_lap_str = f"{best_lap:.3f}s" if best_lap else "N/A"

            print(f"\nSession {session_id}: {start_time}")
            print(f"  Game: {game} | Track: {track} | Car: {car}")
            print(f"  Laps: {laps} | Best Lap: {best_lap_str}")

        print("\n" + "="*100 + "\n")

    def session_info(self, session_id: int):
        """
        Display detailed session information.

        Args:
            session_id: Session ID
        """
        cursor = self.db.cursor()

        # Get session metadata
        cursor.execute("""
            SELECT
                session_id,
                datetime(start_time, 'unixepoch', 'localtime') as start_time,
                datetime(end_time, 'unixepoch', 'localtime') as end_time,
                game,
                track_name,
                car_model,
                player_name,
                total_laps,
                best_lap_time,
                total_distance,
                notes
            FROM sessions
            WHERE session_id = ?
        """, (session_id,))

        session = cursor.fetchone()

        if not session:
            print(f"Session {session_id} not found.")
            return

        print("\n" + "="*100)
        print(f"SESSION {session_id} - DETAILS")
        print("="*100)

        print(f"\nGame: {session['game'].upper()}")
        print(f"Track: {session['track_name'] or 'Unknown'}")
        print(f"Car: {session['car_model'] or 'Unknown'}")
        print(f"Player: {session['player_name'] or 'Unknown'}")
        print(f"\nStart Time: {session['start_time']}")
        print(f"End Time: {session['end_time'] or 'In Progress'}")
        print(f"\nTotal Laps: {session['total_laps']}")
        print(f"Best Lap Time: {session['best_lap_time']:.3f}s" if session['best_lap_time'] else "N/A")
        print(f"Total Distance: {session['total_distance']:.2f}m" if session['total_distance'] else "N/A")

        # Count telemetry samples
        cursor.execute("SELECT COUNT(*) as count FROM telemetry WHERE session_id = ?", (session_id,))
        telemetry_count = cursor.fetchone()["count"]
        print(f"\nTelemetry Samples: {telemetry_count:,}")

        # Count AI commentary
        cursor.execute("SELECT COUNT(*) as count FROM ai_commentary WHERE session_id = ?", (session_id,))
        ai_count = cursor.fetchone()["count"]
        if ai_count > 0:
            print(f"AI Commentary Messages: {ai_count}")

        # Count voice queries
        cursor.execute("SELECT COUNT(*) as count FROM voice_queries WHERE session_id = ?", (session_id,))
        voice_count = cursor.fetchone()["count"]
        if voice_count > 0:
            print(f"Voice Queries: {voice_count}")

        print("\n" + "="*100 + "\n")

    def session_laps(self, session_id: int):
        """
        Display lap times for a session.

        Args:
            session_id: Session ID
        """
        cursor = self.db.cursor()

        cursor.execute("""
            SELECT
                lap_number,
                lap_time,
                fuel_start,
                fuel_end,
                avg_speed,
                max_speed,
                valid
            FROM laps
            WHERE session_id = ?
            ORDER BY lap_number
        """, (session_id,))

        laps = cursor.fetchall()

        if not laps:
            print(f"No laps found for session {session_id}.")
            return

        print("\n" + "="*100)
        print(f"SESSION {session_id} - LAP TIMES")
        print("="*100)

        print(f"\n{'Lap':<6} {'Time':<10} {'Fuel Used':<12} {'Avg Speed':<12} {'Max Speed':<12} {'Valid':<6}")
        print("-"*100)

        for lap in laps:
            lap_num = lap["lap_number"]
            lap_time = f"{lap['lap_time']:.3f}s" if lap["lap_time"] else "N/A"

            fuel_used = ""
            if lap["fuel_start"] and lap["fuel_end"]:
                used = lap["fuel_start"] - lap["fuel_end"]
                fuel_used = f"{used:.2f}L"

            avg_speed = f"{lap['avg_speed']:.1f} km/h" if lap["avg_speed"] else "N/A"
            max_speed = f"{lap['max_speed']:.1f} km/h" if lap["max_speed"] else "N/A"
            valid = "Yes" if lap["valid"] else "No"

            print(f"{lap_num:<6} {lap_time:<10} {fuel_used:<12} {avg_speed:<12} {max_speed:<12} {valid:<6}")

        # Calculate best lap
        valid_laps = [lap for lap in laps if lap["valid"] and lap["lap_time"]]
        if valid_laps:
            best_lap = min(valid_laps, key=lambda x: x["lap_time"])
            print("\n" + "-"*100)
            print(f"Best Lap: Lap {best_lap['lap_number']} - {best_lap['lap_time']:.3f}s")

        print("\n" + "="*100 + "\n")

    def export_session(self, session_id: int, output_dir: str = "data/exports"):
        """
        Export session data to CSV files.

        Args:
            session_id: Session ID
            output_dir: Output directory for CSV files
        """
        import csv

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cursor = self.db.cursor()

        # Export telemetry
        cursor.execute("""
            SELECT * FROM telemetry
            WHERE session_id = ?
            ORDER BY lap_number, elapsed_time
        """, (session_id,))

        telemetry_file = output_path / f"session_{session_id}_telemetry.csv"
        with open(telemetry_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                "lap_number", "elapsed_time", "pos_x", "pos_z", "speed", "gear", "rpm",
                "throttle", "brake", "fuel",
                "tyre_pressure_fl", "tyre_pressure_fr", "tyre_pressure_rl", "tyre_pressure_rr",
                "tyre_temp_fl", "tyre_temp_fr", "tyre_temp_rl", "tyre_temp_rr"
            ])

            # Write data
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

        print(f"✅ Telemetry exported to: {telemetry_file}")

        # Export laps
        cursor.execute("""
            SELECT * FROM laps
            WHERE session_id = ?
            ORDER BY lap_number
        """, (session_id,))

        laps_file = output_path / f"session_{session_id}_laps.csv"
        with open(laps_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["lap_number", "lap_time", "fuel_start", "fuel_end", "avg_speed", "max_speed", "valid"])

            for row in cursor:
                writer.writerow([
                    row["lap_number"], row["lap_time"], row["fuel_start"], row["fuel_end"],
                    row["avg_speed"], row["max_speed"], row["valid"]
                ])

        print(f"✅ Laps exported to: {laps_file}")

        # Export AI commentary (if exists)
        cursor.execute("SELECT COUNT(*) as count FROM ai_commentary WHERE session_id = ?", (session_id,))
        if cursor.fetchone()["count"] > 0:
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
                    writer.writerow([row["timestamp"], row["message"], row["trigger"], row["priority"], row["lap_number"]])

            print(f"✅ AI commentary exported to: {ai_file}")

        print(f"\n✅ Session {session_id} exported to {output_dir}/")

    def close(self):
        """Close database connection."""
        if self.db:
            self.db.close()


def main():
    """Command-line interface."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    try:
        viewer = SessionViewer()

        if command == "list":
            viewer.list_sessions()

        elif command == "info":
            if len(sys.argv) < 3:
                print("Usage: python -m data.session_viewer info <session_id>")
                sys.exit(1)
            session_id = int(sys.argv[2])
            viewer.session_info(session_id)

        elif command == "laps":
            if len(sys.argv) < 3:
                print("Usage: python -m data.session_viewer laps <session_id>")
                sys.exit(1)
            session_id = int(sys.argv[2])
            viewer.session_laps(session_id)

        elif command == "export":
            if len(sys.argv) < 3:
                print("Usage: python -m data.session_viewer export <session_id>")
                sys.exit(1)
            session_id = int(sys.argv[2])
            viewer.export_session(session_id)

        else:
            print(f"Unknown command: {command}")
            print(__doc__)
            sys.exit(1)

        viewer.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
