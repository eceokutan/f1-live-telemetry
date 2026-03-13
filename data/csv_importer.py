"""
Import sample CSV session data into the SQLite database.

Usage:
    python -m data.csv_importer
"""
import csv
import sqlite3
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = "data/telemetry_sessions.db"
SAMPLE_DIR = Path("sample_data")


def ensure_tables(db: sqlite3.Connection):
    """Create tables if they don't exist (mirrors SessionRecorder schema)."""
    cursor = db.cursor()

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
        tyre_pressure_fl REAL,
        tyre_pressure_fr REAL,
        tyre_pressure_rl REAL,
        tyre_pressure_rr REAL,
        tyre_temp_fl REAL,
        tyre_temp_fr REAL,
        tyre_temp_rl REAL,
        tyre_temp_rr REAL,
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

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS voice_queries (
        query_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        timestamp REAL NOT NULL,
        query_text TEXT NOT NULL,
        response_text TEXT,
        lap_number INTEGER,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_laps_session ON laps(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_session ON telemetry(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_lap ON telemetry(lap_number)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_session ON ai_commentary(session_id)")

    db.commit()


def import_session(db: sqlite3.Connection, telemetry_csv: Path, laps_csv: Path,
                   ai_csv: Path = None, track_name: str = "Unknown Track",
                   car_model: str = "Unknown Car") -> int:
    """Import a single session from CSV files into the database."""
    cursor = db.cursor()
    now = time.time()

    # Read laps to compute session stats
    laps = []
    with open(laps_csv, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            laps.append(row)

    total_laps = len(laps)
    best_lap_time = None
    if laps:
        valid_times = [float(l["lap_time"]) for l in laps if int(l["valid"]) == 1]
        if valid_times:
            best_lap_time = min(valid_times)

    # Estimate start/end times from AI commentary timestamps if available
    start_time = now - 3600  # default: 1 hour ago
    end_time = now - 3000
    if ai_csv and ai_csv.exists():
        try:
            with open(ai_csv, newline='', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                timestamps = []
                for row in reader:
                    try:
                        timestamps.append(float(row["timestamp"]))
                    except (ValueError, KeyError):
                        pass
                if timestamps:
                    start_time = min(timestamps) - 60
                    end_time = max(timestamps) + 60
        except Exception:
            pass

    # Determine if AI was enabled
    ai_enabled = 1 if (ai_csv and ai_csv.exists()) else 0

    # Insert session
    cursor.execute("""
    INSERT INTO sessions (start_time, end_time, game, track_name, car_model,
                          player_name, total_laps, best_lap_time, ai_enabled)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (start_time, end_time, "ac", track_name, car_model,
          "Sample Driver", total_laps, best_lap_time, ai_enabled))
    session_id = cursor.lastrowid

    # Import laps
    for lap in laps:
        cursor.execute("""
        INSERT INTO laps (session_id, lap_number, lap_time, fuel_start, fuel_end,
                          avg_speed, max_speed, valid, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            int(lap["lap_number"]),
            float(lap["lap_time"]),
            float(lap["fuel_start"]),
            float(lap["fuel_end"]),
            float(lap["avg_speed"]),
            float(lap["max_speed"]),
            int(lap["valid"]),
            now,
        ))

    # Import telemetry in batches
    batch = []
    with open(telemetry_csv, newline='', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch.append((
                session_id,
                int(row["lap_number"]),
                float(row["elapsed_time"]),
                float(row["pos_x"]),
                float(row["pos_z"]),
                float(row["speed"]),
                int(row["gear"]),
                int(row["rpm"]),
                float(row["throttle"]),
                float(row["brake"]),
                float(row["fuel"]),
                float(row["tyre_pressure_fl"]),
                float(row["tyre_pressure_fr"]),
                float(row["tyre_pressure_rl"]),
                float(row["tyre_pressure_rr"]),
                float(row["tyre_temp_fl"]),
                float(row["tyre_temp_fr"]),
                float(row["tyre_temp_rl"]),
                float(row["tyre_temp_rr"]),
                now,
            ))

            if len(batch) >= 1000:
                cursor.executemany("""
                INSERT INTO telemetry (
                    session_id, lap_number, elapsed_time, pos_x, pos_z, speed,
                    gear, rpm, throttle, brake, fuel,
                    tyre_pressure_fl, tyre_pressure_fr, tyre_pressure_rl, tyre_pressure_rr,
                    tyre_temp_fl, tyre_temp_fr, tyre_temp_rl, tyre_temp_rr, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch)
                batch.clear()

    if batch:
        cursor.executemany("""
        INSERT INTO telemetry (
            session_id, lap_number, elapsed_time, pos_x, pos_z, speed,
            gear, rpm, throttle, brake, fuel,
            tyre_pressure_fl, tyre_pressure_fr, tyre_pressure_rl, tyre_pressure_rr,
            tyre_temp_fl, tyre_temp_fr, tyre_temp_rl, tyre_temp_rr, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, batch)

    # Import AI commentary
    if ai_csv and ai_csv.exists():
        with open(ai_csv, newline='', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    cursor.execute("""
                    INSERT INTO ai_commentary (session_id, timestamp, message, trigger, priority, lap_number)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        session_id,
                        float(row["timestamp"]),
                        row["message"],
                        row["trigger"],
                        row["priority"],
                        int(row["lap_number"]),
                    ))
                except (ValueError, KeyError):
                    continue

    db.commit()
    return session_id


def main():
    """Import all sample sessions."""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    ensure_tables(db)

    # Check if sessions already imported
    cursor = db.cursor()
    cursor.execute("SELECT COUNT(*) FROM sessions")
    existing = cursor.fetchone()[0]
    if existing > 0:
        print(f"Database already has {existing} session(s). Skipping import.")
        print("Delete data/telemetry_sessions.db and re-run to reimport.")
        db.close()
        return

    sessions_to_import = [
        {
            "telemetry": SAMPLE_DIR / "session_14_telemetry.csv",
            "laps": SAMPLE_DIR / "session_14_laps.csv",
            "ai": SAMPLE_DIR / "session_14_ai_commentary.csv",
            "track": "Mugello Circuit",
            "car": "Formula Hybrid 2024",
        },
        {
            "telemetry": SAMPLE_DIR / "session_19_telemetry.csv",
            "laps": SAMPLE_DIR / "session_19_laps.csv",
            "ai": SAMPLE_DIR / "session_19_ai_commentary.csv",
            "track": "Mugello Circuit",
            "car": "Formula Hybrid 2024",
        },
    ]

    for s in sessions_to_import:
        if not s["telemetry"].exists():
            print(f"Skipping: {s['telemetry']} not found")
            continue

        print(f"Importing {s['telemetry'].name}...")
        sid = import_session(
            db,
            telemetry_csv=s["telemetry"],
            laps_csv=s["laps"],
            ai_csv=s["ai"],
            track_name=s["track"],
            car_model=s["car"],
        )
        print(f"  -> Session {sid} imported successfully")

    db.close()
    print("Done! Sample sessions are now available in the session picker.")


if __name__ == "__main__":
    main()
