"""
Session Recorder for F1 Telemetry Dashboard.

Persistent storage of telemetry sessions using SQLite for:
- Session replay
- Post-race analysis
- Lap-by-lap comparisons
- AI commentary history
- Voice query/response history

Architecture:
- SQLite database (portable, serverless, fast)
- Batch inserts for high-frequency telemetry (60 samples/sec)
- QThread worker for non-blocking writes
- Signals for progress feedback
"""

import sqlite3
import logging
import time
import threading
from typing import Optional, List, Dict, Any
from pathlib import Path
from PyQt5 import QtCore
from datetime import datetime

logger = logging.getLogger(__name__)


class SessionRecorder(QtCore.QThread):
    """
    Session recorder worker thread.

    Records telemetry, laps, AI commentary, and voice interactions to SQLite.
    Uses batch inserts for performance (buffers telemetry samples).

    Signals:
        status_update(str message) - Status messages
        error_occurred(str error) - Error messages
        session_started(int session_id) - Emitted when new session starts
    """

    status_update = QtCore.pyqtSignal(str)
    error_occurred = QtCore.pyqtSignal(str)
    session_started = QtCore.pyqtSignal(int)

    # Batch size for telemetry inserts (buffer 60 samples = 1 second)
    TELEMETRY_BATCH_SIZE = 60

    def __init__(self, db_path: str = "data/telemetry_sessions.db"):
        """
        Initialize session recorder.

        Args:
            db_path: Path to SQLite database file
        """
        super().__init__()

        self.db_path = db_path
        self.db: Optional[sqlite3.Connection] = None

        # Current session
        self.session_id: Optional[int] = None
        self.session_start_time: Optional[float] = None

        # Telemetry batch buffer
        self.telemetry_buffer: List[Dict[str, Any]] = []
        self._last_valid_lap_number: int = 0

        # State
        self._running = False
        self._db_lock = threading.RLock()

        logger.info(f"SessionRecorder initialized with db_path={db_path}")

    def run(self):
        """Main thread execution loop."""
        self._running = True
        self.status_update.emit("Session recorder starting...")

        try:
            # Initialize database
            self._initialize_database()

            self.status_update.emit("💾 Session recorder ready")

            # Main loop - flush batch periodically
            while self._running:
                time.sleep(1.0)  # Flush every second
                self._flush_telemetry_batch()

        except Exception as e:
            logger.error(f"Session recorder error: {e}", exc_info=True)
            self.error_occurred.emit(f"Session recorder failed: {e}")
        finally:
            self._cleanup()
            self.status_update.emit("Session recorder stopped")

    def _initialize_database(self):
        """Create database and tables if they don't exist."""
        try:
            with self._db_lock:
                self.status_update.emit("Initializing database...")

                # Ensure directory exists
                db_dir = Path(self.db_path).parent
                db_dir.mkdir(parents=True, exist_ok=True)

                # Connect to database
                self.db = sqlite3.connect(self.db_path, check_same_thread=False)
                self.db.row_factory = sqlite3.Row  # Enable column access by name

                cursor = self.db.cursor()

                # Create sessions table
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

                # Create laps table
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

                # Create telemetry table (high-frequency data)
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
                    ride_height_front REAL,
                    ride_height_rear REAL,
                    car_damage_front REAL,
                    car_damage_rear REAL,
                    car_damage_left REAL,
                    car_damage_right REAL,
                    car_damage_centre REAL,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
                """)

                # Create AI commentary table
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

                # Create voice queries table
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

                # Create indices for common queries
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_laps_session ON laps(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_session ON telemetry(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_lap ON telemetry(lap_number)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_ai_session ON ai_commentary(session_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_voice_session ON voice_queries(session_id)")

                # Migration: add session_type column if it doesn't exist
                try:
                    cursor.execute("ALTER TABLE sessions ADD COLUMN session_type TEXT DEFAULT ''")
                except sqlite3.OperationalError:
                    pass  # column already exists

                # Migration: add extended telemetry columns for existing databases
                new_telemetry_cols = [
                    ("steer_angle", "REAL"), ("g_force_lat", "REAL"), ("g_force_lon", "REAL"),
                    ("tyre_wear_fl", "REAL"), ("tyre_wear_fr", "REAL"),
                    ("tyre_wear_rl", "REAL"), ("tyre_wear_rr", "REAL"),
                    ("wheel_slip_fl", "REAL"), ("wheel_slip_fr", "REAL"),
                    ("wheel_slip_rl", "REAL"), ("wheel_slip_rr", "REAL"),
                    ("suspension_fl", "REAL"), ("suspension_fr", "REAL"),
                    ("suspension_rl", "REAL"), ("suspension_rr", "REAL"),
                    ("ride_height_front", "REAL"), ("ride_height_rear", "REAL"),
                    ("car_damage_front", "REAL"), ("car_damage_rear", "REAL"),
                    ("car_damage_left", "REAL"), ("car_damage_right", "REAL"),
                    ("car_damage_centre", "REAL"),
                ]
                for col_name, col_type in new_telemetry_cols:
                    try:
                        cursor.execute(f"ALTER TABLE telemetry ADD COLUMN {col_name} {col_type}")
                    except sqlite3.OperationalError:
                        pass  # column already exists

                self.db.commit()
            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}", exc_info=True)
            raise RuntimeError(f"Database initialization failed: {e}")

    def start_session(
        self,
        game: str,
        track_name: str = "",
        car_model: str = "",
        player_name: str = "",
        ai_enabled: bool = False,
        session_type: str = ""
    ) -> int:
        """
        Start a new recording session.

        Args:
            game: Game name ("ac" or "acc")
            track_name: Track name
            car_model: Car model
            player_name: Player name
            ai_enabled: Whether AI race engineer is enabled
            session_type: Game mode (e.g. "Hotlap", "Practice", "Race", "Qualify")

        Returns:
            Session ID
        """
        try:
            with self._db_lock:
                self.session_start_time = time.time()

                cursor = self.db.cursor()
                cursor.execute("""
                INSERT INTO sessions (
                    start_time, game, track_name, car_model, player_name, ai_enabled, session_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.session_start_time,
                    game,
                    track_name,
                    car_model,
                    player_name,
                    1 if ai_enabled else 0,
                    session_type
                ))

                self.session_id = cursor.lastrowid
                self.db.commit()

            logger.info(f"Session started: session_id={self.session_id}, game={game}, track={track_name}")
            self.status_update.emit(f"Recording session {self.session_id}")
            self.session_started.emit(self.session_id)

            return self.session_id

        except Exception as e:
            logger.error(f"Failed to start session: {e}", exc_info=True)
            self.error_occurred.emit(f"Failed to start session: {e}")
            raise

    def _save_incomplete_laps(self):
        """Save lap entries for any laps that have telemetry but no laps row."""
        if not self.session_id or not self.db:
            return

        try:
            cursor = self.db.cursor()

            # Find lap numbers in telemetry that don't have a laps entry yet
            cursor.execute("""
                SELECT DISTINCT t.lap_number
                FROM telemetry t
                LEFT JOIN laps l
                    ON l.session_id = t.session_id AND l.lap_number = t.lap_number
                WHERE t.session_id = ? AND l.lap_id IS NULL
                ORDER BY t.lap_number
            """, (self.session_id,))

            incomplete_laps = [row["lap_number"] for row in cursor.fetchall()]

            for lap_number in incomplete_laps:
                cursor.execute("""
                    SELECT
                        MIN(elapsed_time) as t_start,
                        MAX(elapsed_time) as t_end,
                        AVG(speed) as avg_speed,
                        MAX(speed) as max_speed,
                        MIN(speed) as min_speed,
                        COUNT(*) as sample_count
                    FROM telemetry
                    WHERE session_id = ? AND lap_number = ?
                """, (self.session_id, lap_number))

                stats = cursor.fetchone()
                if not stats or stats["sample_count"] == 0:
                    continue

                lap_time = max(0.0, stats["t_end"] - stats["t_start"])

                # Get fuel at start and end
                cursor.execute("""
                    SELECT fuel FROM telemetry
                    WHERE session_id = ? AND lap_number = ?
                    ORDER BY elapsed_time ASC LIMIT 1
                """, (self.session_id, lap_number))
                fuel_start_row = cursor.fetchone()
                fuel_start = fuel_start_row["fuel"] if fuel_start_row else 0.0

                cursor.execute("""
                    SELECT fuel FROM telemetry
                    WHERE session_id = ? AND lap_number = ?
                    ORDER BY elapsed_time DESC LIMIT 1
                """, (self.session_id, lap_number))
                fuel_end_row = cursor.fetchone()
                fuel_end = fuel_end_row["fuel"] if fuel_end_row else 0.0

                cursor.execute("""
                    INSERT INTO laps (
                        session_id, lap_number, lap_time, fuel_start, fuel_end,
                        avg_speed, max_speed, min_speed, valid, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.session_id,
                    lap_number,
                    lap_time,
                    fuel_start,
                    fuel_end,
                    stats["avg_speed"],
                    stats["max_speed"],
                    stats["min_speed"],
                    0,  # incomplete laps marked as invalid
                    time.time()
                ))

                logger.info(f"Saved incomplete lap: lap={lap_number}, time={lap_time:.1f}s, samples={stats['sample_count']}")

            self.db.commit()

        except Exception as e:
            logger.error(f"Failed to save incomplete laps: {e}", exc_info=True)

    def end_session(self):
        """End the current recording session."""
        if not self.session_id:
            return

        try:
            with self._db_lock:
                # Flush any remaining telemetry
                self._flush_telemetry_batch()

                # Save incomplete laps before calculating totals
                self._save_incomplete_laps()

                # Update session end time
                cursor = self.db.cursor()

                # Calculate session statistics (count all laps, not just valid)
                cursor.execute("""
                SELECT
                    COUNT(*) as total_laps,
                    MIN(CASE WHEN valid = 1 THEN lap_time END) as best_lap_time
                FROM laps
                WHERE session_id = ?
                """, (self.session_id,))

                row = cursor.fetchone()
                total_laps = row["total_laps"] if row else 0
                best_lap_time = row["best_lap_time"] if row else None

                # Update session record
                cursor.execute("""
                UPDATE sessions
                SET end_time = ?, total_laps = ?, best_lap_time = ?
                WHERE session_id = ?
                """, (time.time(), total_laps, best_lap_time, self.session_id))

                self.db.commit()

            logger.info(f"Session ended: session_id={self.session_id}, laps={total_laps}")
            self.status_update.emit(f"Session {self.session_id} saved ({total_laps} laps)")

            self.session_id = None
            self.session_start_time = None

        except Exception as e:
            logger.error(f"Failed to end session: {e}", exc_info=True)
            self.error_occurred.emit(f"Failed to end session: {e}")

    def record_telemetry_sample(self, sample: Dict[str, Any]):
        """
        Record a telemetry sample (buffered for batch insert).

        Args:
            sample: Telemetry sample dict with keys: lap_id, t, x, z, speed, gear, etc.
        """
        if not self.session_id:
            return

        try:
            with self._db_lock:
                # Sanitize lap_number — reject negative values from garbage shared memory reads
                lap_number = sample.get("lap_id", 0)
                if lap_number < 0:
                    lap_number = self._last_valid_lap_number
                else:
                    self._last_valid_lap_number = lap_number

                # Add to buffer
                self.telemetry_buffer.append({
                    "session_id": self.session_id,
                    "lap_number": lap_number,
                    "elapsed_time": sample.get("t", 0.0),
                    "pos_x": sample.get("x", 0.0),
                    "pos_z": sample.get("z", 0.0),
                    "speed": sample.get("speed", 0.0),
                    "gear": sample.get("gear", 0),
                    "rpm": sample.get("rpms", 0),  # AC sends "rpms" (with s)
                    "throttle": sample.get("throttle", 0.0),
                    "brake": sample.get("brake", 0.0),
                    "fuel": sample.get("fuel", 0.0),
                    "steer_angle": sample.get("steer_angle", 0.0),
                    "g_force_lat": sample.get("g_force_lat", 0.0),
                    "g_force_lon": sample.get("g_force_lon", 0.0),
                    "tyre_pressure_fl": sample.get("tyre_pressure_fl", 0.0),
                    "tyre_pressure_fr": sample.get("tyre_pressure_fr", 0.0),
                    "tyre_pressure_rl": sample.get("tyre_pressure_rl", 0.0),
                    "tyre_pressure_rr": sample.get("tyre_pressure_rr", 0.0),
                    "tyre_temp_fl": sample.get("tyre_temp_fl", 0.0),
                    "tyre_temp_fr": sample.get("tyre_temp_fr", 0.0),
                    "tyre_temp_rl": sample.get("tyre_temp_rl", 0.0),
                    "tyre_temp_rr": sample.get("tyre_temp_rr", 0.0),
                    "tyre_wear_fl": sample.get("tyre_wear_fl", 0.0),
                    "tyre_wear_fr": sample.get("tyre_wear_fr", 0.0),
                    "tyre_wear_rl": sample.get("tyre_wear_rl", 0.0),
                    "tyre_wear_rr": sample.get("tyre_wear_rr", 0.0),
                    "wheel_slip_fl": sample.get("wheel_slip_fl", 0.0),
                    "wheel_slip_fr": sample.get("wheel_slip_fr", 0.0),
                    "wheel_slip_rl": sample.get("wheel_slip_rl", 0.0),
                    "wheel_slip_rr": sample.get("wheel_slip_rr", 0.0),
                    "suspension_fl": sample.get("suspension_fl", 0.0),
                    "suspension_fr": sample.get("suspension_fr", 0.0),
                    "suspension_rl": sample.get("suspension_rl", 0.0),
                    "suspension_rr": sample.get("suspension_rr", 0.0),
                    "ride_height_front": sample.get("ride_height_front", 0.0),
                    "ride_height_rear": sample.get("ride_height_rear", 0.0),
                    "car_damage_front": sample.get("car_damage_front", 0.0),
                    "car_damage_rear": sample.get("car_damage_rear", 0.0),
                    "car_damage_left": sample.get("car_damage_left", 0.0),
                    "car_damage_right": sample.get("car_damage_right", 0.0),
                    "car_damage_centre": sample.get("car_damage_centre", 0.0),
                    "timestamp": time.time()
                })

                # Flush if batch is full
                if len(self.telemetry_buffer) >= self.TELEMETRY_BATCH_SIZE:
                    self._flush_telemetry_batch()

        except Exception as e:
            logger.error(f"Failed to record telemetry sample: {e}")

    def _flush_telemetry_batch(self):
        """Flush buffered telemetry samples to database."""
        with self._db_lock:
            if not self.telemetry_buffer:
                return

            try:
                cursor = self.db.cursor()

                cursor.executemany("""
                INSERT INTO telemetry (
                    session_id, lap_number, elapsed_time, pos_x, pos_z, speed,
                    gear, rpm, throttle, brake, fuel,
                    steer_angle, g_force_lat, g_force_lon,
                    tyre_pressure_fl, tyre_pressure_fr, tyre_pressure_rl, tyre_pressure_rr,
                    tyre_temp_fl, tyre_temp_fr, tyre_temp_rl, tyre_temp_rr,
                    tyre_wear_fl, tyre_wear_fr, tyre_wear_rl, tyre_wear_rr,
                    wheel_slip_fl, wheel_slip_fr, wheel_slip_rl, wheel_slip_rr,
                    suspension_fl, suspension_fr, suspension_rl, suspension_rr,
                    ride_height_front, ride_height_rear,
                    car_damage_front, car_damage_rear, car_damage_left, car_damage_right, car_damage_centre,
                    timestamp
                ) VALUES (
                    :session_id, :lap_number, :elapsed_time, :pos_x, :pos_z, :speed,
                    :gear, :rpm, :throttle, :brake, :fuel,
                    :steer_angle, :g_force_lat, :g_force_lon,
                    :tyre_pressure_fl, :tyre_pressure_fr, :tyre_pressure_rl, :tyre_pressure_rr,
                    :tyre_temp_fl, :tyre_temp_fr, :tyre_temp_rl, :tyre_temp_rr,
                    :tyre_wear_fl, :tyre_wear_fr, :tyre_wear_rl, :tyre_wear_rr,
                    :wheel_slip_fl, :wheel_slip_fr, :wheel_slip_rl, :wheel_slip_rr,
                    :suspension_fl, :suspension_fr, :suspension_rl, :suspension_rr,
                    :ride_height_front, :ride_height_rear,
                    :car_damage_front, :car_damage_rear, :car_damage_left, :car_damage_right, :car_damage_centre,
                    :timestamp
                )
                """, self.telemetry_buffer)

                self.db.commit()

                logger.debug(f"Flushed {len(self.telemetry_buffer)} telemetry samples to database")
                self.telemetry_buffer.clear()

            except Exception as e:
                logger.error(f"Failed to flush telemetry batch: {e}", exc_info=True)

    def record_lap(
        self,
        lap_number: int,
        lap_time: Optional[float] = None,
        fuel_start: Optional[float] = None,
        fuel_end: Optional[float] = None,
        avg_speed: Optional[float] = None,
        max_speed: Optional[float] = None,
        min_speed: Optional[float] = None,
        valid: bool = True
    ):
        """
        Record a completed lap.

        Args:
            lap_number: Lap number
            lap_time: Lap time in seconds
            fuel_start: Fuel at lap start
            fuel_end: Fuel at lap end
            avg_speed: Average speed
            max_speed: Maximum speed
            min_speed: Minimum speed
            valid: Whether lap is valid (no cuts, etc.)
        """
        if not self.session_id:
            return

        try:
            with self._db_lock:
                cursor = self.db.cursor()
                cursor.execute("""
                INSERT INTO laps (
                    session_id, lap_number, lap_time, fuel_start, fuel_end,
                    avg_speed, max_speed, min_speed, valid, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.session_id,
                    lap_number,
                    lap_time,
                    fuel_start,
                    fuel_end,
                    avg_speed,
                    max_speed,
                    min_speed,
                    1 if valid else 0,
                    time.time()
                ))

                self.db.commit()
            logger.info(f"Lap recorded: lap={lap_number}, time={lap_time}")

        except Exception as e:
            logger.error(f"Failed to record lap: {e}", exc_info=True)
            self.error_occurred.emit(f"Failed to record lap: {e}")

    def record_ai_commentary(
        self,
        message: str,
        trigger: str = "",
        priority: str = "",
        lap_number: int = 0
    ):
        """
        Record AI race engineer commentary.

        Args:
            message: Commentary message
            trigger: Event trigger
            priority: Message priority
            lap_number: Current lap number
        """
        if not self.session_id:
            return

        try:
            with self._db_lock:
                cursor = self.db.cursor()
                cursor.execute("""
                INSERT INTO ai_commentary (
                    session_id, timestamp, message, trigger, priority, lap_number
                ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    self.session_id,
                    time.time(),
                    message,
                    trigger,
                    priority,
                    lap_number
                ))

                self.db.commit()
            logger.debug(f"AI commentary recorded: {message[:50]}...")

        except Exception as e:
            logger.error(f"Failed to record AI commentary: {e}", exc_info=True)

    def record_voice_query(
        self,
        query_text: str,
        response_text: str = "",
        lap_number: int = 0
    ):
        """
        Record voice query and response.

        Args:
            query_text: Driver's query
            response_text: AI's response
            lap_number: Current lap number
        """
        if not self.session_id:
            return

        try:
            with self._db_lock:
                cursor = self.db.cursor()
                cursor.execute("""
                INSERT INTO voice_queries (
                    session_id, timestamp, query_text, response_text, lap_number
                ) VALUES (?, ?, ?, ?, ?)
                """, (
                    self.session_id,
                    time.time(),
                    query_text,
                    response_text,
                    lap_number
                ))

                self.db.commit()
            logger.debug(f"Voice query recorded: {query_text[:50]}...")

        except Exception as e:
            logger.error(f"Failed to record voice query: {e}", exc_info=True)

    def _cleanup(self):
        """Clean up database resources."""
        with self._db_lock:
            # Flush remaining telemetry
            self._flush_telemetry_batch()

            # End current session if active
            if self.session_id:
                self.end_session()

            # Close database
            if self.db:
                try:
                    self.db.close()
                except Exception:
                    pass

        logger.info("Database resources cleaned up")

    def stop(self):
        """Stop the session recorder worker."""
        logger.info("Stopping session recorder...")
        self._running = False
