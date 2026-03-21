"""
CSV telemetry loader - loads exported CSV files into Session/Lap objects.
"""
import json
from datetime import datetime
import pandas as pd
from pathlib import Path
from typing import List

from .models import SessionMetadata, LapSummary, AIComment
from .lap import Lap
from .session import Session


class TelemetryLoader:
    """
    Load and parse CSV telemetry exports.

    Expected CSV files:
    - session_X_telemetry.csv - High-frequency telemetry samples
    - session_X_laps.csv - Lap summary statistics
    - session_X_ai_commentary.csv (optional) - AI comments
    - session_X_metadata.json (optional) - Session metadata from SQLite export
    """

    @staticmethod
    def load_session(session_path: str) -> Session:
        """
        Load a complete session from CSV files.

        Args:
            session_path: Path to directory containing CSVs OR path to telemetry.csv file

        Returns:
            Session object with all data loaded
        """
        path = Path(session_path)

        if path.is_file():
            directory = path.parent
            base_name = path.stem.replace('_telemetry', '')
        else:
            directory = path
            telemetry_files = list(directory.glob('*_telemetry.csv'))
            if not telemetry_files:
                raise FileNotFoundError(f"No telemetry CSV found in {directory}")
            base_name = telemetry_files[0].stem.replace('_telemetry', '')

        telemetry_path = directory / f"{base_name}_telemetry.csv"
        laps_path = directory / f"{base_name}_laps.csv"
        ai_path = directory / f"{base_name}_ai_commentary.csv"
        metadata_path = directory / f"{base_name}_metadata.json"

        if not telemetry_path.exists():
            raise FileNotFoundError(f"Telemetry file not found: {telemetry_path}")

        telemetry_df = pd.read_csv(telemetry_path)
        TelemetryLoader._validate_telemetry(telemetry_df)

        laps_list: List[Lap] = []
        lap_summaries = {}

        if laps_path.exists():
            laps_df = pd.read_csv(laps_path)
            lap_summaries = TelemetryLoader._parse_lap_summaries(laps_df)

        laps_list = TelemetryLoader._create_laps(telemetry_df, lap_summaries)

        metadata = TelemetryLoader._extract_metadata(
            telemetry_df,
            base_name,
            len(laps_list),
            metadata_path,
        )

        ai_commentary = []
        if ai_path.exists():
            ai_commentary = TelemetryLoader._load_ai_commentary(ai_path)

        return Session(
            metadata=metadata,
            laps=laps_list,
            telemetry=telemetry_df,
            ai_commentary=ai_commentary
        )

    @staticmethod
    def _validate_telemetry(df: pd.DataFrame) -> None:
        """Validate telemetry DataFrame has required columns."""
        required_columns = ['lap_number', 'elapsed_time', 'speed']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Telemetry CSV missing required columns: {missing}")
        if len(df) == 0:
            raise ValueError("Telemetry CSV is empty")

    @staticmethod
    def _parse_lap_summaries(laps_df: pd.DataFrame) -> dict:
        """Parse lap summaries DataFrame into LapSummary objects."""
        summaries = {}
        for _, row in laps_df.iterrows():
            lap_num = int(row['lap_number'])
            summaries[lap_num] = LapSummary(
                lap_number=lap_num,
                lap_time=float(row['lap_time']),
                fuel_start=float(row.get('fuel_start', 0.0)),
                fuel_end=float(row.get('fuel_end', 0.0)),
                avg_speed=float(row.get('avg_speed', 0.0)),
                max_speed=float(row.get('max_speed', 0.0)),
                min_speed=float(row.get('min_speed', 0.0)),
                valid=bool(row.get('valid', True))
            )
        return summaries

    @staticmethod
    def _create_laps(telemetry_df: pd.DataFrame, lap_summaries: dict) -> List[Lap]:
        """Split telemetry DataFrame by lap number and create Lap objects."""
        laps = []
        for lap_num, lap_data in telemetry_df.groupby('lap_number'):
            lap_num = int(lap_num)
            summary = lap_summaries.get(lap_num)
            lap = Lap(
                lap_number=lap_num,
                telemetry=lap_data.reset_index(drop=True),
                summary=summary
            )
            laps.append(lap)
        return laps

    @staticmethod
    def _extract_metadata(
        telemetry_df: pd.DataFrame,
        base_name: str,
        total_laps: int,
        metadata_path: Path,
    ) -> SessionMetadata:
        """Extract session metadata from telemetry DataFrame."""
        session_id = 0
        if base_name.startswith('session_'):
            try:
                session_id = int(base_name.split('_')[1])
            except (IndexError, ValueError):
                pass

        metadata_payload = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata_payload = json.load(f) or {}
            except Exception:
                metadata_payload = {}

        start_time = None
        start_raw = metadata_payload.get("start_time")
        if isinstance(start_raw, (int, float)):
            try:
                start_time = datetime.fromtimestamp(start_raw)
            except (OSError, OverflowError, ValueError):
                start_time = None

        end_time = None
        end_raw = metadata_payload.get("end_time")
        if isinstance(end_raw, (int, float)):
            try:
                end_time = datetime.fromtimestamp(end_raw)
            except (OSError, OverflowError, ValueError):
                end_time = None

        return SessionMetadata(
            session_id=int(metadata_payload.get("session_id", session_id) or session_id),
            game=str(metadata_payload.get("game") or "unknown"),
            track_name=str(metadata_payload.get("track_name") or "Unknown Track"),
            car_model=str(metadata_payload.get("car_model") or "Unknown Car"),
            player_name=str(metadata_payload.get("player_name") or "Unknown Player"),
            start_time=start_time,
            end_time=end_time,
            total_laps=int(metadata_payload.get("total_laps", total_laps) or total_laps),
            ai_enabled=bool(metadata_payload.get("ai_enabled", False)),
            notes=str(metadata_payload.get("notes") or ""),
        )

    @staticmethod
    def _load_ai_commentary(ai_path: Path) -> List[AIComment]:
        """Load AI commentary from CSV."""
        df = pd.read_csv(ai_path, encoding='latin-1')
        comments = []
        model_column = next((col for col in ("model", "role", "agent", "ai_model") if col in df.columns), None)

        for _, row in df.iterrows():
            timestamp_value = row.get('timestamp')
            timestamp = pd.to_datetime(timestamp_value, unit='s', errors='coerce')
            if pd.isna(timestamp):
                timestamp = pd.to_datetime(timestamp_value, errors='coerce')
            if pd.isna(timestamp):
                timestamp = pd.Timestamp.now()

            priority_value = row.get('priority', 2)
            try:
                priority = int(priority_value)
            except (TypeError, ValueError):
                priority_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
                priority = priority_map.get(str(priority_value).strip().lower(), 2)

            model = None
            if model_column:
                model_value = row.get(model_column)
                if pd.notna(model_value):
                    model_text = str(model_value).strip().lower()
                    model = model_text or None

            comment = AIComment(
                timestamp=timestamp,
                message=str(row['message']),
                trigger=str(row['trigger']),
                priority=priority,
                lap_number=int(row['lap_number']),
                model=model,
            )
            comments.append(comment)

        return comments
