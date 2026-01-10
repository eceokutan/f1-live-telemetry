"""
Session context for Jarvis-Granite Live Telemetry.

LiveSessionContext maintains all state during an active racing session:
- Vehicle state (speed, rpm, gear, inputs)
- Resource state (fuel, tires)
- Race position (position, gaps)
- Telemetry buffer (rolling 60s window)
- Conversation history (last 3 exchanges)
- Active alerts
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from collections import deque
from typing import Any, Deque, Dict, List, Optional


def _utcnow() -> datetime:
    """Get current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)

from jarvis_granite.schemas.telemetry import TelemetryData


@dataclass
class LiveSessionContext:
    """
    In-memory context maintained during a live racing session.

    This context is updated with each telemetry message and provides
    the state needed for AI response generation.

    Attributes:
        session_id: Unique session identifier
        source: Telemetry data source (torcs, assetto_corsa, can_bus)
        track_name: Name of the current track
        started_at: Session start timestamp
    """

    # Session identification (required)
    session_id: str
    source: str
    track_name: str

    # Session start time
    started_at: datetime = field(default_factory=_utcnow)

    # Current vehicle state
    current_lap: int = 0
    current_sector: int = 1
    speed_kmh: float = 0.0
    rpm: int = 0
    gear: int = 0
    throttle: float = 0.0
    brake: float = 0.0

    # Resource state
    fuel_remaining: float = 100.0
    fuel_consumption_per_lap: float = 0.0
    tire_wear: Dict[str, float] = field(
        default_factory=lambda: {"fl": 0, "fr": 0, "rl": 0, "rr": 0}
    )
    tire_temps: Dict[str, float] = field(
        default_factory=lambda: {"fl": 80, "fr": 80, "rl": 80, "rr": 80}
    )

    # Race position
    position: int = 1
    gap_ahead: Optional[float] = None
    gap_behind: Optional[float] = None

    # Lap history
    lap_times: List[float] = field(default_factory=list)
    best_lap: Optional[float] = None
    last_lap: Optional[float] = None

    # Rolling telemetry buffer (60 seconds at 10Hz = 600 samples)
    telemetry_buffer: Deque[Dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=600)
    )

    # Conversation history (last 3 exchanges)
    conversation_history: Deque[Dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=3)
    )

    # Active alerts
    active_alerts: List[Dict[str, Any]] = field(default_factory=list)

    # Proactive message timing
    last_proactive_message_time: Optional[datetime] = None

    # Internal tracking for fuel consumption calculation
    _lap_start_fuel: float = field(default=100.0, repr=False)

    def update(self, telemetry: TelemetryData) -> None:
        """
        Update context from telemetry data.

        Args:
            telemetry: TelemetryData snapshot from the platform
        """
        # Vehicle state
        self.speed_kmh = telemetry.speed_kmh
        self.rpm = telemetry.rpm
        self.gear = telemetry.gear
        self.throttle = telemetry.throttle
        self.brake = telemetry.brake

        # Resources
        self.fuel_remaining = telemetry.fuel_remaining
        self.tire_temps = {
            "fl": telemetry.tire_temps.fl,
            "fr": telemetry.tire_temps.fr,
            "rl": telemetry.tire_temps.rl,
            "rr": telemetry.tire_temps.rr,
        }
        self.tire_wear = {
            "fl": telemetry.tire_wear.fl,
            "fr": telemetry.tire_wear.fr,
            "rl": telemetry.tire_wear.rl,
            "rr": telemetry.tire_wear.rr,
        }

        # Lap and sector
        self.current_lap = telemetry.lap_number
        self.current_sector = telemetry.sector

        # Race position (may be None)
        if telemetry.position is not None:
            self.position = telemetry.position
        self.gap_ahead = telemetry.gap_ahead
        self.gap_behind = telemetry.gap_behind

        # Add to buffer
        self.add_telemetry(telemetry)

    def add_telemetry(self, telemetry: TelemetryData) -> None:
        """
        Add telemetry snapshot to the rolling buffer.

        Args:
            telemetry: TelemetryData to add
        """
        self.telemetry_buffer.append(telemetry.model_dump())

    def add_exchange(self, query: str, response: str) -> None:
        """
        Add a conversation exchange to history.

        Args:
            query: Driver's question or command
            response: AI's response
        """
        self.conversation_history.append({
            "query": query,
            "response": response,
            "timestamp": _utcnow()
        })

    def record_lap_time(self, lap_time: float) -> None:
        """
        Record a completed lap time.

        Updates lap_times list, last_lap, best_lap, and calculates
        fuel consumption if tracking.

        Args:
            lap_time: Lap time in seconds
        """
        self.lap_times.append(lap_time)
        self.last_lap = lap_time

        # Update best lap
        if self.best_lap is None or lap_time < self.best_lap:
            self.best_lap = lap_time

        # Calculate fuel consumption
        if self._lap_start_fuel > 0:
            fuel_used = self._lap_start_fuel - self.fuel_remaining
            if fuel_used > 0:
                self.fuel_consumption_per_lap = fuel_used

        # Reset lap start fuel for next lap
        self._lap_start_fuel = self.fuel_remaining

    def get_fuel_laps_remaining(self) -> float:
        """
        Calculate estimated laps remaining based on fuel.

        Returns:
            Number of laps that can be completed with remaining fuel.
            Returns infinity if consumption is zero.
        """
        if self.fuel_consumption_per_lap <= 0:
            return float('inf')
        return self.fuel_remaining / self.fuel_consumption_per_lap

    def add_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        """
        Add an active alert.

        Args:
            alert_type: Type of alert (e.g., "fuel_warning")
            data: Alert-specific data
        """
        self.active_alerts.append({
            "type": alert_type,
            "data": data,
            "timestamp": _utcnow()
        })

    def clear_alert(self, alert_type: str) -> None:
        """
        Clear alerts of a specific type.

        Args:
            alert_type: Type of alert to clear
        """
        self.active_alerts = [
            alert for alert in self.active_alerts
            if alert["type"] != alert_type
        ]

    def can_send_proactive(self, min_interval_seconds: float) -> bool:
        """
        Check if enough time has passed to send a proactive message.

        Args:
            min_interval_seconds: Minimum seconds between proactive messages

        Returns:
            True if a proactive message can be sent
        """
        if self.last_proactive_message_time is None:
            return True

        elapsed = _utcnow() - self.last_proactive_message_time
        return elapsed.total_seconds() >= min_interval_seconds

    def mark_proactive_sent(self) -> None:
        """Mark that a proactive message was just sent."""
        self.last_proactive_message_time = _utcnow()

    def get_session_duration(self) -> timedelta:
        """
        Get the duration of the current session.

        Returns:
            Time elapsed since session started
        """
        return _utcnow() - self.started_at

    def to_prompt_context(self) -> str:
        """
        Format context for LLM prompt injection.

        Returns:
            Formatted string containing current session state
            suitable for including in LLM prompts.
        """
        # Format fuel laps
        fuel_laps = self.get_fuel_laps_remaining()
        fuel_laps_str = f"{fuel_laps:.1f}" if fuel_laps != float('inf') else "N/A"

        # Format gaps
        gap_ahead_str = f"{self.gap_ahead:.2f}s" if self.gap_ahead is not None else "N/A"
        gap_behind_str = f"{self.gap_behind:.2f}s" if self.gap_behind is not None else "N/A"

        # Format lap times
        best_lap_str = self._format_lap_time(self.best_lap) if self.best_lap else "N/A"
        last_lap_str = self._format_lap_time(self.last_lap) if self.last_lap else "N/A"

        return f"""Track: {self.track_name}
Lap: {self.current_lap} | Position: P{self.position}
Gap Ahead: {gap_ahead_str} | Gap Behind: {gap_behind_str}
Fuel: {self.fuel_remaining:.1f}L ({fuel_laps_str} laps)
Tires: FL:{self.tire_temps['fl']:.0f}°C FR:{self.tire_temps['fr']:.0f}°C RL:{self.tire_temps['rl']:.0f}°C RR:{self.tire_temps['rr']:.0f}°C
Best Lap: {best_lap_str} | Last Lap: {last_lap_str}"""

    def _format_lap_time(self, seconds: float) -> str:
        """Format lap time as M:SS.mmm."""
        minutes = int(seconds // 60)
        remaining = seconds % 60
        return f"{minutes}:{remaining:06.3f}"
