"""
Session context for Jarvis-Granite Live Telemetry.

LiveSessionContext maintains all state during an active racing session:
- Vehicle state (speed, rpm, gear, inputs)
- Resource state (fuel, tires)
- Race position (position, gaps)
- Telemetry buffer (rolling 60s window)
- Conversation history (last exchange)
- Active alerts
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from collections import deque
from typing import Any, Deque, Dict, List, Optional


def _utcnow() -> datetime:
    """Get current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)

from ai.race_engineer_core.telemetry import OpponentSnapshot, TelemetryData

# Keyword groups for query-aware context pruning.
# Each group maps a set of query keywords to the context line labels it unlocks.
# Lines whose label is in the "core" group are always included.
_CONTEXT_GROUPS = {
    "core": {
        "keywords": set(),  # always included
        "labels": {"track", "lap_position", "speed_gear_rpm"},
    },
    "inputs": {
        "keywords": {"throttle", "brake", "pedal", "g-force", "steering", "braking", "accelerat"},
        "labels": {"throttle_brake", "g_forces"},
    },
    "gaps": {
        "keywords": {"gap", "ahead", "behind", "opponent", "defend", "attack", "overtake", "pass", "close", "catch"},
        "labels": {"gap_ahead_behind", "nearby_opponents"},
    },
    "fuel": {
        "keywords": {"fuel", "range"},
        "labels": {"fuel"},
    },
    "tires": {
        "keywords": {"tire", "tyre", "temp", "temperature", "pressure", "wear", "grip", "deg", "degradation"},
        "labels": {"tire_temps", "tire_pressures", "tire_wear"},
    },
    "damage": {
        "keywords": {"damage", "crash", "contact", "hit", "broken", "wing"},
        "labels": {"car_damage"},
    },
    "pit": {
        "keywords": {"pit", "pitting", "box", "stop", "stint", "refuel"},
        "labels": {"pit_summary", "fuel"},
    },
    "laptimes": {
        "keywords": {"lap", "time", "pace", "fast", "slow", "delta", "best", "sector", "improve"},
        "labels": {"best_last_lap"},
    },
}


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
    tire_pressures: Dict[str, float] = field(
        default_factory=lambda: {"fl": 28.0, "fr": 28.0, "rl": 28.0, "rr": 28.0}
    )

    # Race position
    position: int = 1
    gap_ahead: Optional[float] = None
    gap_behind: Optional[float] = None
    opponents: List[OpponentSnapshot] = field(default_factory=list)

    # Steering and physics
    steering_angle: float = 0.0
    g_force_lat: float = 0.0
    g_force_lon: float = 0.0

    # Wheel slip
    wheel_slip: Dict[str, float] = field(
        default_factory=lambda: {"fl": 0.0, "fr": 0.0, "rl": 0.0, "rr": 0.0}
    )

    # Suspension and ride height
    suspension_travel: Dict[str, float] = field(
        default_factory=lambda: {"fl": 0.0, "fr": 0.0, "rl": 0.0, "rr": 0.0}
    )
    ride_height_front: float = 0.0
    ride_height_rear: float = 0.0

    # Car damage (5 zones: front, rear, left, right, centre; 0.0 = no damage)
    car_damage: Dict[str, float] = field(
        default_factory=lambda: {"front": 0.0, "rear": 0.0, "left": 0.0, "right": 0.0, "centre": 0.0}
    )

    # Lap history
    lap_times: List[float] = field(default_factory=list)
    best_lap: Optional[float] = None
    last_lap: Optional[float] = None

    # Rolling telemetry buffer (60 seconds at 10Hz = 600 samples)
    telemetry_buffer: Deque[Dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=600)
    )

    # Conversation history (last exchange)
    conversation_history: Deque[Dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=1)
    )

    # Active alerts
    active_alerts: List[Dict[str, Any]] = field(default_factory=list)

    # Proactive message timing
    last_proactive_message_time: Optional[datetime] = None

    # Internal tracking for fuel consumption calculation
    # Starts at -1 to indicate "not yet initialized from telemetry"
    _lap_start_fuel: float = field(default=-1.0, repr=False)

    # Internal tracking for lap time calculation from telemetry timestamps
    _lap_start_time: float = field(default=0.0, repr=False)

    def update(self, telemetry: TelemetryData) -> None:
        """
        Update context from telemetry data.

        Args:
            telemetry: TelemetryData snapshot from the platform
        """
        # Vehicle state
        self.speed_kmh = telemetry.speed  # AC uses 'speed' not 'speed_kmh'
        self.rpm = telemetry.rpms  # AC uses 'rpms' not 'rpm'
        self.gear = telemetry.gear
        self.throttle = telemetry.throttle
        self.brake = telemetry.brake

        # Resources
        self.fuel_remaining = telemetry.fuel if telemetry.fuel is not None else 0.0
        # Initialize lap start fuel from first real telemetry reading
        if self._lap_start_fuel < 0 and self.fuel_remaining > 0:
            self._lap_start_fuel = self.fuel_remaining
        self.tire_temps = {
            "fl": telemetry.tire_temps.fl,
            "fr": telemetry.tire_temps.fr,
            "rl": telemetry.tire_temps.rl,
            "rr": telemetry.tire_temps.rr,
        }
        if telemetry.tire_pressure is not None:
            self.tire_pressures = {
                "fl": telemetry.tire_pressure.fl,
                "fr": telemetry.tire_pressure.fr,
                "rl": telemetry.tire_pressure.rl,
                "rr": telemetry.tire_pressure.rr,
            }
        # Tire wear may not be available in AC
        if telemetry.tire_wear is not None:
            self.tire_wear = {
                "fl": telemetry.tire_wear.fl,
                "fr": telemetry.tire_wear.fr,
                "rl": telemetry.tire_wear.rl,
                "rr": telemetry.tire_wear.rr,
            }
        else:
            self.tire_wear = {"fl": 0.0, "fr": 0.0, "rl": 0.0, "rr": 0.0}

        # Steering and physics
        self.steering_angle = telemetry.steering_angle or 0.0
        if telemetry.g_forces is not None:
            self.g_force_lat = telemetry.g_forces.lateral
            self.g_force_lon = telemetry.g_forces.longitudinal

        # Wheel slip
        if telemetry.wheel_slip is not None:
            self.wheel_slip = {
                "fl": telemetry.wheel_slip.fl,
                "fr": telemetry.wheel_slip.fr,
                "rl": telemetry.wheel_slip.rl,
                "rr": telemetry.wheel_slip.rr,
            }

        # Suspension and ride height
        if telemetry.suspension_travel is not None:
            self.suspension_travel = {
                "fl": telemetry.suspension_travel.fl,
                "fr": telemetry.suspension_travel.fr,
                "rl": telemetry.suspension_travel.rl,
                "rr": telemetry.suspension_travel.rr,
            }
        if telemetry.ride_height is not None:
            self.ride_height_front = telemetry.ride_height.front
            self.ride_height_rear = telemetry.ride_height.rear

        # Car damage
        if telemetry.car_damage is not None:
            self.car_damage = telemetry.car_damage

        # Lap and sector (keep current values if None in AC)
        if telemetry.lap_number is not None:
            # Detect lap transition and record lap time
            if telemetry.lap_number > self.current_lap and self.current_lap > 0:
                # Prefer the game's authoritative lap time (lastTimeMs) over
                # computed timestamps which are imprecise due to telemetry throttling.
                if telemetry.last_time_ms and telemetry.last_time_ms > 0:
                    lap_time = telemetry.last_time_ms / 1000.0
                else:
                    lap_time = telemetry.t - self._lap_start_time
                if lap_time > 0:
                    self.record_lap_time(lap_time)
            # Reset lap start time on any lap change (including first lap)
            if telemetry.lap_number != self.current_lap:
                self._lap_start_time = telemetry.t
            self.current_lap = telemetry.lap_number
        if telemetry.sector is not None:
            self.current_sector = telemetry.sector

        # Race position (may be None)
        if telemetry.position is not None:
            self.position = telemetry.position
        self.gap_ahead = telemetry.gap_ahead
        self.gap_behind = telemetry.gap_behind
        self.opponents = telemetry.opponents or []

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

        # Calculate fuel consumption (only if we have a valid start reading)
        if self._lap_start_fuel > 0:
            fuel_used = self._lap_start_fuel - self.fuel_remaining
            # Sanity check: consumption should be reasonable (< 20L/lap for any car)
            if 0 < fuel_used < 20:
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

    def to_prompt_context(self, query: Optional[str] = None, grounding_only: bool = False) -> str:
        """
        Format context for LLM prompt injection.

        Args:
            query: Optional driver query. When provided, only context lines
                   relevant to the query keywords are included (plus core
                   lines that are always present). When *None*, all lines
                   are returned (backwards-compatible).

        Returns:
            Formatted string containing current session state
            suitable for including in LLM prompts.
        """
        # Determine which labels to include
        active_labels: set = set(_CONTEXT_GROUPS["core"]["labels"])

        if query is None:
            # No query → return full context (backwards compatible)
            for group in _CONTEXT_GROUPS.values():
                active_labels |= group["labels"]
        elif query is not None:
            query_lower = query.lower()
            for group_name, group in _CONTEXT_GROUPS.items():
                if group_name == "core":
                    continue
                for kw in group["keywords"]:
                    if kw in query_lower:
                        active_labels |= group["labels"]
                        break
            # No keyword match -> core labels only (query-agnostic minimum context).
            # grounding_only has identical behavior in this case.

        # Pre-format values used by multiple lines
        fuel_laps = self.get_fuel_laps_remaining()
        if fuel_laps == float('inf'):
            fuel_laps_str = "unknown (no completed lap yet)"
        else:
            fuel_laps_str = f"{fuel_laps:.1f}"
        gap_ahead_str = f"{self.gap_ahead:.2f}s" if self.gap_ahead is not None else "N/A"
        gap_behind_str = f"{self.gap_behind:.2f}s" if self.gap_behind is not None else "N/A"
        nearby_str = self._format_nearby_opponents()
        best_lap_str = self._format_lap_time(self.best_lap) if self.best_lap else "N/A"
        last_lap_str = self._format_lap_time(self.last_lap) if self.last_lap else "N/A"
        total_damage = sum(self.car_damage.values())
        if total_damage > 0:
            damage_parts = [f"{zone}: {val:.0f}%" for zone, val in self.car_damage.items() if val > 0]
            damage_str = ", ".join(damage_parts)
        else:
            damage_str = "No damage"
        wear_str = f"FL:{self.tire_wear['fl']:.0f}% FR:{self.tire_wear['fr']:.0f}% RL:{self.tire_wear['rl']:.0f}% RR:{self.tire_wear['rr']:.0f}%"
        pressure_str = f"FL:{self.tire_pressures['fl']:.1f}psi FR:{self.tire_pressures['fr']:.1f}psi RL:{self.tire_pressures['rl']:.1f}psi RR:{self.tire_pressures['rr']:.1f}psi"

        # Pit summary: condensed single-line overview for pit-strategy queries
        max_tire_temp = max(self.tire_temps.values()) if self.tire_temps else 0.0
        max_temp_corner = max(self.tire_temps, key=self.tire_temps.get) if self.tire_temps else "N/A"
        max_tire_wear_val = max(self.tire_wear.values()) if self.tire_wear else 0.0
        max_wear_corner = max(self.tire_wear, key=self.tire_wear.get) if self.tire_wear else "N/A"
        pit_summary_str = (
            f"Fuel {fuel_laps_str} laps"
            f" | Max Tire Temp: {max_tire_temp:.0f}C ({max_temp_corner.upper()})"
            f" | Max Tire Wear: {max_tire_wear_val:.0f}% ({max_wear_corner.upper()})"
            f" | Damage: {total_damage:.0f}% total"
        )

        # Build context lines conditionally
        all_lines = [
            ("track", f"Track: {self.track_name}"),
            ("lap_position", f"Lap: {self.current_lap} | Position: P{self.position}"),
            ("speed_gear_rpm", f"Speed: {self.speed_kmh:.0f} km/h | Gear: {self.gear} | RPM: {self.rpm}"),
            ("throttle_brake", f"Throttle: {self.throttle:.0%} | Brake: {self.brake:.0%}"),
            ("g_forces", f"G-Forces: Lat {self.g_force_lat:.2f}g | Lon {self.g_force_lon:.2f}g"),
            ("gap_ahead_behind", f"Gap Ahead: {gap_ahead_str} | Gap Behind: {gap_behind_str}"),
            ("nearby_opponents", f"Nearby Opponents: {nearby_str}"),
            ("fuel", f"Fuel: {self.fuel_remaining:.1f}L ({fuel_laps_str} laps)"),
            ("pit_summary", f"Pit Summary: {pit_summary_str}"),
            ("tire_temps", f"Tire Temps: FL:{self.tire_temps['fl']:.0f}°C FR:{self.tire_temps['fr']:.0f}°C RL:{self.tire_temps['rl']:.0f}°C RR:{self.tire_temps['rr']:.0f}°C"),
            ("tire_pressures", f"Tire Pressures: {pressure_str}"),
            ("tire_wear", f"Tire Wear: {wear_str}"),
            ("car_damage", f"Car Damage: {damage_str}"),
            ("best_last_lap", f"Best Lap: {best_lap_str} | Last Lap: {last_lap_str}"),
        ]

        lines = [text for label, text in all_lines if label in active_labels]
        return "\n".join(lines)

    def _format_nearby_opponents(self) -> str:
        """Format up to three opponents nearest in race position."""
        if not self.opponents:
            return "N/A"

        sorted_cars = sorted(
            self.opponents,
            key=lambda c: abs(c.position - self.position),
        )[:3]
        return ", ".join(
            f"P{car.position}(car {car.car_index}, {car.speed:.0f} km/h)"
            for car in sorted_cars
        )

    def _format_lap_time(self, seconds: float) -> str:
        """Format lap time as M:SS.mmm."""
        minutes = int(seconds // 60)
        remaining = seconds % 60
        return f"{minutes}:{remaining:06.3f}"
