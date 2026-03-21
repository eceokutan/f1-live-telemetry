"""
Telemetry Agent for Jarvis-Granite Live Telemetry.

Rule-based telemetry processing with no LLM dependency.
Latency budget: <50ms

This agent:
- Parses and validates incoming telemetry data
- Detects events based on configurable thresholds
- Returns a list of prioritized events for the orchestrator

Event Types:
- fuel_critical: Fuel < 2 laps remaining (CRITICAL)
- fuel_warning: Fuel < 5 laps remaining (HIGH)
- tire_critical: Tire temperature > 110C (CRITICAL)
- tire_warning: Tire temperature > 100C (MEDIUM)
- tire_wear_critical: Tire wear > 85% (HIGH)
- wheel_slip_critical: Wheel slip > 10.0 (HIGH)
- wheel_slip_warning: Wheel slip > 5.0 (MEDIUM)
- gap_change: Gap changes > 1s (MEDIUM)
- opponent_close_behind: Car behind is within close gap threshold (HIGH)
- car_damage_alert: Car damage crossed warning/critical threshold (MEDIUM/HIGH)
- lap_complete: Lap number increased (MEDIUM)
- pit_window_open: Fuel or tire wear at warning levels (HIGH)
"""

import time
from typing import List, Optional

from ai.race_engineer_core.context import LiveSessionContext
from ai.race_engineer_core.events import (
    Event,
    Priority,
    create_fuel_critical_event,
    create_fuel_warning_event,
    create_tire_critical_event,
    create_tire_warning_event,
    create_gap_change_event,
    create_lap_complete_event,
    create_pit_window_event,
    create_wheel_slip_warning_event,
    create_wheel_slip_critical_event,
    create_opponent_close_behind_event,
    create_car_damage_event,
)
from ai.race_engineer_core.telemetry import TelemetryData
from ai.race_engineer_core.config import ThresholdsConfig


class TelemetryAgent:
    """
    Rule-based telemetry processing agent.

    Detects events from telemetry data based on configurable thresholds.
    No LLM calls - pure rule-based logic for <50ms latency.

    Attributes:
        thresholds: Configuration for event trigger thresholds
    """

    # Cooldown periods (seconds) to avoid spamming the driver
    FUEL_CRITICAL_COOLDOWN = 45.0
    FUEL_WARNING_COOLDOWN = 60.0
    OPPONENT_CLOSE_COOLDOWN = 15.0
    CAR_DAMAGE_COOLDOWN = 20.0

    def __init__(self, thresholds: Optional[ThresholdsConfig] = None):
        """
        Initialize TelemetryAgent.

        Args:
            thresholds: Threshold configuration. Uses defaults if not provided.
        """
        self.thresholds = thresholds or ThresholdsConfig()
        self._last_fuel_critical_time: float = 0.0
        self._last_fuel_warning_time: float = 0.0
        self._last_opponent_close_time: float = 0.0
        self._opponent_close_active: bool = False
        self._last_car_damage_time: float = 0.0

    def detect_events(
        self,
        telemetry: TelemetryData,
        context: LiveSessionContext
    ) -> List[Event]:
        """
        Detect events from telemetry data.

        This is the main entry point called by the orchestrator.
        Checks all event conditions and returns a list of detected events.

        Args:
            telemetry: Current telemetry snapshot
            context: Session context with previous state

        Returns:
            List of Event objects, sorted by priority (highest first)
        """
        events: List[Event] = []

        # Check fuel events
        fuel_events = self._check_fuel_events(telemetry, context)
        events.extend(fuel_events)

        # Check tire temperature events
        tire_temp_events = self._check_tire_temp_events(telemetry)
        events.extend(tire_temp_events)

        # Check tire wear events
        tire_wear_events = self._check_tire_wear_events(telemetry)
        events.extend(tire_wear_events)

        # Check wheel slip events
        wheel_slip_events = self._check_wheel_slip_events(telemetry)
        events.extend(wheel_slip_events)

        # Check gap change events
        gap_events = self._check_gap_events(telemetry, context)
        events.extend(gap_events)

        # Check close-behind opponent threat
        opponent_events = self._check_opponent_events(telemetry, context)
        events.extend(opponent_events)

        # Check car damage alerts
        damage_events = self._check_car_damage_events(telemetry, context)
        events.extend(damage_events)

        # Check lap completion
        lap_events = self._check_lap_completion(telemetry, context)
        events.extend(lap_events)

        # Check pit window
        pit_events = self._check_pit_window(telemetry, context, events)
        events.extend(pit_events)

        # Enforce priority ordering before handing off to the worker.
        # Lower enum values are higher priority (CRITICAL=0).
        return sorted(events, key=lambda e: (int(e.priority), e.timestamp))

    def _check_fuel_events(
        self,
        telemetry: TelemetryData,
        context: LiveSessionContext
    ) -> List[Event]:
        """
        Check for fuel-related events.

        Returns:
            List of fuel events (either critical OR warning, not both)
        """
        events = []

        # Can't calculate fuel laps if consumption is unknown or fuel data missing
        if context.fuel_consumption_per_lap <= 0 or telemetry.fuel is None:
            return events

        fuel_laps_remaining = telemetry.fuel / context.fuel_consumption_per_lap
        now = time.time()

        # Check critical first (takes precedence) — cooldown ~45s
        if fuel_laps_remaining <= self.thresholds.fuel_critical_laps:
            if now - self._last_fuel_critical_time >= self.FUEL_CRITICAL_COOLDOWN:
                events.append(create_fuel_critical_event(fuel_laps_remaining))
                self._last_fuel_critical_time = now
        # Only check warning if not critical — cooldown ~60s
        elif fuel_laps_remaining <= self.thresholds.fuel_warning_laps:
            if now - self._last_fuel_warning_time >= self.FUEL_WARNING_COOLDOWN:
                events.append(create_fuel_warning_event(fuel_laps_remaining))
                self._last_fuel_warning_time = now

        return events

    def _check_tire_temp_events(self, telemetry: TelemetryData) -> List[Event]:
        """
        Check for tire temperature events.

        Returns:
            List of tire temperature events (critical or warning per tire)
        """
        events = []

        tire_positions = {
            "fl": telemetry.tire_temps.fl,
            "fr": telemetry.tire_temps.fr,
            "rl": telemetry.tire_temps.rl,
            "rr": telemetry.tire_temps.rr,
        }

        for position, temp in tire_positions.items():
            # Check critical first (takes precedence for each tire)
            if temp >= self.thresholds.tire_temp_critical:
                events.append(create_tire_critical_event(temp, position))
            elif temp >= self.thresholds.tire_temp_warning:
                events.append(create_tire_warning_event(temp, position))

        return events

    def _check_tire_wear_events(self, telemetry: TelemetryData) -> List[Event]:
        """
        Check for tire wear events.

        Returns:
            List of tire wear critical events
        """
        events = []

        # AC doesn't provide tire wear data - skip if not available
        if telemetry.tire_wear is None:
            return events

        tire_wear = {
            "fl": telemetry.tire_wear.fl,
            "fr": telemetry.tire_wear.fr,
            "rl": telemetry.tire_wear.rl,
            "rr": telemetry.tire_wear.rr,
        }

        for position, wear in tire_wear.items():
            if wear >= self.thresholds.tire_wear_critical:
                events.append(Event(
                    type="tire_wear_critical",
                    priority=Priority.HIGH,
                    data={"wear": wear, "position": position},
                    timestamp=time.time()
                ))

        return events

    def _check_wheel_slip_events(self, telemetry: TelemetryData) -> List[Event]:
        """
        Check for excessive wheel slip events.

        Returns:
            List of wheel slip events (critical or warning per tire)
        """
        events = []

        if telemetry.wheel_slip is None:
            return events

        # Slip ratio is meaningless at low speed (denominator ≈ 0 → huge values)
        if telemetry.speed < 10.0:
            return events

        slip_positions = {
            "fl": abs(telemetry.wheel_slip.fl),
            "fr": abs(telemetry.wheel_slip.fr),
            "rl": abs(telemetry.wheel_slip.rl),
            "rr": abs(telemetry.wheel_slip.rr),
        }

        for position, slip in slip_positions.items():
            if slip >= self.thresholds.wheel_slip_critical:
                events.append(create_wheel_slip_critical_event(slip, position))
            elif slip >= self.thresholds.wheel_slip_warning:
                events.append(create_wheel_slip_warning_event(slip, position))

        return events

    def _check_gap_events(
        self,
        telemetry: TelemetryData,
        context: LiveSessionContext
    ) -> List[Event]:
        """
        Check for gap change events.

        Returns:
            List of gap change events
        """
        events = []

        # Check gap ahead
        if context.gap_ahead is not None and telemetry.gap_ahead is not None:
            gap_change = abs(telemetry.gap_ahead - context.gap_ahead)
            if gap_change >= self.thresholds.gap_change_threshold:
                events.append(create_gap_change_event(
                    gap_change=gap_change,
                    direction="ahead",
                    new_gap=telemetry.gap_ahead
                ))

        # Check gap behind
        if context.gap_behind is not None and telemetry.gap_behind is not None:
            gap_change = abs(telemetry.gap_behind - context.gap_behind)
            if gap_change >= self.thresholds.gap_change_threshold:
                events.append(create_gap_change_event(
                    gap_change=gap_change,
                    direction="behind",
                    new_gap=telemetry.gap_behind
                ))

        return events

    def _check_lap_completion(
        self,
        telemetry: TelemetryData,
        context: LiveSessionContext
    ) -> List[Event]:
        """
        Check for lap completion event.

        Returns:
            List containing lap_complete event if lap changed
        """
        events = []

        # Check for lap completion (skip if lap_number not available in AC)
        if (telemetry.lap_number is not None and
            context.current_lap is not None and
            telemetry.lap_number > context.current_lap):
            # Prefer the game's authoritative lap time (lastTimeMs) over
            # computed timestamps which are imprecise due to telemetry throttling.
            if telemetry.last_time_ms and telemetry.last_time_ms > 0:
                lap_time = telemetry.last_time_ms / 1000.0
            else:
                lap_time = telemetry.t - context._lap_start_time if context._lap_start_time > 0 else 0.0
            # AC uses 0-indexed lap IDs; display as 1-indexed for the driver
            display_lap = context.current_lap + 1
            events.append(create_lap_complete_event(
                lap_number=display_lap,
                lap_time=lap_time,
                best_lap=context.best_lap
            ))

        return events

    def _check_opponent_events(
        self,
        telemetry: TelemetryData,
        context: LiveSessionContext
    ) -> List[Event]:
        """
        Check for a close opponent behind.

        Returns:
            List containing opponent_close_behind event if conditions met
        """
        events = []
        gap_behind = telemetry.gap_behind

        if gap_behind is None:
            return events

        # Invalid/negative gaps are ignored.
        if gap_behind < 0:
            return events

        if gap_behind <= self.thresholds.opponent_close_behind_gap:
            now = time.time()
            should_emit = (
                not self._opponent_close_active
                or now - self._last_opponent_close_time >= self.OPPONENT_CLOSE_COOLDOWN
            )

            if should_emit:
                own_position = telemetry.position if telemetry.position is not None else context.position
                car_index = None
                opponent_position = None
                opponent_speed = None

                if telemetry.opponents:
                    closest_behind = None
                    if own_position is not None:
                        candidates = [
                            car for car in telemetry.opponents
                            if car.position >= own_position + 1
                        ]
                        if candidates:
                            closest_behind = min(candidates, key=lambda c: c.position)
                    if closest_behind is None:
                        closest_behind = min(
                            telemetry.opponents,
                            key=lambda c: c.position,
                        )

                    car_index = closest_behind.car_index
                    opponent_position = closest_behind.position
                    opponent_speed = closest_behind.speed

                events.append(
                    create_opponent_close_behind_event(
                        gap=gap_behind,
                        car_index=car_index,
                        position=opponent_position,
                        speed=opponent_speed,
                    )
                )
                self._last_opponent_close_time = now

            self._opponent_close_active = True
        elif gap_behind >= self.thresholds.opponent_close_reset_gap:
            # Rear threat has backed off enough to allow a fresh trigger next time.
            self._opponent_close_active = False

        return events

    def _check_car_damage_events(
        self,
        telemetry: TelemetryData,
        context: LiveSessionContext
    ) -> List[Event]:
        """
        Check for new car damage that should be reported.

        Returns:
            List containing car_damage_alert event if thresholds are crossed
        """
        events = []

        if telemetry.car_damage is None:
            return events

        zones = {zone: max(0.0, float(value)) for zone, value in telemetry.car_damage.items()}
        total_damage = sum(zones.values())
        previous_total = sum(max(0.0, float(value)) for value in context.car_damage.values())
        damage_delta = max(0.0, total_damage - previous_total)

        warning_threshold = self.thresholds.car_damage_warning_total
        critical_threshold = self.thresholds.car_damage_critical_total
        delta_threshold = self.thresholds.car_damage_delta_threshold

        crossed_warning = previous_total < warning_threshold <= total_damage
        crossed_critical = previous_total < critical_threshold <= total_damage
        significant_increase = damage_delta >= delta_threshold and total_damage >= warning_threshold

        if not (crossed_warning or crossed_critical or significant_increase):
            return events

        now = time.time()
        if now - self._last_car_damage_time < self.CAR_DAMAGE_COOLDOWN:
            return events

        severity = "critical" if total_damage >= critical_threshold else "warning"
        events.append(
            create_car_damage_event(
                total_damage=total_damage,
                zones=zones,
                severity=severity,
            )
        )
        self._last_car_damage_time = now

        return events

    def _check_pit_window(
        self,
        telemetry: TelemetryData,
        context: LiveSessionContext,
        existing_events: List[Event]
    ) -> List[Event]:
        """
        Check if pit window should open.

        Pit window opens when:
        - Fuel is at warning level (but not critical - that's more urgent)
        - Tire wear exceeds warning threshold

        Args:
            existing_events: Already detected events (to avoid duplication)

        Returns:
            List containing pit_window_open event if conditions met
        """
        events = []

        # Get existing event types to check for duplicates
        existing_types = {e.type for e in existing_events}

        # Don't open pit window if there's already a critical fuel event
        if "fuel_critical" in existing_types:
            return events

        # Check if fuel is at warning level (triggers pit window)
        if "fuel_warning" in existing_types:
            events.append(create_pit_window_event(reason="fuel"))
            return events  # Return early, one pit window reason is enough

        # Check tire wear for pit window (skip if not available in AC)
        if telemetry.tire_wear is not None:
            tire_wear = {
                "fl": telemetry.tire_wear.fl,
                "fr": telemetry.tire_wear.fr,
                "rl": telemetry.tire_wear.rl,
                "rr": telemetry.tire_wear.rr,
            }

            max_wear = max(tire_wear.values())
            if max_wear >= self.thresholds.tire_wear_warning:
                events.append(create_pit_window_event(reason="tires"))

        return events
