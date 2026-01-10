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
- gap_change: Gap changes > 1s (MEDIUM)
- lap_complete: Lap number increased (MEDIUM)
- sector_complete: Sector changed (LOW)
- pit_window_open: Fuel or tire wear at warning levels (HIGH)
"""

import time
from typing import List, Optional

from jarvis_granite.schemas.telemetry import TelemetryData
from jarvis_granite.schemas.events import (
    Event,
    Priority,
    create_fuel_critical_event,
    create_fuel_warning_event,
    create_tire_critical_event,
    create_tire_warning_event,
    create_gap_change_event,
    create_lap_complete_event,
    create_sector_complete_event,
    create_pit_window_event,
)
from jarvis_granite.live.context import LiveSessionContext
from config.config import ThresholdsConfig


class TelemetryAgent:
    """
    Rule-based telemetry processing agent.

    Detects events from telemetry data based on configurable thresholds.
    No LLM calls - pure rule-based logic for <50ms latency.

    Attributes:
        thresholds: Configuration for event trigger thresholds
    """

    def __init__(self, thresholds: Optional[ThresholdsConfig] = None):
        """
        Initialize TelemetryAgent.

        Args:
            thresholds: Threshold configuration. Uses defaults if not provided.
        """
        self.thresholds = thresholds or ThresholdsConfig()

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

        # Check gap change events
        gap_events = self._check_gap_events(telemetry, context)
        events.extend(gap_events)

        # Check lap completion
        lap_events = self._check_lap_completion(telemetry, context)
        events.extend(lap_events)

        # Check sector completion
        sector_events = self._check_sector_completion(telemetry, context)
        events.extend(sector_events)

        # Check pit window
        pit_events = self._check_pit_window(telemetry, context, events)
        events.extend(pit_events)

        return events

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

        # Can't calculate fuel laps if consumption is unknown
        if context.fuel_consumption_per_lap <= 0:
            return events

        fuel_laps_remaining = telemetry.fuel_remaining / context.fuel_consumption_per_lap

        # Check critical first (takes precedence)
        if fuel_laps_remaining <= self.thresholds.fuel_critical_laps:
            events.append(create_fuel_critical_event(fuel_laps_remaining))
        # Only check warning if not critical
        elif fuel_laps_remaining <= self.thresholds.fuel_warning_laps:
            events.append(create_fuel_warning_event(fuel_laps_remaining))

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

        if telemetry.lap_number > context.current_lap:
            events.append(create_lap_complete_event(
                lap_number=telemetry.lap_number,
                lap_time=context.last_lap or 0.0,
                best_lap=context.best_lap
            ))

        return events

    def _check_sector_completion(
        self,
        telemetry: TelemetryData,
        context: LiveSessionContext
    ) -> List[Event]:
        """
        Check for sector completion event.

        Returns:
            List containing sector_complete event if sector changed
        """
        events = []

        # Sector changed
        if telemetry.sector != context.current_sector:
            # The completed sector is the previous sector
            completed_sector = context.current_sector

            # Handle wrap-around (sector 3 -> sector 1)
            if telemetry.sector == 1 and context.current_sector == 3:
                completed_sector = 3

            events.append(create_sector_complete_event(
                sector=completed_sector,
                sector_time=0.0  # Would need sector timing data
            ))

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

        # Check tire wear for pit window
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
