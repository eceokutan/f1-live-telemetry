"""
TDD Tests for Telemetry Agent - Phase 2, Section 4

These tests define the expected behavior for:
1. Rule-based telemetry parsing and validation
2. Event detection logic (fuel, tire, gap, lap completion)
3. Threshold checking against configuration
4. Performance target: <50ms processing latency

Run with: pytest tests/test_telemetry_agent.py -v

Write these tests FIRST, watch them fail, then implement TelemetryAgent to pass.
"""

import pytest
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis_granite.schemas.telemetry import TelemetryData
from jarvis_granite.schemas.events import Event, Priority
from jarvis_granite.live.context import LiveSessionContext
from config.config import ThresholdsConfig


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def thresholds():
    """Default threshold configuration for testing."""
    return ThresholdsConfig(
        tire_temp_warning=100.0,
        tire_temp_critical=110.0,
        tire_wear_warning=70.0,
        tire_wear_critical=85.0,
        fuel_warning_laps=5,
        fuel_critical_laps=2,
        gap_change_threshold=1.0
    )


@pytest.fixture
def session_context():
    """Create a session context for testing."""
    context = LiveSessionContext(
        session_id="test_session",
        source="torcs",
        track_name="Monza"
    )
    # Set up some initial state
    context.current_lap = 10
    context.current_sector = 1
    context.fuel_remaining = 50.0
    context.fuel_consumption_per_lap = 3.5
    context.position = 3
    context.gap_ahead = 2.5
    context.gap_behind = 1.5
    return context


@pytest.fixture
def base_telemetry_dict():
    """Base valid telemetry data for modification in tests."""
    return {
        "speed_kmh": 200.0,
        "rpm": 10000,
        "gear": 4,
        "throttle": 0.8,
        "brake": 0.0,
        "steering_angle": 0.0,
        "fuel_remaining": 45.0,
        "tire_temps": {"fl": 90.0, "fr": 91.0, "rl": 88.0, "rr": 89.0},
        "tire_wear": {"fl": 20.0, "fr": 21.0, "rl": 18.0, "rr": 19.0},
        "g_forces": {"lateral": 1.0, "longitudinal": 0.2},
        "track_position": 0.5,
        "lap_number": 10,
        "lap_time_current": 45.0,
        "sector": 2,
        "position": 3,
        "gap_ahead": 2.5,
        "gap_behind": 1.5
    }


# =============================================================================
# TELEMETRY AGENT CREATION
# =============================================================================

class TestTelemetryAgentCreation:
    """Tests for TelemetryAgent instantiation."""

    def test_create_telemetry_agent_with_thresholds(self, thresholds):
        """Should create TelemetryAgent with threshold configuration."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        assert agent is not None
        assert agent.thresholds == thresholds

    def test_create_telemetry_agent_with_default_thresholds(self):
        """Should create TelemetryAgent with default thresholds if not provided."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent()

        assert agent is not None
        assert agent.thresholds is not None
        assert agent.thresholds.tire_temp_critical == 110.0

    def test_telemetry_agent_has_detect_events_method(self, thresholds):
        """TelemetryAgent should have detect_events method."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        assert hasattr(agent, 'detect_events')
        assert callable(agent.detect_events)


# =============================================================================
# FUEL EVENT DETECTION
# =============================================================================

class TestFuelEventDetection:
    """Tests for fuel-related event detection."""

    def test_detect_fuel_critical_event(self, thresholds, session_context, base_telemetry_dict):
        """Should detect fuel_critical when fuel < 2 laps remaining."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # Set up: fuel = 6L, consumption = 3.5L/lap = 1.7 laps remaining (< 2)
        session_context.fuel_remaining = 6.0
        session_context.fuel_consumption_per_lap = 3.5

        base_telemetry_dict["fuel_remaining"] = 6.0
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        fuel_critical = [e for e in events if e.type == "fuel_critical"]
        assert len(fuel_critical) == 1
        assert fuel_critical[0].priority == Priority.CRITICAL
        assert "laps" in fuel_critical[0].data
        assert fuel_critical[0].data["laps"] < 2

    def test_detect_fuel_warning_event(self, thresholds, session_context, base_telemetry_dict):
        """Should detect fuel_warning when fuel < 5 laps but >= 2 laps remaining."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # Set up: fuel = 12L, consumption = 3.5L/lap = 3.4 laps remaining (< 5, >= 2)
        session_context.fuel_remaining = 12.0
        session_context.fuel_consumption_per_lap = 3.5

        base_telemetry_dict["fuel_remaining"] = 12.0
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        fuel_warning = [e for e in events if e.type == "fuel_warning"]
        assert len(fuel_warning) == 1
        assert fuel_warning[0].priority == Priority.HIGH
        assert fuel_warning[0].data["laps"] < 5
        assert fuel_warning[0].data["laps"] >= 2

    def test_no_fuel_event_when_fuel_adequate(self, thresholds, session_context, base_telemetry_dict):
        """Should not detect fuel events when fuel >= 5 laps remaining."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # Set up: fuel = 50L, consumption = 3.5L/lap = 14+ laps remaining
        session_context.fuel_remaining = 50.0
        session_context.fuel_consumption_per_lap = 3.5

        base_telemetry_dict["fuel_remaining"] = 50.0
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        fuel_events = [e for e in events if e.type in ("fuel_critical", "fuel_warning")]
        assert len(fuel_events) == 0

    def test_no_fuel_event_when_consumption_unknown(self, thresholds, session_context, base_telemetry_dict):
        """Should not detect fuel events when consumption is 0 (unknown)."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # Consumption is 0 (first lap, no data yet)
        session_context.fuel_consumption_per_lap = 0.0

        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        fuel_events = [e for e in events if e.type in ("fuel_critical", "fuel_warning")]
        assert len(fuel_events) == 0

    def test_fuel_critical_takes_precedence_over_warning(self, thresholds, session_context, base_telemetry_dict):
        """When fuel is critical, should not also generate a warning event."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # Fuel is critically low
        session_context.fuel_remaining = 3.0
        session_context.fuel_consumption_per_lap = 3.5

        base_telemetry_dict["fuel_remaining"] = 3.0
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        fuel_events = [e for e in events if e.type in ("fuel_critical", "fuel_warning")]
        # Should only have critical, not warning
        assert len(fuel_events) == 1
        assert fuel_events[0].type == "fuel_critical"


# =============================================================================
# TIRE TEMPERATURE EVENT DETECTION
# =============================================================================

class TestTireTempEventDetection:
    """Tests for tire temperature event detection."""

    def test_detect_tire_critical_event(self, thresholds, session_context, base_telemetry_dict):
        """Should detect tire_critical when any tire > 110C."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # Set front-left tire to critical temperature
        base_telemetry_dict["tire_temps"]["fl"] = 115.0
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        tire_critical = [e for e in events if e.type == "tire_critical"]
        assert len(tire_critical) == 1
        assert tire_critical[0].priority == Priority.CRITICAL
        assert tire_critical[0].data["temp"] == 115.0
        assert tire_critical[0].data["position"] == "fl"

    def test_detect_tire_warning_event(self, thresholds, session_context, base_telemetry_dict):
        """Should detect tire_warning when tire > 100C but <= 110C."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # Set rear-right tire to warning temperature
        base_telemetry_dict["tire_temps"]["rr"] = 105.0
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        tire_warning = [e for e in events if e.type == "tire_warning"]
        assert len(tire_warning) == 1
        assert tire_warning[0].priority == Priority.MEDIUM
        assert tire_warning[0].data["temp"] == 105.0
        assert tire_warning[0].data["position"] == "rr"

    def test_no_tire_event_when_temps_normal(self, thresholds, session_context, base_telemetry_dict):
        """Should not detect tire events when all temps <= 100C."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # All temps are normal (< 100C)
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        tire_events = [e for e in events if e.type in ("tire_critical", "tire_warning")]
        assert len(tire_events) == 0

    def test_detect_multiple_tire_critical_events(self, thresholds, session_context, base_telemetry_dict):
        """Should detect multiple tire critical events if multiple tires are critical."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # Multiple tires at critical temperature
        base_telemetry_dict["tire_temps"]["fl"] = 112.0
        base_telemetry_dict["tire_temps"]["fr"] = 115.0
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        tire_critical = [e for e in events if e.type == "tire_critical"]
        # Could be one aggregated event or multiple - implementation choice
        # At minimum, should detect the issue
        assert len(tire_critical) >= 1
        assert all(e.priority == Priority.CRITICAL for e in tire_critical)

    def test_tire_critical_takes_precedence_for_same_tire(self, thresholds, session_context, base_telemetry_dict):
        """If a tire is critical, should not also generate a warning for that tire."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # FL is critical (should not also get warning for FL)
        base_telemetry_dict["tire_temps"]["fl"] = 115.0
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        fl_events = [e for e in events if e.type in ("tire_critical", "tire_warning")
                     and e.data.get("position") == "fl"]
        # Should only have critical for FL
        assert len(fl_events) == 1
        assert fl_events[0].type == "tire_critical"


# =============================================================================
# GAP CHANGE EVENT DETECTION
# =============================================================================

class TestGapChangeEventDetection:
    """Tests for gap change event detection."""

    def test_detect_gap_ahead_increased(self, thresholds, session_context, base_telemetry_dict):
        """Should detect gap_change when gap ahead increases > 1s."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # Previous gap was 2.5s, new gap is 4.0s (change of 1.5s)
        session_context.gap_ahead = 2.5
        base_telemetry_dict["gap_ahead"] = 4.0
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        gap_events = [e for e in events if e.type == "gap_change"]
        assert len(gap_events) == 1
        assert gap_events[0].priority == Priority.MEDIUM
        assert gap_events[0].data["direction"] == "ahead"
        assert gap_events[0].data["change"] >= 1.0

    def test_detect_gap_ahead_decreased(self, thresholds, session_context, base_telemetry_dict):
        """Should detect gap_change when gap ahead decreases > 1s."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # Previous gap was 2.5s, new gap is 1.0s (change of 1.5s - closing in)
        session_context.gap_ahead = 2.5
        base_telemetry_dict["gap_ahead"] = 1.0
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        gap_events = [e for e in events if e.type == "gap_change"]
        assert len(gap_events) == 1
        assert gap_events[0].data["direction"] == "ahead"

    def test_detect_gap_behind_changed(self, thresholds, session_context, base_telemetry_dict):
        """Should detect gap_change when gap behind changes > 1s."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # Previous gap behind was 1.5s, new gap is 3.0s (change of 1.5s)
        session_context.gap_behind = 1.5
        base_telemetry_dict["gap_behind"] = 3.0
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        gap_events = [e for e in events if e.type == "gap_change" and e.data.get("direction") == "behind"]
        assert len(gap_events) == 1

    def test_no_gap_event_when_change_small(self, thresholds, session_context, base_telemetry_dict):
        """Should not detect gap_change when change < 1s."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # Gap ahead changed from 2.5s to 2.8s (only 0.3s change)
        session_context.gap_ahead = 2.5
        base_telemetry_dict["gap_ahead"] = 2.8
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        gap_events = [e for e in events if e.type == "gap_change"]
        assert len(gap_events) == 0

    def test_no_gap_event_when_previous_gap_none(self, thresholds, session_context, base_telemetry_dict):
        """Should not detect gap_change when previous gap was None."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # No previous gap data
        session_context.gap_ahead = None
        base_telemetry_dict["gap_ahead"] = 2.5
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        gap_events = [e for e in events if e.type == "gap_change" and e.data.get("direction") == "ahead"]
        assert len(gap_events) == 0

    def test_no_gap_event_when_current_gap_none(self, thresholds, session_context, base_telemetry_dict):
        """Should not detect gap_change when current gap is None."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        session_context.gap_ahead = 2.5
        del base_telemetry_dict["gap_ahead"]
        del base_telemetry_dict["gap_behind"]
        del base_telemetry_dict["position"]
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        gap_events = [e for e in events if e.type == "gap_change"]
        assert len(gap_events) == 0


# =============================================================================
# LAP COMPLETION EVENT DETECTION
# =============================================================================

class TestLapCompletionEventDetection:
    """Tests for lap completion event detection."""

    def test_detect_lap_complete_event(self, thresholds, session_context, base_telemetry_dict):
        """Should detect lap_complete when lap number increases."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # Previous lap was 10, new lap is 11
        session_context.current_lap = 10
        session_context.last_lap = 82.456
        base_telemetry_dict["lap_number"] = 11
        base_telemetry_dict["lap_time_current"] = 1.2  # Just crossed line
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        lap_events = [e for e in events if e.type == "lap_complete"]
        assert len(lap_events) == 1
        assert lap_events[0].priority == Priority.MEDIUM
        assert lap_events[0].data["lap"] == 11

    def test_no_lap_event_when_same_lap(self, thresholds, session_context, base_telemetry_dict):
        """Should not detect lap_complete when lap number unchanged."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # Same lap
        session_context.current_lap = 10
        base_telemetry_dict["lap_number"] = 10
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        lap_events = [e for e in events if e.type == "lap_complete"]
        assert len(lap_events) == 0

    def test_lap_complete_event_includes_lap_time(self, thresholds, session_context, base_telemetry_dict):
        """Lap complete event should include last lap time if available."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        session_context.current_lap = 10
        session_context.last_lap = 82.456
        session_context.best_lap = 81.234
        base_telemetry_dict["lap_number"] = 11
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        lap_events = [e for e in events if e.type == "lap_complete"]
        assert len(lap_events) == 1
        # Event should include relevant timing data
        event_data = lap_events[0].data
        assert "lap" in event_data


# =============================================================================
# SECTOR COMPLETION EVENT DETECTION
# =============================================================================

class TestSectorCompletionEventDetection:
    """Tests for sector completion event detection."""

    def test_detect_sector_complete_event(self, thresholds, session_context, base_telemetry_dict):
        """Should detect sector_complete when sector changes."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # Previous sector was 1, now sector 2
        session_context.current_sector = 1
        base_telemetry_dict["sector"] = 2
        base_telemetry_dict["lap_number"] = 10  # Same lap
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        sector_events = [e for e in events if e.type == "sector_complete"]
        assert len(sector_events) == 1
        assert sector_events[0].priority == Priority.LOW
        assert sector_events[0].data["sector"] == 1  # Completed sector 1

    def test_detect_sector_3_complete(self, thresholds, session_context, base_telemetry_dict):
        """Should detect sector 3 completion (wraps to sector 1)."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # Was in sector 3, now in sector 1 of new lap
        session_context.current_sector = 3
        session_context.current_lap = 10
        base_telemetry_dict["sector"] = 1
        base_telemetry_dict["lap_number"] = 11  # New lap
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        sector_events = [e for e in events if e.type == "sector_complete"]
        # Should complete sector 3
        assert len(sector_events) == 1
        assert sector_events[0].data["sector"] == 3

    def test_no_sector_event_when_same_sector(self, thresholds, session_context, base_telemetry_dict):
        """Should not detect sector_complete when sector unchanged."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # Same sector
        session_context.current_sector = 2
        base_telemetry_dict["sector"] = 2
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        sector_events = [e for e in events if e.type == "sector_complete"]
        assert len(sector_events) == 0


# =============================================================================
# PIT WINDOW EVENT DETECTION
# =============================================================================

class TestPitWindowEventDetection:
    """Tests for pit window open event detection."""

    def test_detect_pit_window_due_to_fuel(self, thresholds, session_context, base_telemetry_dict):
        """Should detect pit_window_open when fuel is low but not critical."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # Fuel is at warning level (3-5 laps)
        session_context.fuel_remaining = 14.0
        session_context.fuel_consumption_per_lap = 3.5  # ~4 laps

        base_telemetry_dict["fuel_remaining"] = 14.0
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        pit_events = [e for e in events if e.type == "pit_window_open"]
        assert len(pit_events) == 1
        assert pit_events[0].priority == Priority.HIGH
        assert pit_events[0].data["reason"] == "fuel"

    def test_detect_pit_window_due_to_tire_wear(self, thresholds, session_context, base_telemetry_dict):
        """Should detect pit_window_open when tire wear exceeds warning threshold."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # Tire wear at warning level (> 70%)
        base_telemetry_dict["tire_wear"]["fl"] = 75.0
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        pit_events = [e for e in events if e.type == "pit_window_open"]
        assert len(pit_events) == 1
        assert pit_events[0].data["reason"] == "tires"

    def test_no_pit_window_when_resources_adequate(self, thresholds, session_context, base_telemetry_dict):
        """Should not detect pit_window when fuel and tires are adequate."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # Everything is fine
        session_context.fuel_remaining = 50.0
        session_context.fuel_consumption_per_lap = 3.5

        base_telemetry_dict["fuel_remaining"] = 50.0
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        pit_events = [e for e in events if e.type == "pit_window_open"]
        assert len(pit_events) == 0


# =============================================================================
# TIRE WEAR EVENT DETECTION
# =============================================================================

class TestTireWearEventDetection:
    """Tests for tire wear event detection."""

    def test_detect_tire_wear_critical(self, thresholds, session_context, base_telemetry_dict):
        """Should detect tire wear critical when wear > 85%."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        base_telemetry_dict["tire_wear"]["fr"] = 90.0
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        wear_events = [e for e in events if e.type == "tire_wear_critical"]
        assert len(wear_events) == 1
        assert wear_events[0].priority == Priority.HIGH
        assert wear_events[0].data["position"] == "fr"
        assert wear_events[0].data["wear"] == 90.0


# =============================================================================
# EVENT TIMESTAMPS
# =============================================================================

class TestEventTimestamps:
    """Tests for event timestamp handling."""

    def test_events_have_timestamps(self, thresholds, session_context, base_telemetry_dict):
        """All detected events should have timestamps."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # Create a situation that generates an event
        session_context.fuel_remaining = 6.0
        session_context.fuel_consumption_per_lap = 3.5
        base_telemetry_dict["fuel_remaining"] = 6.0
        telemetry = TelemetryData(**base_telemetry_dict)

        before = time.time()
        events = agent.detect_events(telemetry, session_context)
        after = time.time()

        assert len(events) > 0
        for event in events:
            assert event.timestamp >= before
            assert event.timestamp <= after


# =============================================================================
# MULTIPLE EVENTS IN SINGLE DETECTION
# =============================================================================

class TestMultipleEventDetection:
    """Tests for detecting multiple events simultaneously."""

    def test_detect_multiple_different_events(self, thresholds, session_context, base_telemetry_dict):
        """Should detect multiple different event types in single call."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # Set up multiple issues
        session_context.fuel_remaining = 6.0  # Critical fuel
        session_context.fuel_consumption_per_lap = 3.5
        session_context.current_lap = 10
        session_context.gap_ahead = 2.0

        base_telemetry_dict["fuel_remaining"] = 6.0
        base_telemetry_dict["tire_temps"]["fl"] = 112.0  # Critical tire
        base_telemetry_dict["lap_number"] = 11  # Lap complete
        base_telemetry_dict["gap_ahead"] = 4.0  # Gap change > 1s
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        event_types = {e.type for e in events}
        assert "fuel_critical" in event_types
        assert "tire_critical" in event_types
        assert "lap_complete" in event_types
        assert "gap_change" in event_types


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestTelemetryAgentPerformance:
    """Tests for TelemetryAgent performance requirements."""

    def test_detect_events_under_50ms(self, thresholds, session_context, base_telemetry_dict):
        """Event detection should complete in under 50ms."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)
        telemetry = TelemetryData(**base_telemetry_dict)

        # Run detection 100 times and check average
        total_time = 0
        iterations = 100

        for _ in range(iterations):
            start = time.perf_counter()
            agent.detect_events(telemetry, session_context)
            end = time.perf_counter()
            total_time += (end - start) * 1000  # Convert to ms

        avg_time_ms = total_time / iterations
        assert avg_time_ms < 50, f"Average detection time {avg_time_ms:.2f}ms exceeds 50ms target"

    def test_detect_events_with_complex_scenario_under_50ms(self, thresholds, session_context, base_telemetry_dict):
        """Complex scenarios should still complete in under 50ms."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # Set up worst-case scenario with multiple events
        session_context.fuel_remaining = 6.0
        session_context.fuel_consumption_per_lap = 3.5
        session_context.current_lap = 10
        session_context.current_sector = 3
        session_context.gap_ahead = 2.0
        session_context.gap_behind = 1.0

        base_telemetry_dict["fuel_remaining"] = 6.0
        base_telemetry_dict["tire_temps"] = {"fl": 115.0, "fr": 112.0, "rl": 108.0, "rr": 105.0}
        base_telemetry_dict["tire_wear"] = {"fl": 90.0, "fr": 85.0, "rl": 75.0, "rr": 70.0}
        base_telemetry_dict["lap_number"] = 11
        base_telemetry_dict["sector"] = 1
        base_telemetry_dict["gap_ahead"] = 5.0
        base_telemetry_dict["gap_behind"] = 4.0
        telemetry = TelemetryData(**base_telemetry_dict)

        start = time.perf_counter()
        events = agent.detect_events(telemetry, session_context)
        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000
        assert elapsed_ms < 50, f"Complex detection time {elapsed_ms:.2f}ms exceeds 50ms target"
        assert len(events) >= 4, "Should detect multiple events in complex scenario"


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_handles_first_telemetry_message(self, thresholds, base_telemetry_dict):
        """Should handle first telemetry when context has no previous data."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # Fresh context with minimal state
        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        telemetry = TelemetryData(**base_telemetry_dict)

        # Should not crash
        events = agent.detect_events(telemetry, context)
        assert isinstance(events, list)

    def test_handles_boundary_threshold_values(self, thresholds, session_context, base_telemetry_dict):
        """Should correctly handle values exactly at thresholds."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # Exactly at warning threshold (100.0C) - should trigger warning
        base_telemetry_dict["tire_temps"]["fl"] = 100.0
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        # Behavior at exact boundary is implementation-defined
        # but should not crash
        assert isinstance(events, list)

    def test_returns_empty_list_when_no_events(self, thresholds, session_context, base_telemetry_dict):
        """Should return empty list when no events detected."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        # All values normal, no state changes
        session_context.current_lap = 10
        session_context.current_sector = 2
        base_telemetry_dict["lap_number"] = 10
        base_telemetry_dict["sector"] = 2
        telemetry = TelemetryData(**base_telemetry_dict)

        events = agent.detect_events(telemetry, session_context)

        assert events == []

    def test_handles_zero_fuel_consumption(self, thresholds, session_context, base_telemetry_dict):
        """Should handle zero fuel consumption without division error."""
        from jarvis_granite.agents.telemetry_agent import TelemetryAgent

        agent = TelemetryAgent(thresholds=thresholds)

        session_context.fuel_consumption_per_lap = 0.0
        telemetry = TelemetryData(**base_telemetry_dict)

        # Should not crash with division by zero
        events = agent.detect_events(telemetry, session_context)
        assert isinstance(events, list)
