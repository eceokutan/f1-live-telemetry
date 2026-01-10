"""
TDD Tests for Session Context - Phase 1, Section 3

These tests define the expected behavior for:
1. LiveSessionContext creation and defaults
2. Rolling telemetry buffer (60s window at 10Hz = 600 samples)
3. Conversation history tracking (last 3 exchanges)
4. Context updates from telemetry
5. to_prompt_context() method for LLM injection

Run with: pytest tests/test_context.py -v
"""

import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys


def _utcnow() -> datetime:
    """Get current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# LIVE SESSION CONTEXT - CREATION AND DEFAULTS
# =============================================================================

class TestLiveSessionContextCreation:
    """Tests for creating LiveSessionContext."""

    def test_create_context_with_required_fields(self):
        """Should create context with session_id, source, and track_name."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="race_001",
            source="torcs",
            track_name="Monza"
        )

        assert context.session_id == "race_001"
        assert context.source == "torcs"
        assert context.track_name == "Monza"
        assert context.started_at is not None

    def test_context_has_sensible_defaults(self):
        """Should initialize with sensible default values."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test Track"
        )

        # Vehicle state defaults
        assert context.current_lap == 0
        assert context.current_sector == 1
        assert context.speed_kmh == 0.0
        assert context.rpm == 0
        assert context.gear == 0
        assert context.throttle == 0.0
        assert context.brake == 0.0

        # Resource defaults
        assert context.fuel_remaining == 100.0
        assert context.fuel_consumption_per_lap == 0.0

        # Race position defaults
        assert context.position == 1
        assert context.gap_ahead is None
        assert context.gap_behind is None

        # Lap history defaults
        assert context.lap_times == []
        assert context.best_lap is None
        assert context.last_lap is None

    def test_context_started_at_is_set_automatically(self):
        """Should set started_at to current time if not provided."""
        from jarvis_granite.live.context import LiveSessionContext

        before = _utcnow()
        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )
        after = _utcnow()

        assert before <= context.started_at <= after

    def test_context_can_override_started_at(self):
        """Should allow overriding started_at."""
        from jarvis_granite.live.context import LiveSessionContext

        custom_time = datetime(2025, 1, 15, 14, 30, 0)
        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test",
            started_at=custom_time
        )

        assert context.started_at == custom_time


class TestTireStateDefaults:
    """Tests for tire state initialization."""

    def test_tire_wear_defaults_to_zero(self):
        """Tire wear should default to 0% for all corners."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        assert context.tire_wear == {"fl": 0, "fr": 0, "rl": 0, "rr": 0}

    def test_tire_temps_defaults_to_ambient(self):
        """Tire temps should default to ~80C (ambient/warmup)."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        assert context.tire_temps == {"fl": 80, "fr": 80, "rl": 80, "rr": 80}


# =============================================================================
# TELEMETRY BUFFER
# =============================================================================

class TestTelemetryBuffer:
    """Tests for rolling telemetry buffer (60s at 10Hz = 600 samples)."""

    def test_buffer_exists_and_is_empty_initially(self):
        """Should have an empty telemetry buffer on creation."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        assert len(context.telemetry_buffer) == 0

    def test_buffer_max_length_is_600(self):
        """Buffer should hold max 600 samples (60s at 10Hz)."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        # Add 700 items
        for i in range(700):
            context.telemetry_buffer.append({"sample": i})

        # Should only keep last 600
        assert len(context.telemetry_buffer) == 600
        assert context.telemetry_buffer[0]["sample"] == 100  # First 100 dropped
        assert context.telemetry_buffer[-1]["sample"] == 699

    def test_buffer_is_fifo(self):
        """Buffer should be FIFO - oldest samples dropped first."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        # Fill buffer
        for i in range(600):
            context.telemetry_buffer.append({"index": i})

        # Add one more
        context.telemetry_buffer.append({"index": 600})

        # First item should now be index 1 (index 0 dropped)
        assert context.telemetry_buffer[0]["index"] == 1
        assert context.telemetry_buffer[-1]["index"] == 600

    def test_add_telemetry_method(self):
        """Should have method to add telemetry snapshot."""
        from jarvis_granite.live.context import LiveSessionContext
        from jarvis_granite.schemas.telemetry import TelemetryData

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        telemetry = TelemetryData(
            speed_kmh=200.0,
            rpm=10000,
            gear=4,
            throttle=0.8,
            brake=0.0,
            steering_angle=0.1,
            fuel_remaining=45.0,
            tire_temps={"fl": 95, "fr": 96, "rl": 92, "rr": 93},
            tire_wear={"fl": 15, "fr": 16, "rl": 12, "rr": 13},
            g_forces={"lateral": 1.5, "longitudinal": 0.2},
            track_position=0.5,
            lap_number=5,
            lap_time_current=30.0,
            sector=2
        )

        context.add_telemetry(telemetry)

        assert len(context.telemetry_buffer) == 1


# =============================================================================
# CONVERSATION HISTORY
# =============================================================================

class TestConversationHistory:
    """Tests for conversation history tracking (last 3 exchanges)."""

    def test_conversation_history_empty_initially(self):
        """Should have empty conversation history on creation."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        assert len(context.conversation_history) == 0

    def test_conversation_history_max_length_is_3(self):
        """Should keep only last 3 exchanges."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        # Add 5 exchanges
        for i in range(5):
            context.add_exchange(
                query=f"Question {i}",
                response=f"Answer {i}"
            )

        # Should only have last 3
        assert len(context.conversation_history) == 3
        assert context.conversation_history[0]["query"] == "Question 2"
        assert context.conversation_history[-1]["query"] == "Question 4"

    def test_add_exchange_stores_query_and_response(self):
        """add_exchange should store both query and response."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        context.add_exchange(
            query="How are my tires?",
            response="Tires are in good condition, about 15% wear."
        )

        assert len(context.conversation_history) == 1
        assert context.conversation_history[0]["query"] == "How are my tires?"
        assert context.conversation_history[0]["response"] == "Tires are in good condition, about 15% wear."

    def test_add_exchange_includes_timestamp(self):
        """Each exchange should have a timestamp."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        before = _utcnow()
        context.add_exchange(query="Test?", response="Test.")
        after = _utcnow()

        exchange = context.conversation_history[0]
        assert "timestamp" in exchange
        assert before <= exchange["timestamp"] <= after


# =============================================================================
# CONTEXT UPDATE FROM TELEMETRY
# =============================================================================

class TestContextUpdate:
    """Tests for updating context from telemetry data."""

    @pytest.fixture
    def sample_telemetry(self):
        """Create sample telemetry data."""
        from jarvis_granite.schemas.telemetry import TelemetryData

        return TelemetryData(
            speed_kmh=245.5,
            rpm=12500,
            gear=5,
            throttle=0.95,
            brake=0.0,
            steering_angle=-0.12,
            fuel_remaining=45.2,
            tire_temps={"fl": 95.2, "fr": 96.1, "rl": 92.4, "rr": 93.8},
            tire_wear={"fl": 15.2, "fr": 16.1, "rl": 12.4, "rr": 13.8},
            g_forces={"lateral": 1.8, "longitudinal": 0.3},
            track_position=0.342,
            lap_number=12,
            lap_time_current=45.234,
            sector=2,
            position=3,
            gap_ahead=2.456,
            gap_behind=1.234
        )

    def test_update_updates_vehicle_state(self, sample_telemetry):
        """update() should update vehicle state fields."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        context.update(sample_telemetry)

        assert context.speed_kmh == 245.5
        assert context.rpm == 12500
        assert context.gear == 5
        assert context.throttle == 0.95
        assert context.brake == 0.0

    def test_update_updates_resource_state(self, sample_telemetry):
        """update() should update resource fields."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        context.update(sample_telemetry)

        assert context.fuel_remaining == 45.2
        assert context.tire_temps == {"fl": 95.2, "fr": 96.1, "rl": 92.4, "rr": 93.8}
        assert context.tire_wear == {"fl": 15.2, "fr": 16.1, "rl": 12.4, "rr": 13.8}

    def test_update_updates_race_position(self, sample_telemetry):
        """update() should update race position fields."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        context.update(sample_telemetry)

        assert context.position == 3
        assert context.gap_ahead == 2.456
        assert context.gap_behind == 1.234

    def test_update_updates_lap_and_sector(self, sample_telemetry):
        """update() should update lap and sector."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        context.update(sample_telemetry)

        assert context.current_lap == 12
        assert context.current_sector == 2

    def test_update_adds_to_telemetry_buffer(self, sample_telemetry):
        """update() should add telemetry to buffer."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        context.update(sample_telemetry)

        assert len(context.telemetry_buffer) == 1

    def test_update_handles_optional_position_fields(self):
        """update() should handle missing position/gap fields."""
        from jarvis_granite.live.context import LiveSessionContext
        from jarvis_granite.schemas.telemetry import TelemetryData

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        # Set initial values
        context.position = 5
        context.gap_ahead = 3.0

        # Telemetry without position data
        telemetry = TelemetryData(
            speed_kmh=200.0,
            rpm=10000,
            gear=4,
            throttle=0.8,
            brake=0.0,
            steering_angle=0.0,
            fuel_remaining=50.0,
            tire_temps={"fl": 90, "fr": 90, "rl": 90, "rr": 90},
            tire_wear={"fl": 10, "fr": 10, "rl": 10, "rr": 10},
            g_forces={"lateral": 0, "longitudinal": 0},
            track_position=0.5,
            lap_number=5,
            lap_time_current=30.0,
            sector=1
            # position, gap_ahead, gap_behind are None
        )

        context.update(telemetry)

        # Should keep previous values or set to None
        assert context.gap_ahead is None
        assert context.gap_behind is None


# =============================================================================
# LAP COMPLETION TRACKING
# =============================================================================

class TestLapTracking:
    """Tests for lap time tracking."""

    def test_record_lap_time(self):
        """Should record lap times."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        context.record_lap_time(82.456)

        assert len(context.lap_times) == 1
        assert context.lap_times[0] == 82.456
        assert context.last_lap == 82.456

    def test_best_lap_updated_on_improvement(self):
        """Should update best_lap when a faster lap is set."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        context.record_lap_time(85.0)
        assert context.best_lap == 85.0

        context.record_lap_time(83.0)  # Faster
        assert context.best_lap == 83.0

        context.record_lap_time(84.0)  # Slower
        assert context.best_lap == 83.0  # Unchanged

    def test_fuel_consumption_calculated(self):
        """Should calculate fuel consumption per lap."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        # Start of lap
        context.fuel_remaining = 50.0
        context._lap_start_fuel = 50.0

        # End of lap (used 3.5L)
        context.fuel_remaining = 46.5
        context.record_lap_time(82.0)

        assert context.fuel_consumption_per_lap == 3.5

    def test_get_fuel_laps_remaining(self):
        """Should calculate laps remaining based on fuel."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        context.fuel_remaining = 35.0
        context.fuel_consumption_per_lap = 3.5

        assert context.get_fuel_laps_remaining() == 10.0

    def test_get_fuel_laps_remaining_handles_zero_consumption(self):
        """Should handle zero fuel consumption gracefully."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        context.fuel_remaining = 50.0
        context.fuel_consumption_per_lap = 0.0

        # Should return infinity or a large number, not crash
        result = context.get_fuel_laps_remaining()
        assert result == float('inf') or result > 1000


# =============================================================================
# ACTIVE ALERTS
# =============================================================================

class TestActiveAlerts:
    """Tests for active alert tracking."""

    def test_active_alerts_empty_initially(self):
        """Should have no active alerts on creation."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        assert context.active_alerts == []

    def test_add_alert(self):
        """Should be able to add active alerts."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        context.add_alert("fuel_warning", {"laps": 5})

        assert len(context.active_alerts) == 1
        assert context.active_alerts[0]["type"] == "fuel_warning"

    def test_clear_alert(self):
        """Should be able to clear alerts by type."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        context.add_alert("fuel_warning", {"laps": 5})
        context.add_alert("tire_warning", {"temp": 105})

        context.clear_alert("fuel_warning")

        assert len(context.active_alerts) == 1
        assert context.active_alerts[0]["type"] == "tire_warning"


# =============================================================================
# TO_PROMPT_CONTEXT METHOD
# =============================================================================

class TestToPromptContext:
    """Tests for to_prompt_context() method used in LLM prompts."""

    def test_to_prompt_context_returns_string(self):
        """Should return a formatted string."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Monza"
        )

        result = context.to_prompt_context()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_to_prompt_context_includes_track_name(self):
        """Should include track name."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Silverstone"
        )

        result = context.to_prompt_context()

        assert "Silverstone" in result

    def test_to_prompt_context_includes_lap_info(self):
        """Should include current lap and position."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )
        context.current_lap = 15
        context.position = 3

        result = context.to_prompt_context()

        assert "15" in result  # Lap number
        assert "P3" in result or "3" in result  # Position

    def test_to_prompt_context_includes_fuel_info(self):
        """Should include fuel remaining and laps estimate."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )
        context.fuel_remaining = 45.2
        context.fuel_consumption_per_lap = 3.5

        result = context.to_prompt_context()

        assert "45" in result  # Fuel amount
        # Should include laps remaining estimate

    def test_to_prompt_context_includes_tire_temps(self):
        """Should include tire temperatures."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )
        context.tire_temps = {"fl": 95, "fr": 96, "rl": 92, "rr": 93}

        result = context.to_prompt_context()

        assert "FL" in result or "fl" in result.lower()
        assert "95" in result

    def test_to_prompt_context_includes_gap_info(self):
        """Should include gap ahead and behind."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )
        context.gap_ahead = 2.5
        context.gap_behind = 1.2

        result = context.to_prompt_context()

        assert "2.5" in result or "2.50" in result
        assert "1.2" in result or "1.20" in result

    def test_to_prompt_context_includes_best_and_last_lap(self):
        """Should include best and last lap times."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )
        context.best_lap = 81.234
        context.last_lap = 82.456

        result = context.to_prompt_context()

        # Lap times might be formatted as MM:SS.mmm
        assert "81" in result or "1:21" in result
        assert "82" in result or "1:22" in result

    def test_to_prompt_context_handles_none_values(self):
        """Should handle None values gracefully."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )
        # gap_ahead, gap_behind, best_lap, last_lap are all None

        result = context.to_prompt_context()

        # Should not crash, should show N/A or similar
        assert isinstance(result, str)
        assert "N/A" in result or "-" in result or "None" not in result


# =============================================================================
# PROACTIVE MESSAGE TIMING
# =============================================================================

class TestProactiveMessageTiming:
    """Tests for proactive message interval tracking."""

    def test_last_proactive_message_time_is_none_initially(self):
        """Should have no last proactive message time initially."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        assert context.last_proactive_message_time is None

    def test_can_send_proactive_when_none_sent(self):
        """Should allow proactive message when none have been sent."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        assert context.can_send_proactive(min_interval_seconds=10.0) is True

    def test_cannot_send_proactive_too_soon(self):
        """Should block proactive message if sent too recently."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        context.mark_proactive_sent()

        # Immediately after, should not allow
        assert context.can_send_proactive(min_interval_seconds=10.0) is False

    def test_can_send_proactive_after_interval(self):
        """Should allow proactive message after interval has passed."""
        from jarvis_granite.live.context import LiveSessionContext

        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test"
        )

        # Set last message time to 15 seconds ago
        context.last_proactive_message_time = _utcnow() - timedelta(seconds=15)

        assert context.can_send_proactive(min_interval_seconds=10.0) is True


# =============================================================================
# SESSION DURATION
# =============================================================================

class TestSessionDuration:
    """Tests for session duration tracking."""

    def test_get_session_duration(self):
        """Should calculate session duration."""
        from jarvis_granite.live.context import LiveSessionContext

        start_time = _utcnow() - timedelta(minutes=30)
        context = LiveSessionContext(
            session_id="test",
            source="torcs",
            track_name="Test",
            started_at=start_time
        )

        duration = context.get_session_duration()

        assert duration.total_seconds() >= 30 * 60 - 1  # Allow 1 second tolerance
        assert duration.total_seconds() <= 30 * 60 + 1
