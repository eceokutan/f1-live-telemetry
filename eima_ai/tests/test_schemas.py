"""
TDD Tests for Data Schemas - Phase 1, Section 2

These tests define the expected behavior for:
1. Telemetry data schemas (TelemetryData, TireTemps, TireWear, GForces)
2. WebSocket message schemas (session_init, telemetry, ai_response, etc.)
3. Event schemas (Event, Priority enum)

Run with: pytest tests/test_schemas.py -v

Write these tests FIRST, watch them fail, then implement schemas to pass.
"""

import pytest
from datetime import datetime
from typing import Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# TELEMETRY DATA SCHEMAS
# =============================================================================

class TestTireTemps:
    """Tests for TireTemps schema - tire temperatures in Celsius."""

    def test_create_tire_temps_with_valid_data(self):
        """Should create TireTemps with all four corners."""
        from jarvis_granite.schemas.telemetry import TireTemps

        temps = TireTemps(fl=95.2, fr=96.1, rl=92.4, rr=93.8)

        assert temps.fl == 95.2
        assert temps.fr == 96.1
        assert temps.rl == 92.4
        assert temps.rr == 93.8

    def test_tire_temps_requires_all_corners(self):
        """Should require all four tire positions."""
        from jarvis_granite.schemas.telemetry import TireTemps
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TireTemps(fl=95.0, fr=96.0)  # Missing rl, rr

    def test_tire_temps_must_be_non_negative(self):
        """Tire temperatures cannot be negative."""
        from jarvis_granite.schemas.telemetry import TireTemps
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TireTemps(fl=-10.0, fr=96.0, rl=92.0, rr=93.0)

    def test_tire_temps_to_dict(self):
        """Should convert to dictionary."""
        from jarvis_granite.schemas.telemetry import TireTemps

        temps = TireTemps(fl=95.0, fr=96.0, rl=92.0, rr=93.0)
        data = temps.model_dump()

        assert data == {"fl": 95.0, "fr": 96.0, "rl": 92.0, "rr": 93.0}


class TestTireWear:
    """Tests for TireWear schema - tire wear percentages (0-100)."""

    def test_create_tire_wear_with_valid_data(self):
        """Should create TireWear with all four corners."""
        from jarvis_granite.schemas.telemetry import TireWear

        wear = TireWear(fl=15.2, fr=16.1, rl=12.4, rr=13.8)

        assert wear.fl == 15.2
        assert wear.fr == 16.1
        assert wear.rl == 12.4
        assert wear.rr == 13.8

    def test_tire_wear_must_be_percentage(self):
        """Tire wear must be between 0 and 100."""
        from jarvis_granite.schemas.telemetry import TireWear
        from pydantic import ValidationError

        # Over 100
        with pytest.raises(ValidationError):
            TireWear(fl=105.0, fr=16.0, rl=12.0, rr=13.0)

        # Negative
        with pytest.raises(ValidationError):
            TireWear(fl=-5.0, fr=16.0, rl=12.0, rr=13.0)

    def test_tire_wear_edge_values(self):
        """Should accept edge values 0 and 100."""
        from jarvis_granite.schemas.telemetry import TireWear

        wear = TireWear(fl=0.0, fr=100.0, rl=50.0, rr=50.0)

        assert wear.fl == 0.0
        assert wear.fr == 100.0


class TestGForces:
    """Tests for GForces schema - lateral and longitudinal G-forces."""

    def test_create_g_forces(self):
        """Should create GForces with lateral and longitudinal."""
        from jarvis_granite.schemas.telemetry import GForces

        g = GForces(lateral=1.8, longitudinal=0.3)

        assert g.lateral == 1.8
        assert g.longitudinal == 0.3

    def test_g_forces_can_be_negative(self):
        """G-forces can be negative (deceleration, turning)."""
        from jarvis_granite.schemas.telemetry import GForces

        g = GForces(lateral=-2.5, longitudinal=-1.2)

        assert g.lateral == -2.5
        assert g.longitudinal == -1.2


class TestTelemetryData:
    """Tests for the main TelemetryData schema."""

    @pytest.fixture
    def valid_telemetry_dict(self):
        """Valid telemetry data as dictionary."""
        return {
            "speed_kmh": 245.5,
            "rpm": 12500,
            "gear": 5,
            "throttle": 0.95,
            "brake": 0.0,
            "steering_angle": -0.12,
            "fuel_remaining": 45.2,
            "tire_temps": {"fl": 95.2, "fr": 96.1, "rl": 92.4, "rr": 93.8},
            "tire_wear": {"fl": 15.2, "fr": 16.1, "rl": 12.4, "rr": 13.8},
            "g_forces": {"lateral": 1.8, "longitudinal": 0.3},
            "track_position": 0.342,
            "lap_number": 12,
            "lap_time_current": 45.234,
            "sector": 2,
            "position": 3,
            "gap_ahead": 2.456,
            "gap_behind": 1.234
        }

    def test_create_telemetry_data_from_dict(self, valid_telemetry_dict):
        """Should create TelemetryData from valid dictionary."""
        from jarvis_granite.schemas.telemetry import TelemetryData

        data = TelemetryData(**valid_telemetry_dict)

        assert data.speed_kmh == 245.5
        assert data.rpm == 12500
        assert data.gear == 5
        assert data.throttle == 0.95
        assert data.tire_temps.fl == 95.2
        assert data.tire_wear.rl == 12.4
        assert data.g_forces.lateral == 1.8

    def test_speed_must_be_non_negative(self):
        """Speed cannot be negative."""
        from jarvis_granite.schemas.telemetry import TelemetryData, TireTemps, TireWear, GForces
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TelemetryData(
                speed_kmh=-10.0,
                rpm=5000,
                gear=3,
                throttle=0.5,
                brake=0.0,
                steering_angle=0.0,
                fuel_remaining=50.0,
                tire_temps=TireTemps(fl=80, fr=80, rl=80, rr=80),
                tire_wear=TireWear(fl=10, fr=10, rl=10, rr=10),
                g_forces=GForces(lateral=0, longitudinal=0),
                track_position=0.5,
                lap_number=1,
                lap_time_current=0.0,
                sector=1
            )

    def test_throttle_must_be_0_to_1(self):
        """Throttle must be between 0 and 1."""
        from jarvis_granite.schemas.telemetry import TelemetryData
        from pydantic import ValidationError

        base_data = {
            "speed_kmh": 100.0,
            "rpm": 5000,
            "gear": 3,
            "brake": 0.0,
            "steering_angle": 0.0,
            "fuel_remaining": 50.0,
            "tire_temps": {"fl": 80, "fr": 80, "rl": 80, "rr": 80},
            "tire_wear": {"fl": 10, "fr": 10, "rl": 10, "rr": 10},
            "g_forces": {"lateral": 0, "longitudinal": 0},
            "track_position": 0.5,
            "lap_number": 1,
            "lap_time_current": 0.0,
            "sector": 1
        }

        # Over 1
        with pytest.raises(ValidationError):
            TelemetryData(**{**base_data, "throttle": 1.5})

        # Negative
        with pytest.raises(ValidationError):
            TelemetryData(**{**base_data, "throttle": -0.1})

    def test_brake_must_be_0_to_1(self):
        """Brake must be between 0 and 1."""
        from jarvis_granite.schemas.telemetry import TelemetryData
        from pydantic import ValidationError

        base_data = {
            "speed_kmh": 100.0,
            "rpm": 5000,
            "gear": 3,
            "throttle": 0.5,
            "steering_angle": 0.0,
            "fuel_remaining": 50.0,
            "tire_temps": {"fl": 80, "fr": 80, "rl": 80, "rr": 80},
            "tire_wear": {"fl": 10, "fr": 10, "rl": 10, "rr": 10},
            "g_forces": {"lateral": 0, "longitudinal": 0},
            "track_position": 0.5,
            "lap_number": 1,
            "lap_time_current": 0.0,
            "sector": 1
        }

        with pytest.raises(ValidationError):
            TelemetryData(**{**base_data, "brake": 1.5})

    def test_steering_angle_must_be_minus1_to_1(self):
        """Steering angle must be between -1 and 1."""
        from jarvis_granite.schemas.telemetry import TelemetryData
        from pydantic import ValidationError

        base_data = {
            "speed_kmh": 100.0,
            "rpm": 5000,
            "gear": 3,
            "throttle": 0.5,
            "brake": 0.0,
            "fuel_remaining": 50.0,
            "tire_temps": {"fl": 80, "fr": 80, "rl": 80, "rr": 80},
            "tire_wear": {"fl": 10, "fr": 10, "rl": 10, "rr": 10},
            "g_forces": {"lateral": 0, "longitudinal": 0},
            "track_position": 0.5,
            "lap_number": 1,
            "lap_time_current": 0.0,
            "sector": 1
        }

        with pytest.raises(ValidationError):
            TelemetryData(**{**base_data, "steering_angle": 1.5})

        with pytest.raises(ValidationError):
            TelemetryData(**{**base_data, "steering_angle": -1.5})

    def test_gear_must_be_0_to_8(self):
        """Gear must be between 0 (neutral) and 8."""
        from jarvis_granite.schemas.telemetry import TelemetryData
        from pydantic import ValidationError

        base_data = {
            "speed_kmh": 100.0,
            "rpm": 5000,
            "throttle": 0.5,
            "brake": 0.0,
            "steering_angle": 0.0,
            "fuel_remaining": 50.0,
            "tire_temps": {"fl": 80, "fr": 80, "rl": 80, "rr": 80},
            "tire_wear": {"fl": 10, "fr": 10, "rl": 10, "rr": 10},
            "g_forces": {"lateral": 0, "longitudinal": 0},
            "track_position": 0.5,
            "lap_number": 1,
            "lap_time_current": 0.0,
            "sector": 1
        }

        # Negative gear
        with pytest.raises(ValidationError):
            TelemetryData(**{**base_data, "gear": -1})

        # Gear too high
        with pytest.raises(ValidationError):
            TelemetryData(**{**base_data, "gear": 9})

    def test_track_position_must_be_0_to_1(self):
        """Track position is a percentage (0-1) around the lap."""
        from jarvis_granite.schemas.telemetry import TelemetryData
        from pydantic import ValidationError

        base_data = {
            "speed_kmh": 100.0,
            "rpm": 5000,
            "gear": 3,
            "throttle": 0.5,
            "brake": 0.0,
            "steering_angle": 0.0,
            "fuel_remaining": 50.0,
            "tire_temps": {"fl": 80, "fr": 80, "rl": 80, "rr": 80},
            "tire_wear": {"fl": 10, "fr": 10, "rl": 10, "rr": 10},
            "g_forces": {"lateral": 0, "longitudinal": 0},
            "lap_number": 1,
            "lap_time_current": 0.0,
            "sector": 1
        }

        with pytest.raises(ValidationError):
            TelemetryData(**{**base_data, "track_position": 1.5})

    def test_sector_must_be_1_to_3(self):
        """Sector must be 1, 2, or 3."""
        from jarvis_granite.schemas.telemetry import TelemetryData
        from pydantic import ValidationError

        base_data = {
            "speed_kmh": 100.0,
            "rpm": 5000,
            "gear": 3,
            "throttle": 0.5,
            "brake": 0.0,
            "steering_angle": 0.0,
            "fuel_remaining": 50.0,
            "tire_temps": {"fl": 80, "fr": 80, "rl": 80, "rr": 80},
            "tire_wear": {"fl": 10, "fr": 10, "rl": 10, "rr": 10},
            "g_forces": {"lateral": 0, "longitudinal": 0},
            "track_position": 0.5,
            "lap_number": 1,
            "lap_time_current": 0.0,
        }

        with pytest.raises(ValidationError):
            TelemetryData(**{**base_data, "sector": 0})

        with pytest.raises(ValidationError):
            TelemetryData(**{**base_data, "sector": 4})

    def test_optional_fields_can_be_none(self, valid_telemetry_dict):
        """Position, gap_ahead, gap_behind are optional."""
        from jarvis_granite.schemas.telemetry import TelemetryData

        del valid_telemetry_dict["position"]
        del valid_telemetry_dict["gap_ahead"]
        del valid_telemetry_dict["gap_behind"]

        data = TelemetryData(**valid_telemetry_dict)

        assert data.position is None
        assert data.gap_ahead is None
        assert data.gap_behind is None

    def test_telemetry_json_serialization(self, valid_telemetry_dict):
        """Should serialize to JSON and back."""
        from jarvis_granite.schemas.telemetry import TelemetryData

        data = TelemetryData(**valid_telemetry_dict)
        json_str = data.model_dump_json()
        restored = TelemetryData.model_validate_json(json_str)

        assert restored.speed_kmh == data.speed_kmh
        assert restored.tire_temps.fl == data.tire_temps.fl


# =============================================================================
# WEBSOCKET MESSAGE SCHEMAS
# =============================================================================

class TestSessionInitMessage:
    """Tests for session initialization message."""

    def test_create_session_init(self):
        """Should create valid session init message."""
        from jarvis_granite.schemas.messages import SessionInitMessage

        msg = SessionInitMessage(
            session_id="race_001",
            source="torcs",
            track_name="Monza",
            config={"verbosity": "moderate", "driver_name": "Driver 1"}
        )

        assert msg.type == "session_init"
        assert msg.session_id == "race_001"
        assert msg.source == "torcs"
        assert msg.track_name == "Monza"
        assert msg.config["verbosity"] == "moderate"

    def test_session_init_type_is_fixed(self):
        """Message type should always be 'session_init'."""
        from jarvis_granite.schemas.messages import SessionInitMessage

        msg = SessionInitMessage(
            session_id="test",
            source="torcs",
            track_name="Test Track"
        )

        assert msg.type == "session_init"

    def test_source_must_be_valid(self):
        """Source must be one of the allowed values."""
        from jarvis_granite.schemas.messages import SessionInitMessage
        from pydantic import ValidationError

        # Valid sources
        for source in ["torcs", "assetto_corsa", "can_bus"]:
            msg = SessionInitMessage(
                session_id="test",
                source=source,
                track_name="Test"
            )
            assert msg.source == source

        # Invalid source
        with pytest.raises(ValidationError):
            SessionInitMessage(
                session_id="test",
                source="invalid_source",
                track_name="Test"
            )


class TestTelemetryMessage:
    """Tests for telemetry WebSocket message."""

    def test_create_telemetry_message(self):
        """Should create telemetry message with timestamp and data."""
        from jarvis_granite.schemas.messages import TelemetryMessage
        from jarvis_granite.schemas.telemetry import TelemetryData

        msg = TelemetryMessage(
            timestamp="2026-01-15T14:32:05.123Z",
            data={
                "speed_kmh": 200.0,
                "rpm": 10000,
                "gear": 4,
                "throttle": 0.8,
                "brake": 0.0,
                "steering_angle": 0.1,
                "fuel_remaining": 40.0,
                "tire_temps": {"fl": 90, "fr": 90, "rl": 90, "rr": 90},
                "tire_wear": {"fl": 20, "fr": 20, "rl": 20, "rr": 20},
                "g_forces": {"lateral": 1.0, "longitudinal": 0.2},
                "track_position": 0.5,
                "lap_number": 5,
                "lap_time_current": 30.0,
                "sector": 2
            }
        )

        assert msg.type == "telemetry"
        assert msg.data.speed_kmh == 200.0
        assert msg.data.tire_temps.fl == 90


class TestTextQueryMessage:
    """Tests for text query message (alternative to voice)."""

    def test_create_text_query(self):
        """Should create text query message."""
        from jarvis_granite.schemas.messages import TextQueryMessage

        msg = TextQueryMessage(
            timestamp="2026-01-15T14:32:10.456Z",
            query="How are my tires looking?"
        )

        assert msg.type == "text_query"
        assert msg.query == "How are my tires looking?"

    def test_query_cannot_be_empty(self):
        """Query text cannot be empty."""
        from jarvis_granite.schemas.messages import TextQueryMessage
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TextQueryMessage(
                timestamp="2026-01-15T14:32:10.456Z",
                query=""
            )


class TestConfigUpdateMessage:
    """Tests for runtime configuration update message."""

    def test_create_config_update(self):
        """Should create config update message."""
        from jarvis_granite.schemas.messages import ConfigUpdateMessage

        msg = ConfigUpdateMessage(
            config={"verbosity": "verbose"}
        )

        assert msg.type == "config_update"
        assert msg.config["verbosity"] == "verbose"


class TestSessionEndMessage:
    """Tests for session end message."""

    def test_create_session_end(self):
        """Should create session end message."""
        from jarvis_granite.schemas.messages import SessionEndMessage

        msg = SessionEndMessage()

        assert msg.type == "session_end"


class TestSessionConfirmedMessage:
    """Tests for session confirmed response from server."""

    def test_create_session_confirmed(self):
        """Should create session confirmed with LiveKit details."""
        from jarvis_granite.schemas.messages import SessionConfirmedMessage, LiveKitDetails

        msg = SessionConfirmedMessage(
            session_id="race_001",
            config={"verbosity": "moderate"},
            livekit=LiveKitDetails(
                url="wss://livekit.example.com",
                token="eyJ...",
                room_name="race_001_voice"
            )
        )

        assert msg.type == "session_confirmed"
        assert msg.session_id == "race_001"
        assert msg.livekit.url == "wss://livekit.example.com"
        assert msg.livekit.room_name == "race_001_voice"


class TestAIResponseMessage:
    """Tests for AI response message."""

    def test_create_ai_response(self):
        """Should create AI response with all fields."""
        from jarvis_granite.schemas.messages import AIResponseMessage, ResponseMetadata

        msg = AIResponseMessage(
            response_id="resp_abc123",
            timestamp="2026-01-15T14:32:11.789Z",
            trigger="pit_window_open",
            text="Pit window is open. Box this lap for fresh mediums.",
            priority="high",
            metadata=ResponseMetadata(latency_ms=1850, tokens_used=45)
        )

        assert msg.type == "ai_response"
        assert msg.response_id == "resp_abc123"
        assert msg.trigger == "pit_window_open"
        assert msg.priority == "high"
        assert msg.metadata.latency_ms == 1850

    def test_priority_must_be_valid(self):
        """Priority must be critical, high, medium, or low."""
        from jarvis_granite.schemas.messages import AIResponseMessage
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AIResponseMessage(
                response_id="test",
                timestamp="2026-01-15T14:32:11.789Z",
                trigger="test",
                text="Test",
                priority="invalid"
            )


class TestErrorMessage:
    """Tests for error message."""

    def test_create_error_message(self):
        """Should create error message."""
        from jarvis_granite.schemas.messages import ErrorMessage

        msg = ErrorMessage(
            error_code="LLM_TIMEOUT",
            message="AI response timed out, please try again",
            timestamp="2026-01-15T14:32:15.000Z"
        )

        assert msg.type == "error"
        assert msg.error_code == "LLM_TIMEOUT"
        assert "timed out" in msg.message

    def test_error_code_must_be_valid(self):
        """Error code must be from defined set."""
        from jarvis_granite.schemas.messages import ErrorMessage
        from pydantic import ValidationError

        valid_codes = [
            "LLM_TIMEOUT", "LLM_ERROR", "TTS_ERROR", "STT_ERROR",
            "LIVEKIT_ERROR", "VALIDATION_ERROR", "SESSION_NOT_FOUND", "CONFIG_ERROR"
        ]

        for code in valid_codes:
            msg = ErrorMessage(
                error_code=code,
                message="Test error",
                timestamp="2026-01-15T14:32:15.000Z"
            )
            assert msg.error_code == code


class TestHeartbeatMessage:
    """Tests for heartbeat message."""

    def test_create_heartbeat(self):
        """Should create heartbeat message."""
        from jarvis_granite.schemas.messages import HeartbeatMessage

        msg = HeartbeatMessage(
            timestamp="2026-01-15T14:32:30.000Z",
            session_active=True
        )

        assert msg.type == "heartbeat"
        assert msg.session_active is True


class TestMessageParsing:
    """Tests for parsing incoming WebSocket messages."""

    def test_parse_session_init(self):
        """Should parse session_init from JSON."""
        from jarvis_granite.schemas.messages import parse_client_message

        json_data = {
            "type": "session_init",
            "session_id": "race_001",
            "source": "torcs",
            "track_name": "Monza"
        }

        msg = parse_client_message(json_data)

        assert msg.type == "session_init"
        assert msg.session_id == "race_001"

    def test_parse_telemetry(self):
        """Should parse telemetry message from JSON."""
        from jarvis_granite.schemas.messages import parse_client_message

        json_data = {
            "type": "telemetry",
            "timestamp": "2026-01-15T14:32:05.123Z",
            "data": {
                "speed_kmh": 200.0,
                "rpm": 10000,
                "gear": 4,
                "throttle": 0.8,
                "brake": 0.0,
                "steering_angle": 0.1,
                "fuel_remaining": 40.0,
                "tire_temps": {"fl": 90, "fr": 90, "rl": 90, "rr": 90},
                "tire_wear": {"fl": 20, "fr": 20, "rl": 20, "rr": 20},
                "g_forces": {"lateral": 1.0, "longitudinal": 0.2},
                "track_position": 0.5,
                "lap_number": 5,
                "lap_time_current": 30.0,
                "sector": 2
            }
        }

        msg = parse_client_message(json_data)

        assert msg.type == "telemetry"
        assert msg.data.speed_kmh == 200.0

    def test_parse_unknown_type_raises_error(self):
        """Should raise error for unknown message type."""
        from jarvis_granite.schemas.messages import parse_client_message

        json_data = {"type": "unknown_type"}

        with pytest.raises(ValueError, match="Unknown message type"):
            parse_client_message(json_data)


# =============================================================================
# EVENT SCHEMAS
# =============================================================================

class TestPriorityEnum:
    """Tests for Priority enumeration."""

    def test_priority_values(self):
        """Priority enum should have correct integer values."""
        from jarvis_granite.schemas.events import Priority

        assert Priority.CRITICAL == 0
        assert Priority.HIGH == 1
        assert Priority.MEDIUM == 2
        assert Priority.LOW == 3

    def test_priority_ordering(self):
        """Lower value = higher priority (for heapq)."""
        from jarvis_granite.schemas.events import Priority

        assert Priority.CRITICAL < Priority.HIGH
        assert Priority.HIGH < Priority.MEDIUM
        assert Priority.MEDIUM < Priority.LOW

    def test_priority_comparison(self):
        """Priorities should be comparable."""
        from jarvis_granite.schemas.events import Priority

        assert Priority.CRITICAL < Priority.HIGH
        assert Priority.HIGH <= Priority.HIGH
        assert Priority.LOW > Priority.MEDIUM


class TestEvent:
    """Tests for Event dataclass/model."""

    def test_create_event(self):
        """Should create event with all fields."""
        from jarvis_granite.schemas.events import Event, Priority

        event = Event(
            type="fuel_critical",
            priority=Priority.CRITICAL,
            data={"laps": 1.5},
            timestamp=1705329125.123
        )

        assert event.type == "fuel_critical"
        assert event.priority == Priority.CRITICAL
        assert event.data["laps"] == 1.5
        assert event.timestamp == 1705329125.123

    def test_event_comparison_by_priority(self):
        """Events should be sortable by priority for heapq."""
        from jarvis_granite.schemas.events import Event, Priority

        critical = Event(
            type="fuel_critical",
            priority=Priority.CRITICAL,
            data={},
            timestamp=1.0
        )
        medium = Event(
            type="gap_change",
            priority=Priority.MEDIUM,
            data={},
            timestamp=2.0
        )

        # For heapq: (priority, timestamp, event)
        assert (critical.priority, critical.timestamp) < (medium.priority, medium.timestamp)

    def test_event_types_are_strings(self):
        """Event types should be descriptive strings."""
        from jarvis_granite.schemas.events import Event, Priority

        valid_types = [
            "pit_window_open",
            "tire_critical",
            "fuel_critical",
            "fuel_warning",
            "gap_change",
            "lap_complete",
            "sector_complete",
            "tire_warning"
        ]

        for event_type in valid_types:
            event = Event(
                type=event_type,
                priority=Priority.MEDIUM,
                data={},
                timestamp=1.0
            )
            assert event.type == event_type


class TestEventFactory:
    """Tests for event factory/creation helpers."""

    def test_create_fuel_critical_event(self):
        """Should create fuel critical event with proper defaults."""
        from jarvis_granite.schemas.events import create_fuel_critical_event, Priority

        event = create_fuel_critical_event(laps_remaining=1.5)

        assert event.type == "fuel_critical"
        assert event.priority == Priority.CRITICAL
        assert event.data["laps"] == 1.5
        assert event.timestamp > 0

    def test_create_tire_warning_event(self):
        """Should create tire warning event."""
        from jarvis_granite.schemas.events import create_tire_warning_event, Priority

        event = create_tire_warning_event(temp=105.0, position="fl")

        assert event.type == "tire_warning"
        assert event.priority == Priority.MEDIUM
        assert event.data["temp"] == 105.0
        assert event.data["position"] == "fl"

    def test_create_gap_change_event(self):
        """Should create gap change event."""
        from jarvis_granite.schemas.events import create_gap_change_event, Priority

        event = create_gap_change_event(gap_change=1.5, direction="ahead")

        assert event.type == "gap_change"
        assert event.priority == Priority.MEDIUM
        assert event.data["change"] == 1.5

    def test_create_lap_complete_event(self):
        """Should create lap complete event."""
        from jarvis_granite.schemas.events import create_lap_complete_event, Priority

        event = create_lap_complete_event(
            lap_number=10,
            lap_time=82.456,
            best_lap=81.234
        )

        assert event.type == "lap_complete"
        assert event.priority == Priority.MEDIUM
        assert event.data["lap"] == 10
        assert event.data["time"] == 82.456
        assert event.data["best"] == 81.234
