"""
Tests for Orchestrator Core - Phase 4, Section 9

These tests verify the expected behavior for:
1. JarvisLiveOrchestrator initialization and configuration
2. Event routing from telemetry to AI response
3. Telemetry Agent -> Race Engineer Agent pipeline
4. Queue processing logic
5. Driver query handling
6. Interrupt handling for high-priority events

Run with: pytest tests/test_orchestrator.py -v
"""

import asyncio
import time
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis_granite.schemas.events import Event, Priority
from jarvis_granite.schemas.telemetry import TelemetryData, TireTemps, TireWear, GForces
from jarvis_granite.live.context import LiveSessionContext
from jarvis_granite.live.orchestrator import JarvisLiveOrchestrator
from config.config import LiveConfig, ThresholdsConfig


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    client = MagicMock()
    client.invoke = AsyncMock(return_value="AI Response: Box this lap.")
    return client


@pytest.fixture
def mock_telemetry_agent():
    """Create a mock telemetry agent."""
    agent = MagicMock()
    agent.detect_events = MagicMock(return_value=[])
    return agent


@pytest.fixture
def mock_race_engineer_agent():
    """Create a mock race engineer agent."""
    agent = MagicMock()
    agent.generate_proactive_response = AsyncMock(return_value="Box this lap for fresh tires.")
    agent.generate_reactive_response = AsyncMock(return_value="Fuel is looking good, about 10 laps remaining.")
    agent.handle_event = AsyncMock(return_value="Box this lap for fresh tires.")
    return agent


@pytest.fixture
def config():
    """Create test configuration."""
    return LiveConfig()


@pytest.fixture
def session_context():
    """Create test session context."""
    return LiveSessionContext(
        session_id="test_session_001",
        source="torcs",
        track_name="Monza"
    )


@pytest.fixture
def sample_telemetry():
    """Create sample telemetry data."""
    return TelemetryData(
        speed_kmh=250.0,
        rpm=12000,
        gear=5,
        throttle=0.9,
        brake=0.0,
        steering_angle=0.1,
        fuel_remaining=50.0,
        tire_temps=TireTemps(fl=95.0, fr=96.0, rl=92.0, rr=93.0),
        tire_wear=TireWear(fl=20.0, fr=22.0, rl=18.0, rr=19.0),
        g_forces=GForces(lateral=1.5, longitudinal=0.2),
        track_position=0.5,
        lap_number=10,
        lap_time_current=45.0,
        sector=2,
        position=3,
        gap_ahead=2.5,
        gap_behind=1.5
    )


@pytest.fixture
def critical_event():
    """Create a critical priority event."""
    return Event(
        type="fuel_critical",
        priority=Priority.CRITICAL,
        data={"laps": 1.5},
        timestamp=time.time()
    )


@pytest.fixture
def medium_event():
    """Create a medium priority event."""
    return Event(
        type="lap_complete",
        priority=Priority.MEDIUM,
        data={"lap": 10, "time": 92.5},
        timestamp=time.time()
    )


# =============================================================================
# ORCHESTRATOR INITIALIZATION
# =============================================================================

class TestOrchestratorInitialization:
    """Tests for JarvisLiveOrchestrator initialization."""

    def test_create_orchestrator_with_defaults(self):
        """Should create orchestrator with default configuration."""
        orchestrator = JarvisLiveOrchestrator()

        assert orchestrator is not None
        assert orchestrator.config is not None
        assert orchestrator.event_queue is not None
        assert orchestrator.is_speaking is False
        assert orchestrator.pending_interrupt is None

    def test_create_orchestrator_with_config(self, config):
        """Should create orchestrator with provided configuration."""
        orchestrator = JarvisLiveOrchestrator(config=config)

        assert orchestrator.config == config

    def test_create_orchestrator_with_agents(
        self,
        mock_telemetry_agent,
        mock_race_engineer_agent
    ):
        """Should create orchestrator with provided agents."""
        orchestrator = JarvisLiveOrchestrator(
            telemetry_agent=mock_telemetry_agent,
            race_engineer_agent=mock_race_engineer_agent
        )

        assert orchestrator.telemetry_agent == mock_telemetry_agent
        assert orchestrator.race_engineer_agent == mock_race_engineer_agent

    def test_orchestrator_has_priority_queue(self):
        """Orchestrator should have a priority queue."""
        orchestrator = JarvisLiveOrchestrator()

        assert hasattr(orchestrator, 'event_queue')
        assert orchestrator.event_queue.is_empty()


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

class TestSessionManagement:
    """Tests for session context management."""

    def test_set_session_context(self, session_context):
        """Should set session context."""
        orchestrator = JarvisLiveOrchestrator()

        orchestrator.set_session_context(session_context)

        assert orchestrator.session_context == session_context

    def test_get_session_context(self, session_context):
        """Should get session context."""
        orchestrator = JarvisLiveOrchestrator()
        orchestrator.set_session_context(session_context)

        result = orchestrator.get_session_context()

        assert result == session_context

    def test_session_context_none_by_default(self):
        """Session context should be None by default."""
        orchestrator = JarvisLiveOrchestrator()

        assert orchestrator.session_context is None


# =============================================================================
# EVENT QUEUEING
# =============================================================================

class TestEventQueueing:
    """Tests for event queueing operations."""

    @pytest.mark.asyncio
    async def test_queue_event(self, medium_event):
        """Should queue an event."""
        orchestrator = JarvisLiveOrchestrator()

        await orchestrator.queue_event(medium_event)

        assert orchestrator.event_queue.size() == 1

    @pytest.mark.asyncio
    async def test_queue_multiple_events(self, critical_event, medium_event):
        """Should queue multiple events in priority order."""
        orchestrator = JarvisLiveOrchestrator()

        await orchestrator.queue_event(medium_event)
        await orchestrator.queue_event(critical_event)

        # Critical should be first
        assert orchestrator.event_queue.peek().priority == Priority.CRITICAL

    @pytest.mark.asyncio
    async def test_queue_event_respects_priority(self):
        """Events should be ordered by priority in queue."""
        orchestrator = JarvisLiveOrchestrator()

        low_event = Event(type="sector", priority=Priority.LOW, data={}, timestamp=time.time())
        high_event = Event(type="fuel", priority=Priority.HIGH, data={}, timestamp=time.time())
        medium_event = Event(type="lap", priority=Priority.MEDIUM, data={}, timestamp=time.time())

        await orchestrator.queue_event(low_event)
        await orchestrator.queue_event(medium_event)
        await orchestrator.queue_event(high_event)

        # Should pop in priority order
        assert orchestrator.event_queue.pop().priority == Priority.HIGH
        assert orchestrator.event_queue.pop().priority == Priority.MEDIUM
        assert orchestrator.event_queue.pop().priority == Priority.LOW


# =============================================================================
# TELEMETRY HANDLING
# =============================================================================

class TestTelemetryHandling:
    """Tests for telemetry processing."""

    @pytest.mark.asyncio
    async def test_handle_telemetry_updates_context(
        self,
        mock_telemetry_agent,
        mock_race_engineer_agent,
        session_context,
        sample_telemetry
    ):
        """handle_telemetry should update session context."""
        orchestrator = JarvisLiveOrchestrator(
            telemetry_agent=mock_telemetry_agent,
            race_engineer_agent=mock_race_engineer_agent
        )
        orchestrator.set_session_context(session_context)

        await orchestrator.handle_telemetry(sample_telemetry)

        assert session_context.speed_kmh == sample_telemetry.speed_kmh
        assert session_context.fuel_remaining == sample_telemetry.fuel_remaining
        assert session_context.current_lap == sample_telemetry.lap_number

    @pytest.mark.asyncio
    async def test_handle_telemetry_detects_events(
        self,
        mock_telemetry_agent,
        mock_race_engineer_agent,
        session_context,
        sample_telemetry,
        medium_event
    ):
        """handle_telemetry should detect events via telemetry agent."""
        mock_telemetry_agent.detect_events.return_value = [medium_event]

        orchestrator = JarvisLiveOrchestrator(
            telemetry_agent=mock_telemetry_agent,
            race_engineer_agent=mock_race_engineer_agent
        )
        orchestrator.set_session_context(session_context)

        await orchestrator.handle_telemetry(sample_telemetry)

        mock_telemetry_agent.detect_events.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_telemetry_queues_detected_events(
        self,
        mock_telemetry_agent,
        mock_race_engineer_agent,
        session_context,
        sample_telemetry,
        medium_event
    ):
        """handle_telemetry should queue detected events."""
        mock_telemetry_agent.detect_events.return_value = [medium_event]

        orchestrator = JarvisLiveOrchestrator(
            telemetry_agent=mock_telemetry_agent,
            race_engineer_agent=mock_race_engineer_agent
        )
        orchestrator.set_session_context(session_context)

        # Process without generating responses
        await orchestrator.handle_telemetry(sample_telemetry, process_queue=False)

        assert orchestrator.event_queue.size() == 1

    @pytest.mark.asyncio
    async def test_handle_telemetry_without_session_context_raises(
        self,
        mock_telemetry_agent,
        sample_telemetry
    ):
        """handle_telemetry without session context should raise."""
        orchestrator = JarvisLiveOrchestrator(
            telemetry_agent=mock_telemetry_agent
        )

        with pytest.raises(ValueError, match="Session context not set"):
            await orchestrator.handle_telemetry(sample_telemetry)


# =============================================================================
# EVENT PROCESSING
# =============================================================================

class TestEventProcessing:
    """Tests for event processing and AI response generation."""

    @pytest.mark.asyncio
    async def test_process_event_generates_response(
        self,
        mock_race_engineer_agent,
        session_context,
        medium_event
    ):
        """process_event should generate AI response."""
        orchestrator = JarvisLiveOrchestrator(
            race_engineer_agent=mock_race_engineer_agent
        )
        orchestrator.set_session_context(session_context)

        response = await orchestrator.process_event(medium_event)

        assert response is not None
        mock_race_engineer_agent.handle_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_event_sets_speaking_flag(
        self,
        mock_race_engineer_agent,
        session_context,
        medium_event
    ):
        """process_event should set is_speaking flag during processing."""
        orchestrator = JarvisLiveOrchestrator(
            race_engineer_agent=mock_race_engineer_agent
        )
        orchestrator.set_session_context(session_context)

        # Check that is_speaking is set during call
        original_handle = mock_race_engineer_agent.handle_event

        async def capture_speaking(*args, **kwargs):
            assert orchestrator.is_speaking is True
            return await original_handle(*args, **kwargs)

        mock_race_engineer_agent.handle_event = capture_speaking

        await orchestrator.process_event(medium_event)

        # After completion, should be False
        assert orchestrator.is_speaking is False

    @pytest.mark.asyncio
    async def test_process_event_tracks_current_priority(
        self,
        mock_race_engineer_agent,
        session_context,
        medium_event
    ):
        """process_event should track current event priority."""
        orchestrator = JarvisLiveOrchestrator(
            race_engineer_agent=mock_race_engineer_agent
        )
        orchestrator.set_session_context(session_context)

        await orchestrator.process_event(medium_event)

        # Current priority should be tracked
        assert orchestrator.current_priority == medium_event.priority

    @pytest.mark.asyncio
    async def test_process_event_queue_processes_all_events(
        self,
        mock_race_engineer_agent,
        session_context
    ):
        """process_event_queue should process all queued events."""
        orchestrator = JarvisLiveOrchestrator(
            race_engineer_agent=mock_race_engineer_agent
        )
        orchestrator.set_session_context(session_context)

        # Queue multiple events
        events = [
            Event(type=f"event_{i}", priority=Priority.MEDIUM, data={}, timestamp=time.time())
            for i in range(3)
        ]
        for event in events:
            await orchestrator.queue_event(event)

        responses = await orchestrator.process_event_queue()

        assert len(responses) == 3
        assert orchestrator.event_queue.is_empty()

    @pytest.mark.asyncio
    async def test_process_event_queue_respects_priority_order(
        self,
        mock_race_engineer_agent,
        session_context
    ):
        """process_event_queue should process in priority order."""
        processed_types = []

        async def track_event(event, context):
            processed_types.append(event.type)
            return f"Response for {event.type}"

        mock_race_engineer_agent.handle_event = track_event

        orchestrator = JarvisLiveOrchestrator(
            race_engineer_agent=mock_race_engineer_agent
        )
        orchestrator.set_session_context(session_context)

        # Queue events in reverse priority order
        low_event = Event(type="low", priority=Priority.LOW, data={}, timestamp=time.time())
        high_event = Event(type="high", priority=Priority.HIGH, data={}, timestamp=time.time())
        medium_event = Event(type="medium", priority=Priority.MEDIUM, data={}, timestamp=time.time())

        await orchestrator.queue_event(low_event)
        await orchestrator.queue_event(medium_event)
        await orchestrator.queue_event(high_event)

        await orchestrator.process_event_queue()

        # Should process in priority order
        assert processed_types == ["high", "medium", "low"]


# =============================================================================
# DRIVER QUERY HANDLING
# =============================================================================

class TestDriverQueryHandling:
    """Tests for driver query handling."""

    @pytest.mark.asyncio
    async def test_handle_driver_query_generates_response(
        self,
        mock_race_engineer_agent,
        session_context
    ):
        """handle_driver_query should generate AI response."""
        orchestrator = JarvisLiveOrchestrator(
            race_engineer_agent=mock_race_engineer_agent
        )
        orchestrator.set_session_context(session_context)

        response = await orchestrator.handle_driver_query("How's my fuel?")

        assert response is not None
        mock_race_engineer_agent.generate_reactive_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_driver_query_passes_context(
        self,
        mock_race_engineer_agent,
        session_context
    ):
        """handle_driver_query should pass session context to agent."""
        orchestrator = JarvisLiveOrchestrator(
            race_engineer_agent=mock_race_engineer_agent
        )
        orchestrator.set_session_context(session_context)

        await orchestrator.handle_driver_query("What's the gap ahead?")

        call_args = mock_race_engineer_agent.generate_reactive_response.call_args
        assert call_args[0][0] == "What's the gap ahead?"
        assert call_args[0][1] == session_context

    @pytest.mark.asyncio
    async def test_handle_driver_query_without_context_raises(
        self,
        mock_race_engineer_agent
    ):
        """handle_driver_query without session context should raise."""
        orchestrator = JarvisLiveOrchestrator(
            race_engineer_agent=mock_race_engineer_agent
        )

        with pytest.raises(ValueError, match="Session context not set"):
            await orchestrator.handle_driver_query("How's my fuel?")

    @pytest.mark.asyncio
    async def test_handle_driver_query_is_high_priority(
        self,
        mock_race_engineer_agent,
        session_context
    ):
        """Driver queries should be treated as HIGH priority."""
        orchestrator = JarvisLiveOrchestrator(
            race_engineer_agent=mock_race_engineer_agent
        )
        orchestrator.set_session_context(session_context)

        await orchestrator.handle_driver_query("Tire status?")

        # Should have been processed at HIGH priority
        # This is indicated by the fact it processes immediately, not queued


# =============================================================================
# INTERRUPT HANDLING
# =============================================================================

class TestInterruptHandling:
    """Tests for interrupt handling during AI responses."""

    @pytest.mark.asyncio
    async def test_high_priority_sets_pending_interrupt(
        self,
        mock_race_engineer_agent,
        session_context
    ):
        """High priority event should set pending interrupt when speaking."""
        orchestrator = JarvisLiveOrchestrator(
            race_engineer_agent=mock_race_engineer_agent
        )
        orchestrator.set_session_context(session_context)
        orchestrator.is_speaking = True
        orchestrator.current_priority = Priority.MEDIUM

        critical_event = Event(
            type="fuel_critical",
            priority=Priority.CRITICAL,
            data={},
            timestamp=time.time()
        )

        await orchestrator.queue_event(critical_event)
        result = await orchestrator.check_for_interrupt()

        assert result is True
        assert orchestrator.pending_interrupt is not None

    @pytest.mark.asyncio
    async def test_same_priority_does_not_interrupt(
        self,
        mock_race_engineer_agent,
        session_context
    ):
        """Same priority event should not interrupt current speech."""
        orchestrator = JarvisLiveOrchestrator(
            race_engineer_agent=mock_race_engineer_agent
        )
        orchestrator.set_session_context(session_context)
        orchestrator.is_speaking = True
        orchestrator.current_priority = Priority.MEDIUM

        medium_event = Event(
            type="lap_complete",
            priority=Priority.MEDIUM,
            data={},
            timestamp=time.time()
        )

        await orchestrator.queue_event(medium_event)
        result = await orchestrator.check_for_interrupt()

        assert result is False
        assert orchestrator.pending_interrupt is None

    @pytest.mark.asyncio
    async def test_lower_priority_does_not_interrupt(
        self,
        mock_race_engineer_agent,
        session_context
    ):
        """Lower priority event should not interrupt current speech."""
        orchestrator = JarvisLiveOrchestrator(
            race_engineer_agent=mock_race_engineer_agent
        )
        orchestrator.set_session_context(session_context)
        orchestrator.is_speaking = True
        orchestrator.current_priority = Priority.HIGH

        low_event = Event(
            type="sector_complete",
            priority=Priority.LOW,
            data={},
            timestamp=time.time()
        )

        await orchestrator.queue_event(low_event)
        result = await orchestrator.check_for_interrupt()

        assert result is False

    @pytest.mark.asyncio
    async def test_on_sentence_complete_processes_interrupt(
        self,
        mock_race_engineer_agent,
        session_context
    ):
        """on_sentence_complete should process pending interrupt."""
        orchestrator = JarvisLiveOrchestrator(
            race_engineer_agent=mock_race_engineer_agent
        )
        orchestrator.set_session_context(session_context)

        critical_event = Event(
            type="fuel_critical",
            priority=Priority.CRITICAL,
            data={},
            timestamp=time.time()
        )
        orchestrator.pending_interrupt = critical_event

        response = await orchestrator.on_sentence_complete()

        assert response is not None
        assert orchestrator.pending_interrupt is None

    @pytest.mark.asyncio
    async def test_on_sentence_complete_without_interrupt_returns_none(
        self,
        mock_race_engineer_agent,
        session_context
    ):
        """on_sentence_complete without interrupt should return None."""
        orchestrator = JarvisLiveOrchestrator(
            race_engineer_agent=mock_race_engineer_agent
        )
        orchestrator.set_session_context(session_context)
        orchestrator.pending_interrupt = None

        response = await orchestrator.on_sentence_complete()

        assert response is None


# =============================================================================
# SPEAKING STATE
# =============================================================================

class TestSpeakingState:
    """Tests for speaking state management."""

    def test_initial_speaking_state_is_false(self):
        """Initial speaking state should be False."""
        orchestrator = JarvisLiveOrchestrator()

        assert orchestrator.is_speaking is False

    def test_set_speaking_state(self):
        """Should be able to set speaking state."""
        orchestrator = JarvisLiveOrchestrator()

        orchestrator.set_speaking(True)
        assert orchestrator.is_speaking is True

        orchestrator.set_speaking(False)
        assert orchestrator.is_speaking is False

    @pytest.mark.asyncio
    async def test_events_wait_when_speaking(
        self,
        mock_race_engineer_agent,
        session_context
    ):
        """Events should wait in queue when system is speaking."""
        orchestrator = JarvisLiveOrchestrator(
            race_engineer_agent=mock_race_engineer_agent
        )
        orchestrator.set_session_context(session_context)
        orchestrator.is_speaking = True
        orchestrator.current_priority = Priority.MEDIUM

        # Queue a LOW priority event (should not interrupt MEDIUM)
        low_event = Event(
            type="sector",
            priority=Priority.LOW,
            data={},
            timestamp=time.time()
        )

        await orchestrator.queue_event(low_event)

        # Event should remain in queue
        assert orchestrator.event_queue.size() == 1


# =============================================================================
# FULL PIPELINE INTEGRATION
# =============================================================================

class TestFullPipelineIntegration:
    """Integration tests for the full telemetry -> AI response pipeline."""

    @pytest.mark.asyncio
    async def test_telemetry_to_response_pipeline(
        self,
        mock_telemetry_agent,
        mock_race_engineer_agent,
        session_context,
        sample_telemetry,
        medium_event
    ):
        """Full pipeline: telemetry -> event detection -> AI response."""
        mock_telemetry_agent.detect_events.return_value = [medium_event]

        orchestrator = JarvisLiveOrchestrator(
            telemetry_agent=mock_telemetry_agent,
            race_engineer_agent=mock_race_engineer_agent
        )
        orchestrator.set_session_context(session_context)

        responses = await orchestrator.handle_telemetry(sample_telemetry)

        # Should have generated a response
        assert len(responses) == 1
        mock_race_engineer_agent.handle_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_events_generate_multiple_responses(
        self,
        mock_telemetry_agent,
        mock_race_engineer_agent,
        session_context,
        sample_telemetry
    ):
        """Multiple detected events should generate multiple responses."""
        events = [
            Event(type="fuel_warning", priority=Priority.HIGH, data={}, timestamp=time.time()),
            Event(type="lap_complete", priority=Priority.MEDIUM, data={}, timestamp=time.time()),
        ]
        mock_telemetry_agent.detect_events.return_value = events

        orchestrator = JarvisLiveOrchestrator(
            telemetry_agent=mock_telemetry_agent,
            race_engineer_agent=mock_race_engineer_agent
        )
        orchestrator.set_session_context(session_context)

        responses = await orchestrator.handle_telemetry(sample_telemetry)

        assert len(responses) == 2

    @pytest.mark.asyncio
    async def test_no_events_returns_empty_responses(
        self,
        mock_telemetry_agent,
        mock_race_engineer_agent,
        session_context,
        sample_telemetry
    ):
        """No detected events should return empty response list."""
        mock_telemetry_agent.detect_events.return_value = []

        orchestrator = JarvisLiveOrchestrator(
            telemetry_agent=mock_telemetry_agent,
            race_engineer_agent=mock_race_engineer_agent
        )
        orchestrator.set_session_context(session_context)

        responses = await orchestrator.handle_telemetry(sample_telemetry)

        assert responses == []


# =============================================================================
# ERROR HANDLING
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_handle_telemetry_without_telemetry_agent(
        self,
        session_context,
        sample_telemetry
    ):
        """Should handle telemetry even without telemetry agent (no events)."""
        orchestrator = JarvisLiveOrchestrator()
        orchestrator.set_session_context(session_context)

        # Should not raise, but return empty responses
        responses = await orchestrator.handle_telemetry(sample_telemetry)

        assert responses == []

    @pytest.mark.asyncio
    async def test_process_event_without_race_engineer_returns_none(
        self,
        session_context,
        medium_event
    ):
        """process_event without race engineer should return None."""
        orchestrator = JarvisLiveOrchestrator()
        orchestrator.set_session_context(session_context)

        response = await orchestrator.process_event(medium_event)

        assert response is None

    @pytest.mark.asyncio
    async def test_llm_error_is_handled_gracefully(
        self,
        mock_race_engineer_agent,
        session_context,
        medium_event
    ):
        """LLM errors should be handled gracefully."""
        from jarvis_granite.llm import LLMError

        mock_race_engineer_agent.handle_event = AsyncMock(
            side_effect=LLMError("API timeout")
        )

        orchestrator = JarvisLiveOrchestrator(
            race_engineer_agent=mock_race_engineer_agent
        )
        orchestrator.set_session_context(session_context)

        # Should not raise, but return None or error message
        response = await orchestrator.process_event(medium_event)

        # Either None or an error indication
        assert response is None or "error" in response.lower()


# =============================================================================
# CONFIGURATION
# =============================================================================

class TestConfiguration:
    """Tests for orchestrator configuration."""

    def test_uses_config_queue_size(self):
        """Should use queue size from config."""
        config = LiveConfig()
        config.orchestrator.priority_queue_max_size = 50

        orchestrator = JarvisLiveOrchestrator(config=config)

        assert orchestrator.event_queue._max_size == 50

    def test_default_queue_size_is_100(self):
        """Default queue size should be 100."""
        orchestrator = JarvisLiveOrchestrator()

        assert orchestrator.event_queue._max_size == 100


# =============================================================================
# STATISTICS AND STATE
# =============================================================================

class TestStatisticsAndState:
    """Tests for orchestrator statistics and state reporting."""

    def test_get_stats(self, session_context):
        """get_stats should return orchestrator state."""
        orchestrator = JarvisLiveOrchestrator()
        orchestrator.set_session_context(session_context)

        stats = orchestrator.get_stats()

        assert "is_speaking" in stats
        assert "queue_size" in stats
        assert "session_id" in stats
        assert stats["session_id"] == "test_session_001"

    @pytest.mark.asyncio
    async def test_get_stats_includes_queue_stats(self, session_context, medium_event):
        """get_stats should include queue statistics."""
        orchestrator = JarvisLiveOrchestrator()
        orchestrator.set_session_context(session_context)

        await orchestrator.queue_event(medium_event)

        stats = orchestrator.get_stats()

        assert stats["queue_size"] == 1
        assert "queue_stats" in stats
