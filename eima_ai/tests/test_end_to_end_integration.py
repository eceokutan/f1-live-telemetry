"""
TDD Tests for End-to-End Integration - Phase 6, Section 14

These tests define the expected behavior for:
1. Full pipeline integration: Telemetry -> Event -> LLM -> Voice
2. Session lifecycle: init -> active -> end
3. LiveKit token generation in session_confirmed
4. All components wired together correctly
5. Error handling and graceful degradation

Run with: pytest tests/test_end_to_end_integration.py -v

Following TDD: Write tests FIRST, watch them fail, then implement IntegrationPipeline.
"""

import asyncio
import time
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis_granite.schemas.events import Event, Priority
from jarvis_granite.schemas.telemetry import TelemetryData, TireTemps, TireWear, GForces
from jarvis_granite.live.context import LiveSessionContext
from config.config import LiveConfig, LiveKitConfig, ThresholdsConfig


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def live_config():
    """Create test live configuration."""
    return LiveConfig(
        livekit=LiveKitConfig(
            url="wss://test-livekit.example.com",
            api_key="test_api_key",
            api_secret="test_api_secret",
            sample_rate=48000,
        )
    )


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
def low_fuel_telemetry():
    """Create telemetry with critical fuel level."""
    return TelemetryData(
        speed_kmh=250.0,
        rpm=12000,
        gear=5,
        throttle=0.9,
        brake=0.0,
        steering_angle=0.1,
        fuel_remaining=3.0,  # Critical - only ~1.5 laps
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
def high_tire_temp_telemetry():
    """Create telemetry with critical tire temperature."""
    return TelemetryData(
        speed_kmh=250.0,
        rpm=12000,
        gear=5,
        throttle=0.9,
        brake=0.0,
        steering_angle=0.1,
        fuel_remaining=50.0,
        tire_temps=TireTemps(fl=115.0, fr=116.0, rl=112.0, rr=113.0),  # Critical temps
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
def mock_llm_client():
    """Create a mock LLM client."""
    client = MagicMock()
    client.invoke = AsyncMock(return_value="Box this lap for fresh tires.")
    return client


@pytest.fixture
def mock_tts_client():
    """Create a mock TTS client."""
    client = MagicMock()
    client.synthesize = AsyncMock(return_value=b'\x00\x00' * 4800)  # Mock audio bytes
    return client


@pytest.fixture
def mock_stt_client():
    """Create a mock STT client."""
    client = MagicMock()
    client.transcribe = AsyncMock(return_value="How are my tires?")
    return client


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    websocket = AsyncMock()
    websocket.send_json = AsyncMock()
    websocket.receive_json = AsyncMock()
    websocket.accept = AsyncMock()
    websocket.close = AsyncMock()
    return websocket


# =============================================================================
# INTEGRATION PIPELINE INITIALIZATION
# =============================================================================

class TestIntegrationPipelineInitialization:
    """Tests for IntegrationPipeline initialization and component wiring."""

    def test_create_integration_pipeline_with_config(self, live_config):
        """Should create integration pipeline with configuration."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        assert pipeline is not None
        assert pipeline.config == live_config
        assert pipeline.orchestrator is not None
        assert pipeline.telemetry_agent is not None

    def test_integration_pipeline_has_all_components(self, live_config):
        """Should have all required components wired together."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        # Verify all components are present
        assert pipeline.orchestrator is not None
        assert pipeline.telemetry_agent is not None
        assert pipeline.voice_pipeline is not None
        assert pipeline.session_manager is not None

    def test_components_are_properly_wired(self, live_config):
        """Should have components properly connected to each other."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        # Orchestrator should have telemetry agent
        assert pipeline.orchestrator.telemetry_agent is pipeline.telemetry_agent

        # Voice pipeline should have orchestrator reference
        assert pipeline.voice_pipeline.orchestrator is pipeline.orchestrator

    def test_integration_pipeline_with_llm_client(self, live_config, mock_llm_client):
        """Should accept and configure LLM client."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(
            config=live_config,
            llm_client=mock_llm_client
        )

        assert pipeline.race_engineer_agent is not None
        assert pipeline.orchestrator.race_engineer_agent is not None


# =============================================================================
# SESSION LIFECYCLE
# =============================================================================

class TestSessionLifecycle:
    """Tests for session lifecycle: init -> active -> end."""

    @pytest.mark.asyncio
    async def test_session_init_creates_context(self, live_config):
        """Should create session context on initialization."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        session_info = await pipeline.init_session(
            session_id="race_001",
            source="torcs",
            track_name="Monza",
            config={"verbosity": "moderate"}
        )

        assert session_info is not None
        assert session_info["session_id"] == "race_001"
        assert "livekit" in session_info
        assert pipeline.get_session("race_001") is not None

    @pytest.mark.asyncio
    async def test_session_init_returns_livekit_token(self, live_config):
        """Should return LiveKit connection info including token."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        session_info = await pipeline.init_session(
            session_id="race_001",
            source="torcs",
            track_name="Monza"
        )

        # Verify LiveKit info is present
        assert "livekit" in session_info
        assert "url" in session_info["livekit"]
        assert "token" in session_info["livekit"]
        assert "room_name" in session_info["livekit"]

        # Token should not be a placeholder
        token = session_info["livekit"]["token"]
        assert token is not None
        assert len(token) > 20  # Real JWT tokens are longer
        assert "placeholder" not in token.lower()

    @pytest.mark.asyncio
    async def test_session_active_accepts_telemetry(self, live_config, sample_telemetry):
        """Should process telemetry during active session."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        # Init session
        await pipeline.init_session(
            session_id="race_001",
            source="torcs",
            track_name="Monza"
        )

        # Send telemetry - should not raise
        result = await pipeline.process_telemetry("race_001", sample_telemetry)

        assert result is not None
        assert result["processed"] is True

    @pytest.mark.asyncio
    async def test_session_end_cleans_up(self, live_config):
        """Should clean up resources on session end."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        # Init and end session
        await pipeline.init_session(
            session_id="race_001",
            source="torcs",
            track_name="Monza"
        )

        await pipeline.end_session("race_001")

        # Session should be gone
        assert pipeline.get_session("race_001") is None

    @pytest.mark.asyncio
    async def test_cannot_process_telemetry_without_session(self, live_config, sample_telemetry):
        """Should reject telemetry for non-existent session."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        with pytest.raises(ValueError, match="Session not found"):
            await pipeline.process_telemetry("nonexistent", sample_telemetry)

    @pytest.mark.asyncio
    async def test_multiple_sessions_are_isolated(self, live_config, sample_telemetry):
        """Should maintain separate state for multiple sessions."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        # Create two sessions
        await pipeline.init_session("race_001", "torcs", "Monza")
        await pipeline.init_session("race_002", "assetto", "Spa")

        # Verify both exist and are separate
        session1 = pipeline.get_session("race_001")
        session2 = pipeline.get_session("race_002")

        assert session1 is not None
        assert session2 is not None
        assert session1["context"].track_name == "Monza"
        assert session2["context"].track_name == "Spa"


# =============================================================================
# FULL PIPELINE: TELEMETRY -> EVENT -> LLM -> VOICE
# =============================================================================

class TestFullPipeline:
    """Tests for complete pipeline: Telemetry -> Event -> LLM -> Voice."""

    @pytest.mark.asyncio
    async def test_telemetry_triggers_event_detection(self, live_config, low_fuel_telemetry):
        """Should detect events from telemetry data."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        await pipeline.init_session("race_001", "torcs", "Monza")

        # Set up fuel consumption rate for accurate detection
        context = pipeline.get_session("race_001")["context"]
        context.fuel_consumption_per_lap = 2.0  # ~1.5 laps remaining with 3L

        result = await pipeline.process_telemetry("race_001", low_fuel_telemetry)

        assert result is not None
        assert "events" in result
        assert len(result["events"]) > 0
        # Should detect fuel critical event
        assert any(e["type"] == "fuel_critical" for e in result["events"])

    @pytest.mark.asyncio
    async def test_event_triggers_llm_response(self, live_config, low_fuel_telemetry, mock_llm_client):
        """Should generate LLM response for detected events."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(
            config=live_config,
            llm_client=mock_llm_client
        )

        await pipeline.init_session("race_001", "torcs", "Monza")

        # Set up fuel consumption
        context = pipeline.get_session("race_001")["context"]
        context.fuel_consumption_per_lap = 2.0

        result = await pipeline.process_telemetry("race_001", low_fuel_telemetry)

        assert "ai_response" in result
        assert result["ai_response"] is not None
        assert len(result["ai_response"]) > 0

    @pytest.mark.asyncio
    async def test_llm_response_sent_to_voice_pipeline(
        self,
        live_config,
        low_fuel_telemetry,
        mock_llm_client,
        mock_tts_client
    ):
        """Should send LLM response to voice pipeline for TTS."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(
            config=live_config,
            llm_client=mock_llm_client
        )
        pipeline.voice_pipeline.tts_client = mock_tts_client

        await pipeline.init_session("race_001", "torcs", "Monza")

        context = pipeline.get_session("race_001")["context"]
        context.fuel_consumption_per_lap = 2.0

        result = await pipeline.process_telemetry("race_001", low_fuel_telemetry)

        # TTS should have been called if voice is enabled
        if pipeline.config.voice.enable_voice:
            mock_tts_client.synthesize.assert_called()

    @pytest.mark.asyncio
    async def test_full_pipeline_latency_tracking(
        self,
        live_config,
        low_fuel_telemetry,
        mock_llm_client
    ):
        """Should track end-to-end latency."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(
            config=live_config,
            llm_client=mock_llm_client
        )

        await pipeline.init_session("race_001", "torcs", "Monza")

        context = pipeline.get_session("race_001")["context"]
        context.fuel_consumption_per_lap = 2.0

        result = await pipeline.process_telemetry("race_001", low_fuel_telemetry)

        assert "latency_ms" in result
        assert isinstance(result["latency_ms"], int)

    @pytest.mark.asyncio
    async def test_driver_query_triggers_response(self, live_config, mock_llm_client):
        """Should generate response for driver voice queries."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        mock_llm_client.invoke = AsyncMock(
            return_value="Tires are looking good, about 15 laps remaining."
        )

        pipeline = IntegrationPipeline(
            config=live_config,
            llm_client=mock_llm_client
        )

        await pipeline.init_session("race_001", "torcs", "Monza")

        result = await pipeline.handle_driver_query(
            "race_001",
            "How are my tires?"
        )

        assert result is not None
        assert "response" in result
        assert len(result["response"]) > 0


# =============================================================================
# LIVEKIT TOKEN GENERATION
# =============================================================================

class TestLiveKitTokenGeneration:
    """Tests for proper LiveKit token generation."""

    @pytest.mark.asyncio
    async def test_token_is_valid_jwt(self, live_config):
        """Should generate valid JWT token for LiveKit."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        session_info = await pipeline.init_session(
            session_id="race_001",
            source="torcs",
            track_name="Monza"
        )

        token = session_info["livekit"]["token"]

        # JWT tokens have 3 parts separated by dots
        parts = token.split(".")
        assert len(parts) == 3

        # Each part should be base64 encoded
        for part in parts:
            assert len(part) > 0

    @pytest.mark.asyncio
    async def test_token_includes_room_name(self, live_config):
        """Should include room name in token grants."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        session_info = await pipeline.init_session(
            session_id="race_001",
            source="torcs",
            track_name="Monza"
        )

        room_name = session_info["livekit"]["room_name"]

        # Room name should be derived from session_id
        assert "race_001" in room_name

    @pytest.mark.asyncio
    async def test_each_session_gets_unique_token(self, live_config):
        """Should generate unique tokens for each session."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        session1 = await pipeline.init_session("race_001", "torcs", "Monza")
        session2 = await pipeline.init_session("race_002", "assetto", "Spa")

        token1 = session1["livekit"]["token"]
        token2 = session2["livekit"]["token"]

        assert token1 != token2


# =============================================================================
# ERROR HANDLING AND GRACEFUL DEGRADATION
# =============================================================================

class TestErrorHandling:
    """Tests for error handling and graceful degradation."""

    @pytest.mark.asyncio
    async def test_continues_without_llm_client(self, live_config, sample_telemetry):
        """Should process telemetry even without LLM client."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        # Create without LLM client
        pipeline = IntegrationPipeline(config=live_config, llm_client=None)

        await pipeline.init_session("race_001", "torcs", "Monza")

        # Should process without error
        result = await pipeline.process_telemetry("race_001", sample_telemetry)

        assert result["processed"] is True
        # No AI response expected
        assert result.get("ai_response") is None

    @pytest.mark.asyncio
    async def test_handles_llm_timeout(self, live_config, low_fuel_telemetry, mock_llm_client):
        """Should handle LLM timeout gracefully."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        # Make LLM timeout
        mock_llm_client.invoke = AsyncMock(side_effect=asyncio.TimeoutError())

        pipeline = IntegrationPipeline(
            config=live_config,
            llm_client=mock_llm_client
        )

        await pipeline.init_session("race_001", "torcs", "Monza")

        context = pipeline.get_session("race_001")["context"]
        context.fuel_consumption_per_lap = 2.0

        # Should not raise, should return error info
        result = await pipeline.process_telemetry("race_001", low_fuel_telemetry)

        assert result["processed"] is True
        assert "error" in result or result.get("ai_response") is None

    @pytest.mark.asyncio
    async def test_handles_voice_pipeline_failure(
        self,
        live_config,
        low_fuel_telemetry,
        mock_llm_client,
        mock_tts_client
    ):
        """Should handle voice pipeline failure gracefully."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        # Make TTS fail
        mock_tts_client.synthesize = AsyncMock(side_effect=Exception("TTS error"))

        pipeline = IntegrationPipeline(
            config=live_config,
            llm_client=mock_llm_client
        )
        pipeline.voice_pipeline.tts_client = mock_tts_client

        await pipeline.init_session("race_001", "torcs", "Monza")

        context = pipeline.get_session("race_001")["context"]
        context.fuel_consumption_per_lap = 2.0

        # Should not raise
        result = await pipeline.process_telemetry("race_001", low_fuel_telemetry)

        assert result["processed"] is True
        # AI response text should still be available
        assert result.get("ai_response") is not None


# =============================================================================
# INTERRUPT HANDLING IN FULL PIPELINE
# =============================================================================

class TestInterruptHandling:
    """Tests for interrupt handling across the full pipeline."""

    @pytest.mark.asyncio
    async def test_critical_event_interrupts_medium_response(
        self,
        live_config,
        sample_telemetry,
        high_tire_temp_telemetry,
        mock_llm_client
    ):
        """Should interrupt medium priority response with critical event."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(
            config=live_config,
            llm_client=mock_llm_client
        )

        await pipeline.init_session("race_001", "torcs", "Monza")

        context = pipeline.get_session("race_001")["context"]

        # Simulate being in the middle of a medium priority response
        pipeline.orchestrator.is_speaking = True
        pipeline.orchestrator.current_priority = Priority.MEDIUM

        # Send critical tire temp telemetry
        result = await pipeline.process_telemetry("race_001", high_tire_temp_telemetry)

        # Should have detected interrupt condition
        assert "interrupt_triggered" in result or pipeline.orchestrator.pending_interrupt is not None

    @pytest.mark.asyncio
    async def test_driver_query_interrupts_current_response(
        self,
        live_config,
        mock_llm_client
    ):
        """Should interrupt current AI response for driver query."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(
            config=live_config,
            llm_client=mock_llm_client
        )

        await pipeline.init_session("race_001", "torcs", "Monza")

        # Simulate being in the middle of a response
        pipeline.orchestrator.is_speaking = True
        pipeline.orchestrator.current_priority = Priority.MEDIUM

        # Driver query should trigger interrupt
        result = await pipeline.handle_driver_query("race_001", "What about fuel?")

        # Response should be generated
        assert "response" in result


# =============================================================================
# STATISTICS AND MONITORING
# =============================================================================

class TestStatisticsAndMonitoring:
    """Tests for statistics and monitoring capabilities."""

    @pytest.mark.asyncio
    async def test_pipeline_tracks_statistics(self, live_config, sample_telemetry):
        """Should track pipeline statistics."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        await pipeline.init_session("race_001", "torcs", "Monza")
        await pipeline.process_telemetry("race_001", sample_telemetry)

        stats = pipeline.get_stats()

        assert "active_sessions" in stats
        assert "telemetry_processed" in stats
        assert "events_detected" in stats
        assert "ai_responses_generated" in stats

    @pytest.mark.asyncio
    async def test_session_stats_are_tracked(self, live_config, sample_telemetry):
        """Should track per-session statistics."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        await pipeline.init_session("race_001", "torcs", "Monza")

        # Process multiple telemetry messages
        for _ in range(5):
            await pipeline.process_telemetry("race_001", sample_telemetry)

        session = pipeline.get_session("race_001")

        assert "stats" in session
        assert session["stats"]["telemetry_count"] == 5

    def test_health_check_returns_status(self, live_config):
        """Should return health check status."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        health = pipeline.health_check()

        assert "status" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        assert "components" in health


# =============================================================================
# WEBSOCKET INTEGRATION
# =============================================================================

class TestWebSocketIntegration:
    """Tests for WebSocket integration with the pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_handles_websocket_session_init(
        self,
        live_config,
        mock_websocket
    ):
        """Should handle WebSocket session_init message."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        message = {
            "type": "session_init",
            "session_id": "race_001",
            "source": "torcs",
            "track_name": "Monza",
            "config": {"verbosity": "moderate"}
        }

        result = await pipeline.handle_websocket_message(message, mock_websocket)

        assert result is not None
        assert result.get("type") == "session_confirmed"
        assert "livekit" in result

    @pytest.mark.asyncio
    async def test_pipeline_handles_websocket_telemetry(
        self,
        live_config,
        mock_websocket,
        sample_telemetry
    ):
        """Should handle WebSocket telemetry message."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        # Init session first
        init_message = {
            "type": "session_init",
            "session_id": "race_001",
            "source": "torcs",
            "track_name": "Monza"
        }
        await pipeline.handle_websocket_message(init_message, mock_websocket)

        # Now send telemetry
        telemetry_message = {
            "type": "telemetry",
            "session_id": "race_001",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": sample_telemetry.model_dump()
        }

        result = await pipeline.handle_websocket_message(
            telemetry_message,
            mock_websocket,
            session_id="race_001"
        )

        assert result is None or result.get("error") is None

    @pytest.mark.asyncio
    async def test_pipeline_handles_websocket_session_end(
        self,
        live_config,
        mock_websocket
    ):
        """Should handle WebSocket session_end message."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        # Init session
        init_message = {
            "type": "session_init",
            "session_id": "race_001",
            "source": "torcs",
            "track_name": "Monza"
        }
        await pipeline.handle_websocket_message(init_message, mock_websocket)

        # End session
        end_message = {"type": "session_end"}
        await pipeline.handle_websocket_message(
            end_message,
            mock_websocket,
            session_id="race_001"
        )

        # Session should be ended
        assert pipeline.get_session("race_001") is None


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

class TestConfigurationValidation:
    """Tests for configuration validation in the pipeline."""

    def test_rejects_invalid_livekit_config(self):
        """Should reject configuration with missing LiveKit credentials."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        config = LiveConfig(
            livekit=LiveKitConfig(
                url="",  # Invalid - empty URL
                api_key="test_key",
                api_secret="test_secret"
            )
        )

        pipeline = IntegrationPipeline(config=config)

        # Health check should indicate degraded state
        health = pipeline.health_check()
        assert health["components"]["livekit"] != "healthy"

    def test_accepts_valid_configuration(self, live_config):
        """Should accept valid configuration."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        health = pipeline.health_check()
        assert health["status"] in ["healthy", "degraded"]
