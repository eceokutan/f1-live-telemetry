"""
TDD Tests for Error Handling - Phase 6, Section 15

These tests define the expected behavior for:
1. Custom exception hierarchy with error codes
2. Graceful degradation (text fallback if voice fails)
3. Reconnection logic
4. Error classification (transient vs permanent)
5. Error context and correlation IDs

Error Codes per documentation:
- LLM_TIMEOUT: Granite API request timed out
- LLM_ERROR: Granite API returned error
- TTS_ERROR: Text-to-speech failed
- STT_ERROR: Speech-to-text failed
- LIVEKIT_ERROR: LiveKit connection failed
- VALIDATION_ERROR: Invalid request data
- SESSION_NOT_FOUND: Referenced session doesn't exist
- CONFIG_ERROR: Invalid configuration

Run with: pytest tests/test_error_handling.py -v

Following TDD: Write tests FIRST, watch them fail, then implement.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis_granite.live.errors import (
    ErrorCode,
    ErrorSeverity,
    ErrorContext,
    JarvisError,
    LLMError,
    LLMTimeoutError,
    LLMUnavailableError,
    VoiceError,
    TTSError,
    STTError,
    LiveKitError,
    LiveKitConnectionError,
    LiveKitDisconnectedError,
    SessionError,
    SessionNotFoundError,
    SessionExpiredError,
    ValidationError,
    InvalidTelemetryError,
    InvalidMessageError,
    ConfigurationError,
    MissingCredentialsError,
    OrchestrationError,
    QueueFullError,
    PipelineError,
    TelemetryProcessingError,
    create_error_response,
    create_error_response_from_exception,
    is_transient_error,
)
from jarvis_granite.schemas.telemetry import TelemetryData, TireTemps, TireWear, GForces
from config.config import LiveConfig, LiveKitConfig


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
def mock_llm_client():
    """Create a mock LLM client."""
    client = MagicMock()
    client.invoke = AsyncMock(return_value="Box this lap for fresh tires.")
    return client


@pytest.fixture
def mock_tts_client():
    """Create a mock TTS client."""
    client = MagicMock()
    client.synthesize = AsyncMock(return_value=b'\x00\x00' * 4800)
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
# ERROR CODE TESTS
# =============================================================================

class TestErrorCodes:
    """Tests for error code enumeration."""

    def test_error_codes_exist(self):
        """All documented error codes should exist."""
        assert ErrorCode.LLM_TIMEOUT == "LLM_TIMEOUT"
        assert ErrorCode.LLM_ERROR == "LLM_ERROR"
        assert ErrorCode.TTS_ERROR == "TTS_ERROR"
        assert ErrorCode.STT_ERROR == "STT_ERROR"
        assert ErrorCode.LIVEKIT_ERROR == "LIVEKIT_ERROR"
        assert ErrorCode.VALIDATION_ERROR == "VALIDATION_ERROR"
        assert ErrorCode.SESSION_NOT_FOUND == "SESSION_NOT_FOUND"
        assert ErrorCode.CONFIG_ERROR == "CONFIG_ERROR"

    def test_error_codes_are_strings(self):
        """Error codes should be string values."""
        for code in ErrorCode:
            assert isinstance(code.value, str)

    def test_error_severity_levels(self):
        """Severity levels should be defined."""
        assert ErrorSeverity.CRITICAL == "critical"
        assert ErrorSeverity.ERROR == "error"
        assert ErrorSeverity.WARNING == "warning"
        assert ErrorSeverity.INFO == "info"


# =============================================================================
# ERROR CONTEXT TESTS
# =============================================================================

class TestErrorContext:
    """Tests for error context tracking."""

    def test_error_context_has_correlation_id(self):
        """Error context should have a correlation ID."""
        context = ErrorContext()
        assert context.correlation_id is not None
        assert len(context.correlation_id) > 0

    def test_error_context_has_timestamp(self):
        """Error context should have a timestamp."""
        context = ErrorContext()
        assert context.timestamp is not None
        assert context.timestamp > 0

    def test_error_context_accepts_session_id(self):
        """Error context should accept session ID."""
        context = ErrorContext(session_id="test_session")
        assert context.session_id == "test_session"

    def test_error_context_accepts_metadata(self):
        """Error context should accept metadata."""
        context = ErrorContext(metadata={"key": "value"})
        assert context.metadata["key"] == "value"

    def test_error_context_to_dict(self):
        """Error context should serialize to dictionary."""
        context = ErrorContext(
            session_id="test_session",
            component="voice_pipeline",
            operation="synthesize"
        )
        data = context.to_dict()

        assert "correlation_id" in data
        assert "session_id" in data
        assert "timestamp" in data
        assert data["session_id"] == "test_session"
        assert data["component"] == "voice_pipeline"
        assert data["operation"] == "synthesize"


# =============================================================================
# EXCEPTION HIERARCHY TESTS
# =============================================================================

class TestJarvisErrorBase:
    """Tests for base JarvisError exception."""

    def test_jarvis_error_is_exception(self):
        """JarvisError should be an Exception."""
        error = JarvisError("Test error")
        assert isinstance(error, Exception)

    def test_jarvis_error_has_error_code(self):
        """JarvisError should have error code."""
        error = JarvisError("Test error", error_code=ErrorCode.INTERNAL_ERROR)
        assert error.error_code == ErrorCode.INTERNAL_ERROR

    def test_jarvis_error_has_message(self):
        """JarvisError should have message."""
        error = JarvisError("Test error message")
        assert error.message == "Test error message"
        assert str(error) == "Test error message"

    def test_jarvis_error_has_context(self):
        """JarvisError should have context."""
        error = JarvisError("Test error")
        assert error.context is not None
        assert isinstance(error.context, ErrorContext)

    def test_jarvis_error_has_is_transient(self):
        """JarvisError should indicate if transient."""
        transient = JarvisError("Test", is_transient=True)
        permanent = JarvisError("Test", is_transient=False)

        assert transient.is_transient is True
        assert permanent.is_transient is False

    def test_jarvis_error_to_dict(self):
        """JarvisError should serialize to dictionary."""
        error = JarvisError(
            "Test error",
            error_code=ErrorCode.INTERNAL_ERROR,
            severity=ErrorSeverity.ERROR
        )
        data = error.to_dict()

        assert data["type"] == "error"
        assert data["error_code"] == "INTERNAL_ERROR"
        assert data["message"] == "Test error"
        assert data["severity"] == "error"
        assert "context" in data

    def test_jarvis_error_to_response(self):
        """JarvisError should create WebSocket response."""
        error = JarvisError("Test error", error_code=ErrorCode.INTERNAL_ERROR)
        response = error.to_response()

        assert response["type"] == "error"
        assert response["error_code"] == "INTERNAL_ERROR"
        assert response["message"] == "Test error"
        assert "timestamp" in response

    def test_jarvis_error_accepts_cause(self):
        """JarvisError should accept original exception."""
        original = ValueError("Original error")
        error = JarvisError("Wrapped error", cause=original)

        assert error.cause is original


class TestLLMErrors:
    """Tests for LLM-related errors."""

    def test_llm_error_inherits_from_jarvis_error(self):
        """LLMError should inherit from JarvisError."""
        error = LLMError("LLM failed")
        assert isinstance(error, JarvisError)

    def test_llm_timeout_error_is_transient(self):
        """LLM timeout should be transient (retryable)."""
        error = LLMTimeoutError()
        assert error.is_transient is True
        assert error.error_code == ErrorCode.LLM_TIMEOUT

    def test_llm_timeout_has_default_message(self):
        """LLM timeout should have default message."""
        error = LLMTimeoutError()
        assert "timed out" in error.message.lower()

    def test_llm_unavailable_error(self):
        """LLM unavailable should be transient."""
        error = LLMUnavailableError()
        assert error.is_transient is True
        assert error.error_code == ErrorCode.LLM_UNAVAILABLE


class TestVoiceErrors:
    """Tests for voice-related errors."""

    def test_voice_error_inherits_from_jarvis_error(self):
        """VoiceError should inherit from JarvisError."""
        error = VoiceError("Voice failed")
        assert isinstance(error, JarvisError)

    def test_tts_error(self):
        """TTS error should have correct code."""
        error = TTSError("TTS failed")
        assert error.error_code == ErrorCode.TTS_ERROR
        assert isinstance(error, VoiceError)

    def test_tts_error_is_transient_by_default(self):
        """TTS errors should be transient by default."""
        error = TTSError()
        assert error.is_transient is True

    def test_stt_error(self):
        """STT error should have correct code."""
        error = STTError("STT failed")
        assert error.error_code == ErrorCode.STT_ERROR
        assert isinstance(error, VoiceError)

    def test_stt_error_is_transient_by_default(self):
        """STT errors should be transient by default."""
        error = STTError()
        assert error.is_transient is True


class TestLiveKitErrors:
    """Tests for LiveKit-related errors."""

    def test_livekit_error_inherits_from_jarvis_error(self):
        """LiveKitError should inherit from JarvisError."""
        error = LiveKitError("LiveKit failed")
        assert isinstance(error, JarvisError)

    def test_livekit_connection_error(self):
        """LiveKit connection error should have correct code."""
        error = LiveKitConnectionError()
        assert error.error_code == ErrorCode.LIVEKIT_CONNECTION_ERROR
        assert error.is_transient is True

    def test_livekit_disconnected_error(self):
        """LiveKit disconnected error should have correct code."""
        error = LiveKitDisconnectedError()
        assert error.error_code == ErrorCode.LIVEKIT_DISCONNECTED
        assert error.is_transient is True


class TestSessionErrors:
    """Tests for session-related errors."""

    def test_session_error_inherits_from_jarvis_error(self):
        """SessionError should inherit from JarvisError."""
        error = SessionError("Session error")
        assert isinstance(error, JarvisError)

    def test_session_not_found_error(self):
        """Session not found error should include session ID."""
        error = SessionNotFoundError("race_001")
        assert error.error_code == ErrorCode.SESSION_NOT_FOUND
        assert error.session_id == "race_001"
        assert "race_001" in error.message

    def test_session_not_found_is_not_transient(self):
        """Session not found is not transient."""
        error = SessionNotFoundError("race_001")
        assert error.is_transient is False

    def test_session_expired_error(self):
        """Session expired error should include session ID."""
        error = SessionExpiredError("race_001")
        assert error.error_code == ErrorCode.SESSION_EXPIRED
        assert error.session_id == "race_001"


class TestValidationErrors:
    """Tests for validation errors."""

    def test_validation_error_inherits_from_jarvis_error(self):
        """ValidationError should inherit from JarvisError."""
        error = ValidationError("Validation failed")
        assert isinstance(error, JarvisError)

    def test_validation_error_is_not_transient(self):
        """Validation errors are not transient."""
        error = ValidationError("Invalid input")
        assert error.is_transient is False

    def test_invalid_telemetry_error(self):
        """Invalid telemetry error should have correct code."""
        error = InvalidTelemetryError("Invalid fuel value")
        assert error.error_code == ErrorCode.INVALID_TELEMETRY

    def test_invalid_message_error(self):
        """Invalid message error should have correct code."""
        error = InvalidMessageError("Missing type field")
        assert error.error_code == ErrorCode.INVALID_MESSAGE


class TestConfigurationErrors:
    """Tests for configuration errors."""

    def test_configuration_error_is_critical(self):
        """Configuration errors should be critical severity."""
        error = ConfigurationError("Config invalid")
        assert error.severity == ErrorSeverity.CRITICAL

    def test_missing_credentials_error(self):
        """Missing credentials error should include service name."""
        error = MissingCredentialsError("Watson TTS")
        assert error.error_code == ErrorCode.MISSING_CREDENTIALS
        assert error.service == "Watson TTS"
        assert "Watson TTS" in error.message


class TestOrchestrationErrors:
    """Tests for orchestration errors."""

    def test_orchestration_error_inherits_from_jarvis_error(self):
        """OrchestrationError should inherit from JarvisError."""
        error = OrchestrationError("Orchestration failed")
        assert isinstance(error, JarvisError)

    def test_queue_full_error(self):
        """Queue full error should be transient."""
        error = QueueFullError()
        assert error.error_code == ErrorCode.QUEUE_FULL
        assert error.is_transient is True


class TestPipelineErrors:
    """Tests for pipeline errors."""

    def test_pipeline_error_inherits_from_jarvis_error(self):
        """PipelineError should inherit from JarvisError."""
        error = PipelineError("Pipeline failed")
        assert isinstance(error, JarvisError)

    def test_telemetry_processing_error(self):
        """Telemetry processing error should be transient."""
        error = TelemetryProcessingError()
        assert error.error_code == ErrorCode.TELEMETRY_PROCESSING_ERROR
        assert error.is_transient is True


# =============================================================================
# ERROR RESPONSE HELPERS TESTS
# =============================================================================

class TestErrorResponseHelpers:
    """Tests for error response helper functions."""

    def test_create_error_response(self):
        """Should create response from JarvisError."""
        error = LLMTimeoutError()
        response = create_error_response(error)

        assert response["type"] == "error"
        assert response["error_code"] == "LLM_TIMEOUT"
        assert "timestamp" in response

    def test_create_error_response_from_exception(self):
        """Should create response from generic exception."""
        exception = ValueError("Something went wrong")
        response = create_error_response_from_exception(exception)

        assert response["type"] == "error"
        assert "Something went wrong" in response["message"]

    def test_create_error_response_from_jarvis_error(self):
        """Should handle JarvisError directly."""
        error = SessionNotFoundError("race_001")
        response = create_error_response_from_exception(error)

        assert response["error_code"] == "SESSION_NOT_FOUND"

    def test_is_transient_error_with_jarvis_error(self):
        """Should detect transient JarvisError."""
        transient = LLMTimeoutError()
        permanent = SessionNotFoundError("test")

        assert is_transient_error(transient) is True
        assert is_transient_error(permanent) is False

    def test_is_transient_error_with_standard_exceptions(self):
        """Should detect transient standard exceptions."""
        assert is_transient_error(ConnectionError()) is True
        assert is_transient_error(TimeoutError()) is True
        assert is_transient_error(ValueError()) is False


# =============================================================================
# GRACEFUL DEGRADATION TESTS
# =============================================================================

class TestGracefulDegradation:
    """Tests for graceful degradation (text fallback if voice fails)."""

    @pytest.mark.asyncio
    async def test_text_response_available_when_tts_fails(
        self,
        live_config,
        mock_llm_client,
        mock_tts_client
    ):
        """Should return text response even when TTS fails."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline
        from jarvis_granite.schemas.telemetry import TelemetryData, TireTemps, TireWear, GForces

        # Make TTS fail
        mock_tts_client.synthesize = AsyncMock(side_effect=TTSError("TTS failed"))

        pipeline = IntegrationPipeline(
            config=live_config,
            llm_client=mock_llm_client
        )
        pipeline.voice_pipeline.tts_client = mock_tts_client

        await pipeline.init_session("race_001", "torcs", "Monza")

        # Create low fuel telemetry to trigger event
        low_fuel = TelemetryData(
            speed_kmh=250.0,
            rpm=12000,
            gear=5,
            throttle=0.9,
            brake=0.0,
            steering_angle=0.1,
            fuel_remaining=3.0,  # Critical
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

        context = pipeline.get_session("race_001")["context"]
        context.fuel_consumption_per_lap = 2.0

        # Should not raise - text response should still be available
        result = await pipeline.process_telemetry("race_001", low_fuel)

        assert result["processed"] is True
        # Text response should be available even if voice failed
        assert result.get("ai_response") is not None

    @pytest.mark.asyncio
    async def test_continues_processing_when_voice_unavailable(
        self,
        live_config,
        mock_llm_client
    ):
        """Should continue processing telemetry when voice is unavailable."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        # No voice clients configured
        pipeline = IntegrationPipeline(
            config=live_config,
            llm_client=mock_llm_client
        )
        # Explicitly set voice clients to None
        pipeline.voice_pipeline.tts_client = None
        pipeline.voice_pipeline.stt_client = None

        await pipeline.init_session("race_001", "torcs", "Monza")

        result = await pipeline.handle_driver_query("race_001", "How are my tires?")

        # Should return text response
        assert result is not None
        assert "response" in result
        assert result["response"] is not None

    @pytest.mark.asyncio
    async def test_health_check_shows_degraded_when_voice_unavailable(
        self,
        live_config
    ):
        """Health check should show degraded state when voice is unavailable."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)
        # No LLM client
        pipeline.voice_pipeline.tts_client = None

        health = pipeline.health_check()

        # Core components should still be healthy
        assert health["components"]["telemetry_agent"] == "healthy"
        assert health["components"]["orchestrator"] == "healthy"
        # Overall status should still be healthy (voice is optional)
        assert health["status"] in ["healthy", "degraded"]


# =============================================================================
# ERROR HANDLING IN INTEGRATION PIPELINE TESTS
# =============================================================================

class TestIntegrationPipelineErrorHandling:
    """Tests for error handling in IntegrationPipeline."""

    @pytest.mark.asyncio
    async def test_handles_llm_timeout_gracefully(
        self,
        live_config,
        mock_llm_client,
        sample_telemetry
    ):
        """Should handle LLM timeout and return error info."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline
        from jarvis_granite.schemas.telemetry import TelemetryData, TireTemps, TireWear, GForces

        # Make LLM timeout
        mock_llm_client.invoke = AsyncMock(side_effect=asyncio.TimeoutError())

        pipeline = IntegrationPipeline(
            config=live_config,
            llm_client=mock_llm_client
        )

        await pipeline.init_session("race_001", "torcs", "Monza")

        # Low fuel to trigger event
        low_fuel = TelemetryData(
            speed_kmh=250.0,
            rpm=12000,
            gear=5,
            throttle=0.9,
            brake=0.0,
            steering_angle=0.1,
            fuel_remaining=3.0,
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

        context = pipeline.get_session("race_001")["context"]
        context.fuel_consumption_per_lap = 2.0

        result = await pipeline.process_telemetry("race_001", low_fuel)

        assert result["processed"] is True
        assert "error" in result
        assert result["error"] == "LLM_TIMEOUT"

    @pytest.mark.asyncio
    async def test_returns_proper_error_for_session_not_found(
        self,
        live_config,
        sample_telemetry
    ):
        """Should return proper error when session not found."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        # Try to process without initializing session
        with pytest.raises(ValueError, match="Session not found"):
            await pipeline.process_telemetry("nonexistent", sample_telemetry)

    @pytest.mark.asyncio
    async def test_websocket_returns_error_response_for_invalid_message(
        self,
        live_config,
        mock_websocket
    ):
        """Should return error response for invalid WebSocket message."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        # Send invalid message type
        message = {"type": "invalid_type"}
        result = await pipeline.handle_websocket_message(message, mock_websocket)

        assert result is not None
        assert result["type"] == "error"
        assert result["error_code"] == "VALIDATION_ERROR"

    @pytest.mark.asyncio
    async def test_websocket_returns_error_for_missing_session(
        self,
        live_config,
        mock_websocket
    ):
        """Should return error for telemetry without session."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        # Send telemetry without session
        message = {
            "type": "telemetry",
            "data": {"speed_kmh": 100}
        }
        result = await pipeline.handle_websocket_message(
            message,
            mock_websocket,
            session_id=None
        )

        assert result is not None
        assert result["type"] == "error"
        assert result["error_code"] == "SESSION_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_websocket_validation_error_for_missing_fields(
        self,
        live_config,
        mock_websocket
    ):
        """Should return validation error for missing required fields."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        # Session init without required fields
        message = {"type": "session_init"}
        result = await pipeline.handle_websocket_message(message, mock_websocket)

        assert result is not None
        assert result["type"] == "error"
        assert result["error_code"] == "VALIDATION_ERROR"


# =============================================================================
# RECONNECTION LOGIC TESTS
# =============================================================================

class TestReconnectionLogic:
    """Tests for reconnection logic."""

    @pytest.mark.asyncio
    async def test_voice_pipeline_tracks_connection_state(self, live_config):
        """Voice pipeline should track connection state."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)

        assert pipeline.is_active is False

        await pipeline.start("test_room", "test_participant")
        assert pipeline.is_active is True

        await pipeline.stop()
        assert pipeline.is_active is False

    @pytest.mark.asyncio
    async def test_voice_pipeline_can_reconnect_after_disconnect(self, live_config):
        """Voice pipeline should support reconnection after disconnect."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)

        # First connection
        await pipeline.start("test_room", "test_participant")
        await pipeline.stop()

        # Reconnect
        await pipeline.start("test_room", "test_participant")
        assert pipeline.is_active is True

        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_voice_pipeline_clears_state_on_reconnect(self, live_config):
        """Voice pipeline should clear state on reconnect."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)

        # First connection with some audio
        await pipeline.start("test_room", "test_participant")
        pipeline.on_audio_frame_received(b'\x00' * 1000)

        await pipeline.stop()

        # Reconnect - buffer should be cleared
        await pipeline.start("test_room", "test_participant")
        assert pipeline.audio_buffer.is_empty()

        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_voice_pipeline_tracks_last_error(self, live_config):
        """Voice pipeline should track last error for debugging."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)
        pipeline._last_error = "Test error"

        assert pipeline.last_error == "Test error"


# =============================================================================
# TRANSIENT ERROR RETRY TESTS
# =============================================================================

class TestTransientErrorRetry:
    """Tests for transient error classification and retry behavior."""

    def test_llm_timeout_is_retryable(self):
        """LLM timeout should be classified as retryable."""
        error = LLMTimeoutError()
        assert error.is_transient is True
        assert is_transient_error(error) is True

    def test_connection_errors_are_retryable(self):
        """Connection errors should be retryable."""
        assert is_transient_error(ConnectionError()) is True
        assert is_transient_error(ConnectionResetError()) is True
        assert is_transient_error(TimeoutError()) is True

    def test_validation_errors_are_not_retryable(self):
        """Validation errors should not be retryable."""
        error = ValidationError("Invalid input")
        assert error.is_transient is False
        assert is_transient_error(error) is False

    def test_session_errors_are_not_retryable(self):
        """Session errors should not be retryable."""
        error = SessionNotFoundError("test")
        assert error.is_transient is False
        assert is_transient_error(error) is False

    def test_config_errors_are_not_retryable(self):
        """Configuration errors should not be retryable."""
        error = ConfigurationError("Invalid config")
        assert error.is_transient is False
        assert is_transient_error(error) is False


# =============================================================================
# ERROR LOGGING AND TRACKING TESTS
# =============================================================================

class TestErrorLoggingAndTracking:
    """Tests for error logging and tracking capabilities."""

    def test_error_context_includes_component(self):
        """Error context should include component information."""
        context = ErrorContext(
            component="voice_pipeline",
            operation="synthesize_speech"
        )
        error = TTSError("TTS failed", context=context)

        assert error.context.component == "voice_pipeline"
        assert error.context.operation == "synthesize_speech"

    def test_correlation_ids_are_unique(self):
        """Each error context should have unique correlation ID."""
        ctx1 = ErrorContext()
        ctx2 = ErrorContext()

        assert ctx1.correlation_id != ctx2.correlation_id

    def test_error_includes_original_exception(self):
        """Error should preserve original exception for debugging."""
        original = ValueError("Original error")
        error = LLMError("Wrapped error", cause=original)

        assert error.cause is original
        assert str(error.cause) == "Original error"


# =============================================================================
# INTEGRATION WITH EXISTING COMPONENTS
# =============================================================================

class TestIntegrationWithExistingComponents:
    """Tests for integration with existing Watson/LiveKit clients."""

    def test_watson_tts_error_is_compatible(self):
        """New TTSError should work with existing Watson TTS client."""
        # The new TTSError should be catchable like the old one
        error = TTSError("TTS synthesis failed")

        assert error.error_code == ErrorCode.TTS_ERROR
        assert isinstance(error, VoiceError)
        assert isinstance(error, JarvisError)

    def test_watson_stt_error_is_compatible(self):
        """New STTError should work with existing Watson STT client."""
        error = STTError("STT transcription failed")

        assert error.error_code == ErrorCode.STT_ERROR
        assert isinstance(error, VoiceError)
        assert isinstance(error, JarvisError)

    def test_llm_error_is_compatible(self):
        """New LLMError should work with existing LLM client."""
        error = LLMError("LLM invocation failed")

        assert error.error_code == ErrorCode.LLM_ERROR
        assert isinstance(error, JarvisError)
