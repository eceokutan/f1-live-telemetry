"""
Tests for Watson TTS/STT Clients - Phase 5, Section 11

These tests verify the expected behavior for:
1. WatsonTTSClient initialization and configuration
2. WatsonSTTClient initialization and configuration
3. Tenacity retry logic with exponential backoff
4. Audio synthesis (TTS) functionality
5. Audio transcription (STT) functionality
6. Error handling and retry behavior

Run with: pytest tests/test_watson_voice.py -v
"""

import asyncio
import time
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis_granite.voice.watson_tts import WatsonTTSClient, TTSError
from jarvis_granite.voice.watson_stt import WatsonSTTClient, STTError
from config.config import RetryConfig


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def tts_config():
    """Create TTS configuration."""
    return {
        "api_key": "test_api_key",
        "service_url": "https://api.us-south.text-to-speech.watson.cloud.ibm.com",
        "voice": "en-GB_JamesV3Voice",
    }


@pytest.fixture
def stt_config():
    """Create STT configuration."""
    return {
        "api_key": "test_api_key",
        "service_url": "https://api.us-south.speech-to-text.watson.cloud.ibm.com",
        "model": "en-GB_BroadbandModel",
    }


@pytest.fixture
def retry_config():
    """Create retry configuration."""
    return RetryConfig(
        max_attempts=3,
        multiplier=1.0,
        min_wait=0.5,
        max_wait=4.0,
    )


@pytest.fixture
def sample_audio_bytes():
    """Create sample audio bytes (WAV header + minimal data)."""
    # Minimal WAV header for testing
    return b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00'


# =============================================================================
# WATSON TTS CLIENT INITIALIZATION
# =============================================================================

class TestWatsonTTSClientInitialization:
    """Tests for WatsonTTSClient initialization."""

    def test_create_tts_client(self, tts_config):
        """Should create a WatsonTTSClient instance."""
        client = WatsonTTSClient(
            api_key=tts_config["api_key"],
            service_url=tts_config["service_url"],
        )

        assert client is not None
        assert client.api_key == tts_config["api_key"]
        assert client.service_url == tts_config["service_url"]

    def test_tts_client_default_voice(self, tts_config):
        """Should use default voice if not specified."""
        client = WatsonTTSClient(
            api_key=tts_config["api_key"],
            service_url=tts_config["service_url"],
        )

        assert client.voice == "en-GB_JamesV3Voice"

    def test_tts_client_custom_voice(self, tts_config):
        """Should accept custom voice."""
        client = WatsonTTSClient(
            api_key=tts_config["api_key"],
            service_url=tts_config["service_url"],
            voice="en-US_MichaelV3Voice",
        )

        assert client.voice == "en-US_MichaelV3Voice"

    def test_tts_client_has_synthesize_method(self, tts_config):
        """TTS client should have synthesize method."""
        client = WatsonTTSClient(
            api_key=tts_config["api_key"],
            service_url=tts_config["service_url"],
        )

        assert hasattr(client, 'synthesize')
        assert asyncio.iscoroutinefunction(client.synthesize)

    def test_tts_client_accepts_retry_config(self, tts_config, retry_config):
        """Should accept retry configuration."""
        client = WatsonTTSClient(
            api_key=tts_config["api_key"],
            service_url=tts_config["service_url"],
            retry_config=retry_config,
        )

        assert client.retry_config == retry_config


# =============================================================================
# WATSON STT CLIENT INITIALIZATION
# =============================================================================

class TestWatsonSTTClientInitialization:
    """Tests for WatsonSTTClient initialization."""

    def test_create_stt_client(self, stt_config):
        """Should create a WatsonSTTClient instance."""
        client = WatsonSTTClient(
            api_key=stt_config["api_key"],
            service_url=stt_config["service_url"],
        )

        assert client is not None
        assert client.api_key == stt_config["api_key"]
        assert client.service_url == stt_config["service_url"]

    def test_stt_client_default_model(self, stt_config):
        """Should use default model if not specified."""
        client = WatsonSTTClient(
            api_key=stt_config["api_key"],
            service_url=stt_config["service_url"],
        )

        assert client.model == "en-GB_BroadbandModel"

    def test_stt_client_custom_model(self, stt_config):
        """Should accept custom model."""
        client = WatsonSTTClient(
            api_key=stt_config["api_key"],
            service_url=stt_config["service_url"],
            model="en-US_BroadbandModel",
        )

        assert client.model == "en-US_BroadbandModel"

    def test_stt_client_has_transcribe_method(self, stt_config):
        """STT client should have transcribe method."""
        client = WatsonSTTClient(
            api_key=stt_config["api_key"],
            service_url=stt_config["service_url"],
        )

        assert hasattr(client, 'transcribe')
        assert asyncio.iscoroutinefunction(client.transcribe)

    def test_stt_client_accepts_retry_config(self, stt_config, retry_config):
        """Should accept retry configuration."""
        client = WatsonSTTClient(
            api_key=stt_config["api_key"],
            service_url=stt_config["service_url"],
            retry_config=retry_config,
        )

        assert client.retry_config == retry_config


# =============================================================================
# TTS SYNTHESIS
# =============================================================================

class TestTTSSynthesis:
    """Tests for TTS synthesis functionality."""

    @pytest.mark.asyncio
    async def test_synthesize_returns_bytes(self, tts_config, sample_audio_bytes):
        """synthesize should return audio bytes."""
        client = WatsonTTSClient(
            api_key=tts_config["api_key"],
            service_url=tts_config["service_url"],
        )

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = sample_audio_bytes

            result = await client.synthesize("Hello, world!")

            assert isinstance(result, bytes)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_synthesize_calls_api_with_text(self, tts_config, sample_audio_bytes):
        """synthesize should call API with correct text."""
        client = WatsonTTSClient(
            api_key=tts_config["api_key"],
            service_url=tts_config["service_url"],
        )

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = sample_audio_bytes

            await client.synthesize("Box this lap.")

            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert "Box this lap." in str(call_args)

    @pytest.mark.asyncio
    async def test_synthesize_uses_configured_voice(self, tts_config, sample_audio_bytes):
        """synthesize should use configured voice."""
        client = WatsonTTSClient(
            api_key=tts_config["api_key"],
            service_url=tts_config["service_url"],
            voice="en-US_AllisonV3Voice",
        )

        # Verify voice is stored on client
        assert client.voice == "en-US_AllisonV3Voice"

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = sample_audio_bytes

            await client.synthesize("Test message")

            # Verify request was made
            mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_empty_text_raises_error(self, tts_config):
        """synthesize with empty text should raise error."""
        client = WatsonTTSClient(
            api_key=tts_config["api_key"],
            service_url=tts_config["service_url"],
        )

        with pytest.raises(ValueError, match="empty"):
            await client.synthesize("")

    @pytest.mark.asyncio
    async def test_synthesize_accepts_audio_format(self, tts_config, sample_audio_bytes):
        """synthesize should accept audio format parameter."""
        client = WatsonTTSClient(
            api_key=tts_config["api_key"],
            service_url=tts_config["service_url"],
        )

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = sample_audio_bytes

            await client.synthesize("Test", audio_format="audio/ogg;codecs=opus")

            call_args = mock_request.call_args
            assert "audio/ogg" in str(call_args) or "opus" in str(call_args)


# =============================================================================
# STT TRANSCRIPTION
# =============================================================================

class TestSTTTranscription:
    """Tests for STT transcription functionality."""

    @pytest.mark.asyncio
    async def test_transcribe_returns_string(self, stt_config, sample_audio_bytes):
        """transcribe should return text string."""
        client = WatsonSTTClient(
            api_key=stt_config["api_key"],
            service_url=stt_config["service_url"],
        )

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "results": [{"alternatives": [{"transcript": "What about fuel?"}]}]
            }

            result = await client.transcribe(sample_audio_bytes)

            assert isinstance(result, str)
            assert result == "What about fuel?"

    @pytest.mark.asyncio
    async def test_transcribe_calls_api_with_audio(self, stt_config, sample_audio_bytes):
        """transcribe should call API with audio data."""
        client = WatsonSTTClient(
            api_key=stt_config["api_key"],
            service_url=stt_config["service_url"],
        )

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "results": [{"alternatives": [{"transcript": "Test"}]}]
            }

            await client.transcribe(sample_audio_bytes)

            mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_uses_configured_model(self, stt_config, sample_audio_bytes):
        """transcribe should use configured model."""
        client = WatsonSTTClient(
            api_key=stt_config["api_key"],
            service_url=stt_config["service_url"],
            model="en-US_NarrowbandModel",
        )

        # Verify model is stored on client
        assert client.model == "en-US_NarrowbandModel"

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "results": [{"alternatives": [{"transcript": "Test"}]}]
            }

            await client.transcribe(sample_audio_bytes)

            # Verify request was made
            mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_empty_audio_raises_error(self, stt_config):
        """transcribe with empty audio should raise error."""
        client = WatsonSTTClient(
            api_key=stt_config["api_key"],
            service_url=stt_config["service_url"],
        )

        with pytest.raises(ValueError, match="empty"):
            await client.transcribe(b"")

    @pytest.mark.asyncio
    async def test_transcribe_handles_no_results(self, stt_config, sample_audio_bytes):
        """transcribe should handle empty results gracefully."""
        client = WatsonSTTClient(
            api_key=stt_config["api_key"],
            service_url=stt_config["service_url"],
        )

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"results": []}

            result = await client.transcribe(sample_audio_bytes)

            assert result == ""

    @pytest.mark.asyncio
    async def test_transcribe_returns_best_alternative(self, stt_config, sample_audio_bytes):
        """transcribe should return the best (first) alternative."""
        client = WatsonSTTClient(
            api_key=stt_config["api_key"],
            service_url=stt_config["service_url"],
        )

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "results": [{
                    "alternatives": [
                        {"transcript": "Best transcription", "confidence": 0.95},
                        {"transcript": "Alternative", "confidence": 0.80},
                    ]
                }]
            }

            result = await client.transcribe(sample_audio_bytes)

            assert result == "Best transcription"


# =============================================================================
# RETRY LOGIC
# =============================================================================

class TestRetryLogic:
    """Tests for Tenacity retry logic."""

    @pytest.mark.asyncio
    async def test_tts_retries_on_timeout(self, tts_config, sample_audio_bytes):
        """TTS should retry on timeout errors."""
        client = WatsonTTSClient(
            api_key=tts_config["api_key"],
            service_url=tts_config["service_url"],
            retry_config=RetryConfig(max_attempts=3, min_wait=0.01, max_wait=0.02),
        )

        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("Connection timeout")
            return sample_audio_bytes

        with patch.object(client, '_make_request', side_effect=mock_request):
            result = await client.synthesize("Test")

            assert call_count == 3
            assert result == sample_audio_bytes

    @pytest.mark.asyncio
    async def test_stt_retries_on_timeout(self, stt_config, sample_audio_bytes):
        """STT should retry on timeout errors."""
        client = WatsonSTTClient(
            api_key=stt_config["api_key"],
            service_url=stt_config["service_url"],
            retry_config=RetryConfig(max_attempts=3, min_wait=0.01, max_wait=0.02),
        )

        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("Connection timeout")
            return {"results": [{"alternatives": [{"transcript": "Test"}]}]}

        with patch.object(client, '_make_request', side_effect=mock_request):
            result = await client.transcribe(sample_audio_bytes)

            assert call_count == 3
            assert result == "Test"

    @pytest.mark.asyncio
    async def test_tts_raises_after_max_retries(self, tts_config):
        """TTS should raise error after max retries exceeded."""
        client = WatsonTTSClient(
            api_key=tts_config["api_key"],
            service_url=tts_config["service_url"],
            retry_config=RetryConfig(max_attempts=3, min_wait=0.01, max_wait=0.02),
        )

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = TimeoutError("Connection timeout")

            with pytest.raises(TTSError):
                await client.synthesize("Test")

    @pytest.mark.asyncio
    async def test_stt_raises_after_max_retries(self, stt_config, sample_audio_bytes):
        """STT should raise error after max retries exceeded."""
        client = WatsonSTTClient(
            api_key=stt_config["api_key"],
            service_url=stt_config["service_url"],
            retry_config=RetryConfig(max_attempts=3, min_wait=0.01, max_wait=0.02),
        )

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = TimeoutError("Connection timeout")

            with pytest.raises(STTError):
                await client.transcribe(sample_audio_bytes)

    @pytest.mark.asyncio
    async def test_tts_retries_on_http_error(self, tts_config, sample_audio_bytes):
        """TTS should retry on HTTP errors (5xx)."""
        client = WatsonTTSClient(
            api_key=tts_config["api_key"],
            service_url=tts_config["service_url"],
            retry_config=RetryConfig(max_attempts=3, min_wait=0.01, max_wait=0.02),
        )

        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("503 Service Unavailable")
            return sample_audio_bytes

        with patch.object(client, '_make_request', side_effect=mock_request):
            result = await client.synthesize("Test")

            assert call_count == 2
            assert result == sample_audio_bytes

    @pytest.mark.asyncio
    async def test_retry_uses_exponential_backoff(self, tts_config, sample_audio_bytes):
        """Retry should use exponential backoff."""
        client = WatsonTTSClient(
            api_key=tts_config["api_key"],
            service_url=tts_config["service_url"],
            retry_config=RetryConfig(max_attempts=3, multiplier=1.0, min_wait=0.1, max_wait=0.5),
        )

        timestamps = []

        async def mock_request(*args, **kwargs):
            timestamps.append(time.time())
            if len(timestamps) < 3:
                raise TimeoutError("Timeout")
            return sample_audio_bytes

        with patch.object(client, '_make_request', side_effect=mock_request):
            await client.synthesize("Test")

            # Check that delays increase (exponential backoff)
            if len(timestamps) >= 3:
                delay1 = timestamps[1] - timestamps[0]
                delay2 = timestamps[2] - timestamps[1]
                # Second delay should be >= first delay (exponential)
                assert delay2 >= delay1 * 0.9  # Allow small margin


# =============================================================================
# ERROR HANDLING
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_tts_error_includes_message(self, tts_config):
        """TTSError should include error message."""
        client = WatsonTTSClient(
            api_key=tts_config["api_key"],
            service_url=tts_config["service_url"],
            retry_config=RetryConfig(max_attempts=1, min_wait=0.01, max_wait=0.02),
        )

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = Exception("API error: Invalid credentials")

            with pytest.raises(TTSError) as exc_info:
                await client.synthesize("Test")

            assert "Invalid credentials" in str(exc_info.value) or "API error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stt_error_includes_message(self, stt_config, sample_audio_bytes):
        """STTError should include error message."""
        client = WatsonSTTClient(
            api_key=stt_config["api_key"],
            service_url=stt_config["service_url"],
            retry_config=RetryConfig(max_attempts=1, min_wait=0.01, max_wait=0.02),
        )

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = Exception("API error: Rate limited")

            with pytest.raises(STTError) as exc_info:
                await client.transcribe(sample_audio_bytes)

            assert "Rate limited" in str(exc_info.value) or "API error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_tts_handles_malformed_response(self, tts_config):
        """TTS should handle malformed response."""
        client = WatsonTTSClient(
            api_key=tts_config["api_key"],
            service_url=tts_config["service_url"],
        )

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = None

            with pytest.raises(TTSError):
                await client.synthesize("Test")

    @pytest.mark.asyncio
    async def test_stt_handles_malformed_response(self, stt_config, sample_audio_bytes):
        """STT should handle malformed response."""
        client = WatsonSTTClient(
            api_key=stt_config["api_key"],
            service_url=stt_config["service_url"],
        )

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"invalid": "response"}

            # Should return empty string for malformed response
            result = await client.transcribe(sample_audio_bytes)
            assert result == ""


# =============================================================================
# AUDIO FORMAT HANDLING
# =============================================================================

class TestAudioFormatHandling:
    """Tests for audio format handling."""

    @pytest.mark.asyncio
    async def test_tts_default_audio_format(self, tts_config, sample_audio_bytes):
        """TTS should use WAV as default format."""
        client = WatsonTTSClient(
            api_key=tts_config["api_key"],
            service_url=tts_config["service_url"],
        )

        assert client.default_audio_format == "audio/wav"

    @pytest.mark.asyncio
    async def test_stt_supports_multiple_formats(self, stt_config, sample_audio_bytes):
        """STT should support multiple audio formats."""
        client = WatsonSTTClient(
            api_key=stt_config["api_key"],
            service_url=stt_config["service_url"],
        )

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "results": [{"alternatives": [{"transcript": "Test"}]}]
            }

            # Should accept content type parameter
            await client.transcribe(sample_audio_bytes, content_type="audio/ogg")

            call_args = mock_request.call_args
            assert "audio/ogg" in str(call_args)


# =============================================================================
# CLIENT CONFIGURATION
# =============================================================================

class TestClientConfiguration:
    """Tests for client configuration options."""

    def test_tts_timeout_configuration(self, tts_config):
        """TTS client should accept timeout configuration."""
        client = WatsonTTSClient(
            api_key=tts_config["api_key"],
            service_url=tts_config["service_url"],
            timeout=30.0,
        )

        assert client.timeout == 30.0

    def test_stt_timeout_configuration(self, stt_config):
        """STT client should accept timeout configuration."""
        client = WatsonSTTClient(
            api_key=stt_config["api_key"],
            service_url=stt_config["service_url"],
            timeout=30.0,
        )

        assert client.timeout == 30.0

    def test_tts_default_timeout(self, tts_config):
        """TTS client should have sensible default timeout."""
        client = WatsonTTSClient(
            api_key=tts_config["api_key"],
            service_url=tts_config["service_url"],
        )

        assert client.timeout == 10.0  # Default 10 seconds

    def test_stt_default_timeout(self, stt_config):
        """STT client should have sensible default timeout."""
        client = WatsonSTTClient(
            api_key=stt_config["api_key"],
            service_url=stt_config["service_url"],
        )

        assert client.timeout == 30.0  # Default 30 seconds for longer audio


# =============================================================================
# STATISTICS AND STATE
# =============================================================================

class TestStatisticsAndState:
    """Tests for client statistics."""

    @pytest.mark.asyncio
    async def test_tts_tracks_request_count(self, tts_config, sample_audio_bytes):
        """TTS should track request count."""
        client = WatsonTTSClient(
            api_key=tts_config["api_key"],
            service_url=tts_config["service_url"],
        )

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = sample_audio_bytes

            await client.synthesize("Test 1")
            await client.synthesize("Test 2")

            stats = client.get_stats()
            assert stats["total_requests"] == 2

    @pytest.mark.asyncio
    async def test_stt_tracks_request_count(self, stt_config, sample_audio_bytes):
        """STT should track request count."""
        client = WatsonSTTClient(
            api_key=stt_config["api_key"],
            service_url=stt_config["service_url"],
        )

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "results": [{"alternatives": [{"transcript": "Test"}]}]
            }

            await client.transcribe(sample_audio_bytes)
            await client.transcribe(sample_audio_bytes)

            stats = client.get_stats()
            assert stats["total_requests"] == 2

    @pytest.mark.asyncio
    async def test_tts_tracks_error_count(self, tts_config):
        """TTS should track error count."""
        client = WatsonTTSClient(
            api_key=tts_config["api_key"],
            service_url=tts_config["service_url"],
            retry_config=RetryConfig(max_attempts=1, min_wait=0.01, max_wait=0.02),
        )

        with patch.object(client, '_make_request', new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = Exception("Error")

            try:
                await client.synthesize("Test")
            except TTSError:
                pass

            stats = client.get_stats()
            assert stats["total_errors"] >= 1
