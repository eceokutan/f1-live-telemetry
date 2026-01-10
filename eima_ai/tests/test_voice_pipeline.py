"""
Tests for Voice Pipeline - Phase 5, Section 13

These tests verify the expected behavior for:
1. VoicePipeline initialization and configuration
2. Audio buffering and silence detection
3. STT -> Orchestrator flow
4. Orchestrator -> TTS flow
5. Audio frame streaming
6. Interrupt handling in the pipeline

Run with: pytest tests/test_voice_pipeline.py -v
"""

import asyncio
import pytest
import struct
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock livekit modules before importing our code
mock_rtc = MagicMock()
mock_api = MagicMock()

# Configure mock classes
mock_rtc.Room = MagicMock
mock_rtc.AudioSource = MagicMock
mock_rtc.LocalAudioTrack = MagicMock
mock_rtc.LocalAudioTrack.create_audio_track = MagicMock(return_value=MagicMock())
mock_rtc.AudioStream = MagicMock
mock_rtc.TrackKind = MagicMock()
mock_rtc.TrackKind.KIND_AUDIO = "audio"
mock_rtc.AudioFrame = MagicMock

# Mock access token
mock_access_token = MagicMock()
mock_access_token.to_jwt.return_value = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.signature"
mock_access_token.with_identity.return_value = mock_access_token
mock_access_token.with_name.return_value = mock_access_token
mock_access_token.with_grants.return_value = mock_access_token
mock_api.AccessToken = MagicMock(return_value=mock_access_token)
mock_api.VideoGrants = MagicMock

# Patch at module level
sys.modules['livekit'] = MagicMock()
sys.modules['livekit.rtc'] = mock_rtc
sys.modules['livekit.api'] = mock_api
sys.modules['livekit'].rtc = mock_rtc
sys.modules['livekit'].api = mock_api

# Clear cached imports
for mod_name in list(sys.modules.keys()):
    if 'jarvis_granite.voice' in mod_name:
        del sys.modules[mod_name]

from config.config import LiveConfig, LiveKitConfig


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def live_config():
    """Create full LiveConfig."""
    return LiveConfig(
        livekit=LiveKitConfig(
            url="wss://livekit.example.com",
            api_key="test_api_key",
            api_secret="test_api_secret",
            room_prefix="jarvis_live_",
            audio_codec="opus",
            sample_rate=48000,
        )
    )


@pytest.fixture
def sample_audio_bytes():
    """Create sample audio bytes (WAV header + minimal data)."""
    return b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00'


@pytest.fixture
def sample_audio_with_data():
    """Create sample audio with actual PCM data."""
    # WAV header
    header = b'RIFF'
    header += struct.pack('<I', 36 + 1920)  # File size - 8
    header += b'WAVE'
    header += b'fmt '
    header += struct.pack('<I', 16)  # Subchunk1Size
    header += struct.pack('<H', 1)   # AudioFormat (PCM)
    header += struct.pack('<H', 1)   # NumChannels
    header += struct.pack('<I', 48000)  # SampleRate
    header += struct.pack('<I', 96000)  # ByteRate
    header += struct.pack('<H', 2)   # BlockAlign
    header += struct.pack('<H', 16)  # BitsPerSample
    header += b'data'
    header += struct.pack('<I', 1920)  # Subchunk2Size

    # PCM data (960 samples = 20ms at 48kHz)
    pcm_data = bytes([0] * 1920)  # Silence

    return header + pcm_data


@pytest.fixture
def loud_audio_frame():
    """Create audio frame with loud signal (non-silence)."""
    # Generate 20ms of 1kHz tone at 48kHz
    samples = 960
    data = bytearray()
    for i in range(samples):
        # Generate 16-bit signed sample
        import math
        value = int(16000 * math.sin(2 * math.pi * 1000 * i / 48000))
        data.extend(struct.pack('<h', value))
    return bytes(data)


@pytest.fixture
def silent_audio_frame():
    """Create audio frame with silence."""
    # 960 samples of silence
    return bytes([0] * 1920)


@pytest.fixture
def mock_tts_client(sample_audio_bytes):
    """Create mock TTS client."""
    client = MagicMock()
    client.synthesize = AsyncMock(return_value=sample_audio_bytes)
    return client


@pytest.fixture
def mock_stt_client():
    """Create mock STT client."""
    client = MagicMock()
    client.transcribe = AsyncMock(return_value="What about my tires?")
    return client


@pytest.fixture
def mock_orchestrator():
    """Create mock orchestrator."""
    orchestrator = MagicMock()
    orchestrator.handle_driver_query = AsyncMock(
        return_value="Your tires are in good condition."
    )
    orchestrator.is_speaking = False
    orchestrator.set_speaking = MagicMock()
    return orchestrator


@pytest.fixture
def mock_livekit_client():
    """Create mock LiveKit client."""
    client = MagicMock()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.send_audio = AsyncMock()
    client.is_connected = True
    client.on_audio_received = None
    return client


# =============================================================================
# VOICE PIPELINE INITIALIZATION
# =============================================================================

class TestVoicePipelineInitialization:
    """Tests for VoicePipeline initialization."""

    def test_create_voice_pipeline(self, live_config):
        """Should create a VoicePipeline instance."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)

        assert pipeline is not None
        assert pipeline.config == live_config

    def test_pipeline_has_audio_buffer(self, live_config):
        """Pipeline should have an audio buffer."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)

        assert hasattr(pipeline, 'audio_buffer')
        assert pipeline.audio_buffer is not None

    def test_pipeline_has_silence_detector(self, live_config):
        """Pipeline should have silence detection."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)

        assert hasattr(pipeline, 'silence_detector')

    def test_pipeline_has_required_components(self, live_config):
        """Pipeline should have slots for TTS, STT, orchestrator."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)

        assert hasattr(pipeline, 'tts_client')
        assert hasattr(pipeline, 'stt_client')
        assert hasattr(pipeline, 'orchestrator')
        assert hasattr(pipeline, 'livekit_client')

    def test_pipeline_default_state(self, live_config):
        """Pipeline should start in inactive state."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)

        assert pipeline.is_active is False
        assert pipeline.is_processing is False


# =============================================================================
# SILENCE DETECTION
# =============================================================================

class TestSilenceDetection:
    """Tests for silence detection functionality."""

    def test_silence_detector_initialization(self, live_config):
        """SilenceDetector should be properly initialized."""
        from jarvis_granite.voice.voice_pipeline import SilenceDetector

        detector = SilenceDetector(
            threshold_db=-40.0,
            min_silence_duration_ms=300,
            sample_rate=48000
        )

        assert detector.threshold_db == -40.0
        assert detector.min_silence_duration_ms == 300

    def test_detect_silence_in_silent_frame(self, silent_audio_frame):
        """Should detect silence in silent audio frame."""
        from jarvis_granite.voice.voice_pipeline import SilenceDetector

        detector = SilenceDetector(
            threshold_db=-40.0,
            min_silence_duration_ms=300,
            sample_rate=48000
        )

        is_silent = detector.is_frame_silent(silent_audio_frame)

        assert is_silent is True

    def test_detect_non_silence_in_loud_frame(self, loud_audio_frame):
        """Should detect non-silence in loud audio frame."""
        from jarvis_granite.voice.voice_pipeline import SilenceDetector

        detector = SilenceDetector(
            threshold_db=-40.0,
            min_silence_duration_ms=300,
            sample_rate=48000
        )

        is_silent = detector.is_frame_silent(loud_audio_frame)

        assert is_silent is False

    def test_silence_duration_tracking(self, silent_audio_frame):
        """Should track silence duration."""
        from jarvis_granite.voice.voice_pipeline import SilenceDetector

        detector = SilenceDetector(
            threshold_db=-40.0,
            min_silence_duration_ms=300,
            sample_rate=48000
        )

        # Feed multiple silent frames
        for _ in range(20):  # 20 * 20ms = 400ms
            detector.process_frame(silent_audio_frame)

        assert detector.current_silence_duration_ms >= 300

    def test_end_of_speech_detection(self, silent_audio_frame, loud_audio_frame):
        """Should detect end of speech after sufficient silence."""
        from jarvis_granite.voice.voice_pipeline import SilenceDetector

        detector = SilenceDetector(
            threshold_db=-40.0,
            min_silence_duration_ms=300,
            sample_rate=48000
        )

        # Simulate speech followed by silence
        detector.process_frame(loud_audio_frame)  # Speech

        # Feed silent frames
        end_detected = False
        for _ in range(20):  # 400ms of silence
            result = detector.process_frame(silent_audio_frame)
            if result:
                end_detected = True
                break

        assert end_detected is True

    def test_reset_silence_on_speech(self, silent_audio_frame, loud_audio_frame):
        """Should reset silence counter when speech detected."""
        from jarvis_granite.voice.voice_pipeline import SilenceDetector

        detector = SilenceDetector(
            threshold_db=-40.0,
            min_silence_duration_ms=300,
            sample_rate=48000
        )

        # Feed silent frames
        for _ in range(10):
            detector.process_frame(silent_audio_frame)

        # Feed loud frame
        detector.process_frame(loud_audio_frame)

        assert detector.current_silence_duration_ms == 0


# =============================================================================
# AUDIO BUFFERING
# =============================================================================

class TestAudioBuffering:
    """Tests for audio buffering functionality."""

    def test_audio_buffer_initialization(self, live_config):
        """AudioBuffer should be properly initialized."""
        from jarvis_granite.voice.voice_pipeline import AudioBuffer

        buffer = AudioBuffer(max_duration_ms=5000, sample_rate=48000)

        assert buffer.max_duration_ms == 5000
        assert len(buffer) == 0

    def test_buffer_add_frame(self, sample_audio_bytes):
        """Should add audio frame to buffer."""
        from jarvis_granite.voice.voice_pipeline import AudioBuffer

        buffer = AudioBuffer(max_duration_ms=5000, sample_rate=48000)

        buffer.add_frame(sample_audio_bytes)

        assert len(buffer) == 1

    def test_buffer_get_and_clear(self, sample_audio_bytes):
        """Should get all buffered audio and clear."""
        from jarvis_granite.voice.voice_pipeline import AudioBuffer

        buffer = AudioBuffer(max_duration_ms=5000, sample_rate=48000)

        buffer.add_frame(sample_audio_bytes)
        buffer.add_frame(sample_audio_bytes)

        audio = buffer.get_and_clear()

        assert audio is not None
        assert len(audio) > 0
        assert len(buffer) == 0

    def test_buffer_max_duration_limit(self, sample_audio_bytes):
        """Buffer should respect max duration limit."""
        from jarvis_granite.voice.voice_pipeline import AudioBuffer

        # Very short max duration
        buffer = AudioBuffer(max_duration_ms=100, sample_rate=48000)

        # Add many frames
        for _ in range(100):
            buffer.add_frame(sample_audio_bytes)

        # Should not exceed max duration
        assert buffer.total_duration_ms <= 100

    def test_buffer_is_empty(self):
        """Should correctly report empty state."""
        from jarvis_granite.voice.voice_pipeline import AudioBuffer

        buffer = AudioBuffer(max_duration_ms=5000, sample_rate=48000)

        assert buffer.is_empty() is True

    def test_buffer_has_speech(self, loud_audio_frame, live_config):
        """Should track whether buffer contains speech."""
        from jarvis_granite.voice.voice_pipeline import AudioBuffer

        buffer = AudioBuffer(max_duration_ms=5000, sample_rate=48000)

        buffer.add_frame(loud_audio_frame)

        assert buffer.has_content() is True


# =============================================================================
# STT -> ORCHESTRATOR FLOW
# =============================================================================

class TestSTTToOrchestratorFlow:
    """Tests for STT to Orchestrator flow."""

    @pytest.mark.asyncio
    async def test_transcribe_and_forward_to_orchestrator(
        self, live_config, mock_stt_client, mock_orchestrator, sample_audio_with_data
    ):
        """Should transcribe audio and forward to orchestrator."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)
        pipeline.stt_client = mock_stt_client
        pipeline.orchestrator = mock_orchestrator

        # Process audio through pipeline
        await pipeline.process_audio_input(sample_audio_with_data)

        # Verify STT was called
        mock_stt_client.transcribe.assert_called()

    @pytest.mark.asyncio
    async def test_empty_transcription_not_forwarded(
        self, live_config, mock_stt_client, mock_orchestrator, sample_audio_with_data
    ):
        """Should not forward empty transcriptions."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)
        mock_stt_client.transcribe = AsyncMock(return_value="")
        pipeline.stt_client = mock_stt_client
        pipeline.orchestrator = mock_orchestrator

        await pipeline.process_audio_input(sample_audio_with_data)

        # Orchestrator should not be called for empty transcription
        mock_orchestrator.handle_driver_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_transcription_forwarded_as_driver_query(
        self, live_config, mock_stt_client, mock_orchestrator, sample_audio_with_data
    ):
        """Transcription should be forwarded as driver query."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)
        pipeline.stt_client = mock_stt_client
        pipeline.orchestrator = mock_orchestrator

        await pipeline.process_audio_input(sample_audio_with_data)

        mock_orchestrator.handle_driver_query.assert_called_with("What about my tires?")


# =============================================================================
# ORCHESTRATOR -> TTS FLOW
# =============================================================================

class TestOrchestratorToTTSFlow:
    """Tests for Orchestrator to TTS flow."""

    @pytest.mark.asyncio
    async def test_response_sent_to_tts(
        self, live_config, mock_tts_client, mock_livekit_client
    ):
        """Orchestrator response should be sent to TTS."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)
        pipeline.tts_client = mock_tts_client
        pipeline.livekit_client = mock_livekit_client

        await pipeline.deliver_response("Box this lap.")

        mock_tts_client.synthesize.assert_called()

    @pytest.mark.asyncio
    async def test_tts_audio_sent_to_livekit(
        self, live_config, mock_tts_client, mock_livekit_client, sample_audio_bytes
    ):
        """TTS audio should be sent through LiveKit."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)
        pipeline.tts_client = mock_tts_client
        pipeline.livekit_client = mock_livekit_client

        await pipeline.deliver_response("Box this lap.")

        mock_livekit_client.send_audio.assert_called()

    @pytest.mark.asyncio
    async def test_sentence_splitting_in_delivery(
        self, live_config, mock_tts_client, mock_livekit_client
    ):
        """Should split response into sentences for delivery."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)
        pipeline.tts_client = mock_tts_client
        pipeline.livekit_client = mock_livekit_client

        await pipeline.deliver_response("Box this lap. Fresh tires ready.")

        # TTS should be called multiple times (once per sentence)
        assert mock_tts_client.synthesize.call_count >= 2

    @pytest.mark.asyncio
    async def test_empty_response_not_processed(
        self, live_config, mock_tts_client, mock_livekit_client
    ):
        """Empty response should not be processed."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)
        pipeline.tts_client = mock_tts_client
        pipeline.livekit_client = mock_livekit_client

        await pipeline.deliver_response("")

        mock_tts_client.synthesize.assert_not_called()


# =============================================================================
# FULL PIPELINE FLOW
# =============================================================================

class TestFullPipelineFlow:
    """Tests for complete pipeline flow."""

    @pytest.mark.asyncio
    async def test_full_voice_interaction(
        self, live_config, mock_stt_client, mock_tts_client,
        mock_orchestrator, mock_livekit_client, sample_audio_with_data
    ):
        """Should handle full voice interaction cycle."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)
        pipeline.stt_client = mock_stt_client
        pipeline.tts_client = mock_tts_client
        pipeline.orchestrator = mock_orchestrator
        pipeline.livekit_client = mock_livekit_client

        # Simulate driver speaking
        await pipeline.process_audio_input(sample_audio_with_data)

        # Verify full flow
        mock_stt_client.transcribe.assert_called()
        mock_orchestrator.handle_driver_query.assert_called()
        mock_tts_client.synthesize.assert_called()
        mock_livekit_client.send_audio.assert_called()

    @pytest.mark.asyncio
    async def test_pipeline_start_and_stop(self, live_config, mock_livekit_client):
        """Should start and stop pipeline properly."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)
        pipeline.livekit_client = mock_livekit_client

        await pipeline.start("test_room", "ai_engineer")

        assert pipeline.is_active is True

        await pipeline.stop()

        assert pipeline.is_active is False


# =============================================================================
# INTERRUPT HANDLING
# =============================================================================

class TestInterruptHandling:
    """Tests for interrupt handling in pipeline."""

    @pytest.mark.asyncio
    async def test_interrupt_during_delivery(
        self, live_config, mock_tts_client, mock_livekit_client
    ):
        """Should handle interrupt during response delivery."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)
        pipeline.tts_client = mock_tts_client
        pipeline.livekit_client = mock_livekit_client

        # Request interrupt before delivery starts
        pipeline.request_interrupt()

        # Start delivery - should detect interrupt after first sentence
        await pipeline.deliver_response("This is a long response. With multiple sentences. And more content.")

        # Pipeline should have stopped early due to interrupt
        assert pipeline.was_interrupted is True
        # TTS should have been called at most once (first sentence before interrupt detected)
        assert mock_tts_client.synthesize.call_count <= 1

    def test_pending_interrupt_flag(self, live_config):
        """Should track pending interrupt state."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)

        assert pipeline.has_pending_interrupt is False

        pipeline.request_interrupt()

        assert pipeline.has_pending_interrupt is True

    @pytest.mark.asyncio
    async def test_interrupt_clears_after_handling(
        self, live_config, mock_tts_client, mock_livekit_client
    ):
        """Interrupt flag should clear after being handled."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)
        pipeline.tts_client = mock_tts_client
        pipeline.livekit_client = mock_livekit_client

        pipeline.request_interrupt()

        # Deliver a response (will check for interrupt)
        await pipeline.deliver_response("Test.")

        assert pipeline.has_pending_interrupt is False


# =============================================================================
# AUDIO FRAME STREAMING
# =============================================================================

class TestAudioFrameStreaming:
    """Tests for audio frame streaming."""

    @pytest.mark.asyncio
    async def test_incoming_audio_buffered(
        self, live_config, silent_audio_frame
    ):
        """Incoming audio should be buffered."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)

        pipeline.on_audio_frame_received(silent_audio_frame)

        assert not pipeline.audio_buffer.is_empty()

    @pytest.mark.asyncio
    async def test_buffer_processed_on_silence(
        self, live_config, mock_stt_client, mock_orchestrator,
        loud_audio_frame, silent_audio_frame
    ):
        """Buffer should be processed when silence detected."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)
        pipeline.stt_client = mock_stt_client
        pipeline.orchestrator = mock_orchestrator
        pipeline.is_active = True

        # Add speech frames (using async method to also update silence detector)
        for _ in range(5):
            await pipeline.on_audio_frame_received_async(loud_audio_frame)

        # Add enough silence to trigger processing
        for _ in range(20):  # 400ms of silence
            await pipeline.on_audio_frame_received_async(silent_audio_frame)

        # STT should have been called
        mock_stt_client.transcribe.assert_called()


# =============================================================================
# ERROR HANDLING
# =============================================================================

class TestPipelineErrorHandling:
    """Tests for pipeline error handling."""

    @pytest.mark.asyncio
    async def test_stt_error_handled(self, live_config, mock_stt_client, sample_audio_with_data):
        """Should handle STT errors gracefully."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)
        mock_stt_client.transcribe = AsyncMock(side_effect=Exception("STT Error"))
        pipeline.stt_client = mock_stt_client

        # Should not raise
        await pipeline.process_audio_input(sample_audio_with_data)

        assert pipeline.last_error is not None

    @pytest.mark.asyncio
    async def test_tts_error_handled(
        self, live_config, mock_tts_client, mock_livekit_client
    ):
        """Should handle TTS errors gracefully."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)
        mock_tts_client.synthesize = AsyncMock(side_effect=Exception("TTS Error"))
        pipeline.tts_client = mock_tts_client
        pipeline.livekit_client = mock_livekit_client

        # Should not raise
        await pipeline.deliver_response("Test message")

        assert pipeline.last_error is not None

    @pytest.mark.asyncio
    async def test_livekit_error_handled(
        self, live_config, mock_tts_client, mock_livekit_client, sample_audio_bytes
    ):
        """Should handle LiveKit errors gracefully."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)
        pipeline.tts_client = mock_tts_client
        mock_livekit_client.send_audio = AsyncMock(side_effect=Exception("LiveKit Error"))
        pipeline.livekit_client = mock_livekit_client

        # Should not raise
        await pipeline.deliver_response("Test message")

        assert pipeline.last_error is not None


# =============================================================================
# PIPELINE STATISTICS
# =============================================================================

class TestPipelineStatistics:
    """Tests for pipeline statistics."""

    def test_get_stats(self, live_config):
        """Should return pipeline statistics."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)

        stats = pipeline.get_stats()

        assert isinstance(stats, dict)
        assert "is_active" in stats
        assert "is_processing" in stats
        assert "buffer_frames" in stats
        assert "messages_processed" in stats

    @pytest.mark.asyncio
    async def test_stats_track_processed_messages(
        self, live_config, mock_stt_client, mock_tts_client,
        mock_orchestrator, mock_livekit_client, sample_audio_with_data
    ):
        """Stats should track processed messages."""
        from jarvis_granite.voice.voice_pipeline import VoicePipeline

        pipeline = VoicePipeline(config=live_config)
        pipeline.stt_client = mock_stt_client
        pipeline.tts_client = mock_tts_client
        pipeline.orchestrator = mock_orchestrator
        pipeline.livekit_client = mock_livekit_client

        await pipeline.process_audio_input(sample_audio_with_data)

        stats = pipeline.get_stats()
        assert stats["messages_processed"] >= 1
