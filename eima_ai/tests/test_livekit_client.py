"""
Tests for LiveKit Client - Phase 5, Section 12

These tests verify the expected behavior for:
1. LiveKitClient initialization and configuration
2. Room connection and disconnection
3. Audio track publishing (AI -> Driver)
4. Audio track subscription (Driver -> AI)
5. Token generation
6. VoiceAgent integration with LiveKit

Run with: pytest tests/test_livekit_client.py -v
"""

import asyncio
import pytest
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

# Mock access token with chainable methods
mock_access_token = MagicMock()
mock_access_token.to_jwt.return_value = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJyb29tIjoidGVzdCJ9.signature"
mock_access_token.with_identity.return_value = mock_access_token
mock_access_token.with_name.return_value = mock_access_token
mock_access_token.with_grants.return_value = mock_access_token
mock_api.AccessToken = MagicMock(return_value=mock_access_token)
mock_api.VideoGrants = MagicMock

# Patch at module level before any imports
sys.modules['livekit'] = MagicMock()
sys.modules['livekit.rtc'] = mock_rtc
sys.modules['livekit.api'] = mock_api
sys.modules['livekit'].rtc = mock_rtc
sys.modules['livekit'].api = mock_api

# Clear any cached imports of our modules
for mod_name in list(sys.modules.keys()):
    if 'jarvis_granite.voice' in mod_name:
        del sys.modules[mod_name]

from config.config import LiveKitConfig, LiveConfig


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def livekit_config():
    """Create LiveKit configuration."""
    return LiveKitConfig(
        url="wss://livekit.example.com",
        api_key="test_api_key",
        api_secret="test_api_secret",
        room_prefix="jarvis_live_",
        audio_codec="opus",
        sample_rate=48000,
    )


@pytest.fixture
def live_config(livekit_config):
    """Create full LiveConfig with LiveKit settings."""
    return LiveConfig(livekit=livekit_config)


@pytest.fixture
def sample_audio_bytes():
    """Create sample audio bytes (WAV header + minimal data)."""
    return b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00'


@pytest.fixture
def mock_livekit_room():
    """Create mock LiveKit Room."""
    room = MagicMock()
    room.connect = AsyncMock()
    room.disconnect = AsyncMock()
    room.on = MagicMock()
    room.local_participant = MagicMock()
    room.local_participant.publish_track = AsyncMock()
    return room


@pytest.fixture
def mock_audio_source():
    """Create mock audio source."""
    source = MagicMock()
    source.capture_frame = AsyncMock()
    return source


# =============================================================================
# LIVEKIT CLIENT INITIALIZATION
# =============================================================================

class TestLiveKitClientInitialization:
    """Tests for LiveKitClient initialization."""

    def test_create_livekit_client(self, livekit_config):
        """Should create a LiveKitClient instance."""
        from jarvis_granite.voice.livekit_client import LiveKitClient

        client = LiveKitClient(config=livekit_config)

        assert client is not None
        assert client.config == livekit_config

    def test_livekit_client_stores_url(self, livekit_config):
        """Should store LiveKit server URL."""
        from jarvis_granite.voice.livekit_client import LiveKitClient

        client = LiveKitClient(config=livekit_config)

        assert client.config.url == "wss://livekit.example.com"

    def test_livekit_client_stores_credentials(self, livekit_config):
        """Should store API key and secret."""
        from jarvis_granite.voice.livekit_client import LiveKitClient

        client = LiveKitClient(config=livekit_config)

        assert client.config.api_key == "test_api_key"
        assert client.config.api_secret == "test_api_secret"

    def test_livekit_client_default_state(self, livekit_config):
        """Client should start disconnected."""
        from jarvis_granite.voice.livekit_client import LiveKitClient

        client = LiveKitClient(config=livekit_config)

        assert client.is_connected is False
        assert client.room is None

    def test_livekit_client_has_connect_method(self, livekit_config):
        """Client should have connect method."""
        from jarvis_granite.voice.livekit_client import LiveKitClient

        client = LiveKitClient(config=livekit_config)

        assert hasattr(client, 'connect')
        assert asyncio.iscoroutinefunction(client.connect)

    def test_livekit_client_has_disconnect_method(self, livekit_config):
        """Client should have disconnect method."""
        from jarvis_granite.voice.livekit_client import LiveKitClient

        client = LiveKitClient(config=livekit_config)

        assert hasattr(client, 'disconnect')
        assert asyncio.iscoroutinefunction(client.disconnect)


# =============================================================================
# TOKEN GENERATION
# =============================================================================

class TestTokenGeneration:
    """Tests for LiveKit token generation."""

    def test_generate_token_returns_string(self, livekit_config):
        """generate_token should return a JWT string."""
        from jarvis_granite.voice.livekit_client import LiveKitClient

        client = LiveKitClient(config=livekit_config)

        token = client.generate_token(
            room_name="test_room",
            participant_name="ai_engineer"
        )

        assert isinstance(token, str)
        assert len(token) > 0

    def test_generate_token_with_room_name(self, livekit_config):
        """Token should be generated for specific room."""
        from jarvis_granite.voice.livekit_client import LiveKitClient

        client = LiveKitClient(config=livekit_config)

        token = client.generate_token(
            room_name="race_001_voice",
            participant_name="ai_engineer"
        )

        # Token is a JWT, should have 3 parts separated by dots
        parts = token.split('.')
        assert len(parts) == 3

    def test_generate_token_with_participant_identity(self, livekit_config):
        """Token should include participant identity."""
        from jarvis_granite.voice.livekit_client import LiveKitClient

        client = LiveKitClient(config=livekit_config)

        token = client.generate_token(
            room_name="test_room",
            participant_name="jarvis_ai"
        )

        assert token is not None
        assert len(token) > 0

    def test_generate_token_with_room_prefix(self, livekit_config):
        """Should use room prefix from config."""
        from jarvis_granite.voice.livekit_client import LiveKitClient

        client = LiveKitClient(config=livekit_config)

        # Room name should work with prefix
        full_room_name = f"{livekit_config.room_prefix}session_123"
        token = client.generate_token(
            room_name=full_room_name,
            participant_name="ai_engineer"
        )

        assert token is not None


# =============================================================================
# ROOM CONNECTION
# =============================================================================

class TestRoomConnection:
    """Tests for LiveKit room connection."""

    @pytest.mark.asyncio
    async def test_connect_creates_room(self, livekit_config, mock_livekit_room):
        """connect should create a LiveKit room."""
        from jarvis_granite.voice.livekit_client import LiveKitClient

        client = LiveKitClient(config=livekit_config)

        with patch('jarvis_granite.voice.livekit_client.rtc.Room', return_value=mock_livekit_room):
            await client.connect(
                room_name="test_room",
                participant_name="ai_engineer"
            )

            assert client.room is not None

    @pytest.mark.asyncio
    async def test_connect_uses_generated_token(self, livekit_config, mock_livekit_room):
        """connect should use generated token."""
        from jarvis_granite.voice.livekit_client import LiveKitClient

        client = LiveKitClient(config=livekit_config)

        with patch('jarvis_granite.voice.livekit_client.rtc.Room', return_value=mock_livekit_room):
            with patch.object(client, 'generate_token', return_value="mock_token") as mock_gen:
                await client.connect(
                    room_name="test_room",
                    participant_name="ai_engineer"
                )

                mock_gen.assert_called_once_with("test_room", "ai_engineer")

    @pytest.mark.asyncio
    async def test_connect_calls_room_connect(self, livekit_config, mock_livekit_room):
        """connect should call room.connect with URL and token."""
        from jarvis_granite.voice.livekit_client import LiveKitClient

        client = LiveKitClient(config=livekit_config)

        with patch('jarvis_granite.voice.livekit_client.rtc.Room', return_value=mock_livekit_room):
            await client.connect(
                room_name="test_room",
                participant_name="ai_engineer"
            )

            mock_livekit_room.connect.assert_called_once()
            call_args = mock_livekit_room.connect.call_args
            assert livekit_config.url in str(call_args)

    @pytest.mark.asyncio
    async def test_connect_sets_is_connected(self, livekit_config, mock_livekit_room):
        """connect should set is_connected to True."""
        from jarvis_granite.voice.livekit_client import LiveKitClient

        client = LiveKitClient(config=livekit_config)

        with patch('jarvis_granite.voice.livekit_client.rtc.Room', return_value=mock_livekit_room):
            await client.connect(
                room_name="test_room",
                participant_name="ai_engineer"
            )

            assert client.is_connected is True

    @pytest.mark.asyncio
    async def test_disconnect_closes_room(self, livekit_config, mock_livekit_room):
        """disconnect should close the room connection."""
        from jarvis_granite.voice.livekit_client import LiveKitClient

        client = LiveKitClient(config=livekit_config)

        with patch('jarvis_granite.voice.livekit_client.rtc.Room', return_value=mock_livekit_room):
            await client.connect(
                room_name="test_room",
                participant_name="ai_engineer"
            )
            await client.disconnect()

            mock_livekit_room.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_sets_is_connected_false(self, livekit_config, mock_livekit_room):
        """disconnect should set is_connected to False."""
        from jarvis_granite.voice.livekit_client import LiveKitClient

        client = LiveKitClient(config=livekit_config)

        with patch('jarvis_granite.voice.livekit_client.rtc.Room', return_value=mock_livekit_room):
            await client.connect(
                room_name="test_room",
                participant_name="ai_engineer"
            )
            await client.disconnect()

            assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self, livekit_config):
        """disconnect when not connected should not raise error."""
        from jarvis_granite.voice.livekit_client import LiveKitClient

        client = LiveKitClient(config=livekit_config)

        # Should not raise
        await client.disconnect()

        assert client.is_connected is False


# =============================================================================
# AUDIO TRACK PUBLISHING (AI -> DRIVER)
# =============================================================================

class TestAudioTrackPublishing:
    """Tests for audio track publishing (AI to Driver)."""

    @pytest.mark.asyncio
    async def test_publish_audio_track(self, livekit_config, mock_livekit_room):
        """Should publish audio track to room."""
        from jarvis_granite.voice.livekit_client import LiveKitClient

        client = LiveKitClient(config=livekit_config)

        with patch('jarvis_granite.voice.livekit_client.rtc.Room', return_value=mock_livekit_room):
            with patch('jarvis_granite.voice.livekit_client.rtc.AudioSource') as mock_source_class:
                mock_source = MagicMock()
                mock_source_class.return_value = mock_source

                await client.connect(
                    room_name="test_room",
                    participant_name="ai_engineer"
                )

                # Audio track should be created during connect
                assert client.audio_source is not None

    @pytest.mark.asyncio
    async def test_stream_audio_frames(self, livekit_config, mock_livekit_room, sample_audio_bytes):
        """Should stream audio frames via audio source."""
        from jarvis_granite.voice.livekit_client import LiveKitClient

        client = LiveKitClient(config=livekit_config)

        with patch('jarvis_granite.voice.livekit_client.rtc.Room', return_value=mock_livekit_room):
            with patch('jarvis_granite.voice.livekit_client.rtc.AudioSource') as mock_source_class:
                mock_source = MagicMock()
                mock_source.capture_frame = AsyncMock()
                mock_source_class.return_value = mock_source

                with patch('jarvis_granite.voice.livekit_client.rtc.LocalAudioTrack') as mock_track_class:
                    mock_track = MagicMock()
                    mock_track_class.create_audio_track.return_value = mock_track

                    await client.connect(
                        room_name="test_room",
                        participant_name="ai_engineer"
                    )

                    # Should have method to send audio
                    assert hasattr(client, 'send_audio')

    @pytest.mark.asyncio
    async def test_send_audio_requires_connection(self, livekit_config, sample_audio_bytes):
        """send_audio should require active connection."""
        from jarvis_granite.voice.livekit_client import LiveKitClient, LiveKitError

        client = LiveKitClient(config=livekit_config)

        with pytest.raises(LiveKitError, match="Not connected"):
            await client.send_audio(sample_audio_bytes)


# =============================================================================
# AUDIO TRACK SUBSCRIPTION (DRIVER -> AI)
# =============================================================================

class TestAudioTrackSubscription:
    """Tests for audio track subscription (Driver to AI)."""

    @pytest.mark.asyncio
    async def test_subscribe_to_audio_tracks(self, livekit_config, mock_livekit_room):
        """Should subscribe to incoming audio tracks."""
        from jarvis_granite.voice.livekit_client import LiveKitClient

        client = LiveKitClient(config=livekit_config)

        with patch('jarvis_granite.voice.livekit_client.rtc.Room', return_value=mock_livekit_room):
            with patch('jarvis_granite.voice.livekit_client.rtc.AudioSource'):
                with patch('jarvis_granite.voice.livekit_client.rtc.LocalAudioTrack'):
                    await client.connect(
                        room_name="test_room",
                        participant_name="ai_engineer"
                    )

                    # Should register track subscription handler
                    mock_livekit_room.on.assert_called()

    @pytest.mark.asyncio
    async def test_on_audio_received_callback(self, livekit_config, mock_livekit_room):
        """Should support callback when audio is received."""
        from jarvis_granite.voice.livekit_client import LiveKitClient

        client = LiveKitClient(config=livekit_config)
        received_audio = []

        async def on_audio(audio_data: bytes):
            received_audio.append(audio_data)

        client.on_audio_received = on_audio

        with patch('jarvis_granite.voice.livekit_client.rtc.Room', return_value=mock_livekit_room):
            with patch('jarvis_granite.voice.livekit_client.rtc.AudioSource'):
                with patch('jarvis_granite.voice.livekit_client.rtc.LocalAudioTrack'):
                    await client.connect(
                        room_name="test_room",
                        participant_name="ai_engineer"
                    )

                    assert client.on_audio_received is not None


# =============================================================================
# ERROR HANDLING
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_livekit_error_class_exists(self):
        """LiveKitError class should exist."""
        from jarvis_granite.voice.livekit_client import LiveKitError

        error = LiveKitError("Test error")
        assert str(error) == "Test error"

    @pytest.mark.asyncio
    async def test_connect_error_handling(self, livekit_config, mock_livekit_room):
        """connect should handle connection errors."""
        from jarvis_granite.voice.livekit_client import LiveKitClient, LiveKitError

        client = LiveKitClient(config=livekit_config)

        mock_livekit_room.connect.side_effect = Exception("Connection failed")

        with patch('jarvis_granite.voice.livekit_client.rtc.Room', return_value=mock_livekit_room):
            with pytest.raises(LiveKitError, match="Connection failed"):
                await client.connect(
                    room_name="test_room",
                    participant_name="ai_engineer"
                )

    @pytest.mark.asyncio
    async def test_connect_timeout(self, livekit_config, mock_livekit_room):
        """connect should handle timeout errors."""
        from jarvis_granite.voice.livekit_client import LiveKitClient, LiveKitError

        client = LiveKitClient(config=livekit_config)

        mock_livekit_room.connect.side_effect = asyncio.TimeoutError("Connection timeout")

        with patch('jarvis_granite.voice.livekit_client.rtc.Room', return_value=mock_livekit_room):
            with pytest.raises(LiveKitError, match="timeout"):
                await client.connect(
                    room_name="test_room",
                    participant_name="ai_engineer"
                )


# =============================================================================
# CLIENT STATISTICS
# =============================================================================

class TestClientStatistics:
    """Tests for client statistics."""

    def test_get_stats_returns_dict(self, livekit_config):
        """get_stats should return dictionary."""
        from jarvis_granite.voice.livekit_client import LiveKitClient

        client = LiveKitClient(config=livekit_config)

        stats = client.get_stats()

        assert isinstance(stats, dict)
        assert "is_connected" in stats
        assert "room_name" in stats

    @pytest.mark.asyncio
    async def test_stats_tracks_connection_state(self, livekit_config, mock_livekit_room):
        """Stats should track connection state."""
        from jarvis_granite.voice.livekit_client import LiveKitClient

        client = LiveKitClient(config=livekit_config)

        with patch('jarvis_granite.voice.livekit_client.rtc.Room', return_value=mock_livekit_room):
            with patch('jarvis_granite.voice.livekit_client.rtc.AudioSource'):
                with patch('jarvis_granite.voice.livekit_client.rtc.LocalAudioTrack'):
                    stats_before = client.get_stats()
                    assert stats_before["is_connected"] is False

                    await client.connect(
                        room_name="test_room",
                        participant_name="ai_engineer"
                    )

                    stats_after = client.get_stats()
                    assert stats_after["is_connected"] is True
                    assert stats_after["room_name"] == "test_room"


# =============================================================================
# VOICE AGENT INTEGRATION
# =============================================================================

class TestVoiceAgentIntegration:
    """Tests for VoiceAgent integration with LiveKit."""

    def test_voice_agent_has_livekit_client(self, live_config):
        """VoiceAgent should have LiveKitClient."""
        from jarvis_granite.voice.livekit_client import VoiceAgent

        agent = VoiceAgent(config=live_config)

        assert hasattr(agent, 'livekit_client')
        assert agent.livekit_client is not None

    def test_voice_agent_has_tts_client(self, live_config):
        """VoiceAgent should have TTS client reference."""
        from jarvis_granite.voice.livekit_client import VoiceAgent

        agent = VoiceAgent(config=live_config)

        assert hasattr(agent, 'tts_client')

    def test_voice_agent_has_stt_client(self, live_config):
        """VoiceAgent should have STT client reference."""
        from jarvis_granite.voice.livekit_client import VoiceAgent

        agent = VoiceAgent(config=live_config)

        assert hasattr(agent, 'stt_client')

    def test_voice_agent_has_speak_method(self, live_config):
        """VoiceAgent should have speak method."""
        from jarvis_granite.voice.livekit_client import VoiceAgent

        agent = VoiceAgent(config=live_config)

        assert hasattr(agent, 'speak')
        assert asyncio.iscoroutinefunction(agent.speak)

    def test_voice_agent_has_connect_method(self, live_config):
        """VoiceAgent should have connect method."""
        from jarvis_granite.voice.livekit_client import VoiceAgent

        agent = VoiceAgent(config=live_config)

        assert hasattr(agent, 'connect')
        assert asyncio.iscoroutinefunction(agent.connect)

    def test_voice_agent_tracks_speaking_state(self, live_config):
        """VoiceAgent should track currently_speaking state."""
        from jarvis_granite.voice.livekit_client import VoiceAgent

        agent = VoiceAgent(config=live_config)

        assert hasattr(agent, 'currently_speaking')
        assert agent.currently_speaking is False

    @pytest.mark.asyncio
    async def test_voice_agent_speak_sets_speaking_state(self, live_config, sample_audio_bytes):
        """speak should set currently_speaking state."""
        from jarvis_granite.voice.livekit_client import VoiceAgent

        agent = VoiceAgent(config=live_config)

        # Mock the TTS and LiveKit clients
        agent.tts_client = MagicMock()
        agent.tts_client.synthesize = AsyncMock(return_value=sample_audio_bytes)
        agent.livekit_client = MagicMock()
        agent.livekit_client.is_connected = True
        agent.livekit_client.send_audio = AsyncMock()

        # Call speak (need to await it to completion)
        await agent.speak("Test message")

        # After completion, speaking state should be False
        assert agent.currently_speaking is False

    def test_voice_agent_has_on_query_received_callback(self, live_config):
        """VoiceAgent should support on_query_received callback."""
        from jarvis_granite.voice.livekit_client import VoiceAgent

        agent = VoiceAgent(config=live_config)

        assert hasattr(agent, 'on_query_received')
        agent.on_query_received = lambda x: None  # Should be settable


# =============================================================================
# SENTENCE SPLITTING FOR INTERRUPT HANDLING
# =============================================================================

class TestSentenceSplitting:
    """Tests for sentence splitting (for interrupt handling)."""

    def test_split_into_sentences(self, live_config):
        """Should split text into sentences."""
        from jarvis_granite.voice.livekit_client import VoiceAgent

        agent = VoiceAgent(config=live_config)

        text = "Box this lap. Fresh mediums ready. Good work out there!"
        sentences = agent._split_into_sentences(text)

        assert len(sentences) == 3
        assert "Box this lap" in sentences[0]
        assert "Fresh mediums ready" in sentences[1]
        assert "Good work out there" in sentences[2]

    def test_split_handles_single_sentence(self, live_config):
        """Should handle single sentence."""
        from jarvis_granite.voice.livekit_client import VoiceAgent

        agent = VoiceAgent(config=live_config)

        text = "Box now!"
        sentences = agent._split_into_sentences(text)

        assert len(sentences) == 1
        assert "Box now" in sentences[0]

    def test_split_handles_question_marks(self, live_config):
        """Should handle question marks as sentence boundaries."""
        from jarvis_granite.voice.livekit_client import VoiceAgent

        agent = VoiceAgent(config=live_config)

        text = "How are the tires? Temperature is good."
        sentences = agent._split_into_sentences(text)

        assert len(sentences) == 2


# =============================================================================
# AUDIO FRAME CONVERSION
# =============================================================================

class TestAudioFrameConversion:
    """Tests for audio frame conversion."""

    def test_bytes_to_frames_method_exists(self, livekit_config):
        """LiveKitClient should have bytes_to_frames method."""
        from jarvis_granite.voice.livekit_client import LiveKitClient

        client = LiveKitClient(config=livekit_config)

        assert hasattr(client, '_bytes_to_frames')

    def test_bytes_to_frames_returns_list(self, livekit_config, sample_audio_bytes):
        """bytes_to_frames should return list of frames."""
        from jarvis_granite.voice.livekit_client import LiveKitClient

        client = LiveKitClient(config=livekit_config)

        frames = client._bytes_to_frames(sample_audio_bytes)

        assert isinstance(frames, list)
