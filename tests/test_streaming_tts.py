"""
Tests for Streaming TTS Client.

TDD tests for Watson Text-to-Speech streaming implementation.
Tests the StreamingTTSClient class and its integration with TTSOutputWorker.
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock


class TestStreamingTTSClientInit:
    """Tests for StreamingTTSClient initialization."""

    def test_init_with_required_params(self):
        """Client initializes with required API key and URL."""
        from ai.streaming_tts import StreamingTTSClient

        client = StreamingTTSClient(
            api_key="test-api-key",
            service_url="https://api.us-south.text-to-speech.watson.cloud.ibm.com"
        )

        assert client.api_key == "test-api-key"
        assert client.service_url == "https://api.us-south.text-to-speech.watson.cloud.ibm.com"

    def test_init_with_custom_voice(self):
        """Client accepts custom voice."""
        from ai.streaming_tts import StreamingTTSClient

        client = StreamingTTSClient(
            api_key="test-api-key",
            service_url="https://api.test.com",
            voice="en-US_AllisonV3Voice"
        )

        assert client.voice == "en-US_AllisonV3Voice"

    def test_init_default_voice(self):
        """Client uses en-GB_JamesV3Voice by default (British male race engineer)."""
        from ai.streaming_tts import StreamingTTSClient

        client = StreamingTTSClient(
            api_key="test-api-key",
            service_url="https://api.test.com"
        )

        assert client.voice == "en-GB_JamesV3Voice"

    def test_init_with_custom_chunk_size(self):
        """Client accepts custom chunk size for streaming."""
        from ai.streaming_tts import StreamingTTSClient

        client = StreamingTTSClient(
            api_key="test-api-key",
            service_url="https://api.test.com",
            chunk_size=8192
        )

        assert client.chunk_size == 8192

    def test_init_default_chunk_size(self):
        """Client uses 4096 byte chunks by default."""
        from ai.streaming_tts import StreamingTTSClient

        client = StreamingTTSClient(
            api_key="test-api-key",
            service_url="https://api.test.com"
        )

        assert client.chunk_size == 4096


class TestStreamingTTSClientCallbacks:
    """Tests for callback registration."""

    def test_register_on_audio_chunk_callback(self):
        """Client registers callback for audio chunks."""
        from ai.streaming_tts import StreamingTTSClient

        client = StreamingTTSClient(
            api_key="test-api-key",
            service_url="https://api.test.com"
        )

        callback = Mock()
        client.on_audio_chunk(callback)

        assert client._on_audio_chunk_callback == callback

    def test_register_on_synthesis_start_callback(self):
        """Client registers callback for synthesis start."""
        from ai.streaming_tts import StreamingTTSClient

        client = StreamingTTSClient(
            api_key="test-api-key",
            service_url="https://api.test.com"
        )

        callback = Mock()
        client.on_synthesis_start(callback)

        assert client._on_synthesis_start_callback == callback

    def test_register_on_synthesis_complete_callback(self):
        """Client registers callback for synthesis complete."""
        from ai.streaming_tts import StreamingTTSClient

        client = StreamingTTSClient(
            api_key="test-api-key",
            service_url="https://api.test.com"
        )

        callback = Mock()
        client.on_synthesis_complete(callback)

        assert client._on_synthesis_complete_callback == callback

    def test_register_on_error_callback(self):
        """Client registers callback for errors."""
        from ai.streaming_tts import StreamingTTSClient

        client = StreamingTTSClient(
            api_key="test-api-key",
            service_url="https://api.test.com"
        )

        callback = Mock()
        client.on_error(callback)

        assert client._on_error_callback == callback


class TestStreamingTTSClientSynthesis:
    """Tests for streaming synthesis."""

    @pytest.mark.asyncio
    async def test_synthesize_stream_calls_start_callback(self):
        """Synthesis calls on_synthesis_start callback."""
        from ai.streaming_tts import StreamingTTSClient

        client = StreamingTTSClient(
            api_key="test-api-key",
            service_url="https://api.test.com"
        )

        start_callback = Mock()
        client.on_synthesis_start(start_callback)

        # Create proper async context manager mocks
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content.iter_chunked = lambda size: AsyncIterator([b'audio_chunk'])

        mock_post_cm = AsyncMock()
        mock_post_cm.__aenter__.return_value = mock_response

        mock_session = MagicMock()
        mock_session.post.return_value = mock_post_cm

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session

        with patch('aiohttp.ClientSession', return_value=mock_session_cm):
            await client.synthesize_stream("Hello")

        start_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_stream_calls_chunk_callback(self):
        """Synthesis calls on_audio_chunk for each chunk received."""
        from ai.streaming_tts import StreamingTTSClient

        client = StreamingTTSClient(
            api_key="test-api-key",
            service_url="https://api.test.com"
        )

        chunks_received = []
        client.on_audio_chunk(lambda chunk: chunks_received.append(chunk))

        # Create proper async context manager mocks
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content.iter_chunked = lambda size: AsyncIterator([b'chunk1', b'chunk2', b'chunk3'])

        mock_post_cm = AsyncMock()
        mock_post_cm.__aenter__.return_value = mock_response

        mock_session = MagicMock()
        mock_session.post.return_value = mock_post_cm

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session

        with patch('aiohttp.ClientSession', return_value=mock_session_cm):
            await client.synthesize_stream("Hello")

        assert len(chunks_received) == 3

    @pytest.mark.asyncio
    async def test_synthesize_stream_calls_complete_callback(self):
        """Synthesis calls on_synthesis_complete when done."""
        from ai.streaming_tts import StreamingTTSClient

        client = StreamingTTSClient(
            api_key="test-api-key",
            service_url="https://api.test.com"
        )

        complete_callback = Mock()
        client.on_synthesis_complete(complete_callback)

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.content.iter_chunked = lambda size: AsyncIterator([b'audio'])

        mock_post_cm = AsyncMock()
        mock_post_cm.__aenter__.return_value = mock_response

        mock_session = MagicMock()
        mock_session.post.return_value = mock_post_cm

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session

        with patch('aiohttp.ClientSession', return_value=mock_session_cm):
            await client.synthesize_stream("Hello")

        complete_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_synthesize_stream_calls_error_callback_on_failure(self):
        """Synthesis calls on_error callback when API fails."""
        from ai.streaming_tts import StreamingTTSClient

        client = StreamingTTSClient(
            api_key="test-api-key",
            service_url="https://api.test.com"
        )

        error_callback = Mock()
        client.on_error(error_callback)

        mock_response = MagicMock()
        mock_response.status = 401
        mock_response.text = AsyncMock(return_value="Unauthorized")

        mock_post_cm = AsyncMock()
        mock_post_cm.__aenter__.return_value = mock_response

        mock_session = MagicMock()
        mock_session.post.return_value = mock_post_cm

        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session

        with patch('aiohttp.ClientSession', return_value=mock_session_cm):
            await client.synthesize_stream("Hello")

        error_callback.assert_called_once()


class TestStreamingTTSClientAudioFormat:
    """Tests for audio format handling."""

    def test_get_synthesis_url(self):
        """URL is correctly constructed."""
        from ai.streaming_tts import StreamingTTSClient

        client = StreamingTTSClient(
            api_key="test-api-key",
            service_url="https://api.us-south.text-to-speech.watson.cloud.ibm.com"
        )

        url = client._get_synthesis_url()

        assert "/v1/synthesize" in url

    def test_audio_format_is_wav(self):
        """Client requests WAV format for streaming compatibility."""
        from ai.streaming_tts import StreamingTTSClient

        client = StreamingTTSClient(
            api_key="test-api-key",
            service_url="https://api.test.com"
        )

        headers = client._get_headers()

        assert headers.get("Accept") == "audio/wav"

    def test_request_includes_voice_parameter(self):
        """Request includes the voice parameter."""
        from ai.streaming_tts import StreamingTTSClient

        client = StreamingTTSClient(
            api_key="test-api-key",
            service_url="https://api.test.com",
            voice="en-GB_JamesV3Voice"
        )

        params = client._get_params()

        assert params.get("voice") == "en-GB_JamesV3Voice"


class TestStreamingTTSClientChainedCallbacks:
    """Tests for method chaining on callbacks."""

    def test_callbacks_return_self_for_chaining(self):
        """Callback methods return self for method chaining."""
        from ai.streaming_tts import StreamingTTSClient

        client = StreamingTTSClient(
            api_key="test-api-key",
            service_url="https://api.test.com"
        )

        result = client.on_audio_chunk(Mock())
        assert result is client

        result = client.on_synthesis_start(Mock())
        assert result is client

        result = client.on_synthesis_complete(Mock())
        assert result is client

        result = client.on_error(Mock())
        assert result is client


class TestTTSOutputWorkerStreamingMode:
    """Tests for TTSOutputWorker with streaming mode."""

    def test_tts_output_worker_accepts_streaming_mode_flag(self):
        """TTSOutputWorker accepts use_streaming parameter."""
        from ai.tts_output import TTSOutputWorker

        worker = TTSOutputWorker(
            watson_api_key="test-key",
            watson_url="https://api.test.com",
            use_streaming=True
        )

        assert worker.use_streaming is True

    def test_tts_output_worker_streaming_mode_default_false(self):
        """TTSOutputWorker defaults to batch mode for backward compatibility."""
        from ai.tts_output import TTSOutputWorker

        worker = TTSOutputWorker(
            watson_api_key="test-key",
            watson_url="https://api.test.com"
        )

        assert worker.use_streaming is False

    def test_tts_output_worker_has_streaming_client_attribute(self):
        """TTSOutputWorker has streaming client attribute when enabled."""
        from ai.tts_output import TTSOutputWorker

        worker = TTSOutputWorker(
            watson_api_key="test-key",
            watson_url="https://api.test.com",
            use_streaming=True
        )

        assert hasattr(worker, '_streaming_tts_client')

    def test_tts_output_worker_has_synthesize_and_play_streaming_method(self):
        """TTSOutputWorker has method for streaming synthesis and playback."""
        from ai.tts_output import TTSOutputWorker

        worker = TTSOutputWorker(
            watson_api_key="test-key",
            watson_url="https://api.test.com",
            use_streaming=True
        )

        assert hasattr(worker, '_synthesize_and_play_streaming')
        assert callable(worker._synthesize_and_play_streaming)


class TestStreamingAudioPlayer:
    """Tests for the streaming audio player component."""

    def test_streaming_audio_player_init(self):
        """StreamingAudioPlayer initializes correctly."""
        from ai.streaming_tts import StreamingAudioPlayer

        player = StreamingAudioPlayer()

        assert player is not None

    def test_streaming_audio_player_has_play_chunk_method(self):
        """StreamingAudioPlayer has method to play audio chunks."""
        from ai.streaming_tts import StreamingAudioPlayer

        player = StreamingAudioPlayer()

        assert hasattr(player, 'play_chunk')
        assert callable(player.play_chunk)

    def test_streaming_audio_player_has_start_method(self):
        """StreamingAudioPlayer has method to start playback."""
        from ai.streaming_tts import StreamingAudioPlayer

        player = StreamingAudioPlayer()

        assert hasattr(player, 'start')
        assert callable(player.start)

    def test_streaming_audio_player_has_stop_method(self):
        """StreamingAudioPlayer has method to stop playback."""
        from ai.streaming_tts import StreamingAudioPlayer

        player = StreamingAudioPlayer()

        assert hasattr(player, 'stop')
        assert callable(player.stop)

    def test_streaming_audio_player_has_is_playing_property(self):
        """StreamingAudioPlayer has is_playing property."""
        from ai.streaming_tts import StreamingAudioPlayer

        player = StreamingAudioPlayer()

        assert hasattr(player, 'is_playing')
        assert player.is_playing is False


# Helper class for async iteration in tests
class AsyncIterator:
    """Helper to create async iterators for testing."""

    def __init__(self, items):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item
