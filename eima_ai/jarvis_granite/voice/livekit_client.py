"""
LiveKit Client for Jarvis-Granite Live Telemetry.

Provides WebRTC transport using LiveKit for real-time audio streaming
between the AI race engineer and the driver.

Phase 5, Section 12: LiveKit Integration
- Room connection and management
- Audio track publishing (AI -> Driver)
- Audio track subscription (Driver -> AI)
- Token generation for authentication
- VoiceAgent integration with Watson TTS/STT

Example:
    client = LiveKitClient(config=livekit_config)
    await client.connect(room_name="race_001", participant_name="ai_engineer")
    await client.send_audio(audio_bytes)
    await client.disconnect()
"""

import asyncio
import logging
import re
import struct
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Awaitable

# Try to import livekit, provide stubs for testing if not available
try:
    from livekit import rtc, api
    LIVEKIT_AVAILABLE = True
except ImportError:
    # When livekit is not installed, we still need to be able to
    # define the class. The actual modules will be mocked in tests.
    import sys
    if 'livekit' in sys.modules:
        # Mocked in tests
        rtc = sys.modules.get('livekit.rtc') or sys.modules['livekit'].rtc
        api = sys.modules.get('livekit.api') or sys.modules['livekit'].api
        LIVEKIT_AVAILABLE = True
    else:
        rtc = None  # type: ignore
        api = None  # type: ignore
        LIVEKIT_AVAILABLE = False

from config.config import LiveKitConfig, LiveConfig

logger = logging.getLogger(__name__)


class LiveKitError(Exception):
    """Exception raised for LiveKit-related errors."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.cause = cause


@dataclass
class AudioFrame:
    """Represents an audio frame for streaming."""
    data: bytes
    sample_rate: int
    num_channels: int
    samples_per_channel: int


class LiveKitClient:
    """
    Client for LiveKit WebRTC transport.

    Handles room connection, audio track publishing/subscription,
    and token generation for the real-time voice pipeline.

    Features:
    - WebRTC room connection via LiveKit
    - Audio track publishing for AI voice output
    - Audio track subscription for driver voice input
    - JWT token generation for authentication
    - Connection state management

    Attributes:
        config: LiveKit configuration
        room: Connected LiveKit room (None if not connected)
        is_connected: Whether currently connected to a room
        audio_source: Audio source for publishing frames
        on_audio_received: Callback for received audio

    Example:
        client = LiveKitClient(config=livekit_config)

        # Connect to room
        await client.connect(
            room_name="race_001_voice",
            participant_name="ai_engineer"
        )

        # Send audio
        await client.send_audio(audio_bytes)

        # Disconnect
        await client.disconnect()
    """

    # Audio configuration
    DEFAULT_SAMPLE_RATE = 48000
    DEFAULT_NUM_CHANNELS = 1
    FRAME_DURATION_MS = 20  # 20ms frames for low latency

    def __init__(self, config: LiveKitConfig):
        """
        Initialize LiveKit client.

        Args:
            config: LiveKit configuration with URL, API key, and secret
        """
        self.config = config
        self.room: Optional[Any] = None  # rtc.Room when connected
        self.audio_source: Optional[Any] = None  # rtc.AudioSource
        self.audio_track: Optional[Any] = None  # rtc.LocalAudioTrack
        self._is_connected: bool = False
        self._room_name: Optional[str] = None

        # Callback for received audio
        self.on_audio_received: Optional[Callable[[bytes], Awaitable[None]]] = None

        # Statistics
        self._frames_sent: int = 0
        self._frames_received: int = 0

        logger.info(
            f"Initialized LiveKitClient with URL={config.url}, "
            f"sample_rate={config.sample_rate}"
        )

    @property
    def is_connected(self) -> bool:
        """Return whether client is connected to a room."""
        return self._is_connected

    def generate_token(self, room_name: str, participant_name: str) -> str:
        """
        Generate a LiveKit access token for room connection.

        Args:
            room_name: Name of the room to join
            participant_name: Identity of the participant

        Returns:
            JWT token string for authentication

        Example:
            token = client.generate_token("race_001", "ai_engineer")
        """
        token = api.AccessToken(
            api_key=self.config.api_key,
            api_secret=self.config.api_secret,
        )

        token.with_identity(participant_name)
        token.with_name(participant_name)

        # Grant room join permission
        token.with_grants(api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
        ))

        return token.to_jwt()

    async def connect(
        self,
        room_name: str,
        participant_name: str,
    ) -> None:
        """
        Connect to a LiveKit room.

        Creates a room connection, sets up audio publishing,
        and subscribes to incoming audio tracks.

        Args:
            room_name: Name of the room to join
            participant_name: Identity of this participant

        Raises:
            LiveKitError: If connection fails

        Example:
            await client.connect("race_001_voice", "ai_engineer")
        """
        try:
            # Generate access token
            token = self.generate_token(room_name, participant_name)

            # Create room
            self.room = rtc.Room()

            # Set up event handlers
            self.room.on("track_subscribed", self._on_track_subscribed)

            # Connect to server
            logger.info(f"Connecting to LiveKit room: {room_name}")
            await self.room.connect(
                url=self.config.url,
                token=token,
            )

            # Create audio source for publishing
            self.audio_source = rtc.AudioSource(
                sample_rate=self.config.sample_rate,
                num_channels=self.DEFAULT_NUM_CHANNELS,
            )

            # Create and publish audio track
            self.audio_track = rtc.LocalAudioTrack.create_audio_track(
                "ai-engineer-audio",
                self.audio_source,
            )

            await self.room.local_participant.publish_track(self.audio_track)

            self._is_connected = True
            self._room_name = room_name

            logger.info(
                f"Connected to LiveKit room '{room_name}' as '{participant_name}'"
            )

        except asyncio.TimeoutError as e:
            logger.error(f"Connection timeout: {e}")
            raise LiveKitError(f"Connection timeout: {e}", cause=e) from e
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise LiveKitError(f"Connection failed: {e}", cause=e) from e

    async def disconnect(self) -> None:
        """
        Disconnect from the LiveKit room.

        Safely closes the room connection and cleans up resources.

        Example:
            await client.disconnect()
        """
        if self.room:
            try:
                await self.room.disconnect()
                logger.info(f"Disconnected from room '{self._room_name}'")
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
            finally:
                self.room = None
                self.audio_source = None
                self.audio_track = None

        self._is_connected = False
        self._room_name = None

    async def send_audio(self, audio_bytes: bytes) -> None:
        """
        Send audio data through the LiveKit connection.

        Converts audio bytes to frames and captures them
        through the audio source.

        Args:
            audio_bytes: Raw audio data to send

        Raises:
            LiveKitError: If not connected to a room

        Example:
            await client.send_audio(audio_bytes)
        """
        if not self._is_connected or not self.audio_source:
            raise LiveKitError("Not connected to a room")

        frames = self._bytes_to_frames(audio_bytes)

        for frame in frames:
            await self.audio_source.capture_frame(frame)
            self._frames_sent += 1

            # Small delay to maintain timing
            frame_duration = frame.samples_per_channel / frame.sample_rate
            await asyncio.sleep(frame_duration)

    def _on_track_subscribed(
        self,
        track: Any,
        publication: Any,
        participant: Any,
    ) -> None:
        """
        Handle subscribed track events.

        Called when a remote participant publishes a track.
        Sets up audio processing for incoming audio tracks.

        Args:
            track: The subscribed track
            publication: Track publication info
            participant: Remote participant who published
        """
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(
                f"Subscribed to audio track from {participant.identity}"
            )
            asyncio.create_task(self._process_audio_stream(track))

    async def _process_audio_stream(self, track: Any) -> None:
        """
        Process incoming audio stream from a track.

        Args:
            track: Audio track to process
        """
        audio_stream = rtc.AudioStream(track)

        async for frame in audio_stream:
            self._frames_received += 1

            # Convert frame to bytes and notify callback
            if self.on_audio_received:
                audio_bytes = self._frame_to_bytes(frame)
                await self.on_audio_received(audio_bytes)

    def _bytes_to_frames(self, audio_bytes: bytes) -> List[Any]:
        """
        Convert audio bytes to LiveKit audio frames.

        Handles WAV format parsing and frame creation.

        Args:
            audio_bytes: Raw audio data (WAV format)

        Returns:
            List of AudioFrame objects
        """
        frames = []

        # Parse WAV header if present
        if audio_bytes[:4] == b'RIFF':
            # Skip WAV header (44 bytes for standard WAV)
            audio_data = audio_bytes[44:]
        else:
            audio_data = audio_bytes

        if not audio_data:
            return frames

        # Calculate frame size
        sample_rate = self.config.sample_rate
        samples_per_frame = int(sample_rate * self.FRAME_DURATION_MS / 1000)
        bytes_per_sample = 2  # 16-bit audio
        frame_size = samples_per_frame * bytes_per_sample

        # Split into frames
        for i in range(0, len(audio_data), frame_size):
            frame_data = audio_data[i:i + frame_size]

            if len(frame_data) == frame_size:
                # Convert to int16 samples
                samples = struct.unpack(f'<{samples_per_frame}h', frame_data)

                frame = rtc.AudioFrame(
                    data=bytes(frame_data),
                    sample_rate=sample_rate,
                    num_channels=self.DEFAULT_NUM_CHANNELS,
                    samples_per_channel=samples_per_frame,
                )
                frames.append(frame)

        return frames

    def _frame_to_bytes(self, frame: Any) -> bytes:
        """
        Convert LiveKit audio frame to bytes.

        Args:
            frame: LiveKit audio frame

        Returns:
            Raw audio bytes
        """
        return bytes(frame.data)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get client statistics.

        Returns:
            Dictionary with connection and streaming statistics

        Example:
            stats = client.get_stats()
            print(f"Frames sent: {stats['frames_sent']}")
        """
        return {
            "is_connected": self._is_connected,
            "room_name": self._room_name,
            "frames_sent": self._frames_sent,
            "frames_received": self._frames_received,
            "url": self.config.url,
            "sample_rate": self.config.sample_rate,
        }

    def __repr__(self) -> str:
        """String representation of client."""
        return (
            f"LiveKitClient(connected={self._is_connected}, "
            f"room={self._room_name})"
        )


class VoiceAgent:
    """
    Voice agent for AI race engineer voice communication.

    Integrates LiveKit for WebRTC transport with Watson TTS/STT
    for voice synthesis and recognition.

    Features:
    - Text-to-speech via Watson TTS
    - Speech-to-text via Watson STT
    - WebRTC streaming via LiveKit
    - Sentence-based interrupt handling
    - Speaking state management

    Attributes:
        config: Live configuration
        livekit_client: LiveKit client for WebRTC
        tts_client: Watson TTS client (optional, set externally)
        stt_client: Watson STT client (optional, set externally)
        currently_speaking: Whether agent is currently speaking
        on_query_received: Callback for driver queries

    Example:
        agent = VoiceAgent(config=live_config)
        agent.tts_client = tts_client
        agent.stt_client = stt_client

        await agent.connect("race_001", "ai_engineer")
        await agent.speak("Box this lap for fresh tires.")
    """

    def __init__(self, config: LiveConfig):
        """
        Initialize voice agent.

        Args:
            config: Live configuration with LiveKit and voice settings
        """
        self.config = config
        self.livekit_client = LiveKitClient(config=config.livekit)

        # TTS/STT clients (set externally to allow mocking)
        self.tts_client: Optional[Any] = None
        self.stt_client: Optional[Any] = None

        # State management
        self.currently_speaking: bool = False
        self.pending_query: Optional[str] = None
        self.current_sentence_complete = asyncio.Event()

        # Callbacks
        self.on_query_received: Optional[Callable[[str], Awaitable[None]]] = None

        # Set up audio received callback
        self.livekit_client.on_audio_received = self._on_audio_received

        logger.info("Initialized VoiceAgent")

    async def connect(
        self,
        room_name: str,
        participant_name: str = "ai_engineer",
    ) -> None:
        """
        Connect to a LiveKit room.

        Args:
            room_name: Name of the room to join
            participant_name: Identity of this participant

        Example:
            await agent.connect("race_001_voice", "ai_engineer")
        """
        await self.livekit_client.connect(room_name, participant_name)

    async def disconnect(self) -> None:
        """
        Disconnect from the LiveKit room.

        Example:
            await agent.disconnect()
        """
        await self.livekit_client.disconnect()

    async def speak(
        self,
        text: str,
        priority: str = "medium",
    ) -> None:
        """
        Convert text to speech and stream via LiveKit.

        Splits text into sentences for interrupt handling,
        synthesizes each sentence, and streams audio.

        Args:
            text: Text to speak
            priority: Priority level (critical, high, medium, low)

        Example:
            await agent.speak("Box this lap. Fresh mediums ready.")
        """
        if not text or not text.strip():
            return

        sentences = self._split_into_sentences(text)

        for sentence in sentences:
            self.currently_speaking = True

            try:
                # Synthesize speech via TTS
                if self.tts_client:
                    audio_bytes = await self.tts_client.synthesize(sentence)

                    # Stream via LiveKit
                    if self.livekit_client.is_connected:
                        await self.livekit_client.send_audio(audio_bytes)

                # Signal sentence complete
                self.current_sentence_complete.set()
                self.current_sentence_complete.clear()

                # Check for pending query (interrupt)
                if self.pending_query:
                    logger.info("Interrupt detected, stopping speech")
                    break

            except Exception as e:
                logger.error(f"Speech synthesis/streaming error: {e}")
                raise

        self.currently_speaking = False

    async def _on_audio_received(self, audio_bytes: bytes) -> None:
        """
        Handle received audio from driver.

        Transcribes audio and notifies orchestrator.

        Args:
            audio_bytes: Received audio data
        """
        if not self.stt_client:
            return

        try:
            # Transcribe audio
            text = await self.stt_client.transcribe(audio_bytes)

            if text and text.strip():
                # Handle interrupt if currently speaking
                if self.currently_speaking:
                    self.pending_query = text
                    await self.current_sentence_complete.wait()

                # Notify orchestrator
                if self.on_query_received:
                    await self.on_query_received(text)

                self.pending_query = None

        except Exception as e:
            logger.error(f"Audio transcription error: {e}")

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for interrupt handling.

        Allows graceful interruption at sentence boundaries.

        Args:
            text: Text to split

        Returns:
            List of sentences

        Example:
            sentences = agent._split_into_sentences("Box now. Tires are good.")
            # Returns: ["Box now.", "Tires are good."]
        """
        # Split on sentence-ending punctuation followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get voice agent statistics.

        Returns:
            Dictionary with agent statistics
        """
        return {
            "currently_speaking": self.currently_speaking,
            "livekit_stats": self.livekit_client.get_stats(),
            "has_tts_client": self.tts_client is not None,
            "has_stt_client": self.stt_client is not None,
        }

    def __repr__(self) -> str:
        """String representation of agent."""
        return (
            f"VoiceAgent(speaking={self.currently_speaking}, "
            f"connected={self.livekit_client.is_connected})"
        )
