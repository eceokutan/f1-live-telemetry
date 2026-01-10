"""
Voice Pipeline for Jarvis-Granite Live Telemetry.

Connects STT -> Orchestrator -> TTS flow with audio frame streaming,
silence detection, and interrupt handling.

Phase 5, Section 13: Voice Pipeline
- Complete voice interaction flow
- Audio buffering with silence detection
- Speech segmentation for natural pauses
- Interrupt handling at sentence boundaries

Example:
    pipeline = VoicePipeline(config=live_config)
    pipeline.stt_client = stt_client
    pipeline.tts_client = tts_client
    pipeline.orchestrator = orchestrator

    await pipeline.start("race_001", "ai_engineer")

    # Audio is processed automatically
    pipeline.on_audio_frame_received(audio_frame)
"""

import asyncio
import logging
import math
import re
import struct
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Awaitable

from config.config import LiveConfig

logger = logging.getLogger(__name__)


class SilenceDetector:
    """
    Detects silence in audio streams for speech segmentation.

    Uses RMS amplitude analysis to determine if audio frames
    contain speech or silence. Tracks silence duration to
    detect end-of-speech events.

    Attributes:
        threshold_db: Silence threshold in decibels (default -40dB)
        min_silence_duration_ms: Minimum silence to trigger end-of-speech
        sample_rate: Audio sample rate
        current_silence_duration_ms: Current tracked silence duration

    Example:
        detector = SilenceDetector(
            threshold_db=-40.0,
            min_silence_duration_ms=300,
            sample_rate=48000
        )

        for frame in audio_stream:
            if detector.process_frame(frame):
                # End of speech detected
                process_accumulated_audio()
    """

    # Default configuration
    DEFAULT_THRESHOLD_DB = -40.0
    DEFAULT_MIN_SILENCE_MS = 300
    FRAME_DURATION_MS = 20  # 20ms frames

    def __init__(
        self,
        threshold_db: float = DEFAULT_THRESHOLD_DB,
        min_silence_duration_ms: int = DEFAULT_MIN_SILENCE_MS,
        sample_rate: int = 48000,
    ):
        """
        Initialize silence detector.

        Args:
            threshold_db: RMS threshold below which audio is silent
            min_silence_duration_ms: Duration of silence to trigger detection
            sample_rate: Audio sample rate in Hz
        """
        self.threshold_db = threshold_db
        self.min_silence_duration_ms = min_silence_duration_ms
        self.sample_rate = sample_rate

        # Convert dB threshold to linear amplitude
        self.threshold_linear = 10 ** (threshold_db / 20)

        # State tracking
        self.current_silence_duration_ms: float = 0
        self._had_speech: bool = False

        logger.debug(
            f"Initialized SilenceDetector: threshold={threshold_db}dB, "
            f"min_duration={min_silence_duration_ms}ms"
        )

    def is_frame_silent(self, frame_data: bytes) -> bool:
        """
        Check if an audio frame is silent.

        Args:
            frame_data: Raw PCM audio data (16-bit signed)

        Returns:
            True if frame is below silence threshold
        """
        if not frame_data:
            return True

        # Calculate RMS amplitude
        rms = self._calculate_rms(frame_data)

        return rms < self.threshold_linear

    def process_frame(self, frame_data: bytes) -> bool:
        """
        Process an audio frame and detect end-of-speech.

        Updates internal silence tracking and returns True
        when sufficient silence is detected after speech.

        Args:
            frame_data: Raw PCM audio data

        Returns:
            True if end-of-speech detected (after min silence duration)
        """
        is_silent = self.is_frame_silent(frame_data)

        if is_silent:
            # Accumulate silence duration
            self.current_silence_duration_ms += self.FRAME_DURATION_MS

            # Check if we should trigger end-of-speech
            if (self._had_speech and
                self.current_silence_duration_ms >= self.min_silence_duration_ms):
                logger.debug(
                    f"End-of-speech detected after "
                    f"{self.current_silence_duration_ms}ms silence"
                )
                self._had_speech = False
                return True
        else:
            # Speech detected, reset silence counter
            self.current_silence_duration_ms = 0
            self._had_speech = True

        return False

    def reset(self) -> None:
        """Reset detector state."""
        self.current_silence_duration_ms = 0
        self._had_speech = False

    def _calculate_rms(self, frame_data: bytes) -> float:
        """
        Calculate RMS amplitude of audio frame.

        Args:
            frame_data: Raw PCM audio data (16-bit signed)

        Returns:
            RMS amplitude normalized to 0-1 range
        """
        if len(frame_data) < 2:
            return 0.0

        # Number of 16-bit samples
        num_samples = len(frame_data) // 2

        if num_samples == 0:
            return 0.0

        # Unpack as signed 16-bit integers
        try:
            samples = struct.unpack(f'<{num_samples}h', frame_data[:num_samples * 2])
        except struct.error:
            return 0.0

        # Calculate RMS
        sum_squares = sum(s * s for s in samples)
        rms = math.sqrt(sum_squares / num_samples)

        # Normalize to 0-1 (max 16-bit value is 32767)
        return rms / 32767.0


class AudioBuffer:
    """
    Buffer for accumulating audio frames.

    Collects audio frames until processed, with duration limits
    to prevent memory issues.

    Attributes:
        max_duration_ms: Maximum buffer duration
        sample_rate: Audio sample rate

    Example:
        buffer = AudioBuffer(max_duration_ms=5000, sample_rate=48000)

        buffer.add_frame(frame1)
        buffer.add_frame(frame2)

        audio = buffer.get_and_clear()
    """

    DEFAULT_MAX_DURATION_MS = 5000  # 5 seconds
    FRAME_DURATION_MS = 20  # 20ms frames

    def __init__(
        self,
        max_duration_ms: int = DEFAULT_MAX_DURATION_MS,
        sample_rate: int = 48000,
    ):
        """
        Initialize audio buffer.

        Args:
            max_duration_ms: Maximum buffer duration in milliseconds
            sample_rate: Audio sample rate
        """
        self.max_duration_ms = max_duration_ms
        self.sample_rate = sample_rate
        self._frames: Deque[bytes] = deque()
        self._total_duration_ms: float = 0

    def add_frame(self, frame_data: bytes) -> None:
        """
        Add an audio frame to the buffer.

        Args:
            frame_data: Raw audio data
        """
        self._frames.append(frame_data)
        self._total_duration_ms += self.FRAME_DURATION_MS

        # Trim if exceeding max duration
        while self._total_duration_ms > self.max_duration_ms and self._frames:
            self._frames.popleft()
            self._total_duration_ms -= self.FRAME_DURATION_MS

    def get_and_clear(self) -> bytes:
        """
        Get all buffered audio and clear the buffer.

        Returns:
            Concatenated audio data
        """
        if not self._frames:
            return b''

        audio_data = b''.join(self._frames)
        self._frames.clear()
        self._total_duration_ms = 0

        return audio_data

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self._frames) == 0

    def has_content(self) -> bool:
        """Check if buffer has content."""
        return len(self._frames) > 0

    @property
    def total_duration_ms(self) -> float:
        """Get total buffered duration in milliseconds."""
        return self._total_duration_ms

    def __len__(self) -> int:
        """Return number of frames in buffer."""
        return len(self._frames)

    def clear(self) -> None:
        """Clear the buffer."""
        self._frames.clear()
        self._total_duration_ms = 0


class VoicePipeline:
    """
    Complete voice pipeline connecting STT -> Orchestrator -> TTS.

    Manages the full voice interaction flow:
    1. Receives audio frames from driver
    2. Buffers and detects end-of-speech
    3. Transcribes audio via STT
    4. Forwards query to orchestrator
    5. Receives response from orchestrator
    6. Synthesizes response via TTS
    7. Streams audio back to driver via LiveKit

    Attributes:
        config: Live configuration
        stt_client: Speech-to-text client
        tts_client: Text-to-speech client
        orchestrator: Response orchestrator
        livekit_client: WebRTC client for audio streaming
        audio_buffer: Buffer for incoming audio
        silence_detector: Detector for end-of-speech

    Example:
        pipeline = VoicePipeline(config=live_config)
        pipeline.stt_client = stt_client
        pipeline.tts_client = tts_client
        pipeline.orchestrator = orchestrator

        await pipeline.start("race_001", "ai_engineer")

        # Process incoming audio
        await pipeline.on_audio_frame_received_async(frame)
    """

    def __init__(self, config: LiveConfig):
        """
        Initialize voice pipeline.

        Args:
            config: Live configuration
        """
        self.config = config

        # Audio processing components
        self.audio_buffer = AudioBuffer(
            max_duration_ms=5000,
            sample_rate=config.livekit.sample_rate
        )
        self.silence_detector = SilenceDetector(
            threshold_db=-40.0,
            min_silence_duration_ms=300,
            sample_rate=config.livekit.sample_rate
        )

        # External components (set after initialization)
        self.stt_client: Optional[Any] = None
        self.tts_client: Optional[Any] = None
        self.orchestrator: Optional[Any] = None
        self.livekit_client: Optional[Any] = None

        # State management
        self._is_active: bool = False
        self._is_processing: bool = False
        self._pending_interrupt: bool = False
        self._was_interrupted: bool = False
        self._last_error: Optional[str] = None

        # Statistics
        self._messages_processed: int = 0
        self._responses_delivered: int = 0

        logger.info("Initialized VoicePipeline")

    @property
    def is_active(self) -> bool:
        """Whether pipeline is active."""
        return self._is_active

    @is_active.setter
    def is_active(self, value: bool) -> None:
        """Set pipeline active state."""
        self._is_active = value

    @property
    def is_processing(self) -> bool:
        """Whether pipeline is currently processing audio."""
        return self._is_processing

    @property
    def has_pending_interrupt(self) -> bool:
        """Whether there is a pending interrupt."""
        return self._pending_interrupt

    @property
    def was_interrupted(self) -> bool:
        """Whether last delivery was interrupted."""
        return self._was_interrupted

    @property
    def last_error(self) -> Optional[str]:
        """Last error message."""
        return self._last_error

    async def start(
        self,
        room_name: str,
        participant_name: str = "ai_engineer"
    ) -> None:
        """
        Start the voice pipeline.

        Connects to LiveKit room and begins processing audio.

        Args:
            room_name: LiveKit room name
            participant_name: Participant identity
        """
        logger.info(f"Starting voice pipeline for room: {room_name}")

        if self.livekit_client:
            await self.livekit_client.connect(room_name, participant_name)

        self._is_active = True
        self._last_error = None

        logger.info("Voice pipeline started")

    async def stop(self) -> None:
        """
        Stop the voice pipeline.

        Disconnects from LiveKit and clears state.
        """
        logger.info("Stopping voice pipeline")

        self._is_active = False

        if self.livekit_client:
            await self.livekit_client.disconnect()

        self.audio_buffer.clear()
        self.silence_detector.reset()

        logger.info("Voice pipeline stopped")

    def on_audio_frame_received(self, frame_data: bytes) -> None:
        """
        Handle incoming audio frame (synchronous).

        Buffers the frame for later processing.

        Args:
            frame_data: Raw audio frame
        """
        self.audio_buffer.add_frame(frame_data)

    async def on_audio_frame_received_async(self, frame_data: bytes) -> None:
        """
        Handle incoming audio frame with silence detection.

        Buffers the frame and processes when end-of-speech detected.

        Args:
            frame_data: Raw audio frame
        """
        self.audio_buffer.add_frame(frame_data)

        # Check for end of speech
        if self.silence_detector.process_frame(frame_data):
            # End of speech detected, process buffer
            audio = self.audio_buffer.get_and_clear()
            if audio:
                await self.process_audio_input(audio)

    async def process_audio_input(self, audio_data: bytes) -> None:
        """
        Process accumulated audio through the pipeline.

        Transcribes audio and forwards to orchestrator for response.

        Args:
            audio_data: Accumulated audio data
        """
        if not audio_data:
            return

        self._is_processing = True
        self._last_error = None

        try:
            # Transcribe audio
            transcript = await self._transcribe(audio_data)

            if not transcript or not transcript.strip():
                return

            logger.info(f"Transcribed: {transcript[:50]}...")

            # Forward to orchestrator
            response = await self._get_orchestrator_response(transcript)

            if response:
                # Deliver response through TTS
                await self.deliver_response(response)

            self._messages_processed += 1

        except Exception as e:
            logger.error(f"Error processing audio input: {e}")
            self._last_error = str(e)

        finally:
            self._is_processing = False

    async def _transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe audio using STT client.

        Args:
            audio_data: Audio data to transcribe

        Returns:
            Transcribed text or empty string
        """
        if not self.stt_client:
            logger.warning("No STT client configured")
            return ""

        try:
            transcript = await self.stt_client.transcribe(audio_data)
            return transcript or ""

        except Exception as e:
            logger.error(f"STT error: {e}")
            self._last_error = str(e)
            return ""

    async def _get_orchestrator_response(self, query: str) -> str:
        """
        Get response from orchestrator.

        Args:
            query: Driver query

        Returns:
            Orchestrator response or empty string
        """
        if not self.orchestrator:
            logger.warning("No orchestrator configured")
            return ""

        try:
            response = await self.orchestrator.handle_driver_query(query)
            return response or ""

        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            self._last_error = str(e)
            return ""

    async def deliver_response(self, response: str) -> None:
        """
        Deliver response through TTS and LiveKit.

        Splits response into sentences for interrupt handling.

        Args:
            response: Text response to deliver
        """
        if not response or not response.strip():
            return

        self._was_interrupted = False
        self._last_error = None

        sentences = self._split_into_sentences(response)

        for sentence in sentences:
            # Check for interrupt
            if self._pending_interrupt:
                logger.info("Response delivery interrupted")
                self._was_interrupted = True
                self._pending_interrupt = False
                break

            try:
                # Synthesize sentence
                audio_bytes = await self._synthesize(sentence)

                if audio_bytes:
                    # Stream through LiveKit
                    await self._stream_audio(audio_bytes)

            except Exception as e:
                logger.error(f"Delivery error: {e}")
                self._last_error = str(e)
                break

        if not self._was_interrupted:
            self._responses_delivered += 1

        # Clear any pending interrupt
        self._pending_interrupt = False

    async def _synthesize(self, text: str) -> bytes:
        """
        Synthesize text to speech.

        Args:
            text: Text to synthesize

        Returns:
            Audio bytes or empty bytes
        """
        if not self.tts_client:
            logger.warning("No TTS client configured")
            return b''

        try:
            audio = await self.tts_client.synthesize(text)
            return audio or b''

        except Exception as e:
            logger.error(f"TTS error: {e}")
            self._last_error = str(e)
            return b''

    async def _stream_audio(self, audio_bytes: bytes) -> None:
        """
        Stream audio through LiveKit.

        Args:
            audio_bytes: Audio to stream
        """
        if not self.livekit_client:
            logger.warning("No LiveKit client configured")
            return

        if not audio_bytes:
            return

        try:
            await self.livekit_client.send_audio(audio_bytes)

        except Exception as e:
            logger.error(f"LiveKit streaming error: {e}")
            self._last_error = str(e)

    def request_interrupt(self) -> None:
        """
        Request an interrupt of current response delivery.

        The interrupt will be honored at the next sentence boundary.
        """
        self._pending_interrupt = True
        logger.debug("Interrupt requested")

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.

        Returns:
            Dictionary with pipeline stats
        """
        return {
            "is_active": self._is_active,
            "is_processing": self._is_processing,
            "buffer_frames": len(self.audio_buffer),
            "buffer_duration_ms": self.audio_buffer.total_duration_ms,
            "silence_duration_ms": self.silence_detector.current_silence_duration_ms,
            "messages_processed": self._messages_processed,
            "responses_delivered": self._responses_delivered,
            "has_pending_interrupt": self._pending_interrupt,
            "last_error": self._last_error,
            "has_stt_client": self.stt_client is not None,
            "has_tts_client": self.tts_client is not None,
            "has_orchestrator": self.orchestrator is not None,
            "has_livekit_client": self.livekit_client is not None,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"VoicePipeline("
            f"active={self._is_active}, "
            f"processing={self._is_processing}, "
            f"messages={self._messages_processed})"
        )
