"""
TTS Output Worker for F1 Telemetry Dashboard.

Provides audio output for AI race engineer responses using:
- IBM Watson Text-to-Speech for synthesis (cloud API)
- PyAudio for audio playback (local)

Synthesizes AI responses and plays them through the default audio output device.
"""

import asyncio
import io
import logging
import wave
from typing import Optional
from PyQt5 import QtCore
import pyaudio
import aiohttp

logger = logging.getLogger(__name__)


class SimpleTTSClient:
    """Simple Watson TTS client with minimal dependencies."""

    def __init__(self, api_key: str, service_url: str, voice: str = "en-GB_JamesV3Voice"):
        self.api_key = api_key
        self.service_url = service_url.rstrip("/")
        self.voice = voice

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio using Watson TTS."""
        url = f"{self.service_url}/v1/synthesize"

        headers = {
            "Accept": "audio/wav",
            "Content-Type": "application/json",
        }

        params = {
            "voice": self.voice,
        }

        payload = {
            "text": text,
        }

        auth = aiohttp.BasicAuth("apikey", self.api_key)
        timeout = aiohttp.ClientTimeout(total=10.0)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url,
                headers=headers,
                params=params,
                json=payload,
                auth=auth,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"TTS API error {response.status}: {error_text}")

                return await response.read()


class TTSOutputWorker(QtCore.QThread):
    """
    TTS output worker thread for race engineer audio.

    Synthesizes AI responses using IBM Watson TTS and plays them
    through the default audio output device.

    Signals:
        status_update(str message) - Status messages for logging
        error_occurred(str error) - Error messages
        playback_started() - Emitted when audio playback starts
        playback_finished() - Emitted when audio playback finishes
    """

    status_update = QtCore.pyqtSignal(str)
    error_occurred = QtCore.pyqtSignal(str)
    playback_started = QtCore.pyqtSignal()
    playback_finished = QtCore.pyqtSignal()

    def __init__(
        self,
        watson_api_key: str,
        watson_url: str,
        voice: str = "en-GB_JamesV3Voice"
    ):
        """
        Initialize TTS output worker.

        Args:
            watson_api_key: IBM Watson TTS API key
            watson_url: Watson TTS service URL
            voice: Watson TTS voice to use (default: British male race engineer)
        """
        super().__init__()

        self.watson_api_key = watson_api_key
        self.watson_url = watson_url
        self.voice = voice

        # Watson TTS client
        self.tts_client = None

        # Audio playback
        self.audio = None

        # State
        self._running = False
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Message queue (initialized in run() after event loop is created)
        self.message_queue: Optional[asyncio.Queue] = None

        logger.info(f"TTSOutputWorker initialized with voice={voice}")

    def run(self):
        """Main thread execution loop."""
        self._running = True
        self.status_update.emit("TTS output starting...")

        try:
            # Create asyncio event loop for this thread
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)

            # Create message queue (must be done AFTER event loop is set)
            self.message_queue = asyncio.Queue()

            # Initialize components
            self._initialize_tts_client()
            self._initialize_audio()

            self.status_update.emit("🔊 TTS output ready")

            # Run async processing loop
            self._event_loop.run_until_complete(self._process_loop())

        except Exception as e:
            logger.error(f"TTS output error: {e}", exc_info=True)
            self.error_occurred.emit(f"TTS output failed: {e}")
        finally:
            if self._event_loop:
                self._event_loop.close()
            self._cleanup()
            self.status_update.emit("TTS output stopped")

    def _initialize_tts_client(self):
        """Initialize Watson TTS client."""
        try:
            self.status_update.emit("Connecting to IBM Watson TTS...")

            self.tts_client = SimpleTTSClient(
                api_key=self.watson_api_key,
                service_url=self.watson_url,
                voice=self.voice
            )

            logger.info("Watson TTS client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Watson TTS: {e}", exc_info=True)
            raise RuntimeError(f"Watson TTS initialization failed: {e}")

    def _initialize_audio(self):
        """Initialize audio output."""
        try:
            self.status_update.emit("Opening audio output...")

            self.audio = pyaudio.PyAudio()

            logger.info("Audio output initialized")

        except Exception as e:
            logger.error(f"Failed to initialize audio output: {e}", exc_info=True)
            raise RuntimeError(f"Audio output initialization failed: {e}")

    async def _process_loop(self):
        """Async processing loop - synthesizes and plays TTS messages."""
        while self._running:
            try:
                # Get message from queue (with timeout)
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )

                logger.info(f"Synthesizing TTS for: {message[:50]}...")
                await self._synthesize_and_play(message)

            except asyncio.TimeoutError:
                # No message, that's ok
                continue
            except Exception as e:
                logger.error(f"Error in TTS processing loop: {e}", exc_info=True)
                self.error_occurred.emit(f"TTS error: {e}")
                continue

    async def _synthesize_and_play(self, text: str):
        """
        Synthesize text to speech and play it.

        Args:
            text: Text to synthesize and play
        """
        try:
            # Synthesize using Watson TTS
            self.status_update.emit("Synthesizing speech...")
            audio_bytes = await self.tts_client.synthesize(text)

            logger.info(f"Synthesized {len(audio_bytes)} bytes of audio")

            # Play audio
            self.status_update.emit("Playing audio...")
            self.playback_started.emit()

            await self._play_audio_async(audio_bytes)

            self.playback_finished.emit()
            logger.info("Audio playback completed")

        except Exception as e:
            logger.error(f"TTS synthesis/playback failed: {e}", exc_info=True)
            self.error_occurred.emit(f"TTS error: {e}")

    async def _play_audio_async(self, audio_bytes: bytes):
        """
        Play WAV audio bytes through PyAudio.

        Args:
            audio_bytes: WAV audio data
        """
        # Run blocking audio playback in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._play_audio_sync, audio_bytes)

    def _play_audio_sync(self, audio_bytes: bytes):
        """
        Synchronously play WAV audio bytes.

        Args:
            audio_bytes: WAV audio data
        """
        try:
            # Parse WAV file
            with io.BytesIO(audio_bytes) as wav_io:
                with wave.open(wav_io, 'rb') as wav_file:
                    # Get audio parameters
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    frame_rate = wav_file.getframerate()

                    # Open audio stream
                    stream = self.audio.open(
                        format=self.audio.get_format_from_width(sample_width),
                        channels=channels,
                        rate=frame_rate,
                        output=True
                    )

                    # Play audio in chunks
                    chunk_size = 1024
                    data = wav_file.readframes(chunk_size)

                    while data and self._running:
                        stream.write(data)
                        data = wav_file.readframes(chunk_size)

                    # Clean up stream
                    stream.stop_stream()
                    stream.close()

        except Exception as e:
            logger.error(f"Error playing audio: {e}", exc_info=True)
            raise

    def speak(self, text: str):
        """
        Queue text for TTS synthesis and playback (called from main thread).

        Args:
            text: Text to speak
        """
        if self._event_loop and self._running and text.strip():
            # Thread-safe: put message in queue
            asyncio.run_coroutine_threadsafe(
                self.message_queue.put(text),
                self._event_loop
            )

    def _cleanup(self):
        """Clean up audio resources."""
        if self.audio:
            try:
                self.audio.terminate()
            except:
                pass

        logger.info("Audio resources cleaned up")

    def stop(self):
        """Stop the TTS output worker."""
        logger.info("Stopping TTS output...")
        self._running = False
