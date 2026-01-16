"""
TTS Output Worker for F1 Telemetry Dashboard.

Provides audio output for AI race engineer responses using:
- IBM Watson Text-to-Speech for synthesis (cloud API)
- PyAudio for audio playback (local)

Supports three modes:
- Batch mode (default): Full synthesis before playback
- Streaming mode: Start playback while still receiving audio (~300-500ms faster)
- Sentence pipelining mode: Speak sentence-by-sentence (~500-1000ms faster for multi-sentence)

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

# Streaming TTS client
from ai.streaming_tts import StreamingTTSClient, StreamingAudioPlayer

# Sentence pipelining
from ai.sentence_pipelining import SentencePipelinedTTS

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
        voice: str = "en-GB_JamesV3Voice",
        use_streaming: bool = False,
        use_sentence_pipelining: bool = False
    ):
        """
        Initialize TTS output worker.

        Args:
            watson_api_key: IBM Watson TTS API key
            watson_url: Watson TTS service URL
            voice: Watson TTS voice to use (default: British male race engineer)
            use_streaming: Use streaming playback for lower latency (default: False)
            use_sentence_pipelining: Speak sentence-by-sentence for multi-sentence
                                     responses (~500-1000ms faster). Default: False.
        """
        super().__init__()

        self.watson_api_key = watson_api_key
        self.watson_url = watson_url
        self.voice = voice
        self.use_streaming = use_streaming
        self.use_sentence_pipelining = use_sentence_pipelining

        # Watson TTS client (batch mode)
        self.tts_client = None

        # Streaming TTS client and player (streaming mode)
        self._streaming_tts_client: Optional[StreamingTTSClient] = None
        self._streaming_player: Optional[StreamingAudioPlayer] = None

        # Sentence pipelining (pipelining mode)
        self._pipelined_tts: Optional[SentencePipelinedTTS] = None

        # Audio playback
        self.audio = None

        # State
        self._running = False
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Message queue (initialized in run() after event loop is created)
        self.message_queue: Optional[asyncio.Queue] = None

        mode_str = "pipelined" if use_sentence_pipelining else ("streaming" if use_streaming else "batch")
        logger.info(f"TTSOutputWorker initialized with voice={voice}, mode={mode_str}")

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

            # Initialize components based on mode
            if self.use_sentence_pipelining:
                self._initialize_tts_client()  # Need batch client for sentence TTS
                self._initialize_pipelined_tts()
            elif self.use_streaming:
                self._initialize_streaming_client()
            else:
                self._initialize_tts_client()
            self._initialize_audio()

            mode_str = "pipelined" if self.use_sentence_pipelining else ("streaming" if self.use_streaming else "batch")
            self.status_update.emit(f"TTS output ready ({mode_str})")

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

                # Use pipelined, streaming, or batch mode
                if self.use_sentence_pipelining:
                    await self._synthesize_and_play_pipelined(message)
                elif self.use_streaming:
                    await self._synthesize_and_play_streaming(message)
                else:
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
        logger.info(f"[TTS] speak() called with: {text[:50]}...")
        logger.info(f"[TTS] _event_loop={self._event_loop}, _running={self._running}, text.strip()={bool(text.strip())}")

        if self._event_loop and self._running and text.strip():
            logger.info(f"[TTS] Queueing message for TTS: {text}")
            # Thread-safe: put message in queue
            asyncio.run_coroutine_threadsafe(
                self.message_queue.put(text),
                self._event_loop
            )
        else:
            logger.warning(f"[TTS] Message NOT queued - conditions not met")

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

        # Stop streaming player if active
        if self._streaming_player and self._streaming_player.is_playing:
            self._streaming_player.stop()

    # =========================================================================
    # STREAMING MODE METHODS
    # =========================================================================

    def _initialize_streaming_client(self):
        """Initialize the streaming TTS client."""
        try:
            self.status_update.emit("Initializing streaming TTS...")

            self._streaming_tts_client = StreamingTTSClient(
                api_key=self.watson_api_key,
                service_url=self.watson_url,
                voice=self.voice
            )

            # Set up callbacks
            self._streaming_tts_client.on_synthesis_start(self._on_streaming_start)
            self._streaming_tts_client.on_audio_chunk(self._on_audio_chunk)
            self._streaming_tts_client.on_synthesis_complete(self._on_streaming_complete)
            self._streaming_tts_client.on_error(self._on_streaming_error)

            # Initialize streaming player
            self._streaming_player = StreamingAudioPlayer()

            logger.info("Streaming TTS client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize streaming TTS: {e}", exc_info=True)
            raise RuntimeError(f"Streaming TTS initialization failed: {e}")

    async def _synthesize_and_play_streaming(self, text: str):
        """
        Synthesize and play using streaming mode.

        Starts playback as soon as first audio chunk arrives,
        rather than waiting for full synthesis.

        Args:
            text: Text to synthesize and play
        """
        try:
            self.status_update.emit("Streaming synthesis...")

            # Start the streaming player (will be configured when first chunk arrives)
            # Using Watson TTS default format: 22050Hz, mono, 16-bit
            self._streaming_player.start(
                sample_rate=22050,
                channels=1,
                sample_width=2
            )

            # Start streaming synthesis (callbacks handle playback)
            await self._streaming_tts_client.synthesize_stream(text)

        except Exception as e:
            logger.error(f"Streaming TTS error: {e}", exc_info=True)
            self.error_occurred.emit(f"Streaming TTS error: {e}")
        finally:
            # Ensure player is stopped
            if self._streaming_player:
                self._streaming_player.stop()

    def _on_streaming_start(self):
        """Callback when streaming synthesis starts."""
        logger.info("[TTS Streaming] Synthesis started")
        self.playback_started.emit()
        self.status_update.emit("Playing (streaming)...")

    def _on_audio_chunk(self, chunk: bytes):
        """
        Callback for each audio chunk received.

        Args:
            chunk: Audio data bytes
        """
        if self._streaming_player and self._streaming_player.is_playing:
            self._streaming_player.play_chunk(chunk)

    def _on_streaming_complete(self):
        """Callback when streaming synthesis completes."""
        logger.info("[TTS Streaming] Synthesis complete")
        self.playback_finished.emit()

    def _on_streaming_error(self, error: str):
        """
        Callback for streaming errors.

        Args:
            error: Error message
        """
        logger.error(f"[TTS Streaming] Error: {error}")
        self.error_occurred.emit(f"Streaming TTS error: {error}")

    # =========================================================================
    # SENTENCE PIPELINING MODE METHODS
    # =========================================================================

    def _initialize_pipelined_tts(self):
        """Initialize the sentence-pipelined TTS handler."""
        try:
            self.status_update.emit("Initializing sentence-pipelined TTS...")

            # Create pipelined TTS with our sentence speak callback
            self._pipelined_tts = SentencePipelinedTTS(
                speak_callback=self._speak_single_sentence
            )

            # Set up callbacks
            self._pipelined_tts.on_sentence_started(self._on_sentence_started)
            self._pipelined_tts.on_sentence_completed(self._on_sentence_completed)
            self._pipelined_tts.on_all_completed(self._on_all_sentences_completed)

            logger.info("Sentence-pipelined TTS initialized")

        except Exception as e:
            logger.error(f"Failed to initialize pipelined TTS: {e}", exc_info=True)
            raise RuntimeError(f"Pipelined TTS initialization failed: {e}")

    async def _synthesize_and_play_pipelined(self, text: str):
        """
        Synthesize and play using sentence pipelining.

        Splits text into sentences and speaks them one by one,
        reducing perceived latency for multi-sentence responses.

        Args:
            text: Text to synthesize and play
        """
        try:
            self.status_update.emit("Pipelined synthesis...")
            self.playback_started.emit()

            # Use pipelined TTS to speak sentence by sentence
            await self._pipelined_tts.speak(text)

        except Exception as e:
            logger.error(f"Pipelined TTS error: {e}", exc_info=True)
            self.error_occurred.emit(f"Pipelined TTS error: {e}")

    async def _speak_single_sentence(self, sentence: str):
        """
        Speak a single sentence (callback for SentencePipelinedTTS).

        Args:
            sentence: Single sentence to synthesize and play
        """
        try:
            # Synthesize using Watson TTS
            audio_bytes = await self.tts_client.synthesize(sentence)

            logger.debug(f"Synthesized sentence ({len(audio_bytes)} bytes): {sentence[:30]}...")

            # Play audio
            await self._play_audio_async(audio_bytes)

        except Exception as e:
            logger.error(f"Error speaking sentence: {e}", exc_info=True)
            raise

    def _on_sentence_started(self, sentence: str):
        """Callback when a sentence starts speaking."""
        logger.debug(f"[TTS Pipelined] Sentence started: {sentence[:30]}...")
        self.status_update.emit(f"Speaking: {sentence[:25]}...")

    def _on_sentence_completed(self, sentence: str):
        """Callback when a sentence finishes speaking."""
        logger.debug(f"[TTS Pipelined] Sentence completed: {sentence[:30]}...")

    def _on_all_sentences_completed(self):
        """Callback when all sentences are done."""
        logger.info("[TTS Pipelined] All sentences complete")
        self.playback_finished.emit()
