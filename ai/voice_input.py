"""
Voice Input Worker for F1 Telemetry Dashboard.

Provides hands-free voice input using:
- Continuous audio capture (pyaudio)
- Silero VAD for voice activity detection (local, neural network)
- IBM Watson Speech-to-Text for transcription (cloud API)

Supports two modes:
- Batch mode (default): Buffers speech, sends after silence detected
- Streaming mode: Real-time WebSocket streaming for lower latency (~500-1000ms faster)

No button press required - automatically detects when driver is speaking.
"""

import asyncio
import os
import io
import time
import logging
import numpy as np
import pyaudio
import torch
from typing import Optional
from PyQt5 import QtCore

# IBM Watson STT
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# Streaming STT client
from ai.streaming_stt import StreamingSTTClient

logger = logging.getLogger(__name__)


class VoiceInputWorker(QtCore.QThread):
    """
    Voice input worker thread for hands-free driver queries.

    Uses Silero VAD to detect speech, then transcribes with IBM Watson STT.
    Runs continuously without requiring button presses.

    Signals:
        speech_detected(str text) - Emitted when speech is transcribed
        vad_state_changed(bool is_speaking) - Emitted when VAD state changes
        status_update(str message) - Status messages for logging
        error_occurred(str error) - Error messages
    """

    speech_detected = QtCore.pyqtSignal(str)  # Transcribed text
    vad_state_changed = QtCore.pyqtSignal(bool)  # True=speaking, False=silent
    status_update = QtCore.pyqtSignal(str)
    error_occurred = QtCore.pyqtSignal(str)

    # Audio configuration
    SAMPLE_RATE = 16000  # Hz (required by Silero VAD)
    CHUNK_SIZE = 512  # Samples per chunk (~32ms at 16kHz)
    CHANNELS = 1  # Mono audio
    FORMAT = pyaudio.paInt16  # 16-bit PCM

    # VAD configuration
    VAD_THRESHOLD = 0.3  # Speech probability threshold (lowered from 0.5 for better detection)
    SPEECH_PAD_MS = 300  # Padding before/after speech (ms)
    MIN_SPEECH_DURATION_MS = 500  # Minimum speech duration to process

    def __init__(
        self,
        watson_api_key: str,
        watson_url: str,
        model: str = "en-US_BroadbandModel",
        use_streaming: bool = False
    ):
        """
        Initialize voice input worker.

        Args:
            watson_api_key: IBM Watson STT API key
            watson_url: Watson STT service URL
            model: Watson STT model to use
            use_streaming: Use WebSocket streaming for lower latency (default: False)
        """
        super().__init__()

        self.watson_api_key = watson_api_key
        self.watson_url = watson_url
        self.model = model
        self.use_streaming = use_streaming

        # Audio stream
        self.audio = None
        self.stream = None

        # Silero VAD model
        self.vad_model = None

        # Watson STT client (batch mode)
        self.stt_client = None

        # Streaming STT client (streaming mode)
        self._streaming_client: Optional[StreamingSTTClient] = None
        self._streaming_event_loop: Optional[asyncio.AbstractEventLoop] = None

        # State
        self._running = False
        self._paused = False  # Pause listening during TTS playback
        self._is_speaking = False
        self._speech_buffer = []
        self._silence_chunks = 0

        logger.info(f"VoiceInputWorker initialized (streaming={use_streaming})")

    def run(self):
        """Main thread execution loop."""
        self._running = True
        self.status_update.emit("Voice input starting...")

        try:
            # Initialize components
            self._initialize_vad()

            # Initialize STT based on mode
            if self.use_streaming:
                self._init_streaming_client()
                # Create event loop for streaming
                self._streaming_event_loop = asyncio.new_event_loop()
            else:
                self._initialize_watson_stt()

            self._initialize_audio()

            mode_str = "streaming" if self.use_streaming else "batch"
            self.status_update.emit(f"[Voice] Voice input ready ({mode_str}) - speak naturally!")

            # Main audio processing loop
            while self._running:
                try:
                    # Read audio chunk
                    audio_data = self.stream.read(self.CHUNK_SIZE, exception_on_overflow=False)

                    # Skip processing if paused (TTS is playing)
                    if self._paused:
                        # Reset speech state when paused
                        if self._is_speaking:
                            self._is_speaking = False
                            self._speech_buffer = []
                            self._silence_chunks = 0
                            self.vad_state_changed.emit(False)
                        continue

                    audio_int16 = np.frombuffer(audio_data, dtype=np.int16)

                    # Convert to float32 for VAD
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0

                    # Detect speech
                    speech_prob = self._detect_speech(audio_float32)

                    # Debug: Print speech probability periodically
                    if hasattr(self, '_vad_debug_counter'):
                        self._vad_debug_counter += 1
                    else:
                        self._vad_debug_counter = 0

                    # Print VAD probability every 2 seconds (~125 chunks at 16kHz)
                    if self._vad_debug_counter % 125 == 0:
                        logger.info(f"[VAD] Speech probability: {speech_prob:.3f} (threshold: {self.VAD_THRESHOLD})")

                    # Process based on VAD state
                    if speech_prob > self.VAD_THRESHOLD:
                        # Speech detected
                        if not self._is_speaking:
                            self._is_speaking = True
                            self._speech_buffer = []
                            self.vad_state_changed.emit(True)
                            logger.info("[VAD] Speech started - recording...")
                            self.status_update.emit("[Voice] Listening...")

                            # Start streaming session if in streaming mode
                            if self.use_streaming:
                                self._start_streaming_session()

                        # In streaming mode, send audio immediately
                        if self.use_streaming:
                            self._send_audio_streaming(audio_int16)
                        else:
                            self._speech_buffer.append(audio_int16)

                        self._silence_chunks = 0

                    else:
                        # Silence detected
                        if self._is_speaking:
                            self._silence_chunks += 1

                            # Continue buffering/streaming for a bit (padding)
                            if self._silence_chunks < self._chunks_for_ms(self.SPEECH_PAD_MS):
                                if self.use_streaming:
                                    self._send_audio_streaming(audio_int16)
                                else:
                                    self._speech_buffer.append(audio_int16)
                            else:
                                # End of speech
                                self._is_speaking = False
                                self.vad_state_changed.emit(False)
                                logger.info("[VAD] Speech ended")

                                if self.use_streaming:
                                    # Stop streaming session - transcript comes via callback
                                    self._stop_streaming_session()
                                    self.status_update.emit("[Voice] Processing...")
                                else:
                                    # Batch mode - check minimum duration and transcribe
                                    duration_ms = len(self._speech_buffer) * (self.CHUNK_SIZE / self.SAMPLE_RATE) * 1000
                                    logger.info(f"[VAD] Recorded {duration_ms:.0f}ms of speech (min: {self.MIN_SPEECH_DURATION_MS}ms)")
                                    if duration_ms >= self.MIN_SPEECH_DURATION_MS:
                                        self.status_update.emit("[Voice] Transcribing speech...")
                                        self._transcribe_speech()
                                    else:
                                        logger.info(f"[VAD] Speech too short ({duration_ms:.0f}ms), ignoring")
                                        self.status_update.emit(f"[Voice] Speech too short ({duration_ms:.0f}ms)")

                                self._speech_buffer = []
                                self._silence_chunks = 0

                except Exception as e:
                    logger.error(f"Error in audio processing loop: {e}")
                    continue

        except Exception as e:
            logger.error(f"Voice input error: {e}", exc_info=True)
            self.error_occurred.emit(f"Voice input failed: {e}")
        finally:
            self._cleanup()
            self.status_update.emit("Voice input stopped")

    def _initialize_vad(self):
        """Initialize Silero VAD model."""
        try:
            self.status_update.emit("Loading Silero VAD model...")

            # Load Silero VAD from torch hub
            self.vad_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )

            self.vad_model.eval()
            logger.info("Silero VAD model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load VAD model: {e}")

    def _initialize_watson_stt(self):
        """Initialize IBM Watson Speech-to-Text client."""
        try:
            self.status_update.emit("Connecting to IBM Watson STT...")

            # Configure SSL certificate path for macOS Homebrew Python
            import certifi
            os.environ['SSL_CERT_FILE'] = certifi.where()
            os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

            authenticator = IAMAuthenticator(self.watson_api_key)
            self.stt_client = SpeechToTextV1(authenticator=authenticator)
            self.stt_client.set_service_url(self.watson_url)

            # Set SSL verify to use certifi bundle explicitly
            self.stt_client.set_http_config({'verify': certifi.where()})

            # Test connection
            self.stt_client.list_models()

            logger.info("Watson STT client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Watson STT: {e}", exc_info=True)
            raise RuntimeError(f"Watson STT initialization failed: {e}")

    def _initialize_audio(self):
        """Initialize audio capture stream."""
        try:
            self.status_update.emit("Opening microphone...")

            self.audio = pyaudio.PyAudio()

            # Find default input device
            device_info = self.audio.get_default_input_device_info()
            device_name = device_info['name']
            device_rate = int(device_info['defaultSampleRate'])

            logger.info(f"Using microphone: {device_name}")
            logger.info(f"Device default sample rate: {device_rate} Hz")
            logger.info(f"Requested sample rate: {self.SAMPLE_RATE} Hz")

            self.status_update.emit(f"Using microphone: {device_name}")

            # Open audio stream with error handling for sample rate
            try:
                self.stream = self.audio.open(
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=self.CHUNK_SIZE,
                    input_device_index=device_info['index']
                )
                logger.info(f"Audio stream opened successfully at {self.SAMPLE_RATE} Hz")
                self.status_update.emit("Microphone ready - speak naturally!")

            except Exception as stream_error:
                # Try with device's native sample rate if 16kHz fails
                logger.warning(f"Failed at {self.SAMPLE_RATE} Hz: {stream_error}")
                logger.info(f"Retrying with device native rate: {device_rate} Hz")

                # Update sample rate to match device
                self.SAMPLE_RATE = device_rate

                self.stream = self.audio.open(
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=self.CHUNK_SIZE,
                    input_device_index=device_info['index']
                )
                logger.info(f"Audio stream opened at native rate: {self.SAMPLE_RATE} Hz")
                self.status_update.emit(f"Microphone ready ({self.SAMPLE_RATE} Hz) - speak naturally!")

        except Exception as e:
            logger.error(f"Failed to open audio stream: {e}", exc_info=True)
            self.error_occurred.emit(f"Microphone initialization failed: {e}")
            raise RuntimeError(f"Microphone initialization failed: {e}")

    def _detect_speech(self, audio_chunk: np.ndarray) -> float:
        """
        Detect speech in audio chunk using Silero VAD.

        Args:
            audio_chunk: Audio data as float32 array

        Returns:
            Speech probability (0.0 to 1.0)
        """
        try:
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_chunk)

            # Run VAD
            with torch.no_grad():
                speech_prob = self.vad_model(audio_tensor, self.SAMPLE_RATE).item()

            return speech_prob

        except Exception as e:
            logger.error(f"VAD error: {e}")
            return 0.0

    def _transcribe_speech(self):
        """Transcribe buffered speech using Watson STT."""
        try:
            # Concatenate all buffered chunks
            audio_data = np.concatenate(self._speech_buffer)

            # Convert to bytes
            audio_bytes = audio_data.tobytes()

            logger.debug(f"Transcribing {len(audio_bytes)} bytes of audio...")
            self.status_update.emit("Transcribing...")

            # Call Watson STT
            response = self.stt_client.recognize(
                audio=audio_bytes,
                content_type=f'audio/l16;rate={self.SAMPLE_RATE}',
                model=self.model,
                max_alternatives=1
            ).get_result()

            # Extract transcript
            if response['results']:
                transcript = response['results'][0]['alternatives'][0]['transcript'].strip()

                if transcript:
                    logger.info(f"[STT] Transcribed: {transcript}")
                    logger.info(f"[STT] Emitting speech_detected signal with transcript: {transcript}")
                    self.speech_detected.emit(transcript)
                    logger.info(f"[STT] Signal emitted successfully")
                    self.status_update.emit(f"[Voice] Heard: \"{transcript}\"")
                else:
                    logger.info("[STT] Empty transcript received")
                    self.status_update.emit("[Voice] No speech detected")
            else:
                logger.info("[STT] No speech recognized in audio")
                self.status_update.emit("[Voice] No speech detected")

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            self.error_occurred.emit(f"Transcription failed: {e}")

            # Retry once
            try:
                time.sleep(0.5)
                logger.info("Retrying transcription...")
                self._transcribe_speech()
            except:
                pass

    def _chunks_for_ms(self, milliseconds: int) -> int:
        """Calculate number of chunks for given milliseconds."""
        return int((milliseconds / 1000) * self.SAMPLE_RATE / self.CHUNK_SIZE)

    def _cleanup(self):
        """Clean up audio resources."""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass

        if self.audio:
            try:
                self.audio.terminate()
            except:
                pass

        logger.info("Audio resources cleaned up")

    def pause(self):
        """
        Pause voice input processing (e.g., during TTS playback).
        Thread-safe - can be called from main thread via Qt signal.
        """
        if not self._paused:
            self._paused = True
            logger.debug("Voice input paused (TTS playing)")

    def resume(self):
        """
        Resume voice input processing after TTS playback.
        Thread-safe - can be called from main thread via Qt signal.
        """
        if self._paused:
            self._paused = False
            logger.debug("Voice input resumed (TTS finished)")

    def stop(self):
        """Stop the voice input worker."""
        logger.info("Stopping voice input...")
        self._running = False

        # Stop streaming session if active
        if self.use_streaming and self._streaming_client:
            self._stop_streaming_session()

    # =========================================================================
    # STREAMING MODE METHODS
    # =========================================================================

    def _init_streaming_client(self):
        """Initialize the streaming STT client."""
        if self._streaming_client is not None:
            return

        self._streaming_client = StreamingSTTClient(
            api_key=self.watson_api_key,
            service_url=self.watson_url,
            model=self.model,
            sample_rate=self.SAMPLE_RATE
        )

        # Register callbacks
        self._streaming_client.on_transcript(self._on_streaming_transcript)
        self._streaming_client.on_partial(self._on_streaming_partial)
        self._streaming_client.on_error(self._on_streaming_error)

        logger.info("Streaming STT client initialized")

    def _start_streaming_session(self):
        """Start a streaming STT session."""
        if not self.use_streaming:
            return

        self._init_streaming_client()

        # Create event loop for async operations if needed
        if self._streaming_event_loop is None:
            self._streaming_event_loop = asyncio.new_event_loop()

        # Start streaming in the event loop
        try:
            self._streaming_event_loop.run_until_complete(
                self._streaming_client.start_streaming()
            )
            logger.info("Streaming STT session started")
            self.status_update.emit("[Voice] Streaming mode active")
        except Exception as e:
            logger.error(f"Failed to start streaming session: {e}")
            self.error_occurred.emit(f"Streaming start failed: {e}")

    def _stop_streaming_session(self):
        """Stop the streaming STT session."""
        if not self._streaming_client:
            return

        try:
            if self._streaming_event_loop:
                self._streaming_event_loop.run_until_complete(
                    self._streaming_client.stop_streaming()
                )
            logger.info("Streaming STT session stopped")
        except Exception as e:
            logger.error(f"Error stopping streaming session: {e}")

    def _send_audio_streaming(self, audio_data: np.ndarray):
        """
        Send audio chunk to streaming STT.

        Args:
            audio_data: Audio data as numpy int16 array
        """
        if not self._streaming_client or not self._streaming_client.is_streaming:
            return

        try:
            if self._streaming_event_loop:
                self._streaming_event_loop.run_until_complete(
                    self._streaming_client.send_audio(audio_data)
                )
        except Exception as e:
            logger.error(f"Error sending audio to streaming STT: {e}")

    def _on_streaming_transcript(self, transcript: str):
        """
        Handle final transcript from streaming STT.

        Args:
            transcript: Final transcribed text
        """
        if transcript:
            logger.info(f"[STT Streaming] Final transcript: {transcript}")
            self.speech_detected.emit(transcript)
            self.status_update.emit(f"[Voice] Heard: \"{transcript}\"")

    def _on_streaming_partial(self, transcript: str):
        """
        Handle partial (interim) transcript from streaming STT.

        Args:
            transcript: Partial transcribed text
        """
        if transcript:
            logger.debug(f"[STT Streaming] Partial: {transcript}")
            # Optionally emit status update for partial results
            # self.status_update.emit(f"[Voice] Hearing: \"{transcript}...\"")

    def _on_streaming_error(self, error: str):
        """
        Handle error from streaming STT.

        Args:
            error: Error message
        """
        logger.error(f"[STT Streaming] Error: {error}")
        self.error_occurred.emit(f"STT streaming error: {error}")
