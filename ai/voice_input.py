"""
Voice Input Worker for F1 Telemetry Dashboard.

Provides hands-free voice input using:
- Continuous audio capture (pyaudio)
- Silero VAD for voice activity detection (local, neural network)
- IBM Watson Speech-to-Text for transcription (cloud API)

No button press required - automatically detects when driver is speaking.
"""

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
    VAD_THRESHOLD = 0.5  # Speech probability threshold
    SPEECH_PAD_MS = 300  # Padding before/after speech (ms)
    MIN_SPEECH_DURATION_MS = 500  # Minimum speech duration to process

    def __init__(
        self,
        watson_api_key: str,
        watson_url: str,
        model: str = "en-US_BroadbandModel"
    ):
        """
        Initialize voice input worker.

        Args:
            watson_api_key: IBM Watson STT API key
            watson_url: Watson STT service URL
            model: Watson STT model to use
        """
        super().__init__()

        self.watson_api_key = watson_api_key
        self.watson_url = watson_url
        self.model = model

        # Audio stream
        self.audio = None
        self.stream = None

        # Silero VAD model
        self.vad_model = None

        # Watson STT client
        self.stt_client = None

        # State
        self._running = False
        self._is_speaking = False
        self._speech_buffer = []
        self._silence_chunks = 0

        logger.info("VoiceInputWorker initialized")

    def run(self):
        """Main thread execution loop."""
        self._running = True
        self.status_update.emit("Voice input starting...")

        try:
            # Initialize components
            self._initialize_vad()
            self._initialize_watson_stt()
            self._initialize_audio()

            self.status_update.emit("🎤 Voice input ready - speak naturally!")

            # Main audio processing loop
            while self._running:
                try:
                    # Read audio chunk
                    audio_data = self.stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                    audio_int16 = np.frombuffer(audio_data, dtype=np.int16)

                    # Convert to float32 for VAD
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0

                    # Detect speech
                    speech_prob = self._detect_speech(audio_float32)

                    # Process based on VAD state
                    if speech_prob > self.VAD_THRESHOLD:
                        # Speech detected
                        if not self._is_speaking:
                            self._is_speaking = True
                            self._speech_buffer = []
                            self.vad_state_changed.emit(True)
                            logger.debug("Speech started")

                        self._speech_buffer.append(audio_int16)
                        self._silence_chunks = 0

                    else:
                        # Silence detected
                        if self._is_speaking:
                            self._silence_chunks += 1

                            # Continue buffering for a bit (padding)
                            if self._silence_chunks < self._chunks_for_ms(self.SPEECH_PAD_MS):
                                self._speech_buffer.append(audio_int16)
                            else:
                                # End of speech - transcribe
                                self._is_speaking = False
                                self.vad_state_changed.emit(False)
                                logger.debug("Speech ended")

                                # Check minimum duration
                                duration_ms = len(self._speech_buffer) * (self.CHUNK_SIZE / self.SAMPLE_RATE) * 1000
                                if duration_ms >= self.MIN_SPEECH_DURATION_MS:
                                    self._transcribe_speech()
                                else:
                                    logger.debug(f"Speech too short ({duration_ms:.0f}ms), ignoring")

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
            logger.info(f"Using microphone: {device_info['name']}")

            self.stream = self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.CHUNK_SIZE
            )

            logger.info("Audio stream opened successfully")

        except Exception as e:
            logger.error(f"Failed to open audio stream: {e}", exc_info=True)
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
                    logger.info(f"Transcribed: {transcript}")
                    self.speech_detected.emit(transcript)
                    self.status_update.emit(f"Heard: \"{transcript}\"")
                else:
                    logger.debug("Empty transcript")
            else:
                logger.debug("No speech recognized")

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

    def stop(self):
        """Stop the voice input worker."""
        logger.info("Stopping voice input...")
        self._running = False
