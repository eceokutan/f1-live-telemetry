"""
Voice Input Worker for F1 Telemetry Dashboard.

Provides hands-free voice input using:
- Continuous audio capture (pyaudio)
- Silero VAD for voice activity detection (local, neural network)
- faster-whisper for local speech-to-text transcription (CTranslate2)

No button press required - automatically detects when driver is speaking.
"""

import logging
import numpy as np
import pyaudio
import torch
from PyQt5 import QtCore

logger = logging.getLogger(__name__)


class VoiceInputWorker(QtCore.QThread):
    """
    Voice input worker thread for hands-free driver queries.

    Uses Silero VAD to detect speech, then transcribes locally with faster-whisper.
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
    SAMPLE_RATE = 16000  # Hz (required by Silero VAD and Whisper)
    CHUNK_SIZE = 512  # Samples per chunk (~32ms at 16kHz)
    CHANNELS = 1  # Mono audio
    FORMAT = pyaudio.paInt16  # 16-bit PCM

    # VAD configuration
    VAD_THRESHOLD = 0.5  # Speech probability threshold
    SPEECH_PAD_MS = 150  # Padding before/after speech (ms)
    MIN_SPEECH_DURATION_MS = 300  # Minimum speech duration to process

    def __init__(self, whisper_model_size: str = "base"):
        """
        Initialize voice input worker.

        Args:
            whisper_model_size: Whisper model size (tiny, base, small, medium, large).
                                Default "base" (~140MB, good speed/accuracy balance).
        """
        super().__init__()

        self.whisper_model_size = whisper_model_size

        # Audio stream
        self.audio = None
        self.stream = None

        # Silero VAD model
        self.vad_model = None

        # Whisper model
        self.whisper_model = None

        # State
        self._running = False
        self._paused = False  # Pause listening during TTS playback
        self._is_speaking = False
        self._speech_buffer = []
        self._silence_chunks = 0

        logger.info(f"VoiceInputWorker initialized (whisper_model={whisper_model_size})")

    def run(self):
        """Main thread execution loop."""
        self._running = True
        self.status_update.emit("Voice input starting...")

        try:
            # Initialize components
            self._initialize_vad()
            self._initialize_whisper()
            self._initialize_audio()

            self.status_update.emit("[Voice] Voice input ready (faster-whisper) - speak naturally!")

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
                                # End of speech
                                self._is_speaking = False
                                self.vad_state_changed.emit(False)
                                logger.info("[VAD] Speech ended")

                                # Check minimum duration and transcribe
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

    def _initialize_whisper(self):
        """Initialize faster-whisper model for local transcription."""
        try:
            self.status_update.emit(f"Loading Whisper model ({self.whisper_model_size})...")

            from faster_whisper import WhisperModel

            self.whisper_model = WhisperModel(
                self.whisper_model_size,
                device="cpu",
                compute_type="int8",
            )

            logger.info(f"Whisper model '{self.whisper_model_size}' loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
            raise RuntimeError(f"Whisper model initialization failed: {e}")

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
        """Transcribe buffered speech using faster-whisper."""
        try:
            # Concatenate all buffered chunks and convert to float32 for Whisper
            audio_data = np.concatenate(self._speech_buffer).astype(np.float32) / 32768.0

            logger.debug(f"Transcribing {len(audio_data)} samples ({len(audio_data) / self.SAMPLE_RATE:.1f}s) of audio...")
            self.status_update.emit("Transcribing...")

            # Run Whisper transcription
            segments, info = self.whisper_model.transcribe(
                audio_data,
                language="en",
                beam_size=5,
                initial_prompt="F1 racing, telemetry, tires, brakes, fuel, pit stop, lap time, sector",
            )

            # Collect transcript from segments
            transcript = " ".join(seg.text for seg in segments).strip()

            if transcript:
                logger.info(f"[STT] Transcribed: {transcript}")
                self.speech_detected.emit(transcript)
                self.status_update.emit(f"[Voice] Heard: \"{transcript}\"")
            else:
                logger.info("[STT] No speech recognized in audio")
                self.status_update.emit("[Voice] No speech detected")

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            self.error_occurred.emit(f"Transcription failed: {e}")

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
