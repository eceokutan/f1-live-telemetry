"""
Voice Input Worker for F1 Telemetry Dashboard.

Provides voice input using either:
- VAD mode (default): Continuous audio capture with webrtcvad for automatic
  voice activity detection. No button press required.
- PTT mode (--ptt flag): Push-to-talk. Records only while PTT button is held.
  Skips VAD model loading entirely (saves memory).

Both modes use faster-whisper for local speech-to-text transcription (CTranslate2).
"""

import logging
import threading
import numpy as np
import pyaudio
from PyQt5 import QtCore

logger = logging.getLogger(__name__)


class VoiceInputWorker(QtCore.QThread):
    """
    Voice input worker thread for driver queries.

    Supports two modes:
    - VAD mode (default): Uses webrtcvad to auto-detect speech, then transcribes
      with faster-whisper. No button press required.
    - PTT mode: Records audio only while push-to-talk is active (controlled via
      start_recording/stop_recording). Skips VAD model loading.

    Signals:
        speech_detected(str text) - Emitted when speech is transcribed
        vad_state_changed(bool is_speaking) - Emitted when recording state changes
        status_update(str message) - Status messages for logging
        error_occurred(str error) - Error messages
    """

    speech_detected = QtCore.pyqtSignal(str)  # Transcribed text
    vad_state_changed = QtCore.pyqtSignal(bool)  # True=speaking, False=silent
    status_update = QtCore.pyqtSignal(str)
    error_occurred = QtCore.pyqtSignal(str)

    # Audio configuration
    SAMPLE_RATE = 16000  # Hz (required by webrtcvad and Whisper)
    CHUNK_SIZE = 480  # Samples per chunk (30ms at 16kHz, required by webrtcvad)
    CHANNELS = 1  # Mono audio
    FORMAT = pyaudio.paInt16  # 16-bit PCM

    # VAD configuration
    VAD_AGGRESSIVENESS = 2  # webrtcvad aggressiveness (0-3, higher = more aggressive filtering)
    SPEECH_PAD_MS = 150  # Padding before/after speech (ms)
    MIN_SPEECH_DURATION_MS = 300  # Minimum speech duration to process

    def __init__(self, whisper_model_size: str = "base", ptt_mode: bool = False):
        """
        Initialize voice input worker.

        Args:
            whisper_model_size: Whisper model size (tiny, base, small, medium, large).
                                Default "base" (~140MB, good speed/accuracy balance).
            ptt_mode: If True, use push-to-talk mode (skip VAD, record only when
                      start_recording() is called). If False, use VAD auto-detection.
        """
        super().__init__()

        self.whisper_model_size = whisper_model_size
        self.ptt_mode = ptt_mode

        # Audio stream
        self.audio = None
        self.stream = None

        # webrtcvad instance (not loaded in PTT mode)
        self.vad = None

        # Whisper model
        self.whisper_model = None

        # State
        self._running = False
        self._pause_count = 0  # Reference-counted pause (TTS playback + AI processing)
        self._pause_lock = threading.Lock()
        self._is_speaking = False
        self._speech_buffer = []
        self._silence_chunks = 0

        # PTT-specific state
        self._ptt_recording = False
        self._ptt_lock = threading.Lock()

        mode_str = "PTT" if ptt_mode else "VAD"
        logger.info(f"VoiceInputWorker initialized (whisper_model={whisper_model_size}, mode={mode_str})")

    def run(self):
        """Main thread execution loop."""
        self._running = True
        self.status_update.emit("Voice input starting...")

        try:
            # Initialize components (skip VAD in PTT mode to save memory)
            if not self.ptt_mode:
                self._initialize_vad()
            self._initialize_whisper()
            self._initialize_audio()

            if self.ptt_mode:
                self.status_update.emit("[Voice] PTT mode ready — hold V or joystick button to talk")
                self._run_ptt_loop()
            else:
                self.status_update.emit("[Voice] Voice input ready (faster-whisper) - speak naturally!")
                self._run_vad_loop()

        except Exception as e:
            logger.error(f"Voice input error: {e}", exc_info=True)
            self.error_occurred.emit(f"Voice input failed: {e}")
        finally:
            self._cleanup()
            self.status_update.emit("Voice input stopped")

    def _run_vad_loop(self):
        """Audio processing loop for VAD mode. Auto-detects speech using webrtcvad."""
        while self._running:
            try:
                # Read audio chunk
                audio_data = self.stream.read(self.CHUNK_SIZE, exception_on_overflow=False)

                # Skip processing if paused (TTS playing or AI processing)
                if self._pause_count > 0:
                    # Reset speech state when paused
                    if self._is_speaking:
                        self._is_speaking = False
                        self._speech_buffer = []
                        self._silence_chunks = 0
                        self.vad_state_changed.emit(False)
                    continue

                audio_int16 = np.frombuffer(audio_data, dtype=np.int16)

                # Detect speech using raw PCM bytes
                is_speech = self._detect_speech(audio_data)

                # Process based on VAD state
                if is_speech:
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

    def _run_ptt_loop(self):
        """Audio processing loop for PTT mode. Records only while PTT button is held."""
        while self._running:
            try:
                # Read audio chunk (keeps stream alive even when not recording)
                audio_data = self.stream.read(self.CHUNK_SIZE, exception_on_overflow=False)

                # Skip if paused (TTS playing or AI processing)
                if self._pause_count > 0:
                    if self._is_speaking:
                        self._is_speaking = False
                        self._speech_buffer = []
                        self.vad_state_changed.emit(False)
                    continue

                # Check PTT state
                with self._ptt_lock:
                    is_recording = self._ptt_recording

                if is_recording:
                    audio_int16 = np.frombuffer(audio_data, dtype=np.int16)

                    # First chunk of a new recording
                    if not self._is_speaking:
                        self._is_speaking = True
                        self._speech_buffer = []
                        self.vad_state_changed.emit(True)
                        logger.info("[PTT] Recording started")
                        self.status_update.emit("[Voice] Recording (PTT held)...")

                    self._speech_buffer.append(audio_int16)

                else:
                    # PTT released — if we were recording, transcribe
                    if self._is_speaking:
                        self._is_speaking = False
                        self.vad_state_changed.emit(False)
                        logger.info("[PTT] Recording stopped")

                        duration_ms = len(self._speech_buffer) * (self.CHUNK_SIZE / self.SAMPLE_RATE) * 1000
                        logger.info(f"[PTT] Recorded {duration_ms:.0f}ms of audio")

                        if duration_ms >= self.MIN_SPEECH_DURATION_MS:
                            self.status_update.emit("[Voice] Transcribing speech...")
                            self._transcribe_speech()
                        else:
                            logger.info(f"[PTT] Recording too short ({duration_ms:.0f}ms), ignoring")
                            self.status_update.emit(f"[Voice] Too short ({duration_ms:.0f}ms)")

                        self._speech_buffer = []

            except Exception as e:
                logger.error(f"Error in PTT audio loop: {e}")
                continue

    def start_recording(self):
        """Start recording audio (called when PTT button is pressed). Thread-safe."""
        if self._pause_count > 0:
            logger.debug("PTT pressed but voice input is paused (TTS playing)")
            return
        with self._ptt_lock:
            self._ptt_recording = True
        logger.debug("PTT recording started")

    def stop_recording(self):
        """Stop recording audio (called when PTT button is released). Thread-safe."""
        with self._ptt_lock:
            self._ptt_recording = False
        logger.debug("PTT recording stopped")

    def _initialize_vad(self):
        """Initialize webrtcvad. Only called in VAD mode (not PTT)."""
        try:
            import webrtcvad

            self.status_update.emit("Initializing voice activity detection...")

            self.vad = webrtcvad.Vad(self.VAD_AGGRESSIVENESS)

            logger.info("webrtcvad initialized (aggressiveness=%d)", self.VAD_AGGRESSIVENESS)

        except Exception as e:
            logger.error(f"Failed to initialize webrtcvad: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize VAD: {e}")

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
                # Try webrtcvad-compatible rates before giving up
                # webrtcvad supports: 8000, 16000, 32000, 48000 Hz
                logger.warning(f"Failed at {self.SAMPLE_RATE} Hz: {stream_error}")
                fallback_rates = [48000, 32000, 8000]
                opened = False
                for rate in fallback_rates:
                    try:
                        logger.info(f"Trying fallback rate: {rate} Hz")
                        self.SAMPLE_RATE = rate
                        # Recalculate chunk size for 30ms at the new rate
                        self.CHUNK_SIZE = int(rate * 30 / 1000)
                        self.stream = self.audio.open(
                            format=self.FORMAT,
                            channels=self.CHANNELS,
                            rate=self.SAMPLE_RATE,
                            input=True,
                            frames_per_buffer=self.CHUNK_SIZE,
                            input_device_index=device_info['index']
                        )
                        logger.info(f"Audio stream opened at {self.SAMPLE_RATE} Hz")
                        self.status_update.emit(f"Microphone ready ({self.SAMPLE_RATE} Hz) - speak naturally!")
                        opened = True
                        break
                    except Exception:
                        continue
                if not opened:
                    raise RuntimeError(
                        f"Could not open audio stream at any supported sample rate. "
                        f"Device native rate: {device_rate} Hz"
                    )

        except Exception as e:
            logger.error(f"Failed to open audio stream: {e}", exc_info=True)
            self.error_occurred.emit(f"Microphone initialization failed: {e}")
            raise RuntimeError(f"Microphone initialization failed: {e}")

    def _detect_speech(self, audio_bytes: bytes) -> bool:
        """
        Detect speech in audio chunk using webrtcvad.

        Args:
            audio_bytes: Raw 16-bit PCM audio bytes (30ms frame at 16kHz)

        Returns:
            True if speech is detected, False otherwise
        """
        try:
            return self.vad.is_speech(audio_bytes, self.SAMPLE_RATE)
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return False

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
        Pause voice input processing (e.g., during TTS playback or AI processing).
        Reference-counted: each pause() must be matched by a resume().
        Thread-safe - can be called from any thread.
        """
        with self._pause_lock:
            self._pause_count += 1
            count = self._pause_count
        if count == 1 and self.ptt_mode:
            # Force explicit re-press after pause so an old held PTT state
            # cannot restart recording automatically on resume.
            with self._ptt_lock:
                self._ptt_recording = False
        logger.debug("Voice input paused (pause_count=%d)", count)

    def resume(self):
        """
        Resume voice input processing. Only actually resumes when all
        pause sources have called resume (pause_count reaches 0).
        Thread-safe - can be called from any thread.
        """
        with self._pause_lock:
            self._pause_count = max(0, self._pause_count - 1)
            count = self._pause_count
        logger.debug("Voice input resume requested (pause_count=%d)", count)

    def stop(self):
        """Stop the voice input worker."""
        logger.info("Stopping voice input...")
        self._running = False
