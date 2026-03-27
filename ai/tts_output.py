"""
TTS Output Worker for F1 Telemetry Dashboard.

Provides audio output for AI race engineer responses using:
- Kokoro Text-to-Speech for synthesis (local, default)
- PyAudio for audio playback (local)

Synthesizes the full AI response, then plays it through the default audio output device.
"""

import asyncio
import io
import logging
import re
import sys
import time
import wave
import platform
from pathlib import Path
from typing import Any, Callable, Optional
from PyQt5 import QtCore
import pyaudio
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_KOKORO_VOICE_ID = "bm_lewis"
DEFAULT_KOKORO_LANG = "en-gb"
DEFAULT_KOKORO_SPEED = 1.3
if getattr(sys, "frozen", False):
    DEFAULT_KOKORO_CACHE_DIR = Path(sys.executable).parent / "data" / "kokoro_cache"
else:
    DEFAULT_KOKORO_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "kokoro_cache"


class KokoroTTSClient:
    """Local Kokoro TTS client (pykokoro) with automatic model/voice caching."""

    def __init__(
        self,
        voice_id: str = DEFAULT_KOKORO_VOICE_ID,
        lang: str = DEFAULT_KOKORO_LANG,
        speed: float = DEFAULT_KOKORO_SPEED,
        use_cuda: bool = False,
        cache_dir: str = "",
        status_callback: Optional[Callable[[str], None]] = None,
    ):
        self.voice_id = (voice_id or DEFAULT_KOKORO_VOICE_ID).strip()
        self.lang = (lang or DEFAULT_KOKORO_LANG).strip().lower()
        self.speed = float(speed)
        self.use_cuda = use_cuda
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else DEFAULT_KOKORO_CACHE_DIR
        self.status_callback = status_callback
        self._pipeline = None

    def initialize(self):
        """
        Initialize Kokoro pipeline.

        First run downloads model/voice assets into cache automatically.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._emit_status(f"Loading Kokoro voice '{self.voice_id}'...")
        self._clear_stale_locks()

        from pykokoro import GenerationConfig, PipelineConfig, build_pipeline
        from pykokoro import onnx_backend as kokoro_onnx_backend
        from pykokoro import utils as kokoro_utils
        from pykokoro.ssmd_parser import (
            DEFAULT_PAUSE_NONE,
            DEFAULT_PAUSE_WEAK,
            parse_ssmd_to_segments,
        )
        from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser
        from pykokoro.stages.protocols import DocumentResult
        from pykokoro.tokenizer import TokenizerConfig

        class NoSpacySsmdDocumentParser(SsmdDocumentParser):
            """Doc parser variant that forces non-spaCy sentence splitting."""

            def parse(self, text: str, cfg, trace):
                generation = cfg.generation
                initial_pause, segments = parse_ssmd_to_segments(
                    text,
                    lang=generation.lang,
                    pause_none=DEFAULT_PAUSE_NONE,
                    pause_weak=DEFAULT_PAUSE_WEAK,
                    pause_clause=generation.pause_clause,
                    pause_sentence=generation.pause_sentence,
                    pause_paragraph=generation.pause_paragraph,
                    use_spacy=False,
                )
                clean_text, spans, boundaries, doc_segments = self._build_document(
                    segments, initial_pause, trace
                )
                if generation.pause_mode == "auto":
                    boundaries.extend(self._sentence_boundaries(doc_segments, boundaries))
                return DocumentResult(
                    clean_text=clean_text,
                    annotation_spans=spans,
                    boundary_events=boundaries,
                    segments=doc_segments,
                )

        def _project_cache_path(folder: str | None = None) -> Path:
            base = self.cache_dir.resolve()
            base.mkdir(parents=True, exist_ok=True)
            if folder:
                out = base / folder
                out.mkdir(parents=True, exist_ok=True)
                return out
            return base

        # Force pykokoro's internal cache resolver to use project-local cache.
        kokoro_utils.get_user_cache_path = _project_cache_path
        kokoro_onnx_backend.get_user_cache_path = _project_cache_path

        provider = "cuda" if self.use_cuda else "cpu"
        cfg = PipelineConfig(
            voice=self.voice_id,
            model_source="huggingface",
            provider=provider,
            cache_dir=str(self.cache_dir.resolve()),
            tokenizer_config=TokenizerConfig(
                use_spacy=False,
                spacy_model_size="sm",
            ),
            generation=GenerationConfig(
                lang=self.lang,
                speed=self.speed,
                pause_mode="tts",
            ),
        )
        self._pipeline = build_pipeline(
            config=cfg,
            eager=True,
            doc_parser=NoSpacySsmdDocumentParser(),
        )

        # Warm up once so model downloads/locks happen during initialization.
        self._emit_status("Preparing Kokoro models...")
        _ = self._pipeline.run("System ready.")

        logger.info(
            "Kokoro pipeline initialized (voice=%s, lang=%s, speed=%.2f, provider=%s)",
            self.voice_id,
            self.lang,
            self.speed,
            provider,
        )

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to WAV bytes using Kokoro."""
        if self._pipeline is None:
            raise RuntimeError("Kokoro pipeline is not initialized")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> bytes:
        result = self._pipeline.run(text)
        audio = np.asarray(result.audio, dtype=np.float32)
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767.0).astype(np.int16)

        with io.BytesIO() as wav_io:
            with wave.open(wav_io, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(int(result.sample_rate))
                wav_file.writeframes(audio_int16.tobytes())
            return wav_io.getvalue()

    def close(self):
        if self._pipeline is not None:
            close_fn = getattr(self._pipeline, "close", None)
            if callable(close_fn):
                close_fn()
            self._pipeline = None

    def _emit_status(self, message: str):
        if self.status_callback:
            self.status_callback(message)

    def _clear_stale_locks(self, max_age_seconds: int = 1200):
        """Remove stale Kokoro .lock files in cache dir from interrupted downloads."""
        now = time.time()
        for lock_file in self.cache_dir.rglob("*.lock"):
            try:
                age = now - lock_file.stat().st_mtime
                if age > max_age_seconds:
                    lock_file.unlink(missing_ok=True)
            except Exception:
                continue


class TTSOutputWorker(QtCore.QThread):
    """
    TTS output worker thread for race engineer audio.

    Synthesizes AI responses using Kokoro only and plays them through
    the default audio output device.

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
        kokoro_voice_id: str = DEFAULT_KOKORO_VOICE_ID,
        kokoro_lang: str = DEFAULT_KOKORO_LANG,
        kokoro_speed: float = DEFAULT_KOKORO_SPEED,
        kokoro_use_cuda: bool = False,
        kokoro_cache_dir: str = "",
        playback_backend: str = "auto",
    ):
        """
        Initialize TTS output worker.

        Args:
            kokoro_voice_id: Kokoro voice id (e.g., "bm_lewis", "bf_emma", "af_nova").
            kokoro_lang: Kokoro language code (default "en-gb").
            kokoro_speed: Kokoro speaking speed multiplier.
            kokoro_use_cuda: Use CUDA for Kokoro if available.
            kokoro_cache_dir: Directory for Kokoro model/voice cache.
            playback_backend: "auto", "winsound", or "pyaudio" for local playback.
        """
        super().__init__()

        self.kokoro_voice_id = (kokoro_voice_id or DEFAULT_KOKORO_VOICE_ID).strip()
        self.kokoro_lang = (kokoro_lang or DEFAULT_KOKORO_LANG).strip().lower()
        self.kokoro_speed = float(kokoro_speed)
        self.kokoro_use_cuda = kokoro_use_cuda
        self.kokoro_cache_dir = kokoro_cache_dir.strip()
        self.playback_backend = playback_backend

        # Active TTS client
        self.tts_client = None

        # Audio playback
        self.audio = None

        # State
        self._running = False
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Message queue (initialized in run() after event loop is created)
        self.message_queue: Optional[asyncio.Queue] = None
        self._enqueue_sequence = 0

        logger.info(
            "TTSOutputWorker initialized with voice=%s",
            self.kokoro_voice_id,
        )

    def run(self):
        """Main thread execution loop."""
        self._running = True
        self.status_update.emit("TTS output starting...")

        try:
            # Create asyncio event loop for this thread
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)

            # Create message queue (must be done AFTER event loop is set)
            self.message_queue = asyncio.PriorityQueue()

            # Initialize components
            self._initialize_tts_client()
            self._initialize_audio()

            self.status_update.emit("TTS output ready")

            # Run async processing loop
            self._event_loop.run_until_complete(self._process_loop())

        except Exception as e:
            logger.error(f"TTS output error: {e}", exc_info=True)
            self.error_occurred.emit(f"TTS output failed: {e}")
        finally:
            self._running = False
            loop = self._event_loop
            self._event_loop = None
            self.message_queue = None
            if loop:
                loop.close()
            self._cleanup()
            self.status_update.emit("TTS output stopped")

    def _initialize_tts_client(self):
        """Initialize Kokoro TTS client."""
        try:
            self.status_update.emit("Initializing Kokoro TTS...")
            self.tts_client = KokoroTTSClient(
                voice_id=self.kokoro_voice_id,
                lang=self.kokoro_lang,
                speed=self.kokoro_speed,
                use_cuda=self.kokoro_use_cuda,
                cache_dir=self.kokoro_cache_dir,
                status_callback=self.status_update.emit,
            )
            self.tts_client.initialize()
            logger.info("Kokoro TTS client initialized")
        except Exception as e:
            logger.error("Failed to initialize Kokoro TTS: %s", e, exc_info=True)
            raise RuntimeError(f"Kokoro TTS initialization failed: {e}")

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
                queued_item = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                # Priority queue item shape: (priority, sequence, payload)
                message = queued_item[2] if isinstance(queued_item, tuple) and len(queued_item) == 3 else queued_item

                if not isinstance(message, str) or not message.strip():
                    continue

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
            self.status_update.emit("Synthesizing speech...")
            text = re.sub(r'(\d)\.(\d)', r'\1 point \2', text)
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
            backend = self.playback_backend.lower().strip()
            if backend not in {"auto", "winsound", "pyaudio"}:
                backend = "auto"

            should_use_winsound = (
                platform.system().lower().startswith("win")
                and backend in {"auto", "winsound"}
            )
            if should_use_winsound:
                try:
                    import winsound
                    winsound.PlaySound(audio_bytes, winsound.SND_MEMORY)
                    return
                except Exception:
                    if backend == "winsound":
                        raise
                    logger.warning("winsound playback failed, falling back to PyAudio", exc_info=True)

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

    @staticmethod
    def _normalize_priority(priority: Optional[int]) -> int:
        """Normalize priority to 0..3 where 0 is highest urgency."""
        try:
            value = int(priority) if priority is not None else 2
        except (TypeError, ValueError):
            value = 2
        return max(0, min(3, value))

    def _make_queue_item(self, payload: Any, priority: Optional[int] = None) -> tuple[int, int, Any]:
        """Build a stable priority-queue tuple."""
        normalized = self._normalize_priority(priority)
        self._enqueue_sequence += 1
        return (normalized, self._enqueue_sequence, payload)

    def speak(self, text: str, priority: int = 2):
        """
        Queue text for TTS synthesis and playback (called from main thread).

        Args:
            text: Text to speak
            priority: Message priority (0=critical, 3=low)
        """
        logger.info(f"[TTS] speak() called with priority={priority}: {text[:50]}...")
        loop = self._event_loop
        text_ok = bool(text and text.strip())
        loop_running = bool(loop and loop.is_running() and not loop.is_closed())
        queue_ready = self.message_queue is not None
        thread_running = self._running and self.isRunning()

        logger.info(
            "[TTS] thread_running=%s loop_running=%s queue_ready=%s text_ok=%s",
            thread_running,
            loop_running,
            queue_ready,
            text_ok,
        )

        if not (thread_running and loop_running and queue_ready and text_ok):
            logger.warning("[TTS] Message NOT queued - worker not ready")
            return

        logger.info(f"[TTS] Queueing message for TTS: {text}")
        try:
            # Thread-safe: put message in queue.
            asyncio.run_coroutine_threadsafe(
                self.message_queue.put(self._make_queue_item(text, priority)),
                loop,
            )
        except RuntimeError as e:
            # Can happen during shutdown races; do not crash caller thread.
            logger.warning("[TTS] Failed to queue message during shutdown: %s", e)

    def _cleanup(self):
        """Clean up audio resources."""
        if self.tts_client and hasattr(self.tts_client, "close"):
            try:
                self.tts_client.close()
            except Exception:
                pass

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
