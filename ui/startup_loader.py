"""
Startup loading screen with stage-by-stage progress.

Shows a dark-themed splash screen while heavy imports and model loading
happen in a background thread, so the user sees activity within ~200ms.
"""

import os
import sys
import logging

from PyQt5 import QtWidgets, QtCore, QtGui, QtSvg
from ui.styles import BG_COLOR, TEXT_COLOR, TEXT_COLOR_DIM, ACCENT_PRIMARY, FONT_HEADING, load_fonts

logger = logging.getLogger(__name__)

# Stage status constants
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

STAGE_NAMES = [
    "Init app shell",
    "Native runtimes",
    "Dashboard UI stack",
    "Telemetry backends",
    "AI live stack",
    "Local models",
    "Voice stack",
    "TTS stack",
    "Persistence",
    "Fonts & config",
    "Complete",
]


class StageRow(QtWidgets.QWidget):
    """Single row showing status icon + stage name + detail text."""

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 2, 0, 2)
        main_layout.setSpacing(2)

        # Top row: icon + name + detail
        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(8)

        self.icon_label = QtWidgets.QLabel("...")
        self.icon_label.setFixedWidth(24)
        self.icon_label.setAlignment(QtCore.Qt.AlignCenter)
        self.icon_label.setStyleSheet(f"color: #888888; font-size: 11pt;")
        top_row.addWidget(self.icon_label)

        self.name_label = QtWidgets.QLabel(name)
        self.name_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 11pt;")
        top_row.addWidget(self.name_label)

        self.detail_label = QtWidgets.QLabel("")
        self.detail_label.setStyleSheet("color: #888888; font-size: 10pt;")
        top_row.addWidget(self.detail_label, 1)

        main_layout.addLayout(top_row)


    def set_status(self, status: str, detail: str = ""):
        if status == STATUS_PENDING:
            self.icon_label.setText("...")
            self.icon_label.setStyleSheet("color: #888888; font-size: 11pt;")
        elif status == STATUS_RUNNING:
            self.icon_label.setText("...")
            self.icon_label.setStyleSheet(f"color: #FFD93D; font-size: 11pt;")
        elif status == STATUS_DONE:
            self.icon_label.setText("\u2713")
            self.icon_label.setStyleSheet("color: #6BCB77; font-size: 11pt;")
        elif status == STATUS_FAILED:
            self.icon_label.setText("X")
            self.icon_label.setStyleSheet("color: #FF6B6B; font-size: 11pt;")
        elif status == STATUS_SKIPPED:
            self.icon_label.setText("-")
            self.icon_label.setStyleSheet("color: #888888; font-size: 11pt;")

        self.detail_label.setText(detail)


class LoadingScreen(QtWidgets.QWidget):
    """Frameless, centered, dark-themed loading screen."""

    def __init__(self, parent=None):
        super().__init__(parent)
        load_fonts()
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, False)
        self.setFixedSize(500, 620)
        self.setStyleSheet(f"background-color: {BG_COLOR};")

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(40, 12, 40, 30)
        main_layout.setSpacing(8)

        # SVG logo
        svg_path = os.path.join(os.path.dirname(__file__), "img", "f1_jarvis_topdown_massive_tyres.svg")
        if os.path.exists(svg_path):
            svg_widget = QtSvg.QSvgWidget(svg_path)
            svg_widget.setFixedSize(180, 180)
            logo_container = QtWidgets.QHBoxLayout()
            logo_container.addStretch()
            logo_container.addWidget(svg_widget)
            logo_container.addStretch()
            main_layout.addLayout(logo_container)
        else:
            main_layout.addSpacing(180)

        # Title
        title = QtWidgets.QLabel("F1 JARVIS GRANITE")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet(
            f"font-family: '{FONT_HEADING}'; "
            f"color: {TEXT_COLOR_DIM}; font-size: 24px; "
            f"letter-spacing: 4px;"
        )
        main_layout.addWidget(title)

        # Subtitle
        subtitle = QtWidgets.QLabel("Loading components...")
        subtitle.setAlignment(QtCore.Qt.AlignCenter)
        subtitle.setStyleSheet("color: #888888; font-size: 10pt; margin-bottom: 12px;")
        main_layout.addWidget(subtitle)
        self._subtitle = subtitle

        # Separator
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setStyleSheet(f"color: #333333;")
        main_layout.addWidget(sep)

        main_layout.addSpacing(4)

        # Stage rows
        self._stage_rows = []
        for name in STAGE_NAMES:
            row = StageRow(name, self)
            self._stage_rows.append(row)
            main_layout.addWidget(row)

        main_layout.addStretch()

        # Error panel (hidden by default)
        self._error_panel = QtWidgets.QWidget(self)
        self._error_panel.setVisible(False)
        error_layout = QtWidgets.QVBoxLayout(self._error_panel)
        error_layout.setContentsMargins(0, 8, 0, 0)

        self._error_label = QtWidgets.QLabel("")
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #FF6B6B; font-size: 10pt;")
        error_layout.addWidget(self._error_label)

        quit_btn = QtWidgets.QPushButton("Quit")
        quit_btn.setFixedWidth(100)
        quit_btn.setStyleSheet(
            f"background-color: {ACCENT_PRIMARY}; color: white; "
            f"font-size: 11pt; font-weight: bold; padding: 6px 12px; border-radius: 4px;"
        )
        quit_btn.clicked.connect(lambda: sys.exit(1))
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(quit_btn)
        error_layout.addLayout(btn_layout)

        main_layout.addWidget(self._error_panel)

        # Always-visible quit button
        quit_always_btn = QtWidgets.QPushButton("Quit")
        quit_always_btn.setFixedWidth(100)
        quit_always_btn.setCursor(QtCore.Qt.PointingHandCursor)
        quit_always_btn.setStyleSheet(
            "background-color: #444444; color: #CCCCCC; "
            "font-size: 10pt; padding: 5px 12px; border-radius: 4px;"
        )
        quit_always_btn.clicked.connect(lambda: sys.exit(0))
        quit_btn_layout = QtWidgets.QHBoxLayout()
        quit_btn_layout.addStretch()
        quit_btn_layout.addWidget(quit_always_btn)
        quit_btn_layout.addStretch()
        main_layout.addLayout(quit_btn_layout)

        # Center on screen
        self._center_on_screen()

    def _center_on_screen(self):
        screen = QtWidgets.QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = (geo.width() - self.width()) // 2 + geo.x()
            y = (geo.height() - self.height()) // 2 + geo.y()
            self.move(x, y)

    def set_stage_status(self, index: int, status: str, detail: str = ""):
        """Update a stage row's status and detail text."""
        if 0 <= index < len(self._stage_rows):
            self._stage_rows[index].set_status(status, detail)


    def show_fatal_error(self, index: int, message: str):
        """Show error panel with Quit button for hard failures."""
        if 0 <= index < len(self._stage_rows):
            self._stage_rows[index].set_status(STATUS_FAILED, "Fatal error")
        self._error_label.setText(message)
        self._error_panel.setVisible(True)
        self._subtitle.setText("Startup failed")


class StartupLoaderThread(QtCore.QThread):
    """Background thread that runs stages 2-9 sequentially."""

    stage_update = QtCore.pyqtSignal(int, str, str)  # index, status, detail
all_done = QtCore.pyqtSignal(dict)
    fatal_error = QtCore.pyqtSignal(int, str)

    def run(self):
        results = {
            "MainWindow": None,
            "AcTelemetryWorker": None,
            "AIRaceEngineerWorker": None,
            "AI_AVAILABLE": False,
            "VoiceInputWorker": None,
            "VOICE_AVAILABLE": False,
            "TTSOutputWorker": None,
            "TTS_AVAILABLE": False,
            "SessionRecorder": None,
            "RECORDER_AVAILABLE": False,
            "PTTController": None,
            "PTT_AVAILABLE": False,
        }

        # Stage 2: Native runtimes (optional)
        self._run_stage_2(results)

        # Stage 3: Dashboard UI stack (fatal)
        if not self._run_stage_3(results):
            return

        # Stage 4: Telemetry backends (fatal)
        if not self._run_stage_4(results):
            return

        # Stage 5: AI live stack (optional)
        self._run_stage_5(results)

        # Stage 6: Local models (optional)
        self._run_stage_6(results)

        # Stage 7: Voice stack (optional)
        self._run_stage_7(results)

        # Stage 8: TTS stack (optional)
        self._run_stage_8(results)

        # Stage 9: Persistence (optional)
        self._run_stage_9(results)

        self.all_done.emit(results)

    def _run_stage_2(self, results: dict):
        """Native runtimes — check if onnxruntime/ctranslate2 were pre-loaded."""
        self.stage_update.emit(1, STATUS_RUNNING, "Checking native runtimes...")
        loaded = []
        for mod_name in ("onnxruntime", "ctranslate2"):
            if mod_name in sys.modules:
                loaded.append(mod_name)

        if loaded:
            self.stage_update.emit(1, STATUS_DONE, ", ".join(loaded))
        else:
            self.stage_update.emit(1, STATUS_SKIPPED, "Not installed")

    def _run_stage_3(self, results: dict) -> bool:
        """Dashboard UI stack — imports MainWindow (pulls numpy, matplotlib)."""
        self.stage_update.emit(2, STATUS_RUNNING, "Loading UI components...")
        try:
            from ui.main_window import MainWindow
            results["MainWindow"] = MainWindow
            self.stage_update.emit(2, STATUS_DONE, "UI ready")
            return True
        except Exception as e:
            logger.error("Fatal: Failed to import MainWindow: %s", e, exc_info=True)
            self.fatal_error.emit(2, f"Failed to load UI components:\n{e}")
            return False

    def _run_stage_4(self, results: dict) -> bool:
        """Telemetry backend — import AC worker."""
        self.stage_update.emit(3, STATUS_RUNNING, "Loading telemetry backends...")
        try:
            from telemetry.backends.ac_backend import AcTelemetryWorker
            results["AcTelemetryWorker"] = AcTelemetryWorker
            self.stage_update.emit(3, STATUS_DONE, "AC")
            return True
        except Exception as e:
            logger.error("Fatal: Failed to import telemetry backends: %s", e, exc_info=True)
            self.fatal_error.emit(3, f"Failed to load telemetry backends:\n{e}")
            return False

    def _run_stage_5(self, results: dict):
        """AI live stack — import AIRaceEngineerWorker."""
        self.stage_update.emit(4, STATUS_RUNNING, "Loading AI stack...")
        try:
            from ai.race_engineer import AIRaceEngineerWorker
            results["AIRaceEngineerWorker"] = AIRaceEngineerWorker
            results["AI_AVAILABLE"] = True
            self.stage_update.emit(4, STATUS_DONE, "AI Race Engineer ready")
        except Exception as e:
            logger.warning("AI Race Engineer not available: %s", e)
            self.stage_update.emit(4, STATUS_FAILED, str(e)[:60])

    def _download_model_with_progress(self, stage_idx, label, ensure_fn,
                                      dest_path=None, expected_size_mb=2000):
        """Download a model in a thread while showing real download progress.

        Args:
            stage_idx: Stage index for UI updates
            label: Display label for the download
            ensure_fn: Function that performs the download
            dest_path: Path to the destination file (for size tracking)
            expected_size_mb: Expected final file size in MB (for progress %)
        """
        import threading as _th
        import time as _time
        from pathlib import Path

        error = [None]
        done = _th.Event()

        def _do_download():
            try:
                ensure_fn()
            except Exception as e:
                error[0] = e
            finally:
                done.set()

        _th.Thread(target=_do_download, daemon=True).start()
        t0 = _time.time()
        while not done.is_set():
            elapsed = int(_time.time() - t0)
            size_mb = 0.0
            if dest_path:
                try:
                    p = Path(dest_path)
                    # Check final file, parent dir globs, AND the HF local_dir
                    # cache where partial downloads live during hf_hub_download
                    hf_cache_dir = p.parent / ".cache" / "huggingface" / "download"
                    candidates = [p]
                    candidates += list(p.parent.glob(f"{p.name}.*"))
                    candidates += list(p.parent.glob("*.incomplete"))
                    if hf_cache_dir.exists():
                        candidates += list(hf_cache_dir.glob("*.incomplete"))
                    for f in candidates:
                        if f.is_file() and f.stat().st_size > 0:
                            size_mb = f.stat().st_size / (1024 * 1024)
                            if size_mb > 1:
                                break
                except Exception:
                    pass

            if size_mb > 1:
                percent = min(int((size_mb / expected_size_mb) * 100), 99)
                progress_str = f" {percent}%"
            else:
                progress_str = f" ({elapsed}s)"

            self.stage_update.emit(
                stage_idx, STATUS_RUNNING,
                f"{label}...{progress_str}"
            )
            done.wait(timeout=1.0)

        if error[0]:
            raise error[0]

    def _run_stage_6(self, results: dict):
        """Local models — check/download GGUF models, prewarm LLM."""
        self.stage_update.emit(5, STATUS_RUNNING, "Checking local models...")
        try:
            from ai.model_downloader import (
                ensure_model,
                ensure_postrace_model,
                is_model_available,
                DEFAULT_LOCAL_PATH,
                POSTRACE_LOCAL_PATH,
            )

            from ai.model_downloader import get_model_path

            # Download live model if missing
            if not is_model_available(DEFAULT_LOCAL_PATH):
                self._download_model_with_progress(
                    5, "Downloading AI Race Engineer model (~2GB)", ensure_model,
                    dest_path=str(get_model_path(DEFAULT_LOCAL_PATH))
                )
            else:
                ensure_model()

            # Download post-race model if missing
            if not is_model_available(POSTRACE_LOCAL_PATH):
                self._download_model_with_progress(
                    5, "Downloading Post-Race Analyst model (~2GB)", ensure_postrace_model,
                    dest_path=str(get_model_path(POSTRACE_LOCAL_PATH))
                )
            else:
                ensure_postrace_model()

            # Prewarm LLM into memory
            try:
                from ai.model_prewarm import (
                    is_local_llm_model_available,
                    needs_local_llm_prewarm,
                    prewarm_local_llm,
                )

                local_model_path = DEFAULT_LOCAL_PATH
                if is_local_llm_model_available(local_model_path) and needs_local_llm_prewarm():
                    self._download_model_with_progress(
                        5, "Loading LLM into memory",
                        lambda: prewarm_local_llm(model_path=local_model_path)
                    )
                    self.stage_update.emit(5, STATUS_DONE, "Models downloaded, LLM loaded")
                else:
                    self.stage_update.emit(5, STATUS_DONE, "Models ready")
            except Exception as e:
                logger.warning("LLM prewarm failed (non-fatal): %s", e)
                self.stage_update.emit(5, STATUS_DONE, "Models ready, LLM loads on first use")

        except Exception as e:
            logger.error("Model download failed: %s", e, exc_info=True)
            self.stage_update.emit(5, STATUS_FAILED, f"Download failed: {str(e)[:80]}")

    def _run_stage_7(self, results: dict):
        """Voice stack — import VoiceInputWorker, prewarm faster-whisper."""
        self.stage_update.emit(6, STATUS_RUNNING, "Loading voice stack...")
        try:
            from ai.voice_input import VoiceInputWorker
            results["VoiceInputWorker"] = VoiceInputWorker
            results["VOICE_AVAILABLE"] = True

            # Prewarm faster-whisper cache
            try:
                from ai.model_prewarm import needs_faster_whisper_prewarm, prewarm_faster_whisper
                if needs_faster_whisper_prewarm(model_size="base"):
                    self.stage_update.emit(6, STATUS_RUNNING, "Caching Whisper model...")
                    prewarm_faster_whisper(model_size="base")
            except Exception as e:
                logger.warning("Faster-whisper prewarm failed (non-fatal): %s", e)

            self.stage_update.emit(6, STATUS_DONE, "Voice input ready")
        except Exception as e:
            logger.warning("Voice input not available: %s", e)
            self.stage_update.emit(6, STATUS_FAILED, str(e)[:60])

    def _run_stage_8(self, results: dict):
        """TTS stack — import TTSOutputWorker, prewarm Kokoro."""
        self.stage_update.emit(7, STATUS_RUNNING, "Loading TTS stack...")
        try:
            from ai.tts_output import TTSOutputWorker
            results["TTSOutputWorker"] = TTSOutputWorker
            results["TTS_AVAILABLE"] = True

            # Prewarm Kokoro cache
            try:
                from ai.model_prewarm import needs_kokoro_prewarm, prewarm_kokoro
                if needs_kokoro_prewarm():
                    self.stage_update.emit(7, STATUS_RUNNING, "Caching Kokoro model...")
                    prewarm_kokoro()
            except Exception as e:
                logger.warning("Kokoro prewarm failed (non-fatal): %s", e)

            self.stage_update.emit(7, STATUS_DONE, "TTS ready")
        except Exception as e:
            logger.warning("TTS output not available: %s", e)
            self.stage_update.emit(7, STATUS_FAILED, str(e)[:60])

    def _run_stage_9(self, results: dict):
        """Persistence — import SessionRecorder, PTTController."""
        self.stage_update.emit(8, STATUS_RUNNING, "Loading persistence...")
        try:
            from data.session_recorder import SessionRecorder
            results["SessionRecorder"] = SessionRecorder
            results["RECORDER_AVAILABLE"] = True
        except Exception as e:
            logger.warning("Session recorder not available: %s", e)

        try:
            from ai.ptt_controller import PTTController
            results["PTTController"] = PTTController
            results["PTT_AVAILABLE"] = True
        except Exception as e:
            logger.warning("PTT controller not available: %s", e)

        if results["RECORDER_AVAILABLE"] or results["PTT_AVAILABLE"]:
            parts = []
            if results["RECORDER_AVAILABLE"]:
                parts.append("recorder")
            if results["PTT_AVAILABLE"]:
                parts.append("PTT")
            self.stage_update.emit(8, STATUS_DONE, ", ".join(parts))
        else:
            self.stage_update.emit(8, STATUS_FAILED, "Not available")
