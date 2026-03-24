"""
Startup loading screen with stage-by-stage progress.

Shows a dark-themed splash screen while heavy imports and model loading
happen in a background thread, so the user sees activity within ~200ms.
"""

import os
import sys
import logging

from PyQt5 import QtWidgets, QtCore, QtGui, QtSvg
from ui.styles import BG_COLOR, TEXT_COLOR, ACCENT_PRIMARY

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
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)

        self.icon_label = QtWidgets.QLabel("...")
        self.icon_label.setFixedWidth(24)
        self.icon_label.setAlignment(QtCore.Qt.AlignCenter)
        self.icon_label.setStyleSheet(f"color: #888888; font-size: 11pt;")
        layout.addWidget(self.icon_label)

        self.name_label = QtWidgets.QLabel(name)
        self.name_label.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 11pt;")
        layout.addWidget(self.name_label)

        self.detail_label = QtWidgets.QLabel("")
        self.detail_label.setStyleSheet("color: #888888; font-size: 10pt;")
        layout.addWidget(self.detail_label, 1)

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
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, False)
        self.setFixedSize(500, 580)
        self.setStyleSheet(f"background-color: {BG_COLOR};")

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(40, 30, 40, 30)
        main_layout.setSpacing(8)

        # SVG logo
        svg_path = os.path.join(os.path.dirname(__file__), "img", "f1_jarvis_topdown_massive_tyres.svg")
        if os.path.exists(svg_path):
            svg_widget = QtSvg.QSvgWidget(svg_path)
            svg_widget.setFixedSize(140, 140)
            logo_container = QtWidgets.QHBoxLayout()
            logo_container.addStretch()
            logo_container.addWidget(svg_widget)
            logo_container.addStretch()
            main_layout.addLayout(logo_container)
        else:
            main_layout.addSpacing(140)

        # Title
        title = QtWidgets.QLabel("F1 JARVIS")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet(
            f"color: {TEXT_COLOR}; font-size: 28pt; font-weight: bold; "
            f"letter-spacing: 4px; margin-top: 8px; margin-bottom: 4px;"
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

            # Download live model if missing
            if not is_model_available(DEFAULT_LOCAL_PATH):
                self.stage_update.emit(5, STATUS_RUNNING, "Downloading live model...")
            ensure_model()

            # Download post-race model if missing
            if not is_model_available(POSTRACE_LOCAL_PATH):
                self.stage_update.emit(5, STATUS_RUNNING, "Downloading post-race model...")
            ensure_postrace_model()

            # Prewarm LLM into memory (with progress updates so UI doesn't look frozen)
            try:
                from ai.model_prewarm import (
                    is_local_llm_model_available,
                    needs_local_llm_prewarm,
                    prewarm_local_llm,
                )
                import threading as _th
                import time as _time

                local_model_path = DEFAULT_LOCAL_PATH
                if is_local_llm_model_available(local_model_path) and needs_local_llm_prewarm():
                    self.stage_update.emit(5, STATUS_RUNNING, "Loading LLM into memory...")
                    prewarm_error = [None]
                    prewarm_done = _th.Event()

                    def _do_prewarm():
                        try:
                            prewarm_local_llm(model_path=local_model_path)
                        except Exception as e:
                            prewarm_error[0] = e
                        finally:
                            prewarm_done.set()

                    _th.Thread(target=_do_prewarm, daemon=True).start()
                    t0 = _time.time()
                    while not prewarm_done.is_set():
                        elapsed = int(_time.time() - t0)
                        self.stage_update.emit(
                            5, STATUS_RUNNING,
                            f"Loading LLM into memory... ({elapsed}s)"
                        )
                        prewarm_done.wait(timeout=1.0)

                    if prewarm_error[0]:
                        raise prewarm_error[0]
                    self.stage_update.emit(5, STATUS_DONE, "Models downloaded, LLM loaded")
                else:
                    self.stage_update.emit(5, STATUS_DONE, "Models ready")
            except Exception as e:
                logger.warning("LLM prewarm failed (non-fatal): %s", e)
                self.stage_update.emit(5, STATUS_DONE, "Models ready, LLM loads on first use")

        except Exception as e:
            logger.warning("Model download/prewarm failed: %s", e)
            self.stage_update.emit(5, STATUS_FAILED, f"Will use rule-based fallback")

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
