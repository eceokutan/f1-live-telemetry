"""
Setup & Settings window for Jarvis Granite.

Lets the user configure voice mode and keyboard PTT key.
"""

import logging
import os
from PyQt5 import QtWidgets, QtCore, QtGui
from ui.config_manager import load_config, save_config
from ui.styles import (
    BG_COLOR, BG_COLOR_LIGHT, TEXT_COLOR,
    BORDER_COLOR, ACCENT_PRIMARY, FONT_HEADING, FONT_BODY,
)

logger = logging.getLogger(__name__)


class KeyCaptureButton(QtWidgets.QPushButton):
    """Button that captures the next key press and displays its name."""

    key_captured = QtCore.pyqtSignal(str)

    def __init__(self, current_key: str = "v", parent=None):
        super().__init__(parent)
        self._capturing = False
        self.key_name = current_key
        self._update_text()

    def _update_text(self):
        if self._capturing:
            self.setText("Press any key...")
        else:
            self.setText(f"  {self.key_name.upper()}  (click to change)")

    def mousePressEvent(self, event):
        self._capturing = True
        self._update_text()
        self.setFocus()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if not self._capturing:
            super().keyPressEvent(event)
            return

        key = event.key()
        if key in (QtCore.Qt.Key_Shift, QtCore.Qt.Key_Control,
                   QtCore.Qt.Key_Alt, QtCore.Qt.Key_Meta):
            return

        seq = QtGui.QKeySequence(key)
        name = seq.toString().lower()
        if name:
            self.key_name = name
            self.key_captured.emit(name)

        self._capturing = False
        self._update_text()

    def focusOutEvent(self, event):
        self._capturing = False
        self._update_text()
        super().focusOutEvent(event)


class LauncherWindow(QtWidgets.QDialog):
    """Setup & Settings dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Jarvis Granite - Setup & Settings")
        self.setMinimumSize(780, 640)
        self.resize(820, 680)
        self.setModal(True)

        self.config = load_config()
        self._accepted = False
        self._build_ui()
        self._apply_theme()
        self._load_from_config()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 16, 20, 16)

        # ---- Title ----
        title = QtWidgets.QLabel("SETUP & SETTINGS")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet(f"""
            font-family: '{FONT_HEADING}';
            font-size: 26px;
            padding: 4px;
            background-color: transparent;
        """)
        layout.addWidget(title)

        # Keep content scrollable so action buttons are always reachable.
        content_scroll = QtWidgets.QScrollArea()
        content_scroll.setWidgetResizable(True)
        content_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setSpacing(10)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # ---- About ----
        about_group = QtWidgets.QGroupBox("ABOUT")
        about_layout = QtWidgets.QVBoxLayout(about_group)
        about_layout.setContentsMargins(12, 16, 12, 10)
        about_text = QtWidgets.QLabel(
            "<b>F1 Jarvis Granite</b> is a real-time telemetry visualisation "
            "tool for Assetto Corsa. It displays live lap data, track maps, "
            "speed/RPM/brake graphs, and tire information as you drive.<br><br>"
            "<b>How to set up:</b><br>"
            "1. Download <a href='https://assettocorsa.club/content-manager.html' "
            "style='color: #6FA8FF;'>Content Manager</a> for Assetto Corsa<br>"
            "2. In Content Manager go to Settings &gt; Assetto Corsa &gt; System "
            "&gt; Allow Developer Apps and Settings &gt; Assetto Corsa &gt; "
            "Python Apps &gt; Enable Python Apps and Developer Apps<br>"
            "3. Launch Assetto Corsa and load into a track session<br>"
            "4. Configure voice settings below and click <b>Start Jarvis Live</b> "
            "(make sure your preferred voice input device is the default in "
            "system settings)<br>"
            "5. Drive! The dashboard will update in real time<br>"
            "6. Once you are done recording a session, exit Jarvis Live and "
            "click <b>Start Jarvis Post</b> to see your Post Race Analysis"
        )
        about_text.setOpenExternalLinks(True)
        about_text.setWordWrap(True)
        about_text.setMinimumHeight(130)
        about_text.setStyleSheet(f"""
            color: {TEXT_COLOR};
            font-family: '{FONT_BODY}';
            font-size: 10pt;
            background-color: transparent;
        """)
        about_layout.addWidget(about_text)
        content_layout.addWidget(about_group)

        # ---- Lower settings row ----
        lower_row = QtWidgets.QHBoxLayout()
        lower_row.setSpacing(10)

        # ---- Voice mode ----
        voice_group = QtWidgets.QGroupBox("VOICE INPUT")
        voice_layout = QtWidgets.QVBoxLayout(voice_group)
        voice_layout.setContentsMargins(12, 16, 12, 10)

        self.voice_disabled_radio = QtWidgets.QRadioButton("Disabled")
        self.voice_ptt_radio = QtWidgets.QRadioButton("Push-to-Talk")
        self.voice_continuous_radio = QtWidgets.QRadioButton("Continuous (always listening)")

        self.voice_btn_group = QtWidgets.QButtonGroup(self)
        self.voice_btn_group.addButton(self.voice_disabled_radio)
        self.voice_btn_group.addButton(self.voice_ptt_radio)
        self.voice_btn_group.addButton(self.voice_continuous_radio)

        voice_layout.addWidget(self.voice_disabled_radio)
        voice_layout.addWidget(self.voice_ptt_radio)
        voice_layout.addWidget(self.voice_continuous_radio)

        # PTT keyboard selector
        self.ptt_key_widget = QtWidgets.QWidget()
        ptt_key_layout = QtWidgets.QVBoxLayout(self.ptt_key_widget)
        ptt_key_layout.setContentsMargins(20, 4, 0, 0)
        ptt_key_layout.setSpacing(4)

        ptt_key_row = QtWidgets.QHBoxLayout()
        ptt_key_row.addWidget(QtWidgets.QLabel("Keyboard Key:"))
        self.ptt_key_button = KeyCaptureButton("v")
        ptt_key_row.addWidget(self.ptt_key_button)
        ptt_key_row.addStretch()
        ptt_key_layout.addLayout(ptt_key_row)

        self.ptt_key_hint_label = QtWidgets.QLabel(
            "This picker changes keyboard PTT only. "
        )
        self.ptt_key_hint_label.setWordWrap(True)
        self.ptt_key_hint_label.setStyleSheet(
            f"color: {TEXT_COLOR}; font-size: 10pt;"
        )
        ptt_key_layout.addWidget(self.ptt_key_hint_label)
        voice_layout.addWidget(self.ptt_key_widget)

        # Show/hide PTT key based on radio selection
        self.voice_ptt_radio.toggled.connect(self.ptt_key_widget.setVisible)
        self.ptt_key_widget.setVisible(False)

        lower_row.addWidget(voice_group, 1)

        # ---- AI mode ----
        ai_group = QtWidgets.QGroupBox("AI RACE ENGINEER")
        ai_layout = QtWidgets.QVBoxLayout(ai_group)
        ai_layout.setContentsMargins(12, 16, 12, 10)
        ai_layout.setSpacing(8)

        self.ai_enabled_checkbox = QtWidgets.QCheckBox("Enable AI Race Engineer")
        ai_layout.addWidget(self.ai_enabled_checkbox)

        self.use_local_llm_checkbox = QtWidgets.QCheckBox("Use Local Model (GGUF)")
        ai_layout.addWidget(self.use_local_llm_checkbox)

        model_row = QtWidgets.QHBoxLayout()
        model_row.addWidget(QtWidgets.QLabel("GGUF Model Path:"))
        self.local_model_path_edit = QtWidgets.QLineEdit()
        self.local_model_path_edit.setPlaceholderText("race_engineer_gguf/granite-race-engineer-Q4_K_M.gguf")
        model_row.addWidget(self.local_model_path_edit, 1)
        self.local_model_browse_button = QtWidgets.QPushButton("Browse")
        self.local_model_browse_button.setFixedWidth(90)
        self.local_model_browse_button.clicked.connect(self._browse_local_model_path)
        model_row.addWidget(self.local_model_browse_button)
        ai_layout.addLayout(model_row)

        ai_hint = QtWidgets.QLabel(
            "If the GGUF model file is not found, "
            "Jarvis Live will use rule-based fallback responses."
        )
        ai_hint.setWordWrap(True)
        ai_hint.setStyleSheet(f"color: {TEXT_COLOR}; font-size: 10pt;")
        ai_layout.addWidget(ai_hint)

        self.use_local_llm_checkbox.toggled.connect(self._update_local_controls_visibility)
        lower_row.addWidget(ai_group, 1)

        content_layout.addLayout(lower_row)
        content_layout.addStretch()
        content_scroll.setWidget(content)
        layout.addWidget(content_scroll, 1)

        # ---- Buttons ----
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()

        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setFixedSize(140, 44)
        self.cancel_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {BG_COLOR_LIGHT};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 12pt;
            }}
            QPushButton:hover {{ background-color: #333333; }}
            QPushButton:pressed {{ background-color: #2a2a2a; }}
        """)
        self.cancel_button.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_button)

        self.start_button = QtWidgets.QPushButton("Save")
        self.start_button.setFixedSize(140, 44)
        self.start_button.clicked.connect(self._on_start)
        btn_layout.addWidget(self.start_button)

        layout.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # Config load / save
    # ------------------------------------------------------------------

    def _load_from_config(self):
        c = self.config

        voice = c.get("voice_mode", "push_to_talk")
        if voice == "push_to_talk":
            self.voice_ptt_radio.setChecked(True)
        elif voice == "continuous":
            self.voice_continuous_radio.setChecked(True)
        else:
            self.voice_disabled_radio.setChecked(True)

        ptt_key = c.get("ptt_key", "v")
        self.ptt_key_button.key_name = ptt_key
        self.ptt_key_button._update_text()

        self.ai_enabled_checkbox.setChecked(c.get("ai_enabled", True))
        self.use_local_llm_checkbox.setChecked(c.get("use_local_llm", False))
        self.local_model_path_edit.setText(
            c.get("local_model_path",
                   c.get("local_adapter_path", "race_engineer_gguf/granite-race-engineer-Q4_K_M.gguf"))
        )
        self._update_local_controls_visibility()

    def _save_to_config(self):
        if self.voice_ptt_radio.isChecked():
            voice_mode = "push_to_talk"
        elif self.voice_continuous_radio.isChecked():
            voice_mode = "continuous"
        else:
            voice_mode = "disabled"

        self.config.update({
            "ai_enabled": self.ai_enabled_checkbox.isChecked(),
            "voice_mode": voice_mode,
            "ptt_key": self.ptt_key_button.key_name,
            "use_local_llm": self.use_local_llm_checkbox.isChecked(),
            "local_model_path": self.local_model_path_edit.text().strip() or "race_engineer_gguf/granite-race-engineer-Q4_K_M.gguf",
        })
        save_config(self.config)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_start(self):
        self._save_to_config()
        self._accepted = True
        self.accept()

    def get_settings(self) -> dict:
        """Return the current settings dict (call after exec_())."""
        return dict(self.config)

    def was_accepted(self) -> bool:
        return self._accepted

    def _browse_local_model_path(self):
        selected, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select GGUF Model File",
            self.local_model_path_edit.text() or os.getcwd(),
            "GGUF Models (*.gguf);;All Files (*)",
        )
        if selected:
            self.local_model_path_edit.setText(selected)

    def _update_local_controls_visibility(self):
        enabled = self.use_local_llm_checkbox.isChecked()
        self.local_model_path_edit.setEnabled(enabled)
        self.local_model_browse_button.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self):
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {BG_COLOR};
                color: {TEXT_COLOR};
                font-family: '{FONT_BODY}';
            }}
            QGroupBox {{
                background-color: {BG_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 20px;
                font-family: '{FONT_HEADING}';
                font-size: 16pt;
                color: {TEXT_COLOR};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 6px;
            }}
            QLabel {{
                color: {TEXT_COLOR};
                font-size: 11pt;
                background-color: transparent;
            }}
            QCheckBox, QRadioButton {{
                color: {TEXT_COLOR};
                font-size: 11pt;
                spacing: 8px;
                background-color: transparent;
            }}
            QPushButton {{
                background-color: {ACCENT_PRIMARY};
                color: #FFFFFF;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 12pt;
            }}
            QPushButton:hover {{
                background-color: #C00500;
            }}
            QPushButton:pressed {{
                background-color: #A00400;
            }}
        """)
