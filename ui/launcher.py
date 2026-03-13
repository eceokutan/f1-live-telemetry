"""
Launcher / Settings window shown before the main dashboard.

Lets the user configure AI, voice mode, API credentials, and PTT key
without needing command-line flags.
"""

import logging
from PyQt5 import QtWidgets, QtCore, QtGui
from ui.config_manager import load_config, save_config
from ui.styles import (
    BG_COLOR, BG_COLOR_LIGHT, TEXT_COLOR, TEXT_COLOR_DIM,
    BORDER_COLOR, ACCENT_BLUE,
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
        # Ignore modifier-only presses
        if key in (QtCore.Qt.Key_Shift, QtCore.Qt.Key_Control,
                   QtCore.Qt.Key_Alt, QtCore.Qt.Key_Meta):
            return

        # Map key to a readable name
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
    """Settings / launcher dialog shown on startup."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("F1 Telemetry Dashboard")
        self.setMinimumSize(520, 620)
        self.setModal(True)

        self.config = load_config()
        self._accepted = False

        self._build_ui()
        self._apply_theme()
        self._load_from_config()
        self._toggle_ai_fields()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        # ---- About / How to use ----
        about_group = QtWidgets.QGroupBox("About && How to Use")
        about_layout = QtWidgets.QVBoxLayout(about_group)
        about_text = QtWidgets.QLabel(
            "<b>F1 Telemetry Dashboard</b> is a real-time telemetry visualisation "
            "tool for Assetto Corsa. It displays live lap data, track maps, "
            "speed/RPM/brake graphs, and tire information as you drive.<br><br>"
            "<b>How to use:</b><br>"
            "1. Install <b>Content Manager</b> for Assetto Corsa and enable "
            "Python apps in Settings &gt; Assetto Corsa &gt; Apps.<br>"
            "2. Launch Assetto Corsa and load into a track session.<br>"
            "3. Configure settings below and click <b>Start</b>.<br>"
            "4. Drive! The dashboard updates in real time.<br><br>"
            "<b>AI Race Engineer</b> (optional): Provides live commentary and "
            "answers voice questions about your race. Requires API credentials below. "
            "Speech-to-text and text-to-speech both run locally. "
            "Voice output uses Kokoro and auto-downloads a default voice on first run."
        )
        about_text.setWordWrap(True)
        about_layout.addWidget(about_text)
        layout.addWidget(about_group)

        # ---- AI Race Engineer ----
        ai_group = QtWidgets.QGroupBox("AI Race Engineer")
        ai_layout = QtWidgets.QVBoxLayout(ai_group)

        self.ai_checkbox = QtWidgets.QCheckBox("Enable AI Race Engineer")
        self.ai_checkbox.toggled.connect(self._toggle_ai_fields)
        ai_layout.addWidget(self.ai_checkbox)

        # Credentials container (shown/hidden with AI toggle)
        self.creds_widget = QtWidgets.QWidget()
        creds_layout = QtWidgets.QFormLayout(self.creds_widget)
        creds_layout.setContentsMargins(0, 4, 0, 0)

        # HuggingFace
        creds_layout.addRow(QtWidgets.QLabel("<b>HuggingFace (LLM)</b>"))
        self.hf_token_edit = QtWidgets.QLineEdit()
        self.hf_token_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.hf_token_edit.setPlaceholderText("hf_...")
        creds_layout.addRow("Token:", self.hf_token_edit)

        self.hf_model_edit = QtWidgets.QLineEdit()
        self.hf_model_edit.setPlaceholderText("e.g. mistralai/Mistral-7B-Instruct-v0.2")
        creds_layout.addRow("Model ID:", self.hf_model_edit)

        # Kokoro TTS settings (shown when voice is enabled)
        self.voice_creds_label = QtWidgets.QLabel(
            "<b>Kokoro Text-to-Speech</b> (local voice output)"
        )
        creds_layout.addRow(self.voice_creds_label)

        self.kokoro_voice_edit = QtWidgets.QLineEdit()
        self.kokoro_voice_edit.setPlaceholderText("bm_lewis")
        creds_layout.addRow("Voice ID:", self.kokoro_voice_edit)

        self.kokoro_lang_edit = QtWidgets.QLineEdit()
        self.kokoro_lang_edit.setPlaceholderText("en-gb")
        creds_layout.addRow("Language:", self.kokoro_lang_edit)

        self.kokoro_speed_spin = QtWidgets.QDoubleSpinBox()
        self.kokoro_speed_spin.setRange(0.6, 1.4)
        self.kokoro_speed_spin.setSingleStep(0.01)
        self.kokoro_speed_spin.setDecimals(2)
        self.kokoro_speed_spin.setValue(0.97)
        creds_layout.addRow("Speed:", self.kokoro_speed_spin)

        self.kokoro_cuda_checkbox = QtWidgets.QCheckBox("Use CUDA (if available)")
        creds_layout.addRow(self.kokoro_cuda_checkbox)

        self.remember_creds_checkbox = QtWidgets.QCheckBox("Remember credentials")
        creds_layout.addRow(self.remember_creds_checkbox)

        ai_layout.addWidget(self.creds_widget)
        layout.addWidget(ai_group)

        # ---- Voice mode ----
        self.voice_group = QtWidgets.QGroupBox("Voice Input")
        voice_layout = QtWidgets.QVBoxLayout(self.voice_group)

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

        # PTT key selector
        self.ptt_key_widget = QtWidgets.QWidget()
        ptt_key_layout = QtWidgets.QHBoxLayout(self.ptt_key_widget)
        ptt_key_layout.setContentsMargins(20, 0, 0, 0)
        ptt_key_layout.addWidget(QtWidgets.QLabel("Key:"))
        self.ptt_key_button = KeyCaptureButton("v")
        ptt_key_layout.addWidget(self.ptt_key_button)
        ptt_key_layout.addStretch()
        voice_layout.addWidget(self.ptt_key_widget)

        # Show/hide PTT key based on radio selection
        self.voice_ptt_radio.toggled.connect(self.ptt_key_widget.setVisible)
        self.ptt_key_widget.setVisible(False)

        layout.addWidget(self.voice_group)

        # ---- Buttons ----
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()

        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.setFixedSize(120, 40)
        self.start_button.clicked.connect(self._on_start)
        btn_layout.addWidget(self.start_button)

        layout.addWidget(QtWidgets.QWidget())  # spacer
        layout.addLayout(btn_layout)

    # ------------------------------------------------------------------
    # Visibility toggles
    # ------------------------------------------------------------------

    def _toggle_ai_fields(self):
        enabled = self.ai_checkbox.isChecked()
        self.creds_widget.setVisible(enabled)
        self.voice_group.setVisible(enabled)
        if not enabled:
            self.voice_disabled_radio.setChecked(True)
        self.adjustSize()

    def _toggle_voice_creds(self, disabled_checked):
        """Show/hide local TTS settings based on voice mode."""
        voice_enabled = not disabled_checked
        self.voice_creds_label.setVisible(voice_enabled)
        self.kokoro_voice_edit.setVisible(voice_enabled)
        self.kokoro_lang_edit.setVisible(voice_enabled)
        self.kokoro_speed_spin.setVisible(voice_enabled)
        self.kokoro_cuda_checkbox.setVisible(voice_enabled)
        # Also hide the form row labels
        form = self.creds_widget.layout()
        if isinstance(form, QtWidgets.QFormLayout):
            for edit in (self.kokoro_voice_edit, self.kokoro_lang_edit, self.kokoro_speed_spin):
                label = form.labelForField(edit)
                if label:
                    label.setVisible(voice_enabled)

    # ------------------------------------------------------------------
    # Config load / save
    # ------------------------------------------------------------------

    def _load_from_config(self):
        c = self.config
        self.ai_checkbox.setChecked(c.get("ai_enabled", False))
        self.hf_token_edit.setText(c.get("huggingface_token", ""))
        self.hf_model_edit.setText(c.get("huggingface_model_id", ""))
        self.kokoro_voice_edit.setText(c.get("kokoro_voice", "bm_lewis"))
        self.kokoro_lang_edit.setText(c.get("kokoro_lang", "en-gb"))
        self.kokoro_speed_spin.setValue(float(c.get("kokoro_speed", 0.97)))
        self.kokoro_cuda_checkbox.setChecked(bool(c.get("kokoro_use_cuda", False)))
        self.remember_creds_checkbox.setChecked(c.get("remember_credentials", False))

        voice = c.get("voice_mode", "disabled")
        if voice == "push_to_talk":
            self.voice_ptt_radio.setChecked(True)
        elif voice == "continuous":
            self.voice_continuous_radio.setChecked(True)
        else:
            self.voice_disabled_radio.setChecked(True)

        ptt_key = c.get("ptt_key", "v")
        self.ptt_key_button.key_name = ptt_key
        self.ptt_key_button._update_text()

    def _save_to_config(self):
        if self.voice_ptt_radio.isChecked():
            voice_mode = "push_to_talk"
        elif self.voice_continuous_radio.isChecked():
            voice_mode = "continuous"
        else:
            voice_mode = "disabled"

        self.config.update({
            "ai_enabled": self.ai_checkbox.isChecked(),
            "voice_mode": voice_mode,
            "ptt_key": self.ptt_key_button.key_name,
            "remember_credentials": self.remember_creds_checkbox.isChecked(),
            "huggingface_token": self.hf_token_edit.text().strip(),
            "huggingface_model_id": self.hf_model_edit.text().strip(),
            "kokoro_voice": self.kokoro_voice_edit.text().strip() or "bm_lewis",
            "kokoro_lang": self.kokoro_lang_edit.text().strip() or "en-gb",
            "kokoro_speed": float(self.kokoro_speed_spin.value()),
            "kokoro_use_cuda": self.kokoro_cuda_checkbox.isChecked(),
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

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------

    def _apply_theme(self):
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {BG_COLOR};
                color: {TEXT_COLOR};
            }}
            QGroupBox {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 12px;
                font-weight: bold;
                color: {TEXT_COLOR};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }}
            QLabel {{
                color: {TEXT_COLOR};
                font-size: 10pt;
            }}
            QCheckBox, QRadioButton {{
                color: {TEXT_COLOR};
                font-size: 10pt;
                spacing: 6px;
            }}
            QLineEdit {{
                background-color: {BG_COLOR_LIGHT};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                padding: 4px 6px;
                font-size: 10pt;
            }}
            QLineEdit:focus {{
                border-color: {ACCENT_BLUE};
            }}
            QPushButton {{
                background-color: {ACCENT_BLUE};
                color: #FFFFFF;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11pt;
            }}
            QPushButton:hover {{
                background-color: #5A98EF;
            }}
            QPushButton:pressed {{
                background-color: #4A88DF;
            }}
        """)
