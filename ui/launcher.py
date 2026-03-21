"""
Setup & Settings window for Jarvis Granite.

Lets the user configure voice mode and keyboard PTT key.
"""

import logging
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


class JoystickCaptureButton(QtWidgets.QPushButton):
    """Button that captures the next wheel/joystick button press and displays its index."""

    button_captured = QtCore.pyqtSignal(int)

    def __init__(self, current_button: int = 11, parent=None):
        super().__init__(parent)
        self._capturing = False
        self._poll_timer = None
        self.button_index = current_button
        self._pygame_available = False
        self._joystick = None
        self._update_text()

    def _update_text(self):
        if self._capturing:
            self.setText("Press any wheel/joystick button...")
        else:
            self.setText(f"  Button {self.button_index}  (click to change)")

    def mousePressEvent(self, event):
        if self._capturing:
            # Second click cancels capture
            self._stop_capture()
            return
        self._start_capture()

    def _start_capture(self):
        """Initialize pygame and start polling for button presses."""
        try:
            import os
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            import pygame
            pygame.init()
            pygame.joystick.init()

            if pygame.joystick.get_count() == 0:
                self.setText("No device found!")
                QtCore.QTimer.singleShot(2000, self._update_text)
                return

            self._joystick = pygame.joystick.Joystick(0)
            self._joystick.init()
            self._pygame_available = True

        except ImportError:
            self.setText("pygame not installed!")
            QtCore.QTimer.singleShot(2000, self._update_text)
            return
        except Exception:
            self.setText("Device error!")
            QtCore.QTimer.singleShot(2000, self._update_text)
            return

        self._capturing = True
        self._update_text()

        # Drain any already-held buttons so we only detect fresh presses
        try:
            import pygame
            pygame.event.pump()
        except Exception:
            pass

        # Poll at 60Hz via QTimer (stays on the UI thread, no threading needed)
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.timeout.connect(self._poll_joystick)
        self._poll_timer.start(16)

    def _poll_joystick(self):
        """Check if any joystick/wheel button was pressed."""
        if not self._capturing or not self._joystick:
            return
        try:
            import pygame
            pygame.event.pump()
            num_buttons = self._joystick.get_numbuttons()
            for i in range(num_buttons):
                if self._joystick.get_button(i):
                    self.button_index = i
                    self.button_captured.emit(i)
                    self._stop_capture()
                    return
        except Exception:
            self._stop_capture()

    def _stop_capture(self):
        """Stop polling and clean up pygame."""
        self._capturing = False
        if self._poll_timer:
            self._poll_timer.stop()
            self._poll_timer = None
        if self._joystick:
            try:
                import pygame
                pygame.joystick.quit()
                pygame.quit()
            except Exception:
                pass
            self._joystick = None
        self._pygame_available = False
        self._update_text()


class PTTBindingWidget(QtWidgets.QWidget):
    """A single PTT binding slot: type selector (Keyboard/Wheel-Joystick/Disabled) + value input."""

    def __init__(self, label: str, allow_disabled: bool = False, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        slot_label = QtWidgets.QLabel(label)
        slot_label.setFixedWidth(46)
        layout.addWidget(slot_label)

        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItem("Keyboard", "keyboard")
        self.type_combo.addItem("Wheel / Joystick", "joystick")
        if allow_disabled:
            self.type_combo.addItem("--", "disabled")
        self.type_combo.setFixedWidth(150)
        layout.addWidget(self.type_combo)

        # Stacked widget: keyboard capture / joystick capture / disabled placeholder
        self.stack = QtWidgets.QStackedWidget()

        self.key_button = KeyCaptureButton("v")
        self.stack.addWidget(self.key_button)  # index 0 = keyboard

        self.joy_button = JoystickCaptureButton(11)
        self.stack.addWidget(self.joy_button)  # index 1 = joystick

        if allow_disabled:
            disabled_label = QtWidgets.QLabel("  --")
            self.stack.addWidget(disabled_label)  # index 2 = disabled

        layout.addWidget(self.stack, 1)

        self.type_combo.currentIndexChanged.connect(self._on_type_changed)

    def _on_type_changed(self, index):
        self.stack.setCurrentIndex(index)

    def get_binding(self):
        """Return (type_str, value_str)."""
        btype = self.type_combo.currentData()
        if btype == "keyboard":
            return ("keyboard", self.key_button.key_name)
        elif btype == "joystick":
            return ("joystick", str(self.joy_button.button_index))
        else:
            return ("disabled", "")

    def set_binding(self, btype: str, value: str):
        """Set the binding type and value."""
        idx = self.type_combo.findData(btype)
        if idx >= 0:
            self.type_combo.setCurrentIndex(idx)
        if btype == "keyboard" and value:
            self.key_button.key_name = value
            self.key_button._update_text()
        elif btype == "joystick" and value:
            try:
                self.joy_button.button_index = int(value)
                self.joy_button._update_text()
            except ValueError:
                pass


class LauncherWindow(QtWidgets.QDialog):
    """Setup & Settings dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Jarvis Granite - Setup & Settings")
        self.setMinimumSize(520, 420)
        self.resize(600, 520)
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
        layout.setSpacing(12)
        layout.setContentsMargins(24, 20, 24, 20)

        # ---- Title ----
        title = QtWidgets.QLabel("SETUP & SETTINGS")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet(f"""
            font-family: '{FONT_HEADING}';
            font-size: 24px;
            padding: 2px;
            background-color: transparent;
        """)
        layout.addWidget(title)

        # Keep content scrollable so buttons are always reachable.
        content_scroll = QtWidgets.QScrollArea()
        content_scroll.setWidgetResizable(True)
        content_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        content = QtWidgets.QWidget()
        content.setObjectName("scrollContent")
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setSpacing(12)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # ---- About ----
        about_group = QtWidgets.QGroupBox("ABOUT")
        about_layout = QtWidgets.QVBoxLayout(about_group)
        about_layout.setContentsMargins(14, 20, 14, 12)
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
        about_layout.addWidget(about_text)
        content_layout.addWidget(about_group)

        # ---- Voice mode ----
        voice_group = QtWidgets.QGroupBox("VOICE INPUT")
        voice_layout = QtWidgets.QVBoxLayout(voice_group)
        voice_layout.setContentsMargins(14, 20, 14, 12)

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

        # PTT binding slots
        self.ptt_key_widget = QtWidgets.QWidget()
        ptt_key_layout = QtWidgets.QVBoxLayout(self.ptt_key_widget)
        ptt_key_layout.setContentsMargins(24, 6, 0, 0)
        ptt_key_layout.setSpacing(6)

        self.ptt_slot_1 = PTTBindingWidget("Key 1:", allow_disabled=False)
        ptt_key_layout.addWidget(self.ptt_slot_1)

        self.ptt_slot_2 = PTTBindingWidget("Key 2:", allow_disabled=True)
        ptt_key_layout.addWidget(self.ptt_slot_2)

        voice_layout.addWidget(self.ptt_key_widget)

        # Show/hide PTT key based on radio selection
        self.voice_ptt_radio.toggled.connect(self.ptt_key_widget.setVisible)
        self.ptt_key_widget.setVisible(False)

        content_layout.addWidget(voice_group)
        content_layout.addStretch()
        content_scroll.setWidget(content)
        layout.addWidget(content_scroll, 1)

        # ---- Buttons (always visible, outside scroll) ----
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addStretch()

        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setFixedSize(140, 44)
        self.cancel_button.setObjectName("cancelBtn")
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

        self.ptt_slot_1.set_binding(
            c.get("ptt_slot_1_type", "keyboard"),
            c.get("ptt_slot_1_value", c.get("ptt_key", "v")),
        )
        self.ptt_slot_2.set_binding(
            c.get("ptt_slot_2_type", "disabled"),
            c.get("ptt_slot_2_value", ""),
        )

    def _save_to_config(self):
        if self.voice_ptt_radio.isChecked():
            voice_mode = "push_to_talk"
        elif self.voice_continuous_radio.isChecked():
            voice_mode = "continuous"
        else:
            voice_mode = "disabled"

        s1_type, s1_value = self.ptt_slot_1.get_binding()
        s2_type, s2_value = self.ptt_slot_2.get_binding()

        self.config.update({
            "voice_mode": voice_mode,
            "ptt_slot_1_type": s1_type,
            "ptt_slot_1_value": s1_value,
            "ptt_slot_2_type": s2_type,
            "ptt_slot_2_value": s2_value,
            "ptt_key": s1_value if s1_type == "keyboard" else (s2_value if s2_type == "keyboard" else "v"),
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
                font-family: '{FONT_BODY}';
            }}
            /* Ensure scroll area and its contents inherit dark background */
            QScrollArea {{
                background-color: {BG_COLOR};
                border: none;
            }}
            QScrollArea > QWidget > QWidget#scrollContent {{
                background-color: {BG_COLOR};
            }}
            QWidget {{
                background-color: {BG_COLOR};
            }}
            QGroupBox {{
                background-color: {BG_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                margin-top: 12px;
                padding: 18px 14px 14px 14px;
                font-family: '{FONT_HEADING}';
                font-size: 18pt;
                color: {TEXT_COLOR};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                top: 8px;
                padding: 0 6px;
            }}
            QLabel {{
                color: {TEXT_COLOR};
                font-size: 13pt;
                background-color: transparent;
            }}
            QCheckBox, QRadioButton {{
                color: {TEXT_COLOR};
                font-size: 13pt;
                spacing: 10px;
                padding: 4px 0;
                background-color: transparent;
            }}
            QCheckBox::indicator, QRadioButton::indicator {{
                width: 18px;
                height: 18px;
            }}
            QPushButton {{
                background-color: {ACCENT_PRIMARY};
                color: #FFFFFF;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: bold;
                font-size: 13pt;
            }}
            QPushButton:hover {{
                background-color: #C00500;
            }}
            QPushButton:pressed {{
                background-color: #A00400;
            }}
            QPushButton#cancelBtn {{
                background-color: {BG_COLOR_LIGHT};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
            }}
            QPushButton#cancelBtn:hover {{
                background-color: #333333;
            }}
            QPushButton#cancelBtn:pressed {{
                background-color: #2a2a2a;
            }}
        """)
