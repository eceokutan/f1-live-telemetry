"""
Unified launcher - starting point for Jarvis Live and Jarvis Post.

Provides three options:
1. Start Jarvis Live - Launch live telemetry dashboard
2. Start Jarvis Post - Open session picker then post-race analysis
3. Settings - Configure AI, voice, credentials
"""
import logging
import os
from PyQt5 import QtWidgets, QtCore, QtGui, QtSvg
from ui.styles import (
    BG_COLOR, BG_COLOR_LIGHT, TEXT_COLOR, TEXT_COLOR_DIM,
    BORDER_COLOR, FONT_HEADING, FONT_BODY,
    load_fonts,
)

logger = logging.getLogger(__name__)

# Red palette for launcher
_RED_DIM = "#8B0000"         # Muted red for buttons


class UnifiedLauncher(QtWidgets.QDialog):
    """Main launcher dialog with options for Live, Post-Race, and Settings."""

    ACTION_LIVE = "live"
    ACTION_POST = "post"
    ACTION_SETTINGS = "settings"
    ACTION_QUIT = "quit"

    def __init__(self, parent=None):
        super().__init__(parent)
        load_fonts()
        self.setWindowTitle("Jarvis - F1 Telemetry Suite")
        self.setFixedSize(500, 520)
        self.setModal(True)

        self._action = self.ACTION_QUIT

        self._build_ui()
        self._apply_theme()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(40, 36, 40, 28)

        # Logo
        logo_path = os.path.join(os.path.dirname(__file__), "img", "f1_jarvis_topdown_massive_tyres.svg")
        logo = QtSvg.QSvgWidget(logo_path)
        logo.setFixedSize(180, 180)
        logo_container = QtWidgets.QHBoxLayout()
        logo_container.addStretch()
        logo_container.addWidget(logo)
        logo_container.addStretch()
        layout.addLayout(logo_container)

        subtitle = QtWidgets.QLabel("F1 JARVIS GRANITE")
        subtitle.setAlignment(QtCore.Qt.AlignCenter)
        subtitle.setStyleSheet(f"""
            font-family: '{FONT_HEADING}';
            font-size: 24px;
            color: {TEXT_COLOR_DIM};
            letter-spacing: 4px;
        """)
        layout.addWidget(subtitle)

        layout.addSpacing(6)

        # Button template
        btn_style = """
            QPushButton {{
                font-family: '""" + FONT_BODY + """';
                background-color: {bg};
                color: #FFFFFF;
                border: none;
                border-radius: 8px;
                padding: 14px;
                font-size: 16pt;
                font-weight: bold;
                text-align: left;
                padding-left: 24px;
            }}
            QPushButton:hover {{
                background-color: {hover};
            }}
            QPushButton:pressed {{
                background-color: {pressed};
            }}
        """

        # Start Live
        self.live_btn = QtWidgets.QPushButton("Start Jarvis Live")
        self.live_btn.setFixedHeight(62)
        self.live_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.live_btn.setStyleSheet(btn_style.format(
            bg=_RED_DIM, hover="#7A0000", pressed="#600000"
        ))
        self.live_btn.setToolTip("Launch real-time telemetry dashboard for Assetto Corsa")
        self.live_btn.clicked.connect(self._on_live)
        layout.addWidget(self.live_btn)

        # Start Post
        self.post_btn = QtWidgets.QPushButton("Start Jarvis Post")
        self.post_btn.setFixedHeight(62)
        self.post_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.post_btn.setStyleSheet(btn_style.format(
            bg=_RED_DIM, hover="#7A0000", pressed="#600000"
        ))
        self.post_btn.setToolTip("Analyze a recorded session with post-race telemetry viewer")
        self.post_btn.clicked.connect(self._on_post)
        layout.addWidget(self.post_btn)

        # Settings
        self.settings_btn = QtWidgets.QPushButton("Setup & Settings")
        self.settings_btn.setFixedHeight(48)
        self.settings_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.settings_btn.setStyleSheet(f"""
            QPushButton {{
                font-family: '{FONT_BODY}';
                background-color: {BG_COLOR_LIGHT};
                color: {TEXT_COLOR_DIM};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                padding: 10px;
                font-size: 11pt;
                text-align: left;
                padding-left: 24px;
            }}
            QPushButton:hover {{ background-color: #252525; color: {TEXT_COLOR}; }}
            QPushButton:pressed {{ background-color: #202020; }}
        """)
        self.settings_btn.setToolTip("Setup & Settings - Configure AI Race Engineer, voice input, and API credentials")
        self.settings_btn.clicked.connect(self._on_settings)
        layout.addWidget(self.settings_btn)

        layout.addStretch()


    def _on_live(self):
        self._action = self.ACTION_LIVE
        self.accept()

    def _on_post(self):
        self._action = self.ACTION_POST
        self.accept()

    def _on_settings(self):
        self._action = self.ACTION_SETTINGS
        self.accept()

    def get_action(self) -> str:
        """Return which action the user chose."""
        return self._action

    def _apply_theme(self):
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {BG_COLOR};
                color: {TEXT_COLOR};
            }}
            QLabel {{
                color: {TEXT_COLOR};
                background-color: transparent;
            }}
        """)
