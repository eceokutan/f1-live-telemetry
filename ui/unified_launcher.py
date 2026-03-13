"""
Unified launcher - starting point for Jarvis Live and Jarvis Post.

Provides three options:
1. Start Jarvis Live - Launch live telemetry dashboard
2. Start Jarvis Post - Open session picker then post-race analysis
3. Settings - Configure AI, voice, credentials
"""
import logging
from PyQt5 import QtWidgets, QtCore
from ui.styles import (
    BG_COLOR, BG_COLOR_LIGHT, TEXT_COLOR, TEXT_COLOR_DIM,
    BORDER_COLOR, ACCENT_BLUE, ACCENT_GREEN, ACCENT_CYAN,
)

logger = logging.getLogger(__name__)


class UnifiedLauncher(QtWidgets.QDialog):
    """Main launcher dialog with options for Live, Post-Race, and Settings."""

    # Signals to communicate user choice back to main.py
    ACTION_LIVE = "live"
    ACTION_POST = "post"
    ACTION_SETTINGS = "settings"
    ACTION_QUIT = "quit"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Jarvis - F1 Telemetry Suite")
        self.setFixedSize(480, 400)
        self.setModal(True)

        self._action = self.ACTION_QUIT

        self._build_ui()
        self._apply_theme()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(30, 30, 30, 30)

        # Title
        title = QtWidgets.QLabel("JARVIS")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet(f"""
            font-size: 32px;
            font-weight: bold;
            color: {ACCENT_BLUE};
            letter-spacing: 6px;
            padding: 8px;
        """)
        layout.addWidget(title)

        subtitle = QtWidgets.QLabel("F1 Telemetry Suite")
        subtitle.setAlignment(QtCore.Qt.AlignCenter)
        subtitle.setStyleSheet(f"""
            font-size: 13px;
            color: {TEXT_COLOR_DIM};
            padding-bottom: 16px;
        """)
        layout.addWidget(subtitle)

        # Buttons
        btn_style_template = """
            QPushButton {{
                background-color: {bg};
                color: #FFFFFF;
                border: none;
                border-radius: 6px;
                padding: 14px;
                font-size: 13pt;
                font-weight: bold;
                text-align: left;
                padding-left: 20px;
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
        self.live_btn.setFixedHeight(60)
        self.live_btn.setStyleSheet(btn_style_template.format(
            bg=ACCENT_BLUE, hover="#5A98EF", pressed="#4A88DF"
        ))
        self.live_btn.setToolTip("Launch real-time telemetry dashboard for Assetto Corsa")
        self.live_btn.clicked.connect(self._on_live)
        layout.addWidget(self.live_btn)

        # Start Post
        self.post_btn = QtWidgets.QPushButton("Start Jarvis Post")
        self.post_btn.setFixedHeight(60)
        self.post_btn.setStyleSheet(btn_style_template.format(
            bg=ACCENT_GREEN, hover="#5BBB67", pressed="#4BAA57"
        ))
        self.post_btn.setToolTip("Analyze a recorded session with post-race telemetry viewer")
        self.post_btn.clicked.connect(self._on_post)
        layout.addWidget(self.post_btn)

        # Settings
        self.settings_btn = QtWidgets.QPushButton("Settings")
        self.settings_btn.setFixedHeight(50)
        self.settings_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {BG_COLOR_LIGHT};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                padding: 10px;
                font-size: 11pt;
                text-align: left;
                padding-left: 20px;
            }}
            QPushButton:hover {{ background-color: #282828; }}
            QPushButton:pressed {{ background-color: #222222; }}
        """)
        self.settings_btn.setToolTip("Configure AI Race Engineer, voice input, and API credentials")
        self.settings_btn.clicked.connect(self._on_settings)
        layout.addWidget(self.settings_btn)

        layout.addStretch()

        # Version / footer
        footer = QtWidgets.QLabel("Team 17 - Systems Course")
        footer.setAlignment(QtCore.Qt.AlignCenter)
        footer.setStyleSheet(f"font-size: 9px; color: {TEXT_COLOR_DIM}; padding: 4px;")
        layout.addWidget(footer)

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
