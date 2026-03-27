"""
Unified launcher - starting point for Jarvis Live and Jarvis Post.

Provides three options:
1. Start Jarvis Live - Launch live telemetry dashboard
2. Start Jarvis Post - Open session picker then post-race analysis
3. Settings - Configure voice input and controls
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

# Google Drive download constants
_GDRIVE_FILE_ID = "1YaFC-wgbjQ1v6dSYlWyufD7g33qlhoEs"
_GDRIVE_BASE_URL = "https://drive.usercontent.google.com/download"


class _GDriveDownloadThread(QtCore.QThread):
    """Download a file from Google Drive with large-file confirmation handling."""

    progress = QtCore.pyqtSignal(int, int)      # downloaded_bytes, total_bytes
    finished_ok = QtCore.pyqtSignal(str)         # dest_path
    error = QtCore.pyqtSignal(str)               # error message

    def __init__(self, file_id: str, dest_path: str, parent=None):
        super().__init__(parent)
        self.file_id = file_id
        self.dest_path = dest_path
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        import requests
        try:
            session = requests.Session()
            # confirm=t bypasses the virus-scan interstitial for large files
            resp = session.get(
                _GDRIVE_BASE_URL,
                params={"id": self.file_id, "export": "download", "confirm": "t"},
                stream=True,
                timeout=30,
            )
            resp.raise_for_status()

            # Validate we got a file, not an HTML error page
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type:
                resp.close()
                self.error.emit(
                    "Google Drive did not serve the file. "
                    "It may have been removed or the link may be invalid."
                )
                return

            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 64 * 1024

            with open(self.dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size):
                    if self._abort:
                        break
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        self.progress.emit(downloaded, total)

            resp.close()

            if self._abort:
                try:
                    os.remove(self.dest_path)
                except OSError:
                    pass
                return

            self.finished_ok.emit(self.dest_path)

        except Exception as e:
            try:
                os.remove(self.dest_path)
            except OSError:
                pass
            self.error.emit(str(e))


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
        self.setFixedSize(500, 600)
        self.setModal(True)

        self._action = self.ACTION_QUIT

        self._build_ui()
        self._apply_theme()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(40, 36, 40, 28)

        header_row = QtWidgets.QHBoxLayout()
        header_row.addStretch()

        self.quit_btn = QtWidgets.QPushButton("Quit")
        self.quit_btn.setFixedSize(72, 32)
        self.quit_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.quit_btn.setToolTip("Exit Jarvis")
        self.quit_btn.setStyleSheet(f"""
            QPushButton {{
                font-family: '{FONT_BODY}';
                background-color: {BG_COLOR_LIGHT};
                color: {TEXT_COLOR_DIM};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                font-size: 10pt;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: #252525; color: {TEXT_COLOR}; }}
            QPushButton:pressed {{ background-color: #202020; }}
        """)
        self.quit_btn.clicked.connect(self._on_quit)
        header_row.addWidget(self.quit_btn)
        layout.addLayout(header_row)

        # Logo + title grouped so they move together when layout shifts
        header_group = QtWidgets.QVBoxLayout()
        header_group.setSpacing(0)

        logo_path = os.path.join(os.path.dirname(__file__), "img", "f1_jarvis_topdown_massive_tyres.svg")
        logo = QtSvg.QSvgWidget(logo_path)
        logo.setFixedSize(180, 180)
        logo_container = QtWidgets.QHBoxLayout()
        logo_container.addStretch()
        logo_container.addWidget(logo)
        logo_container.addStretch()
        header_group.addLayout(logo_container)

        subtitle = QtWidgets.QLabel("F1 JARVIS GRANITE")
        subtitle.setAlignment(QtCore.Qt.AlignCenter)
        subtitle.setStyleSheet(f"""
            font-family: '{FONT_HEADING}';
            font-size: 24px;
            color: {TEXT_COLOR_DIM};
            letter-spacing: 4px;
        """)
        header_group.addWidget(subtitle)

        layout.addLayout(header_group)
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
            QPushButton:disabled {{
                background-color: #333333;
                color: #666666;
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

        # Jarvis VR Download
        self.vr_btn = QtWidgets.QPushButton("Download Jarvis VR (~3.6GB)")
        self.vr_btn.setFixedHeight(62)
        self.vr_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.vr_btn.setStyleSheet(btn_style.format(
            bg=_RED_DIM, hover="#7A0000", pressed="#600000"
        ))
        self.vr_btn.setToolTip("Download Jarvis VR application (zip file)")
        self.vr_btn.clicked.connect(self._on_vr_download)
        layout.addWidget(self.vr_btn)

        # Download progress widgets (hidden by default)
        self._dl_progress = QtWidgets.QProgressBar()
        self._dl_progress.setFixedHeight(18)
        self._dl_progress.setVisible(False)
        self._dl_progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: {BG_COLOR_LIGHT};
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                text-align: center;
                color: {TEXT_COLOR_DIM};
                font-size: 9pt;
            }}
            QProgressBar::chunk {{
                background-color: {_RED_DIM};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(self._dl_progress)

        self._dl_status = QtWidgets.QLabel("")
        self._dl_status.setAlignment(QtCore.Qt.AlignCenter)
        self._dl_status.setStyleSheet(f"color: {TEXT_COLOR_DIM}; font-size: 9pt;")
        self._dl_status.setVisible(False)
        layout.addWidget(self._dl_status)

        self._dl_cancel_btn = QtWidgets.QPushButton("Cancel Download")
        self._dl_cancel_btn.setFixedHeight(28)
        self._dl_cancel_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self._dl_cancel_btn.setStyleSheet(f"""
            QPushButton {{
                font-family: '{FONT_BODY}';
                background-color: {BG_COLOR_LIGHT};
                color: {TEXT_COLOR_DIM};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                font-size: 9pt;
            }}
            QPushButton:hover {{ background-color: #252525; color: {TEXT_COLOR}; }}
            QPushButton:pressed {{ background-color: #202020; }}
        """)
        self._dl_cancel_btn.setVisible(False)
        self._dl_cancel_btn.clicked.connect(self._on_dl_cancel)
        layout.addWidget(self._dl_cancel_btn)

        self._dl_thread = None

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
        self.settings_btn.setToolTip("Setup & Settings - Configure voice input and controls")
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

    def _on_quit(self):
        self._action = self.ACTION_QUIT
        self.accept()

    # -- VR download handlers --------------------------------------------------

    def _on_vr_download(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Jarvis VR",
            os.path.join(os.path.expanduser("~"), "Downloads", "JarvisVR.zip"),
            "ZIP Archive (*.zip);;All Files (*)",
        )
        if not path:
            return

        # Disable buttons during download
        for btn in (self.live_btn, self.post_btn, self.vr_btn, self.settings_btn):
            btn.setEnabled(False)

        # Show progress widgets
        self._dl_progress.setValue(0)
        self._dl_progress.setRange(0, 0)  # indeterminate until we know total
        self._dl_progress.setVisible(True)
        self._dl_status.setText("Starting download...")
        self._dl_status.setVisible(True)
        self._dl_cancel_btn.setVisible(True)

        self._dl_thread = _GDriveDownloadThread(_GDRIVE_FILE_ID, path, parent=self)
        self._dl_thread.progress.connect(self._on_dl_progress)
        self._dl_thread.finished_ok.connect(self._on_dl_success)
        self._dl_thread.error.connect(self._on_dl_error)
        self._dl_thread.start()

    def _on_dl_progress(self, downloaded: int, total: int):
        if total > 0:
            self._dl_progress.setRange(0, total)
            self._dl_progress.setValue(downloaded)
            self._dl_status.setText(
                f"Downloading... {downloaded / 1048576:.1f} MB / {total / 1048576:.1f} MB"
            )
        else:
            self._dl_progress.setRange(0, 0)
            self._dl_status.setText(f"Downloading... {downloaded / 1048576:.1f} MB")

    def _on_dl_success(self, path: str):
        self._dl_hide_progress()
        QtWidgets.QMessageBox.information(
            self, "Download Complete", f"Jarvis VR saved to:\n{path}"
        )

    def _on_dl_error(self, message: str):
        self._dl_hide_progress()
        QtWidgets.QMessageBox.warning(
            self, "Download Failed", f"Could not download Jarvis VR:\n{message}"
        )

    def _on_dl_cancel(self):
        if self._dl_thread and self._dl_thread.isRunning():
            self._dl_thread.abort()
            self._dl_thread.wait(5000)
        self._dl_hide_progress()

    def _dl_hide_progress(self):
        self._dl_progress.setVisible(False)
        self._dl_status.setVisible(False)
        self._dl_cancel_btn.setVisible(False)
        for btn in (self.live_btn, self.post_btn, self.vr_btn, self.settings_btn):
            btn.setEnabled(True)

    def reject(self):
        if self._dl_thread and self._dl_thread.isRunning():
            self._dl_thread.abort()
            self._dl_thread.wait(5000)
        super().reject()

    # --------------------------------------------------------------------------

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
