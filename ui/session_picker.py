"""
Session picker dialog - lets the user select a recorded session for post-race analysis.
"""
import os
import logging
from PyQt5 import QtWidgets, QtCore
from data.session_exporter import SessionExporter
from ui.styles import (
    BG_COLOR, BG_COLOR_LIGHT, TEXT_COLOR,
    BORDER_COLOR, ACCENT_PRIMARY, FONT_HEADING, FONT_BODY,
)

logger = logging.getLogger(__name__)


class SessionPickerDialog(QtWidgets.QDialog):
    """Dialog for selecting a recorded session to analyze in Jarvis Post."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Session - Jarvis Post")
        self.setMinimumSize(900, 500)
        self.setModal(True)

        self._selected_session_id = None
        self._use_last = False
        self._accepted = False

        self.exporter = SessionExporter()

        self._build_ui()
        self._apply_theme()
        self._load_sessions()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        # Title
        title = QtWidgets.QLabel("SELECT A SESSION")
        title.setStyleSheet(f"font-family: '{FONT_HEADING}'; font-size: 20px; padding: 8px;")
        layout.addWidget(title)

        # Quick action: use last recorded
        self.use_last_btn = QtWidgets.QPushButton("Use Last Recorded Session")
        self.use_last_btn.setFixedHeight(40)
        self.use_last_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_PRIMARY};
                color: #FFFFFF;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 11pt;
            }}
            QPushButton:hover {{ background-color: #C00500; }}
            QPushButton:pressed {{ background-color: #A00400; }}
            QPushButton:disabled {{ background-color: #555555; color: #888888; }}
        """)
        self.use_last_btn.clicked.connect(self._on_use_last)
        layout.addWidget(self.use_last_btn)

        # Separator
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setStyleSheet(f"color: {BORDER_COLOR};")
        layout.addWidget(separator)

        # Or pick from list
        pick_label = QtWidgets.QLabel("Or select from all sessions:")
        pick_label.setStyleSheet("font-size: 11px; color: #AAAAAA; padding: 4px;")
        layout.addWidget(pick_label)

        # Session table
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "ID", "Name", "Date", "Track", "Car", "Mode", "Duration", "Laps", "Best Lap"
        ])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.doubleClicked.connect(self._on_double_click)

        # Set column widths
        header = self.table.horizontalHeader()
        header.resizeSection(0, 35)   # ID
        header.resizeSection(1, 120)  # Name
        header.resizeSection(2, 120)  # Date
        header.resizeSection(3, 110)  # Track
        header.resizeSection(4, 90)   # Car
        header.resizeSection(5, 60)   # Mode
        header.resizeSection(6, 60)   # Duration
        header.resizeSection(7, 40)   # Laps
        header.resizeSection(8, 75)   # Best Lap
        header.resizeSection(9, 30)   # AI

        layout.addWidget(self.table)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()

        # Destructive / session management buttons (left side)
        action_btn_style = f"""
            QPushButton {{
                background-color: {BG_COLOR_LIGHT};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                font-size: 10pt;
            }}
            QPushButton:hover {{ background-color: #282828; }}
            QPushButton:pressed {{ background-color: #202020; }}
            QPushButton:disabled {{ background-color: #333333; color: #666666; }}
        """

        self.delete_btn = QtWidgets.QPushButton("Delete")
        self.delete_btn.setFixedSize(90, 36)
        self.delete_btn.setEnabled(False)
        self.delete_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #8B2020;
                color: {TEXT_COLOR};
                border: none;
                border-radius: 4px;
                font-size: 10pt;
            }}
            QPushButton:hover {{ background-color: #A03030; }}
            QPushButton:pressed {{ background-color: #701818; }}
            QPushButton:disabled {{ background-color: #333333; color: #666666; }}
        """)
        self.delete_btn.clicked.connect(self._on_delete)
        btn_layout.addWidget(self.delete_btn)

        self.rename_btn = QtWidgets.QPushButton("Rename")
        self.rename_btn.setFixedSize(90, 36)
        self.rename_btn.setEnabled(False)
        self.rename_btn.setStyleSheet(action_btn_style)
        self.rename_btn.clicked.connect(self._on_rename)
        btn_layout.addWidget(self.rename_btn)

        self.export_session_btn = QtWidgets.QPushButton("Export Session")
        self.export_session_btn.setFixedSize(120, 36)
        self.export_session_btn.setEnabled(False)
        self.export_session_btn.setStyleSheet(action_btn_style)
        self.export_session_btn.clicked.connect(self._on_export_session)
        btn_layout.addWidget(self.export_session_btn)

        self.import_session_btn = QtWidgets.QPushButton("Import Session")
        self.import_session_btn.setFixedSize(120, 36)
        self.import_session_btn.setStyleSheet(action_btn_style)
        self.import_session_btn.clicked.connect(self._on_import_session)
        btn_layout.addWidget(self.import_session_btn)

        btn_layout.addStretch()

        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setFixedSize(100, 36)
        self.cancel_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {BG_COLOR_LIGHT};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                font-size: 10pt;
            }}
            QPushButton:hover {{ background-color: #282828; }}
        """)
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        self.open_btn = QtWidgets.QPushButton("Open Session")
        self.open_btn.setFixedSize(130, 36)
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._on_open)
        btn_layout.addWidget(self.open_btn)

        layout.addLayout(btn_layout)

        # Enable buttons when row is selected
        self.table.itemSelectionChanged.connect(self._on_selection_changed)

    def _load_sessions(self):
        """Load sessions from database into the table."""
        sessions = self.exporter.list_sessions()

        if not sessions:
            self.use_last_btn.setEnabled(False)
            self.table.setRowCount(1)
            no_data = QtWidgets.QTableWidgetItem("No recorded sessions found")
            no_data.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(0, 0, no_data)
            self.table.setSpan(0, 0, 1, 10)
            return

        self.table.setRowCount(len(sessions))

        for row_idx, session in enumerate(sessions):
            # ID
            id_item = QtWidgets.QTableWidgetItem(str(session["session_id"]))
            id_item.setData(QtCore.Qt.UserRole, session["session_id"])
            id_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(row_idx, 0, id_item)

            # Name (from notes field)
            custom_name = (session.get("notes", "") or "").strip()
            name_str = custom_name or f"Session {session['session_id']}"
            self.table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(name_str))

            # Date
            date_str = ""
            if session["start_time"]:
                date_str = session["start_time"].strftime("%Y-%m-%d %H:%M")
            self.table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(date_str))

            # Track
            self.table.setItem(row_idx, 3, QtWidgets.QTableWidgetItem(session["track_name"]))

            # Car
            self.table.setItem(row_idx, 4, QtWidgets.QTableWidgetItem(session["car_model"]))

            # Mode
            mode_item = QtWidgets.QTableWidgetItem(session.get("session_type", "") or "")
            mode_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(row_idx, 5, mode_item)

            # Duration
            duration_str = ""
            if session["start_time"] and session["end_time"]:
                delta = session["end_time"] - session["start_time"]
                total_secs = int(delta.total_seconds())
                if total_secs >= 3600:
                    duration_str = f"{total_secs // 3600}h {(total_secs % 3600) // 60}m"
                elif total_secs >= 60:
                    duration_str = f"{total_secs // 60}m {total_secs % 60}s"
                else:
                    duration_str = f"{total_secs}s"
            duration_item = QtWidgets.QTableWidgetItem(duration_str)
            duration_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(row_idx, 6, duration_item)

            # Laps
            laps_item = QtWidgets.QTableWidgetItem(str(session["total_laps"]))
            laps_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(row_idx, 7, laps_item)

            # Best Lap
            best_str = ""
            if session["best_lap_time"]:
                best_str = f"{session['best_lap_time']:.3f}s"
            best_item = QtWidgets.QTableWidgetItem(best_str)
            best_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(row_idx, 8, best_item)

    def _on_selection_changed(self):
        selected = self.table.selectedItems()
        has_selection = len(selected) > 0
        self.open_btn.setEnabled(has_selection)
        self.delete_btn.setEnabled(has_selection)
        self.rename_btn.setEnabled(has_selection)
        self.export_session_btn.setEnabled(has_selection)

    def _on_double_click(self):
        """Open session on double-click."""
        self._on_open()

    def _on_use_last(self):
        """Use the most recent session."""
        last_id = self.exporter.get_last_session_id()
        if last_id is not None:
            self._selected_session_id = last_id
            self._use_last = True
            self._accepted = True
            self.accept()

    def _on_open(self):
        """Open the selected session."""
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        id_item = self.table.item(row, 0)
        if id_item:
            session_id = id_item.data(QtCore.Qt.UserRole)
            if session_id is not None:
                self._selected_session_id = session_id
                self._accepted = True
                self.accept()

    def _on_delete(self):
        """Delete the selected session after confirmation."""
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        id_item = self.table.item(row, 0)
        if not id_item:
            return

        session_id = id_item.data(QtCore.Qt.UserRole)
        if session_id is None:
            return

        # Confirm deletion
        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete Session",
            f"Delete session {session_id}? This will permanently remove all telemetry, laps, and AI commentary for this session.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )

        if reply == QtWidgets.QMessageBox.Yes:
            self.exporter.delete_session(session_id)
            # Reload the table
            self.table.clearContents()
            self.table.setRowCount(0)
            self.table.clearSpans()
            self._load_sessions()

    def _on_rename(self):
        """Rename the selected session."""
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        id_item = self.table.item(row, 0)
        if not id_item:
            return

        session_id = id_item.data(QtCore.Qt.UserRole)
        if session_id is None:
            return

        current_name = self.table.item(row, 1).text() if self.table.item(row, 1) else ""
        new_name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename Session",
            f"Enter a name for session {session_id}:",
            text=current_name
        )
        if ok and new_name is not None:
            self.exporter.rename_session(session_id, new_name.strip())
            self.table.clearContents()
            self.table.setRowCount(0)
            self.table.clearSpans()
            self._load_sessions()

    def _on_export_session(self):
        """Export the selected session to a .jsession file."""
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return

        row = selected_rows[0].row()
        id_item = self.table.item(row, 0)
        if not id_item:
            return

        session_id = id_item.data(QtCore.Qt.UserRole)
        if session_id is None:
            return

        track = self.table.item(row, 3).text() if self.table.item(row, 3) else "session"
        default_name = f"session_{session_id}_{track}.jsession"
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Session", default_name,
            "Jarvis Session File (*.jsession);;All Files (*)"
        )
        if not file_path:
            return

        try:
            self.exporter.export_session_bundle(session_id, file_path)
            QtWidgets.QMessageBox.information(
                self, "Export Complete",
                f"Session {session_id} exported to {os.path.basename(file_path)}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export session:\n{str(e)}")

    def _on_import_session(self):
        """Import a .jsession file into the database."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import Session", "",
            "Jarvis Session File (*.jsession);;All Files (*)"
        )
        if not file_path:
            return

        try:
            new_id = self.exporter.import_session_bundle(file_path)
            QtWidgets.QMessageBox.information(
                self, "Import Complete",
                f"Session imported as ID {new_id} from {os.path.basename(file_path)}"
            )
            self.table.clearContents()
            self.table.setRowCount(0)
            self.table.clearSpans()
            self._load_sessions()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Import Error", f"Failed to import session:\n{str(e)}")

    def get_selected_session_id(self) -> int:
        """Return the selected session ID (call after exec_())."""
        return self._selected_session_id

    def was_accepted(self) -> bool:
        return self._accepted

    def _apply_theme(self):
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {BG_COLOR};
                color: {TEXT_COLOR};
                font-family: '{FONT_BODY}';
            }}
            QLabel {{
                color: {TEXT_COLOR};
            }}
            QTableWidget {{
                background-color: {BG_COLOR_LIGHT};
                color: {TEXT_COLOR};
                gridline-color: {BORDER_COLOR};
                border: 1px solid {BORDER_COLOR};
                alternate-background-color: #1E1E1E;
            }}
            QTableWidget::item:selected {{
                background-color: {ACCENT_PRIMARY};
            }}
            QHeaderView::section {{
                background-color: {BG_COLOR};
                color: {TEXT_COLOR};
                padding: 6px;
                border: 1px solid {BORDER_COLOR};
                font-weight: bold;
            }}
            QFrame {{
                color: {BORDER_COLOR};
            }}
        """)
