"""
Session picker dialog - lets the user select a recorded session for post-race analysis.
"""
import logging
from PyQt5 import QtWidgets, QtCore
from data.session_exporter import SessionExporter
from ui.styles import (
    BG_COLOR, BG_COLOR_LIGHT, TEXT_COLOR,
    BORDER_COLOR, ACCENT_PRIMARY,
)

logger = logging.getLogger(__name__)


class SessionPickerDialog(QtWidgets.QDialog):
    """Dialog for selecting a recorded session to analyze in Jarvis Post."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Session - Jarvis Post")
        self.setMinimumSize(800, 500)
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
        title = QtWidgets.QLabel("Select a recorded session for post-race analysis")
        title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 8px;")
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
            "ID", "Date", "Track", "Car", "Mode", "Duration", "Laps", "Best Lap", "AI"
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
        header.resizeSection(0, 40)   # ID
        header.resizeSection(1, 130)  # Date
        header.resizeSection(2, 120)  # Track
        header.resizeSection(3, 100)  # Car
        header.resizeSection(4, 70)   # Mode
        header.resizeSection(5, 70)   # Duration
        header.resizeSection(6, 45)   # Laps
        header.resizeSection(7, 80)   # Best Lap
        header.resizeSection(8, 35)   # AI

        layout.addWidget(self.table)

        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()

        self.delete_btn = QtWidgets.QPushButton("Delete Session")
        self.delete_btn.setFixedSize(130, 36)
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
            self.table.setSpan(0, 0, 1, 9)
            return

        self.table.setRowCount(len(sessions))

        for row_idx, session in enumerate(sessions):
            # ID
            id_item = QtWidgets.QTableWidgetItem(str(session["session_id"]))
            id_item.setData(QtCore.Qt.UserRole, session["session_id"])
            id_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(row_idx, 0, id_item)

            # Date
            date_str = ""
            if session["start_time"]:
                date_str = session["start_time"].strftime("%Y-%m-%d %H:%M")
            self.table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(date_str))

            # Track
            self.table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(session["track_name"]))

            # Car
            self.table.setItem(row_idx, 3, QtWidgets.QTableWidgetItem(session["car_model"]))

            # Mode
            mode_item = QtWidgets.QTableWidgetItem(session.get("session_type", "") or "")
            mode_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(row_idx, 4, mode_item)

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
            self.table.setItem(row_idx, 5, duration_item)

            # Laps
            laps_item = QtWidgets.QTableWidgetItem(str(session["total_laps"]))
            laps_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(row_idx, 6, laps_item)

            # Best Lap
            best_str = ""
            if session["best_lap_time"]:
                best_str = f"{session['best_lap_time']:.3f}s"
            best_item = QtWidgets.QTableWidgetItem(best_str)
            best_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(row_idx, 7, best_item)

            # AI
            ai_str = "Yes" if session["ai_enabled"] else "No"
            ai_item = QtWidgets.QTableWidgetItem(ai_str)
            ai_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(row_idx, 8, ai_item)

    def _on_selection_changed(self):
        selected = self.table.selectedItems()
        has_selection = len(selected) > 0
        self.open_btn.setEnabled(has_selection)
        self.delete_btn.setEnabled(has_selection)

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
