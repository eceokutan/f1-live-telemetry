"""
Main window for F1 Telemetry Dashboard.
"""
import sys
import numpy as np
from datetime import datetime

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QGroupBox,
    QTextEdit,
)

from ui.canvases import TrackMapCanvas, TimeSeriesCanvas, MultiLineCanvas
from ui.styles import DARK_STYLESHEET


class MainWindow(QMainWindow):
    """
    Main dashboard window for F1 telemetry visualization.

    Displays:
    - Track map with speed-colored path
    - Live telemetry graphs (speed, gear, RPM, brake, tire pressure/temp)
    - Lap times table
    - Session information
    - AI commentary transcripts
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("F1 Telemetry Dashboard")
        self.resize(1600, 900)

        # Real-time data buffers for current lap
        self.current_lap_samples = []
        self.current_lap_id = None

        # Create central widget and root layout
        central = QWidget()
        self.setCentralWidget(central)

        # Create menu bar
        self._create_menu_bar()

        # Root layout: horizontal split into left / middle / right
        root_layout = QHBoxLayout()
        root_layout.setContentsMargins(8, 24, 8, 8)
        root_layout.setSpacing(10)
        central.setLayout(root_layout)

        # Build UI sections
        left_col = self._build_left_column()
        mid_col = self._build_middle_column()
        right_col = self._build_right_column()

        # Add columns to root layout
        root_layout.addLayout(left_col, 3)   # Track map + lap table
        root_layout.addLayout(mid_col, 5)    # Graphs
        root_layout.addLayout(right_col, 2)  # Session info + transcripts

        # Apply dark theme
        self.setStyleSheet(DARK_STYLESHEET)

    def _build_left_column(self):
        """Build left column: track map + lap times table."""
        left_col = QVBoxLayout()
        left_col.setSpacing(10)

        # Track map
        track_group = QGroupBox("Location Map")
        track_layout = QVBoxLayout()
        track_group.setLayout(track_layout)
        self.track_canvas = TrackMapCanvas(self, width=5, height=4, dpi=100)
        track_layout.addWidget(self.track_canvas)

        # Lap times table
        lap_group = QGroupBox("Lap Times")
        lap_layout = QVBoxLayout()
        lap_group.setLayout(lap_layout)

        self.lap_table = QTableWidget(8, 2)
        self.lap_table.setHorizontalHeaderLabels(["Lap Time", "Delta"])
        self.lap_table.verticalHeader().setVisible(True)
        self.lap_table.horizontalHeader().setStretchLastSection(True)
        self.lap_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.lap_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.lap_table.setColumnWidth(0, 100)

        # Initialize empty rows
        for i in range(8):
            lap_label = QTableWidgetItem(f"{i+1}.")
            self.lap_table.setVerticalHeaderItem(i, lap_label)
            self.lap_table.setItem(i, 0, QTableWidgetItem("--:--:---"))
            self.lap_table.setItem(i, 1, QTableWidgetItem("----"))

        lap_layout.addWidget(self.lap_table)

        left_col.addWidget(track_group)
        left_col.addWidget(lap_group)

        return left_col

    def _build_middle_column(self):
        """Build middle column: telemetry graphs."""
        mid_col = QVBoxLayout()
        mid_col.setSpacing(4)

        title_label = QLabel("Live Telemetry Analysis")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        mid_col.addWidget(title_label)

        # Create canvases
        self.speed_canvas = TimeSeriesCanvas("Speed [km/h]", self)
        self.gear_canvas = TimeSeriesCanvas("Gear", self)
        self.rpm_canvas = TimeSeriesCanvas("RPM", self)
        self.brake_canvas = TimeSeriesCanvas("Brake [%]", self)

        self.tyre_pressure_canvas = MultiLineCanvas(
            "Tyre Pressure [PSI]", ["FL", "FR", "RL", "RR"], self
        )
        self.tyre_temp_canvas = MultiLineCanvas(
            "Tyre Temperature [°C]", ["FL", "FR", "RL", "RR"], self
        )

        # Add to layout
        mid_col.addWidget(self.speed_canvas)
        mid_col.addWidget(self.gear_canvas)
        mid_col.addWidget(self.rpm_canvas)
        mid_col.addWidget(self.brake_canvas)
        mid_col.addWidget(self.tyre_pressure_canvas)
        mid_col.addWidget(self.tyre_temp_canvas)

        # Add graph button (placeholder)
        add_graph_btn = QPushButton("+ Add Graph")
        add_graph_btn.setFixedHeight(24)
        mid_col.addWidget(add_graph_btn)

        return mid_col

    def _build_right_column(self):
        """Build right column: session info + transcripts."""
        right_col = QVBoxLayout()
        right_col.setSpacing(10)

        # Session info panel
        driver_group = QGroupBox("Session Info")
        driver_layout = QVBoxLayout()
        driver_layout.setSpacing(2)
        driver_group.setLayout(driver_layout)

        self.driver_name_label = QLabel("Driver: ----")
        self.car_label = QLabel("Car: ----")
        self.track_label = QLabel("Track: ----")

        driver_layout.addWidget(self.driver_name_label)
        driver_layout.addWidget(self.car_label)
        driver_layout.addWidget(self.track_label)

        # Separator
        driver_layout.addWidget(QLabel("─" * 30))

        # Live data labels
        self.lap_label = QLabel("Lap: --")
        self.position_label = QLabel("Position: --")
        self.status_label = QLabel("Status: ⏸️ WAITING")

        driver_layout.addWidget(self.lap_label)
        driver_layout.addWidget(self.position_label)
        driver_layout.addWidget(self.status_label)

        driver_layout.addWidget(QLabel("─" * 30))

        # Telemetry labels
        self.speed_label = QLabel("Speed: -- km/h")
        self.gear_label = QLabel("Gear: --")
        self.rpm_label = QLabel("RPM: --")
        self.fuel_label = QLabel("Fuel: -- L")

        driver_layout.addWidget(self.speed_label)
        driver_layout.addWidget(self.gear_label)
        driver_layout.addWidget(self.rpm_label)
        driver_layout.addWidget(self.fuel_label)

        driver_layout.addWidget(QLabel("─" * 30))

        # Lap time labels
        self.last_lap_label = QLabel("Last: --:--:---")
        self.best_lap_label = QLabel("Best: --:--:---")

        driver_layout.addWidget(self.last_lap_label)
        driver_layout.addWidget(self.best_lap_label)

        driver_layout.addStretch()

        # Communications transcript
        comms_group = QGroupBox("Communications Transcript")
        comms_layout = QVBoxLayout()
        comms_group.setLayout(comms_layout)

        self.comms_text = QTextEdit()
        self.comms_text.setReadOnly(True)
        self.comms_text.setPlaceholderText("Radio messages will appear here...")
        comms_layout.addWidget(self.comms_text)

        # Commentator transcript
        comment_group = QGroupBox("Commentator Transcript")
        comment_layout = QVBoxLayout()
        comment_group.setLayout(comment_layout)

        self.comment_text = QTextEdit()
        self.comment_text.setReadOnly(True)
        self.comment_text.setPlaceholderText("Commentary will appear here...")
        comment_layout.addWidget(self.comment_text)

        right_col.addWidget(driver_group)
        right_col.addWidget(comms_group)
        right_col.addWidget(comment_group)

        return right_col

    def _create_menu_bar(self):
        """Create menu bar."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        settings_menu = menu_bar.addMenu("Settings")
        view_menu = menu_bar.addMenu("View")

    # ==========================================================================
    # Data Update Methods
    # ==========================================================================

    def update_session_info(self, session_data):
        """
        Update session information panel.

        Args:
            session_data: Dict with track, car_model, player_name, etc.
        """
        track = session_data.get("track", "Unknown")
        track_config = session_data.get("track_config", "")
        car = session_data.get("car_model", "Unknown")
        player_name = session_data.get("player_name", "")
        player_surname = session_data.get("player_surname", "")
        player_nick = session_data.get("player_nick", "")

        track_name = f"{track} ({track_config})" if track_config else track

        self.driver_name_label.setText(f"Driver: {player_name} {player_surname} ({player_nick})")
        self.car_label.setText(f"Car: {car}")
        self.track_label.setText(f"Track: {track_name}")

    def update_live_data(self, live_data):
        """
        Update live telemetry data panel.

        Args:
            live_data: Dict with current_lap, speed, gear, rpm, fuel, etc.
        """
        current_lap = live_data.get("current_lap", 1)
        speed = live_data.get("speed", 0)
        gear = live_data.get("gear", 0)
        rpm = live_data.get("rpm", 0)
        fuel = live_data.get("fuel", 0)
        position = live_data.get("position", 0)
        is_in_pit = live_data.get("is_in_pit", 0)
        best_time = live_data.get("best_time", "")
        last_time = live_data.get("last_time", "")

        pit_status = "🏁 IN PIT" if is_in_pit else "🏎️ ON TRACK"

        # Format gear display
        if gear == 0:
            gear_display = "R"
        elif gear == 1:
            gear_display = "N"
        else:
            gear_display = str(gear - 1)

        self.lap_label.setText(f"Lap: {current_lap}")
        self.position_label.setText(f"Position: P{position}")
        self.status_label.setText(f"Status: {pit_status}")
        self.speed_label.setText(f"Speed: {speed:.1f} km/h")
        self.gear_label.setText(f"Gear: {gear_display}")
        self.rpm_label.setText(f"RPM: {rpm:,}")
        self.fuel_label.setText(f"Fuel: {fuel:.1f} L")
        self.last_lap_label.setText(f"Last: {last_time if last_time else '--:--:---'}")
        self.best_lap_label.setText(f"Best: {best_time if best_time else '--:--:---'}")

    def handle_realtime_sample(self, sample):
        """
        Handle real-time telemetry sample for live visualization.

        Args:
            sample: Dict with telemetry data
        """
        lap_id = sample.get("lap_id", 0)

        # Check if new lap started
        if self.current_lap_id is None or lap_id != self.current_lap_id:
            self.current_lap_samples = []
            self.current_lap_id = lap_id
            print(f"🏁 UI: New lap {lap_id} started")

        self.current_lap_samples.append(sample)

        # Throttled updates (every 5 samples ~12Hz at 60Hz telemetry)
        if len(self.current_lap_samples) % 5 == 0:
            self._update_realtime_visualizations()

    def _update_realtime_visualizations(self):
        """Update all visualizations with current lap data."""
        if len(self.current_lap_samples) < 2:
            return

        try:
            # Extract arrays from samples
            xs = np.array([s["x"] for s in self.current_lap_samples], dtype=float)
            zs = np.array([s["z"] for s in self.current_lap_samples], dtype=float)
            speeds = np.array([s["speed"] for s in self.current_lap_samples], dtype=float)
            times = np.array([s["t"] for s in self.current_lap_samples], dtype=float)
            gears = np.array([s.get("gear", 0) for s in self.current_lap_samples], dtype=float)
            rpms = np.array([s.get("rpms", 0) for s in self.current_lap_samples], dtype=float)
            brakes = np.array([s.get("brake", 0) for s in self.current_lap_samples], dtype=float)

            # Tire data arrays
            tyre_pressure_fl = np.array([s.get("tyre_pressure_fl", 0) for s in self.current_lap_samples], dtype=float)
            tyre_pressure_fr = np.array([s.get("tyre_pressure_fr", 0) for s in self.current_lap_samples], dtype=float)
            tyre_pressure_rl = np.array([s.get("tyre_pressure_rl", 0) for s in self.current_lap_samples], dtype=float)
            tyre_pressure_rr = np.array([s.get("tyre_pressure_rr", 0) for s in self.current_lap_samples], dtype=float)

            tyre_temp_fl = np.array([s.get("tyre_temp_fl", 0) for s in self.current_lap_samples], dtype=float)
            tyre_temp_fr = np.array([s.get("tyre_temp_fr", 0) for s in self.current_lap_samples], dtype=float)
            tyre_temp_rl = np.array([s.get("tyre_temp_rl", 0) for s in self.current_lap_samples], dtype=float)
            tyre_temp_rr = np.array([s.get("tyre_temp_rr", 0) for s in self.current_lap_samples], dtype=float)

            # Normalize times to start from 0
            times = times - times[0]

            # Update visualizations
            self.track_canvas.plot_track(xs, zs, speeds)
            self.speed_canvas.update_data(times, speeds)
            self.gear_canvas.update_data(times, gears)
            self.rpm_canvas.update_data(times, rpms)
            self.brake_canvas.update_data(times, brakes * 100)  # Scale to percentage

            self.tyre_pressure_canvas.update_data(times, [
                tyre_pressure_fl, tyre_pressure_fr, tyre_pressure_rl, tyre_pressure_rr
            ])
            self.tyre_temp_canvas.update_data(times, [
                tyre_temp_fl, tyre_temp_fr, tyre_temp_rl, tyre_temp_rr
            ])

        except Exception as e:
            print(f"❌ Error in visualization update: {e}")

    def handle_lap_complete(self, lap_id, samples):
        """
        Handle lap completion event.

        Args:
            lap_id: Completed lap number
            samples: List of telemetry samples for the lap
        """
        if not samples:
            return

        print(f"🏁 UI: Lap {lap_id} completed with {len(samples)} samples")

        times = np.array([s["t"] for s in samples], dtype=float)
        times = times - times[0]  # Normalize to start from 0

        # Update lap table
        row = min(lap_id - 1, self.lap_table.rowCount() - 1)
        if row >= 0 and len(times) > 0:
            lap_time_seconds = times[-1]
            minutes = int(lap_time_seconds // 60)
            seconds = lap_time_seconds % 60
            lap_time = f"{minutes}:{seconds:06.3f}"
            self.lap_table.setItem(row, 0, QTableWidgetItem(lap_time))
            self.lap_table.setItem(row, 1, QTableWidgetItem("--"))

    def handle_ai_commentary(self, message: str, trigger: str, priority: int):
        """
        Handle AI-generated commentary.

        Args:
            message: AI commentary text
            trigger: Event that triggered the commentary
            priority: Priority level (0=CRITICAL, 1=HIGH, 2=MEDIUM, 3=LOW)
        """
        priority_labels = {0: "🔴 CRITICAL", 1: "🟠 HIGH", 2: "🟡 MEDIUM", 3: "⚪ LOW"}
        priority_label = priority_labels.get(priority, "⚪ INFO")

        timestamp = datetime.now().strftime("%H:%M:%S")

        formatted_message = (
            f"<div style='margin-bottom: 8px;'>"
            f"<span style='color: #888;'>[{timestamp}]</span> "
            f"<span style='font-weight: bold;'>{priority_label}</span> "
            f"<span style='color: #AAA;'>({trigger})</span><br>"
            f"<span style='color: #EEEEEE;'>{message}</span>"
            f"</div>"
        )

        self.comment_text.append(formatted_message)

        # Auto-scroll to bottom
        cursor = self.comment_text.textCursor()
        cursor.movePosition(cursor.End)
        self.comment_text.setTextCursor(cursor)

        print(f"💬 AI Commentary [{trigger}]: {message[:80]}...")
