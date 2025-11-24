import sys
import numpy as np

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
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm


# ------------------ Generic Matplotlib Canvas for Time Series ------------------ #

class TimeSeriesCanvas(FigureCanvas):
    def __init__(self, title: str, parent=None, width=4, height=2, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        bg = "#111111"
        ax_bg = "#181818"

        self.fig.patch.set_facecolor(bg)
        self.ax.set_facecolor(ax_bg)

        for spine in self.ax.spines.values():
            spine.set_color("#CCCCCC")
        self.ax.tick_params(colors="#CCCCCC")
        self.ax.xaxis.label.set_color("#CCCCCC")
        self.ax.yaxis.label.set_color("#CCCCCC")
        self.ax.title.set_color("#FFFFFF")

        self.title = title
        self.ax.set_title(title, fontsize=9)
        self.ax.grid(True, color="#333333", alpha=0.6)
        self.ax.set_xlabel("Time [s]", fontsize=8)

        self.line, = self.ax.plot([], [], linewidth=1.5, color="#6FA8FF")

        self.fig.tight_layout(pad=1.0)

    def update_data(self, t: np.ndarray, y: np.ndarray):
        """
        Update the line data and rescale axes.
        """
        if t.size == 0 or y.size == 0:
            return
        self.line.set_data(t, y)
        self.ax.relim()
        self.ax.autoscale_view()
        self.draw()


# ------------------ Track Map Canvas ------------------ #

class TrackMapCanvas(FigureCanvas):
    def __init__(self, parent=None, width=4, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        # --- DARK BACKGROUND ---
        bg = "#111111"      # figure background
        ax_bg = "#181818"   # axes background

        self.fig.patch.set_facecolor(bg)
        self.ax.set_facecolor(ax_bg)

        # Make lines / text readable on dark bg
        for spine in self.ax.spines.values():
            spine.set_color("#CCCCCC")
        self.ax.tick_params(colors="#CCCCCC")
        self.ax.xaxis.label.set_color("#CCCCCC")
        self.ax.yaxis.label.set_color("#CCCCCC")
        self.ax.title.set_color("#FFFFFF")

        # General style
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.set_title("Location Map", fontsize=10)
        self.ax.set_xlabel("X [m]", fontsize=8)
        self.ax.set_ylabel("Z [m]", fontsize=8)
        self.ax.grid(True, color="#333333", alpha=0.6)

        self.line_collection = None
        self.colorbar = None

        self.fig.tight_layout(pad=1.0)

    def plot_track(self, xs: np.ndarray, zs: np.ndarray, speeds: np.ndarray):
        """
        Plot the track as a series of line segments colored by speed.
        xs, zs, speeds: 1D numpy arrays of same length.
        """
        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.set_title("Location Map", fontsize=10)
        self.ax.set_xlabel("X [m]", fontsize=8)
        self.ax.set_ylabel("Z [m]", fontsize=8)
        self.ax.grid(True, color="#333333", alpha=0.6)

        if xs.size < 2:
            self.draw()
            return

        points = np.array([xs, zs]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = Normalize(vmin=np.min(speeds), vmax=np.max(speeds))
        cmap = cm.get_cmap("Blues")

        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidth=2.5,
        )
        lc.set_array(speeds[:-1])

        self.line_collection = lc
        self.ax.add_collection(lc)

        self.ax.set_xlim(xs.min() - 10, xs.max() + 10)
        self.ax.set_ylim(zs.min() - 10, zs.max() + 10)

        if self.colorbar is not None:
            self.colorbar.remove()
        self.colorbar = self.fig.colorbar(
            lc, ax=self.ax, fraction=0.046, pad=0.04, label="Speed [km/h]"
        )

        self.draw()


# ------------------ Main Dashboard Window ------------------ #

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AC Telemetry Dashboard – Prototype")
        self.resize(1400, 800)

        # Real-time data buffers for current lap
        self.current_lap_samples = []
        self.current_lap_id = None

        central = QWidget()
        self.setCentralWidget(central)

        # Top "menu bar" text style (you can also use real QMenuBar if you want)
        self._create_menu_bar()

        # Root layout: horizontal split into left / middle / right
        root_layout = QHBoxLayout()
        root_layout.setContentsMargins(8, 24, 8, 8)
        root_layout.setSpacing(10)
        central.setLayout(root_layout)

        # ----- Left column: Track map + Lap times -----
        left_col = QVBoxLayout()
        left_col.setSpacing(10)

        # Track map group
        track_group = QGroupBox("Location Map")
        track_group_layout = QVBoxLayout()
        track_group.setLayout(track_group_layout)

        self.track_canvas = TrackMapCanvas(self, width=5, height=4, dpi=100)
        track_group_layout.addWidget(self.track_canvas)

        # Lap times group
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

        # demo data for table
        for i in range(8):
            lap_label = QTableWidgetItem(f"{i+1}.")
            self.lap_table.setVerticalHeaderItem(i, lap_label)
            self.lap_table.setItem(i, 0, QTableWidgetItem("--:--:---"))
            self.lap_table.setItem(i, 1, QTableWidgetItem("----"))

        lap_layout.addWidget(self.lap_table)

        left_col.addWidget(track_group)
        left_col.addWidget(lap_group)

        # ----- Middle column: Graphs -----
        mid_col = QVBoxLayout()
        mid_col.setSpacing(6)

        title_label = QLabel("Live Telemetry Analysis")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        mid_col.addWidget(title_label)

        # Graph canvases
        self.speed_canvas = TimeSeriesCanvas("Speed", self)
        self.gear_canvas = TimeSeriesCanvas("Gear", self)
        self.rpm_canvas = TimeSeriesCanvas("RPM", self)
        self.brake_canvas = TimeSeriesCanvas("Brake", self)

        mid_col.addWidget(self.speed_canvas)
        mid_col.addWidget(self.gear_canvas)
        mid_col.addWidget(self.rpm_canvas)
        mid_col.addWidget(self.brake_canvas)

        # Add Graph button
        add_graph_btn = QPushButton("+ Add Graph")
        add_graph_btn.setFixedHeight(32)
        mid_col.addWidget(add_graph_btn)

        # ----- Right column: Driver info + transcripts -----
        right_col = QVBoxLayout()
        right_col.setSpacing(10)

        # Driver info - now more compact with vertical layout
        driver_group = QGroupBox("Session Info")
        driver_layout = QVBoxLayout()
        driver_layout.setSpacing(2)
        driver_group.setLayout(driver_layout)

        # Individual labels for better control
        self.driver_name_label = QLabel("Driver: ----")
        self.car_label = QLabel("Car: ----")
        self.track_label = QLabel("Track: ----")
        
        driver_layout.addWidget(self.driver_name_label)
        driver_layout.addWidget(self.car_label)
        driver_layout.addWidget(self.track_label)
        
        # Separator
        separator = QLabel("─" * 30)
        separator.setStyleSheet("color: #555555;")
        driver_layout.addWidget(separator)
        
        # Live data labels
        self.lap_label = QLabel("Lap: --")
        self.position_label = QLabel("Position: --")
        self.status_label = QLabel("Status: ⏸️ WAITING")
        
        driver_layout.addWidget(self.lap_label)
        driver_layout.addWidget(self.position_label)
        driver_layout.addWidget(self.status_label)
        
        # Another separator
        separator2 = QLabel("─" * 30)
        separator2.setStyleSheet("color: #555555;")
        driver_layout.addWidget(separator2)
        
        # Telemetry labels
        self.speed_label = QLabel("Speed: -- km/h")
        self.gear_label = QLabel("Gear: --")
        self.rpm_label = QLabel("RPM: --")
        self.fuel_label = QLabel("Fuel: -- L")
        
        driver_layout.addWidget(self.speed_label)
        driver_layout.addWidget(self.gear_label)
        driver_layout.addWidget(self.rpm_label)
        driver_layout.addWidget(self.fuel_label)
        
        # Another separator
        separator3 = QLabel("─" * 30)
        separator3.setStyleSheet("color: #555555;")
        driver_layout.addWidget(separator3)
        
        # Lap time labels
        self.last_lap_label = QLabel("Last: --:--:---")
        self.best_lap_label = QLabel("Best: --:--:---")
        
        driver_layout.addWidget(self.last_lap_label)
        driver_layout.addWidget(self.best_lap_label)
        
        # Add stretch to push everything to the top
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

        # ----- Add columns to root layout -----
        root_layout.addLayout(left_col, 3)   # Increased from 2
        root_layout.addLayout(mid_col, 4)    # Increased from 3
        root_layout.addLayout(right_col, 2)  # Same

        # Optional dark-ish background for sci-fi vibes
        self._apply_basic_style()

    # ------------------ Session Info Updates ------------------ #

    def update_session_info(self, session_data):
        """
        Called once when session starts with static info.
        session_data: dict with track, car_model, player_name, etc.
        """
        track = session_data.get("track", "Unknown")
        track_config = session_data.get("track_config", "")
        car = session_data.get("car_model", "Unknown")
        player_name = session_data.get("player_name", "")
        player_surname = session_data.get("player_surname", "")
        player_nick = session_data.get("player_nick", "")
        
        # Format track name (remove config if empty)
        track_name = f"{track} ({track_config})" if track_config else track
        
        # Update labels
        self.driver_name_label.setText(f"Driver: {player_name} {player_surname} ({player_nick})")
        self.car_label.setText(f"Car: {car}")
        self.track_label.setText(f"Track: {track_name}")

    def update_live_data(self, live_data):
        """
        Called periodically (~6 times/sec) with live telemetry.
        live_data: dict with current_lap, speed, gear, rpm, fuel, etc.
        """
        # Extract live data
        current_lap = live_data.get("current_lap", 1)
        speed = live_data.get("speed", 0)
        gear = live_data.get("gear", 0)
        rpm = live_data.get("rpm", 0)
        fuel = live_data.get("fuel", 0)
        position = live_data.get("position", 0)
        is_in_pit = live_data.get("is_in_pit", 0)
        best_time = live_data.get("best_time", "")
        last_time = live_data.get("last_time", "")
        
        # Format pit status
        pit_status = "🏁 IN PIT" if is_in_pit else "🏎️ ON TRACK"
        
        # Format gear display
        if gear == 0:
            gear_display = "R"
        elif gear == 1:
            gear_display = "N"
        else:
            gear_display = str(gear - 1)
        
        # Update all labels
        self.lap_label.setText(f"Lap: {current_lap}")
        self.position_label.setText(f"Position: P{position}")
        self.status_label.setText(f"Status: {pit_status}")
        self.speed_label.setText(f"Speed: {speed:.1f} km/h")
        self.gear_label.setText(f"Gear: {gear_display}")
        self.rpm_label.setText(f"RPM: {rpm:,}")
        self.fuel_label.setText(f"Fuel: {fuel:.1f} L")
        self.last_lap_label.setText(f"Last: {last_time if last_time else '--:--:---'}")
        self.best_lap_label.setText(f"Best: {best_time if best_time else '--:--:---'}")

    # ------------------ Wiring methods ------------------ #

    def _create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        settings_menu = menu_bar.addMenu("Settings")
        view_menu = menu_bar.addMenu("View")
        # You can add real actions later

    def _apply_basic_style(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #111111;
                color: #EEEEEE;
            }
            QGroupBox {
                border: 1px solid #555555;
                margin-top: 6px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 3px;
            }
            QLabel {
                color: #EEEEEE;
                font-size: 11px;
            }
            QTableWidget, QTextEdit {
                background-color: #1B1B1B;
                color: #EEEEEE;
                border: 1px solid #555555;
            }
            QPushButton {
                background-color: #333333;
                color: #FFFFFF;
                border: 1px solid #777777;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #444444;
            }
            """
        )

    # ------------------ Public API for backend ------------------ #

    def handle_realtime_sample(self, sample):
        """
        Handle real-time telemetry sample (called every frame ~60Hz).
        sample: dict with lap_id, t, x, z, speed, gear, rpms, brake, throttle
        """
        lap_id = sample.get("lap_id", 0)

        # If lap changed, clear current lap buffer
        if self.current_lap_id is None or lap_id != self.current_lap_id:
            self.current_lap_samples = []
            self.current_lap_id = lap_id

        # Add sample to current lap buffer
        self.current_lap_samples.append(sample)

        # Update visualizations with current lap data (throttle updates to avoid overload)
        # Only update every 5 samples (~12 updates/sec at 60Hz)
        if len(self.current_lap_samples) % 5 == 0:
            self._update_realtime_visualizations()

    def _update_realtime_visualizations(self):
        """Update all visualizations with current lap data"""
        if len(self.current_lap_samples) < 2:
            return

        # Extract arrays from samples
        xs = np.array([s["x"] for s in self.current_lap_samples], dtype=float)
        zs = np.array([s["z"] for s in self.current_lap_samples], dtype=float)
        speeds = np.array([s["speed"] for s in self.current_lap_samples], dtype=float)
        times = np.array([s["t"] for s in self.current_lap_samples], dtype=float)
        gears = np.array([s.get("gear", 0) for s in self.current_lap_samples], dtype=float)
        rpms = np.array([s.get("rpms", 0) for s in self.current_lap_samples], dtype=float)
        brakes = np.array([s.get("brake", 0) for s in self.current_lap_samples], dtype=float)

        # Update track map
        self.track_canvas.plot_track(xs, zs, speeds)

        # Normalize times to start from 0
        times = times - times[0]

        # Update time-series graphs
        self.speed_canvas.update_data(times, speeds)
        self.gear_canvas.update_data(times, gears)
        self.rpm_canvas.update_data(times, rpms)
        self.brake_canvas.update_data(times, brakes * 100)  # Scale 0-1 to 0-100%

    def handle_lap_complete(self, lap_id, samples):
        """
        Entry point for the LapBuffer callback.
        Called when a lap is completed.
        samples: list of dicts with:
                 t, x, z, speed, gear, rpms, brake, throttle
        """
        if not samples:
            return

        times = np.array([s["t"] for s in samples], dtype=float)

        # Update window title
        self.setWindowTitle(f"AC Telemetry Dashboard – Lap {lap_id}")

        # Normalize times to start from 0 for each lap
        times = times - times[0]

        # Update lap table with lap time
        row = min(lap_id - 1, self.lap_table.rowCount() - 1)
        if row >= 0 and len(times) > 0:
            lap_time_seconds = times[-1]
            minutes = int(lap_time_seconds // 60)
            seconds = lap_time_seconds % 60
            lap_time = f"{minutes}:{seconds:06.3f}"
            self.lap_table.setItem(row, 0, QTableWidgetItem(lap_time))
            self.lap_table.setItem(row, 1, QTableWidgetItem("--"))

        # The visualization is already showing the current lap in real-time,
        # so we don't need to redraw here. The real-time handler will
        # automatically start showing the next lap.


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
