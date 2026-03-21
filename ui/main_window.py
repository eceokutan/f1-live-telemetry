"""
Main window for F1 Telemetry Dashboard.
"""
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

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

from ui.canvases import TrackMapCanvas, TimeSeriesCanvas, MultiLineCanvas
from ui.styles import DARK_STYLESHEET, FONT_HEADING


class DeltaCanvas(FigureCanvas):
    """
    Live delta-to-best-lap canvas.

    Shows time delta vs distance: green = ahead of best, red = behind.
    Zero line marks the best lap reference.
    """

    def __init__(self, parent=None, width=4, height=1.5, dpi=100):
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
        self.ax.tick_params(colors="#CCCCCC", labelsize=7)
        self.ax.xaxis.label.set_color("#CCCCCC")
        self.ax.yaxis.label.set_color("#CCCCCC")
        self.ax.title.set_color("#FFFFFF")

        self.ax.set_title("Delta to Best", fontsize=8)
        self.ax.set_xlabel("Distance [m]", fontsize=7)
        self.ax.set_ylabel("Delta [s]", fontsize=7)
        self.ax.axhline(0, color="#555555", linewidth=1, linestyle="-")
        self.ax.grid(True, color="#333333", alpha=0.6)

        self._line_collection = None
        self._delta_label = self.ax.text(
            0.98, 0.92, "", transform=self.ax.transAxes,
            ha="right", va="top", fontsize=12, fontweight="bold",
            color="#FFFFFF",
        )

        self.fig.tight_layout(pad=0.5)

    def update_delta(self, distances, deltas):
        """
        Update delta trace with green/red coloring.

        Args:
            distances: cumulative distance array (meters)
            deltas: time delta array (seconds, negative = ahead)
        """
        if len(distances) < 2:
            return

        try:
            # Remove old line collection
            if self._line_collection is not None:
                try:
                    self._line_collection.remove()
                except Exception:
                    pass

            # Build colored segments
            points = np.column_stack([distances, deltas])
            segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)

            # Color: green if delta < 0 (ahead), red if delta > 0 (behind)
            colors = []
            for i in range(len(segments)):
                avg_delta = (deltas[i] + deltas[i + 1]) / 2
                if avg_delta <= 0:
                    colors.append((0.42, 0.80, 0.47, 1.0))  # green
                else:
                    colors.append((1.0, 0.42, 0.42, 1.0))   # red

            lc = LineCollection(segments, colors=colors, linewidths=2)
            self._line_collection = self.ax.add_collection(lc)

            self.ax.set_xlim(0, distances[-1])

            y_max = max(abs(deltas.min()), abs(deltas.max()), 0.5)
            self.ax.set_ylim(-y_max * 1.2, y_max * 1.2)

            # Update delta label
            current_delta = deltas[-1]
            if current_delta <= 0:
                self._delta_label.set_text(f"{current_delta:+.3f}s")
                self._delta_label.set_color("#6BCB77")
            else:
                self._delta_label.set_text(f"+{current_delta:.3f}s")
                self._delta_label.set_color("#FF6B6B")

            self.draw_idle()
        except Exception:
            pass

    def clear_delta(self):
        """Clear the delta trace."""
        if self._line_collection is not None:
            try:
                self._line_collection.remove()
            except Exception:
                pass
            self._line_collection = None
        self._delta_label.set_text("")
        self.draw_idle()


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

        self.setWindowTitle("Jarvis Live - F1 Telemetry Dashboard")
        self.resize(1600, 900)
        self.exit_application_requested = False

        # Real-time data buffers for current lap
        self.current_lap_samples = []
        self.current_lap_id = None

        # Lap time tracking
        self._lap_times_ms = []       # list of lap times in ms, index = lap-1
        self._best_time_ms = 0        # best lap time in ms
        self._last_completed_laps = 0 # last seen completedLaps value
        self._lap_counter_initialized = False

        # Best lap reference for delta calculation
        self._best_lap_samples = None  # list of sample dicts from best lap
        self._best_lap_distances = None  # cumulative distance array
        self._best_lap_times = None      # elapsed time array (from t=0)
        self._previous_lap_samples = None  # samples from last completed lap (for best-lap saving)

        # Menu bar
        self._create_menu_bar()

        # Create central widget and root layout
        central = QWidget()
        self.setCentralWidget(central)

        # Root layout: horizontal split into left / middle / right
        root_layout = QHBoxLayout()
        root_layout.setContentsMargins(8, 8, 8, 8)
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

    def _create_menu_bar(self):
        """Create menu bar with File menu."""
        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)

        file_menu = menu_bar.addMenu("File")

        back_action = QtWidgets.QAction("Back to Launcher", self)
        back_action.setShortcut("Ctrl+W")
        back_action.triggered.connect(self._back_to_launcher)
        file_menu.addAction(back_action)

        file_menu.addSeparator()

        exit_action = QtWidgets.QAction("Exit Application", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self._exit_application)
        file_menu.addAction(exit_action)

    def _back_to_launcher(self):
        """Close live window and return to launcher loop."""
        self.exit_application_requested = False
        self.close()

    def _exit_application(self):
        """Close live window and request full app shutdown."""
        self.exit_application_requested = True
        self.close()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.quit()

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

        self.lap_table = QTableWidget(0, 2)
        self.lap_table.setHorizontalHeaderLabels(["Lap Time", "Delta"])
        self.lap_table.verticalHeader().setVisible(True)
        self.lap_table.horizontalHeader().setStretchLastSection(True)
        self.lap_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.lap_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.lap_table.setColumnWidth(0, 100)

        self.lap_table.setMaximumHeight(150)
        lap_layout.addWidget(self.lap_table)

        left_col.addWidget(track_group)
        left_col.addWidget(lap_group)

        # Delta to best lap graph
        delta_group = QGroupBox("Delta to Best Lap")
        delta_layout = QVBoxLayout()
        delta_group.setLayout(delta_layout)
        self.delta_canvas = DeltaCanvas(self, width=4, height=1.5, dpi=100)
        delta_layout.addWidget(self.delta_canvas)
        left_col.addWidget(delta_group)

        return left_col

    def _build_middle_column(self):
        """Build middle column: telemetry graphs."""
        mid_col = QVBoxLayout()
        mid_col.setSpacing(4)

        title_label = QLabel("Live Telemetry Analysis")
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        title_label.setStyleSheet(f"font-family: '{FONT_HEADING}'; font-size: 18px;")
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

        # Add to layout with equal stretch so the available space is
        # distributed across the remaining telemetry plots.
        canvases = [
            self.speed_canvas,
            self.gear_canvas,
            self.rpm_canvas,
            self.brake_canvas,
            self.tyre_pressure_canvas,
            self.tyre_temp_canvas,
        ]
        for canvas in canvases:
            mid_col.addWidget(canvas, 1)

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

        driver_layout.addWidget(QLabel("─" * 30))

        # Microphone status indicator
        self.mic_status_label = QLabel("🎤 Mic: Ready")
        self.mic_status_label.setStyleSheet("color: #888888;")  # Gray when idle
        driver_layout.addWidget(self.mic_status_label)

        # AI readiness/status indicator
        self.ai_status_label = QLabel("🤖 AI: Off")
        self.ai_status_label.setStyleSheet("color: #888888;")  # Gray by default
        driver_layout.addWidget(self.ai_status_label)

        driver_layout.addStretch()

        # Communications transcript
        comms_group = QGroupBox("Communications Transcript")
        comms_layout = QVBoxLayout()
        comms_group.setLayout(comms_layout)

        self.comms_text = QTextEdit()
        self.comms_text.setReadOnly(True)
        self.comms_text.setAcceptRichText(True)
        self.comms_text.setPlaceholderText("Radio messages will appear here...")
        comms_layout.addWidget(self.comms_text)

        right_col.addWidget(driver_group)
        right_col.addWidget(comms_group)

        return right_col

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
        ac_status = live_data.get("ac_status", 2)  # default to LIVE
        best_time = live_data.get("best_time", "")
        last_time = live_data.get("last_time", "")

        # AC status: 0=OFF, 1=REPLAY, 2=LIVE, 3=PAUSE
        if ac_status == 0:
            pit_status = "⏸️ OFF / IN MENU"
        elif ac_status == 1:
            pit_status = "🎬 REPLAY"
        elif ac_status == 3:
            pit_status = "⏸️ PAUSED"
        elif is_in_pit:
            pit_status = "🏁 IN PIT"
        else:
            pit_status = "🏎️ ON TRACK"

        # Format gear display (display_gear: -1=R, 0=N, 1+=gear number)
        if gear <= -1:
            gear_display = "R"
        elif gear == 0:
            gear_display = "N"
        else:
            gear_display = str(gear)

        self.lap_label.setText(f"Lap: {current_lap}")
        self.position_label.setText(f"Position: P{position}")
        self.status_label.setText(f"Status: {pit_status}")
        self.speed_label.setText(f"Speed: {speed:.1f} km/h")
        self.gear_label.setText(f"Gear: {gear_display}")
        self.rpm_label.setText(f"RPM: {rpm:,}")
        self.fuel_label.setText(f"Fuel: {fuel:.1f} L")
        self.last_lap_label.setText(f"Last: {last_time if last_time else '--:--:---'}")
        self.best_lap_label.setText(f"Best: {best_time if best_time else '--:--:---'}")

        # Detect lap completion from AC's completedLaps counter
        try:
            completed_laps = int(live_data.get("completed_laps", 0))
        except (TypeError, ValueError):
            completed_laps = 0

        try:
            last_time_ms = int(live_data.get("last_time_ms", 0) or 0)
        except (TypeError, ValueError):
            last_time_ms = 0

        try:
            best_time_ms = int(live_data.get("best_time_ms", 0) or 0)
        except (TypeError, ValueError):
            best_time_ms = 0

        completed_laps = max(0, completed_laps)
        last_seen = self._last_completed_laps

        # Late join safety: first packet only establishes baseline so we don't
        # backfill historical laps when opening mid-session.
        if not self._lap_counter_initialized:
            self._lap_counter_initialized = True
            self._last_completed_laps = completed_laps
            return

        # Session reset/restart: resync baseline without creating table rows.
        if completed_laps < last_seen:
            logger.info(
                "Lap counter reset detected (%d -> %d), resyncing baseline",
                last_seen,
                completed_laps,
            )
            self._last_completed_laps = completed_laps
            return

        if completed_laps == last_seen:
            return

        # Ignore jumps greater than one lap; treat as desync and resync.
        if completed_laps > last_seen + 1:
            logger.warning(
                "Lap counter jump detected (%d -> %d), resyncing baseline",
                last_seen,
                completed_laps,
            )
            self._last_completed_laps = completed_laps
            return

        # Valid one-lap increment.
        self._last_completed_laps = completed_laps
        if last_time_ms <= 0:
            return

        # Check if this lap set a new best — save its samples as reference
        is_new_best = (
            best_time_ms != self._best_time_ms
            and best_time_ms > 0
            and (self._best_time_ms == 0 or best_time_ms <= self._best_time_ms)
        )
        self._best_time_ms = best_time_ms

        if is_new_best and self._previous_lap_samples:
            self._save_best_lap_reference(self._previous_lap_samples)

        self._add_lap_to_table(completed_laps, last_time_ms)

    def handle_realtime_sample(self, sample):
        """
        Handle real-time telemetry sample for live visualization.

        Args:
            sample: Dict with telemetry data
        """
        # Skip stationary samples before the car starts moving
        # (prevents plotting zero-speed data from garage/grid/loading)
        speed = sample.get("speed", 0.0)
        if not self.current_lap_samples and speed < 1.0:
            return

        lap_id = sample.get("lap_id", 0)

        # Check if new lap started
        if self.current_lap_id is None or lap_id != self.current_lap_id:
            # Save previous lap samples before clearing (needed for best-lap reference)
            if self.current_lap_samples:
                self._previous_lap_samples = list(self.current_lap_samples)
            self.current_lap_samples = []
            self.current_lap_id = lap_id
            # Reset track map initialization flag so it redraws properly
            if hasattr(self.track_canvas, '_initialized'):
                del self.track_canvas._initialized
            self.delta_canvas.clear_delta()
            logger.info("UI: New lap %d started", lap_id)

        # Detect large position jumps (hotlap teleport or lap reset)
        if len(self.current_lap_samples) > 0:
            last_x = self.current_lap_samples[-1].get("x", 0)
            last_z = self.current_lap_samples[-1].get("z", 0)
            curr_x = sample.get("x", 0)
            curr_z = sample.get("z", 0)
            distance = ((curr_x - last_x) ** 2 + (curr_z - last_z) ** 2) ** 0.5

            # If position jumped more than 100m, clear the buffer (hotlap reset)
            if distance > 100:
                logger.warning("Large position jump detected (%.0fm), clearing buffer", distance)
                self.current_lap_samples = []
                if hasattr(self.track_canvas, '_initialized'):
                    del self.track_canvas._initialized

        self.current_lap_samples.append(sample)

        # Throttled updates (every 20 samples ~3Hz at 60Hz telemetry)
        # Reduced from 5 to prevent UI overload
        if len(self.current_lap_samples) % 20 == 0:
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

            # Note: Position (x,z) may be (0,0) in some AC configurations
            # We still plot the graphs (speed, RPM, etc.) even if position is unavailable
            # Track map will just show origin, but telemetry data is still valid

            # Check for monotonically increasing times (no backwards jumps)
            if len(times) > 1 and not np.all(np.diff(times) >= 0):
                # Times went backwards - lap was reset, clear buffer
                logger.warning("Time went backwards, clearing buffer")
                self.current_lap_samples = []
                return

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

            # Debug: Print when actually updating graphs
            if len(self.current_lap_samples) % 60 == 0:
                logger.debug("Updating graphs: %d samples, pos=(%.1f, %.1f)",
                             len(self.current_lap_samples), xs[-1], zs[-1])

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

            # Update delta to best lap
            if self._best_lap_distances is not None:
                eval_dist, deltas = self._compute_delta(self.current_lap_samples)
                if eval_dist is not None and deltas is not None:
                    self.delta_canvas.update_delta(eval_dist, deltas)

        except Exception as e:
            logger.error("Visualization update error: %s", e)

    def _add_lap_to_table(self, lap_number: int, lap_time_ms: int):
        """Add a completed lap to the table with proper delta calculation."""
        from PyQt5.QtGui import QColor

        self._lap_times_ms.append(lap_time_ms)

        row = self.lap_table.rowCount()
        self.lap_table.insertRow(row)
        self.lap_table.setVerticalHeaderItem(row, QTableWidgetItem(f"{lap_number}"))

        # Lap time
        time_str = self._format_time_ms(lap_time_ms)
        time_item = QTableWidgetItem(time_str)

        # Highlight best lap in green
        if lap_time_ms == self._best_time_ms:
            time_item.setForeground(QColor(107, 203, 119))  # green

        self.lap_table.setItem(row, 0, time_item)

        # Delta vs best lap
        if self._best_time_ms > 0 and len(self._lap_times_ms) > 1:
            delta_ms = lap_time_ms - self._best_time_ms
            if delta_ms == 0:
                delta_str = "BEST"
                delta_item = QTableWidgetItem(delta_str)
                delta_item.setForeground(QColor(107, 203, 119))  # green
            else:
                delta_seconds = delta_ms / 1000.0
                delta_str = f"+{delta_seconds:.3f}"
                delta_item = QTableWidgetItem(delta_str)
                delta_item.setForeground(QColor(255, 107, 107))  # red
        else:
            delta_item = QTableWidgetItem("--")

        self.lap_table.setItem(row, 1, delta_item)

        # Scroll to the new row
        self.lap_table.scrollToBottom()

        # If best time changed, update all previous deltas
        if lap_time_ms == self._best_time_ms and len(self._lap_times_ms) > 1:
            self._refresh_deltas()

    def _refresh_deltas(self):
        """Refresh all delta values in the table after a new best lap."""
        from PyQt5.QtGui import QColor
        for i, t_ms in enumerate(self._lap_times_ms):
            delta_ms = t_ms - self._best_time_ms
            if delta_ms == 0:
                delta_item = QTableWidgetItem("BEST")
                delta_item.setForeground(QColor(107, 203, 119))
            else:
                delta_item = QTableWidgetItem(f"+{delta_ms / 1000.0:.3f}")
                delta_item.setForeground(QColor(255, 107, 107))
            self.lap_table.setItem(i, 1, delta_item)

    @staticmethod
    def _format_time_ms(ms: int) -> str:
        """Format milliseconds as M:SS.mmm"""
        total_seconds = ms / 1000.0
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        return f"{minutes}:{seconds:06.3f}"

    def _save_best_lap_reference(self, samples):
        """Save a completed lap's samples as the best-lap reference for delta calculation."""
        if len(samples) < 2:
            return

        self._best_lap_samples = samples

        xs = np.array([s["x"] for s in samples], dtype=float)
        zs = np.array([s["z"] for s in samples], dtype=float)
        ts = np.array([s["t"] for s in samples], dtype=float)
        ts = ts - ts[0]  # normalize to start from 0

        # Compute cumulative distance
        dx = np.diff(xs)
        dz = np.diff(zs)
        seg_dist = np.sqrt(dx ** 2 + dz ** 2)
        self._best_lap_distances = np.concatenate([[0], np.cumsum(seg_dist)])
        self._best_lap_times = ts

        logger.info("Saved best lap reference: %.1fm total, %.3fs",
                     self._best_lap_distances[-1], ts[-1])

    def _compute_delta(self, current_samples):
        """Compute time delta between current lap and best lap at each distance point."""
        if (self._best_lap_distances is None or self._best_lap_times is None
                or len(current_samples) < 2):
            return None, None

        xs = np.array([s["x"] for s in current_samples], dtype=float)
        zs = np.array([s["z"] for s in current_samples], dtype=float)
        ts = np.array([s["t"] for s in current_samples], dtype=float)
        ts = ts - ts[0]

        # Cumulative distance for current lap
        dx = np.diff(xs)
        dz = np.diff(zs)
        seg_dist = np.sqrt(dx ** 2 + dz ** 2)
        curr_distances = np.concatenate([[0], np.cumsum(seg_dist)])

        # Only compare up to the shorter of current distance or best lap distance
        max_dist = min(curr_distances[-1], self._best_lap_distances[-1])
        if max_dist < 10:  # need at least 10m of data
            return None, None

        # Sample at regular distance intervals for smooth comparison
        n_points = min(200, len(curr_distances))
        eval_distances = np.linspace(0, max_dist, n_points)

        # Interpolate: time at each distance for both laps
        curr_time_at_dist = np.interp(eval_distances, curr_distances, ts)
        best_time_at_dist = np.interp(eval_distances, self._best_lap_distances, self._best_lap_times)

        # Delta = current time - best time (positive = slower, negative = faster)
        deltas = curr_time_at_dist - best_time_at_dist

        return eval_distances, deltas

    def handle_lap_complete(self, lap_id, samples):
        """
        Handle lap completion event.

        Args:
            lap_id: Completed lap number
            samples: List of telemetry samples for the lap
        """
        if not samples:
            return
        logger.info("Lap %d completed with %d samples", lap_id, len(samples))

    def handle_ai_commentary(self, message: str, trigger: str, priority: int):
        """
        Handle AI-generated commentary.

        Args:
            message: AI commentary text
            trigger: Event that triggered the commentary
            priority: Priority level (0=CRITICAL, 1=HIGH, 2=MEDIUM, 3=LOW)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Driver query responses go to Communications Transcript
        if trigger == "driver_query" or trigger == "driver_query_error" or trigger == "driver_query_timeout":
            formatted_response = (
                f"<div style='margin-bottom: 6px; font-size: 14pt;'>"
                f"<span style='color: #888;'>[{timestamp}]</span> "
                f"<span style='font-weight: bold; color: #FF6B6B;'>RACE ENGINEER:</span><br>"
                f"<span style='color: #EEEEEE;'>{message}</span>"
                f"</div><br>"
            )

            # Use insertHtml for proper HTML rendering
            cursor = self.comms_text.textCursor()
            cursor.movePosition(cursor.End)
            cursor.insertHtml(formatted_response)
            self.comms_text.setTextCursor(cursor)
            self.comms_text.ensureCursorVisible()

            logger.info("AI Response: %s...", message[:80])

        # All other AI commentary also goes to Communications Transcript
        else:
            priority_labels = {0: "🔴 CRITICAL", 1: "🟠 HIGH", 2: "🟡 MEDIUM", 3: "⚪ LOW"}
            priority_label = priority_labels.get(priority, "⚪ INFO")

            formatted_message = (
                f"<div style='margin-bottom: 6px; font-size: 14pt;'>"
                f"<span style='color: #888;'>[{timestamp}]</span> "
                f"<span style='font-weight: bold;'>{priority_label}</span> "
                f"<span style='color: #AAA;'>({trigger})</span><br>"
                f"<span style='color: #EEEEEE;'>{message}</span>"
                f"</div><br>"
            )

            # Use insertHtml for proper HTML rendering
            cursor = self.comms_text.textCursor()
            cursor.movePosition(cursor.End)
            cursor.insertHtml(formatted_message)
            self.comms_text.setTextCursor(cursor)
            self.comms_text.ensureCursorVisible()

            logger.info("AI Commentary [%s]: %s...", trigger, message[:80])

    def handle_driver_query(self, query: str):
        """
        Handle driver query (display in Communications panel).

        Args:
            query: Driver's question/command
        """
        timestamp = datetime.now().strftime("%H:%M:%S")

        formatted_query = (
            f"<div style='margin-bottom: 4px;'>"
            f"<span style='color: #888;'>[{timestamp}]</span> "
            f"<span style='font-weight: bold; color: #6FA8FF;'>DRIVER:</span><br>"
            f"<span style='color: #EEEEEE;'>{query}</span>"
            f"</div><br>"
        )

        # Use insertHtml for proper HTML rendering
        cursor = self.comms_text.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertHtml(formatted_query)
        self.comms_text.setTextCursor(cursor)
        self.comms_text.ensureCursorVisible()

        logger.info("Driver Query: %s", query)

    def set_ptt_controller(self, controller):
        """Set PTT controller reference for Qt key-event forwarding (macOS).

        Installs an application-level event filter so key events are captured
        regardless of which child widget has focus.
        """
        self._ptt_controller = controller
        QtWidgets.QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, event):
        """Application-level event filter to forward key events to PTT controller."""
        ctrl = getattr(self, "_ptt_controller", None)
        if ctrl:
            if event.type() == QtCore.QEvent.KeyPress and not event.isAutoRepeat():
                if event.text():
                    ctrl.handle_key_press(event.text())
            elif event.type() == QtCore.QEvent.KeyRelease and not event.isAutoRepeat():
                if event.text():
                    ctrl.handle_key_release(event.text())
        return super().eventFilter(obj, event)

    def handle_vad_state_change(self, is_speaking: bool):
        """
        Handle voice activity detection state change.

        Args:
            is_speaking: True if driver is speaking, False if silent
        """
        if is_speaking:
            self.mic_status_label.setText("🎤 Mic: Speaking...")
            self.mic_status_label.setStyleSheet("color: #FF6B6B;")  # Red when speaking
        else:
            self.mic_status_label.setText("🎤 Mic: Ready")
            self.mic_status_label.setStyleSheet("color: #888888;")  # Gray when idle

    def handle_ai_status_update(self, status: str):
        """
        Handle AI worker status updates and expose clear readiness state in UI.

        Args:
            status: Status message emitted by AI worker
        """
        message = (status or "").strip()
        status_lower = message.lower()

        if "ready" in status_lower:
            self.ai_status_label.setText("🤖 AI: Ready")
            self.ai_status_label.setStyleSheet("color: #6BCB77;")
            return

        if "starting" in status_lower or "loading" in status_lower or "warming up" in status_lower:
            self.ai_status_label.setText("🤖 AI: Loading...")
            self.ai_status_label.setStyleSheet("color: #FFD166;")
            return

        if "processing" in status_lower:
            self.ai_status_label.setText("🤖 AI: Processing query...")
            self.ai_status_label.setStyleSheet("color: #6FA8FF;")
            return

        if "fallback" in status_lower:
            self.ai_status_label.setText("🤖 AI: Rule-based fallback")
            self.ai_status_label.setStyleSheet("color: #FFD166;")
            return

        if "error" in status_lower or "failed" in status_lower:
            self.ai_status_label.setText("🤖 AI: Error")
            self.ai_status_label.setStyleSheet("color: #FF6B6B;")
            return

        if "stopped" in status_lower:
            self.ai_status_label.setText("🤖 AI: Stopped")
            self.ai_status_label.setStyleSheet("color: #888888;")
            return

        # Generic passthrough for uncategorized statuses.
        self.ai_status_label.setText(f"🤖 AI: {message}")
        self.ai_status_label.setStyleSheet("color: #AAAAAA;")
