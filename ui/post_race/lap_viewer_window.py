"""
Main lap viewer window - post-race telemetry visualization.
"""
from datetime import datetime
import threading
from PyQt5 import QtWidgets, QtCore
from typing import Optional

from data import Session, Lap
from analysis import AIPipelineBridge
from .timeline_controller import TimelineController
from ui.styles import DARK_STYLESHEET, TIRE_COLORS, TIRE_LABELS
from .canvases import TrackMapCanvas, TimeSeriesCanvas


class _StayOpenMenu(QtWidgets.QMenu):
    """QMenu subclass that stays open when checkable actions are toggled."""

    def mouseReleaseEvent(self, event):
        action = self.activeAction()
        if action and action.isCheckable():
            action.trigger()
            return  # Don't close
        super().mouseReleaseEvent(event)


class LapViewerWindow(QtWidgets.QMainWindow):
    """
    Main window for post-race telemetry analysis.

    Features:
    - YouTube-style timeline scrubber
    - Track map with position marker
    - Scrollable time-series graphs
    - Lap selection
    - Play/pause controls
    """
    analysis_finished = QtCore.pyqtSignal(int, object, str)

    def __init__(self):
        super().__init__()

        # Data
        self.session: Optional[Session] = None
        self.current_lap: Optional[Lap] = None

        # Canvases (created when lap is loaded)
        self.track_map: Optional[TrackMapCanvas] = None
        self._active_canvases: list = []

        # Timeline controller
        self.timeline = TimelineController(self)
        self.timeline.time_changed.connect(self.on_time_changed)
        self.ai_pipeline = AIPipelineBridge()
        self.analysis_finished.connect(self._on_analysis_finished)
        self._analysis_request_id = 0

        # Analysis tab widgets
        self.analysis_context_label: Optional[QtWidgets.QLabel] = None
        self.analysis_source_label: Optional[QtWidgets.QLabel] = None
        self.analysis_refresh_button: Optional[QtWidgets.QPushButton] = None
        self.coach_output: Optional[QtWidgets.QTextEdit] = None
        self.analyst_output: Optional[QtWidgets.QTextEdit] = None

        # Build UI
        self.setWindowTitle("Jarvis Post - Post-Race Telemetry Analysis")
        self.setGeometry(100, 100, 1400, 900)

        self._create_menu_bar()
        self._create_central_widget()
        self._create_status_bar()

        self._apply_dark_theme()

    # All available graph definitions: (key, title, type, config)
    # type: "single" or "multi"
    # config: dict with plot parameters
    GRAPH_DEFS = [
        ("speed", "Speed", "single", {"col": "speed", "ylabel": "Speed [km/h]"}),
        ("gear", "Gear", "single", {"col": "gear", "ylabel": "Gear"}),
        ("rpm", "Engine RPM", "single", {"col": "rpm", "ylabel": "RPM"}),
        ("throttle_brake", "Throttle & Brake", "multi_tb", {}),
        ("fuel", "Fuel", "single", {"col": "fuel", "ylabel": "Fuel [L]"}),
        ("steer_angle", "Steering Angle", "single", {"col": "steer_angle", "ylabel": "Steer [rad]"}),
        ("g_forces", "G-Forces", "multi_gforce", {}),
        ("tyre_temp", "Tire Temperatures", "multi_tyre", {"prefix": "tyre_temp", "ylabel": "Temperature [C]"}),
        ("tyre_pressure", "Tire Pressures", "multi_tyre", {"prefix": "tyre_pressure", "ylabel": "Pressure [PSI]"}),
        ("tyre_wear", "Tire Wear", "multi_tyre", {"prefix": "tyre_wear", "ylabel": "Wear [%]"}),
        ("wheel_slip", "Wheel Slip", "multi_tyre", {"prefix": "wheel_slip", "ylabel": "Slip"}),
        ("suspension", "Suspension Travel", "multi_tyre", {"prefix": "suspension", "ylabel": "Travel [m]"}),
        ("ride_height", "Ride Height", "multi_rh", {}),
        ("car_damage", "Car Damage", "multi_damage", {}),
    ]

    # Default enabled graphs
    DEFAULT_GRAPHS = {"speed", "gear", "rpm", "throttle_brake", "tyre_temp", "tyre_pressure"}

    def _create_menu_bar(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("File")

        open_action = QtWidgets.QAction("Open Session...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_session)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        exit_action = QtWidgets.QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = menu_bar.addMenu("View")
        fullscreen_action = QtWidgets.QAction("Toggle Fullscreen", self)
        fullscreen_action.setShortcut("F11")
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)

        # Graphs menu that stays open when clicking checkboxes
        graphs_menu = _StayOpenMenu("Graphs", self)
        menu_bar.addMenu(graphs_menu)

        # Header label showing max limit
        header_action = QtWidgets.QWidgetAction(self)
        header_label = QtWidgets.QLabel(f"  Select up to {self.MAX_GRAPHS} graphs:")
        header_label.setStyleSheet("color: #aaa; padding: 4px 8px; font-size: 11px;")
        header_action.setDefaultWidget(header_label)
        graphs_menu.addAction(header_action)
        graphs_menu.addSeparator()

        self._graph_actions = {}
        for key, title, _, _ in self.GRAPH_DEFS:
            action = QtWidgets.QAction(title, self)
            action.setCheckable(True)
            action.setChecked(key in self.DEFAULT_GRAPHS)
            action.triggered.connect(self._on_graph_toggled)
            graphs_menu.addAction(action)
            self._graph_actions[key] = action

        self._enabled_graphs = set(self.DEFAULT_GRAPHS)

    def _create_central_widget(self) -> None:
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.tab_widget = QtWidgets.QTabWidget()
        main_layout.addWidget(self.tab_widget)

        self.lap_review_tab = self._create_lap_review_tab()
        self.tab_widget.addTab(self.lap_review_tab, "Lap Review")

        self.analysis_tab = self._create_analysis_tab()
        self.tab_widget.addTab(self.analysis_tab, "Analysis")

        timeline_widget = self._create_timeline_widget()
        main_layout.addWidget(timeline_widget)

    def _create_lap_review_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.left_panel = self._create_left_panel()
        layout.addWidget(self.left_panel)

        self.graph_container = self._create_graph_container()
        layout.addWidget(self.graph_container, stretch=1)

        return tab

    def _create_analysis_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        header_row = QtWidgets.QHBoxLayout()
        self.analysis_context_label = QtWidgets.QLabel("No session loaded")
        self.analysis_context_label.setStyleSheet("font-size: 12px; color: #AAAAAA;")
        header_row.addWidget(self.analysis_context_label)
        header_row.addStretch()

        self.analysis_refresh_button = QtWidgets.QPushButton("Refresh AI Analysis")
        self.analysis_refresh_button.setEnabled(False)
        self.analysis_refresh_button.clicked.connect(self.refresh_analysis)
        header_row.addWidget(self.analysis_refresh_button)
        layout.addLayout(header_row)

        self.analysis_source_label = QtWidgets.QLabel("Source: --")
        self.analysis_source_label.setStyleSheet("font-size: 11px; color: #888888;")
        layout.addWidget(self.analysis_source_label)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        coach_group = QtWidgets.QGroupBox("Coach")
        coach_layout = QtWidgets.QVBoxLayout(coach_group)
        self.coach_output = QtWidgets.QTextEdit()
        self.coach_output.setReadOnly(True)
        self.coach_output.setPlaceholderText("Coach model output will appear here.")
        coach_layout.addWidget(self.coach_output)

        analyst_group = QtWidgets.QGroupBox("Analyst")
        analyst_layout = QtWidgets.QVBoxLayout(analyst_group)
        self.analyst_output = QtWidgets.QTextEdit()
        self.analyst_output.setReadOnly(True)
        self.analyst_output.setPlaceholderText("Analyst model output will appear here.")
        analyst_layout.addWidget(self.analyst_output)

        splitter.addWidget(coach_group)
        splitter.addWidget(analyst_group)
        splitter.setSizes([700, 700])
        layout.addWidget(splitter, stretch=1)

        self._set_analysis_text(
            coach_text="Load a session and select a lap to run AI analysis.",
            analyst_text="Load a session and select a lap to run AI analysis.",
            source_text="Source: --",
        )

        return tab

    def _create_left_panel(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        widget.setMinimumWidth(250)
        widget.setMaximumWidth(280)

        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(4)

        title = QtWidgets.QLabel("Laps")
        title.setStyleSheet("font-size: 14px; font-weight: bold; padding: 4px;")
        layout.addWidget(title)

        self.lap_list = QtWidgets.QListWidget()
        self.lap_list.setMaximumHeight(120)
        self.lap_list.itemClicked.connect(self.on_lap_selected)
        layout.addWidget(self.lap_list)

        track_label = QtWidgets.QLabel("Track Map")
        track_label.setStyleSheet("font-size: 13px; font-weight: bold; padding: 4px;")
        layout.addWidget(track_label)

        self.track_map_container = QtWidgets.QVBoxLayout()
        placeholder = QtWidgets.QLabel("Load a lap to view track map")
        placeholder.setAlignment(QtCore.Qt.AlignCenter)
        placeholder.setStyleSheet("color: #888; padding: 20px;")
        self.track_map_container.addWidget(placeholder)
        layout.addLayout(self.track_map_container)

        # Lap info section (moved here from the separate right panel)
        info_label = QtWidgets.QLabel("Lap Info")
        info_label.setStyleSheet("font-size: 13px; font-weight: bold; padding: 4px;")
        layout.addWidget(info_label)

        self.time_label = QtWidgets.QLabel("Time: 0.00s")
        self.time_label.setStyleSheet("font-size: 12px; padding: 2px 8px;")
        layout.addWidget(self.time_label)

        self.speed_label = QtWidgets.QLabel("Speed: -- km/h")
        self.gear_label = QtWidgets.QLabel("Gear: --")
        self.rpm_label = QtWidgets.QLabel("RPM: ----")
        self.throttle_label = QtWidgets.QLabel("Throttle: --%")
        self.brake_label = QtWidgets.QLabel("Brake: --%")

        for label in [self.speed_label, self.gear_label, self.rpm_label,
                      self.throttle_label, self.brake_label]:
            label.setStyleSheet("padding: 1px 8px; font-size: 12px;")
            layout.addWidget(label)

        layout.addStretch()

        return widget

    def _create_graph_container(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        outer_layout = QtWidgets.QHBoxLayout(container)
        outer_layout.setSpacing(10)
        outer_layout.setContentsMargins(10, 10, 10, 10)

        # Left column: Speed, Gear, RPM
        self.graph_layout_left = QtWidgets.QVBoxLayout()
        self.graph_layout_left.setSpacing(6)
        outer_layout.addLayout(self.graph_layout_left)

        # Right column: Throttle/Brake, Tire Temps, Tire Pressures
        self.graph_layout_right = QtWidgets.QVBoxLayout()
        self.graph_layout_right.setSpacing(6)
        outer_layout.addLayout(self.graph_layout_right)

        # Placeholder (spans both columns initially)
        self.graph_placeholder = QtWidgets.QLabel("Load a session to view telemetry")
        self.graph_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.graph_placeholder.setStyleSheet("font-size: 14px; color: #888; padding: 50px;")
        self.graph_layout_left.addWidget(self.graph_placeholder)

        return container

    def _create_timeline_widget(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        widget.setFixedHeight(110)

        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(10, 5, 10, 5)

        time_row = QtWidgets.QHBoxLayout()
        self.current_time_label = QtWidgets.QLabel("0:00.000")
        self.duration_label = QtWidgets.QLabel("0:00.000")
        time_row.addWidget(self.current_time_label)
        time_row.addStretch()
        time_row.addWidget(self.duration_label)
        layout.addLayout(time_row)

        self.timeline_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(10000)
        self.timeline_slider.setValue(0)
        self.timeline_slider.sliderMoved.connect(self.on_slider_moved)
        layout.addWidget(self.timeline_slider)

        controls_row = QtWidgets.QHBoxLayout()

        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.setFixedSize(60, 30)
        self.play_button.clicked.connect(self.timeline.toggle_play_pause)
        controls_row.addWidget(self.play_button)

        controls_row.addWidget(QtWidgets.QLabel("Speed:"))
        self.speed_combo = QtWidgets.QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentTextChanged.connect(self.on_speed_changed)
        controls_row.addWidget(self.speed_combo)

        controls_row.addSpacing(20)
        controls_row.addWidget(QtWidgets.QLabel("Zoom:"))
        self.zoom_combo = QtWidgets.QComboBox()
        self.zoom_combo.addItems(["15s", "30s", "45s", "60s", "Full Lap"])
        self.zoom_combo.setCurrentText("45s")
        self.zoom_combo.currentTextChanged.connect(self.on_zoom_changed)
        controls_row.addWidget(self.zoom_combo)

        controls_row.addStretch()
        layout.addLayout(controls_row)

        self.timeline.playback_started.connect(lambda: self.play_button.setText("Pause"))
        self.timeline.playback_paused.connect(lambda: self.play_button.setText("Play"))
        self.timeline.playback_finished.connect(lambda: self.play_button.setText("Play"))

        return widget

    def _create_status_bar(self) -> None:
        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(DARK_STYLESHEET)

    # Slots
    def on_time_changed(self, time: float) -> None:
        self.current_time_label.setText(self._format_time(time))
        self.time_label.setText(f"Time: {time:.3f}s")

        if self.timeline.duration > 0:
            slider_pos = int((time / self.timeline.duration) * 10000)
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(slider_pos)
            self.timeline_slider.blockSignals(False)

        if self.current_lap:
            self._update_visualizations_at_time(time)

    def on_slider_moved(self, value: int) -> None:
        if self.timeline.duration > 0:
            time = (value / 10000.0) * self.timeline.duration
            self.timeline.seek(time)

    def on_speed_changed(self, text: str) -> None:
        speed_map = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "2x": 2.0, "4x": 4.0}
        self.timeline.set_playback_speed(speed_map.get(text, 1.0))

    def on_zoom_changed(self, text: str) -> None:
        zoom_map = {"15s": 15.0, "30s": 30.0, "45s": 45.0, "60s": 60.0, "Full Lap": None}
        window = zoom_map.get(text)
        for canvas in self._active_canvases:
            canvas.window_duration = window
            canvas.update_sliding_window(self.timeline.current_time if self.timeline else 0.0)

    MAX_GRAPHS = 6

    def _on_graph_toggled(self) -> None:
        """Rebuild enabled graphs set and repopulate canvases (max 6)."""
        new_enabled = set()
        for key, action in self._graph_actions.items():
            if action.isChecked():
                new_enabled.add(key)

        if len(new_enabled) > self.MAX_GRAPHS:
            # Find which graph was just toggled on and uncheck it
            newly_added = new_enabled - self._enabled_graphs
            for key in newly_added:
                self._graph_actions[key].setChecked(False)
            self.statusBar().showMessage(f"Maximum {self.MAX_GRAPHS} graphs allowed (3 per column)", 3000)
            return

        self._enabled_graphs = new_enabled
        if self.current_lap:
            self._populate_canvases()
            self._update_visualizations_at_time(self.timeline.current_time if self.timeline else 0.0)

    def on_lap_selected(self, item: QtWidgets.QListWidgetItem) -> None:
        if not self.session:
            return
        lap_number = item.data(QtCore.Qt.UserRole)
        self.load_lap(lap_number)

    def open_session(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Telemetry Session", "",
            "Telemetry CSV (*.csv);;All Files (*)"
        )
        if file_path:
            self.load_session_from_file(file_path)

    def load_session_from_file(self, file_path: str) -> None:
        """Load session from CSV file."""
        try:
            from data import TelemetryLoader
            self.session = TelemetryLoader.load_session(file_path)
            self.status_bar.showMessage(f"Loaded session: {self.session.metadata.track_name}")
            if self.analysis_refresh_button:
                self.analysis_refresh_button.setEnabled(True)

            self.lap_list.clear()
            for lap in self.session.laps:
                item = QtWidgets.QListWidgetItem(f"Lap {lap.lap_number} - {lap.lap_time:.3f}s")
                item.setData(QtCore.Qt.UserRole, lap.lap_number)
                self.lap_list.addItem(item)

            if self.session.laps:
                self.load_lap(self.session.laps[0].lap_number)

        except Exception as e:
            if self.analysis_refresh_button:
                self.analysis_refresh_button.setEnabled(False)
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load session:\n{str(e)}")

    def load_lap(self, lap_number: int) -> None:
        """Load specific lap for visualization."""
        if not self.session:
            return

        self.current_lap = self.session.get_lap(lap_number)
        if not self.current_lap:
            return

        self.timeline.set_duration(self.current_lap.lap_time)
        self.duration_label.setText(self._format_time(self.current_lap.lap_time))

        self.timeline.stop()
        self._populate_canvases()
        self._update_visualizations_at_time(0.0)
        self.refresh_analysis()

        self.status_bar.showMessage(f"Loaded Lap {lap_number}")

    def refresh_analysis(self) -> None:
        """Run AI analysis for the currently loaded lap."""
        if not self.session or not self.current_lap:
            self._set_analysis_text(
                coach_text="Load a session and select a lap to run AI analysis.",
                analyst_text="Load a session and select a lap to run AI analysis.",
                source_text="Source: --",
            )
            if self.analysis_context_label:
                self.analysis_context_label.setText("No session loaded")
            return

        if self.analysis_context_label:
            self.analysis_context_label.setText(
                f"Session {self.session.metadata.session_id} | Lap {self.current_lap.lap_number} | {self.current_lap.lap_time:.3f}s"
            )

        self._analysis_request_id += 1
        request_id = self._analysis_request_id
        session = self.session
        lap = self.current_lap

        if self.analysis_refresh_button:
            self.analysis_refresh_button.setEnabled(False)

        self._set_analysis_text(
            coach_text="Running coach analysis...\nIf the Hugging Face Space is cold, the first request can take a few minutes.",
            analyst_text="Running analyst analysis...\nIf the Hugging Face Space is cold, the first request can take a few minutes.",
            source_text="Source: running...",
        )

        threading.Thread(
            target=self._run_analysis_worker,
            args=(request_id, session, lap),
            daemon=True,
        ).start()

    def _run_analysis_worker(self, request_id: int, session: Session, lap: Lap) -> None:
        try:
            result = self.ai_pipeline.generate(session, lap)
            self.analysis_finished.emit(request_id, result, "")
        except Exception as exc:
            self.analysis_finished.emit(request_id, None, str(exc))

    def _on_analysis_finished(self, request_id: int, result: object, error_text: str) -> None:
        if request_id != self._analysis_request_id:
            return

        if self.analysis_refresh_button:
            self.analysis_refresh_button.setEnabled(bool(self.session and self.current_lap))

        if error_text:
            self._set_analysis_text(
                coach_text=f"Coach analysis failed:\n{error_text}",
                analyst_text=f"Analyst analysis failed:\n{error_text}",
                source_text="Source: error",
            )
            return

        if result is None:
            self._set_analysis_text(
                coach_text="Coach analysis returned no data.",
                analyst_text="Analyst analysis returned no data.",
                source_text="Source: empty",
            )
            return

        generated_at = datetime.now().strftime("%H:%M:%S")
        self._set_analysis_text(
            coach_text=result.coach,
            analyst_text=result.analyst,
            source_text=f"Source: {result.source} ({generated_at})",
        )

    def _set_analysis_text(self, coach_text: str, analyst_text: str, source_text: str) -> None:
        if self.coach_output:
            self.coach_output.setPlainText(coach_text)
        if self.analyst_output:
            self.analyst_output.setPlainText(analyst_text)
        if self.analysis_source_label:
            self.analysis_source_label.setText(source_text)

    def _clear_layout(self, layout):
        """Remove all widgets from a layout."""
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def _col_exists(self, df, col):
        """Check if column exists and has non-zero data."""
        return col in df.columns and df[col].abs().sum() > 0

    def _populate_canvases(self) -> None:
        """Create and populate canvases based on enabled graphs."""
        if not self.current_lap:
            return

        from ui.styles import ACCENT_GREEN, ACCENT_RED, ACCENT_CYAN, ACCENT_YELLOW

        df = self.current_lap.telemetry
        times = df['elapsed_time'].values

        self._clear_layout(self.graph_layout_left)
        self._clear_layout(self.graph_layout_right)
        self._clear_layout(self.track_map_container)
        self._active_canvases = []

        # Track map (always shown)
        self.track_map = TrackMapCanvas(width=3, height=3)
        x, z = self.current_lap.get_racing_line()
        speeds = self.current_lap.get_speed_trace()
        self.track_map.load_lap(x, z, speeds)
        self.track_map_container.addWidget(self.track_map)

        # Build canvases for enabled graphs
        canvases = []
        for key, title, gtype, config in self.GRAPH_DEFS:
            if key not in self._enabled_graphs:
                continue

            canvas = TimeSeriesCanvas(width=5, height=2.5)

            if gtype == "single":
                col = config["col"]
                if not self._col_exists(df, col):
                    continue
                canvas.plot_single_line(times, df[col].values, ylabel=config["ylabel"], title=title)

            elif gtype == "multi_tb":
                if not (self._col_exists(df, "throttle") or self._col_exists(df, "brake")):
                    continue
                canvas.plot_multi_line(
                    times,
                    [df['throttle'].values * 100, df['brake'].values * 100],
                    labels=['Throttle', 'Brake'],
                    colors=[ACCENT_GREEN, ACCENT_RED],
                    ylabel="Input [%]", title=title
                )

            elif gtype == "multi_tyre":
                prefix = config["prefix"]
                cols = [f"{prefix}_fl", f"{prefix}_fr", f"{prefix}_rl", f"{prefix}_rr"]
                if not any(self._col_exists(df, c) for c in cols):
                    continue
                canvas.plot_multi_line(
                    times,
                    [df[c].values if c in df.columns else times * 0 for c in cols],
                    labels=TIRE_LABELS, colors=TIRE_COLORS,
                    ylabel=config["ylabel"], title=title
                )

            elif gtype == "multi_gforce":
                if not (self._col_exists(df, "g_force_lat") or self._col_exists(df, "g_force_lon")):
                    continue
                vals = []
                labels = []
                colors = []
                if self._col_exists(df, "g_force_lat"):
                    vals.append(df["g_force_lat"].values)
                    labels.append("Lateral")
                    colors.append(ACCENT_CYAN)
                if self._col_exists(df, "g_force_lon"):
                    vals.append(df["g_force_lon"].values)
                    labels.append("Longitudinal")
                    colors.append(ACCENT_YELLOW)
                canvas.plot_multi_line(times, vals, labels=labels, colors=colors,
                                       ylabel="G-Force", title=title)

            elif gtype == "multi_rh":
                if not (self._col_exists(df, "ride_height_front") or self._col_exists(df, "ride_height_rear")):
                    continue
                vals, labels, colors = [], [], []
                if self._col_exists(df, "ride_height_front"):
                    vals.append(df["ride_height_front"].values)
                    labels.append("Front")
                    colors.append(ACCENT_CYAN)
                if self._col_exists(df, "ride_height_rear"):
                    vals.append(df["ride_height_rear"].values)
                    labels.append("Rear")
                    colors.append(ACCENT_YELLOW)
                canvas.plot_multi_line(times, vals, labels=labels, colors=colors,
                                       ylabel="Height [m]", title=title)

            elif gtype == "multi_damage":
                dmg_cols = ["car_damage_front", "car_damage_rear", "car_damage_left",
                            "car_damage_right", "car_damage_centre"]
                if not any(self._col_exists(df, c) for c in dmg_cols):
                    continue
                dmg_labels = ["Front", "Rear", "Left", "Right", "Centre"]
                dmg_colors = [ACCENT_RED, ACCENT_YELLOW, ACCENT_CYAN, ACCENT_GREEN, "#FFFFFF"]
                canvas.plot_multi_line(
                    times,
                    [df[c].values if c in df.columns else times * 0 for c in dmg_cols],
                    labels=dmg_labels, colors=dmg_colors,
                    ylabel="Damage", title=title
                )
            else:
                continue

            canvases.append(canvas)
            self._active_canvases.append(canvas)

        # Split canvases evenly between left and right columns
        mid = (len(canvases) + 1) // 2
        for canvas in canvases[:mid]:
            self.graph_layout_left.addWidget(canvas)
        for canvas in canvases[mid:]:
            self.graph_layout_right.addWidget(canvas)

    def _update_visualizations_at_time(self, time: float) -> None:
        """Update all visualizations to show data at the given timestamp."""
        if not self.current_lap:
            return

        df = self.current_lap.telemetry
        idx = (df['elapsed_time'] - time).abs().idxmin()
        sample = df.loc[idx]
        sample_index = df.index.get_loc(idx)

        self.speed_label.setText(f"Speed: {sample['speed']:.1f} km/h")
        self.gear_label.setText(f"Gear: {int(sample['gear'])}")
        self.rpm_label.setText(f"RPM: {int(sample['rpm'])}")
        self.throttle_label.setText(f"Throttle: {sample['throttle']*100:.0f}%")
        self.brake_label.setText(f"Brake: {sample['brake']*100:.0f}%")

        if self.track_map:
            self.track_map.update_position(sample_index)

        for canvas in self._active_canvases:
            canvas.update_timeline_marker(time)

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    @staticmethod
    def _format_time(seconds: float) -> str:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}:{secs:06.3f}"
