"""
Main lap viewer window - post-race telemetry visualization.
"""
from datetime import datetime
import json
import os
import threading
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from typing import Optional

from data import Session, Lap
from analysis import AIPipelineBridge
from .timeline_controller import TimelineController
from ui.styles import (
    ACCENT_PRIMARY,
    BG_COLOR,
    BG_COLOR_LIGHT,
    BORDER_COLOR,
    DARK_STYLESHEET,
    FONT_HEADING,
    TEXT_COLOR,
    TEXT_COLOR_DIM,
    TIRE_COLORS,
    TIRE_LABELS,
)
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
    analyst_finished = QtCore.pyqtSignal(int, str, str)  # (request_id, analyst_text, source)
    analyst_chunk = QtCore.pyqtSignal(int, str)  # (request_id, text)
    coach_finished = QtCore.pyqtSignal(int, str, str)  # (request_id, coach_text, error)
    coach_chunk = QtCore.pyqtSignal(int, str)   # (request_id, text)
    POSTRACE_RENDER_HZ = 20.0

    def __init__(self):
        super().__init__()

        # Data
        self.session: Optional[Session] = None
        self.current_lap: Optional[Lap] = None
        self._render_telemetry_df = None
        self._elapsed_times: Optional[np.ndarray] = None
        self._telemetry_columns: dict[str, Optional[np.ndarray]] = {}
        self._last_sample_index: int = -1

        # Canvases (created when lap is loaded)
        self.track_map: Optional[TrackMapCanvas] = None
        self._active_canvases: list = []

        # Timeline controller
        self.timeline = TimelineController(self)
        self.timeline.time_changed.connect(self.on_time_changed)
        self.ai_pipeline = AIPipelineBridge()
        self.analyst_finished.connect(self._on_analyst_finished)
        self.analyst_chunk.connect(self._on_analyst_chunk)
        self.coach_finished.connect(self._on_coach_finished)
        self.coach_chunk.connect(self._on_coach_chunk)
        self._analysis_request_id = 0
        self._analyst_request_id = 0
        self._analyst_running = False
        self._analyst_done_event = threading.Event()
        self._analyst_done_event.set()  # not blocking initially
        self._analyst_cancel_event = threading.Event()

        # Analysis tab widgets
        self.analysis_context_label: Optional[QtWidgets.QLabel] = None
        self.analysis_source_label: Optional[QtWidgets.QLabel] = None
        self.coach_output: Optional[QtWidgets.QTextEdit] = None
        self.analyst_output: Optional[QtWidgets.QTextEdit] = None

        # Coach chatbot state
        self._coach_conversation: list[dict] = []
        self._coach_streaming = False
        self._coach_input: Optional[QtWidgets.QLineEdit] = None
        self._coach_send_button: Optional[QtWidgets.QPushButton] = None
        self._current_stream_text: list[str] = []
        self.exit_application_requested: bool = False
        self._pending_scrub_time: Optional[float] = None
        self._scrub_timer = QtCore.QTimer(self)
        self._scrub_timer.setSingleShot(True)
        self._scrub_timer.setInterval(33)  # cap scrub redraw rate to ~30 FPS
        self._scrub_timer.timeout.connect(self._flush_pending_scrub_seek)

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
        menu_bar.setNativeMenuBar(False)

        file_menu = menu_bar.addMenu("File")

        open_action = QtWidgets.QAction("Open Session...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_session)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        import_session_action = QtWidgets.QAction("Import Session...", self)
        import_session_action.setShortcut("Ctrl+I")
        import_session_action.triggered.connect(self.import_session)
        file_menu.addAction(import_session_action)

        export_session_action = QtWidgets.QAction("Export Session...", self)
        export_session_action.setShortcut("Ctrl+E")
        export_session_action.triggered.connect(self.export_session)
        file_menu.addAction(export_session_action)

        file_menu.addSeparator()

        back_action = QtWidgets.QAction("Back to Launcher", self)
        back_action.setShortcut("Ctrl+W")
        back_action.triggered.connect(self.close)
        file_menu.addAction(back_action)

        exit_action = QtWidgets.QAction("Exit Application", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self._exit_application)
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
        header_label.setStyleSheet("color: #aaa; padding: 6px 12px; font-size: 13pt;")
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

        self.timeline_widget = self._create_timeline_widget()
        main_layout.addWidget(self.timeline_widget)

        self._style_main_tabs()
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

    def _style_main_tabs(self) -> None:
        """Make active tab state obvious for Lap Review vs Analysis."""
        self.tab_widget.setStyleSheet(
            f"""
            QTabBar::tab {{
                background-color: {BG_COLOR_LIGHT};
                color: {TEXT_COLOR_DIM};
                border: 1px solid {BORDER_COLOR};
                border-bottom: 2px solid {BORDER_COLOR};
                padding: 10px 22px;
                margin-right: 4px;
                font-size: 8pt;
                font-weight: 600;
            }}
            QTabBar::tab:hover:!selected {{
                color: {TEXT_COLOR};
                border-bottom: 2px solid {TEXT_COLOR_DIM};
            }}
            QTabBar::tab:selected {{
                background-color: {BG_COLOR};
                color: {TEXT_COLOR};
                border: 1px solid {ACCENT_PRIMARY};
                border-bottom: 3px solid {ACCENT_PRIMARY};
                font-weight: 700;
            }}
            """
        )

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
        self.analysis_context_label.setStyleSheet("font-size: 14pt; color: #AAAAAA;")
        header_row.addWidget(self.analysis_context_label)
        header_row.addStretch()
        layout.addLayout(header_row)

        # Analyst source section hidden per UI request.
        # self.analysis_source_label = QtWidgets.QLabel("Source: --")
        # self.analysis_source_label.setStyleSheet("font-size: 13pt; color: #888888;")
        # layout.addWidget(self.analysis_source_label)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        coach_group = QtWidgets.QGroupBox("Coach")
        coach_layout = QtWidgets.QVBoxLayout(coach_group)
        self.coach_output = QtWidgets.QTextEdit()
        self.coach_output.setReadOnly(True)
        self.coach_output.setPlaceholderText("Coach model output will appear here.")
        coach_layout.addWidget(self.coach_output, stretch=1)

        chat_row = QtWidgets.QHBoxLayout()
        self._coach_input = QtWidgets.QLineEdit()
        self._coach_input.setPlaceholderText("Ask the coach...")
        self._coach_input.setEnabled(False)
        self._coach_input.returnPressed.connect(self._send_coach_followup)
        chat_row.addWidget(self._coach_input)
        self._coach_send_button = QtWidgets.QPushButton("Send")
        self._coach_send_button.setEnabled(False)
        self._coach_send_button.clicked.connect(self._send_coach_followup)
        chat_row.addWidget(self._coach_send_button)
        coach_layout.addLayout(chat_row)

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
            coach_text="Load a session and select a lap to start coaching.",
            analyst_text="Session analyst will run automatically when a session is loaded.",
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

        title = QtWidgets.QLabel("LAPS")
        title.setStyleSheet(f"font-family: '{FONT_HEADING}'; font-size: 18px; padding: 4px;")
        layout.addWidget(title)

        self.lap_list = QtWidgets.QListWidget()
        self.lap_list.setMaximumHeight(120)
        self.lap_list.itemClicked.connect(self.on_lap_selected)
        layout.addWidget(self.lap_list)

        track_label = QtWidgets.QLabel("TRACK MAP")
        track_label.setStyleSheet(f"font-family: '{FONT_HEADING}'; font-size: 16px; padding: 4px;")
        layout.addWidget(track_label)

        self.track_map_container = QtWidgets.QVBoxLayout()
        placeholder = QtWidgets.QLabel("Load a lap to view track map")
        placeholder.setAlignment(QtCore.Qt.AlignCenter)
        placeholder.setStyleSheet("color: #888; padding: 20px;")
        self.track_map_container.addWidget(placeholder)
        layout.addLayout(self.track_map_container)

        # Lap info section (moved here from the separate right panel)
        info_label = QtWidgets.QLabel("LAP INFO")
        info_label.setStyleSheet(f"font-family: '{FONT_HEADING}'; font-size: 16px; padding: 4px;")
        layout.addWidget(info_label)

        self.selected_lap_label = QtWidgets.QLabel("Viewing: Lap --")
        self.selected_lap_label.setStyleSheet(
            f"font-family: '{FONT_HEADING}'; font-size: 14px; padding: 2px 8px; color: {ACCENT_PRIMARY};"
        )
        layout.addWidget(self.selected_lap_label)

        self.time_label = QtWidgets.QLabel("Time: 0.00s")
        self.time_label.setStyleSheet("font-size: 14pt; padding: 2px 8px;")
        layout.addWidget(self.time_label)

        self.speed_label = QtWidgets.QLabel("Speed: -- km/h")
        self.gear_label = QtWidgets.QLabel("Gear: --")
        self.rpm_label = QtWidgets.QLabel("RPM: ----")
        self.throttle_label = QtWidgets.QLabel("Throttle: --%")
        self.brake_label = QtWidgets.QLabel("Brake: --%")

        for label in [self.speed_label, self.gear_label, self.rpm_label,
                      self.throttle_label, self.brake_label]:
            label.setStyleSheet("padding: 1px 8px; font-size: 14pt;")
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
        outer_layout.addLayout(self.graph_layout_left, 1)

        # Right column: Throttle/Brake, Tire Temps, Tire Pressures
        self.graph_layout_right = QtWidgets.QVBoxLayout()
        self.graph_layout_right.setSpacing(6)
        outer_layout.addLayout(self.graph_layout_right, 1)

        # Placeholder (spans both columns initially)
        self.graph_placeholder = QtWidgets.QLabel("Load a session to view telemetry")
        self.graph_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.graph_placeholder.setStyleSheet(f"font-family: '{FONT_HEADING}'; font-size: 16px; color: #888; padding: 50px;")
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
        self.timeline_slider.sliderReleased.connect(self.on_slider_released)
        layout.addWidget(self.timeline_slider)

        controls_row = QtWidgets.QHBoxLayout()

        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.setFixedSize(60, 30)
        self.play_button.clicked.connect(self.timeline.toggle_play_pause)
        controls_row.addWidget(self.play_button)

        controls_row.addSpacing(24)
        controls_row.addWidget(QtWidgets.QLabel("Speed:"))
        self.speed_combo = QtWidgets.QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self.speed_combo.setCurrentText("1x")
        self.speed_combo.currentTextChanged.connect(self.on_speed_changed)
        controls_row.addWidget(self.speed_combo)

        controls_row.addStretch()
        controls_row.addWidget(QtWidgets.QLabel("Graphs X Zoom:"))
        self.zoom_combo = QtWidgets.QComboBox()
        self.zoom_combo.addItems(["15s", "30s", "45s", "60s", "Full Lap"])
        self.zoom_combo.setCurrentText("45s")
        self.zoom_combo.currentTextChanged.connect(self.on_zoom_changed)
        controls_row.addWidget(self.zoom_combo)

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

    def _exit_application(self) -> None:
        """Close this window and request full app shutdown."""
        self.exit_application_requested = True
        self.close()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.quit()

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
            self._request_scrub_seek(time)

    def on_slider_released(self) -> None:
        self._scrub_timer.stop()
        self._flush_pending_scrub_seek()

    def _request_scrub_seek(self, time: float) -> None:
        self._pending_scrub_time = time
        if not self._scrub_timer.isActive():
            # Apply immediately for low perceived latency, then throttle follow-ups.
            self._flush_pending_scrub_seek()
            self._scrub_timer.start()

    def _flush_pending_scrub_seek(self) -> None:
        if self._pending_scrub_time is None:
            return
        time = self._pending_scrub_time
        self._pending_scrub_time = None
        self.timeline.seek(time)
        if self._pending_scrub_time is not None and not self._scrub_timer.isActive():
            self._scrub_timer.start()

    def on_speed_changed(self, text: str) -> None:
        speed_map = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "2x": 2.0, "4x": 4.0}
        self.timeline.set_playback_speed(speed_map.get(text, 1.0))

    def on_zoom_changed(self, text: str) -> None:
        zoom_map = {"15s": 15.0, "30s": 30.0, "45s": 45.0, "60s": 60.0, "Full Lap": None}
        window = zoom_map.get(text)
        current_time = self.timeline.current_time if self.timeline else 0.0
        for canvas in self._active_canvases:
            canvas.window_duration = window
            # Force immediate visual update even when playback is paused.
            canvas.update_timeline_marker(current_time)

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
            # New session always supersedes any in-flight analyst run.
            self._cancel_analyst_run()
            from data import TelemetryLoader
            self.session = TelemetryLoader.load_session(file_path)
            self.status_bar.showMessage(f"Loaded session: {self.session.metadata.track_name}")

            self.lap_list.clear()
            self.selected_lap_label.setText("Viewing: Lap --")
            for lap in self.session.laps:
                item = QtWidgets.QListWidgetItem(f"Lap {lap.lap_number} - {lap.lap_time:.3f}s")
                item.setData(QtCore.Qt.UserRole, lap.lap_number)
                self.lap_list.addItem(item)

            # Start session-level analyst immediately in background
            self._start_session_analyst()

            if self.session.laps:
                self.load_lap(self.session.laps[0].lap_number)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load session:\n{str(e)}")

    def load_lap(self, lap_number: int) -> None:
        """Load specific lap for visualization."""
        if not self.session:
            return

        self.current_lap = self.session.get_lap(lap_number)
        if not self.current_lap:
            return

        self._sync_lap_list_selection(lap_number)
        self.selected_lap_label.setText(f"Viewing: Lap {lap_number}")

        self._scrub_timer.stop()
        self._pending_scrub_time = None
        self._last_sample_index = -1
        telemetry_df = self._downsample_telemetry_for_render(self.current_lap.telemetry)
        self._render_telemetry_df = telemetry_df
        self._elapsed_times = (
            telemetry_df["elapsed_time"].to_numpy(copy=False)
            if "elapsed_time" in telemetry_df.columns
            else None
        )
        self._telemetry_columns = {
            "speed": telemetry_df["speed"].to_numpy(copy=False) if "speed" in telemetry_df.columns else None,
            "gear": telemetry_df["gear"].to_numpy(copy=False) if "gear" in telemetry_df.columns else None,
            "rpm": telemetry_df["rpm"].to_numpy(copy=False) if "rpm" in telemetry_df.columns else None,
            "throttle": telemetry_df["throttle"].to_numpy(copy=False) if "throttle" in telemetry_df.columns else None,
            "brake": telemetry_df["brake"].to_numpy(copy=False) if "brake" in telemetry_df.columns else None,
        }

        self.timeline.set_duration(self.current_lap.lap_time)
        self.duration_label.setText(self._format_time(self.current_lap.lap_time))

        self.timeline.stop()
        self._populate_canvases()
        self._update_visualizations_at_time(0.0)
        self.refresh_analysis()

        if self.analysis_context_label:
            self.analysis_context_label.setText(self._get_analysis_session_name())

        self.status_bar.showMessage(f"Loaded Lap {lap_number}")

    def _downsample_telemetry_for_render(self, telemetry_df):
        """Downsample lap telemetry for post-race rendering only (keeps recording fidelity unchanged)."""
        if telemetry_df is None or telemetry_df.empty:
            return telemetry_df
        if "elapsed_time" not in telemetry_df.columns:
            return telemetry_df
        if self.POSTRACE_RENDER_HZ <= 0:
            return telemetry_df

        times = telemetry_df["elapsed_time"].to_numpy(copy=False)
        if times.size <= 2:
            return telemetry_df

        sample_period = 1.0 / self.POSTRACE_RENDER_HZ
        if sample_period <= 0:
            return telemetry_df

        # Keep first sample per 20 Hz time bucket and always include the final sample.
        bucket_ids = np.floor(times / sample_period).astype(np.int64)
        keep_mask = np.empty(times.size, dtype=bool)
        keep_mask[0] = True
        keep_mask[1:] = bucket_ids[1:] != bucket_ids[:-1]
        keep_indices = np.flatnonzero(keep_mask)

        last_idx = times.size - 1
        if keep_indices.size == 0 or int(keep_indices[-1]) != last_idx:
            keep_indices = np.append(keep_indices, last_idx)

        if keep_indices.size >= times.size:
            return telemetry_df

        return telemetry_df.iloc[keep_indices].reset_index(drop=True)

    def _sync_lap_list_selection(self, lap_number: int) -> None:
        """Keep lap list highlight aligned with the currently loaded lap."""
        for i in range(self.lap_list.count()):
            item = self.lap_list.item(i)
            if item and item.data(QtCore.Qt.UserRole) == lap_number:
                self.lap_list.setCurrentItem(item)
                return

    def _start_session_analyst(self) -> None:
        """Kick off session-level analyst in background as soon as session loads."""
        if not self.session:
            return

        self._cancel_analyst_run()
        self._analyst_cancel_event = threading.Event()
        request_id = self._analyst_request_id
        self._analyst_running = True
        self._analyst_done_event.clear()
        if self.analyst_output:
            self.analyst_output.clear()
        session = self.session
        threading.Thread(
            target=self._run_analyst_worker,
            args=(request_id, session, self._analyst_cancel_event),
            daemon=True,
        ).start()

    def _cancel_analyst_run(self) -> None:
        """Cancel the current analyst run and invalidate stale callbacks."""
        self._analyst_request_id += 1
        self._analyst_cancel_event.set()
        self._analyst_running = False
        self._analyst_done_event.set()

    def _run_analyst_worker(
        self,
        request_id: int,
        session: Session,
        cancel_event: threading.Event,
    ) -> None:
        try:
            stream = self.ai_pipeline.generate_analyst_stream(session)
            if stream is not None:
                for chunk in stream:
                    if cancel_event.is_set() or request_id != self._analyst_request_id:
                        close_fn = getattr(stream, "close", None)
                        if callable(close_fn):
                            close_fn()
                        return
                    self.analyst_chunk.emit(request_id, chunk)
                source = f"external:{self.ai_pipeline._external_source}"
                self.analyst_finished.emit(request_id, "", source)
            else:
                # Streaming unavailable — fall back to non-streaming
                text = self.ai_pipeline.generate_analyst(session)
                if cancel_event.is_set() or request_id != self._analyst_request_id:
                    return
                if text:
                    source = f"external:{self.ai_pipeline._external_source}"
                    self.analyst_finished.emit(request_id, text, source)
                else:
                    self.analyst_finished.emit(request_id, "", "")
        except Exception as exc:
            if cancel_event.is_set() or request_id != self._analyst_request_id:
                return
            self.analyst_finished.emit(request_id, f"Analyst failed:\n{exc}", "error")

    def _on_analyst_chunk(self, request_id: int, text: str) -> None:
        if request_id != self._analyst_request_id:
            return
        if self.analyst_output:
            self.analyst_output.moveCursor(QtGui.QTextCursor.End)
            self.analyst_output.insertPlainText(text)

    def _on_coach_chunk(self, request_id: int, text: str) -> None:
        if request_id != self._analysis_request_id:
            return
        self._current_stream_text.append(text)
        if self.coach_output:
            self.coach_output.moveCursor(QtGui.QTextCursor.End)
            self.coach_output.insertPlainText(text)

    def _on_analyst_finished(self, request_id: int, analyst_text: str, source: str) -> None:
        if request_id != self._analyst_request_id:
            return
        self._analyst_running = False
        self._analyst_done_event.set()
        if analyst_text:
            # Non-streaming path — replace content with full text
            if self.analyst_output:
                self.analyst_output.setPlainText(analyst_text)
        elif not source:
            # No text and no source — analyst unavailable
            if self.analyst_output:
                current = self.analyst_output.toPlainText().strip()
                if not current:
                    self.analyst_output.setPlainText("Session analyst not available.")
        # else: streaming path finished (text already appended via chunks)
        if source and source != "error":
            generated_at = datetime.now().strftime("%H:%M:%S")
            if self.analysis_source_label:
                self.analysis_source_label.setText(f"Analyst source: {source} ({generated_at})")

    def refresh_analysis(self) -> None:
        """Run coach analysis for the currently loaded lap (analyst runs on session load)."""
        if not self.session or not self.current_lap:
            if self.coach_output:
                self.coach_output.setPlainText("Load a session and select a lap to start coaching.")
            if self.analysis_context_label:
                self.analysis_context_label.setText("No session loaded")
            return

        if self.analysis_context_label:
            self.analysis_context_label.setText(self._get_analysis_session_name())

        # Reset conversation state for new lap
        self._coach_conversation = []
        self._current_stream_text = []
        self._coach_streaming = True
        if self._coach_input:
            self._coach_input.setEnabled(False)
        if self._coach_send_button:
            self._coach_send_button.setEnabled(False)

        self._analysis_request_id += 1
        request_id = self._analysis_request_id
        session = self.session
        lap = self.current_lap

        if self.coach_output:
            self.coach_output.clear()

        threading.Thread(
            target=self._run_coach_worker,
            args=(request_id, session, lap),
            daemon=True,
        ).start()

    def _get_analysis_session_name(self) -> str:
        """Return the session name shown in the analysis header."""
        if not self.session:
            return "No session loaded"

        metadata = self.session.metadata
        for attr in ("notes", "session_name", "display_name", "name"):
            value = getattr(metadata, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if value:
                return str(value)

        return f"Session {metadata.session_id}"

    def _run_coach_worker(self, request_id: int, session: Session, lap: Lap) -> None:
        # Wait for analyst to finish so it always predates the coach.
        # Both share one LLM lock; without this the coach's lighter setup
        # lets it grab the lock first.
        self._analyst_done_event.wait()
        try:
            stream = self.ai_pipeline.generate_coach_stream(session, lap)
            if stream is not None:
                for chunk in stream:
                    self.coach_chunk.emit(request_id, chunk)
                self.coach_finished.emit(request_id, "", "")
            else:
                # Streaming unavailable — fall back to non-streaming
                coach_text = self.ai_pipeline.generate_coach(session, lap)
                if not coach_text:
                    fallback = self.ai_pipeline._generate_fallback(session, lap)
                    coach_text = fallback.coach
                self.coach_finished.emit(request_id, coach_text or "", "")
        except Exception as exc:
            self.coach_finished.emit(request_id, "", str(exc))

    def _on_coach_finished(self, request_id: int, coach_text: str, error_text: str) -> None:
        """Handle coach analysis or follow-up completion."""
        if request_id != self._analysis_request_id:
            return

        self._coach_streaming = False

        if error_text:
            if self.coach_output:
                self.coach_output.moveCursor(QtGui.QTextCursor.End)
                self.coach_output.insertPlainText(f"\n\nCoach analysis failed:\n{error_text}")
            self._enable_coach_input()
            return

        # Capture the assistant's response for conversation history
        if coach_text:
            # Non-streaming path — replace content with full text
            response = coach_text
            if self.coach_output:
                self.coach_output.setPlainText(coach_text)
        else:
            # Streaming path finished (text already appended via chunks)
            response = "".join(self._current_stream_text).strip()
            if not response and self.coach_output:
                current = self.coach_output.toPlainText().strip()
                if not current:
                    self.coach_output.setPlainText("Coach analysis returned no data.")

        if response:
            self._coach_conversation.append({"role": "assistant", "content": response})

        self._enable_coach_input()

    def _enable_coach_input(self) -> None:
        """Enable the chat input field and send button."""
        if self._coach_input:
            self._coach_input.setEnabled(True)
            self._coach_input.setFocus()
        if self._coach_send_button:
            self._coach_send_button.setEnabled(True)

    def _send_coach_followup(self) -> None:
        """Send a follow-up question to the coach."""
        if not self.session or not self.current_lap:
            return
        if self._coach_streaming:
            return
        if not self._coach_input:
            return

        question = self._coach_input.text().strip()
        if not question:
            return

        # Add user question to conversation history
        self._coach_conversation.append({"role": "user", "content": question})

        # Display in widget with turquoise colour for user messages
        if self.coach_output:
            cursor = self.coach_output.textCursor()
            cursor.movePosition(QtGui.QTextCursor.End)
            cursor.insertText("\n\n")
            fmt = QtGui.QTextCharFormat()
            fmt.setForeground(QtGui.QColor("#40E0D0"))
            cursor.insertText(f"You: {question}", fmt)
            default_fmt = QtGui.QTextCharFormat()
            default_fmt.setForeground(QtGui.QColor("#DDDDDD"))
            cursor.insertText("\n\nCoach:\n", default_fmt)
            self.coach_output.setTextCursor(cursor)

        # Clear input and disable during generation
        self._coach_input.clear()
        self._coach_input.setEnabled(False)
        if self._coach_send_button:
            self._coach_send_button.setEnabled(False)
        self._coach_streaming = True
        self._current_stream_text = []

        self._analysis_request_id += 1
        request_id = self._analysis_request_id
        session = self.session
        lap = self.current_lap
        history_snapshot = list(self._coach_conversation)

        threading.Thread(
            target=self._run_coach_followup_worker,
            args=(request_id, session, lap, history_snapshot),
            daemon=True,
        ).start()

    def _run_coach_followup_worker(
        self, request_id: int, session: Session, lap: Lap, history: list[dict]
    ) -> None:
        """Background worker for coach follow-up streaming."""
        try:
            stream = self.ai_pipeline.generate_coach_followup_stream(session, lap, history)
            if stream is not None:
                for chunk in stream:
                    self.coach_chunk.emit(request_id, chunk)
                self.coach_finished.emit(request_id, "", "")
            else:
                self.coach_finished.emit(request_id, "", "Follow-up not available.")
        except Exception as exc:
            self.coach_finished.emit(request_id, "", str(exc))

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

        df = self._render_telemetry_df
        if df is None:
            df = self.current_lap.telemetry
        times = df['elapsed_time'].values

        self._clear_layout(self.graph_layout_left)
        self._clear_layout(self.graph_layout_right)
        self._clear_layout(self.track_map_container)
        self._active_canvases = []

        # Track map (always shown)
        self.track_map = TrackMapCanvas(width=3, height=3)
        x = df["pos_x"].to_numpy(copy=False) if "pos_x" in df.columns else np.array([])
        z = df["pos_z"].to_numpy(copy=False) if "pos_z" in df.columns else np.array([])
        speeds = df["speed"].to_numpy(copy=False) if "speed" in df.columns else np.array([])
        self.track_map.load_lap(x, z, speeds)
        self.track_map_container.addWidget(self.track_map)

        # Build canvases for enabled graphs
        canvases = []
        for key, title, gtype, config in self.GRAPH_DEFS:
            if key not in self._enabled_graphs:
                continue

            canvas = TimeSeriesCanvas(width=5, height=2.5)
            canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

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
        if not self.current_lap or self._elapsed_times is None or self._elapsed_times.size == 0:
            return

        sample_index = self._find_nearest_sample_index(time)
        if sample_index < 0:
            return

        speed_vals = self._telemetry_columns.get("speed")
        if speed_vals is not None:
            self.speed_label.setText(f"Speed: {float(speed_vals[sample_index]):.1f} km/h")

        gear_vals = self._telemetry_columns.get("gear")
        if gear_vals is not None:
            self.gear_label.setText(f"Gear: {int(gear_vals[sample_index])}")

        rpm_vals = self._telemetry_columns.get("rpm")
        if rpm_vals is not None:
            self.rpm_label.setText(f"RPM: {int(rpm_vals[sample_index])}")

        throttle_vals = self._telemetry_columns.get("throttle")
        if throttle_vals is not None:
            self.throttle_label.setText(f"Throttle: {float(throttle_vals[sample_index]) * 100:.0f}%")

        brake_vals = self._telemetry_columns.get("brake")
        if brake_vals is not None:
            self.brake_label.setText(f"Brake: {float(brake_vals[sample_index]) * 100:.0f}%")

        if self.track_map and sample_index != self._last_sample_index:
            self.track_map.update_position(sample_index)
            self._last_sample_index = sample_index

        for canvas in self._active_canvases:
            canvas.update_timeline_marker(time)

    def _find_nearest_sample_index(self, time: float) -> int:
        """Find nearest telemetry sample index in O(log n) using cached elapsed times."""
        if self._elapsed_times is None or self._elapsed_times.size == 0:
            return -1

        insert_pos = int(np.searchsorted(self._elapsed_times, time, side="left"))
        if insert_pos <= 0:
            return 0
        if insert_pos >= self._elapsed_times.size:
            return int(self._elapsed_times.size - 1)

        prev_time = float(self._elapsed_times[insert_pos - 1])
        next_time = float(self._elapsed_times[insert_pos])
        if abs(next_time - time) < abs(time - prev_time):
            return insert_pos
        return insert_pos - 1

    def _on_tab_changed(self, index: int) -> None:
        """Show timeline only on Lap Review tab."""
        is_lap_review = (index == 0)
        self.timeline_widget.setVisible(is_lap_review)
        if not is_lap_review:
            self.timeline.pause()

    def export_session(self) -> None:
        """Export the full loaded session to a .jsession file."""
        if not self.session:
            QtWidgets.QMessageBox.warning(self, "Export", "No session loaded to export.")
            return

        default_name = f"session_{self.session.metadata.session_id}_{self.session.metadata.track_name}.jsession"
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Session", default_name,
            "Jarvis Session File (*.jsession);;All Files (*)"
        )
        if not file_path:
            return

        try:
            bundle = self._build_session_bundle_from_memory()
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(bundle, f)
            self.status_bar.showMessage(f"Session exported to {os.path.basename(file_path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export session:\n{str(e)}")

    def import_session(self) -> None:
        """Import a .jsession file and load it."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import Session", "",
            "Jarvis Session File (*.jsession);;All Files (*)"
        )
        if not file_path:
            return

        try:
            from data.session_exporter import SessionExporter
            exporter = SessionExporter()
            new_id = exporter.import_session_bundle(file_path)

            # Export to CSV so TelemetryLoader can load it
            export_dir = exporter.export_session(new_id)
            self.load_session_from_file(export_dir)
            self.status_bar.showMessage(f"Imported session {new_id} from {os.path.basename(file_path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Import Error", f"Failed to import session:\n{str(e)}")

    def _build_session_bundle_from_memory(self) -> dict:
        """Build a .jsession payload from the currently loaded in-memory session."""
        if not self.session:
            raise ValueError("No session loaded")

        metadata = self.session.metadata
        fastest_lap = self.session.get_fastest_lap()

        laps_payload = []
        for lap in self.session.laps:
            summary = lap.summary
            laps_payload.append(
                {
                    "lap_number": int(lap.lap_number),
                    "lap_time": float(lap.lap_time),
                    "fuel_start": float(summary.fuel_start),
                    "fuel_end": float(summary.fuel_end),
                    "avg_speed": float(summary.avg_speed),
                    "max_speed": float(summary.max_speed),
                    "min_speed": float(summary.min_speed),
                    "valid": bool(summary.valid),
                }
            )

        telemetry_payload: dict[str, list] = {}
        if self.session.telemetry is not None and not self.session.telemetry.empty:
            for col in self.session.telemetry.columns:
                if col in {"telemetry_id", "session_id", "timestamp"}:
                    continue
                telemetry_payload[col] = [
                    self._to_json_compatible(value)
                    for value in self.session.telemetry[col].tolist()
                ]

        ai_payload = []
        for comment in self.session.ai_commentary:
            timestamp = getattr(comment, "timestamp", None)
            if isinstance(timestamp, datetime):
                timestamp_value = float(timestamp.timestamp())
            elif hasattr(timestamp, "timestamp"):
                timestamp_value = float(timestamp.timestamp())
            else:
                timestamp_value = self._to_json_compatible(timestamp)

            ai_payload.append(
                {
                    "timestamp": timestamp_value,
                    "message": str(getattr(comment, "message", "")),
                    "trigger": str(getattr(comment, "trigger", "")),
                    "priority": int(getattr(comment, "priority", 2) or 2),
                    "lap_number": int(getattr(comment, "lap_number", 0) or 0),
                }
            )

        return {
            "version": 1,
            "metadata": {
                "game": metadata.game,
                "track_name": metadata.track_name,
                "car_model": metadata.car_model,
                "player_name": metadata.player_name,
                "total_laps": len(self.session.laps),
                "best_lap_time": float(fastest_lap.lap_time) if fastest_lap else None,
                "notes": str(getattr(metadata, "notes", "") or ""),
            },
            "laps": laps_payload,
            "telemetry": telemetry_payload,
            "ai_commentary": ai_payload,
        }

    @staticmethod
    def _to_json_compatible(value):
        """Convert numpy/pandas scalar types into plain JSON-compatible values."""
        if value is None:
            return None

        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass

        if isinstance(value, datetime):
            return float(value.timestamp())

        if isinstance(value, float):
            if np.isnan(value) or np.isinf(value):
                return None
            return value

        return value

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
