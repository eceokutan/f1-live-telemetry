"""
Time-series canvas with vertical timeline marker for scrubbing.
"""
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from ui.styles import (
    BG_COLOR, BG_COLOR_LIGHT, TEXT_COLOR_DIM, GRID_COLOR,
    ACCENT_BLUE, ACCENT_RED, ACCENT_CYAN, ACCENT_YELLOW, ACCENT_GREEN
)


class TimeSeriesCanvas(FigureCanvas):
    """
    Time-series graph with vertical marker showing current timeline position.

    Used for speed, RPM, gear, throttle, brake, etc.
    """

    def __init__(self, parent=None, width=8, height=2.5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        # Enforce minimum height so graphs don't get squished
        self.setMinimumHeight(150)

        # Apply dark theme
        self.fig.patch.set_facecolor(BG_COLOR)
        self.ax.set_facecolor(BG_COLOR_LIGHT)

        for spine in self.ax.spines.values():
            spine.set_color(TEXT_COLOR_DIM)
        self.ax.tick_params(colors=TEXT_COLOR_DIM, labelsize=8)
        self.ax.xaxis.label.set_color(TEXT_COLOR_DIM)
        self.ax.yaxis.label.set_color(TEXT_COLOR_DIM)
        self.ax.title.set_color("#FFFFFF")

        self.ax.set_xlabel("Time [s]", fontsize=9)
        self.ax.grid(True, color=GRID_COLOR, alpha=0.6)

        # Data
        self.times = None
        self.values = None
        self.line = None
        self.timeline_marker = None

        # Sliding window configuration
        self.window_duration = 45.0  # seconds of lap visible at once
        self.lap_duration = 0.0

        self.fig.tight_layout(pad=0.5)

    def plot_single_line(self, times, values, ylabel, color=ACCENT_BLUE, title=""):
        """Plot a single time-series line."""
        self.times = times
        self.values = values
        self.lap_duration = float(times[-1]) if len(times) > 0 else 0.0

        self.ax.clear()
        self.ax.set_facecolor(BG_COLOR_LIGHT)
        for spine in self.ax.spines.values():
            spine.set_color(TEXT_COLOR_DIM)
        self.ax.tick_params(colors=TEXT_COLOR_DIM, labelsize=8)
        self.ax.xaxis.label.set_color(TEXT_COLOR_DIM)
        self.ax.yaxis.label.set_color(TEXT_COLOR_DIM)
        self.ax.title.set_color("#FFFFFF")

        self.line = self.ax.plot(times, values, color=color, linewidth=2)[0]

        self.ax.set_xlabel("Time [s]", fontsize=9)
        self.ax.set_ylabel(ylabel, fontsize=9)
        if title:
            self.ax.set_title(title, fontsize=10, fontweight="bold")
        self.ax.grid(True, color=GRID_COLOR, alpha=0.6)

        if self.window_duration and self.lap_duration > self.window_duration:
            self.ax.set_xlim(0, self.window_duration)
        else:
            self.ax.set_xlim(0, self.lap_duration)

        self.fig.tight_layout(pad=0.5)
        self.draw_idle()

    def plot_multi_line(self, times, values_list, labels, colors, ylabel, title=""):
        """Plot multiple time-series lines (e.g., for tire data)."""
        self.times = times
        self.lap_duration = float(times[-1]) if len(times) > 0 else 0.0

        self.ax.clear()
        self.ax.set_facecolor(BG_COLOR_LIGHT)
        for spine in self.ax.spines.values():
            spine.set_color(TEXT_COLOR_DIM)
        self.ax.tick_params(colors=TEXT_COLOR_DIM, labelsize=8)
        self.ax.xaxis.label.set_color(TEXT_COLOR_DIM)
        self.ax.yaxis.label.set_color(TEXT_COLOR_DIM)
        self.ax.title.set_color("#FFFFFF")

        for values, label, color in zip(values_list, labels, colors):
            self.ax.plot(times, values, color=color, linewidth=2, label=label)

        self.ax.set_xlabel("Time [s]", fontsize=9)
        self.ax.set_ylabel(ylabel, fontsize=9)
        if title:
            self.ax.set_title(title, fontsize=10, fontweight="bold")
        self.ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
        self.ax.grid(True, color=GRID_COLOR, alpha=0.6)

        if self.window_duration and self.lap_duration > self.window_duration:
            self.ax.set_xlim(0, self.window_duration)
        else:
            self.ax.set_xlim(0, self.lap_duration)

        self.fig.tight_layout(pad=0.5)
        self.draw_idle()

    def update_sliding_window(self, current_time: float):
        """Update x-axis limits to show a window centered on current_time."""
        if self.times is None or self.lap_duration <= 0:
            return

        # No sliding window — always show full lap
        if not self.window_duration:
            self.ax.set_xlim(0, self.lap_duration)
            return

        half_window = self.window_duration / 2.0

        x_min = current_time - half_window
        x_max = current_time + half_window

        if x_min < 0:
            x_min = 0
            x_max = min(self.window_duration, self.lap_duration)
        elif x_max > self.lap_duration:
            x_max = self.lap_duration
            x_min = max(0, self.lap_duration - self.window_duration)

        self.ax.set_xlim(x_min, x_max)

    def update_timeline_marker(self, current_time: float):
        """Update vertical line marker and sliding window to current time."""
        if self.times is None:
            return

        if self.timeline_marker is not None:
            try:
                self.timeline_marker.remove()
            except:
                pass

        self.timeline_marker = self.ax.axvline(
            current_time,
            color=ACCENT_RED,
            linewidth=2,
            linestyle='--',
            alpha=0.7,
            zorder=10
        )

        self.update_sliding_window(current_time)
        self.draw_idle()

    def clear_plot(self):
        """Clear the graph."""
        self.ax.clear()
        self.ax.set_facecolor(BG_COLOR_LIGHT)
        for spine in self.ax.spines.values():
            spine.set_color(TEXT_COLOR_DIM)
        self.ax.tick_params(colors=TEXT_COLOR_DIM, labelsize=8)
        self.ax.xaxis.label.set_color(TEXT_COLOR_DIM)
        self.ax.yaxis.label.set_color(TEXT_COLOR_DIM)

        self.ax.set_xlabel("Time [s]", fontsize=9)
        self.ax.grid(True, color=GRID_COLOR, alpha=0.6)

        self.times = None
        self.values = None
        self.line = None
        self.timeline_marker = None

        self.fig.tight_layout(pad=0.5)
        self.draw_idle()
