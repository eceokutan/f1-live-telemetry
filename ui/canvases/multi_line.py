"""
Multi-line time series canvas for tire data visualization.
"""
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MultiLineCanvas(FigureCanvas):
    """
    Matplotlib canvas for multi-line time-series visualization.

    Displays multiple colored lines on the same plot (e.g., FL, FR, RL, RR tires).
    """

    def __init__(self, title: str, labels: list, parent=None, width=4, height=1.5, dpi=100, window_seconds: float = 30.0):
        """
        Initialize multi-line canvas.

        Args:
            title: Chart title
            labels: List of line labels (e.g., ["FL", "FR", "RL", "RR"])
            parent: Parent QWidget
            width: Figure width in inches
            height: Figure height in inches
            dpi: Dots per inch resolution
            window_seconds: Sliding window duration in seconds (0 = show all)
        """
        self.window_seconds = window_seconds
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        # Dark theme colors
        bg = "#111111"
        ax_bg = "#181818"

        self.fig.patch.set_facecolor(bg)
        self.ax.set_facecolor(ax_bg)

        # Style axes
        for spine in self.ax.spines.values():
            spine.set_color("#CCCCCC")
        self.ax.tick_params(colors="#CCCCCC", labelsize=7)
        self.ax.xaxis.label.set_color("#CCCCCC")
        self.ax.yaxis.label.set_color("#CCCCCC")
        self.ax.title.set_color("#FFFFFF")

        # Configure plot
        self.title = title
        self.ax.set_title(title, fontsize=8, pad=3)
        self.ax.grid(True, color="#333333", alpha=0.6)
        self.ax.set_xlabel("Time [s]", fontsize=7)

        # Create lines for each data series (FL, FR, RL, RR)
        colors = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#6BCB77"]
        self.lines = []
        for label, color in zip(labels, colors):
            line, = self.ax.plot([], [], linewidth=1.5, color=color, label=label)
            self.lines.append(line)

        self.ax.legend(loc="upper right", fontsize=6, framealpha=0.8)
        self._apply_uniform_layout()

    def _apply_uniform_layout(self):
        """Use fixed margins so all live telemetry graphs render consistently."""
        self.fig.subplots_adjust(left=0.10, right=0.995, top=0.82, bottom=0.24)

    def update_data(self, t: np.ndarray, y_data: list):
        """
        Update multiple lines.

        Args:
            t: Time array (X-axis, shared across all lines)
            y_data: List of numpy arrays, one for each line
        """
        if t.size == 0:
            return

        try:
            if self.window_seconds > 0 and t[-1] > self.window_seconds:
                t_min = t[-1] - self.window_seconds
                mask = t >= t_min
                t = t[mask]
                y_data = [y[mask] if y.size == mask.size else y for y in y_data]

            for line, y in zip(self.lines, y_data):
                if y.size > 0:
                    if t.size != y.size:
                        print(f"⚠️  Warning: Array size mismatch in {self.title}: t={t.size}, y={y.size}")
                        continue
                    line.set_data(t, y)

            self.ax.relim()
            self.ax.autoscale_view()
            if self.window_seconds > 0 and t.size > 0:
                self.ax.set_xlim(t[0], t[-1])
            self.draw_idle()
        except Exception:
            # Matplotlib can throw errors during rendering in multithreaded environments
            # Just skip this update and wait for the next one
            pass
