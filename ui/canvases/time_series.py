"""
Time series canvas for telemetry visualization.
"""
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class TimeSeriesCanvas(FigureCanvas):
    """
    Generic matplotlib canvas for time-series data visualization.

    Displays a single line plot with time on the X-axis and telemetry
    data on the Y-axis (speed, RPM, gear, brake, etc.).
    """

    def __init__(self, title: str, parent=None, width=4, height=1.5, dpi=100):
        """
        Initialize time series canvas.

        Args:
            title: Chart title
            parent: Parent QWidget
            width: Figure width in inches
            height: Figure height in inches
            dpi: Dots per inch resolution
        """
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
        self.ax.set_title(title, fontsize=8)
        self.ax.grid(True, color="#333333", alpha=0.6)
        self.ax.set_xlabel("Time [s]", fontsize=7)

        # Create line
        self.line, = self.ax.plot([], [], linewidth=1.5, color="#6FA8FF")

        self.fig.tight_layout(pad=0.5)

    def update_data(self, t: np.ndarray, y: np.ndarray):
        """
        Update the line data and rescale axes.

        Args:
            t: Time array (X-axis)
            y: Data array (Y-axis)
        """
        if t.size == 0 or y.size == 0:
            return
        self.line.set_data(t, y)
        self.ax.relim()
        self.ax.autoscale_view()
        self.draw()
