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

    def __init__(self, title: str, labels: list, parent=None, width=4, height=1.5, dpi=100):
        """
        Initialize multi-line canvas.

        Args:
            title: Chart title
            labels: List of line labels (e.g., ["FL", "FR", "RL", "RR"])
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

        # Create lines for each data series (FL, FR, RL, RR)
        colors = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#6BCB77"]
        self.lines = []
        for label, color in zip(labels, colors):
            line, = self.ax.plot([], [], linewidth=1.5, color=color, label=label)
            self.lines.append(line)

        self.ax.legend(loc="upper right", fontsize=6, framealpha=0.8)
        self.fig.tight_layout(pad=0.5)

    def update_data(self, t: np.ndarray, y_data: list):
        """
        Update multiple lines.

        Args:
            t: Time array (X-axis, shared across all lines)
            y_data: List of numpy arrays, one for each line
        """
        if t.size == 0:
            return

        for line, y in zip(self.lines, y_data):
            if y.size > 0:
                line.set_data(t, y)

        self.ax.relim()
        self.ax.autoscale_view()
        self.draw()
