"""
Camber gain canvas: plots camber angle vs suspension travel for all four tires.
"""
import math
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class CamberGainCanvas(FigureCanvas):
    """
    XY scatter canvas showing camber angle (deg) vs suspension travel (mm)
    for FL, FR, RL, RR tires.
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

        self.ax.set_title("Camber Gain", fontsize=8)
        self.ax.grid(True, color="#333333", alpha=0.6)
        self.ax.set_xlabel("Suspension Travel [mm]", fontsize=7)
        self.ax.set_ylabel("Camber [°]", fontsize=7)

        colors = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#6BCB77"]
        labels = ["FL", "FR", "RL", "RR"]
        self.lines = []
        for label, color in zip(labels, colors):
            line, = self.ax.plot([], [], linewidth=1.5, color=color, label=label, alpha=0.7)
            self.lines.append(line)

        self.ax.legend(loc="upper right", fontsize=6, framealpha=0.8)
        self.fig.tight_layout(pad=0.5)

    def update_data(self, suspension: list, camber: list):
        """
        Update the plot.

        Args:
            suspension: List of 4 numpy arrays (FL, FR, RL, RR) — travel in meters
            camber: List of 4 numpy arrays (FL, FR, RL, RR) — angle in radians
        """
        try:
            for line, susp, cam in zip(self.lines, suspension, camber):
                if susp.size == 0 or cam.size == 0 or susp.size != cam.size:
                    continue
                # Convert: meters → mm, radians → degrees
                line.set_data(susp * 1000, np.degrees(cam))

            self.ax.relim()
            self.ax.autoscale_view()
            self.draw_idle()
        except Exception:
            pass
