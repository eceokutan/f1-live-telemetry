"""
Track map canvas with position marker for scrubbing through laps.
"""
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm

from ui.styles import BG_COLOR, BG_COLOR_LIGHT, TEXT_COLOR_DIM, GRID_COLOR, ACCENT_RED


class TrackMapCanvas(FigureCanvas):
    """
    Track map visualization with position marker for timeline scrubbing.

    Shows the full lap racing line colored by speed, with a marker
    indicating the current position based on timeline scrubber.
    """

    def __init__(self, parent=None, width=6, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        # Apply dark theme
        self.fig.patch.set_facecolor(BG_COLOR)
        self.ax.set_facecolor(BG_COLOR_LIGHT)

        for spine in self.ax.spines.values():
            spine.set_color(TEXT_COLOR_DIM)
        self.ax.tick_params(colors=TEXT_COLOR_DIM)
        self.ax.xaxis.label.set_color(TEXT_COLOR_DIM)
        self.ax.yaxis.label.set_color(TEXT_COLOR_DIM)
        self.ax.title.set_color("#FFFFFF")

        # Configure plot
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.set_title("Track Map", fontsize=12, fontweight="bold")
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_xlabel("")
        self.ax.set_ylabel("")
        self.ax.grid(True, color=GRID_COLOR, alpha=0.6)

        # Track data
        self.xs = None
        self.zs = None
        self.speeds = None
        self.line_collection = None
        self.colorbar = None
        self.position_marker = None

        self.fig.tight_layout(pad=0.3)

    def load_lap(self, xs: np.ndarray, zs: np.ndarray, speeds: np.ndarray):
        """Load lap data and plot the full racing line."""
        self.xs = xs
        self.zs = zs
        self.speeds = speeds

        if self.line_collection is not None:
            try:
                self.line_collection.remove()
            except:
                pass

        if self.position_marker is not None:
            try:
                self.position_marker.remove()
            except:
                pass
            self.position_marker = None

        if xs.size < 2:
            self.draw_idle()
            return

        points = np.array([xs, zs]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = Normalize(vmin=np.min(speeds), vmax=np.max(speeds))
        cmap = cm.get_cmap("plasma")

        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2.5)
        lc.set_array(speeds[:-1])

        self.line_collection = lc
        self.ax.add_collection(lc)

        self.ax.set_xlim(xs.min() - 10, xs.max() + 10)
        self.ax.set_ylim(zs.min() - 10, zs.max() + 10)

        if self.colorbar is None:
            try:
                self.colorbar = self.fig.colorbar(
                    lc, ax=self.ax, fraction=0.035, pad=0.02, label="Speed [km/h]"
                )
                self.colorbar.ax.yaxis.label.set_color(TEXT_COLOR_DIM)
                self.colorbar.ax.tick_params(colors=TEXT_COLOR_DIM)
            except:
                pass
        else:
            try:
                self.colorbar.update_normal(lc)
            except:
                pass

        self.draw_idle()

    def update_position(self, sample_index: int):
        """Update position marker to show car position at given sample index."""
        if self.xs is None or self.zs is None:
            return

        sample_index = max(0, min(sample_index, len(self.xs) - 1))

        if self.position_marker is not None:
            try:
                self.position_marker.remove()
            except:
                pass

        x = self.xs[sample_index]
        z = self.zs[sample_index]

        self.position_marker = self.ax.plot(
            x, z,
            marker='o',
            markersize=12,
            color=ACCENT_RED,
            markeredgecolor='white',
            markeredgewidth=2,
            zorder=10
        )[0]

        self.draw_idle()

    def clear_plot(self):
        """Clear the track map."""
        self.ax.clear()
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.set_title("Track Map", fontsize=12, fontweight="bold")
        self.ax.set_xlabel("X [m]", fontsize=9)
        self.ax.set_ylabel("Z [m]", fontsize=9)
        self.ax.grid(True, color=GRID_COLOR, alpha=0.6)

        self.xs = None
        self.zs = None
        self.speeds = None
        self.line_collection = None
        self.position_marker = None
        self.colorbar = None

        self.draw_idle()
