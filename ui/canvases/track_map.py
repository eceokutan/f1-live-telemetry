"""
Track map canvas for visualizing lap trajectory colored by speed.
"""
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm


class TrackMapCanvas(FigureCanvas):
    """
    Matplotlib canvas for track map visualization.

    Displays the car's path (X, Z coordinates) colored by speed,
    giving a visual representation of the racing line and speed zones.
    """

    def __init__(self, parent=None, width=4, height=4, dpi=100):
        """
        Initialize track map canvas.

        Args:
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
        bg = "#111111"      # figure background
        ax_bg = "#181818"   # axes background

        self.fig.patch.set_facecolor(bg)
        self.ax.set_facecolor(ax_bg)

        # Make lines / text readable on dark background
        for spine in self.ax.spines.values():
            spine.set_color("#CCCCCC")
        self.ax.tick_params(colors="#CCCCCC")
        self.ax.xaxis.label.set_color("#CCCCCC")
        self.ax.yaxis.label.set_color("#CCCCCC")
        self.ax.title.set_color("#FFFFFF")

        # Configure plot
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

        Args:
            xs: X coordinates array
            zs: Z coordinates array
            speeds: Speed values at each point (for coloring)
        """
        try:
            # Remove old line collection if it exists
            if self.line_collection is not None:
                try:
                    self.line_collection.remove()
                except:
                    pass

            # Clear axis only if needed (for style reset)
            if not hasattr(self, '_initialized'):
                self.ax.clear()
                self.ax.set_aspect("equal", adjustable="datalim")
                self.ax.set_title("Location Map", fontsize=10)
                self.ax.set_xlabel("X [m]", fontsize=8)
                self.ax.set_ylabel("Z [m]", fontsize=8)
                self.ax.grid(True, color="#333333", alpha=0.6)
                self._initialized = True

            if xs.size < 2:
                self.draw_idle()
                return

            # Create line segments
            points = np.array([xs, zs]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Color by speed
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

            # Set axis limits with padding
            self.ax.set_xlim(xs.min() - 10, xs.max() + 10)
            self.ax.set_ylim(zs.min() - 10, zs.max() + 10)

            # Only create colorbar once
            if self.colorbar is None:
                try:
                    self.colorbar = self.fig.colorbar(
                        lc, ax=self.ax, fraction=0.046, pad=0.04, label="Speed [km/h]"
                    )
                except:
                    pass  # Skip colorbar if it fails
            else:
                # Update existing colorbar
                try:
                    self.colorbar.update_normal(lc)
                except:
                    pass  # Skip if update fails

            self.draw_idle()  # Use draw_idle instead of draw for better performance
        except Exception as e:
            # Silently fail to avoid crashing the UI
            pass
