"""
Matplotlib canvas widgets for telemetry visualization.
"""
from ui.canvases.track_map import TrackMapCanvas
from ui.canvases.time_series import TimeSeriesCanvas
from ui.canvases.multi_line import MultiLineCanvas
from ui.canvases.camber_gain import CamberGainCanvas

__all__ = ['TrackMapCanvas', 'TimeSeriesCanvas', 'MultiLineCanvas', 'CamberGainCanvas']
