"""
Matplotlib canvas widgets for post-race telemetry visualization.
"""
from .track_map_canvas import TrackMapCanvas
from .time_series_canvas import TimeSeriesCanvas

__all__ = [
    'TrackMapCanvas',
    'TimeSeriesCanvas',
]
