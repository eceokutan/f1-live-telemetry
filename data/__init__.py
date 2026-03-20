"""
Data persistence module for F1 Telemetry Dashboard.

Provides session recording, playback, and post-race data models.

Heavy imports (Lap, Session, TelemetryLoader) are lazy to avoid pulling
pandas/numpy transitively when only SessionRecorder is needed.
"""

from .models import (
    TireCorners,
    SessionMetadata,
    LapSummary,
    AIComment,
    BrakePoint,
    Corner,
)

__all__ = [
    "SessionRecorder",
    "TireCorners",
    "SessionMetadata",
    "LapSummary",
    "AIComment",
    "BrakePoint",
    "Corner",
    "Lap",
    "Session",
    "TelemetryLoader",
]

# Lazy imports for heavy modules that pull in pandas/numpy
_LAZY_IMPORTS = {
    "SessionRecorder": ".session_recorder",
    "Lap": ".lap",
    "Session": ".session",
    "TelemetryLoader": ".telemetry_loader",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib
        mod = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
