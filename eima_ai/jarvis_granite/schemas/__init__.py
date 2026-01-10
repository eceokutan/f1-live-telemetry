"""
Schema module for Jarvis-Granite Live Telemetry.

Provides Pydantic models and dataclasses for:
- Telemetry data (TelemetryData, TireTemps, TireWear, GForces)
- WebSocket messages (SessionInitMessage, TelemetryMessage, etc.)
- Events (Event, Priority)
"""

# Telemetry schemas
from jarvis_granite.schemas.telemetry import (
    TireTemps,
    TireWear,
    GForces,
    TelemetryData,
)

# Message schemas
from jarvis_granite.schemas.messages import (
    # Types
    SourceType,
    PriorityLevel,
    ErrorCode,
    # Client messages
    SessionInitMessage,
    TelemetryMessage,
    TextQueryMessage,
    ConfigUpdateMessage,
    SessionEndMessage,
    ClientMessage,
    # Server messages
    LiveKitDetails,
    SessionConfirmedMessage,
    ResponseMetadata,
    AIResponseMessage,
    ErrorMessage,
    HeartbeatMessage,
    ServerMessage,
    # Parser
    parse_client_message,
)

# Event schemas
from jarvis_granite.schemas.events import (
    Priority,
    Event,
    # Factory functions
    create_fuel_critical_event,
    create_fuel_warning_event,
    create_tire_critical_event,
    create_tire_warning_event,
    create_gap_change_event,
    create_lap_complete_event,
    create_sector_complete_event,
    create_pit_window_event,
)

__all__ = [
    # Telemetry
    "TireTemps",
    "TireWear",
    "GForces",
    "TelemetryData",
    # Message types
    "SourceType",
    "PriorityLevel",
    "ErrorCode",
    # Client messages
    "SessionInitMessage",
    "TelemetryMessage",
    "TextQueryMessage",
    "ConfigUpdateMessage",
    "SessionEndMessage",
    "ClientMessage",
    # Server messages
    "LiveKitDetails",
    "SessionConfirmedMessage",
    "ResponseMetadata",
    "AIResponseMessage",
    "ErrorMessage",
    "HeartbeatMessage",
    "ServerMessage",
    # Parser
    "parse_client_message",
    # Events
    "Priority",
    "Event",
    "create_fuel_critical_event",
    "create_fuel_warning_event",
    "create_tire_critical_event",
    "create_tire_warning_event",
    "create_gap_change_event",
    "create_lap_complete_event",
    "create_sector_complete_event",
    "create_pit_window_event",
]
