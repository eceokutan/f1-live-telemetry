"""
WebSocket message schemas for Jarvis-Granite Live Telemetry.

Defines Pydantic models for all client-server WebSocket messages:
- Client → Server: session_init, telemetry, text_query, config_update, session_end
- Server → Client: session_confirmed, ai_response, error, heartbeat
"""

from typing import Any, Dict, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator

from jarvis_granite.schemas.telemetry import TelemetryData


# =============================================================================
# ENUMS AND LITERALS
# =============================================================================

# Valid data sources
SourceType = Literal["torcs", "assetto_corsa", "can_bus"]

# Valid priority levels for responses
PriorityLevel = Literal["critical", "high", "medium", "low"]

# Valid error codes
ErrorCode = Literal[
    "LLM_TIMEOUT",
    "LLM_ERROR",
    "TTS_ERROR",
    "STT_ERROR",
    "LIVEKIT_ERROR",
    "VALIDATION_ERROR",
    "SESSION_NOT_FOUND",
    "CONFIG_ERROR"
]


# =============================================================================
# CLIENT → SERVER MESSAGES
# =============================================================================

class SessionInitMessage(BaseModel):
    """
    Session initialization message sent by client to start a session.

    Example:
        {
            "type": "session_init",
            "session_id": "race_001",
            "source": "torcs",
            "track_name": "Monza",
            "config": {"verbosity": "moderate"}
        }
    """
    type: Literal["session_init"] = "session_init"
    session_id: str = Field(..., min_length=1, description="Unique session identifier")
    source: SourceType = Field(..., description="Telemetry data source")
    track_name: str = Field(..., min_length=1, description="Name of the track")
    config: Dict[str, Any] = Field(default_factory=dict, description="Session configuration")


class TelemetryMessage(BaseModel):
    """
    Telemetry data message sent by client during active session.

    Example:
        {
            "type": "telemetry",
            "timestamp": "2026-01-15T14:32:05.123Z",
            "data": { ... telemetry fields ... }
        }
    """
    type: Literal["telemetry"] = "telemetry"
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    data: TelemetryData = Field(..., description="Telemetry data snapshot")


class TextQueryMessage(BaseModel):
    """
    Text query message (alternative to voice input).

    Example:
        {
            "type": "text_query",
            "timestamp": "2026-01-15T14:32:10.456Z",
            "query": "How are my tires looking?"
        }
    """
    type: Literal["text_query"] = "text_query"
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    query: str = Field(..., min_length=1, description="Query text from driver")


class ConfigUpdateMessage(BaseModel):
    """
    Runtime configuration update message.

    Example:
        {
            "type": "config_update",
            "config": {"verbosity": "verbose"}
        }
    """
    type: Literal["config_update"] = "config_update"
    config: Dict[str, Any] = Field(..., description="Configuration updates to apply")


class SessionEndMessage(BaseModel):
    """
    Session end message to terminate the current session.

    Example:
        {
            "type": "session_end"
        }
    """
    type: Literal["session_end"] = "session_end"


# =============================================================================
# SERVER → CLIENT MESSAGES
# =============================================================================

class LiveKitDetails(BaseModel):
    """LiveKit connection details included in session confirmation."""
    url: str = Field(..., description="LiveKit server URL")
    token: str = Field(..., description="JWT access token")
    room_name: str = Field(..., description="Room name to join")


class SessionConfirmedMessage(BaseModel):
    """
    Session confirmation response from server.

    Includes LiveKit connection details for voice communication.

    Example:
        {
            "type": "session_confirmed",
            "session_id": "race_001",
            "config": { ... },
            "livekit": {
                "url": "wss://livekit.example.com",
                "token": "eyJ...",
                "room_name": "race_001_voice"
            }
        }
    """
    type: Literal["session_confirmed"] = "session_confirmed"
    session_id: str = Field(..., description="Confirmed session ID")
    config: Dict[str, Any] = Field(..., description="Active configuration")
    livekit: LiveKitDetails = Field(..., description="LiveKit connection details")


class ResponseMetadata(BaseModel):
    """Metadata about AI response generation."""
    latency_ms: int = Field(..., ge=0, description="Response latency in milliseconds")
    tokens_used: Optional[int] = Field(default=None, ge=0, description="LLM tokens consumed")


class AIResponseMessage(BaseModel):
    """
    AI response message containing generated text.

    Note: Audio is delivered via LiveKit WebRTC, not WebSocket.

    Example:
        {
            "type": "ai_response",
            "response_id": "resp_abc123",
            "timestamp": "2026-01-15T14:32:11.789Z",
            "trigger": "pit_window_open",
            "text": "Pit window is open. Box this lap for fresh mediums.",
            "priority": "high",
            "metadata": {"latency_ms": 1850, "tokens_used": 45}
        }
    """
    type: Literal["ai_response"] = "ai_response"
    response_id: str = Field(..., description="Unique response identifier")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    trigger: str = Field(..., description="Event or query that triggered response")
    text: str = Field(..., description="Generated response text")
    priority: PriorityLevel = Field(..., description="Response priority level")
    metadata: Optional[ResponseMetadata] = Field(default=None, description="Response metadata")


class ErrorMessage(BaseModel):
    """
    Error message for communicating failures to client.

    Example:
        {
            "type": "error",
            "error_code": "LLM_TIMEOUT",
            "message": "AI response timed out, please try again",
            "timestamp": "2026-01-15T14:32:15.000Z"
        }
    """
    type: Literal["error"] = "error"
    error_code: ErrorCode = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    timestamp: str = Field(..., description="ISO 8601 timestamp")


class HeartbeatMessage(BaseModel):
    """
    Heartbeat message for connection monitoring.

    Example:
        {
            "type": "heartbeat",
            "timestamp": "2026-01-15T14:32:30.000Z",
            "session_active": true
        }
    """
    type: Literal["heartbeat"] = "heartbeat"
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    session_active: bool = Field(..., description="Whether session is still active")


# =============================================================================
# MESSAGE PARSING
# =============================================================================

# Union type for all client messages
ClientMessage = Union[
    SessionInitMessage,
    TelemetryMessage,
    TextQueryMessage,
    ConfigUpdateMessage,
    SessionEndMessage
]

# Union type for all server messages
ServerMessage = Union[
    SessionConfirmedMessage,
    AIResponseMessage,
    ErrorMessage,
    HeartbeatMessage
]

# Message type to class mapping for client messages
CLIENT_MESSAGE_TYPES = {
    "session_init": SessionInitMessage,
    "telemetry": TelemetryMessage,
    "text_query": TextQueryMessage,
    "config_update": ConfigUpdateMessage,
    "session_end": SessionEndMessage
}


def parse_client_message(data: Dict[str, Any]) -> ClientMessage:
    """
    Parse a client WebSocket message from JSON data.

    Args:
        data: Dictionary containing the message data

    Returns:
        Parsed message object

    Raises:
        ValueError: If message type is unknown
        ValidationError: If message data is invalid
    """
    message_type = data.get("type")

    if message_type not in CLIENT_MESSAGE_TYPES:
        raise ValueError(f"Unknown message type: {message_type}")

    message_class = CLIENT_MESSAGE_TYPES[message_type]
    return message_class(**data)
