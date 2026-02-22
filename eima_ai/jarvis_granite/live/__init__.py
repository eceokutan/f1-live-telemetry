"""
Live mode module for Jarvis-Granite Live Telemetry.

Provides real-time race engineering during active sessions.
"""

from jarvis_granite.live.context import LiveSessionContext
from jarvis_granite.live.websocket_handler import WebSocketHandler
from jarvis_granite.live.main import create_app
from jarvis_granite.live.priority_queue import PriorityQueue
from jarvis_granite.live.orchestrator import JarvisLiveOrchestrator
from jarvis_granite.live.interrupt_handler import InterruptHandler, InterruptType
from jarvis_granite.live.errors import (
    ErrorCode,
    ErrorSeverity,
    ErrorContext,
    JarvisError,
    LLMError,
    LLMTimeoutError,
    LLMUnavailableError,
    VoiceError,
    TTSError,
    STTError,
    LiveKitError,
    LiveKitConnectionError,
    LiveKitDisconnectedError,
    SessionError,
    SessionNotFoundError,
    SessionExpiredError,
    ValidationError,
    InvalidTelemetryError,
    InvalidMessageError,
    ConfigurationError,
    MissingCredentialsError,
    OrchestrationError,
    QueueFullError,
    PipelineError,
    TelemetryProcessingError,
    create_error_response,
    create_error_response_from_exception,
    is_transient_error,
)
from jarvis_granite.live.performance import (
    PerformanceMetrics,
    profile_latency,
    profile_latency_async,
    profile,
    profile_async,
)

__all__ = [
    "LiveSessionContext",
    "WebSocketHandler",
    "create_app",
    "PriorityQueue",
    "JarvisLiveOrchestrator",
    "InterruptHandler",
    "InterruptType",
    # Error handling
    "ErrorCode",
    "ErrorSeverity",
    "ErrorContext",
    "JarvisError",
    "LLMError",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "VoiceError",
    "TTSError",
    "STTError",
    "LiveKitError",
    "LiveKitConnectionError",
    "LiveKitDisconnectedError",
    "SessionError",
    "SessionNotFoundError",
    "SessionExpiredError",
    "ValidationError",
    "InvalidTelemetryError",
    "InvalidMessageError",
    "ConfigurationError",
    "MissingCredentialsError",
    "OrchestrationError",
    "QueueFullError",
    "PipelineError",
    "TelemetryProcessingError",
    "create_error_response",
    "create_error_response_from_exception",
    "is_transient_error",
    # Performance monitoring (Phase 6, Section 16)
    "PerformanceMetrics",
    "profile_latency",
    "profile_latency_async",
    "profile",
    "profile_async",
]
