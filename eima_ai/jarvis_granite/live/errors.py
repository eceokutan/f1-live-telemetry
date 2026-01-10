"""
Error Handling Module for Jarvis-Granite Live Telemetry.

Phase 6, Section 15: Error Handling
- Custom exception hierarchy with error codes
- Error classification (transient vs permanent)
- Error context and correlation IDs
- Support for graceful degradation

Error Codes defined in documentation:
- LLM_TIMEOUT: Granite API request timed out
- LLM_ERROR: Granite API returned error
- TTS_ERROR: Text-to-speech failed
- STT_ERROR: Speech-to-text failed
- LIVEKIT_ERROR: LiveKit connection failed
- VALIDATION_ERROR: Invalid request data
- SESSION_NOT_FOUND: Referenced session doesn't exist
- CONFIG_ERROR: Invalid configuration

Example:
    from jarvis_granite.live.errors import (
        JarvisError,
        LLMTimeoutError,
        TTSError,
        SessionNotFoundError,
    )

    try:
        response = await llm_client.invoke(prompt)
    except LLMTimeoutError as e:
        logger.error(f"LLM timeout: {e.error_code}")
        return create_error_response(e)
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ErrorCode(str, Enum):
    """Standard error codes for Jarvis-Granite Live."""

    # LLM Errors
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_ERROR = "LLM_ERROR"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"

    # Voice Errors
    TTS_ERROR = "TTS_ERROR"
    STT_ERROR = "STT_ERROR"
    VOICE_PIPELINE_ERROR = "VOICE_PIPELINE_ERROR"

    # LiveKit Errors
    LIVEKIT_ERROR = "LIVEKIT_ERROR"
    LIVEKIT_CONNECTION_ERROR = "LIVEKIT_CONNECTION_ERROR"
    LIVEKIT_DISCONNECTED = "LIVEKIT_DISCONNECTED"

    # Session Errors
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    SESSION_INVALID = "SESSION_INVALID"

    # Validation Errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_TELEMETRY = "INVALID_TELEMETRY"
    INVALID_MESSAGE = "INVALID_MESSAGE"

    # Configuration Errors
    CONFIG_ERROR = "CONFIG_ERROR"
    MISSING_CREDENTIALS = "MISSING_CREDENTIALS"

    # Orchestration Errors
    ORCHESTRATION_ERROR = "ORCHESTRATION_ERROR"
    QUEUE_FULL = "QUEUE_FULL"
    INTERRUPT_ERROR = "INTERRUPT_ERROR"

    # Pipeline Errors
    PIPELINE_ERROR = "PIPELINE_ERROR"
    TELEMETRY_PROCESSING_ERROR = "TELEMETRY_PROCESSING_ERROR"

    # Generic Errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""

    CRITICAL = "critical"  # System unusable, immediate attention required
    ERROR = "error"  # Operation failed, but system continues
    WARNING = "warning"  # Degraded operation, attention needed
    INFO = "info"  # Informational, no action needed


@dataclass
class ErrorContext:
    """
    Context information for error tracking and debugging.

    Attributes:
        correlation_id: Unique ID for tracing across components
        session_id: Associated session ID if applicable
        timestamp: When the error occurred
        component: Component where error originated
        operation: Operation that was being performed
        metadata: Additional contextual information
    """

    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    component: Optional[str] = None
    operation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for logging/serialization."""
        return {
            "correlation_id": self.correlation_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "component": self.component,
            "operation": self.operation,
            "metadata": self.metadata,
        }


class JarvisError(Exception):
    """
    Base exception for all Jarvis-Granite errors.

    Provides:
    - Standard error codes for classification
    - Error context for debugging
    - Transient vs permanent error classification
    - Support for error chaining

    Attributes:
        message: Human-readable error message
        error_code: Standardized error code
        severity: Error severity level
        is_transient: Whether the error is transient (retryable)
        context: Error context for debugging
        cause: Original exception that caused this error
    """

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        is_transient: bool = False,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.is_transient = is_transient
        self.context = context or ErrorContext()
        self.cause = cause

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API responses."""
        return {
            "type": "error",
            "error_code": self.error_code.value,
            "message": self.message,
            "severity": self.severity.value,
            "is_transient": self.is_transient,
            "context": self.context.to_dict(),
            "timestamp": self.context.timestamp,
        }

    def to_response(self) -> Dict[str, Any]:
        """Create a WebSocket-compatible error response."""
        from datetime import datetime, timezone

        return {
            "type": "error",
            "error_code": self.error_code.value,
            "message": self.message,
            "timestamp": datetime.fromtimestamp(
                self.context.timestamp, tz=timezone.utc
            ).isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"code={self.error_code.value}, "
            f"message='{self.message[:50]}...', "
            f"transient={self.is_transient})"
        )


# =============================================================================
# LLM ERRORS
# =============================================================================


class LLMError(JarvisError):
    """Base exception for LLM-related errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.LLM_ERROR,
        is_transient: bool = False,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            severity=ErrorSeverity.ERROR,
            is_transient=is_transient,
            context=context,
            cause=cause,
        )


class LLMTimeoutError(LLMError):
    """LLM request timed out - transient, can retry."""

    def __init__(
        self,
        message: str = "LLM request timed out",
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.LLM_TIMEOUT,
            is_transient=True,  # Timeouts are transient
            context=context,
            cause=cause,
        )


class LLMUnavailableError(LLMError):
    """LLM service is unavailable."""

    def __init__(
        self,
        message: str = "LLM service unavailable",
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.LLM_UNAVAILABLE,
            is_transient=True,
            context=context,
            cause=cause,
        )


# =============================================================================
# VOICE ERRORS
# =============================================================================


class VoiceError(JarvisError):
    """Base exception for voice-related errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.VOICE_PIPELINE_ERROR,
        is_transient: bool = False,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            severity=ErrorSeverity.ERROR,
            is_transient=is_transient,
            context=context,
            cause=cause,
        )


class TTSError(VoiceError):
    """Text-to-speech synthesis failed."""

    def __init__(
        self,
        message: str = "TTS synthesis failed",
        is_transient: bool = True,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.TTS_ERROR,
            is_transient=is_transient,
            context=context,
            cause=cause,
        )


class STTError(VoiceError):
    """Speech-to-text transcription failed."""

    def __init__(
        self,
        message: str = "STT transcription failed",
        is_transient: bool = True,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.STT_ERROR,
            is_transient=is_transient,
            context=context,
            cause=cause,
        )


# =============================================================================
# LIVEKIT ERRORS
# =============================================================================


class LiveKitError(JarvisError):
    """Base exception for LiveKit-related errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.LIVEKIT_ERROR,
        is_transient: bool = True,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            severity=ErrorSeverity.ERROR,
            is_transient=is_transient,
            context=context,
            cause=cause,
        )


class LiveKitConnectionError(LiveKitError):
    """LiveKit connection failed."""

    def __init__(
        self,
        message: str = "LiveKit connection failed",
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.LIVEKIT_CONNECTION_ERROR,
            is_transient=True,
            context=context,
            cause=cause,
        )


class LiveKitDisconnectedError(LiveKitError):
    """LiveKit connection was disconnected."""

    def __init__(
        self,
        message: str = "LiveKit disconnected",
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.LIVEKIT_DISCONNECTED,
            is_transient=True,
            context=context,
            cause=cause,
        )


# =============================================================================
# SESSION ERRORS
# =============================================================================


class SessionError(JarvisError):
    """Base exception for session-related errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.SESSION_INVALID,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            severity=ErrorSeverity.ERROR,
            is_transient=False,  # Session errors are usually not transient
            context=context,
            cause=cause,
        )


class SessionNotFoundError(SessionError):
    """Session does not exist."""

    def __init__(
        self,
        session_id: str,
        message: Optional[str] = None,
        context: Optional[ErrorContext] = None,
    ):
        msg = message or f"Session not found: {session_id}"
        ctx = context or ErrorContext(session_id=session_id)
        ctx.session_id = session_id
        super().__init__(
            message=msg,
            error_code=ErrorCode.SESSION_NOT_FOUND,
            context=ctx,
        )
        self.session_id = session_id


class SessionExpiredError(SessionError):
    """Session has expired."""

    def __init__(
        self,
        session_id: str,
        message: Optional[str] = None,
        context: Optional[ErrorContext] = None,
    ):
        msg = message or f"Session expired: {session_id}"
        ctx = context or ErrorContext(session_id=session_id)
        ctx.session_id = session_id
        super().__init__(
            message=msg,
            error_code=ErrorCode.SESSION_EXPIRED,
            context=ctx,
        )
        self.session_id = session_id


# =============================================================================
# VALIDATION ERRORS
# =============================================================================


class ValidationError(JarvisError):
    """Base exception for validation errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.VALIDATION_ERROR,
        field: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            severity=ErrorSeverity.WARNING,
            is_transient=False,
            context=context,
            cause=cause,
        )
        self.field = field


class InvalidTelemetryError(ValidationError):
    """Telemetry data is invalid."""

    def __init__(
        self,
        message: str = "Invalid telemetry data",
        field: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.INVALID_TELEMETRY,
            field=field,
            context=context,
            cause=cause,
        )


class InvalidMessageError(ValidationError):
    """WebSocket message is invalid."""

    def __init__(
        self,
        message: str = "Invalid message",
        field: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.INVALID_MESSAGE,
            field=field,
            context=context,
            cause=cause,
        )


# =============================================================================
# CONFIGURATION ERRORS
# =============================================================================


class ConfigurationError(JarvisError):
    """Configuration is invalid or missing."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.CONFIG_ERROR,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            severity=ErrorSeverity.CRITICAL,
            is_transient=False,
            context=context,
            cause=cause,
        )


class MissingCredentialsError(ConfigurationError):
    """Required credentials are missing."""

    def __init__(
        self,
        service: str,
        message: Optional[str] = None,
        context: Optional[ErrorContext] = None,
    ):
        msg = message or f"Missing credentials for {service}"
        super().__init__(
            message=msg,
            error_code=ErrorCode.MISSING_CREDENTIALS,
            context=context,
        )
        self.service = service


# =============================================================================
# ORCHESTRATION ERRORS
# =============================================================================


class OrchestrationError(JarvisError):
    """Orchestration operation failed."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.ORCHESTRATION_ERROR,
        is_transient: bool = False,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            severity=ErrorSeverity.ERROR,
            is_transient=is_transient,
            context=context,
            cause=cause,
        )


class QueueFullError(OrchestrationError):
    """Event queue is at capacity."""

    def __init__(
        self,
        message: str = "Event queue is full",
        context: Optional[ErrorContext] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.QUEUE_FULL,
            is_transient=True,  # Queue may clear
            context=context,
        )


# =============================================================================
# PIPELINE ERRORS
# =============================================================================


class PipelineError(JarvisError):
    """Pipeline processing failed."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.PIPELINE_ERROR,
        is_transient: bool = False,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            severity=ErrorSeverity.ERROR,
            is_transient=is_transient,
            context=context,
            cause=cause,
        )


class TelemetryProcessingError(PipelineError):
    """Telemetry processing failed."""

    def __init__(
        self,
        message: str = "Telemetry processing failed",
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.TELEMETRY_PROCESSING_ERROR,
            is_transient=True,
            context=context,
            cause=cause,
        )


# =============================================================================
# ERROR RESPONSE HELPERS
# =============================================================================


def create_error_response(
    error: JarvisError,
) -> Dict[str, Any]:
    """
    Create a standardized error response for WebSocket messages.

    Args:
        error: JarvisError instance

    Returns:
        Dictionary suitable for WebSocket JSON response
    """
    return error.to_response()


def create_error_response_from_exception(
    exception: Exception,
    error_code: ErrorCode = ErrorCode.INTERNAL_ERROR,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create error response from a generic exception.

    Args:
        exception: Any exception
        error_code: Error code to use
        session_id: Optional session ID for context

    Returns:
        Dictionary suitable for WebSocket JSON response
    """
    if isinstance(exception, JarvisError):
        return exception.to_response()

    context = ErrorContext(session_id=session_id)
    error = JarvisError(
        message=str(exception),
        error_code=error_code,
        context=context,
        cause=exception,
    )
    return error.to_response()


def is_transient_error(error: Exception) -> bool:
    """
    Check if an error is transient (should be retried).

    Args:
        error: Exception to check

    Returns:
        True if error is transient
    """
    if isinstance(error, JarvisError):
        return error.is_transient

    # Standard Python exceptions that are typically transient
    transient_types = (
        ConnectionError,
        TimeoutError,
        ConnectionResetError,
        ConnectionRefusedError,
    )
    return isinstance(error, transient_types)
