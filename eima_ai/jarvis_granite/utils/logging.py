"""
Logging infrastructure for Jarvis-Granite Live Telemetry.

This module provides structured logging with support for:
- Console and file output
- JSON formatting for production
- Request/session context tracking
- Performance metrics logging
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from contextvars import ContextVar
from functools import wraps
import time

# Context variables for request tracking
session_id_var: ContextVar[Optional[str]] = ContextVar('session_id', default=None)
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)


class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs log records as JSON objects for easy parsing
    and integration with log aggregation systems.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add context variables
        session_id = session_id_var.get()
        if session_id:
            log_data["session_id"] = session_id

        request_id = request_id_var.get()
        if request_id:
            log_data["request_id"] = request_id

        # Add extra fields from record
        if hasattr(record, 'extra_data'):
            log_data["extra"] = record.extra_data

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class ConsoleFormatter(logging.Formatter):
    """
    Human-readable formatter for console output.

    Uses colors for different log levels when terminal supports it.
    """

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        level = record.levelname

        if self.use_colors:
            level_color = self.COLORS.get(level, '')
            level_str = f"{level_color}{level:8s}{self.RESET}"
        else:
            level_str = f"{level:8s}"

        # Build context string
        context_parts = []
        session_id = session_id_var.get()
        if session_id:
            context_parts.append(f"session={session_id[:8]}")

        context_str = f" [{', '.join(context_parts)}]" if context_parts else ""

        message = record.getMessage()

        return f"{timestamp} | {level_str} | {record.name}{context_str} | {message}"


class PerformanceLogger:
    """
    Logger for performance metrics.

    Tracks and logs timing information for critical operations
    to ensure latency targets are met.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_latency(
        self,
        operation: str,
        latency_ms: float,
        target_ms: Optional[float] = None,
        **extra: Any
    ) -> None:
        """
        Log latency for an operation.

        Args:
            operation: Name of the operation
            latency_ms: Measured latency in milliseconds
            target_ms: Target latency threshold (optional)
            **extra: Additional context to include
        """
        data = {
            "operation": operation,
            "latency_ms": round(latency_ms, 2),
            **extra
        }

        if target_ms is not None:
            data["target_ms"] = target_ms
            data["within_target"] = latency_ms <= target_ms

        if target_ms and latency_ms > target_ms:
            self.logger.warning(
                f"Latency exceeded target: {operation} took {latency_ms:.2f}ms (target: {target_ms}ms)",
                extra={"extra_data": data}
            )
        else:
            self.logger.debug(
                f"Operation timing: {operation} took {latency_ms:.2f}ms",
                extra={"extra_data": data}
            )


def setup_logging(
    log_level: str = "INFO",
    log_directory: Optional[str] = None,
    json_format: bool = False,
    app_name: str = "jarvis-granite-live"
) -> logging.Logger:
    """
    Set up logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_directory: Directory for log files (None for console only)
        json_format: Use JSON formatting for logs
        app_name: Application name for the root logger

    Returns:
        Configured root logger instance
    """
    # Get or create the root logger for the application
    logger = logging.getLogger(app_name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))

    if json_format:
        console_handler.setFormatter(JsonFormatter())
    else:
        console_handler.setFormatter(ConsoleFormatter())

    logger.addHandler(console_handler)

    # File handler (if log directory specified)
    if log_directory:
        log_path = Path(log_directory)
        log_path.mkdir(parents=True, exist_ok=True)

        # Main log file
        file_handler = logging.FileHandler(
            log_path / f"{app_name}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)

        # Error log file
        error_handler = logging.FileHandler(
            log_path / f"{app_name}.error.log",
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JsonFormatter())
        logger.addHandler(error_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"jarvis-granite-live.{name}")


def log_execution_time(
    operation_name: Optional[str] = None,
    target_ms: Optional[float] = None,
    logger: Optional[logging.Logger] = None
):
    """
    Decorator to log execution time of a function.

    Args:
        operation_name: Name for the operation (defaults to function name)
        target_ms: Target latency threshold
        logger: Logger instance (defaults to function's module logger)

    Example:
        @log_execution_time(target_ms=50)
        def process_telemetry(data):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            nonlocal logger, operation_name
            if logger is None:
                logger = get_logger(func.__module__)
            if operation_name is None:
                operation_name = func.__name__

            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                perf_logger = PerformanceLogger(logger)
                perf_logger.log_latency(operation_name, elapsed_ms, target_ms)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            nonlocal logger, operation_name
            if logger is None:
                logger = get_logger(func.__module__)
            if operation_name is None:
                operation_name = func.__name__

            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                perf_logger = PerformanceLogger(logger)
                perf_logger.log_latency(operation_name, elapsed_ms, target_ms)

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class SessionLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that automatically includes session context.

    Use this adapter when you have a session_id available and want
    it automatically included in all log messages.
    """

    def __init__(self, logger: logging.Logger, session_id: str):
        super().__init__(logger, {"session_id": session_id})
        self.session_id = session_id

    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        # Set context variable for other formatters
        session_id_var.set(self.session_id)
        return msg, kwargs
