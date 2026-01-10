"""
Performance Monitoring and Optimization - Phase 6, Section 16

Provides performance metrics tracking, latency profiling, and bottleneck
identification for the Jarvis-Granite Live Telemetry system.

Performance Targets (from documentation):
- Telemetry processing: <50ms (rule-based, no LLM)
- LLM response: <2000ms (Granite generation via LangChain)
- TTS conversion: <500ms (Watson TTS with Tenacity retry)
- LiveKit transport: <100ms (WebRTC jitter buffering)
- End-to-end (voice): <3000ms (Total pipeline latency)

Example:
    from jarvis_granite.live.performance import PerformanceMetrics, profile_latency

    metrics = PerformanceMetrics()
    metrics.set_target("telemetry_processing", 50.0)

    with profile_latency(metrics, "telemetry_processing"):
        # Process telemetry
        pass

    report = metrics.get_summary()
"""

import asyncio
import json
import logging
import statistics
import time
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

# Type variable for generic decorators
F = TypeVar('F', bound=Callable[..., Any])


# =============================================================================
# DEFAULT PERFORMANCE TARGETS
# =============================================================================

DEFAULT_TARGETS = {
    "telemetry_processing": 50.0,   # <50ms
    "event_detection": 50.0,        # <50ms (part of telemetry)
    "llm_response": 2000.0,         # <2000ms
    "tts_conversion": 500.0,        # <500ms
    "stt_transcription": 500.0,     # <500ms
    "livekit_transport": 100.0,     # <100ms
    "end_to_end": 3000.0,           # <3000ms total
}


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

@dataclass
class MetricStats:
    """Statistics for a single metric."""
    count: int = 0
    total: float = 0.0
    min: float = float('inf')
    max: float = 0.0
    values: List[float] = field(default_factory=list)
    last_updated: Optional[datetime] = None

    def record(self, value: float) -> None:
        """Record a new measurement."""
        self.count += 1
        self.total += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.values.append(value)
        self.last_updated = datetime.now(timezone.utc)

        # Keep only last 1000 values to prevent memory growth
        if len(self.values) > 1000:
            self.values = self.values[-1000:]

    @property
    def avg(self) -> float:
        """Calculate average."""
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def percentile(self, p: float) -> float:
        """Calculate percentile (0-100)."""
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        idx = int(len(sorted_values) * p / 100)
        idx = min(idx, len(sorted_values) - 1)
        return sorted_values[idx]

    def reset(self) -> None:
        """Reset all statistics."""
        self.count = 0
        self.total = 0.0
        self.min = float('inf')
        self.max = 0.0
        self.values = []
        self.last_updated = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "count": self.count,
            "total": self.total,
            "min": self.min if self.count > 0 else 0.0,
            "max": self.max,
            "avg": self.avg,
            "p50": self.percentile(50),
            "p95": self.percentile(95),
            "p99": self.percentile(99),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


class PerformanceMetrics:
    """
    Performance metrics tracking for latency profiling and optimization.

    Tracks latency measurements for different operations, calculates
    statistics, and identifies bottlenecks.

    Attributes:
        targets: Target thresholds for each operation
        metrics: Recorded metrics for each operation
        log_file: Optional file path for logging metrics

    Example:
        metrics = PerformanceMetrics()
        metrics.set_target("llm_response", 2000.0)
        metrics.record("llm_response", 1500.0)
        stats = metrics.get_stats("llm_response")
    """

    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize performance metrics.

        Args:
            log_file: Optional path to log file for metrics persistence
        """
        self._metrics: Dict[str, MetricStats] = {}
        self._targets: Dict[str, float] = DEFAULT_TARGETS.copy()
        self._log_file = log_file
        self._log_buffer: List[Dict[str, Any]] = []

    def record(self, operation: str, latency_ms: float) -> None:
        """
        Record a latency measurement.

        Args:
            operation: Name of the operation
            latency_ms: Latency in milliseconds
        """
        if operation not in self._metrics:
            self._metrics[operation] = MetricStats()

        self._metrics[operation].record(latency_ms)

        # Add to log buffer
        self._log_buffer.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "latency_ms": latency_ms,
        })

        # Log warning if exceeds target
        if operation in self._targets and latency_ms > self._targets[operation]:
            logger.warning(
                f"Performance warning: {operation} took {latency_ms:.2f}ms "
                f"(target: {self._targets[operation]:.2f}ms)"
            )

    def set_target(self, operation: str, target_ms: float) -> None:
        """
        Set target threshold for an operation.

        Args:
            operation: Name of the operation
            target_ms: Target maximum latency in milliseconds
        """
        self._targets[operation] = target_ms

    def get_stats(self, operation: str) -> Dict[str, Any]:
        """
        Get statistics for an operation.

        Args:
            operation: Name of the operation

        Returns:
            Dictionary with count, min, max, avg, percentiles
        """
        if operation not in self._metrics:
            return {
                "count": 0,
                "total": 0.0,
                "min": 0.0,
                "max": 0.0,
                "avg": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "last_updated": None,
            }

        return self._metrics[operation].to_dict()

    def get_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get summary of all metrics.

        Returns:
            Dictionary mapping operation names to their statistics
        """
        return {
            operation: stats.to_dict()
            for operation, stats in self._metrics.items()
        }

    def check_compliance(self) -> Dict[str, Dict[str, Any]]:
        """
        Check compliance with target thresholds.

        Returns:
            Dictionary with compliance status for each operation
        """
        result = {}

        for operation, target in self._targets.items():
            stats = self.get_stats(operation)

            if stats["count"] == 0:
                result[operation] = {
                    "target_ms": target,
                    "compliant": True,  # No data = compliant
                    "avg_ms": 0.0,
                    "p95_ms": 0.0,
                }
            else:
                # Check if p95 is under target
                compliant = stats["p95"] <= target

                result[operation] = {
                    "target_ms": target,
                    "compliant": compliant,
                    "avg_ms": stats["avg"],
                    "p95_ms": stats["p95"],
                }

        return result

    def identify_bottlenecks(
        self,
        threshold_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Identify performance bottlenecks.

        Args:
            threshold_only: If True, only return operations exceeding thresholds

        Returns:
            List of bottlenecks sorted by latency (highest first)
        """
        bottlenecks = []

        for operation, stats in self._metrics.items():
            if stats.count == 0:
                continue

            exceeds_threshold = (
                operation in self._targets and
                stats.avg > self._targets[operation]
            )

            if threshold_only and not exceeds_threshold:
                continue

            bottlenecks.append({
                "operation": operation,
                "avg_ms": stats.avg,
                "p95_ms": stats.percentile(95),
                "count": stats.count,
                "target_ms": self._targets.get(operation),
                "exceeds_threshold": exceeds_threshold,
            })

        # Sort by average latency (highest first)
        bottlenecks.sort(key=lambda x: x["avg_ms"], reverse=True)

        return bottlenecks

    def reset(self) -> None:
        """Reset all metrics."""
        for stats in self._metrics.values():
            stats.reset()
        self._log_buffer = []

    def export_json(self) -> str:
        """
        Export metrics as JSON string.

        Returns:
            JSON string with all metrics
        """
        data = {
            operation: stats.to_dict()
            for operation, stats in self._metrics.items()
        }
        return json.dumps(data, indent=2)

    def flush_log(self) -> None:
        """Flush log buffer to file."""
        if not self._log_file or not self._log_buffer:
            return

        try:
            log_path = Path(self._log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            with open(log_path, 'a') as f:
                for entry in self._log_buffer:
                    f.write(json.dumps(entry) + '\n')

            self._log_buffer = []
        except Exception as e:
            logger.error(f"Failed to flush metrics log: {e}")


# =============================================================================
# LATENCY PROFILING UTILITIES
# =============================================================================

@contextmanager
def profile_latency(
    metrics: PerformanceMetrics,
    operation: str
):
    """
    Context manager for profiling code block latency.

    Args:
        metrics: PerformanceMetrics instance to record to
        operation: Name of the operation being profiled

    Example:
        with profile_latency(metrics, "telemetry_processing"):
            process_telemetry(data)
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics.record(operation, elapsed_ms)


@asynccontextmanager
async def profile_latency_async(
    metrics: PerformanceMetrics,
    operation: str
):
    """
    Async context manager for profiling async code block latency.

    Args:
        metrics: PerformanceMetrics instance to record to
        operation: Name of the operation being profiled

    Example:
        async with profile_latency_async(metrics, "llm_response"):
            response = await llm.invoke(prompt)
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        metrics.record(operation, elapsed_ms)


def profile(
    metrics: PerformanceMetrics,
    operation: str
) -> Callable[[F], F]:
    """
    Decorator for profiling function latency.

    Args:
        metrics: PerformanceMetrics instance to record to
        operation: Name of the operation being profiled

    Returns:
        Decorated function

    Example:
        @profile(metrics, "compute_strategy")
        def compute_strategy(data):
            # ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with profile_latency(metrics, operation):
                return func(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator


def profile_async(
    metrics: PerformanceMetrics,
    operation: str
) -> Callable[[F], F]:
    """
    Decorator for profiling async function latency.

    Args:
        metrics: PerformanceMetrics instance to record to
        operation: Name of the operation being profiled

    Returns:
        Decorated async function

    Example:
        @profile_async(metrics, "generate_response")
        async def generate_response(prompt):
            # ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with profile_latency_async(metrics, operation):
                return await func(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator
