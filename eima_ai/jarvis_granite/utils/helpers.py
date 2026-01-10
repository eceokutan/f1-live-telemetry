"""
Common helper functions for Jarvis-Granite Live Telemetry.

This module provides utility functions used across the application.
"""

import uuid
from datetime import datetime
from typing import Optional


def generate_session_id() -> str:
    """
    Generate a unique session ID.

    Returns:
        Unique session ID string in format: session_<uuid4>
    """
    return f"session_{uuid.uuid4().hex[:12]}"


def generate_response_id() -> str:
    """
    Generate a unique response ID.

    Returns:
        Unique response ID string in format: resp_<uuid4>
    """
    return f"resp_{uuid.uuid4().hex[:12]}"


def timestamp_iso() -> str:
    """
    Get current timestamp in ISO format.

    Returns:
        ISO 8601 formatted timestamp string with Z suffix
    """
    return datetime.utcnow().isoformat() + "Z"


def timestamp_ms() -> int:
    """
    Get current timestamp in milliseconds.

    Returns:
        Unix timestamp in milliseconds
    """
    return int(datetime.utcnow().timestamp() * 1000)


def format_lap_time(seconds: float) -> str:
    """
    Format lap time from seconds to MM:SS.mmm format.

    Args:
        seconds: Lap time in seconds

    Returns:
        Formatted string in MM:SS.mmm format

    Example:
        >>> format_lap_time(85.234)
        '1:25.234'
    """
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}:{remaining_seconds:06.3f}"


def format_gap(seconds: float) -> str:
    """
    Format time gap for display.

    Args:
        seconds: Gap in seconds

    Returns:
        Formatted gap string

    Example:
        >>> format_gap(1.234)
        '+1.234s'
        >>> format_gap(-0.5)
        '-0.500s'
    """
    sign = '+' if seconds >= 0 else ''
    return f"{sign}{seconds:.3f}s"


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value to a specified range.

    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Perform safe division with a default value for division by zero.

    Args:
        numerator: Dividend
        denominator: Divisor
        default: Default value if denominator is zero

    Returns:
        Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def calculate_fuel_laps_remaining(
    fuel_remaining: float,
    fuel_per_lap: float,
    safety_margin: float = 0.0
) -> float:
    """
    Calculate estimated laps remaining based on fuel.

    Args:
        fuel_remaining: Current fuel amount in liters
        fuel_per_lap: Average fuel consumption per lap
        safety_margin: Fuel to reserve (default 0)

    Returns:
        Estimated laps remaining
    """
    usable_fuel = max(0, fuel_remaining - safety_margin)
    return safe_division(usable_fuel, fuel_per_lap)


def tire_condition_label(wear_percentage: float) -> str:
    """
    Get human-readable tire condition label.

    Args:
        wear_percentage: Tire wear percentage (0-100)

    Returns:
        Condition label string
    """
    if wear_percentage < 20:
        return "Fresh"
    elif wear_percentage < 40:
        return "Good"
    elif wear_percentage < 60:
        return "Used"
    elif wear_percentage < 80:
        return "Worn"
    else:
        return "Critical"


def tire_temp_status(temp_celsius: float, warning: float = 100.0, critical: float = 110.0) -> str:
    """
    Get tire temperature status.

    Args:
        temp_celsius: Tire temperature in Celsius
        warning: Warning threshold
        critical: Critical threshold

    Returns:
        Status string: "optimal", "warning", or "critical"
    """
    if temp_celsius >= critical:
        return "critical"
    elif temp_celsius >= warning:
        return "warning"
    return "optimal"
