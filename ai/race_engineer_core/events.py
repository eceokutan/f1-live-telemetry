"""
Event schemas for Jarvis-Granite Live Telemetry.

Defines:
- Priority enum for event prioritization (used with heapq)
- Event dataclass for telemetry events
- Factory functions for creating common event types
"""

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional


class Priority(IntEnum):
    """
    Event priority levels for the orchestrator queue.

    Lower values = higher priority (for heapq min-heap).

    CRITICAL (0): Immediate interrupt - brake failure, collision warning
    HIGH (1): Interrupt medium/low - pit now, fuel critical, tire failure
    MEDIUM (2): Queue normally - pit window, gap changes, lap summary
    LOW (3): Skip if busy - sector times, minor updates
    """
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class Event:
    """
    Telemetry event detected by the TelemetryAgent.

    Events are queued by the orchestrator based on priority
    and processed to generate AI responses.

    Attributes:
        type: Event type identifier (e.g., "fuel_critical", "lap_complete")
        priority: Priority level for queue ordering
        data: Event-specific data payload
        timestamp: Unix timestamp when event was detected
    """
    type: str
    priority: Priority
    data: Dict[str, Any]
    timestamp: float


# =============================================================================
# EVENT FACTORY FUNCTIONS
# =============================================================================

def create_fuel_critical_event(laps_remaining: float) -> Event:
    """
    Create a fuel critical event.

    Args:
        laps_remaining: Estimated laps of fuel remaining

    Returns:
        Event with CRITICAL priority
    """
    return Event(
        type="fuel_critical",
        priority=Priority.CRITICAL,
        data={"laps": laps_remaining},
        timestamp=time.time()
    )


def create_fuel_warning_event(laps_remaining: float) -> Event:
    """
    Create a fuel warning event.

    Args:
        laps_remaining: Estimated laps of fuel remaining

    Returns:
        Event with HIGH priority
    """
    return Event(
        type="fuel_warning",
        priority=Priority.HIGH,
        data={"laps": laps_remaining},
        timestamp=time.time()
    )


def create_tire_critical_event(temp: float, position: str) -> Event:
    """
    Create a tire critical temperature event.

    Args:
        temp: Tire temperature in Celsius
        position: Tire position (fl, fr, rl, rr)

    Returns:
        Event with CRITICAL priority
    """
    return Event(
        type="tire_critical",
        priority=Priority.CRITICAL,
        data={"temp": temp, "position": position},
        timestamp=time.time()
    )


def create_tire_warning_event(temp: float, position: str) -> Event:
    """
    Create a tire warning temperature event.

    Args:
        temp: Tire temperature in Celsius
        position: Tire position (fl, fr, rl, rr)

    Returns:
        Event with MEDIUM priority
    """
    return Event(
        type="tire_warning",
        priority=Priority.MEDIUM,
        data={"temp": temp, "position": position},
        timestamp=time.time()
    )


def create_gap_change_event(
    gap_change: float,
    direction: str,
    new_gap: Optional[float] = None
) -> Event:
    """
    Create a gap change event.

    Args:
        gap_change: Magnitude of gap change in seconds
        direction: "ahead" or "behind"
        new_gap: New gap value (optional)

    Returns:
        Event with MEDIUM priority
    """
    data = {"change": gap_change, "direction": direction}
    if new_gap is not None:
        data["new_gap"] = new_gap

    return Event(
        type="gap_change",
        priority=Priority.MEDIUM,
        data=data,
        timestamp=time.time()
    )


def create_lap_complete_event(
    lap_number: int,
    lap_time: float,
    best_lap: Optional[float] = None,
    delta: Optional[float] = None
) -> Event:
    """
    Create a lap completion event.

    Args:
        lap_number: Completed lap number
        lap_time: Lap time in seconds
        best_lap: Best lap time (optional)
        delta: Delta to best lap (optional)

    Returns:
        Event with MEDIUM priority
    """
    data = {"lap": lap_number, "time": lap_time}
    if best_lap is not None:
        data["best"] = best_lap
    if delta is not None:
        data["delta"] = delta

    return Event(
        type="lap_complete",
        priority=Priority.MEDIUM,
        data=data,
        timestamp=time.time()
    )


def create_sector_complete_event(
    sector: int,
    sector_time: float,
    best_sector: Optional[float] = None
) -> Event:
    """
    Create a sector completion event.

    Args:
        sector: Completed sector number (1-3)
        sector_time: Sector time in seconds
        best_sector: Best sector time (optional)

    Returns:
        Event with LOW priority
    """
    data = {"sector": sector, "time": sector_time}
    if best_sector is not None:
        data["best"] = best_sector

    return Event(
        type="sector_complete",
        priority=Priority.LOW,
        data=data,
        timestamp=time.time()
    )


def create_pit_window_event(
    reason: str,
    recommended_lap: Optional[int] = None
) -> Event:
    """
    Create a pit window open event.

    Args:
        reason: Why pit window opened (fuel, tires, strategy)
        recommended_lap: Recommended lap to pit (optional)

    Returns:
        Event with HIGH priority
    """
    data = {"reason": reason}
    if recommended_lap is not None:
        data["recommended_lap"] = recommended_lap

    return Event(
        type="pit_window_open",
        priority=Priority.HIGH,
        data=data,
        timestamp=time.time()
    )
