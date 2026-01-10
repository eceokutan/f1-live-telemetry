"""
Priority Queue System for Jarvis-Granite Live Telemetry.

Implements a heapq-based priority queue for event processing in the
custom orchestrator. Events are ordered by priority level, then by
timestamp for FIFO ordering within the same priority.

Phase 4, Section 8: Priority Queue System
- heapq-based priority queue
- Priority levels (Critical, High, Medium, Low)
- Event queueing logic with interrupt support
"""

import heapq
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from jarvis_granite.schemas.events import Event, Priority


@dataclass
class QueueEntry:
    """
    Wrapper for events in the priority queue.

    Provides proper ordering for heapq:
    - Primary sort by priority (lower value = higher priority)
    - Secondary sort by timestamp (earlier = higher priority, FIFO)
    - Counter for tie-breaking when priority and timestamp are equal

    Attributes:
        priority: Priority level (0=CRITICAL to 3=LOW)
        timestamp: Unix timestamp when event was queued
        counter: Insertion order counter for stable sorting
        event: The actual Event object
    """
    priority: int
    timestamp: float
    counter: int
    event: Event

    def __lt__(self, other: 'QueueEntry') -> bool:
        """Compare entries for heap ordering."""
        # Primary: priority (lower = higher priority)
        if self.priority != other.priority:
            return self.priority < other.priority
        # Secondary: timestamp (earlier = higher priority)
        if self.timestamp != other.timestamp:
            return self.timestamp < other.timestamp
        # Tertiary: counter (lower = earlier insertion)
        return self.counter < other.counter


class PriorityQueue:
    """
    Heapq-based priority queue for event processing.

    Events are ordered by priority level (CRITICAL > HIGH > MEDIUM > LOW)
    and within the same priority by timestamp (FIFO order).

    The queue supports:
    - Efficient push/pop operations (O(log n))
    - Peeking at highest priority event without removal
    - Filtering by priority level
    - Skip logic for low-priority events when system is busy

    Example:
        queue = PriorityQueue()

        # Push events
        queue.push(fuel_critical_event)
        queue.push(lap_complete_event)

        # Pop highest priority
        event = queue.pop()

        # Check for critical events
        if queue.has_critical_event():
            # Handle interrupt
            pass
    """

    def __init__(self, max_size: int = 100):
        """
        Initialize the priority queue.

        Args:
            max_size: Maximum number of events in queue (default 100).
                      Oldest low-priority events are dropped when exceeded.
        """
        self._heap: List[QueueEntry] = []
        self._counter: int = 0  # Monotonic counter for stable sorting
        self._max_size: int = max_size

    def push(self, event: Event) -> bool:
        """
        Add an event to the priority queue.

        Events are ordered by priority (CRITICAL=0 highest, LOW=3 lowest)
        and within the same priority by timestamp (FIFO).

        If the queue is at max capacity:
        - LOW priority events are skipped
        - Higher priority events cause oldest LOW events to be dropped

        Args:
            event: Event to add to the queue

        Returns:
            True if event was added, False if skipped due to capacity
        """
        # Check capacity
        if len(self._heap) >= self._max_size:
            if event.priority == Priority.LOW:
                # Skip LOW priority events when at capacity
                return False

            # Try to make room by removing oldest LOW priority event
            if not self._drop_lowest_priority():
                # Queue is full of higher priority events
                if event.priority == Priority.LOW:
                    return False

        # Create queue entry
        entry = QueueEntry(
            priority=event.priority,
            timestamp=event.timestamp,
            counter=self._counter,
            event=event
        )

        self._counter += 1
        heapq.heappush(self._heap, entry)
        return True

    def pop(self) -> Optional[Event]:
        """
        Remove and return the highest priority event.

        Returns:
            The highest priority event, or None if queue is empty
        """
        if not self._heap:
            return None

        entry = heapq.heappop(self._heap)
        return entry.event

    def peek(self) -> Optional[Event]:
        """
        Return the highest priority event without removing it.

        Returns:
            The highest priority event, or None if queue is empty
        """
        if not self._heap:
            return None

        return self._heap[0].event

    def peek_priority(self) -> Optional[Priority]:
        """
        Return the priority of the highest priority event.

        Returns:
            Priority of next event, or None if queue is empty
        """
        if not self._heap:
            return None

        return Priority(self._heap[0].priority)

    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return len(self._heap) == 0

    def size(self) -> int:
        """Return the number of events in the queue."""
        return len(self._heap)

    def clear(self) -> None:
        """Remove all events from the queue."""
        self._heap.clear()
        # Keep counter to maintain insertion ordering for future events

    def has_critical_event(self) -> bool:
        """
        Check if there is a CRITICAL priority event in the queue.

        Used for interrupt handling - if a critical event is pending,
        the current response should be interrupted.

        Returns:
            True if a CRITICAL event is in the queue
        """
        return self.peek_priority() == Priority.CRITICAL

    def has_high_priority_event(self) -> bool:
        """
        Check if there is a HIGH or CRITICAL priority event.

        Returns:
            True if a HIGH or CRITICAL event is in the queue
        """
        priority = self.peek_priority()
        return priority is not None and priority <= Priority.HIGH

    def should_interrupt(self, current_priority: Priority) -> bool:
        """
        Determine if the current response should be interrupted.

        A response should be interrupted if:
        - There is a CRITICAL event (always interrupts)
        - There is a HIGH event and current priority is MEDIUM or LOW

        Args:
            current_priority: Priority of the currently processing event

        Returns:
            True if the current response should be interrupted
        """
        next_priority = self.peek_priority()

        if next_priority is None:
            return False

        # CRITICAL always interrupts
        if next_priority == Priority.CRITICAL:
            return True

        # HIGH interrupts MEDIUM and LOW
        if next_priority == Priority.HIGH and current_priority > Priority.HIGH:
            return True

        return False

    def pop_if_higher_priority(
        self,
        current_priority: Priority
    ) -> Optional[Event]:
        """
        Pop the next event only if it has higher priority than current.

        Used for interrupt handling - only interrupt for higher priority.

        Args:
            current_priority: Priority of currently processing event

        Returns:
            The higher priority event, or None if not applicable
        """
        next_priority = self.peek_priority()

        if next_priority is None:
            return None

        # Only pop if strictly higher priority (lower value)
        if next_priority < current_priority:
            return self.pop()

        return None

    def get_events_by_priority(self, priority: Priority) -> List[Event]:
        """
        Get all events of a specific priority without removing them.

        Args:
            priority: Priority level to filter by

        Returns:
            List of events matching the priority
        """
        return [
            entry.event
            for entry in self._heap
            if entry.priority == priority
        ]

    def drop_below_priority(self, min_priority: Priority) -> int:
        """
        Remove all events below a minimum priority level.

        Useful for clearing low-priority events when system is busy.

        Args:
            min_priority: Minimum priority to keep (events with higher
                         priority values are dropped)

        Returns:
            Number of events dropped
        """
        original_size = len(self._heap)

        # Keep only events at or above min_priority
        self._heap = [
            entry for entry in self._heap
            if entry.priority <= min_priority
        ]

        # Re-heapify after filtering
        heapq.heapify(self._heap)

        return original_size - len(self._heap)

    def _drop_lowest_priority(self) -> bool:
        """
        Remove the oldest LOW priority event to make room.

        Returns:
            True if an event was dropped, False if no LOW priority events
        """
        # Find LOW priority events
        low_indices = [
            i for i, entry in enumerate(self._heap)
            if entry.priority == Priority.LOW
        ]

        if not low_indices:
            return False

        # Remove the oldest LOW priority event (highest counter = oldest)
        # Actually, lowest counter = oldest in our scheme
        oldest_idx = min(low_indices, key=lambda i: self._heap[i].counter)

        # Remove and re-heapify
        self._heap[oldest_idx] = self._heap[-1]
        self._heap.pop()
        heapq.heapify(self._heap)

        return True

    def get_queue_stats(self) -> dict:
        """
        Get statistics about the current queue state.

        Returns:
            Dictionary with queue statistics
        """
        stats = {
            "total": len(self._heap),
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
        }

        for entry in self._heap:
            if entry.priority == Priority.CRITICAL:
                stats["critical"] += 1
            elif entry.priority == Priority.HIGH:
                stats["high"] += 1
            elif entry.priority == Priority.MEDIUM:
                stats["medium"] += 1
            elif entry.priority == Priority.LOW:
                stats["low"] += 1

        return stats

    def __len__(self) -> int:
        """Return queue size."""
        return len(self._heap)

    def __bool__(self) -> bool:
        """Return True if queue is not empty."""
        return len(self._heap) > 0

    def __repr__(self) -> str:
        """String representation of queue state."""
        stats = self.get_queue_stats()
        return (
            f"PriorityQueue(size={stats['total']}, "
            f"critical={stats['critical']}, high={stats['high']}, "
            f"medium={stats['medium']}, low={stats['low']})"
        )
