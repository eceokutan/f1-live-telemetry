"""
Tests for Priority Queue System - Phase 4, Section 8

These tests verify the expected behavior for:
1. Heapq-based priority queue ordering
2. Priority levels (CRITICAL > HIGH > MEDIUM > LOW)
3. Event queueing logic with push/pop operations
4. Queue management (peek, clear, size, is_empty)
5. Priority-based filtering and interrupt logic
6. Capacity handling and LOW priority skip logic

Run with: pytest tests/test_priority_queue.py -v
"""

import time
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis_granite.schemas.events import Event, Priority
from jarvis_granite.live.priority_queue import PriorityQueue, QueueEntry


# =============================================================================
# PRIORITY QUEUE CREATION AND BASICS
# =============================================================================

class TestPriorityQueueCreation:
    """Tests for PriorityQueue initialization and basic properties."""

    def test_create_empty_queue(self):
        """Should create an empty priority queue."""
        queue = PriorityQueue()

        assert queue.is_empty()
        assert queue.size() == 0
        assert len(queue) == 0

    def test_create_queue_with_max_size(self):
        """Should create queue with specified max size."""
        queue = PriorityQueue(max_size=50)

        assert queue._max_size == 50

    def test_default_max_size_is_100(self):
        """Should default to max size of 100."""
        queue = PriorityQueue()

        assert queue._max_size == 100

    def test_queue_bool_false_when_empty(self):
        """Empty queue should be falsy."""
        queue = PriorityQueue()

        assert not bool(queue)

    def test_queue_bool_true_when_not_empty(self):
        """Non-empty queue should be truthy."""
        queue = PriorityQueue()
        event = Event(
            type="test",
            priority=Priority.MEDIUM,
            data={},
            timestamp=time.time()
        )
        queue.push(event)

        assert bool(queue)


# =============================================================================
# PUSH AND POP OPERATIONS
# =============================================================================

class TestPushPopOperations:
    """Tests for push and pop operations."""

    def test_push_adds_event_to_queue(self):
        """Push should add event to queue."""
        queue = PriorityQueue()
        event = Event(
            type="fuel_warning",
            priority=Priority.HIGH,
            data={"laps": 5},
            timestamp=time.time()
        )

        result = queue.push(event)

        assert result is True
        assert queue.size() == 1

    def test_pop_returns_event(self):
        """Pop should return the pushed event."""
        queue = PriorityQueue()
        event = Event(
            type="lap_complete",
            priority=Priority.MEDIUM,
            data={"lap": 10},
            timestamp=time.time()
        )
        queue.push(event)

        popped = queue.pop()

        assert popped is not None
        assert popped.type == "lap_complete"
        assert queue.is_empty()

    def test_pop_on_empty_queue_returns_none(self):
        """Pop on empty queue should return None."""
        queue = PriorityQueue()

        result = queue.pop()

        assert result is None

    def test_push_multiple_events(self):
        """Should push multiple events."""
        queue = PriorityQueue()

        for i in range(5):
            event = Event(
                type=f"event_{i}",
                priority=Priority.MEDIUM,
                data={"index": i},
                timestamp=time.time()
            )
            queue.push(event)

        assert queue.size() == 5


# =============================================================================
# PRIORITY ORDERING
# =============================================================================

class TestPriorityOrdering:
    """Tests for priority-based ordering."""

    def test_critical_pops_before_high(self):
        """CRITICAL priority events should pop before HIGH."""
        queue = PriorityQueue()

        high_event = Event(
            type="high_event",
            priority=Priority.HIGH,
            data={},
            timestamp=time.time()
        )
        critical_event = Event(
            type="critical_event",
            priority=Priority.CRITICAL,
            data={},
            timestamp=time.time() + 1  # Added later
        )

        queue.push(high_event)
        queue.push(critical_event)

        first = queue.pop()
        assert first.priority == Priority.CRITICAL

    def test_high_pops_before_medium(self):
        """HIGH priority events should pop before MEDIUM."""
        queue = PriorityQueue()

        medium_event = Event(
            type="medium_event",
            priority=Priority.MEDIUM,
            data={},
            timestamp=time.time()
        )
        high_event = Event(
            type="high_event",
            priority=Priority.HIGH,
            data={},
            timestamp=time.time() + 1
        )

        queue.push(medium_event)
        queue.push(high_event)

        first = queue.pop()
        assert first.priority == Priority.HIGH

    def test_medium_pops_before_low(self):
        """MEDIUM priority events should pop before LOW."""
        queue = PriorityQueue()

        low_event = Event(
            type="low_event",
            priority=Priority.LOW,
            data={},
            timestamp=time.time()
        )
        medium_event = Event(
            type="medium_event",
            priority=Priority.MEDIUM,
            data={},
            timestamp=time.time() + 1
        )

        queue.push(low_event)
        queue.push(medium_event)

        first = queue.pop()
        assert first.priority == Priority.MEDIUM

    def test_fifo_order_within_same_priority(self):
        """Events with same priority should be FIFO ordered."""
        queue = PriorityQueue()

        # Add events with same priority but different timestamps
        for i in range(3):
            event = Event(
                type=f"event_{i}",
                priority=Priority.MEDIUM,
                data={"order": i},
                timestamp=time.time() + i * 0.001
            )
            queue.push(event)

        # Should pop in insertion order
        first = queue.pop()
        second = queue.pop()
        third = queue.pop()

        assert first.data["order"] == 0
        assert second.data["order"] == 1
        assert third.data["order"] == 2

    def test_full_priority_ordering(self):
        """Should pop in full priority order."""
        queue = PriorityQueue()

        # Add events in reverse priority order
        events = [
            Event(type="low", priority=Priority.LOW, data={}, timestamp=time.time()),
            Event(type="medium", priority=Priority.MEDIUM, data={}, timestamp=time.time()),
            Event(type="high", priority=Priority.HIGH, data={}, timestamp=time.time()),
            Event(type="critical", priority=Priority.CRITICAL, data={}, timestamp=time.time()),
        ]

        for event in events:
            queue.push(event)

        # Should pop in priority order
        assert queue.pop().type == "critical"
        assert queue.pop().type == "high"
        assert queue.pop().type == "medium"
        assert queue.pop().type == "low"


# =============================================================================
# PEEK OPERATIONS
# =============================================================================

class TestPeekOperations:
    """Tests for peek operations."""

    def test_peek_returns_highest_priority(self):
        """Peek should return highest priority event."""
        queue = PriorityQueue()

        medium_event = Event(
            type="medium",
            priority=Priority.MEDIUM,
            data={},
            timestamp=time.time()
        )
        high_event = Event(
            type="high",
            priority=Priority.HIGH,
            data={},
            timestamp=time.time()
        )

        queue.push(medium_event)
        queue.push(high_event)

        peeked = queue.peek()

        assert peeked is not None
        assert peeked.priority == Priority.HIGH

    def test_peek_does_not_remove_event(self):
        """Peek should not remove the event."""
        queue = PriorityQueue()

        event = Event(
            type="test",
            priority=Priority.MEDIUM,
            data={},
            timestamp=time.time()
        )
        queue.push(event)

        queue.peek()
        queue.peek()

        assert queue.size() == 1

    def test_peek_on_empty_returns_none(self):
        """Peek on empty queue should return None."""
        queue = PriorityQueue()

        assert queue.peek() is None

    def test_peek_priority_returns_priority_level(self):
        """peek_priority should return priority enum."""
        queue = PriorityQueue()

        event = Event(
            type="critical",
            priority=Priority.CRITICAL,
            data={},
            timestamp=time.time()
        )
        queue.push(event)

        priority = queue.peek_priority()

        assert priority == Priority.CRITICAL

    def test_peek_priority_on_empty_returns_none(self):
        """peek_priority on empty queue should return None."""
        queue = PriorityQueue()

        assert queue.peek_priority() is None


# =============================================================================
# QUEUE MANAGEMENT
# =============================================================================

class TestQueueManagement:
    """Tests for queue management operations."""

    def test_clear_removes_all_events(self):
        """Clear should remove all events."""
        queue = PriorityQueue()

        for i in range(5):
            event = Event(
                type=f"event_{i}",
                priority=Priority.MEDIUM,
                data={},
                timestamp=time.time()
            )
            queue.push(event)

        queue.clear()

        assert queue.is_empty()
        assert queue.size() == 0

    def test_size_returns_correct_count(self):
        """Size should return correct event count."""
        queue = PriorityQueue()

        assert queue.size() == 0

        for i in range(3):
            event = Event(
                type=f"event_{i}",
                priority=Priority.MEDIUM,
                data={},
                timestamp=time.time()
            )
            queue.push(event)

        assert queue.size() == 3

    def test_is_empty_when_empty(self):
        """is_empty should return True when empty."""
        queue = PriorityQueue()

        assert queue.is_empty() is True

    def test_is_empty_when_not_empty(self):
        """is_empty should return False when not empty."""
        queue = PriorityQueue()

        event = Event(
            type="test",
            priority=Priority.MEDIUM,
            data={},
            timestamp=time.time()
        )
        queue.push(event)

        assert queue.is_empty() is False


# =============================================================================
# INTERRUPT LOGIC
# =============================================================================

class TestInterruptLogic:
    """Tests for interrupt handling logic."""

    def test_has_critical_event_true(self):
        """has_critical_event should return True when CRITICAL exists."""
        queue = PriorityQueue()

        event = Event(
            type="brake_failure",
            priority=Priority.CRITICAL,
            data={},
            timestamp=time.time()
        )
        queue.push(event)

        assert queue.has_critical_event() is True

    def test_has_critical_event_false(self):
        """has_critical_event should return False when no CRITICAL."""
        queue = PriorityQueue()

        event = Event(
            type="high_event",
            priority=Priority.HIGH,
            data={},
            timestamp=time.time()
        )
        queue.push(event)

        assert queue.has_critical_event() is False

    def test_has_high_priority_event_for_critical(self):
        """has_high_priority_event should return True for CRITICAL."""
        queue = PriorityQueue()

        event = Event(
            type="critical",
            priority=Priority.CRITICAL,
            data={},
            timestamp=time.time()
        )
        queue.push(event)

        assert queue.has_high_priority_event() is True

    def test_has_high_priority_event_for_high(self):
        """has_high_priority_event should return True for HIGH."""
        queue = PriorityQueue()

        event = Event(
            type="high",
            priority=Priority.HIGH,
            data={},
            timestamp=time.time()
        )
        queue.push(event)

        assert queue.has_high_priority_event() is True

    def test_has_high_priority_event_false_for_medium(self):
        """has_high_priority_event should return False for MEDIUM."""
        queue = PriorityQueue()

        event = Event(
            type="medium",
            priority=Priority.MEDIUM,
            data={},
            timestamp=time.time()
        )
        queue.push(event)

        assert queue.has_high_priority_event() is False

    def test_should_interrupt_critical_always(self):
        """CRITICAL should always interrupt."""
        queue = PriorityQueue()

        critical = Event(
            type="critical",
            priority=Priority.CRITICAL,
            data={},
            timestamp=time.time()
        )
        queue.push(critical)

        # Should interrupt even HIGH priority
        assert queue.should_interrupt(Priority.HIGH) is True
        # Should interrupt MEDIUM
        assert queue.should_interrupt(Priority.MEDIUM) is True
        # Should interrupt LOW
        assert queue.should_interrupt(Priority.LOW) is True

    def test_should_interrupt_high_on_medium(self):
        """HIGH should interrupt MEDIUM."""
        queue = PriorityQueue()

        high = Event(
            type="high",
            priority=Priority.HIGH,
            data={},
            timestamp=time.time()
        )
        queue.push(high)

        assert queue.should_interrupt(Priority.MEDIUM) is True
        assert queue.should_interrupt(Priority.LOW) is True

    def test_should_not_interrupt_high_on_high(self):
        """HIGH should not interrupt HIGH."""
        queue = PriorityQueue()

        high = Event(
            type="high",
            priority=Priority.HIGH,
            data={},
            timestamp=time.time()
        )
        queue.push(high)

        assert queue.should_interrupt(Priority.HIGH) is False

    def test_should_not_interrupt_medium_on_medium(self):
        """MEDIUM should not interrupt MEDIUM."""
        queue = PriorityQueue()

        medium = Event(
            type="medium",
            priority=Priority.MEDIUM,
            data={},
            timestamp=time.time()
        )
        queue.push(medium)

        assert queue.should_interrupt(Priority.MEDIUM) is False

    def test_pop_if_higher_priority_returns_event(self):
        """pop_if_higher_priority should return higher priority event."""
        queue = PriorityQueue()

        critical = Event(
            type="critical",
            priority=Priority.CRITICAL,
            data={},
            timestamp=time.time()
        )
        queue.push(critical)

        result = queue.pop_if_higher_priority(Priority.MEDIUM)

        assert result is not None
        assert result.priority == Priority.CRITICAL

    def test_pop_if_higher_priority_returns_none_for_same(self):
        """pop_if_higher_priority should return None for same priority."""
        queue = PriorityQueue()

        medium = Event(
            type="medium",
            priority=Priority.MEDIUM,
            data={},
            timestamp=time.time()
        )
        queue.push(medium)

        result = queue.pop_if_higher_priority(Priority.MEDIUM)

        assert result is None
        assert queue.size() == 1  # Event still in queue


# =============================================================================
# CAPACITY HANDLING
# =============================================================================

class TestCapacityHandling:
    """Tests for queue capacity limits."""

    def test_low_priority_skipped_at_capacity(self):
        """LOW priority events should be skipped at capacity."""
        queue = PriorityQueue(max_size=3)

        # Fill queue with MEDIUM events
        for i in range(3):
            event = Event(
                type=f"medium_{i}",
                priority=Priority.MEDIUM,
                data={},
                timestamp=time.time()
            )
            queue.push(event)

        # Try to add LOW priority
        low = Event(
            type="low",
            priority=Priority.LOW,
            data={},
            timestamp=time.time()
        )

        result = queue.push(low)

        assert result is False
        assert queue.size() == 3

    def test_higher_priority_drops_low_at_capacity(self):
        """Higher priority should drop LOW when at capacity."""
        queue = PriorityQueue(max_size=3)

        # Fill with LOW events
        for i in range(3):
            event = Event(
                type=f"low_{i}",
                priority=Priority.LOW,
                data={},
                timestamp=time.time() + i
            )
            queue.push(event)

        # Add HIGH priority
        high = Event(
            type="high",
            priority=Priority.HIGH,
            data={},
            timestamp=time.time()
        )

        result = queue.push(high)

        assert result is True
        assert queue.size() == 3
        # HIGH should be at top
        assert queue.peek().priority == Priority.HIGH

    def test_drop_below_priority(self):
        """drop_below_priority should remove lower priority events."""
        queue = PriorityQueue()

        # Add events of all priorities
        for priority in [Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.LOW]:
            event = Event(
                type=f"{priority.name}",
                priority=priority,
                data={},
                timestamp=time.time()
            )
            queue.push(event)

        # Drop LOW and MEDIUM
        dropped = queue.drop_below_priority(Priority.HIGH)

        assert dropped == 2  # MEDIUM and LOW dropped
        assert queue.size() == 2
        # Only CRITICAL and HIGH remain
        assert queue.pop().priority == Priority.CRITICAL
        assert queue.pop().priority == Priority.HIGH


# =============================================================================
# STATISTICS AND REPR
# =============================================================================

class TestQueueStats:
    """Tests for queue statistics."""

    def test_get_queue_stats_empty(self):
        """get_queue_stats should work on empty queue."""
        queue = PriorityQueue()

        stats = queue.get_queue_stats()

        assert stats["total"] == 0
        assert stats["critical"] == 0
        assert stats["high"] == 0
        assert stats["medium"] == 0
        assert stats["low"] == 0

    def test_get_queue_stats_with_events(self):
        """get_queue_stats should count events by priority."""
        queue = PriorityQueue()

        # Add various priority events
        priorities = [
            Priority.CRITICAL, Priority.CRITICAL,
            Priority.HIGH,
            Priority.MEDIUM, Priority.MEDIUM, Priority.MEDIUM,
            Priority.LOW,
        ]

        for i, priority in enumerate(priorities):
            event = Event(
                type=f"event_{i}",
                priority=priority,
                data={},
                timestamp=time.time()
            )
            queue.push(event)

        stats = queue.get_queue_stats()

        assert stats["total"] == 7
        assert stats["critical"] == 2
        assert stats["high"] == 1
        assert stats["medium"] == 3
        assert stats["low"] == 1

    def test_repr(self):
        """__repr__ should show queue state."""
        queue = PriorityQueue()

        event = Event(
            type="test",
            priority=Priority.HIGH,
            data={},
            timestamp=time.time()
        )
        queue.push(event)

        repr_str = repr(queue)

        assert "PriorityQueue" in repr_str
        assert "size=1" in repr_str
        assert "high=1" in repr_str

    def test_get_events_by_priority(self):
        """get_events_by_priority should filter correctly."""
        queue = PriorityQueue()

        # Add events
        for priority in [Priority.HIGH, Priority.MEDIUM, Priority.HIGH]:
            event = Event(
                type=f"{priority.name}",
                priority=priority,
                data={},
                timestamp=time.time()
            )
            queue.push(event)

        high_events = queue.get_events_by_priority(Priority.HIGH)

        assert len(high_events) == 2
        assert all(e.priority == Priority.HIGH for e in high_events)


# =============================================================================
# QUEUE ENTRY COMPARISON
# =============================================================================

class TestQueueEntry:
    """Tests for QueueEntry ordering."""

    def test_entry_comparison_by_priority(self):
        """Entries should compare by priority first."""
        entry1 = QueueEntry(
            priority=Priority.HIGH,
            timestamp=100.0,
            counter=0,
            event=None
        )
        entry2 = QueueEntry(
            priority=Priority.MEDIUM,
            timestamp=100.0,
            counter=1,
            event=None
        )

        assert entry1 < entry2  # HIGH (1) < MEDIUM (2)

    def test_entry_comparison_by_timestamp(self):
        """Entries with same priority should compare by timestamp."""
        entry1 = QueueEntry(
            priority=Priority.MEDIUM,
            timestamp=100.0,
            counter=0,
            event=None
        )
        entry2 = QueueEntry(
            priority=Priority.MEDIUM,
            timestamp=101.0,
            counter=1,
            event=None
        )

        assert entry1 < entry2  # Earlier timestamp first

    def test_entry_comparison_by_counter(self):
        """Entries with same priority and timestamp compare by counter."""
        entry1 = QueueEntry(
            priority=Priority.MEDIUM,
            timestamp=100.0,
            counter=0,
            event=None
        )
        entry2 = QueueEntry(
            priority=Priority.MEDIUM,
            timestamp=100.0,
            counter=1,
            event=None
        )

        assert entry1 < entry2  # Lower counter first
