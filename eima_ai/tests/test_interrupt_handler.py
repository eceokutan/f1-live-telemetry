"""
Tests for Interrupt Handler - Phase 4, Section 10

These tests verify the expected behavior for:
1. Sentence completion detection
2. Pending interrupt queue management
3. Priority-based interrupt logic (Critical/High interrupts Medium/Low)
4. Driver query interrupt handling
5. Graceful speech completion before interrupt

Run with: pytest tests/test_interrupt_handler.py -v
"""

import asyncio
import time
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis_granite.schemas.events import Event, Priority
from jarvis_granite.live.interrupt_handler import InterruptHandler, InterruptType


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def interrupt_handler():
    """Create an InterruptHandler instance."""
    return InterruptHandler()


@pytest.fixture
def critical_event():
    """Create a critical priority event."""
    return Event(
        type="fuel_critical",
        priority=Priority.CRITICAL,
        data={"laps": 1.5},
        timestamp=time.time()
    )


@pytest.fixture
def high_event():
    """Create a high priority event."""
    return Event(
        type="fuel_warning",
        priority=Priority.HIGH,
        data={"laps": 4.0},
        timestamp=time.time()
    )


@pytest.fixture
def medium_event():
    """Create a medium priority event."""
    return Event(
        type="lap_complete",
        priority=Priority.MEDIUM,
        data={"lap": 10},
        timestamp=time.time()
    )


@pytest.fixture
def low_event():
    """Create a low priority event."""
    return Event(
        type="sector_complete",
        priority=Priority.LOW,
        data={"sector": 2},
        timestamp=time.time()
    )


# =============================================================================
# INTERRUPT HANDLER INITIALIZATION
# =============================================================================

class TestInterruptHandlerInitialization:
    """Tests for InterruptHandler initialization."""

    def test_create_interrupt_handler(self):
        """Should create an InterruptHandler instance."""
        handler = InterruptHandler()

        assert handler is not None
        assert handler.is_speaking is False
        assert handler.current_priority is None

    def test_interrupt_handler_has_required_methods(self):
        """InterruptHandler should have required methods."""
        handler = InterruptHandler()

        assert hasattr(handler, 'start_speaking')
        assert hasattr(handler, 'on_sentence_complete')
        assert hasattr(handler, 'stop_speaking')
        assert hasattr(handler, 'request_interrupt')
        assert hasattr(handler, 'should_interrupt')
        assert hasattr(handler, 'get_pending_interrupt')

    def test_no_pending_interrupt_initially(self):
        """Should have no pending interrupt initially."""
        handler = InterruptHandler()

        assert handler.get_pending_interrupt() is None


# =============================================================================
# SPEAKING STATE MANAGEMENT
# =============================================================================

class TestSpeakingStateManagement:
    """Tests for speaking state management."""

    def test_start_speaking_sets_state(self, interrupt_handler, medium_event):
        """start_speaking should set speaking state."""
        interrupt_handler.start_speaking(medium_event.priority)

        assert interrupt_handler.is_speaking is True
        assert interrupt_handler.current_priority == medium_event.priority

    def test_stop_speaking_clears_state(self, interrupt_handler, medium_event):
        """stop_speaking should clear speaking state."""
        interrupt_handler.start_speaking(medium_event.priority)
        interrupt_handler.stop_speaking()

        assert interrupt_handler.is_speaking is False
        assert interrupt_handler.current_priority is None

    def test_stop_speaking_when_not_speaking(self, interrupt_handler):
        """stop_speaking when not speaking should not raise."""
        interrupt_handler.stop_speaking()

        assert interrupt_handler.is_speaking is False

    def test_get_speaking_state(self, interrupt_handler, medium_event):
        """Should be able to get current speaking state."""
        assert interrupt_handler.is_speaking is False

        interrupt_handler.start_speaking(medium_event.priority)

        assert interrupt_handler.is_speaking is True


# =============================================================================
# SENTENCE COMPLETION DETECTION
# =============================================================================

class TestSentenceCompletionDetection:
    """Tests for sentence completion detection."""

    def test_split_text_into_sentences(self, interrupt_handler):
        """Should split text into sentences."""
        text = "First sentence. Second sentence! Third sentence?"

        sentences = interrupt_handler.split_into_sentences(text)

        assert len(sentences) == 3
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence!"
        assert sentences[2] == "Third sentence?"

    def test_split_preserves_sentence_endings(self, interrupt_handler):
        """Should preserve sentence ending punctuation."""
        text = "Box this lap. Fuel is critical!"

        sentences = interrupt_handler.split_into_sentences(text)

        assert sentences[0].endswith(".")
        assert sentences[1].endswith("!")

    def test_split_handles_single_sentence(self, interrupt_handler):
        """Should handle single sentence text."""
        text = "Box this lap for fresh tires."

        sentences = interrupt_handler.split_into_sentences(text)

        assert len(sentences) == 1
        assert sentences[0] == "Box this lap for fresh tires."

    def test_split_handles_empty_text(self, interrupt_handler):
        """Should handle empty text."""
        sentences = interrupt_handler.split_into_sentences("")

        assert len(sentences) == 0

    def test_split_handles_abbreviations(self, interrupt_handler):
        """Should handle common abbreviations correctly."""
        text = "P1 is 2.5s ahead. P2 is gaining."

        sentences = interrupt_handler.split_into_sentences(text)

        # Should recognize these as two sentences
        assert len(sentences) == 2

    def test_split_handles_numbers_with_decimals(self, interrupt_handler):
        """Should not split on decimal points in numbers."""
        text = "Gap is 2.5 seconds. Fuel at 45.2 liters."

        sentences = interrupt_handler.split_into_sentences(text)

        assert len(sentences) == 2


# =============================================================================
# INTERRUPT REQUEST HANDLING
# =============================================================================

class TestInterruptRequestHandling:
    """Tests for interrupt request handling."""

    def test_request_event_interrupt(self, interrupt_handler, critical_event):
        """Should accept event interrupt request."""
        result = interrupt_handler.request_interrupt(
            interrupt_type=InterruptType.EVENT,
            event=critical_event
        )

        assert result is True

    def test_request_driver_query_interrupt(self, interrupt_handler):
        """Should accept driver query interrupt request."""
        result = interrupt_handler.request_interrupt(
            interrupt_type=InterruptType.DRIVER_QUERY,
            query="What's my fuel status?"
        )

        assert result is True

    def test_pending_interrupt_stored(self, interrupt_handler, critical_event):
        """Interrupt request should be stored as pending."""
        interrupt_handler.request_interrupt(
            interrupt_type=InterruptType.EVENT,
            event=critical_event
        )

        pending = interrupt_handler.get_pending_interrupt()

        assert pending is not None
        assert pending['type'] == InterruptType.EVENT
        assert pending['event'] == critical_event

    def test_driver_query_stored_as_pending(self, interrupt_handler):
        """Driver query should be stored as pending interrupt."""
        query = "How are my tires?"
        interrupt_handler.request_interrupt(
            interrupt_type=InterruptType.DRIVER_QUERY,
            query=query
        )

        pending = interrupt_handler.get_pending_interrupt()

        assert pending is not None
        assert pending['type'] == InterruptType.DRIVER_QUERY
        assert pending['query'] == query

    def test_higher_priority_replaces_pending(
        self,
        interrupt_handler,
        high_event,
        critical_event
    ):
        """Higher priority interrupt should replace lower priority pending."""
        interrupt_handler.request_interrupt(
            interrupt_type=InterruptType.EVENT,
            event=high_event
        )
        interrupt_handler.request_interrupt(
            interrupt_type=InterruptType.EVENT,
            event=critical_event
        )

        pending = interrupt_handler.get_pending_interrupt()

        assert pending['event'].priority == Priority.CRITICAL

    def test_lower_priority_does_not_replace(
        self,
        interrupt_handler,
        high_event,
        medium_event
    ):
        """Lower priority interrupt should not replace higher priority pending."""
        interrupt_handler.request_interrupt(
            interrupt_type=InterruptType.EVENT,
            event=high_event
        )
        interrupt_handler.request_interrupt(
            interrupt_type=InterruptType.EVENT,
            event=medium_event
        )

        pending = interrupt_handler.get_pending_interrupt()

        assert pending['event'].priority == Priority.HIGH


# =============================================================================
# PRIORITY-BASED INTERRUPT LOGIC
# =============================================================================

class TestPriorityBasedInterruptLogic:
    """Tests for priority-based interrupt logic."""

    def test_critical_interrupts_high(self, interrupt_handler, critical_event):
        """CRITICAL should interrupt HIGH priority speech."""
        interrupt_handler.start_speaking(Priority.HIGH)

        should_interrupt = interrupt_handler.should_interrupt(critical_event.priority)

        assert should_interrupt is True

    def test_critical_interrupts_medium(self, interrupt_handler, critical_event):
        """CRITICAL should interrupt MEDIUM priority speech."""
        interrupt_handler.start_speaking(Priority.MEDIUM)

        should_interrupt = interrupt_handler.should_interrupt(critical_event.priority)

        assert should_interrupt is True

    def test_critical_interrupts_low(self, interrupt_handler, critical_event):
        """CRITICAL should interrupt LOW priority speech."""
        interrupt_handler.start_speaking(Priority.LOW)

        should_interrupt = interrupt_handler.should_interrupt(critical_event.priority)

        assert should_interrupt is True

    def test_high_interrupts_medium(self, interrupt_handler, high_event):
        """HIGH should interrupt MEDIUM priority speech."""
        interrupt_handler.start_speaking(Priority.MEDIUM)

        should_interrupt = interrupt_handler.should_interrupt(high_event.priority)

        assert should_interrupt is True

    def test_high_interrupts_low(self, interrupt_handler, high_event):
        """HIGH should interrupt LOW priority speech."""
        interrupt_handler.start_speaking(Priority.LOW)

        should_interrupt = interrupt_handler.should_interrupt(high_event.priority)

        assert should_interrupt is True

    def test_high_does_not_interrupt_high(self, interrupt_handler, high_event):
        """HIGH should not interrupt HIGH priority speech."""
        interrupt_handler.start_speaking(Priority.HIGH)

        should_interrupt = interrupt_handler.should_interrupt(high_event.priority)

        assert should_interrupt is False

    def test_high_does_not_interrupt_critical(self, interrupt_handler, high_event):
        """HIGH should not interrupt CRITICAL priority speech."""
        interrupt_handler.start_speaking(Priority.CRITICAL)

        should_interrupt = interrupt_handler.should_interrupt(high_event.priority)

        assert should_interrupt is False

    def test_medium_does_not_interrupt_medium(self, interrupt_handler, medium_event):
        """MEDIUM should not interrupt MEDIUM priority speech."""
        interrupt_handler.start_speaking(Priority.MEDIUM)

        should_interrupt = interrupt_handler.should_interrupt(medium_event.priority)

        assert should_interrupt is False

    def test_low_does_not_interrupt_anything(self, interrupt_handler, low_event):
        """LOW should not interrupt any priority speech."""
        for priority in [Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.LOW]:
            interrupt_handler.start_speaking(priority)

            should_interrupt = interrupt_handler.should_interrupt(low_event.priority)

            assert should_interrupt is False

    def test_driver_query_treated_as_high_priority(self, interrupt_handler):
        """Driver query should be treated as HIGH priority for interrupts."""
        interrupt_handler.start_speaking(Priority.MEDIUM)

        # Driver query should interrupt MEDIUM
        should_interrupt = interrupt_handler.should_interrupt_for_query()

        assert should_interrupt is True

    def test_driver_query_does_not_interrupt_critical(self, interrupt_handler):
        """Driver query should not interrupt CRITICAL priority speech."""
        interrupt_handler.start_speaking(Priority.CRITICAL)

        should_interrupt = interrupt_handler.should_interrupt_for_query()

        assert should_interrupt is False


# =============================================================================
# SENTENCE COMPLETION CALLBACKS
# =============================================================================

class TestSentenceCompletionCallbacks:
    """Tests for sentence completion callback handling."""

    def test_on_sentence_complete_returns_pending(
        self,
        interrupt_handler,
        medium_event,
        critical_event
    ):
        """on_sentence_complete should return pending interrupt."""
        interrupt_handler.start_speaking(medium_event.priority)
        interrupt_handler.request_interrupt(
            interrupt_type=InterruptType.EVENT,
            event=critical_event
        )

        pending = interrupt_handler.on_sentence_complete()

        assert pending is not None
        assert pending['event'] == critical_event

    def test_on_sentence_complete_clears_pending(
        self,
        interrupt_handler,
        medium_event,
        critical_event
    ):
        """on_sentence_complete should clear pending interrupt."""
        interrupt_handler.start_speaking(medium_event.priority)
        interrupt_handler.request_interrupt(
            interrupt_type=InterruptType.EVENT,
            event=critical_event
        )

        interrupt_handler.on_sentence_complete()
        pending = interrupt_handler.get_pending_interrupt()

        assert pending is None

    def test_on_sentence_complete_returns_none_without_pending(
        self,
        interrupt_handler,
        medium_event
    ):
        """on_sentence_complete should return None without pending interrupt."""
        interrupt_handler.start_speaking(medium_event.priority)

        pending = interrupt_handler.on_sentence_complete()

        assert pending is None

    def test_on_sentence_complete_only_returns_if_should_interrupt(
        self,
        interrupt_handler,
        high_event,
        medium_event
    ):
        """on_sentence_complete should only return if priority warrants interrupt."""
        interrupt_handler.start_speaking(high_event.priority)
        interrupt_handler.request_interrupt(
            interrupt_type=InterruptType.EVENT,
            event=medium_event  # Lower priority than current
        )

        # Should not return interrupt since MEDIUM doesn't interrupt HIGH
        pending = interrupt_handler.on_sentence_complete()

        # Pending should still be stored but not returned
        assert pending is None
        # But it should still be in the queue
        assert interrupt_handler.get_pending_interrupt() is not None


# =============================================================================
# RESPONSE CHUNKING FOR INTERRUPTS
# =============================================================================

class TestResponseChunking:
    """Tests for response chunking to support interrupts."""

    def test_chunk_response_into_sentences(self, interrupt_handler):
        """Should chunk response into sentences for interrupt points."""
        response = "Your tires are looking good. About ten laps remaining on them. No need to pit yet."

        chunks = interrupt_handler.chunk_response(response)

        assert len(chunks) == 3
        assert chunks[0] == "Your tires are looking good."
        assert chunks[1] == "About ten laps remaining on them."
        assert chunks[2] == "No need to pit yet."

    def test_chunk_short_response_single_chunk(self, interrupt_handler):
        """Short response should be single chunk."""
        response = "Box this lap."

        chunks = interrupt_handler.chunk_response(response)

        assert len(chunks) == 1

    def test_chunk_preserves_content(self, interrupt_handler):
        """Chunking should preserve all content."""
        response = "First. Second. Third."

        chunks = interrupt_handler.chunk_response(response)
        reconstructed = " ".join(chunks)

        assert reconstructed == response


# =============================================================================
# INTERRUPT FLOW
# =============================================================================

class TestInterruptFlow:
    """Tests for complete interrupt flow."""

    def test_full_interrupt_flow(
        self,
        interrupt_handler,
        medium_event,
        critical_event
    ):
        """Test full interrupt flow from request to completion."""
        # Start speaking at MEDIUM priority
        interrupt_handler.start_speaking(medium_event.priority)
        assert interrupt_handler.is_speaking is True

        # Request CRITICAL interrupt
        interrupt_handler.request_interrupt(
            interrupt_type=InterruptType.EVENT,
            event=critical_event
        )

        # Should have pending interrupt
        assert interrupt_handler.get_pending_interrupt() is not None

        # On sentence complete, should return interrupt
        pending = interrupt_handler.on_sentence_complete()
        assert pending is not None
        assert pending['event'].priority == Priority.CRITICAL

        # Pending should be cleared
        assert interrupt_handler.get_pending_interrupt() is None

    def test_driver_query_interrupt_flow(self, interrupt_handler, medium_event):
        """Test driver query interrupt flow."""
        # Start speaking at MEDIUM priority
        interrupt_handler.start_speaking(medium_event.priority)

        # Driver asks a question
        query = "What about fuel?"
        interrupt_handler.request_interrupt(
            interrupt_type=InterruptType.DRIVER_QUERY,
            query=query
        )

        # On sentence complete, should return query
        pending = interrupt_handler.on_sentence_complete()

        assert pending is not None
        assert pending['type'] == InterruptType.DRIVER_QUERY
        assert pending['query'] == query

    def test_no_interrupt_when_not_speaking(
        self,
        interrupt_handler,
        critical_event
    ):
        """Should accept interrupt even when not speaking."""
        # Request interrupt when not speaking
        result = interrupt_handler.request_interrupt(
            interrupt_type=InterruptType.EVENT,
            event=critical_event
        )

        assert result is True
        # Should be stored for immediate processing
        assert interrupt_handler.get_pending_interrupt() is not None


# =============================================================================
# STATISTICS AND STATE
# =============================================================================

class TestStatisticsAndState:
    """Tests for interrupt handler statistics and state."""

    def test_get_state(self, interrupt_handler, medium_event, critical_event):
        """get_state should return handler state."""
        interrupt_handler.start_speaking(medium_event.priority)
        interrupt_handler.request_interrupt(
            interrupt_type=InterruptType.EVENT,
            event=critical_event
        )

        state = interrupt_handler.get_state()

        assert state['is_speaking'] is True
        assert state['current_priority'] == Priority.MEDIUM
        assert state['has_pending_interrupt'] is True

    def test_get_state_when_idle(self, interrupt_handler):
        """get_state should reflect idle state."""
        state = interrupt_handler.get_state()

        assert state['is_speaking'] is False
        assert state['current_priority'] is None
        assert state['has_pending_interrupt'] is False

    def test_interrupt_count_tracking(self, interrupt_handler, critical_event):
        """Should track interrupt statistics."""
        interrupt_handler.start_speaking(Priority.MEDIUM)
        interrupt_handler.request_interrupt(
            interrupt_type=InterruptType.EVENT,
            event=critical_event
        )
        interrupt_handler.on_sentence_complete()

        stats = interrupt_handler.get_stats()

        assert stats['total_interrupts'] >= 1


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases in interrupt handling."""

    def test_multiple_interrupts_queued(
        self,
        interrupt_handler,
        high_event,
        critical_event
    ):
        """Multiple interrupts should keep highest priority."""
        interrupt_handler.start_speaking(Priority.LOW)

        interrupt_handler.request_interrupt(
            interrupt_type=InterruptType.EVENT,
            event=high_event
        )
        interrupt_handler.request_interrupt(
            interrupt_type=InterruptType.EVENT,
            event=critical_event
        )

        pending = interrupt_handler.get_pending_interrupt()

        assert pending['event'].priority == Priority.CRITICAL

    def test_clear_all_interrupts(self, interrupt_handler, critical_event):
        """Should be able to clear all pending interrupts."""
        interrupt_handler.request_interrupt(
            interrupt_type=InterruptType.EVENT,
            event=critical_event
        )

        interrupt_handler.clear_pending()

        assert interrupt_handler.get_pending_interrupt() is None

    def test_concurrent_event_and_query_interrupt(
        self,
        interrupt_handler,
        high_event
    ):
        """Event and query interrupts should be prioritized correctly."""
        interrupt_handler.start_speaking(Priority.MEDIUM)

        # Both HIGH priority event and driver query
        interrupt_handler.request_interrupt(
            interrupt_type=InterruptType.EVENT,
            event=high_event
        )
        interrupt_handler.request_interrupt(
            interrupt_type=InterruptType.DRIVER_QUERY,
            query="How's the gap?"
        )

        # Driver query should take precedence (more recent, same effective priority)
        pending = interrupt_handler.get_pending_interrupt()
        assert pending['type'] == InterruptType.DRIVER_QUERY

    def test_text_with_ellipsis(self, interrupt_handler):
        """Should handle text with ellipsis correctly."""
        text = "Your tires are looking good... about ten laps remaining."

        sentences = interrupt_handler.split_into_sentences(text)

        # Should be one or two sentences, not split on each dot
        assert len(sentences) <= 2

    def test_handles_text_without_punctuation(self, interrupt_handler):
        """Should handle text without sentence-ending punctuation."""
        text = "Box this lap"

        sentences = interrupt_handler.split_into_sentences(text)

        assert len(sentences) == 1
        assert sentences[0] == "Box this lap"
