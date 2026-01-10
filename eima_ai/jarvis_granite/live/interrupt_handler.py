"""
Interrupt Handler for Jarvis-Granite Live Telemetry.

Manages graceful interruption of AI speech for higher-priority events
and driver queries. Implements sentence-based interruption to ensure
natural speech completion before responding to interrupts.

Phase 4, Section 10: Interrupt Handler
- Sentence completion detection
- Pending interrupt queue with priority management
- Priority-based interrupt logic (Critical/High interrupts Medium/Low)
- Driver query handling

Interrupt Behavior:
    AI Speaking: "Your tires are looking good, about ten laps—"
    Driver Speaks: "What about fuel?"
                        │
                        ▼
                AI finishes sentence: "—remaining on them."
                        │
                        ▼
                AI responds to query: "Fuel is at 45 liters, roughly 8 laps at current pace."
"""

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from jarvis_granite.schemas.events import Event, Priority

logger = logging.getLogger(__name__)


class InterruptType(Enum):
    """Types of interrupts that can occur."""
    EVENT = auto()        # Telemetry event (fuel critical, tire warning, etc.)
    DRIVER_QUERY = auto() # Driver voice/text query


@dataclass
class PendingInterrupt:
    """
    Represents a pending interrupt waiting to be processed.

    Attributes:
        interrupt_type: Type of interrupt (EVENT or DRIVER_QUERY)
        priority: Effective priority for interrupt comparison
        event: Event object if type is EVENT
        query: Query string if type is DRIVER_QUERY
        timestamp: When the interrupt was requested
    """
    interrupt_type: InterruptType
    priority: Priority
    event: Optional[Event] = None
    query: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'type': self.interrupt_type,
            'priority': self.priority,
            'event': self.event,
            'query': self.query,
            'timestamp': self.timestamp,
        }


class InterruptHandler:
    """
    Manages graceful interruption of AI speech.

    Handles:
    - Tracking speaking state and current priority
    - Queueing pending interrupts
    - Priority-based interrupt decisions
    - Sentence splitting for natural interrupt points

    Priority Rules:
    - CRITICAL: Always interrupts (brake failure, collision warning)
    - HIGH: Interrupts MEDIUM and LOW (pit now, fuel critical)
    - MEDIUM: Does not interrupt (normal updates)
    - LOW: Never interrupts (sector times)

    Example:
        handler = InterruptHandler()

        # Start speaking at MEDIUM priority
        handler.start_speaking(Priority.MEDIUM)

        # Process response in chunks
        for sentence in handler.chunk_response(ai_response):
            # Speak sentence...

            # Check for pending interrupt after each sentence
            interrupt = handler.on_sentence_complete()
            if interrupt:
                # Handle interrupt
                break

        handler.stop_speaking()
    """

    # Sentence splitting pattern - matches sentence-ending punctuation
    # Handles common abbreviations and decimal numbers
    SENTENCE_PATTERN = re.compile(
        r'(?<!\d)(?<![A-Z])(?<!\s[A-Z])(?<![A-Z]\.)(?<!\.\.)(?<!\.\.\.)[\.\!\?]+(?=\s+[A-Z]|\s*$)'
    )

    # Simpler fallback pattern for basic sentence splitting
    SIMPLE_SENTENCE_PATTERN = re.compile(r'(?<=[.!?])\s+')

    # Driver query effective priority (treated as HIGH)
    DRIVER_QUERY_PRIORITY = Priority.HIGH

    def __init__(self):
        """Initialize the InterruptHandler."""
        self._is_speaking: bool = False
        self._current_priority: Optional[Priority] = None
        self._pending_interrupt: Optional[PendingInterrupt] = None

        # Statistics
        self._total_interrupts: int = 0
        self._interrupts_by_type: Dict[InterruptType, int] = {
            InterruptType.EVENT: 0,
            InterruptType.DRIVER_QUERY: 0,
        }

        logger.info("InterruptHandler initialized")

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def is_speaking(self) -> bool:
        """Whether the AI is currently speaking."""
        return self._is_speaking

    @property
    def current_priority(self) -> Optional[Priority]:
        """Priority of current speech, or None if not speaking."""
        return self._current_priority

    # =========================================================================
    # SPEAKING STATE MANAGEMENT
    # =========================================================================

    def start_speaking(self, priority: Priority) -> None:
        """
        Mark the start of AI speech at a given priority.

        Args:
            priority: Priority level of the response being spoken
        """
        self._is_speaking = True
        self._current_priority = priority
        logger.debug(f"Started speaking at {priority.name} priority")

    def stop_speaking(self) -> None:
        """Mark the end of AI speech."""
        self._is_speaking = False
        self._current_priority = None
        logger.debug("Stopped speaking")

    # =========================================================================
    # SENTENCE COMPLETION
    # =========================================================================

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for interrupt handling.

        Splits on sentence-ending punctuation (., !, ?) while
        attempting to handle abbreviations and decimal numbers.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        if not text or not text.strip():
            return []

        text = text.strip()

        # Try simple pattern first (most reliable)
        sentences = self.SIMPLE_SENTENCE_PATTERN.split(text)

        # Filter empty strings and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]

        # If no split occurred, return original text as single sentence
        if not sentences:
            return [text]

        return sentences

    def chunk_response(self, response: str) -> List[str]:
        """
        Chunk a response into sentences for interrupt-aware delivery.

        Each chunk represents a natural break point where an
        interrupt can be processed.

        Args:
            response: Full AI response text

        Returns:
            List of sentence chunks
        """
        return self.split_into_sentences(response)

    def on_sentence_complete(self) -> Optional[Dict[str, Any]]:
        """
        Callback when a sentence is completed during speech.

        If there is a pending interrupt with sufficient priority,
        it is returned and cleared. Otherwise returns None.

        Returns:
            Pending interrupt dict if should interrupt, None otherwise
        """
        if self._pending_interrupt is None:
            return None

        # Check if pending interrupt should actually interrupt
        if not self.should_interrupt(self._pending_interrupt.priority):
            # Keep pending but don't return it
            logger.debug(
                f"Pending interrupt {self._pending_interrupt.priority.name} "
                f"does not interrupt {self._current_priority.name}"
            )
            return None

        # Get and clear pending interrupt
        interrupt = self._pending_interrupt
        self._pending_interrupt = None

        # Update statistics
        self._total_interrupts += 1
        self._interrupts_by_type[interrupt.interrupt_type] += 1

        logger.info(
            f"Processing interrupt: {interrupt.interrupt_type.name} "
            f"at {interrupt.priority.name} priority"
        )

        return interrupt.to_dict()

    # =========================================================================
    # INTERRUPT REQUESTS
    # =========================================================================

    def request_interrupt(
        self,
        interrupt_type: InterruptType,
        event: Optional[Event] = None,
        query: Optional[str] = None,
    ) -> bool:
        """
        Request an interrupt for an event or driver query.

        The interrupt is queued as pending and will be processed
        at the next sentence boundary if priority warrants.

        Args:
            interrupt_type: Type of interrupt (EVENT or DRIVER_QUERY)
            event: Event object if type is EVENT
            query: Query string if type is DRIVER_QUERY

        Returns:
            True if interrupt was accepted
        """
        # Determine effective priority
        if interrupt_type == InterruptType.EVENT and event:
            priority = event.priority
        elif interrupt_type == InterruptType.DRIVER_QUERY:
            priority = self.DRIVER_QUERY_PRIORITY
        else:
            logger.warning("Invalid interrupt request - missing event or query")
            return False

        # Check if we should replace existing pending interrupt
        if self._pending_interrupt is not None:
            if priority >= self._pending_interrupt.priority:
                # Same or lower priority - check if it's a query (takes precedence)
                if interrupt_type != InterruptType.DRIVER_QUERY:
                    logger.debug(
                        f"Keeping existing {self._pending_interrupt.priority.name} "
                        f"interrupt over {priority.name}"
                    )
                    return True

        # Create pending interrupt
        self._pending_interrupt = PendingInterrupt(
            interrupt_type=interrupt_type,
            priority=priority,
            event=event,
            query=query,
        )

        logger.info(
            f"Interrupt requested: {interrupt_type.name} "
            f"at {priority.name} priority"
        )

        return True

    def get_pending_interrupt(self) -> Optional[Dict[str, Any]]:
        """
        Get the current pending interrupt without clearing it.

        Returns:
            Pending interrupt dict or None
        """
        if self._pending_interrupt is None:
            return None
        return self._pending_interrupt.to_dict()

    def clear_pending(self) -> None:
        """Clear any pending interrupt."""
        self._pending_interrupt = None
        logger.debug("Cleared pending interrupt")

    # =========================================================================
    # INTERRUPT LOGIC
    # =========================================================================

    def should_interrupt(self, interrupt_priority: Priority) -> bool:
        """
        Determine if an interrupt should occur based on priorities.

        Rules:
        - CRITICAL: Always interrupts
        - HIGH: Interrupts MEDIUM and LOW
        - MEDIUM: Does not interrupt
        - LOW: Never interrupts

        Args:
            interrupt_priority: Priority of the interrupting event

        Returns:
            True if interrupt should occur
        """
        if not self._is_speaking or self._current_priority is None:
            # Not speaking, interrupt can proceed immediately
            return True

        # CRITICAL always interrupts
        if interrupt_priority == Priority.CRITICAL:
            return True

        # HIGH interrupts MEDIUM and LOW
        if interrupt_priority == Priority.HIGH:
            return self._current_priority > Priority.HIGH

        # MEDIUM and LOW never interrupt
        return False

    def should_interrupt_for_query(self) -> bool:
        """
        Determine if a driver query should interrupt current speech.

        Driver queries are treated as HIGH priority.

        Returns:
            True if query should interrupt
        """
        return self.should_interrupt(self.DRIVER_QUERY_PRIORITY)

    # =========================================================================
    # STATISTICS AND STATE
    # =========================================================================

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the interrupt handler.

        Returns:
            Dictionary with current state information
        """
        return {
            'is_speaking': self._is_speaking,
            'current_priority': self._current_priority,
            'has_pending_interrupt': self._pending_interrupt is not None,
            'pending_priority': (
                self._pending_interrupt.priority
                if self._pending_interrupt else None
            ),
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Get interrupt statistics.

        Returns:
            Dictionary with interrupt statistics
        """
        return {
            'total_interrupts': self._total_interrupts,
            'event_interrupts': self._interrupts_by_type[InterruptType.EVENT],
            'query_interrupts': self._interrupts_by_type[InterruptType.DRIVER_QUERY],
        }

    def __repr__(self) -> str:
        """String representation of handler state."""
        state = "speaking" if self._is_speaking else "idle"
        pending = "pending" if self._pending_interrupt else "no pending"
        return f"InterruptHandler({state}, {pending})"
