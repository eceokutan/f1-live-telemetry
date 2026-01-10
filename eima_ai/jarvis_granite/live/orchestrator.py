"""
Custom Orchestrator Core for Jarvis-Granite Live Telemetry.

Lightweight orchestrator (~200-300 LOC) for real-time race engineering.
Handles event routing, priority queue management, and interrupt handling.

Phase 4, Section 9: Orchestrator Core
- JarvisLiveOrchestrator class
- Event router connecting Telemetry Agent -> Race Engineer Agent
- Queue processing logic with priority handling
- Interrupt handling for high-priority events

Phase 4, Section 10: Interrupt Handler Integration
- InterruptHandler for sentence-level interrupt management
- Driver query interrupt handling
- Response chunking for graceful interrupts

Design principles:
- Minimal framework overhead for <2s latency
- Racing-specific priority queue (Critical > High > Medium > Low)
- Interrupt handling for graceful speech completion
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from jarvis_granite.schemas.events import Event, Priority
from jarvis_granite.schemas.telemetry import TelemetryData
from jarvis_granite.live.context import LiveSessionContext
from jarvis_granite.live.priority_queue import PriorityQueue
from jarvis_granite.live.interrupt_handler import InterruptHandler, InterruptType
from jarvis_granite.agents.telemetry_agent import TelemetryAgent
from jarvis_granite.agents.race_engineer_agent import RaceEngineerAgent
from config.config import LiveConfig

logger = logging.getLogger(__name__)


class JarvisLiveOrchestrator:
    """
    Custom lightweight orchestrator for real-time race engineering.

    ~200-300 LOC focused on:
    - Event routing from telemetry to AI response
    - Priority queue management
    - Interrupt handling for critical events

    Pipeline: Telemetry -> TelemetryAgent (events) -> Queue -> RaceEngineerAgent (response)

    Attributes:
        config: Live configuration settings
        event_queue: Priority-based event queue
        is_speaking: Whether AI is currently outputting audio
        pending_interrupt: Event waiting to interrupt current speech
        current_priority: Priority of currently processing event
        session_context: Current racing session state
        telemetry_agent: Rule-based event detection
        race_engineer_agent: LLM-powered response generation

    Example:
        orchestrator = JarvisLiveOrchestrator(
            telemetry_agent=telemetry_agent,
            race_engineer_agent=race_engineer_agent
        )
        orchestrator.set_session_context(session_context)

        # Process telemetry
        responses = await orchestrator.handle_telemetry(telemetry_data)

        # Handle driver query
        response = await orchestrator.handle_driver_query("How's my fuel?")
    """

    def __init__(
        self,
        config: Optional[LiveConfig] = None,
        telemetry_agent: Optional[TelemetryAgent] = None,
        race_engineer_agent: Optional[RaceEngineerAgent] = None,
        interrupt_handler: Optional[InterruptHandler] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            config: Live configuration. Uses defaults if not provided.
            telemetry_agent: Agent for rule-based event detection.
            race_engineer_agent: Agent for LLM response generation.
            interrupt_handler: Handler for interrupt management (created if not provided).
        """
        self.config = config or LiveConfig()

        # Priority queue for event management
        queue_size = self.config.orchestrator.priority_queue_max_size
        self.event_queue = PriorityQueue(max_size=queue_size)

        # Interrupt handler for sentence-level interrupt management
        self.interrupt_handler = interrupt_handler or InterruptHandler()

        # Speaking state for interrupt handling (delegated to interrupt_handler but kept for compatibility)
        self.is_speaking: bool = False
        self.pending_interrupt: Optional[Event] = None
        self.current_priority: Priority = Priority.LOW

        # Session context (set externally)
        self.session_context: Optional[LiveSessionContext] = None

        # Agent references
        self.telemetry_agent = telemetry_agent
        self.race_engineer_agent = race_engineer_agent

        # Callback for voice agent integration
        self.on_response_generated: Optional[Callable[[str, Priority], None]] = None

        logger.info(
            f"Initialized JarvisLiveOrchestrator with queue_size={queue_size}"
        )

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    def set_session_context(self, context: LiveSessionContext) -> None:
        """
        Set the session context for the orchestrator.

        Args:
            context: LiveSessionContext instance
        """
        self.session_context = context
        logger.info(f"Session context set for session {context.session_id}")

    def get_session_context(self) -> Optional[LiveSessionContext]:
        """
        Get the current session context.

        Returns:
            Current session context or None
        """
        return self.session_context

    def _ensure_session_context(self) -> LiveSessionContext:
        """
        Ensure session context is set.

        Returns:
            Session context

        Raises:
            ValueError: If session context is not set
        """
        if self.session_context is None:
            raise ValueError("Session context not set")
        return self.session_context

    # =========================================================================
    # EVENT QUEUEING
    # =========================================================================

    async def queue_event(self, event: Event) -> bool:
        """
        Add an event to the priority queue.

        Events are ordered by priority (CRITICAL > HIGH > MEDIUM > LOW)
        and within the same priority by timestamp (FIFO).

        Args:
            event: Event to queue

        Returns:
            True if event was queued, False if skipped
        """
        result = self.event_queue.push(event)

        if result:
            logger.debug(
                f"Queued event {event.type} with priority {event.priority.name}"
            )
        else:
            logger.debug(
                f"Skipped event {event.type} - queue at capacity"
            )

        return result

    # =========================================================================
    # TELEMETRY HANDLING
    # =========================================================================

    async def handle_telemetry(
        self,
        telemetry: TelemetryData,
        process_queue: bool = True
    ) -> List[str]:
        """
        Process incoming telemetry and generate AI responses.

        This is the main entry point for the telemetry pipeline:
        1. Update session context with telemetry
        2. Detect events via TelemetryAgent
        3. Queue detected events
        4. Process queue and generate responses

        Args:
            telemetry: Incoming telemetry data
            process_queue: Whether to process queued events (default True)

        Returns:
            List of AI response strings

        Raises:
            ValueError: If session context is not set
        """
        context = self._ensure_session_context()

        # Update session context
        context.update(telemetry)

        # Detect events via telemetry agent
        events = []
        if self.telemetry_agent:
            events = self.telemetry_agent.detect_events(telemetry, context)
            logger.debug(f"Detected {len(events)} events from telemetry")

        # Queue detected events
        for event in events:
            await self.queue_event(event)

        # Process queue if requested
        if process_queue and events:
            return await self.process_event_queue()

        return []

    # =========================================================================
    # EVENT PROCESSING
    # =========================================================================

    async def process_event(self, event: Event) -> Optional[str]:
        """
        Process a single event and generate AI response.

        Args:
            event: Event to process

        Returns:
            AI response string or None if no response generated
        """
        context = self._ensure_session_context()

        if not self.race_engineer_agent:
            logger.warning("No race engineer agent - skipping response generation")
            return None

        # Set speaking state
        self.is_speaking = True
        self.current_priority = event.priority

        try:
            # Generate response via race engineer agent
            response = await self.race_engineer_agent.handle_event(event, context)

            if response:
                logger.info(
                    f"Generated response for {event.type}: {len(response)} chars"
                )

                # Notify callback if registered
                if self.on_response_generated:
                    self.on_response_generated(response, event.priority)

            return response

        except Exception as e:
            logger.error(f"Error processing event {event.type}: {e}")
            return None

        finally:
            # Clear speaking state
            self.is_speaking = False

    async def process_event_queue(self) -> List[str]:
        """
        Process all events in the queue and generate responses.

        Events are processed in priority order. Processing stops
        if an interrupt is triggered by a higher priority event.

        Returns:
            List of AI response strings
        """
        responses = []

        while not self.event_queue.is_empty():
            # Check for interrupts
            if self.pending_interrupt:
                break

            # Pop and process next event
            event = self.event_queue.pop()
            if event:
                response = await self.process_event(event)
                if response:
                    responses.append(response)

        return responses

    # =========================================================================
    # DRIVER QUERY HANDLING
    # =========================================================================

    async def handle_driver_query(self, query: str) -> str:
        """
        Handle a driver voice query (reactive mode).

        Driver queries are treated as HIGH priority and processed
        immediately, potentially interrupting lower priority responses.

        Args:
            query: Driver's question or command

        Returns:
            AI response string

        Raises:
            ValueError: If session context is not set
        """
        context = self._ensure_session_context()

        if not self.race_engineer_agent:
            logger.warning("No race engineer agent - cannot respond to query")
            return "Sorry, I'm unable to respond at the moment."

        logger.info(f"Handling driver query: {query[:50]}...")

        # Generate response via race engineer agent
        response = await self.race_engineer_agent.generate_reactive_response(
            query, context
        )

        return response

    # =========================================================================
    # INTERRUPT HANDLING
    # =========================================================================

    async def check_for_interrupt(self) -> bool:
        """
        Check if current speech should be interrupted.

        A response should be interrupted if:
        - There is a CRITICAL event (always interrupts)
        - There is a HIGH event and current priority is MEDIUM or LOW

        Returns:
            True if interrupt should occur
        """
        if not self.is_speaking:
            return False

        # Check if queue has higher priority event
        if self.event_queue.should_interrupt(self.current_priority):
            # Set pending interrupt
            self.pending_interrupt = self.event_queue.pop()
            logger.info(
                f"Interrupt triggered by {self.pending_interrupt.type} "
                f"({self.pending_interrupt.priority.name})"
            )
            return True

        return False

    async def on_sentence_complete(self) -> Optional[str]:
        """
        Callback when a sentence is completed during speech.

        If there is a pending interrupt, it is processed after
        the current sentence completes (graceful interruption).

        Returns:
            Response for the interrupt event, or None
        """
        if self.pending_interrupt is None:
            return None

        event = self.pending_interrupt
        self.pending_interrupt = None

        logger.info(f"Processing pending interrupt: {event.type}")

        return await self.process_event(event)

    # =========================================================================
    # SPEAKING STATE MANAGEMENT
    # =========================================================================

    def set_speaking(self, is_speaking: bool) -> None:
        """
        Set the speaking state.

        Args:
            is_speaking: Whether AI is currently speaking
        """
        self.is_speaking = is_speaking

        if not is_speaking:
            # Clear current priority when done speaking
            self.current_priority = Priority.LOW

    # =========================================================================
    # INTERRUPT-AWARE RESPONSE DELIVERY
    # =========================================================================

    async def deliver_response_with_interrupts(
        self,
        response: str,
        priority: Priority,
        on_sentence: Optional[Callable[[str], None]] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Deliver a response with interrupt support at sentence boundaries.

        Chunks the response into sentences and checks for interrupts
        after each sentence. If an interrupt occurs, remaining sentences
        are skipped and the interrupt is returned.

        Args:
            response: Full AI response text
            priority: Priority of this response
            on_sentence: Callback to invoke for each sentence (e.g., TTS)

        Returns:
            Tuple of (completed, interrupt_info):
            - completed: True if full response was delivered
            - interrupt_info: Interrupt data if interrupted, None otherwise
        """
        # Start speaking at given priority
        self.interrupt_handler.start_speaking(priority)
        self.is_speaking = True
        self.current_priority = priority

        try:
            # Chunk response into sentences
            sentences = self.interrupt_handler.chunk_response(response)

            for sentence in sentences:
                # Deliver sentence (e.g., to TTS)
                if on_sentence:
                    on_sentence(sentence)

                # Check for pending interrupt after each sentence
                interrupt = self.interrupt_handler.on_sentence_complete()
                if interrupt:
                    logger.info(f"Response interrupted after: {sentence[:30]}...")
                    return False, interrupt

            # Full response delivered
            return True, None

        finally:
            # Clear speaking state
            self.interrupt_handler.stop_speaking()
            self.is_speaking = False
            self.current_priority = Priority.LOW

    async def request_event_interrupt(self, event: Event) -> bool:
        """
        Request an interrupt for a telemetry event.

        Args:
            event: Event that should interrupt current speech

        Returns:
            True if interrupt was accepted
        """
        return self.interrupt_handler.request_interrupt(
            interrupt_type=InterruptType.EVENT,
            event=event
        )

    async def request_query_interrupt(self, query: str) -> bool:
        """
        Request an interrupt for a driver query.

        Args:
            query: Driver's question

        Returns:
            True if interrupt was accepted
        """
        return self.interrupt_handler.request_interrupt(
            interrupt_type=InterruptType.DRIVER_QUERY,
            query=query
        )

    # =========================================================================
    # STATISTICS AND STATE
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get orchestrator statistics and state.

        Returns:
            Dictionary with orchestrator state information
        """
        session_id = None
        if self.session_context:
            session_id = self.session_context.session_id

        return {
            "is_speaking": self.is_speaking,
            "current_priority": self.current_priority.name if self.is_speaking else None,
            "pending_interrupt": self.pending_interrupt.type if self.pending_interrupt else None,
            "queue_size": self.event_queue.size(),
            "queue_stats": self.event_queue.get_queue_stats(),
            "session_id": session_id,
            "has_telemetry_agent": self.telemetry_agent is not None,
            "has_race_engineer_agent": self.race_engineer_agent is not None,
            "interrupt_handler_state": self.interrupt_handler.get_state(),
            "interrupt_stats": self.interrupt_handler.get_stats(),
        }

    def __repr__(self) -> str:
        """String representation of orchestrator state."""
        stats = self.get_stats()
        return (
            f"JarvisLiveOrchestrator("
            f"session={stats['session_id']}, "
            f"queue={stats['queue_size']}, "
            f"speaking={stats['is_speaking']})"
        )
