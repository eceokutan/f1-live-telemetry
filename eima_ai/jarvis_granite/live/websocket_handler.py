"""
WebSocket Handler for Jarvis-Granite Live Telemetry.

Handles WebSocket connections, message routing, and session management.
Integrates TelemetryAgent for event detection and RaceEngineerAgent
for AI response generation.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from jarvis_granite.live.context import LiveSessionContext
from jarvis_granite.schemas.telemetry import TelemetryData
from jarvis_granite.schemas.events import Event, Priority
from jarvis_granite.schemas.messages import (
    parse_client_message,
    SessionInitMessage,
    TelemetryMessage,
    TextQueryMessage,
    ConfigUpdateMessage,
    SessionEndMessage,
)
from jarvis_granite.agents.telemetry_agent import TelemetryAgent
from jarvis_granite.agents.race_engineer_agent import RaceEngineerAgent
from jarvis_granite.llm import LLMClient, LLMError
from config.config import ThresholdsConfig, LiveConfig

logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    """Get current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _generate_response_id() -> str:
    """Generate a unique response ID."""
    import uuid
    return f"resp_{uuid.uuid4().hex[:12]}"


class WebSocketHandler:
    """
    Handles WebSocket connections and message routing.

    Manages active sessions, routes messages to appropriate handlers,
    and coordinates between the telemetry agent and race engineer agent.

    Attributes:
        active_sessions: Dictionary of active session data keyed by session_id
        telemetry_agent: Agent for rule-based telemetry processing
        race_engineer_agent: Agent for LLM-powered response generation
    """

    def __init__(
        self,
        thresholds: Optional[ThresholdsConfig] = None,
        llm_client: Optional[LLMClient] = None,
        config: Optional[LiveConfig] = None,
    ):
        """
        Initialize WebSocketHandler.

        Args:
            thresholds: Optional threshold configuration for telemetry agent
            llm_client: Optional LLM client for race engineer agent
            config: Optional live configuration
        """
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.telemetry_agent = TelemetryAgent(thresholds=thresholds)
        self.config = config or LiveConfig()

        # Initialize race engineer agent if LLM client provided
        if llm_client is not None:
            self.race_engineer_agent = RaceEngineerAgent(
                llm_client=llm_client,
                verbosity=self.config.verbosity.level,
            )
        else:
            self.race_engineer_agent = None
            logger.warning(
                "No LLM client provided - AI responses will be disabled"
            )

    def set_race_engineer_agent(self, agent: RaceEngineerAgent) -> None:
        """
        Set the race engineer agent.

        Args:
            agent: RaceEngineerAgent instance
        """
        self.race_engineer_agent = agent

    async def handle_connection(self, websocket) -> None:
        """
        Handle a new WebSocket connection.

        Accepts the connection and begins message processing loop.

        Args:
            websocket: FastAPI WebSocket instance
        """
        await websocket.accept()

        session_id: Optional[str] = None

        try:
            while True:
                # Receive and parse message
                try:
                    data = await websocket.receive_json()
                except json.JSONDecodeError:
                    await self._send_error(
                        websocket,
                        "VALIDATION_ERROR",
                        "Invalid JSON format"
                    )
                    continue

                # Route message
                result = await self.handle_message(data, websocket, session_id)

                # Update session_id if session was initialized
                if result and result.get("session_id"):
                    session_id = result["session_id"]

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            # Clean up session on disconnect
            if session_id and session_id in self.active_sessions:
                del self.active_sessions[session_id]

    async def handle_message(
        self,
        data: Dict[str, Any],
        websocket,
        session_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Route and handle an incoming message.

        Args:
            data: Parsed JSON message data
            websocket: WebSocket connection
            session_id: Current session ID (if any)

        Returns:
            Result dictionary with session_id if applicable
        """
        message_type = data.get("type")

        # Route based on message type
        if message_type == "session_init":
            return await self._handle_session_init(data, websocket)

        elif message_type == "telemetry":
            return await self._handle_telemetry(data, websocket, session_id)

        elif message_type == "text_query":
            return await self._handle_text_query(data, websocket, session_id)

        elif message_type == "config_update":
            return await self._handle_config_update(data, websocket, session_id)

        elif message_type == "session_end":
            return await self._handle_session_end(data, websocket, session_id)

        else:
            await self._send_error(
                websocket,
                "VALIDATION_ERROR",
                f"Unknown message type: {message_type}"
            )
            return None

    async def _handle_session_init(
        self,
        data: Dict[str, Any],
        websocket
    ) -> Optional[Dict[str, Any]]:
        """Handle session initialization message."""
        try:
            # Validate message
            message = SessionInitMessage(**data)
        except ValidationError as e:
            await self._send_error(
                websocket,
                "VALIDATION_ERROR",
                f"Invalid session_init: {str(e)}"
            )
            return None

        # Check for duplicate session
        if message.session_id in self.active_sessions:
            await self._send_error(
                websocket,
                "VALIDATION_ERROR",
                f"Session {message.session_id} already exists"
            )
            return None

        # Create session context
        context = LiveSessionContext(
            session_id=message.session_id,
            source=message.source,
            track_name=message.track_name
        )

        # Update verbosity if provided in config
        verbosity = message.config.get("verbosity", self.config.verbosity.level)
        if self.race_engineer_agent:
            self.race_engineer_agent.set_verbosity(verbosity)

        # Store session
        self.active_sessions[message.session_id] = {
            "context": context,
            "config": message.config.copy(),
            "websocket": websocket,
            "verbosity": verbosity,
        }

        # Generate LiveKit token (placeholder - real implementation would use LiveKit SDK)
        livekit_token = self._generate_livekit_token(message.session_id)

        # Send session confirmed response
        response = {
            "type": "session_confirmed",
            "session_id": message.session_id,
            "config": message.config,
            "livekit": {
                "url": self.config.livekit.url or "wss://livekit.example.com",
                "token": livekit_token,
                "room_name": f"{message.session_id}_voice"
            }
        }

        await websocket.send_json(response)
        logger.info(f"Session {message.session_id} initialized for {message.track_name}")

        return {"session_id": message.session_id}

    async def _handle_telemetry(
        self,
        data: Dict[str, Any],
        websocket,
        session_id: Optional[str]
    ) -> None:
        """Handle telemetry message and generate AI responses for events."""
        # Check session exists
        if not session_id or session_id not in self.active_sessions:
            await self._send_error(
                websocket,
                "SESSION_NOT_FOUND",
                "No active session. Send session_init first."
            )
            return None

        try:
            # Validate message
            message = TelemetryMessage(**data)
        except ValidationError as e:
            await self._send_error(
                websocket,
                "VALIDATION_ERROR",
                f"Invalid telemetry: {str(e)}"
            )
            return None

        session = self.active_sessions[session_id]
        context = session["context"]

        # Update context with telemetry
        context.update(message.data)

        # Detect events using telemetry agent
        events = self.telemetry_agent.detect_events(message.data, context)

        # Process detected events
        if events and self.race_engineer_agent:
            # Process highest priority event only to avoid message spam
            await self._process_events(events, context, websocket)

        return None

    async def _process_events(
        self,
        events: List[Event],
        context: LiveSessionContext,
        websocket
    ) -> None:
        """
        Process detected events and generate AI responses.

        Only processes the highest priority event to avoid message spam.

        Args:
            events: List of detected events
            context: Session context
            websocket: WebSocket connection
        """
        if not events or not self.race_engineer_agent:
            return

        # Sort by priority (lower value = higher priority)
        events.sort(key=lambda e: e.priority)

        # Get highest priority event
        event = events[0]

        # Check proactive message rate limiting
        min_interval = self.config.min_proactive_interval_seconds
        if not context.can_send_proactive(min_interval):
            # Skip if we recently sent a message (unless critical)
            if event.priority != Priority.CRITICAL:
                logger.debug(f"Skipping {event.type} - rate limited")
                return

        start_time = time.time()

        try:
            # Generate response
            response_text = await self.race_engineer_agent.generate_proactive_response(
                event=event,
                context=context,
            )

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)

            # Send AI response
            ai_response = {
                "type": "ai_response",
                "response_id": _generate_response_id(),
                "timestamp": _utcnow_iso(),
                "trigger": event.type,
                "text": response_text,
                "priority": event.priority.name.lower(),
                "metadata": {
                    "latency_ms": latency_ms,
                    "tokens_used": None,  # Would be populated by actual LLM
                }
            }

            await websocket.send_json(ai_response)

            # Mark proactive message sent
            context.mark_proactive_sent()

            # Add to conversation history
            context.add_exchange(f"[Event: {event.type}]", response_text)

            logger.info(
                f"AI response for {event.type} sent in {latency_ms}ms"
            )

        except LLMError as e:
            logger.error(f"LLM error for {event.type}: {e}")
            await self._send_error(
                websocket,
                "LLM_ERROR",
                f"Failed to generate response: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error processing event {event.type}: {e}")

    async def _handle_text_query(
        self,
        data: Dict[str, Any],
        websocket,
        session_id: Optional[str]
    ) -> None:
        """Handle text query message and generate AI response."""
        # Check session exists
        if not session_id or session_id not in self.active_sessions:
            await self._send_error(
                websocket,
                "SESSION_NOT_FOUND",
                "No active session. Send session_init first."
            )
            return None

        try:
            # Validate message
            message = TextQueryMessage(**data)
        except ValidationError as e:
            await self._send_error(
                websocket,
                "VALIDATION_ERROR",
                f"Invalid text_query: {str(e)}"
            )
            return None

        session = self.active_sessions[session_id]
        context = session["context"]

        # Check if race engineer agent is available
        if not self.race_engineer_agent:
            await self._send_error(
                websocket,
                "LLM_ERROR",
                "AI responses not available - no LLM configured"
            )
            return None

        start_time = time.time()

        try:
            # Generate response
            response_text = await self.race_engineer_agent.generate_reactive_response(
                query=message.query,
                context=context,
            )

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)

            # Send AI response
            ai_response = {
                "type": "ai_response",
                "response_id": _generate_response_id(),
                "timestamp": _utcnow_iso(),
                "trigger": "text_query",
                "text": response_text,
                "priority": "high",  # Driver queries are high priority
                "metadata": {
                    "latency_ms": latency_ms,
                    "tokens_used": None,
                }
            }

            await websocket.send_json(ai_response)

            logger.info(
                f"AI response for query '{message.query[:30]}...' sent in {latency_ms}ms"
            )

        except LLMError as e:
            logger.error(f"LLM error for query: {e}")
            await self._send_error(
                websocket,
                "LLM_ERROR",
                f"Failed to generate response: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            await self._send_error(
                websocket,
                "LLM_ERROR",
                f"Unexpected error: {str(e)}"
            )

        return None

    async def _handle_config_update(
        self,
        data: Dict[str, Any],
        websocket,
        session_id: Optional[str]
    ) -> None:
        """Handle configuration update message."""
        # Check session exists
        if not session_id or session_id not in self.active_sessions:
            await self._send_error(
                websocket,
                "SESSION_NOT_FOUND",
                "No active session. Send session_init first."
            )
            return None

        try:
            # Validate message
            message = ConfigUpdateMessage(**data)
        except ValidationError as e:
            await self._send_error(
                websocket,
                "VALIDATION_ERROR",
                f"Invalid config_update: {str(e)}"
            )
            return None

        session = self.active_sessions[session_id]

        # Update session config
        session["config"].update(message.config)

        # Update verbosity if changed
        if "verbosity" in message.config and self.race_engineer_agent:
            new_verbosity = message.config["verbosity"]
            self.race_engineer_agent.set_verbosity(new_verbosity)
            session["verbosity"] = new_verbosity
            logger.info(f"Updated verbosity to {new_verbosity}")

        return None

    async def _handle_session_end(
        self,
        data: Dict[str, Any],
        websocket,
        session_id: Optional[str]
    ) -> None:
        """Handle session end message."""
        if session_id and session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Session {session_id} ended")

        return None

    async def send_heartbeat(self, session_id: str, websocket) -> None:
        """
        Send heartbeat message to client.

        Args:
            session_id: Session to send heartbeat for
            websocket: WebSocket connection
        """
        is_active = session_id in self.active_sessions

        heartbeat = {
            "type": "heartbeat",
            "timestamp": _utcnow_iso(),
            "session_active": is_active
        }

        await websocket.send_json(heartbeat)

    async def _send_error(
        self,
        websocket,
        error_code: str,
        message: str
    ) -> None:
        """Send error message to client."""
        error = {
            "type": "error",
            "error_code": error_code,
            "message": message,
            "timestamp": _utcnow_iso()
        }

        await websocket.send_json(error)
        logger.warning(f"Sent error {error_code}: {message}")

    def _generate_livekit_token(self, session_id: str) -> str:
        """
        Generate LiveKit access token.

        In production, this would use the LiveKit SDK to generate
        a proper JWT token. For now, returns a placeholder.

        Args:
            session_id: Session ID for the token

        Returns:
            JWT token string
        """
        # Placeholder - real implementation would use:
        # from livekit import api
        # token = api.AccessToken(api_key, api_secret)
        # token.add_grant(api.VideoGrant(room_join=True, room=f"{session_id}_voice"))
        # return token.to_jwt()

        return f"placeholder_token_{session_id}_{int(time.time())}"
