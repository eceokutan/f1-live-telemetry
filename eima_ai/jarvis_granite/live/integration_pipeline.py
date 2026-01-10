"""
Integration Pipeline for Jarvis-Granite Live Telemetry.

Wires all components together for end-to-end integration:
- Telemetry Agent: Rule-based event detection
- Orchestrator: Event routing and priority queue
- Race Engineer Agent: LLM-powered response generation
- Voice Pipeline: STT -> Orchestrator -> TTS flow

Phase 6, Section 14: End-to-End Integration
- Wire all components together
- Implement session lifecycle (init -> active -> end)
- Add LiveKit token to session_confirmed response
- Full pipeline: Telemetry -> Event -> LLM -> Voice

Phase 6, Section 16: Performance Optimization
- Profile end-to-end latency (target: <3000ms)
- Optimize bottlenecks
- Add metrics logging

Performance Targets:
- Telemetry processing: <50ms (rule-based, no LLM)
- LLM response: <2000ms (Granite generation via LangChain)
- TTS conversion: <500ms (Watson TTS with Tenacity retry)
- LiveKit transport: <100ms (WebRTC jitter buffering)
- End-to-end (voice): <3000ms (Total pipeline latency)

Example:
    pipeline = IntegrationPipeline(config=live_config, llm_client=llm_client)

    # Initialize session
    session_info = await pipeline.init_session(
        session_id="race_001",
        source="torcs",
        track_name="Monza"
    )

    # Process telemetry
    result = await pipeline.process_telemetry("race_001", telemetry_data)

    # Handle driver query
    response = await pipeline.handle_driver_query("race_001", "How are my tires?")

    # Get performance report
    report = pipeline.get_performance_report()

    # End session
    await pipeline.end_session("race_001")
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from jarvis_granite.live.context import LiveSessionContext
from jarvis_granite.live.orchestrator import JarvisLiveOrchestrator
from jarvis_granite.live.errors import (
    ErrorCode,
    JarvisError,
    LLMTimeoutError,
    TTSError,
    SessionNotFoundError,
    ValidationError,
    create_error_response,
)
from jarvis_granite.live.performance import (
    PerformanceMetrics,
    profile_latency,
    profile_latency_async,
)
from jarvis_granite.agents.telemetry_agent import TelemetryAgent
from jarvis_granite.agents.race_engineer_agent import RaceEngineerAgent
from jarvis_granite.schemas.telemetry import TelemetryData
from jarvis_granite.schemas.events import Event, Priority
from jarvis_granite.voice.voice_pipeline import VoicePipeline
from config.config import LiveConfig

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Information about an active session."""
    session_id: str
    context: LiveSessionContext
    config: Dict[str, Any]
    created_at: datetime
    stats: Dict[str, Any] = field(default_factory=lambda: {
        "telemetry_count": 0,
        "events_detected": 0,
        "ai_responses": 0,
        "queries_handled": 0,
    })


class IntegrationPipeline:
    """
    End-to-end integration pipeline for Jarvis-Granite Live Telemetry.

    Wires together all components for complete telemetry-to-voice flow:
    1. Telemetry Agent - Rule-based event detection
    2. Orchestrator - Event routing with priority queue
    3. Race Engineer Agent - LLM-powered response generation
    4. Voice Pipeline - STT/TTS processing

    Features:
    - Session lifecycle management (init -> active -> end)
    - LiveKit token generation for voice communication
    - Full pipeline: Telemetry -> Event -> LLM -> Voice
    - Interrupt handling for high-priority events
    - Statistics tracking and health monitoring

    Attributes:
        config: Live configuration
        orchestrator: Event routing and priority management
        telemetry_agent: Rule-based telemetry processing
        race_engineer_agent: LLM-powered response generation
        voice_pipeline: Voice processing pipeline
        session_manager: Active session management

    Example:
        pipeline = IntegrationPipeline(config=live_config)
        session = await pipeline.init_session("race_001", "torcs", "Monza")
        result = await pipeline.process_telemetry("race_001", telemetry)
    """

    def __init__(
        self,
        config: Optional[LiveConfig] = None,
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize the integration pipeline.

        Args:
            config: Live configuration. Uses defaults if not provided.
            llm_client: Optional LLM client for AI response generation.
        """
        self.config = config or LiveConfig()

        # Phase 6, Section 16: Performance Metrics
        self.performance_metrics = PerformanceMetrics()

        # Create telemetry agent
        self.telemetry_agent = TelemetryAgent(thresholds=self.config.thresholds)

        # Create race engineer agent if LLM client provided
        self.race_engineer_agent: Optional[RaceEngineerAgent] = None
        if llm_client is not None:
            self.race_engineer_agent = RaceEngineerAgent(
                llm_client=llm_client,
                verbosity=self.config.verbosity.level,
            )

        # Create orchestrator with agents
        self.orchestrator = JarvisLiveOrchestrator(
            config=self.config,
            telemetry_agent=self.telemetry_agent,
            race_engineer_agent=self.race_engineer_agent,
        )

        # Create voice pipeline
        self.voice_pipeline = VoicePipeline(config=self.config)
        self.voice_pipeline.orchestrator = self.orchestrator

        # Session management
        self._sessions: Dict[str, SessionInfo] = {}

        # Statistics
        self._stats = {
            "telemetry_processed": 0,
            "events_detected": 0,
            "ai_responses_generated": 0,
            "sessions_created": 0,
            "sessions_ended": 0,
        }

        logger.info("Initialized IntegrationPipeline")

    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================

    @property
    def session_manager(self) -> Dict[str, SessionInfo]:
        """Get the session manager (active sessions)."""
        return self._sessions

    async def init_session(
        self,
        session_id: str,
        source: str,
        track_name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Initialize a new racing session.

        Creates session context, generates LiveKit token, and prepares
        components for telemetry processing.

        Args:
            session_id: Unique session identifier
            source: Telemetry source (torcs, assetto, etc.)
            track_name: Name of the race track
            config: Optional session configuration

        Returns:
            Session info including LiveKit connection details

        Example:
            session = await pipeline.init_session(
                session_id="race_001",
                source="torcs",
                track_name="Monza",
                config={"verbosity": "moderate"}
            )
        """
        # Create session context
        context = LiveSessionContext(
            session_id=session_id,
            source=source,
            track_name=track_name,
        )

        # Apply config if provided
        session_config = config or {}
        if "verbosity" in session_config and self.race_engineer_agent:
            self.race_engineer_agent.set_verbosity(session_config["verbosity"])

        # Create session info
        session = SessionInfo(
            session_id=session_id,
            context=context,
            config=session_config,
            created_at=datetime.now(timezone.utc),
        )

        # Store session
        self._sessions[session_id] = session
        self._stats["sessions_created"] += 1

        # Generate LiveKit token
        room_name = f"{session_id}_voice"
        livekit_token = self._generate_livekit_token(room_name, "ai_engineer")

        logger.info(f"Initialized session {session_id} for {track_name}")

        return {
            "session_id": session_id,
            "config": session_config,
            "livekit": {
                "url": self.config.livekit.url or "wss://livekit.example.com",
                "token": livekit_token,
                "room_name": room_name,
            }
        }

    async def end_session(self, session_id: str) -> None:
        """
        End a racing session and clean up resources.

        Args:
            session_id: Session to end

        Example:
            await pipeline.end_session("race_001")
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._stats["sessions_ended"] += 1
            logger.info(f"Ended session {session_id}")

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session information.

        Args:
            session_id: Session to look up

        Returns:
            Session info dict or None if not found
        """
        session = self._sessions.get(session_id)
        if session:
            return {
                "context": session.context,
                "config": session.config,
                "created_at": session.created_at,
                "stats": session.stats,
            }
        return None

    # =========================================================================
    # TELEMETRY PROCESSING
    # =========================================================================

    async def process_telemetry(
        self,
        session_id: str,
        telemetry: TelemetryData,
    ) -> Dict[str, Any]:
        """
        Process telemetry data through the full pipeline.

        Pipeline flow:
        1. Validate session exists
        2. Update session context
        3. Detect events via TelemetryAgent
        4. Generate AI response via RaceEngineerAgent (if events)
        5. Send to voice pipeline for TTS (if enabled)

        Performance tracking (Phase 6, Section 16):
        - telemetry_processing: Time for context update + event detection
        - llm_response: Time for LLM response generation
        - end_to_end: Total pipeline time

        Args:
            session_id: Session to process for
            telemetry: Telemetry data to process

        Returns:
            Processing result with events and AI response

        Raises:
            ValueError: If session not found

        Example:
            result = await pipeline.process_telemetry("race_001", telemetry)
        """
        start_time = time.time()

        # Validate session
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        context = session.context
        result: Dict[str, Any] = {
            "processed": True,
            "events": [],
            "ai_response": None,
            "latency_ms": 0,
        }

        try:
            # Track telemetry processing time (target: <50ms)
            telemetry_start = time.time()

            # Update context with telemetry
            context.update(telemetry)
            session.stats["telemetry_count"] += 1
            self._stats["telemetry_processed"] += 1

            # Detect events
            events = self.telemetry_agent.detect_events(telemetry, context)
            result["events"] = [{"type": e.type, "priority": e.priority.name} for e in events]

            # Record telemetry processing latency
            telemetry_latency_ms = (time.time() - telemetry_start) * 1000
            self.performance_metrics.record("telemetry_processing", telemetry_latency_ms)

            if events:
                session.stats["events_detected"] += len(events)
                self._stats["events_detected"] += len(events)

            # Check for interrupt condition
            if events and self.orchestrator.is_speaking:
                highest_priority = min(e.priority for e in events)
                if highest_priority <= Priority.HIGH and self.orchestrator.current_priority > highest_priority:
                    result["interrupt_triggered"] = True
                    self.orchestrator.pending_interrupt = events[0]

            # Generate AI response if we have events and agent
            if events and self.race_engineer_agent:
                # Set context on orchestrator
                self.orchestrator.set_session_context(context)

                # Process highest priority event
                events.sort(key=lambda e: e.priority)
                highest_event = events[0]

                # Check rate limiting (unless critical)
                min_interval = self.config.min_proactive_interval_seconds
                if context.can_send_proactive(min_interval) or highest_event.priority == Priority.CRITICAL:
                    try:
                        # Track LLM response time (target: <2000ms)
                        llm_start = time.time()

                        response = await self.race_engineer_agent.generate_proactive_response(
                            event=highest_event,
                            context=context,
                        )

                        # Record LLM latency
                        llm_latency_ms = (time.time() - llm_start) * 1000
                        self.performance_metrics.record("llm_response", llm_latency_ms)

                        if response:
                            result["ai_response"] = response
                            session.stats["ai_responses"] += 1
                            self._stats["ai_responses_generated"] += 1
                            context.mark_proactive_sent()

                            # Send to voice pipeline if enabled
                            if self.config.voice.enable_voice and self.voice_pipeline.tts_client:
                                try:
                                    # Track TTS conversion time (target: <500ms)
                                    tts_start = time.time()
                                    await self.voice_pipeline.deliver_response(response)
                                    tts_latency_ms = (time.time() - tts_start) * 1000
                                    self.performance_metrics.record("tts_conversion", tts_latency_ms)
                                except Exception as e:
                                    logger.error(f"Voice pipeline error: {e}")
                                    # Continue - AI response is still available

                    except asyncio.TimeoutError as e:
                        logger.error("LLM timeout during response generation")
                        result["error"] = ErrorCode.LLM_TIMEOUT.value
                        result["error_details"] = {
                            "code": ErrorCode.LLM_TIMEOUT.value,
                            "message": "LLM request timed out",
                            "is_transient": True,
                        }
                    except Exception as e:
                        logger.error(f"Error generating AI response: {e}")
                        if isinstance(e, JarvisError):
                            result["error"] = e.error_code.value
                            result["error_details"] = e.to_dict()
                        else:
                            result["error"] = ErrorCode.LLM_ERROR.value
                            result["error_details"] = {
                                "code": ErrorCode.LLM_ERROR.value,
                                "message": str(e),
                                "is_transient": False,
                            }

        except Exception as e:
            logger.error(f"Error processing telemetry: {e}")
            if isinstance(e, JarvisError):
                result["error"] = e.error_code.value
                result["error_details"] = e.to_dict()
            else:
                result["error"] = ErrorCode.TELEMETRY_PROCESSING_ERROR.value
                result["error_details"] = {
                    "code": ErrorCode.TELEMETRY_PROCESSING_ERROR.value,
                    "message": str(e),
                    "is_transient": True,
                }

        # Calculate and record end-to-end latency (target: <3000ms)
        result["latency_ms"] = int((time.time() - start_time) * 1000)
        self.performance_metrics.record("end_to_end", float(result["latency_ms"]))

        return result

    # =========================================================================
    # DRIVER QUERY HANDLING
    # =========================================================================

    async def handle_driver_query(
        self,
        session_id: str,
        query: str,
    ) -> Dict[str, Any]:
        """
        Handle a driver voice/text query.

        Args:
            session_id: Session to handle query for
            query: Driver's question or command

        Returns:
            Response with AI answer

        Raises:
            ValueError: If session not found

        Example:
            result = await pipeline.handle_driver_query("race_001", "How are my tires?")
        """
        start_time = time.time()

        # Validate session
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")

        context = session.context
        result: Dict[str, Any] = {"response": None}

        if not self.race_engineer_agent:
            result["response"] = "AI responses not available."
            return result

        try:
            # Set context on orchestrator
            self.orchestrator.set_session_context(context)

            # Generate response
            response = await self.race_engineer_agent.generate_reactive_response(
                query=query,
                context=context,
            )

            result["response"] = response
            session.stats["queries_handled"] += 1

            # Send to voice pipeline if enabled
            if self.config.voice.enable_voice and self.voice_pipeline.tts_client:
                try:
                    await self.voice_pipeline.deliver_response(response)
                except Exception as e:
                    logger.error(f"Voice pipeline error: {e}")

        except asyncio.TimeoutError:
            logger.error("LLM timeout handling query")
            result["error"] = ErrorCode.LLM_TIMEOUT.value
            result["error_details"] = {
                "code": ErrorCode.LLM_TIMEOUT.value,
                "message": "LLM request timed out",
                "is_transient": True,
            }
            result["response"] = "Sorry, the AI is taking too long. Please try again."

        except Exception as e:
            logger.error(f"Error handling query: {e}")
            if isinstance(e, JarvisError):
                result["error"] = e.error_code.value
                result["error_details"] = e.to_dict()
            else:
                result["error"] = ErrorCode.LLM_ERROR.value
                result["error_details"] = {
                    "code": ErrorCode.LLM_ERROR.value,
                    "message": str(e),
                    "is_transient": False,
                }
            result["response"] = "Sorry, I couldn't process your request."

        result["latency_ms"] = int((time.time() - start_time) * 1000)

        return result

    # =========================================================================
    # WEBSOCKET MESSAGE HANDLING
    # =========================================================================

    async def handle_websocket_message(
        self,
        message: Dict[str, Any],
        websocket: Any,
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Handle a WebSocket message.

        Routes messages to appropriate handlers based on type.

        Args:
            message: Parsed WebSocket message
            websocket: WebSocket connection
            session_id: Current session ID (if established)

        Returns:
            Response message or None
        """
        message_type = message.get("type")

        if message_type == "session_init":
            return await self._handle_ws_session_init(message, websocket)

        elif message_type == "telemetry":
            return await self._handle_ws_telemetry(message, websocket, session_id)

        elif message_type == "text_query":
            return await self._handle_ws_text_query(message, websocket, session_id)

        elif message_type == "config_update":
            return await self._handle_ws_config_update(message, websocket, session_id)

        elif message_type == "session_end":
            return await self._handle_ws_session_end(message, websocket, session_id)

        else:
            return self._create_error_response(
                "VALIDATION_ERROR",
                f"Unknown message type: {message_type}"
            )

    def _create_error_response(
        self,
        error_code: str,
        message: str,
    ) -> Dict[str, Any]:
        """
        Create a standardized error response.

        Args:
            error_code: Error code (e.g., VALIDATION_ERROR, SESSION_NOT_FOUND)
            message: Human-readable error message

        Returns:
            Error response dictionary with timestamp
        """
        return {
            "type": "error",
            "error_code": error_code,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    async def _handle_ws_session_init(
        self,
        message: Dict[str, Any],
        websocket: Any,
    ) -> Dict[str, Any]:
        """Handle WebSocket session_init message with validation."""
        # Validate required fields
        session_id = message.get("session_id")
        source = message.get("source")
        track_name = message.get("track_name")

        if not session_id:
            return self._create_error_response(
                "VALIDATION_ERROR",
                "session_id is required"
            )

        if not track_name:
            return self._create_error_response(
                "VALIDATION_ERROR",
                "track_name is required"
            )

        # Validate source - must be one of known sources
        valid_sources = ["torcs", "assetto_corsa", "assetto", "can_bus", "simulator"]
        if source and source not in valid_sources:
            return self._create_error_response(
                "VALIDATION_ERROR",
                f"Invalid source '{source}'. Must be one of: {', '.join(valid_sources)}"
            )

        # Check for duplicate session
        if session_id in self._sessions:
            return self._create_error_response(
                "VALIDATION_ERROR",
                f"Session {session_id} already exists"
            )

        session_info = await self.init_session(
            session_id=session_id,
            source=source or "unknown",
            track_name=track_name,
            config=message.get("config", {}),
        )

        return {
            "type": "session_confirmed",
            **session_info,
        }

    async def _handle_ws_telemetry(
        self,
        message: Dict[str, Any],
        websocket: Any,
        session_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Handle WebSocket telemetry message."""
        if not session_id:
            return self._create_error_response(
                "SESSION_NOT_FOUND",
                "No active session. Send session_init first."
            )

        # Verify session exists
        if session_id not in self._sessions:
            return self._create_error_response(
                "SESSION_NOT_FOUND",
                f"Session {session_id} not found"
            )

        try:
            telemetry_data = message.get("data", {})
            telemetry = TelemetryData(**telemetry_data)

            result = await self.process_telemetry(session_id, telemetry)

            # Send AI response if generated
            if result.get("ai_response"):
                await websocket.send_json({
                    "type": "ai_response",
                    "text": result["ai_response"],
                    "trigger": result["events"][0]["type"] if result["events"] else "telemetry",
                    "latency_ms": result["latency_ms"],
                })

        except Exception as e:
            logger.error(f"Error processing telemetry: {e}")
            return self._create_error_response(
                "VALIDATION_ERROR",
                str(e)
            )

        return None

    async def _handle_ws_text_query(
        self,
        message: Dict[str, Any],
        websocket: Any,
        session_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Handle WebSocket text_query message with validation."""
        if not session_id:
            return self._create_error_response(
                "SESSION_NOT_FOUND",
                "No active session. Send session_init first."
            )

        # Verify session exists
        if session_id not in self._sessions:
            return self._create_error_response(
                "SESSION_NOT_FOUND",
                f"Session {session_id} not found"
            )

        # Validate query is not empty
        query = message.get("query", "")
        if not query or not query.strip():
            return self._create_error_response(
                "VALIDATION_ERROR",
                "Query cannot be empty"
            )

        result = await self.handle_driver_query(session_id, query)

        await websocket.send_json({
            "type": "ai_response",
            "text": result["response"],
            "trigger": "text_query",
            "latency_ms": result.get("latency_ms", 0),
        })

        return None

    async def _handle_ws_config_update(
        self,
        message: Dict[str, Any],
        websocket: Any,
        session_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Handle WebSocket config_update message."""
        if not session_id:
            return self._create_error_response(
                "SESSION_NOT_FOUND",
                "No active session. Send session_init first."
            )

        # Verify session exists
        if session_id not in self._sessions:
            return self._create_error_response(
                "SESSION_NOT_FOUND",
                f"Session {session_id} not found"
            )

        session = self._sessions[session_id]
        new_config = message.get("config", {})

        # Update session config
        session.config.update(new_config)

        # Update verbosity if changed
        if "verbosity" in new_config and self.race_engineer_agent:
            self.race_engineer_agent.set_verbosity(new_config["verbosity"])

        logger.info(f"Updated config for session {session_id}")

        return None

    async def _handle_ws_session_end(
        self,
        message: Dict[str, Any],
        websocket: Any,
        session_id: Optional[str],
    ) -> None:
        """Handle WebSocket session_end message."""
        if session_id:
            await self.end_session(session_id)

        return None

    # =========================================================================
    # LIVEKIT TOKEN GENERATION
    # =========================================================================

    def _generate_livekit_token(
        self,
        room_name: str,
        participant_name: str,
    ) -> str:
        """
        Generate a LiveKit access token.

        Creates a JWT token for LiveKit room authentication.

        Args:
            room_name: Room to join
            participant_name: Participant identity

        Returns:
            JWT token string
        """
        api_key = self.config.livekit.api_key or "devkey"
        api_secret = self.config.livekit.api_secret or "secret"

        # Create JWT header
        header = {
            "alg": "HS256",
            "typ": "JWT"
        }

        # Create claims
        now = int(time.time())
        claims = {
            "iss": api_key,
            "sub": participant_name,
            "name": participant_name,
            "iat": now,
            "nbf": now,
            "exp": now + 86400,  # 24 hour expiry
            "video": {
                "room": room_name,
                "roomJoin": True,
                "canPublish": True,
                "canSubscribe": True,
            }
        }

        # Encode header and claims
        header_b64 = self._base64url_encode(json.dumps(header))
        claims_b64 = self._base64url_encode(json.dumps(claims))

        # Create signature
        message = f"{header_b64}.{claims_b64}"
        signature = hmac.new(
            api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        ).digest()
        signature_b64 = self._base64url_encode_bytes(signature)

        return f"{header_b64}.{claims_b64}.{signature_b64}"

    def _base64url_encode(self, data: str) -> str:
        """Base64url encode a string."""
        return self._base64url_encode_bytes(data.encode("utf-8"))

    def _base64url_encode_bytes(self, data: bytes) -> str:
        """Base64url encode bytes."""
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")

    # =========================================================================
    # STATISTICS AND HEALTH
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.

        Returns:
            Dictionary with pipeline statistics
        """
        return {
            "active_sessions": len(self._sessions),
            **self._stats,
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get performance report with metrics, compliance, and bottlenecks.

        Phase 6, Section 16: Performance Optimization

        Returns comprehensive performance report including:
        - Summary of all metrics (avg, p50, p95, p99)
        - Compliance status against targets
        - Identified bottlenecks

        Returns:
            Performance report dictionary

        Example:
            report = pipeline.get_performance_report()
            if report["compliance"]["llm_response"]["compliant"]:
                print("LLM latency within target")
        """
        return {
            "summary": self.performance_metrics.get_summary(),
            "compliance": self.performance_metrics.check_compliance(),
            "targets": {
                "telemetry_processing": "<50ms",
                "llm_response": "<2000ms",
                "tts_conversion": "<500ms",
                "end_to_end": "<3000ms",
            },
            "bottlenecks": self.performance_metrics.identify_bottlenecks(),
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on pipeline components.

        Returns:
            Health status for all components

        Status levels:
        - healthy: Core telemetry pipeline working
        - degraded: Optional components (LLM, voice) unavailable
        - unhealthy: Critical components broken

        Core components (required for healthy):
        - telemetry_agent: Event detection from telemetry
        - orchestrator: Event routing and priority queue

        Optional components (unavailable -> still healthy):
        - llm: AI response generation
        - voice_pipeline: TTS/STT processing
        - livekit: WebRTC transport
        """
        components = {}

        # Core components (must be working)
        components["telemetry_agent"] = "healthy"
        components["orchestrator"] = "healthy" if self.orchestrator else "unhealthy"

        # Optional: LLM for AI responses
        components["llm"] = "healthy" if self.race_engineer_agent else "unavailable"

        # Optional: LiveKit configuration for voice
        livekit_configured = bool(
            self.config.livekit.url and
            self.config.livekit.api_key and
            self.config.livekit.api_secret
        )
        components["livekit"] = "healthy" if livekit_configured else "unavailable"

        # Optional: Voice pipeline
        voice_available = (
            self.voice_pipeline.tts_client is not None or
            not self.config.voice.enable_voice
        )
        components["voice_pipeline"] = "healthy" if voice_available else "unavailable"

        # Overall status based on core components only
        # "unavailable" is okay - system works without optional components
        core_healthy = all(
            components[c] == "healthy"
            for c in ["telemetry_agent", "orchestrator"]
        )
        any_unhealthy = any(s == "unhealthy" for s in components.values())

        if any_unhealthy:
            status = "unhealthy"
        elif core_healthy:
            status = "healthy"
        else:
            status = "degraded"

        return {
            "status": status,
            "components": components,
            "active_sessions": len(self._sessions),
        }

    def __repr__(self) -> str:
        """String representation of pipeline."""
        return (
            f"IntegrationPipeline("
            f"sessions={len(self._sessions)}, "
            f"llm={'enabled' if self.race_engineer_agent else 'disabled'})"
        )
