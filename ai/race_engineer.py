"""
AI Race Engineer Worker for F1 Telemetry Dashboard.

Integrates the local race engineer pipeline with telemetry ingestion.
Runs in a separate QThread to avoid blocking the UI.

Receives telemetry samples, detects events, generates AI commentary.
"""

import asyncio
import logging
import math
import re
import time
from typing import Optional, Dict, Any
from PyQt5 import QtCore

import os
from ai.race_engineer_core import (
    Event,
    GForces,
    LLMClient,
    LiveSessionContext,
    OpponentSnapshot,
    Priority,
    RaceEngineerAgent,
    RideHeight,
    SuspensionTravel,
    TelemetryAgent,
    TelemetryData,
    ThresholdsConfig,
    TirePressure,
    TireTemps,
    TireWear,
    WheelSlip,
)


logger = logging.getLogger(__name__)


class AIRaceEngineerWorker(QtCore.QThread):
    """
    AI Race Engineer worker thread.

    Processes telemetry samples, detects events, and generates AI commentary
    using local race engineer agents.

    Signals:
        ai_commentary(str message, str trigger, int priority) - AI-generated commentary
        status_update(str message) - Status messages for logging
    """

    ai_commentary = QtCore.pyqtSignal(str, str, int)  # message, trigger, priority
    driver_query_received = QtCore.pyqtSignal(str)  # driver query text (for UI display)
    processing_query = QtCore.pyqtSignal(bool)  # True=LLM busy, False=done
    status_update = QtCore.pyqtSignal(str)

    def __init__(
        self,
        track_name: str = "Unknown Track",
        session_id: str = "ac_session_001",
        verbosity: str = "minimal",
    ):
        """
        Initialize AI Race Engineer.

        Args:
            track_name: Name of the track
            session_id: Unique session identifier
            verbosity: AI verbosity level (minimal, moderate, verbose)
        """
        super().__init__()

        # Configuration
        self.track_name = track_name
        self.session_id = session_id
        self.verbosity = verbosity

        # Agents (initialized in run())
        self.telemetry_agent: Optional[TelemetryAgent] = None
        self.race_engineer_agent: Optional[RaceEngineerAgent] = None

        # Session context
        self.context: Optional[LiveSessionContext] = None

        # Thread control
        self._running = False
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._on_track = False  # True only when AC status is LIVE (2)
        self._ready = False
        # Prevent repeated spam of identical CRITICAL events while still allowing
        # different critical event types to bypass the generic cooldown.
        self._last_critical_event_sent: Dict[str, float] = {}

        # Queues (initialized in run() after event loop is created)
        self.telemetry_queue: Optional[asyncio.Queue] = None
        self.query_queue: Optional[asyncio.Queue] = None

        logger.info("AIRaceEngineerWorker initialized")

    def update_session_info(self, info: dict):
        """
        Update session info from telemetry backend.

        Called when session_info_update signal fires. Seeds fuel consumption
        estimate from datasheet if no real telemetry data is available yet.

        Args:
            info: Dict with keys 'track', 'car_model', 'max_fuel', etc.
        """
        track = info.get("track", "Unknown Track")
        car_model = info.get("car_model", "")

        # Update track name
        self.track_name = track
        if self.context:
            self.context.track_name = track

        # Look up fuel data from datasheet
        try:
            from ai.fuel_lookup import lookup_fuel_consumption
            fuel_data = lookup_fuel_consumption(car_model, track)

            # Seed fuel consumption estimate only if no real telemetry data yet
            if self.context and self.context.fuel_consumption_per_lap <= 0:
                self.context.fuel_consumption_per_lap = fuel_data["fuel_per_lap"]
                logger.info(
                    "Seeded fuel estimate from datasheet: %.2f L/lap (%s @ %s)",
                    fuel_data["fuel_per_lap"],
                    fuel_data["matched_car"],
                    fuel_data["matched_track"],
                )
        except Exception as e:
            logger.warning("Failed to look up fuel data: %s", e)

        logger.info("Session info updated: track=%s, car=%s", track, car_model)

    def run(self):
        """Main thread execution loop."""
        self._running = True
        self.status_update.emit("AI Race Engineer starting...")

        try:
            # Create asyncio event loop for this thread
            self._event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._event_loop)

            # Create queues (must be done AFTER event loop is set)
            # Limit queue size to prevent infinite backlog
            # Keep only last 5 telemetry samples (at 60Hz = ~80ms buffer)
            self.telemetry_queue = asyncio.Queue(maxsize=5)
            self.query_queue = asyncio.Queue()

            # Initialize agents
            self._initialize_agents()

            # Initialize session context
            self.context = LiveSessionContext(
                session_id=self.session_id,
                source="assetto_corsa",
                track_name=self.track_name
            )

            self._ready = True
            self.status_update.emit("AI Race Engineer ready")

            # Run async processing loop
            self._event_loop.run_until_complete(self._process_loop())

        except Exception as e:
            logger.error(f"AI Race Engineer error: {e}", exc_info=True)
            self.status_update.emit(f"AI Race Engineer error: {e}")
        finally:
            self._ready = False
            if self._event_loop:
                self._event_loop.close()
            self.status_update.emit("AI Race Engineer stopped")

    def _initialize_agents(self):
        """Initialize TelemetryAgent and RaceEngineerAgent."""
        from pathlib import Path

        # Live mode targets short radio replies with low latency.
        # Low-latency default for live radio replies (one short sentence).
        live_max_tokens = int(os.getenv("LIVE_LLM_MAX_TOKENS", "24"))
        # Lower default temperature to reduce speculative/hallucinated claims.
        live_temperature = float(os.getenv("LIVE_LLM_TEMPERATURE", "0.15"))
        local_max_time_seconds = float(os.getenv("LOCAL_LLM_MAX_TIME_SECONDS", "5.0"))
        # Support new env var with fallback to old one for backwards compat
        local_model_path = os.getenv(
            "LOCAL_MODEL_PATH",
            os.getenv("LOCAL_ADAPTER_PATH", "race_engineer_gguf/granite-race-engineer-Q4_K_M.gguf"),
        )

        # Check if GGUF model file exists
        force_rule_based_fallback = False
        model_file = Path(local_model_path)
        if not model_file.is_absolute():
            model_file = Path(__file__).parent.parent / model_file

        if not model_file.exists():
            # Try auto-downloading from Hugging Face Hub
            try:
                from ai.model_downloader import ensure_model
                logger.info("Model not found locally, attempting auto-download...")
                self.status_update.emit("Downloading AI model (first run)...")
                downloaded = ensure_model(local_model_path)
                model_file = downloaded
                local_model_path = str(downloaded)
                os.environ["LOCAL_MODEL_PATH"] = local_model_path
                logger.info("Model downloaded: %s", model_file)
            except Exception as dl_err:
                logger.warning(
                    "GGUF model not found at %s and auto-download failed: %s; "
                    "falling back to rule-based responses.",
                    model_file, dl_err,
                )
                self.status_update.emit(
                    "GGUF model not found. Using rule-based fallback responses."
                )
                force_rule_based_fallback = True

        # Pre-download post-race model too so it's ready when the user opens Jarvis Post
        try:
            from ai.model_downloader import ensure_postrace_model, is_model_available, POSTRACE_LOCAL_PATH
            if not is_model_available(POSTRACE_LOCAL_PATH):
                self.status_update.emit("Downloading Post-Race model (first run)...")
                ensure_postrace_model()
                logger.info("Post-race model downloaded")
        except Exception as dl_err:
            logger.warning("Post-race model download failed: %s (will retry on first use)", dl_err)

        llm_client = LLMClient(
            max_tokens=live_max_tokens,
            temperature=live_temperature,
            local_model_path=local_model_path,
            local_max_time_seconds=local_max_time_seconds,
            force_rule_based_fallback=force_rule_based_fallback,
        )
        # Wire status callback so LLM load/generation failures surface in the UI
        llm_client.set_status_callback(
            lambda msg: self.status_update.emit(msg)
        )

        if not force_rule_based_fallback:
            logger.info("Using local GGUF model from: %s", local_model_path)
        else:
            logger.info("Rule-based responses only (GGUF model unavailable)")

        # Create race engineer agent
        self.race_engineer_agent = RaceEngineerAgent(
            llm_client=llm_client,
            verbosity=self.verbosity
        )

        # Create telemetry agent with default thresholds
        thresholds = ThresholdsConfig(
            tire_temp_warning=100.0,  # Celsius
            tire_temp_critical=110.0,
            fuel_warning_laps=5,
            fuel_critical_laps=2,
            gap_change_threshold=1.0
        )
        self.telemetry_agent = TelemetryAgent(thresholds=thresholds)

        logger.info("AI agents initialized")

    async def _process_loop(self):
        """Async processing loop - processes telemetry, queries, and generates AI responses."""
        # Run telemetry and query processors as separate concurrent tasks
        # This ensures query processing (which can take 5-30s for LLM) is never cancelled
        telemetry_task = asyncio.create_task(self._telemetry_processor())
        query_task = asyncio.create_task(self._query_processor())

        try:
            # Wait for both tasks (they run until self._running is False)
            await asyncio.gather(telemetry_task, query_task)
        except Exception as e:
            logger.error(f"Error in process loop: {e}", exc_info=True)

    async def _telemetry_processor(self):
        """Continuously process telemetry samples."""
        logger.info("Telemetry processor started")
        while self._running:
            await self._process_telemetry_queue()
        logger.info("Telemetry processor stopped")

    async def _query_processor(self):
        """Continuously process driver queries."""
        logger.info("Query processor started")
        while self._running:
            await self._process_query_queue()
        logger.info("Query processor stopped")

    async def _process_telemetry_queue(self):
        """Process telemetry from queue."""
        try:
            # Get telemetry from queue (with timeout)
            telemetry_dict = await asyncio.wait_for(
                self.telemetry_queue.get(),
                timeout=0.1
            )

            # Convert dict to TelemetryData
            telemetry = self._dict_to_telemetry(telemetry_dict)

            # Detect events BEFORE updating context so that delta-based
            # checks (gap_change, lap_complete, sector_complete) compare
            # new telemetry against the previous state, not itself.
            events = self.telemetry_agent.detect_events(telemetry, self.context)

            # Update context with current telemetry
            self.context.update(telemetry)

            # Process each event
            for event in events:
                await self._handle_event(event)

        except asyncio.TimeoutError:
            # No telemetry, that's ok
            pass
        except Exception as e:
            logger.error(f"Error processing telemetry: {e}", exc_info=True)

    async def _process_query_queue(self):
        """Process driver queries from queue."""
        try:
            # Get query from queue (with timeout)
            query = await asyncio.wait_for(
                self.query_queue.get(),
                timeout=0.1
            )

            # Clear any pending queries (only process the latest)
            # This prevents queue backup if driver speaks multiple times
            while not self.query_queue.empty():
                try:
                    stale_query = self.query_queue.get_nowait()
                    logger.debug(f"Dropping stale query: {stale_query}")
                    query = stale_query  # Use the newest query
                except asyncio.QueueEmpty:
                    break

            # Emit signal to show query in UI
            self.driver_query_received.emit(query)

            # If not on track, don't waste an LLM call
            if not self._on_track:
                msg = "We're not on track right now. Get out there and I'll have your data ready."
                self.ai_commentary.emit(msg, "driver_query", 2)
                self.status_update.emit("AI Race Engineer ready")
                return

            logger.info(f"Processing driver query: {query}")
            self.status_update.emit(f"Processing: \"{query}\"")

            # Signal voice input to pause during LLM processing
            self.processing_query.emit(True)
            try:
                # Generate AI response (full response, then send to TTS)
                try:
                    query_timeout_seconds = float(os.getenv("LIVE_LLM_QUERY_TIMEOUT_SECONDS", "10.0"))
                    response = await asyncio.wait_for(
                        self.race_engineer_agent.generate_reactive_response(
                            query=query,
                            context=self.context,
                        ),
                        timeout=query_timeout_seconds,
                    )
                    logger.debug(f"LLM response received: {response[:100]}...")
                except asyncio.TimeoutError:
                    logger.error("LLM response timed out after %.1f seconds", query_timeout_seconds)
                    if self._is_pit_query(query):
                        response = self._build_pit_query_fallback_response()
                    else:
                        prompt = self._build_reactive_prompt(query)
                        response = self.race_engineer_agent.llm_client._generate_fallback_response(prompt)
                    trigger = "driver_query_timeout"

                response = self._clean_llm_response(response)

                # Guardrail chain: only one fires (first match wins).
                if self._has_ungrounded_numbers(response, query):
                    logger.warning("Response contains ungrounded numbers; using fallback")
                    response = self._build_reactive_fallback_response(query)
                elif self._should_force_pit_fallback(query, response):
                    logger.warning("Replacing low-confidence pit response with deterministic fallback")
                    response = self._build_pit_query_fallback_response()
                elif self._is_low_quality_response(response):
                    logger.warning("Low-quality reactive response detected; using deterministic fallback")
                    response = self._build_reactive_fallback_response(query)

                # Check for empty response and provide fallback
                if not response or not response.strip():
                    logger.warning("LLM returned empty response, using fallback")
                    response = "I heard your question but I'm having trouble generating a response right now. Could you please rephrase your question about your race situation?"

                # Emit as AI commentary — triggers both UI display and TTS
                self.ai_commentary.emit(response, "driver_query", 2)  # MEDIUM priority
                logger.info(f"AI response to query: {response[:50]}...")
                self.status_update.emit("AI Race Engineer ready")
            finally:
                self.processing_query.emit(False)

        except asyncio.TimeoutError:
            # No query, that's ok
            pass
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            self.ai_commentary.emit(
                "Sorry, I couldn't process that. Ask again in a moment.",
                "driver_query_error",
                1  # HIGH priority for errors
            )
            self.status_update.emit("AI Race Engineer ready")

    def _dict_to_telemetry(self, data: Dict[str, Any]) -> TelemetryData:
        """
        Convert AC telemetry dict to TelemetryData Pydantic model.

        Args:
            data: Telemetry dict from AC worker

        Returns:
            TelemetryData object
        """
        # DEBUG: Log incoming data keys
        logger.debug(f"Converting telemetry dict with keys: {list(data.keys())}")
        logger.debug(f"speed={data.get('speed')}, rpms={data.get('rpms')}, rpm={data.get('rpm')}")

        # Extract tire data
        tire_temps = TireTemps(
            fl=data.get("tyre_temp_fl", 80.0),
            fr=data.get("tyre_temp_fr", 80.0),
            rl=data.get("tyre_temp_rl", 80.0),
            rr=data.get("tyre_temp_rr", 80.0)
        )

        tire_pressure = TirePressure(
            fl=data.get("tyre_pressure_fl", 28.0),
            fr=data.get("tyre_pressure_fr", 28.0),
            rl=data.get("tyre_pressure_rl", 28.0),
            rr=data.get("tyre_pressure_rr", 28.0)
        )

        # G-forces from AC shared memory
        g_forces = GForces(
            lateral=data.get("g_force_lat", 0.0),
            longitudinal=data.get("g_force_lon", 0.0)
        )

        # Wheel slip
        wheel_slip = None
        if "wheel_slip_fl" in data:
            wheel_slip = WheelSlip(
                fl=data.get("wheel_slip_fl", 0.0),
                fr=data.get("wheel_slip_fr", 0.0),
                rl=data.get("wheel_slip_rl", 0.0),
                rr=data.get("wheel_slip_rr", 0.0)
            )

        # Suspension travel
        suspension_travel = None
        if "suspension_fl" in data:
            suspension_travel = SuspensionTravel(
                fl=data.get("suspension_fl", 0.0),
                fr=data.get("suspension_fr", 0.0),
                rl=data.get("suspension_rl", 0.0),
                rr=data.get("suspension_rr", 0.0)
            )

        # Ride height
        ride_height = None
        if "ride_height_front" in data:
            ride_height = RideHeight(
                front=data.get("ride_height_front", 0.0),
                rear=data.get("ride_height_rear", 0.0)
            )

        # Tire wear: AC's tyreWear = remaining life (100=fresh, decreases with wear)
        # AI context expects percent worn (0=fresh, 100=worn), so invert.
        tire_wear = None
        if "tyre_wear_fl" in data:
            tire_wear = TireWear(
                fl=max(0.0, min(100.0, 100.0 - data.get("tyre_wear_fl", 100.0))),
                fr=max(0.0, min(100.0, 100.0 - data.get("tyre_wear_fr", 100.0))),
                rl=max(0.0, min(100.0, 100.0 - data.get("tyre_wear_rl", 100.0))),
                rr=max(0.0, min(100.0, 100.0 - data.get("tyre_wear_rr", 100.0)))
            )

        # Car damage (AC: 5 zones, values 0.0 = no damage)
        car_damage = None
        if "car_damage_front" in data:
            car_damage = {
                "front": data.get("car_damage_front", 0.0),
                "rear": data.get("car_damage_rear", 0.0),
                "left": data.get("car_damage_left", 0.0),
                "right": data.get("car_damage_right", 0.0),
                "centre": data.get("car_damage_centre", 0.0),
            }

        opponents = None
        if data.get("opponents"):
            opponents = []
            for car in data.get("opponents", []):
                try:
                    opponents.append(
                        OpponentSnapshot(
                            car_index=int(car.get("car_index", 0)),
                            position=max(1, int(car.get("position", 1))),
                            lap_number=max(0, int(car.get("lap_number", 0))),
                            speed=max(0.0, float(car.get("speed", 0.0))),
                            track_position=car.get("track_position", None),
                        )
                    )
                except Exception:
                    continue

        lap_id = data.get("lap_id", 0)
        sector = data.get("sector")
        try:
            sector = int(sector) if sector is not None else None
        except (TypeError, ValueError):
            sector = None
        if sector is not None and not (1 <= sector <= 3):
            sector = None

        return TelemetryData(
            speed=data.get("speed", 0.0),
            rpms=data.get("rpms", 0),
            gear=data.get("gear", 0),
            throttle=data.get("throttle", 0.0),
            brake=data.get("brake", 0.0),
            steering_angle=data.get("steer_angle", 0.0),
            fuel=data.get("fuel", None),
            tire_temps=tire_temps,
            tire_pressure=tire_pressure,
            tire_wear=tire_wear,
            g_forces=g_forces,
            wheel_slip=wheel_slip,
            suspension_travel=suspension_travel,
            ride_height=ride_height,
            car_damage=car_damage,
            x=data.get("x", 0.0),
            z=data.get("z", 0.0),
            lap_id=lap_id,
            lap_number=lap_id,  # AI uses lap_number, AC provides lap_id
            t=data.get("t", 0.0),
            sector=sector,
            position=data.get("position"),
            gap_ahead=data.get("gap_ahead"),
            gap_behind=data.get("gap_behind"),
            opponents=opponents,
            last_time_ms=data.get("last_time_ms"),
        )

    async def _handle_event(self, event: Event):
        """
        Handle detected event - generate AI response.

        Args:
            event: Detected event from TelemetryAgent
        """
        try:
            # Event-triggered alerts are fully rule-based for deterministic low latency.
            # Do not call LLM for proactive event commentary.
            if not self.context:
                return

            min_interval = 10.0
            if hasattr(self.context, "config") and self.context.config:
                min_interval = getattr(
                    self.context.config,
                    "min_proactive_interval_seconds",
                    10.0,
                )

            # CRITICAL alerts must not be blocked by generic anti-spam cooldown.
            is_critical = event.priority == Priority.CRITICAL
            now_monotonic = time.monotonic()
            if is_critical:
                last_sent = self._last_critical_event_sent.get(event.type, 0.0)
                if (now_monotonic - last_sent) < min_interval:
                    logger.debug(
                        "Suppressing repeated CRITICAL event %s (cooldown %.1fs)",
                        event.type,
                        min_interval,
                    )
                    return

            if not is_critical and not self.context.can_send_proactive(min_interval):
                logger.debug("Skipping proactive message for %s - too soon", event.type)
                return
            if is_critical and not self.context.can_send_proactive(min_interval):
                logger.warning(
                    "Bypassing proactive cooldown for CRITICAL event: %s",
                    event.type,
                )

            response = self._build_event_fallback_response(event)
            response = self._clean_llm_response(response)
            if not response:
                response = "Copy that. Monitoring."

            self.context.mark_proactive_sent()
            self.context.add_exchange(f"[Event: {event.type}]", response)
            if is_critical:
                self._last_critical_event_sent[event.type] = now_monotonic

            self.ai_commentary.emit(response, event.type, event.priority.value)
            logger.info("Rule-based commentary for %s: %s...", event.type, response[:80])

        except Exception as e:
            logger.error(f"Error generating AI response for {event.type}: {e}", exc_info=True)

    def update_ac_status(self, ac_status: int):
        """Update whether the car is on track (AC status 2=LIVE)."""
        self._on_track = (ac_status == 2)

    def process_telemetry(self, telemetry_dict: Dict[str, Any]):
        """
        Process incoming telemetry sample (called from main thread).

        Args:
            telemetry_dict: Telemetry data dictionary from AC worker
        """
        if self._event_loop and self._running and self.telemetry_queue is not None:
            # Receiving realtime samples implies we're on track.
            if not self._on_track:
                self._on_track = True

            # Thread-safe enqueue into event-loop-owned queue.
            asyncio.run_coroutine_threadsafe(
                self._enqueue_telemetry(telemetry_dict),
                self._event_loop
            )

    async def _enqueue_telemetry(self, telemetry_dict: Dict[str, Any]):
        """
        Thread-safe enqueue helper.

        Keeps the queue bounded and drops oldest telemetry when full.
        """
        if self.telemetry_queue is None:
            return

        if self.telemetry_queue.full():
            try:
                self.telemetry_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass

        await self.telemetry_queue.put(telemetry_dict)

    def process_driver_query(self, query: str):
        """
        Process driver query (called from voice input or main thread).

        Args:
            query: Driver's question/command as text
        """
        if not self._ready:
            self.status_update.emit("AI still warming up local model...")
            self.ai_commentary.emit(
                "AI is still loading locally. Give me a few seconds and ask again.",
                "driver_query_warmup",
                1,
            )
            return

        # Filter out empty/noise transcriptions (e.g. accidental PTT press)
        # Show in transcript but don't invoke the LLM pipeline
        if not query.strip().strip(".…,!? "):
            logger.info(f"Ignoring noise transcription: {query}")
            self.driver_query_received.emit(query)
            return

        if self._event_loop and self._running and self.query_queue is not None:
            # Thread-safe: put query in queue
            asyncio.run_coroutine_threadsafe(
                self.query_queue.put(query),
                self._event_loop
            )

    @staticmethod
    def _clean_llm_response(response: str) -> str:
        """
        Post-process LLM output to strip meta-commentary, references,
        and prompt template leakage that the model sometimes outputs.

        Args:
            response: Raw LLM response text

        Returns:
            Cleaned response suitable for TTS/display
        """
        # Split into lines for line-level filtering
        lines = response.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip instruction leakage lines (model echoing its own rules)
            if stripped.startswith("- Do NOT") or stripped.startswith("Do NOT"):
                continue
            if stripped.startswith("- Do not") or stripped.startswith("Do not"):
                continue
            if re.match(r"^-\s+(Keep|Reply|Answer|Speak|Focus|Provide|Use)\b", stripped, re.IGNORECASE):
                continue
            # Skip lines that look like rule headers
            if re.match(r"^(RULES|CONSTRAINTS|INSTRUCTIONS|GUIDELINES)\s*:", stripped, re.IGNORECASE):
                continue
            # Skip lines that are just bullet points with instructions
            if re.match(r"^[-•]\s+(You are|The driver|Responses?|Lead with|Match)\b", stripped, re.IGNORECASE):
                continue
            cleaned_lines.append(line)
        response = "\n".join(cleaned_lines)

        # Remove "Driver's Question: ..." echo (prompt leakage)
        response = re.sub(r"Driver'?s?\s*Question\s*:\s*\"?[^\"]*\"?\s*", "", response, flags=re.IGNORECASE)

        # Remove "Driver: [Event: ...] Engineer:" pattern (prompt leakage)
        response = re.sub(r"Driver:\s*\[Event:\s*[^\]]*\]\s*Engineer:\s*", "", response, flags=re.IGNORECASE)

        # Remove "Radio Message:" / "Engineer:" / "Alert:" prefix
        response = re.sub(r"^(Radio\s*Message|Engineer|Response|Alert|Answer)\s*:\s*", "", response, flags=re.IGNORECASE)

        # Remove "References:", "Sources:", etc. sections and everything after
        response = re.sub(r"\n?\s*(References|Sources|Notes?|Context|Data|Observations?)\s*:.*", "", response, flags=re.IGNORECASE | re.DOTALL)

        # Remove numbered data lists like "1. Tire Wear Data: 100%"
        response = re.sub(r"\n\s*\d+\.\s+\w[\w\s]*?:\s*[\d.]+[%°CLs]*\s*", "", response)

        # Remove filler openers
        response = re.sub(r"^(Understood|Copy that|Roger|Noted)[.,]?\s*", "", response, flags=re.IGNORECASE)

        # Remove meta-commentary markers
        response = re.sub(r"\n\s*---+\s*\n?", "", response)

        # Remove asterisks used for emphasis
        response = response.replace("*", "")

        # Strip surrounding quotes
        response = response.strip().strip('"').strip("'")

        # Collapse multiple newlines and trim
        response = re.sub(r"\n{2,}", "\n", response).strip()

        # Final safety: if response is still very long (>200 chars), take first sentence only
        if len(response) > 200:
            first_sentence = re.split(r'(?<=[.!?])\s', response, maxsplit=1)
            if first_sentence:
                response = first_sentence[0]

        return response

    @staticmethod
    def _extract_numeric_values(text: str) -> list[float]:
        """Extract numeric values from text for grounding checks."""
        if not text:
            return []

        values = []
        for token in re.findall(r"\d+(?:\.\d+)?", text):
            try:
                values.append(float(token))
            except ValueError:
                continue
        return values

    def _has_ungrounded_numbers(self, response: str, query: str) -> bool:
        """
        Check if response contains numeric claims not grounded in the
        current query/context snapshot.

        Returns True if the response has fabricated numbers that should
        trigger a deterministic fallback.
        """
        if not response or not self.context:
            return False

        response_nums = self._extract_numeric_values(response)
        if not response_nums:
            return False

        source_text = f"{query}\n{self.context.to_prompt_context(query=query, grounding_only=True)}"
        source_nums = self._extract_numeric_values(source_text)

        if not source_nums:
            return True

        tolerance = 0.6  # allow small rounding differences
        for val in response_nums:
            if min(abs(val - src) for src in source_nums) > tolerance:
                logger.warning(
                    "Ungrounded numeric claim in response (%.3f); triggering fallback",
                    val,
                )
                return True

        return False

    def _build_reactive_prompt(self, query: str) -> str:
        """
        Build the reactive prompt from query + session context.

        Used to generate the same prompt that would have been sent to the LLM,
        so the rule-based fallback can keyword-match on full context.
        """
        from ai.race_engineer_core.prompts import format_conversation_history

        session_context_str = self.context.to_prompt_context(query=query) if self.context else ""
        conversation_str = format_conversation_history(
            list(self.context.conversation_history) if self.context else []
        )
        return self.race_engineer_agent.reactive_prompt.format(
            query=query,
            session_context=session_context_str,
            conversation_history=conversation_str,
        )

    @staticmethod
    def _is_pit_query(query: str) -> bool:
        """Return True for pitting/strategy timing questions."""
        if not query:
            return False
        query_lower = query.lower()
        return any(token in query_lower for token in (
            "pit", "pits", "pitting",
            "pet", "pay", "bet", "bit",  # common STT misrecognitions of "pit"
            "box", "stop", "stint", "refuel",
        ))

    @staticmethod
    def _signals_insufficient_data(response: str) -> bool:
        """Detect weak model replies that claim missing info."""
        if not response:
            return False
        lowered = response.lower()
        markers = (
            "not enough information",
            "not enough data",
            "insufficient data",
            "lack enough",
            "can't determine",
            "cannot determine",
            "unable to determine",
            "don't have enough",
            "do not have enough",
        )
        return any(marker in lowered for marker in markers)

    def _should_force_pit_fallback(self, query: str, response: str) -> bool:
        """Return True when a pit query got an insufficient-data response
        that the context-aware deterministic fallback can answer better."""
        if not self._is_pit_query(query):
            return False
        return self._signals_insufficient_data(response)

    @staticmethod
    def _is_low_quality_response(response: str) -> bool:
        """Detect unusable outputs like lone numbers or placeholders."""
        if not response:
            return True

        cleaned = response.strip().lower()
        if cleaned in {"0", "0.", "n/a", "na", "none", "null", "unknown", "idk", "...", "insufficient data.", "insufficient data"}:
            return True

        if re.fullmatch(r"[0-9\W]+", cleaned):
            return True

        # Very short single-token outputs are almost always poor radio responses.
        tokens = cleaned.split()
        return len(tokens) == 1 and len(cleaned) <= 3

    def _build_reactive_fallback_response(self, query: str) -> str:
        """Context-aware deterministic fallback for failed/low-quality reactive responses."""
        if self._is_pit_query(query):
            return self._build_pit_query_fallback_response()

        if not self.context:
            return "Copy that. Monitoring the situation."

        query_lower = query.lower()

        # Damage queries
        if any(kw in query_lower for kw in ("damage", "crash", "contact", "hit", "broken", "wing")):
            total_damage = sum(self.context.car_damage.values())
            if total_damage > 0:
                parts = [f"{zone} {val:.0f}%" for zone, val in self.context.car_damage.items() if val > 0]
                return f"Car damage: {', '.join(parts)}. Monitor handling."
            return "No damage reported. Car is clean."

        # Tire queries
        if any(kw in query_lower for kw in ("tire", "tyre", "temp", "temperature", "pressure", "wear", "grip")):
            max_temp = max(self.context.tire_temps.values())
            max_corner = max(self.context.tire_temps, key=self.context.tire_temps.get).upper()
            wear_values = list(self.context.tire_wear.values())
            has_wear = any(w > 0 for w in wear_values)
            if has_wear:
                max_wear = max(wear_values)
                return f"Hottest tire is {max_corner} at {max_temp:.0f} C. Max wear {max_wear:.0f} percent."
            return f"Hottest tire is {max_corner} at {max_temp:.0f} C. No wear data available."

        # Gap / position queries
        if any(kw in query_lower for kw in ("gap", "ahead", "behind", "position", "opponent")):
            gap_ahead_str = f"{self.context.gap_ahead:.1f}s" if self.context.gap_ahead is not None else "no car"
            gap_behind_str = f"{self.context.gap_behind:.1f}s" if self.context.gap_behind is not None else "no car"
            return f"P{self.context.position}. Gap ahead {gap_ahead_str}, behind {gap_behind_str}."

        # Fuel queries
        if any(kw in query_lower for kw in ("fuel", "range")):
            fuel_laps = self.context.get_fuel_laps_remaining()
            if math.isfinite(fuel_laps):
                return f"Fuel at {self.context.fuel_remaining:.1f} litres, about {int(fuel_laps)} laps remaining."
            return f"Fuel at {self.context.fuel_remaining:.1f} litres. Need a completed lap to estimate range."

        # Lap time queries
        if any(kw in query_lower for kw in ("lap", "time", "pace", "fast", "best")):
            if self.context.last_lap:
                best_str = f" Best is {self.context._format_lap_time(self.context.best_lap)}." if self.context.best_lap else ""
                return f"Last lap {self.context._format_lap_time(self.context.last_lap)}.{best_str}"
            return "No lap time recorded yet. Complete a lap first."

        # Speed queries
        if any(kw in query_lower for kw in ("speed", "rpm")):
            return f"Currently {self.context.speed_kmh:.0f} km/h in gear {self.context.gear} at {self.context.rpm} RPM."

        return "Copy that. Monitoring the situation."

    def _build_pit_query_fallback_response(self) -> str:
        """Deterministic pit guidance from live context and thresholds."""
        if not self.context:
            return "Hold for one lap while I gather data, then ask again for pit timing."

        thresholds = self.telemetry_agent.thresholds if self.telemetry_agent else ThresholdsConfig()
        fuel_laps = self.context.get_fuel_laps_remaining()

        wear_values = list(self.context.tire_wear.values()) if self.context.tire_wear else []
        has_wear_data = any(wear > 0 for wear in wear_values)
        max_wear = max(wear_values) if wear_values else 0.0
        max_temp = max(self.context.tire_temps.values()) if self.context.tire_temps else 0.0
        total_damage = sum(self.context.car_damage.values()) if self.context.car_damage else 0.0

        if not math.isfinite(fuel_laps) or self.context.fuel_consumption_per_lap <= 0:
            if total_damage >= thresholds.car_damage_critical_total:
                parts = [f"{zone} {val:.0f}%" for zone, val in self.context.car_damage.items() if val > 0]
                return f"Heavy damage: {', '.join(parts)}. Box this lap for repairs."
            if has_wear_data and max_wear >= thresholds.tire_wear_critical:
                return f"Tire wear is critical at {max_wear:.0f} percent. Box this lap."
            if max_temp >= thresholds.tire_temp_critical:
                return f"Tire temperatures are critical at {max_temp:.0f} C. Box this lap."
            if total_damage >= thresholds.car_damage_warning_total:
                parts = [f"{zone} {val:.0f}%" for zone, val in self.context.car_damage.items() if val > 0]
                return f"Car damage: {', '.join(parts)}. Consider pitting for repairs."
            if has_wear_data and max_wear >= thresholds.tire_wear_warning:
                return f"Tire wear is high at {max_wear:.0f} percent. Pit window is open."
            if max_temp >= thresholds.tire_temp_warning:
                return f"Tire temperatures are high at {max_temp:.0f} C. Pit soon if they do not recover."
            return "Need one clean lap to calibrate fuel burn before I can call pit timing. Tires and damage look good, push on."

        # Critical checks first (box this lap)
        if fuel_laps <= thresholds.fuel_critical_laps:
            return f"Fuel critical, box this lap. About {int(fuel_laps)} laps remaining."

        if total_damage >= thresholds.car_damage_critical_total:
            parts = [f"{zone} {val:.0f}%" for zone, val in self.context.car_damage.items() if val > 0]
            return f"Heavy damage: {', '.join(parts)}. Box this lap for repairs."

        if has_wear_data and max_wear >= thresholds.tire_wear_critical:
            return f"Tire wear is critical at {max_wear:.0f} percent. Box this lap."

        if max_temp >= thresholds.tire_temp_critical:
            return f"Tire temperatures are critical at {max_temp:.0f} C. Box this lap."

        # Warning checks (pit soon)
        if fuel_laps <= thresholds.fuel_warning_laps:
            return f"Pit window open now. Fuel projects about {int(fuel_laps)} laps."

        if total_damage >= thresholds.car_damage_warning_total:
            parts = [f"{zone} {val:.0f}%" for zone, val in self.context.car_damage.items() if val > 0]
            return f"Car damage: {', '.join(parts)}. Consider pitting for repairs."

        if has_wear_data and max_wear >= thresholds.tire_wear_warning:
            return f"Pit window open on tires, max wear {max_wear:.0f} percent."

        if max_temp >= thresholds.tire_temp_warning:
            return f"Tire temperatures are high at {max_temp:.0f} C. Manage pace and plan a stop."

        return f"No stop needed yet. Fuel projects about {int(fuel_laps)} laps remaining."

    def _build_event_fallback_response(self, event: Event) -> str:
        """
        Build deterministic one-line race engineer responses for proactive events.

        This path intentionally avoids LLM invocation for event-triggered commentary.
        """
        data = event.data or {}
        event_type = event.type

        def _pos_label(code: str) -> str:
            return {
                "fl": "front left",
                "fr": "front right",
                "rl": "rear left",
                "rr": "rear right",
            }.get((code or "").lower(), "that tire")

        if event_type == "fuel_critical":
            laps = data.get("laps")
            if laps is not None:
                return f"Fuel critical, box this lap. About {laps:.1f} laps remaining."
            return "Fuel critical, box this lap."

        if event_type == "fuel_warning":
            laps = data.get("laps")
            if laps is not None:
                return f"Fuel is getting low, around {laps:.1f} laps remaining. Plan the stop."
            return "Fuel is getting low. Plan your stop."

        if event_type in {"tire_critical", "tire_warning"}:
            temp = data.get("temp")
            pos = _pos_label(data.get("position", ""))
            if temp is not None:
                if event_type == "tire_critical":
                    return f"{pos.title()} tire is critical at {temp:.0f} C. Back off and cool it now."
                return f"{pos.title()} tire temp high at {temp:.0f} C. Manage pace to cool it."
            return "Tire temperature warning. Manage pace and cool the tires."

        if event_type == "tire_wear_critical":
            wear = data.get("wear")
            pos = _pos_label(data.get("position", ""))
            if wear is not None:
                return f"{pos.title()} wear is critical at {wear:.0f} percent. Box this lap."
            return "Tire wear is critical. Box this lap."

        if event_type in {"wheel_slip_critical", "wheel_slip_warning"}:
            pos = _pos_label(data.get("position", ""))
            if event_type == "wheel_slip_critical":
                return f"Big slide on {pos}. Smooth throttle now."
            return f"Wheel slip on {pos}. Be progressive on throttle."

        if event_type == "gap_change":
            direction = data.get("direction", "ahead")
            new_gap = data.get("new_gap")
            change = data.get("change")
            if new_gap is not None:
                return f"Gap {direction} changed by {change:.1f}s. Current gap {new_gap:.2f}s."
            return f"Gap {direction} changed. Adjust pace."

        if event_type == "pit_window_open":
            reason = data.get("reason", "strategy")
            if reason == "fuel":
                return "Pit window open for fuel. Prepare to box soon."
            if reason == "tires":
                return "Pit window open for tires. Prepare to box soon."
            return "Pit window open. Prepare to box."

        if event_type == "opponent_close_behind":
            gap = data.get("gap")
            opp_pos = data.get("position")
            if gap is not None and opp_pos is not None:
                return f"Car behind in P{opp_pos}, gap {gap:.2f} seconds. Defend smart."
            if gap is not None:
                return f"Car close behind, gap {gap:.2f} seconds. Defend smart."
            return "Car close behind. Defend smart."

        if event_type == "car_damage_alert":
            total = data.get("total")
            severity = str(data.get("severity", "warning")).lower()
            zones = data.get("zones") or {}
            worst_zone = None
            worst_value = None
            if isinstance(zones, dict) and zones:
                worst_zone, worst_value = max(
                    zones.items(),
                    key=lambda item: float(item[1]),
                )
            severity_label = "Critical" if severity == "critical" else "Damage"
            if total is not None and worst_zone is not None and worst_value is not None and worst_value > 0:
                zone_name = str(worst_zone).replace("_", " ")
                return f"{severity_label} on {zone_name}, {worst_value:.0f} percent. Total damage {total:.0f}."
            if total is not None:
                return f"{severity_label} reported. Total damage {total:.0f} percent."
            return "Damage reported. Adjust risk."

        if event_type == "lap_complete":
            lap = data.get("lap")
            lap_time = data.get("time")
            if lap is not None and lap_time is not None:
                return f"Lap {lap} complete, {lap_time:.3f} seconds. Keep building."
            return "Lap complete. Keep building."

        return "Copy that. Monitoring."

    def stop(self):
        """Stop the worker thread."""
        logger.info("Stopping AI Race Engineer...")
        self._running = False
