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
from typing import Optional, Dict, Any
from PyQt5 import QtCore

import os
from ai.race_engineer_core import (
    Event,
    GForces,
    LLMClient,
    LiveSessionContext,
    OpponentSnapshot,
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

        # Queues (initialized in run() after event loop is created)
        self.telemetry_queue: Optional[asyncio.Queue] = None
        self.query_queue: Optional[asyncio.Queue] = None

        logger.info("AIRaceEngineerWorker initialized")

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
        # Live mode targets short radio replies with low latency.
        live_max_tokens = int(os.getenv("LIVE_LLM_MAX_TOKENS", "24"))
        live_temperature = float(os.getenv("LIVE_LLM_TEMPERATURE", "0.3"))
        local_max_time_seconds = float(os.getenv("LOCAL_LLM_MAX_TIME_SECONDS", "5.0"))
        local_adapter_path = os.getenv("LOCAL_ADAPTER_PATH", "race_engineer_llm")

        # Check CUDA availability — local LLM requires CUDA.
        # If CUDA is unavailable, fall back to rule-based responses only.
        force_rule_based_fallback = False
        cuda_available = False
        cuda_device = ""
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                cuda_device = torch.cuda.get_device_name(0)
        except Exception as e:
            logger.warning("Unable to check CUDA status: %s", e)

        if cuda_available:
            logger.info("Local LLM CUDA active: %s", cuda_device or "unknown device")
        else:
            logger.warning(
                "CUDA is unavailable; falling back to rule-based responses."
            )
            self.status_update.emit(
                "CUDA unavailable. Using rule-based fallback responses."
            )
            force_rule_based_fallback = True

        llm_client = LLMClient(
            max_tokens=live_max_tokens,
            temperature=live_temperature,
            local_adapter_path=local_adapter_path,
            local_max_time_seconds=local_max_time_seconds,
            force_rule_based_fallback=force_rule_based_fallback,
        )
        # Wire status callback so LLM load/generation failures surface in the UI
        llm_client.set_status_callback(
            lambda msg: self.status_update.emit(msg)
        )

        if not force_rule_based_fallback:
            logger.info("Using local QLoRA-finetuned Granite model from: %s", local_adapter_path)
        else:
            logger.info("Rule-based responses only (no CUDA)")

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
                self.ai_commentary.emit(
                    "We're not on track right now. Get out there and I'll have your data ready.",
                    "driver_query", 2
                )
                return

            logger.info(f"Processing driver query: {query}")
            self.status_update.emit(f"Processing: \"{query}\"")

            # Generate AI response using reactive mode
            try:
                query_timeout_seconds = float(os.getenv("LIVE_LLM_QUERY_TIMEOUT_SECONDS", "12.0"))
                response = await asyncio.wait_for(
                    self.race_engineer_agent.generate_reactive_response(
                        query=query,
                        context=self.context
                    ),
                    timeout=query_timeout_seconds,
                )
                logger.debug(f"LLM response received: {response[:100]}...")
            except asyncio.TimeoutError:
                logger.error("LLM response timed out after %.1f seconds", query_timeout_seconds)
                fallback = self._build_timeout_fallback_response(query)
                self.ai_commentary.emit(
                    fallback,
                    "driver_query_timeout",
                    1
                )
                return
            except Exception as llm_error:
                logger.error(f"LLM error: {llm_error}", exc_info=True)
                raise

            # Clean LLM response (strip meta-commentary, references, prompt leakage)
            response = self._clean_llm_response(response)

            # Check for empty response and provide fallback
            if not response or not response.strip():
                logger.warning("LLM returned empty response, using fallback")
                response = "I heard your question but I'm having trouble generating a response right now. Could you please rephrase your question about your race situation?"

            # Emit as AI commentary with special trigger
            self.ai_commentary.emit(response, "driver_query", 2)  # MEDIUM priority
            logger.info(f"AI response to query: {response[:50]}...")

        except asyncio.TimeoutError:
            # No query, that's ok
            pass
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            self.ai_commentary.emit(
                f"Sorry, I couldn't process that question. Error: {str(e)}",
                "driver_query_error",
                1  # HIGH priority for errors
            )

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
            position=data.get("position"),
            gap_ahead=data.get("gap_ahead"),
            gap_behind=data.get("gap_behind"),
            opponents=opponents,
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

            if not self.context.can_send_proactive(min_interval):
                logger.debug("Skipping proactive message for %s - too soon", event.type)
                return

            response = self._build_event_fallback_response(event)
            response = self._clean_llm_response(response)
            if not response:
                response = "Copy that. Monitoring."

            self.context.mark_proactive_sent()
            self.context.add_exchange(f"[Event: {event.type}]", response)

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

    def _build_timeout_fallback_response(self, query: str) -> str:
        """
        Build a fast telemetry-based fallback when LLM query response times out.

        Keeps driver comms useful and concise under latency pressure.
        """
        if not self.context:
            return "No quick model reply. Ask again in a second."

        q = (query or "").lower()
        c = self.context

        if "wear" in q and "tire" in q:
            return (
                f"Tire wear is FL {c.tire_wear['fl']:.0f}, FR {c.tire_wear['fr']:.0f}, "
                f"RL {c.tire_wear['rl']:.0f}, RR {c.tire_wear['rr']:.0f} percent."
            )

        if "temp" in q and "tire" in q:
            return (
                f"Tire temps are FL {c.tire_temps['fl']:.0f}, FR {c.tire_temps['fr']:.0f}, "
                f"RL {c.tire_temps['rl']:.0f}, RR {c.tire_temps['rr']:.0f} C."
            )

        if "fuel" in q:
            fuel_laps = c.get_fuel_laps_remaining()
            if math.isfinite(fuel_laps):
                return f"Fuel is {c.fuel_remaining:.1f} liters, about {fuel_laps:.1f} laps remaining."
            return f"Fuel is {c.fuel_remaining:.1f} liters. Consumption estimate is not ready yet."

        if "damage" in q:
            total_damage = sum(c.car_damage.values())
            if total_damage <= 0:
                return "Car looks clean, no damage reported."
            return f"Damage detected, total around {total_damage:.0f} percent across zones."

        if "gap" in q or "ahead" in q or "behind" in q:
            gap_ahead = f"{c.gap_ahead:.2f}s" if c.gap_ahead is not None else "N/A"
            gap_behind = f"{c.gap_behind:.2f}s" if c.gap_behind is not None else "N/A"
            return f"Gap ahead {gap_ahead}, gap behind {gap_behind}."

        return (
            f"Model reply timed out. You're P{c.position}, fuel {c.fuel_remaining:.1f} liters, "
            f"speed {c.speed_kmh:.0f}."
        )

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
            slip = data.get("slip")
            pos = _pos_label(data.get("position", ""))
            if slip is not None:
                if event_type == "wheel_slip_critical":
                    return f"Critical wheel slip on {pos}, value {slip:.2f}. Smooth throttle now."
                return f"Wheel slip rising on {pos}, value {slip:.2f}. Be progressive on throttle."
            return "Wheel slip detected. Smooth your inputs."

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

        if event_type == "lap_complete":
            lap = data.get("lap")
            lap_time = data.get("time")
            if lap is not None and lap_time is not None:
                return f"Lap {lap} complete, {lap_time:.3f} seconds. Keep building."
            return "Lap complete. Keep building."

        if event_type == "sector_complete":
            sector = data.get("sector")
            sector_time = data.get("time")
            if sector is not None and sector_time is not None:
                return f"Sector {sector} complete in {sector_time:.3f} seconds."
            return "Sector complete."

        return "Copy that. Monitoring."

    def stop(self):
        """Stop the worker thread."""
        logger.info("Stopping AI Race Engineer...")
        self._running = False
