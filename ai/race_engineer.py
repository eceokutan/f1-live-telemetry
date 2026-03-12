"""
AI Race Engineer Worker for F1 Telemetry Dashboard.

Integrates Eima's AI race engineer (TelemetryAgent + RaceEngineerAgent)
with the AC telemetry system. Runs in a separate QThread to avoid blocking the UI.

Receives telemetry samples, detects events, generates AI commentary.
"""

import asyncio
import logging
import re
import sys
from typing import Optional, Dict, Any
from PyQt5 import QtCore

# Add eima_ai to Python path
import os
import importlib.util

eima_path = os.path.join(os.path.dirname(__file__), '..', 'eima_ai')
sys.path.insert(0, os.path.abspath(eima_path))

# Import modules directly to avoid circular imports from __init__.py files
def import_from_file(module_name, file_path):
    """Import module directly from file path without triggering __init__.py"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import schemas
telemetry_module = import_from_file(
    "jarvis_granite.schemas.telemetry",
    os.path.join(eima_path, "jarvis_granite/schemas/telemetry.py")
)
events_module = import_from_file(
    "jarvis_granite.schemas.events",
    os.path.join(eima_path, "jarvis_granite/schemas/events.py")
)

# Import config
config_module = import_from_file(
    "config.config",
    os.path.join(eima_path, "config/config.py")
)

# Import live context
context_module = import_from_file(
    "jarvis_granite.live.context",
    os.path.join(eima_path, "jarvis_granite/live/context.py")
)

# Import LLM client
llm_module = import_from_file(
    "jarvis_granite.llm.llm_client",
    os.path.join(eima_path, "jarvis_granite/llm/llm_client.py")
)

# Import agents
telemetry_agent_module = import_from_file(
    "jarvis_granite.agents.telemetry_agent",
    os.path.join(eima_path, "jarvis_granite/agents/telemetry_agent.py")
)
race_engineer_module = import_from_file(
    "jarvis_granite.agents.race_engineer_agent",
    os.path.join(eima_path, "jarvis_granite/agents/race_engineer_agent.py")
)

# Extract classes
TelemetryData = telemetry_module.TelemetryData
TireTemps = telemetry_module.TireTemps
TireWear = telemetry_module.TireWear
TirePressure = telemetry_module.TirePressure
GForces = telemetry_module.GForces
Event = events_module.Event
TelemetryAgent = telemetry_agent_module.TelemetryAgent
RaceEngineerAgent = race_engineer_module.RaceEngineerAgent
LiveSessionContext = context_module.LiveSessionContext
LLMClient = llm_module.LLMClient
ThresholdsConfig = config_module.ThresholdsConfig


logger = logging.getLogger(__name__)


class AIRaceEngineerWorker(QtCore.QThread):
    """
    AI Race Engineer worker thread.

    Processes telemetry samples, detects events, and generates AI commentary
    using Eima's race engineer agents.

    Signals:
        ai_commentary(str message, str trigger, int priority) - AI-generated commentary
        status_update(str message) - Status messages for logging
    """

    ai_commentary = QtCore.pyqtSignal(str, str, int)  # message, trigger, priority
    driver_query_received = QtCore.pyqtSignal(str)  # driver query text (for UI display)
    status_update = QtCore.pyqtSignal(str)

    def __init__(
        self,
        huggingface_token: str,
        hf_model_id: str,
        track_name: str = "Unknown Track",
        session_id: str = "ac_session_001",
        verbosity: str = "minimal",
    ):
        """
        Initialize AI Race Engineer.

        Args:
            huggingface_token: Hugging Face access token
            hf_model_id: Hugging Face model id for the race engineer LLM
            track_name: Name of the track
            session_id: Unique session identifier
            verbosity: AI verbosity level (minimal, moderate, verbose)
        """
        super().__init__()

        # Configuration
        self.huggingface_token = huggingface_token
        self.hf_model_id = hf_model_id
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

            self.status_update.emit("AI Race Engineer ready")

            # Run async processing loop
            self._event_loop.run_until_complete(self._process_loop())

        except Exception as e:
            logger.error(f"AI Race Engineer error: {e}", exc_info=True)
            self.status_update.emit(f"AI Race Engineer error: {e}")
        finally:
            if self._event_loop:
                self._event_loop.close()
            self.status_update.emit("AI Race Engineer stopped")

    def _initialize_agents(self):
        """Initialize TelemetryAgent and RaceEngineerAgent."""
        # Create LLM client
        # max_tokens=75 for faster responses (~200-400ms savings over 150 tokens)
        # Racing comms should be brief anyway

        # Optional: support dedicated Hugging Face Inference Endpoint
        endpoint_url = os.getenv("HUGGINGFACE_ENDPOINT_URL", None)

        # Optional: support custom Hugging Face Space backend
        space_url = os.getenv("HUGGINGFACE_SPACE_URL", None)
        space_skip_ssl = os.getenv("HUGGINGFACE_SPACE_SKIP_SSL_VERIFY", "").lower() in ("1", "true", "yes")

        llm_client = LLMClient(
            huggingface_token=self.huggingface_token,
            model_id=self.hf_model_id,
            max_tokens=75,  # Latency optimization: shorter responses
            max_retries=2,
            endpoint_url=endpoint_url,
            space_url=space_url,
            space_skip_ssl_verify=space_skip_ssl,
        )

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

            # Update context
            self.context.update(telemetry)

            # Detect events
            events = self.telemetry_agent.detect_events(telemetry, self.context)

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
                response = await asyncio.wait_for(
                    self.race_engineer_agent.generate_reactive_response(
                        query=query,
                        context=self.context
                    ),
                    timeout=30.0  # 30 second timeout for LLM response
                )
                logger.debug(f"LLM response received: {response[:100]}...")
            except asyncio.TimeoutError:
                logger.error("LLM response timed out after 30 seconds")
                self.ai_commentary.emit(
                    "Sorry, the AI took too long to respond. Please try again.",
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

        # G-forces (optional)
        g_forces = GForces(
            lateral=0.0,  # AC doesn't provide this directly
            longitudinal=0.0
        )

        # Tire wear: AC's tyreWear = remaining life (100=fresh, decreases with wear)
        # eima_ai expects percent worn (0=fresh, 100=worn), so invert.
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

        lap_id = data.get("lap_id", 0)
        return TelemetryData(
            speed=data.get("speed", 0.0),
            rpms=data.get("rpms", 0),
            gear=data.get("gear", 0),
            throttle=data.get("throttle", 0.0),
            brake=data.get("brake", 0.0),
            fuel=data.get("fuel", None),
            tire_temps=tire_temps,
            tire_pressure=tire_pressure,
            tire_wear=tire_wear,
            car_damage=car_damage,
            x=data.get("x", 0.0),
            z=data.get("z", 0.0),
            lap_id=lap_id,
            lap_number=lap_id,  # AI uses lap_number, AC provides lap_id
            t=data.get("t", 0.0)
        )

    async def _handle_event(self, event: Event):
        """
        Handle detected event - generate AI response.

        Args:
            event: Detected event from TelemetryAgent
        """
        try:
            # Generate AI response
            response = await self.race_engineer_agent.handle_event(event, self.context)

            if response:
                # Clean LLM response (strip meta-commentary, references, prompt leakage)
                response = self._clean_llm_response(response)

                # Emit AI commentary signal
                self.ai_commentary.emit(response, event.type, event.priority.value)
                logger.info(f"AI commentary generated for {event.type}: {response[:50]}...")

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
        if self._event_loop and self._running:
            # Thread-safe: put telemetry in queue
            # If queue is full, drop oldest sample (non-blocking)
            try:
                self.telemetry_queue.put_nowait(telemetry_dict)
            except asyncio.QueueFull:
                # Queue full - drop this sample to prevent backlog
                # This is acceptable for real-time telemetry
                pass

    def process_driver_query(self, query: str):
        """
        Process driver query (called from voice input or main thread).

        Args:
            query: Driver's question/command as text
        """
        if self._event_loop and self._running:
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

    def stop(self):
        """Stop the worker thread."""
        logger.info("Stopping AI Race Engineer...")
        self._running = False
