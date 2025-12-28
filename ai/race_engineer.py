"""
AI Race Engineer Worker for F1 Telemetry Dashboard.

Integrates Eima's AI race engineer (TelemetryAgent + RaceEngineerAgent)
with the AC telemetry system. Runs in a separate QThread to avoid blocking the UI.

Receives telemetry samples, detects events, generates AI commentary.
"""

import asyncio
import logging
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
        watsonx_url: str,
        watsonx_project_id: str,
        watsonx_api_key: str,
        track_name: str = "Unknown Track",
        session_id: str = "ac_session_001",
        verbosity: str = "moderate"
    ):
        """
        Initialize AI Race Engineer.

        Args:
            watsonx_url: IBM WatsonX API URL
            watsonx_project_id: WatsonX project ID
            watsonx_api_key: WatsonX API key
            track_name: Name of the track
            session_id: Unique session identifier
            verbosity: AI verbosity level (minimal, moderate, verbose)
        """
        super().__init__()

        # Configuration
        self.watsonx_url = watsonx_url
        self.watsonx_project_id = watsonx_project_id
        self.watsonx_api_key = watsonx_api_key
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
            self.telemetry_queue = asyncio.Queue()
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
        llm_client = LLMClient(
            watsonx_url=self.watsonx_url,
            watsonx_project_id=self.watsonx_project_id,
            watsonx_api_key=self.watsonx_api_key,
            model_id="ibm/granite-3-8b-instruct",  # granite-4-h-small not available in this environment
            max_retries=2
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
        while self._running:
            # Create tasks for both telemetry and query processing
            tasks = [
                asyncio.create_task(self._process_telemetry_queue()),
                asyncio.create_task(self._process_query_queue())
            ]

            # Wait for either task with timeout
            try:
                done, pending = await asyncio.wait(
                    tasks,
                    timeout=1.0,
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()

            except Exception as e:
                logger.error(f"Error in process loop: {e}", exc_info=True)
                continue

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

        # Create TelemetryData
        return TelemetryData(
            speed=data.get("speed", 0.0),
            rpms=data.get("rpm", 0),
            gear=data.get("gear", 0),
            throttle=data.get("throttle", 0.0),
            brake=data.get("brake", 0.0),
            fuel=data.get("fuel", None),
            tire_temps=tire_temps,
            tire_pressure=tire_pressure,
            x=data.get("x", 0.0),
            z=data.get("z", 0.0),
            lap_id=data.get("lap_id", 0),
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
                # Emit AI commentary signal
                self.ai_commentary.emit(response, event.type, event.priority.value)
                logger.info(f"AI commentary generated for {event.type}: {response[:50]}...")

        except Exception as e:
            logger.error(f"Error generating AI response for {event.type}: {e}", exc_info=True)

    def process_telemetry(self, telemetry_dict: Dict[str, Any]):
        """
        Process incoming telemetry sample (called from main thread).

        Args:
            telemetry_dict: Telemetry data dictionary from AC worker
        """
        if self._event_loop and self._running:
            # Thread-safe: put telemetry in queue
            asyncio.run_coroutine_threadsafe(
                self.telemetry_queue.put(telemetry_dict),
                self._event_loop
            )

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

    def stop(self):
        """Stop the worker thread."""
        logger.info("Stopping AI Race Engineer...")
        self._running = False
