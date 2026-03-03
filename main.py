#!/usr/bin/env python3
"""
F1 Telemetry Dashboard - Main Entry Point

Real-time telemetry visualization and AI race engineering for sim racing.
Supports Assetto Corsa and Assetto Corsa Competizione.

Usage:
    python main.py              # Run with Assetto Corsa
    python main.py --acc        # Run with ACC
    python main.py --ai         # Enable AI race engineer
    python main.py --acc --ai   # ACC with AI
"""
import sys
import os
import logging

# Import torch BEFORE PyQt5 to avoid DLL conflict on Windows
# (PyQt5 changes DLL search paths, breaking torch's c10.dll loading)
try:
    import torch
except ImportError:
    pass

from PyQt5 import QtWidgets

# Configure logging FIRST - before any other imports
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

logger.info("F1 Telemetry Dashboard starting")

# Import UI
from ui.main_window import MainWindow

# Import telemetry backends
from telemetry.backends.ac_backend import AcTelemetryWorker
from telemetry.backends.acc_backend import AccTelemetryWorker

# Try to import AI worker (optional)
try:
    from ai.race_engineer import AIRaceEngineerWorker
    AI_AVAILABLE = True
    logger.info("AI Race Engineer module available")
except ImportError as e:
    AI_AVAILABLE = False
    logger.warning("AI Race Engineer not available: %s", e)

# Try to import voice input worker (optional)
try:
    from ai.voice_input import VoiceInputWorker
    VOICE_AVAILABLE = True
    logger.info("Voice Input module available")
except ImportError as e:
    VOICE_AVAILABLE = False
    logger.warning("Voice Input not available: %s", e)

# Try to import TTS output worker (optional)
try:
    from ai.tts_output import TTSOutputWorker
    TTS_AVAILABLE = True
    logger.info("TTS Output module available")
except ImportError as e:
    TTS_AVAILABLE = False
    logger.warning("TTS Output not available: %s", e)

# Try to import session recorder (optional)
try:
    from data.session_recorder import SessionRecorder
    RECORDER_AVAILABLE = True
    logger.info("Session Recorder module available")
except ImportError as e:
    RECORDER_AVAILABLE = False
    logger.warning("Session Recorder not available: %s", e)

logger.info("All core modules imported")


def main(game: str = "ac", enable_ai: bool = False):
    """
    Entry point for the telemetry dashboard.

    Args:
        game: "ac" for Assetto Corsa, "acc" for Assetto Corsa Competizione
        enable_ai: Enable AI race engineer (requires IBM WatsonX credentials)
    """
    logger.info("Starting dashboard for: %s", game.upper())
    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()

    # Choose backend
    logger.info("Initializing %s telemetry backend", game.upper())
    if game == "ac":
        telemetry_thread = AcTelemetryWorker()
    elif game == "acc":
        telemetry_thread = AccTelemetryWorker(host="127.0.0.1", port=9232, password="")
    else:
        raise ValueError(f"Unknown game '{game}'. Use 'ac' or 'acc'.")

    # Connect signals
    telemetry_thread.lap_completed.connect(window.handle_lap_complete)
    telemetry_thread.status_update.connect(lambda msg: logger.info("Status: %s", msg))

    # Connect telemetry signals to UI (real-time, critical path)
    # IMPORTANT: UI visualization must receive ALL samples at full 60Hz rate
    # for smooth graphs and track map updates. Never throttle these connections.
    if hasattr(telemetry_thread, 'session_info_update'):
        telemetry_thread.session_info_update.connect(window.update_session_info)
    if hasattr(telemetry_thread, 'live_data_update'):
        telemetry_thread.live_data_update.connect(window.update_live_data)
    if hasattr(telemetry_thread, 'realtime_sample'):
        telemetry_thread.realtime_sample.connect(window.handle_realtime_sample)

    logger.info("Signals connected")

    # Initialize AI race engineer (optional)
    ai_thread = None
    voice_thread = None
    tts_thread = None
    if enable_ai and AI_AVAILABLE:
        logger.info("Initializing AI Race Engineer")

        # Load credentials from environment
        watsonx_url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        watsonx_project_id = os.getenv("WATSONX_PROJECT_ID", "")
        watsonx_api_key = os.getenv("WATSONX_API_KEY", "")
        watson_tts_api_key = os.getenv("WATSON_TTS_API_KEY", "")
        watson_tts_url = os.getenv("WATSON_TTS_URL", "")

        if not watsonx_api_key or not watsonx_project_id:
            logger.warning("AI Race Engineer requires WATSONX_API_KEY and WATSONX_PROJECT_ID in .env")
        else:
            try:
                ai_thread = AIRaceEngineerWorker(
                    watsonx_url=watsonx_url,
                    watsonx_project_id=watsonx_project_id,
                    watsonx_api_key=watsonx_api_key,
                    track_name="Unknown Track",
                    session_id="ac_session_001",
                    verbosity="moderate"
                )

                # Connect AI signals
                ai_thread.ai_commentary.connect(window.handle_ai_commentary)
                ai_thread.driver_query_received.connect(window.handle_driver_query)
                ai_thread.status_update.connect(lambda msg: logger.info("AI: %s", msg))

                # Connect telemetry to AI worker with throttling
                # AI doesn't need 60Hz telemetry - throttle to ~5Hz to avoid delays
                ai_sample_counter = [0]  # Mutable counter for lambda
                AI_SAMPLE_RATE = 12  # Process every 12th sample (60Hz / 12 = 5Hz)

                def throttled_ai_telemetry(sample):
                    """Send telemetry to AI at reduced rate to prevent delays."""
                    ai_sample_counter[0] += 1
                    if ai_sample_counter[0] >= AI_SAMPLE_RATE:
                        ai_sample_counter[0] = 0
                        # Process in background - never block UI thread
                        try:
                            ai_thread.process_telemetry(sample)
                        except Exception:
                            # Silently ignore AI processing errors to not affect visualization
                            pass

                if hasattr(telemetry_thread, 'realtime_sample'):
                    telemetry_thread.realtime_sample.connect(throttled_ai_telemetry)

                # Keep AI informed of AC status (on track / in menu / paused)
                if hasattr(telemetry_thread, 'live_data_update'):
                    telemetry_thread.live_data_update.connect(
                        lambda data: ai_thread.update_ac_status(data.get("ac_status", 0))
                    )

                # Start AI thread
                ai_thread.start()
                logger.info("AI Race Engineer started")

                # Initialize voice input (if available)
                if VOICE_AVAILABLE:
                    logger.info("Initializing Voice Input (faster-whisper)")
                    try:
                        voice_thread = VoiceInputWorker(
                            whisper_model_size="base"
                        )

                        # Connect voice signals
                        voice_thread.speech_detected.connect(ai_thread.process_driver_query)
                        voice_thread.vad_state_changed.connect(window.handle_vad_state_change)
                        voice_thread.status_update.connect(lambda msg: logger.info("Voice: %s", msg))
                        voice_thread.error_occurred.connect(lambda err: logger.error("Voice: %s", err))

                        # Start voice thread
                        voice_thread.start()
                        logger.info("Voice Input started")

                    except Exception as e:
                        logger.error("Failed to initialize Voice Input: %s", e, exc_info=True)
                        voice_thread = None
                else:
                    logger.warning("Voice Input module not available (missing dependencies)")

                # Initialize TTS output (if available and credentials present)
                if TTS_AVAILABLE and watson_tts_api_key and watson_tts_url:
                    logger.info("Initializing TTS Output (sentence pipelining mode)")
                    try:
                        tts_thread = TTSOutputWorker(
                            watson_api_key=watson_tts_api_key,
                            watson_url=watson_tts_url,
                            voice="en-GB_JamesV3Voice",  # British male race engineer
                            use_sentence_pipelining=True  # Latency optimization: ~500-1000ms savings
                        )

                        # Connect TTS signals
                        tts_thread.status_update.connect(lambda msg: logger.info("TTS: %s", msg))
                        tts_thread.error_occurred.connect(lambda err: logger.error("TTS: %s", err))

                        # Connect AI commentary to TTS playback
                        def on_ai_commentary_for_tts(msg, trigger, priority):
                            logger.debug("AI commentary for TTS: %s...", msg[:50])
                            tts_thread.speak(msg)

                        ai_thread.ai_commentary.connect(on_ai_commentary_for_tts)

                        # Pause voice input during TTS playback to prevent echo/feedback
                        if voice_thread:
                            tts_thread.playback_started.connect(voice_thread.pause)
                            tts_thread.playback_finished.connect(voice_thread.resume)

                        # Start TTS thread
                        tts_thread.start()
                        logger.info("TTS Output started")

                    except Exception as e:
                        logger.error("Failed to initialize TTS Output: %s", e, exc_info=True)
                        tts_thread = None
                elif TTS_AVAILABLE:
                    logger.warning("TTS Output requires WATSON_TTS_API_KEY and WATSON_TTS_URL in .env")
                else:
                    logger.warning("TTS Output module not available (missing dependencies)")

            except Exception as e:
                logger.error("Failed to initialize AI Race Engineer: %s", e, exc_info=True)
                ai_thread = None
    elif enable_ai:
        logger.warning("AI requested but AIRaceEngineerWorker module not available")

    # Initialize session recorder (optional)
    recorder_thread = None
    current_lap_number = [0]  # Track current lap in list (mutable for lambda)
    session_info = {"track": "", "car": "", "player": ""}  # Store session metadata

    if RECORDER_AVAILABLE:
        logger.info("Initializing Session Recorder")
        try:
            recorder_thread = SessionRecorder(db_path="data/telemetry_sessions.db")

            # Connect recorder signals
            recorder_thread.status_update.connect(lambda msg: logger.info("Recorder: %s", msg))
            recorder_thread.error_occurred.connect(lambda err: logger.error("Recorder: %s", err))

            # Start recorder thread
            recorder_thread.start()

            # Helper to start session when we get session info
            def on_session_info(info: dict):
                """Start recording session when we get track/car info."""
                session_info["track"] = info.get("track", "")
                session_info["car"] = info.get("car_model", "")  # AC sends "car_model"
                session_info["player"] = info.get("player_name", "")  # AC sends "player_name"

                # Start recording session
                if recorder_thread and not recorder_thread.session_id:
                    recorder_thread.start_session(
                        game=game,
                        track_name=session_info["track"],
                        car_model=session_info["car"],
                        player_name=session_info["player"],
                        ai_enabled=enable_ai
                    )

            # Connect to session info update
            if hasattr(telemetry_thread, 'session_info_update'):
                telemetry_thread.session_info_update.connect(on_session_info)

            # Record telemetry samples
            if hasattr(telemetry_thread, 'realtime_sample'):
                telemetry_thread.realtime_sample.connect(
                    lambda sample: recorder_thread.record_telemetry_sample(sample)
                )

            # Record completed laps
            def on_lap_complete(lap_id: int, samples: list):
                """Record lap completion with statistics."""
                if not samples:
                    return

                current_lap_number[0] = lap_id

                # Calculate lap statistics
                lap_time = samples[-1].get("t", 0.0) if samples else 0.0
                speeds = [s.get("speed", 0.0) for s in samples]
                avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
                max_speed = max(speeds) if speeds else 0.0
                min_speed = min(speeds) if speeds else 0.0

                fuel_start = samples[0].get("fuel", 0.0) if samples else 0.0
                fuel_end = samples[-1].get("fuel", 0.0) if samples else 0.0

                # Record lap
                if recorder_thread:
                    recorder_thread.record_lap(
                        lap_number=lap_id,
                        lap_time=lap_time,
                        fuel_start=fuel_start,
                        fuel_end=fuel_end,
                        avg_speed=avg_speed,
                        max_speed=max_speed,
                        min_speed=min_speed,
                        valid=True
                    )

            telemetry_thread.lap_completed.connect(on_lap_complete)

            # Record AI commentary (if AI enabled)
            if ai_thread:
                def on_ai_commentary(message: str, trigger: str, priority: str):
                    """Record AI commentary."""
                    if recorder_thread:
                        recorder_thread.record_ai_commentary(
                            message=message,
                            trigger=trigger,
                            priority=priority,
                            lap_number=current_lap_number[0]
                        )

                ai_thread.ai_commentary.connect(on_ai_commentary)

            # Record voice queries (if voice enabled)
            if voice_thread and ai_thread:
                # Track query/response pairs
                last_query = [""]  # Mutable for lambda

                def on_driver_query(query: str):
                    """Store driver query for later pairing with response."""
                    last_query[0] = query

                def on_query_response(message: str, trigger: str, priority: str):
                    """Record voice query/response pair."""
                    if recorder_thread and last_query[0] and trigger == "driver_query":
                        recorder_thread.record_voice_query(
                            query_text=last_query[0],
                            response_text=message,
                            lap_number=current_lap_number[0]
                        )
                        last_query[0] = ""  # Clear after recording

                ai_thread.driver_query_received.connect(on_driver_query)
                ai_thread.ai_commentary.connect(on_query_response)

            logger.info("Session Recorder started")

        except Exception as e:
            logger.error("Failed to initialize Session Recorder: %s", e, exc_info=True)
            recorder_thread = None
    else:
        logger.warning("Session Recorder module not available")

    # Start telemetry thread
    logger.info("Starting telemetry worker thread")
    telemetry_thread.start()

    # Show window
    window.show()

    if ai_thread and voice_thread and tts_thread:
        logger.info("Dashboard ready — AI Race Engineer + Voice I/O active")
    elif ai_thread and voice_thread:
        logger.info("Dashboard ready — AI Race Engineer + Voice Input active")
    elif ai_thread and tts_thread:
        logger.info("Dashboard ready — AI Race Engineer + TTS Output active")
    elif ai_thread:
        logger.info("Dashboard ready — AI Race Engineer active")
    else:
        logger.info("Dashboard ready — telemetry only")

    if recorder_thread:
        logger.info("Session Recording enabled — all data saved to database")

    # Run Qt event loop
    result = app.exec_()

    # Clean shutdown
    logger.info("Shutting down...")

    try:
        telemetry_thread.stop()
        telemetry_thread.wait(2000)  # 2 second timeout
    except Exception as e:
        logger.warning("Error stopping telemetry thread: %s", e)

    if ai_thread:
        try:
            ai_thread.stop()
            ai_thread.wait(2000)
        except Exception as e:
            logger.warning("Error stopping AI thread: %s", e)

    if voice_thread:
        try:
            voice_thread.stop()
            voice_thread.wait(2000)
        except Exception as e:
            logger.warning("Error stopping voice thread: %s", e)

    if tts_thread:
        try:
            tts_thread.stop()
            tts_thread.wait(2000)
        except Exception as e:
            logger.warning("Error stopping TTS thread: %s", e)

    if recorder_thread:
        try:
            recorder_thread.stop()
            recorder_thread.wait(2000)
        except Exception as e:
            logger.warning("Error stopping recorder thread: %s", e)

    logger.info("Goodbye!")
    sys.exit(result)


if __name__ == "__main__":
    # Parse command line arguments
    game = "ac"
    enable_ai = False

    if "--acc" in sys.argv:
        game = "acc"
    if "--ai" in sys.argv:
        enable_ai = True

    logger.info("Game: %s | AI: %s", game, enable_ai)

    try:
        main(game, enable_ai)
    except Exception as e:
        logger.critical("Fatal error: %s", e, exc_info=True)
        sys.exit(1)
