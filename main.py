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
from PyQt5 import QtWidgets

# Configure logging FIRST - before any other imports
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(name)s: %(message)s',
    stream=sys.stdout
)

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

print("="*60)
print("🚀 F1 TELEMETRY DASHBOARD STARTING...")
print("="*60)

# Import UI
from ui.main_window import MainWindow

# Import telemetry backends
from telemetry.backends.ac_backend import AcTelemetryWorker
from telemetry.backends.acc_backend import AccTelemetryWorker

# Try to import AI worker (optional)
try:
    from ai.race_engineer import AIRaceEngineerWorker
    AI_AVAILABLE = True
    print("✅ AI Race Engineer module available")
except ImportError as e:
    AI_AVAILABLE = False
    print(f"⚠️  AI Race Engineer not available: {e}")

# Try to import voice input worker (optional)
try:
    from ai.voice_input import VoiceInputWorker
    VOICE_AVAILABLE = True
    print("✅ Voice Input module available")
except ImportError as e:
    VOICE_AVAILABLE = False
    print(f"⚠️  Voice Input not available: {e}")

# Try to import TTS output worker (optional)
try:
    from ai.tts_output import TTSOutputWorker
    TTS_AVAILABLE = True
    print("✅ TTS Output module available")
except ImportError as e:
    TTS_AVAILABLE = False
    print(f"⚠️  TTS Output not available: {e}")

# Try to import session recorder (optional)
try:
    from data.session_recorder import SessionRecorder
    RECORDER_AVAILABLE = True
    print("✅ Session Recorder module available")
except ImportError as e:
    RECORDER_AVAILABLE = False
    print(f"⚠️  Session Recorder not available: {e}")

print("✅ All core modules imported successfully")


def main(game: str = "ac", enable_ai: bool = False):
    """
    Entry point for the telemetry dashboard.

    Args:
        game: "ac" for Assetto Corsa, "acc" for Assetto Corsa Competizione
        enable_ai: Enable AI race engineer (requires IBM WatsonX credentials)
    """
    print(f"\n📋 Starting dashboard for: {game.upper()}")
    print("🔧 Creating Qt application...")
    app = QtWidgets.QApplication(sys.argv)

    print("🖥️  Creating main window...")
    window = MainWindow()

    # Choose backend
    print(f"🎮 Initializing {game.upper()} telemetry backend...")
    if game == "ac":
        telemetry_thread = AcTelemetryWorker()
    elif game == "acc":
        telemetry_thread = AccTelemetryWorker(host="127.0.0.1", port=9232, password="")
    else:
        raise ValueError(f"Unknown game '{game}'. Use 'ac' or 'acc'.")

    print("✅ Backend initialized")

    # Connect signals
    print("🔗 Connecting Qt signals...")
    telemetry_thread.lap_completed.connect(window.handle_lap_complete)
    telemetry_thread.status_update.connect(lambda msg: print(f"[Status] {msg}"))

    # Connect telemetry signals
    if hasattr(telemetry_thread, 'session_info_update'):
        telemetry_thread.session_info_update.connect(window.update_session_info)
    if hasattr(telemetry_thread, 'live_data_update'):
        telemetry_thread.live_data_update.connect(window.update_live_data)
    if hasattr(telemetry_thread, 'realtime_sample'):
        telemetry_thread.realtime_sample.connect(window.handle_realtime_sample)

    print("✅ Signals connected")

    # Initialize AI race engineer (optional)
    ai_thread = None
    voice_thread = None
    tts_thread = None
    if enable_ai and AI_AVAILABLE:
        print("🤖 Initializing AI Race Engineer...")

        # Load credentials from environment
        watsonx_url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        watsonx_project_id = os.getenv("WATSONX_PROJECT_ID", "")
        watsonx_api_key = os.getenv("WATSONX_API_KEY", "")
        watson_stt_api_key = os.getenv("WATSON_STT_API_KEY", "")
        watson_stt_url = os.getenv("WATSON_STT_URL", "")
        watson_tts_api_key = os.getenv("WATSON_TTS_API_KEY", "")
        watson_tts_url = os.getenv("WATSON_TTS_URL", "")

        if not watsonx_api_key or not watsonx_project_id:
            print("⚠️  AI Race Engineer requires WATSONX_API_KEY and WATSONX_PROJECT_ID")
            print("⚠️  Skipping AI initialization. Set these in .env file to enable AI.")
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
                ai_thread.status_update.connect(lambda msg: print(f"[AI] {msg}"))

                # Connect telemetry to AI worker
                if hasattr(telemetry_thread, 'realtime_sample'):
                    telemetry_thread.realtime_sample.connect(
                        lambda sample: ai_thread.process_telemetry(sample)
                    )

                # Start AI thread
                ai_thread.start()
                print("✅ AI Race Engineer started")

                # Initialize voice input (if available and credentials present)
                if VOICE_AVAILABLE and watson_stt_api_key and watson_stt_url:
                    print("🎤 Initializing Voice Input...")
                    try:
                        voice_thread = VoiceInputWorker(
                            watson_api_key=watson_stt_api_key,
                            watson_url=watson_stt_url,
                            model="en-US_BroadbandModel"
                        )

                        # Connect voice signals
                        voice_thread.speech_detected.connect(ai_thread.process_driver_query)
                        voice_thread.vad_state_changed.connect(window.handle_vad_state_change)
                        voice_thread.status_update.connect(lambda msg: print(f"[Voice] {msg}"))
                        voice_thread.error_occurred.connect(lambda err: print(f"[Voice Error] {err}"))

                        # Start voice thread
                        voice_thread.start()
                        print("✅ Voice Input started")

                    except Exception as e:
                        print(f"⚠️  Failed to initialize Voice Input: {e}")
                        import traceback
                        traceback.print_exc()
                        voice_thread = None
                elif VOICE_AVAILABLE:
                    print("⚠️  Voice Input requires WATSON_STT_API_KEY and WATSON_STT_URL in .env")
                else:
                    print("⚠️  Voice Input module not available (missing dependencies)")

                # Initialize TTS output (if available and credentials present)
                if TTS_AVAILABLE and watson_tts_api_key and watson_tts_url:
                    print("🔊 Initializing TTS Output...")
                    try:
                        tts_thread = TTSOutputWorker(
                            watson_api_key=watson_tts_api_key,
                            watson_url=watson_tts_url,
                            voice="en-GB_JamesV3Voice"  # British male race engineer
                        )

                        # Connect TTS signals
                        tts_thread.status_update.connect(lambda msg: print(f"[TTS] {msg}"))
                        tts_thread.error_occurred.connect(lambda err: print(f"[TTS Error] {err}"))

                        # Connect AI commentary to TTS playback
                        ai_thread.ai_commentary.connect(lambda msg, _trigger, _priority: tts_thread.speak(msg))

                        # Pause voice input during TTS playback to prevent echo/feedback
                        if voice_thread:
                            tts_thread.playback_started.connect(voice_thread.pause)
                            tts_thread.playback_finished.connect(voice_thread.resume)

                        # Start TTS thread
                        tts_thread.start()
                        print("✅ TTS Output started")

                    except Exception as e:
                        print(f"⚠️  Failed to initialize TTS Output: {e}")
                        import traceback
                        traceback.print_exc()
                        tts_thread = None
                elif TTS_AVAILABLE:
                    print("⚠️  TTS Output requires WATSON_TTS_API_KEY and WATSON_TTS_URL in .env")
                else:
                    print("⚠️  TTS Output module not available (missing dependencies)")

            except Exception as e:
                print(f"⚠️  Failed to initialize AI Race Engineer: {e}")
                import traceback
                traceback.print_exc()
                ai_thread = None
    elif enable_ai:
        print("⚠️  AI requested but AIRaceEngineerWorker module not available")

    # Initialize session recorder (optional)
    recorder_thread = None
    current_lap_number = [0]  # Track current lap in list (mutable for lambda)
    session_info = {"track": "", "car": "", "player": ""}  # Store session metadata

    if RECORDER_AVAILABLE:
        print("💾 Initializing Session Recorder...")
        try:
            recorder_thread = SessionRecorder(db_path="data/telemetry_sessions.db")

            # Connect recorder signals
            recorder_thread.status_update.connect(lambda msg: print(f"[Recorder] {msg}"))
            recorder_thread.error_occurred.connect(lambda err: print(f"[Recorder Error] {err}"))

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

            print("✅ Session Recorder started")

        except Exception as e:
            print(f"⚠️  Failed to initialize Session Recorder: {e}")
            import traceback
            traceback.print_exc()
            recorder_thread = None
    else:
        print("⚠️  Session Recorder module not available")

    # Start telemetry thread
    print("🚀 Starting telemetry worker thread...")
    telemetry_thread.start()

    # Show window
    print("🪟 Showing UI window...")
    window.show()

    print("\n" + "="*60)
    if ai_thread and voice_thread and tts_thread:
        print("✅ DASHBOARD READY - AI Race Engineer + Voice I/O ACTIVE")
    elif ai_thread and voice_thread:
        print("✅ DASHBOARD READY - AI Race Engineer + Voice Input ACTIVE")
    elif ai_thread and tts_thread:
        print("✅ DASHBOARD READY - AI Race Engineer + TTS Output ACTIVE")
    elif ai_thread:
        print("✅ DASHBOARD READY - AI Race Engineer ACTIVE")
    else:
        print("✅ DASHBOARD READY - Check AC for shared memory connection")

    if recorder_thread:
        print("💾 Session Recording ENABLED - All data will be saved to database")

    print("="*60 + "\n")

    # Run Qt event loop
    result = app.exec_()

    # Clean shutdown
    print("\n🛑 Shutting down...")
    telemetry_thread.stop()
    telemetry_thread.wait()

    if ai_thread:
        ai_thread.stop()
        ai_thread.wait()

    if voice_thread:
        voice_thread.stop()
        voice_thread.wait()

    if tts_thread:
        tts_thread.stop()
        tts_thread.wait()

    if recorder_thread:
        recorder_thread.stop()
        recorder_thread.wait()

    print("👋 Goodbye!")
    sys.exit(result)


if __name__ == "__main__":
    # Parse command line arguments
    game = "ac"
    enable_ai = False

    if "--acc" in sys.argv:
        game = "acc"
    if "--ai" in sys.argv:
        enable_ai = True

    print(f"🎯 Command line args: {sys.argv}")
    print(f"🎮 Selected game: {game}")
    print(f"🤖 AI enabled: {enable_ai}\n")

    try:
        main(game, enable_ai)
    except Exception as e:
        print("\n" + "="*60)
        print("❌ FATAL ERROR:")
        print("="*60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        print("="*60)
        sys.exit(1)
