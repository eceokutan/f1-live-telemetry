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
from PyQt5 import QtWidgets

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
    if enable_ai and AI_AVAILABLE:
        print("🤖 Initializing AI Race Engineer...")

        # Load credentials from environment
        watsonx_url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        watsonx_project_id = os.getenv("WATSONX_PROJECT_ID", "")
        watsonx_api_key = os.getenv("WATSONX_API_KEY", "")

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
                ai_thread.status_update.connect(lambda msg: print(f"[AI] {msg}"))

                # Connect telemetry to AI worker
                if hasattr(telemetry_thread, 'realtime_sample'):
                    telemetry_thread.realtime_sample.connect(
                        lambda sample: ai_thread.process_telemetry(sample)
                    )

                # Start AI thread
                ai_thread.start()
                print("✅ AI Race Engineer started")

            except Exception as e:
                print(f"⚠️  Failed to initialize AI Race Engineer: {e}")
                import traceback
                traceback.print_exc()
                ai_thread = None
    elif enable_ai:
        print("⚠️  AI requested but AIRaceEngineerWorker module not available")

    # Start telemetry thread
    print("🚀 Starting telemetry worker thread...")
    telemetry_thread.start()

    # Show window
    print("🪟 Showing UI window...")
    window.show()

    print("\n" + "="*60)
    if ai_thread:
        print("✅ DASHBOARD READY - AI Race Engineer ACTIVE")
    else:
        print("✅ DASHBOARD READY - Check AC for shared memory connection")
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
