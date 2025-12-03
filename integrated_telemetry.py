# integrated_telemetry.py
import sys
from PyQt5 import QtWidgets

print("="*60)
print("🚀 F1 TELEMETRY DASHBOARD STARTING...")
print("="*60)

from dashboard import MainWindow
from telemetry.ac_shared_memory import AcTelemetryWorker
from telemetry.acc_backend import AccTelemetryWorker  # ← FIXED

print("✅ All modules imported successfully")


def main(game: str = "ac"):
    """
    Entry point for the telemetry dashboard.

    :param game: "ac" for Assetto Corsa, "acc" for Assetto Corsa Competizione
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
        telemetry_thread = AccTelemetryWorker(host="127.0.0.1", port=9232, password="")  # ← Fixed port
    else:
        raise ValueError(f"Unknown game '{game}'. Use 'ac' or 'acc'.")

    print("✅ Backend initialized")

    # Connect signals
    print("🔗 Connecting Qt signals...")
    telemetry_thread.lap_completed.connect(window.handle_lap_complete)
    telemetry_thread.status_update.connect(lambda msg: print(f"[Status] {msg}"))

    # Connect new signals (these exist in both backends now)
    if hasattr(telemetry_thread, 'session_info_update'):
        telemetry_thread.session_info_update.connect(window.update_session_info)
    if hasattr(telemetry_thread, 'live_data_update'):
        telemetry_thread.live_data_update.connect(window.update_live_data)
    if hasattr(telemetry_thread, 'realtime_sample'):
        telemetry_thread.realtime_sample.connect(window.handle_realtime_sample)

    print("✅ Signals connected")

    # Start telemetry thread
    print("🚀 Starting telemetry worker thread...")
    telemetry_thread.start()

    # Show window
    print("🪟 Showing UI window...")
    window.show()

    print("\n" + "="*60)
    print("✅ DASHBOARD READY - Check AC for shared memory connection")
    print("="*60 + "\n")

    # Run Qt event loop
    result = app.exec_()

    # Clean shutdown
    telemetry_thread.stop()
    telemetry_thread.wait()

    sys.exit(result)


if __name__ == "__main__":
    # Parse command line argument
    import sys
    game = "ac"
    if "--acc" in sys.argv:
        game = "acc"

    print(f"🎯 Command line args: {sys.argv}")
    print(f"🎮 Selected game: {game}\n")

    try:
        main(game)
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