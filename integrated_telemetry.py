# integrated_telemetry.py
import sys
from PyQt5 import QtWidgets

from dashboard import MainWindow
from telemetry.ac_shared_memory import AcTelemetryWorker
from telemetry.acc_backend import AccTelemetryWorker  # ← FIXED


def main(game: str = "ac"):
    """
    Entry point for the telemetry dashboard.

    :param game: "ac" for Assetto Corsa, "acc" for Assetto Corsa Competizione
    """
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()

    # Choose backend
    if game == "ac":
        telemetry_thread = AcTelemetryWorker()
    elif game == "acc":
        telemetry_thread = AccTelemetryWorker(host="127.0.0.1", port=9232, password="")  # ← Fixed port
    else:
        raise ValueError(f"Unknown game '{game}'. Use 'ac' or 'acc'.")

    # Connect signals
    telemetry_thread.lap_completed.connect(window.handle_lap_complete)
    telemetry_thread.status_update.connect(lambda msg: print(f"[Status] {msg}"))
    
    # Connect new signals (these exist in both backends now)
    if hasattr(telemetry_thread, 'session_info_update'):
        telemetry_thread.session_info_update.connect(window.update_session_info)
    if hasattr(telemetry_thread, 'live_data_update'):
        telemetry_thread.live_data_update.connect(window.update_live_data)

    # Start telemetry thread
    telemetry_thread.start()

    # Show window
    window.show()

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
    
    main(game)