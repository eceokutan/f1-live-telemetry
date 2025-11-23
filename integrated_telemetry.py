# integrated_telemetry.py
import sys

from PyQt5 import QtWidgets

from dashboard import MainWindow
from telemetry.ac_shared_memory import AcTelemetryWorker
from telemetry.acc_udp import AccTelemetryWorker  # new import

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
        telemetry_thread = AccTelemetryWorker(host="127.0.0.1", port=9000, password="")
    else:
        raise ValueError(f"Unknown game '{game}'. Use 'ac' or 'acc'.")

    # Connect signals
    telemetry_thread.lap_completed.connect(window.handle_lap_complete)
    telemetry_thread.status_update.connect(lambda msg: print(f"[Status] {msg}"))

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
    # For now, default to AC. You can later change this or parse CLI args.
    main("ac")
