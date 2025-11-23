# telemetry/acc_udp.py
"""
Assetto Corsa Competizione telemetry backend (skeleton).

ACC provides telemetry via a UDP "broadcasting" interface.
This file defines an AccTelemetryWorker that:

- Listens on a UDP socket
- Parses incoming packets into normalized sample dicts
- Uses LapBuffer to detect lap boundaries
- Emits lap_completed & status_update signals compatible with the UI

NOTE: The actual ACC packet parsing is left as a TODO. You should fill
      `parse_acc_packet()` with logic from the official ACC broadcasting SDK.
"""

import socket
import time
from typing import Dict, Any, Optional

from PyQt5 import QtCore

from .lap_buffer import LapBuffer


class AccTelemetryWorker(QtCore.QThread):
    """
    Assetto Corsa Competizione telemetry adapter (UDP skeleton).

    Once you implement parse_acc_packet(), this will behave similarly to
    the AcTelemetryWorker: it will emit lap_completed(lap_id, samples)
    using the same sample dict format that the dashboard already expects.
    """

    lap_completed = QtCore.pyqtSignal(int, list)  # (lap_id, samples: List[Dict])
    status_update = QtCore.pyqtSignal(str)        # status message

    def __init__(self, host: str = "127.0.0.1", port: int = 9000, parent=None):
        """
        :param host: Host/IP to bind to for receiving ACC broadcast packets.
                     Usually 127.0.0.1 if ACC is on the same machine.
        :param port: UDP port that ACC is configured to broadcast on.
                     Default here is 9000 (you must match ACC's settings).
        """
        super().__init__(parent)
        self.host = host
        self.port = port
        self.running = False

        self._sock: Optional[socket.socket] = None

    # ------------------ Core QThread loop ------------------ #

    def run(self):
        self.status_update.emit(
            f"Starting ACC telemetry listener on {self.host}:{self.port}..."
        )

        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind((self.host, self.port))
            self._sock.settimeout(1.0)  # 1-second timeout to allow clean shutdown
        except Exception as e:
            self.status_update.emit(f"ERROR: Could not open UDP socket: {e}")
            return

        self.status_update.emit("ACC UDP socket ready. Make sure ACC is broadcasting.")

        # LapBuffer with callback that emits Qt signal
        lap_buffer = LapBuffer(
            on_lap_complete=lambda lap_id, samples: self.lap_completed.emit(lap_id, samples)
        )

        t0 = time.time()
        self.running = True

        try:
            while self.running:
                try:
                    data, addr = self._sock.recvfrom(2048)  # buffer size is arbitrary
                except socket.timeout:
                    # Periodic timeout just so we can check self.running
                    continue
                except OSError:
                    # Socket closed during shutdown
                    break

                now = time.time()
                elapsed = now - t0

                # Parse raw ACC packet into a normalized dict
                try:
                    parsed = parse_acc_packet(data, elapsed)
                except NotImplementedError:
                    # Parsing not implemented yet: warn once and bail out
                    self.status_update.emit(
                        "parse_acc_packet() not implemented. "
                        "Fill this in using the ACC broadcasting SDK."
                    )
                    break
                except Exception as e:
                    self.status_update.emit(f"ACC parse error: {e}")
                    continue

                if parsed is None:
                    # Packet type we don't care about
                    continue

                # Expected keys in 'parsed':
                #   lap_id, t, x, z, speed_kmh, gear, rpms, brake, throttle
                lap_buffer.add_sample(
                    lap_id=parsed["lap_id"],
                    t=parsed["t"],
                    x=parsed["x"],
                    z=parsed["z"],
                    speed_kmh=parsed["speed_kmh"],
                    gear=parsed.get("gear", 0),
                    rpms=parsed.get("rpms", 0),
                    brake=parsed.get("brake", 0.0),
                    throttle=parsed.get("throttle", 0.0),
                )

        finally:
            if self._sock is not None:
                try:
                    self._sock.close()
                except Exception:
                    pass
            self.status_update.emit("ACC telemetry listener stopped.")

    def stop(self):
        self.running = False
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass


# ------------------ Packet parsing (TODO) ------------------ #

def parse_acc_packet(data: bytes, t: float) -> Optional[Dict[str, Any]]:
    """
    Parse a raw ACC UDP packet into a normalized telemetry dict.

    This function MUST be implemented using the official ACC broadcasting
    protocol documentation. For now, it's a stub so you don't accidentally
    think it's working when it's not.

    Expected return format (dict):
        {
            "t": t,                    # float, seconds since start (we pass it in)
            "lap_id": int,             # some lap counter / index
            "x": float,                # world X (meters)
            "z": float,                # world Z (meters)
            "speed_kmh": float,        # km/h
            "gear": int,               # as game encodes it
            "rpms": int,               # engine RPM
            "brake": float,            # 0.0–1.0
            "throttle": float,         # 0.0–1.0
        }

    If the packet is a type you don't care about (e.g. track info, session info),
    return None so the worker can just skip it.
    """
    # TODO: Implement using ACC broadcasting SDK / documentation
    raise NotImplementedError("ACC packet parsing not implemented yet.")
