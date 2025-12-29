"""
TORCS Telemetry Backend
Connects to TORCS via UDP socket and uses torcs_jm_par driver
"""
import sys
import os
import time
import numpy as np
from PyQt5 import QtCore
from ..lap_buffer import LapBuffer

# Import TORCS client from same directory
from .torcs_jm_par import Client, drive_modular


class TorcsTelemetryWorker(QtCore.QThread):
    """Background thread that reads TORCS telemetry and uses torcs_jm_par driver"""
    lap_completed = QtCore.pyqtSignal(int, list)
    status_update = QtCore.pyqtSignal(str)
    session_info_update = QtCore.pyqtSignal(dict)
    live_data_update = QtCore.pyqtSignal(dict)
    realtime_sample = QtCore.pyqtSignal(dict)
    
    def __init__(self, port=3001, host='localhost', use_ai_driver=False, parent=None):
        super().__init__(parent)
        self.port = port
        self.host = host
        self.use_ai_driver = use_ai_driver
        self.running = False
        self.client = None
        
        # Lap tracking
        self.current_lap_id = 0
        self.last_dist_raced = 0
        
        # Track approximation for visualization
        self.track_length = 5000  # meters
        
    def run(self):
        print("\n" + "="*60)
        print("🏎️  TORCS TELEMETRY WORKER STARTING")
        print("="*60)
        self.status_update.emit("Connecting to TORCS...")
        
        # Create TORCS client from torcs_jm_par
        try:
            self.client = Client(p=self.port, H=self.host, parse_args=False)
            print(f"✅ Connected to TORCS on port {self.port}")
            self.status_update.emit("Connected! AI driver active...")
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            print("\n💡 Make sure TORCS is running:")
            print("   1. Start TORCS GUI")
            print("   2. Go to: Race -> Practice -> Configure Race")
            print("   3. Set driver to 'scr_server 1'")
            print("   4. Then: Race -> Practice -> New Race")
            self.status_update.emit(f"ERROR: Could not connect to TORCS on port {self.port}")
            return
        
        # Emit session info
        session_data = {
            "track": "TORCS Track",
            "track_config": "",
            "car_model": "TORCS Race Car",
            "player_name": "AI",
            "player_surname": "Driver",
            "player_nick": "UDP",
            "max_rpm": 10000,
            "max_fuel": 100,
        }
        self.session_info_update.emit(session_data)
        
        # Setup lap buffer
        lap_buffer = LapBuffer(
            on_lap_complete=lambda lap_id, samples: self.lap_completed.emit(lap_id, samples)
        )
        
        t0 = time.time()
        self.running = True
        frame_count = 0
        
        print("\n🏁 Starting telemetry loop (reading at ~60Hz)...")
        print("🤖 Using drive_modular from torcs_jm_par!\n")
        
        try:
            while self.running:
                # Get TORCS telemetry using Client
                self.client.get_servers_input()
                torcs_data = self.client.S.d
                
                elapsed = time.time() - t0
                
                # Detect lap changes
                dist_raced = torcs_data.get('distRaced', 0)
                if dist_raced < self.last_dist_raced - 100:
                    self.current_lap_id += 1
                    print(f"\n🏁 LAP {self.current_lap_id} COMPLETED!\n")
                self.last_dist_raced = dist_raced
                
                # Build sample
                sample = self._build_sample(torcs_data, elapsed)
                
                # Feed to lap buffer
                lap_buffer.add_sample(
                    lap_id=sample["lap_id"],
                    t=sample["t"],
                    x=sample["x"],
                    z=sample["z"],
                    speed_kmh=sample["speed"],
                    gear=sample["gear"],
                    rpms=sample["rpms"],
                    brake=sample["brake"],
                    throttle=sample["throttle"],
                    tyre_pressure_fl=sample["tyre_pressure_fl"],
                    tyre_pressure_fr=sample["tyre_pressure_fr"],
                    tyre_pressure_rl=sample["tyre_pressure_rl"],
                    tyre_pressure_rr=sample["tyre_pressure_rr"],
                    tyre_temp_fl=sample["tyre_temp_fl"],
                    tyre_temp_fr=sample["tyre_temp_fr"],
                    tyre_temp_rl=sample["tyre_temp_rl"],
                    tyre_temp_rr=sample["tyre_temp_rr"],
                )
                
                # Emit real-time sample for dashboard visualization
                self.realtime_sample.emit(sample)
                
                # Emit live data every 10 frames (~6 times/sec)
                frame_count += 1
                if frame_count % 10 == 0:
                    live_data = {
                        "current_lap": self.current_lap_id + 1,
                        "speed": sample["speed"],
                        "gear": sample["gear"],
                        "rpm": sample["rpms"],
                        "fuel": torcs_data.get('fuel', 0),
                        "position": int(torcs_data.get('racePos', 1)),
                        "is_in_pit": 0,
                        "current_time": "",
                        "last_time": "",
                        "best_time": "",
                    }
                    self.live_data_update.emit(live_data)
                
                # Use drive_modular from torcs_jm_par to control the car (if AI driver enabled)
                if self.use_ai_driver:
                    drive_modular(self.client)
                self.client.respond_to_server()
                
                # 60Hz loop
                time.sleep(1/60.0)
                
        except Exception as e:
            self.status_update.emit(f"Error: {str(e)}")
            print(f"❌ Error in telemetry loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.client:
                print("\n👋 Shutting down TORCS client...")
                self.client.shutdown()
            self.status_update.emit("Disconnected from TORCS")
    
    def _build_sample(self, torcs_data, elapsed_time):
        """Convert TORCS data to dashboard format"""
        speed_ms = torcs_data.get('speedX', 0)
        speed_kmh = speed_ms * 3.6
        
        dist = torcs_data.get('distFromStart', 0)
        trackPos = torcs_data.get('trackPos', 0)
        
        # Circular track approximation for visualization
        angle_on_track = (dist / self.track_length) * (2 * np.pi)
        radius = 200
        x_base = radius * np.cos(angle_on_track)
        z_base = radius * np.sin(angle_on_track)
        
        # Offset based on track position
        track_width = 15
        x_offset = -trackPos * track_width * np.sin(angle_on_track)
        z_offset = trackPos * track_width * np.cos(angle_on_track)
        
        return {
            "lap_id": self.current_lap_id,
            "t": elapsed_time,
            "x": x_base + x_offset,
            "z": z_base + z_offset,
            "speed": speed_kmh,
            "gear": int(torcs_data.get('gear', 0)),
            "rpms": int(torcs_data.get('rpm', 0)),
            "brake": 0.0,  # Not available in TORCS sensor data
            "throttle": 0.0,  # Not available in TORCS sensor data
            
            # Tire data not available - set to 0
            "tyre_pressure_fl": 0.0,
            "tyre_pressure_fr": 0.0,
            "tyre_pressure_rl": 0.0,
            "tyre_pressure_rr": 0.0,
            "tyre_temp_fl": 0.0,
            "tyre_temp_fr": 0.0,
            "tyre_temp_rl": 0.0,
            "tyre_temp_rr": 0.0,
        }
    
    def stop(self):
        """Stop the telemetry worker"""
        self.running = False
