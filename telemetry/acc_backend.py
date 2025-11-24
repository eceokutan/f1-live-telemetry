import socket
import struct
import time
from typing import Dict, Any, List
from PyQt5 import QtCore
from enum import IntEnum


# ===================== ACC BROADCASTING PROTOCOL ENUMS =====================

class InboundMessageType(IntEnum):
    REGISTRATION_RESULT = 1
    REALTIME_UPDATE = 2
    REALTIME_CAR_UPDATE = 3
    ENTRY_LIST = 4
    ENTRY_LIST_CAR = 6
    BROADCASTING_EVENT = 5
    TRACK_DATA = 7


class OutboundMessageType(IntEnum):
    REGISTER_COMMAND_APPLICATION = 1
    UNREGISTER_COMMAND_APPLICATION = 9
    REQUEST_ENTRY_LIST = 10
    REQUEST_TRACK_DATA = 11
    CHANGE_HUD_PAGE = 49
    CHANGE_FOCUS = 50
    INSTANT_REPLAY_REQUEST = 51
    PLAY_MANUAL_REPLAY_HIGHLIGHT = 52
    SET_FOCUS = 53
    SET_CAMERA = 54


class LapType(IntEnum):
    ERROR = 0
    OUTLAP = 1
    REGULAR = 2
    INLAP = 3


# ===================== ACC PACKET PARSER =====================

class AccPacketParser:
    """Parses ACC broadcasting UDP packets"""
    
    @staticmethod
    def read_string(data: bytes, offset: int) -> tuple:
        """Read UTF-8 string from binary data. Returns (string, new_offset)"""
        if offset + 2 > len(data):
            return "", offset
        
        length = struct.unpack_from('<H', data, offset)[0]
        offset += 2
        
        if offset + length > len(data):
            return "", offset
            
        string = data[offset:offset + length].decode('utf-8', errors='ignore')
        offset += length
        return string, offset
    
    @staticmethod
    def parse_registration_result(data: bytes) -> Dict[str, Any]:
        """Parse registration result packet"""
        if len(data) < 5:
            return {"success": False, "error": "Packet too short"}
        
        connection_id = struct.unpack_from('<I', data, 1)[0]
        success = struct.unpack_from('<B', data, 5)[0] == 1
        is_readonly = struct.unpack_from('<B', data, 6)[0] == 1 if len(data) > 6 else False
        
        error_msg, _ = AccPacketParser.read_string(data, 7) if len(data) > 7 else ("", 7)
        
        return {
            "success": success,
            "connection_id": connection_id,
            "readonly": is_readonly,
            "error": error_msg
        }
    
    @staticmethod
    def parse_realtime_update(data: bytes) -> Dict[str, Any]:
        """Parse realtime update packet (session info)"""
        if len(data) < 100:
            return {}
        
        offset = 1  # Skip message type
        
        event_index = struct.unpack_from('<H', data, offset)[0]
        offset += 2
        
        session_index = struct.unpack_from('<H', data, offset)[0]
        offset += 2
        
        session_type = struct.unpack_from('<B', data, offset)[0]
        offset += 1
        
        phase = struct.unpack_from('<B', data, offset)[0]
        offset += 1
        
        session_time_ms = struct.unpack_from('<f', data, offset)[0]
        offset += 4
        
        session_end_time_ms = struct.unpack_from('<f', data, offset)[0]
        offset += 4
        
        focused_car_index = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        active_camera_set, offset = AccPacketParser.read_string(data, offset)
        active_camera, offset = AccPacketParser.read_string(data, offset)
        current_hud_page, offset = AccPacketParser.read_string(data, offset)
        
        return {
            "event_index": event_index,
            "session_index": session_index,
            "session_type": session_type,
            "phase": phase,
            "session_time_ms": session_time_ms,
            "session_end_time_ms": session_end_time_ms,
            "focused_car_index": focused_car_index,
        }
    
    @staticmethod
    def parse_realtime_car_update(data: bytes) -> Dict[str, Any]:
        """Parse realtime car update packet (telemetry for one car)"""
        if len(data) < 100:
            return {}
        
        offset = 1  # Skip message type
        
        car_index = struct.unpack_from('<H', data, offset)[0]
        offset += 2
        
        driver_index = struct.unpack_from('<H', data, offset)[0]
        offset += 2
        
        driver_count = struct.unpack_from('<B', data, offset)[0]
        offset += 1
        
        gear = struct.unpack_from('<B', data, offset)[0] - 1  # ACC sends gear+1
        offset += 1
        
        # World position (X, Y, Z)
        world_pos_x = struct.unpack_from('<f', data, offset)[0]
        offset += 4
        world_pos_y = struct.unpack_from('<f', data, offset)[0]
        offset += 4
        world_pos_z = struct.unpack_from('<f', data, offset)[0]
        offset += 4
        
        # Velocity (m/s)
        velocity = struct.unpack_from('<f', data, offset)[0]
        offset += 4
        
        # Position in race
        position = struct.unpack_from('<H', data, offset)[0]
        offset += 2
        
        # Track position (spline, 0-1)
        track_position = struct.unpack_from('<f', data, offset)[0]
        offset += 4
        
        # Spline position (normalized lap position)
        spline_position = struct.unpack_from('<f', data, offset)[0]
        offset += 4
        
        # Lap count
        laps = struct.unpack_from('<H', data, offset)[0]
        offset += 2
        
        # Delta
        delta = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        # Best session lap
        best_session_lap_ms = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        # Last lap time
        last_lap_ms = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        # Current lap time
        current_lap_ms = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        # Skip car location (1 byte)
        offset += 1
        
        # Kmh
        kmh = struct.unpack_from('<H', data, offset)[0]
        offset += 2
        
        # Cup position
        cup_position = struct.unpack_from('<H', data, offset)[0]
        offset += 2
        
        # Track status (probably)
        track_status = struct.unpack_from('<B', data, offset)[0]
        offset += 1
        
        return {
            "car_index": car_index,
            "driver_index": driver_index,
            "gear": gear,
            "world_pos_x": world_pos_x,
            "world_pos_y": world_pos_y,
            "world_pos_z": world_pos_z,
            "velocity": velocity,
            "position": position,
            "track_position": track_position,
            "spline_position": spline_position,
            "laps": laps,
            "delta": delta,
            "best_session_lap_ms": best_session_lap_ms,
            "last_lap_ms": last_lap_ms,
            "current_lap_ms": current_lap_ms,
            "kmh": kmh,
            "cup_position": cup_position,
        }
    
    @staticmethod
    def parse_track_data(data: bytes) -> Dict[str, Any]:
        """Parse track data packet"""
        if len(data) < 10:
            return {}
        
        offset = 1
        connection_id = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        track_name, offset = AccPacketParser.read_string(data, offset)
        track_id = struct.unpack_from('<I', data, offset)[0] if offset + 4 <= len(data) else 0
        offset += 4
        
        track_meters = struct.unpack_from('<I', data, offset)[0] if offset + 4 <= len(data) else 0
        offset += 4
        
        camera_set_count = struct.unpack_from('<B', data, offset)[0] if offset + 1 <= len(data) else 0
        offset += 1
        
        # Skip camera sets for now
        
        return {
            "track_name": track_name,
            "track_id": track_id,
            "track_meters": track_meters,
        }
    
    @staticmethod
    def parse_entry_list_car(data: bytes) -> Dict[str, Any]:
        """Parse entry list car packet"""
        if len(data) < 10:
            return {}
        
        offset = 1
        car_id = struct.unpack_from('<H', data, offset)[0]
        offset += 2
        
        car_model_type = struct.unpack_from('<B', data, offset)[0]
        offset += 1
        
        team_name, offset = AccPacketParser.read_string(data, offset)
        race_number = struct.unpack_from('<I', data, offset)[0]
        offset += 4
        
        cup_category = struct.unpack_from('<B', data, offset)[0]
        offset += 1
        
        current_driver_index = struct.unpack_from('<B', data, offset)[0]
        offset += 1
        
        # Driver count
        driver_count = struct.unpack_from('<B', data, offset)[0] if offset < len(data) else 0
        offset += 1
        
        drivers = []
        for _ in range(driver_count):
            if offset >= len(data):
                break
            first_name, offset = AccPacketParser.read_string(data, offset)
            last_name, offset = AccPacketParser.read_string(data, offset)
            short_name, offset = AccPacketParser.read_string(data, offset)
            category = struct.unpack_from('<B', data, offset)[0] if offset < len(data) else 0
            offset += 1
            nationality = struct.unpack_from('<H', data, offset)[0] if offset + 2 <= len(data) else 0
            offset += 2
            
            drivers.append({
                "first_name": first_name,
                "last_name": last_name,
                "short_name": short_name,
            })
        
        return {
            "car_id": car_id,
            "car_model_type": car_model_type,
            "team_name": team_name,
            "race_number": race_number,
            "drivers": drivers,
        }


# ===================== ACC TELEMETRY WORKER =====================

class AccTelemetryWorker(QtCore.QThread):
    """Background thread that connects to ACC via UDP broadcasting"""
    lap_completed = QtCore.pyqtSignal(int, list)
    status_update = QtCore.pyqtSignal(str)
    session_info_update = QtCore.pyqtSignal(dict)
    live_data_update = QtCore.pyqtSignal(dict)
    realtime_sample = QtCore.pyqtSignal(dict)  # realtime telemetry sample (every frame)

    def __init__(self, host="127.0.0.1", port=9000, password="", display_name="PythonTelemetry"):
        super().__init__()
        self.host = host
        self.port = port
        self.password = password
        self.display_name = display_name
        self.running = False
        self.connection_id = None
        self.focused_car_index = 0
        self.car_data = {}
        self.track_info = {}
        self.current_lap_samples = {}  # car_index -> [samples]
        self.last_lap_count = {}  # car_index -> lap_count
        
    def run(self):
        self.status_update.emit("Connecting to ACC...")
        
        # Create UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
        sock.bind(('', self.port))
        
        # Send registration request
        self._send_registration(sock)
        
        self.running = True
        registered = False
        last_realtime = time.time()
        
        try:
            while self.running:
                try:
                    data, addr = sock.recvfrom(2048)
                    
                    if len(data) < 1:
                        continue
                    
                    msg_type = data[0]
                    
                    if msg_type == InboundMessageType.REGISTRATION_RESULT:
                        result = AccPacketParser.parse_registration_result(data)
                        if result.get("success"):
                            self.connection_id = result["connection_id"]
                            registered = True
                            self.status_update.emit("Connected to ACC!")
                            # Request track data and entry list
                            self._send_request_track_data(sock)
                            self._send_request_entry_list(sock)
                        else:
                            self.status_update.emit(f"Connection failed: {result.get('error', 'Unknown error')}")
                    
                    elif msg_type == InboundMessageType.TRACK_DATA:
                        track_data = AccPacketParser.parse_track_data(data)
                        self.track_info = track_data
                        # Emit session info with track data
                        session_data = {
                            "track": track_data.get("track_name", "Unknown"),
                            "track_config": "",
                            "car_model": "Unknown",
                            "player_name": "ACC",
                            "player_surname": "Driver",
                            "player_nick": "",
                            "max_rpm": 9000,
                            "max_fuel": 100,
                        }
                        self.session_info_update.emit(session_data)
                    
                    elif msg_type == InboundMessageType.ENTRY_LIST_CAR:
                        car_info = AccPacketParser.parse_entry_list_car(data)
                        car_id = car_info.get("car_id")
                        self.car_data[car_id] = car_info
                    
                    elif msg_type == InboundMessageType.REALTIME_CAR_UPDATE:
                        car_update = AccPacketParser.parse_realtime_car_update(data)
                        self._handle_car_update(car_update)
                        
                        # Emit live data every ~0.1s
                        if time.time() - last_realtime > 0.1:
                            self._emit_live_data(car_update)
                            last_realtime = time.time()
                    
                except socket.timeout:
                    if not registered and self.running:
                        # Retry registration
                        self._send_registration(sock)
                    continue
                    
        except Exception as e:
            self.status_update.emit(f"Error: {str(e)}")
        finally:
            if self.connection_id:
                self._send_unregistration(sock)
            sock.close()
            self.status_update.emit("Disconnected")
    
    def _send_registration(self, sock):
        """Send registration packet to ACC"""
        msg = bytearray()
        msg.append(OutboundMessageType.REGISTER_COMMAND_APPLICATION)
        
        # Protocol version
        msg.append(4)
        
        # Display name
        name_bytes = self.display_name.encode('utf-8')
        msg.extend(struct.pack('<H', len(name_bytes)))
        msg.extend(name_bytes)
        
        # Connection password
        pwd_bytes = self.password.encode('utf-8')
        msg.extend(struct.pack('<H', len(pwd_bytes)))
        msg.extend(pwd_bytes)
        
        # Update interval (ms)
        msg.extend(struct.pack('<I', 100))
        
        # Command password (empty)
        msg.extend(struct.pack('<H', 0))
        
        sock.sendto(bytes(msg), (self.host, self.port))
    
    def _send_unregistration(self, sock):
        """Send unregistration packet"""
        msg = bytearray()
        msg.append(OutboundMessageType.UNREGISTER_COMMAND_APPLICATION)
        msg.extend(struct.pack('<I', self.connection_id))
        sock.sendto(bytes(msg), (self.host, self.port))
    
    def _send_request_track_data(self, sock):
        """Request track data"""
        msg = bytearray()
        msg.append(OutboundMessageType.REQUEST_TRACK_DATA)
        msg.extend(struct.pack('<I', self.connection_id))
        sock.sendto(bytes(msg), (self.host, self.port))
    
    def _send_request_entry_list(self, sock):
        """Request entry list"""
        msg = bytearray()
        msg.append(OutboundMessageType.REQUEST_ENTRY_LIST)
        msg.extend(struct.pack('<I', self.connection_id))
        sock.sendto(bytes(msg), (self.host, self.port))
    
    def _handle_car_update(self, car_update: Dict[str, Any]):
        """Handle car telemetry update and buffer lap data"""
        car_index = car_update.get("car_index", 0)
        laps = car_update.get("laps", 0)

        # Initialize if first time seeing this car
        if car_index not in self.current_lap_samples:
            self.current_lap_samples[car_index] = []
            self.last_lap_count[car_index] = laps

        # Add sample to current lap
        sample = {
            "t": time.time(),
            "lap_id": laps,
            "x": car_update.get("world_pos_x", 0),
            "z": car_update.get("world_pos_z", 0),
            "speed": car_update.get("kmh", 0),
            "gear": car_update.get("gear", 0),
            "rpms": 0,  # ACC doesn't expose RPM directly
            "brake": 0,  # Would need physics data
            "throttle": 0,  # Would need physics data
        }
        self.current_lap_samples[car_index].append(sample)

        # Emit real-time sample for live visualization
        self.realtime_sample.emit(sample)

        # Check for lap completion
        if laps > self.last_lap_count[car_index]:
            # Lap completed!
            if self.current_lap_samples[car_index]:
                self.lap_completed.emit(laps, self.current_lap_samples[car_index])

            # Reset for next lap
            self.current_lap_samples[car_index] = [sample]
            self.last_lap_count[car_index] = laps
    
    def _emit_live_data(self, car_update: Dict[str, Any]):
        """Emit live telemetry data for UI"""
        live_data = {
            "current_lap": car_update.get("laps", 0) + 1,
            "speed": car_update.get("kmh", 0),
            "gear": car_update.get("gear", 0),
            "rpm": 0,  # Not available
            "fuel": 0,  # Would need physics data
            "position": car_update.get("position", 0),
            "is_in_pit": 0,  # Would need to detect from data
            "current_time": "",
            "last_time": self._format_lap_time(car_update.get("last_lap_ms", 0)),
            "best_time": self._format_lap_time(car_update.get("best_session_lap_ms", 0)),
        }
        self.live_data_update.emit(live_data)
    
    def _format_lap_time(self, ms: int) -> str:
        """Format milliseconds to lap time string"""
        if ms == 0 or ms > 2147483647:  # Max int, means no time
            return ""
        total_seconds = ms / 1000.0
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        millis = int((total_seconds % 1) * 1000)
        return f"{minutes:02d}:{seconds:02d}.{millis:03d}"
    
    def stop(self):
        self.running = False