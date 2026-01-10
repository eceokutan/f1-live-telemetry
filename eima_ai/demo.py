#!/usr/bin/env python3
"""
Jarvis-Granite Live Demo Script

This script demonstrates the live telemetry system by:
1. Connecting to the WebSocket endpoint
2. Initializing a racing session
3. Sending simulated telemetry that triggers AI responses
4. Displaying the AI race engineer responses in real-time

Run the server first:
    python -m uvicorn jarvis_granite.live.main:app --reload --port 8000

Then run this demo:
    python demo.py

For blog screenshots, you can capture:
- The terminal output showing AI responses
- The /health endpoint in browser
- The /stats endpoint in browser
- The /config endpoint in browser
"""

import asyncio
import json
import sys
from datetime import datetime, timezone

# Check for websockets library
try:
    import websockets
except ImportError:
    print("Installing websockets...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
    import websockets


# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print a styled header."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}\n")


def print_telemetry(lap: int, fuel: float, tire_temp: float, position: int, gap: float):
    """Print telemetry status."""
    print(f"{Colors.CYAN}[TELEMETRY]{Colors.ENDC} Lap {lap} | "
          f"Fuel: {fuel:.1f}L | Tire: {tire_temp:.0f}C | "
          f"P{position} | Gap: {gap:.2f}s")


def print_event(event_type: str, priority: str):
    """Print detected event."""
    color = Colors.RED if priority in ["CRITICAL", "HIGH"] else Colors.YELLOW
    print(f"{color}[EVENT DETECTED]{Colors.ENDC} {event_type} ({priority})")


def print_ai_response(text: str, trigger: str, latency: int):
    """Print AI response."""
    print(f"\n{Colors.GREEN}{Colors.BOLD}[RACE ENGINEER]{Colors.ENDC}")
    print(f"{Colors.GREEN}>>> {text}{Colors.ENDC}")
    print(f"{Colors.BLUE}    Trigger: {trigger} | Latency: {latency}ms{Colors.ENDC}\n")


def print_error(error_code: str, message: str):
    """Print error message."""
    print(f"{Colors.RED}[ERROR] {error_code}: {message}{Colors.ENDC}")


async def run_demo():
    """Run the live demo."""
    uri = "ws://localhost:8000/live"

    print_header("JARVIS-GRANITE LIVE DEMO")
    print(f"Connecting to {uri}...")

    try:
        async with websockets.connect(uri) as websocket:
            print(f"{Colors.GREEN}Connected!{Colors.ENDC}\n")

            # ============================================================
            # STEP 1: Initialize Session
            # ============================================================
            print_header("STEP 1: Initialize Session")

            session_init = {
                "type": "session_init",
                "session_id": "demo_race_001",
                "source": "simulator",
                "track_name": "Monza",
                "config": {
                    "verbosity": "moderate",
                    "driver_name": "Demo Driver"
                }
            }

            print(f"Sending: {json.dumps(session_init, indent=2)}")
            await websocket.send(json.dumps(session_init))

            response = await websocket.recv()
            data = json.loads(response)

            if data.get("type") == "session_confirmed":
                print(f"\n{Colors.GREEN}Session confirmed!{Colors.ENDC}")
                print(f"  Session ID: {data.get('session_id')}")
                print(f"  LiveKit Room: {data.get('livekit', {}).get('room_name')}")
                print(f"  LiveKit Token: {data.get('livekit', {}).get('token', '')[:50]}...")
            else:
                print_error(data.get("error_code", "UNKNOWN"), data.get("message", ""))
                return

            await asyncio.sleep(1)

            # ============================================================
            # STEP 2: Send Normal Telemetry (No Events)
            # ============================================================
            print_header("STEP 2: Normal Telemetry (No Events)")

            normal_telemetry = {
                "type": "telemetry",
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "data": {
                    "speed_kmh": 285.5,
                    "rpm": 12500,
                    "gear": 6,
                    "throttle": 0.95,
                    "brake": 0.0,
                    "steering_angle": 0.05,
                    "fuel_remaining": 45.0,
                    "tire_temps": {"fl": 92.0, "fr": 93.0, "rl": 90.0, "rr": 91.0},
                    "tire_wear": {"fl": 15.0, "fr": 16.0, "rl": 14.0, "rr": 15.0},
                    "g_forces": {"lateral": 1.2, "longitudinal": 0.1},
                    "track_position": 0.45,
                    "lap_number": 15,
                    "lap_time_current": 42.5,
                    "sector": 2,
                    "position": 3,
                    "gap_ahead": 2.5,
                    "gap_behind": 1.8
                }
            }

            print_telemetry(15, 45.0, 92.0, 3, 2.5)
            await websocket.send(json.dumps(normal_telemetry))
            print(f"{Colors.BLUE}No events triggered - normal operation{Colors.ENDC}")

            await asyncio.sleep(2)

            # ============================================================
            # STEP 3: Low Fuel Warning
            # ============================================================
            print_header("STEP 3: Fuel Warning Event")

            low_fuel_telemetry = {
                "type": "telemetry",
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "data": {
                    "speed_kmh": 265.0,
                    "rpm": 11800,
                    "gear": 5,
                    "throttle": 0.85,
                    "brake": 0.0,
                    "steering_angle": -0.12,
                    "fuel_remaining": 8.0,  # Low fuel!
                    "tire_temps": {"fl": 95.0, "fr": 96.0, "rl": 93.0, "rr": 94.0},
                    "tire_wear": {"fl": 25.0, "fr": 27.0, "rl": 22.0, "rr": 24.0},
                    "g_forces": {"lateral": 2.1, "longitudinal": -0.3},
                    "track_position": 0.72,
                    "lap_number": 18,
                    "lap_time_current": 68.2,
                    "sector": 3,
                    "position": 3,
                    "gap_ahead": 1.8,
                    "gap_behind": 2.2
                }
            }

            print_telemetry(18, 8.0, 95.0, 3, 1.8)
            await websocket.send(json.dumps(low_fuel_telemetry))

            # Wait for and display AI response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                data = json.loads(response)
                if data.get("type") == "ai_response":
                    print_event("fuel_warning", "HIGH")
                    print_ai_response(
                        data.get("text", ""),
                        data.get("trigger", ""),
                        data.get("latency_ms", 0)
                    )
            except asyncio.TimeoutError:
                print(f"{Colors.YELLOW}(No fuel event - requires lap completion to calculate consumption rate){Colors.ENDC}")

            await asyncio.sleep(2)

            # ============================================================
            # STEP 4: Critical Tire Temperature
            # ============================================================
            print_header("STEP 4: Critical Tire Temperature")

            hot_tires_telemetry = {
                "type": "telemetry",
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "data": {
                    "speed_kmh": 245.0,
                    "rpm": 11200,
                    "gear": 5,
                    "throttle": 0.75,
                    "brake": 0.15,
                    "steering_angle": 0.25,
                    "fuel_remaining": 6.5,
                    "tire_temps": {"fl": 115.0, "fr": 118.0, "rl": 112.0, "rr": 114.0},  # Critical!
                    "tire_wear": {"fl": 45.0, "fr": 48.0, "rl": 42.0, "rr": 44.0},
                    "g_forces": {"lateral": 3.2, "longitudinal": -1.5},
                    "track_position": 0.15,
                    "lap_number": 19,
                    "lap_time_current": 15.8,
                    "sector": 1,
                    "position": 3,
                    "gap_ahead": 1.2,
                    "gap_behind": 2.8
                }
            }

            print_telemetry(19, 6.5, 118.0, 3, 1.2)
            await websocket.send(json.dumps(hot_tires_telemetry))

            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                data = json.loads(response)
                if data.get("type") == "ai_response":
                    print_event("tire_critical", "CRITICAL")
                    print_ai_response(
                        data.get("text", ""),
                        data.get("trigger", ""),
                        data.get("latency_ms", 0)
                    )
            except asyncio.TimeoutError:
                print(f"{Colors.YELLOW}(AI response timeout - LLM may be slow or unavailable){Colors.ENDC}")

            await asyncio.sleep(2)

            # ============================================================
            # STEP 5: Driver Query
            # ============================================================
            print_header("STEP 5: Driver Query")

            query = {
                "type": "text_query",
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "query": "What's my tire situation?"
            }

            print(f'{Colors.CYAN}[DRIVER]{Colors.ENDC} "What\'s my tire situation?"')
            await websocket.send(json.dumps(query))

            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                data = json.loads(response)
                if data.get("type") == "ai_response":
                    print_ai_response(
                        data.get("text", ""),
                        "driver_query",
                        data.get("latency_ms", 0)
                    )
            except asyncio.TimeoutError:
                print(f"{Colors.YELLOW}(AI response timeout - LLM may be slow or unavailable){Colors.ENDC}")

            await asyncio.sleep(1)

            # ============================================================
            # STEP 6: End Session
            # ============================================================
            print_header("STEP 6: End Session")

            end_session = {"type": "session_end"}
            await websocket.send(json.dumps(end_session))
            print(f"{Colors.GREEN}Session ended successfully{Colors.ENDC}")

            print_header("DEMO COMPLETE")
            # print("Screenshots you can capture:")
            # print("  1. This terminal output")
            # print("  2. http://localhost:8000/health - Health status")
            # print("  3. http://localhost:8000/stats - Statistics")
            # print("  4. http://localhost:8000/config - Configuration")
            # print("  5. http://localhost:8000/docs - API Documentation (Swagger UI)")

    except ConnectionRefusedError:
        print(f"\n{Colors.RED}ERROR: Could not connect to server{Colors.ENDC}")
        print(f"\nMake sure the server is running:")
        print(f"  {Colors.CYAN}python -m uvicorn jarvis_granite.live.main:app --reload --port 8000{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.RED}ERROR: {e}{Colors.ENDC}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("JARVIS-GRANITE LIVE TELEMETRY DEMO")
    print("="*60)
    print("\nThis demo will simulate a racing session with:")
    print("  - Session initialization with LiveKit token")
    print("  - Normal telemetry (no events)")
    print("  - Low fuel warning event")
    print("  - Critical tire temperature event")
    print("  - Driver voice query simulation")
    print("  - Session end")
    print("\n" + "-"*60)

    asyncio.run(run_demo())
