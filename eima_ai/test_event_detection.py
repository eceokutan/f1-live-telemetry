"""
Test event detection with the same telemetry as demo.py
"""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Import in specific order to avoid circular imports
from config.config import ThresholdsConfig
from jarvis_granite.schemas.telemetry import TelemetryData

# Import context directly (not through __init__.py)
import jarvis_granite.live.context as context_module
LiveSessionContext = context_module.LiveSessionContext

# Import telemetry agent directly
import jarvis_granite.agents.telemetry_agent as ta_module
TelemetryAgent = ta_module.TelemetryAgent

def test_event_detection():
    print("=" * 60)
    print("Testing Event Detection")
    print("=" * 60)

    # Create telemetry agent with default thresholds
    thresholds = ThresholdsConfig()
    agent = TelemetryAgent(thresholds=thresholds)

    print(f"\nThresholds:")
    print(f"  Tire temp warning: {thresholds.tire_temp_warning}C")
    print(f"  Tire temp critical: {thresholds.tire_temp_critical}C")
    print(f"  Fuel warning laps: {thresholds.fuel_warning_laps}")
    print(f"  Fuel critical laps: {thresholds.fuel_critical_laps}")

    # Create session context (simulating the demo)
    context = LiveSessionContext(
        session_id="test_001",
        source="simulator",
        track_name="Monza"
    )

    print(f"\nContext initial state:")
    print(f"  fuel_consumption_per_lap: {context.fuel_consumption_per_lap}")
    print(f"  current_lap: {context.current_lap}")

    # Test 1: Normal telemetry (should NOT trigger events)
    print("\n" + "-" * 60)
    print("TEST 1: Normal Telemetry (Lap 15, Fuel 45L, Tire 92C)")
    print("-" * 60)

    normal_telemetry = TelemetryData(
        speed_kmh=285.5,
        rpm=12500,
        gear=6,
        throttle=0.95,
        brake=0.0,
        steering_angle=0.05,
        fuel_remaining=45.0,
        tire_temps={"fl": 92.0, "fr": 93.0, "rl": 90.0, "rr": 91.0},
        tire_wear={"fl": 15.0, "fr": 16.0, "rl": 14.0, "rr": 15.0},
        g_forces={"lateral": 1.2, "longitudinal": 0.1},
        track_position=0.45,
        lap_number=15,
        lap_time_current=42.5,
        sector=2,
        position=3,
        gap_ahead=2.5,
        gap_behind=1.8
    )

    context.update(normal_telemetry)
    events = agent.detect_events(normal_telemetry, context)
    print(f"Events detected: {len(events)}")
    for e in events:
        print(f"  - {e.type} (priority: {e.priority.name})")

    # Test 2: Low fuel warning (should trigger fuel_warning if consumption is set)
    print("\n" + "-" * 60)
    print("TEST 2: Low Fuel (Lap 18, Fuel 8L, Tire 95C)")
    print("-" * 60)

    print(f"Context fuel_consumption_per_lap: {context.fuel_consumption_per_lap}")
    print(">>> NOTE: Fuel events require fuel_consumption_per_lap > 0 <<<")

    low_fuel_telemetry = TelemetryData(
        speed_kmh=265.0,
        rpm=11800,
        gear=5,
        throttle=0.85,
        brake=0.0,
        steering_angle=-0.12,
        fuel_remaining=8.0,
        tire_temps={"fl": 95.0, "fr": 96.0, "rl": 93.0, "rr": 94.0},
        tire_wear={"fl": 25.0, "fr": 27.0, "rl": 22.0, "rr": 24.0},
        g_forces={"lateral": 2.1, "longitudinal": -0.3},
        track_position=0.72,
        lap_number=18,
        lap_time_current=68.2,
        sector=3,
        position=3,
        gap_ahead=1.8,
        gap_behind=2.2
    )

    context.update(low_fuel_telemetry)
    events = agent.detect_events(low_fuel_telemetry, context)
    print(f"Events detected: {len(events)}")
    for e in events:
        print(f"  - {e.type} (priority: {e.priority.name}, data: {e.data})")

    # Test 3: Critical tire temp (should trigger tire_critical)
    print("\n" + "-" * 60)
    print("TEST 3: Critical Tires (Lap 19, Fuel 6.5L, Tire 118C)")
    print("-" * 60)

    print(f"Tire temps being sent: FL=115, FR=118, RL=112, RR=114")
    print(f"Critical threshold: {thresholds.tire_temp_critical}C")

    hot_tires_telemetry = TelemetryData(
        speed_kmh=245.0,
        rpm=11200,
        gear=5,
        throttle=0.75,
        brake=0.15,
        steering_angle=0.25,
        fuel_remaining=6.5,
        tire_temps={"fl": 115.0, "fr": 118.0, "rl": 112.0, "rr": 114.0},
        tire_wear={"fl": 45.0, "fr": 48.0, "rl": 42.0, "rr": 44.0},
        g_forces={"lateral": 3.2, "longitudinal": -1.5},
        track_position=0.15,
        lap_number=19,
        lap_time_current=15.8,
        sector=1,
        position=3,
        gap_ahead=1.2,
        gap_behind=2.8
    )

    context.update(hot_tires_telemetry)
    events = agent.detect_events(hot_tires_telemetry, context)
    print(f"Events detected: {len(events)}")
    for e in events:
        print(f"  - {e.type} (priority: {e.priority.name}, data: {e.data})")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if not events:
        print("\n❌ No events detected for critical tire temperatures!")
        print("   This is unexpected. Check the threshold values and tire temps.")
    else:
        print(f"\n✅ {len(events)} events detected for critical tire temps")
        print("   The tire events ARE being detected correctly.")
        print("\n   The issue is likely:")
        print("   1. LLM response taking longer than 5 second timeout")
        print("   2. Rate limiting (but CRITICAL should bypass)")
        print("   3. An exception during AI response generation")

if __name__ == "__main__":
    test_event_detection()
