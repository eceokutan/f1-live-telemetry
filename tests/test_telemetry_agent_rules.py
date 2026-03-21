from ai.race_engineer_core.config import ThresholdsConfig
from ai.race_engineer_core.context import LiveSessionContext
from ai.race_engineer_core.telemetry import OpponentSnapshot, TelemetryData, TireTemps
from ai.race_engineer_core.telemetry_agent import TelemetryAgent


def _make_telemetry(**overrides) -> TelemetryData:
    base = {
        "speed": 150.0,
        "rpms": 7500,
        "gear": 4,
        "throttle": 0.6,
        "brake": 0.0,
        "fuel": 30.0,
        "tire_temps": TireTemps(fl=85.0, fr=86.0, rl=88.0, rr=87.0),
        "lap_id": 0,
        "lap_number": 0,
        "t": 10.0,
    }
    base.update(overrides)
    return TelemetryData(**base)


def test_sector_completion_event_removed():
    agent = TelemetryAgent()
    context = LiveSessionContext(
        session_id="s1",
        source="assetto_corsa",
        track_name="spa",
    )
    context.current_sector = 1

    telemetry = _make_telemetry(sector=2)
    events = agent.detect_events(telemetry, context)

    assert all(event.type != "sector_complete" for event in events)


def test_opponent_close_behind_event_emitted():
    thresholds = ThresholdsConfig(
        opponent_close_behind_gap=0.8,
        opponent_close_reset_gap=1.2,
    )
    agent = TelemetryAgent(thresholds=thresholds)
    context = LiveSessionContext(
        session_id="s2",
        source="assetto_corsa",
        track_name="monza",
    )
    context.position = 5
    context.gap_behind = 1.2

    telemetry = _make_telemetry(
        position=5,
        gap_behind=0.6,
        opponents=[
            OpponentSnapshot(
                car_index=42,
                position=6,
                lap_number=3,
                speed=220.0,
                track_position=0.54,
            )
        ],
    )
    events = agent.detect_events(telemetry, context)
    close_events = [event for event in events if event.type == "opponent_close_behind"]

    assert len(close_events) == 1
    assert close_events[0].data.get("gap") == 0.6
    assert close_events[0].data.get("car_index") == 42


def test_car_damage_alert_emitted_on_threshold_cross():
    thresholds = ThresholdsConfig(
        car_damage_warning_total=5.0,
        car_damage_critical_total=20.0,
        car_damage_delta_threshold=2.0,
    )
    agent = TelemetryAgent(thresholds=thresholds)
    context = LiveSessionContext(
        session_id="s3",
        source="assetto_corsa",
        track_name="silverstone",
    )

    telemetry = _make_telemetry(
        car_damage={
            "front": 6.0,
            "rear": 0.0,
            "left": 0.0,
            "right": 0.0,
            "centre": 0.0,
        }
    )
    events = agent.detect_events(telemetry, context)
    damage_events = [event for event in events if event.type == "car_damage_alert"]

    assert len(damage_events) == 1
    assert damage_events[0].data.get("severity") == "warning"
    assert damage_events[0].data.get("total") == 6.0
