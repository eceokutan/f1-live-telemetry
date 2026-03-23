from types import SimpleNamespace

from ai.race_engineer import AIRaceEngineerWorker
from ai.race_engineer_core import LiveSessionContext, ThresholdsConfig


def _build_worker_for_tests() -> AIRaceEngineerWorker:
    """Create a lightweight worker instance without starting threads/event loops."""
    worker = AIRaceEngineerWorker.__new__(AIRaceEngineerWorker)
    worker.context = LiveSessionContext(
        session_id="test-session",
        source="assetto_corsa",
        track_name="Monza",
    )
    worker.telemetry_agent = SimpleNamespace(thresholds=ThresholdsConfig())
    worker.race_engineer_agent = None
    return worker


def test_unmatched_query_context_is_core_only():
    ctx = LiveSessionContext(
        session_id="test-session",
        source="assetto_corsa",
        track_name="Monza",
    )
    prompt_ctx = ctx.to_prompt_context(query="I love you")
    assert "Track:" in prompt_ctx
    assert "Lap:" in prompt_ctx
    assert "Speed:" in prompt_ctx
    assert "Fuel:" not in prompt_ctx
    assert "Tire Temps:" not in prompt_ctx
    assert "Gap Ahead:" not in prompt_ctx


def test_keyword_query_context_includes_matching_group():
    ctx = LiveSessionContext(
        session_id="test-session",
        source="assetto_corsa",
        track_name="Monza",
    )
    prompt_ctx = ctx.to_prompt_context(query="When should I pit?")
    # Pit queries get condensed pit summary + fuel, not individual tire/damage breakdowns
    assert "Fuel:" in prompt_ctx
    assert "Pit Summary:" in prompt_ctx
    # Per-tire breakdown lines should NOT appear (use line prefixes to avoid
    # matching substrings inside the pit summary line)
    assert "Tire Temps: FL:" not in prompt_ctx
    assert "Tire Pressures: FL:" not in prompt_ctx
    assert "Tire Wear: FL:" not in prompt_ctx
    assert "Car Damage:" not in prompt_ctx


def test_low_quality_response_detects_numeric_stub():
    assert AIRaceEngineerWorker._is_low_quality_response("0")
    assert not AIRaceEngineerWorker._is_low_quality_response("Fuel looks good for 6 laps.")


def test_pit_fallback_asks_for_one_clean_lap_when_fuel_burn_unknown():
    worker = _build_worker_for_tests()
    msg = worker._build_pit_query_fallback_response()
    assert "calibrate fuel burn" in msg.lower()


def test_pit_fallback_uses_fuel_projection_when_available():
    worker = _build_worker_for_tests()
    worker.context.fuel_consumption_per_lap = 2.0
    worker.context.fuel_remaining = 6.0
    msg = worker._build_pit_query_fallback_response()
    assert "pit window open" in msg.lower()
    assert "3.0 laps" in msg.lower()


def test_general_fallback_safe_when_agent_missing():
    worker = _build_worker_for_tests()
    msg = worker._build_reactive_fallback_response("How are my gaps?")
    assert msg == "Copy that. Monitoring the situation."


def test_pit_query_gets_condensed_pit_summary():
    ctx = LiveSessionContext(
        session_id="test-session",
        source="assetto_corsa",
        track_name="Monza",
    )
    ctx.fuel_remaining = 10.0
    ctx.fuel_consumption_per_lap = 2.5
    ctx.tire_temps = {"fl": 95.0, "fr": 102.0, "rl": 88.0, "rr": 90.0}
    prompt_ctx = ctx.to_prompt_context(query="should I pit?")
    assert "Pit Summary:" in prompt_ctx
    assert "102" in prompt_ctx  # max tire temp value
    assert "FR" in prompt_ctx   # corner with max temp


def test_tire_query_still_gets_full_breakdowns():
    ctx = LiveSessionContext(
        session_id="test-session",
        source="assetto_corsa",
        track_name="Monza",
    )
    prompt_ctx = ctx.to_prompt_context(query="how are my tires?")
    assert "Tire Temps:" in prompt_ctx
    assert "Tire Pressures:" in prompt_ctx
    assert "Tire Wear:" in prompt_ctx
    assert "Pit Summary:" not in prompt_ctx


def test_should_force_pit_fallback_catches_insufficient_data():
    worker = _build_worker_for_tests()
    # Exact phrase the system prompt tells the LLM to emit
    assert worker._should_force_pit_fallback(
        "should I box?",
        "Insufficient data.",
    )
    # Verbose variant the model may still produce
    assert worker._should_force_pit_fallback(
        "when should I pit?",
        "I don't have enough data to determine pit timing.",
    )


def test_should_force_pit_fallback_passes_specific_response():
    worker = _build_worker_for_tests()
    assert not worker._should_force_pit_fallback(
        "when should I pit?",
        "Fuel projects 3.0 laps. Pit window open now.",
    )


def test_should_force_pit_fallback_false_for_non_pit_query():
    worker = _build_worker_for_tests()
    assert not worker._should_force_pit_fallback(
        "how are my tires?",
        "Insufficient data.",
    )
