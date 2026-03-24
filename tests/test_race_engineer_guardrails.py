import pytest
from types import SimpleNamespace

pytest.importorskip("PyQt5")

from ai.race_engineer import AIRaceEngineerWorker
from ai.race_engineer_core import LiveSessionContext, ThresholdsConfig

pytestmark = [pytest.mark.component, pytest.mark.regression]


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


def test_clean_llm_response_preserves_data_mid_sentence():
    """The word 'Data' mid-sentence should not be stripped."""
    result = AIRaceEngineerWorker._clean_llm_response("Data looks good, no damage detected.")
    assert "Data looks good" in result
    result2 = AIRaceEngineerWorker._clean_llm_response("Tire data shows normal temps across all four corners.")
    assert "data shows normal" in result2.lower()


def test_clean_llm_response_strips_data_header_at_line_start():
    """A 'Data:' header line should still be removed."""
    result = AIRaceEngineerWorker._clean_llm_response("Tires are fine.\nData: speed 250, rpm 8500")
    assert "Data:" not in result
    assert "Tires are fine" in result


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
    assert "3 laps" in msg.lower()


def test_general_fallback_default_for_unknown_query():
    worker = _build_worker_for_tests()
    msg = worker._build_reactive_fallback_response("Tell me a joke")
    assert msg == "Copy that. Monitoring the situation."


def test_reactive_fallback_damage_no_damage():
    worker = _build_worker_for_tests()
    msg = worker._build_reactive_fallback_response("How's my damage?")
    assert "no damage" in msg.lower()


def test_reactive_fallback_damage_with_damage():
    worker = _build_worker_for_tests()
    worker.context.car_damage = {"front": 15.0, "rear": 0.0, "left": 0.0, "right": 0.0, "centre": 0.0}
    msg = worker._build_reactive_fallback_response("How's my damage?")
    assert "front 15%" in msg.lower()


def test_reactive_fallback_tire_query():
    worker = _build_worker_for_tests()
    worker.context.tire_temps = {"fl": 85.0, "fr": 102.0, "rl": 88.0, "rr": 90.0}
    msg = worker._build_reactive_fallback_response("How are my tires?")
    assert "fr" in msg.lower()
    assert "102" in msg


def test_reactive_fallback_gap_query():
    worker = _build_worker_for_tests()
    worker.context.position = 3
    worker.context.gap_ahead = 1.5
    worker.context.gap_behind = None
    msg = worker._build_reactive_fallback_response("What's my gap?")
    assert "p3" in msg.lower()
    assert "1.5s" in msg
    assert "no car" in msg.lower()


def test_reactive_fallback_fuel_query():
    worker = _build_worker_for_tests()
    worker.context.fuel_remaining = 45.0
    worker.context.fuel_consumption_per_lap = 3.0
    msg = worker._build_reactive_fallback_response("How's my fuel?")
    assert "45.0" in msg
    assert "15 laps" in msg


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


# --- _has_ungrounded_numbers tests ---

def test_has_ungrounded_numbers_detects_fabricated():
    worker = _build_worker_for_tests()
    # Context has speed ~0, rpm 0, gear 0, etc.  "3894" is nowhere in context.
    assert worker._has_ungrounded_numbers("Your lap time is 3894 seconds.", "how am I doing?")


def test_has_ungrounded_numbers_passes_grounded():
    worker = _build_worker_for_tests()
    worker.context.speed_kmh = 250.0
    worker.context.rpm = 8500
    # Response only references numbers present in context
    assert not worker._has_ungrounded_numbers("Speed is 250 km/h.", "how fast am I?")


def test_has_ungrounded_numbers_false_when_no_numbers():
    worker = _build_worker_for_tests()
    assert not worker._has_ungrounded_numbers("Tires look fine, push on.", "how are my tires?")


def test_has_ungrounded_numbers_passes_trivial_values():
    """Numbers like 0, 1, 100 should not trigger ungrounded fallback."""
    worker = _build_worker_for_tests()
    assert not worker._has_ungrounded_numbers("0 damage, 100 percent condition.", "how's my damage?")
    assert not worker._has_ungrounded_numbers("Push for 1 more lap.", "should I push?")
    assert not worker._has_ungrounded_numbers("All 4 tires look fine, 10 laps to go.", "how are my tires?")


def test_has_ungrounded_numbers_still_detects_fabricated_with_trivial():
    """Mixing trivial and fabricated numbers should still trigger fallback."""
    worker = _build_worker_for_tests()
    assert worker._has_ungrounded_numbers("0 damage but lap time was 9876 seconds.", "how am I doing?")


def test_has_ungrounded_numbers_trivial_only_with_empty_source():
    """Response with only trivial numbers should pass even if source has no numbers."""
    worker = _build_worker_for_tests()
    assert not worker._has_ungrounded_numbers("0 damage.", "any damage?")


# --- _is_pit_query STT broadening tests ---

def test_is_pit_query_catches_stt_misrecognitions():
    assert AIRaceEngineerWorker._is_pit_query("when should I pay?")
    assert AIRaceEngineerWorker._is_pit_query("when should I bet?")
    assert AIRaceEngineerWorker._is_pit_query("when should I bit?")
    assert AIRaceEngineerWorker._is_pit_query("when should I pet?")
    assert AIRaceEngineerWorker._is_pit_query("when should I pit?")
    assert AIRaceEngineerWorker._is_pit_query("should I box now?")


def test_is_pit_query_rejects_unrelated():
    assert not AIRaceEngineerWorker._is_pit_query("how are my tires?")
    assert not AIRaceEngineerWorker._is_pit_query("what is my gap?")


# --- Pit context pruning with STT keywords ---

def test_stt_pit_query_gets_pit_context():
    """Garbled pit queries should still get fuel/pit context lines."""
    ctx = LiveSessionContext(
        session_id="test-session",
        source="assetto_corsa",
        track_name="Monza",
    )
    ctx.fuel_remaining = 10.0
    ctx.fuel_consumption_per_lap = 2.5
    prompt_ctx = ctx.to_prompt_context(query="when should I pay?")
    assert "Fuel:" in prompt_ctx
    assert "Pit Summary:" in prompt_ctx


# --- Fuel lookup tests ---

def test_fuel_lookup_known_car_and_track():
    from ai.fuel_lookup import lookup_fuel_consumption
    result = lookup_fuel_consumption("ks_ferrari_458", "monza")
    assert not result["is_default"]
    assert "Ferrari" in result["matched_car"]
    assert "Monza" in result["matched_track"]
    # Ferrari 458: base=3.50, Monza: sf=1.20, km=5.79
    expected = 3.50 * 1.20 * (5.79 / 5.0)
    assert abs(result["fuel_per_lap"] - expected) < 0.01


def test_fuel_lookup_unknown_falls_back_to_default():
    from ai.fuel_lookup import lookup_fuel_consumption
    result = lookup_fuel_consumption("totally_unknown_car", "nonexistent_track")
    assert result["is_default"]
    assert "Lotus" in result["matched_car"]
    assert "Monza" in result["matched_track"]


def test_fuel_lookup_track_with_ks_prefix():
    from ai.fuel_lookup import lookup_fuel_consumption
    result = lookup_fuel_consumption("ks_lotus_exos_125", "ks_nurburgring")
    assert "Nurburgring" in result["matched_track"] or "Nürburgring" in result["matched_track"]
