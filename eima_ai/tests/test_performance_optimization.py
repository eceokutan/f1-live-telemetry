"""
TDD Tests for Performance Optimization - Phase 6, Section 16

These tests define the expected behavior for:
1. End-to-end latency profiling (target: <3000ms)
2. Bottleneck identification and optimization
3. Metrics logging and monitoring

Performance Targets (from documentation):
- Telemetry processing: <50ms (rule-based, no LLM)
- LLM response: <2000ms (Granite generation via LangChain)
- TTS conversion: <500ms (Watson TTS with Tenacity retry)
- LiveKit transport: <100ms (WebRTC jitter buffering)
- End-to-end (voice): <3000ms (Total pipeline latency)

Run with: pytest tests/test_performance_optimization.py -v

Following TDD: Write tests FIRST, watch them fail, then implement.
"""

import asyncio
import time
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis_granite.schemas.telemetry import TelemetryData, TireTemps, TireWear, GForces
from config.config import LiveConfig, LiveKitConfig


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def live_config():
    """Create test live configuration."""
    return LiveConfig(
        livekit=LiveKitConfig(
            url="wss://test-livekit.example.com",
            api_key="test_api_key",
            api_secret="test_api_secret",
        )
    )


@pytest.fixture
def sample_telemetry():
    """Create sample telemetry data."""
    return TelemetryData(
        speed_kmh=250.0,
        rpm=12000,
        gear=5,
        throttle=0.9,
        brake=0.0,
        steering_angle=0.1,
        fuel_remaining=50.0,
        tire_temps=TireTemps(fl=95.0, fr=96.0, rl=92.0, rr=93.0),
        tire_wear=TireWear(fl=20.0, fr=22.0, rl=18.0, rr=19.0),
        g_forces=GForces(lateral=1.5, longitudinal=0.2),
        track_position=0.5,
        lap_number=10,
        lap_time_current=45.0,
        sector=2,
        position=3,
        gap_ahead=2.5,
        gap_behind=1.5
    )


@pytest.fixture
def low_fuel_telemetry():
    """Create telemetry with critical fuel level."""
    return TelemetryData(
        speed_kmh=250.0,
        rpm=12000,
        gear=5,
        throttle=0.9,
        brake=0.0,
        steering_angle=0.1,
        fuel_remaining=3.0,  # Critical - only ~1.5 laps
        tire_temps=TireTemps(fl=95.0, fr=96.0, rl=92.0, rr=93.0),
        tire_wear=TireWear(fl=20.0, fr=22.0, rl=18.0, rr=19.0),
        g_forces=GForces(lateral=1.5, longitudinal=0.2),
        track_position=0.5,
        lap_number=10,
        lap_time_current=45.0,
        sector=2,
        position=3,
        gap_ahead=2.5,
        gap_behind=1.5
    )


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client with realistic latency."""
    client = MagicMock()

    async def slow_invoke(*args, **kwargs):
        await asyncio.sleep(0.05)  # 50ms simulated LLM latency
        return "Box this lap for fresh tires."

    client.invoke = AsyncMock(side_effect=slow_invoke)
    return client


# =============================================================================
# PERFORMANCE METRICS CLASS TESTS
# =============================================================================

class TestPerformanceMetrics:
    """Tests for the PerformanceMetrics class."""

    def test_create_performance_metrics(self):
        """Should create PerformanceMetrics instance."""
        from jarvis_granite.live.performance import PerformanceMetrics

        metrics = PerformanceMetrics()

        assert metrics is not None

    def test_record_latency_measurement(self):
        """Should record latency measurements."""
        from jarvis_granite.live.performance import PerformanceMetrics

        metrics = PerformanceMetrics()

        metrics.record("telemetry_processing", 45.0)
        metrics.record("telemetry_processing", 50.0)
        metrics.record("telemetry_processing", 48.0)

        stats = metrics.get_stats("telemetry_processing")

        assert stats is not None
        assert stats["count"] == 3
        assert stats["min"] == 45.0
        assert stats["max"] == 50.0

    def test_calculate_average_latency(self):
        """Should calculate average latency."""
        from jarvis_granite.live.performance import PerformanceMetrics

        metrics = PerformanceMetrics()

        metrics.record("llm_response", 100.0)
        metrics.record("llm_response", 200.0)
        metrics.record("llm_response", 150.0)

        stats = metrics.get_stats("llm_response")

        assert stats["avg"] == 150.0

    def test_calculate_percentiles(self):
        """Should calculate percentile latencies (p50, p95, p99)."""
        from jarvis_granite.live.performance import PerformanceMetrics

        metrics = PerformanceMetrics()

        # Record 100 measurements
        for i in range(100):
            metrics.record("test_operation", float(i))

        stats = metrics.get_stats("test_operation")

        assert "p50" in stats
        assert "p95" in stats
        assert "p99" in stats

    def test_check_target_compliance(self):
        """Should check if metrics meet target thresholds."""
        from jarvis_granite.live.performance import PerformanceMetrics

        metrics = PerformanceMetrics()

        # Set target thresholds
        metrics.set_target("telemetry_processing", 50.0)  # <50ms target
        metrics.set_target("llm_response", 2000.0)  # <2000ms target

        # Record compliant measurements
        metrics.record("telemetry_processing", 45.0)
        metrics.record("llm_response", 1500.0)

        compliance = metrics.check_compliance()

        assert compliance["telemetry_processing"]["compliant"] is True
        assert compliance["llm_response"]["compliant"] is True

    def test_detect_non_compliant_metrics(self):
        """Should detect when metrics exceed targets."""
        from jarvis_granite.live.performance import PerformanceMetrics

        metrics = PerformanceMetrics()

        metrics.set_target("telemetry_processing", 50.0)

        # Record non-compliant measurement
        metrics.record("telemetry_processing", 75.0)

        compliance = metrics.check_compliance()

        assert compliance["telemetry_processing"]["compliant"] is False

    def test_reset_metrics(self):
        """Should reset all metrics."""
        from jarvis_granite.live.performance import PerformanceMetrics

        metrics = PerformanceMetrics()

        metrics.record("test", 100.0)
        metrics.reset()

        stats = metrics.get_stats("test")

        assert stats["count"] == 0

    def test_get_all_metrics_summary(self):
        """Should return summary of all metrics."""
        from jarvis_granite.live.performance import PerformanceMetrics

        metrics = PerformanceMetrics()

        metrics.record("telemetry_processing", 45.0)
        metrics.record("llm_response", 1500.0)
        metrics.record("tts_conversion", 400.0)

        summary = metrics.get_summary()

        assert "telemetry_processing" in summary
        assert "llm_response" in summary
        assert "tts_conversion" in summary


# =============================================================================
# LATENCY PROFILER TESTS
# =============================================================================

class TestLatencyProfiler:
    """Tests for latency profiling utilities."""

    def test_profile_context_manager(self):
        """Should profile code block with context manager."""
        from jarvis_granite.live.performance import PerformanceMetrics, profile_latency

        metrics = PerformanceMetrics()

        with profile_latency(metrics, "test_operation"):
            time.sleep(0.01)  # 10ms

        stats = metrics.get_stats("test_operation")

        assert stats["count"] == 1
        assert stats["min"] >= 10.0  # At least 10ms

    @pytest.mark.asyncio
    async def test_profile_async_context_manager(self):
        """Should profile async code block."""
        from jarvis_granite.live.performance import PerformanceMetrics, profile_latency_async

        metrics = PerformanceMetrics()

        async with profile_latency_async(metrics, "async_operation"):
            await asyncio.sleep(0.01)  # 10ms

        stats = metrics.get_stats("async_operation")

        assert stats["count"] == 1
        assert stats["min"] >= 10.0

    def test_profile_decorator(self):
        """Should profile function with decorator."""
        from jarvis_granite.live.performance import PerformanceMetrics, profile

        metrics = PerformanceMetrics()

        @profile(metrics, "decorated_function")
        def slow_function():
            time.sleep(0.01)
            return "result"

        result = slow_function()

        assert result == "result"
        stats = metrics.get_stats("decorated_function")
        assert stats["count"] == 1

    @pytest.mark.asyncio
    async def test_profile_async_decorator(self):
        """Should profile async function with decorator."""
        from jarvis_granite.live.performance import PerformanceMetrics, profile_async

        metrics = PerformanceMetrics()

        @profile_async(metrics, "async_decorated")
        async def slow_async_function():
            await asyncio.sleep(0.01)
            return "async result"

        result = await slow_async_function()

        assert result == "async result"
        stats = metrics.get_stats("async_decorated")
        assert stats["count"] == 1


# =============================================================================
# PIPELINE PERFORMANCE INTEGRATION TESTS
# =============================================================================

class TestPipelinePerformance:
    """Tests for performance tracking in IntegrationPipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_has_performance_metrics(self, live_config):
        """Pipeline should have performance metrics instance."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        assert hasattr(pipeline, 'performance_metrics')
        assert pipeline.performance_metrics is not None

    @pytest.mark.asyncio
    async def test_pipeline_tracks_telemetry_latency(self, live_config, sample_telemetry):
        """Should track telemetry processing latency."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        await pipeline.init_session("race_001", "torcs", "Monza")
        await pipeline.process_telemetry("race_001", sample_telemetry)

        stats = pipeline.performance_metrics.get_stats("telemetry_processing")

        assert stats["count"] >= 1

    @pytest.mark.asyncio
    async def test_pipeline_tracks_llm_latency(self, live_config, low_fuel_telemetry, mock_llm_client):
        """Should track LLM response latency."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(
            config=live_config,
            llm_client=mock_llm_client
        )

        await pipeline.init_session("race_001", "torcs", "Monza")

        context = pipeline.get_session("race_001")["context"]
        context.fuel_consumption_per_lap = 2.0

        await pipeline.process_telemetry("race_001", low_fuel_telemetry)

        stats = pipeline.performance_metrics.get_stats("llm_response")

        # Should have recorded at least one LLM call
        assert stats["count"] >= 1

    @pytest.mark.asyncio
    async def test_pipeline_tracks_end_to_end_latency(self, live_config, low_fuel_telemetry, mock_llm_client):
        """Should track total end-to-end pipeline latency."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(
            config=live_config,
            llm_client=mock_llm_client
        )

        await pipeline.init_session("race_001", "torcs", "Monza")

        context = pipeline.get_session("race_001")["context"]
        context.fuel_consumption_per_lap = 2.0

        await pipeline.process_telemetry("race_001", low_fuel_telemetry)

        stats = pipeline.performance_metrics.get_stats("end_to_end")

        assert stats["count"] >= 1

    @pytest.mark.asyncio
    async def test_get_performance_report(self, live_config, sample_telemetry):
        """Should generate performance report."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        await pipeline.init_session("race_001", "torcs", "Monza")

        # Process several telemetry messages
        for _ in range(10):
            await pipeline.process_telemetry("race_001", sample_telemetry)

        report = pipeline.get_performance_report()

        assert "summary" in report
        assert "compliance" in report
        assert "targets" in report


# =============================================================================
# PERFORMANCE TARGET TESTS
# =============================================================================

class TestPerformanceTargets:
    """Tests for performance target validation."""

    @pytest.mark.asyncio
    async def test_telemetry_processing_under_50ms(self, live_config, sample_telemetry):
        """Telemetry processing should be under 50ms (no events)."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(config=live_config)

        await pipeline.init_session("race_001", "torcs", "Monza")

        result = await pipeline.process_telemetry("race_001", sample_telemetry)

        # Should complete under 50ms for simple telemetry (no events)
        assert result["latency_ms"] < 50

    @pytest.mark.asyncio
    async def test_end_to_end_under_3000ms(self, live_config, low_fuel_telemetry, mock_llm_client):
        """Full pipeline should complete under 3000ms."""
        from jarvis_granite.live.integration_pipeline import IntegrationPipeline

        pipeline = IntegrationPipeline(
            config=live_config,
            llm_client=mock_llm_client
        )

        await pipeline.init_session("race_001", "torcs", "Monza")

        context = pipeline.get_session("race_001")["context"]
        context.fuel_consumption_per_lap = 2.0

        result = await pipeline.process_telemetry("race_001", low_fuel_telemetry)

        # End-to-end should be under 3000ms
        assert result["latency_ms"] < 3000


# =============================================================================
# METRICS LOGGING TESTS
# =============================================================================

class TestMetricsLogging:
    """Tests for metrics logging functionality."""

    def test_metrics_can_be_exported(self):
        """Should export metrics in JSON format."""
        from jarvis_granite.live.performance import PerformanceMetrics

        metrics = PerformanceMetrics()

        metrics.record("test", 100.0)
        metrics.record("test", 200.0)

        export = metrics.export_json()

        assert isinstance(export, str)
        import json
        data = json.loads(export)
        assert "test" in data

    def test_metrics_log_to_file(self, tmp_path):
        """Should log metrics to file."""
        from jarvis_granite.live.performance import PerformanceMetrics

        log_file = tmp_path / "metrics.log"
        metrics = PerformanceMetrics(log_file=str(log_file))

        metrics.record("test", 100.0)
        metrics.flush_log()

        assert log_file.exists()

    def test_metrics_include_timestamp(self):
        """Should include timestamp with each measurement."""
        from jarvis_granite.live.performance import PerformanceMetrics

        metrics = PerformanceMetrics()

        metrics.record("test", 100.0)

        export = metrics.export_json()
        import json
        data = json.loads(export)

        # Each metric should have timestamp info
        assert "last_updated" in data["test"]


# =============================================================================
# BOTTLENECK IDENTIFICATION TESTS
# =============================================================================

class TestBottleneckIdentification:
    """Tests for bottleneck identification."""

    def test_identify_slowest_operation(self):
        """Should identify the slowest operation."""
        from jarvis_granite.live.performance import PerformanceMetrics

        metrics = PerformanceMetrics()

        metrics.record("fast_op", 10.0)
        metrics.record("slow_op", 500.0)
        metrics.record("medium_op", 100.0)

        bottlenecks = metrics.identify_bottlenecks()

        assert bottlenecks[0]["operation"] == "slow_op"

    def test_identify_operations_exceeding_threshold(self):
        """Should identify operations exceeding their thresholds."""
        from jarvis_granite.live.performance import PerformanceMetrics

        metrics = PerformanceMetrics()

        metrics.set_target("telemetry", 50.0)
        metrics.set_target("llm", 2000.0)

        metrics.record("telemetry", 75.0)  # Exceeds 50ms
        metrics.record("llm", 1500.0)  # Within 2000ms

        bottlenecks = metrics.identify_bottlenecks(threshold_only=True)

        assert len(bottlenecks) == 1
        assert bottlenecks[0]["operation"] == "telemetry"
