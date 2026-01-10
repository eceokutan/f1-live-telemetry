"""
Tests for configuration module.

Run with: pytest tests/test_config.py -v
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch

# Add project root to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfigModels:
    """Test configuration Pydantic models."""

    def test_live_config_defaults(self):
        """LiveConfig should have sensible defaults."""
        from config.config import LiveConfig

        config = LiveConfig()

        assert config.verbosity.level == "moderate"
        assert config.thresholds.tire_temp_critical == 110.0
        assert config.thresholds.fuel_critical_laps == 2
        assert config.telemetry_buffer_seconds == 60
        assert config.orchestrator.priority_queue_max_size == 100

    def test_thresholds_config_custom_values(self):
        """ThresholdsConfig should accept custom values."""
        from config.config import ThresholdsConfig

        thresholds = ThresholdsConfig(
            tire_temp_warning=95.0,
            tire_temp_critical=105.0,
            fuel_warning_laps=3
        )

        assert thresholds.tire_temp_warning == 95.0
        assert thresholds.tire_temp_critical == 105.0
        assert thresholds.fuel_warning_laps == 3

    def test_verbosity_levels(self):
        """VerbosityConfig should accept valid levels."""
        from config.config import VerbosityConfig

        for level in ["minimal", "moderate", "verbose"]:
            config = VerbosityConfig(level=level)
            assert config.level == level

    def test_retry_config_defaults(self):
        """RetryConfig should have appropriate defaults."""
        from config.config import RetryConfig

        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.min_wait == 0.5
        assert config.max_wait == 4.0


class TestEnvironmentSubstitution:
    """Test environment variable substitution in config loader."""

    def test_simple_substitution(self):
        """${VAR} should be replaced with env value."""
        from config.config_loader import _substitute_env_vars

        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = _substitute_env_vars("${TEST_VAR}")
            assert result == "test_value"

    def test_default_value_substitution(self):
        """${VAR:-default} should use default when VAR is unset."""
        from config.config_loader import _substitute_env_vars

        # Ensure var is not set
        os.environ.pop("UNSET_VAR", None)

        result = _substitute_env_vars("${UNSET_VAR:-fallback}")
        assert result == "fallback"

    def test_default_value_when_set(self):
        """${VAR:-default} should use VAR value when set."""
        from config.config_loader import _substitute_env_vars

        with patch.dict(os.environ, {"SET_VAR": "actual_value"}):
            result = _substitute_env_vars("${SET_VAR:-fallback}")
            assert result == "actual_value"

    def test_conditional_value_when_set(self):
        """${VAR:+value} should return value when VAR is set."""
        from config.config_loader import _substitute_env_vars

        with patch.dict(os.environ, {"COND_VAR": "anything"}):
            result = _substitute_env_vars("${COND_VAR:+replacement}")
            assert result == "replacement"

    def test_conditional_value_when_unset(self):
        """${VAR:+value} should return empty when VAR is unset."""
        from config.config_loader import _substitute_env_vars

        os.environ.pop("UNSET_COND", None)

        result = _substitute_env_vars("${UNSET_COND:+replacement}")
        assert result == ""

    def test_nested_dict_substitution(self):
        """Substitution should work in nested dicts."""
        from config.config_loader import _substitute_env_vars

        with patch.dict(os.environ, {"NESTED_VAR": "nested_value"}):
            data = {
                "level1": {
                    "level2": "${NESTED_VAR}"
                }
            }
            result = _substitute_env_vars(data)
            assert result["level1"]["level2"] == "nested_value"

    def test_list_substitution(self):
        """Substitution should work in lists."""
        from config.config_loader import _substitute_env_vars

        with patch.dict(os.environ, {"LIST_VAR": "list_value"}):
            data = ["${LIST_VAR}", "static"]
            result = _substitute_env_vars(data)
            assert result == ["list_value", "static"]


class TestYamlLoading:
    """Test YAML configuration loading."""

    def test_load_live_yaml(self):
        """Should load live.yaml successfully."""
        from config.config_loader import load_yaml_config

        config = load_yaml_config()

        assert "orchestrator" in config
        assert "verbosity" in config
        assert "thresholds" in config
        assert config["verbosity"]["level"] == "moderate"

    def test_load_live_config_model(self):
        """Should create LiveConfig from YAML."""
        from config.config_loader import load_live_config

        config = load_live_config()

        assert config.verbosity.level == "moderate"
        assert config.thresholds.tire_temp_critical == 110.0

    def test_missing_yaml_raises_error(self):
        """Should raise FileNotFoundError for missing YAML."""
        from config.config_loader import load_yaml_config

        with pytest.raises(FileNotFoundError):
            load_yaml_config(Path("/nonexistent/config.yaml"))


class TestConfigurationManager:
    """Test ConfigurationManager singleton."""

    def test_singleton_pattern(self):
        """ConfigurationManager should be a singleton."""
        from config.config_loader import ConfigurationManager

        manager1 = ConfigurationManager()
        manager2 = ConfigurationManager()

        assert manager1 is manager2

    def test_lazy_initialization(self):
        """Config should be loaded lazily on first access."""
        from config.config_loader import ConfigurationManager

        manager = ConfigurationManager()
        # Access triggers initialization
        config = manager.live_config

        assert config is not None
        assert config.verbosity.level == "moderate"


class TestHelpers:
    """Test helper utility functions."""

    def test_format_lap_time(self):
        """Should format lap time correctly."""
        from jarvis_granite.utils.helpers import format_lap_time

        assert format_lap_time(85.234) == "1:25.234"
        assert format_lap_time(60.0) == "1:00.000"
        assert format_lap_time(45.5) == "0:45.500"

    def test_format_gap(self):
        """Should format gap correctly."""
        from jarvis_granite.utils.helpers import format_gap

        assert format_gap(1.234) == "+1.234s"
        assert format_gap(-0.5) == "-0.500s"
        assert format_gap(0) == "+0.000s"

    def test_calculate_fuel_laps(self):
        """Should calculate fuel laps correctly."""
        from jarvis_granite.utils.helpers import calculate_fuel_laps_remaining

        assert calculate_fuel_laps_remaining(50.0, 5.0) == 10.0
        assert calculate_fuel_laps_remaining(50.0, 5.0, safety_margin=10.0) == 8.0
        assert calculate_fuel_laps_remaining(10.0, 0) == 0.0  # Division by zero

    def test_tire_condition_label(self):
        """Should return correct tire condition labels."""
        from jarvis_granite.utils.helpers import tire_condition_label

        assert tire_condition_label(10) == "Fresh"
        assert tire_condition_label(30) == "Good"
        assert tire_condition_label(50) == "Used"
        assert tire_condition_label(70) == "Worn"
        assert tire_condition_label(90) == "Critical"

    def test_tire_temp_status(self):
        """Should return correct temperature status."""
        from jarvis_granite.utils.helpers import tire_temp_status

        assert tire_temp_status(85) == "optimal"
        assert tire_temp_status(105) == "warning"
        assert tire_temp_status(115) == "critical"


class TestLogging:
    """Test logging infrastructure."""

    def test_setup_logging(self):
        """Should create logger successfully."""
        from jarvis_granite.utils.logging import setup_logging

        logger = setup_logging(log_level="DEBUG")

        assert logger is not None
        assert logger.name == "jarvis-granite-live"

    def test_get_logger(self):
        """Should get module logger."""
        from jarvis_granite.utils.logging import get_logger

        logger = get_logger("test_module")

        assert "test_module" in logger.name

    def test_session_id_context(self):
        """Session ID context var should work."""
        from jarvis_granite.utils.logging import session_id_var

        session_id_var.set("test_session_123")
        assert session_id_var.get() == "test_session_123"
