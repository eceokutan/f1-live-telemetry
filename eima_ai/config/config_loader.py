"""
Configuration loader for Jarvis-Granite Live Telemetry.

This module handles loading configuration from YAML files and
environment variables, with support for variable substitution.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from config.config import LiveConfig, EnvironmentSettings


def _substitute_env_vars(value: Any) -> Any:
    """
    Recursively substitute environment variables in configuration values.

    Supports patterns:
    - ${VAR_NAME} - required variable
    - ${VAR_NAME:-default} - variable with default value
    - ${VAR_NAME:+value_if_set} - value if variable is set
    """
    if isinstance(value, str):
        # Pattern for ${VAR_NAME}, ${VAR_NAME:-default}, ${VAR_NAME:+value}
        pattern = r'\$\{([^}]+)\}'

        def replace_var(match):
            var_expr = match.group(1)

            # Handle ${VAR:+value_if_set}
            if ':+' in var_expr:
                var_name, value_if_set = var_expr.split(':+', 1)
                env_value = os.getenv(var_name)
                return value_if_set if env_value else ''

            # Handle ${VAR:-default}
            if ':-' in var_expr:
                var_name, default = var_expr.split(':-', 1)
                return os.getenv(var_name, default)

            # Handle simple ${VAR}
            return os.getenv(var_expr, '')

        return re.sub(pattern, replace_var, value)

    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]

    return value


def load_yaml_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file with environment variable substitution.

    Args:
        config_path: Path to the YAML configuration file.
                    Defaults to config/live.yaml.

    Returns:
        Dictionary containing the configuration values.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "live.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    # Substitute environment variables
    return _substitute_env_vars(raw_config)


def load_live_config(
    config_path: Optional[Path] = None,
    env_file: Optional[Path] = None
) -> LiveConfig:
    """
    Load LiveConfig from YAML file with environment variable substitution.

    Args:
        config_path: Path to the YAML configuration file.
        env_file: Path to the .env file. Defaults to .env in project root.

    Returns:
        LiveConfig instance with all configuration values.
    """
    # Load environment variables from .env file
    if env_file is None:
        env_file = Path(__file__).parent.parent / ".env"

    load_dotenv(env_file)

    # Load and parse YAML configuration
    yaml_config = load_yaml_config(config_path)

    # Create LiveConfig from the parsed configuration
    return LiveConfig(**yaml_config)


def load_environment_settings(env_file: Optional[Path] = None) -> EnvironmentSettings:
    """
    Load environment settings from .env file and environment variables.

    Args:
        env_file: Path to the .env file. Defaults to .env in project root.

    Returns:
        EnvironmentSettings instance with all environment values.
    """
    if env_file is None:
        env_file = Path(__file__).parent.parent / ".env"

    load_dotenv(env_file)

    return EnvironmentSettings()


def get_config() -> tuple[LiveConfig, EnvironmentSettings]:
    """
    Get both LiveConfig and EnvironmentSettings.

    This is a convenience function that loads both configuration
    sources in a single call.

    Returns:
        Tuple of (LiveConfig, EnvironmentSettings).
    """
    env_settings = load_environment_settings()
    live_config = load_live_config()

    return live_config, env_settings


class ConfigurationManager:
    """
    Singleton manager for application configuration.

    Provides centralized access to both YAML configuration
    and environment settings.
    """

    _instance: Optional['ConfigurationManager'] = None
    _live_config: Optional[LiveConfig] = None
    _env_settings: Optional[EnvironmentSettings] = None

    def __new__(cls) -> 'ConfigurationManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(
        self,
        config_path: Optional[Path] = None,
        env_file: Optional[Path] = None
    ) -> None:
        """
        Initialize the configuration manager.

        Args:
            config_path: Path to the YAML configuration file.
            env_file: Path to the .env file.
        """
        self._env_settings = load_environment_settings(env_file)
        self._live_config = load_live_config(config_path, env_file)

    @property
    def live_config(self) -> LiveConfig:
        """Get the LiveConfig instance."""
        if self._live_config is None:
            self.initialize()
        return self._live_config

    @property
    def env_settings(self) -> EnvironmentSettings:
        """Get the EnvironmentSettings instance."""
        if self._env_settings is None:
            self.initialize()
        return self._env_settings

    def reload(
        self,
        config_path: Optional[Path] = None,
        env_file: Optional[Path] = None
    ) -> None:
        """
        Reload configuration from files.

        Useful for runtime configuration updates.

        Args:
            config_path: Path to the YAML configuration file.
            env_file: Path to the .env file.
        """
        self.initialize(config_path, env_file)


# Global configuration manager instance
config_manager = ConfigurationManager()
