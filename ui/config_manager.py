"""
Configuration manager for persisting user settings.

Loads and saves settings to a JSON file next to the executable/script.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Default config file location (next to the script/executable)
CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.json"

DEFAULTS = {
    "voice_mode": "disabled",       # "disabled", "push_to_talk", "continuous"
    "ptt_key": "v",                 # Key name for push-to-talk
    "use_local_llm": True,
    "local_model_path": "race_engineer_gguf/granite-race-engineer-Q4_K_M.gguf",
}


def load_config() -> Dict[str, Any]:
    """Load config from disk, falling back to defaults for missing keys."""
    config = dict(DEFAULTS)
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r") as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                # Only accept recognized keys so stale settings are dropped.
                for key in DEFAULTS:
                    if key in saved:
                        config[key] = saved[key]
    except Exception as e:
        logger.warning("Failed to load config: %s", e)
    return config


def save_config(config: Dict[str, Any]) -> None:
    """Save config to disk."""
    # Persist only recognized keys so legacy/deprecated settings are dropped.
    to_save = {key: config.get(key, default) for key, default in DEFAULTS.items()}
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(to_save, f, indent=2)
    except Exception as e:
        logger.warning("Failed to save config: %s", e)
