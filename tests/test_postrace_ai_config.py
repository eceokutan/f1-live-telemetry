"""
Tests for Jarvis Post local GGUF model detection in the pipeline bridge.
"""

import os
from pathlib import Path
from unittest.mock import patch

from analysis.ai_pipeline_bridge import AIPipelineBridge


def _bridge_for_config_tests() -> AIPipelineBridge:
    # Avoid running discovery in __init__; config helpers only read env/filesystem.
    return AIPipelineBridge.__new__(AIPipelineBridge)


def test_has_postrace_local_model_true_when_gguf_exists(monkeypatch, tmp_path):
    monkeypatch.delenv("POSTRACE_GGUF_MODEL_PATH", raising=False)
    model_file = tmp_path / "granite-postrace-analyst-Q5_K_M.gguf"
    model_file.write_bytes(b"fake")

    monkeypatch.setenv("POSTRACE_GGUF_MODEL_PATH", str(model_file))

    bridge = _bridge_for_config_tests()
    assert bridge._has_postrace_local_model() is True


def test_has_postrace_local_model_env_override(monkeypatch, tmp_path):
    model_file = tmp_path / "custom-model.gguf"
    model_file.write_bytes(b"fake")
    monkeypatch.setenv("POSTRACE_GGUF_MODEL_PATH", str(model_file))

    bridge = _bridge_for_config_tests()
    assert bridge._has_postrace_local_model() is True


def test_has_postrace_local_model_false_when_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("POSTRACE_GGUF_MODEL_PATH", str(tmp_path / "nonexistent.gguf"))

    bridge = _bridge_for_config_tests()
    assert bridge._has_postrace_local_model() is False


def test_has_postrace_local_model_false_default_path_missing(monkeypatch):
    monkeypatch.delenv("POSTRACE_GGUF_MODEL_PATH", raising=False)

    bridge = _bridge_for_config_tests()
    # Patch the default path resolution to point to a non-existent location
    with patch.object(Path, "exists", return_value=False):
        assert bridge._has_postrace_local_model() is False
