"""
Tests for Jarvis Post AI credential/config resolution.
"""

from analysis.ai_pipeline_bridge import AIPipelineBridge
from jarvis_post.llm.client import HFClient


POSTRACE_ENV_KEYS = (
    "POSTRACE_HF_API_TOKEN",
    "POSTRACE_HF_SPACE_URL",
    "POSTRACE_HF_MODEL_ID",
    "HF_API_TOKEN",
    "HF_SPACE_URL",
    "HF_MODEL_ID",
    "HUGGINGFACE_TOKEN",
    "HUGGINGFACE_SPACE_URL",
)


def _clear_env(monkeypatch) -> None:
    for key in POSTRACE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _bridge_for_config_tests() -> AIPipelineBridge:
    # Avoid running discovery in __init__; config helpers only read env.
    return AIPipelineBridge.__new__(AIPipelineBridge)


def test_postrace_env_reads_only_postrace_keys(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("POSTRACE_HF_API_TOKEN", "postrace_token")
    monkeypatch.setenv("POSTRACE_HF_SPACE_URL", "https://postrace-space.hf.space")
    monkeypatch.setenv("HF_API_TOKEN", "legacy_token")
    monkeypatch.setenv("HF_SPACE_URL", "https://legacy-space.hf.space")

    bridge = _bridge_for_config_tests()
    settings = bridge._get_postrace_hf_settings()

    assert settings["api_token"] == "postrace_token"
    assert settings["space_url"] == "https://postrace-space.hf.space"


def test_postrace_config_allows_public_space_without_token(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("POSTRACE_HF_SPACE_URL", "https://public-space.hf.space")

    bridge = _bridge_for_config_tests()
    settings = bridge._get_postrace_hf_settings()

    assert settings["api_token"] == ""
    assert settings["space_url"] == "https://public-space.hf.space"
    assert bridge._has_postrace_hf_config(settings) is True


def test_postrace_config_ignores_live_huggingface_keys(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "live_token")
    monkeypatch.setenv("HUGGINGFACE_SPACE_URL", "https://live-space.hf.space/chat")

    bridge = _bridge_for_config_tests()
    settings = bridge._get_postrace_hf_settings()

    assert settings["api_token"] == ""
    assert settings["space_url"] == ""
    assert bridge._has_postrace_hf_config(settings) is False


def test_postrace_config_ignores_legacy_hf_keys(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HF_API_TOKEN", "legacy_token")
    monkeypatch.setenv("HF_SPACE_URL", "https://legacy-space.hf.space")

    bridge = _bridge_for_config_tests()
    settings = bridge._get_postrace_hf_settings()

    assert settings["api_token"] == ""
    assert settings["space_url"] == ""
    assert bridge._has_postrace_hf_config(settings) is False


def test_hf_client_reads_postrace_keys_by_default(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("POSTRACE_HF_API_TOKEN", "postrace_token")
    monkeypatch.setenv("POSTRACE_HF_SPACE_URL", "https://postrace-space.hf.space")

    client = HFClient()

    assert client.api_token == "postrace_token"
    assert client.space_url == "https://postrace-space.hf.space"
    assert client.endpoint == "https://postrace-space.hf.space/v1/chat/completions"


def test_hf_client_ignores_legacy_hf_keys(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("HF_API_TOKEN", "legacy_token")
    monkeypatch.setenv("HF_SPACE_URL", "https://legacy-space.hf.space")

    client = HFClient()

    assert client.api_token == ""
    assert client.space_url == ""


def test_hf_client_constructor_args_override_env(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("POSTRACE_HF_API_TOKEN", "env_token")
    monkeypatch.setenv("POSTRACE_HF_SPACE_URL", "https://env-space.hf.space")

    client = HFClient(api_token="arg_token", space_url="https://arg-space.hf.space")

    assert client.api_token == "arg_token"
    assert client.space_url == "https://arg-space.hf.space"
