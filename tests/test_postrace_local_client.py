"""
Tests for LocalGGUFClient (mocked llama_cpp to avoid needing real GGUF in CI).
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from jarvis_post.llm.local_client import LocalGGUFClient


@pytest.fixture
def mock_llama():
    """Create a mock Llama instance with realistic create_completion output."""
    mock_instance = MagicMock()
    mock_instance.create_completion.return_value = {
        "choices": [
            {
                "text": "Lap 3 showed improved braking consistency.",
                "finish_reason": "stop",
            }
        ],
        "usage": {"completion_tokens": 12},
    }
    return mock_instance


@pytest.fixture
def client_with_mock(tmp_path, monkeypatch, mock_llama):
    """Create a LocalGGUFClient with a fake model file and mocked Llama."""
    model_file = tmp_path / "test-model.gguf"
    model_file.write_bytes(b"fake-gguf")
    monkeypatch.setenv("POSTRACE_GGUF_MODEL_PATH", str(model_file))

    client = LocalGGUFClient()

    # Bypass actual model loading — inject mock directly
    client._model = mock_llama
    client._loaded = True
    return client


def test_generate_returns_correct_shape(client_with_mock, mock_llama):
    result = asyncio.run(
        client_with_mock.generate(
            prompt="Analyse lap 3",
            system_prompt="You are a race analyst.",
        )
    )

    assert isinstance(result, dict)
    assert "content" in result
    assert "tokens_used" in result
    assert "finish_reason" in result
    assert result["content"] == "Lap 3 showed improved braking consistency."
    assert result["tokens_used"] == 12
    assert result["finish_reason"] == "stop"


def test_generate_finish_reason_length_propagated(client_with_mock, mock_llama):
    mock_llama.create_completion.return_value = {
        "choices": [{"text": "truncated output", "finish_reason": "length"}],
        "usage": {"completion_tokens": 1000},
    }

    result = asyncio.run(
        client_with_mock.generate(
            prompt="Long analysis request",
            system_prompt="You are a race analyst.",
            max_tokens=1000,
        )
    )

    assert result["finish_reason"] == "length"
    assert result["tokens_used"] == 1000


def test_prompt_formatting_uses_granite_template(client_with_mock, mock_llama):
    asyncio.run(
        client_with_mock.generate(
            prompt="How was lap 5?",
            system_prompt="You are a coach.",
        )
    )

    call_args = mock_llama.create_completion.call_args
    formatted_prompt = call_args[0][0]

    assert "<|start_of_role|>system<|end_of_role|>" in formatted_prompt
    assert "You are a coach." in formatted_prompt
    assert "<|start_of_role|>user<|end_of_role|>" in formatted_prompt
    assert "How was lap 5?" in formatted_prompt
    assert "<|start_of_role|>assistant<|end_of_role|>" in formatted_prompt


def test_prompt_formatting_without_system_prompt(client_with_mock, mock_llama):
    asyncio.run(
        client_with_mock.generate(
            prompt="Quick check",
            system_prompt="",
        )
    )

    call_args = mock_llama.create_completion.call_args
    formatted_prompt = call_args[0][0]

    assert "<|start_of_role|>system<|end_of_role|>" not in formatted_prompt
    assert "<|start_of_role|>user<|end_of_role|>" in formatted_prompt


def test_model_not_found_raises_file_not_found(monkeypatch, tmp_path):
    monkeypatch.setenv("POSTRACE_GGUF_MODEL_PATH", str(tmp_path / "nonexistent.gguf"))

    client = LocalGGUFClient()

    with pytest.raises(FileNotFoundError, match="Post-race GGUF model not found"):
        client._load_model()


def test_warmup_sync_returns_true_on_success(client_with_mock):
    assert client_with_mock.warmup_sync() is True


def test_warmup_sync_returns_false_on_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("POSTRACE_GGUF_MODEL_PATH", str(tmp_path / "missing.gguf"))

    client = LocalGGUFClient()
    assert client.warmup_sync() is False
