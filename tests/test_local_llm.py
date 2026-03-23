"""
Deterministic tests for the live local LLM wrappers.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import ai.local_llm_inference as local_llm_module
from ai.local_llm_inference import LocalLLMInference
from ai.race_engineer_core.llm_client import LLMClient

pytestmark = [pytest.mark.unit, pytest.mark.component, pytest.mark.regression]


class _FakeModel:
    def __init__(self):
        self.last_prompt = None

    def tokenize(self, _payload: bytes):
        return list(range(100))

    def detokenize(self, _tokens):
        return b"TRUNCATED_PROMPT"

    def create_completion(self, prompt, **_kwargs):
        self.last_prompt = prompt
        return {"choices": [{"text": "  Copy that.  "}]}


def _fake_model_file(tmp_path: Path) -> Path:
    model_path = tmp_path / "fake.gguf"
    model_path.write_bytes(b"fake")
    return model_path


def test_local_llm_formats_prompt_with_granite_roles(tmp_path):
    model_path = _fake_model_file(tmp_path)
    llm = LocalLLMInference(model_path=str(model_path))

    prompt = llm._format_with_chat_template("Fuel?")

    assert "<|start_of_role|>system<|end_of_role|>" in prompt
    assert "<|start_of_role|>user<|end_of_role|>Fuel?" in prompt
    assert prompt.endswith("<|start_of_role|>assistant<|end_of_role|>")


def test_local_llm_generate_truncates_prompt_and_returns_text(tmp_path):
    model_path = _fake_model_file(tmp_path)
    llm = LocalLLMInference(model_path=str(model_path), max_prompt_tokens=4)
    llm._model = _FakeModel()
    llm._loaded = True

    result = llm.generate("How are my tires?")

    assert result == "Copy that."
    assert llm._model.last_prompt == "TRUNCATED_PROMPT"


def test_get_shared_reuses_singleton_instance(monkeypatch, tmp_path):
    model_path = _fake_model_file(tmp_path)

    def _fake_load(self):
        self._loaded = True
        self._model = object()

    monkeypatch.setattr(LocalLLMInference, "load", _fake_load)
    local_llm_module._shared_instance = None

    try:
        first = LocalLLMInference.get_shared(model_path=str(model_path))
        second = LocalLLMInference.get_shared(model_path=str(model_path))
        assert first is second
    finally:
        local_llm_module._shared_instance = None


def test_llm_client_force_fallback_returns_deterministic_response():
    client = LLMClient(force_rule_based_fallback=True)

    response = asyncio.run(client.invoke("fuel critical"))

    assert "Box box box" in response


def test_llm_client_local_generation_failure_falls_back(monkeypatch):
    class _BadLocal:
        def generate(self, _prompt):
            raise RuntimeError("boom")

    statuses = []
    monkeypatch.setattr(LLMClient, "_init_local_llm", lambda self: None)
    client = LLMClient(force_rule_based_fallback=False)
    client.set_status_callback(statuses.append)
    monkeypatch.setattr(client, "_get_local_llm", lambda: _BadLocal())

    response = asyncio.run(client.invoke("gap update"))

    assert "Gap has changed" in response
    assert any("generation failed" in msg.lower() for msg in statuses)


def test_llm_client_clean_response_strips_role_prefix():
    client = LLMClient(force_rule_based_fallback=True)

    cleaned = client._clean_response("Engineer: Keep pushing.")

    assert cleaned == "Keep pushing."
