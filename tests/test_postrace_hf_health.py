"""
Tests for Jarvis Post HF health-check behavior.
"""

import pytest
import httpx

import jarvis_post.llm.client as hf_module
from jarvis_post.llm.client import HFClient, LLMError

pytestmark = [pytest.mark.component, pytest.mark.regression]


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://unit.test/health")
            response = httpx.Response(self.status_code, request=request, json=self._payload)
            raise httpx.HTTPStatusError("health failed", request=request, response=response)

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, response: _FakeResponse | None = None, error: Exception | None = None):
        self._response = response
        self._error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, *_args, **_kwargs):
        if self._error is not None:
            raise self._error
        return self._response


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://unit.test/health")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("health failed", request=request, response=response)


@pytest.mark.asyncio
async def test_probe_health_returns_false_on_500_response(monkeypatch):
    client = HFClient(api_token="", space_url="https://unit.test")
    monkeypatch.setattr(
        hf_module.httpx,
        "AsyncClient",
        lambda timeout: _FakeAsyncClient(response=_FakeResponse(500)),
    )

    result = await client._probe_health()

    assert result is False


@pytest.mark.asyncio
async def test_probe_health_returns_false_on_500_http_error(monkeypatch):
    client = HFClient(api_token="", space_url="https://unit.test")
    monkeypatch.setattr(
        hf_module.httpx,
        "AsyncClient",
        lambda timeout: _FakeAsyncClient(error=_http_status_error(500)),
    )

    result = await client._probe_health()

    assert result is False


@pytest.mark.asyncio
async def test_probe_health_raises_on_401_http_error(monkeypatch):
    client = HFClient(api_token="", space_url="https://unit.test")
    monkeypatch.setattr(
        hf_module.httpx,
        "AsyncClient",
        lambda timeout: _FakeAsyncClient(error=_http_status_error(401)),
    )

    with pytest.raises(LLMError):
        await client._probe_health()
