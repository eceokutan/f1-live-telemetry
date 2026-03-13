"""HuggingFace Space inference client for Jarvis Post."""

from __future__ import annotations

import asyncio
import os
import time

import httpx


class LLMTimeoutError(Exception):
    """Raised when LLM request times out."""

    pass


class LLMError(Exception):
    """Raised when LLM request fails."""

    pass


class HFClient:
    """Client for a model served via a HuggingFace Space (Docker FastAPI)."""

    def __init__(self):
        self.api_token = (os.getenv("HF_API_TOKEN") or "").strip()
        space_url = os.getenv("HF_SPACE_URL", "https://ecsy9-f1-granite-post.hf.space")
        self.space_url = space_url.rstrip("/")
        self.endpoint = f"{self.space_url}/v1/chat/completions"
        self.health_endpoint = f"{self.space_url}/health"

        # Defaults tuned for Spaces that may cold-start for several minutes.
        self.request_timeout = float(os.getenv("HF_REQUEST_TIMEOUT_SECONDS", "300"))
        self.max_retries = int(os.getenv("HF_MAX_RETRIES", "3"))
        self.health_timeout = float(os.getenv("HF_HEALTH_TIMEOUT_SECONDS", "20"))
        self.health_poll_interval = float(os.getenv("HF_HEALTH_POLL_INTERVAL_SECONDS", "8"))
        self.max_warmup_wait = float(os.getenv("HF_MAX_WARMUP_WAIT_SECONDS", "420"))
        self._space_ready = False
        self._health_supported: bool | None = None

    async def generate(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.5,
    ) -> dict:
        await self._wait_for_space_ready()

        messages = [
            {"role": "user", "content": prompt},
        ]
        if system_prompt.strip():
            messages.insert(0, {"role": "system", "content": system_prompt})

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.request_timeout) as client:
                    response = await client.post(self.endpoint, json=payload, headers=headers)

                # During cold starts the API often returns 5xx while loading.
                if response.status_code in (502, 503, 504):
                    if attempt < self.max_retries:
                        await asyncio.sleep(min(20, attempt * 5))
                        continue
                    raise LLMTimeoutError(
                        f"Space is still warming up (HTTP {response.status_code}) after "
                        f"{self.max_retries} attempts."
                    )

                response.raise_for_status()
                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    raise LLMError("Space response did not include choices.")
                first_choice = choices[0]
                content = first_choice.get("message", {}).get("content", "") or ""
                finish_reason = first_choice.get("finish_reason")
                tokens_used = data.get("usage", {}).get("completion_tokens", 0) or 0
                return {
                    "content": content,
                    "tokens_used": tokens_used,
                    "finish_reason": finish_reason,
                }
            except httpx.TimeoutException as e:
                last_error = e
                if attempt == self.max_retries:
                    raise LLMTimeoutError(
                        f"Space did not respond after {self.request_timeout:.0f}s "
                        f"({attempt} attempts). It may still be waking up."
                    ) from e
            except httpx.HTTPStatusError as e:
                raise LLMError(f"Space returned HTTP {e.response.status_code}: {e.response.text}") from e
            except Exception as e:
                raise LLMError(f"Space request failed: {e}") from e
        raise LLMError(f"All {self.max_retries} attempts failed: {last_error}")

    async def warmup(self) -> bool:
        """
        Warm up the Space model in advance.

        Returns:
            True when ready or no health endpoint is available; False on timeout.
        """
        try:
            await self._wait_for_space_ready()
            return True
        except LLMTimeoutError:
            return False

    def warmup_sync(self) -> bool:
        """
        Synchronous wrapper for startup warmup from non-async code.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            return False

        try:
            return asyncio.run(self.warmup())
        except Exception:
            return False

    async def _wait_for_space_ready(self) -> None:
        """
        Poll /health when available to avoid first-call failures on cold starts.
        """
        if self._space_ready:
            return
        if self._health_supported is False:
            return

        deadline = time.monotonic() + self.max_warmup_wait
        while time.monotonic() < deadline:
            health = await self._probe_health()

            if health is None:
                # /health unsupported or unavailable: proceed without explicit warmup.
                self._health_supported = False
                return
            if health:
                self._space_ready = True
                self._health_supported = True
                return

            await asyncio.sleep(self.health_poll_interval)

        raise LLMTimeoutError(
            f"Space model did not report ready within {self.max_warmup_wait:.0f}s. "
            "It may still be waking up."
        )

    async def _probe_health(self) -> bool | None:
        """
        Returns:
            - True: health says model is ready
            - False: health reachable but model still loading
            - None: health endpoint unsupported/unavailable
        """
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        try:
            async with httpx.AsyncClient(timeout=self.health_timeout) as client:
                response = await client.get(self.health_endpoint, headers=headers)

            if response.status_code in (404, 405):
                return None
            if response.status_code in (502, 503, 504):
                return False
            response.raise_for_status()

            data = response.json()
            if isinstance(data, dict) and "model_loaded" in data:
                return bool(data.get("model_loaded"))

            # If the endpoint responds but doesn't provide model_loaded,
            # treat it as ready and let generation perform the final check.
            return True
        except httpx.TimeoutException:
            return False
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code if e.response else "unknown"
            raise LLMError(f"Space health check failed with HTTP {status_code}.") from e
        except Exception:
            return None
