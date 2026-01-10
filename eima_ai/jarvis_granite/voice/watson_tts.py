"""
Watson Text-to-Speech Client for Jarvis-Granite Live Telemetry.

Provides async TTS synthesis using IBM Watson Text-to-Speech service
with Tenacity retry logic and exponential backoff.

Phase 5, Section 11: Watson TTS Client
- Async HTTP client for Watson TTS API
- Tenacity retry with exponential backoff (3 attempts, 0.5-4s wait)
- Target latency: <500ms for synthesis

Example:
    client = WatsonTTSClient(
        api_key="your-api-key",
        service_url="https://api.us-south.text-to-speech.watson.cloud.ibm.com"
    )
    audio_bytes = await client.synthesize("Box this lap for fresh tires.")
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

import aiohttp
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.config import RetryConfig

logger = logging.getLogger(__name__)


class TTSError(Exception):
    """Exception raised for TTS synthesis errors."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.cause = cause


class WatsonTTSClient:
    """
    Async client for IBM Watson Text-to-Speech service.

    Features:
    - Async HTTP requests via aiohttp
    - Tenacity retry with exponential backoff
    - Configurable voice and audio format
    - Request statistics tracking

    Attributes:
        api_key: Watson TTS API key
        service_url: Watson TTS service URL
        voice: Voice model to use (default: en-GB_JamesV3Voice)
        timeout: Request timeout in seconds
        retry_config: Retry configuration for Tenacity

    Example:
        client = WatsonTTSClient(
            api_key="your-api-key",
            service_url="https://api.us-south.text-to-speech.watson.cloud.ibm.com",
            voice="en-GB_JamesV3Voice"
        )

        # Synthesize text to audio
        audio = await client.synthesize("Hello, driver!")

        # Get statistics
        stats = client.get_stats()
    """

    # Default configuration
    DEFAULT_VOICE = "en-GB_JamesV3Voice"
    DEFAULT_AUDIO_FORMAT = "audio/wav"
    DEFAULT_TIMEOUT = 10.0  # seconds

    # Retry configuration defaults
    DEFAULT_MAX_ATTEMPTS = 3
    DEFAULT_MIN_WAIT = 0.5  # seconds
    DEFAULT_MAX_WAIT = 4.0  # seconds

    def __init__(
        self,
        api_key: str,
        service_url: str,
        voice: str = DEFAULT_VOICE,
        timeout: float = DEFAULT_TIMEOUT,
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        Initialize Watson TTS client.

        Args:
            api_key: Watson TTS API key
            service_url: Watson TTS service URL
            voice: Voice model to use
            timeout: Request timeout in seconds
            retry_config: Retry configuration (uses defaults if not provided)
        """
        self.api_key = api_key
        self.service_url = service_url.rstrip("/")
        self.voice = voice
        self.timeout = timeout
        self.default_audio_format = self.DEFAULT_AUDIO_FORMAT

        # Retry configuration
        self.retry_config = retry_config or RetryConfig(
            max_attempts=self.DEFAULT_MAX_ATTEMPTS,
            min_wait=self.DEFAULT_MIN_WAIT,
            max_wait=self.DEFAULT_MAX_WAIT,
        )

        # Statistics
        self._total_requests = 0
        self._total_errors = 0
        self._total_retries = 0
        self._total_latency_ms = 0.0

        logger.info(
            f"Initialized WatsonTTSClient with voice={voice}, timeout={timeout}s"
        )

    async def synthesize(
        self,
        text: str,
        audio_format: Optional[str] = None,
    ) -> bytes:
        """
        Synthesize text to audio.

        Args:
            text: Text to synthesize
            audio_format: Audio format (default: audio/wav)

        Returns:
            Audio bytes in requested format

        Raises:
            ValueError: If text is empty
            TTSError: If synthesis fails after retries
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        audio_format = audio_format or self.default_audio_format
        start_time = time.time()

        try:
            # Use Tenacity for retry logic
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.retry_config.max_attempts),
                wait=wait_exponential(
                    multiplier=self.retry_config.multiplier,
                    min=self.retry_config.min_wait,
                    max=self.retry_config.max_wait,
                ),
                retry=retry_if_exception_type((TimeoutError, ConnectionError, aiohttp.ClientError)),
                reraise=True,
            ):
                with attempt:
                    if attempt.retry_state.attempt_number > 1:
                        self._total_retries += 1
                        logger.warning(
                            f"TTS retry attempt {attempt.retry_state.attempt_number}"
                        )

                    result = await self._make_request(text, audio_format)

                    if result is None:
                        raise TTSError("Empty response from TTS service")

                    # Update statistics
                    self._total_requests += 1
                    elapsed_ms = (time.time() - start_time) * 1000
                    self._total_latency_ms += elapsed_ms

                    logger.debug(
                        f"TTS synthesis completed in {elapsed_ms:.0f}ms, "
                        f"{len(result)} bytes"
                    )

                    return result

        except Exception as e:
            self._total_errors += 1
            logger.error(f"TTS synthesis failed: {e}")
            raise TTSError(f"TTS synthesis failed: {e}", cause=e) from e

    async def _make_request(
        self,
        text: str,
        audio_format: str,
    ) -> bytes:
        """
        Make HTTP request to Watson TTS API.

        Args:
            text: Text to synthesize
            audio_format: Audio format

        Returns:
            Audio bytes

        Raises:
            Various exceptions on network/API errors
        """
        url = f"{self.service_url}/v1/synthesize"

        headers = {
            "Accept": audio_format,
            "Content-Type": "application/json",
        }

        params = {
            "voice": self.voice,
        }

        payload = {
            "text": text,
        }

        auth = aiohttp.BasicAuth("apikey", self.api_key)
        timeout = aiohttp.ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url,
                headers=headers,
                params=params,
                json=payload,
                auth=auth,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ConnectionError(
                        f"TTS API error {response.status}: {error_text}"
                    )

                return await response.read()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get client statistics.

        Returns:
            Dictionary with request statistics
        """
        avg_latency = 0.0
        if self._total_requests > 0:
            avg_latency = self._total_latency_ms / self._total_requests

        return {
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "total_retries": self._total_retries,
            "average_latency_ms": avg_latency,
            "voice": self.voice,
            "service_url": self.service_url,
        }

    def __repr__(self) -> str:
        """String representation of client."""
        return (
            f"WatsonTTSClient(voice={self.voice}, "
            f"requests={self._total_requests})"
        )
