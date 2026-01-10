"""
Watson Speech-to-Text Client for Jarvis-Granite Live Telemetry.

Provides async STT transcription using IBM Watson Speech-to-Text service
with Tenacity retry logic and exponential backoff.

Phase 5, Section 11: Watson STT Client
- Async HTTP client for Watson STT API
- Tenacity retry with exponential backoff (3 attempts, 0.5-4s wait)
- Support for multiple audio formats

Example:
    client = WatsonSTTClient(
        api_key="your-api-key",
        service_url="https://api.us-south.speech-to-text.watson.cloud.ibm.com"
    )
    transcript = await client.transcribe(audio_bytes)
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import aiohttp
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.config import RetryConfig

logger = logging.getLogger(__name__)


class STTError(Exception):
    """Exception raised for STT transcription errors."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.cause = cause


class WatsonSTTClient:
    """
    Async client for IBM Watson Speech-to-Text service.

    Features:
    - Async HTTP requests via aiohttp
    - Tenacity retry with exponential backoff
    - Configurable model and audio format
    - Request statistics tracking

    Attributes:
        api_key: Watson STT API key
        service_url: Watson STT service URL
        model: Speech recognition model (default: en-GB_BroadbandModel)
        timeout: Request timeout in seconds
        retry_config: Retry configuration for Tenacity

    Example:
        client = WatsonSTTClient(
            api_key="your-api-key",
            service_url="https://api.us-south.speech-to-text.watson.cloud.ibm.com",
            model="en-GB_BroadbandModel"
        )

        # Transcribe audio
        transcript = await client.transcribe(audio_bytes)

        # Get statistics
        stats = client.get_stats()
    """

    # Default configuration
    DEFAULT_MODEL = "en-GB_BroadbandModel"
    DEFAULT_CONTENT_TYPE = "audio/wav"
    DEFAULT_TIMEOUT = 30.0  # seconds (longer for audio processing)

    # Retry configuration defaults
    DEFAULT_MAX_ATTEMPTS = 3
    DEFAULT_MIN_WAIT = 0.5  # seconds
    DEFAULT_MAX_WAIT = 4.0  # seconds

    def __init__(
        self,
        api_key: str,
        service_url: str,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        Initialize Watson STT client.

        Args:
            api_key: Watson STT API key
            service_url: Watson STT service URL
            model: Speech recognition model to use
            timeout: Request timeout in seconds
            retry_config: Retry configuration (uses defaults if not provided)
        """
        self.api_key = api_key
        self.service_url = service_url.rstrip("/")
        self.model = model
        self.timeout = timeout

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
            f"Initialized WatsonSTTClient with model={model}, timeout={timeout}s"
        )

    async def transcribe(
        self,
        audio_data: bytes,
        content_type: str = DEFAULT_CONTENT_TYPE,
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio_data: Audio bytes to transcribe
            content_type: Audio content type (default: audio/wav)

        Returns:
            Transcribed text string

        Raises:
            ValueError: If audio data is empty
            STTError: If transcription fails after retries
        """
        if not audio_data:
            raise ValueError("Audio data cannot be empty")

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
                            f"STT retry attempt {attempt.retry_state.attempt_number}"
                        )

                    result = await self._make_request(audio_data, content_type)

                    # Extract transcript from response
                    transcript = self._extract_transcript(result)

                    # Update statistics
                    self._total_requests += 1
                    elapsed_ms = (time.time() - start_time) * 1000
                    self._total_latency_ms += elapsed_ms

                    logger.debug(
                        f"STT transcription completed in {elapsed_ms:.0f}ms: "
                        f"'{transcript[:50]}...'" if len(transcript) > 50 else f"'{transcript}'"
                    )

                    return transcript

        except Exception as e:
            self._total_errors += 1
            logger.error(f"STT transcription failed: {e}")
            raise STTError(f"STT transcription failed: {e}", cause=e) from e

    async def _make_request(
        self,
        audio_data: bytes,
        content_type: str,
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Watson STT API.

        Args:
            audio_data: Audio bytes to transcribe
            content_type: Audio content type

        Returns:
            API response dictionary

        Raises:
            Various exceptions on network/API errors
        """
        url = f"{self.service_url}/v1/recognize"

        headers = {
            "Content-Type": content_type,
        }

        params = {
            "model": self.model,
        }

        auth = aiohttp.BasicAuth("apikey", self.api_key)
        timeout = aiohttp.ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url,
                headers=headers,
                params=params,
                data=audio_data,
                auth=auth,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ConnectionError(
                        f"STT API error {response.status}: {error_text}"
                    )

                return await response.json()

    def _extract_transcript(self, response: Dict[str, Any]) -> str:
        """
        Extract transcript from API response.

        Args:
            response: API response dictionary

        Returns:
            Transcribed text string
        """
        if not response:
            return ""

        results = response.get("results", [])
        if not results:
            return ""

        # Get all transcripts and concatenate
        transcripts = []
        for result in results:
            alternatives = result.get("alternatives", [])
            if alternatives:
                # Use the first (best) alternative
                transcript = alternatives[0].get("transcript", "")
                if transcript:
                    transcripts.append(transcript.strip())

        return " ".join(transcripts)

    async def transcribe_with_confidence(
        self,
        audio_data: bytes,
        content_type: str = DEFAULT_CONTENT_TYPE,
    ) -> Dict[str, Any]:
        """
        Transcribe audio and return transcript with confidence scores.

        Args:
            audio_data: Audio bytes to transcribe
            content_type: Audio content type

        Returns:
            Dictionary with transcript and confidence

        Raises:
            ValueError: If audio data is empty
            STTError: If transcription fails
        """
        if not audio_data:
            raise ValueError("Audio data cannot be empty")

        start_time = time.time()

        try:
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
                    result = await self._make_request(audio_data, content_type)

                    # Extract transcript with confidence
                    transcript_data = self._extract_transcript_with_confidence(result)

                    self._total_requests += 1
                    elapsed_ms = (time.time() - start_time) * 1000
                    self._total_latency_ms += elapsed_ms

                    return transcript_data

        except Exception as e:
            self._total_errors += 1
            raise STTError(f"STT transcription failed: {e}", cause=e) from e

    def _extract_transcript_with_confidence(
        self,
        response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract transcript with confidence from API response.

        Args:
            response: API response dictionary

        Returns:
            Dictionary with transcript and confidence
        """
        if not response:
            return {"transcript": "", "confidence": 0.0}

        results = response.get("results", [])
        if not results:
            return {"transcript": "", "confidence": 0.0}

        # Get all transcripts
        transcripts = []
        confidences = []

        for result in results:
            alternatives = result.get("alternatives", [])
            if alternatives:
                best = alternatives[0]
                transcript = best.get("transcript", "").strip()
                confidence = best.get("confidence", 0.0)

                if transcript:
                    transcripts.append(transcript)
                    confidences.append(confidence)

        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "transcript": " ".join(transcripts),
            "confidence": avg_confidence,
            "alternatives": len(transcripts),
        }

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
            "model": self.model,
            "service_url": self.service_url,
        }

    def __repr__(self) -> str:
        """String representation of client."""
        return (
            f"WatsonSTTClient(model={self.model}, "
            f"requests={self._total_requests})"
        )
