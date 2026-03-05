"""
LLM Client for Jarvis-Granite Live Telemetry.

Provides an interface to a text generation model hosted on Hugging Face
using the `huggingface_hub` Inference API.

Features:
- Hugging Face InferenceClient integration
- Tenacity retry with exponential backoff
- Async support for non-blocking calls
- Configurable model parameters
"""

import asyncio
import logging
from typing import Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Exception raised for LLM-related errors."""

    pass


class LLMClient:
    """
    Client for a Hugging Face-hosted LLM via `huggingface_hub.InferenceClient`.

    This is a lightweight wrapper that handles:
    - Connection configuration
    - Retry logic with Tenacity
    - Async invocation

    The actual prompt formatting is handled by RaceEngineerAgent.

    Attributes:
        model_id: Hugging Face model identifier
        max_tokens: Maximum tokens for response generation
        temperature: LLM temperature for response variety
        max_retries: Maximum retry attempts for failed requests
    """

    def __init__(
        self,
        huggingface_token: str,
        model_id: str,
        max_tokens: int = 75,  # Reduced from 150 for faster responses (~200-400ms savings)
        temperature: float = 0.7,
        max_retries: int = 3,
        min_retry_wait: float = 1.0,
        max_retry_wait: float = 5.0,
    ):
        """
        Initialize LLM Client.

        Args:
            huggingface_token: Hugging Face access token (HUGGINGFACE_TOKEN / HUGGINGFACE_API_KEY)
            model_id: Model identifier on Hugging Face (e.g. "org/custom-race-engineer")
            max_tokens: Maximum response tokens (default: 75, reduced for racing brevity)
            temperature: Response temperature (default: 0.7)
            max_retries: Max retry attempts (default: 3)
            min_retry_wait: Minimum wait between retries in seconds (default: 1.0)
            max_retry_wait: Maximum wait between retries in seconds (default: 5.0)
        """
        self.huggingface_token = huggingface_token
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.min_retry_wait = min_retry_wait
        self.max_retry_wait = max_retry_wait

        # Initialize client instance (lazy initialization)
        self._client = None
        self._client_initialized = False

    def _get_client(self):
        """
        Get or create the InferenceClient instance.

        Uses lazy initialization to avoid import errors during testing.

        Returns:
            InferenceClient instance or None if not available
        """
        if self._client_initialized:
            return self._client

        try:
            from huggingface_hub import InferenceClient

            if not self.huggingface_token:
                logger.warning(
                    "Hugging Face token not provided. LLM calls will use fallback."
                )
                self._client = None
            else:
                self._client = InferenceClient(
                    model=self.model_id,
                    token=self.huggingface_token,
                )
                logger.info(
                    f"Initialized Hugging Face InferenceClient with model {self.model_id}"
                )

            self._client_initialized = True

        except ImportError:
            logger.warning(
                "huggingface_hub not installed. LLM calls will use fallback."
            )
            self._client = None
            self._client_initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize Hugging Face InferenceClient: {e}")
            self._client = None
            self._client_initialized = True

        return self._client

    async def invoke(self, prompt: str) -> str:
        """
        Invoke the LLM with the given prompt.

        Uses retry logic with exponential backoff for resilience.

        Args:
            prompt: The formatted prompt to send to the LLM

        Returns:
            Generated response text

        Raises:
            LLMError: If LLM invocation fails after all retries
        """
        return await self._invoke_with_retry(prompt)

    async def _invoke_with_retry(self, prompt: str) -> str:
        """
        Invoke LLM with Tenacity retry logic.

        Retries on transient errors with exponential backoff.

        Args:
            prompt: The prompt to send

        Returns:
            Generated response text

        Raises:
            LLMError: If all retries are exhausted
        """
        # Create retry decorator dynamically to use instance config
        retry_decorator = retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(
                multiplier=1,
                min=self.min_retry_wait,
                max=self.max_retry_wait,
            ),
            retry=retry_if_exception_type(
                (ConnectionError, TimeoutError, Exception)
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

        @retry_decorator
        async def _do_invoke():
            return await self._invoke_llm(prompt)

        try:
            response = await _do_invoke()
            return self._clean_response(response)
        except Exception as e:
            logger.error(
                f"LLM invocation failed after {self.max_retries} attempts: {e}"
            )
            raise LLMError(f"Failed to invoke LLM: {e}") from e

    async def _invoke_llm(self, prompt: str) -> str:
        """
        Internal method to invoke the LLM.

        Args:
            prompt: Formatted prompt

        Returns:
            Raw LLM response text
        """
        client = self._get_client()

        if client is None:
            # Fallback for testing or when LLM is not available
            logger.warning("LLM not available, using fallback response")
            return self._generate_fallback_response(prompt)

        # Use asyncio to run the sync LLM call in a thread pool
        loop = asyncio.get_event_loop()

        def _call_hf():
            # We intentionally keep parameters minimal so this works
            # for both hosted inference endpoints and public models.
            try:
                return client.text_generation(
                    prompt,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                )
            except TypeError:
                # Older versions of huggingface_hub may not support
                # all kwargs; retry with the bare minimum.
                return client.text_generation(
                    prompt,
                    max_new_tokens=self.max_tokens,
                )

        response = await loop.run_in_executor(None, _call_hf)

        return response

    def _generate_fallback_response(self, prompt: str) -> str:
        """
        Generate a fallback response when LLM is not available.

        This is used for testing or when the LLM service is unavailable.

        Args:
            prompt: The original prompt

        Returns:
            Fallback response text
        """
        # Extract key information from prompt for contextual fallback
        prompt_lower = prompt.lower()

        if "fuel" in prompt_lower and "critical" in prompt_lower:
            return "Box box box! Fuel critical, pit this lap."
        elif "fuel" in prompt_lower:
            return "Fuel looking tight. Consider your pit window."
        elif "tire" in prompt_lower and "critical" in prompt_lower:
            return "Tires are gone! Box immediately."
        elif "tire" in prompt_lower:
            return "Tires are showing wear. Monitor carefully."
        elif "gap" in prompt_lower:
            return "Gap has changed. Adjust your pace accordingly."
        elif "lap" in prompt_lower:
            return "Good lap. Keep pushing."
        elif "position" in prompt_lower:
            return "Position update noted. Stay focused."
        else:
            return "Copy that. Monitoring the situation."

    def _clean_response(self, response: str) -> str:
        """
        Clean and format LLM response.

        Args:
            response: Raw LLM response

        Returns:
            Cleaned response string
        """
        if not response:
            return ""

        # Strip whitespace and normalize
        cleaned = response.strip()

        # Remove any potential prompt echoing
        # Some models may echo parts of the prompt
        if ":" in cleaned and cleaned.index(":") < 20:
            # Check if it looks like a role prefix (e.g., "Engineer:")
            potential_prefix = cleaned.split(":")[0].lower()
            if any(role in potential_prefix for role in ["engineer", "ai", "assistant", "response"]):
                cleaned = ":".join(cleaned.split(":")[1:]).strip()

        return cleaned

    def invoke_sync(self, prompt: str) -> str:
        """
        Synchronous version of invoke for non-async contexts.

        Args:
            prompt: The formatted prompt

        Returns:
            Generated response text

        Raises:
            LLMError: If LLM invocation fails
        """
        return asyncio.run(self.invoke(prompt))
