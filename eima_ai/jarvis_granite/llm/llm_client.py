"""
LLM Client for Jarvis-Granite Live Telemetry.

Provides interface to IBM Granite LLM via WatsonX using LangChain.
Implements retry logic with Tenacity for robust API calls.

Features:
- LangChain integration with WatsonxLLM
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
    Client for IBM Granite LLM via WatsonX using LangChain.

    This is a lightweight wrapper around WatsonxLLM that handles:
    - Connection configuration
    - Retry logic with Tenacity
    - Async invocation

    The actual prompt formatting is handled by RaceEngineerAgent.

    Attributes:
        model_id: WatsonX model identifier
        max_tokens: Maximum tokens for response generation
        temperature: LLM temperature for response variety
        max_retries: Maximum retry attempts for failed requests
    """

    def __init__(
        self,
        watsonx_url: str,
        watsonx_project_id: str,
        watsonx_api_key: str,
        model_id: str = "ibm/granite-3-8b-instruct",
        max_tokens: int = 150,
        temperature: float = 0.7,
        max_retries: int = 3,
        min_retry_wait: float = 1.0,
        max_retry_wait: float = 5.0,
    ):
        """
        Initialize LLM Client.

        Args:
            watsonx_url: WatsonX API URL
            watsonx_project_id: WatsonX project ID
            watsonx_api_key: WatsonX API key
            model_id: Model identifier (default: ibm/granite-3-8b-instruct)
            max_tokens: Maximum response tokens (default: 150)
            temperature: Response temperature (default: 0.7)
            max_retries: Max retry attempts (default: 3)
            min_retry_wait: Minimum wait between retries in seconds (default: 1.0)
            max_retry_wait: Maximum wait between retries in seconds (default: 5.0)
        """
        self.watsonx_url = watsonx_url
        self.watsonx_project_id = watsonx_project_id
        self.watsonx_api_key = watsonx_api_key
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.min_retry_wait = min_retry_wait
        self.max_retry_wait = max_retry_wait

        # Initialize LLM instance (lazy initialization)
        self._llm = None
        self._llm_initialized = False

    def _get_llm(self):
        """
        Get or create the WatsonxLLM instance.

        Uses lazy initialization to avoid import errors during testing.

        Returns:
            WatsonxLLM instance or None if not available
        """
        if self._llm_initialized:
            return self._llm

        try:
            from langchain_ibm import WatsonxLLM

            self._llm = WatsonxLLM(
                model_id=self.model_id,
                url=self.watsonx_url,
                project_id=self.watsonx_project_id,
                apikey=self.watsonx_api_key,
                params={
                    "max_new_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "decoding_method": "greedy",
                },
            )
            self._llm_initialized = True
            logger.info(f"Initialized WatsonxLLM with model {self.model_id}")

        except ImportError:
            logger.warning(
                "langchain_ibm not installed. LLM calls will use fallback."
            )
            self._llm = None
            self._llm_initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize WatsonxLLM: {e}")
            self._llm = None
            self._llm_initialized = True

        return self._llm

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
                max=self.max_retry_wait
            ),
            retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)),
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
            logger.error(f"LLM invocation failed after {self.max_retries} attempts: {e}")
            raise LLMError(f"Failed to invoke LLM: {e}") from e

    async def _invoke_llm(self, prompt: str) -> str:
        """
        Internal method to invoke the LLM.

        Args:
            prompt: Formatted prompt

        Returns:
            Raw LLM response text
        """
        llm = self._get_llm()

        if llm is None:
            # Fallback for testing or when LLM is not available
            logger.warning("LLM not available, using fallback response")
            return self._generate_fallback_response(prompt)

        # Use asyncio to run the sync LLM call in a thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, llm.invoke, prompt)

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
