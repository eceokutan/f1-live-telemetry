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

import requests
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
        endpoint_url: Optional[str] = None,
        space_url: Optional[str] = None,
        space_skip_ssl_verify: bool = False,
        use_local_llm: bool = False,
        local_adapter_path: str = "race_engineer_llm",
        force_rule_based_fallback: bool = False,
    ):
        """
        Initialize LLM Client.

        Args:
            huggingface_token: Hugging Face access token (HUGGINGFACE_TOKEN / HUGGINGFACE_API_KEY)
            model_id: Model identifier on Hugging Face (e.g. "org/custom-race-engineer").
                      Used when no endpoint_url is provided.
            max_tokens: Maximum response tokens (default: 75, reduced for racing brevity)
            temperature: Response temperature (default: 0.7)
            max_retries: Max retry attempts (default: 3)
            min_retry_wait: Minimum wait between retries in seconds (default: 1.0)
            max_retry_wait: Maximum wait between retries in seconds (default: 5.0)
            endpoint_url: Optional Hugging Face Inference Endpoint URL. When set,
                          requests will be sent to this endpoint instead of the
                          generic model API.
            space_url: Optional Hugging Face Space URL (FastAPI-style backend)
                       that exposes a /chat endpoint. When set, this is preferred
                       over endpoint_url/model_id and will be called via HTTP POST.
            space_skip_ssl_verify: If True, skip SSL cert verification for Space
                                  requests only (use when HF Space hostname/cert mismatch).
            use_local_llm: If True, use local QLoRA-finetuned model instead of HF APIs.
            local_adapter_path: Path to QLoRA adapter directory (relative to project root).
            force_rule_based_fallback: If True, bypass all LLM backends and use
                                      rule-based fallback responses only.
        """
        self.huggingface_token = huggingface_token
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.min_retry_wait = min_retry_wait
        self.max_retry_wait = max_retry_wait
        self.endpoint_url = endpoint_url
        self.space_url = space_url
        self.space_skip_ssl_verify = space_skip_ssl_verify
        self.use_local_llm = use_local_llm
        self.local_adapter_path = local_adapter_path
        self.force_rule_based_fallback = force_rule_based_fallback

        # Initialize client instance (lazy initialization)
        self._client = None
        self._client_initialized = False

        # Local LLM inference
        self._local_llm = None
        self._local_llm_initialized = False

        # Pre-load local LLM if enabled
        if self.use_local_llm:
            self._init_local_llm()

    def _init_local_llm(self) -> None:
        """Initialize local LLM inference (pre-loaded on startup)."""
        if self._local_llm_initialized:
            return

        try:
            from ai.local_llm_inference import LocalLLMInference

            logger.info("Pre-loading local LLM (Granite-4.0-micro + QLoRA)...")
            self._local_llm = LocalLLMInference(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                adapter_path=self.local_adapter_path,
            )
            self._local_llm.load()
            self._local_llm_initialized = True
            logger.info("Local LLM ready")

        except Exception as e:
            logger.error(f"Failed to initialize local LLM: {e}")
            self._local_llm = None
            self._local_llm_initialized = True

    def _get_local_llm(self):
        """Get local LLM instance (if available)."""
        if not self.use_local_llm:
            return None
        if not self._local_llm_initialized:
            self._init_local_llm()
        return self._local_llm

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
                # Prefer a dedicated Hugging Face Inference Endpoint if configured.
                if self.endpoint_url:
                    self._client = InferenceClient(
                        base_url=self.endpoint_url,
                        token=self.huggingface_token,
                    )
                    logger.info(
                        f"Initialized Hugging Face InferenceClient with endpoint {self.endpoint_url}"
                    )
                else:
                    # Fallback to generic model API using model_id.
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

        Priority order:
        1. Local LLM (if enabled and loaded)
        2. HF Space (if configured)
        3. HF Inference Endpoint (if configured)
        4. Generic HF Model API

        Args:
            prompt: Formatted prompt

        Returns:
            Raw LLM response text
        """
        if self.force_rule_based_fallback:
            logger.info("LLM disabled by configuration, using rule-based fallback response")
            return self._generate_fallback_response(prompt)

        # Try local LLM first
        local_llm = self._get_local_llm()
        if local_llm is not None:
            loop = asyncio.get_event_loop()
            try:
                response = await loop.run_in_executor(None, local_llm.generate, prompt)
                return response
            except Exception as e:
                logger.error(f"Local LLM generation failed: {e}")
                raise

        # If a custom Space backend is configured, prefer calling it directly.
        if self.space_url:
            loop = asyncio.get_event_loop()

            def _call_space():
                try:
                    resp = requests.post(
                        self.space_url,
                        json={
                            "prompt": prompt,
                            "max_new_tokens": self.max_tokens,
                            "temperature": self.temperature,
                        },
                        timeout=30,
                        verify=not self.space_skip_ssl_verify,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    # Expecting {"response": "..."} from the Space
                    return data.get("response", "").strip()
                except Exception as e:
                    logger.error(f"Error calling Hugging Face Space at {self.space_url}: {e}")
                    raise

            return await loop.run_in_executor(None, _call_space)

        # Otherwise, fall back to Hugging Face model / endpoint APIs.
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
