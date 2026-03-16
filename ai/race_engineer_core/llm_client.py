"""
LLM Client for Jarvis-Granite Live Telemetry.

Provides an interface to a locally-hosted GGUF model (via llama-cpp-python).
Falls back to deterministic rule-based responses when the local model
is unavailable (e.g. GGUF file missing, load failure).
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Exception raised for LLM-related errors."""

    pass


class LLMClient:
    """
    Client for the local GGUF race engineer LLM (via llama-cpp-python).

    Priority order in _invoke_llm:
    1. Force rule-based fallback (if configured, e.g. GGUF missing)
    2. Local LLM (GGUF quantized model)
    3. Rule-based fallback (if local LLM unavailable or generation fails)

    Attributes:
        max_tokens: Maximum tokens for response generation
        temperature: LLM temperature for response variety
    """

    def __init__(
        self,
        max_tokens: int = 48,
        temperature: float = 0.3,
        local_model_path: str = "race_engineer_gguf/granite-race-engineer-Q4_K_M.gguf",
        local_max_time_seconds: float = 5.0,
        force_rule_based_fallback: bool = False,
    ):
        """
        Initialize LLM Client.

        Args:
            max_tokens: Maximum response tokens (default: 24, short for racing brevity)
            temperature: Response temperature (default: 0.3)
            local_model_path: Path to GGUF model file (relative to project root).
            local_max_time_seconds: Max generation time for local model responses.
            force_rule_based_fallback: If True, bypass the local LLM and use
                                      rule-based fallback responses only.
        """
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.local_model_path = local_model_path
        self.local_max_time_seconds = local_max_time_seconds
        self.force_rule_based_fallback = force_rule_based_fallback

        # Local LLM inference
        self._local_llm = None
        self._local_llm_initialized = False
        self._local_llm_failed = False

        # Status callback (set by AIRaceEngineerWorker to relay UI updates)
        self._status_callback = None

        # Pre-load local LLM if not in forced-fallback mode
        if not self.force_rule_based_fallback:
            self._init_local_llm()

    def set_status_callback(self, callback):
        """Set a callback for status updates (e.g. to emit Qt signals)."""
        self._status_callback = callback

    def _emit_status(self, message: str):
        """Emit a status update through the callback if set."""
        if self._status_callback:
            self._status_callback(message)

    def _init_local_llm(self) -> None:
        """Initialize local LLM inference, reusing the shared singleton if available."""
        if self._local_llm_initialized:
            return

        try:
            from ai.local_llm_inference import LocalLLMInference

            logger.info("Acquiring shared local LLM instance (GGUF)...")
            self._local_llm = LocalLLMInference.get_shared(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                model_path=self.local_model_path,
                max_time_seconds=self.local_max_time_seconds,
            )
            self._local_llm_initialized = True
            logger.info("Local LLM ready (shared instance)")

        except Exception as e:
            logger.error(f"Failed to initialize local LLM: {e}")
            self._local_llm = None
            self._local_llm_initialized = True
            self._local_llm_failed = True
            self._emit_status(
                "Local LLM failed to load. Using rule-based fallback responses."
            )

    def _get_local_llm(self):
        """Get local LLM instance (if available)."""
        if not self._local_llm_initialized:
            self._init_local_llm()
        return self._local_llm

    async def invoke(self, prompt: str) -> str:
        """
        Invoke the LLM with the given prompt.

        Args:
            prompt: The formatted prompt to send to the LLM

        Returns:
            Generated response text

        Raises:
            LLMError: If LLM invocation fails and no fallback is possible
        """
        try:
            response = await self._invoke_llm(prompt)
            return self._clean_response(response)
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            raise LLMError(f"Failed to invoke LLM: {e}") from e

    async def _invoke_llm(self, prompt: str) -> str:
        """
        Internal method to invoke the LLM.

        Priority order:
        1. Force rule-based fallback (if configured)
        2. Local LLM (if loaded)
        3. Rule-based fallback (if local LLM unavailable or generation fails)

        Args:
            prompt: Formatted prompt

        Returns:
            Raw LLM response text
        """
        if self.force_rule_based_fallback:
            logger.info("LLM disabled by configuration, using rule-based fallback response")
            return self._generate_fallback_response(prompt)

        # Try local LLM
        local_llm = self._get_local_llm()
        if local_llm is not None:
            loop = asyncio.get_event_loop()
            try:
                response = await loop.run_in_executor(None, local_llm.generate, prompt)
                return response
            except Exception as e:
                logger.error(f"Local LLM generation failed: {e}")
                self._emit_status(
                    "Local LLM generation failed. Using rule-based fallback."
                )
                return self._generate_fallback_response(prompt)

        # Local LLM not available — use rule-based fallback
        logger.warning("Local LLM not available, using rule-based fallback response")
        return self._generate_fallback_response(prompt)

    def _generate_fallback_response(self, prompt: str) -> str:
        """
        Generate a deterministic fallback response when LLM is not available.

        Args:
            prompt: The original prompt

        Returns:
            Fallback response text
        """
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

        cleaned = response.strip()

        # Remove any potential prompt echoing (e.g., "Engineer: ...")
        if ":" in cleaned and cleaned.index(":") < 20:
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
