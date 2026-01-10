"""
LLM module for Jarvis-Granite Live Telemetry.

This module contains the LLM client for IBM Granite via WatsonX:

- LLMClient: Wrapper around WatsonxLLM with retry logic
- LLMError: Exception for LLM-related errors
"""

from jarvis_granite.llm.llm_client import LLMClient, LLMError

__all__ = ["LLMClient", "LLMError"]
