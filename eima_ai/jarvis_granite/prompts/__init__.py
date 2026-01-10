"""
Prompts module for Jarvis-Granite Live Telemetry.

Contains LangChain PromptTemplate objects for race engineer responses.
"""

from jarvis_granite.prompts.live_prompts import (
    LIVE_SYSTEM_PROMPT,
    VERBOSITY_INSTRUCTIONS,
    PROACTIVE_PROMPTS,
    REACTIVE_PROMPTS,
    get_proactive_prompt,
    get_reactive_prompt,
    format_conversation_history,
)

__all__ = [
    "LIVE_SYSTEM_PROMPT",
    "VERBOSITY_INSTRUCTIONS",
    "PROACTIVE_PROMPTS",
    "REACTIVE_PROMPTS",
    "get_proactive_prompt",
    "get_reactive_prompt",
    "format_conversation_history",
]
