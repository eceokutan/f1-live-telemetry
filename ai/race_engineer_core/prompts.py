"""
Prompt templates for Jarvis-Granite Live Telemetry.

Defines lightweight prompt template objects used to format prompts
for the race engineer LLM. This avoids a hard dependency on LangChain
while preserving the same `.format(**kwargs)` interface.

All prompts support three verbosity levels: minimal, moderate, verbose.
"""


class PromptTemplate:
    def __init__(self, input_variables, template: str):
        self.input_variables = input_variables
        self.template = template

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

LIVE_SYSTEM_PROMPT = """You are an expert F1 race engineer communicating with your driver over team radio during a live race.

CRITICAL CONSTRAINTS:
- Driver is actively racing and cannot read text
- Responses must be CONCISE (1-3 sentences max)
- Lead with the most important information
- Use precise numbers when helpful
- Match urgency to the situation

CURRENT SESSION:
{session_context}

VERBOSITY: {verbosity_level}
{verbosity_instructions}

CONVERSATION HISTORY:
{conversation_history}
"""


# =============================================================================
# VERBOSITY INSTRUCTIONS
# =============================================================================

VERBOSITY_INSTRUCTIONS = {
    "minimal": "Keep responses under 15 words. Be extremely brief.",
    "moderate": "Keep responses to 1-2 sentences. Be direct but informative.",
    "verbose": "Provide detailed responses up to 4 sentences with reasoning."
}


# =============================================================================
# PROACTIVE PROMPTS (Event-Triggered)
# =============================================================================

PROACTIVE_PROMPT_MINIMAL = PromptTemplate(
    input_variables=["event_type", "event_data", "session_context", "conversation_history"],
    template="""Alert: {event_type}. Data: {event_data}

Alert:"""
)

PROACTIVE_PROMPT_MODERATE = PromptTemplate(
    input_variables=["event_type", "event_data", "session_context", "conversation_history"],
    template="""Event: {event_type}
Details: {event_data}

Data:
{session_context}

Alert:"""
)

PROACTIVE_PROMPT_VERBOSE = PromptTemplate(
    input_variables=["event_type", "event_data", "session_context", "conversation_history"],
    template="""Event Type: {event_type}
Event Details: {event_data}

Session Data:
{session_context}

Recent conversation:
{conversation_history}

Reply in under 4 sentences.
Alert:"""
)


# =============================================================================
# REACTIVE PROMPTS (Query-Driven)
# =============================================================================

REACTIVE_PROMPT_MINIMAL = PromptTemplate(
    input_variables=["query", "session_context", "conversation_history"],
    template="""Driver asks: "{query}"
Data: {session_context}

Answer:"""
)

REACTIVE_PROMPT_MODERATE = PromptTemplate(
    input_variables=["query", "session_context", "conversation_history"],
    template="""Question: "{query}"

Data:
{session_context}

Answer:"""
)

REACTIVE_PROMPT_VERBOSE = PromptTemplate(
    input_variables=["query", "session_context", "conversation_history"],
    template="""Driver's Question: "{query}"

Session Data:
{session_context}

Recent conversation:
{conversation_history}

Reply in under 4 sentences.
Answer:"""
)


# =============================================================================
# PROMPT REGISTRY
# =============================================================================

PROACTIVE_PROMPTS = {
    "minimal": PROACTIVE_PROMPT_MINIMAL,
    "moderate": PROACTIVE_PROMPT_MODERATE,
    "verbose": PROACTIVE_PROMPT_VERBOSE,
}

REACTIVE_PROMPTS = {
    "minimal": REACTIVE_PROMPT_MINIMAL,
    "moderate": REACTIVE_PROMPT_MODERATE,
    "verbose": REACTIVE_PROMPT_VERBOSE,
}


def get_proactive_prompt(verbosity: str = "moderate") -> PromptTemplate:
    """
    Get the proactive prompt template for the given verbosity level.

    Args:
        verbosity: Verbosity level (minimal, moderate, verbose)

    Returns:
        PromptTemplate for proactive responses
    """
    return PROACTIVE_PROMPTS.get(verbosity, PROACTIVE_PROMPT_MODERATE)


def get_reactive_prompt(verbosity: str = "moderate") -> PromptTemplate:
    """
    Get the reactive prompt template for the given verbosity level.

    Args:
        verbosity: Verbosity level (minimal, moderate, verbose)

    Returns:
        PromptTemplate for reactive responses
    """
    return REACTIVE_PROMPTS.get(verbosity, REACTIVE_PROMPT_MODERATE)


def format_conversation_history(history: list) -> str:
    """
    Format conversation history for prompt injection.

    Args:
        history: List of conversation exchanges with 'query' and 'response' keys

    Returns:
        Formatted string of conversation history
    """
    if not history:
        return "(No previous conversation)"

    formatted = []
    for exchange in history:
        query = exchange.get("query", "")
        response = exchange.get("response", "")
        formatted.append(f"Driver: {query}\nEngineer: {response}")

    return "\n\n".join(formatted)
