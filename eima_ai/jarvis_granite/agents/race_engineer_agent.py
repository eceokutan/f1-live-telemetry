"""
Race Engineer Agent for Jarvis-Granite Live Telemetry.

LLM-powered agent that generates race engineer responses using
IBM Granite via WatsonX. Handles both proactive (event-triggered)
and reactive (query-driven) response generation.

This agent:
- Formats prompts using LangChain templates
- Invokes LLM via LLMClient
- Manages conversation context
- Respects verbosity settings

Target latency: <2000ms for LLM response
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional, Union

from jarvis_granite.llm import LLMClient, LLMError
from jarvis_granite.schemas.events import Event
from jarvis_granite.live.context import LiveSessionContext
from jarvis_granite.prompts import (
    get_proactive_prompt,
    get_reactive_prompt,
    format_conversation_history,
)

logger = logging.getLogger(__name__)


class RaceEngineerAgent:
    """
    LLM-powered race engineer agent.

    Generates contextual responses for both proactive alerts
    and reactive driver queries using IBM Granite LLM.

    Attributes:
        llm_client: LLMClient instance for LLM invocation
        verbosity: Response verbosity level (minimal, moderate, verbose)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        verbosity: str = "moderate",
    ):
        """
        Initialize Race Engineer Agent.

        Args:
            llm_client: LLMClient instance for LLM invocation
            verbosity: Response verbosity - minimal, moderate, verbose (default: moderate)
        """
        self.llm_client = llm_client
        self.verbosity = verbosity

        # Get prompt templates based on verbosity
        self.proactive_prompt = get_proactive_prompt(verbosity)
        self.reactive_prompt = get_reactive_prompt(verbosity)

        logger.info(f"Initialized RaceEngineerAgent with verbosity={verbosity}")

    def set_verbosity(self, verbosity: str) -> None:
        """
        Update verbosity level and refresh prompt templates.

        Args:
            verbosity: New verbosity level
        """
        self.verbosity = verbosity
        self.proactive_prompt = get_proactive_prompt(verbosity)
        self.reactive_prompt = get_reactive_prompt(verbosity)
        logger.info(f"Updated verbosity to {verbosity}")

    async def generate_proactive_response(
        self,
        event: Event,
        context: LiveSessionContext,
    ) -> str:
        """
        Generate proactive response for detected event.

        Called when the telemetry agent detects an event that
        requires driver notification (e.g., fuel warning, tire critical).

        Args:
            event: Detected event with type, priority, and data
            context: Current session context

        Returns:
            Generated response text for the driver

        Raises:
            LLMError: If LLM invocation fails
        """
        start_time = time.time()

        # Format event data for the prompt
        event_data_str = self._format_event_data(event.data)

        # Format session context
        session_context_str = context.to_prompt_context()

        # Format conversation history
        conversation_str = format_conversation_history(
            list(context.conversation_history)
        )

        # Format the prompt
        prompt = self.proactive_prompt.format(
            event_type=event.type,
            event_data=event_data_str,
            session_context=session_context_str,
            conversation_history=conversation_str,
        )

        logger.debug(f"Proactive prompt for {event.type}: {len(prompt)} chars")

        try:
            # Invoke LLM
            response = await self.llm_client.invoke(prompt)

            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Proactive response for {event.type} generated in {elapsed_ms:.0f}ms"
            )

            return response

        except LLMError:
            raise
        except asyncio.TimeoutError:
            # Propagate timeout errors for proper handling upstream
            logger.error("Timeout during proactive response generation")
            raise
        except Exception as e:
            logger.error(f"Error generating proactive response: {e}")
            raise LLMError(f"Failed to generate proactive response: {e}") from e

    async def generate_reactive_response(
        self,
        query: str,
        context: LiveSessionContext,
    ) -> str:
        """
        Generate reactive response for driver query.

        Called when the driver asks a question via voice or text.

        Args:
            query: Driver's question or command
            context: Current session context

        Returns:
            Generated response text for the driver

        Raises:
            LLMError: If LLM invocation fails
        """
        start_time = time.time()

        # Format session context
        session_context_str = context.to_prompt_context()

        # Format conversation history
        conversation_str = format_conversation_history(
            list(context.conversation_history)
        )

        # Format the prompt
        prompt = self.reactive_prompt.format(
            query=query,
            session_context=session_context_str,
            conversation_history=conversation_str,
        )

        logger.debug(f"Reactive prompt for query: {len(prompt)} chars")

        try:
            # Invoke LLM
            response = await self.llm_client.invoke(prompt)

            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"Reactive response generated in {elapsed_ms:.0f}ms")

            # Store exchange in conversation history
            context.add_exchange(query, response)

            return response

        except LLMError:
            raise
        except asyncio.TimeoutError:
            # Propagate timeout errors for proper handling upstream
            logger.error("Timeout during reactive response generation")
            raise
        except Exception as e:
            logger.error(f"Error generating reactive response: {e}")
            raise LLMError(f"Failed to generate reactive response: {e}") from e

    def _format_event_data(self, data: Dict[str, Any]) -> str:
        """
        Format event data for prompt injection.

        Args:
            data: Event data dictionary

        Returns:
            Human-readable string representation
        """
        if not data:
            return "(No additional data)"

        # Create readable format
        parts = []
        for key, value in data.items():
            # Format key for readability
            readable_key = key.replace("_", " ").title()

            # Format value based on type
            if isinstance(value, float):
                formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)

            parts.append(f"{readable_key}: {formatted_value}")

        return ", ".join(parts)

    async def handle_event(
        self,
        event: Event,
        context: LiveSessionContext,
    ) -> Optional[str]:
        """
        Handle an event and generate response if appropriate.

        This method adds rate limiting to prevent message spam.

        Args:
            event: Detected event
            context: Current session context

        Returns:
            Generated response or None if skipped
        """
        # Check if we can send a proactive message
        min_interval = 10.0  # Default minimum interval
        if hasattr(context, 'config') and context.config:
            min_interval = getattr(
                context.config,
                'min_proactive_interval_seconds',
                10.0
            )

        if not context.can_send_proactive(min_interval):
            logger.debug(
                f"Skipping proactive message for {event.type} - too soon"
            )
            return None

        # Generate response
        response = await self.generate_proactive_response(event, context)

        # Mark that we sent a proactive message
        context.mark_proactive_sent()

        # Add to conversation history (proactive messages count as exchanges)
        context.add_exchange(f"[Event: {event.type}]", response)

        return response


def create_race_engineer_agent(
    watsonx_url: str,
    watsonx_project_id: str,
    watsonx_api_key: str,
    model_id: str = "ibm/granite-3-8b-instruct",
    verbosity: str = "moderate",
    max_retries: int = 3,
) -> RaceEngineerAgent:
    """
    Factory function to create a RaceEngineerAgent with LLMClient.

    Args:
        watsonx_url: WatsonX API URL
        watsonx_project_id: WatsonX project ID
        watsonx_api_key: WatsonX API key
        model_id: Model identifier
        verbosity: Response verbosity level
        max_retries: Maximum retry attempts

    Returns:
        Configured RaceEngineerAgent instance
    """
    llm_client = LLMClient(
        watsonx_url=watsonx_url,
        watsonx_project_id=watsonx_project_id,
        watsonx_api_key=watsonx_api_key,
        model_id=model_id,
        max_retries=max_retries,
    )

    return RaceEngineerAgent(
        llm_client=llm_client,
        verbosity=verbosity,
    )
