"""
TDD Tests for LLM Client and Race Engineer Agent - Phase 3, Section 6-7

These tests define the expected behavior for:
1. LLM Client initialization with WatsonX configuration
2. LLM invocation with retry logic
3. Race Engineer Agent prompt formatting and response generation
4. Context formatting for prompts
5. Verbosity levels

Run with: pytest tests/test_llm_client.py -v
"""

import pytest
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from jarvis_granite.schemas.events import Event, Priority
from jarvis_granite.live.context import LiveSessionContext


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_config():
    """Mock configuration for LLM client."""
    return {
        "watsonx_url": "https://us-south.ml.cloud.ibm.com",
        "watsonx_project_id": "test-project-id",
        "watsonx_api_key": "test-api-key",
        "model_id": "ibm/granite-3-8b-instruct",
        "max_tokens": 150,
        "temperature": 0.7
    }


@pytest.fixture
def session_context():
    """Create a session context for testing."""
    context = LiveSessionContext(
        session_id="race_001",
        source="torcs",
        track_name="Monza"
    )
    context.current_lap = 15
    context.current_sector = 2
    context.position = 3
    context.fuel_remaining = 35.0
    context.fuel_consumption_per_lap = 3.5
    context.tire_temps = {"fl": 95, "fr": 96, "rl": 92, "rr": 93}
    context.tire_wear = {"fl": 25, "fr": 26, "rl": 22, "rr": 23}
    context.gap_ahead = 2.5
    context.gap_behind = 1.8
    context.best_lap = 81.234
    context.last_lap = 82.456
    return context


@pytest.fixture
def fuel_critical_event():
    """Create a fuel critical event."""
    return Event(
        type="fuel_critical",
        priority=Priority.CRITICAL,
        data={"laps": 1.5},
        timestamp=1705329125.123
    )


@pytest.fixture
def tire_warning_event():
    """Create a tire warning event."""
    return Event(
        type="tire_warning",
        priority=Priority.MEDIUM,
        data={"temp": 105.0, "position": "fl"},
        timestamp=1705329125.456
    )


@pytest.fixture
def lap_complete_event():
    """Create a lap complete event."""
    return Event(
        type="lap_complete",
        priority=Priority.MEDIUM,
        data={"lap": 15, "time": 82.456, "best": 81.234},
        timestamp=1705329125.789
    )


# =============================================================================
# LLM CLIENT CREATION
# =============================================================================

class TestLLMClientCreation:
    """Tests for LLMClient instantiation."""

    def test_create_llm_client(self, mock_config):
        """Should create LLMClient instance."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        assert client is not None

    def test_llm_client_stores_config(self, mock_config):
        """LLMClient should store configuration."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        assert client.model_id == "ibm/granite-3-8b-instruct"
        assert client.max_tokens == 150

    def test_llm_client_has_invoke_method(self, mock_config):
        """LLMClient should have invoke method."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        assert hasattr(client, 'invoke')
        assert callable(client.invoke)

    def test_llm_client_default_values(self):
        """LLMClient should have sensible defaults."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(
            watsonx_url="https://test.com",
            watsonx_project_id="test-project",
            watsonx_api_key="test-key"
        )

        assert client.model_id == "ibm/granite-3-8b-instruct"
        assert client.max_tokens == 150
        assert client.temperature == 0.7


# =============================================================================
# LLM CLIENT INVOCATION
# =============================================================================

class TestLLMClientInvocation:
    """Tests for LLM invocation."""

    @pytest.mark.asyncio
    async def test_invoke_returns_string(self, mock_config):
        """invoke should return a string."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        # Mock the internal LLM call
        with patch.object(client, '_invoke_llm', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = "Test response"

            response = await client.invoke("Test prompt")

            assert isinstance(response, str)
            assert len(response) > 0

    @pytest.mark.asyncio
    async def test_invoke_cleans_response(self, mock_config):
        """invoke should clean the response."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        with patch.object(client, '_invoke_llm', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = "  Response with spaces  \n\n"

            response = await client.invoke("Test prompt")

            assert response == "Response with spaces"

    @pytest.mark.asyncio
    async def test_invoke_handles_empty_response(self, mock_config):
        """invoke should handle empty responses."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config)

        with patch.object(client, '_invoke_llm', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = ""

            response = await client.invoke("Test prompt")

            assert isinstance(response, str)


# =============================================================================
# RETRY LOGIC
# =============================================================================

class TestRetryLogic:
    """Tests for retry logic with Tenacity."""

    def test_retry_config_stored(self, mock_config):
        """Should store retry configuration."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config, max_retries=5)

        assert client.max_retries == 5

    def test_max_retries_configurable(self, mock_config):
        """Max retries should be configurable."""
        from jarvis_granite.llm import LLMClient

        client = LLMClient(**mock_config, max_retries=5)

        assert client.max_retries == 5

    @pytest.mark.asyncio
    async def test_raises_llm_error_on_failure(self, mock_config):
        """Should raise LLMError after retries exhausted."""
        from jarvis_granite.llm import LLMClient, LLMError

        client = LLMClient(**mock_config, max_retries=1)

        with patch.object(client, '_invoke_llm', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = Exception("LLM unavailable")

            with pytest.raises(LLMError):
                await client.invoke("Test prompt")


# =============================================================================
# RACE ENGINEER AGENT CREATION
# =============================================================================

class TestRaceEngineerAgentCreation:
    """Tests for RaceEngineerAgent instantiation."""

    def test_create_race_engineer_agent(self, mock_config):
        """Should create RaceEngineerAgent instance."""
        from jarvis_granite.llm import LLMClient
        from jarvis_granite.agents import RaceEngineerAgent

        llm_client = LLMClient(**mock_config)
        agent = RaceEngineerAgent(llm_client=llm_client)

        assert agent is not None

    def test_agent_has_required_methods(self, mock_config):
        """RaceEngineerAgent should have required methods."""
        from jarvis_granite.llm import LLMClient
        from jarvis_granite.agents import RaceEngineerAgent

        llm_client = LLMClient(**mock_config)
        agent = RaceEngineerAgent(llm_client=llm_client)

        assert hasattr(agent, 'generate_proactive_response')
        assert callable(agent.generate_proactive_response)
        assert hasattr(agent, 'generate_reactive_response')
        assert callable(agent.generate_reactive_response)

    def test_agent_has_prompt_templates(self, mock_config):
        """RaceEngineerAgent should have prompt templates."""
        from jarvis_granite.llm import LLMClient
        from jarvis_granite.agents import RaceEngineerAgent

        llm_client = LLMClient(**mock_config)
        agent = RaceEngineerAgent(llm_client=llm_client)

        assert hasattr(agent, 'proactive_prompt')
        assert agent.proactive_prompt is not None
        assert hasattr(agent, 'reactive_prompt')
        assert agent.reactive_prompt is not None


# =============================================================================
# PROACTIVE RESPONSE GENERATION
# =============================================================================

class TestProactiveResponseGeneration:
    """Tests for proactive (event-triggered) response generation."""

    @pytest.mark.asyncio
    async def test_generate_proactive_response_returns_string(
        self, mock_config, fuel_critical_event, session_context
    ):
        """generate_proactive_response should return a string."""
        from jarvis_granite.llm import LLMClient
        from jarvis_granite.agents import RaceEngineerAgent

        llm_client = LLMClient(**mock_config)
        agent = RaceEngineerAgent(llm_client=llm_client)

        with patch.object(llm_client, 'invoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = "Box box box! Fuel critical."

            response = await agent.generate_proactive_response(
                event=fuel_critical_event,
                context=session_context
            )

            assert isinstance(response, str)
            assert len(response) > 0

    @pytest.mark.asyncio
    async def test_proactive_response_uses_event_data(
        self, mock_config, fuel_critical_event, session_context
    ):
        """Proactive response should incorporate event data."""
        from jarvis_granite.llm import LLMClient
        from jarvis_granite.agents import RaceEngineerAgent

        llm_client = LLMClient(**mock_config)
        agent = RaceEngineerAgent(llm_client=llm_client)

        with patch.object(llm_client, 'invoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = "Fuel critical, 1.5 laps remaining!"

            await agent.generate_proactive_response(
                event=fuel_critical_event,
                context=session_context
            )

            # Check the prompt was formatted with event data
            call_args = mock_invoke.call_args[0][0]
            assert "fuel" in call_args.lower() or "critical" in call_args.lower()

    @pytest.mark.asyncio
    async def test_proactive_response_uses_context(
        self, mock_config, lap_complete_event, session_context
    ):
        """Proactive response should incorporate session context."""
        from jarvis_granite.llm import LLMClient
        from jarvis_granite.agents import RaceEngineerAgent

        llm_client = LLMClient(**mock_config)
        agent = RaceEngineerAgent(llm_client=llm_client)

        with patch.object(llm_client, 'invoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = "Lap 15 complete."

            await agent.generate_proactive_response(
                event=lap_complete_event,
                context=session_context
            )

            # Check context was included
            call_args = mock_invoke.call_args[0][0]
            assert "Monza" in call_args or "15" in call_args

    @pytest.mark.asyncio
    async def test_proactive_response_different_event_types(
        self, mock_config, session_context
    ):
        """Should handle different event types appropriately."""
        from jarvis_granite.llm import LLMClient
        from jarvis_granite.agents import RaceEngineerAgent

        llm_client = LLMClient(**mock_config)
        agent = RaceEngineerAgent(llm_client=llm_client)

        events = [
            Event(type="fuel_critical", priority=Priority.CRITICAL,
                  data={"laps": 1.5}, timestamp=1.0),
            Event(type="tire_warning", priority=Priority.MEDIUM,
                  data={"temp": 105, "position": "fl"}, timestamp=2.0),
            Event(type="gap_change", priority=Priority.MEDIUM,
                  data={"change": 1.5, "direction": "ahead"}, timestamp=3.0),
        ]

        with patch.object(llm_client, 'invoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = "Response"

            for event in events:
                response = await agent.generate_proactive_response(
                    event=event,
                    context=session_context
                )
                assert isinstance(response, str)


# =============================================================================
# REACTIVE RESPONSE GENERATION
# =============================================================================

class TestReactiveResponseGeneration:
    """Tests for reactive (query-driven) response generation."""

    @pytest.mark.asyncio
    async def test_generate_reactive_response_returns_string(
        self, mock_config, session_context
    ):
        """generate_reactive_response should return a string."""
        from jarvis_granite.llm import LLMClient
        from jarvis_granite.agents import RaceEngineerAgent

        llm_client = LLMClient(**mock_config)
        agent = RaceEngineerAgent(llm_client=llm_client)

        with patch.object(llm_client, 'invoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = "Tires are looking good."

            response = await agent.generate_reactive_response(
                query="How are my tires?",
                context=session_context
            )

            assert isinstance(response, str)
            assert len(response) > 0

    @pytest.mark.asyncio
    async def test_reactive_response_uses_query(
        self, mock_config, session_context
    ):
        """Reactive response should incorporate the query."""
        from jarvis_granite.llm import LLMClient
        from jarvis_granite.agents import RaceEngineerAgent

        llm_client = LLMClient(**mock_config)
        agent = RaceEngineerAgent(llm_client=llm_client)

        with patch.object(llm_client, 'invoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = "Fuel is at 35 liters."

            await agent.generate_reactive_response(
                query="What's my fuel status?",
                context=session_context
            )

            call_args = mock_invoke.call_args[0][0]
            assert "fuel" in call_args.lower()

    @pytest.mark.asyncio
    async def test_reactive_response_uses_context(
        self, mock_config, session_context
    ):
        """Reactive response should incorporate session context."""
        from jarvis_granite.llm import LLMClient
        from jarvis_granite.agents import RaceEngineerAgent

        llm_client = LLMClient(**mock_config)
        agent = RaceEngineerAgent(llm_client=llm_client)

        with patch.object(llm_client, 'invoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = "You're in P3."

            await agent.generate_reactive_response(
                query="What's my position?",
                context=session_context
            )

            call_args = mock_invoke.call_args[0][0]
            # Context should be included
            assert "P3" in call_args or "3" in call_args or "Monza" in call_args

    @pytest.mark.asyncio
    async def test_reactive_response_stores_exchange(
        self, mock_config, session_context
    ):
        """Reactive response should store exchange in conversation history."""
        from jarvis_granite.llm import LLMClient
        from jarvis_granite.agents import RaceEngineerAgent

        llm_client = LLMClient(**mock_config)
        agent = RaceEngineerAgent(llm_client=llm_client)

        with patch.object(llm_client, 'invoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = "Tires are good."

            await agent.generate_reactive_response(
                query="How are my tires?",
                context=session_context
            )

            # Check conversation history was updated
            assert len(session_context.conversation_history) == 1
            exchange = session_context.conversation_history[0]
            assert "tires" in exchange["query"].lower()


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

class TestContextFormatting:
    """Tests for session context formatting."""

    def test_context_to_prompt_context(self, session_context):
        """Session context should format to prompt-friendly string."""
        formatted = session_context.to_prompt_context()

        assert isinstance(formatted, str)
        assert "Monza" in formatted
        assert "15" in formatted  # Lap number

    def test_context_includes_fuel_info(self, session_context):
        """Formatted context should include fuel information."""
        formatted = session_context.to_prompt_context()

        assert "35" in formatted or "fuel" in formatted.lower()

    def test_context_includes_tire_info(self, session_context):
        """Formatted context should include tire information."""
        formatted = session_context.to_prompt_context()

        assert "tire" in formatted.lower() or "95" in formatted

    def test_context_includes_position_info(self, session_context):
        """Formatted context should include position information."""
        formatted = session_context.to_prompt_context()

        assert "P3" in formatted or "position" in formatted.lower()


# =============================================================================
# RACE ENGINEER PERSONALITY
# =============================================================================

class TestRaceEngineerPersonality:
    """Tests for race engineer personality in prompts."""

    def test_proactive_prompt_has_race_engineer_context(self, mock_config):
        """Proactive prompt should establish race engineer role."""
        from jarvis_granite.llm import LLMClient
        from jarvis_granite.agents import RaceEngineerAgent

        llm_client = LLMClient(**mock_config)
        agent = RaceEngineerAgent(llm_client=llm_client)

        # Check the template includes race engineer role
        template_str = agent.proactive_prompt.template
        prompt_lower = template_str.lower()
        assert any(term in prompt_lower for term in
                   ["race engineer", "engineer", "driver", "f1"])

    def test_reactive_prompt_has_race_engineer_context(self, mock_config):
        """Reactive prompt should establish race engineer role."""
        from jarvis_granite.llm import LLMClient
        from jarvis_granite.agents import RaceEngineerAgent

        llm_client = LLMClient(**mock_config)
        agent = RaceEngineerAgent(llm_client=llm_client)

        template_str = agent.reactive_prompt.template
        prompt_lower = template_str.lower()
        assert any(term in prompt_lower for term in
                   ["race engineer", "engineer", "driver", "respond"])


# =============================================================================
# VERBOSITY LEVELS
# =============================================================================

class TestVerbosityLevels:
    """Tests for response verbosity control."""

    def test_verbosity_affects_prompt(self, mock_config):
        """Verbosity setting should affect prompt templates."""
        from jarvis_granite.llm import LLMClient
        from jarvis_granite.agents import RaceEngineerAgent

        llm_client = LLMClient(**mock_config)

        agent_minimal = RaceEngineerAgent(llm_client=llm_client, verbosity="minimal")
        agent_verbose = RaceEngineerAgent(llm_client=llm_client, verbosity="verbose")

        # Templates should be different based on verbosity
        assert agent_minimal.reactive_prompt.template != agent_verbose.reactive_prompt.template

    def test_default_verbosity_is_moderate(self, mock_config):
        """Default verbosity should be moderate."""
        from jarvis_granite.llm import LLMClient
        from jarvis_granite.agents import RaceEngineerAgent

        llm_client = LLMClient(**mock_config)
        agent = RaceEngineerAgent(llm_client=llm_client)

        assert agent.verbosity == "moderate"

    def test_set_verbosity_changes_templates(self, mock_config):
        """set_verbosity should update prompt templates."""
        from jarvis_granite.llm import LLMClient
        from jarvis_granite.agents import RaceEngineerAgent

        llm_client = LLMClient(**mock_config)
        agent = RaceEngineerAgent(llm_client=llm_client, verbosity="minimal")

        original_template = agent.reactive_prompt.template

        agent.set_verbosity("verbose")

        assert agent.verbosity == "verbose"
        assert agent.reactive_prompt.template != original_template


# =============================================================================
# ERROR HANDLING
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in Race Engineer Agent."""

    @pytest.mark.asyncio
    async def test_handles_llm_error_gracefully(self, mock_config, session_context):
        """Should propagate LLMError from client."""
        from jarvis_granite.llm import LLMClient, LLMError
        from jarvis_granite.agents import RaceEngineerAgent

        llm_client = LLMClient(**mock_config)
        agent = RaceEngineerAgent(llm_client=llm_client)

        with patch.object(llm_client, 'invoke', new_callable=AsyncMock) as mock_invoke:
            mock_invoke.side_effect = LLMError("LLM unavailable")

            with pytest.raises(LLMError):
                await agent.generate_reactive_response(
                    query="Test",
                    context=session_context
                )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

class TestFactoryFunction:
    """Tests for create_race_engineer_agent factory function."""

    def test_create_race_engineer_agent_factory(self):
        """Factory should create configured agent."""
        from jarvis_granite.agents import create_race_engineer_agent

        agent = create_race_engineer_agent(
            watsonx_url="https://test.com",
            watsonx_project_id="test-project",
            watsonx_api_key="test-key",
            verbosity="verbose"
        )

        assert agent is not None
        assert agent.verbosity == "verbose"

    def test_factory_creates_llm_client(self):
        """Factory should create LLM client internally."""
        from jarvis_granite.agents import create_race_engineer_agent

        agent = create_race_engineer_agent(
            watsonx_url="https://test.com",
            watsonx_project_id="test-project",
            watsonx_api_key="test-key"
        )

        assert agent.llm_client is not None
