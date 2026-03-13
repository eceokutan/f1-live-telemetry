"""Base agent class for Jarvis Post."""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Base class for analysis agents."""

    @abstractmethod
    async def analyse(self, session_data: dict) -> dict:
        """Analyse session data and return results."""
        pass
