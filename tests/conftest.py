"""
Pytest configuration and fixtures for F1 Telemetry tests.

Provides mocking for hardware dependencies that may not be available
in all test environments (pyaudio, torch).
"""

import sys
from unittest.mock import MagicMock
import pytest


def mock_hardware_modules():
    """Mock hardware-dependent modules before they're imported."""
    # Create mock modules
    mock_pyaudio = MagicMock()
    mock_pyaudio.paInt16 = 8  # pyaudio constant

    mock_torch = MagicMock()

    # Add to sys.modules
    sys.modules['pyaudio'] = mock_pyaudio
    sys.modules['torch'] = mock_torch


# Mock modules at import time for tests that need VoiceInputWorker
mock_hardware_modules()


@pytest.fixture
def mock_pyaudio():
    """Fixture providing mocked pyaudio module."""
    return sys.modules['pyaudio']


@pytest.fixture
def mock_torch():
    """Fixture providing mocked torch module."""
    return sys.modules['torch']
