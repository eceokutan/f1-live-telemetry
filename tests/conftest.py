"""
Pytest configuration and fixtures for F1 Telemetry tests.
"""

from __future__ import annotations

import os
import shutil
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _set_test_environment_defaults() -> None:
    """Set deterministic environment defaults for tests."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("POSTRACE_GGUF_AUTO_DOWNLOAD", "0")
    os.environ.setdefault("LOCAL_LLM_AUTO_DOWNLOAD", "0")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _mock_hardware_modules() -> None:
    """Mock hardware-dependent modules before they are imported."""
    mock_pyaudio = MagicMock()
    mock_pyaudio.paInt16 = 8  # pyaudio constant
    sys.modules["pyaudio"] = mock_pyaudio


_set_test_environment_defaults()
_mock_hardware_modules()


@pytest.fixture
def mock_pyaudio():
    """Fixture providing mocked pyaudio module."""
    return sys.modules["pyaudio"]


@pytest.fixture
def tmp_path() -> Path:
    """
    Local temporary directory fixture.

    This intentionally avoids pytest's built-in tmp_path fixture because
    the execution environment can deny access to OS-level temp roots.
    """
    root = Path("tmp_test_artifacts") / "pytest-local"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"case-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
