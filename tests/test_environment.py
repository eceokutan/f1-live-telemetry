"""
Manual environment/system checks.

These are intentionally opt-in and excluded from the default pytest run.
Run with:
    RUN_ENVIRONMENT_CHECKS=1 pytest -m "system and manual" tests/test_environment.py
"""

from __future__ import annotations

import importlib
import os
import platform
import sys

import pytest

pytestmark = [pytest.mark.system, pytest.mark.manual]

if os.getenv("RUN_ENVIRONMENT_CHECKS", "0") != "1":
    pytest.skip(
        "Environment checks are manual/opt-in. Set RUN_ENVIRONMENT_CHECKS=1 to run.",
        allow_module_level=True,
    )


def test_python_version_supported():
    assert sys.version_info >= (3, 10)


def test_core_dependencies_import():
    required = ["PyQt5", "matplotlib", "numpy", "pydantic", "httpx"]
    for module in required:
        assert importlib.import_module(module) is not None


def test_platform_is_supported_for_live_ac_integration():
    # Assetto Corsa shared memory integration is Windows-only.
    assert platform.system() == "Windows"
