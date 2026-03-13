"""
Shared types and constants for AI pipeline integration.
"""
from __future__ import annotations

from dataclasses import dataclass


JARVIS_POST_ROOT_CANDIDATES = (
    "f1-post-analysis-main",
    "f1_post_analysis_main",
    "post-analysis-main",
)

EXTERNAL_PIPELINE_MODULES = (
    "explorer.ai.pipeline",
    "explorer.ai_pipeline",
    "explorer.pipeline",
)


@dataclass
class AIAnalysisResult:
    """Normalized model outputs for the analysis tab."""

    coach: str
    analyst: str
    source: str
