"""
Configuration models for Jarvis-Granite Live Telemetry.

This module defines Pydantic models for all configuration options
including orchestrator, LiveKit, LangChain, retry, verbosity,
thresholds, and voice settings.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OrchestratorConfig(BaseModel):
    """Custom orchestrator settings."""
    priority_queue_max_size: int = Field(default=100, description="Maximum size of priority queue")
    interrupt_on_critical: bool = Field(default=True, description="Allow critical events to interrupt")
    sentence_completion_timeout_ms: int = Field(default=2000, description="Timeout for sentence completion")


class LiveKitConfig(BaseModel):
    """LiveKit WebRTC configuration."""
    url: str = Field(default="", description="LiveKit server URL")
    api_key: str = Field(default="", description="LiveKit API key")
    api_secret: str = Field(default="", description="LiveKit API secret")
    room_prefix: str = Field(default="jarvis_live_", description="Prefix for room names")
    audio_codec: str = Field(default="opus", description="Audio codec to use")
    sample_rate: int = Field(default=48000, description="Audio sample rate")


class LangChainCallbackConfig(BaseModel):
    """LangChain callback configuration."""
    type: str = Field(description="Callback type (langsmith, console)")
    enabled: bool = Field(default=False, description="Whether callback is enabled")


class LangChainConfig(BaseModel):
    """LangChain configuration for LLM management."""
    enable_langsmith: bool = Field(default=False, description="Enable LangSmith observability")
    langsmith_project: str = Field(default="jarvis-granite-live", description="LangSmith project name")
    callbacks: List[LangChainCallbackConfig] = Field(default_factory=list)


class RetryConfig(BaseModel):
    """Retry configuration for external services."""
    max_attempts: int = Field(default=3, description="Maximum retry attempts")
    multiplier: float = Field(default=1.0, description="Exponential backoff multiplier")
    min_wait: float = Field(default=0.5, description="Minimum wait time between retries")
    max_wait: float = Field(default=4.0, description="Maximum wait time between retries")


class RetrySettings(BaseModel):
    """Retry settings for all external services."""
    watson_tts: RetryConfig = Field(default_factory=RetryConfig)
    watson_stt: RetryConfig = Field(default_factory=RetryConfig)
    granite_llm: RetryConfig = Field(default_factory=lambda: RetryConfig(max_attempts=2, min_wait=1, max_wait=5))


class VerbosityConfig(BaseModel):
    """Verbosity settings for AI responses."""
    level: str = Field(default="moderate", description="Verbosity level: minimal, moderate, verbose")
    announce_lap_times: bool = Field(default=True, description="Announce lap times on completion")
    announce_gap_changes: bool = Field(default=True, description="Announce significant gap changes")
    announce_tire_status: bool = Field(default=True, description="Announce tire status updates")
    announce_fuel_status: bool = Field(default=True, description="Announce fuel status updates")
    proactive_coaching: bool = Field(default=False, description="Enable proactive coaching suggestions")


class ThresholdsConfig(BaseModel):
    """Threshold configuration for event triggers."""
    tire_temp_warning: float = Field(default=100.0, description="Tire temperature warning threshold (C)")
    tire_temp_critical: float = Field(default=110.0, description="Tire temperature critical threshold (C)")
    tire_wear_warning: float = Field(default=70.0, description="Tire wear warning threshold (%)")
    tire_wear_critical: float = Field(default=85.0, description="Tire wear critical threshold (%)")
    fuel_warning_laps: int = Field(default=5, description="Laps remaining for fuel warning")
    fuel_critical_laps: int = Field(default=2, description="Laps remaining for fuel critical alert")
    gap_change_threshold: float = Field(default=1.0, description="Gap change threshold (seconds)")


class VoiceConfig(BaseModel):
    """Voice settings for TTS/STT."""
    tts_voice: str = Field(default="en-GB_JamesV3Voice", description="Watson TTS voice")
    stt_model: str = Field(default="en-GB_BroadbandModel", description="Watson STT model")
    enable_voice: bool = Field(default=True, description="Enable voice output")


class LiveConfig(BaseModel):
    """
    Complete configuration for Jarvis-Granite Live Telemetry.

    This aggregates all sub-configurations into a single model
    that can be loaded from live.yaml.
    """
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    livekit: LiveKitConfig = Field(default_factory=LiveKitConfig)
    langchain: LangChainConfig = Field(default_factory=LangChainConfig)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    verbosity: VerbosityConfig = Field(default_factory=VerbosityConfig)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)

    # Buffer settings
    telemetry_buffer_seconds: int = Field(default=60, description="Telemetry history window in seconds")
    conversation_history_length: int = Field(default=3, description="Number of conversation exchanges to retain")
    min_proactive_interval_seconds: float = Field(default=10.0, description="Minimum time between AI alerts")


class EnvironmentSettings(BaseSettings):
    """
    Environment variables for Jarvis-Granite Live.

    These are loaded from .env file or environment variables.
    """
    # IBM watsonx.ai
    watsonx_api_key: str = Field(default="", alias="WATSONX_API_KEY")
    watsonx_project_id: str = Field(default="", alias="WATSONX_PROJECT_ID")
    watsonx_url: str = Field(default="https://us-south.ml.cloud.ibm.com", alias="WATSONX_URL")

    # IBM Watson Speech Services
    watson_tts_api_key: str = Field(default="", alias="WATSON_TTS_API_KEY")
    watson_tts_url: str = Field(
        default="https://api.us-south.text-to-speech.watson.cloud.ibm.com",
        alias="WATSON_TTS_URL"
    )
    watson_stt_api_key: str = Field(default="", alias="WATSON_STT_API_KEY")
    watson_stt_url: str = Field(
        default="https://api.us-south.speech-to-text.watson.cloud.ibm.com",
        alias="WATSON_STT_URL"
    )

    # LiveKit
    livekit_api_key: str = Field(default="", alias="LIVEKIT_API_KEY")
    livekit_api_secret: str = Field(default="", alias="LIVEKIT_API_SECRET")
    livekit_url: str = Field(default="", alias="LIVEKIT_URL")

    # LangSmith (optional)
    langsmith_api_key: Optional[str] = Field(default=None, alias="LANGSMITH_API_KEY")

    # HuggingFace (for development)
    huggingface_token: Optional[str] = Field(default=None, alias="HUGGINGFACE_TOKEN")

    # Application
    llm_provider: str = Field(default="watsonx", alias="LLM_PROVIDER")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_directory: str = Field(default="logs", alias="LOG_DIRECTORY")
    debug: bool = Field(default=False, alias="DEBUG")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
