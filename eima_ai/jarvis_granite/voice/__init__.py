"""
Voice module for Jarvis-Granite Live Telemetry.

Provides Text-to-Speech (TTS) and Speech-to-Text (STT) clients
using IBM Watson services, WebRTC transport via LiveKit, and
the complete voice pipeline.

Phase 5, Sections 11-13:
- Watson TTS/STT Clients (Section 11)
- LiveKit Integration (Section 12)
- Voice Pipeline (Section 13)
"""

from jarvis_granite.voice.watson_tts import WatsonTTSClient, TTSError
from jarvis_granite.voice.watson_stt import WatsonSTTClient, STTError
from jarvis_granite.voice.livekit_client import LiveKitClient, LiveKitError, VoiceAgent
from jarvis_granite.voice.voice_pipeline import VoicePipeline, SilenceDetector, AudioBuffer

__all__ = [
    # Watson TTS/STT
    "WatsonTTSClient",
    "TTSError",
    "WatsonSTTClient",
    "STTError",
    # LiveKit
    "LiveKitClient",
    "LiveKitError",
    "VoiceAgent",
    # Voice Pipeline
    "VoicePipeline",
    "SilenceDetector",
    "AudioBuffer",
]
