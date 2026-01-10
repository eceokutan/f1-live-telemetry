# Jarvis-Granite Live Telemetry - Environment Variables
# Copy this file to .env and fill in your credentials

# =============================================================================
# IBM watsonx.ai (Granite LLM)
# =============================================================================
WATSONX_API_KEY=your_api_key
WATSONX_PROJECT_ID=your_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com

# =============================================================================
# IBM Watson Speech Services
# =============================================================================
# Text-to-Speech
WATSON_TTS_API_KEY=your_tts_api_key
WATSON_TTS_URL=https://api.us-south.text-to-speech.watson.cloud.ibm.com

# Speech-to-Text
WATSON_STT_API_KEY=your_stt_api_key
WATSON_STT_URL=https://api.us-south.speech-to-text.watson.cloud.ibm.com

# =============================================================================
# LiveKit (WebRTC Transport)
# =============================================================================
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
LIVEKIT_URL=wss://your-livekit-server.com

# =============================================================================
# LangSmith (Optional - for LLM observability)
# =============================================================================
LANGSMITH_API_KEY=

# =============================================================================
# HuggingFace (Optional - for development/testing)
# =============================================================================
HUGGINGFACE_TOKEN=

# =============================================================================
# Application Settings
# =============================================================================
# LLM provider: "watsonx" or "huggingface"
LLM_PROVIDER=watsonx

# Logging configuration
LOG_LEVEL=INFO
LOG_DIRECTORY=logs

# Debug mode (enables additional logging)
DEBUG=false
