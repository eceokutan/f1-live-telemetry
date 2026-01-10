# Jarvis-Granite Live Telemetry

**Developer Documentation**

Version 2.1 | December 2025

---

## 1. Overview

Jarvis-Granite-Live provides real-time AI race engineering during active racing sessions. It connects via LiveKit WebRTC, processes telemetry continuously, and delivers voice-based strategic advice with sub-2-second latency.

| Aspect | Details |
|--------|---------|
| **Connection Type** | LiveKit WebRTC (persistent bi-directional) |
| **Optimization** | Speed (<2s end-to-end latency) |
| **Output Format** | Voice (Watson TTS via LiveKit) + Text |
| **Telemetry Buffer** | Rolling 60-second window (~600 samples at 10Hz) |
| **Conversation History** | Last 3 exchanges retained |
| **AI Trigger Mode** | Event-driven (proactive) + Query-driven (reactive) |
| **Orchestration** | Custom lightweight (~200-300 LOC) |
| **LLM Management** | LangChain (prompt templates, retry logic, observability) |

### 1.1 Design Principles

1. **Stateless for telemetry** — Platforms handle their own data storage; jarvis-granite processes data on-demand
2. **Event-driven** — AI analysis triggers automatically based on telemetry events
3. **Configurable** — Designed for iterative LLM fine-tuning and experimentation
4. **Observable** — All AI interactions logged for model improvement
5. **Layered architecture** — Specialized tools for each layer (LiveKit for transport, LangChain for LLM, custom for business logic)
6. **Performance-aware orchestration** — Custom code where performance demands it (<2s latency requirement)

### 1.2 Live vs Analysis Mode Comparison

| Aspect | Live Mode | Analysis Mode |
|--------|-----------|---------------|
| **Orchestration** | Custom lightweight (~200-300 LOC) | LangGraph |
| **Rationale** | <2s latency demands minimal overhead | <15s latency allows framework overhead |
| **Voice Transport** | LiveKit WebRTC | N/A |
| **LLM Management** | LangChain | LangChain |
| **Retry Logic** | Tenacity | Tenacity |
| **State Management** | In-memory session context | LangGraph checkpointer |
| **Agent Count** | 3 (Telemetry, Race Engineer, Voice) | 4+ (Data Fetch, Compare, Insight, Recommendation) |
| **Complexity** | Low (event → response) | High (multi-step workflows) |
| **Connection Type** | WebSocket + LiveKit WebRTC | REST API |
| **Primary Output** | Voice + Text | Text |
| **Optimization Goal** | Speed | Depth |

---

## 2. Architecture

### 2.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    JARVIS-GRANITE-LIVE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              LiveKit WebRTC Transport                   │   │
│   │  • Connection negotiation, jitter buffering             │   │
│   │  • Packet loss recovery, real-time audio streaming      │   │
│   │  Endpoint: wss://livekit.host/live                      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              CUSTOM ORCHESTRATOR (~200-300 LOC)         │   │
│   │                                                         │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│   │  │   Event     │  │  Priority   │  │  Interrupt  │     │   │
│   │  │   Router    │  │   Queue     │  │  Handler    │     │   │
│   │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│   │                                                         │   │
│   │  Racing-specific logic, minimal framework overhead      │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│          ┌───────────────────┼───────────────────┐              │
│          ▼                   ▼                   ▼              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │  Telemetry  │───▶│    Race     │───▶│    Voice    │        │
│   │    Agent    │    │  Engineer   │    │    Agent    │        │
│   │             │    │ (LangChain) │    │  (LiveKit)  │        │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│          │                   │                 │                │
│          ▼                   ▼                 ▼                │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │  Rolling    │    │   Granite   │    │   Watson    │        │
│   │  Buffer     │    │  LLM via    │    │  TTS / STT  │        │
│   │  (60s)      │    │  watsonx.ai │    │  (Tenacity) │        │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  SESSION CONTEXT                        │   │
│   │  • Current state (lap, fuel, tires, position, gaps)     │   │
│   │  • Telemetry buffer (last 60 seconds)                   │   │
│   │  • Conversation history (last 3 exchanges)              │   │
│   │  • Active alerts                                        │   │
│   │  • Configuration (verbosity, thresholds)                │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Runtime | Python 3.11+ | Primary language |
| Web Framework | FastAPI | WebSocket + REST support |
| **WebRTC Transport** | **LiveKit** | **Connection negotiation, jitter buffering, real-time audio** |
| LLM | IBM Granite 3.x (watsonx.ai) | Race engineer intelligence |
| **LLM Management** | **LangChain + langchain-ibm** | **Prompt templates, retry logic, observability** |
| **Retry Logic** | **Tenacity** | **Exponential backoff for API failures** |
| Speech-to-Text | IBM Watson STT | Driver voice input |
| Text-to-Speech | IBM Watson TTS | AI voice output |
| **Orchestration** | **Custom lightweight** | **Event routing, priority queue, racing logic** |
| Deployment | IBM Cloud Code Engine | Container hosting |

### 2.3 Layer Separation

| Layer | Technology | Responsibility |
|-------|------------|----------------|
| **Transport** | LiveKit | WebRTC connection, jitter buffering, packet loss recovery |
| **Speech AI** | Watson TTS/STT | Voice synthesis, speech recognition |
| **Orchestration** | Custom (~200-300 LOC) | Event routing, priority queue, interrupt handling |
| **Intelligence** | Granite via LangChain | Strategic insights, natural language generation |

### 2.4 External Services

| Service | Provider | Purpose | Endpoint |
|---------|----------|---------|----------|
| Granite LLM | IBM watsonx.ai | Text generation | `https://us-south.ml.cloud.ibm.com` |
| Speech-to-Text | IBM Watson | Voice transcription | `https://api.us-south.speech-to-text.watson.cloud.ibm.com` |
| Text-to-Speech | IBM Watson | Voice synthesis | `https://api.us-south.text-to-speech.watson.cloud.ibm.com` |
| WebRTC Transport | LiveKit | Real-time audio/video | Self-hosted or LiveKit Cloud |

#### Why LiveKit over IBM Watson Voice Gateway?

| Aspect | LiveKit | Watson Voice Gateway |
|--------|---------|---------------------|
| **Primary Use Case** | WebRTC for web/mobile apps | SIP/PSTN telephony |
| **Latency** | <100ms transport | ~500ms overhead |
| **Protocol** | WebRTC (native browser) | SIP (telephony) |
| **Fit for Racing** | ✅ Real-time, low latency | ❌ Telephony-focused |

---

## 3. Orchestration Strategy

### 3.1 Why Custom Orchestration for Live Mode?

The <2s end-to-end latency requirement demands minimal framework overhead. A custom lightweight orchestrator (~200-300 LOC) provides:

- **Direct event routing** — No framework abstraction layers
- **Racing-specific priority queue** — Critical events interrupt lower-priority responses
- **Interrupt handling** — Graceful sentence completion before new responses
- **Minimal dependencies** — Reduced failure points

**What we use frameworks for:**
- **LangChain** — LLM calls only (prompt templates, retry logic, observability via LangSmith)
- **Tenacity** — Watson TTS/STT retries with exponential backoff

**What we explicitly avoid:**
- LangGraph (overhead not justified for single-turn interactions)
- CrewAI (multi-agent complexity unnecessary)
- AutoGen (conversational agents not applicable)

### 3.2 JarvisLiveOrchestrator Implementation

```python
import asyncio
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Callable, Awaitable
from collections import deque
import heapq

from langchain_ibm import WatsonxLLM
from langchain.prompts import PromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential


class Priority(IntEnum):
    CRITICAL = 0  # Immediate interrupt
    HIGH = 1      # Interrupt medium/low
    MEDIUM = 2    # Queue normally
    LOW = 3       # Skip if busy


@dataclass
class Event:
    type: str
    priority: Priority
    data: dict
    timestamp: float


class JarvisLiveOrchestrator:
    """
    Custom lightweight orchestrator for real-time race engineering.
    ~200-300 LOC focused on event routing, priority queue, and interrupt handling.
    """
    
    def __init__(self, config: 'LiveConfig'):
        self.config = config
        self.event_queue: list = []  # heapq priority queue
        self.is_speaking = False
        self.pending_interrupt: Optional[str] = None
        self.session_context: Optional['LiveSessionContext'] = None
        
        # LangChain for LLM calls only
        self.llm = WatsonxLLM(
            model_id="ibm/granite-3-8b-instruct",
            url=config.watsonx_url,
            project_id=config.watsonx_project_id,
        )
        
        # Prompt templates managed by LangChain
        self.proactive_prompt = PromptTemplate(
            input_variables=["event_type", "event_data", "session_context"],
            template=PROACTIVE_TRIGGER_PROMPT
        )
        self.reactive_prompt = PromptTemplate(
            input_variables=["query", "session_context"],
            template=DRIVER_QUERY_PROMPT
        )
        
        # Agent references
        self.telemetry_agent: Optional['TelemetryAgent'] = None
        self.voice_agent: Optional['VoiceAgent'] = None
    
    async def handle_telemetry(self, telemetry: 'TelemetryMessage') -> None:
        """Process incoming telemetry and detect events."""
        # Update session context
        self.session_context.update(telemetry)
        
        # Detect events (rule-based, no LLM)
        events = self.telemetry_agent.detect_events(telemetry, self.session_context)
        
        for event in events:
            await self.queue_event(event)
        
        # Process queue
        await self.process_event_queue()
    
    async def queue_event(self, event: Event) -> None:
        """Add event to priority queue."""
        # heapq uses (priority, timestamp, event) for ordering
        heapq.heappush(self.event_queue, (event.priority, event.timestamp, event))
    
    async def process_event_queue(self) -> None:
        """Process events respecting priority and interrupt logic."""
        while self.event_queue:
            priority, _, event = self.event_queue[0]
            
            # Check if we should interrupt current speech
            if self.is_speaking:
                if priority <= Priority.HIGH and self.current_priority > priority:
                    # High priority event interrupts lower priority speech
                    self.pending_interrupt = event
                    return  # Wait for sentence completion
                else:
                    return  # Don't interrupt, wait for completion
            
            # Pop and process event
            heapq.heappop(self.event_queue)
            await self.handle_event(event)
    
    async def handle_event(self, event: Event) -> None:
        """Generate and deliver AI response for event."""
        self.current_priority = event.priority
        
        # Generate response via LangChain
        prompt = self.proactive_prompt.format(
            event_type=event.type,
            event_data=event.data,
            session_context=self.session_context.to_prompt_context()
        )
        
        response_text = await self.llm.ainvoke(prompt)
        
        # Deliver via voice agent
        await self.voice_agent.speak(response_text, priority=event.priority)
    
    async def handle_driver_query(self, query: str) -> None:
        """Handle driver voice query (reactive mode)."""
        prompt = self.reactive_prompt.format(
            query=query,
            session_context=self.session_context.to_prompt_context()
        )
        
        response_text = await self.llm.ainvoke(prompt)
        await self.voice_agent.speak(response_text, priority=Priority.HIGH)
    
    def on_sentence_complete(self) -> None:
        """Callback from voice agent when sentence finishes."""
        if self.pending_interrupt:
            event = self.pending_interrupt
            self.pending_interrupt = None
            asyncio.create_task(self.handle_event(event))
```

### 3.3 LangChain Integration

LangChain is used exclusively for LLM interactions, not orchestration:

```python
from langchain_ibm import WatsonxLLM
from langchain.prompts import PromptTemplate
from langchain.callbacks import LangSmithCallbackHandler

# Initialize with observability
llm = WatsonxLLM(
    model_id="ibm/granite-3-8b-instruct",
    url=os.getenv("WATSONX_URL"),
    project_id=os.getenv("WATSONX_PROJECT_ID"),
    callbacks=[LangSmithCallbackHandler()] if os.getenv("LANGSMITH_API_KEY") else []
)

# Prompt templates for consistency
SYSTEM_TEMPLATE = PromptTemplate(
    input_variables=["track_name", "current_lap", "position", "fuel_state", "tire_state"],
    template=LIVE_SYSTEM_PROMPT
)

# Invoke with retry logic handled by Tenacity at the HTTP layer
response = await llm.ainvoke(formatted_prompt)
```

### 3.4 Tenacity Retry Configuration

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx

# Retry configuration for Watson TTS/STT
watson_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=0.5, max=4),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError))
)

@watson_retry
async def synthesize_speech(text: str) -> bytes:
    """TTS with automatic retry on failure."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{WATSON_TTS_URL}/v1/synthesize",
            headers={"Content-Type": "application/json", "Accept": "audio/wav"},
            auth=("apikey", WATSON_TTS_API_KEY),
            json={"text": text},
            params={"voice": "en-GB_JamesV3Voice"},
            timeout=10.0
        )
        response.raise_for_status()
        return response.content
```

---

## 4. Data Flow

### 4.1 Telemetry Processing Flow

Incoming telemetry follows this processing pipeline:

```
Platform ──LiveKit──▶ Telemetry Agent ──▶ Update Context
                                │
                                ▼
                        Event Detected?
                                │
                    ┌───────────┴───────────┐
                    │ Yes                   │ No
                    ▼                       ▼
           Custom Orchestrator      Continue monitoring
           (Priority Queue)
                    │
                    ▼
            Race Engineer Agent
               (LangChain)
                    │
                    ▼
              Voice Agent
               (LiveKit)
                    │
                    ▼
         ◀──LiveKit WebRTC──  Platform (Audio)
```

1. Platform sends telemetry via WebSocket
2. Telemetry Agent parses and validates data (<50ms, rule-based)
3. Session Context is updated with current state
4. Telemetry Agent checks for event triggers
5. If event detected: Custom Orchestrator queues by priority
6. Race Engineer Agent generates response via LangChain → Granite
7. Voice Agent converts to speech (Watson TTS with Tenacity retry)
8. Audio streamed to platform via LiveKit WebRTC

### 4.2 Voice Query Flow

Driver voice queries are processed as follows:

```
Platform ──LiveKit──▶ Voice Agent (Watson STT)
                                │
                                ▼
                        Transcribed Text
                                │
                                ▼
                      Custom Orchestrator
                      (Interrupt Handler)
                                │
                                ▼
                      Race Engineer Agent
                         (LangChain)
                                │
                                ▼
                        Voice Agent
                      (Watson TTS → LiveKit)
                                │
                                ▼
         ◀──LiveKit WebRTC──  Platform (Audio)
```

1. Platform sends audio via LiveKit WebRTC
2. Voice Agent transcribes audio (Watson STT with Tenacity retry)
3. Custom Orchestrator handles interrupt logic
4. Race Engineer Agent generates contextual response via LangChain
5. Voice Agent converts response to speech
6. Audio streamed to platform via LiveKit

---

## 5. LiveKit Voice Infrastructure

### 5.1 Why LiveKit?

LiveKit provides WebRTC transport optimized for real-time applications:

- **Connection negotiation** — Automatic ICE candidate handling
- **Jitter buffering** — Smooth audio despite network variance
- **Packet loss recovery** — Maintains quality on unreliable networks
- **Native browser support** — No plugins required
- **Low latency** — <100ms transport overhead

### 5.2 Voice Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     VOICE PIPELINE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│   │  Driver  │───▶│  LiveKit │───▶│  Watson  │───▶│   Text   │ │
│   │  Audio   │    │  WebRTC  │    │   STT    │    │  Output  │ │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│                                                                 │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│   │  Granite │───▶│  Watson  │───▶│  LiveKit │───▶│  Driver  │ │
│   │ Response │    │   TTS    │    │  WebRTC  │    │ Speakers │ │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 VoiceAgent Implementation

```python
import asyncio
from livekit import rtc
from tenacity import retry, stop_after_attempt, wait_exponential

class VoiceAgent:
    """
    Voice agent using LiveKit for WebRTC transport
    and Watson for TTS/STT processing.
    """
    
    def __init__(self, config: 'LiveConfig'):
        self.config = config
        self.room: Optional[rtc.Room] = None
        self.audio_track: Optional[rtc.LocalAudioTrack] = None
        self.currently_speaking = False
        self.pending_query: Optional[str] = None
        self.current_sentence_complete = asyncio.Event()
        
        # Watson clients with Tenacity retry
        self.tts_client = WatsonTTSClient(
            api_key=config.watson_tts_api_key,
            service_url=config.watson_tts_url
        )
        self.stt_client = WatsonSTTClient(
            api_key=config.watson_stt_api_key,
            service_url=config.watson_stt_url
        )
        
        # Orchestrator callback
        self.on_query_received: Optional[Callable] = None
    
    async def connect(self, room_name: str, participant_name: str) -> None:
        """Connect to LiveKit room."""
        self.room = rtc.Room()
        
        # Generate access token
        token = self._generate_token(room_name, participant_name)
        
        # Connect to LiveKit server
        await self.room.connect(
            url=self.config.livekit_url,
            token=token
        )
        
        # Set up audio track for TTS output
        self.audio_track = rtc.LocalAudioTrack.create_audio_track(
            "ai-engineer-audio",
            rtc.AudioSource()
        )
        await self.room.local_participant.publish_track(self.audio_track)
        
        # Subscribe to incoming audio (driver voice)
        self.room.on("track_subscribed", self._on_track_subscribed)
    
    async def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant
    ) -> None:
        """Handle incoming audio from driver."""
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            audio_stream = rtc.AudioStream(track)
            asyncio.create_task(self._process_audio_stream(audio_stream))
    
    async def _process_audio_stream(self, stream: rtc.AudioStream) -> None:
        """Process incoming audio and transcribe."""
        audio_buffer = []
        silence_threshold = 0.5  # seconds
        
        async for frame in stream:
            audio_buffer.append(frame.data)
            
            # Detect end of speech (simplified)
            if self._detect_silence(frame, silence_threshold):
                if audio_buffer:
                    audio_bytes = self._combine_frames(audio_buffer)
                    text = await self._transcribe_with_retry(audio_bytes)
                    
                    if text and self.on_query_received:
                        await self._handle_driver_input(text)
                    
                    audio_buffer = []
    
    async def _handle_driver_input(self, text: str) -> None:
        """Handle transcribed driver speech."""
        if self.currently_speaking:
            # Queue the query, wait for current sentence to complete
            self.pending_query = text
            await self.current_sentence_complete.wait()
        
        # Notify orchestrator
        if self.on_query_received:
            await self.on_query_received(text)
    
    async def speak(self, text: str, priority: str = "medium") -> None:
        """Convert text to speech and stream via LiveKit."""
        sentences = self._split_into_sentences(text)
        
        for sentence in sentences:
            self.currently_speaking = True
            
            # Synthesize with retry
            audio_bytes = await self._synthesize_with_retry(sentence)
            
            # Stream audio via LiveKit
            await self._stream_audio(audio_bytes)
            
            # Signal sentence complete for interrupt handling
            self.current_sentence_complete.set()
            self.current_sentence_complete.clear()
            
            # Check for pending query after each sentence
            if self.pending_query:
                break
        
        self.currently_speaking = False
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=0.5, max=4)
    )
    async def _transcribe_with_retry(self, audio: bytes) -> str:
        """Transcribe audio with automatic retry."""
        return await self.stt_client.transcribe(audio)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=0.5, max=4)
    )
    async def _synthesize_with_retry(self, text: str) -> bytes:
        """Synthesize speech with automatic retry."""
        return await self.tts_client.synthesize(text)
    
    async def _stream_audio(self, audio_bytes: bytes) -> None:
        """Stream audio frames via LiveKit."""
        frames = self._bytes_to_frames(audio_bytes)
        for frame in frames:
            await self.audio_track.source.capture_frame(frame)
            await asyncio.sleep(frame.duration)
    
    def _generate_token(self, room_name: str, participant_name: str) -> str:
        """Generate LiveKit access token."""
        from livekit.api import AccessToken, VideoGrants
        
        token = AccessToken(
            api_key=self.config.livekit_api_key,
            api_secret=self.config.livekit_api_secret
        )
        token.with_identity(participant_name)
        token.with_grants(VideoGrants(
            room_join=True,
            room=room_name
        ))
        return token.to_jwt()
    
    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences for interrupt handling."""
        import re
        return re.split(r'(?<=[.!?])\s+', text)
    
    async def disconnect(self) -> None:
        """Disconnect from LiveKit room."""
        if self.room:
            await self.room.disconnect()
```

### 5.4 LiveKit Deployment Options

| Option | Description | Cost |
|--------|-------------|------|
| **Self-hosted** | Deploy on IBM Cloud Kubernetes | Infrastructure costs only |
| **LiveKit Cloud** | Managed service | ~$0.004/minute |

**Self-hosted on IBM Cloud:**

```bash
# Deploy LiveKit on IBM Cloud Kubernetes
kubectl apply -f https://raw.githubusercontent.com/livekit/livekit-helm/main/livekit-server.yaml

# Configure with IBM Cloud Load Balancer
kubectl expose deployment livekit-server --type=LoadBalancer --port=7880
```

---

## 6. Event System

### 6.1 Event Types and Priorities

Events are categorized by priority level which determines interrupt behavior:

| Priority | Level | Events | Behavior |
|----------|-------|--------|----------|
| **Critical** | 0 | Brake failure, collision warning | Immediate interrupt |
| **High** | 1 | Pit now, fuel critical, tire failure | Interrupt medium/low |
| **Medium** | 2 | Pit window, gap changes, lap summary | Queue normally |
| **Low** | 3 | Sector times, minor updates | Skip if busy |

### 6.2 Event Triggers

The Telemetry Agent monitors the following conditions:

| Event | Condition | Priority |
|-------|-----------|----------|
| `pit_window_open` | Fuel < threshold OR tire wear > threshold | High |
| `tire_critical` | Tire temperature > 110°C | Critical |
| `fuel_critical` | Fuel < 2 laps remaining | Critical |
| `gap_change` | Gap to car ahead/behind changes > 1s | Medium |
| `lap_complete` | Track position crosses start/finish | Medium |
| `sector_complete` | Sector boundary crossed | Low |

### 6.3 Event Detection Logic

```python
class TelemetryAgent:
    """
    Rule-based telemetry processing (no LLM).
    Latency budget: <50ms
    """
    
    def detect_events(self, telemetry: TelemetryMessage, context: SessionContext) -> List[Event]:
        events = []
        
        # Fuel events
        fuel_laps_remaining = self.calculate_fuel_laps(telemetry, context)
        if fuel_laps_remaining <= context.config.fuel_critical_laps:
            events.append(Event(
                type="fuel_critical",
                priority=Priority.CRITICAL,
                data={"laps": fuel_laps_remaining}
            ))
        elif fuel_laps_remaining <= context.config.fuel_warning_laps:
            events.append(Event(
                type="fuel_warning",
                priority=Priority.HIGH,
                data={"laps": fuel_laps_remaining}
            ))
        
        # Tire events
        max_tire_temp = max(telemetry.data.tire_temps.values())
        if max_tire_temp >= context.config.tire_temp_critical:
            events.append(Event(
                type="tire_critical",
                priority=Priority.CRITICAL,
                data={"temp": max_tire_temp}
            ))
        elif max_tire_temp >= context.config.tire_temp_warning:
            events.append(Event(
                type="tire_warning",
                priority=Priority.MEDIUM,
                data={"temp": max_tire_temp}
            ))
        
        # Gap events
        if context.gap_ahead is not None and telemetry.data.gap_ahead is not None:
            gap_change = abs(telemetry.data.gap_ahead - context.gap_ahead)
            if gap_change >= context.config.gap_change_threshold:
                events.append(Event(
                    type="gap_change",
                    priority=Priority.MEDIUM,
                    data={"change": gap_change}
                ))
        
        # Lap completion
        if telemetry.data.lap_number > context.current_lap:
            events.append(Event(
                type="lap_complete",
                priority=Priority.MEDIUM,
                data={"lap": telemetry.data.lap_number}
            ))
        
        return events
```

---

## 7. Interrupt Handling

When the driver speaks while the AI is outputting audio, the system handles the interrupt gracefully.

### 7.1 Interrupt Behavior

```
AI Speaking: "Your tires are looking good, about ten laps—"
Driver Speaks: "What about fuel?"
                    │
                    ▼
            AI finishes sentence: "—remaining on them."
                    │
                    ▼
            AI responds to query: "Fuel is at 45 liters, roughly 8 laps at current pace."
```

**Process:**

1. LiveKit detects incoming driver audio
2. Voice Agent queues the transcribed query
3. Current sentence is completed (natural finish point)
4. Remaining queued output is cleared
5. Driver query is processed by orchestrator
6. New response is generated and delivered

### 7.2 Priority-Based Interrupts

```python
async def process_event_queue(self) -> None:
    """Process events respecting priority and interrupt logic."""
    while self.event_queue:
        priority, _, event = self.event_queue[0]
        
        if self.is_speaking:
            # Critical/High events can interrupt Medium/Low speech
            if priority <= Priority.HIGH and self.current_priority > priority:
                self.pending_interrupt = event
                return  # Wait for sentence completion
            else:
                return  # Don't interrupt, wait
        
        heapq.heappop(self.event_queue)
        await self.handle_event(event)
```

---

## 8. Session Context

The `LiveSessionContext` maintains all state during an active racing session.

### 8.1 Context Fields

| Category | Fields | Description |
|----------|--------|-------------|
| Session | session_id, source, track_name, started_at | Session identification |
| Vehicle State | current_lap, sector, speed, rpm, gear, throttle, brake | Real-time vehicle data |
| Resources | fuel_remaining, fuel_consumption_per_lap, tire_wear, tire_temps | Consumables tracking |
| Race Position | position, gap_ahead, gap_behind | Track position data |
| Lap History | lap_times, best_lap, last_lap | Timing data |
| Telemetry Buffer | deque(maxlen=600) | Rolling 60s window at 10Hz |
| Conversation | deque(maxlen=3) | Last 3 driver exchanges |
| Configuration | verbosity, thresholds, voice settings | Runtime settings |

### 8.2 Context Definition

```python
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from typing import Optional, List, Dict, Any

@dataclass
class LiveSessionContext:
    """In-memory context maintained during a live racing session."""
    
    # Session identification
    session_id: str
    source: str  # "torcs", "assetto_corsa", "can_bus"
    track_name: str
    started_at: datetime
    
    # Current vehicle state
    current_lap: int = 0
    current_sector: int = 1
    speed_kmh: float = 0.0
    rpm: int = 0
    gear: int = 0
    throttle: float = 0.0
    brake: float = 0.0
    
    # Resource state
    fuel_remaining: float = 100.0
    fuel_consumption_per_lap: float = 0.0
    tire_wear: Dict[str, float] = field(default_factory=lambda: {"fl": 0, "fr": 0, "rl": 0, "rr": 0})
    tire_temps: Dict[str, float] = field(default_factory=lambda: {"fl": 80, "fr": 80, "rl": 80, "rr": 80})
    
    # Race position
    position: int = 1
    gap_ahead: Optional[float] = None
    gap_behind: Optional[float] = None
    
    # Lap history
    lap_times: List[float] = field(default_factory=list)
    best_lap: Optional[float] = None
    last_lap: Optional[float] = None
    
    # Rolling telemetry buffer (last 60 seconds at ~10Hz = 600 points)
    telemetry_buffer: deque = field(default_factory=lambda: deque(maxlen=600))
    
    # Conversation history (last 3 exchanges)
    conversation_history: deque = field(default_factory=lambda: deque(maxlen=3))
    
    # Active state
    active_alerts: List[Dict[str, Any]] = field(default_factory=list)
    last_proactive_message_time: Optional[datetime] = None
    
    # Configuration
    config: 'LiveConfig' = field(default_factory=lambda: LiveConfig())
    
    def to_prompt_context(self) -> str:
        """Format context for LLM prompt injection."""
        return f"""Track: {self.track_name}
Lap: {self.current_lap} | Position: P{self.position}
Gap Ahead: {self.gap_ahead}s | Gap Behind: {self.gap_behind}s
Fuel: {self.fuel_remaining:.1f}L ({self._fuel_laps_remaining():.1f} laps)
Tires: FL:{self.tire_temps['fl']:.0f}°C FR:{self.tire_temps['fr']:.0f}°C RL:{self.tire_temps['rl']:.0f}°C RR:{self.tire_temps['rr']:.0f}°C
Best Lap: {self.best_lap or 'N/A'} | Last Lap: {self.last_lap or 'N/A'}"""
```

---

## 9. WebSocket API Reference

### 9.1 Connection

**Telemetry Endpoint:** `ws://host:port/live`
**Voice Endpoint:** LiveKit room connection

**Connection Flow:**

```
Client                              Server
   │                                   │
   │──── WebSocket Connect ───────────▶│
   │                                   │
   │◀─── Connection Accepted ──────────│
   │                                   │
   │──── Session Init Message ────────▶│
   │                                   │
   │◀─── Session Confirmed ────────────│
   │     (includes LiveKit token)      │
   │                                   │
   │──── LiveKit Room Connect ────────▶│ (separate WebRTC connection)
   │                                   │
   │──── Telemetry Stream ────────────▶│
   │◀─── AI Responses (LiveKit audio)──│
   │                                   │
```

### 9.2 Client → Server Messages

#### Session Initialization

```json
{
  "type": "session_init",
  "session_id": "race_001",
  "source": "torcs",
  "track_name": "Monza",
  "config": {
    "verbosity": "moderate",
    "driver_name": "Driver 1"
  }
}
```

#### Telemetry

```json
{
  "type": "telemetry",
  "timestamp": "2026-01-15T14:32:05.123Z",
  "data": {
    "speed_kmh": 245.5,
    "rpm": 12500,
    "gear": 5,
    "throttle": 0.95,
    "brake": 0.0,
    "steering_angle": -0.12,
    "fuel_remaining": 45.2,
    "tire_temps": {"fl": 95.2, "fr": 96.1, "rl": 92.4, "rr": 93.8},
    "tire_wear": {"fl": 15.2, "fr": 16.1, "rl": 12.4, "rr": 13.8},
    "g_forces": {"lateral": 1.8, "longitudinal": 0.3},
    "track_position": 0.342,
    "lap_number": 12,
    "lap_time_current": 45.234,
    "sector": 2,
    "position": 3,
    "gap_ahead": 2.456,
    "gap_behind": 1.234
  }
}
```

#### Text Query (Alternative to Voice)

```json
{
  "type": "text_query",
  "timestamp": "2026-01-15T14:32:10.456Z",
  "query": "How are my tires looking?"
}
```

#### Configuration Update

```json
{
  "type": "config_update",
  "config": {
    "verbosity": "verbose"
  }
}
```

#### Session End

```json
{
  "type": "session_end"
}
```

### 9.3 Server → Client Messages

#### Session Confirmed

```json
{
  "type": "session_confirmed",
  "session_id": "race_001",
  "config": { /* active config */ },
  "livekit": {
    "url": "wss://livekit.example.com",
    "token": "eyJ...",
    "room_name": "race_001_voice"
  }
}
```

#### AI Response (Text)

```json
{
  "type": "ai_response",
  "response_id": "resp_abc123",
  "timestamp": "2026-01-15T14:32:11.789Z",
  "trigger": "pit_window_open",
  "text": "Pit window is open. Box this lap for fresh mediums.",
  "priority": "high",
  "metadata": {
    "latency_ms": 1850,
    "tokens_used": 45
  }
}
```

*Note: Audio is delivered via LiveKit WebRTC, not WebSocket.*

#### Error

```json
{
  "type": "error",
  "error_code": "LLM_TIMEOUT",
  "message": "AI response timed out, please try again",
  "timestamp": "2026-01-15T14:32:15.000Z"
}
```

#### Heartbeat

```json
{
  "type": "heartbeat",
  "timestamp": "2026-01-15T14:32:30.000Z",
  "session_active": true
}
```

---

## 10. Telemetry Data Schema

Each telemetry message must conform to the following structure:

| Field | Type | Range/Format | Description |
|-------|------|--------------|-------------|
| `speed_kmh` | float | ≥0 | Current speed in km/h |
| `rpm` | int | ≥0 | Engine RPM |
| `gear` | int | 0-8 | Current gear (0=neutral) |
| `throttle` | float | 0-1 | Throttle position |
| `brake` | float | 0-1 | Brake pressure |
| `steering_angle` | float | -1 to 1 | Steering input |
| `fuel_remaining` | float | ≥0 | Fuel in liters |
| `tire_temps` | object | fl, fr, rl, rr | Tire temperatures (°C) |
| `tire_wear` | object | fl, fr, rl, rr (0-100) | Tire wear percentages |
| `g_forces` | object | lateral, longitudinal | G-force values |
| `track_position` | float | 0-1 | Position on track |
| `lap_number` | int | ≥0 | Current lap |
| `lap_time_current` | float | ≥0 | Current lap time (seconds) |
| `sector` | int | 1-3 | Current sector |
| `position` | int | ≥1 | Race position (optional) |
| `gap_ahead` | float | seconds | Gap to car ahead (optional) |
| `gap_behind` | float | seconds | Gap to car behind (optional) |

---

## 11. Configuration

### 11.1 Live Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `verbosity.level` | moderate | minimal / moderate / verbose |
| `verbosity.announce_lap_times` | true | Report lap times on completion |
| `verbosity.announce_gap_changes` | true | Report significant gap changes |
| `verbosity.proactive_coaching` | false | Enable coaching suggestions |
| `thresholds.tire_temp_warning` | 100.0 | Warning temperature (°C) |
| `thresholds.tire_temp_critical` | 110.0 | Critical temperature (°C) |
| `thresholds.fuel_warning_laps` | 5 | Laps before fuel warning |
| `thresholds.fuel_critical_laps` | 2 | Laps before critical alert |
| `thresholds.gap_change_threshold` | 1.0 | Gap change trigger (seconds) |
| `voice.tts_voice` | en-GB_JamesV3Voice | Watson TTS voice |
| `voice.enable_voice` | true | Enable voice output |
| `livekit.url` | (required) | LiveKit server URL |
| `telemetry_buffer_seconds` | 60 | Telemetry history window |
| `min_proactive_interval_seconds` | 10.0 | Minimum time between AI alerts |

### 11.2 Configuration Files

**config/live.yaml:**

```yaml
# Custom Orchestrator Settings
orchestrator:
  priority_queue_max_size: 100
  interrupt_on_critical: true
  sentence_completion_timeout_ms: 2000

# LiveKit Configuration
livekit:
  url: ${LIVEKIT_URL}
  api_key: ${LIVEKIT_API_KEY}
  api_secret: ${LIVEKIT_API_SECRET}
  room_prefix: "jarvis_live_"
  audio_codec: "opus"
  sample_rate: 48000

# LangChain Callbacks
langchain:
  enable_langsmith: ${LANGSMITH_API_KEY:+true}
  langsmith_project: "jarvis-granite-live"
  callbacks:
    - type: "langsmith"
      enabled: ${LANGSMITH_API_KEY:+true}
    - type: "console"
      enabled: ${DEBUG:-false}

# Tenacity Retry Configuration
retry:
  watson_tts:
    max_attempts: 3
    multiplier: 1
    min_wait: 0.5
    max_wait: 4
  watson_stt:
    max_attempts: 3
    multiplier: 1
    min_wait: 0.5
    max_wait: 4
  granite_llm:
    max_attempts: 2
    multiplier: 1
    min_wait: 1
    max_wait: 5

# Verbosity Settings
verbosity:
  level: "moderate"
  announce_lap_times: true
  announce_gap_changes: true
  announce_tire_status: true
  announce_fuel_status: true
  proactive_coaching: false

# Thresholds
thresholds:
  tire_temp_warning: 100.0
  tire_temp_critical: 110.0
  tire_wear_warning: 70.0
  tire_wear_critical: 85.0
  fuel_warning_laps: 5
  fuel_critical_laps: 2
  gap_change_threshold: 1.0

# Voice Settings
voice:
  tts_voice: "en-GB_JamesV3Voice"
  stt_model: "en-GB_BroadbandModel"
  enable_voice: true

# Buffer Settings
telemetry_buffer_seconds: 60
conversation_history_length: 3
min_proactive_interval_seconds: 10.0
```

### 11.3 Environment Variables

```bash
# IBM watsonx.ai
WATSONX_API_KEY=your_api_key
WATSONX_PROJECT_ID=your_project_id
WATSONX_URL=https://us-south.ml.cloud.ibm.com

# IBM Watson Speech Services
WATSON_TTS_API_KEY=your_tts_api_key
WATSON_TTS_URL=https://api.us-south.text-to-speech.watson.cloud.ibm.com
WATSON_STT_API_KEY=your_stt_api_key
WATSON_STT_URL=https://api.us-south.speech-to-text.watson.cloud.ibm.com

# LiveKit
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
LIVEKIT_URL=wss://your-livekit-server.com

# LangSmith (optional, for observability)
LANGSMITH_API_KEY=your_langsmith_key

# HuggingFace (for development)
HUGGINGFACE_TOKEN=your_hf_token

# Application
LLM_PROVIDER=watsonx
LOG_LEVEL=INFO
LOG_DIRECTORY=logs
DEBUG=false
```

---

## 12. LLM Prompts

### 12.1 System Prompt

```python
LIVE_SYSTEM_PROMPT = """You are an expert F1 race engineer communicating with your driver over team radio during a live race.

CRITICAL CONSTRAINTS:
- Driver is actively racing and cannot read text
- Responses must be CONCISE (1-3 sentences max)
- Lead with the most important information
- Use precise numbers when helpful
- Match urgency to the situation

CURRENT SESSION:
{session_context}

VERBOSITY: {verbosity_level}
{verbosity_instructions}

CONVERSATION HISTORY:
{conversation_history}
"""
```

### 12.2 Proactive Trigger Prompt

```python
PROACTIVE_TRIGGER_PROMPT = """EVENT DETECTED: {event_type}
EVENT DATA: {event_data}

Based on the current session state and this event, provide a brief message for the driver.

Guidelines:
- Be direct and actionable
- If pit stop recommended, say "Box" clearly
- Include relevant numbers (laps, temperatures, gaps)
- Critical events require urgent tone
- Non-critical events can be informational

Your radio message to the driver:"""
```

### 12.3 Driver Query Prompt

```python
DRIVER_QUERY_PROMPT = """DRIVER ASKED: "{query}"

Answer the driver's question using the current session data. Be concise but complete.

If the question involves strategy, provide your recommendation with reasoning.
If you need to reference data, use specific numbers.

Your radio response:"""
```

---

## 13. Integration Guide

### 13.1 Python Client Example

```python
import asyncio
import websockets
import json
from livekit import rtc

class JarvisLiveClient:
    def __init__(
        self,
        telemetry_url: str = "ws://localhost:8001/live"
    ):
        self.telemetry_url = telemetry_url
        self.websocket = None
        self.livekit_room = None
    
    async def connect(self, session_config: dict):
        # Connect telemetry WebSocket
        self.websocket = await websockets.connect(self.telemetry_url)
        
        # Initialize session
        await self.websocket.send(json.dumps({
            "type": "session_init",
            **session_config
        }))
        
        response = json.loads(await self.websocket.recv())
        
        # Connect to LiveKit for voice
        if "livekit" in response:
            self.livekit_room = rtc.Room()
            await self.livekit_room.connect(
                url=response["livekit"]["url"],
                token=response["livekit"]["token"]
            )
            
            # Set up audio handling
            self.livekit_room.on("track_subscribed", self._on_ai_audio)
        
        return response
    
    async def _on_ai_audio(self, track, publication, participant):
        """Handle incoming AI audio from LiveKit."""
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            # Play audio through speakers
            audio_stream = rtc.AudioStream(track)
            async for frame in audio_stream:
                self.play_audio_frame(frame)
    
    async def send_telemetry(self, telemetry: dict):
        await self.websocket.send(json.dumps({
            "type": "telemetry",
            **telemetry
        }))
    
    async def send_voice(self, audio_track: rtc.LocalAudioTrack):
        """Publish driver audio to LiveKit room."""
        await self.livekit_room.local_participant.publish_track(audio_track)
    
    async def receive_responses(self):
        """Generator for receiving text responses."""
        while True:
            message = await self.websocket.recv()
            yield json.loads(message)
    
    async def close(self):
        await self.websocket.send(json.dumps({"type": "session_end"}))
        await self.websocket.close()
        if self.livekit_room:
            await self.livekit_room.disconnect()


# Usage
async def main():
    client = JarvisLiveClient()
    
    await client.connect({
        "session_id": "race_001",
        "source": "torcs",
        "track_name": "Monza",
        "config": {"verbosity": "moderate"}
    })
    
    # Handle text responses in background
    async def handle_responses():
        async for response in client.receive_responses():
            if response["type"] == "ai_response":
                print(f"AI: {response['text']}")
    
    asyncio.create_task(handle_responses())
    
    # Send telemetry in game loop
    while racing:
        telemetry = get_current_telemetry()
        await client.send_telemetry(telemetry)
        await asyncio.sleep(0.1)  # 10Hz
```

### 13.2 Integration Checklist

- [ ] Establish WebSocket connection to `/live` endpoint
- [ ] Send `session_init` with source, track, and config
- [ ] Connect to LiveKit room using token from `session_confirmed`
- [ ] Stream telemetry at 10Hz (100ms intervals)
- [ ] Publish driver audio to LiveKit room
- [ ] Subscribe to AI audio track from LiveKit
- [ ] Handle `ai_response` text messages
- [ ] Implement error handling for disconnects
- [ ] Handle `heartbeat` messages for connection monitoring
- [ ] Send `session_end` when race concludes

---

## 14. Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Telemetry processing | <50ms | Rule-based, no LLM |
| LLM response | <2000ms | Granite generation via LangChain |
| TTS conversion | <500ms | Watson TTS with Tenacity retry |
| LiveKit transport | <100ms | WebRTC jitter buffering |
| End-to-end (voice) | <3000ms | Total pipeline latency |
| Telemetry rate | 10Hz | 100ms between samples |

---

## 15. Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `LLM_TIMEOUT` | Granite API request timed out | Retry (handled by LangChain) |
| `LLM_ERROR` | Granite API returned error | Check credentials, quota |
| `TTS_ERROR` | Text-to-speech failed | Retry (handled by Tenacity) |
| `STT_ERROR` | Speech-to-text failed | Check audio format, retry |
| `LIVEKIT_ERROR` | LiveKit connection failed | Check LiveKit server, credentials |
| `VALIDATION_ERROR` | Invalid request data | Check request schema |
| `SESSION_NOT_FOUND` | Referenced session doesn't exist | Initialize session first |
| `CONFIG_ERROR` | Invalid configuration | Check config values |

---

## 16. Project Structure

```
jarvis-granite/
├── jarvis_granite/
│   ├── __init__.py
│   ├── live/                      # Live mode implementation
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI app for live mode
│   │   ├── orchestrator.py        # Custom lightweight orchestrator
│   │   ├── websocket_handler.py   # WebSocket management
│   │   └── context.py             # LiveSessionContext
│   │
│   ├── agents/                    # Shared agents
│   │   ├── __init__.py
│   │   ├── base.py                # Base agent class
│   │   ├── telemetry_agent.py     # Telemetry processing (rule-based)
│   │   ├── race_engineer_agent.py # LLM-powered insights (LangChain)
│   │   └── voice_agent.py         # TTS/STT handling (LiveKit + Watson)
│   │
│   ├── llm/                       # LLM client abstraction
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract base client
│   │   ├── granite_client.py      # LangChain + watsonx.ai client
│   │   └── huggingface_client.py  # HuggingFace client (dev)
│   │
│   ├── voice/                     # Voice services
│   │   ├── __init__.py
│   │   ├── livekit_client.py      # LiveKit WebRTC client
│   │   ├── watson_tts.py          # Text-to-speech (Tenacity)
│   │   └── watson_stt.py          # Speech-to-text (Tenacity)
│   │
│   ├── schemas/                   # Pydantic models
│   │   ├── __init__.py
│   │   └── telemetry.py           # Telemetry schemas
│   │
│   ├── prompts/                   # Prompt templates
│   │   ├── __init__.py
│   │   └── live_prompts.py        # Live mode prompts (LangChain)
│   │
│   └── utils/                     # Utilities
│       ├── __init__.py
│       ├── logging.py             # Logging service
│       └── helpers.py             # Common helpers
│
├── config/
│   ├── __init__.py
│   ├── config.py                  # Configuration models
│   ├── config_loader.py           # Configuration loading
│   └── live.yaml                  # Live mode config
│
├── tests/
│   ├── __init__.py
│   ├── test_telemetry_agent.py
│   ├── test_orchestrator.py
│   ├── test_voice_agent.py
│   └── fixtures/
│       └── telemetry_samples.json
│
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## 17. Dependencies

```
# requirements.txt

# Web Framework
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
websockets>=12.0

# Data Validation
pydantic>=2.5.0

# HTTP Client
httpx>=0.25.0

# IBM Services
ibm-watson>=8.0.0
ibm-cloud-sdk-core>=3.18.0

# LangChain (LLM management)
langchain>=0.1.0
langchain-ibm>=0.1.0

# LiveKit (WebRTC transport)
livekit>=0.9.0
livekit-api>=0.4.0

# Retry Logic
tenacity>=8.2.0

# Configuration
python-dotenv>=1.0.0
pyyaml>=6.0.0
```

---

## 18. Local Development

```bash
# Clone repository
git clone https://github.com/your-org/jarvis-granite.git
cd jarvis-granite

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Copy environment template
cp .env.example .env
# Edit .env with your credentials (including LiveKit)

# Run LiveKit locally (optional, for voice testing)
docker run -d \
  -p 7880:7880 \
  -p 7881:7881 \
  -e LIVEKIT_KEYS="devkey: secret" \
  livekit/livekit-server

# Run live service
uvicorn jarvis_granite.live.main:app --reload --port 8001
```

---

## 19. Future Considerations

1. **LiveKit Spatial Audio** — For VR platform integration, implement spatial audio positioning based on race position
2. **LangSmith Integration** — Production LLM monitoring and prompt optimization
3. **Fine-tuned Models** — Once Granite is fine-tuned on motorsport data, update `model_id` in configuration
4. **Multi-session Support** — Current design supports single session; scaling requires session management layer
5. **Caching Layer** — Add Redis for response caching if latency becomes an issue

---

## 20. Glossary

| Term | Definition |
|------|------------|
| **LiveKit** | Open-source WebRTC platform for real-time audio/video communication |
| **LangChain** | Framework for building LLM applications; used here for prompt management and observability |
| **Tenacity** | Python library for retry logic with exponential backoff |
| **Proactive** | AI-initiated communication based on telemetry events |
| **Reactive** | AI response to driver queries |
| **Event** | Telemetry condition that may trigger AI response |
| **Verbosity** | Configuration for how "chatty" the AI is |
| **Rolling Buffer** | In-memory storage of recent telemetry (60 seconds) |
| **Custom Orchestrator** | Lightweight (~200-300 LOC) event routing and priority management |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 2.0 | December 2025 | Team 17 | Initial Live Telemetry documentation |
| 2.1 | December 2025 | Team 17 | Added LiveKit, LangChain, custom orchestration |

---

*F1 Jarvis TORCS Project | Team 17*

Development Order for Jarvis-Granite Live Telemetry
Phase 1: Core Infrastructure
1. Configuration & Environment Setup

Set up project structure
Create configuration models (config.py, live.yaml)
Implement environment variable loading
Set up logging infrastructure

2. Data Schemas

Eima - may need to adjust based on exact telemetry data schema received from AC, TORCS, FS

Define Pydantic models for telemetry data (TelemetryData, TireTemps, TireWear, GForces)
Define WebSocket message schemas (session_init, telemetry, ai_response, error)
Define event schemas (Event, Priority enum)

3. Session Context

Implement LiveSessionContext dataclass
Build rolling telemetry buffer (deque with 600-point capacity)
Implement conversation history tracking
Create to_prompt_context() method for LLM injection


Phase 2: Telemetry Processing (No External Dependencies)
4. Telemetry Agent

Implement rule-based telemetry parsing and validation
Build event detection logic (fuel, tire, gap, lap completion)
Implement threshold checking against configuration
Target: <50ms processing latency

5. WebSocket Server

Set up FastAPI WebSocket endpoint (/live)
Implement connection handling
Build message routing (session_init, telemetry, text_query, config_update, session_end)
Add heartbeat mechanism


Phase 3: LLM Integration
6. LLM Client (LangChain + Granite)

Implement GraniteClient with LangChain wrapper
Set up WatsonxLLM integration
Create prompt templates (PromptTemplate objects)
Add LangSmith callbacks for observability (optional)

7. Race Engineer Agent

Implement proactive response generation (event-triggered)
Implement reactive response generation (query-triggered)
Connect to LLM client
Target: <2000ms LLM response time


Phase 4: Custom Orchestrator
8. Priority Queue System

Implement heapq-based priority queue
Define priority levels (Critical, High, Medium, Low)
Build event queueing logic

9. Orchestrator Core

Implement JarvisLiveOrchestrator class (~200-300 LOC)
Build event router
Connect Telemetry Agent → Race Engineer Agent pipeline
Implement queue processing logic

10. Interrupt Handler

Implement sentence completion detection
Build pending interrupt queue
Implement priority-based interrupt logic (Critical/High interrupts Medium/Low)


Phase 5: Voice Infrastructure
11. Watson TTS/STT Clients

Implement WatsonTTSClient with Tenacity retry
Implement WatsonSTTClient with Tenacity retry
Configure exponential backoff (3 attempts, 0.5-4s wait)
Target: <500ms TTS latency

12. LiveKit Integration

Implement VoiceAgent with LiveKit room connection
Build audio track publishing (AI → Driver)
Build audio track subscription (Driver → AI)
Implement token generation

13. Voice Pipeline

Connect STT → Orchestrator → TTS flow
Implement audio frame streaming
Add silence detection for speech segmentation
Implement sentence splitting for interrupt handling


Phase 6: Integration & Testing
14. End-to-End Integration

Wire all components together in main.py
Implement session lifecycle (init → active → end)
Add LiveKit token to session_confirmed response
Test full pipeline: Telemetry → Event → LLM → Voice

15. Error Handling

Implement error codes (LLM_TIMEOUT, TTS_ERROR, LIVEKIT_ERROR, etc.)
Add graceful degradation (text fallback if voice fails)
Implement reconnection logic

16. Performance Optimization

Profile end-to-end latency (target: <3000ms)
Optimize bottlenecks
Add metrics logging


Phase 7: Deployment
17. Containerization

Create Dockerfile
Configure for IBM Cloud Code Engine
Set up secrets management

18. LiveKit Deployment

Deploy LiveKit server (self-hosted on IBM Cloud K8s or LiveKit Cloud)
Configure TURN servers for NAT traversal
Test WebRTC connectivity

run tests with python -m pytest tests/ -v

livekit can't be installed through pip