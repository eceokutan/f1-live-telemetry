# Jarvis-Granite Developer Documentation

**Project:** F1 Jarvis TORCS - AI Race Engineer Backend  
**Component:** jarvis-granite  
**Version:** 2.0 (Updated with Orchestration Architecture)  
**Last Updated:** December 2025

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Live Mode Orchestration Strategy](#3-live-mode-orchestration-strategy)
4. [Analysis Mode Orchestration Strategy](#4-analysis-mode-orchestration-strategy)
5. [Voice Infrastructure Layer - LiveKit](#5-voice-infrastructure-layer---livekit)
6. [System Components](#6-system-components)
7. [Jarvis-Granite Live (Real-Time)](#7-jarvis-granite-live-real-time)
8. [Jarvis-Granite Analysis (Post-Race)](#8-jarvis-granite-analysis-post-race)
9. [Shared Components](#9-shared-components)
10. [API Reference](#10-api-reference)
11. [Data Schemas](#11-data-schemas)
12. [Configuration](#12-configuration)
13. [Logging & Observability](#13-logging--observability)
14. [Deployment](#14-deployment)
15. [Development Guide](#15-development-guide)
16. [Integration Guide](#16-integration-guide)
17. [Appendices](#17-appendices)

---

## 1. Executive Summary

### 1.1 Purpose

jarvis-granite is the AI race engineer backend for the F1 Jarvis TORCS project. It provides intelligent, real-time race strategy advice and comprehensive post-race analysis powered by IBM Granite large language models.

> **Note:** This project uses a **custom proprietary orchestrator**, not Microsoft's Jarvis framework. The name "jarvis-granite" refers to our project codename combined with IBM's Granite LLM.

### 1.2 Key Characteristics

| Aspect | Description |
|--------|-------------|
| **Primary Function** | AI-powered race engineering insights via voice and text |
| **LLM Backend** | IBM Granite via watsonx.ai API |
| **Deployment** | IBM Cloud |
| **Architecture** | Custom lightweight orchestrator (Live) / LangGraph (Analysis) |
| **Consumers** | 2D Telemetry Platform, VR Analysis Platform |

### 1.3 Two Distinct Implementations

jarvis-granite consists of **two separate implementations** optimized for different use cases:

| Implementation | Purpose | Connection | Optimization |
|----------------|---------|------------|--------------|
| **jarvis-granite-live** | Real-time race engineer during racing | WebSocket + WebRTC (LiveKit) | Speed (<2s latency) |
| **jarvis-granite-analysis** | Post-race detailed analysis | REST API | Depth & thoroughness |

These implementations share common components (LLM client, prompts, schemas) but operate independently with different performance characteristics and interaction patterns.

### 1.4 Live vs Analysis Mode Comparison

| Aspect | Live Mode | Analysis Mode |
|--------|-----------|---------------|
| **Orchestration** | Custom lightweight (~200 LOC) | LangGraph (full framework) |
| **Rationale** | <2s latency critical | <15s allows framework overhead |
| **Voice Transport** | LiveKit (WebRTC) | N/A (text-only, REST API) |
| **LLM Management** | LangChain (component-only) | LangChain (via LangGraph) |
| **Retry Logic** | Tenacity (minimal) | Framework-native (LangGraph) |
| **State Management** | In-memory, manual | Framework-managed, persistent |
| **Agent Count** | 3 simple agents | 4+ analysis agents |
| **Complexity** | Low (fast decisions) | High (multi-step workflows) |
| **Connection Type** | WebSocket + WebRTC | REST API |
| **Primary Output** | Voice (with text fallback) | Text (structured JSON) |
| **Optimization Goal** | Speed & responsiveness | Depth & thoroughness |

### 1.5 Design Principles

1. **Stateless for telemetry** — Platforms handle their own data storage; jarvis-granite processes data on-demand
2. **Event-driven (live)** — AI analysis triggers automatically based on telemetry events
3. **On-demand (analysis)** — AI analysis runs only when explicitly requested by users
4. **Layered architecture** — Specialized tools for each layer (LiveKit for transport, LangChain for LLM, custom for business logic)
5. **Configurable** — Designed for iterative LLM fine-tuning and experimentation
6. **Observable** — All AI interactions logged for model improvement
7. **Performance-aware orchestration** — Use frameworks where they add value (Analysis), custom code where performance demands it (Live)

---

## 2. Architecture Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                    DATA SOURCES                                      │
├─────────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐         ┌──────────┐         ┌──────────┐         ┌──────────┐       │
│  │  TORCS   │         │ Assetto  │         │ CAN Bus  │         │  Driver  │       │
│  │Simulator │         │  Corsa   │         │(Formula  │         │  Voice   │       │
│  │          │         │          │         │ Student) │         │  Input   │       │
│  └────┬─────┘         └────┬─────┘         └────┬─────┘         └────┬─────┘       │
└───────┴────────────────────┴────────────────────┴────────────────────┴──────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
                    ▼                                       ▼
┌─────────────────────────────────────┐   ┌─────────────────────────────────────┐
│      JARVIS-GRANITE-LIVE            │   │      JARVIS-GRANITE-ANALYSIS        │
│         (Real-Time)                 │   │          (Post-Race)                │
├─────────────────────────────────────┤   ├─────────────────────────────────────┤
│  • WebSocket API                    │   │  • REST API                         │
│  • LiveKit WebRTC transport         │   │  • LangGraph orchestration          │
│  • Custom orchestrator              │   │  • Multi-agent workflows            │
│  • LangChain for LLM calls          │   │  • On-demand AI analysis            │
│  • Event-driven AI triggers         │   │  • Text responses                   │
│  • Voice I/O (Watson TTS/STT)       │   │  • Full session context             │
│  • Rolling 60s telemetry buffer     │   │  • Detailed insights                │
│  • <2s response latency             │   │  • Comprehensive outputs            │
│  • Concise outputs                  │   │                                     │
└─────────────────────────────────────┘   └─────────────────────────────────────┘
                    │                                       │
                    ▼                                       ▼
┌─────────────────────────────────────┐   ┌─────────────────────────────────────┐
│        LIVE PLATFORM                │   │       ANALYSIS PLATFORMS            │
│      (2D Real-Time Mode)            │   │      (2D + VR Post-Race)            │
├─────────────────────────────────────┤   ├─────────────────────────────────────┤
│  • Real-time telemetry display      │   │  • Session replay                   │
│  • Voice interaction with AI        │   │  • On-demand AI insights            │
│  • Proactive AI alerts              │   │  • Lap comparison                   │
│  • Handles own data storage         │   │  • Improvement recommendations      │
└─────────────────────────────────────┘   └─────────────────────────────────────┘
```

### 2.2 Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Runtime** | Python 3.11+ | Primary language |
| **Web Framework** | FastAPI | REST API + WebSocket support |
| **Voice Infrastructure** | LiveKit | WebRTC transport, real-time audio streaming |
| **LLM** | IBM Granite 3.x via watsonx.ai | Race engineer intelligence |
| **LLM Management** | LangChain | Prompt management, retry logic, observability |
| **Retry Logic** | Tenacity | Exponential backoff for API failures |
| **Orchestration (Live)** | Custom lightweight | Event routing, priority queue, racing logic |
| **Orchestration (Analysis)** | LangGraph | Multi-step workflow management, state persistence |
| **Speech-to-Text** | IBM Watson STT | Driver voice input |
| **Text-to-Speech** | IBM Watson TTS | AI voice output |
| **Deployment** | IBM Cloud (Code Engine / Kubernetes) | Hosting |
| **Logging** | Structured JSON logs | AI interaction tracking |

### 2.3 External Services

| Service | Provider | Purpose | Endpoint |
|---------|----------|---------|----------|
| Granite LLM | IBM watsonx.ai | Text generation | `https://us-south.ml.cloud.ibm.com` |
| Speech-to-Text | IBM Watson | Voice transcription | `https://api.us-south.speech-to-text.watson.cloud.ibm.com` |
| Text-to-Speech | IBM Watson | Voice synthesis | `https://api.us-south.text-to-speech.watson.cloud.ibm.com` |
| Voice Transport | LiveKit | WebRTC infrastructure | `wss://livekit-server.yourproject.com` |

---

## 3. Live Mode Orchestration Strategy

### 3.1 Implementation Approach

**Custom Lightweight Orchestrator with Component-Level Frameworks**

**Rationale:** <2s latency requirement demands minimal overhead and precise control over interrupt handling.

### 3.2 Core Orchestrator

- **Custom orchestrator core** (~200-300 lines of code)
- Handles event routing between 3 agents (Telemetry, Race Engineer, Voice)
- Manages priority queue for interrupt handling
- Direct control over audio buffer flushing and cancellation

### 3.3 Framework Usage (Component-Level ONLY)

#### LangChain - For Race Engineer Agent LLM calls ONLY

- Prompt template management and versioning
- Automatic retry logic for IBM watsonx.ai API
- Built-in observability callbacks for logging
- Streaming support for future optimizations

#### Tenacity - For retry logic on Watson TTS/STT calls

- Exponential backoff on API failures
- Lightweight (<10KB), no orchestration overhead

#### Pydantic - For data validation

- Request/response schema validation
- Type safety for telemetry data

### 3.4 What We Explicitly DON'T Use

- ❌ **LangGraph** (too heavy for 3-agent system)
- ❌ **CrewAI** (unnecessary abstraction)
- ❌ **AutoGen** (performance overhead)
- ❌ **Full orchestration frameworks**

### 3.5 Live Orchestrator Code Example

```python
class JarvisLiveOrchestrator:
    def __init__(self):
        self.telemetry_agent = TelemetryAgent()
        self.engineer_agent = RaceEngineerAgent(llm_via_langchain=True)
        self.voice_agent = VoiceAgent(retry_via_tenacity=True)
        self.priority_queue = CustomPriorityQueue()
    
    async def handle_event(self, event):
        # Custom routing logic for racing scenarios
        if event.priority == "CRITICAL":
            await self.interrupt_and_handle(event)
        else:
            await self.queue_and_process(event)
```

---

## 4. Analysis Mode Orchestration Strategy

### 4.1 Implementation Approach

**Full Orchestration Framework (LangGraph)**

**Rationale:** <15s latency budget allows framework overhead. Complexity of multi-step analysis workflows benefits from established orchestration patterns.

### 4.2 LangGraph Capabilities

- State management for multi-step analysis workflows
- Agent-to-agent communication for complex comparisons
- Built-in error recovery and retry logic
- Graph-based workflow visualization for debugging

### 4.3 Why Framework Makes Sense for Analysis Mode

#### Complex Workflows Required:

- Lap comparison (requires multiple data fetches)
- Sector-by-sector analysis (multi-step reasoning)
- Historical pattern recognition (stateful analysis)
- User Q&A with context retention across queries

#### Adequate Latency Budget:

- 10-15 seconds allows for framework initialization overhead
- Multiple LLM calls in sequence
- Sophisticated error handling and retries

#### Maintainability Benefits:

- Visual workflow graphs for debugging
- Standardized debugging tools
- Easier onboarding for new developers
- Industry-standard patterns

### 4.4 Framework Options

**Primary Recommendation: LangGraph**
- Native multi-agent support
- State persistence between analysis steps
- Excellent visualization tools
- Strong IBM watsonx.ai integration

**Alternative: CrewAI**
- Use if simpler agent delegation patterns are sufficient
- Lighter than LangGraph but more structured than custom
- Good for task-oriented agent teams

### 4.5 Analysis Orchestrator Code Example

```python
from langgraph.graph import StateGraph
from langchain_ibm import WatsonxLLM

class JarvisAnalysisOrchestrator:
    def __init__(self):
        self.graph = StateGraph()
        
        # Define analysis workflow
        self.graph.add_node("data_fetch_agent", DataFetchAgent())
        self.graph.add_node("comparison_agent", ComparisonAgent())
        self.graph.add_node("insight_agent", InsightAgent())
        self.graph.add_node("recommendation_agent", RecommendationAgent())
        
        # Framework handles routing, state management, retries
        self.graph.add_edge("data_fetch_agent", "comparison_agent")
        self.graph.add_edge("comparison_agent", "insight_agent")
        self.graph.add_edge("insight_agent", "recommendation_agent")
    
    async def analyze_lap(self, session_id, lap_number):
        # Framework manages state across all agents
        result = await self.graph.invoke({
            "session_id": session_id,
            "lap_number": lap_number
        })
        return result
```

---

## 5. Voice Infrastructure Layer - LiveKit

### 5.1 Why LiveKit

WebRTC transport complexity should be handled by specialized infrastructure, not custom code.

### 5.2 Core Capabilities

- WebRTC connection negotiation and management
- Low-latency audio streaming (bidirectional)
- Jitter buffering and packet loss recovery
- Audio synchronization
- Network adaptation for varying conditions
- Audio mixing and routing

### 5.3 Why We Need It

- Current architecture requires bidirectional audio streaming (driver ↔ AI)
- WebSocket-based voice communication specified in document
- Sub-3 second end-to-end voice latency requirement
- Audio buffer management and interrupt handling during speech
- These are complex real-time transport problems that LiveKit solves

### 5.4 Updated Voice Architecture

```
┌─────────────────────────────────────────────────────┐
│           DRIVER (Browser/VR Headset)               │
│              WebRTC Audio Stream                    │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              LIVEKIT SERVER                         │
│  • Manages WebRTC connections                       │
│  • Audio mixing/routing                             │
│  • Low-latency streaming (<100ms)                   │
│  • Handles network issues gracefully                │
│  • Provides clean audio streams to services         │
└─────────────────────┬───────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
┌──────────────────┐      ┌──────────────────┐
│   Watson STT     │      │   Watson TTS     │
│  (or Granite STT)│      │  (or Granite TTS)│
│  Voice → Text    │      │  Text → Voice    │
└────────┬─────────┘      └─────────┬────────┘
         │                          │
         └──────────┬───────────────┘
                    ▼
        ┌─────────────────────────┐
        │  Custom Orchestrator    │
        │  • Event routing        │
        │  • Priority queue       │
        │  • Interrupt logic      │
        │  • LLM calls (Granite)  │
        └─────────────────────────┘
```

### 5.5 Layer Separation

| Layer | Technology | Responsibility |
|-------|------------|----------------|
| **Transport** | LiveKit | WebRTC, audio streaming, connection management, network issues |
| **Speech AI** | Watson STT/TTS (or Granite) | Voice ↔ Text conversion |
| **Orchestration** | Custom (with LangChain components) | Racing logic, event routing, priority decisions |
| **Intelligence** | Granite LLM | Race strategy, insights, analysis |

### 5.6 IBM Alternative Analysis

**IBM Watson Voice Gateway:**
- Built on telephony infrastructure (SIP/PSTN focus)
- Designed for customer service IVR, not racing telemetry
- ~500ms overhead - too much for <2s target
- **Verdict:** Not suitable for our use case

**IBM Has No Direct LiveKit Equivalent:**
- IBM focuses on API-based services (Watson STT/TTS)
- Does not provide low-latency WebRTC infrastructure
- **Conclusion:** LiveKit fills a necessary gap in the stack

### 5.7 Deployment Considerations

**LiveKit Hosting Options:**
- Self-hosted (open source) on IBM Cloud Kubernetes
- LiveKit Cloud (managed service)
- Cost estimate: ~$0.004/minute (~$24/hour for 100 concurrent sessions)

**Integration with Existing Stack:**
- LiveKit handles transport layer ONLY
- Does NOT replace custom orchestrator
- Complements Watson/Granite services
- No impact on Analysis Mode (which uses REST API, no voice)

### 5.8 Voice Agent Code Example

```python
from livekit import rtc
from tenacity import retry, stop_after_attempt

class VoiceAgent:
    def __init__(self):
        self.livekit_room = rtc.Room()
        self.watson_stt = WatsonSTTClient()
        self.watson_tts = WatsonTTSClient()
        self.orchestrator = None  # Injected
    
    async def connect_driver(self, driver_id, room_name):
        """Establish LiveKit WebRTC connection with driver"""
        await self.livekit_room.connect(
            url="wss://livekit-server.yourproject.com",
            token=generate_token(driver_id, room_name)
        )
        
        # Subscribe to driver's audio track
        self.livekit_room.on("track_subscribed", self.handle_driver_audio)
    
    async def handle_driver_audio(self, track: rtc.RemoteAudioTrack):
        """Process incoming driver voice via LiveKit"""
        async for audio_frame in track:
            # LiveKit provides clean audio frames
            text = await self.watson_stt.recognize(audio_frame)
            
            if text:
                # Send to orchestrator for decision making
                response = await self.orchestrator.process_driver_input(text)
                await self.speak_to_driver(response)
    
    @retry(stop=stop_after_attempt(3))
    async def speak_to_driver(self, text: str):
        """Send AI response back to driver via LiveKit"""
        # Convert text to audio
        audio_data = await self.watson_tts.synthesize(text)
        
        # Publish to LiveKit room
        audio_track = rtc.LocalAudioTrack.create_audio_track(
            "ai-engineer",
            audio_data
        )
        await self.livekit_room.local_participant.publish_track(audio_track)
    
    async def interrupt_current_speech(self):
        """Interrupt AI mid-speech (critical alerts)"""
        # LiveKit allows instant track unpublishing
        await self.livekit_room.local_participant.unpublish_track("ai-engineer")
```

---

## 6. System Components

### 6.1 Component Overview

Both jarvis-granite implementations share a common component structure:

```
┌─────────────────────────────────────────────────────────────────┐
│                         SHARED COMPONENTS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    ORCHESTRATOR                         │   │
│   │  • Routes events to appropriate agents                  │   │
│   │  • Manages priority queue (live only)                   │   │
│   │  • Handles mode-specific logic                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│          ┌───────────────────┼───────────────────┐              │
│          │                   │                   │              │
│          ▼                   ▼                   ▼              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │  TELEMETRY  │    │    RACE     │    │    VOICE    │        │
│   │    AGENT    │    │  ENGINEER   │    │    AGENT    │        │
│   │             │    │    AGENT    │    │             │        │
│   │ • Parse     │    │ • Granite   │    │ • LiveKit   │        │
│   │ • Validate  │    │   LLM       │    │   WebRTC    │        │
│   │ • Detect    │    │ • Strategy  │    │ • Watson    │        │
│   │   events    │    │ • Insights  │    │   TTS/STT   │        │
│   │             │    │             │    │ • Audio     │        │
│   │ LLM: No     │    │ LLM: Yes    │    │   streaming │        │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    LLM CLIENT                           │   │
│   │  • IBM watsonx.ai integration via LangChain             │   │
│   │  • Swappable backend (HuggingFace for dev)              │   │
│   │  • Retry logic via Tenacity                             │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                   LOGGING SERVICE                       │   │
│   │  • AI interaction logs (JSON lines)                     │   │
│   │  • Performance metrics                                  │   │
│   │  • Debug logging                                        │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Agent Responsibilities

#### 6.2.1 Telemetry Agent

**Purpose:** Parse, validate, and analyze incoming telemetry data.

| Aspect | Details |
|--------|---------|
| **LLM Usage** | None — rule-based processing |
| **Latency Budget** | <50ms |
| **Input** | Raw telemetry JSON from platforms |
| **Output** | Parsed context, event triggers |

**Responsibilities:**
- Parse incoming telemetry messages
- Validate data integrity and ranges
- Update session context (current lap, fuel, tires, position)
- Maintain rolling telemetry buffer (live mode only)
- Detect threshold breaches and trigger events
- Calculate derived metrics (degradation rates, fuel consumption)

**Event Triggers (Live Mode):**

| Event | Condition | Priority |
|-------|-----------|----------|
| `pit_window_open` | Fuel < threshold OR tire wear > threshold | High |
| `tire_critical` | Tire temp > 110°C | Critical |
| `fuel_critical` | Fuel < 2 laps remaining | Critical |
| `gap_change` | Gap to car ahead/behind changes > 1s | Medium |
| `lap_complete` | Track position crosses start/finish | Medium |
| `sector_complete` | Sector boundary crossed | Low |

#### 6.2.2 Race Engineer Agent

**Purpose:** Generate strategic insights and answer driver queries using IBM Granite LLM.

| Aspect | Details |
|--------|---------|
| **LLM Usage** | Primary — all outputs via Granite |
| **Latency Budget** | <2s (live), <10s (analysis) |
| **Input** | Session context, telemetry, queries |
| **Output** | Strategic advice, analysis, answers |

**Responsibilities:**
- Generate proactive strategic recommendations (live mode)
- Answer driver questions with contextual awareness
- Provide comprehensive post-race analysis (analysis mode)
- Adapt communication style to verbosity setting
- Maintain conversation coherence

**Operating Modes:**

| Mode | Trigger | Response Style |
|------|---------|----------------|
| **Proactive** | Telemetry events | Brief, actionable |
| **Reactive** | Driver query | Conversational, complete |
| **Analysis** | User request | Detailed, comprehensive |

#### 6.2.3 Voice Agent

**Purpose:** Handle real-time voice transport and speech conversion.

| Aspect | Details |
|--------|---------|
| **LLM Usage** | None — uses LiveKit + IBM Watson services |
| **Latency Budget** | <500ms |
| **Used In** | Live mode only |

**Responsibilities:**
- Manage LiveKit WebRTC connections
- Convert driver speech to text (Watson STT)
- Convert AI responses to speech (Watson TTS)
- Manage audio streaming
- Handle interrupt logic (finish sentence before new response)
- Priority queue management for concurrent outputs

---

## 7. Jarvis-Granite Live (Real-Time)

### 7.1 Overview

jarvis-granite-live provides real-time AI race engineering during active racing sessions. It connects via WebSocket, processes telemetry continuously, and delivers voice-based strategic advice.

### 7.2 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    JARVIS-GRANITE-LIVE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                 WebSocket Server                        │   │
│   │  Endpoint: ws://host:port/live                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              CUSTOM ORCHESTRATOR                        │   │
│   │                                                         │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│   │  │   Event     │  │  Priority   │  │  Interrupt  │     │   │
│   │  │   Router    │  │   Queue     │  │  Handler    │     │   │
│   │  └─────────────┘  └─────────────┘  └─────────────┘     │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│          ┌───────────────────┼───────────────────┐              │
│          ▼                   ▼                   ▼              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│   │  Telemetry  │───▶│    Race     │───▶│    Voice    │        │
│   │    Agent    │    │  Engineer   │    │    Agent    │        │
│   │             │    │ (LangChain) │    │  (LiveKit)  │        │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│          │                                     │                │
│          ▼                                     ▼                │
│   ┌─────────────┐                      ┌─────────────┐         │
│   │  Rolling    │                      │   Watson    │         │
│   │  Buffer     │                      │  TTS / STT  │         │
│   │  (60s)      │                      │             │         │
│   └─────────────┘                      └─────────────┘         │
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

### 7.3 Data Flow

```
TELEMETRY FLOW:
Platform ──WebSocket──▶ Telemetry Agent ──▶ Update Context
                                │
                                ▼
                        Event Detected?
                                │
                    ┌───────────┴───────────┐
                    │ Yes                   │ No
                    ▼                       ▼
            Race Engineer           Continue monitoring
            (via LangChain)
                    │
                    ▼
            Generate Response
                    │
                    ▼
              Voice Agent
              (via LiveKit)
                    │
                    ▼
         ◀──WebSocket──  Platform (Audio)


VOICE QUERY FLOW:
Platform ──LiveKit/WebRTC──▶ Voice Agent (STT)
                                │
                                ▼
                        Transcribed Text
                                │
                                ▼
                        Race Engineer
                        (via LangChain)
                                │
                                ▼
                        Generate Response
                                │
                                ▼
                        Voice Agent (TTS)
                                │
                                ▼
         ◀──LiveKit/WebRTC──  Platform (Audio)
```

### 7.4 Event Processing

#### 7.4.1 Event Types and Priorities

| Priority | Level | Events | Behavior |
|----------|-------|--------|----------|
| **Critical** | 0 | Brake failure, collision warning | Immediate interrupt |
| **High** | 1 | Pit now, fuel critical, tire failure | Interrupt medium/low |
| **Medium** | 2 | Pit window, gap changes, lap summary | Queue normally |
| **Low** | 3 | Sector times, minor updates | Skip if busy |

#### 7.4.2 Event Detection Logic

```python
class TelemetryAgent:
    def detect_events(self, telemetry: TelemetryMessage, context: SessionContext) -> List[Event]:
        events = []
        
        # Fuel events
        fuel_laps_remaining = self.calculate_fuel_laps(telemetry, context)
        if fuel_laps_remaining <= context.config.fuel_critical_laps:
            events.append(Event(type="fuel_critical", priority="critical", data={"laps": fuel_laps_remaining}))
        elif fuel_laps_remaining <= context.config.fuel_warning_laps:
            events.append(Event(type="fuel_warning", priority="high", data={"laps": fuel_laps_remaining}))
        
        # Tire events
        max_tire_temp = max(telemetry.data.tire_temps.values())
        if max_tire_temp >= context.config.tire_temp_critical:
            events.append(Event(type="tire_critical", priority="critical", data={"temp": max_tire_temp}))
        elif max_tire_temp >= context.config.tire_temp_warning:
            events.append(Event(type="tire_warning", priority="medium", data={"temp": max_tire_temp}))
        
        # Gap events
        if context.gap_ahead is not None and telemetry.data.gap_ahead is not None:
            gap_change = abs(telemetry.data.gap_ahead - context.gap_ahead)
            if gap_change >= context.config.gap_change_threshold:
                events.append(Event(type="gap_change", priority="medium", data={"change": gap_change}))
        
        # Lap completion
        if telemetry.data.lap_number > context.current_lap:
            events.append(Event(type="lap_complete", priority="medium", data={"lap": telemetry.data.lap_number}))
        
        return events
```

### 7.5 Interrupt Handling

When the driver speaks while the AI is outputting audio:

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

**Implementation:**

```python
class VoiceAgent:
    def __init__(self):
        self.livekit_room = rtc.Room()
        self.currently_speaking = False
        self.pending_query: Optional[str] = None
        self.current_sentence_complete = asyncio.Event()
    
    async def handle_driver_input(self, audio_frame) -> None:
        text = await self.watson_stt.transcribe(audio_frame)
        
        if self.currently_speaking:
            # Queue the query, wait for current sentence to complete
            self.pending_query = text
            await self.current_sentence_complete.wait()
        
        # Process the query
        response = await self.orchestrator.handle_driver_query(text)
        await self.speak(response)
    
    async def speak(self, text: str, priority: str = "medium") -> None:
        sentences = self.split_into_sentences(text)
        
        for sentence in sentences:
            self.currently_speaking = True
            audio = await self.watson_tts.synthesize(sentence)
            await self.stream_audio_via_livekit(audio)
            
            # Signal sentence complete for interrupt handling
            self.current_sentence_complete.set()
            self.current_sentence_complete.clear()
            
            # Check for pending query after each sentence
            if self.pending_query:
                break
        
        self.currently_speaking = False
```

### 7.6 Session Context (Live)

```python
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from typing import Optional, List, Dict, Any

@dataclass
class LiveSessionContext:
    """Context maintained during a live racing session"""
    
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
```

### 7.7 Prompts (Live Mode)

#### 7.7.1 System Prompt

```python
LIVE_SYSTEM_PROMPT = """You are an expert F1 race engineer communicating with your driver over team radio during a live race.

CRITICAL CONSTRAINTS:
- Driver is actively racing and cannot read text
- Responses must be CONCISE (1-3 sentences max)
- Lead with the most important information
- Use precise numbers when helpful
- Match urgency to the situation

CURRENT SESSION:
- Track: {track_name}
- Current Lap: {current_lap}
- Position: P{position}
- Gap Ahead: {gap_ahead}s | Gap Behind: {gap_behind}s

VEHICLE STATE:
- Fuel: {fuel_remaining}L ({fuel_laps_remaining} laps)
- Tire Temps: FL:{tire_fl}°C FR:{tire_fr}°C RL:{tire_rl}°C RR:{tire_rr}°C
- Tire Wear: FL:{wear_fl}% FR:{wear_fr}% RL:{wear_rl}% RR:{wear_rr}%

RECENT LAPS:
{recent_lap_times}

VERBOSITY: {verbosity_level}
{verbosity_instructions}

CONVERSATION HISTORY:
{conversation_history}
"""
```

#### 7.7.2 Proactive Trigger Prompt

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

#### 7.7.3 Driver Query Prompt

```python
DRIVER_QUERY_PROMPT = """DRIVER ASKED: "{query}"

Answer the driver's question using the current session data. Be concise but complete.

If the question involves strategy, provide your recommendation with reasoning.
If you need to reference data, use specific numbers.

Your radio response:"""
```

---

## 8. Jarvis-Granite Analysis (Post-Race)

### 8.1 Overview

jarvis-granite-analysis provides comprehensive post-race analysis via REST API. Analysis runs **on-demand** when users explicitly request it, allowing for thorough LLM-powered insights without blocking the UI.

### 8.2 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  JARVIS-GRANITE-ANALYSIS                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    REST API Server                      │   │
│   │                                                         │   │
│   │  POST /analysis/summary      - Session summary          │   │
│   │  POST /analysis/lap/{n}      - Lap analysis             │   │
│   │  POST /analysis/compare      - Lap comparison           │   │
│   │  POST /analysis/query        - Natural language query   │   │
│   │  POST /analysis/improvements - Recommendations          │   │
│   │                                                         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              LANGGRAPH ORCHESTRATOR                     │   │
│   │                                                         │   │
│   │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐ │   │
│   │  │  Data   │──▶│Compare  │──▶│ Insight │──▶│ Recomm. │ │   │
│   │  │  Fetch  │   │  Agent  │   │  Agent  │   │  Agent  │ │   │
│   │  └─────────┘   └─────────┘   └─────────┘   └─────────┘ │   │
│   │                                                         │   │
│   │  • State persistence between steps                      │   │
│   │  • Automatic retry and error recovery                   │   │
│   │  • Workflow visualization for debugging                 │   │
│   │                                                         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                  RACE ENGINEER AGENT                    │   │
│   │                                                         │   │
│   │  • Full context processing                              │   │
│   │  • Detailed analysis generation                         │   │
│   │  • Multi-lap comparison logic                           │   │
│   │  • Improvement identification                           │   │
│   │                                                         │   │
│   │  Optimized for DEPTH over speed                         │   │
│   │                                                         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.3 Data Flow

```
USER ACTION                          JARVIS-GRANITE-ANALYSIS
    │
    │  User clicks "Analyze Lap 5"
    │
    ▼
Platform retrieves lap 5 telemetry from its storage
    │
    ▼
Platform sends POST /analysis/lap/5 with telemetry data
    │
    ▼
                                    LangGraph Orchestrator receives request
                                            │
                                            ▼
                                    Data Fetch Agent assembles context
                                            │
                                            ▼
                                    Comparison Agent analyzes vs baseline
                                            │
                                            ▼
                                    Insight Agent generates observations
                                            │
                                            ▼
                                    Recommendation Agent creates action items
                                            │
    ◀───────────────────────────────────────┘
    │
    ▼
Platform displays analysis to user
```

### 8.4 Prompts (Analysis Mode)

#### 8.4.1 System Prompt

```python
ANALYSIS_SYSTEM_PROMPT = """You are an expert motorsport data analyst and race engineer reviewing telemetry data from a completed session.

Your role is to provide thorough, insightful analysis that helps the driver improve. Unlike live race communication, you have time for detailed explanations.

ANALYSIS PRINCIPLES:
1. Be specific - reference exact lap numbers, corners, times
2. Be actionable - every observation should lead to a recommendation
3. Be balanced - acknowledge strengths while identifying improvements
4. Use data - support claims with telemetry evidence
5. Prioritize - focus on changes with biggest impact first

SESSION CONTEXT:
{session_context}

OUTPUT FORMAT:
Provide structured, detailed analysis. Use clear sections and bullet points where helpful.
Include specific numbers and comparisons to support your insights.
"""
```

#### 8.4.2 Lap Analysis Prompt

```python
LAP_ANALYSIS_PROMPT = """Analyze lap {lap_number} in detail.

LAP DATA:
- Lap Time: {lap_time}
- Sector Times: S1: {sector_1}, S2: {sector_2}, S3: {sector_3}

TELEMETRY SUMMARY:
{telemetry_summary}

COMPARISON BASELINE (Best Lap):
{best_lap_data}

Provide:
1. Overall assessment of the lap
2. Sector-by-sector breakdown with specific observations
3. Key corners where time was gained or lost
4. Tire and fuel usage observations
5. Specific recommendations for improvement

Be thorough and specific. Reference exact speeds, distances, and times."""
```

---

## 9. Shared Components

### 9.1 LLM Client

The LLM client uses LangChain to abstract the connection to IBM Granite, providing prompt management, retry logic, and observability.

```python
# llm/langchain_client.py
from langchain_ibm import WatsonxLLM
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import get_openai_callback
from tenacity import retry, stop_after_attempt, wait_exponential

class GraniteLangChainClient:
    def __init__(self, api_key: str, project_id: str):
        self.llm = WatsonxLLM(
            model_id="ibm/granite-3-8b-instruct",
            url="https://us-south.ml.cloud.ibm.com",
            apikey=api_key,
            project_id=project_id,
            params={
                "max_new_tokens": 150,
                "temperature": 0.7
            }
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def generate(
        self,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 150
    ) -> str:
        template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", prompt)
        ])
        
        chain = template | self.llm
        response = await chain.ainvoke({})
        
        return response
```

```python
# llm/huggingface_client.py (for development)
from huggingface_hub import InferenceClient

class HuggingFaceClient:
    """Development client using HuggingFace Inference API"""
    
    def __init__(self, token: str, model: str = "ibm-granite/granite-3.0-8b-instruct"):
        self.client = InferenceClient(model=model, token=token)
        self.model = model
    
    async def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 1000
    ) -> str:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        response = self.client.text_generation(
            full_prompt,
            max_new_tokens=max_tokens,
            temperature=0.7
        )
        
        return response
```

### 9.2 Voice Services

```python
# voice/watson_tts.py
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

class WatsonTTSClient:
    def __init__(self, api_key: str, service_url: str):
        self.api_key = api_key
        self.service_url = service_url
        self.voice = "en-GB_JamesV3Voice"  # British male voice for race engineer feel
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def synthesize(self, text: str, voice: str = None) -> bytes:
        voice = voice or self.voice
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.service_url}/v1/synthesize",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "audio/wav"
                },
                auth=("apikey", self.api_key),
                json={"text": text},
                params={"voice": voice},
                timeout=10.0
            )
            response.raise_for_status()
            return response.content
```

```python
# voice/watson_stt.py
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

class WatsonSTTClient:
    def __init__(self, api_key: str, service_url: str):
        self.api_key = api_key
        self.service_url = service_url
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def transcribe(self, audio: bytes, content_type: str = "audio/wav") -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.service_url}/v1/recognize",
                headers={"Content-Type": content_type},
                auth=("apikey", self.api_key),
                content=audio,
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
        
        if data.get("results"):
            return data["results"][0]["alternatives"][0]["transcript"]
        return ""
```

---

## 10. API Reference

### 10.1 Jarvis-Granite-Live WebSocket API

#### Connection

**Endpoint:** `ws://host:port/live`

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
   │                                   │
   │──── Telemetry Stream ────────────▶│
   │                                   │
   │◀─── AI Responses (voice/text) ────│
```

### 10.2 Jarvis-Granite-Analysis REST API

#### Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/analysis/summary` | POST | Session summary |
| `/analysis/lap/{n}` | POST | Lap analysis |
| `/analysis/compare` | POST | Lap comparison |
| `/analysis/query` | POST | Natural language query |
| `/analysis/improvements` | POST | Recommendations |

---

## 11. Data Schemas

### 11.1 Telemetry Message

```python
from pydantic import BaseModel
from typing import Dict, Optional
from datetime import datetime

class TelemetryData(BaseModel):
    timestamp: datetime
    lap_number: int
    sector: int
    speed_kmh: float
    rpm: int
    gear: int
    throttle: float  # 0.0 - 1.0
    brake: float     # 0.0 - 1.0
    steering: float  # -1.0 to 1.0
    fuel_remaining: float
    tire_temps: Dict[str, float]  # fl, fr, rl, rr
    tire_wear: Dict[str, float]
    position: Optional[int] = None
    gap_ahead: Optional[float] = None
    gap_behind: Optional[float] = None

class TelemetryMessage(BaseModel):
    type: str = "telemetry"
    session_id: str
    data: TelemetryData
```

---

## 12. Configuration

### 12.1 Dependencies

```toml
[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.104.0"
uvicorn = "^0.24.0"

# LLM and AI
langchain = "^0.1.0"
langchain-ibm = "^0.1.0"
langgraph = "^0.0.20"  # For Analysis Mode only

# Voice Infrastructure
livekit = "^0.10.0"
livekit-api = "^0.5.0"

# Retry and resilience
tenacity = "^8.2.0"

# IBM Services
ibm-watson = "^7.0.0"
ibm-watsonx-ai = "^0.1.0"

# Data validation
pydantic = "^2.5.0"

# Logging and observability
structlog = "^23.2.0"
```

### 12.2 Live Mode Configuration

```yaml
# config/live.yaml
orchestration:
  type: "custom"
  agents:
    - telemetry
    - race_engineer
    - voice
  priority_queue:
    max_size: 100
    timeout_ms: 2000

livekit:
  url: "wss://livekit-server.yourproject.com"
  api_key: "${LIVEKIT_API_KEY}"
  api_secret: "${LIVEKIT_API_SECRET}"
  room_prefix: "race_"

langchain:
  llm:
    provider: "ibm_watsonx"
    model_id: "ibm/granite-3-8b-instruct"
    max_tokens: 150
    temperature: 0.7
  callbacks:
    - logging
    - metrics

tenacity:
  max_attempts: 3
  wait_exponential:
    multiplier: 1
    min: 1
    max: 10
```

### 12.3 Analysis Mode Configuration

```yaml
# config/analysis.yaml
orchestration:
  type: "langgraph"
  workflow:
    - data_fetch_agent
    - comparison_agent
    - insight_agent
    - recommendation_agent
  state_persistence: true
  
langgraph:
  checkpointer: "memory"  # or "redis" for production
  max_iterations: 20
  timeout_seconds: 15

langchain:
  llm:
    provider: "ibm_watsonx"
    model_id: "ibm/granite-3-8b-instruct"
    max_tokens: 1000
    temperature: 0.7
```

### 12.4 Environment Variables

```bash
# .env.example

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
LIVEKIT_URL=wss://livekit-server.yourproject.com

# HuggingFace (for development)
HUGGINGFACE_TOKEN=your_hf_token

# Application
LLM_PROVIDER=watsonx  # or "huggingface" for development
LOG_LEVEL=INFO
LOG_DIRECTORY=logs
```

---

## 13. Logging & Observability

### 13.1 Logging Strategy

| Log Type | Format | Location | Purpose |
|----------|--------|----------|---------|
| AI Interactions | JSON Lines | `logs/ai_interactions/YYYY-MM-DD.jsonl` | Fine-tuning dataset, model improvement |
| Application Logs | Structured JSON | `logs/app.log` | Debugging, monitoring |
| Error Logs | Structured JSON | `logs/errors.log` | Error tracking |
| Performance Metrics | JSON | `logs/metrics/YYYY-MM-DD.jsonl` | Latency analysis |

### 13.2 AI Interaction Log Schema

```json
{
  "timestamp": "2026-01-15T14:32:11.789Z",
  "session_id": "race_001",
  "interaction_type": "proactive",
  "trigger": "pit_window_open",
  "input_context": {
    "current_lap": 15,
    "fuel_remaining": 12.5,
    "tire_wear": {"fl": 68, "fr": 70, "rl": 62, "rr": 64},
    "position": 3,
    "gap_ahead": 2.4
  },
  "prompt": "EVENT DETECTED: pit_window_open...",
  "response": "Pit window is open. Box this lap for fresh mediums.",
  "model": "ibm/granite-3-8b-instruct",
  "tokens_used": 45,
  "latency_ms": 1850,
  "priority": "high"
}
```

---

## 14. Deployment

### 14.1 IBM Cloud Deployment

#### Prerequisites

- IBM Cloud account with Code Engine access
- IBM watsonx.ai project
- IBM Watson Speech services provisioned
- LiveKit server deployed (self-hosted or LiveKit Cloud)
- Docker installed locally

#### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY jarvis_granite/ ./jarvis_granite/

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "jarvis_granite.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### IBM Code Engine Deployment

```bash
# Create project
ibmcloud ce project create --name jarvis-granite

# Select project
ibmcloud ce project select --name jarvis-granite

# Create build
ibmcloud ce build create --name jarvis-granite-build \
  --source https://github.com/your-org/jarvis-granite.git \
  --strategy dockerfile

# Deploy application
ibmcloud ce application create --name jarvis-granite-live \
  --build jarvis-granite-build \
  --port 8000 \
  --min-scale 1 \
  --env-from-secret jarvis-secrets

# Create secrets
ibmcloud ce secret create --name jarvis-secrets \
  --from-literal WATSONX_API_KEY=xxx \
  --from-literal WATSONX_PROJECT_ID=xxx \
  --from-literal WATSON_TTS_API_KEY=xxx \
  --from-literal WATSON_STT_API_KEY=xxx \
  --from-literal LIVEKIT_API_KEY=xxx \
  --from-literal LIVEKIT_API_SECRET=xxx
```

---

## 15. Development Guide

### 15.1 Local Setup

```bash
# Clone repository
git clone https://github.com/your-org/jarvis-granite.git
cd jarvis-granite

# Install dependencies
poetry install

# Copy environment template
cp .env.example .env
# Edit .env with your credentials

# Run live mode server
uvicorn jarvis_granite.live.main:app --reload --port 8001

# Run analysis mode server
uvicorn jarvis_granite.analysis.main:app --reload --port 8002
```

### 15.2 Using HuggingFace for Development

Set `LLM_PROVIDER=huggingface` in your `.env` file to use HuggingFace Inference API instead of watsonx.ai during development. This allows testing without IBM Cloud credentials.

---

## 16. Integration Guide

### 16.1 Integrating with 2D Platform (React)

```typescript
// Example WebSocket client for Live Mode
class JarvisLiveClient {
  private ws: WebSocket;
  private lkRoom: Room;  // LiveKit room for voice
  
  async connect(sessionConfig: SessionConfig) {
    // WebSocket for telemetry
    this.ws = new WebSocket('ws://jarvis-granite-live/live');
    
    // LiveKit for voice
    this.lkRoom = new Room();
    await this.lkRoom.connect(LIVEKIT_URL, token);
    
    this.ws.onopen = () => {
      this.ws.send(JSON.stringify({
        type: 'session_init',
        ...sessionConfig
      }));
    };
  }
  
  sendTelemetry(data: TelemetryData) {
    this.ws.send(JSON.stringify({
      type: 'telemetry',
      data
    }));
  }
}
```

---

## 17. Appendices

### 17.1 Glossary

| Term | Definition |
|------|------------|
| **jarvis-granite** | The AI race engineer backend system |
| **Live Mode** | Real-time operation during active racing |
| **Analysis Mode** | Post-race detailed analysis operation |
| **Proactive** | AI-initiated communication based on events |
| **Reactive** | AI response to driver queries |
| **Event** | Telemetry condition that may trigger AI response |
| **Verbosity** | Configuration for how "chatty" the AI is |
| **Rolling Buffer** | In-memory storage of recent telemetry |
| **LiveKit** | WebRTC infrastructure for real-time voice |
| **LangChain** | LLM tooling for prompt management and observability |
| **LangGraph** | Multi-agent workflow orchestration framework |

### 17.2 Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `LLM_TIMEOUT` | Granite API request timed out | Retry, check network |
| `LLM_ERROR` | Granite API returned error | Check credentials, quota |
| `TTS_ERROR` | Text-to-speech failed | Check Watson TTS service |
| `STT_ERROR` | Speech-to-text failed | Check audio format, Watson STT service |
| `LIVEKIT_ERROR` | WebRTC connection failed | Check LiveKit server, credentials |
| `VALIDATION_ERROR` | Invalid request data | Check request schema |
| `SESSION_NOT_FOUND` | Referenced session doesn't exist | Initialize session first |
| `CONFIG_ERROR` | Invalid configuration | Check config values |

### 17.3 Performance Targets

| Metric | Live Mode Target | Analysis Mode Target |
|--------|------------------|---------------------|
| Telemetry processing | <50ms | N/A |
| LLM response | <2000ms | <10000ms |
| TTS conversion | <500ms | N/A |
| End-to-end (voice) | <3000ms | N/A |
| API response | N/A | <15000ms |

### 17.4 Future Considerations

1. **Fine-tuned Models**: Once Granite is fine-tuned on motorsport data, update `model_id` in configuration
2. **Multi-session Support**: Current design supports single session; scaling requires session management layer
3. **Granite TTS/STT**: If IBM releases Granite TTS/STT, swap Watson clients (architecture supports this via Voice Agent abstraction)
4. **LiveKit Spatial Audio**: For VR platform, implement spatial audio positioning (engineer voice from pit wall location)
5. **LangGraph Advanced Features**: Explore human-in-the-loop workflows for Analysis Mode (driver corrects AI insights)
6. **Plugin Architecture**: Clean module boundaries allow future plugin system if needed
7. **Caching Layer**: Add Redis for response caching if latency becomes an issue
8. **LangSmith Integration**: Add LangSmith for production LLM monitoring and prompt optimization

---

## Summary of Architecture Changes (v1.0 → v2.0)

This document reflects 10 key architectural changes:

1. ✅ Clarified naming: Custom orchestrator, not Microsoft Jarvis framework
2. ✅ Live Mode: Custom lightweight orchestrator with LangChain/Tenacity components
3. ✅ Analysis Mode: Full LangGraph framework for multi-step workflows
4. ✅ Added LiveKit for WebRTC voice transport infrastructure
5. ✅ Updated Technology Stack table with new components
6. ✅ Added Live vs Analysis comparison table
7. ✅ Updated dependencies and configuration schemas
8. ✅ Updated architecture diagrams
9. ✅ Added layered architecture and performance-aware orchestration to design principles
10. ✅ Added future considerations for LiveKit spatial audio, LangGraph features, and LangSmith

**Key Principle:** Use specialized frameworks where they add value, custom code where performance demands it.

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | December 2025 | Team 17 | Initial documentation |
| 2.0 | December 2025 | Team 17 | Orchestration architecture updates (10 changes) |

---

*This document is part of the F1 Jarvis TORCS project. For questions or contributions, contact the development team.*