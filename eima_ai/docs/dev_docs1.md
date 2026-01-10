# Jarvis-Granite Developer Documentation

**Project:** F1 Jarvis TORCS - AI Race Engineer Backend  
**Component:** jarvis-granite  
**Version:** 1.0  
**Last Updated:** December 2025

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [System Components](#3-system-components)
4. [Jarvis-Granite Live (Real-Time)](#4-jarvis-granite-live-real-time)
5. [Jarvis-Granite Analysis (Post-Race)](#5-jarvis-granite-analysis-post-race)
6. [Shared Components](#6-shared-components)
7. [API Reference](#7-api-reference)
8. [Data Schemas](#8-data-schemas)
9. [Configuration](#9-configuration)
10. [Logging & Observability](#10-logging--observability)
11. [Deployment](#11-deployment)
12. [Development Guide](#12-development-guide)
13. [Integration Guide](#13-integration-guide)
14. [Appendices](#14-appendices)

---

## 1. Executive Summary

### 1.1 Purpose

jarvis-granite is the AI race engineer backend for the F1 Jarvis TORCS project. It provides intelligent, real-time race strategy advice and comprehensive post-race analysis powered by IBM Granite large language models.

### 1.2 Key Characteristics

| Aspect | Description |
|--------|-------------|
| **Primary Function** | AI-powered race engineering insights via voice and text |
| **LLM Backend** | IBM Granite via watsonx.ai API |
| **Deployment** | IBM Cloud |
| **Architecture** | Custom lightweight multi-agent orchestrator |
| **Consumers** | 2D Telemetry Platform, VR Analysis Platform |

### 1.3 Two Distinct Implementations

jarvis-granite consists of **two separate implementations** optimized for different use cases:

| Implementation | Purpose | Connection | Optimization |
|----------------|---------|------------|--------------|
| **jarvis-granite-live** | Real-time race engineer during racing | WebSocket | Speed (<2s latency) |
| **jarvis-granite-analysis** | Post-race detailed analysis | REST API | Depth & thoroughness |

These implementations share common components (LLM client, prompts, schemas) but operate independently with different performance characteristics and interaction patterns.

### 1.4 Design Principles

1. **Stateless for telemetry** — Platforms handle their own data storage; jarvis-granite processes data on-demand
2. **Event-driven (live)** — AI analysis triggers automatically based on telemetry events
3. **On-demand (analysis)** — AI analysis runs only when explicitly requested by users
4. **Configurable** — Designed for iterative LLM fine-tuning and experimentation
5. **Observable** — All AI interactions logged for model improvement

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
│  • Event-driven AI triggers         │   │  • On-demand AI analysis            │
│  • Voice I/O (Watson TTS/STT)       │   │  • Text responses                   │
│  • Rolling 60s telemetry buffer     │   │  • Full session context             │
│  • <2s response latency             │   │  • Detailed insights                │
│  • Concise outputs                  │   │  • Comprehensive outputs            │
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
| **LLM** | IBM Granite 3.x via watsonx.ai | Race engineer intelligence |
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

---

## 3. System Components

### 3.1 Component Overview

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
│   │ • Parse     │    │ • Granite   │    │ • Watson    │        │
│   │ • Validate  │    │   LLM       │    │   TTS/STT   │        │
│   │ • Detect    │    │ • Strategy  │    │ • Audio     │        │
│   │   events    │    │ • Insights  │    │   streaming │        │
│   │             │    │             │    │             │        │
│   │ LLM: No     │    │ LLM: Yes    │    │ LLM: No     │        │
│   └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                    LLM CLIENT                           │   │
│   │  • IBM watsonx.ai integration                           │   │
│   │  • Swappable backend (HuggingFace for dev)              │   │
│   │  • Retry logic and error handling                       │   │
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

### 3.2 Agent Responsibilities

#### 3.2.1 Telemetry Agent

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

#### 3.2.2 Race Engineer Agent

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

#### 3.2.3 Voice Agent

**Purpose:** Handle speech-to-text and text-to-speech conversion.

| Aspect | Details |
|--------|---------|
| **LLM Usage** | None — uses IBM Watson services |
| **Latency Budget** | <500ms |
| **Used In** | Live mode only |

**Responsibilities:**
- Convert driver speech to text (Watson STT)
- Convert AI responses to speech (Watson TTS)
- Manage audio streaming
- Handle interrupt logic (finish sentence before new response)
- Priority queue management for concurrent outputs

---

## 4. Jarvis-Granite Live (Real-Time)

### 4.1 Overview

jarvis-granite-live provides real-time AI race engineering during active racing sessions. It connects via WebSocket, processes telemetry continuously, and delivers voice-based strategic advice.

### 4.2 Architecture

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
│   │                    ORCHESTRATOR                         │   │
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

### 4.3 Data Flow

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
                    │
                    ▼
            Generate Response
                    │
                    ▼
              Voice Agent
                    │
                    ▼
         ◀──WebSocket──  Platform (Audio)


VOICE QUERY FLOW:
Platform ──WebSocket──▶ Voice Agent (STT)
                                │
                                ▼
                        Transcribed Text
                                │
                                ▼
                        Race Engineer
                                │
                                ▼
                        Generate Response
                                │
                                ▼
                        Voice Agent (TTS)
                                │
                                ▼
         ◀──WebSocket──  Platform (Audio)
```

### 4.4 Event Processing

#### 4.4.1 Event Types and Priorities

| Priority | Level | Events | Behavior |
|----------|-------|--------|----------|
| **Critical** | 0 | Brake failure, collision warning | Immediate interrupt |
| **High** | 1 | Pit now, fuel critical, tire failure | Interrupt medium/low |
| **Medium** | 2 | Pit window, gap changes, lap summary | Queue normally |
| **Low** | 3 | Sector times, minor updates | Skip if busy |

#### 4.4.2 Event Detection Logic

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

### 4.5 Interrupt Handling

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
        self.currently_speaking = False
        self.pending_query: Optional[str] = None
        self.current_sentence_complete = asyncio.Event()
    
    async def handle_driver_input(self, audio: bytes) -> None:
        text = await self.stt_client.transcribe(audio)
        
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
            audio = await self.tts_client.synthesize(sentence)
            await self.stream_audio(audio)
            
            # Signal sentence complete for interrupt handling
            self.current_sentence_complete.set()
            self.current_sentence_complete.clear()
            
            # Check for pending query after each sentence
            if self.pending_query:
                break
        
        self.currently_speaking = False
```

### 4.6 Session Context (Live)

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

### 4.7 Prompts (Live Mode)

#### 4.7.1 System Prompt

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

#### 4.7.2 Proactive Trigger Prompt

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

#### 4.7.3 Driver Query Prompt

```python
DRIVER_QUERY_PROMPT = """DRIVER ASKED: "{query}"

Answer the driver's question using the current session data. Be concise but complete.

If the question involves strategy, provide your recommendation with reasoning.
If you need to reference data, use specific numbers.

Your radio response:"""
```

---

## 5. Jarvis-Granite Analysis (Post-Race)

### 5.1 Overview

jarvis-granite-analysis provides comprehensive post-race analysis via REST API. Analysis runs **on-demand** when users explicitly request it, allowing for thorough LLM-powered insights without blocking the UI.

### 5.2 Architecture

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
│   │                    ORCHESTRATOR                         │   │
│   │                                                         │   │
│   │  • Request validation                                   │   │
│   │  • Context assembly                                     │   │
│   │  • Response formatting                                  │   │
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

### 5.3 Data Flow

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
                                    Orchestrator receives request
                                            │
                                            ▼
                                    Validates and assembles context
                                            │
                                            ▼
                                    Race Engineer Agent
                                    (Granite LLM - detailed prompt)
                                            │
                                            ▼
                                    Comprehensive analysis generated
                                            │
    ◀───────────────────────────────────────┘
    │
    ▼
Platform displays analysis to user
```

### 5.4 API Endpoints

#### 5.4.1 Session Summary

**Endpoint:** `POST /analysis/summary`

**Purpose:** Generate a high-level summary of an entire session.

**Request:**
```json
{
  "session_id": "race_001",
  "session_metadata": {
    "track_name": "Monza",
    "date": "2026-01-15",
    "conditions": "dry",
    "total_laps": 25
  },
  "lap_summaries": [
    {
      "lap_number": 1,
      "lap_time": 92.456,
      "sector_times": [28.123, 35.891, 28.442],
      "avg_speed_kmh": 215.4,
      "fuel_used": 2.1,
      "tire_wear_delta": {"fl": 2.1, "fr": 2.3, "rl": 1.8, "rr": 1.9},
      "position": 5,
      "incidents": []
    }
    // ... more laps
  ],
  "final_state": {
    "position": 3,
    "total_time": "38:42.156",
    "best_lap": 89.234,
    "fuel_remaining": 12.5
  }
}
```

**Response:**
```json
{
  "analysis_id": "analysis_abc123",
  "type": "session_summary",
  "summary": {
    "overall_assessment": "Strong performance with consistent improvement...",
    "key_highlights": [
      "Best lap of 89.234s on lap 18",
      "Gained 2 positions over the session",
      "Excellent tire management in final stint"
    ],
    "areas_for_improvement": [
      "Sector 2 times inconsistent - braking points varied",
      "Fuel consumption higher than optimal in opening laps"
    ],
    "statistics": {
      "avg_lap_time": 90.125,
      "consistency_rating": 0.87,
      "positions_gained": 2
    }
  },
  "metadata": {
    "processing_time_ms": 3420,
    "model_used": "granite-3.1-8b-instruct",
    "tokens_used": 2156
  }
}
```

#### 5.4.2 Lap Analysis

**Endpoint:** `POST /analysis/lap/{lap_number}`

**Purpose:** Deep analysis of a specific lap.

**Request:**
```json
{
  "session_id": "race_001",
  "lap_number": 5,
  "telemetry": {
    "samples": [
      {
        "timestamp": "2026-01-15T14:32:05.100Z",
        "distance": 0,
        "speed_kmh": 45.2,
        "throttle": 0.95,
        "brake": 0.0,
        "steering": 0.02,
        "rpm": 8500,
        "gear": 2,
        "g_lat": 0.1,
        "g_lon": 0.8
      }
      // ... telemetry at ~10-100Hz
    ],
    "lap_time": 90.234,
    "sector_times": [27.891, 35.123, 27.220]
  },
  "context": {
    "best_lap_telemetry": { /* optional: for comparison */ },
    "track_reference": { /* optional: ideal line data */ }
  }
}
```

**Response:**
```json
{
  "analysis_id": "analysis_def456",
  "type": "lap_analysis",
  "lap_number": 5,
  "analysis": {
    "overall_assessment": "Solid lap with room for improvement in sector 2...",
    "sector_breakdown": [
      {
        "sector": 1,
        "time": 27.891,
        "assessment": "Good entry speed, clean exit",
        "delta_to_best": "+0.15s"
      },
      {
        "sector": 2,
        "time": 35.123,
        "assessment": "Braking 8m too early into turn 4, lost momentum",
        "delta_to_best": "+0.42s"
      },
      {
        "sector": 3,
        "time": 27.220,
        "assessment": "Excellent final chicane, best sector of the lap",
        "delta_to_best": "-0.08s"
      }
    ],
    "corner_analysis": [
      {
        "corner": "Turn 4 (Variante della Roggia)",
        "issue": "Early braking",
        "suggestion": "Brake marker at 75m board instead of 85m",
        "time_loss": "0.3s"
      }
    ],
    "tire_usage": "Rear tires sliding slightly on exit of slow corners",
    "fuel_efficiency": "Consumption normal for this lap type"
  },
  "metadata": {
    "processing_time_ms": 4521,
    "model_used": "granite-3.1-8b-instruct",
    "tokens_used": 3842
  }
}
```

#### 5.4.3 Lap Comparison

**Endpoint:** `POST /analysis/compare`

**Purpose:** Compare two or more laps to identify differences.

**Request:**
```json
{
  "session_id": "race_001",
  "comparison_type": "laps",
  "items": [
    {
      "label": "Lap 5",
      "lap_number": 5,
      "telemetry": { /* full telemetry */ }
    },
    {
      "label": "Lap 18 (Best)",
      "lap_number": 18,
      "telemetry": { /* full telemetry */ }
    }
  ]
}
```

**Response:**
```json
{
  "analysis_id": "analysis_ghi789",
  "type": "comparison",
  "comparison": {
    "summary": "Lap 18 was 1.0s faster primarily due to improved sector 2...",
    "time_delta": {
      "total": -1.0,
      "sector_1": -0.15,
      "sector_2": -0.72,
      "sector_3": -0.13
    },
    "key_differences": [
      {
        "location": "Turn 4 entry",
        "lap_5": "Braking at 85m, minimum speed 62 km/h",
        "lap_18": "Braking at 75m, minimum speed 58 km/h",
        "impact": "Later braking allowed higher entry speed, better exit"
      },
      {
        "location": "Turn 7 exit",
        "lap_5": "75% throttle due to wheelspin",
        "lap_18": "95% throttle with smooth application",
        "impact": "Better traction = faster acceleration"
      }
    ],
    "recommendations": [
      "Focus on replicating Lap 18 braking points",
      "Smoother throttle application on corner exit"
    ]
  },
  "metadata": {
    "processing_time_ms": 5123,
    "model_used": "granite-3.1-8b-instruct",
    "tokens_used": 4521
  }
}
```

#### 5.4.4 Natural Language Query

**Endpoint:** `POST /analysis/query`

**Purpose:** Answer specific questions about session data.

**Request:**
```json
{
  "session_id": "race_001",
  "query": "Why was my pace slower in the middle stint compared to the opening laps?",
  "context": {
    "lap_range": [8, 16],
    "telemetry_summary": { /* aggregated data for context */ }
  }
}
```

**Response:**
```json
{
  "analysis_id": "analysis_jkl012",
  "type": "query_response",
  "query": "Why was my pace slower in the middle stint compared to the opening laps?",
  "response": {
    "answer": "Your middle stint (laps 8-16) averaged 91.2s compared to 89.8s in the opening stint. The primary factors were:\n\n1. **Tire degradation**: Rear tire temps increased from 92°C to 104°C, causing oversteer and requiring earlier braking.\n\n2. **Fuel load**: Higher fuel weight in opening laps actually helped rear traction. As fuel burned off, the rear became lighter and more prone to sliding.\n\n3. **Traffic**: Laps 11-13 involved lapping slower cars, costing approximately 0.5-0.8s per lap.\n\nThe pace recovered in the final stint after your pit stop provided fresh tires.",
    "confidence": 0.89,
    "data_references": [
      "Tire temp data laps 8-16",
      "Fuel consumption curve",
      "Gap data showing traffic"
    ]
  },
  "metadata": {
    "processing_time_ms": 2891,
    "model_used": "granite-3.1-8b-instruct",
    "tokens_used": 2103
  }
}
```

#### 5.4.5 Improvement Recommendations

**Endpoint:** `POST /analysis/improvements`

**Purpose:** Generate specific, actionable improvement recommendations.

**Request:**
```json
{
  "session_id": "race_001",
  "focus_areas": ["braking", "tire_management"],
  "session_data": {
    "lap_summaries": [ /* ... */ ],
    "telemetry_highlights": { /* key moments */ }
  }
}
```

**Response:**
```json
{
  "analysis_id": "analysis_mno345",
  "type": "improvements",
  "recommendations": [
    {
      "area": "Braking",
      "priority": "high",
      "issue": "Inconsistent braking points at Turn 4",
      "current_behavior": "Braking distance varies by 15m between laps",
      "recommendation": "Use the 75m board as a consistent reference",
      "expected_gain": "0.3-0.5s per lap",
      "practice_drill": "Focus next session on hitting exact braking points"
    },
    {
      "area": "Tire Management",
      "priority": "medium",
      "issue": "Rear tire overheating in long stints",
      "current_behavior": "Aggressive throttle application causing wheelspin",
      "recommendation": "Progressive throttle in slow corners, especially turns 4 and 7",
      "expected_gain": "Extend tire life by 3-5 laps",
      "practice_drill": "Practice slow corner exits with smooth throttle"
    }
  ],
  "overall_focus": "Consistency is your biggest opportunity. Your best laps show the pace is there - focus on replicating them.",
  "metadata": {
    "processing_time_ms": 3654,
    "model_used": "granite-3.1-8b-instruct",
    "tokens_used": 2987
  }
}
```

### 5.5 Prompts (Analysis Mode)

#### 5.5.1 System Prompt

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

#### 5.5.2 Lap Analysis Prompt

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

## 6. Shared Components

### 6.1 LLM Client

The LLM client abstracts the connection to IBM Granite, allowing for backend swapping during development.

```python
# llm/base.py
from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel

class LLMResponse(BaseModel):
    text: str
    tokens_used: int
    model: str
    latency_ms: float

class BaseLLMClient(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> LLMResponse:
        pass
```

```python
# llm/granite_client.py
import httpx
import time
from .base import BaseLLMClient, LLMResponse

class GraniteClient(BaseLLMClient):
    def __init__(self, api_key: str, project_id: str):
        self.api_key = api_key
        self.project_id = project_id
        self.base_url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation"
        self.model_id = "ibm/granite-3-8b-instruct"
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> LLMResponse:
        start_time = time.time()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model_id": self.model_id,
                    "project_id": self.project_id,
                    "messages": messages,
                    "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": temperature
                    }
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LLMResponse(
            text=data["results"][0]["generated_text"],
            tokens_used=data["results"][0]["generated_token_count"],
            model=self.model_id,
            latency_ms=latency_ms
        )
```

```python
# llm/huggingface_client.py (for development)
from huggingface_hub import InferenceClient
from .base import BaseLLMClient, LLMResponse

class HuggingFaceClient(BaseLLMClient):
    """Development client using HuggingFace Inference API"""
    
    def __init__(self, token: str, model: str = "ibm-granite/granite-3.0-8b-instruct"):
        self.client = InferenceClient(model=model, token=token)
        self.model = model
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> LLMResponse:
        start_time = time.time()
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        response = self.client.text_generation(
            full_prompt,
            max_new_tokens=max_tokens,
            temperature=temperature
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LLMResponse(
            text=response,
            tokens_used=len(response.split()) * 1.3,  # Rough estimate
            model=self.model,
            latency_ms=latency_ms
        )
```

### 6.2 Voice Services

```python
# tts/watson_tts.py
import httpx
from typing import Optional

class WatsonTTSClient:
    def __init__(self, api_key: str, service_url: str):
        self.api_key = api_key
        self.service_url = service_url
        self.voice = "en-GB_JamesV3Voice"  # British male voice for race engineer feel
    
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None
    ) -> bytes:
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
# stt/watson_stt.py
import httpx

class WatsonSTTClient:
    def __init__(self, api_key: str, service_url: str):
        self.api_key = api_key
        self.service_url = service_url
    
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

### 6.3 Logging Service

```python
# utils/logging.py
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel

class AIInteractionLog(BaseModel):
    timestamp: datetime
    session_id: str
    interaction_type: str  # "proactive", "reactive", "analysis"
    trigger: Optional[str]
    input_context: Dict[str, Any]
    prompt: str
    response: str
    model: str
    tokens_used: int
    latency_ms: float
    priority: Optional[str]

class AIInteractionLogger:
    def __init__(self, log_dir: str = "logs/ai_interactions"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def log(self, interaction: AIInteractionLog) -> None:
        date_str = interaction.timestamp.strftime("%Y-%m-%d")
        log_file = self.log_dir / f"{date_str}.jsonl"
        
        with open(log_file, "a") as f:
            f.write(interaction.model_dump_json() + "\n")
    
    def get_logs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        session_id: Optional[str] = None
    ) -> list[AIInteractionLog]:
        logs = []
        
        for log_file in sorted(self.log_dir.glob("*.jsonl")):
            with open(log_file) as f:
                for line in f:
                    log = AIInteractionLog.model_validate_json(line)
                    
                    if start_date and log.timestamp < start_date:
                        continue
                    if end_date and log.timestamp > end_date:
                        continue
                    if session_id and log.session_id != session_id:
                        continue
                    
                    logs.append(log)
        
        return logs
```

---

## 7. API Reference

### 7.1 Jarvis-Granite-Live WebSocket API

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
   │◀─── AI Responses (when triggered)─│
   │                                   │
```

#### Message Types (Client → Server)

**Session Initialization:**
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

**Telemetry:**
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

**Voice Input:**
```json
{
  "type": "voice_input",
  "timestamp": "2026-01-15T14:32:10.456Z",
  "audio_base64": "UklGRiQAAABXQVZF...",
  "audio_format": "wav"
}
```

**Text Query (Alternative to Voice):**
```json
{
  "type": "text_query",
  "timestamp": "2026-01-15T14:32:10.456Z",
  "query": "How are my tires looking?"
}
```

**Configuration Update:**
```json
{
  "type": "config_update",
  "config": {
    "verbosity": "verbose"
  }
}
```

**Session End:**
```json
{
  "type": "session_end"
}
```

#### Message Types (Server → Client)

**Session Confirmed:**
```json
{
  "type": "session_confirmed",
  "session_id": "race_001",
  "config": { /* active config */ }
}
```

**AI Response (Voice):**
```json
{
  "type": "ai_response",
  "response_id": "resp_abc123",
  "timestamp": "2026-01-15T14:32:11.789Z",
  "trigger": "pit_window_open",
  "text": "Pit window is open. Box this lap for fresh mediums.",
  "audio_base64": "UklGRiQAAABXQVZF...",
  "audio_format": "wav",
  "priority": "high",
  "metadata": {
    "latency_ms": 1850,
    "tokens_used": 45
  }
}
```

**AI Response (Text Only):**
```json
{
  "type": "ai_response",
  "response_id": "resp_def456",
  "timestamp": "2026-01-15T14:32:12.123Z",
  "trigger": "driver_query",
  "text": "Rear tires at 94 degrees, manageable for another 8 laps.",
  "audio_base64": null,
  "priority": "medium",
  "metadata": {
    "latency_ms": 1200,
    "tokens_used": 38
  }
}
```

**Error:**
```json
{
  "type": "error",
  "error_code": "LLM_TIMEOUT",
  "message": "AI response timed out, please try again",
  "timestamp": "2026-01-15T14:32:15.000Z"
}
```

**Heartbeat:**
```json
{
  "type": "heartbeat",
  "timestamp": "2026-01-15T14:32:30.000Z",
  "session_active": true
}
```

### 7.2 Jarvis-Granite-Analysis REST API

#### Base URL

`http://host:port/api/v1`

#### Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analysis/summary` | Generate session summary |
| POST | `/analysis/lap/{lap_number}` | Analyze specific lap |
| POST | `/analysis/compare` | Compare laps |
| POST | `/analysis/query` | Natural language query |
| POST | `/analysis/improvements` | Get improvement recommendations |
| GET | `/health` | Health check |
| GET | `/config` | Get current configuration |
| PUT | `/config` | Update configuration |

#### Common Response Format

All analysis endpoints return:

```json
{
  "analysis_id": "string",
  "type": "string",
  "/* type-specific data */": {},
  "metadata": {
    "processing_time_ms": 0,
    "model_used": "string",
    "tokens_used": 0
  }
}
```

#### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Detailed error message",
    "details": {}
  }
}
```

---

## 8. Data Schemas

### 8.1 Telemetry Schema

```python
# schemas/telemetry.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, List

class TireTemps(BaseModel):
    fl: float = Field(..., description="Front left tire temperature (Celsius)")
    fr: float = Field(..., description="Front right tire temperature (Celsius)")
    rl: float = Field(..., description="Rear left tire temperature (Celsius)")
    rr: float = Field(..., description="Rear right tire temperature (Celsius)")

class TireWear(BaseModel):
    fl: float = Field(..., ge=0, le=100, description="Front left tire wear percentage")
    fr: float = Field(..., ge=0, le=100, description="Front right tire wear percentage")
    rl: float = Field(..., ge=0, le=100, description="Rear left tire wear percentage")
    rr: float = Field(..., ge=0, le=100, description="Rear right tire wear percentage")

class GForces(BaseModel):
    lateral: float = Field(..., description="Lateral G-force")
    longitudinal: float = Field(..., description="Longitudinal G-force")

class TelemetryData(BaseModel):
    speed_kmh: float = Field(..., ge=0, description="Current speed in km/h")
    rpm: int = Field(..., ge=0, description="Engine RPM")
    gear: int = Field(..., ge=0, le=8, description="Current gear (0=neutral)")
    throttle: float = Field(..., ge=0, le=1, description="Throttle position (0-1)")
    brake: float = Field(..., ge=0, le=1, description="Brake pressure (0-1)")
    steering_angle: float = Field(..., ge=-1, le=1, description="Steering angle (-1 to 1)")
    fuel_remaining: float = Field(..., ge=0, description="Fuel remaining in liters")
    tire_temps: TireTemps
    tire_wear: TireWear
    g_forces: GForces
    track_position: float = Field(..., ge=0, le=1, description="Position on track (0-1)")
    lap_number: int = Field(..., ge=0, description="Current lap number")
    lap_time_current: float = Field(..., ge=0, description="Current lap time in seconds")
    sector: int = Field(..., ge=1, le=3, description="Current sector")
    position: Optional[int] = Field(None, ge=1, description="Race position")
    gap_ahead: Optional[float] = Field(None, description="Gap to car ahead in seconds")
    gap_behind: Optional[float] = Field(None, description="Gap to car behind in seconds")

class TelemetryMessage(BaseModel):
    type: str = "telemetry"
    timestamp: datetime
    session_id: str
    source: str
    data: TelemetryData
```

### 8.2 Analysis Request Schemas

```python
# schemas/analysis.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class LapSummary(BaseModel):
    lap_number: int
    lap_time: float
    sector_times: List[float]
    avg_speed_kmh: float
    fuel_used: float
    tire_wear_delta: Dict[str, float]
    position: Optional[int]
    incidents: List[str] = []

class SessionMetadata(BaseModel):
    track_name: str
    date: str
    conditions: str
    total_laps: int

class SessionSummaryRequest(BaseModel):
    session_id: str
    session_metadata: SessionMetadata
    lap_summaries: List[LapSummary]
    final_state: Dict[str, Any]

class LapTelemetry(BaseModel):
    samples: List[Dict[str, Any]]
    lap_time: float
    sector_times: List[float]

class LapAnalysisRequest(BaseModel):
    session_id: str
    lap_number: int
    telemetry: LapTelemetry
    context: Optional[Dict[str, Any]] = None

class ComparisonItem(BaseModel):
    label: str
    lap_number: int
    telemetry: LapTelemetry

class CompareRequest(BaseModel):
    session_id: str
    comparison_type: str = "laps"
    items: List[ComparisonItem]

class QueryRequest(BaseModel):
    session_id: str
    query: str
    context: Optional[Dict[str, Any]] = None

class ImprovementsRequest(BaseModel):
    session_id: str
    focus_areas: Optional[List[str]] = None
    session_data: Dict[str, Any]
```

### 8.3 Response Schemas

```python
# schemas/responses.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class AnalysisMetadata(BaseModel):
    processing_time_ms: float
    model_used: str
    tokens_used: int

class SessionSummaryResponse(BaseModel):
    analysis_id: str
    type: str = "session_summary"
    summary: Dict[str, Any]
    metadata: AnalysisMetadata

class LapAnalysisResponse(BaseModel):
    analysis_id: str
    type: str = "lap_analysis"
    lap_number: int
    analysis: Dict[str, Any]
    metadata: AnalysisMetadata

class CompareResponse(BaseModel):
    analysis_id: str
    type: str = "comparison"
    comparison: Dict[str, Any]
    metadata: AnalysisMetadata

class QueryResponse(BaseModel):
    analysis_id: str
    type: str = "query_response"
    query: str
    response: Dict[str, Any]
    metadata: AnalysisMetadata

class ImprovementsResponse(BaseModel):
    analysis_id: str
    type: str = "improvements"
    recommendations: List[Dict[str, Any]]
    overall_focus: str
    metadata: AnalysisMetadata
```

---

## 9. Configuration

### 9.1 Configuration Schema

```python
# config.py
from pydantic import BaseModel, Field
from typing import Literal, Optional

class LLMConfig(BaseModel):
    provider: Literal["watsonx", "huggingface"] = "watsonx"
    model_id: str = "ibm/granite-3-8b-instruct"
    max_tokens_live: int = 150
    max_tokens_analysis: int = 2000
    temperature_live: float = 0.7
    temperature_analysis: float = 0.5
    timeout_seconds: float = 10.0

class VoiceConfig(BaseModel):
    tts_voice: str = "en-GB_JamesV3Voice"
    stt_model: str = "en-GB_BroadbandModel"
    enable_voice: bool = True

class ThresholdsConfig(BaseModel):
    tire_temp_warning: float = 100.0
    tire_temp_critical: float = 110.0
    tire_wear_warning: float = 70.0
    tire_wear_critical: float = 85.0
    fuel_warning_laps: int = 5
    fuel_critical_laps: int = 2
    gap_change_threshold: float = 1.0

class VerbosityConfig(BaseModel):
    level: Literal["minimal", "moderate", "verbose"] = "moderate"
    announce_lap_times: bool = True
    announce_gap_changes: bool = True
    announce_tire_status: bool = True
    announce_fuel_status: bool = True
    proactive_coaching: bool = False

class LiveConfig(BaseModel):
    verbosity: VerbosityConfig = VerbosityConfig()
    thresholds: ThresholdsConfig = ThresholdsConfig()
    voice: VoiceConfig = VoiceConfig()
    telemetry_buffer_seconds: int = 60
    conversation_history_length: int = 3
    min_proactive_interval_seconds: float = 10.0

class AnalysisConfig(BaseModel):
    default_comparison_laps: int = 5
    include_telemetry_charts: bool = False
    max_recommendations: int = 5

class JarvisGraniteConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    live: LiveConfig = LiveConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    log_ai_interactions: bool = True
    log_directory: str = "logs"
```

### 9.2 Environment Variables

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

# HuggingFace (for development)
HUGGINGFACE_TOKEN=your_hf_token

# Application
LLM_PROVIDER=watsonx  # or "huggingface" for development
LOG_LEVEL=INFO
LOG_DIRECTORY=logs
```

### 9.3 Configuration Loading

```python
# config_loader.py
import os
from pathlib import Path
from dotenv import load_dotenv
from .config import JarvisGraniteConfig, LLMConfig

def load_config() -> JarvisGraniteConfig:
    load_dotenv()
    
    config = JarvisGraniteConfig()
    
    # Override from environment
    if os.getenv("LLM_PROVIDER"):
        config.llm.provider = os.getenv("LLM_PROVIDER")
    
    if os.getenv("LOG_DIRECTORY"):
        config.log_directory = os.getenv("LOG_DIRECTORY")
    
    return config

def get_llm_credentials() -> dict:
    provider = os.getenv("LLM_PROVIDER", "watsonx")
    
    if provider == "watsonx":
        return {
            "api_key": os.getenv("WATSONX_API_KEY"),
            "project_id": os.getenv("WATSONX_PROJECT_ID"),
            "url": os.getenv("WATSONX_URL")
        }
    elif provider == "huggingface":
        return {
            "token": os.getenv("HUGGINGFACE_TOKEN")
        }
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
```

---

## 10. Logging & Observability

### 10.1 Logging Strategy

| Log Type | Format | Location | Purpose |
|----------|--------|----------|---------|
| AI Interactions | JSON Lines | `logs/ai_interactions/YYYY-MM-DD.jsonl` | Fine-tuning dataset, model improvement |
| Application Logs | Structured JSON | `logs/app.log` | Debugging, monitoring |
| Error Logs | Structured JSON | `logs/errors.log` | Error tracking |
| Performance Metrics | JSON | `logs/metrics/YYYY-MM-DD.jsonl` | Latency analysis |

### 10.2 AI Interaction Log Schema

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

### 10.3 Using Logs for Fine-Tuning

The AI interaction logs are designed to be converted into fine-tuning datasets:

```python
# scripts/export_finetuning_data.py
import json
from pathlib import Path
from typing import List, Dict

def export_for_finetuning(
    log_dir: str,
    output_file: str,
    min_quality_score: float = 0.8
) -> None:
    """Export AI interaction logs as fine-tuning dataset"""
    
    examples = []
    log_path = Path(log_dir)
    
    for log_file in log_path.glob("*.jsonl"):
        with open(log_file) as f:
            for line in f:
                interaction = json.loads(line)
                
                # Convert to fine-tuning format
                example = {
                    "messages": [
                        {"role": "system", "content": get_system_prompt(interaction)},
                        {"role": "user", "content": interaction["prompt"]},
                        {"role": "assistant", "content": interaction["response"]}
                    ],
                    "metadata": {
                        "source": "jarvis-granite",
                        "interaction_type": interaction["interaction_type"],
                        "session_id": interaction["session_id"]
                    }
                }
                examples.append(example)
    
    with open(output_file, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"Exported {len(examples)} examples to {output_file}")
```

---

## 11. Deployment

### 11.1 IBM Cloud Deployment

#### 11.1.1 Prerequisites

- IBM Cloud account with Code Engine access
- IBM watsonx.ai project
- IBM Watson Speech services provisioned
- Docker installed locally

#### 11.1.2 Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY jarvis_granite/ ./jarvis_granite/
COPY config/ ./config/

# Create log directory
RUN mkdir -p /app/logs

# Environment
ENV PYTHONPATH=/app
ENV LOG_DIRECTORY=/app/logs

# Expose ports
EXPOSE 8000

# Run
CMD ["uvicorn", "jarvis_granite.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 11.1.3 Requirements

```
# requirements.txt
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
websockets>=12.0
pydantic>=2.5.0
httpx>=0.25.0
python-dotenv>=1.0.0
ibm-watson>=8.0.0
ibm-cloud-sdk-core>=3.18.0
```

#### 11.1.4 IBM Cloud Code Engine Deployment

```bash
# Login to IBM Cloud
ibmcloud login

# Target your resource group
ibmcloud target -g your-resource-group

# Create Code Engine project
ibmcloud ce project create --name jarvis-granite

# Select the project
ibmcloud ce project select --name jarvis-granite

# Build and deploy
ibmcloud ce build create --name jarvis-granite-build \
  --source . \
  --strategy dockerfile

ibmcloud ce application create --name jarvis-granite-live \
  --build jarvis-granite-build \
  --port 8000 \
  --min-scale 1 \
  --max-scale 3 \
  --env-from-secret jarvis-secrets

# Create secrets for credentials
ibmcloud ce secret create --name jarvis-secrets \
  --from-literal WATSONX_API_KEY=xxx \
  --from-literal WATSONX_PROJECT_ID=xxx \
  --from-literal WATSON_TTS_API_KEY=xxx \
  --from-literal WATSON_STT_API_KEY=xxx
```

### 11.2 Local Development

```bash
# Clone repository
git clone https://github.com/your-org/jarvis-granite.git
cd jarvis-granite

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Copy environment template
cp .env.example .env
# Edit .env with your credentials

# Run live service
uvicorn jarvis_granite.live.main:app --reload --port 8001

# Run analysis service (separate terminal)
uvicorn jarvis_granite.analysis.main:app --reload --port 8002
```

---

## 12. Development Guide

### 12.1 Project Structure

```
jarvis-granite/
├── jarvis_granite/
│   ├── __init__.py
│   ├── live/                      # Live mode implementation
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI app for live mode
│   │   ├── orchestrator.py        # Live orchestrator
│   │   ├── websocket_handler.py   # WebSocket management
│   │   └── context.py             # LiveSessionContext
│   │
│   ├── analysis/                  # Analysis mode implementation
│   │   ├── __init__.py
│   │   ├── main.py                # FastAPI app for analysis mode
│   │   ├── orchestrator.py        # Analysis orchestrator
│   │   └── endpoints.py           # REST endpoint handlers
│   │
│   ├── agents/                    # Shared agents
│   │   ├── __init__.py
│   │   ├── base.py                # Base agent class
│   │   ├── telemetry_agent.py     # Telemetry processing
│   │   ├── race_engineer_agent.py # LLM-powered insights
│   │   └── voice_agent.py         # TTS/STT handling
│   │
│   ├── llm/                       # LLM client abstraction
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract base client
│   │   ├── granite_client.py      # IBM watsonx.ai client
│   │   └── huggingface_client.py  # HuggingFace client (dev)
│   │
│   ├── voice/                     # Voice services
│   │   ├── __init__.py
│   │   ├── watson_tts.py          # Text-to-speech
│   │   └── watson_stt.py          # Speech-to-text
│   │
│   ├── schemas/                   # Pydantic models
│   │   ├── __init__.py
│   │   ├── telemetry.py           # Telemetry schemas
│   │   ├── analysis.py            # Analysis request schemas
│   │   └── responses.py           # Response schemas
│   │
│   ├── prompts/                   # Prompt templates
│   │   ├── __init__.py
│   │   ├── live_prompts.py        # Live mode prompts
│   │   └── analysis_prompts.py    # Analysis mode prompts
│   │
│   └── utils/                     # Utilities
│       ├── __init__.py
│       ├── logging.py             # Logging service
│       └── helpers.py             # Common helpers
│
├── config/
│   ├── __init__.py
│   ├── config.py                  # Configuration models
│   └── config_loader.py           # Configuration loading
│
├── tests/
│   ├── __init__.py
│   ├── test_telemetry_agent.py
│   ├── test_race_engineer_agent.py
│   ├── test_live_orchestrator.py
│   ├── test_analysis_endpoints.py
│   └── fixtures/                  # Test data
│       ├── telemetry_samples.json
│       └── session_data.json
│
├── scripts/
│   ├── export_finetuning_data.py  # Export logs for fine-tuning
│   └── simulate_session.py        # Test session simulation
│
├── logs/                          # Log output directory
│   ├── ai_interactions/
│   └── app.log
│
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── README.md
└── pyproject.toml
```

### 12.2 Adding a New Event Type

To add a new event that triggers AI responses:

1. **Define the event in telemetry_agent.py:**

```python
# In TelemetryAgent.detect_events()
def detect_events(self, telemetry: TelemetryMessage, context: SessionContext) -> List[Event]:
    events = []
    
    # ... existing events ...
    
    # New event: Yellow flag detection
    if telemetry.data.yellow_flag and not context.yellow_flag_active:
        events.append(Event(
            type="yellow_flag",
            priority="critical",
            data={"sector": telemetry.data.sector}
        ))
    
    return events
```

2. **Add prompt handling in race_engineer_agent.py:**

```python
# In RaceEngineerAgent
EVENT_PROMPTS = {
    # ... existing prompts ...
    "yellow_flag": """
        YELLOW FLAG in sector {sector}.
        
        Advise the driver on:
        - Speed reduction required
        - Overtaking restrictions
        - Any strategic implications
        
        Be clear and urgent.
    """
}
```

3. **Update configuration if needed:**

```python
# In config.py
class ThresholdsConfig(BaseModel):
    # ... existing thresholds ...
    yellow_flag_response_delay: float = 0.0  # Immediate
```

### 12.3 Swapping LLM Backends

The LLM client abstraction allows easy backend swapping:

```python
# In your orchestrator or main.py
from jarvis_granite.llm import GraniteClient, HuggingFaceClient
from config import load_config, get_llm_credentials

config = load_config()
credentials = get_llm_credentials()

if config.llm.provider == "watsonx":
    llm_client = GraniteClient(
        api_key=credentials["api_key"],
        project_id=credentials["project_id"]
    )
elif config.llm.provider == "huggingface":
    llm_client = HuggingFaceClient(
        token=credentials["token"]
    )
```

### 12.4 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_telemetry_agent.py

# Run with coverage
pytest --cov=jarvis_granite --cov-report=html

# Run integration tests (requires credentials)
pytest tests/integration/ --integration
```

---

## 13. Integration Guide

### 13.1 Integrating with 2D Telemetry Platform

#### Live Mode Integration

```python
# Example: Python client for 2D platform
import asyncio
import websockets
import json

class JarvisLiveClient:
    def __init__(self, url: str = "ws://localhost:8001/live"):
        self.url = url
        self.websocket = None
    
    async def connect(self, session_config: dict):
        self.websocket = await websockets.connect(self.url)
        
        # Initialize session
        await self.websocket.send(json.dumps({
            "type": "session_init",
            **session_config
        }))
        
        response = await self.websocket.recv()
        return json.loads(response)
    
    async def send_telemetry(self, telemetry: dict):
        await self.websocket.send(json.dumps({
            "type": "telemetry",
            **telemetry
        }))
    
    async def send_voice_query(self, audio_base64: str):
        await self.websocket.send(json.dumps({
            "type": "voice_input",
            "audio_base64": audio_base64,
            "audio_format": "wav"
        }))
    
    async def receive_responses(self):
        """Generator for receiving AI responses"""
        while True:
            message = await self.websocket.recv()
            yield json.loads(message)
    
    async def close(self):
        await self.websocket.send(json.dumps({"type": "session_end"}))
        await self.websocket.close()

# Usage
async def main():
    client = JarvisLiveClient()
    
    await client.connect({
        "session_id": "race_001",
        "source": "torcs",
        "track_name": "Monza",
        "config": {"verbosity": "moderate"}
    })
    
    # Start receiving responses in background
    async def handle_responses():
        async for response in client.receive_responses():
            if response["type"] == "ai_response":
                print(f"AI: {response['text']}")
                # Play audio if available
                if response.get("audio_base64"):
                    play_audio(response["audio_base64"])
    
    asyncio.create_task(handle_responses())
    
    # Send telemetry in your game loop
    while racing:
        telemetry = get_current_telemetry()
        await client.send_telemetry(telemetry)
        await asyncio.sleep(0.1)  # 10Hz
```

#### Analysis Mode Integration

```python
# Example: Python client for post-race analysis
import httpx

class JarvisAnalysisClient:
    def __init__(self, base_url: str = "http://localhost:8002/api/v1"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def analyze_session(self, session_data: dict) -> dict:
        response = await self.client.post(
            f"{self.base_url}/analysis/summary",
            json=session_data
        )
        response.raise_for_status()
        return response.json()
    
    async def analyze_lap(self, session_id: str, lap_number: int, telemetry: dict) -> dict:
        response = await self.client.post(
            f"{self.base_url}/analysis/lap/{lap_number}",
            json={
                "session_id": session_id,
                "lap_number": lap_number,
                "telemetry": telemetry
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def compare_laps(self, session_id: str, laps: list) -> dict:
        response = await self.client.post(
            f"{self.base_url}/analysis/compare",
            json={
                "session_id": session_id,
                "comparison_type": "laps",
                "items": laps
            }
        )
        response.raise_for_status()
        return response.json()
    
    async def ask_question(self, session_id: str, query: str, context: dict = None) -> dict:
        response = await self.client.post(
            f"{self.base_url}/analysis/query",
            json={
                "session_id": session_id,
                "query": query,
                "context": context
            }
        )
        response.raise_for_status()
        return response.json()

# Usage
async def analyze_race():
    client = JarvisAnalysisClient()
    
    # User clicks "Analyze Lap 5"
    lap_analysis = await client.analyze_lap(
        session_id="race_001",
        lap_number=5,
        telemetry=load_lap_telemetry(5)
    )
    display_analysis(lap_analysis)
    
    # User asks a question
    answer = await client.ask_question(
        session_id="race_001",
        query="Why was sector 2 slower than my best lap?"
    )
    display_answer(answer)
```

### 13.2 Integrating with VR Platform (Unreal Engine)

The VR platform uses REST API for post-race analysis. Example C++ integration:

```cpp
// JarvisAnalysisClient.h
#pragma once

#include "CoreMinimal.h"
#include "Http.h"

DECLARE_DELEGATE_OneParam(FOnAnalysisComplete, const FString& /* JsonResponse */);

class VRPLATFORM_API FJarvisAnalysisClient
{
public:
    FJarvisAnalysisClient(const FString& InBaseUrl);
    
    void AnalyzeLap(
        const FString& SessionId,
        int32 LapNumber,
        const FString& TelemetryJson,
        FOnAnalysisComplete OnComplete
    );
    
    void AskQuestion(
        const FString& SessionId,
        const FString& Query,
        FOnAnalysisComplete OnComplete
    );

private:
    FString BaseUrl;
    
    void SendRequest(
        const FString& Endpoint,
        const FString& JsonBody,
        FOnAnalysisComplete OnComplete
    );
};

// JarvisAnalysisClient.cpp
#include "JarvisAnalysisClient.h"

FJarvisAnalysisClient::FJarvisAnalysisClient(const FString& InBaseUrl)
    : BaseUrl(InBaseUrl)
{
}

void FJarvisAnalysisClient::AnalyzeLap(
    const FString& SessionId,
    int32 LapNumber,
    const FString& TelemetryJson,
    FOnAnalysisComplete OnComplete)
{
    FString Endpoint = FString::Printf(TEXT("/analysis/lap/%d"), LapNumber);
    
    TSharedPtr<FJsonObject> RequestBody = MakeShareable(new FJsonObject);
    RequestBody->SetStringField("session_id", SessionId);
    RequestBody->SetNumberField("lap_number", LapNumber);
    // Add telemetry...
    
    FString JsonBody;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&JsonBody);
    FJsonSerializer::Serialize(RequestBody.ToSharedRef(), Writer);
    
    SendRequest(Endpoint, JsonBody, OnComplete);
}

void FJarvisAnalysisClient::SendRequest(
    const FString& Endpoint,
    const FString& JsonBody,
    FOnAnalysisComplete OnComplete)
{
    TSharedRef<IHttpRequest> Request = FHttpModule::Get().CreateRequest();
    Request->SetURL(BaseUrl + Endpoint);
    Request->SetVerb("POST");
    Request->SetHeader("Content-Type", "application/json");
    Request->SetContentAsString(JsonBody);
    
    Request->OnProcessRequestComplete().BindLambda(
        [OnComplete](FHttpRequestPtr Request, FHttpResponsePtr Response, bool bSuccess)
        {
            if (bSuccess && Response.IsValid())
            {
                OnComplete.ExecuteIfBound(Response->GetContentAsString());
            }
        }
    );
    
    Request->ProcessRequest();
}
```

---

## 14. Appendices

### 14.1 Glossary

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

### 14.2 Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `LLM_TIMEOUT` | Granite API request timed out | Retry, check network |
| `LLM_ERROR` | Granite API returned error | Check credentials, quota |
| `TTS_ERROR` | Text-to-speech failed | Check Watson TTS service |
| `STT_ERROR` | Speech-to-text failed | Check audio format, Watson STT service |
| `VALIDATION_ERROR` | Invalid request data | Check request schema |
| `SESSION_NOT_FOUND` | Referenced session doesn't exist | Initialize session first |
| `CONFIG_ERROR` | Invalid configuration | Check config values |

### 14.3 Performance Targets

| Metric | Live Mode Target | Analysis Mode Target |
|--------|------------------|---------------------|
| Telemetry processing | <50ms | N/A |
| LLM response | <2000ms | <10000ms |
| TTS conversion | <500ms | N/A |
| End-to-end (voice) | <3000ms | N/A |
| API response | N/A | <15000ms |

### 14.4 Future Considerations

1. **Fine-tuned Models**: Once Granite is fine-tuned on motorsport data, update `model_id` in configuration
2. **Multi-session Support**: Current design supports single session; scaling requires session management layer
3. **Granite TTS**: If IBM releases Granite TTS, swap Watson TTS client
4. **Plugin Architecture**: Clean module boundaries allow future plugin system if needed
5. **Caching Layer**: Add Redis for response caching if latency becomes an issue

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | December 2025 | Team 17 | Initial documentation |

---

*This document is part of the F1 Jarvis TORCS project. For questions or contributions, contact the development team.*