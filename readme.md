# Jarvis Granite

An AI-enhanced motorsport telemetry analysis and live strategist platform designed to act as a real-time race engineer, strategist, and post-race performance analyst and coach

---

## What It Does

### Jarvis Live — Real-Time Telemetry Dashboard

Drive with a live dashboard that updates as you race:

- **Track map** with speed-colored racing line, updating ~12 times/sec
- **Live telemetry graphs** — speed, RPM, gear, throttle, brake, tire temps (4 tires), tire pressures (4 tires)
- **Lap times table** with delta-to-best for every completed lap
- **Delta-to-best graph** — real-time green/red trace comparing your current lap against your session best
- **Session info** — track, car, driver, session type displayed at a glance
- **Automatic lap switching** — graphs and track map reset when you cross the start/finish line

### AI Race Engineer (Optional)

An AI co-driver that watches your telemetry and talks to you:

- **Proactive alerts** — fuel warnings, tire temperature/wear alerts, excessive wheel slip, gap changes, pit window suggestions
- **Voice interaction** — ask questions hands-free via push-to-talk or continuous listening mode (webrtcvad + faster-whisper, runs locally)
- **Text-to-speech responses** — Kokoro local TTS with configurable voice
- **Commentary transcript** — all AI messages displayed in a live panel

### Jarvis Post — Post-Race Analysis

Review and analyse completed sessions:

- **Lap review with timeline scrubber** — play/pause, seek, variable speed (0.25x–4x), zoom windows (15s–full lap)
- **Configurable graph layout** — choose up to 6 graphs at once from: speed, RPM, gear, fuel, throttle & brake, tire temps, tire pressures, tire wear, wheel slip, suspension travel, g-forces, ride height, car damage, steering angle
- **Track map replay** — car position marker moves along the speed-colored path in sync with the timeline
- **AI coaching** — dual-model analysis (coach + analyst) powered by Hugging Face, with detailed per-lap feedback on driving and telemetry
- **Export/import sessions** — share full sessions as `.jsession` files
- **Fullscreen mode** (F11)

### Session Management

- **Automatic recording** — all telemetry saved to SQLite automatically, no setup needed
- **Session browser** — view all sessions with track, car, player, duration, best lap, session type
- **Rename sessions** for organisation
- **Delete sessions** you no longer need
- **Export sessions** — raw telemetry, lap data, and AI commentary as CSV files

---

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the app
```bash
python main.py          # Launch Jarvis (launcher opens first)
python main.py --ai     # Launch with AI race engineer enabled
```

### Environment variables (for AI features)

Copy `.env.example` to `.env` and fill in:

```bash
# Hugging Face (AI race engineer)
HUGGINGFACE_TOKEN=hf_your_api_key_here
HUGGINGFACE_MODEL_ID=your_model_id

# Post-race AI (Hugging Face Space)
POSTRACE_HF_API_TOKEN=your_token
POSTRACE_HF_SPACE_URL=https://your-space.hf.space

# Kokoro TTS voice (optional)
KOKORO_VOICE=bm_lewis
```

AI features are optional — the app works without them for pure telemetry.

### Local LLM model files (GGUF)

- `*.gguf` files are intentionally **not tracked in Git**.
- Place your local model under `race_engineer_gguf/` (or any local path).
- Point the app at it via `LOCAL_MODEL_PATH` or `config.json`.
- Use `config.example.json` as the tracked template for local runtime config.
- Large training/sample artifacts are also local-only by default (`*.safetensors`, large sample telemetry CSVs).

---

## Architecture Overview

```
Game (AC)
    |
Telemetry Worker (QThread @ 60Hz)
    |--- realtime_sample --> Live UI + Session Recorder
    |--- lap_completed   --> Lap table + Session Recorder
    |--- live_data       --> Live stats panel
    |--- session_info    --> Session info panel
    |
AI Race Engineer (optional)
    |--- TelemetryAgent (rule-based event detection, <50ms)
    |--- RaceEngineerAgent (LLM response generation, ~2s)
    |--- ai_commentary --> UI transcript + Session Recorder
    |
Voice (optional)
    |--- webrtcvad + faster-whisper --> driver queries
    |--- Kokoro TTS --> spoken responses
    |
Session Recorder (automatic)
    |--- SQLite database (telemetry_sessions.db)
    |--- Batch inserts for performance
```

### Project Structure

```
jarvis-granite/
├── main.py                         # Entry point
├── ui/
│   ├── launcher.py                 # Launcher with settings
│   ├── main_window.py              # Jarvis Live dashboard
│   ├── session_picker.py           # Session browser
│   ├── canvases/                   # Matplotlib graph widgets
│   └── post_race/
│       └── lap_viewer_window.py    # Jarvis Post viewer
│
├── telemetry/
│   ├── lap_buffer.py               # Lap detection logic
│   └── backends/
│       ├── ac_backend.py           # Assetto Corsa (shared memory)
│
├── ai/
│   ├── race_engineer.py            # Live AI worker thread
│   ├── voice_input.py              # Speech-to-text (local)
│   ├── tts_output.py               # Text-to-speech (local)
│   └── race_engineer_core/         # AI core: models, agents, prompts, LLM client
│
├── analysis/
│   └── ai_pipeline_bridge.py       # Post-race AI analysis bridge
│
├── data/
│   ├── session_recorder.py         # SQLite recording
│   ├── session_exporter.py         # Export/import sessions
│   └── session_viewer.py           # CLI session viewer
│
└── CLAUDE.md                       # Detailed technical docs
```

---

## File Formats

| Format | Extension | Contents |
|--------|-----------|----------|
| Session file | `.jsession` | Full session: all laps, telemetry, AI commentary (JSON) |
| CSV export | `.csv` | Raw telemetry, lap summaries, AI commentary |
| Database | `.db` | SQLite with all recorded sessions |

---

## Telemetry Data Collected

When using Assetto Corsa, the following data is captured at ~60Hz:

- Position (X, Z coordinates), speed, gear, RPM
- Throttle, brake, steering angle
- Fuel remaining
- Tire temperatures (FL, FR, RL, RR)
- Tire pressures (FL, FR, RL, RR)
- Tire wear (FL, FR, RL, RR)
- Wheel slip (FL, FR, RL, RR)
- Suspension travel (FL, FR, RL, RR)
- Ride height (front, rear)
- G-forces (lateral, longitudinal)
- Car damage (front, rear, left, right, centre)

---

## AI Event Detection

The live AI race engineer monitors telemetry and triggers alerts:

| Event | Threshold | Priority |
|-------|-----------|----------|
| Fuel critical | < 2 laps remaining | CRITICAL |
| Fuel warning | < 5 laps remaining | HIGH |
| Tire temp critical | > 110 C | CRITICAL |
| Tire temp warning | > 100 C | MEDIUM |
| Tire wear critical | > 85% | HIGH |
| Wheel slip critical | > 10.0 | HIGH |
| Wheel slip warning | > 5.0 | MEDIUM |
| Gap change | > 1.0s | MEDIUM |
| Pit window | Fuel or tire at warning | HIGH |

All thresholds are configurable.

---

## Credits

**Team 17 — Systems Course Project**

- Kokoro — local text-to-speech
- webrtcvad — voice activity detection
- faster-whisper — speech-to-text
- Hugging Face — LLM inference
- Game APIs — AC shared memory

---

## License

Educational project for systems coursework.
