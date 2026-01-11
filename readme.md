# F1 Live Telemetry Dashboard

A real-time telemetry visualization tool for sim racing games with AI race engineer capabilities. Built with PyQt5 and IBM Watson AI services.

**Team 17 - Systems Course Project**

---

## Features

- 📊 **Real-time Telemetry** - Live track maps and performance graphs updating at 12Hz
- 🏎️ **Multi-Game Support** - Assetto Corsa (full telemetry) and ACC (limited)
- 🤖 **AI Race Engineer** - IBM WatsonX-powered voice assistant with proactive alerts
- 🎙️ **Voice Interaction** - Hands-free queries with speech-to-text/text-to-speech
- 💾 **Session Recording** - SQLite database recording all telemetry and AI interactions
- 📈 **Multi-Tire Analysis** - Separate graphs for FL/FR/RL/RR tire temps and pressures

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run with Assetto Corsa (default)
```bash
python main.py
```

### 3. Run with AI Race Engineer (experimental)
```bash
python main.py --ai
```

Combine with game selection:
```bash
python main.py --acc --ai    # ACC with AI features
```

**Prerequisites:**
- **AC:** Windows only, shared memory enabled in game settings
- **ACC:** `broadcasting.json` configured (see [SETUP_GUIDE.md](SETUP_GUIDE.md))
- **AI:** IBM Watson credentials in `.env` file (see `.env.example`)

---

## Project Structure

```
f1_telemetry_app/
├── main.py                     # Entry point
├── requirements.txt            # Dependencies
├── .env.example                # Environment template
│
├── ui/                         # PyQt5 interface
│   ├── main_window.py          # Main window
│   ├── styles.py               # Dark theme
│   └── canvases/               # Matplotlib graphs
│
├── telemetry/                  # Game backends
│   ├── lap_buffer.py           # Lap detection
│   └── backends/
│       ├── ac_backend.py       # AC shared memory
│       └── acc_backend.py      # ACC UDP
│
├── ai/                         # AI features
│   ├── race_engineer.py        # AI worker
│   ├── voice_input.py          # Voice queries
│   └── tts_output.py           # Voice output
│
├── eima_ai/                    # AI race engineer system
│   ├── config/                 # Config models
│   └── jarvis_granite/
│       ├── agents/             # Event detection + LLM
│       ├── llm/                # WatsonX client
│       ├── prompts/            # Prompt templates
│       ├── schemas/            # Data models
│       └── live/
│           └── context.py      # Session state
│
├── data/                       # Recording
│   ├── session_recorder.py     # SQLite recorder
│   ├── session_viewer.py       # CLI viewer
│   └── telemetry_sessions.db   # Database
│
├── CLAUDE.md                   # Technical docs
└── SETUP_GUIDE.md              # Setup instructions
```

---

## Environment Variables

Create `.env` file (copy from `.env.example`):

```bash
# IBM WatsonX (AI race engineer)
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_PROJECT_ID=your_project_id
WATSONX_API_KEY=your_api_key

# IBM Watson STT (voice input)
WATSON_STT_API_KEY=your_stt_key
WATSON_STT_URL=https://api.us-south.speech-to-text.watson.cloud.ibm.com

# IBM Watson TTS (voice output)
WATSON_TTS_API_KEY=your_tts_key
WATSON_TTS_URL=https://api.us-south.text-to-speech.watson.cloud.ibm.com
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed Watson setup.

---

## Session Recording

All telemetry is automatically recorded to `data/telemetry_sessions.db`:

**View recorded sessions:**
```bash
python -m data.session_viewer list               # List all
python -m data.session_viewer info <session_id>  # Details
python -m data.session_viewer laps <session_id>  # Lap times
python -m data.session_viewer export <session_id> # Export CSV
```

**Database tables:**
- `sessions` - Track, car, player, timestamps
- `laps` - Lap times, fuel usage, speed stats
- `telemetry` - High-frequency samples (~60Hz)
- `ai_commentary` - AI messages with timestamps
- `voice_queries` - Driver questions and responses

---

## How It Works

### Data Flow

```
Game (AC/ACC)
    ↓
Telemetry Worker (QThread @ 60Hz)
    ↓ emits signals:
    ├─→ realtime_sample  → UI + SessionRecorder
    ├─→ lap_completed    → UI lap table + SessionRecorder
    ├─→ session_info     → UI info panel
    └─→ live_data        → UI live stats

AI Race Engineer (optional)
    ↓
Receives telemetry
    ↓
TelemetryAgent detects events
(fuel warnings, tire issues, gap changes)
    ↓
RaceEngineerAgent generates response
(IBM WatsonX LLM call ~2s)
    ↓
ai_commentary signal → UI + SessionRecorder

Voice Input (optional)
    ↓
Silero VAD detects speech
    ↓
Watson STT transcribes
    ↓
driver_query signal → AI Race Engineer
```

### Core Components

**[telemetry/backends/ac_backend.py](telemetry/backends/ac_backend.py)**
- Reads AC's Windows shared memory (`acpmf_static`, `acpmf_physics`, `acpmf_graphics`)
- Full telemetry: RPM, throttle, brake, tire pressure/temp for all 4 tires
- Windows only

**[telemetry/backends/acc_backend.py](telemetry/backends/acc_backend.py)**
- Connects via UDP broadcasting protocol
- Limited telemetry: position, speed, gear, lap times (no RPM, brake, throttle, fuel)
- Cross-platform

**[ai/race_engineer.py](ai/race_engineer.py)**
- QThread integrating eima_ai system with telemetry
- Converts dict → Pydantic models → TelemetryAgent → RaceEngineerAgent
- Emits AI commentary via Qt signals

**[eima_ai/](eima_ai/)**
- Standalone AI race engineer system (Eima's project, integrated)
- Event detection (<50ms), LLM generation (~2000ms)
- Configuration-based thresholds and verbosity

**[data/session_recorder.py](data/session_recorder.py)**
- QThread recording all telemetry to SQLite
- Batch inserts (60 samples) for performance
- Runs automatically, no flags needed

---

## Game Setup

### Assetto Corsa (AC)

**Windows only** - Uses shared memory

1. **Enable in AC settings:**
   - Options → General → UI Modules
   - Shared Memory = ON
   - Shared Memory Layout = 1

2. **Start session and get on track**
   - Practice/hotlap/race mode
   - Wait for car to fully load

3. **Run as same user (no admin mismatch)**

### Assetto Corsa Competizione (ACC)

**Cross-platform** - Uses UDP broadcasting

1. **Configure broadcasting:**
   - Edit `Documents\Assetto Corsa Competizione\Config\broadcasting.json`:
   ```json
   {
     "updListenerPort": 9232,
     "connectionPassword": "",
     "commandPassword": ""
   }
   ```

2. **Start ACC BEFORE running script**

3. **Get on track** (broadcasting only active during sessions)

**Limitations:** No RPM, throttle, brake, or fuel data (broadcasting API)

---

## Troubleshooting

### AC Issues

**"Could not open shared memory":**
- AC must be running with session started
- Run Python as same user (no admin mismatch)
- Verify shared memory enabled in settings

### ACC Issues

**"Connection failed":**
- Start ACC BEFORE Python script
- Must be on track (not in menus)
- Check `broadcasting.json` port matches code (default 9232)
- Firewall may block UDP traffic

**"RPM/brake/throttle showing zeros":**
- Expected - ACC broadcasting doesn't provide this data
- Only position, speed, gear, lap timing available

### AI Issues

**"AI not responding":**
- Check `.env` has correct Watson credentials
- Verify internet connection (API calls require network)
- Check console for error messages

**"Microphone not working":**
- Check system microphone permissions
- Verify Watson STT credentials in `.env`
- Look for microphone indicator when speaking

---

## Known Limitations

- **AC:** Windows-only (shared memory limitation)
- **ACC:** Limited telemetry via broadcasting (no RPM, brake, throttle, fuel)
- **AI:** Requires internet (~2s latency for LLM responses)
- **UI:** No overlay mode (separate window)
- **Replay:** No playback UI (use CLI viewer)

---

## Credits

**Team 17 Systems Course Project**

- Telemetry dashboard: Team 17
- AI race engineer (`eima_ai/`): Eima (integrated)
- IBM Watson: WatsonX LLM, STT, TTS
- Silero VAD: Voice activity detection
- Game APIs: AC shared memory, ACC broadcasting

---

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Detailed technical architecture for AI assistant
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Step-by-step installation and Watson setup
- **[.env.example](.env.example)** - Environment variable template

---

## License

Educational project for systems coursework.
