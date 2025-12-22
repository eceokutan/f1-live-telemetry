# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

F1 Live Telemetry Dashboard for sim racing games. A PyQt5-based real-time telemetry visualization tool that displays lap data, track maps, and performance metrics.

**Currently supports:**
- **Assetto Corsa (AC)** - Full telemetry via Windows shared memory
- **Assetto Corsa Competizione (ACC)** - Limited telemetry via UDP broadcasting API

This is coursework for team 17 (systems course).

## Running the Application

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run with Assetto Corsa (default)
```bash
python integrated_telemetry.py
```

### Run with ACC
```bash
python integrated_telemetry.py --acc
```

### Run with AI Race Engineer (experimental)
```bash
python integrated_telemetry.py --ai
```

Combine with game selection:
```bash
python integrated_telemetry.py --acc --ai
```

**Prerequisites:**
- For AC: Game must be running with shared memory enabled (Windows only)
- For ACC: Game must be running with `broadcasting.json` configured and you must be on track
- For AI: Requires IBM WatsonX credentials in `.env` file (see `.env.example`)

## Architecture

The codebase follows a modular backend architecture where each sim game has its own telemetry worker:

### Core Components

**[integrated_telemetry.py](integrated_telemetry.py)** - Entry point
- Initializes PyQt5 application
- Selects backend based on command-line args (`--acc` flag)
- Connects backend signals to UI slots
- Manages application lifecycle

**[dashboard.py](dashboard.py)** - Main UI window (game-agnostic)
- `MainWindow` - PyQt5 main window with lap table, session info, and live data panels
- `TrackMapCanvas` - Matplotlib track visualization colored by speed
- `TimeSeriesCanvas` - Generic time-series graphs (speed, RPM, brake, gear)
- Receives lap data via Qt signals and updates all visualizations

**[telemetry/lap_buffer.py](telemetry/lap_buffer.py)** - Lap detection and buffering
- `LapBuffer` - Collects samples until lap completes (when `completedLaps` increments)
- Calls `on_lap_complete(lap_id, samples)` callback with buffered data
- Handles lap resets and backwards time jumps

### Backend Workers (QThread)

Both backends inherit from `QtCore.QThread` and emit these signals:
- `lap_completed(int lap_id, list samples)` - Emitted when a lap finishes
- `status_update(str message)` - Status messages for console logging
- `session_info_update(dict info)` - Session metadata (track, car, etc.)
- `live_data_update(dict data)` - Real-time telemetry for UI panels (current speed, gear, fuel, etc.)
- `realtime_sample(dict sample)` - Emitted every frame (~60Hz) with telemetry sample for live visualization

**[telemetry/ac_shared_memory.py](telemetry/ac_shared_memory.py)** - Assetto Corsa backend
- `AcTelemetryWorker` - Reads from AC's Windows named shared memory blocks:
  - `acpmf_static` - Static session info (track, car model, player name)
  - `acpmf_physics` - Physics data (speed, RPM, throttle, brake, gear, fuel, tire pressure, tire temperature)
  - `acpmf_graphics` - Graphics/session data (lap count, lap times, position X/Y/Z)
- Uses `ctypes.Structure` to map binary shared memory to Python objects
- Polls at ~60Hz and feeds `LapBuffer`
- **Full telemetry available:** RPM, throttle, brake, tire pressure (PSI), tire temperature (°C) for all 4 tires [FL, FR, RL, RR]
- **Windows only** - Requires AC running in same user session

**[telemetry/acc_backend.py](telemetry/acc_backend.py)** - ACC backend
- `AccTelemetryWorker` - Connects to ACC via UDP broadcasting protocol
- `AccPacketParser` - Parses binary UDP packets:
  - `REGISTRATION_RESULT` - Connection handshake
  - `REALTIME_UPDATE` - Session info
  - `REALTIME_CAR_UPDATE` - Car telemetry (position, speed, gear, lap times)
  - `TRACK_DATA` - Track name and metadata
  - `ENTRY_LIST` - Car/driver info
- Connection flow:
  1. Bind to port (default 9232)
  2. Send `REGISTER_COMMAND_APPLICATION` packet
  3. Wait for `REGISTRATION_RESULT`
  4. Request `TRACK_DATA` and `ENTRY_LIST`
  5. Continuously receive `REALTIME_CAR_UPDATE` packets
- **Limitation:** ACC broadcasting API does NOT provide RPM, throttle, brake, fuel, tire pressure, or tire temperature data
  - These fields will show zeros in the UI when using ACC backend
  - Only position (X/Y/Z), speed, gear, and lap timing are available via broadcasting API
  - For full telemetry, would need ACC's physics shared memory plugin (different API)

**[telemetry/ai_race_engineer.py](telemetry/ai_race_engineer.py)** - AI Race Engineer (experimental)
- `AIRaceEngineerWorker` - QThread that integrates Eima's AI race engineer with AC telemetry
- Uses IBM WatsonX (Granite 3-8B-Instruct) for LLM-powered race engineering commentary
- **Components integrated from eima_ai/**:
  - `TelemetryAgent` - Rule-based event detection (<50ms latency)
    - Detects fuel warnings/critical (< 5/2 laps remaining)
    - Detects tire temperature warnings/critical (> 100°C/110°C)
    - Detects gap changes (> 1.0s threshold)
    - Detects lap/sector completions
  - `RaceEngineerAgent` - LLM-powered response generation (~2000ms latency)
    - Generates contextual proactive alerts for detected events
    - Maintains conversation history (last 3 exchanges)
    - Respects configurable verbosity levels (minimal, moderate, verbose)
  - `LiveSessionContext` - In-memory session state
    - Maintains telemetry buffer (60s rolling window)
    - Tracks fuel consumption per lap
    - Manages active alerts and proactive message timing
- **Data flow**:
  1. AC telemetry → AIRaceEngineerWorker.process_telemetry()
  2. Convert dict to Pydantic TelemetryData model
  3. TelemetryAgent detects events (rule-based, <50ms)
  4. RaceEngineerAgent generates AI response (LLM call, ~2s)
  5. Emit ai_commentary signal → UI displays in "Commentator Transcript" panel
- **Environment variables required**: WATSONX_API_KEY, WATSONX_PROJECT_ID
- **Optional**: Works best with AC (full telemetry), degraded with ACC (limited telemetry)

### Data Flow

**Real-time Visualization (every frame ~60Hz):**
```
Game (AC/ACC)
    ↓
Telemetry Worker (QThread)
    ↓ (emits realtime_sample signal)
MainWindow.handle_realtime_sample(sample)
    ↓
Buffer current lap samples
    ↓ (every 5 samples ~12Hz)
Update track map, graphs in real-time
```

**Lap Completion (when lap finishes):**
```
Game (AC/ACC)
    ↓
Telemetry Worker (QThread)
    ↓
LapBuffer.add_sample(lap_id, t, x, z, speed, gear, rpm, brake, ...)
    ↓ (when lap_id increments)
LapBuffer.on_lap_complete(lap_id, samples)
    ↓ (Qt signal)
MainWindow.handle_lap_complete(lap_id, samples)
    ↓
Update lap table with lap time
(visualization already showing next lap in real-time)
```

## Key Implementation Details

### Real-time Visualization Architecture

The dashboard now updates in real-time as you drive:
- **Track map** - Shows current lap path colored by speed, updates ~12 times/sec
- **Time-series graphs** - Speed, gear, RPM, brake, tire pressure (4 tires), tire temperature (4 tires) update live as you drive
- **Multi-line tire graphs** - Each tire (FL, FR, RL, RR) is shown as a separate colored line on the same plot
- **Automatic lap switching** - When you cross start/finish, visualization automatically clears and starts showing the new lap
- **Throttled updates** - Only updates every 5 samples (from 60Hz telemetry) to avoid UI overload

### Adding a New Sim Backend

1. Create `telemetry/your_game.py`
2. Implement a `QThread` subclass with these signals:
   - `lap_completed = QtCore.pyqtSignal(int, list)`
   - `status_update = QtCore.pyqtSignal(str)`
   - `session_info_update = QtCore.pyqtSignal(dict)`
   - `live_data_update = QtCore.pyqtSignal(dict)`
   - `realtime_sample = QtCore.pyqtSignal(dict)` - **Required for real-time visualization**
3. In `run()` method:
   - Read telemetry from your game's API
   - Create sample dict with `lap_id`, `t`, `x`, `z`, `speed`, `gear`, `rpms`, `brake`, `throttle`
   - Emit `realtime_sample` signal every frame for live visualization
   - Feed samples to `LapBuffer.add_sample()` for lap completion tracking
   - Emit `session_info_update` and `live_data_update` for UI panels
4. Add backend selection to [integrated_telemetry.py](integrated_telemetry.py)

### Sample Data Format

Each sample dict must contain:
- `t` (float) - Elapsed time in seconds (relative to lap start)
- `x` (float) - World position X in meters
- `z` (float) - World position Z in meters
- `speed` (float) - Speed in km/h
- `gear` (int) - Current gear (0=neutral, -1=reverse)
- `rpm` (int) - Engine RPM
- `brake` (float) - Brake input (0.0 to 1.0)
- `throttle` (float) - Throttle input (0.0 to 1.0)
- `tyre_pressure_fl/fr/rl/rr` (float) - Tire pressure in PSI for Front Left, Front Right, Rear Left, Rear Right
- `tyre_temp_fl/fr/rl/rr` (float) - Tire core temperature in °C for all 4 tires

Additional fields can be added as kwargs to `add_sample()` and will be preserved in the sample dict.

### ACC Broadcasting Configuration

ACC requires a config file at `Documents\Assetto Corsa Competizione\Config\broadcasting.json`:

```json
{
  "updListenerPort": 9232,
  "connectionPassword": "",
  "commandPassword": ""
}
```

The port must match the port passed to `AccTelemetryWorker(host="127.0.0.1", port=9232, password="")`.

**Important:** ACC must be running and you must be on track BEFORE starting the Python script. The script acts as a UDP client connecting to ACC's UDP server.

## Platform-Specific Notes

### Windows (AC Shared Memory)
- AC shared memory only works on Windows
- Run Python script as the same user that launched AC (avoid admin/non-admin mismatches)
- Shared memory blocks are session-specific and permission-sensitive

### Cross-Platform (ACC UDP)
- ACC broadcasting should work on any platform where ACC runs
- No special permissions required
- Firewall may need to allow UDP traffic on the configured port

## Testing

No automated tests exist yet. Manual testing workflow:

1. Start the game (AC or ACC)
2. Load into a track session
3. Run `python integrated_telemetry.py` (or `--acc` for ACC)
4. Drive a full lap (cross start/finish line)
5. Verify:
   - Lap appears in lap table with correct time
   - Track map shows colored path
   - Speed/gear/RPM/brake graphs populate (RPM/brake will be zero for ACC)
   - Session info panel shows track/car/player name
   - Live data panel updates in real-time

## Known Issues

- ACC backend cannot provide RPM, throttle, brake, or fuel data (broadcasting API limitation)
- AC backend is Windows-only (shared memory implementation)
- No unit tests or integration tests
- Colorbar on track map needs dark theme styling
- No error handling for malformed UDP packets (ACC)
- No reconnection logic if game restarts mid-session
