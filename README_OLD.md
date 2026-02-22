# F1 Live Telemetry Dashboard

## Currently Supports:
- **Assetto Corsa (AC)** ✅
- **Assetto Corsa Competizione (ACC)** ⚠️ limited telemetry via broadcasting API

This is a live telemetry dashboard for sim games (for systems coursework team 17).
Right now the UI + backend are wired for AC, and ACC is being prepared via a separate telemetry backend.

---

## Assetto Corsa (AC) Setup Steps

### 1. Enable Shared Memory in AC
- Go to: **Options → General → Scroll to UI Modules and Shared Memory**
- **Shared Memory** = ON
- **Shared Memory Layout** = 1
- **UDP Plugin** = OFF
- **UDP Frequency** = 333 Hz
- **Enable Data > Apps** = ON

### 2. Start a Session on Track
Telemetry only becomes active when:
- Car is loaded
- You are in the cockpit
- Physics thread is running

So: Start a practice / hotlap / race session and wait until the car is fully loaded. Look around once to make sure physics is running.

### 3. Shared Memory Details (Windows Only)
While AC is running:
- Press `Ctrl+Shift+Esc`
- Go to the "Details" tab
- Look for: `acs.exe` / `acs_x64.exe`
- Right click → Open file location (just to sanity check the game is running)

The script reads these Windows named shared memory blocks (not files on disk):
- `acpmf_static`
- `acpmf_physics`
- `acpmf_graphics`

### 4. Run on the Same User Session
- Run AC normally (no admin needed)
- Run the Python script from the same Windows user account
- Do not run one as admin and the other as non-admin (named shared memory is session/permission sensitive)

---

## Assetto Corsa Competizione (ACC) Setup Steps

ACC support uses UDP broadcasting protocol:
- **File:** `telemetry/acc_udp.py`
- Listens to ACC's broadcasting API and feeds normalized samples into the same UI + lap logic

### ⚠️ Important Limitations:
- **ACC broadcasting provides:** position (x,y,z), speed, gear, lap times, position
- **ACC broadcasting does NOT provide:** RPM, brake input, throttle input, fuel level
- RPM/brake/throttle graphs will show zeros or be empty when using ACC
- For full telemetry, you'd need ACC's physics shared memory plugin (not implemented)

### 1. Configure ACC Broadcasting
- Close ACC completely first
- Navigate to: `Documents\Assetto Corsa Competizione\Config\`
- Create or edit `broadcasting.json`:

```json
{
  "updListenerPort": 9232,
  "connectionPassword": "",
  "commandPassword": ""
}
```

- Save the file
- This tells ACC to listen for broadcasting clients on port 9232

### 2. Make Sure the Port Matches the Code
In `telemetry/acc_udp.py` or when starting the worker:
```python
AccTelemetryWorker(host="127.0.0.1", port=9232, password="")
```

If you change `updListenerPort` in `broadcasting.json`, update the Python code to match. Default is 9232, but you can use any available port.

### 3. Start ACC and Get on Track FIRST
- ACC must be running before you start the Python script
- Start a practice / hotlap / race session
- Telemetry only flows when you're on track and driving

### 4. Run the Python Script
The script will:
1. Bind to the configured port
2. Send a registration packet to ACC
3. Receive registration confirmation
4. Request track data and entry list
5. Continuously receive telemetry updates

If connection fails, check:
- ACC is running and you're on track
- `broadcasting.json` port matches your Python code
- Firewall isn't blocking the port
- No other program is using that port

### How ACC Broadcasting Works:
- ACC acts as the UDP server (listens on port 9232)
- Your Python script acts as the client (connects to ACC)
- **Connection flow:**
  1. Python script binds to port 9232
  2. Python sends REGISTER packet to `127.0.0.1:9232`
  3. ACC responds with REGISTRATION_RESULT
  4. Python requests TRACK_DATA and ENTRY_LIST
  5. ACC continuously broadcasts REALTIME_CAR_UPDATE packets
  6. Python parses packets and feeds data to dashboard

---

## Running the App

### Requirements:
- Python 3.x
- PyQt5
- Matplotlib
- NumPy

See `requirements.txt` for exact versions.

### Basic Run (Assetto Corsa):

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Assetto Corsa and get on track** (see AC setup steps above)

3. **Run the telemetry dashboard**
   ```bash
   python integrated_telemetry.py
   ```
   This currently starts the Assetto Corsa backend (AC) by default.

**You should see:**
- A window called "AC Telemetry Dashboard – Prototype"
- Once you complete a full lap in AC:
  - Track map (colored by speed)
  - Per-lap graphs (speed / gear / RPM / brake)
  - Lap table filled with laptime

### ACC Run:

1. **Configure ACC** (see ACC setup steps above)

2. **Start ACC and get on track**

3. **Modify `integrated_telemetry.py` to use ACC backend:**
   ```python
   # Change from:
   telemetry_thread = AcTelemetryWorker()
   
   # To:
   from telemetry.acc_udp import AccTelemetryWorker
   telemetry_thread = AccTelemetryWorker(host="127.0.0.1", port=9232, password="")
   ```

4. **Run the dashboard**
   ```bash
   python integrated_telemetry.py
   ```

**Expected behavior with ACC:**
- ✅ Track map will work (position data available)
- ✅ Speed graph will work
- ✅ Gear graph will work
- ✅ Lap times will work
- ❌ RPM graph will show zeros (not in broadcasting API)
- ❌ Brake graph will show zeros (not in broadcasting API)
- ❌ Fuel will show "--" (not in broadcasting API)

---

## Current Notes / Roadmap

### Completed:
- ✅ AC backend with full telemetry (shared memory)
- ✅ ACC backend with broadcasting protocol
- ✅ Packet parsing for ACC (registration, realtime updates, track data)
- ✅ Lap detection and buffering for both AC and ACC
- ✅ Live data updates for session info panel

### Short-term:
- Add game selector in UI (choose AC or ACC at startup)
- Polish dark theme (colorbar + minor styling)
- Better error handling and connection status display

### ACC Improvements:
- Investigate ACC physics shared memory plugin for full telemetry
- Add pit status detection from ACC data
- Handle multi-car scenarios (currently focuses on player car)

### Future Ideas:
- Support more sims via their own backend files (e.g. iRacing, F1 games, rFactor 2)
- Multi-driver comparison per lap
- Export laps to CSV / Parquet for offline analysis
- Integrate a custom AI chatbot / engineer inside the app for strategy / driving hints
- Overlay mode (transparent window over game)

---

## About

Live telemetry dashboard for simulation games.

**Designed to be:**
- **Modular** - One backend per sim: AC, ACC, etc.
- **Reusable** - Same UI + lap logic, multiple sources
- **Extensible** - A base for more advanced features (AI engineer, strategy, overlays, etc.)

---

## Troubleshooting

### AC Issues:

**"Could not open shared memory":**
- Make sure AC is running and you're in a session
- Check you're running Python script as same user (no admin mismatch)
- Verify shared memory is enabled in AC settings

### ACC Issues:

**"Connection failed" or no telemetry:**
- Start ACC BEFORE running the Python script
- Make sure you're on track (not in menus)
- Check `broadcasting.json` exists and has correct port
- Verify port isn't blocked by firewall
- Try restarting ACC after editing `broadcasting.json`

**"RPM/brake/throttle showing zeros":**
- This is expected - ACC broadcasting API doesn't provide this data
- Only position, speed, gear, and lap timing are available
- For full telemetry, would need ACC's physics plugin (different API)

### General Issues:

**Graphs not updating:**
- Complete a full lap (cross start/finish line)
- Telemetry is buffered per-lap and displays after lap completion
- Check console output for errors

**"Module not found" errors:**
- Run `pip install -r requirements.txt`
- Make sure you're in the correct Python environment