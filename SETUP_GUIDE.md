# Assetto Corsa Telemetry Setup Guide

## Prerequisites
- Windows OS (shared memory only works on Windows)
- Assetto Corsa installed
- Python 3.x with dependencies installed (`pip install -r requirements.txt`)

## Step 1: Enable Shared Memory in Assetto Corsa

### Option A: Using Content Manager (Recommended)
1. Open **Content Manager**
2. Go to **Settings** (gear icon, top right)
3. Navigate to **Assetto Corsa** → **Video** tab
4. Scroll down to **Developer** section
5. Enable these settings:
   - ✅ **Shared Memory** = ON
   - **Shared Memory Layout** = 1
   - You can leave **UDP Plugin** = OFF (we don't need it)

### Option B: Using Assetto Corsa Directly
1. Launch **Assetto Corsa** (not Content Manager)
2. Go to **Options** → **General**
3. Scroll down to **UI Modules and Shared Memory**
4. Set:
   - ✅ **Shared Memory** = ON
   - **Shared Memory Layout** = 1
   - **UDP Plugin** = OFF
   - **UDP Frequency** = 333 Hz (doesn't matter if UDP is off)
   - **Enable Data > Apps** = ON

## Step 2: Start a Session

**CRITICAL:** Shared memory is ONLY active when:
- You're in a session (Practice, Qualifying, Race, Hotlap)
- The car is loaded on track
- You're in the cockpit view (or any driving view)
- Physics engine is running

**NOT active when:**
- ❌ In menus
- ❌ Paused
- ❌ Replay mode
- ❌ Game not running

### How to Start a Session:
1. Launch Assetto Corsa (via Content Manager or directly)
2. Select **Drive** → **Practice** or **Hotlap**
3. Choose any track and car
4. Click **Drive**
5. Wait for the car to fully load
6. Make sure you can see the steering wheel/cockpit

## Step 3: Run the Dashboard

Open a terminal (Command Prompt or PowerShell) and run:

```bash
cd /path/to/f1_telemetry_app
python integrated_telemetry.py
```

## Step 4: Verify Connection

You should see in the terminal:

```
============================================================
🔍 AC TELEMETRY WORKER STARTING
============================================================
✅ Successfully connected to AC shared memory!
📊 Session Info:
   Track: monza (gp)
   Car: ks_ferrari_458
   Driver: Your Name

🏁 Starting telemetry loop (reading at ~60Hz)...
   Waiting for car data...

📦 Packet #0060 | Lap: 1 | Speed:  120.5 km/h | Gear: 3 | RPM:  7200 | Pos: (150.2, -320.5)
```

## Step 5: Start Driving

Once connected:
1. **Start driving** - the graphs will begin appearing immediately
2. **Track map** will draw your path in real-time
3. **Graphs** will show speed, RPM, gear, brake, tire pressure, tire temp
4. **Complete a lap** - lap time will be recorded in the table

## Troubleshooting

### "Could not connect to AC shared memory"

**Cause:** AC is not running, or you're not in a session.

**Fix:**
1. Make sure AC is running
2. Make sure you're ON TRACK (not in menus)
3. Try exiting to menu and going back on track
4. Restart AC if needed

### "All values showing 0"

**Cause:** You're in a session but physics isn't running (paused, or spectating).

**Fix:**
1. Make sure you're not paused
2. Drive the car - values should start appearing
3. Check that you're in a driving view (not external camera)

### "Permission denied" or "Access denied"

**Cause:** User permission mismatch.

**Fix:**
1. Run both AC and Python script as the **same user**
2. Don't run one as admin and the other as normal user
3. Close both, then restart normally (without admin)

### "Module not found" errors

**Fix:**
```bash
pip install -r requirements.txt
```

## Testing Checklist

- [ ] Shared memory enabled in AC settings
- [ ] AC is running
- [ ] You're in a session (practice/race)
- [ ] You're on track (can see cockpit)
- [ ] Python script started AFTER AC is on track
- [ ] Terminal shows "Successfully connected"
- [ ] Terminal shows packet messages every second
- [ ] UI window opened
- [ ] Graphs start appearing when you drive

## Advanced: Running Order

**Best practice:**
1. Start AC first
2. Load into track
3. Wait for car to fully load
4. THEN run Python script

You can also:
- Leave Python script running
- Restart AC session
- It should reconnect automatically

## Next Steps

Once connected:
- Drive a lap to see the track map
- Watch the graphs update in real-time
- Complete a lap to see lap time in the table
- Try different tracks and cars!
