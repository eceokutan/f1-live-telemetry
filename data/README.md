# Session Recording and Playback

This directory contains the session recording infrastructure for the F1 Telemetry Dashboard.

## Overview

All telemetry sessions are automatically recorded to a SQLite database (`telemetry_sessions.db`) when you run the dashboard. This enables:

- **Session Replay** - Recreate any previous session
- **Post-Race Analysis** - Analyze lap times, fuel usage, tire wear
- **AI Commentary History** - Review all AI race engineer messages
- **Voice Query Log** - See all driver questions and responses

## Database Structure

The SQLite database contains 5 tables:

1. **sessions** - Session metadata (track, car, player, start/end time, lap count, best lap)
2. **laps** - Individual lap data (lap time, fuel usage, average/max speed)
3. **telemetry** - High-frequency telemetry samples (~60 samples/second)
   - Position (x, z), speed, gear, RPM, throttle, brake, fuel
   - Tire pressure and temperature for all 4 tires (FL, FR, RL, RR)
4. **ai_commentary** - AI race engineer messages with timestamps
5. **voice_queries** - Driver voice queries paired with AI responses

## Using the Session Viewer

The session viewer is a command-line tool for querying recorded sessions.

### List All Sessions

```bash
python -m data.session_viewer list
```

Output example:
```
============================================================================================================
RECORDED SESSIONS
============================================================================================================

Session 1: 2025-01-15 14:32:05
  Game: AC | Track: Monza | Car: Ferrari 488 GT3
  Laps: 12 | Best Lap: 1:47.235s | AI: Yes

Session 2: 2025-01-15 15:10:22
  Game: AC | Track: Spa-Francorchamps | Car: McLaren 720S GT3
  Laps: 8 | Best Lap: 2:17.891s | AI: Yes
============================================================================================================
```

### View Session Details

```bash
python -m data.session_viewer info 1
```

Shows:
- Track, car, player name
- Start/end time
- Total laps and best lap time
- Telemetry sample count
- AI commentary count (if AI was enabled)
- Voice query count (if voice input was used)

### View Lap Times

```bash
python -m data.session_viewer laps 1
```

Shows a table with:
- Lap number
- Lap time
- Fuel used per lap
- Average speed
- Maximum speed
- Validity (no track cuts)

### Export to CSV

```bash
python -m data.session_viewer export 1
```

Exports session data to CSV files in `data/exports/`:
- `session_1_telemetry.csv` - All telemetry samples
- `session_1_laps.csv` - Lap times and statistics
- `session_1_ai_commentary.csv` - AI messages (if AI was enabled)

CSV files can be opened in Excel, imported into analysis tools (Python pandas, R, MATLAB), or used for custom visualizations.

## Data Analysis Examples

### Python with pandas

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load telemetry data
df = pd.read_csv('data/exports/session_1_telemetry.csv')

# Filter to a specific lap
lap_3 = df[df['lap_number'] == 3]

# Plot speed trace
plt.plot(lap_3['elapsed_time'], lap_3['speed'])
plt.xlabel('Time (s)')
plt.ylabel('Speed (km/h)')
plt.title('Lap 3 Speed Trace')
plt.show()

# Compare tire temperatures
fig, ax = plt.subplots()
ax.plot(lap_3['elapsed_time'], lap_3['tyre_temp_fl'], label='Front Left')
ax.plot(lap_3['elapsed_time'], lap_3['tyre_temp_fr'], label='Front Right')
ax.plot(lap_3['elapsed_time'], lap_3['tyre_temp_rl'], label='Rear Left')
ax.plot(lap_3['elapsed_time'], lap_3['tyre_temp_rr'], label='Rear Right')
ax.legend()
ax.set_xlabel('Time (s)')
ax.set_ylabel('Tire Temperature (°C)')
plt.show()
```

### Lap Comparison

```python
# Load lap data
laps = pd.read_csv('data/exports/session_1_laps.csv')

# Find best lap
best_lap_number = laps.loc[laps['lap_time'].idxmin(), 'lap_number']

# Load telemetry
df = pd.read_csv('data/exports/session_1_telemetry.csv')

# Compare best lap vs average lap
best_lap_data = df[df['lap_number'] == best_lap_number]
avg_lap_data = df[df['lap_number'] == 5]  # Pick a representative lap

# Plot speed comparison
plt.plot(best_lap_data['elapsed_time'], best_lap_data['speed'], label='Best Lap')
plt.plot(avg_lap_data['elapsed_time'], avg_lap_data['speed'], label='Average Lap', alpha=0.7)
plt.legend()
plt.show()
```

## Direct Database Queries

For advanced analysis, you can query the SQLite database directly:

```python
import sqlite3

conn = sqlite3.connect('data/telemetry_sessions.db')
cursor = conn.cursor()

# Get all sessions for a specific track
cursor.execute("""
    SELECT session_id, track_name, total_laps, best_lap_time
    FROM sessions
    WHERE track_name LIKE '%Monza%'
    ORDER BY best_lap_time ASC
""")

for row in cursor.fetchall():
    print(f"Session {row[0]}: {row[2]} laps, best: {row[3]:.3f}s")

# Get fuel consumption per lap for a session
cursor.execute("""
    SELECT lap_number, fuel_start - fuel_end as fuel_used
    FROM laps
    WHERE session_id = 1
    ORDER BY lap_number
""")

for row in cursor.fetchall():
    print(f"Lap {row[0]}: {row[1]:.2f}L")

conn.close()
```

## Performance Notes

- **Telemetry samples** are recorded at ~60Hz (60 samples per second)
- **Batch inserts** - Samples are buffered and written in batches of 60 for performance
- **Database size** - Expect ~1-2 MB per lap depending on lap duration
- **Indexing** - Database is indexed on session_id and lap_number for fast queries

## Future Enhancements

The session recording system is designed to support future features:

- **Session Replay UI** - Visual playback of recorded sessions
- **Lap Comparison Tool** - Side-by-side comparison of telemetry traces
- **Race Strategy Analysis** - Fuel usage, tire deg, optimal pit windows
- **AI Training Data** - Use recorded sessions to improve AI race engineer
- **Telemetry Sharing** - Export/import sessions for coaching or competition
