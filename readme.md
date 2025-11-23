currently supports:
- assetto corsa (ac) ✅
- assetto corsa competizione (acc) ⚠️ backend skeleton ready, packet parsing still todo

this is a live telemetry dashboard for sim games (for systems coursework team 17).
right now the ui + backend are wired for ac, and acc is being prepared via a separate telemetry backend.

--------------------------------------------------
assetto corsa (ac) setup steps
--------------------------------------------------

1. enable shared memory in ac
   - go to: options -> general -> scroll to ui modules and shared memory
   - shared memory = ON
   - shared memory layout = 1
   - udp plugin = OFF
   - udp frequency = 333 hz
   - enable data > apps = ON

2. start a session on track
   - telemetry only becomes active when:
     - car is loaded
     - you are in the cockpit
     - physics thread is running
   - so: start a practice / hotlap / race session
   - wait until the car is fully loaded and look around once to make sure physics is running

3. shared memory details (windows only)
   - while ac is running:
     - press ctrl+shift+esc
     - go to the "details" tab
     - look for: acs.exe / acs_x64.exe
     - right click -> open file location (just to sanity check the game is running)
   - the script reads these windows named shared memory blocks (not files on disk):
     - `acpmf_static`
     - `acpmf_physics`
     - `acpmf_graphics`

4. run on the same user session
   - run ac normally (no admin needed)
   - run the python script from the same windows user account
   - do not run one as admin and the other as non-admin (named shared memory is session / permission sensitive)

--------------------------------------------------
assetto corsa competizione (acc) setup steps (planned)
--------------------------------------------------

acc support is being added via a separate telemetry backend:
- file: `telemetry/acc_udp.py`
- this listens to acc’s udp broadcasting and feeds normalized samples into the same ui + lap logic

to prepare acc:

1. enable udp broadcasting in acc
   - in acc, go to options -> (network / general, depending on version) -> broadcasting
   - turn broadcasting ON
   - set the target ip + port, e.g.:
     - ip: `127.0.0.1`
     - port: `9000` (this is the default in `AccTelemetryWorker` right now)

2. make sure the port matches the code
   - in `telemetry/acc_udp.py`:
     - `AccTelemetryWorker(host="127.0.0.1", port=9000)`
   - if you change the port in acc, update it here as well

3. start a session on track in acc
   - similar to ac: telemetry is only meaningful when you are on track and driving
   - start a practice / hotlap / race so there is realtime data to broadcast

4. important: packet parsing is still TODO
   - the backend skeleton is already there, but:
     - `parse_acc_packet()` in `telemetry/acc_udp.py` is not implemented yet
   - until you implement this using the official acc broadcasting sdk / docs:
     - acc backend will not actually produce lap data for the dashboard
   - once implemented, it should output a dict like:
     - `{"t", "lap_id", "x", "z", "speed_kmh", "gear", "rpms", "brake", "throttle"}`

--------------------------------------------------
running the app (current status)
--------------------------------------------------

requirements:
- python 3.x
- pyqt5
- matplotlib
- numpy
(see `requirements.txt` for exact versions)

basic run (assetto corsa):

1. install deps
   - `pip install -r requirements.txt`

2. start assetto corsa and get on track (see ac setup steps above)

3. run the telemetry dashboard
   - `python integrated_telemetry.py`
   - this currently starts the assetto corsa backend (ac)

you should see:
- a window called “ac telemetry dashboard – prototype”
- once you complete a full lap in ac:
  - track map (colored by speed)
  - per-lap graphs (speed / gear / rpm / brake)
  - lap table filled with laptime

acc run (future):

- `integrated_telemetry.py` is structured to support multiple backends:
  - `AcTelemetryWorker` (ac)
  - `AccTelemetryWorker` (acc)
- after `parse_acc_packet()` is implemented and the game selector is wired:
  - you’ll be able to start the app with acc as the source instead of ac
  - the ui and lap logic shouldn’t need any changes

--------------------------------------------------
current notes / roadmap
--------------------------------------------------

short-term:
- wire static info & live labels in ui (driver name, car, track, etc.)
- add a real-time `live_data` signal so speed / gear / rpm / fuel update continuously
- polish dark theme (colorbar + minor styling)

acc-related:
- implement `parse_acc_packet()` in `telemetry/acc_udp.py` using the official acc broadcasting sdk
- confirm lap counting logic for acc (equivalent of `completedLaps` in ac)
- verify we can extract world coordinates (x, z) from acc to draw the track map

future ideas:
- support more sims via their own backend files (e.g. iracing, f1 games, rfactor 2)
- multi-driver comparison per lap
- export laps to csv / parquet for offline analysis
- integrate a custom ai chatbot / engineer inside the app for strategy / driving hints

--------------------------------------------------
about
--------------------------------------------------

live telemetry dashboard for simulation games.
designed to be:
- modular (one backend per sim: ac, acc, etc.)
- reusable (same ui + lap logic, multiple sources)
- a base for more advanced stuff (ai engineer, strategy, overlays, etc.)
