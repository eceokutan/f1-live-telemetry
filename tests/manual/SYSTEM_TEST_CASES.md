# System Test Cases

## ST-01: Full Live Telemetry Session (Assetto Corsa -> Jarvis Live)
- Description: Validate full live telemetry workflow in a production-like runtime.
- Preconditions: Windows host, Assetto Corsa running on track, shared memory enabled.
- Test Data: Real driving session, at least 2 completed laps.
- Steps:
1. Launch Assetto Corsa and start a practice/hotlap session.
2. Run `python main.py`.
3. Open Jarvis Live and observe telemetry panels.
4. Complete at least two laps.
5. End session and close app.
- Expected Results:
1. Live speed/gear/throttle/brake graphs update continuously.
2. Lap table records completed laps.
3. Session is persisted to SQLite.

## ST-02: End-to-End Post-Race Flow (Recorded Session -> Jarvis Post)
- Description: Validate complete post-race analysis journey using recorded data.
- Preconditions: At least one recorded session with laps.
- Test Data: Existing session with telemetry and lap summaries.
- Steps:
1. Launch app and open session picker.
2. Select recorded session.
3. Open post-race lap viewer.
4. Scrub timeline and switch laps.
5. Trigger AI coaching/analysis if configured.
- Expected Results:
1. Session loads without crashes.
2. Track map and charts sync with timeline.
3. AI outputs render (or deterministic fallback message appears).
