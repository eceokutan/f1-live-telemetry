# UAT Test Cases

## UAT-01: Driver Can Review Lap Performance
- User Story: As a driver, I can select a recorded session and inspect lap telemetry.
- Acceptance Criteria:
1. Session list shows track/car/lap metadata.
2. Selecting a lap displays charts and track map.
3. Timeline controls play/pause/seek without errors.
- Pass Condition: All criteria met in one end-to-end run.

## UAT-02: Driver Receives Understandable Coaching Output
- User Story: As a driver, I can request analysis and get actionable feedback.
- Acceptance Criteria:
1. Triggering analysis returns text in the UI.
2. Feedback references lap/session context (not generic placeholders).
3. Failure path provides clear fallback message.
- Pass Condition: All criteria met for at least one lap.

## UAT-03: Session Sharing Works Via Bundle Export/Import
- User Story: As a user, I can export a session and import it on another setup.
- Acceptance Criteria:
1. Export creates `.jsession` file.
2. Import creates a new database session entry.
3. Imported session is viewable and usable in post-race tooling.
- Pass Condition: Export/import/view workflow succeeds.
