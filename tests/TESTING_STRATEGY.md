# Testing Strategy (COMP0016)

## Goals
- Validate reliability at multiple levels: unit, component, integration, system, and UAT.
- Keep fast automated checks for frequent regression runs.
- Keep slower/manual checks as explicit opt-in suites with documented outcomes.

## Test Pyramid In This Repository
- `unit`: fast logic checks for small functions and data models.
- `component`: module/class behavior through public interfaces with mocked dependencies.
- `integration`: real component collaboration (database/export/load/bundle workflows).
- `system`: automated end-to-end smoke workflow checks, plus manual environment checks.
- `uat`: executable acceptance tests for key user stories.

## Automated Regression Policy
- Default command: `pytest`
- Default run includes: unit + component + integration + automated system + automated UAT.
- Default run excludes: `manual` and `slow`.
- Regression automation: GitHub Actions workflow at `.github/workflows/tests.yml`.

## Marker Guide
- `unit`
- `component`
- `integration`
- `system`
- `uat`
- `regression`
- `slow`
- `manual`

## Execution Commands
- Fast regression suite: `pytest`
- Include slow automated checks: `pytest -m "not manual"`
- Automated system smoke checks only: `pytest -m "system and not manual"`
- Automated UAT checks only: `pytest -m "uat"`
- Manual system checks: `RUN_ENVIRONMENT_CHECKS=1 pytest -m "system and manual" tests/test_environment.py`

## Test Case Template
- Test Case ID
- Test Case Title
- Description
- Preconditions
- Test Data
- Steps to Carry Out
- Expected Results
- Actual Results
- Pass/Fail Criteria
- Remarks/Comments
- Tested By
- Test Date

## Evidence To Capture For Assessment
- Pytest output (pass/fail summary) from local and CI runs.
- Manual/system/UAT execution logs in `tests/manual/`.
- Notes on any failing tests and root-cause fixes (regression tracking).
