from __future__ import annotations

import pytest

from telemetry.lap_buffer import LapBuffer

pytestmark = [pytest.mark.unit, pytest.mark.component, pytest.mark.regression]


def test_lap_buffer_emits_completed_lap_on_increment():
    completed = []
    buffer = LapBuffer(on_lap_complete=lambda lap_id, samples: completed.append((lap_id, samples)))

    buffer.add_sample(1, t=0.0, x=1.0, z=2.0, speed_kmh=100.0, gear=3)
    buffer.add_sample(1, t=0.5, x=1.5, z=2.5, speed_kmh=110.0, gear=4)
    buffer.add_sample(2, t=0.0, x=2.0, z=3.0, speed_kmh=120.0, gear=4)

    assert len(completed) == 1
    lap_id, samples = completed[0]
    assert lap_id == 1
    assert len(samples) == 2
    assert samples[0]["gear"] == 3
    assert buffer.current_lap_id == 2
    assert len(buffer.samples) == 1


def test_lap_buffer_resets_when_lap_counter_goes_backwards():
    completed = []
    buffer = LapBuffer(on_lap_complete=lambda lap_id, samples: completed.append((lap_id, samples)))

    buffer.add_sample(3, t=10.0, x=0.0, z=0.0, speed_kmh=80.0)
    buffer.add_sample(2, t=0.0, x=1.0, z=1.0, speed_kmh=90.0)

    assert completed == []
    assert buffer.current_lap_id == 2
    assert len(buffer.samples) == 1
