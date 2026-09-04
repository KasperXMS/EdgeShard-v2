"""Dynamic state domain tests (Phase 1 spec §17-18, §51)."""

from __future__ import annotations

import dataclasses

import pytest
from factories import make_worker_state

from edgeshard.cluster.state import (
    DeviceAvailability,
    DeviceState,
    MemoryPoolState,
    WorkerState,
    WorkerStatus,
)


def make_device_state(device_id: str = "gpu-0") -> DeviceState:
    return DeviceState(
        device_id=device_id,
        utilization=None,
        temperature_c=None,
        power_w=None,
        availability=DeviceAvailability.UNKNOWN,
        running_runtime_ids=(),
    )


def test_missing_telemetry_is_none_not_unavailable() -> None:
    """Absent metrics are unknown, not unavailable (spec §17)."""
    state = make_device_state()
    assert state.utilization is None
    assert state.temperature_c is None
    assert state.power_w is None
    assert state.availability is DeviceAvailability.UNKNOWN


def test_worker_status_values() -> None:
    assert [status.value for status in WorkerStatus] == ["online", "suspect", "offline"]


def test_worker_state_carries_no_liveness_status() -> None:
    """Liveness is Master-assigned and never part of the reported state."""
    field_names = {field.name for field in dataclasses.fields(WorkerState)}
    assert "status" not in field_names


def test_worker_state_assembles() -> None:
    state = make_worker_state("worker-1", device_ids=("gpu-a", "gpu-b"), pool_ids=("pool-a",))
    assert state.worker_id == "worker-1"
    assert [device.device_id for device in state.device_states] == ["gpu-a", "gpu-b"]
    assert len(state.runtime_instances) == 1
    assert len(state.models) == 1


def test_worker_state_rejects_empty_worker_id() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        make_worker_state("")


def test_worker_state_rejects_duplicate_device_states() -> None:
    with pytest.raises(ValueError, match="duplicate device_id"):
        make_worker_state("worker-1", device_ids=("gpu-a", "gpu-a"))


def test_worker_state_rejects_duplicate_memory_states() -> None:
    with pytest.raises(ValueError, match="duplicate memory_pool_id"):
        make_worker_state("worker-1", pool_ids=("pool-a", "pool-a"))


def test_device_state_rejects_empty_device_id() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        make_device_state("")


@pytest.mark.parametrize("utilization", [-0.1, 100.1])
def test_device_state_rejects_out_of_range_utilization(utilization: float) -> None:
    with pytest.raises(ValueError, match="utilization"):
        DeviceState(
            device_id="gpu-0",
            utilization=utilization,
            temperature_c=None,
            power_w=None,
            availability=DeviceAvailability.AVAILABLE,
            running_runtime_ids=(),
        )


def test_memory_pool_state_rejects_negative_available_bytes() -> None:
    with pytest.raises(ValueError, match="available_bytes"):
        MemoryPoolState(memory_pool_id="pool-0", available_bytes=-1)


def test_memory_pool_state_unknown_availability_allowed() -> None:
    assert MemoryPoolState(memory_pool_id="pool-0", available_bytes=None).available_bytes is None
