"""Device-lease tests (Phase 2 spec §39).

The lease manager refuses busy devices instead of correcting for them, and
every refusal reason is inspectable. The §39 DoD calls out lease leakage
explicitly: release on success, failure, cancellation and shutdown paths is
tested here at the manager level (the runner's exit paths are covered by
``test_profiling_runner.py``).
"""

from __future__ import annotations

import dataclasses

import pytest

from edgeshard.cluster.state import (
    DeviceAvailability,
    DeviceState,
    MemoryPoolState,
    WorkerState,
)
from edgeshard.control.worker.profiling_leases import (
    DEFAULT_UTILIZATION_FLOOR,
    DeviceBusyError,
    DeviceLeaseManager,
)

WORKER_ID = "w-leases"
GPU_A = "gpu-a"
GPU_B = "gpu-b"
GPU_POOL_A = f"gpu-{GPU_A}-vram"


def make_device(
    device_id: str,
    *,
    utilization: float | None = 0.0,
    availability: DeviceAvailability = DeviceAvailability.AVAILABLE,
    running_runtime_ids: tuple[str, ...] = (),
) -> DeviceState:
    return DeviceState(
        device_id=device_id,
        utilization=utilization,
        temperature_c=40.0,
        power_w=None,
        availability=availability,
        running_runtime_ids=running_runtime_ids,
    )


def make_state(
    devices: tuple[DeviceState, ...] = (make_device(GPU_A), make_device(GPU_B)),
    pools: tuple[MemoryPoolState, ...] = (),
) -> WorkerState:
    return WorkerState(
        worker_id=WORKER_ID,
        device_states=devices,
        memory_states=pools,
        runtime_instances=(),
        models=(),
    )


@pytest.fixture
def manager() -> DeviceLeaseManager:
    return DeviceLeaseManager()


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------


def test_acquire_idle_devices_succeeds(manager: DeviceLeaseManager) -> None:
    leases = manager.acquire("s-1", (GPU_A, GPU_B), make_state())
    assert [lease.device_id for lease in leases] == [GPU_A, GPU_B]
    assert all(lease.session_id == "s-1" for lease in leases)
    assert manager.leased_device_ids == (GPU_A, GPU_B)
    assert manager.leases_for_session("s-1") == leases
    assert manager.leases_for_session("s-other") == ()


def test_acquire_dedupes_requested_devices(manager: DeviceLeaseManager) -> None:
    leases = manager.acquire("s-1", (GPU_A, GPU_A), make_state())
    assert [lease.device_id for lease in leases] == [GPU_A]


def test_acquire_requires_session_and_devices(manager: DeviceLeaseManager) -> None:
    with pytest.raises(ValueError, match="session_id"):
        manager.acquire("", (GPU_A,), make_state())
    with pytest.raises(ValueError, match="at least one device"):
        manager.acquire("s-1", (), make_state())


def test_utilization_at_floor_is_not_busy(manager: DeviceLeaseManager) -> None:
    """The floor is exclusive: exactly DEFAULT_UTILIZATION_FLOOR still leases."""
    state = make_state(
        devices=(make_device(GPU_A, utilization=DEFAULT_UTILIZATION_FLOOR),)
    )
    assert len(manager.acquire("s-1", (GPU_A,), state)) == 1


def test_utilization_none_is_not_busy(manager: DeviceLeaseManager) -> None:
    """§52.2: an unreported utilization is a missing fact, never 'busy'."""
    state = make_state(devices=(make_device(GPU_A, utilization=None),))
    assert len(manager.acquire("s-1", (GPU_A,), state)) == 1


# ---------------------------------------------------------------------------
# §39 busy verdicts
# ---------------------------------------------------------------------------


def test_unknown_device_is_busy(manager: DeviceLeaseManager) -> None:
    with pytest.raises(DeviceBusyError, match="not reported") as exc:
        manager.acquire("s-1", ("gpu-ghost",), make_state())
    assert exc.value.device_id == "gpu-ghost"


def test_unavailable_device_is_busy(manager: DeviceLeaseManager) -> None:
    state = make_state(
        devices=(make_device(GPU_A, availability=DeviceAvailability.UNAVAILABLE),)
    )
    with pytest.raises(DeviceBusyError, match="availability"):
        manager.acquire("s-1", (GPU_A,), state)


def test_unknown_availability_device_is_busy(manager: DeviceLeaseManager) -> None:
    state = make_state(
        devices=(make_device(GPU_A, availability=DeviceAvailability.UNKNOWN),)
    )
    with pytest.raises(DeviceBusyError, match="availability"):
        manager.acquire("s-1", (GPU_A,), state)


def test_device_serving_runtime_is_busy(manager: DeviceLeaseManager) -> None:
    state = make_state(
        devices=(make_device(GPU_A, running_runtime_ids=("runtime-1",)),)
    )
    with pytest.raises(DeviceBusyError, match="runtime-1"):
        manager.acquire("s-1", (GPU_A,), state)


def test_saturated_device_is_busy(manager: DeviceLeaseManager) -> None:
    state = make_state(devices=(make_device(GPU_A, utilization=97.5),))
    with pytest.raises(DeviceBusyError, match="utilization"):
        manager.acquire("s-1", (GPU_A,), state)


def test_custom_utilization_floor_is_honored() -> None:
    manager = DeviceLeaseManager(utilization_floor=50.0)
    state = make_state(devices=(make_device(GPU_A, utilization=40.0),))
    assert len(manager.acquire("s-1", (GPU_A,), state)) == 1
    manager.release_session("s-1")
    with pytest.raises(DeviceBusyError, match="utilization"):
        manager.acquire(
            "s-2",
            (GPU_A,),
            make_state(devices=(make_device(GPU_A, utilization=60.0),)),
        )


def test_device_leased_by_other_session_is_busy(manager: DeviceLeaseManager) -> None:
    manager.acquire("s-1", (GPU_A,), make_state())
    with pytest.raises(DeviceBusyError, match="s-1"):
        manager.acquire("s-2", (GPU_A,), make_state())


def test_same_session_may_reacquire_its_device(manager: DeviceLeaseManager) -> None:
    """Idempotent re-prepare of one session does not fight its own lease."""
    state = make_state()
    manager.acquire("s-1", (GPU_A,), state)
    leases = manager.acquire("s-1", (GPU_A, GPU_B), state)
    assert {lease.device_id for lease in leases} == {GPU_A, GPU_B}


# ---------------------------------------------------------------------------
# All-or-nothing rollback (§39: a partial lease never strands a device)
# ---------------------------------------------------------------------------


def test_failed_multi_device_acquire_rolls_back(manager: DeviceLeaseManager) -> None:
    state = make_state(
        devices=(make_device(GPU_A), make_device(GPU_B, utilization=90.0))
    )
    with pytest.raises(DeviceBusyError):
        manager.acquire("s-1", (GPU_A, GPU_B), state)
    assert manager.leased_device_ids == ()
    # GPU_A is free again for the next session.
    idle = make_state()
    assert len(manager.acquire("s-2", (GPU_A,), idle)) == 1


# ---------------------------------------------------------------------------
# Memory-pressure floor (only with a resolver and a reported pool)
# ---------------------------------------------------------------------------


def memory_manager(min_bytes: int = 1_000) -> DeviceLeaseManager:
    return DeviceLeaseManager(
        min_available_bytes=min_bytes,
        memory_pool_resolver=lambda device_id: (
            (GPU_POOL_A,) if device_id == GPU_A else ()
        ),
    )


def test_low_pool_pressure_is_busy() -> None:
    state = make_state(pools=(MemoryPoolState(GPU_POOL_A, available_bytes=10),))
    with pytest.raises(DeviceBusyError, match="below the"):
        memory_manager().acquire("s-1", (GPU_A,), state)


def test_sufficient_pool_pressure_passes() -> None:
    state = make_state(pools=(MemoryPoolState(GPU_POOL_A, available_bytes=10_000),))
    assert len(memory_manager().acquire("s-1", (GPU_A,), state)) == 1


def test_unreported_pool_bytes_are_not_pressure() -> None:
    """§52.2: available_bytes=None is a missing fact, never an assumption."""
    state = make_state(pools=(MemoryPoolState(GPU_POOL_A, available_bytes=None),))
    assert len(memory_manager().acquire("s-1", (GPU_A,), state)) == 1


def test_missing_pool_is_not_pressure() -> None:
    assert len(memory_manager().acquire("s-1", (GPU_A,), make_state())) == 1


def test_device_without_mapped_pool_skips_memory_check() -> None:
    # GPU_B resolves to no pools; a starved pool of GPU_A must not gate it.
    state = make_state(pools=(MemoryPoolState(GPU_POOL_A, available_bytes=0),))
    assert len(memory_manager().acquire("s-1", (GPU_B,), state)) == 1


# ---------------------------------------------------------------------------
# Release paths (§39: no lease outlives its session or the runner)
# ---------------------------------------------------------------------------


def test_release_session_frees_devices(manager: DeviceLeaseManager) -> None:
    manager.acquire("s-1", (GPU_A,), make_state())
    manager.acquire("s-2", (GPU_B,), make_state())
    assert manager.release_session("s-1") == (GPU_A,)
    assert manager.leased_device_ids == (GPU_B,)
    # The freed device is reservable by another session again.
    assert len(manager.acquire("s-3", (GPU_A,), make_state())) == 1


def test_release_session_is_idempotent(manager: DeviceLeaseManager) -> None:
    manager.acquire("s-1", (GPU_A,), make_state())
    assert manager.release_session("s-1") == (GPU_A,)
    assert manager.release_session("s-1") == ()
    assert manager.release_session("never-leased") == ()


def test_release_all_clears_every_session(manager: DeviceLeaseManager) -> None:
    state = make_state()
    manager.acquire("s-1", (GPU_A,), state)
    manager.acquire("s-2", (GPU_B,), state)
    assert set(manager.release_all()) == {GPU_A, GPU_B}
    assert manager.leased_device_ids == ()
    assert manager.release_all() == ()


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("floor", [-1.0, 100.1])
def test_utilization_floor_must_be_a_percentage(floor: float) -> None:
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        DeviceLeaseManager(utilization_floor=floor)


def test_negative_memory_floor_rejected() -> None:
    with pytest.raises(ValueError, match="negative"):
        DeviceLeaseManager(min_available_bytes=-1, memory_pool_resolver=lambda d: ())


def test_memory_floor_requires_resolver() -> None:
    with pytest.raises(ValueError, match="memory_pool_resolver"):
        DeviceLeaseManager(min_available_bytes=1024)


def test_device_busy_error_carries_device_and_reason() -> None:
    error = DeviceBusyError(GPU_A, "test reason")
    assert error.device_id == GPU_A
    assert error.reason == "test reason"
    assert GPU_A in str(error) and "test reason" in str(error)


def test_dataclass_lease_equality() -> None:
    """Leases are value objects; a re-acquired lease compares equal."""
    manager = DeviceLeaseManager()
    first = manager.acquire("s-1", (GPU_A,), make_state())[0]
    manager.release_session("s-1")
    second = manager.acquire("s-1", (GPU_A,), make_state())[0]
    assert first == second == dataclasses.replace(first)
