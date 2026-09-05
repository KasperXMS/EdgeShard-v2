"""ClusterSnapshot domain tests (Phase 1 spec §38-39, §51)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from edgeshard.cluster.snapshot import ClusterSnapshot, WorkerSnapshot
from edgeshard.cluster.state import (
    DeviceAvailability,
    DeviceState,
    MemoryPoolState,
    WorkerState,
    WorkerStatus,
)
from factories import make_rtx_capability, make_worker_identity, make_worker_state

CREATED_AT = datetime(2026, 9, 5, 12, 0, 0, tzinfo=UTC)
LAST_SEEN_AT = datetime(2026, 9, 5, 11, 59, 58, tzinfo=UTC)


def make_worker_snapshot(worker_id: str = "worker-1") -> WorkerSnapshot:
    return WorkerSnapshot(
        identity=make_worker_identity(worker_id),
        capability=make_rtx_capability(),
        state=make_worker_state(worker_id),
        status=WorkerStatus.ONLINE,
        session_id="session-1",
        last_seen_at=LAST_SEEN_AT,
    )


def make_cluster_snapshot(worker_ids: tuple[str, ...] = ("worker-1",)) -> ClusterSnapshot:
    return ClusterSnapshot(
        snapshot_id="snapshot-1",
        created_at=CREATED_AT,
        workers=tuple(make_worker_snapshot(worker_id) for worker_id in worker_ids),
    )


def test_snapshot_assembles() -> None:
    snapshot = make_cluster_snapshot(("worker-a", "worker-b"))
    assert snapshot.snapshot_id == "snapshot-1"
    assert [worker.identity.worker_id for worker in snapshot.workers] == [
        "worker-a",
        "worker-b",
    ]
    assert all(worker.status is WorkerStatus.ONLINE for worker in snapshot.workers)


def test_snapshot_requires_consistent_worker_id() -> None:
    with pytest.raises(ValueError, match="worker_id mismatch"):
        WorkerSnapshot(
            identity=make_worker_identity("worker-1"),
            capability=make_rtx_capability(),
            state=make_worker_state("worker-2"),
            status=WorkerStatus.ONLINE,
            session_id="session-1",
            last_seen_at=LAST_SEEN_AT,
        )


@pytest.mark.parametrize(
    "timestamp",
    [datetime(2026, 9, 5, 12, 0, 0), None],
    ids=["naive", "absent"],
)
def test_worker_snapshot_rejects_invalid_last_seen(timestamp: datetime | None) -> None:
    if timestamp is None:
        # Absent last_seen_at is valid for never-registered workers.
        snapshot = dataclasses.replace(make_worker_snapshot(), last_seen_at=None)
        assert snapshot.last_seen_at is None
        return
    with pytest.raises(ValueError, match="last_seen_at"):
        dataclasses.replace(make_worker_snapshot(), last_seen_at=timestamp)


def test_worker_snapshot_rejects_empty_session_id() -> None:
    with pytest.raises(ValueError, match="session_id"):
        dataclasses.replace(make_worker_snapshot(), session_id="")


def test_cluster_snapshot_rejects_naive_created_at() -> None:
    with pytest.raises(ValueError, match="created_at"):
        dataclasses.replace(make_cluster_snapshot(), created_at=datetime(2026, 9, 5, 12, 0, 0))


def test_cluster_snapshot_rejects_duplicate_workers() -> None:
    with pytest.raises(ValueError, match="duplicate worker_id"):
        make_cluster_snapshot(("worker-a", "worker-a"))


def test_cluster_snapshot_rejects_empty_snapshot_id() -> None:
    with pytest.raises(ValueError, match="snapshot_id"):
        dataclasses.replace(make_cluster_snapshot(), snapshot_id="")


def test_snapshot_is_immutable_under_new_state() -> None:
    """Heartbeats after snapshot creation must not mutate it (spec §38, §51)."""
    worker_id = "worker-1"
    snapshot_a = make_cluster_snapshot((worker_id,))
    original_state: WorkerState = snapshot_a.workers[0].state

    # A "new heartbeat" produces new state objects elsewhere.
    new_device = dataclasses.replace(original_state.device_states[0], utilization=99.0)
    new_state = dataclasses.replace(original_state, device_states=(new_device,))

    snapshot_b = ClusterSnapshot(
        snapshot_id="snapshot-2",
        created_at=CREATED_AT,
        workers=(dataclasses.replace(snapshot_a.workers[0], state=new_state),),
    )

    assert snapshot_a.workers[0].state is original_state
    assert snapshot_a.workers[0].state.device_states[0].utilization == 21.0
    assert snapshot_b.workers[0].state.device_states[0].utilization == 99.0

    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot_a.workers = ()  # type: ignore[misc]


def test_snapshot_contains_facts_only_shape() -> None:
    """Sanity: snapshots carry reported facts, nothing derived (spec §39)."""
    snapshot = make_cluster_snapshot()
    worker = snapshot.workers[0]
    gpu_state: DeviceState = worker.state.device_states[0]
    assert gpu_state.utilization == 21.0
    assert worker.state.memory_states[0].available_bytes == 18 * 2**30


def test_snapshot_rejects_device_state_for_unknown_device() -> None:
    snapshot = make_worker_snapshot()
    ghost = DeviceState(
        device_id="ghost-device",
        utilization=None,
        temperature_c=None,
        power_w=None,
        availability=DeviceAvailability.UNKNOWN,
        running_runtime_ids=(),
    )
    broken_state = dataclasses.replace(
        snapshot.state, device_states=(*snapshot.state.device_states, ghost)
    )
    with pytest.raises(ValueError, match="unknown device"):
        dataclasses.replace(snapshot, state=broken_state)


def test_snapshot_rejects_memory_state_for_unknown_pool() -> None:
    snapshot = make_worker_snapshot()
    ghost = MemoryPoolState(memory_pool_id="ghost-pool", available_bytes=None)
    broken_state = dataclasses.replace(
        snapshot.state, memory_states=(*snapshot.state.memory_states, ghost)
    )
    with pytest.raises(ValueError, match="unknown memory pool"):
        dataclasses.replace(snapshot, state=broken_state)


def test_snapshot_rejects_runtime_instance_on_unknown_device() -> None:
    snapshot = make_worker_snapshot()
    broken_instance = dataclasses.replace(
        snapshot.state.runtime_instances[0], device_ids=("ghost-device",)
    )
    broken_state = dataclasses.replace(
        snapshot.state, runtime_instances=(broken_instance,)
    )
    with pytest.raises(ValueError, match="unknown device"):
        dataclasses.replace(snapshot, state=broken_state)
