"""StateStore tests (Phase 1 spec §36, §31)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from edgeshard.cluster.state import WorkerState
from edgeshard.control.master.state_store import StateStore
from factories import make_worker_state

WORKER = "worker-1"
WALL = datetime(2026, 9, 5, 12, 0, 0, tzinfo=UTC)


def record(store: StateStore, worker_id: str, state: WorkerState, monotonic: float) -> None:
    store.record(worker_id, state, monotonic=monotonic, wall=WALL)


def test_only_latest_state_is_kept() -> None:
    """§36: latest valid state only — this is not a telemetry history database."""
    store = StateStore()
    first = make_worker_state(WORKER)
    second = dataclasses.replace(first, runtime_instances=())

    record(store, WORKER, first, monotonic=100.0)
    record(store, WORKER, second, monotonic=105.0)

    entry = store.get(WORKER)
    assert entry is not None
    assert entry.state == second
    assert entry.received_monotonic == 105.0


def test_both_timestamps_maintained() -> None:
    """§31: monotonic for liveness, wall clock for debugging."""
    store = StateStore()
    record(store, WORKER, make_worker_state(WORKER), monotonic=42.5)

    entry = store.get(WORKER)
    assert entry is not None
    assert entry.received_monotonic == 42.5
    assert entry.received_wall == WALL


def test_unknown_worker_returns_none() -> None:
    assert StateStore().get("never-seen") is None
    assert StateStore().items() == ()


def test_items_sorted_by_worker_id() -> None:
    store = StateStore()
    for worker_id in ("w-3", "w-1", "w-2"):
        record(store, worker_id, make_worker_state(worker_id), monotonic=1.0)

    assert [worker_id for worker_id, _ in store.items()] == ["w-1", "w-2", "w-3"]


def test_state_worker_id_mismatch_rejected() -> None:
    store = StateStore()
    with pytest.raises(ValueError, match="worker_id mismatch"):
        record(store, "w-1", make_worker_state("w-2"), monotonic=1.0)


def test_naive_wall_clock_rejected() -> None:
    store = StateStore()
    with pytest.raises(ValueError, match="timezone-aware"):
        store.record(
            WORKER,
            make_worker_state(WORKER),
            monotonic=1.0,
            wall=datetime(2026, 9, 5, 12, 0, 0),
        )
