"""LivenessManager tests with a fake clock (Phase 1 spec §32, §37, §51).

No test sleeps in real time for timeouts (spec §37): the monotonic clock
is injected and advanced by hand, so the full ONLINE → SUSPECT → OFFLINE
lifecycle evaluates instantly.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

import pytest

from edgeshard.cluster.state import WorkerStatus
from edgeshard.control.master.config import MasterConfig
from edgeshard.control.master.liveness import LivenessManager
from edgeshard.control.master.state_store import StateStore
from factories import make_worker_state

WALL = datetime(2026, 9, 5, 12, 0, 0, tzinfo=UTC)


class FakeMonotonic:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def set(self, now: float) -> None:
        """Jump to an absolute reading; keeps boundary math float-exact."""
        self.now = now


def make_manager(
    config: MasterConfig | None = None,
) -> tuple[LivenessManager, StateStore, FakeMonotonic]:
    clock = FakeMonotonic()
    store = StateStore()
    manager = LivenessManager(store, config or MasterConfig(), monotonic=clock)
    return manager, store, clock


def heartbeat(store: StateStore, clock: FakeMonotonic, worker_id: str) -> None:
    store.record(worker_id, make_worker_state(worker_id), monotonic=clock(), wall=WALL)


def test_worker_is_online_right_after_heartbeat() -> None:
    manager, store, clock = make_manager()
    heartbeat(store, clock, "w-1")
    assert manager.status_of("w-1") is WorkerStatus.ONLINE


def test_single_missed_heartbeat_stays_online() -> None:
    """§32: one missed heartbeat (5 s interval) must not degrade the Worker."""
    manager, store, clock = make_manager()
    heartbeat(store, clock, "w-1")

    clock.advance(5.0)  # exactly one missed heartbeat at the default interval
    assert manager.status_of("w-1") is WorkerStatus.ONLINE

    clock.advance(5.0)  # boundary: still "within 10 seconds"
    assert manager.status_of("w-1") is WorkerStatus.ONLINE


def test_online_suspect_offline_transitions() -> None:
    """§51 Liveness: ONLINE → SUSPECT → OFFLINE, fake clock only."""
    manager, store, clock = make_manager()
    heartbeat(store, clock, "w-1")  # last seen at t=1000.0

    clock.set(1010.001)  # beyond 10 s → SUSPECT
    assert manager.status_of("w-1") is WorkerStatus.SUSPECT

    clock.set(1015.0)  # 15 s: still within the OFFLINE threshold
    assert manager.status_of("w-1") is WorkerStatus.SUSPECT

    clock.set(1020.0)  # boundary: exactly 20 s is still SUSPECT
    assert manager.status_of("w-1") is WorkerStatus.SUSPECT

    clock.set(1020.001)  # beyond 20 s → OFFLINE
    assert manager.status_of("w-1") is WorkerStatus.OFFLINE

    # §32: OFFLINE is a status, not an eviction — the entry remains.
    clock.set(4620.0)
    assert manager.status_of("w-1") is WorkerStatus.OFFLINE

    # A returning Worker's heartbeat restores ONLINE without re-registration.
    heartbeat(store, clock, "w-1")
    assert manager.status_of("w-1") is WorkerStatus.ONLINE


def test_thresholds_are_configurable() -> None:
    """§32/§52 Test E: shortened test-config thresholds (binary-exact steps)."""
    config = MasterConfig(
        heartbeat_interval_ms=100, suspect_after_ms=250, offline_after_ms=500
    )
    manager, store, clock = make_manager(config)
    heartbeat(store, clock, "w-1")  # last seen at t=1000.0

    clock.set(1000.25)  # boundary: exactly suspect_after is still ONLINE
    assert manager.status_of("w-1") is WorkerStatus.ONLINE
    clock.set(1000.251)
    assert manager.status_of("w-1") is WorkerStatus.SUSPECT
    clock.set(1000.5)  # boundary: exactly offline_after is still SUSPECT
    assert manager.status_of("w-1") is WorkerStatus.SUSPECT
    clock.set(1000.501)
    assert manager.status_of("w-1") is WorkerStatus.OFFLINE


def test_status_of_unknown_worker_raises() -> None:
    manager, _store, _clock = make_manager()
    with pytest.raises(KeyError):
        manager.status_of("ghost")


def test_statuses_covers_every_tracked_worker() -> None:
    manager, store, clock = make_manager()
    heartbeat(store, clock, "w-1")
    heartbeat(store, clock, "w-2")
    clock.advance(15.0)
    heartbeat(store, clock, "w-2")  # only w-2 is fresh

    statuses = manager.statuses()
    assert statuses == {"w-1": WorkerStatus.SUSPECT, "w-2": WorkerStatus.ONLINE}


def test_evaluate_logs_transitions_only(caplog) -> None:
    """§46: WARN on SUSPECT, INFO on OFFLINE — and only on change."""
    manager, store, clock = make_manager()
    heartbeat(store, clock, "w-1")

    with caplog.at_level(logging.INFO, logger="master.liveness"):
        manager.evaluate()  # ONLINE (first observation of this worker)
        manager.evaluate()  # no change → silent
        assert [r.levelname for r in caplog.records] == ["INFO"]
        assert "worker_id=w-1 state=online" in caplog.records[0].getMessage()

        caplog.clear()
        clock.advance(15.0)
        manager.evaluate()
        manager.evaluate()  # unchanged → silent
        assert [r.levelname for r in caplog.records] == ["WARNING"]
        assert "worker_id=w-1 state=suspect" in caplog.records[0].getMessage()

        caplog.clear()
        clock.advance(10.0)
        manager.evaluate()
        assert [r.levelname for r in caplog.records] == ["INFO"]
        assert "worker_id=w-1 state=offline" in caplog.records[0].getMessage()


async def test_periodic_task_runs_and_stops(caplog) -> None:
    """§37: a periodic asyncio task drives evaluate; start/stop are idempotent."""
    config = MasterConfig(liveness_tick_ms=1)
    manager, store, clock = make_manager(config)
    heartbeat(store, clock, "w-1")

    with caplog.at_level(logging.INFO, logger="master.liveness"):
        await manager.start()
        await manager.start()  # idempotent
        clock.advance(15.0)
        await asyncio.sleep(0.05)  # loop mechanics only — not a timeout test
        await manager.stop()
        await manager.stop()  # idempotent

    messages = [record.getMessage() for record in caplog.records]
    assert any("worker_id=w-1 state=suspect" in message for message in messages)
