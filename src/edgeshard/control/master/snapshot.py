"""Master-side cluster snapshot production (Phase 1 spec §38-39, §33).

``SnapshotBuilder`` is the fifth Master component (spec §33): it folds the
four state components — registry (stable identity + capability), sessions
(current session_id), state store (latest accepted dynamic state + receive
timestamps), and liveness (derived ONLINE/SUSPECT/OFFLINE) — into one
immutable :class:`~edgeshard.cluster.snapshot.ClusterSnapshot`.

``build`` is deliberately **synchronous**: it never awaits, so on the
Master's single event loop it runs atomically with respect to registration
and heartbeat handlers (spec §34). A snapshot is therefore a true
point-in-time copy — a heartbeat that lands after ``build`` returns cannot
reach into it, because every domain object it holds is frozen and the
StateStore *replaces* entries rather than mutating them (spec §38, §51).

Snapshots contain facts only (spec §39): identity, capability, dynamic
state, liveness, session and last-seen. No fitting estimates, predictions
or placement recommendations — those belong to later phases and must never
be computed here.

The builder reads through the components' public interfaces and owns no
state of its own; ``wall`` and ``snapshot_factory`` are injectable so
``created_at``/``snapshot_id`` are deterministic under test.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from edgeshard.cluster.snapshot import ClusterSnapshot, WorkerSnapshot
from edgeshard.control.master.liveness import LivenessManager
from edgeshard.control.master.registry import WorkerRecord, WorkerRegistry
from edgeshard.control.master.sessions import SessionManager
from edgeshard.control.master.state_store import StateStore


def _utc_now() -> datetime:
    return datetime.now(UTC)


class SnapshotBuilder:
    """Assembles immutable :class:`ClusterSnapshot` views of Master state (§38)."""

    def __init__(
        self,
        registry: WorkerRegistry,
        sessions: SessionManager,
        states: StateStore,
        liveness: LivenessManager,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wall: Callable[[], datetime] = _utc_now,
        snapshot_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
    ) -> None:
        self._registry = registry
        self._sessions = sessions
        self._states = states
        self._liveness = liveness
        self._monotonic = monotonic
        self._wall = wall
        self._snapshot_factory = snapshot_factory

    def build(self) -> ClusterSnapshot:
        """Copy the Master's current knowledge into one frozen snapshot.

        Synchronous and side-effect free: concurrent heartbeats cannot
        interleave (no ``await``), so the result is internally consistent
        and never changes after it is returned (spec §38, §51). Liveness for
        every Worker is classified against a single ``monotonic`` instant, so
        one snapshot never mixes two clock reads.
        """
        now = self._monotonic()
        workers = tuple(
            self._build_worker(record, now) for record in self._registry.list_workers()
        )
        return ClusterSnapshot(
            snapshot_id=self._snapshot_factory(),
            created_at=self._wall(),
            workers=workers,
        )

    def _build_worker(self, record: WorkerRecord, now: float) -> WorkerSnapshot:
        worker_id = record.worker_id
        stored = self._states.get(worker_id)
        if stored is None:
            # Registration always records an initial state before returning,
            # so a registry entry without state is an internal inconsistency
            # (spec §47): fail loudly rather than emit a half-built snapshot.
            raise ValueError(
                f"registry worker {worker_id!r} has no stored state "
                "(internal inconsistency)"
            )
        session = self._sessions.current(worker_id)
        return WorkerSnapshot(
            identity=record.identity,
            capability=record.capability,
            state=stored.state,
            status=self._liveness.status_at(worker_id, now),
            session_id=session.session_id if session is not None else None,
            last_seen_at=stored.received_wall,
        )


def format_snapshot(snapshot: ClusterSnapshot) -> str:
    """Compact human-readable debug representation of a snapshot (spec §45).

    Phase 1 requires only a debug CLI/log rendering of ``ClusterSnapshot``
    (no REST/dashboard). One header line plus one line per Worker, facts
    only — status, capability revision, device/pool/runtime/model counts,
    current session and the Master wall-clock last-seen timestamp.
    """
    header = (
        f"snapshot_id={snapshot.snapshot_id} "
        f"created_at={snapshot.created_at.isoformat()} "
        f"workers={len(snapshot.workers)}"
    )
    lines = [header]
    for worker in snapshot.workers:
        state = worker.state
        lines.append(
            f"  worker_id={worker.identity.worker_id} "
            f"status={worker.status.value} "
            f"revision={worker.capability.capability_revision[:12]} "
            f"devices={len(state.device_states)} "
            f"pools={len(state.memory_states)} "
            f"runtimes={len(state.runtime_instances)} "
            f"models={len(state.models)} "
            f"session_id={worker.session_id or '-'} "
            f"last_seen_at={worker.last_seen_at.isoformat() if worker.last_seen_at else '-'}"
        )
    return "\n".join(lines)
