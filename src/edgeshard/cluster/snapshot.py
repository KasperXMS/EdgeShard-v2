"""Immutable cluster snapshot domain model (Phase 1 spec §38-39).

A ``ClusterSnapshot`` is a point-in-time copy of everything the Master
knows: identity, capability, and latest state of every registered Worker.
Snapshots contain facts only — no fitting estimates, predictions, or
placement recommendations; those belong to later profiling/scheduling phases.

All members are frozen dataclasses over tuples, so a snapshot created from
the current registry can never be mutated by subsequent heartbeats.
Timestamps must be timezone-aware (UTC at creation time) so snapshots from
different Masters remain comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from edgeshard.cluster.capability import WorkerCapability
from edgeshard.cluster.identity import WorkerIdentity
from edgeshard.cluster.state import WorkerState


def _require_aware(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field} must be timezone-aware")


@dataclass(frozen=True)
class WorkerSnapshot:
    """Everything known about one Worker at snapshot time (spec §38)."""

    identity: WorkerIdentity
    capability: WorkerCapability
    state: WorkerState

    session_id: str | None
    """Current registration session, or ``None`` if never registered."""

    last_seen_at: datetime | None
    """Master wall-clock time of the last accepted heartbeat (debug aid).

    Not used for liveness — that uses Master-local monotonic timestamps.
    """

    def __post_init__(self) -> None:
        if self.identity.worker_id != self.state.worker_id:
            raise ValueError(
                f"identity/state worker_id mismatch: "
                f"{self.identity.worker_id!r} vs {self.state.worker_id!r}"
            )
        if self.session_id is not None and not self.session_id:
            raise ValueError("session_id must not be empty when present")
        if self.last_seen_at is not None:
            _require_aware(self.last_seen_at, "last_seen_at")


@dataclass(frozen=True)
class ClusterSnapshot:
    """Immutable view of the whole cluster at one instant (spec §38)."""

    snapshot_id: str
    created_at: datetime
    workers: tuple[WorkerSnapshot, ...]

    def __post_init__(self) -> None:
        if not self.snapshot_id:
            raise ValueError("snapshot_id must not be empty")
        _require_aware(self.created_at, "created_at")
        worker_ids = [worker.identity.worker_id for worker in self.workers]
        if len(set(worker_ids)) != len(worker_ids):
            raise ValueError("duplicate worker_id in snapshot")
