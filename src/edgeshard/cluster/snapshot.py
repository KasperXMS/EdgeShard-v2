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
from edgeshard.cluster.state import WorkerState, WorkerStatus


def _require_aware(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{field} must be timezone-aware")


@dataclass(frozen=True)
class WorkerSnapshot:
    """Everything known about one Worker at snapshot time (spec §38).

    ``identity``, ``capability`` and ``state`` are Worker-reported facts;
    ``status``, ``session_id`` and ``last_seen_at`` are Master-assigned
    bookkeeping layered on top. Construction cross-validates the reported
    state against the capability and fails loudly on any dangling reference.
    """

    identity: WorkerIdentity
    capability: WorkerCapability
    state: WorkerState

    status: WorkerStatus
    """Master-assigned liveness (spec §32); never reported by the Worker."""

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

        known_devices = {device.identity.device_id for device in self.capability.devices}
        known_pools = {pool.memory_pool_id for pool in self.capability.memory_pools}

        for device_state in self.state.device_states:
            if device_state.device_id not in known_devices:
                raise ValueError(
                    f"device state references unknown device {device_state.device_id!r}"
                )
        for memory_state in self.state.memory_states:
            if memory_state.memory_pool_id not in known_pools:
                raise ValueError(
                    f"memory state references unknown memory pool "
                    f"{memory_state.memory_pool_id!r}"
                )
        for instance in self.state.runtime_instances:
            unknown = [
                device_id
                for device_id in instance.device_ids
                if device_id not in known_devices
            ]
            if unknown:
                raise ValueError(
                    f"runtime instance {instance.runtime_id!r} references "
                    f"unknown device(s): {', '.join(repr(u) for u in unknown)}"
                )


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
