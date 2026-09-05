"""Stable Worker information registry (Phase 1 spec §34).

The registry stores the slowly-changing facts about every Worker that has
ever registered: ``WorkerIdentity``, ``WorkerCapability`` and, carried on
the capability, its ``capability_revision``. Volatile data (sessions,
heartbeat sequences, dynamic state, liveness) deliberately lives in the
other Master components — the registry is what still identifies a Worker
after it goes OFFLINE.

Phase 1 is purely in-memory (spec §34: no database). OFFLINE Workers stay
in the registry forever; nothing is deleted automatically (spec §32).
"""

from __future__ import annotations

from dataclasses import dataclass

from edgeshard.cluster.capability import WorkerCapability
from edgeshard.cluster.identity import WorkerIdentity


@dataclass(frozen=True)
class WorkerRecord:
    """Stable information about one registered Worker (spec §34)."""

    identity: WorkerIdentity
    capability: WorkerCapability

    def __post_init__(self) -> None:
        if not self.identity.worker_id:
            raise ValueError("worker_id must not be empty")

    @property
    def worker_id(self) -> str:
        return self.identity.worker_id

    @property
    def capability_revision(self) -> str:
        """Fingerprint of the stored capability (spec §16)."""
        return self.capability.capability_revision


class WorkerRegistry:
    """In-memory store of stable Worker information (spec §34).

    Re-registration of a known ``worker_id`` updates the same entry
    (spec §52 Test C: "Worker is still same registry entry"); the
    ``worker_id`` is the only key — it is never derived from hostname,
    IP, or any other volatile fact (spec §10).
    """

    def __init__(self) -> None:
        self._records: dict[str, WorkerRecord] = {}

    def upsert(
        self, identity: WorkerIdentity, capability: WorkerCapability
    ) -> WorkerRecord:
        """Insert or replace the record for ``identity.worker_id``."""
        record = WorkerRecord(identity=identity, capability=capability)
        self._records[record.worker_id] = record
        return record

    def get(self, worker_id: str) -> WorkerRecord:
        """Return the record or raise ``KeyError`` if the Worker never registered."""
        return self._records[worker_id]

    def find(self, worker_id: str) -> WorkerRecord | None:
        """Return the record or ``None``; the existence-check flavor of :meth:`get`."""
        return self._records.get(worker_id)

    def list_workers(self) -> tuple[WorkerRecord, ...]:
        """All registered Workers, including OFFLINE ones, ordered by worker_id."""
        return tuple(self._records[key] for key in sorted(self._records))

    def __len__(self) -> int:
        return len(self._records)
