"""Latest accepted Worker state store (Phase 1 spec §36, §31).

The StateStore keeps exactly one entry per Worker: the latest *valid*
(accepted) dynamic state plus the Master-local receive timestamps of that
acceptance. It is explicitly not a historical telemetry database (spec
§36) — each accepted heartbeat replaces the previous entry wholesale.

Two timestamps are maintained per spec §31:

* ``received_monotonic`` — ``time.monotonic()`` at receive time. This is
  the *only* value liveness logic may use; Worker wall clocks are never
  trusted.
* ``received_wall`` — the Master's own UTC wall clock at receive time,
  a debugging aid surfaced as ``WorkerSnapshot.last_seen_at``.

The store never samples clocks itself: ``MasterService`` captures both
values at receive time and passes them in, keeping this component a pure
record and tests deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from edgeshard.cluster.state import WorkerState


@dataclass(frozen=True)
class StoredState:
    """One Worker's latest accepted state with Master-side timestamps (§31, §36)."""

    state: WorkerState
    received_monotonic: float
    received_wall: datetime

    def __post_init__(self) -> None:
        if self.received_wall.tzinfo is None or self.received_wall.tzinfo.utcoffset(
            self.received_wall
        ) is None:
            raise ValueError("received_wall must be timezone-aware")


class StateStore:
    """In-memory latest-state-per-Worker store (spec §36)."""

    def __init__(self) -> None:
        self._entries: dict[str, StoredState] = {}

    def record(
        self,
        worker_id: str,
        state: WorkerState,
        *,
        monotonic: float,
        wall: datetime,
    ) -> StoredState:
        """Replace the Worker's stored state; only accepted states arrive here."""
        if state.worker_id != worker_id:
            raise ValueError(
                f"state worker_id mismatch: {worker_id!r} vs {state.worker_id!r}"
            )
        entry = StoredState(state=state, received_monotonic=monotonic, received_wall=wall)
        self._entries[worker_id] = entry
        return entry

    def get(self, worker_id: str) -> StoredState | None:
        """The Worker's latest accepted entry, or ``None`` if never recorded."""
        return self._entries.get(worker_id)

    def items(self) -> tuple[tuple[str, StoredState], ...]:
        """Every stored entry, ordered by worker_id."""
        return tuple(sorted(self._entries.items()))
