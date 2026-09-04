"""Runtime and model inventory domain model (Phase 1 spec §19-20).

Observational inventory of what already exists on a Worker: EdgeShard-managed
runtime containers (reconstructed from Docker labels/state, never assumed to
be in Agent memory) and ModelStore snapshots. Phase 1 inventory is read-only;
starting, stopping, downloading, or evicting anything is out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RuntimeStatus(StrEnum):
    """Observed lifecycle state of one managed runtime (spec §19)."""

    CREATED = "created"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RuntimeInstanceState:
    """One observed EdgeShard-managed runtime instance (spec §19)."""

    runtime_id: str
    backend: str

    execution_id: str | None

    status: RuntimeStatus

    device_ids: tuple[str, ...]

    container_id: str | None
    endpoint: str | None

    model_local_name: str | None

    def __post_init__(self) -> None:
        if not self.runtime_id:
            raise ValueError("runtime_id must not be empty")
        if not self.backend:
            raise ValueError("backend must not be empty")


class ModelAvailability(StrEnum):
    """Integrity classification of one ModelStore entry (spec §20)."""

    READY = "ready"
    INCOMPLETE = "incomplete"
    INVALID = "invalid"


@dataclass(frozen=True)
class ModelInventoryEntry:
    """One model snapshot observed in the Worker's ModelStore (spec §20).

    Only logical model identity and status cross to the Master; host paths
    never do.
    """

    local_name: str

    model_id: str | None
    revision: str | None

    size_bytes: int | None

    status: ModelAvailability

    def __post_init__(self) -> None:
        if not self.local_name:
            raise ValueError("local_name must not be empty")
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError(f"size_bytes must not be negative, got {self.size_bytes}")
