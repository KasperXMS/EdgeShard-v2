"""Dynamic Worker/Device state domain model (Phase 1 spec §17-18).

State is what changes between heartbeats: device telemetry, memory-pool
availability, observed runtime instances, and model inventory. Missing
telemetry is always ``None`` — an absent ``power_w`` never implies the device
is unavailable (spec §17).

These are Worker-reported facts. Master-local bookkeeping such as monotonic
receive timestamps and liveness thresholds must never be added here (spec
§18); that lives in Master-side records.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from edgeshard.cluster.inventory import ModelInventoryEntry, RuntimeInstanceState


class DeviceAvailability(StrEnum):
    """Availability classification reported per device (spec §17)."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DeviceState:
    """One dynamic device telemetry sample (spec §17).

    ``utilization`` is reported as a percentage (0-100) when the backend
    provides it. Any metric the backend cannot supply is ``None``.
    """

    device_id: str

    utilization: float | None
    temperature_c: float | None
    power_w: float | None

    availability: DeviceAvailability

    running_runtime_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.device_id:
            raise ValueError("device_id must not be empty")
        if self.utilization is not None and not 0.0 <= self.utilization <= 100.0:
            raise ValueError(f"utilization must be within [0, 100], got {self.utilization}")


@dataclass(frozen=True)
class MemoryPoolState:
    """Dynamic availability of one physical memory pool (spec §12)."""

    memory_pool_id: str
    available_bytes: int | None

    def __post_init__(self) -> None:
        if not self.memory_pool_id:
            raise ValueError("memory_pool_id must not be empty")
        if self.available_bytes is not None and self.available_bytes < 0:
            raise ValueError(f"available_bytes must not be negative, got {self.available_bytes}")


class WorkerStatus(StrEnum):
    """Master-assigned liveness status of a Worker (spec §32).

    Derived by the Master from heartbeat receipt; it is carried on
    Master-side records such as ``edgeshard.cluster.snapshot.WorkerSnapshot``
    and never on Worker-reported state.
    """

    ONLINE = "online"
    SUSPECT = "suspect"
    OFFLINE = "offline"


@dataclass(frozen=True)
class WorkerState:
    """Latest Worker-reported dynamic state (spec §18).

    Worker-reported facts only. Master-assigned bookkeeping — liveness
    status, registration session, receive timestamps — lives in Master-side
    records (see ``edgeshard.cluster.snapshot.WorkerSnapshot``), never here.
    """

    worker_id: str

    device_states: tuple[DeviceState, ...]
    memory_states: tuple[MemoryPoolState, ...]

    runtime_instances: tuple[RuntimeInstanceState, ...]
    models: tuple[ModelInventoryEntry, ...]

    def __post_init__(self) -> None:
        if not self.worker_id:
            raise ValueError("worker_id must not be empty")

        device_ids = [state.device_id for state in self.device_states]
        if len(set(device_ids)) != len(device_ids):
            raise ValueError("duplicate device_id in device_states")

        pool_ids = [state.memory_pool_id for state in self.memory_states]
        if len(set(pool_ids)) != len(pool_ids):
            raise ValueError("duplicate memory_pool_id in memory_states")
