"""Static capability probe interface (Phase 1 spec §25).

Probes return ``CapabilityFragment`` values; the Worker Agent merges all
fragments into one ``WorkerCapability`` (``edgeshard.control.worker.agent``).
``None``/empty fields mean "this probe contributes nothing here" - never
"the capability is absent".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from edgeshard.cluster.capability import (
    ContainerRuntimeCapability,
    DeviceCapability,
    MemoryPoolCapability,
    NetworkInterfaceCapability,
    OSInfo,
    RuntimePlatformCapability,
)


@dataclass(frozen=True)
class CapabilityFragment:
    """Static facts one probe contributes to the Worker capability."""

    architecture: str | None = None
    os: OSInfo | None = None
    container_runtime: ContainerRuntimeCapability | None = None
    network_interfaces: tuple[NetworkInterfaceCapability, ...] = ()
    devices: tuple[DeviceCapability, ...] = ()
    memory_pools: tuple[MemoryPoolCapability, ...] = ()
    runtime_platforms: tuple[RuntimePlatformCapability, ...] = ()


class CapabilityProbe(Protocol):
    """Static discovery backend (spec §25): runs once per Agent start."""

    def discover(self) -> CapabilityFragment: ...
