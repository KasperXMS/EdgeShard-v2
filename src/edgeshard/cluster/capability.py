"""Static Worker/Device capability domain model (Phase 1 spec §12-16).

Capabilities are facts about what a Worker *is*: architecture, OS, container
runtime, network interfaces, devices, memory pools, runtime platforms. They
are distinct from dynamic state (``state.py``): capabilities change rarely,
which is why they are fingerprinted into a ``capability_revision`` (spec §16)
instead of being retransmitted on every heartbeat.

Memory is modeled as explicit physical pools, not as properties owned by
devices (spec §12-14). On a discrete-GPU host the CPU's host memory and each
GPU's VRAM are separate pools; on Jetson the CPU and the integrated GPU share
one ``system-memory`` pool so later schedulers never double-count it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from typing import cast

from edgeshard.cluster.identity import DeviceIdentity


class MemoryModel(StrEnum):
    """How a memory pool is physically attached to devices (spec §12)."""

    DISCRETE = "discrete"
    """Pool dedicated to one device (e.g. discrete GPU VRAM)."""

    SHARED = "shared"
    """Pool shared by multiple devices (host RAM, Jetson unified memory)."""


@dataclass(frozen=True)
class MemoryPoolCapability:
    """Static description of one physical memory pool (spec §12)."""

    memory_pool_id: str
    model: MemoryModel
    total_bytes: int

    def __post_init__(self) -> None:
        if not self.memory_pool_id:
            raise ValueError("memory_pool_id must not be empty")
        if self.total_bytes <= 0:
            raise ValueError(f"total_bytes must be positive, got {self.total_bytes}")


@dataclass(frozen=True)
class OSInfo:
    """Operating system identification (spec §15)."""

    name: str
    version: str | None
    kernel: str | None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("os name must not be empty")


@dataclass(frozen=True)
class ContainerRuntimeCapability:
    """Container engine available on the Worker host (spec §15)."""

    runtime: str
    version: str | None
    nvidia_runtime_available: bool

    def __post_init__(self) -> None:
        if not self.runtime:
            raise ValueError("container runtime name must not be empty")


@dataclass(frozen=True)
class NetworkInterfaceCapability:
    """One network interface of the Worker host (spec §15)."""

    interface_id: str
    name: str
    addresses: tuple[str, ...]
    mtu: int | None

    def __post_init__(self) -> None:
        if not self.interface_id:
            raise ValueError("interface_id must not be empty")
        if not self.name:
            raise ValueError("interface name must not be empty")


@dataclass(frozen=True)
class RuntimePlatformCapability:
    """One runtime backend the Worker can host (spec §15).

    E.g. the EdgeShard shard runtime or vLLM, with the platform/image they
    run on.
    """

    backend: str
    platform: str
    image: str | None

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend must not be empty")
        if not self.platform:
            raise ValueError("platform must not be empty")


@dataclass(frozen=True)
class DeviceCapability:
    """Static description of one device (spec §15).

    ``memory_pool_id`` names the physical memory pool this device draws from;
    multiple devices may reference the same shared pool (Jetson CPU+GPU).
    ``None`` means the device has no addressable memory pool (rare).
    """

    identity: DeviceIdentity

    vendor: str
    model: str

    compute_capability: str | None

    memory_pool_id: str | None

    supported_dtypes: tuple[str, ...]

    driver_version: str | None

    platform_tags: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.vendor:
            raise ValueError("device vendor must not be empty")
        if not self.model:
            raise ValueError("device model must not be empty")


@dataclass(frozen=True)
class WorkerCapability:
    """Complete static capability of one Worker (spec §15).

    Constructed with a placeholder ``capability_revision`` and finalized via
    :func:`compute_capability_revision` plus ``dataclasses.replace`` — the
    revision fingerprints the capability content excluding itself.

    Construction validates internal consistency loudly (spec §47): duplicate
    device/pool identities and dangling memory-pool references are rejected.
    """

    architecture: str
    os: OSInfo

    container_runtime: ContainerRuntimeCapability | None

    network_interfaces: tuple[NetworkInterfaceCapability, ...]

    devices: tuple[DeviceCapability, ...]

    memory_pools: tuple[MemoryPoolCapability, ...]

    runtime_platforms: tuple[RuntimePlatformCapability, ...]

    capability_revision: str

    def __post_init__(self) -> None:
        if not self.architecture:
            raise ValueError("architecture must not be empty")

        pool_ids = [pool.memory_pool_id for pool in self.memory_pools]
        if len(set(pool_ids)) != len(pool_ids):
            raise ValueError("duplicate memory_pool_id in memory_pools")

        device_ids = [device.identity.device_id for device in self.devices]
        if len(set(device_ids)) != len(device_ids):
            raise ValueError("duplicate device_id in devices")

        known_pools = set(pool_ids)
        for device in self.devices:
            pool_id = device.memory_pool_id
            if pool_id is not None and pool_id not in known_pools:
                raise ValueError(
                    f"device {device.identity.device_id!r} references unknown "
                    f"memory pool {pool_id!r}"
                )


def _canonical_value(value: object) -> object:
    """Convert a capability structure to plain JSON-serializable data.

    Dataclasses become mappings, tuples become arrays, enums become their
    values; tuple ordering is preserved so it participates in the revision.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _canonical_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    return value


def compute_capability_revision(capability: WorkerCapability) -> str:
    """Deterministic SHA-256 fingerprint of the static capability (spec §16).

    The ``capability_revision`` field itself is excluded so the revision can
    fingerprint a capability built with a placeholder revision. Identical
    canonical capabilities always yield the same revision; any content change
    yields a different one.
    """
    mapping = cast("dict[str, object]", _canonical_value(capability))
    mapping.pop("capability_revision", None)
    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
