"""Cluster identity types (Phase 1 spec §10-11).

Three distinct identity scopes that must never be conflated:

* ``worker_id`` — persistent identity of one Worker installation, a UUIDv4
  string generated on first start and reused across Agent restarts. It must
  never be derived from hostname, IP address, MAC address, or CUDA ordinal.
* ``instance_id`` — one Worker Agent process; carried in wire requests, not
  modeled here.
* ``session_id`` — awarded by the Master per registration; carried in wire
  requests, not modeled here.

Device identity must be stable across reboots and enumeration order (spec
§11, §58): the NVIDIA GPU UUID for discrete GPUs, a persistent worker-derived
key for integrated Jetson devices. ``cuda:0``-style ordinals are never stable
ids; the host-local handle is carried separately as informational
``local_locator``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


@dataclass(frozen=True)
class WorkerIdentity:
    """Stable identity of one Worker installation (spec §10.1)."""

    worker_id: str
    """Persistent UUIDv4 string, loaded from or written to the identity file."""

    hostname: str
    agent_version: str
    protocol_version: str

    def __post_init__(self) -> None:
        if not self.worker_id:
            raise ValueError("worker_id must not be empty")
        if not self.hostname:
            raise ValueError("hostname must not be empty")
        if not self.agent_version:
            raise ValueError("agent_version must not be empty")
        if not self.protocol_version:
            raise ValueError("protocol_version must not be empty")


class DeviceKind(StrEnum):
    """Coarse device classification (spec §11)."""

    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    OTHER = "other"


@dataclass(frozen=True)
class DeviceIdentity:
    """Stable identity of one device (spec §11).

    ``device_id`` must remain valid across reboots and re-enumeration so
    later profiling phases can reference it (spec §58).
    """

    device_id: str
    kind: DeviceKind
    local_locator: str
    """Host-local handle (e.g. a PCI address or CUDA ordinal).

    Informational only — never used as identity and never stable.
    """

    def __post_init__(self) -> None:
        if not self.device_id:
            raise ValueError("device_id must not be empty")
        if not self.local_locator:
            raise ValueError("local_locator must not be empty")
