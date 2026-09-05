"""Host capability probe: stdlib + psutil only (Phase 1 spec §21).

Facts true of every development/production host: architecture, OS, CPU
device, host-memory pool, network interfaces, and the container runtime.
GPU and platform specifics come from the NVIDIA/Jetson probes in milestones
P1C/P1D, whose fragments merge alongside this one.

The CPU vendor/model strings are intentionally coarse on generic hosts
(``platform``-derived); hardware backends report precise values. A missing
Docker daemon is non-fatal (spec §47): the container runtime is reported as
absent instead.
"""

from __future__ import annotations

import logging
import platform
import socket
from collections.abc import Callable
from typing import Any

import docker
import psutil

from edgeshard.cluster.capability import (
    ContainerRuntimeCapability,
    DeviceCapability,
    MemoryModel,
    MemoryPoolCapability,
    NetworkInterfaceCapability,
    OSInfo,
)
from edgeshard.cluster.identity import DeviceIdentity, DeviceKind
from edgeshard.control.worker.discovery.base import CapabilityFragment
from edgeshard.control.worker.identity import derive_cpu_device_id

logger = logging.getLogger("worker.discovery.host")

HOST_MEMORY_POOL_ID = "host-memory"
"""Pool id of host RAM on non-Jetson hosts (spec §13)."""

_MACHINE_ALIASES = {"amd64": "x86_64", "x64": "x86_64", "arm64": "aarch64"}
"""Windows/macOS spellings of ``platform.machine()`` mapped to uname form."""

_LINK_FAMILIES: set[int] = set()
if hasattr(socket, "AF_PACKET"):
    _LINK_FAMILIES.add(socket.AF_PACKET)  # Linux MAC addresses
_PSUTIL_AF_LINK = getattr(psutil, "AF_LINK", None)
if isinstance(_PSUTIL_AF_LINK, int):
    _LINK_FAMILIES.add(_PSUTIL_AF_LINK)  # Windows/BSD MAC addresses


def canonical_architecture(machine: str) -> str:
    """Normalize ``platform.machine()`` spellings (AMD64/ARM64) to uname form."""
    lowered = machine.lower()
    return _MACHINE_ALIASES.get(lowered, lowered)


class HostCapabilityProbe:
    """Standard-library + psutil host discovery (spec §21)."""

    def __init__(
        self,
        worker_id: str,
        docker_client_factory: Callable[[], Any] | None = None,
        *,
        memory_pool_id: str = HOST_MEMORY_POOL_ID,
    ) -> None:
        self._worker_id = worker_id
        self._docker_client_factory = docker_client_factory or docker.from_env
        # Jetson hosts reuse this probe with the shared system-memory pool id
        # (spec §14); discrete/host hosts keep the default.
        self._memory_pool_id = memory_pool_id

    def discover(self) -> CapabilityFragment:
        architecture = canonical_architecture(platform.machine())
        return CapabilityFragment(
            architecture=architecture,
            os=self._discover_os(),
            container_runtime=self._discover_container_runtime(),
            network_interfaces=self._discover_network_interfaces(),
            devices=(self._cpu_device(architecture),),
            memory_pools=(
                MemoryPoolCapability(
                    memory_pool_id=self._memory_pool_id,
                    model=MemoryModel.SHARED,
                    total_bytes=int(psutil.virtual_memory().total),
                ),
            ),
            # Runtime platforms (backend/image facts) are discovered once the
            # container-image plumbing lands; P1B reports none rather than
            # guessing.
            runtime_platforms=(),
        )

    def _discover_os(self) -> OSInfo:
        name = platform.system()
        version: str | None = None
        try:
            freedesktop = platform.freedesktop_os_release()
        except OSError:
            pass  # non-freedesktop host (Windows, older macOS): system() name
        else:
            name = str(freedesktop.get("ID") or name)
            version_id = freedesktop.get("VERSION_ID")
            version = str(version_id) if version_id else None
        return OSInfo(name=name, version=version, kernel=platform.release() or None)

    def _cpu_device(self, architecture: str) -> DeviceCapability:
        machine = platform.machine() or "unknown"
        return DeviceCapability(
            identity=DeviceIdentity(
                device_id=derive_cpu_device_id(self._worker_id),
                kind=DeviceKind.CPU,
                local_locator="cpu",
            ),
            vendor=machine,
            model=platform.processor() or machine,
            compute_capability=None,
            memory_pool_id=self._memory_pool_id,
            # Phase 0 CPU runtimes execute fp32 shards (containers/hf CPU image).
            supported_dtypes=("fp32",),
            driver_version=None,
            platform_tags=(architecture,),
        )

    def _discover_network_interfaces(self) -> tuple[NetworkInterfaceCapability, ...]:
        stats = psutil.net_if_stats()
        interfaces: list[NetworkInterfaceCapability] = []
        for name, snics in psutil.net_if_addrs().items():
            addresses: list[str] = []
            mac: str | None = None
            for snic in snics:
                if snic.family in (socket.AF_INET, socket.AF_INET6) and snic.address:
                    addresses.append(snic.address)
                elif snic.family in _LINK_FAMILIES and snic.address:
                    # Link-layer entry: the MAC address. Windows formats it
                    # with dashes; normalize for stable identity (spec §58).
                    mac = str(snic.address).replace("-", ":").lower()
            interface_id = mac or f"if-{name}"
            nic_stats = stats.get(name)
            interfaces.append(
                NetworkInterfaceCapability(
                    interface_id=interface_id,
                    name=name,
                    addresses=tuple(addresses),
                    mtu=int(nic_stats.mtu) if nic_stats is not None else None,
                )
            )
        return tuple(interfaces)

    def _discover_container_runtime(self) -> ContainerRuntimeCapability | None:
        try:
            client = self._docker_client_factory()
            version_info = client.version()
            runtime_info = client.info()
        except Exception as exc:  # daemon missing is non-fatal (spec §47)
            logger.warning("container runtime not discovered: %s", exc)
            return None
        runtimes = runtime_info.get("Runtimes") or {}
        version = version_info.get("Version")
        return ContainerRuntimeCapability(
            runtime="docker",
            version=str(version) if version else None,
            nvidia_runtime_available="nvidia" in runtimes,
        )
