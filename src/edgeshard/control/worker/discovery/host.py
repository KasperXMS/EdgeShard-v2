"""Host capability probe: stdlib + psutil only (Phase 1 spec §21).

Facts true of every development/production host: architecture, OS, CPU
device, host-memory pool, network interfaces, and the container runtime.
GPU and platform specifics come from the NVIDIA/Jetson probes in milestones
P1C/P1D, whose fragments merge alongside this one.

The CPU vendor/model strings are intentionally coarse on generic hosts
(``platform``-derived); hardware backends report precise values. A missing
Docker daemon is non-fatal (spec §47): the container runtime is reported as
absent instead.

Network discovery reports only *static* host facts (spec §15-16): container
and Pod interfaces (``veth*``, ``docker0``, ``br-*``, CNI/Kubernetes
bridges, …) come and go with every workload and would churn the
``capability_revision`` of an otherwise unchanged host, so they are
filtered; physical NICs and host-level overlays (ZeroTier, Tailscale,
WireGuard, …) are kept. Interface ids must be stable *and* unique: a
usable MAC becomes the id, interfaces whose MAC is invalid (all-zero) or
shared (bonded/aliased NICs) fall back to their name instead.
"""

from __future__ import annotations

import logging
import platform
import re
import socket
from collections import Counter
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

_LOOPBACK_NAMES = frozenset({"lo", "lo0"})
"""Unix loopback interface names: never a static host fact."""

_TRANSIENT_INTERFACE_PREFIXES = (
    "docker",  # Docker NAT bridge (docker0)
    "br-",  # Docker bridge networks (br-<hex>); NOT plain "br0" LAN bridges
    "veth",  # container veth pairs; also Windows "vEthernet" (Hyper-V/WSL)
    "cali",  # Calico CNI endpoints (Kubernetes)
    "tunl",  # Calico IPIP tunnels (Kubernetes)
    "cni",  # CNI bridges (containerd/Kubernetes/podman)
    "flannel",  # Flannel overlay (Kubernetes)
    "kube",  # Kubernetes bridges
    "podman",  # Podman bridge
    "virbr",  # libvirt NAT bridges
    "vnet",  # libvirt/KVM tap devices
    "loopback pseudo-interface",  # Windows loopback
)
"""Lowercased name prefixes of container/Pod/virtualization interfaces.

These appear and disappear with workloads; counting them in the static
capability would churn ``capability_revision`` (§16). Physical NICs and
host-level overlays (``zt*`` ZeroTier, ``tailscale*``, ``wg*`` WireGuard,
``utun*``) deliberately match none of them and are kept.
"""

_MAC_RE = re.compile(r"(?:[0-9a-f]{2}:){5}[0-9a-f]{2}")
_NULL_MAC = "00:00:00:00:00:00"


def canonical_architecture(machine: str) -> str:
    """Normalize ``platform.machine()`` spellings (AMD64/ARM64) to uname form."""
    lowered = machine.lower()
    return _MACHINE_ALIASES.get(lowered, lowered)


def _is_transient_interface(name: str) -> bool:
    """Container/Pod/loopback interfaces are not static capability (§15-16)."""
    if name in _LOOPBACK_NAMES:
        return True
    return name.lower().startswith(_TRANSIENT_INTERFACE_PREFIXES)


def _is_valid_mac(candidate: str) -> bool:
    """A usable link-layer address: well-formed, lowercased, and not all-zero.

    Virtual adapters commonly report ``00:00:00:00:00:00``; treating it as
    identity would collide across interfaces, so it never becomes a stable
    id (§11: ids must be stable *and* unique).
    """
    return bool(_MAC_RE.fullmatch(candidate)) and candidate != _NULL_MAC


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
            # Runtime platforms are operator-declared via worker config
            # (runtime.platforms) and merged in by the inspector; a probe
            # never guesses which engines are actually provisioned (§15).
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
        kept: list[tuple[str, tuple[str, ...], str | None]] = []
        for name, snics in psutil.net_if_addrs().items():
            if _is_transient_interface(name):
                continue
            addresses: list[str] = []
            mac: str | None = None
            for snic in snics:
                if snic.family in (socket.AF_INET, socket.AF_INET6) and snic.address:
                    addresses.append(snic.address)
                elif snic.family in _LINK_FAMILIES and snic.address and mac is None:
                    # Link-layer entry: the MAC address. Windows formats it
                    # with dashes; normalize for stable identity (spec §58).
                    # Invalid MACs (all-zero, malformed) are never ids.
                    candidate = str(snic.address).replace("-", ":").lower()
                    if _is_valid_mac(candidate):
                        mac = candidate
            kept.append((name, tuple(addresses), mac))

        # A MAC is a stable identifier only while exactly one kept interface
        # carries it: bonded/aliased NICs legitimately share one, and picking
        # an enumeration-order winner would churn the revision on reboot.
        mac_uses = Counter(mac for _, _, mac in kept if mac is not None)

        interfaces: list[NetworkInterfaceCapability] = []
        for name, ip_addresses, mac in kept:
            interface_id = mac if mac is not None and mac_uses[mac] == 1 else f"if-{name}"
            nic_stats = stats.get(name)
            interfaces.append(
                NetworkInterfaceCapability(
                    interface_id=interface_id,
                    name=name,
                    addresses=ip_addresses,
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
