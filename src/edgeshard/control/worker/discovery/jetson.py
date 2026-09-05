"""Jetson platform capability probe (Phase 1 spec §14, §23).

Jetson is never modeled as an ordinary discrete NVML GPU: its CPU and
integrated GPU share one physical ``system-memory`` pool, so later
schedulers cannot double-count the unified RAM.

``JetsonPlatformProbe`` is the host-equivalent probe for Jetson hosts. It
delegates the generic host facts (architecture, OS, network interfaces,
container runtime, the single SHARED pool) to ``HostCapabilityProbe``
parameterized with the system-memory pool id, then retargets the CPU device
and adds the integrated GPU device whose identity is derived from the
persistent ``worker_id`` (spec §11) because the SoC exposes no hardware
UUID. L4T/platform facts come from ``/etc/nv_tegra_release`` and the device
tree (spec §23); tegrastats strings never reach this module (spec §60).

Detection (``is_jetson_host``) gates agent probe composition: off-Jetson
hosts never construct this probe, and the probe itself is inert about
detection - it reports whatever a host's platform files say.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from edgeshard.cluster.capability import DeviceCapability
from edgeshard.cluster.identity import DeviceIdentity, DeviceKind
from edgeshard.control.worker.discovery.base import CapabilityFragment
from edgeshard.control.worker.discovery.host import HostCapabilityProbe
from edgeshard.control.worker.discovery.nvidia import dtypes_for_compute_capability
from edgeshard.control.worker.identity import derive_jetson_gpu_device_id

logger = logging.getLogger("worker.discovery.jetson")

SYSTEM_MEMORY_POOL_ID = "system-memory"
"""Pool id of Jetson unified memory shared by CPU and GPU (spec §14)."""

NV_TEGRA_RELEASE = Path("/etc/nv_tegra_release")
DEVICE_TREE_COMPATIBLE = Path("/proc/device-tree/compatible")
DEVICE_TREE_MODEL = Path("/proc/device-tree/model")

# L4T release line, e.g. "# R35 (release), REVISION: 4.1, GCID: 36144147,
# BOARD: t186ref, EABI: aarch64".
_L4T_RELEASE_RE = re.compile(r"#\s*R(\d+)\s*\(release\),\s*REVISION:\s*([\d.]+)")

# CUDA compute capability per Jetson generation (the SoC exposes no NVML
# query path we may use - spec §23 forbids modeling Jetson as a discrete
# NVML GPU). Unknown models report None rather than guessing.
_COMPUTE_CAPABILITIES: tuple[tuple[str, tuple[int, int]], ...] = (
    ("orin", (8, 7)),
    ("xavier", (7, 2)),
    ("tx2", (6, 2)),
    ("tx1", (5, 3)),
    ("nano", (5, 3)),
)


def is_jetson_host() -> bool:
    """True on L4T/Jetson hosts (release file or device-tree compatible)."""
    if NV_TEGRA_RELEASE.exists():
        return True
    try:
        return b"nvidia,tegra" in DEVICE_TREE_COMPATIBLE.read_bytes()
    except OSError:
        return False


def read_l4t_release() -> str | None:
    """L4T release version from ``/etc/nv_tegra_release`` (e.g. ``35.4.1``)."""
    try:
        content = NV_TEGRA_RELEASE.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = _L4T_RELEASE_RE.search(content)
    if match is None:
        logger.warning("could not parse L4T release from %s", NV_TEGRA_RELEASE)
        return None
    return f"{match.group(1)}.{match.group(2)}"


def read_device_tree_model() -> str | None:
    """Board model from the device tree (e.g. ``NVIDIA AGX Orin ...``)."""
    try:
        raw = DEVICE_TREE_MODEL.read_bytes()
    except OSError:
        return None
    model = raw.split(b"\x00", 1)[0].decode("utf-8", "replace").strip()
    return model or None


def compute_capability_for_model(model: str | None) -> tuple[int, int] | None:
    """CUDA compute capability implied by a Jetson board model string."""
    if not model:
        return None
    lowered = model.lower()
    for marker, capability in _COMPUTE_CAPABILITIES:
        if marker in lowered:
            return capability
    return None


class JetsonPlatformProbe:
    """Host-equivalent capability discovery for Jetson hosts (spec §23)."""

    def __init__(
        self,
        worker_id: str,
        docker_client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._worker_id = worker_id
        self._host_probe = HostCapabilityProbe(
            worker_id,
            docker_client_factory,
            memory_pool_id=SYSTEM_MEMORY_POOL_ID,
        )

    def discover(self) -> CapabilityFragment:
        host = self._host_probe.discover()
        devices = tuple(self._retarget_cpu(device) for device in host.devices)
        gpu = self._gpu_device()
        return CapabilityFragment(
            architecture=host.architecture,
            os=host.os,
            container_runtime=host.container_runtime,
            network_interfaces=host.network_interfaces,
            devices=(*devices, gpu),
            memory_pools=host.memory_pools,
            runtime_platforms=host.runtime_platforms,
        )

    def _retarget_cpu(self, device: DeviceCapability) -> DeviceCapability:
        """The SoC CPU: NVIDIA vendor, tegra-tagged, system-memory pool.

        The pool id already comes from the parameterized host probe (spec
        §14); only the coarse generic-host vendor/tags are sharpened here.
        """
        if device.identity.kind is not DeviceKind.CPU:
            return device
        return dataclasses.replace(
            device,
            vendor="nvidia",
            platform_tags=(*device.platform_tags, "tegra"),
        )

    def _gpu_device(self) -> DeviceCapability:
        model = read_device_tree_model()
        capability = compute_capability_for_model(model)
        tags = ["cuda", "tegra"]
        if capability is not None:
            tags.insert(1, f"sm{capability[0]}{capability[1]}")
        l4t = read_l4t_release()
        if l4t is not None:
            tags.append(f"l4t-{l4t}")
        return DeviceCapability(
            identity=DeviceIdentity(
                device_id=derive_jetson_gpu_device_id(self._worker_id),
                kind=DeviceKind.GPU,
                # The integrated GPU is addressed as the SoC GPU, never a
                # CUDA ordinal identity (spec §11).
                local_locator="igpu",
            ),
            vendor="nvidia",
            model=model or "Jetson integrated GPU",
            compute_capability=(
                f"{capability[0]}.{capability[1]}" if capability is not None else None
            ),
            memory_pool_id=SYSTEM_MEMORY_POOL_ID,
            supported_dtypes=(
                dtypes_for_compute_capability(*capability)
                if capability is not None
                else ("fp32",)
            ),
            driver_version=None,
            platform_tags=tuple(tags),
        )
