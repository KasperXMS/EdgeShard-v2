"""Shared builders for cluster/control-plane tests (Phase 1 spec §13-14).

Lives at the ``tests/`` root so every test directory can ``from factories
import ...`` (pytest puts the root conftest's directory on ``sys.path``).

``make_rtx_capability`` and ``make_jetson_capability`` model the two
reference platforms of Phase 1 - a discrete-GPU host with independent VRAM
pools, and a Jetson host whose CPU and GPU share one system-memory pool.
The integration suite reuses the same shapes for fake Worker fixtures
(spec §52 Test B), and the control-protocol tests reuse them as wire
payloads (spec §40).

``make_worker_state`` defaults to ids drawn from the RTX capability so
capability and state compose into a consistent ``WorkerSnapshot``.
"""

from __future__ import annotations

import dataclasses
import uuid

from edgeshard.cluster.capability import (
    ContainerRuntimeCapability,
    DeviceCapability,
    MemoryModel,
    MemoryPoolCapability,
    NetworkInterfaceCapability,
    OSInfo,
    RuntimePlatformCapability,
    WorkerCapability,
    compute_capability_revision,
)
from edgeshard.cluster.identity import DeviceIdentity, DeviceKind, WorkerIdentity
from edgeshard.cluster.inventory import (
    ModelAvailability,
    ModelInventoryEntry,
    RuntimeInstanceState,
    RuntimeStatus,
)
from edgeshard.cluster.state import (
    DeviceAvailability,
    DeviceState,
    MemoryPoolState,
    WorkerState,
)

PLACEHOLDER_REVISION = ""

# Reference identifiers of the RTX-like capability, shared with the state
# factory below so snapshots stay internally consistent.
RTX_CPU_DEVICE_ID = "cpu-host"
RTX_GPU_DEVICE_ID = "GPU-69c27179-5df5-d790-4b75-6cf18a4d2b1c"
RTX_HOST_POOL_ID = "host-memory"
RTX_GPU_POOL_ID = f"gpu-{RTX_GPU_DEVICE_ID}-vram"


def make_worker_identity(worker_id: str | None = None) -> WorkerIdentity:
    return WorkerIdentity(
        worker_id=worker_id or str(uuid.uuid4()),
        hostname="worker-host",
        agent_version="0.1.0",
        protocol_version="1",
    )


def make_rtx_capability() -> WorkerCapability:
    """RTX4090-like capability: discrete VRAM pool per GPU (spec §13)."""
    capability = WorkerCapability(
        architecture="x86_64",
        os=OSInfo(name="ubuntu", version="24.04", kernel="6.8.0"),
        container_runtime=ContainerRuntimeCapability(
            runtime="docker", version="27.0", nvidia_runtime_available=True
        ),
        network_interfaces=(
            NetworkInterfaceCapability(
                interface_id="nic-0", name="eth0", addresses=("192.168.1.100",), mtu=1500
            ),
        ),
        devices=(
            DeviceCapability(
                identity=DeviceIdentity(
                    device_id=RTX_CPU_DEVICE_ID, kind=DeviceKind.CPU, local_locator="cpu"
                ),
                vendor="GenuineIntel",
                model="x86_64 host CPU",
                compute_capability=None,
                memory_pool_id=RTX_HOST_POOL_ID,
                supported_dtypes=(),
                driver_version=None,
                platform_tags=("x86_64",),
            ),
            DeviceCapability(
                identity=DeviceIdentity(
                    device_id=RTX_GPU_DEVICE_ID, kind=DeviceKind.GPU, local_locator="cuda:0"
                ),
                vendor="NVIDIA",
                model="NVIDIA GeForce RTX 4090",
                compute_capability="8.9",
                memory_pool_id=RTX_GPU_POOL_ID,
                supported_dtypes=("fp32", "fp16", "bf16"),
                driver_version="550.90",
                platform_tags=("cuda", "sm89"),
            ),
        ),
        memory_pools=(
            MemoryPoolCapability(
                memory_pool_id=RTX_HOST_POOL_ID, model=MemoryModel.SHARED, total_bytes=64 * 2**30
            ),
            MemoryPoolCapability(
                memory_pool_id=RTX_GPU_POOL_ID,
                model=MemoryModel.DISCRETE,
                total_bytes=24 * 2**30,
            ),
        ),
        runtime_platforms=(
            RuntimePlatformCapability(backend="edgeshard-shard", platform="cuda", image=None),
            RuntimePlatformCapability(backend="vllm", platform="cuda", image=None),
        ),
        capability_revision=PLACEHOLDER_REVISION,
    )
    return finalize(capability)


def make_jetson_capability() -> WorkerCapability:
    """AGX Orin-like capability: CPU and GPU share system-memory (spec §14)."""
    capability = WorkerCapability(
        architecture="aarch64",
        os=OSInfo(name="ubuntu", version="22.04", kernel="5.15.0-tegra"),
        container_runtime=ContainerRuntimeCapability(
            runtime="docker", version="24.0", nvidia_runtime_available=True
        ),
        network_interfaces=(
            NetworkInterfaceCapability(
                interface_id="nic-0", name="eth0", addresses=("192.168.1.101",), mtu=1500
            ),
        ),
        devices=(
            DeviceCapability(
                identity=DeviceIdentity(
                    device_id="cpu-system", kind=DeviceKind.CPU, local_locator="cpu"
                ),
                vendor="NVIDIA",
                model="Arm Cortex-A78AE",
                compute_capability=None,
                memory_pool_id="system-memory",
                supported_dtypes=(),
                driver_version=None,
                platform_tags=("aarch64", "tegra"),
            ),
            DeviceCapability(
                identity=DeviceIdentity(
                    device_id="gpu-system", kind=DeviceKind.GPU, local_locator="igpu"
                ),
                vendor="NVIDIA",
                model="AGX Orin integrated GPU",
                compute_capability="8.7",
                memory_pool_id="system-memory",
                supported_dtypes=("fp32", "fp16", "bf16"),
                driver_version=None,
                platform_tags=("cuda", "sm87", "tegra"),
            ),
        ),
        memory_pools=(
            MemoryPoolCapability(
                memory_pool_id="system-memory", model=MemoryModel.SHARED, total_bytes=32 * 2**30
            ),
        ),
        runtime_platforms=(
            RuntimePlatformCapability(backend="edgeshard-shard", platform="cuda", image=None),
        ),
        capability_revision=PLACEHOLDER_REVISION,
    )
    return finalize(capability)


def finalize(capability: WorkerCapability) -> WorkerCapability:
    """Set the deterministic capability_revision (spec §16 finalize pattern)."""
    return dataclasses.replace(
        capability, capability_revision=compute_capability_revision(capability)
    )


def make_worker_state(
    worker_id: str,
    device_ids: tuple[str, ...] = (RTX_GPU_DEVICE_ID,),
    pool_ids: tuple[str, ...] = (RTX_GPU_POOL_ID,),
) -> WorkerState:
    """Dynamic state consistent with ``make_rtx_capability`` by default."""
    return WorkerState(
        worker_id=worker_id,
        device_states=tuple(
            DeviceState(
                device_id=device_id,
                utilization=21.0,
                temperature_c=45.0,
                power_w=None,
                availability=DeviceAvailability.AVAILABLE,
                running_runtime_ids=(),
            )
            for device_id in device_ids
        ),
        memory_states=tuple(
            MemoryPoolState(memory_pool_id=pool_id, available_bytes=18 * 2**30)
            for pool_id in pool_ids
        ),
        runtime_instances=(
            RuntimeInstanceState(
                runtime_id="runtime-1",
                backend="edgeshard-shard",
                execution_id="execution-1",
                status=RuntimeStatus.RUNNING,
                device_ids=device_ids,
                container_id="abc123",
                endpoint="192.168.1.100:51051",
                model_local_name="tiny-llama",
            ),
        ),
        models=(
            ModelInventoryEntry(
                local_name="tiny-llama",
                model_id="tiny/llama",
                revision="main",
                size_bytes=2**20,
                status=ModelAvailability.READY,
            ),
        ),
    )
