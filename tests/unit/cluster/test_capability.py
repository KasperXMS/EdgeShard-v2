"""Static capability domain tests (Phase 1 spec §12-16, §51)."""

from __future__ import annotations

import dataclasses
import re

import pytest
from factories import PLACEHOLDER_REVISION, finalize, make_jetson_capability, make_rtx_capability

from edgeshard.cluster.capability import (
    MemoryModel,
    MemoryPoolCapability,
    NetworkInterfaceCapability,
    OSInfo,
    WorkerCapability,
    compute_capability_revision,
)
from edgeshard.cluster.identity import DeviceKind


def test_rtx_capability_memory_topology() -> None:
    """Discrete GPU host: independent VRAM pool per GPU plus host memory (spec §13)."""
    capability = make_rtx_capability()
    pools = {pool.memory_pool_id: pool for pool in capability.memory_pools}

    assert pools["host-memory"].model is MemoryModel.SHARED
    gpu_pool = pools["gpu-GPU-69c27179-5df5-d790-4b75-6cf18a4d2b1c-vram"]
    assert gpu_pool.model is MemoryModel.DISCRETE
    assert gpu_pool.total_bytes == 24 * 2**30

    gpu_device = capability.devices[1]
    assert gpu_device.memory_pool_id == gpu_pool.memory_pool_id
    assert gpu_device.identity.kind is DeviceKind.GPU
    assert gpu_device.compute_capability == "8.9"


def test_jetson_shared_memory_topology() -> None:
    """Jetson CPU and GPU reference the same shared pool (spec §14).

    Exactly one system-memory pool exists; it must never look like two
    independent 32 GB resources.
    """
    capability = make_jetson_capability()

    assert len(capability.memory_pools) == 1
    pool = capability.memory_pools[0]
    assert pool.memory_pool_id == "system-memory"
    assert pool.model is MemoryModel.SHARED

    cpu_device, gpu_device = capability.devices
    assert cpu_device.identity.kind is DeviceKind.CPU
    assert gpu_device.identity.kind is DeviceKind.GPU
    assert cpu_device.memory_pool_id == pool.memory_pool_id
    assert gpu_device.memory_pool_id == pool.memory_pool_id


def test_device_referencing_missing_pool_is_rejected() -> None:
    capability = make_rtx_capability()
    broken_device = dataclasses.replace(capability.devices[1], memory_pool_id="no-such-pool")
    with pytest.raises(ValueError, match="unknown memory pool"):
        dataclasses.replace(capability, devices=(capability.devices[0], broken_device))


def test_duplicate_device_ids_are_rejected() -> None:
    capability = make_rtx_capability()
    duplicate = dataclasses.replace(capability.devices[1])
    with pytest.raises(ValueError, match="duplicate device_id"):
        dataclasses.replace(capability, devices=(capability.devices[0], duplicate, duplicate))


def test_duplicate_memory_pool_ids_are_rejected() -> None:
    capability = make_rtx_capability()
    pool = capability.memory_pools[0]
    with pytest.raises(ValueError, match="duplicate memory_pool_id"):
        dataclasses.replace(capability, memory_pools=(pool, pool))


def test_memory_pool_total_bytes_must_be_positive() -> None:
    with pytest.raises(ValueError, match="total_bytes"):
        MemoryPoolCapability(memory_pool_id="pool", model=MemoryModel.SHARED, total_bytes=0)
    with pytest.raises(ValueError, match="total_bytes"):
        MemoryPoolCapability(memory_pool_id="pool", model=MemoryModel.SHARED, total_bytes=-1)


def test_empty_architecture_is_rejected() -> None:
    capability = make_rtx_capability()
    with pytest.raises(ValueError, match="architecture"):
        dataclasses.replace(capability, architecture="")


def test_finalize_sets_deterministic_revision() -> None:
    capability = make_rtx_capability()
    assert capability.capability_revision == compute_capability_revision(capability)
    assert capability.capability_revision != PLACEHOLDER_REVISION


def test_same_canonical_capability_same_revision() -> None:
    """Two independently built identical capabilities hash identically (spec §51)."""
    first = make_rtx_capability()
    second = make_rtx_capability()
    assert first == second
    assert compute_capability_revision(first) == compute_capability_revision(second)


def test_revision_ignores_the_revision_field_itself() -> None:
    capability = make_rtx_capability()
    bare = dataclasses.replace(capability, capability_revision="not-yet-computed")
    assert compute_capability_revision(bare) == compute_capability_revision(capability)


def test_revision_is_sha256_hex() -> None:
    revision = compute_capability_revision(make_rtx_capability())
    assert re.fullmatch(r"[0-9a-f]{64}", revision)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: dataclasses.replace(c, architecture="aarch64"),
        lambda c: dataclasses.replace(
            c,
            memory_pools=(
                dataclasses.replace(c.memory_pools[0], total_bytes=32 * 2**30),
                *c.memory_pools[1:],
            ),
        ),
        lambda c: dataclasses.replace(
            c,
            network_interfaces=(
                *c.network_interfaces,
                NetworkInterfaceCapability(
                    interface_id="nic-1", name="eth1", addresses=(), mtu=None
                ),
            ),
        ),
    ],
    ids=["architecture", "pool_size", "extra_interface"],
)
def test_modified_capability_different_revision(mutate) -> None:
    capability = make_rtx_capability()
    mutated = mutate(capability)
    assert compute_capability_revision(mutated) != capability.capability_revision


def test_revision_is_insensitive_to_collection_order() -> None:
    """Same facts in a different enumeration order give the same revision."""
    base = make_rtx_capability()
    base_revision = compute_capability_revision(base)

    reversed_pools = dataclasses.replace(base, memory_pools=base.memory_pools[::-1])
    assert compute_capability_revision(reversed_pools) == base_revision

    reversed_platforms = dataclasses.replace(
        base, runtime_platforms=base.runtime_platforms[::-1]
    )
    assert compute_capability_revision(reversed_platforms) == base_revision

    gpu = base.devices[1]
    reordered_gpu = dataclasses.replace(
        gpu,
        supported_dtypes=gpu.supported_dtypes[::-1],
        platform_tags=gpu.platform_tags[::-1],
    )
    reordered_devices = dataclasses.replace(base, devices=(reordered_gpu, base.devices[0]))
    assert compute_capability_revision(reordered_devices) == base_revision

    nic = base.network_interfaces[0]
    nic_forward = dataclasses.replace(nic, addresses=("192.168.1.100", "10.0.0.5"))
    nic_backward = dataclasses.replace(nic, addresses=("10.0.0.5", "192.168.1.100"))
    assert compute_capability_revision(
        dataclasses.replace(base, network_interfaces=(nic_forward,))
    ) == compute_capability_revision(dataclasses.replace(base, network_interfaces=(nic_backward,)))

    nic_a = nic
    nic_b = dataclasses.replace(nic, interface_id="nic-1", name="eth1")
    assert compute_capability_revision(
        dataclasses.replace(base, network_interfaces=(nic_a, nic_b))
    ) == compute_capability_revision(
        dataclasses.replace(base, network_interfaces=(nic_b, nic_a))
    )


def test_rtx_and_jetson_revisions_differ() -> None:
    assert compute_capability_revision(
        make_rtx_capability()
    ) != compute_capability_revision(make_jetson_capability())


def test_empty_worker_capability_is_allowed() -> None:
    """A Worker with no devices yet is consistent (P1B probes build incrementally)."""
    capability = WorkerCapability(
        architecture="x86_64",
        os=OSInfo(name="ubuntu", version=None, kernel=None),
        container_runtime=None,
        network_interfaces=(),
        devices=(),
        memory_pools=(),
        runtime_platforms=(),
        capability_revision=PLACEHOLDER_REVISION,
    )
    assert finalize(capability).capability_revision
