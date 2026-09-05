"""Worker-side assembly and local inspection tests (Phase 1 spec §27, §43)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from edgeshard.cluster.capability import (
    DeviceCapability,
    MemoryModel,
    MemoryPoolCapability,
    OSInfo,
    compute_capability_revision,
)
from edgeshard.cluster.identity import DeviceIdentity, DeviceKind
from edgeshard.cluster.inventory import RuntimeInstanceState, RuntimeStatus
from edgeshard.cluster.snapshot import WorkerSnapshot
from edgeshard.cluster.state import (
    DeviceAvailability,
    DeviceState,
    WorkerStatus,
)
from edgeshard.control.worker.agent import (
    assemble_capability,
    enrich_device_states,
    inspect_local_worker,
    to_plain_mapping,
)
from edgeshard.control.worker.config import (
    ModelStoreSection,
    RuntimeSection,
    WorkerConfig,
    WorkerSection,
)
from edgeshard.control.worker.discovery.base import CapabilityFragment
from edgeshard.control.worker.identity import derive_cpu_device_id
from edgeshard.runtime import labels


def make_device(device_id: str, pool_id: str | None = None) -> DeviceCapability:
    return DeviceCapability(
        identity=DeviceIdentity(
            device_id=device_id, kind=DeviceKind.CPU, local_locator=f"loc-{device_id}"
        ),
        vendor="test",
        model="test-device",
        compute_capability=None,
        memory_pool_id=pool_id,
        supported_dtypes=(),
        driver_version=None,
        platform_tags=(),
    )


def make_fragment(
    *,
    architecture: str | None = "x86_64",
    devices: tuple[DeviceCapability, ...] = (),
    pools: tuple[MemoryPoolCapability, ...] = (),
) -> CapabilityFragment:
    return CapabilityFragment(
        architecture=architecture,
        os=OSInfo(name="testos", version="1", kernel="1.0"),
        devices=devices,
        memory_pools=pools,
    )


def make_device_state(device_id: str) -> DeviceState:
    return DeviceState(
        device_id=device_id,
        utilization=None,
        temperature_c=None,
        power_w=None,
        availability=DeviceAvailability.AVAILABLE,
        running_runtime_ids=(),
    )


def make_runtime_instance(
    runtime_id: str, device_ids: tuple[str, ...] = ()
) -> RuntimeInstanceState:
    return RuntimeInstanceState(
        runtime_id=runtime_id,
        backend="edgeshard-shard",
        execution_id=None,
        status=RuntimeStatus.RUNNING,
        device_ids=device_ids,
        container_id=None,
        endpoint=None,
        model_local_name=None,
    )


def test_assemble_finalizes_revision() -> None:
    pool = MemoryPoolCapability(
        memory_pool_id="pool-a", model=MemoryModel.SHARED, total_bytes=1024
    )
    fragment = make_fragment(devices=(make_device("dev-a", "pool-a"),), pools=(pool,))
    capability = assemble_capability([fragment])
    assert capability.capability_revision
    assert capability.capability_revision == compute_capability_revision(capability)


def test_assemble_merges_probe_collections() -> None:
    pool_a = MemoryPoolCapability("pool-a", MemoryModel.SHARED, 1024)
    pool_b = MemoryPoolCapability("pool-b", MemoryModel.DISCRETE, 2048)
    capability = assemble_capability(
        [
            make_fragment(devices=(make_device("dev-a", "pool-a"),), pools=(pool_a,)),
            make_fragment(devices=(make_device("dev-b", "pool-b"),), pools=(pool_b,)),
        ]
    )
    assert {device.identity.device_id for device in capability.devices} == {
        "dev-a",
        "dev-b",
    }
    assert {pool.memory_pool_id for pool in capability.memory_pools} == {
        "pool-a",
        "pool-b",
    }


def test_assemble_rejects_conflicting_scalar_facts() -> None:
    with pytest.raises(ValueError, match="architecture"):
        assemble_capability(
            [make_fragment(architecture="x86_64"), make_fragment(architecture="aarch64")]
        )


def test_assemble_requires_architecture() -> None:
    with pytest.raises(ValueError, match="architecture"):
        assemble_capability([make_fragment(architecture=None)])


def test_assemble_requires_os() -> None:
    with pytest.raises(ValueError, match="OS information"):
        assemble_capability([CapabilityFragment(architecture="x86_64")])


def test_enrich_attaches_running_runtimes_to_devices() -> None:
    states = [make_device_state("dev-a"), make_device_state("dev-b")]
    instances = [
        make_runtime_instance("rt-1", device_ids=("dev-a",)),
        make_runtime_instance("rt-2", device_ids=("dev-a", "dev-b")),
    ]
    enriched = enrich_device_states(states, instances)
    assert enriched[0].running_runtime_ids == ("rt-1", "rt-2")
    assert enriched[1].running_runtime_ids == ("rt-2",)


def test_to_plain_mapping_is_json_safe() -> None:
    pool = MemoryPoolCapability("pool-a", MemoryModel.SHARED, 1024)
    capability = assemble_capability(
        [make_fragment(devices=(make_device("dev-a", "pool-a"),), pools=(pool,))]
    )
    state = make_device_state("dev-a")
    payload = {"capability": to_plain_mapping(capability), "state": to_plain_mapping(state)}

    assert json.loads(json.dumps(payload)) == payload
    assert payload["capability"]["memory_pools"][0]["model"] == "shared"
    assert payload["state"]["availability"] == "available"


def make_config(tmp_path: Path, *, discover: bool = True) -> WorkerConfig:
    return WorkerConfig(
        worker=WorkerSection(identity_path=tmp_path / "worker-id"),
        model_store=ModelStoreSection(root=tmp_path / "models"),
        runtime=RuntimeSection(discover_managed_containers=discover),
    )


class FakeContainer:
    def __init__(self, container_labels: dict[str, str]) -> None:
        self.labels = container_labels
        self.status = "running"
        self.id = "c" * 64
        self.attrs = {"State": {}}


class FakeContainers:
    def __init__(self, containers: list[FakeContainer]) -> None:
        self._containers = containers

    def list(self, **kwargs: Any) -> list[FakeContainer]:
        return list(self._containers)


class FakeDockerClient:
    def __init__(self, containers: list[FakeContainer]) -> None:
        self.containers = FakeContainers(containers)


async def test_inspect_local_worker_end_to_end(tmp_path: Path) -> None:
    """The Master-less lifecycle yields a snapshot-consistent Worker view."""
    snapshot_dir = tmp_path / "models" / "tiny-llama"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text(
        json.dumps({"_name_or_path": "tiny/llama"}), encoding="utf-8"
    )
    (snapshot_dir / "model.safetensors").write_bytes(b"\x00" * 8)

    container = FakeContainer(
        labels.managed_container_labels(
            execution_id="execution-1", runtime_id="runtime-1", backend="edgeshard-shard"
        )
    )
    inspection = await inspect_local_worker(
        make_config(tmp_path), docker_client=FakeDockerClient([container])
    )

    worker_id = inspection.identity.worker_id
    assert (tmp_path / "worker-id").read_text(encoding="utf-8").strip() == worker_id
    assert inspection.state.worker_id == worker_id

    device_ids = {device.identity.device_id for device in inspection.capability.devices}
    pool_ids = {pool.memory_pool_id for pool in inspection.capability.memory_pools}
    assert derive_cpu_device_id(worker_id) in device_ids
    assert "host-memory" in pool_ids
    assert all(
        state.device_id in device_ids for state in inspection.state.device_states
    )
    assert all(
        state.memory_pool_id in pool_ids for state in inspection.state.memory_states
    )

    assert [model.local_name for model in inspection.state.models] == ["tiny-llama"]
    assert [
        instance.runtime_id for instance in inspection.state.runtime_instances
    ] == ["runtime-1"]

    # Cross-validates as a Master-side snapshot would (spec §38).
    WorkerSnapshot(
        identity=inspection.identity,
        capability=inspection.capability,
        state=inspection.state,
        status=WorkerStatus.ONLINE,
        session_id=None,
        last_seen_at=None,
    )

    payload = json.dumps(
        {
            "identity": to_plain_mapping(inspection.identity),
            "capability": to_plain_mapping(inspection.capability),
            "state": to_plain_mapping(inspection.state),
        }
    )
    assert worker_id in payload


async def test_inspect_identity_persists_across_invocations(tmp_path: Path) -> None:
    config = make_config(tmp_path, discover=False)
    first = await inspect_local_worker(config, docker_client=None)
    second = await inspect_local_worker(config, docker_client=None)
    assert first.identity.worker_id == second.identity.worker_id


async def test_inspect_without_runtime_discovery_reports_no_runtimes(tmp_path: Path) -> None:
    inspection = await inspect_local_worker(make_config(tmp_path, discover=False))
    assert inspection.state.runtime_instances == ()


async def test_inspect_tolerates_explicit_none_docker_client(tmp_path: Path) -> None:
    inspection = await inspect_local_worker(make_config(tmp_path), docker_client=None)
    assert inspection.state.runtime_instances == ()
