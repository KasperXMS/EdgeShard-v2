"""MockMaster orchestration tests without a Docker daemon.

A fake Docker client records network management and recording
``RuntimeDriver`` doubles record container lifecycle, so deployment
wiring (network, generated configs, launch order, publish rules,
backend-neutral cleanup) is fully exercised on the CPU development
host — for shard pipelines, standalone vLLM runtimes, and mixed
deployments of both.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import docker
import pytest

from edgeshard.control.mock.deployment import SHARD_LISTEN_PORT, network_name
from edgeshard.control.mock.manifest import DeploymentManifest, ManifestError
from edgeshard.control.mock.master import Deployment, MockMaster, MockMasterError
from edgeshard.model.errors import ModelSourceError
from edgeshard.runtime.config import ShardRuntimeConfig
from edgeshard.runtime.drivers.base import DriverError, RuntimeHandle, RuntimeSpec
from edgeshard.runtime.drivers.vllm import DEFAULT_VLLM_IMAGE, VLLMRuntimeSpec
from edgeshard.runtime.model_store import ModelStore

EXECUTION_ID = "exec-mock"
NETWORK = network_name(EXECUTION_ID)
DEFAULT_IMAGE = "edgeshard/hf-shard:cpu"


class FakeNetwork:
    def __init__(self, name: str) -> None:
        self.name = name
        self.removed = False

    def remove(self) -> None:
        self.removed = True


class FakeNetworks:
    def __init__(self) -> None:
        self.created: list[tuple[str, dict[str, Any]]] = []
        self.by_name: dict[str, FakeNetwork] = {}

    def create(self, name: str, **kwargs: Any) -> FakeNetwork:
        self.created.append((name, kwargs))
        network = FakeNetwork(name)
        self.by_name[name] = network
        return network

    def get(self, name: str) -> FakeNetwork:
        network = self.by_name.get(name)
        if network is None or network.removed:
            raise docker.errors.NotFound(f"no such network: {name}")
        return network


class FakeDockerClient:
    def __init__(self) -> None:
        self.networks = FakeNetworks()


class RecordingDriver:
    """RuntimeDriver double: records lifecycle, never touches Docker."""

    def __init__(self, *, fail_start_number: int | None = None) -> None:
        self.start_calls: list[RuntimeSpec] = []
        self.wait_ready_calls: list[RuntimeHandle] = []
        self.stop_calls: list[RuntimeHandle] = []
        self._fail_start_number = fail_start_number

    async def start(self, spec: RuntimeSpec) -> RuntimeHandle:
        self.start_calls.append(spec)
        if self._fail_start_number == len(self.start_calls):
            raise DriverError("boom")
        port = 40000 + len(self.start_calls)
        return RuntimeHandle(
            runtime_id=spec.runtime_id,
            backend=spec.backend,
            container_id=f"c-{len(self.start_calls)}",
            endpoint=f"127.0.0.1:{port}",
        )

    async def wait_ready(self, handle: RuntimeHandle) -> None:
        self.wait_ready_calls.append(handle)

    async def info(self, handle: RuntimeHandle) -> Any:
        raise AssertionError("info is unused by the Mock Master")

    async def stop(self, handle: RuntimeHandle) -> None:
        self.stop_calls.append(handle)


def manifest_payload(
    model_dir_name: str, *, end: int = 4, second_image: str | None = None
) -> dict[str, Any]:
    """Two contiguous shards over a 4-block model."""
    second: dict[str, Any] = {
        "id": "shard-1",
        "backend": "edgeshard_shard",
        "shard": {"start": 2, "end": end, "include_output_stage": True},
    }
    if second_image:
        second["image"] = second_image
    return {
        "execution_id": EXECUTION_ID,
        "model": {"id": "tiny/llama", "local_name": model_dir_name},
        "runtimes": [
            {
                "id": "shard-0",
                "backend": "edgeshard_shard",
                "shard": {"start": 0, "end": 2, "include_input_stage": True},
            },
            second,
        ],
        "pipeline": ["shard-0", "shard-1"],
    }


def mixed_manifest_payload(
    model_dir_name: str, *, vllm_image: str | None = None
) -> dict[str, Any]:
    """Two shards plus one independent full-model vLLM runtime."""
    payload = manifest_payload(model_dir_name)
    vllm_runtime: dict[str, Any] = {
        "id": "vllm-0",
        "backend": "vllm",
        "device": {"type": "cuda", "index": 0},
        "vllm": {"max_model_len": 2048},
    }
    if vllm_image:
        vllm_runtime["image"] = vllm_image
    payload["runtimes"].append(vllm_runtime)
    return payload


def make_master(
    docker_client: FakeDockerClient,
    drivers: dict[str, RecordingDriver],
    tiny_llama_dir: Path,
    tmp_path: Path,
) -> MockMaster:
    return MockMaster(
        docker_client=docker_client,
        model_store=ModelStore(model_root=tiny_llama_dir.parent),
        work_dir=tmp_path / "work",
        default_image=DEFAULT_IMAGE,
        drivers=drivers,
    )


async def test_deploy_creates_network_configs_and_launches_in_reverse(
    tiny_llama_dir: Path, tmp_path: Path
) -> None:
    docker_client = FakeDockerClient()
    driver = RecordingDriver()
    master = make_master(
        docker_client, {"edgeshard_shard": driver}, tiny_llama_dir, tmp_path
    )
    manifest = DeploymentManifest.model_validate(
        manifest_payload(tiny_llama_dir.name, second_image="custom/shard:v2")
    )

    deployment = await master.deploy(manifest)

    # Dedicated bridge network per execution (spec 23).
    assert docker_client.networks.created == [(NETWORK, {"driver": "bridge"})]
    # Downstream stages launch first so cascaded readiness resolves.
    assert [spec.runtime_id for spec in driver.start_calls] == ["shard-1", "shard-0"]
    specs = {spec.runtime_id: spec for spec in driver.start_calls}
    # Only the entry runtime publishes a host port (spec 23).
    assert specs["shard-0"].host_port == 0
    assert specs["shard-1"].host_port is None
    # Every runtime joins the execution network under its own alias.
    assert all(spec.network == NETWORK for spec in driver.start_calls)
    assert specs["shard-0"].image == DEFAULT_IMAGE
    assert specs["shard-1"].image == "custom/shard:v2"
    # Generated configs exist and are valid runtime configs.
    config_dir = tmp_path / "work" / EXECUTION_ID
    entry_config = ShardRuntimeConfig.from_yaml(config_dir / "shard-0.yaml")
    assert entry_config.pipeline.next_endpoint == f"shard-1:{SHARD_LISTEN_PORT}"
    final_config = ShardRuntimeConfig.from_yaml(config_dir / "shard-1.yaml")
    assert final_config.pipeline.next_endpoint is None
    # Handles come back in pipeline order; readiness waited on the entry.
    assert [handle.runtime_id for handle in deployment.handles] == ["shard-0", "shard-1"]
    assert deployment.entry_endpoint == deployment.handles[0].endpoint
    assert deployment.network == NETWORK
    assert driver.wait_ready_calls == [deployment.handles[0]]


async def test_deploy_mixed_manifest_manages_vllm_and_shards(
    tiny_llama_dir: Path, tmp_path: Path
) -> None:
    """0J gate: one generic lifecycle manages two different backends."""
    docker_client = FakeDockerClient()
    shard_driver = RecordingDriver()
    vllm_driver = RecordingDriver()
    master = make_master(
        docker_client,
        {"edgeshard_shard": shard_driver, "vllm": vllm_driver},
        tiny_llama_dir,
        tmp_path,
    )
    manifest = DeploymentManifest.model_validate(
        mixed_manifest_payload(tiny_llama_dir.name, vllm_image="custom/vllm:v1")
    )

    deployment = await master.deploy(manifest)

    # Standalone runtimes start first; shard stages still reverse.
    assert [spec.runtime_id for spec in vllm_driver.start_calls] == ["vllm-0"]
    assert [spec.runtime_id for spec in shard_driver.start_calls] == [
        "shard-1",
        "shard-0",
    ]
    (vllm_spec,) = vllm_driver.start_calls
    assert isinstance(vllm_spec, VLLMRuntimeSpec)
    assert vllm_spec.image == "custom/vllm:v1"
    assert vllm_spec.model_id == "tiny/llama"
    assert vllm_spec.model_path == Path(f"/models/{tiny_llama_dir.name}")
    assert vllm_spec.device_index == 0
    assert vllm_spec.max_model_len == 2048
    assert vllm_spec.tensor_parallel_size is None
    # Standalone runtimes always publish: test requests come from the host.
    assert vllm_spec.host_port == 0
    assert vllm_spec.network == NETWORK
    # Readiness is waited for per backend: entry shard, then vLLM.
    assert shard_driver.wait_ready_calls == [deployment.handles[0]]
    assert vllm_driver.wait_ready_calls == [deployment.handle("vllm-0")]
    # Handles: pipeline order first, then standalone runtimes.
    assert [handle.runtime_id for handle in deployment.handles] == [
        "shard-0",
        "shard-1",
        "vllm-0",
    ]
    assert deployment.entry_endpoint == deployment.handles[0].endpoint
    with pytest.raises(KeyError):
        deployment.handle("no-such-runtime")
    # Only shard runtimes receive generated runtime configs.
    config_dir = tmp_path / "work" / EXECUTION_ID
    assert sorted(path.name for path in config_dir.glob("*.yaml")) == [
        "shard-0.yaml",
        "shard-1.yaml",
    ]

    # Shutdown goes through each runtime's own driver, in reverse.
    await master.shutdown(deployment)
    assert [handle.runtime_id for handle in vllm_driver.stop_calls] == ["vllm-0"]
    assert [handle.runtime_id for handle in shard_driver.stop_calls] == [
        "shard-1",
        "shard-0",
    ]
    assert docker_client.networks.by_name[NETWORK].removed is True


async def test_deploy_vllm_only_manifest(tiny_llama_dir: Path, tmp_path: Path) -> None:
    docker_client = FakeDockerClient()
    shard_driver = RecordingDriver()
    vllm_driver = RecordingDriver()
    master = make_master(
        docker_client,
        {"edgeshard_shard": shard_driver, "vllm": vllm_driver},
        tiny_llama_dir,
        tmp_path,
    )
    payload = mixed_manifest_payload(tiny_llama_dir.name)
    payload["runtimes"] = [runtime for runtime in payload["runtimes"]
                           if runtime["backend"] == "vllm"]
    payload["pipeline"] = []
    manifest = DeploymentManifest.model_validate(payload)

    deployment = await master.deploy(manifest)

    assert deployment.entry_endpoint is None
    assert [handle.runtime_id for handle in deployment.handles] == ["vllm-0"]
    assert shard_driver.start_calls == []
    assert vllm_driver.wait_ready_calls == [deployment.handles[0]]
    # Without shard runtimes the default vLLM image applies.
    (vllm_spec,) = vllm_driver.start_calls
    assert vllm_spec.image == DEFAULT_VLLM_IMAGE
    config_dir = tmp_path / "work" / EXECUTION_ID
    assert list(config_dir.glob("*.yaml")) == []

    await master.shutdown(deployment)
    assert [handle.runtime_id for handle in vllm_driver.stop_calls] == ["vllm-0"]


async def test_deploy_fails_without_driver_for_backend(
    tiny_llama_dir: Path, tmp_path: Path
) -> None:
    docker_client = FakeDockerClient()
    master = make_master(
        docker_client, {"edgeshard_shard": RecordingDriver()}, tiny_llama_dir, tmp_path
    )
    master._drivers.pop("vllm")
    manifest = DeploymentManifest.model_validate(
        mixed_manifest_payload(tiny_llama_dir.name)
    )
    with pytest.raises(MockMasterError, match="no driver registered"):
        await master.deploy(manifest)
    assert docker_client.networks.by_name[NETWORK].removed is True
    assert not (tmp_path / "work" / EXECUTION_ID).exists()


async def test_deploy_validates_partition_upper_bound(
    tiny_llama_dir: Path, tmp_path: Path
) -> None:
    docker_client = FakeDockerClient()
    driver = RecordingDriver()
    master = make_master(
        docker_client, {"edgeshard_shard": driver}, tiny_llama_dir, tmp_path
    )
    manifest = DeploymentManifest.model_validate(
        manifest_payload(tiny_llama_dir.name, end=5)
    )

    with pytest.raises(ManifestError, match="has 4 blocks"):
        await master.deploy(manifest)
    # Validation happens before anything is created or launched.
    assert docker_client.networks.created == []
    assert driver.start_calls == []


def test_manifest_rejects_local_name_escaping_the_store(
    tiny_llama_dir: Path, tmp_path: Path
) -> None:
    # Plans carry only a local name; anything but a single safe segment is
    # rejected at parse time, before any deployment work starts.
    payload = manifest_payload(tiny_llama_dir.name)
    payload["model"]["local_name"] = "../escape"
    with pytest.raises(ValueError, match="single path segment"):
        DeploymentManifest.model_validate(payload)


async def test_deploy_rejects_missing_model_dir(tmp_path: Path) -> None:
    master = MockMaster(
        docker_client=FakeDockerClient(),
        model_store=ModelStore(model_root=tmp_path / "empty-cache"),
        work_dir=tmp_path / "work",
        default_image=DEFAULT_IMAGE,
        drivers={"edgeshard_shard": RecordingDriver()},
    )
    manifest = DeploymentManifest.model_validate(manifest_payload("no-such-model"))
    with pytest.raises(ModelSourceError, match="no-such-model"):
        await master.deploy(manifest)


async def test_failed_deploy_cleans_up_everything(
    tiny_llama_dir: Path, tmp_path: Path
) -> None:
    docker_client = FakeDockerClient()
    driver = RecordingDriver(fail_start_number=2)  # the entry launch fails
    master = make_master(
        docker_client, {"edgeshard_shard": driver}, tiny_llama_dir, tmp_path
    )
    manifest = DeploymentManifest.model_validate(manifest_payload(tiny_llama_dir.name))

    with pytest.raises(DriverError, match="boom"):
        await master.deploy(manifest)

    # The already-launched downstream container is stopped again.
    assert [handle.runtime_id for handle in driver.stop_calls] == ["shard-1"]
    assert docker_client.networks.by_name[NETWORK].removed is True
    assert not (tmp_path / "work" / EXECUTION_ID).exists()
    assert driver.wait_ready_calls == []


async def test_failed_deploy_stops_standalone_runtimes_too(
    tiny_llama_dir: Path, tmp_path: Path
) -> None:
    docker_client = FakeDockerClient()
    shard_driver = RecordingDriver(fail_start_number=1)  # first shard launch fails
    vllm_driver = RecordingDriver()
    master = make_master(
        docker_client,
        {"edgeshard_shard": shard_driver, "vllm": vllm_driver},
        tiny_llama_dir,
        tmp_path,
    )
    manifest = DeploymentManifest.model_validate(
        mixed_manifest_payload(tiny_llama_dir.name)
    )

    with pytest.raises(DriverError, match="boom"):
        await master.deploy(manifest)

    # The already-started vLLM runtime is cleaned up through its own driver.
    assert shard_driver.stop_calls == []
    assert [handle.runtime_id for handle in vllm_driver.stop_calls] == ["vllm-0"]
    assert docker_client.networks.by_name[NETWORK].removed is True
    assert not (tmp_path / "work" / EXECUTION_ID).exists()


async def test_shutdown_stops_runtimes_removes_network_and_configs(
    tiny_llama_dir: Path, tmp_path: Path
) -> None:
    docker_client = FakeDockerClient()
    driver = RecordingDriver()
    master = make_master(
        docker_client, {"edgeshard_shard": driver}, tiny_llama_dir, tmp_path
    )
    manifest = DeploymentManifest.model_validate(manifest_payload(tiny_llama_dir.name))

    deployment = await master.deploy(manifest)
    await master.shutdown(deployment)

    # Runtimes stop in reverse pipeline order, network and configs go away.
    assert [handle.runtime_id for handle in driver.stop_calls] == ["shard-1", "shard-0"]
    assert docker_client.networks.by_name[NETWORK].removed is True
    assert not (tmp_path / "work" / EXECUTION_ID).exists()

    # Shutdown is idempotent: an already-removed network is not an error.
    await master.shutdown(deployment)


def test_deployment_is_frozen() -> None:
    deployment = Deployment(
        execution_id=EXECUTION_ID, network=NETWORK, entry_endpoint=None, handles=()
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        deployment.network = "other"  # type: ignore[misc]
