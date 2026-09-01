"""MockMaster orchestration tests without a Docker daemon.

A fake Docker client records network management and a recording
``RuntimeDriver`` double records container lifecycle, so deployment
wiring (network, generated configs, launch order, publish rules,
cleanup) is fully exercised on the CPU development host.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import docker
import pytest

from edgeshard.control.mock.deployment import SHARD_LISTEN_PORT, network_name
from edgeshard.control.mock.manifest import DeploymentManifest, ManifestError
from edgeshard.control.mock.master import Deployment, MockMaster
from edgeshard.model.errors import ModelSourceError
from edgeshard.runtime.config import ShardRuntimeConfig
from edgeshard.runtime.drivers.base import DriverError, RuntimeHandle, RuntimeSpec

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
        "model": {"id": "tiny/llama", "path": f"/models/{model_dir_name}"},
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


def make_master(
    docker_client: FakeDockerClient,
    driver: RecordingDriver,
    tiny_llama_dir: Path,
    tmp_path: Path,
) -> MockMaster:
    return MockMaster(
        docker_client=docker_client,
        model_cache_dir=tiny_llama_dir.parent,
        work_dir=tmp_path / "work",
        default_image=DEFAULT_IMAGE,
        driver=driver,
    )


async def test_deploy_creates_network_configs_and_launches_in_reverse(
    tiny_llama_dir: Path, tmp_path: Path
) -> None:
    docker_client = FakeDockerClient()
    driver = RecordingDriver()
    master = make_master(docker_client, driver, tiny_llama_dir, tmp_path)
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


async def test_deploy_validates_partition_upper_bound(
    tiny_llama_dir: Path, tmp_path: Path
) -> None:
    docker_client = FakeDockerClient()
    driver = RecordingDriver()
    master = make_master(docker_client, driver, tiny_llama_dir, tmp_path)
    manifest = DeploymentManifest.model_validate(
        manifest_payload(tiny_llama_dir.name, end=5)
    )

    with pytest.raises(ManifestError, match="has 4 blocks"):
        await master.deploy(manifest)
    # Validation happens before anything is created or launched.
    assert docker_client.networks.created == []
    assert driver.start_calls == []


async def test_deploy_rejects_model_path_outside_mount(
    tiny_llama_dir: Path, tmp_path: Path
) -> None:
    master = make_master(FakeDockerClient(), RecordingDriver(), tiny_llama_dir, tmp_path)
    payload = manifest_payload(tiny_llama_dir.name)
    payload["model"]["path"] = "/weights/tiny-llama"
    manifest = DeploymentManifest.model_validate(payload)
    with pytest.raises(ManifestError, match="model mount"):
        await master.deploy(manifest)


async def test_deploy_rejects_missing_model_dir(tmp_path: Path) -> None:
    master = MockMaster(
        docker_client=FakeDockerClient(),
        model_cache_dir=tmp_path / "empty-cache",
        work_dir=tmp_path / "work",
        default_image=DEFAULT_IMAGE,
        driver=RecordingDriver(),
    )
    manifest = DeploymentManifest.model_validate(manifest_payload("no-such-model"))
    with pytest.raises(ModelSourceError, match="no-such-model"):
        await master.deploy(manifest)


async def test_failed_deploy_cleans_up_everything(
    tiny_llama_dir: Path, tmp_path: Path
) -> None:
    docker_client = FakeDockerClient()
    driver = RecordingDriver(fail_start_number=2)  # the entry launch fails
    master = make_master(docker_client, driver, tiny_llama_dir, tmp_path)
    manifest = DeploymentManifest.model_validate(manifest_payload(tiny_llama_dir.name))

    with pytest.raises(DriverError, match="boom"):
        await master.deploy(manifest)

    # The already-launched downstream container is stopped again.
    assert [handle.runtime_id for handle in driver.stop_calls] == ["shard-1"]
    assert docker_client.networks.by_name[NETWORK].removed is True
    assert not (tmp_path / "work" / EXECUTION_ID).exists()
    assert driver.wait_ready_calls == []


async def test_shutdown_stops_runtimes_removes_network_and_configs(
    tiny_llama_dir: Path, tmp_path: Path
) -> None:
    docker_client = FakeDockerClient()
    driver = RecordingDriver()
    master = make_master(docker_client, driver, tiny_llama_dir, tmp_path)
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
        execution_id=EXECUTION_ID, network=NETWORK, entry_endpoint="127.0.0.1:1", handles=()
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        deployment.network = "other"  # type: ignore[misc]
