"""EdgeShardShardRuntimeDriver unit tests with a fake Docker client.

The fake records launch wiring (labels/volumes/ports/naming, spec 21.5);
``wait_ready``/``info`` are exercised against a real in-process gRPC server
whose bound port the fake Docker "publishes". No Docker daemon required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import docker
import pytest
import yaml

from edgeshard.model.spec import BlockRange
from edgeshard.protocol.domain import ShardMessage
from edgeshard.protocol.grpc_server import start_shard_runtime_server
from edgeshard.runtime.drivers.base import DriverError, RuntimeHandle, RuntimeSpec
from edgeshard.runtime.drivers.edgeshard_shard import (
    EdgeShardShardRuntimeDriver,
    EdgeShardShardRuntimeSpec,
)
from edgeshard.runtime.info import RuntimeInfo, runtime_info_to_wire

EXECUTION_ID = "exec-driver"
RUNTIME_ID = "stage-0"
CONTAINER_PORT = 9100

RUNTIME_INFO = RuntimeInfo(
    runtime_id=RUNTIME_ID,
    model_id="tiny/llama",
    stage_index=0,
    stage_count=1,
    blocks=BlockRange(0, 4),
    include_input_stage=True,
    include_output_stage=True,
)


def write_container_config(
    tmp_path: Path, *, listen_host: str = "0.0.0.0", execution_id: str = EXECUTION_ID,
    runtime_id: str = RUNTIME_ID,
) -> Path:
    payload = {
        "runtime": {
            "backend": "edgeshard_shard",
            "runtime_id": runtime_id,
            "execution_id": execution_id,
        },
        "model": {"id": "tiny/llama", "path": "/models/tiny-llama"},
        "shard": {
            "start_block": 0,
            "end_block": 4,
            "include_input_stage": True,
            "include_output_stage": True,
        },
        "pipeline": {"stage_index": 0, "stage_count": 1},
        "server": {"listen_host": listen_host, "listen_port": CONTAINER_PORT},
    }
    path = tmp_path / "runtime.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


class FakeContainer:
    def __init__(self, container_id: str, bindings: dict[str, int]) -> None:
        self.id = container_id
        self.attrs = {
            "NetworkSettings": {
                "Ports": {
                    port: [{"HostIp": "0.0.0.0", "HostPort": str(host)}]
                    for port, host in bindings.items()
                }
            }
        }
        self.stop_timeout: int | None = None
        self.removed = False

    def reload(self) -> None:
        pass

    def stop(self, timeout: int) -> None:
        self.stop_timeout = timeout

    def remove(self) -> None:
        self.removed = True


class FakeContainers:
    def __init__(self) -> None:
        self.run_calls: list[dict[str, Any]] = []
        self.by_id: dict[str, FakeContainer] = {}

    def run(self, image: str, **kwargs: Any) -> FakeContainer:
        self.run_calls.append({"image": image, **kwargs})
        bindings: dict[str, int] = {}
        ports = kwargs.get("ports")
        if ports:
            (port_key, host_port), = ports.items()
            ephemeral = 49152 + len(self.by_id)
            bindings = {port_key: int(host_port or ephemeral)}
        container = FakeContainer(
            container_id=f"fake-{len(self.by_id)}",
            bindings=bindings,
        )
        self.by_id[container.id] = container
        return container

    def get(self, container_id: str) -> FakeContainer:
        if container_id not in self.by_id:
            raise docker.errors.NotFound(f"no such container: {container_id}")
        return self.by_id[container_id]


class FakeDockerClient:
    def __init__(self) -> None:
        self.containers = FakeContainers()


def make_spec(
    config_path: Path,
    host_port: int | None = 0,
    network: str | None = None,
) -> EdgeShardShardRuntimeSpec:
    return EdgeShardShardRuntimeSpec(
        backend="edgeshard_shard",
        runtime_id=RUNTIME_ID,
        execution_id=EXECUTION_ID,
        image="edgeshard/hf-shard:cpu",
        config_path=config_path,
        host_port=host_port,
        network=network,
    )


async def test_start_wires_labels_volumes_and_ports(tmp_path: Path) -> None:
    docker_client = FakeDockerClient()
    config_path = write_container_config(tmp_path)
    driver = EdgeShardShardRuntimeDriver(
        docker_client=docker_client, model_cache_dir=tmp_path / "model-cache"
    )

    handle = await driver.start(make_spec(config_path, host_port=55555))

    assert handle.runtime_id == RUNTIME_ID
    assert handle.endpoint == "127.0.0.1:55555"
    (call,) = docker_client.containers.run_calls
    assert call["image"] == "edgeshard/hf-shard:cpu"
    assert call["detach"] is True
    assert call["name"] == f"edgeshard-{EXECUTION_ID}-{RUNTIME_ID}"
    assert call["ports"] == {f"{CONTAINER_PORT}/tcp": 55555}
    assert call["labels"] == {
        "io.edgeshard.managed": "true",
        "io.edgeshard.execution_id": EXECUTION_ID,
        "io.edgeshard.runtime_id": RUNTIME_ID,
        "io.edgeshard.backend": "edgeshard_shard",
    }
    volumes = call["volumes"]
    model_cache = str(tmp_path / "model-cache")
    assert volumes[model_cache] == {
        "bind": EdgeShardShardRuntimeDriver.MODEL_MOUNT,
        "mode": "ro",
    }
    assert volumes[str(config_path.resolve())] == {
        "bind": EdgeShardShardRuntimeDriver.CONTAINER_CONFIG_PATH,
        "mode": "ro",
    }


async def test_start_picks_ephemeral_host_port(tmp_path: Path) -> None:
    driver = EdgeShardShardRuntimeDriver(
        docker_client=FakeDockerClient(), model_cache_dir=tmp_path
    )
    handle = await driver.start(make_spec(write_container_config(tmp_path)))
    assert handle.endpoint == "127.0.0.1:49152"


async def test_start_without_host_port_uses_network_endpoint(tmp_path: Path) -> None:
    docker_client = FakeDockerClient()
    driver = EdgeShardShardRuntimeDriver(
        docker_client=docker_client, model_cache_dir=tmp_path
    )
    handle = await driver.start(
        make_spec(write_container_config(tmp_path), host_port=None, network="edgeshard-exec-e")
    )
    # Not published: reachable only inside the Docker network via the alias.
    assert handle.endpoint == f"{RUNTIME_ID}:{CONTAINER_PORT}"
    (call,) = docker_client.containers.run_calls
    assert "ports" not in call


async def test_start_attaches_network_with_runtime_alias(tmp_path: Path) -> None:
    docker_client = FakeDockerClient()
    driver = EdgeShardShardRuntimeDriver(
        docker_client=docker_client, model_cache_dir=tmp_path
    )
    await driver.start(
        make_spec(write_container_config(tmp_path), network="edgeshard-exec-e")
    )
    (call,) = docker_client.containers.run_calls
    assert call["network"] == "edgeshard-exec-e"
    endpoints = call["networking_config"]
    assert set(endpoints) == {"edgeshard-exec-e"}
    assert endpoints["edgeshard-exec-e"]["Aliases"] == [RUNTIME_ID]


async def test_start_rejects_foreign_spec(tmp_path: Path) -> None:
    driver = EdgeShardShardRuntimeDriver(
        docker_client=FakeDockerClient(), model_cache_dir=tmp_path
    )
    foreign = RuntimeSpec(
        backend="edgeshard_shard",
        runtime_id=RUNTIME_ID,
        execution_id=EXECUTION_ID,
        image="edgeshard/hf-shard:cpu",
    )
    with pytest.raises(DriverError, match="EdgeShardShardRuntimeSpec"):
        await driver.start(foreign)


async def test_start_rejects_loopback_config(tmp_path: Path) -> None:
    driver = EdgeShardShardRuntimeDriver(
        docker_client=FakeDockerClient(), model_cache_dir=tmp_path
    )
    config_path = write_container_config(tmp_path, listen_host="127.0.0.1")
    with pytest.raises(DriverError, match="loopback"):
        await driver.start(make_spec(config_path))


async def test_start_rejects_identity_mismatch(tmp_path: Path) -> None:
    driver = EdgeShardShardRuntimeDriver(
        docker_client=FakeDockerClient(), model_cache_dir=tmp_path
    )
    config_path = write_container_config(tmp_path, execution_id="exec-other")
    with pytest.raises(DriverError, match="execution ID"):
        await driver.start(make_spec(config_path))
    config_path = write_container_config(tmp_path, runtime_id="stage-9")
    with pytest.raises(DriverError, match="runtime ID"):
        await driver.start(make_spec(config_path))


class _QuietHandler:
    """Handler good enough for GetRuntimeInfo; forwards are never used here."""

    async def create_session(self, execution_id: str, session_id: str) -> None:
        return None

    async def close_session(self, execution_id: str, session_id: str) -> None:
        return None

    async def prefill(self, message: ShardMessage) -> ShardMessage:
        raise AssertionError("unused")

    async def decode(self, message: ShardMessage) -> ShardMessage:
        raise AssertionError("unused")


async def test_wait_ready_and_info_against_live_server(tmp_path: Path) -> None:
    server, port = await start_shard_runtime_server(
        _QuietHandler(),
        runtime_info_to_wire(RUNTIME_INFO),
        host="127.0.0.1",
        port=0,
    )
    try:
        docker_client = FakeDockerClient()
        config_path = write_container_config(tmp_path)
        driver = EdgeShardShardRuntimeDriver(
            docker_client=docker_client, model_cache_dir=tmp_path
        )
        handle = await driver.start(make_spec(config_path, host_port=port))
        await driver.wait_ready(handle)
        assert await driver.info(handle) == RUNTIME_INFO
    finally:
        await server.stop(None)


async def test_wait_ready_times_out_on_dead_endpoint() -> None:
    driver = EdgeShardShardRuntimeDriver(
        docker_client=FakeDockerClient(),
        model_cache_dir=Path("."),
        ready_timeout_s=0.5,
        poll_interval_s=0.05,
    )
    dead = RuntimeHandle(runtime_id="gone", container_id="fake-x", endpoint="127.0.0.1:1")
    with pytest.raises(DriverError, match="not ready"):
        await driver.wait_ready(dead)


async def test_stop_stops_and_removes_container(tmp_path: Path) -> None:
    docker_client = FakeDockerClient()
    driver = EdgeShardShardRuntimeDriver(
        docker_client=docker_client, model_cache_dir=tmp_path
    )
    handle = await driver.start(
        make_spec(write_container_config(tmp_path), host_port=55555)
    )
    container = docker_client.containers.by_id[handle.container_id]

    await driver.stop(handle)
    assert container.stop_timeout == 10
    assert container.removed

    # Stopping again is not an error: the container is already gone.
    del docker_client.containers.by_id[handle.container_id]
    await driver.stop(handle)
