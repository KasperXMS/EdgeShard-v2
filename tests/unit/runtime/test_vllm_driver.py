"""VLLMRuntimeDriver unit tests with a fake Docker client (spec 20.2).

The fake records launch wiring (entrypoint/command/labels/volumes/ports);
``wait_ready`` is exercised against a real in-process HTTP server whose
bound port the fake Docker "publishes", so the OpenAI-compatible
readiness poll runs for real without any Docker daemon or GPU.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import docker
import pytest

from edgeshard.runtime.drivers.base import DriverError, RuntimeHandle, RuntimeSpec
from edgeshard.runtime.drivers.vllm import (
    DEFAULT_VLLM_IMAGE,
    VLLM_API_PORT,
    VLLMRuntimeDriver,
    VLLMRuntimeSpec,
)

EXECUTION_ID = "exec-vllm"
RUNTIME_ID = "vllm-0"
MODEL_PATH = "/models/tiny-llama"


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


class _ModelsHandler(BaseHTTPRequestHandler):
    """Answers the OpenAI-compatible readiness endpoint only."""

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": MODEL_PATH}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def http_models_server() -> Any:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def make_spec(**overrides: Any) -> VLLMRuntimeSpec:
    fields: dict[str, Any] = {
        "backend": "vllm",
        "runtime_id": RUNTIME_ID,
        "execution_id": EXECUTION_ID,
        "image": DEFAULT_VLLM_IMAGE,
        "model_id": "tiny/llama",
        "model_path": Path(MODEL_PATH),
    }
    fields.update(overrides)
    return VLLMRuntimeSpec(**fields)


async def test_start_wires_vllm_cli_labels_volumes_and_ports(tmp_path: Path) -> None:
    docker_client = FakeDockerClient()
    driver = VLLMRuntimeDriver(
        docker_client=docker_client, model_cache_dir=tmp_path / "model-cache"
    )

    handle = await driver.start(make_spec(host_port=55555))

    assert handle.runtime_id == RUNTIME_ID
    assert handle.backend == "vllm"
    assert handle.endpoint == "127.0.0.1:55555"
    (call,) = docker_client.containers.run_calls
    assert call["image"] == DEFAULT_VLLM_IMAGE
    assert call["detach"] is True
    assert call["name"] == f"edgeshard-{EXECUTION_ID}-{RUNTIME_ID}"
    # vLLM native CLI drives the official image, unmodified (spec 20.2).
    assert call["entrypoint"] == ["vllm"]
    assert call["command"] == [
        "serve",
        MODEL_PATH,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_API_PORT),
    ]
    assert call["ports"] == {f"{VLLM_API_PORT}/tcp": 55555}
    assert call["labels"] == {
        "io.edgeshard.managed": "true",
        "io.edgeshard.execution_id": EXECUTION_ID,
        "io.edgeshard.runtime_id": RUNTIME_ID,
        "io.edgeshard.backend": "vllm",
    }
    volumes = call["volumes"]
    model_cache = str(tmp_path / "model-cache")
    assert volumes[model_cache] == {"bind": "/models", "mode": "ro"}
    # No vLLM knobs set -> no extra CLI args, no device pinning.
    assert "--max-model-len" not in call["command"]
    assert "--tensor-parallel-size" not in call["command"]
    assert "environment" not in call


async def test_start_translates_optional_knobs(tmp_path: Path) -> None:
    docker_client = FakeDockerClient()
    driver = VLLMRuntimeDriver(
        docker_client=docker_client, model_cache_dir=tmp_path
    )
    await driver.start(
        make_spec(device_index=1, max_model_len=2048, tensor_parallel_size=2)
    )
    (call,) = docker_client.containers.run_calls
    assert call["command"][-4:] == [
        "--max-model-len",
        "2048",
        "--tensor-parallel-size",
        "2",
    ]
    assert call["environment"] == {"CUDA_VISIBLE_DEVICES": "1"}


async def test_start_without_host_port_uses_network_endpoint(tmp_path: Path) -> None:
    docker_client = FakeDockerClient()
    driver = VLLMRuntimeDriver(
        docker_client=docker_client, model_cache_dir=tmp_path
    )
    handle = await driver.start(make_spec(host_port=None, network="edgeshard-exec-e"))
    assert handle.endpoint == f"{RUNTIME_ID}:{VLLM_API_PORT}"
    (call,) = docker_client.containers.run_calls
    assert "ports" not in call
    assert call["network"] == "edgeshard-exec-e"
    assert call["networking_config"]["edgeshard-exec-e"]["Aliases"] == [RUNTIME_ID]


async def test_start_rejects_path_outside_model_mount(tmp_path: Path) -> None:
    driver = VLLMRuntimeDriver(
        docker_client=FakeDockerClient(), model_cache_dir=tmp_path
    )
    with pytest.raises(DriverError, match="model mount"):
        await driver.start(make_spec(model_path=Path("/weights/tiny-llama")))


async def test_start_rejects_foreign_spec(tmp_path: Path) -> None:
    driver = VLLMRuntimeDriver(
        docker_client=FakeDockerClient(), model_cache_dir=tmp_path
    )
    foreign = RuntimeSpec(
        backend="vllm",
        runtime_id=RUNTIME_ID,
        execution_id=EXECUTION_ID,
        image=DEFAULT_VLLM_IMAGE,
    )
    with pytest.raises(DriverError, match="VLLMRuntimeSpec"):
        await driver.start(foreign)


async def test_wait_ready_polls_openai_models_endpoint(
    http_models_server: str, tmp_path: Path
) -> None:
    driver = VLLMRuntimeDriver(
        docker_client=FakeDockerClient(), model_cache_dir=tmp_path
    )
    handle = RuntimeHandle(
        runtime_id=RUNTIME_ID,
        backend="vllm",
        container_id="fake-0",
        endpoint=http_models_server,
    )
    await driver.wait_ready(handle)


async def test_wait_ready_times_out_on_dead_endpoint(tmp_path: Path) -> None:
    driver = VLLMRuntimeDriver(
        docker_client=FakeDockerClient(),
        model_cache_dir=tmp_path,
        ready_timeout_s=0.5,
        poll_interval_s=0.05,
    )
    dead = RuntimeHandle(
        runtime_id="gone", backend="vllm", container_id="fake-x", endpoint="127.0.0.1:1"
    )
    with pytest.raises(DriverError, match="not ready"):
        await driver.wait_ready(dead)


async def test_info_reports_full_model_coverage(
    tiny_llama_dir: Path, tmp_path: Path
) -> None:
    cache_dir = tiny_llama_dir.parent
    docker_client = FakeDockerClient()
    driver = VLLMRuntimeDriver(docker_client=docker_client, model_cache_dir=cache_dir)
    handle = await driver.start(
        make_spec(model_path=Path(f"/models/{tiny_llama_dir.name}"))
    )

    info = await driver.info(handle)

    assert info.runtime_id == RUNTIME_ID
    assert info.model_id == "tiny/llama"
    assert info.stage_index == 0
    assert info.stage_count == 1
    assert (info.blocks.start, info.blocks.end) == (0, 4)
    assert info.include_input_stage and info.include_output_stage


async def test_info_rejects_unknown_container(tmp_path: Path) -> None:
    driver = VLLMRuntimeDriver(
        docker_client=FakeDockerClient(), model_cache_dir=tmp_path
    )
    unknown = RuntimeHandle(
        runtime_id=RUNTIME_ID, backend="vllm", container_id="nope", endpoint="x:1"
    )
    with pytest.raises(DriverError, match="not started by this driver"):
        await driver.info(unknown)


async def test_stop_stops_and_removes_container(tmp_path: Path) -> None:
    docker_client = FakeDockerClient()
    driver = VLLMRuntimeDriver(
        docker_client=docker_client, model_cache_dir=tmp_path
    )
    handle = await driver.start(make_spec(host_port=55555))
    container = docker_client.containers.by_id[handle.container_id]

    await driver.stop(handle)
    assert container.stop_timeout == 10
    assert container.removed

    # Stopping again is not an error: the container is already gone.
    del docker_client.containers.by_id[handle.container_id]
    await driver.stop(handle)
