"""RuntimeDriver abstraction (spec 20).

Lifecycle normalization belongs on the control side: drivers start, probe,
describe, and stop runtimes. The native inference protocol of each backend
stays untouched (spec 4.7) — EdgeShard shards speak the ShardRuntime gRPC
service; vLLM keeps its OpenAI-compatible HTTP API (0J).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import docker

from edgeshard.model.errors import EdgeShardError
from edgeshard.runtime.info import RuntimeInfo

MODEL_MOUNT = "/models"
"""Container-side model mount; models are mounted, never baked (spec 21.1)."""


class DriverError(EdgeShardError):
    """Runtime lifecycle failure reported by a driver."""


@dataclass(frozen=True)
class RuntimeSpec:
    """Driver-agnostic request to launch one runtime."""

    backend: str
    runtime_id: str
    execution_id: str
    image: str


@dataclass(frozen=True)
class RuntimeHandle:
    """A started runtime and its primary endpoint.

    The endpoint is host-addressable when the driver published a port; for
    network-only runtimes (spec 23) it is resolvable only inside the
    deployment's Docker network. ``backend`` names the driver that owns
    this runtime's lifecycle.
    """

    runtime_id: str
    backend: str
    container_id: str
    endpoint: str


class RuntimeDriver(Protocol):
    """Normalized lifecycle across runtime backends (spec 20)."""

    async def start(self, spec: RuntimeSpec) -> RuntimeHandle:
        ...

    async def wait_ready(self, handle: RuntimeHandle) -> None:
        ...

    async def info(self, handle: RuntimeHandle) -> RuntimeInfo:
        ...

    async def stop(self, handle: RuntimeHandle) -> None:
        ...


def host_model_path(container_path: Path | str, model_cache_dir: Path | str) -> Path:
    """Map a container-side model path under ``MODEL_MOUNT`` onto the host.

    ``/models/<rel>`` inside the container corresponds to
    ``model_cache_dir/<rel>`` on the host (the drivers mount the cache
    directory at ``MODEL_MOUNT``). Paths outside the mount fail explicitly.
    """
    path = Path(container_path)
    if not path.is_relative_to(MODEL_MOUNT):
        raise DriverError(
            f"model path {path.as_posix()!r} must live under the container "
            f"model mount {MODEL_MOUNT!r}"
        )
    return Path(model_cache_dir) / path.relative_to(MODEL_MOUNT)


def container_network_kwargs(network: str, runtime_id: str) -> dict[str, Any]:
    """docker-py kwargs joining ``network`` with ``runtime_id`` as alias.

    docker-py consumes the endpoint mapping as a plain dict alongside
    ``network``; a pre-wrapped NetworkingConfig fails its sanity check
    there and silently drops the alias.
    """
    return {
        "network": network,
        "networking_config": {
            network: docker.types.EndpointConfig(
                docker.constants.DEFAULT_DOCKER_API_VERSION,
                aliases=[runtime_id],
            )
        },
    }


def nvidia_gpu_device_request() -> docker.types.DeviceRequest:
    """The NVIDIA DeviceRequest equivalent of ``docker run --gpus all``.

    ``count=-1`` with the ``gpu`` capability hands every host GPU to the
    container through the NVIDIA Container Toolkit — the same wiring that
    was validated manually before the drivers gained GPU support. Which
    visible device a runtime actually uses stays its own concern
    (``config.device.index`` inside shard runtimes,
    ``CUDA_VISIBLE_DEVICES`` for vLLM); Phase 0 does no physical GPU
    remapping or placement here.
    """
    return docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])


async def stop_and_remove_container(
    docker_client: Any, container_id: str, *, timeout_s: int = 10
) -> None:
    """Stop and remove a container; an already-gone container is not an error."""

    def stop_and_remove() -> None:
        try:
            container = docker_client.containers.get(container_id)
        except docker.errors.NotFound:
            return
        try:
            container.stop(timeout=timeout_s)
        except docker.errors.NotFound:
            return
        container.remove()

    try:
        await asyncio.to_thread(stop_and_remove)
    except docker.errors.APIError as exc:
        raise DriverError(
            f"failed to stop container {container_id!r}: {exc}"
        ) from exc


async def published_host_port(container: Any, container_port: int) -> int:
    """Host port ``container_port`` is published on; DriverError when absent."""

    def probe() -> int:
        container.reload()
        bindings = (
            container.attrs.get("NetworkSettings", {})
            .get("Ports", {})
            .get(f"{container_port}/tcp")
        )
        if not bindings:
            raise DriverError(f"container port {container_port} is not published")
        return int(bindings[0]["HostPort"])

    return await asyncio.to_thread(probe)
