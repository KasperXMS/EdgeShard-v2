"""EdgeShardShardRuntimeDriver (spec 20.1): shard runtimes in Docker.

The driver launches one container per shard stage via the Docker SDK (spec
5.6: no scattered ``docker run`` shell calls). Models are mounted
read-only, never baked (spec 21.1); managed containers carry the spec 21.5
labels so orphaned Phase 0 containers can be found and cleaned up.
Readiness is the ShardRuntime ``GetRuntimeInfo`` RPC (spec 17.1), matching
the process-level runtime from 0F. Configs with ``device.type: cuda``
launch with every host GPU attached (NVIDIA DeviceRequest, the SDK
equivalent of ``--gpus all``); CPU configs launch without one.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grpc

from edgeshard.protocol.grpc_client import ShardRuntimeClient
from edgeshard.runtime.config import ShardRuntimeConfig
from edgeshard.runtime.drivers.base import (
    DriverError,
    RuntimeHandle,
    RuntimeSpec,
    container_network_kwargs,
    nvidia_gpu_device_request,
    published_host_port,
    stop_and_remove_container,
)
from edgeshard.runtime.info import RuntimeInfo, runtime_info_from_wire

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


@dataclass(frozen=True)
class EdgeShardShardRuntimeSpec(RuntimeSpec):
    """Launch material for one EdgeShard shard runtime container.

    ``config_path`` is a host path: the driver validates it, then mounts it
    read-only into the container. The container binds the config's own
    ``server.listen_port``; the driver publishes it on ``host_port``
    (0 lets Docker pick an ephemeral port, ``None`` publishes nothing —
    spec 23: only the entry runtime needs a host port). When ``network``
    is set, the container joins that Docker network with the runtime ID as
    its alias, so shard configs can name downstream stages as
    ``<runtime-id>:<port>`` without any host IP (spec 23).
    """

    config_path: Path
    host_port: int | None = 0
    network: str | None = None


class EdgeShardShardRuntimeDriver:
    """Container lifecycle for EdgeShard shard runtimes."""

    CONTAINER_CONFIG_PATH = "/runtime/config/runtime.yaml"
    MODEL_MOUNT = "/models"

    def __init__(
        self,
        *,
        docker_client: Any,
        model_cache_dir: Path,
        ready_timeout_s: float = 120.0,
        poll_interval_s: float = 0.25,
    ) -> None:
        self._docker = docker_client
        self._model_cache_dir = Path(model_cache_dir)
        self._ready_timeout_s = ready_timeout_s
        self._poll_interval_s = poll_interval_s

    async def start(self, spec: RuntimeSpec) -> RuntimeHandle:
        if not isinstance(spec, EdgeShardShardRuntimeSpec):
            raise DriverError(
                "EdgeShardShardRuntimeDriver requires an EdgeShardShardRuntimeSpec"
            )
        config = await asyncio.to_thread(ShardRuntimeConfig.from_yaml, spec.config_path)
        if config.server.listen_host in _LOOPBACK_HOSTS:
            raise DriverError(
                f"container runtime config binds loopback host "
                f"{config.server.listen_host!r}; it would be unreachable from "
                f"outside the container"
            )
        if config.runtime.execution_id != spec.execution_id:
            raise DriverError(
                f"config execution ID {config.runtime.execution_id!r} does not "
                f"match spec execution ID {spec.execution_id!r}"
            )
        if config.runtime.runtime_id != spec.runtime_id:
            raise DriverError(
                f"config runtime ID {config.runtime.runtime_id!r} does not "
                f"match spec runtime ID {spec.runtime_id!r}"
            )
        container_port = config.server.listen_port

        def run() -> Any:
            kwargs: dict[str, Any] = {
                "detach": True,
                "name": f"edgeshard-{spec.execution_id}-{spec.runtime_id}",
                "volumes": {
                    str(self._model_cache_dir): {"bind": self.MODEL_MOUNT, "mode": "ro"},
                    str(Path(spec.config_path).resolve()): {
                        "bind": self.CONTAINER_CONFIG_PATH,
                        "mode": "ro",
                    },
                },
                "labels": {
                    "io.edgeshard.managed": "true",
                    "io.edgeshard.execution_id": spec.execution_id,
                    "io.edgeshard.runtime_id": spec.runtime_id,
                    "io.edgeshard.backend": spec.backend,
                },
            }
            if spec.host_port is not None:
                kwargs["ports"] = {f"{container_port}/tcp": spec.host_port or None}
            if spec.network is not None:
                kwargs.update(container_network_kwargs(spec.network, spec.runtime_id))
            if config.device.type == "cuda":
                # All host GPUs, like the manually validated `--gpus all`;
                # the runtime process itself picks `cuda:{device.index}`
                # (no physical remapping here). CPU configs get nothing.
                kwargs["device_requests"] = [nvidia_gpu_device_request()]
            return self._docker.containers.run(spec.image, **kwargs)

        container = await asyncio.to_thread(run)
        if spec.host_port is None:
            # Not published to the host: addressable only inside the Docker
            # network via the runtime-ID alias (spec 23).
            endpoint = f"{spec.runtime_id}:{container_port}"
        else:
            host_port = await published_host_port(container, container_port)
            endpoint = f"127.0.0.1:{host_port}"
        return RuntimeHandle(
            runtime_id=spec.runtime_id,
            backend=spec.backend,
            container_id=str(container.id),
            endpoint=endpoint,
        )

    async def wait_ready(self, handle: RuntimeHandle) -> None:
        """Poll GetRuntimeInfo until the runtime answers (spec 17.1)."""
        deadline = time.monotonic() + self._ready_timeout_s
        async with ShardRuntimeClient(handle.endpoint) as client:
            while True:
                try:
                    await client.get_runtime_info()
                    return
                except grpc.aio.AioRpcError:
                    if time.monotonic() > deadline:
                        raise DriverError(
                            f"runtime {handle.runtime_id!r} at {handle.endpoint} "
                            f"not ready within {self._ready_timeout_s}s"
                        ) from None
                    await asyncio.sleep(self._poll_interval_s)

    async def info(self, handle: RuntimeHandle) -> RuntimeInfo:
        async with ShardRuntimeClient(handle.endpoint) as client:
            wire = await client.get_runtime_info()
        return runtime_info_from_wire(wire)

    async def stop(self, handle: RuntimeHandle) -> None:
        """Stop and remove the container; an already-gone container is not an error."""
        await stop_and_remove_container(self._docker, handle.container_id)
