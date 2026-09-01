"""VLLMRuntimeDriver (spec 20.2): independent full-model vLLM runtimes.

Launches the official pinned vLLM OpenAI image (spec 5.7) through the
Docker SDK, driven with the vLLM native CLI (``vllm serve``). Only the
lifecycle is normalized — start, readiness, describe, stop — while the
data plane stays vLLM-native: an OpenAI-compatible HTTP API that the Mock
Master can issue test requests to (specs 4.7, 22.1). The vLLM image is
never modified to look like an EdgeShard shard runtime (spec 20.2).

Readiness is ``GET /v1/models`` answering 200: vLLM opens its API server
only once the model weights are loaded.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from edgeshard.model.adapters.registry import (
    default_registry,
    resolve_adapter_for_source,
)
from edgeshard.model.source import ModelSource
from edgeshard.model.spec import BlockRange
from edgeshard.runtime.drivers.base import (
    MODEL_MOUNT,
    DriverError,
    RuntimeHandle,
    RuntimeSpec,
    container_network_kwargs,
    host_model_path,
    published_host_port,
    stop_and_remove_container,
)
from edgeshard.runtime.info import RuntimeInfo

DEFAULT_VLLM_IMAGE = "vllm/vllm-openai:v0.28.0"
"""Pinned official vLLM OpenAI image (specs 5.7, 4.10); manifests may override."""

VLLM_API_PORT = 8000
"""In-container port of vLLM's OpenAI-compatible API server."""

_REQUEST_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class VLLMRuntimeSpec(RuntimeSpec):
    """Launch material for one independent full-model vLLM runtime.

    ``model_path`` is the container-side model path under ``MODEL_MOUNT``.
    The container serves the OpenAI-compatible API on ``api_port``; the
    driver publishes it on ``host_port`` like shard runtimes (0 lets
    Docker pick an ephemeral port, ``None`` publishes nothing). Set
    optional fields translate into official vLLM CLI arguments (spec
    19.2); ``device_index`` maps to ``CUDA_VISIBLE_DEVICES``. Unset
    fields keep vLLM's own defaults.
    """

    model_id: str
    model_path: Path
    api_port: int = VLLM_API_PORT
    host_port: int | None = 0
    network: str | None = None
    device_index: int | None = None
    max_model_len: int | None = None
    tensor_parallel_size: int | None = None


class VLLMRuntimeDriver:
    """Container lifecycle for independent vLLM runtimes."""

    def __init__(
        self,
        *,
        docker_client: Any,
        model_cache_dir: Path,
        ready_timeout_s: float = 600.0,
        poll_interval_s: float = 1.0,
    ) -> None:
        self._docker = docker_client
        self._model_cache_dir = Path(model_cache_dir)
        self._ready_timeout_s = ready_timeout_s
        self._poll_interval_s = poll_interval_s
        self._started: dict[str, VLLMRuntimeSpec] = {}

    async def start(self, spec: RuntimeSpec) -> RuntimeHandle:
        if not isinstance(spec, VLLMRuntimeSpec):
            raise DriverError("VLLMRuntimeDriver requires a VLLMRuntimeSpec")
        # Fail before launch when the model path escapes the model mount.
        host_model_path(spec.model_path, self._model_cache_dir)
        command = self._serve_command(spec)

        def run() -> Any:
            kwargs: dict[str, Any] = {
                "detach": True,
                "name": f"edgeshard-{spec.execution_id}-{spec.runtime_id}",
                # Official image, unmodified (spec 20.2): drive it with the
                # vLLM native CLI regardless of the image's own entrypoint.
                "entrypoint": ["vllm"],
                "command": command,
                "volumes": {
                    str(self._model_cache_dir): {"bind": MODEL_MOUNT, "mode": "ro"},
                },
                "labels": {
                    "io.edgeshard.managed": "true",
                    "io.edgeshard.execution_id": spec.execution_id,
                    "io.edgeshard.runtime_id": spec.runtime_id,
                    "io.edgeshard.backend": spec.backend,
                },
            }
            if spec.device_index is not None:
                kwargs["environment"] = {"CUDA_VISIBLE_DEVICES": str(spec.device_index)}
            if spec.host_port is not None:
                kwargs["ports"] = {f"{spec.api_port}/tcp": spec.host_port or None}
            if spec.network is not None:
                kwargs.update(container_network_kwargs(spec.network, spec.runtime_id))
            return self._docker.containers.run(spec.image, **kwargs)

        container = await asyncio.to_thread(run)
        if spec.host_port is None:
            endpoint = f"{spec.runtime_id}:{spec.api_port}"
        else:
            host_port = await published_host_port(container, spec.api_port)
            endpoint = f"127.0.0.1:{host_port}"
        handle = RuntimeHandle(
            runtime_id=spec.runtime_id,
            backend=spec.backend,
            container_id=str(container.id),
            endpoint=endpoint,
        )
        self._started[handle.container_id] = spec
        return handle

    async def wait_ready(self, handle: RuntimeHandle) -> None:
        """Poll ``GET /v1/models`` until it answers 200.

        vLLM opens its OpenAI-compatible API server only after the model
        is loaded, so a 200 here is the readiness signal.
        """
        deadline = time.monotonic() + self._ready_timeout_s
        url = f"http://{handle.endpoint}/v1/models"
        async with httpx.AsyncClient() as client:
            while True:
                try:
                    response = await client.get(url, timeout=_REQUEST_TIMEOUT_S)
                    if response.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                if time.monotonic() > deadline:
                    raise DriverError(
                        f"runtime {handle.runtime_id!r} at {handle.endpoint} "
                        f"not ready within {self._ready_timeout_s}s"
                    ) from None
                await asyncio.sleep(self._poll_interval_s)

    async def info(self, handle: RuntimeHandle) -> RuntimeInfo:
        """Model coverage of a full-model runtime: one stage, every block.

        vLLM does not speak the shard wire protocol; the driver derives
        the coverage from the mounted model's ``config.json`` — never its
        weights.
        """
        spec = self._started.get(handle.container_id)
        if spec is None:
            raise DriverError(
                f"runtime {handle.runtime_id!r} was not started by this driver"
            )
        source = ModelSource(
            path=host_model_path(spec.model_path, self._model_cache_dir),
            model_id=spec.model_id,
        )
        source.ensure_local()
        adapter = resolve_adapter_for_source(source, default_registry())
        num_blocks = adapter.inspect(source).num_blocks
        return RuntimeInfo(
            runtime_id=spec.runtime_id,
            model_id=spec.model_id,
            stage_index=0,
            stage_count=1,
            blocks=BlockRange(0, num_blocks),
            include_input_stage=True,
            include_output_stage=True,
        )

    async def stop(self, handle: RuntimeHandle) -> None:
        """Stop and remove the container; an already-gone container is not an error."""
        self._started.pop(handle.container_id, None)
        await stop_and_remove_container(self._docker, handle.container_id)

    @staticmethod
    def _serve_command(spec: VLLMRuntimeSpec) -> list[str]:
        command = [
            "serve",
            Path(spec.model_path).as_posix(),
            "--host",
            "0.0.0.0",
            "--port",
            str(spec.api_port),
        ]
        if spec.max_model_len is not None:
            command += ["--max-model-len", str(spec.max_model_len)]
        if spec.tensor_parallel_size is not None:
            command += ["--tensor-parallel-size", str(spec.tensor_parallel_size)]
        return command
