"""Mock Master (spec 22): manifest-driven deployment, generation, cleanup.

A thin integration orchestrator: it executes a provided
:class:`DeploymentManifest` — creates the execution's Docker network,
generates per-runtime configs, launches every runtime through its backend
driver, waits for readiness, and tears everything down on shutdown. The
lifecycle itself is backend-neutral (invariant 7): launch order,
readiness, and cleanup run through the :class:`RuntimeDriver` protocol
for every runtime; per-backend knowledge is confined to choosing the
driver and building that backend's launch material. It never discovers
workers, profiles hardware, selects devices, calculates partitions,
optimizes, persists cluster state, or implements production
retry/recovery (spec 22.1).
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import docker

from edgeshard.control.mock.deployment import (
    host_model_path,
    network_name,
    write_runtime_configs,
)
from edgeshard.control.mock.manifest import (
    DeploymentManifest,
    ManifestError,
    ManifestRuntime,
)
from edgeshard.model.adapters.registry import (
    default_registry,
    resolve_adapter_for_source,
)
from edgeshard.model.errors import EdgeShardError
from edgeshard.model.source import ModelSource
from edgeshard.runtime.drivers.base import RuntimeDriver, RuntimeHandle
from edgeshard.runtime.drivers.edgeshard_shard import (
    EdgeShardShardRuntimeDriver,
    EdgeShardShardRuntimeSpec,
)
from edgeshard.runtime.drivers.vllm import (
    DEFAULT_VLLM_IMAGE,
    VLLMRuntimeDriver,
    VLLMRuntimeSpec,
)


class MockMasterError(EdgeShardError):
    """Deployment lifecycle failure of the Mock Master."""


@dataclass(frozen=True)
class Deployment:
    """A deployed execution: network, runtime handles, entry endpoint.

    ``handles`` lists the shard pipeline first (pipeline order, entry
    first), then the standalone runtimes in manifest order.
    ``entry_endpoint`` is the host endpoint of the shard pipeline's entry
    runtime; it is ``None`` for deployments without a shard pipeline.
    """

    execution_id: str
    network: str
    entry_endpoint: str | None
    handles: tuple[RuntimeHandle, ...]

    def handle(self, runtime_id: str) -> RuntimeHandle:
        for handle in self.handles:
            if handle.runtime_id == runtime_id:
                return handle
        raise KeyError(f"unknown runtime id: {runtime_id!r}")


class MockMaster:
    """Deploys and cleans up runtime deployments from a manifest (spec 22)."""

    def __init__(
        self,
        *,
        docker_client: Any,
        model_cache_dir: Path,
        work_dir: Path,
        default_image: str,
        drivers: Mapping[str, RuntimeDriver] | None = None,
    ) -> None:
        self._docker = docker_client
        self._model_cache_dir = Path(model_cache_dir)
        self._work_dir = Path(work_dir)
        self._default_image = default_image
        self._drivers: dict[str, RuntimeDriver] = {
            "edgeshard_shard": EdgeShardShardRuntimeDriver(
                docker_client=docker_client, model_cache_dir=self._model_cache_dir
            ),
            "vllm": VLLMRuntimeDriver(
                docker_client=docker_client, model_cache_dir=self._model_cache_dir
            ),
        }
        if drivers:
            self._drivers.update(drivers)

    async def deploy(
        self, manifest: DeploymentManifest, *, entry_host_port: int = 0
    ) -> Deployment:
        """Deploy the manifest; a failed deploy leaves nothing running.

        Standalone runtimes start first so their (slow) model load
        overlaps with shard startup; shard stages launch in reverse
        pipeline order so the cascaded readiness check resolves
        bottom-up. Only the entry shard runtime and the standalone
        runtimes publish host ports (spec 23): waiting for the entry
        means the whole shard chain is ready, and each standalone runtime
        is waited for through its own driver.
        """
        shard_runtimes = manifest.pipeline_runtimes()
        standalone_runtimes = manifest.standalone_runtimes()
        if shard_runtimes:
            self._validate_partition_bounds(manifest)
        network = network_name(manifest.execution_id)
        await self._create_network(network)
        config_dir = self._config_dir(manifest.execution_id)
        config_dir.mkdir(parents=True, exist_ok=True)
        config_paths = write_runtime_configs(manifest, config_dir)
        shard_handles: list[RuntimeHandle] = []
        standalone_handles: list[RuntimeHandle] = []
        try:
            for runtime in standalone_runtimes:
                standalone_spec = self._standalone_spec(manifest, runtime, network)
                standalone_handles.append(
                    await self._driver_for(runtime.backend).start(standalone_spec)
                )
            for stage_index in reversed(range(len(shard_runtimes))):
                runtime = shard_runtimes[stage_index]
                shard_spec = EdgeShardShardRuntimeSpec(
                    backend=runtime.backend,
                    runtime_id=runtime.id,
                    execution_id=manifest.execution_id,
                    image=runtime.image or self._default_image,
                    config_path=config_paths[runtime.id],
                    host_port=entry_host_port if stage_index == 0 else None,
                    network=network,
                )
                shard_handles.append(
                    await self._driver_for(runtime.backend).start(shard_spec)
                )
            shard_handles.reverse()
            if shard_handles:
                await self._driver_for(shard_runtimes[0].backend).wait_ready(
                    shard_handles[0]
                )
            for runtime, handle in zip(
                standalone_runtimes, standalone_handles, strict=True
            ):
                await self._driver_for(runtime.backend).wait_ready(handle)
        except Exception:
            await self._abort_deploy(
                manifest.execution_id, shard_handles + standalone_handles, network
            )
            raise
        return Deployment(
            execution_id=manifest.execution_id,
            network=network,
            entry_endpoint=shard_handles[0].endpoint if shard_handles else None,
            handles=tuple(shard_handles + standalone_handles),
        )

    async def shutdown(self, deployment: Deployment) -> None:
        """Stop all runtimes, remove the network, delete generated configs."""
        for handle in reversed(deployment.handles):
            await self._driver_for(handle.backend).stop(handle)
        await self._remove_network(deployment.network)
        shutil.rmtree(self._config_dir(deployment.execution_id), ignore_errors=True)

    def _driver_for(self, backend: str) -> RuntimeDriver:
        try:
            return self._drivers[backend]
        except KeyError:
            raise MockMasterError(
                f"no driver registered for backend {backend!r}"
            ) from None

    def _standalone_spec(
        self, manifest: DeploymentManifest, runtime: ManifestRuntime, network: str
    ) -> VLLMRuntimeSpec:
        """Launch material for an independent full-model runtime (spec 19.2).

        Standalone runtimes always publish a host port: the master's test
        requests reach them from the host (spec 22.1).
        """
        options = runtime.vllm
        return VLLMRuntimeSpec(
            backend=runtime.backend,
            runtime_id=runtime.id,
            execution_id=manifest.execution_id,
            image=runtime.image or DEFAULT_VLLM_IMAGE,
            model_id=manifest.model.id,
            model_path=manifest.model.path,
            host_port=0,
            network=network,
            device_index=runtime.device.index,
            max_model_len=options.max_model_len if options else None,
            tensor_parallel_size=options.tensor_parallel_size if options else None,
        )

    def _config_dir(self, execution_id: str) -> Path:
        return self._work_dir / execution_id

    def _validate_partition_bounds(self, manifest: DeploymentManifest) -> None:
        """The provided partition must cover the whole model (spec 22.1).

        Only the upper bound is checked here: contiguity and stage roles
        are static and already enforced when the manifest parses. This
        reads the model's ``config.json`` — never its weights.
        """
        source = ModelSource(
            path=host_model_path(manifest, self._model_cache_dir),
            model_id=manifest.model.id,
        )
        source.ensure_local()
        adapter = resolve_adapter_for_source(source, default_registry())
        num_blocks = adapter.inspect(source).num_blocks
        last = manifest.pipeline_runtimes()[-1]
        shard = last.shard
        if shard is None:  # unreachable: manifests validate shard presence
            raise ManifestError(f"runtime {last.id!r} has no shard section")
        if shard.end != num_blocks:
            raise ManifestError(
                f"partition ends at block {shard.end} but model "
                f"{manifest.model.id!r} has {num_blocks} blocks"
            )

    async def _create_network(self, name: str) -> None:
        def create() -> None:
            self._docker.networks.create(name, driver="bridge")

        try:
            await asyncio.to_thread(create)
        except docker.errors.APIError as exc:
            raise MockMasterError(
                f"failed to create Docker network {name!r}: {exc}"
            ) from exc

    async def _remove_network(self, name: str) -> None:
        """Remove the execution network; an already-gone network is not an error."""

        def remove() -> None:
            try:
                network = self._docker.networks.get(name)
            except docker.errors.NotFound:
                return
            network.remove()

        try:
            await asyncio.to_thread(remove)
        except docker.errors.APIError as exc:
            raise MockMasterError(
                f"failed to remove Docker network {name!r}: {exc}"
            ) from exc

    async def _abort_deploy(
        self, execution_id: str, handles: list[RuntimeHandle], network: str
    ) -> None:
        """Best-effort cleanup after a failed deploy; the primary error wins."""
        for handle in handles:
            with contextlib.suppress(Exception):
                await self._driver_for(handle.backend).stop(handle)
        with contextlib.suppress(Exception):
            await self._remove_network(network)
        shutil.rmtree(self._config_dir(execution_id), ignore_errors=True)
