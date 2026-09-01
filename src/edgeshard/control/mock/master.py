"""Mock Master (spec 22): manifest-driven deployment, generation, cleanup.

A thin integration orchestrator: it executes a provided
:class:`DeploymentManifest` — creates the execution's Docker network,
generates per-runtime configs, launches the containers through the
runtime driver, waits for readiness, and tears everything down on
shutdown. It never discovers workers, profiles hardware, selects devices,
calculates partitions, optimizes, persists cluster state, or implements
production retry/recovery (spec 22.1).
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import docker

from edgeshard.control.mock.deployment import (
    host_model_path,
    network_name,
    write_runtime_configs,
)
from edgeshard.control.mock.manifest import DeploymentManifest, ManifestError
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


class MockMasterError(EdgeShardError):
    """Deployment lifecycle failure of the Mock Master."""


@dataclass(frozen=True)
class Deployment:
    """A deployed execution: network, entry endpoint, runtime handles."""

    execution_id: str
    network: str
    entry_endpoint: str
    handles: tuple[RuntimeHandle, ...]
    """Runtime handles in pipeline order (entry first)."""


class MockMaster:
    """Deploys and cleans up shard pipelines from a manifest (spec 22)."""

    def __init__(
        self,
        *,
        docker_client: Any,
        model_cache_dir: Path,
        work_dir: Path,
        default_image: str,
        driver: RuntimeDriver | None = None,
    ) -> None:
        self._docker = docker_client
        self._model_cache_dir = Path(model_cache_dir)
        self._work_dir = Path(work_dir)
        self._default_image = default_image
        self._driver = driver or EdgeShardShardRuntimeDriver(
            docker_client=docker_client, model_cache_dir=self._model_cache_dir
        )

    async def deploy(
        self, manifest: DeploymentManifest, *, entry_host_port: int = 0
    ) -> Deployment:
        """Deploy the manifest; a failed deploy leaves nothing running.

        Stages launch in reverse pipeline order so the cascaded readiness
        check resolves bottom-up; only the entry runtime publishes a host
        port (spec 23), and waiting for the entry means the whole chain is
        ready.
        """
        self._validate_partition_bounds(manifest)
        network = network_name(manifest.execution_id)
        await self._create_network(network)
        config_dir = self._config_dir(manifest.execution_id)
        config_dir.mkdir(parents=True, exist_ok=True)
        config_paths = write_runtime_configs(manifest, config_dir)
        handles: list[RuntimeHandle] = []
        try:
            for stage_index in reversed(range(len(manifest.pipeline))):
                runtime = manifest.pipeline_runtimes()[stage_index]
                spec = EdgeShardShardRuntimeSpec(
                    backend=runtime.backend,
                    runtime_id=runtime.id,
                    execution_id=manifest.execution_id,
                    image=runtime.image or self._default_image,
                    config_path=config_paths[runtime.id],
                    host_port=entry_host_port if stage_index == 0 else None,
                    network=network,
                )
                handles.append(await self._driver.start(spec))
            handles.reverse()
            await self._driver.wait_ready(handles[0])
        except Exception:
            await self._abort_deploy(manifest.execution_id, handles, network)
            raise
        return Deployment(
            execution_id=manifest.execution_id,
            network=network,
            entry_endpoint=handles[0].endpoint,
            handles=tuple(handles),
        )

    async def shutdown(self, deployment: Deployment) -> None:
        """Stop all runtimes, remove the network, delete generated configs."""
        for handle in reversed(deployment.handles):
            await self._driver.stop(handle)
        await self._remove_network(deployment.network)
        shutil.rmtree(self._config_dir(deployment.execution_id), ignore_errors=True)

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
        if last.shard.end != num_blocks:
            raise ManifestError(
                f"partition ends at block {last.shard.end} but model "
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
                await self._driver.stop(handle)
        with contextlib.suppress(Exception):
            await self._remove_network(network)
        shutil.rmtree(self._config_dir(execution_id), ignore_errors=True)
