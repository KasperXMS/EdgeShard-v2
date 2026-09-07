"""Worker-side assembly of identity, capability, and state (Phase 1 spec §27).

Runs the local portion of the Agent lifecycle - identity, static discovery,
capability revision, telemetry sample, model and runtime inventory - without
any Master contact. ``worker inspect`` (spec §43) prints the result of the
one-shot :func:`inspect_local_worker`; ``worker serve`` (milestone P1G)
keeps a :class:`LocalWorkerInspector` alive instead, so the expensive
*static* work (identity, capability discovery, Docker client, ModelStore
walk, tegrastats subprocess) happens once at startup and every heartbeat
pays only the cheap *dynamic* work (telemetry, runtime inventory, TTL-based
refreshes).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import docker

from edgeshard.cluster.capability import (
    RuntimePlatformCapability,
    WorkerCapability,
    compute_capability_revision,
)
from edgeshard.cluster.identity import WorkerIdentity
from edgeshard.cluster.inventory import ModelInventoryEntry
from edgeshard.cluster.state import DeviceState, WorkerState
from edgeshard.control.worker.config import WorkerConfig
from edgeshard.control.worker.discovery.base import CapabilityFragment, CapabilityProbe
from edgeshard.control.worker.discovery.host import HostCapabilityProbe
from edgeshard.control.worker.discovery.jetson import JetsonPlatformProbe, is_jetson_host
from edgeshard.control.worker.discovery.nvidia import NvidiaCapabilityProbe
from edgeshard.control.worker.identity import IdentityManager, build_worker_identity
from edgeshard.control.worker.model_inventory import scan_model_inventory
from edgeshard.control.worker.runtime_inventory import scan_runtime_inventory
from edgeshard.control.worker.telemetry.base import TelemetryProbe
from edgeshard.control.worker.telemetry.host import HostTelemetryProbe
from edgeshard.control.worker.telemetry.jetson import JetsonTelemetryBackend
from edgeshard.control.worker.telemetry.nvidia import NvidiaTelemetryProbe
from edgeshard.runtime.model_store import ModelStore

logger = logging.getLogger("worker.agent")

_UNSET: Any = object()


@dataclass(frozen=True)
class LocalInspection:
    """Master-less view of one Worker: identity, capability, current state."""

    identity: WorkerIdentity
    capability: WorkerCapability
    state: WorkerState


def assemble_capability(fragments: Sequence[CapabilityFragment]) -> WorkerCapability:
    """Merge probe fragments into one finalized capability (spec §16, §27).

    Scalar facts (architecture, OS, container runtime) may be contributed by
    exactly one probe; conflicting reports are an internal inconsistency and
    fail loudly (spec §47). Collections are concatenated.
    """
    architecture = _single(fragments, "architecture")
    os_info = _single(fragments, "os")
    if not architecture:
        raise ValueError("no probe reported an architecture")
    if os_info is None:
        raise ValueError("no probe reported OS information")
    container_runtime = _single(fragments, "container_runtime")

    capability = WorkerCapability(
        architecture=architecture,
        os=os_info,
        container_runtime=container_runtime,
        network_interfaces=_concat(fragments, "network_interfaces"),
        devices=_concat(fragments, "devices"),
        memory_pools=_concat(fragments, "memory_pools"),
        runtime_platforms=_concat(fragments, "runtime_platforms"),
        capability_revision="",
    )
    return dataclasses.replace(
        capability, capability_revision=compute_capability_revision(capability)
    )


def _single(fragments: Sequence[CapabilityFragment], field: str) -> Any:
    provided = [
        value
        for value in (getattr(fragment, field) for fragment in fragments)
        if value is not None
    ]
    if not provided:
        return None
    first = provided[0]
    if any(value != first for value in provided[1:]):
        raise ValueError(f"conflicting capability facts for {field!r}: {provided!r}")
    return first


def _concat(fragments: Sequence[CapabilityFragment], field: str) -> tuple[Any, ...]:
    merged: list[Any] = []
    for fragment in fragments:
        merged.extend(getattr(fragment, field))
    return tuple(merged)


def enrich_device_states(
    device_states: Sequence[DeviceState],
    runtime_instances: Iterable[Any],
) -> tuple[DeviceState, ...]:
    """Attach observed runtime instances to the devices they occupy (§17)."""
    runtimes_by_device: dict[str, list[str]] = {}
    for instance in runtime_instances:
        for device_id in instance.device_ids:
            runtimes_by_device.setdefault(device_id, []).append(instance.runtime_id)
    return tuple(
        dataclasses.replace(
            state, running_runtime_ids=tuple(runtimes_by_device.get(state.device_id, ()))
        )
        for state in device_states
    )


class LocalWorkerInspector:
    """Long-lived local inspection for ``worker serve`` (spec §27, §24, §16).

    Inverts the one-shot model so a heartbeat every few seconds stays cheap:

    * :meth:`start` runs the *static* work once — identity load, one shared
      Docker client, full capability discovery, the initial ModelStore walk,
      telemetry-probe creation. On Jetson the tegrastats subprocess the
      backend spawns stays up for the whole ``worker serve`` lifetime
      (§24: one long-lived reader, never a process per heartbeat);
    * :meth:`inspect` performs only *dynamic* work — a telemetry sample and
      the runtime-inventory re-read — and refreshes the cached capability /
      model inventory only once their configured intervals
      (``worker.capability_refresh_interval_s``,
      ``model_store.inventory_refresh_interval_s``) have elapsed;
    * :meth:`close` releases the resources this class owns: the Docker
      client it created and default telemetry samplers (tegrastats).

    Operator-declared ``runtime.platforms`` from config are merged into the
    capability (§15) — declared facts, never auto-guessed ones.

    Probes or a Docker client injected through the constructor are owned by
    the caller (injected samplers are *not* closed by :meth:`close`);
    ``docker_client=None`` disables runtime inventory and container-runtime
    discovery. ``monotonic`` is injectable so tests drive refresh intervals
    without sleeping.
    """

    def __init__(
        self,
        config: WorkerConfig,
        *,
        capability_probes: Sequence[CapabilityProbe] | None = None,
        telemetry_probes: Sequence[TelemetryProbe] | None = None,
        docker_client: Any = _UNSET,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._monotonic = monotonic
        self._injected_capability_probes = capability_probes
        self._injected_telemetry_probes = telemetry_probes
        self._requested_docker_client = docker_client

        self._worker_id: str | None = None
        self._identity: WorkerIdentity | None = None
        self._docker_client: Any = None
        self._owns_docker_client = False
        self._capability_probes: Sequence[CapabilityProbe] = ()
        self._telemetry_probes: Sequence[TelemetryProbe] = ()
        self._owns_telemetry = False
        self._nvidia_probe: NvidiaCapabilityProbe | None = None
        self._capability: WorkerCapability | None = None
        self._capability_at = float("-inf")
        self._models: tuple[ModelInventoryEntry, ...] = ()
        self._models_at = float("-inf")
        self._latest_state: WorkerState | None = None
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    def current_state(self) -> WorkerState | None:
        """Latest Phase 1 NVML/tegrastats sample for synchronous instruments."""
        return self._latest_state

    async def start(self) -> None:
        """One-time static discovery; idempotent."""
        if self._started:
            return
        worker_id = IdentityManager(self._config.worker.identity_path).load_or_create()
        self._worker_id = worker_id
        self._identity = build_worker_identity(worker_id)
        self._init_docker_client()
        self._init_probes(worker_id)
        await self._refresh_capability()
        await self._refresh_models()
        self._started = True
        logger.info(
            "inspector started worker_id=%s revision=%s models=%d",
            worker_id,
            self._capability.capability_revision if self._capability else "-",
            len(self._models),
        )

    async def inspect(self) -> LocalInspection:
        """Fresh dynamic state against the cached static capability (§27)."""
        if not self._started:
            raise RuntimeError("LocalWorkerInspector.inspect() called before start()")
        now = self._monotonic()
        if now - self._capability_at >= self._config.worker.capability_refresh_interval_s:
            logger.info("capability refresh interval elapsed; re-discovering")
            await self._refresh_capability()
        if now - self._models_at >= self._config.model_store.inventory_refresh_interval_s:
            await self._refresh_models()

        state_fragments = [await probe.sample() for probe in self._telemetry_probes]
        runtime_instances = ()
        if self._config.runtime.discover_managed_containers:
            runtime_instances = await asyncio.to_thread(self._scan_runtimes)

        assert self._worker_id is not None
        assert self._identity is not None
        assert self._capability is not None
        state = WorkerState(
            worker_id=self._worker_id,
            device_states=enrich_device_states(
                [device for fragment in state_fragments for device in fragment.device_states],
                runtime_instances,
            ),
            memory_states=tuple(
                memory for fragment in state_fragments for memory in fragment.memory_states
            ),
            runtime_instances=runtime_instances,
            models=self._models,
        )
        self._latest_state = state
        logger.info(
            "local inspection complete worker_id=%s devices=%d pools=%d runtimes=%d models=%d",
            self._worker_id,
            len(state.device_states),
            len(state.memory_states),
            len(runtime_instances),
            len(self._models),
        )
        return LocalInspection(
            identity=self._identity, capability=self._capability, state=state
        )

    async def close(self) -> None:
        """Release owned resources; idempotent (spec §24 lifecycle)."""
        if self._owns_telemetry:
            await _close_samplers(self._telemetry_probes)
        self._owns_telemetry = False
        self._latest_state = None
        if self._owns_docker_client and self._docker_client is not None:
            try:
                self._docker_client.close()
            except Exception as exc:  # best-effort cleanup (spec §47)
                logger.warning("docker client close failed: %s", exc)
        self._owns_docker_client = False
        self._started = False

    async def __aenter__(self) -> LocalWorkerInspector:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Startup wiring
    # ------------------------------------------------------------------

    def _init_docker_client(self) -> None:
        requested = self._requested_docker_client
        if requested is not _UNSET:
            self._docker_client = requested  # caller-owned; None disables
            return
        try:
            self._docker_client = docker.from_env()
        except Exception as exc:  # daemon missing is non-fatal (spec §47)
            logger.warning("Docker daemon unavailable; runtime inventory empty: %s", exc)
            self._docker_client = None
        else:
            self._owns_docker_client = True

    def _init_probes(self, worker_id: str) -> None:
        """Platform-appropriate probe sets (spec §22-23).

        Jetson is never modeled as a discrete NVML GPU host: the dedicated
        platform probe/backend replaces the host + NVIDIA pair. Capability
        probes share the inspector's single Docker client; NVIDIA telemetry
        reuses the discovery probe's stable device-id mapping so state can
        never drift from capability ids (spec §11).
        """
        factory = self._docker_client_factory
        if self._injected_capability_probes is not None:
            self._capability_probes = self._injected_capability_probes
        elif is_jetson_host():
            self._capability_probes = (JetsonPlatformProbe(worker_id, factory),)
        else:
            self._nvidia_probe = NvidiaCapabilityProbe()
            self._capability_probes = (
                HostCapabilityProbe(worker_id, factory),
                self._nvidia_probe,
            )

        if self._injected_telemetry_probes is not None:
            self._telemetry_probes = self._injected_telemetry_probes
            self._owns_telemetry = False
            return
        self._owns_telemetry = True
        if is_jetson_host():
            self._telemetry_probes = (
                JetsonTelemetryBackend(worker_id, cpu_sample_interval_s=0.25),
            )
        else:
            nvidia_probe = self._nvidia_probe
            self._telemetry_probes = (
                HostTelemetryProbe(worker_id, cpu_sample_interval_s=0.25),
                NvidiaTelemetryProbe(
                    static_device_ids=(
                        (lambda: nvidia_probe.discovered_device_ids)
                        if nvidia_probe is not None
                        else None
                    )
                ),
            )

    def _docker_client_factory(self) -> Any:
        """Hand the *shared* client to probes; fail into their no-daemon path."""
        if self._docker_client is None:
            raise RuntimeError("Docker client unavailable on this Worker")
        return self._docker_client

    # ------------------------------------------------------------------
    # Cached-fact refresh
    # ------------------------------------------------------------------

    async def _refresh_capability(self) -> None:
        fragments = [
            await asyncio.to_thread(probe.discover) for probe in self._capability_probes
        ]
        fragments.append(self._declared_platforms())
        self._capability = assemble_capability(fragments)
        self._capability_at = self._monotonic()

    def _declared_platforms(self) -> CapabilityFragment:
        """Operator-declared runtime platforms (spec §15).

        Appended after probe-contributed platforms; the Worker reports the
        backends/images an operator provisioned, never a guess derived from
        what happens to be installed.
        """
        return CapabilityFragment(
            runtime_platforms=tuple(
                RuntimePlatformCapability(
                    backend=entry.backend,
                    platform=entry.platform,
                    image=entry.image,
                )
                for entry in self._config.runtime.platforms
            )
        )

    async def _refresh_models(self) -> None:
        store = ModelStore(model_root=self._config.model_store.root)
        self._models = await asyncio.to_thread(scan_model_inventory, store)
        self._models_at = self._monotonic()

    def _scan_runtimes(self) -> tuple[Any, ...]:
        client = self._docker_client
        if client is None:
            return ()
        try:
            return scan_runtime_inventory(client)
        except Exception as exc:  # transient daemon failure is non-fatal (spec §47)
            logger.warning("runtime inventory failed; reporting empty: %s", exc)
            return ()


async def _close_samplers(samplers: Sequence[TelemetryProbe]) -> None:
    """Stop default samplers that hold resources (tegrastats, spec §24)."""
    for sampler in samplers:
        close = getattr(sampler, "close", None)
        if callable(close):
            await close()


async def inspect_local_worker(
    config: WorkerConfig,
    *,
    capability_probes: Sequence[CapabilityProbe] | None = None,
    telemetry_probes: Sequence[TelemetryProbe] | None = None,
    docker_client: Any = _UNSET,
) -> LocalInspection:
    """One-shot convenience over :class:`LocalWorkerInspector` (spec §43).

    ``worker inspect`` and single-shot callers use this; ``worker serve``
    keeps the inspector alive across heartbeats instead. Probe and client
    injection points exist for tests; production callers rely on the
    platform-appropriate defaults (host + NVIDIA probes on generic/RTX
    hosts, the dedicated Jetson backend on Jetson - spec §23). Backends
    without the corresponding hardware report empty fragments rather than
    failing (spec §47). Everything created for the single inspection
    (Docker client, default samplers such as the tegrastats reader) is
    closed before returning.
    """
    inspector = LocalWorkerInspector(
        config,
        capability_probes=capability_probes,
        telemetry_probes=telemetry_probes,
        docker_client=docker_client,
    )
    async with inspector:
        return await inspector.inspect()


def to_plain_mapping(value: object) -> object:
    """Cluster domain structures as plain JSON/YAML-safe data.

    The cluster domain stays free of serialization concerns (spec §60); this
    adapter is the boundary for ``worker inspect`` output.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_plain_mapping(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [to_plain_mapping(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value
