"""Worker-side assembly of identity, capability, and state (Phase 1 spec §27).

Runs the local portion of the Agent lifecycle - identity, static discovery,
capability revision, telemetry sample, model and runtime inventory - without
any Master contact. ``worker inspect`` (spec §43) prints the result;
``worker serve`` (milestone P1G) continues from these same pieces into
registration and heartbeats.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import docker

from edgeshard.cluster.capability import (
    WorkerCapability,
    compute_capability_revision,
)
from edgeshard.cluster.identity import WorkerIdentity
from edgeshard.cluster.state import DeviceState, WorkerState
from edgeshard.control.worker.config import WorkerConfig
from edgeshard.control.worker.discovery.base import CapabilityFragment, CapabilityProbe
from edgeshard.control.worker.discovery.host import HostCapabilityProbe
from edgeshard.control.worker.discovery.nvidia import NvidiaCapabilityProbe
from edgeshard.control.worker.identity import IdentityManager, build_worker_identity
from edgeshard.control.worker.model_inventory import scan_model_inventory
from edgeshard.control.worker.runtime_inventory import scan_runtime_inventory
from edgeshard.control.worker.telemetry.base import TelemetryProbe
from edgeshard.control.worker.telemetry.host import HostTelemetryProbe
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


async def inspect_local_worker(
    config: WorkerConfig,
    *,
    capability_probes: Sequence[CapabilityProbe] | None = None,
    telemetry_probes: Sequence[TelemetryProbe] | None = None,
    docker_client: Any = _UNSET,
) -> LocalInspection:
    """The local portion of the Agent lifecycle (spec §27), Master-less.

    Probe and client injection points exist for tests; production callers
    rely on the defaults (host + NVIDIA probes, psutil/NVML telemetry,
    Docker SDK). Backends without the corresponding hardware report empty
    fragments rather than failing (spec §47).
    """
    worker_id = IdentityManager(config.worker.identity_path).load_or_create()
    identity = build_worker_identity(worker_id)

    if capability_probes is None:
        capability_probes = (HostCapabilityProbe(worker_id), NvidiaCapabilityProbe())
    capability = assemble_capability([probe.discover() for probe in capability_probes])

    samplers = (
        telemetry_probes
        if telemetry_probes is not None
        else (
            HostTelemetryProbe(worker_id, cpu_sample_interval_s=0.25),
            NvidiaTelemetryProbe(),
        )
    )
    state_fragments = [await sampler.sample() for sampler in samplers]

    models = scan_model_inventory(ModelStore(model_root=config.model_store.root))
    runtime_instances = ()
    if config.runtime.discover_managed_containers:
        runtime_instances = _scan_runtimes(docker_client)

    state = WorkerState(
        worker_id=worker_id,
        device_states=enrich_device_states(
            [device for fragment in state_fragments for device in fragment.device_states],
            runtime_instances,
        ),
        memory_states=tuple(
            memory for fragment in state_fragments for memory in fragment.memory_states
        ),
        runtime_instances=runtime_instances,
        models=models,
    )
    logger.info(
        "local inspection complete worker_id=%s devices=%d pools=%d runtimes=%d models=%d",
        worker_id,
        len(state.device_states),
        len(state.memory_states),
        len(runtime_instances),
        len(models),
    )
    return LocalInspection(identity=identity, capability=capability, state=state)


def _scan_runtimes(docker_client: Any) -> tuple[Any, ...]:
    client = docker_client
    if client is _UNSET:
        try:
            client = docker.from_env()
        except Exception as exc:  # daemon missing is non-fatal (spec §47)
            logger.warning("Docker daemon unavailable; runtime inventory empty: %s", exc)
            return ()
    if client is None:
        return ()
    try:
        return scan_runtime_inventory(client)
    except Exception as exc:  # transient daemon failure is non-fatal (spec §47)
        logger.warning("runtime inventory failed; reporting empty: %s", exc)
        return ()


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
