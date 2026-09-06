"""Runtime inventory from Docker labels/state (Phase 1 spec §19).

Observational reconstruction of EdgeShard-managed runtime containers: the
Worker Agent can restart while containers keep running, so inventory is
re-read from Docker each time, never assumed from Agent memory. The labels
are the ones the Phase 0 drivers apply (``edgeshard.runtime.labels``).

Docker SDK objects stay inside this module (spec §60); callers only see
cluster domain types. A container missing its identity labels cannot form a
valid ``RuntimeInstanceState`` and is skipped with a warning.

Best-effort attribution beyond the labels: ``device_ids`` come from explicit
GPU-UUID assignments in the container's device requests or
``NVIDIA_VISIBLE_DEVICES`` (shorthands like ``all`` or CUDA ordinals are
never mapped to stable ids by guessing); ``endpoint`` is the published host
address when exactly one tcp port is mapped; ``model_local_name`` is read
from the optional ``io.edgeshard.model_local_name`` label. Anything not
observable stays empty/``None`` — inventory never invents facts (§19, §47).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from edgeshard.cluster.inventory import RuntimeInstanceState, RuntimeStatus
from edgeshard.runtime import labels

logger = logging.getLogger("worker.inventory.runtimes")

_NVIDIA_VISIBLE_DEVICES = "NVIDIA_VISIBLE_DEVICES"
_WILDCARD_HOST_IPS = frozenset({"", "0.0.0.0", "::"})


def scan_runtime_inventory(docker_client: Any) -> tuple[RuntimeInstanceState, ...]:
    """Reconstruct the inventory of managed containers (spec §19)."""
    containers = docker_client.containers.list(all=True, filters={"label": labels.MANAGED})
    instances: list[RuntimeInstanceState] = []
    for container in containers:
        instance = _to_instance(container)
        if instance is not None:
            instances.append(instance)
    return tuple(instances)


def _to_instance(container: Any) -> RuntimeInstanceState | None:
    container_labels = container.labels or {}
    runtime_id = container_labels.get(labels.RUNTIME_ID)
    backend = container_labels.get(labels.BACKEND)
    if not runtime_id or not backend:
        logger.warning(
            "container %s carries managed label but no runtime identity; skipped",
            getattr(container, "id", "<unknown>"),
        )
        return None
    model_local_name = container_labels.get(labels.MODEL_LOCAL_NAME)
    return RuntimeInstanceState(
        runtime_id=str(runtime_id),
        backend=str(backend),
        execution_id=container_labels.get(labels.EXECUTION_ID),
        status=_runtime_status(container),
        device_ids=_device_ids(container),
        container_id=str(container.id),
        endpoint=_endpoint(container),
        model_local_name=str(model_local_name) if model_local_name else None,
    )


def _attrs(container: Any) -> dict[str, Any]:
    """The container's raw inspect data; ``{}`` when not (yet) loaded."""
    try:
        attrs = container.attrs
    except AttributeError:
        return {}
    return attrs if isinstance(attrs, dict) else {}


def _device_ids(container: Any) -> tuple[str, ...]:
    """Explicit device attribution from inspect data (spec §19).

    Only unambiguous assignments are reported: ``DeviceRequests`` carrying
    concrete ``DeviceIDs``, else ``NVIDIA_VISIBLE_DEVICES`` listing GPU
    UUIDs. ``all``/``none``/CUDA ordinals cannot be resolved to stable
    device ids from inside this module, so attribution stays empty rather
    than guessed (§11: ordinals are never identity).
    """
    attrs = _attrs(container)
    ids: list[str] = []
    host_config = attrs.get("HostConfig") or {}
    for request in host_config.get("DeviceRequests") or []:
        if not isinstance(request, dict):
            continue
        ids.extend(
            device_id
            for device_id in request.get("DeviceIDs") or []
            if isinstance(device_id, str) and device_id
        )
    if not ids:
        config = attrs.get("Config") or {}
        for entry in config.get("Env") or []:
            if (
                isinstance(entry, str)
                and entry.startswith(f"{_NVIDIA_VISIBLE_DEVICES}=")
            ):
                ids.extend(_gpu_uuid_tokens(entry.split("=", 1)[1]))
                break
    # De-duplicate while preserving order.
    return tuple(dict.fromkeys(ids))


def _gpu_uuid_tokens(value: str) -> list[str]:
    """GPU UUID tokens of ``NVIDIA_VISIBLE_DEVICES``; ``[]`` for shorthands."""
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if tokens and all(_is_gpu_uuid(token) for token in tokens):
        return tokens
    return []


def _is_gpu_uuid(token: str) -> bool:
    candidate = token[4:] if token.startswith("GPU-") else token
    try:
        uuid.UUID(candidate)
    except ValueError:
        return False
    return True


def _endpoint(container: Any) -> str | None:
    """Published host endpoint when exactly one tcp port is mapped.

    With zero or multiple mappings there is no single answer, so the
    endpoint stays ``None`` rather than guessed. Wildcard Docker host ips
    are normalized to loopback, the only address Phase 1 runtimes serve on.
    """
    attrs = _attrs(container)
    network = attrs.get("NetworkSettings") or {}
    ports = network.get("Ports") or {}
    found: list[str] = []
    for container_port, bindings in ports.items():
        if not isinstance(container_port, str) or not container_port.endswith("/tcp"):
            continue
        for binding in bindings or []:
            if not isinstance(binding, dict):
                continue
            host_port = binding.get("HostPort")
            if not host_port:
                continue
            host_ip = str(binding.get("HostIp") or "")
            if host_ip in _WILDCARD_HOST_IPS:
                host_ip = "127.0.0.1"
            found.append(f"{host_ip}:{host_port}")
    return found[0] if len(found) == 1 else None


def _runtime_status(container: Any) -> RuntimeStatus:
    status = str(getattr(container, "status", "") or "").lower()
    if status == "running":
        return RuntimeStatus.RUNNING
    if status == "created":
        return RuntimeStatus.CREATED
    if status == "exited":
        exit_code = _exit_code(container)
        if exit_code is not None and exit_code != 0:
            return RuntimeStatus.FAILED
        return RuntimeStatus.STOPPED
    if status == "dead":
        return RuntimeStatus.FAILED
    return RuntimeStatus.UNKNOWN


def _exit_code(container: Any) -> int | None:
    state = _attrs(container).get("State")
    code = state.get("ExitCode") if isinstance(state, dict) else None
    return int(code) if isinstance(code, int) else None
