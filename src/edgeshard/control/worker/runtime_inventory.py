"""Runtime inventory from Docker labels/state (Phase 1 spec §19).

Observational reconstruction of EdgeShard-managed runtime containers: the
Worker Agent can restart while containers keep running, so inventory is
re-read from Docker each time, never assumed from Agent memory. The labels
are the ones the Phase 0 drivers apply (``edgeshard.runtime.labels``).

Docker SDK objects stay inside this module (spec §60); callers only see
cluster domain types. A container missing its identity labels cannot form a
valid ``RuntimeInstanceState`` and is skipped with a warning.

P1B gaps by design: ``device_ids`` attribution needs the NVML/Jetson device
ids (milestones P1C/P1D), and ``endpoint``/``model_local_name`` have no
label source yet - both stay ``None``/empty until then.
"""

from __future__ import annotations

import logging
from typing import Any

from edgeshard.cluster.inventory import RuntimeInstanceState, RuntimeStatus
from edgeshard.runtime import labels

logger = logging.getLogger("worker.inventory.runtimes")


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
    return RuntimeInstanceState(
        runtime_id=str(runtime_id),
        backend=str(backend),
        execution_id=container_labels.get(labels.EXECUTION_ID),
        status=_runtime_status(container),
        device_ids=(),
        container_id=str(container.id),
        endpoint=None,
        model_local_name=None,
    )


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
    try:
        state = container.attrs.get("State") or {}
        code = state.get("ExitCode")
    except AttributeError:
        return None
    return int(code) if isinstance(code, int) else None
