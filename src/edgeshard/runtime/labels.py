"""Container labels identifying EdgeShard-managed runtimes (spec §19 seam).

Phase 0 drivers apply these labels to every container they start. The Phase 1
Worker Agent reconstructs its runtime inventory observationally from them
(Phase 1 spec §19): containers outlive Agent restarts, so inventory must be
re-read from Docker labels rather than assumed to live in Agent memory.
"""

from __future__ import annotations

MANAGED = "io.edgeshard.managed"
"""Presence label marking a container as EdgeShard-managed."""

EXECUTION_ID = "io.edgeshard.execution_id"
"""The execution (deployment) the runtime belongs to."""

RUNTIME_ID = "io.edgeshard.runtime_id"
"""The logical runtime identity assigned by the deployment."""

BACKEND = "io.edgeshard.backend"
"""The runtime backend owning the container's lifecycle."""

MODEL_LOCAL_NAME = "io.edgeshard.model_local_name"
"""Optional label carrying the ModelStore local name a runtime serves.

Phase 0 drivers do not apply it; the Phase 1 runtime inventory reads it
when present so ``RuntimeInstanceState.model_local_name`` can be filled
without guessing (spec §19).
"""


def managed_container_labels(
    *, execution_id: str, runtime_id: str, backend: str
) -> dict[str, str]:
    """The full label set drivers apply to one managed container."""
    return {
        MANAGED: "true",
        EXECUTION_ID: execution_id,
        RUNTIME_ID: runtime_id,
        BACKEND: backend,
    }
