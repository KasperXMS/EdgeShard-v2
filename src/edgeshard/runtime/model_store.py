"""Worker-local model store (spec 24 seam).

Every worker/device owns a :class:`ModelStore` rooted at its own host
directory (``model_root``, default :data:`DEFAULT_MODEL_ROOT`). Deployment
plans identify models by id and local name only — never by host path — and
each worker resolves the local name against its own store
(:meth:`ModelStore.host_path`). Runtime drivers mount the store root
read-only at :data:`MODEL_MOUNT`, so containers and the inference layer see
one uniform container path form (:func:`container_model_path`) regardless of
where a worker keeps its models.

Phase 0 stores are prepared externally (spec 24): snapshots materialized
e.g. via ``huggingface_hub.snapshot_download``. Model download, eviction,
and multi-disk placement are later-phase concerns that build on this
abstraction; nothing here implements them.

This module imports only the leaf error module: ``drivers/__init__.py``
eagerly imports the driver modules, which in turn construct stores, so any
import of :mod:`edgeshard.runtime.drivers` here would close a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from edgeshard.model.errors import EdgeShardError

MODEL_MOUNT = "/models"
"""Container-side model mount; models are mounted, never baked (spec 21.1).

Every driver mounts its store's ``model_root`` at this path, so runtimes
address models as ``/models/<local-name>`` no matter which host directory a
worker chose.
"""

DEFAULT_MODEL_ROOT = Path("/data/edgeshard-models")
"""Default host directory of a worker's model store.

Workers on devices with other layouts override this per store; the default
matches the conventional single-disk cache location.
"""


class ModelStoreError(EdgeShardError):
    """A model store is misconfigured or a local name cannot be resolved."""


def validate_local_name(local_name: str) -> str:
    """A model's local name is a single safe directory segment.

    The name becomes one path segment under ``model_root`` (and under
    ``MODEL_MOUNT`` inside containers), so separators, parent-directory
    names, and whitespace are rejected. Plain dots are allowed — real
    snapshot names carry them (e.g. ``Qwen2.5-7B``).
    """
    if not local_name:
        raise ModelStoreError("model local name must be non-empty")
    if "/" in local_name or "\\" in local_name:
        raise ModelStoreError(
            f"model local name {local_name!r} must be a single path segment"
        )
    if local_name in (".", ".."):
        raise ModelStoreError(
            f"model local name {local_name!r} must not name a parent directory"
        )
    if any(char.isspace() for char in local_name):
        raise ModelStoreError(
            f"model local name {local_name!r} must not contain whitespace"
        )
    return local_name


def container_model_path(local_name: str) -> Path:
    """Uniform container path of one model, independent of any store root.

    Different workers' ``model_root`` directories all map to this same
    form inside their containers.
    """
    return Path(MODEL_MOUNT) / validate_local_name(local_name)


@dataclass(frozen=True)
class ModelStore:
    """One worker's locally cached model snapshots under ``model_root``.

    The store is the worker-side translation point between plan-level model
    identity (id + local name) and host filesystem layout: drivers mount
    ``model_root`` at ``MODEL_MOUNT`` and resolve snapshots through
    :meth:`host_path`. ``model_root`` must exist for container deployments;
    it is not created here and — deliberately — not checked for absoluteness
    (a relative root fails loudly when Docker binds it).
    """

    model_root: Path = DEFAULT_MODEL_ROOT

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_root", Path(self.model_root))

    def host_path(self, local_name: str) -> Path:
        """The snapshot directory of ``local_name`` on this worker's host."""
        return self.model_root / validate_local_name(local_name)

    def container_path(self, local_name: str) -> Path:
        """The path of ``local_name`` inside a container (uniform form)."""
        return container_model_path(local_name)
