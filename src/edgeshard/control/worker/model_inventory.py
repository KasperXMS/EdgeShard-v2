"""Worker-side ModelStore scanning (Phase 1 spec §20).

Reuses :class:`edgeshard.runtime.model_store.ModelStore` (spec §3.5 keeps
the store worker-local) and classifies each snapshot directory. Only
logical model identity and status are produced - host paths never leave the
Worker (spec §20). Phase 1 inventory is observational: nothing is
downloaded, repaired, or evicted.

Classification is deliberately conservative:

* ``ready`` - parseable ``config.json`` and complete weights: when a
  safetensors index (``model.safetensors.index.json``) exists, *every*
  shard its ``weight_map`` references must be present; without an index, at
  least one weight file is the best observable evidence;
* ``incomplete`` - ``config.json`` present but weights missing, or an index
  referencing shards that are not all on disk (e.g. an interrupted
  multi-shard download); an unparseable index also lands here rather than
  claiming readiness that cannot be proven;
* ``invalid`` - ``config.json`` missing/unparseable, or a directory name the
  ModelStore cannot address.

``model_id`` is recovered opportunistically from ``config.json``'s
``_name_or_path`` when it looks like a model identifier rather than a host
path; ``revision`` stays ``None`` until snapshot metadata parsing lands.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from edgeshard.cluster.inventory import ModelAvailability, ModelInventoryEntry
from edgeshard.runtime.model_store import ModelStore, ModelStoreError, validate_local_name

logger = logging.getLogger("worker.inventory.models")

_WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".pt", ".pth", ".gguf"})
_INDEX_FILENAME = "model.safetensors.index.json"
"""Multi-shard safetensors index; when present it is authoritative for
completeness (mirrors :mod:`edgeshard.model.weights.safetensors` but parsed
with plain ``json`` — importing torch here would be absurd overhead for an
inventory scan)."""


def scan_model_inventory(store: ModelStore) -> tuple[ModelInventoryEntry, ...]:
    """Classify every snapshot directory under the store root.

    A missing store root is legitimate on a fresh Worker and yields an empty
    inventory rather than an error.
    """
    root = store.model_root
    if not root.is_dir():
        logger.info("model store root %s does not exist; inventory empty", root)
        return ()
    entries: list[ModelInventoryEntry] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_dir():
            continue
        entries.append(_scan_entry(path))
    return tuple(entries)


def _scan_entry(path: Path) -> ModelInventoryEntry:
    local_name = path.name
    try:
        validate_local_name(local_name)
    except ModelStoreError:
        # Unaddressable directory: report it broken rather than hide it.
        return ModelInventoryEntry(
            local_name=local_name,
            model_id=None,
            revision=None,
            size_bytes=None,
            status=ModelAvailability.INVALID,
        )

    model_id: str | None = None
    has_config = False
    config_path = path / "config.json"
    if config_path.is_file():
        payload = _read_json(config_path)
        if isinstance(payload, dict):
            has_config = True
            candidate = payload.get("_name_or_path")
            if isinstance(candidate, str) and candidate and not _looks_like_host_path(candidate):
                # Never let host paths cross to the Master (spec §20).
                model_id = candidate

    status = _weight_status(path) if has_config else ModelAvailability.INVALID

    return ModelInventoryEntry(
        local_name=local_name,
        model_id=model_id,
        revision=None,
        size_bytes=_directory_size(path),
        status=status,
    )


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _looks_like_host_path(candidate: str) -> bool:
    """POSIX-absolute check plus ``Path.is_absolute`` for Windows forms."""
    return candidate.startswith(("/", "\\")) or Path(candidate).is_absolute()


def _weight_status(path: Path) -> ModelAvailability:
    """READY/INCOMPLETE for a snapshot directory whose config parsed."""
    index_path = path / _INDEX_FILENAME
    if index_path.is_file():
        shards = _index_shards(index_path)
        if shards is None or not shards:
            # Unparseable or empty index: completeness cannot be proven, so
            # never claim READY (spec §20: observational, conservative).
            logger.warning(
                "model %s: safetensors index unreadable or empty; incomplete",
                path.name,
            )
            return ModelAvailability.INCOMPLETE
        missing = sorted(shard for shard in shards if not (path / shard).is_file())
        if missing:
            logger.warning(
                "model %s: %d referenced weight shards missing (e.g. %s); incomplete",
                path.name,
                len(missing),
                missing[0],
            )
            return ModelAvailability.INCOMPLETE
        return ModelAvailability.READY
    if _has_weight_files(path):
        return ModelAvailability.READY
    return ModelAvailability.INCOMPLETE


def _index_shards(index_path: Path) -> frozenset[str] | None:
    """Distinct shard filenames referenced by the index's ``weight_map``.

    ``None`` when the index cannot be parsed or references anything but
    plain relative filenames: an index is untrusted input, so a value with
    a path separator or a leading dot is never joined onto the store root.
    """
    payload = _read_json(index_path)
    if not isinstance(payload, dict):
        return None
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        return None
    shards: set[str] = set()
    for shard in weight_map.values():
        if not isinstance(shard, str) or not shard:
            return None
        if "/" in shard or "\\" in shard or shard.startswith("."):
            return None
        shards.add(shard)
    return frozenset(shards)


def _has_weight_files(path: Path) -> bool:
    try:
        return any(
            item.is_file() and item.suffix in _WEIGHT_SUFFIXES for item in path.rglob("*")
        )
    except OSError:
        return False


def _directory_size(path: Path) -> int | None:
    total = 0
    try:
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
    except OSError:
        return None
    return total
