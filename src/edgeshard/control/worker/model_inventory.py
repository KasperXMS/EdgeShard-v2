"""Worker-side ModelStore scanning (Phase 1 spec §20).

Reuses :class:`edgeshard.runtime.model_store.ModelStore` (spec §3.5 keeps
the store worker-local) and classifies each snapshot directory. Only
logical model identity and status are produced - host paths never leave the
Worker (spec §20). Phase 1 inventory is observational: nothing is
downloaded, repaired, or evicted.

Classification is deliberately conservative:

* ``ready`` - parseable ``config.json`` and at least one weight file;
* ``incomplete`` - ``config.json`` present but no weight files (e.g. an
  interrupted snapshot download);
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

    if has_config and _has_weight_files(path):
        status = ModelAvailability.READY
    elif has_config:
        status = ModelAvailability.INCOMPLETE
    else:
        status = ModelAvailability.INVALID

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
