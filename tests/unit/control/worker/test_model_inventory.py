"""ModelStore scanning tests (Phase 1 spec §20)."""

from __future__ import annotations

import json
from pathlib import Path

from edgeshard.cluster.inventory import ModelAvailability
from edgeshard.control.worker.model_inventory import scan_model_inventory
from edgeshard.runtime.model_store import MODEL_METADATA_FILENAME, ModelStore


def make_snapshot(
    root: Path,
    name: str,
    *,
    config_payload: dict[str, object] | None = None,
    weight_files: tuple[str, ...] = (),
    raw_config: str | None = None,
) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    if config_payload is not None:
        (directory / "config.json").write_text(json.dumps(config_payload), encoding="utf-8")
    if raw_config is not None:
        (directory / "config.json").write_text(raw_config, encoding="utf-8")
    for weight in weight_files:
        (directory / weight).write_bytes(b"\x00" * 16)
    return directory


def write_index(directory: Path, payload: object, *, raw: str | None = None) -> None:
    text = raw if raw is not None else json.dumps(payload)
    (directory / "model.safetensors.index.json").write_text(text, encoding="utf-8")


def write_metadata(directory: Path, payload: object, *, raw: str | None = None) -> None:
    text = raw if raw is not None else json.dumps(payload)
    (directory / MODEL_METADATA_FILENAME).write_text(text, encoding="utf-8")


SHARD_1 = "model-00001-of-00002.safetensors"
SHARD_2 = "model-00002-of-00002.safetensors"
TWO_SHARD_INDEX = {
    "metadata": {"total_size": 32},
    "weight_map": {"layer.a": SHARD_1, "layer.b": SHARD_2, "layer.c": SHARD_1},
}


def test_index_with_all_shards_present_is_ready(tmp_path: Path) -> None:
    """§20: a multi-shard snapshot is READY when every referenced shard exists."""
    root = tmp_path / "models"
    directory = make_snapshot(root, "sharded", config_payload={"_name_or_path": "tiny/llama"})
    write_index(directory, TWO_SHARD_INDEX)
    (directory / SHARD_1).write_bytes(b"\x00" * 16)
    (directory / SHARD_2).write_bytes(b"\x00" * 16)
    (entry,) = scan_model_inventory(ModelStore(model_root=root))
    assert entry.status is ModelAvailability.READY


def test_index_with_missing_shard_is_incomplete(tmp_path: Path) -> None:
    """An interrupted multi-shard download must never claim READY."""
    root = tmp_path / "models"
    directory = make_snapshot(root, "sharded", config_payload={})
    write_index(directory, TWO_SHARD_INDEX)
    (directory / SHARD_1).write_bytes(b"\x00" * 16)
    # SHARD_2 missing — even an unrelated stray weight file does not help:
    # the index is authoritative for completeness.
    (directory / "model.safetensors").write_bytes(b"\x00" * 16)
    (entry,) = scan_model_inventory(ModelStore(model_root=root))
    assert entry.status is ModelAvailability.INCOMPLETE


def test_unparseable_index_is_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "models"
    directory = make_snapshot(root, "sharded", config_payload={}, weight_files=(SHARD_1,))
    write_index(directory, None, raw="{ not json")
    (entry,) = scan_model_inventory(ModelStore(model_root=root))
    assert entry.status is ModelAvailability.INCOMPLETE


def test_index_without_weight_map_is_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "models"
    directory = make_snapshot(root, "sharded", config_payload={}, weight_files=(SHARD_1,))
    write_index(directory, {"metadata": {"total_size": 16}})
    (entry,) = scan_model_inventory(ModelStore(model_root=root))
    assert entry.status is ModelAvailability.INCOMPLETE


def test_empty_weight_map_is_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "models"
    directory = make_snapshot(root, "sharded", config_payload={}, weight_files=(SHARD_1,))
    write_index(directory, {"weight_map": {}})
    (entry,) = scan_model_inventory(ModelStore(model_root=root))
    assert entry.status is ModelAvailability.INCOMPLETE


def test_index_with_non_string_shard_is_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "models"
    directory = make_snapshot(root, "sharded", config_payload={})
    write_index(directory, {"weight_map": {"layer.a": 3}})
    (entry,) = scan_model_inventory(ModelStore(model_root=root))
    assert entry.status is ModelAvailability.INCOMPLETE


def test_index_shard_with_path_separator_is_rejected(tmp_path: Path) -> None:
    """An index is untrusted input: shard names never escape the snapshot dir."""
    root = tmp_path / "models"
    directory = make_snapshot(root, "sharded", config_payload={})
    write_index(directory, {"weight_map": {"layer.a": "../../evil.safetensors"}})
    (entry,) = scan_model_inventory(ModelStore(model_root=root))
    assert entry.status is ModelAvailability.INCOMPLETE

    write_index(directory, {"weight_map": {"layer.a": "shards/model.safetensors"}})
    (entry,) = scan_model_inventory(ModelStore(model_root=root))
    assert entry.status is ModelAvailability.INCOMPLETE

    write_index(directory, {"weight_map": {"layer.a": ".hidden.safetensors"}})
    (entry,) = scan_model_inventory(ModelStore(model_root=root))
    assert entry.status is ModelAvailability.INCOMPLETE


def test_missing_store_root_is_empty(tmp_path: Path) -> None:
    assert scan_model_inventory(ModelStore(model_root=tmp_path / "missing")) == ()


def test_empty_store_root_is_empty(tmp_path: Path) -> None:
    root = tmp_path / "models"
    root.mkdir()
    assert scan_model_inventory(ModelStore(model_root=root)) == ()


def test_ready_model_reports_identity_and_size(tmp_path: Path) -> None:
    root = tmp_path / "models"
    directory = make_snapshot(
        root,
        "tiny-llama",
        config_payload={"_name_or_path": "tiny/llama"},
        weight_files=("model.safetensors",),
    )
    (entry,) = scan_model_inventory(ModelStore(model_root=root))
    assert entry.local_name == "tiny-llama"
    assert entry.model_id == "tiny/llama"
    assert entry.revision is None
    assert entry.status is ModelAvailability.READY
    expected_size = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    assert entry.size_bytes == expected_size


def test_sidecar_identity_takes_precedence_over_config_fallback(tmp_path: Path) -> None:
    root = tmp_path / "models"
    directory = make_snapshot(
        root,
        "tiny-llama",
        config_payload={"_name_or_path": "legacy/model"},
        weight_files=("model.safetensors",),
    )
    write_metadata(
        directory,
        {"model_id": "canonical/model", "revision": "revision-1"},
    )

    (entry,) = scan_model_inventory(ModelStore(model_root=root))

    assert entry.model_id == "canonical/model"
    assert entry.revision == "revision-1"
    assert entry.status is ModelAvailability.READY


def test_invalid_sidecar_marks_entry_invalid_without_config_fallback(tmp_path: Path) -> None:
    root = tmp_path / "models"
    directory = make_snapshot(
        root,
        "tiny-llama",
        config_payload={"_name_or_path": "legacy/model"},
        weight_files=("model.safetensors",),
    )
    write_metadata(directory, None, raw='{ "revision": "revision-1" }')

    (entry,) = scan_model_inventory(ModelStore(model_root=root))

    assert entry.model_id is None
    assert entry.revision is None
    assert entry.status is ModelAvailability.INVALID


def test_absent_sidecar_preserves_name_or_path_fallback(tmp_path: Path) -> None:
    root = tmp_path / "models"
    make_snapshot(
        root,
        "tiny-llama",
        config_payload={"_name_or_path": "legacy/model"},
        weight_files=("model.safetensors",),
    )

    (entry,) = scan_model_inventory(ModelStore(model_root=root))

    assert entry.model_id == "legacy/model"
    assert entry.revision is None
    assert entry.status is ModelAvailability.READY


def test_config_without_weights_is_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "models"
    make_snapshot(root, "partial", config_payload={"_name_or_path": "tiny/llama"})
    (entry,) = scan_model_inventory(ModelStore(model_root=root))
    assert entry.status is ModelAvailability.INCOMPLETE
    assert entry.model_id == "tiny/llama"


def test_weights_without_config_are_invalid(tmp_path: Path) -> None:
    root = tmp_path / "models"
    make_snapshot(root, "loose-weights", weight_files=("model.safetensors",))
    (entry,) = scan_model_inventory(ModelStore(model_root=root))
    assert entry.status is ModelAvailability.INVALID
    assert entry.model_id is None


def test_unparseable_config_is_invalid(tmp_path: Path) -> None:
    root = tmp_path / "models"
    make_snapshot(root, "broken", raw_config="{ not json", weight_files=("model.bin",))
    (entry,) = scan_model_inventory(ModelStore(model_root=root))
    assert entry.status is ModelAvailability.INVALID


def test_host_path_name_is_not_leaked_as_model_id(tmp_path: Path) -> None:
    """Master must never depend on host paths (spec §20)."""
    root = tmp_path / "models"
    make_snapshot(
        root,
        "local-copy",
        config_payload={"_name_or_path": "/home/user/models/tiny"},
        weight_files=("model.safetensors",),
    )
    (entry,) = scan_model_inventory(ModelStore(model_root=root))
    assert entry.model_id is None
    assert entry.status is ModelAvailability.READY


def test_unaddressable_directory_reported_invalid(tmp_path: Path) -> None:
    root = tmp_path / "models"
    (root / "bad name").mkdir(parents=True)
    (entry,) = scan_model_inventory(ModelStore(model_root=root))
    assert entry.local_name == "bad name"
    assert entry.status is ModelAvailability.INVALID
    assert entry.size_bytes is None


def test_non_directory_entries_skipped(tmp_path: Path) -> None:
    root = tmp_path / "models"
    root.mkdir()
    (root / "stray.txt").write_text("not a model", encoding="utf-8")
    assert scan_model_inventory(ModelStore(model_root=root)) == ()


def test_entries_sorted_by_local_name(tmp_path: Path) -> None:
    root = tmp_path / "models"
    make_snapshot(root, "b-model", config_payload={})
    make_snapshot(root, "a-model", config_payload={})
    entries = scan_model_inventory(ModelStore(model_root=root))
    assert [entry.local_name for entry in entries] == ["a-model", "b-model"]
