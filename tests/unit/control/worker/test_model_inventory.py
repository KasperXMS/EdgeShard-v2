"""ModelStore scanning tests (Phase 1 spec §20)."""

from __future__ import annotations

import json
from pathlib import Path

from edgeshard.cluster.inventory import ModelAvailability
from edgeshard.control.worker.model_inventory import scan_model_inventory
from edgeshard.runtime.model_store import ModelStore


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
    assert entry.revision is None  # snapshot metadata parsing is later-phase
    assert entry.status is ModelAvailability.READY
    expected_size = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    assert entry.size_bytes == expected_size


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
