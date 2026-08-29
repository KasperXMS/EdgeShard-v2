"""Safetensors checkpoint index tests (spec 11.2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edgeshard.model.errors import WeightError
from edgeshard.model.source import ModelSource
from edgeshard.model.weights.safetensors import SafetensorsIndex


def test_single_file_checkpoint(tiny_llama_dir: Path) -> None:
    index = SafetensorsIndex.from_source(ModelSource(path=tiny_llama_dir))
    assert len(index) > 0
    assert "model.embed_tokens.weight" in index
    assert "model.layers.0.self_attn.q_proj.weight" in index
    assert index.file_for("model.embed_tokens.weight") == "model.safetensors"


def test_sharded_checkpoint_via_index_json(sharded_llama_dir: Path) -> None:
    index = SafetensorsIndex.from_source(ModelSource(path=sharded_llama_dir))
    files = {index.file_for(name) for name in index.tensor_names}
    assert files == {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }


def test_sharded_index_matches_single_file_names(
    tiny_llama_dir: Path, sharded_llama_dir: Path
) -> None:
    single = SafetensorsIndex.from_source(ModelSource(path=tiny_llama_dir))
    sharded = SafetensorsIndex.from_source(ModelSource(path=sharded_llama_dir))
    assert single.tensor_names == sharded.tensor_names


def test_no_checkpoint_raises(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(WeightError, match="no safetensors checkpoint"):
        SafetensorsIndex.from_source(ModelSource(path=tmp_path))


def test_multiple_files_without_index_raises(tmp_path: Path) -> None:
    (tmp_path / "a.safetensors").write_bytes(b"")
    (tmp_path / "b.safetensors").write_bytes(b"")
    with pytest.raises(WeightError, match=r"without model\.safetensors\.index\.json"):
        SafetensorsIndex.from_source(ModelSource(path=tmp_path))


def test_malformed_index_raises(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}}))
    with pytest.raises(WeightError, match="weight map"):
        SafetensorsIndex.from_source(ModelSource(path=tmp_path))


def test_empty_index_rejected() -> None:
    with pytest.raises(WeightError, match="empty"):
        SafetensorsIndex({})


def test_file_for_unknown_tensor_raises(tiny_llama_dir: Path) -> None:
    index = SafetensorsIndex.from_source(ModelSource(path=tiny_llama_dir))
    with pytest.raises(WeightError, match="not found"):
        index.file_for("does.not.exist")
