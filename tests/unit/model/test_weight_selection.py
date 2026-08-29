"""Shard tensor selection tests: exact tensor sets per ShardSpec (spec 11.2, 11.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from edgeshard.model.adapters.llama import LlamaAdapter
from edgeshard.model.errors import MissingWeightsError
from edgeshard.model.layout import ModelLayout
from edgeshard.model.source import ModelSource
from edgeshard.model.spec import BlockRange, ShardSpec
from edgeshard.model.weights.safetensors import SafetensorsIndex, select_shard_tensors


@pytest.fixture
def llama_layout(tiny_llama_dir: Path) -> ModelLayout:
    return LlamaAdapter().inspect(ModelSource(path=tiny_llama_dir))


@pytest.fixture
def llama_index(tiny_llama_dir: Path) -> SafetensorsIndex:
    return SafetensorsIndex.from_source(ModelSource(path=tiny_llama_dir))


def _spec(
    blocks: BlockRange, *, input_stage: bool = False, output_stage: bool = False
) -> ShardSpec:
    return ShardSpec(
        model_id="tiny/llama",
        blocks=blocks,
        include_input_stage=input_stage,
        include_output_stage=output_stage,
    )


def test_full_shard_selects_entire_checkpoint(
    llama_layout: ModelLayout, llama_index: SafetensorsIndex
) -> None:
    shard = _spec(BlockRange(0, 4), input_stage=True, output_stage=True)
    selected = select_shard_tensors(llama_layout, shard, llama_index)
    assert set(selected) == llama_index.tensor_names


def test_middle_shard_selects_only_its_blocks(
    llama_layout: ModelLayout, llama_index: SafetensorsIndex
) -> None:
    shard = _spec(BlockRange(1, 3))
    selected = select_shard_tensors(llama_layout, shard, llama_index)
    assert selected
    assert all(
        name.startswith(("model.layers.1.", "model.layers.2.")) for name in selected
    )
    for block in (1, 2):
        assert any(name.startswith(f"model.layers.{block}.") for name in selected)
    for other in ("model.layers.0.", "model.layers.3.", "model.embed_tokens.", "lm_head."):
        assert not any(name.startswith(other) for name in selected)


def test_prefix_boundary_is_exact(
    llama_layout: ModelLayout, llama_index: SafetensorsIndex
) -> None:
    """Block 1 selection must not leak into a hypothetical block 10+ (spec 8.3)."""
    shard = _spec(BlockRange(1, 2))
    selected = select_shard_tensors(llama_layout, shard, llama_index)
    assert all(name.startswith("model.layers.1.") for name in selected)


def test_first_shard_includes_embedding(
    llama_layout: ModelLayout, llama_index: SafetensorsIndex
) -> None:
    shard = _spec(BlockRange(0, 2), input_stage=True)
    selected = select_shard_tensors(llama_layout, shard, llama_index)
    assert "model.embed_tokens.weight" in selected
    assert not any(name.startswith("model.norm.") for name in selected)


def test_last_shard_includes_norm_and_lm_head(
    llama_layout: ModelLayout, llama_index: SafetensorsIndex
) -> None:
    shard = _spec(BlockRange(2, 4), output_stage=True)
    selected = select_shard_tensors(llama_layout, shard, llama_index)
    assert "model.norm.weight" in selected
    assert "lm_head.weight" in selected
    assert "model.embed_tokens.weight" not in selected


def test_tied_final_shard_selects_embedding_for_lm_head(tiny_llama_tied_dir: Path) -> None:
    """With tied embeddings the final shard stays independently loadable (spec 11.3)."""
    source = ModelSource(path=tiny_llama_tied_dir)
    layout = LlamaAdapter().inspect(source)
    index = SafetensorsIndex.from_source(source)
    assert layout.tied_word_embeddings is True
    assert "lm_head.weight" not in index

    shard = _spec(BlockRange(1, 2), output_stage=True)
    selected = select_shard_tensors(layout, shard, index)
    assert "model.embed_tokens.weight" in selected


def test_selection_rejects_out_of_bounds(
    llama_layout: ModelLayout, llama_index: SafetensorsIndex
) -> None:
    with pytest.raises(ValueError, match="exceeds"):
        select_shard_tensors(llama_layout, _spec(BlockRange(2, 8)), llama_index)


def test_missing_block_tensors_raise(
    llama_layout: ModelLayout, llama_index: SafetensorsIndex
) -> None:
    partial = SafetensorsIndex(
        {
            name: llama_index.file_for(name)
            for name in llama_index.tensor_names
            if name.startswith("model.layers.0.")
        }
    )
    with pytest.raises(MissingWeightsError, match="block 1"):
        select_shard_tensors(llama_layout, _spec(BlockRange(0, 2)), partial)


def test_selection_on_sharded_checkpoint_maps_files(sharded_llama_dir: Path) -> None:
    source = ModelSource(path=sharded_llama_dir)
    layout = LlamaAdapter().inspect(source)
    index = SafetensorsIndex.from_source(source)
    shard = _spec(BlockRange(1, 3))
    selected = select_shard_tensors(layout, shard, index)
    for name, filename in selected.items():
        assert index.file_for(name) == filename
