"""Llama adapter tests against a generated tiny Llama model (spec 25.1)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch.nn as nn
from transformers import LlamaConfig

from edgeshard.model.adapters.llama import LlamaAdapter
from edgeshard.model.adapters.registry import AdapterRegistry, resolve_adapter_for_source
from edgeshard.model.layout import ModelLayout
from edgeshard.model.source import ModelSource
from edgeshard.model.spec import BlockRange, ShardSpec


@pytest.fixture
def adapter() -> LlamaAdapter:
    return LlamaAdapter()


def test_inspect_layout(adapter: LlamaAdapter, tiny_llama_source: ModelSource) -> None:
    layout = adapter.inspect(tiny_llama_source)
    assert layout == ModelLayout(
        model_type="llama",
        num_blocks=4,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_kv_heads=2,
        embedding_prefix="model.embed_tokens",
        block_prefix_template="model.layers.{index}",
        final_norm_prefix="model.norm",
        lm_head_prefix="lm_head",
        tied_word_embeddings=False,
    )


def test_inspect_without_weight_files(
    adapter: LlamaAdapter, tiny_llama_dir: Path, tmp_path: Path
) -> None:
    """Inspection must not require weights (spec 9.1, 0A gate)."""
    stripped = tmp_path / "stripped"
    shutil.copytree(tiny_llama_dir, stripped)
    for weights in stripped.glob("*.safetensors"):
        weights.unlink()
    layout = adapter.inspect(ModelSource(path=stripped))
    assert layout.num_blocks == 4


def test_automatic_resolution(tiny_llama_source: ModelSource) -> None:
    registry = AdapterRegistry()
    registry.register(LlamaAdapter())
    adapter = resolve_adapter_for_source(tiny_llama_source, registry)
    assert isinstance(adapter, LlamaAdapter)


def test_full_shard_skeleton(adapter: LlamaAdapter, tiny_llama_source: ModelSource) -> None:
    shard = ShardSpec(
        model_id="tiny/llama",
        blocks=BlockRange(0, 4),
        include_input_stage=True,
        include_output_stage=True,
    )
    skeleton = adapter.build_skeleton(tiny_llama_source, shard)
    assert len(skeleton.model.layers) == 4
    assert not isinstance(skeleton.model.embed_tokens, nn.Identity)
    assert not isinstance(skeleton.model.norm, nn.Identity)
    assert not isinstance(skeleton.lm_head, nn.Identity)
    assert all(param.is_meta for param in skeleton.parameters())


def test_middle_shard_skeleton(adapter: LlamaAdapter, tiny_llama_source: ModelSource) -> None:
    shard = ShardSpec(model_id="tiny/llama", blocks=BlockRange(1, 3))
    skeleton = adapter.build_skeleton(tiny_llama_source, shard)
    assert len(skeleton.model.layers) == 2
    assert isinstance(skeleton.model.embed_tokens, nn.Identity)
    assert isinstance(skeleton.model.norm, nn.Identity)
    assert isinstance(skeleton.lm_head, nn.Identity)
    # Skeleton layers are re-indexed from 0: global block 1 -> skeleton index 0.
    names = {name for name, _ in skeleton.named_modules()}
    assert "model.layers.0" in names
    assert "model.layers.1" in names
    assert "model.layers.2" not in names


def test_first_shard_keeps_embedding_only(
    adapter: LlamaAdapter, tiny_llama_source: ModelSource
) -> None:
    shard = ShardSpec(model_id="tiny/llama", blocks=BlockRange(0, 2), include_input_stage=True)
    skeleton = adapter.build_skeleton(tiny_llama_source, shard)
    assert not isinstance(skeleton.model.embed_tokens, nn.Identity)
    assert isinstance(skeleton.model.norm, nn.Identity)
    assert isinstance(skeleton.lm_head, nn.Identity)


def test_skeleton_rejects_out_of_bounds(
    adapter: LlamaAdapter, tiny_llama_source: ModelSource
) -> None:
    shard = ShardSpec(model_id="tiny/llama", blocks=BlockRange(2, 8))
    with pytest.raises(ValueError, match="exceeds"):
        adapter.build_skeleton(tiny_llama_source, shard)


def test_tied_embeddings_reported(adapter: LlamaAdapter, tmp_path: Path) -> None:
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        tie_word_embeddings=True,
    )
    config.save_pretrained(tmp_path)
    layout = adapter.inspect(ModelSource(path=tmp_path))
    assert layout.tied_word_embeddings is True
