"""Qwen2 adapter tests against a generated tiny Qwen2 model (spec 25.1)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch.nn as nn

from edgeshard.model.adapters.qwen2 import Qwen2Adapter
from edgeshard.model.layout import ModelLayout
from edgeshard.model.source import ModelSource
from edgeshard.model.spec import BlockRange, ShardSpec


@pytest.fixture
def adapter() -> Qwen2Adapter:
    return Qwen2Adapter()


def test_inspect_layout(adapter: Qwen2Adapter, tiny_qwen2_source: ModelSource) -> None:
    layout = adapter.inspect(tiny_qwen2_source)
    assert layout == ModelLayout(
        model_type="qwen2",
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
    adapter: Qwen2Adapter, tiny_qwen2_dir: Path, tmp_path: Path
) -> None:
    """Inspection must not require weights (spec 9.1, 0A gate)."""
    stripped = tmp_path / "stripped"
    shutil.copytree(tiny_qwen2_dir, stripped)
    for weights in stripped.glob("*.safetensors"):
        weights.unlink()
    layout = adapter.inspect(ModelSource(path=stripped))
    assert layout.num_blocks == 4


def test_full_shard_skeleton(adapter: Qwen2Adapter, tiny_qwen2_source: ModelSource) -> None:
    shard = ShardSpec(
        model_id="tiny/qwen2",
        blocks=BlockRange(0, 4),
        include_input_stage=True,
        include_output_stage=True,
    )
    skeleton = adapter.build_skeleton(tiny_qwen2_source, shard)
    assert len(skeleton.model.layers) == 4
    assert not isinstance(skeleton.model.embed_tokens, nn.Identity)
    assert not isinstance(skeleton.model.norm, nn.Identity)
    assert not isinstance(skeleton.lm_head, nn.Identity)
    assert all(param.is_meta for param in skeleton.parameters())


def test_middle_shard_skeleton(adapter: Qwen2Adapter, tiny_qwen2_source: ModelSource) -> None:
    shard = ShardSpec(model_id="tiny/qwen2", blocks=BlockRange(1, 3))
    skeleton = adapter.build_skeleton(tiny_qwen2_source, shard)
    assert len(skeleton.model.layers) == 2
    assert isinstance(skeleton.model.embed_tokens, nn.Identity)
    assert isinstance(skeleton.model.norm, nn.Identity)
    assert isinstance(skeleton.lm_head, nn.Identity)
    names = {name for name, _ in skeleton.named_modules()}
    assert "model.layers.0" in names
    assert "model.layers.2" not in names


def test_skeleton_rejects_out_of_bounds(
    adapter: Qwen2Adapter, tiny_qwen2_source: ModelSource
) -> None:
    shard = ShardSpec(model_id="tiny/qwen2", blocks=BlockRange(2, 8))
    with pytest.raises(ValueError, match="exceeds"):
        adapter.build_skeleton(tiny_qwen2_source, shard)
