"""Weight loading tests: shard materialization without the full model (spec 11).

Reference comparisons use ``from_pretrained()`` loading, which is allowed in
tests only (spec 11.1); the loader under test never loads unrelated tensors.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import LlamaForCausalLM

from edgeshard.model.adapters.llama import LlamaAdapter
from edgeshard.model.errors import MissingWeightsError
from edgeshard.model.source import ModelSource
from edgeshard.model.spec import BlockRange, ShardSpec
from edgeshard.model.weights.safetensors import (
    SafetensorsIndex,
    SafetensorsWeightLoader,
    select_shard_tensors,
)


@pytest.fixture
def loader() -> SafetensorsWeightLoader:
    return SafetensorsWeightLoader()


@pytest.fixture
def adapter() -> LlamaAdapter:
    return LlamaAdapter()


def _load_and_check(
    loader: SafetensorsWeightLoader,
    adapter: LlamaAdapter,
    source: ModelSource,
    shard: ShardSpec,
) -> torch.nn.Module:
    layout = adapter.inspect(source)
    skeleton = adapter.build_skeleton(source, shard)
    loader.load_shard(skeleton, source, layout, shard)
    assert not any(param.is_meta for param in skeleton.parameters())
    return skeleton


def test_full_shard_matches_reference(
    loader: SafetensorsWeightLoader, adapter: LlamaAdapter, tiny_llama_dir: Path
) -> None:
    source = ModelSource(path=tiny_llama_dir)
    shard = ShardSpec(
        model_id="tiny/llama",
        blocks=BlockRange(0, 4),
        include_input_stage=True,
        include_output_stage=True,
    )
    skeleton = _load_and_check(loader, adapter, source, shard)
    reference = LlamaForCausalLM.from_pretrained(tiny_llama_dir)
    loaded = dict(skeleton.named_parameters())
    for name, param in reference.named_parameters():
        assert name in loaded, f"missing {name}"
        assert torch.equal(loaded[name], param), name


def test_middle_shard_matches_reference_slice(
    loader: SafetensorsWeightLoader, adapter: LlamaAdapter, tiny_llama_dir: Path
) -> None:
    source = ModelSource(path=tiny_llama_dir)
    shard = ShardSpec(model_id="tiny/llama", blocks=BlockRange(1, 3))
    skeleton = _load_and_check(loader, adapter, source, shard)

    reference = LlamaForCausalLM.from_pretrained(tiny_llama_dir)
    skeleton_params = dict(skeleton.named_parameters())
    assert skeleton_params
    for local_block, global_block in enumerate((1, 2)):
        for name, param in reference.model.layers[global_block].named_parameters():
            skeleton_name = f"model.layers.{local_block}.{name}"
            assert torch.equal(skeleton_params[skeleton_name], param), skeleton_name
    # Nothing outside the shard was materialized.
    assert all(name.startswith("model.layers.") for name in skeleton_params)


def test_shard_loads_without_unrelated_tensors(
    loader: SafetensorsWeightLoader,
    adapter: LlamaAdapter,
    tiny_llama_dir: Path,
    tmp_path: Path,
) -> None:
    """0B gate: a shard materializes when unrelated layer tensors are absent."""
    source = ModelSource(path=tiny_llama_dir)
    shard = ShardSpec(model_id="tiny/llama", blocks=BlockRange(1, 3))
    layout = adapter.inspect(source)
    index = SafetensorsIndex.from_source(source)
    selected = select_shard_tensors(layout, shard, index)

    pruned = tmp_path / "pruned"
    pruned.mkdir()
    shutil.copy(tiny_llama_dir / "config.json", pruned / "config.json")
    with safe_open(tiny_llama_dir / "model.safetensors", framework="pt") as checkpoint:
        tensors = {name: checkpoint.get_tensor(name) for name in selected}
    assert len(tensors) < len(index)  # genuinely pruned
    save_file(tensors, str(pruned / "model.safetensors"))

    skeleton = _load_and_check(loader, adapter, ModelSource(path=pruned), shard)
    reference = LlamaForCausalLM.from_pretrained(tiny_llama_dir)
    skeleton_params = dict(skeleton.named_parameters())
    for local_block, global_block in enumerate((1, 2)):
        for name, param in reference.model.layers[global_block].named_parameters():
            assert torch.equal(
                skeleton_params[f"model.layers.{local_block}.{name}"], param
            )


def test_loading_from_sharded_checkpoint(
    loader: SafetensorsWeightLoader,
    adapter: LlamaAdapter,
    sharded_llama_dir: Path,
    tiny_llama_dir: Path,
) -> None:
    source = ModelSource(path=sharded_llama_dir)
    shard = ShardSpec(model_id="tiny/llama", blocks=BlockRange(1, 3))
    skeleton = _load_and_check(loader, adapter, source, shard)
    reference = LlamaForCausalLM.from_pretrained(tiny_llama_dir)
    skeleton_params = dict(skeleton.named_parameters())
    for local_block, global_block in enumerate((1, 2)):
        for name, param in reference.model.layers[global_block].named_parameters():
            assert torch.equal(
                skeleton_params[f"model.layers.{local_block}.{name}"], param
            )


def test_tied_final_shard_materializes_lm_head(
    loader: SafetensorsWeightLoader, adapter: LlamaAdapter, tiny_llama_tied_dir: Path
) -> None:
    """Tied embeddings: the final shard stays independently loadable (spec 11.3)."""
    source = ModelSource(path=tiny_llama_tied_dir)
    shard = ShardSpec(
        model_id="tiny/llama-tied", blocks=BlockRange(1, 2), include_output_stage=True
    )
    skeleton = _load_and_check(loader, adapter, source, shard)

    reference = LlamaForCausalLM.from_pretrained(tiny_llama_tied_dir)
    skeleton_params = dict(skeleton.named_parameters())
    assert torch.equal(skeleton_params["lm_head.weight"], reference.model.embed_tokens.weight)
    assert torch.equal(skeleton_params["model.norm.weight"], reference.model.norm.weight)


def test_full_tied_shard_matches_reference(
    loader: SafetensorsWeightLoader, adapter: LlamaAdapter, tiny_llama_tied_dir: Path
) -> None:
    source = ModelSource(path=tiny_llama_tied_dir)
    shard = ShardSpec(
        model_id="tiny/llama-tied",
        blocks=BlockRange(0, 2),
        include_input_stage=True,
        include_output_stage=True,
    )
    skeleton = _load_and_check(loader, adapter, source, shard)
    reference = LlamaForCausalLM.from_pretrained(tiny_llama_tied_dir)
    loaded = dict(skeleton.named_parameters())
    for name, param in reference.named_parameters():
        assert name in loaded, f"missing {name}"
        assert torch.equal(loaded[name], param), name


def test_missing_block_weights_raise(
    loader: SafetensorsWeightLoader,
    adapter: LlamaAdapter,
    tiny_llama_dir: Path,
    tmp_path: Path,
) -> None:
    pruned = tmp_path / "pruned"
    pruned.mkdir()
    shutil.copy(tiny_llama_dir / "config.json", pruned / "config.json")
    with safe_open(tiny_llama_dir / "model.safetensors", framework="pt") as checkpoint:
        names = list(checkpoint.keys())
        tensors = {
            name: checkpoint.get_tensor(name)
            for name in names
            if name.startswith("model.layers.0.")
        }
    save_file(tensors, str(pruned / "model.safetensors"))

    source = ModelSource(path=pruned)
    shard = ShardSpec(model_id="tiny/llama", blocks=BlockRange(0, 2))
    layout = adapter.inspect(source)
    skeleton = adapter.build_skeleton(source, shard)
    with pytest.raises(MissingWeightsError, match="block 1"):
        loader.load_shard(skeleton, source, layout, shard)
