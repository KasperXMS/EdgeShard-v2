"""Safetensors checkpoint index and shard tensor selection (spec 11).

safetensors is the formal runtime checkpoint format (spec 11.1). Both
checkpoint shapes are supported:

- sharded checkpoints with ``model.safetensors.index.json``;
- single-file checkpoints (``model.safetensors`` only), whose tensor names are
  read from the file header.

Selection computes the *exact* tensor set a shard needs so that building a
70B shard never requires materializing the full 70B model (spec 4.3, 11.2).
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import torch
from safetensors import safe_open

from edgeshard.model.errors import MissingWeightsError, WeightError
from edgeshard.model.layout import ModelLayout
from edgeshard.model.source import ModelSource
from edgeshard.model.spec import ShardSpec

_INDEX_FILENAME = "model.safetensors.index.json"
_WEIGHT_MAP_KEY = "weight_map"


class SafetensorsIndex:
    """Maps checkpoint tensor names to the safetensors file that stores them."""

    def __init__(self, tensor_files: Mapping[str, str]) -> None:
        if not tensor_files:
            raise WeightError("checkpoint index is empty")
        self._tensor_files: dict[str, str] = dict(tensor_files)

    @classmethod
    def from_source(cls, source: ModelSource) -> SafetensorsIndex:
        """Build the index for a local snapshot without loading tensor data."""
        directory = source.ensure_local()
        index_path = directory / _INDEX_FILENAME
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = payload.get(_WEIGHT_MAP_KEY)
            if not isinstance(weight_map, dict) or not weight_map:
                raise WeightError(f"malformed weight map in {index_path}")
            return cls(weight_map)

        shards = sorted(directory.glob("*.safetensors"))
        if not shards:
            raise WeightError(f"no safetensors checkpoint found in {directory}")
        if len(shards) > 1:
            raise WeightError(
                f"multiple safetensors files without {_INDEX_FILENAME} in {directory}"
            )
        with safe_open(shards[0], framework="pt") as checkpoint:
            names = list(checkpoint.keys())
        return cls({str(name): shards[0].name for name in names})

    @property
    def tensor_names(self) -> frozenset[str]:
        return frozenset(self._tensor_files)

    def file_for(self, tensor_name: str) -> str:
        try:
            return self._tensor_files[tensor_name]
        except KeyError:
            raise WeightError(f"tensor {tensor_name!r} not found in checkpoint") from None

    def __contains__(self, tensor_name: object) -> bool:
        return tensor_name in self._tensor_files

    def __len__(self) -> int:
        return len(self._tensor_files)


def select_shard_tensors(
    layout: ModelLayout,
    shard: ShardSpec,
    index: SafetensorsIndex,
) -> dict[str, str]:
    """Exact checkpoint tensors required for ``shard``, mapped to their files.

    Selection rules (spec 11.2, 11.3):

    - blocks in ``shard.blocks``: all tensors under each block prefix;
    - ``include_input_stage``: embedding tensors;
    - ``include_output_stage``: final norm tensors, plus lm head tensors —
      for tied embeddings the embedding tensors are selected instead so the
      final shard stays independently loadable.
    """
    shard.validate_bounds(layout.num_blocks)
    selected: dict[str, str] = {}

    def take_prefix(prefix: str, *, label: str) -> None:
        matches = {name for name in index.tensor_names if name.startswith(prefix + ".")}
        if not matches:
            raise MissingWeightsError(f"no {label} tensors under prefix {prefix!r}")
        for name in matches:
            selected[name] = index.file_for(name)

    for block in range(shard.blocks.start, shard.blocks.end):
        take_prefix(layout.block_prefix(block), label=f"block {block}")
    if shard.include_input_stage:
        take_prefix(layout.embedding_prefix, label="embedding")
    if shard.include_output_stage:
        take_prefix(layout.final_norm_prefix, label="final norm")
        if layout.tied_word_embeddings:
            take_prefix(layout.embedding_prefix, label="tied lm head (embedding)")
        else:
            take_prefix(layout.lm_head_prefix, label="lm head")
    return selected


def _skeleton_keys(layout: ModelLayout, shard: ShardSpec, tensor_name: str) -> list[str]:
    """Map a selected checkpoint tensor to its skeleton state-dict keys.

    Skeleton layers are re-indexed from 0, so global block ``g`` maps to
    skeleton index ``g - shard.blocks.start``. Tied embeddings materialize the
    lm head from the embedding tensors (spec 11.3).
    """
    for global_block in range(shard.blocks.start, shard.blocks.end):
        prefix = layout.block_prefix(global_block) + "."
        if tensor_name.startswith(prefix):
            suffix = tensor_name[len(prefix) :]
            local_block = global_block - shard.blocks.start
            return [layout.block_prefix(local_block) + "." + suffix]

    if tensor_name.startswith(layout.embedding_prefix + "."):
        suffix = tensor_name[len(layout.embedding_prefix) :]
        keys: list[str] = []
        if shard.include_input_stage:
            keys.append(layout.embedding_prefix + suffix)
        if shard.include_output_stage and layout.tied_word_embeddings:
            keys.append(layout.lm_head_prefix + suffix)
        return keys

    if shard.include_output_stage:
        if tensor_name.startswith(layout.final_norm_prefix + "."):
            return [tensor_name]
        if tensor_name.startswith(layout.lm_head_prefix + "."):
            return [tensor_name]

    raise WeightError(f"tensor {tensor_name!r} is not part of shard {shard!r}")


class SafetensorsWeightLoader:
    """Materializes one shard's weights into a meta-device skeleton (spec 11).

    Only files containing selected tensors are opened and only selected
    tensors are read, so building a shard never touches unrelated weights
    (spec 4.3). ``load_state_dict(strict=True)`` then proves the selection
    exactly covers the skeleton — nothing missing, nothing extra.
    """

    def load_shard(
        self,
        module: torch.nn.Module,
        source: ModelSource,
        layout: ModelLayout,
        shard: ShardSpec,
    ) -> None:
        shard.validate_bounds(layout.num_blocks)
        index = SafetensorsIndex.from_source(source)
        selected = select_shard_tensors(layout, shard, index)

        by_file: dict[str, list[str]] = {}
        for name, filename in selected.items():
            by_file.setdefault(filename, []).append(name)

        directory = source.ensure_local()
        state: dict[str, torch.Tensor] = {}
        for filename in sorted(by_file):
            with safe_open(directory / filename, framework="pt") as checkpoint:
                for name in by_file[filename]:
                    tensor: torch.Tensor = checkpoint.get_tensor(name)
                    for key in _skeleton_keys(layout, shard, name):
                        state[key] = tensor

        module.load_state_dict(state, strict=True, assign=True)
