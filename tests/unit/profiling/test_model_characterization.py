"""P2C characterization-facade tests (spec §17, §7).

The facade chains Phase 0 layout inspection into profiling adapters;
these tests pin the provenance fallbacks, determinism of
``model_signature_id``, provenance exclusion from structural identity,
and typed failure for unknown models.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from edgeshard.model.adapters.base import load_model_config
from edgeshard.model.source import ModelSource
from edgeshard.profiling.domain.experiment import ProfilingErrorCategory
from edgeshard.profiling.domain.model import StageKind, model_signature_id
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.model.adapters.base import ProfilingAdapterRegistry
from edgeshard.profiling.model.characterization import (
    UNRESOLVED_REVISION,
    characterize_model_source,
    model_reference_for,
)


class TestCharacterizeModelSource:
    def test_tiny_qwen2(self, tiny_qwen2_source: ModelSource) -> None:
        characterization = characterize_model_source(tiny_qwen2_source, dtype="fp32")
        assert characterization.model.model_id == "tiny/qwen2"
        assert characterization.model.revision == UNRESOLVED_REVISION
        assert characterization.architecture_family == "qwen2"
        assert characterization.num_layers == 4
        assert characterization.hidden_size == 64
        assert characterization.vocab_size == 128
        assert characterization.head_dim == 16
        assert characterization.dtype == "fp32"
        assert characterization.tied_word_embeddings is False
        assert [stage.kind for stage in characterization.stages] == [
            StageKind.EMBEDDING,
            StageKind.TRANSFORMER_LAYER_GROUP,
            StageKind.FINAL_NORM,
            StageKind.LM_HEAD,
        ]

    def test_tiny_llama_declared_dtype(self, tiny_llama_source: ModelSource) -> None:
        characterization = characterize_model_source(tiny_llama_source, dtype="bf16")
        assert characterization.architecture_family == "llama"
        assert characterization.dtype == "bf16"
        assert characterization.num_layers == 4

    def test_tied_embeddings_observed(self, tiny_llama_tied_dir: Path) -> None:
        source = ModelSource(path=tiny_llama_tied_dir)
        characterization = characterize_model_source(source, dtype="fp32")
        assert characterization.tied_word_embeddings is True
        assert characterization.num_layers == 2
        assert characterization.hidden_size == 32

    def test_deterministic(self, tiny_qwen2_source: ModelSource) -> None:
        first = characterize_model_source(tiny_qwen2_source, dtype="fp32")
        second = characterize_model_source(tiny_qwen2_source, dtype="fp32")
        assert first == second
        assert model_signature_id(first) == model_signature_id(second)

    def test_signature_excludes_provenance(self, tiny_llama_dir: Path) -> None:
        named = characterize_model_source(
            ModelSource(path=tiny_llama_dir, model_id="tiny/llama", revision="aaa"),
            dtype="fp32",
        )
        anonymous = characterize_model_source(
            ModelSource(path=tiny_llama_dir), dtype="fp32"
        )
        assert named.model != anonymous.model
        assert model_signature_id(named) == model_signature_id(anonymous)

    def test_unknown_model_fails_typed(self, tiny_qwen2_source: ModelSource) -> None:
        with pytest.raises(ProfilingError) as excinfo:
            characterize_model_source(
                tiny_qwen2_source,
                dtype="fp32",
                profiling_registry=ProfilingAdapterRegistry(),
            )
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_MODEL


class TestModelReferenceFor:
    def test_source_labels_win(self, tiny_qwen2_source: ModelSource) -> None:
        config = load_model_config(tiny_qwen2_source)
        source = ModelSource(path=tiny_qwen2_source.path, model_id="x/y", revision="deadbeef")
        reference = model_reference_for(source, config)
        assert reference.model_id == "x/y"
        assert reference.revision == "deadbeef"

    def test_fallbacks(self, tiny_qwen2_dir: Path) -> None:
        config = load_model_config(ModelSource(path=tiny_qwen2_dir))
        reference = model_reference_for(ModelSource(path=tiny_qwen2_dir), config)
        assert reference.model_id  # config name_or_path or directory name
        assert reference.revision == UNRESOLVED_REVISION
