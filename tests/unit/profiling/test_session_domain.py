"""Profiling session domain values (spec §38, §41, §47).

Sessions group compatible cases so expensive state is built once; the
request/facts types are the pure-domain contract between the Master-side
strategy and the Worker-side runner. Pinned here: the per-kind requirement
matrix (model sessions need a checkpoint and declared dtype, operator
sessions are model-free, network sessions are device-free) and the
facts-vs-characterization consistency rule that keeps §47 planning honest.
"""

from __future__ import annotations

import pytest

from edgeshard.profiling.domain.hashing import normalized_items
from edgeshard.profiling.domain.model import (
    ModelCharacterization,
    ModelReference,
    ModelStage,
    StageKind,
)
from edgeshard.profiling.domain.session import (
    LayerEntry,
    ModelSessionFacts,
    ModuleEntry,
    ProfilingSessionKind,
    ProfilingSessionRequest,
    profiling_session_id,
)
from edgeshard.profiling.domain.signature import (
    GemmSignature,
    ModuleKind,
    ModuleSignature,
    OperatorKind,
    OperatorSignature,
    TransformerLayerSignature,
)

LAYER_SIG = TransformerLayerSignature(
    architecture_family="llama",
    layer_type="standard_decoder",
    hidden_size=32,
    intermediate_size=64,
    num_attention_heads=4,
    num_kv_heads=2,
    head_dim=8,
    dtype="fp32",
    quantization=None,
)
MODULE_SIG = ModuleSignature(
    kind=ModuleKind.MLP,
    architecture_family="llama",
    structural_parameters=normalized_items({"hidden_size": 32}, "p"),
    dtype="fp32",
    quantization=None,
)
OP_SIG = OperatorSignature(
    kind=OperatorKind.GEMM,
    parameters=GemmSignature(m=8, n=8, k=8, dtype="fp32"),
    backend_family="torch",
)


def _characterization(num_layers: int = 2) -> ModelCharacterization:
    return ModelCharacterization(
        model=ModelReference(model_id="tiny/llama", revision="local"),
        architecture_family="llama",
        num_layers=num_layers,
        hidden_size=32,
        intermediate_size=64,
        vocab_size=64,
        num_attention_heads=4,
        num_kv_heads=2,
        head_dim=8,
        dtype="fp32",
        quantization=None,
        tied_word_embeddings=False,
        stages=(
            ModelStage(kind=StageKind.EMBEDDING),
            ModelStage(kind=StageKind.TRANSFORMER_LAYER_GROUP, layer_count=num_layers),
            ModelStage(kind=StageKind.LM_HEAD),
        ),
    )


def _facts(num_layers: int = 2, **overrides: object) -> ModelSessionFacts:
    base: dict[str, object] = {
        "characterization": _characterization(num_layers),
        "layer_entries": tuple(
            LayerEntry(index, f"model.layers.{index}", LAYER_SIG)
            for index in range(num_layers)
        ),
        "module_entries": (
            ModuleEntry("mlp", "model.layers.0.mlp", ModuleKind.MLP, 0, MODULE_SIG),
        ),
        "operator_signatures": (OP_SIG,),
    }
    base.update(overrides)
    return ModelSessionFacts(**base)  # type: ignore[arg-type]


class TestProfilingSessionRequest:
    def test_model_session_requires_devices_model_and_dtype(self) -> None:
        request = ProfilingSessionRequest(
            kind=ProfilingSessionKind.MODEL,
            device_ids=("GPU-uuid-1",),
            model=ModelReference(model_id="tiny/llama", revision="local"),
            dtype="fp32",
        )
        assert request.backend == "torch"  # default measurement backend
        with pytest.raises(ValueError, match="device_id"):
            ProfilingSessionRequest(
                kind=ProfilingSessionKind.MODEL,
                model=request.model,
                dtype="fp32",
            )
        with pytest.raises(ValueError, match="model reference"):
            ProfilingSessionRequest(
                kind=ProfilingSessionKind.MODEL, device_ids=("g",), dtype="fp32"
            )
        with pytest.raises(ValueError, match="dtype"):
            ProfilingSessionRequest(
                kind=ProfilingSessionKind.MODEL,
                device_ids=("g",),
                model=request.model,
            )

    def test_operator_session_is_model_free_but_device_bound(self) -> None:
        request = ProfilingSessionRequest(
            kind=ProfilingSessionKind.OPERATOR, device_ids=("GPU-uuid-1",)
        )
        assert request.model is None and request.dtype is None
        with pytest.raises(ValueError, match="device_id"):
            ProfilingSessionRequest(kind=ProfilingSessionKind.OPERATOR)
        with pytest.raises(ValueError, match="must not carry a model"):
            ProfilingSessionRequest(
                kind=ProfilingSessionKind.OPERATOR,
                device_ids=("g",),
                model=ModelReference(model_id="tiny/llama", revision="local"),
            )

    def test_network_session_is_device_free(self) -> None:
        request = ProfilingSessionRequest(kind=ProfilingSessionKind.NETWORK)
        assert request.device_ids == () and request.model is None
        with pytest.raises(ValueError, match="must not carry a model"):
            ProfilingSessionRequest(
                kind=ProfilingSessionKind.NETWORK,
                model=ModelReference(model_id="tiny/llama", revision="local"),
            )

    def test_empty_labels_rejected(self) -> None:
        with pytest.raises(ValueError, match="backend"):
            ProfilingSessionRequest(kind=ProfilingSessionKind.NETWORK, backend="")
        with pytest.raises(ValueError, match="device_ids"):
            ProfilingSessionRequest(kind=ProfilingSessionKind.NETWORK, device_ids=("",))
        with pytest.raises(ValueError, match="dtype"):
            ProfilingSessionRequest(
                kind=ProfilingSessionKind.NETWORK, dtype=""
            )
        with pytest.raises(ValueError, match="quantization"):
            ProfilingSessionRequest(
                kind=ProfilingSessionKind.NETWORK, quantization=""
            )


class TestEntries:
    def test_layer_entry_validation(self) -> None:
        entry = LayerEntry(0, "model.layers.0", LAYER_SIG)
        assert entry.signature is LAYER_SIG
        with pytest.raises(ValueError, match="index"):
            LayerEntry(-1, "model.layers.0", LAYER_SIG)
        with pytest.raises(ValueError, match="module_path"):
            LayerEntry(0, "", LAYER_SIG)

    def test_module_entry_validation(self) -> None:
        entry = ModuleEntry(
            "self_attn", "model.layers.1.self_attn", ModuleKind.ATTENTION, 1, MODULE_SIG
        )
        assert entry.layer_index == 1
        with pytest.raises(ValueError, match="name"):
            ModuleEntry("", "p", ModuleKind.MLP, 0, MODULE_SIG)
        with pytest.raises(ValueError, match="module_path"):
            ModuleEntry("mlp", "", ModuleKind.MLP, 0, MODULE_SIG)
        with pytest.raises(ValueError, match="layer_index"):
            ModuleEntry("mlp", "p", ModuleKind.MLP, -1, MODULE_SIG)


class TestModelSessionFacts:
    def test_valid_facts(self) -> None:
        facts = _facts(2)
        assert len(facts.layer_entries) == 2
        assert facts.operator_signatures == (OP_SIG,)

    def test_layer_enumeration_must_match_characterization(self) -> None:
        """§47: facts and characterization can never drift apart silently."""
        with pytest.raises(ValueError, match="characterization declares 2"):
            _facts(2, layer_entries=(LayerEntry(0, "model.layers.0", LAYER_SIG),))

    def test_layer_indices_must_be_unique(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            _facts(
                2,
                layer_entries=(
                    LayerEntry(0, "model.layers.0", LAYER_SIG),
                    LayerEntry(0, "model.layers.1", LAYER_SIG),
                ),
            )


def test_session_id_is_canonical_and_scope_sensitive() -> None:
    """§7: deterministic, and scoped to experiment, worker, and content."""
    request = ProfilingSessionRequest(
        kind=ProfilingSessionKind.OPERATOR, device_ids=("gpu-0",)
    )
    base = profiling_session_id("exp-1", "w-1", request)
    assert len(base) == 64
    assert base == profiling_session_id("exp-1", "w-1", request)
    assert base != profiling_session_id("exp-2", "w-1", request)
    assert base != profiling_session_id("exp-1", "w-2", request)
    other = ProfilingSessionRequest(
        kind=ProfilingSessionKind.OPERATOR, device_ids=("gpu-1",)
    )
    assert base != profiling_session_id("exp-1", "w-1", other)


def test_session_id_requires_non_empty_scope() -> None:
    request = ProfilingSessionRequest(kind=ProfilingSessionKind.NETWORK)
    with pytest.raises(ValueError, match="experiment_id"):
        profiling_session_id("", "w-1", request)
    with pytest.raises(ValueError, match="worker_id"):
        profiling_session_id("exp-1", "", request)
