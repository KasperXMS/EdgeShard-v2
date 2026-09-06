"""P2C profiling-adapter tests (spec §16, §50 P2C DoD).

Covered: registry resolution and typed unknown-model failures, static
characterization from Phase 0 layouts (never a fresh structural
inspection of HF configs), layer/module enumeration on a real tiny
model, structure-mismatch failures, and preservation of unknown layer
children as ``ModuleKind.OTHER``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from edgeshard.model.adapters.base import load_model_config
from edgeshard.model.adapters.registry import default_registry, resolve_adapter_for_source
from edgeshard.model.layout import ModelLayout
from edgeshard.model.source import ModelSource
from edgeshard.profiling.domain.experiment import ProfilingErrorCategory
from edgeshard.profiling.domain.model import ModelReference, StageKind
from edgeshard.profiling.domain.signature import ModuleKind, module_signature_id
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.model.adapters.base import (
    LayerReference,
    ProfilingAdapterRegistry,
    default_profiling_registry,
    observed_module_dtype,
    quantization_from_config,
    resolve_profiling_adapter,
)
from edgeshard.profiling.model.adapters.llama import LlamaProfilingAdapter
from edgeshard.profiling.model.adapters.qwen import QwenProfilingAdapter

MODEL = ModelReference(model_id="tiny/qwen2", revision="rev-1")


@pytest.fixture(scope="module")
def qwen2_model(tiny_qwen2_dir: Path) -> nn.Module:
    return AutoModelForCausalLM.from_pretrained(tiny_qwen2_dir).eval()


@pytest.fixture
def qwen2_layout(tiny_qwen2_source: ModelSource) -> ModelLayout:
    adapter = resolve_adapter_for_source(tiny_qwen2_source, default_registry())
    return adapter.inspect(tiny_qwen2_source)


@pytest.fixture
def qwen2_layer(qwen2_model: nn.Module) -> LayerReference:
    return LayerReference(index=0, module_path="model.layers.0", layer=qwen2_model.model.layers[0])


class TestProfilingAdapterRegistry:
    def test_resolves_builtin_families(self) -> None:
        registry = default_profiling_registry()
        assert isinstance(registry.resolve("qwen2"), QwenProfilingAdapter)
        assert isinstance(registry.resolve("llama"), LlamaProfilingAdapter)
        assert registry.model_types == ("llama", "qwen2")

    def test_resolves_by_architecture(self) -> None:
        registry = default_profiling_registry()
        adapter = registry.resolve(None, ("LlamaForCausalLM",))
        assert isinstance(adapter, LlamaProfilingAdapter)

    def test_unknown_model_fails_typed(self) -> None:
        registry = default_profiling_registry()
        with pytest.raises(ProfilingError) as excinfo:
            registry.resolve("gpt_bigcode", ("GPTBigCodeForCausalLM",))
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_MODEL

    def test_duplicate_registration_rejected(self) -> None:
        registry = ProfilingAdapterRegistry()
        registry.register(QwenProfilingAdapter())
        with pytest.raises(ValueError, match="already registered"):
            registry.register(QwenProfilingAdapter())

    def test_supports_matches_layout(self, qwen2_layout: ModelLayout) -> None:
        assert QwenProfilingAdapter().supports(qwen2_layout)
        assert not LlamaProfilingAdapter().supports(qwen2_layout)

    def test_resolve_profiling_adapter_verifies_support(
        self, qwen2_layout: ModelLayout
    ) -> None:
        config = SimpleNamespace(architectures=("Qwen2ForCausalLM",))
        adapter = resolve_profiling_adapter(qwen2_layout, config)  # type: ignore[arg-type]
        assert isinstance(adapter, QwenProfilingAdapter)

    def test_resolve_profiling_adapter_empty_registry(
        self, qwen2_layout: ModelLayout
    ) -> None:
        with pytest.raises(ProfilingError) as excinfo:
            resolve_profiling_adapter(
                qwen2_layout, SimpleNamespace(architectures=()), ProfilingAdapterRegistry()
            )  # type: ignore[arg-type]
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_MODEL


class TestQuantizationFromConfig:
    def test_none_without_config(self) -> None:
        assert quantization_from_config(SimpleNamespace()) is None  # type: ignore[arg-type]

    def test_dict_quant_method(self) -> None:
        config = SimpleNamespace(quantization_config={"quant_method": "awq"})
        assert quantization_from_config(config) == "awq"  # type: ignore[arg-type]

    def test_object_quant_method(self) -> None:
        config = SimpleNamespace(quantization_config=SimpleNamespace(quant_method="gptq"))
        assert quantization_from_config(config) == "gptq"  # type: ignore[arg-type]

    def test_dict_without_quant_method(self) -> None:
        config = SimpleNamespace(quantization_config={"bits": 4})
        assert quantization_from_config(config) is None  # type: ignore[arg-type]

    def test_empty_method_fails_typed(self) -> None:
        config = SimpleNamespace(quantization_config={"quant_method": ""})
        with pytest.raises(ProfilingError) as excinfo:
            quantization_from_config(config)  # type: ignore[arg-type]
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_MODEL


class TestObservedModuleDtype:
    def test_parameter_dtype(self) -> None:
        assert observed_module_dtype(nn.Linear(4, 4)) == "fp32"

    def test_cast_module(self) -> None:
        assert observed_module_dtype(nn.Linear(4, 4).to(torch.bfloat16)) == "bf16"

    def test_parameterless_module(self) -> None:
        assert observed_module_dtype(nn.Dropout()) is None

    def test_buffer_only_module(self) -> None:
        module = nn.Module()
        module.register_buffer("scale", torch.ones(2))
        assert observed_module_dtype(module) == "fp32"


class TestCharacterize:
    def test_tiny_qwen2_characterization(
        self, qwen2_layout: ModelLayout, tiny_qwen2_source: ModelSource
    ) -> None:
        config = load_model_config(tiny_qwen2_source)
        characterization = QwenProfilingAdapter().characterize(
            qwen2_layout, config, MODEL, dtype="fp32"
        )
        assert characterization.model == MODEL
        assert characterization.architecture_family == "qwen2"
        assert characterization.num_layers == 4
        assert characterization.hidden_size == 64
        assert characterization.intermediate_size == 128
        assert characterization.vocab_size == 128
        assert characterization.num_attention_heads == 4
        assert characterization.num_kv_heads == 2
        assert characterization.head_dim == 16
        assert characterization.dtype == "fp32"
        assert characterization.quantization is None
        assert characterization.tied_word_embeddings is False
        assert [stage.kind for stage in characterization.stages] == [
            StageKind.EMBEDDING,
            StageKind.TRANSFORMER_LAYER_GROUP,
            StageKind.FINAL_NORM,
            StageKind.LM_HEAD,
        ]
        assert characterization.stages[1].layer_count == 4

    def test_missing_vocab_fails_typed(
        self, qwen2_layout: ModelLayout, tiny_qwen2_source: ModelSource
    ) -> None:
        config = load_model_config(tiny_qwen2_source)
        config.vocab_size = 0
        with pytest.raises(ProfilingError) as excinfo:
            QwenProfilingAdapter().characterize(qwen2_layout, config, MODEL, dtype="fp32")
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_MODEL

    def test_head_dim_config_override(
        self, qwen2_layout: ModelLayout, tiny_qwen2_source: ModelSource
    ) -> None:
        config = load_model_config(tiny_qwen2_source)
        config.head_dim = 32
        characterization = QwenProfilingAdapter().characterize(
            qwen2_layout, config, MODEL, dtype="fp32"
        )
        assert characterization.head_dim == 32

    def test_head_dim_derived_without_override(
        self, qwen2_layout: ModelLayout, tiny_qwen2_source: ModelSource
    ) -> None:
        config = load_model_config(tiny_qwen2_source)
        config.head_dim = None
        characterization = QwenProfilingAdapter().characterize(
            qwen2_layout, config, MODEL, dtype="fp32"
        )
        assert characterization.head_dim == 16  # 64 hidden / 4 heads

    def test_quantization_from_config_flows_into_characterization(
        self, qwen2_layout: ModelLayout, tiny_qwen2_source: ModelSource
    ) -> None:
        config = load_model_config(tiny_qwen2_source)
        config.quantization_config = {"quant_method": "awq"}
        characterization = QwenProfilingAdapter().characterize(
            qwen2_layout, config, MODEL, dtype="fp32"
        )
        assert characterization.quantization == "awq"

    def test_explicit_quantization_wins(
        self, qwen2_layout: ModelLayout, tiny_qwen2_source: ModelSource
    ) -> None:
        config = load_model_config(tiny_qwen2_source)
        config.quantization_config = {"quant_method": "awq"}
        characterization = QwenProfilingAdapter().characterize(
            qwen2_layout, config, MODEL, dtype="fp32", quantization="gptq"
        )
        assert characterization.quantization == "gptq"


class TestEnumerateTransformerLayers:
    def test_tiny_qwen2_layers(
        self, qwen2_model: nn.Module, qwen2_layout: ModelLayout
    ) -> None:
        layers = QwenProfilingAdapter().enumerate_transformer_layers(qwen2_model, qwen2_layout)
        assert len(layers) == 4
        assert [layer.index for layer in layers] == [0, 1, 2, 3]
        assert [layer.module_path for layer in layers] == [
            "model.layers.0",
            "model.layers.1",
            "model.layers.2",
            "model.layers.3",
        ]
        assert layers[0].layer is qwen2_model.model.layers[0]  # type: ignore[attr-defined]

    def test_structure_mismatch_fails_typed(
        self, qwen2_model: nn.Module, qwen2_layout: ModelLayout
    ) -> None:
        wrong = qwen2_layout.model_copy(update={"block_prefix_template": "model.blocks.{index}"})
        with pytest.raises(ProfilingError) as excinfo:
            QwenProfilingAdapter().enumerate_transformer_layers(qwen2_model, wrong)
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_MODEL

    def test_non_module_list_fails_typed(
        self, qwen2_model: nn.Module, qwen2_layout: ModelLayout
    ) -> None:
        wrong = qwen2_layout.model_copy(update={"block_prefix_template": "model.norm.{index}"})
        with pytest.raises(ProfilingError) as excinfo:
            QwenProfilingAdapter().enumerate_transformer_layers(qwen2_model, wrong)
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_MODEL
        assert "ModuleList" in str(excinfo.value)


class TestEnumerateProfileModules:
    def test_tiny_qwen2_layer_modules(
        self, qwen2_layer: LayerReference, qwen2_layout: ModelLayout
    ) -> None:
        modules = QwenProfilingAdapter().enumerate_profile_modules(qwen2_layer, qwen2_layout)
        by_name = {module.name: module for module in modules}
        assert set(by_name) == {
            "input_layernorm",
            "self_attn",
            "post_attention_layernorm",
            "mlp",
        }
        assert by_name["self_attn"].kind is ModuleKind.ATTENTION
        assert by_name["mlp"].kind is ModuleKind.MLP
        assert by_name["input_layernorm"].kind is ModuleKind.NORM
        assert by_name["post_attention_layernorm"].kind is ModuleKind.NORM

        attention = by_name["self_attn"].signature
        assert attention.architecture_family == "qwen2"
        assert attention.dtype == "fp32"
        assert attention.quantization is None
        assert attention.parameters == {
            "hidden_size": 64,
            "num_heads": 4,
            "num_kv_heads": 2,
        }
        assert by_name["mlp"].signature.parameters == {
            "hidden_size": 64,
            "intermediate_size": 128,
        }
        assert by_name["input_layernorm"].signature.parameters == {
            "hidden_size": 64,
            "variant": "rms",
        }
        assert by_name["self_attn"].module_path == "model.layers.0.self_attn"

    def test_quantization_flows_into_signatures(
        self, qwen2_layer: LayerReference, qwen2_layout: ModelLayout
    ) -> None:
        modules = QwenProfilingAdapter().enumerate_profile_modules(
            qwen2_layer, qwen2_layout, quantization="awq"
        )
        assert all(module.signature.quantization == "awq" for module in modules)

    def test_unknown_children_preserved_as_other(self, qwen2_layout: ModelLayout) -> None:
        class OddLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.self_attn = nn.Linear(8, 8)
                self.fusion_gate = nn.Linear(8, 1)

        layer = LayerReference(index=0, module_path="model.layers.0", layer=OddLayer())
        modules = QwenProfilingAdapter().enumerate_profile_modules(layer, qwen2_layout)
        by_name = {module.name: module for module in modules}
        assert by_name["fusion_gate"].kind is ModuleKind.OTHER
        assert by_name["fusion_gate"].signature.parameters == {}
        assert by_name["fusion_gate"].signature.dtype == "fp32"

    def test_parameterless_child_fails_typed(self, qwen2_layout: ModelLayout) -> None:
        class DropoutLayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.self_attn = nn.Linear(8, 8)
                self.residual_dropout = nn.Dropout(0.0)

        layer = LayerReference(index=0, module_path="model.layers.0", layer=DropoutLayer())
        with pytest.raises(ProfilingError) as excinfo:
            QwenProfilingAdapter().enumerate_profile_modules(layer, qwen2_layout)
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_MODEL

    def test_module_signatures_identical_across_layers(
        self, qwen2_model: nn.Module, qwen2_layout: ModelLayout
    ) -> None:
        adapter = QwenProfilingAdapter()
        layers = adapter.enumerate_transformer_layers(qwen2_model, qwen2_layout)
        first = adapter.enumerate_profile_modules(layers[0], qwen2_layout)
        second = adapter.enumerate_profile_modules(layers[1], qwen2_layout)
        assert [module_signature_id(module.signature) for module in first] == [
            module_signature_id(module.signature) for module in second
        ]
