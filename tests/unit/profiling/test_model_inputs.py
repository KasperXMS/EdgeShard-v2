"""P2D adapter input-building tests (spec §16.1, §21, §23, §24).

Pinned: shape-correct inputs from case specs, the forward-signature
convention for the primary input name, rotary resolution with typed
failure when absent, seed reproducibility, device placement, and the
typed failures — decode is never approximated from prefill
(``UNSUPPORTED_PHASE``), a case dtype that does not match the observed
checkpoint dtype fails before any benchmark runs (``UNSUPPORTED_MODEL``).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, Qwen2Config

from edgeshard.model.layout import ModelLayout
from edgeshard.profiling.domain.experiment import ModelCaseSpec, ProfilingErrorCategory
from edgeshard.profiling.domain.model import ModelCharacterization, ModelReference
from edgeshard.profiling.domain.signature import (
    InferencePhase,
    ProfilingGranularity,
    TransformerLayerSignature,
)
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.model.adapters.base import (
    LayerReference,
    ProfileModule,
    layer_list_path,
    module_device,
    primary_input_name,
)
from edgeshard.profiling.model.adapters.qwen import QwenProfilingAdapter
from edgeshard.profiling.model.layer_profiler import transformer_layer_signature

MODEL = ModelReference(model_id="tiny/qwen2")


def _layer_case(
    *,
    signature: TransformerLayerSignature,
    batch_size: int = 1,
    sequence_length: int = 8,
    dtype: str = "fp32",
    phase: InferencePhase = InferencePhase.PREFILL,
    context_length: int | None = None,
) -> ModelCaseSpec:
    return ModelCaseSpec(
        granularity=ProfilingGranularity.TRANSFORMER_LAYER,
        device_ids=("cpu-0",),
        dtype=dtype,
        model=MODEL,
        layer_signature=signature,
        phase=phase,
        batch_size=batch_size,
        sequence_length=sequence_length,
        context_length=context_length,
    )


@pytest.fixture
def signature(qwen2_characterization: ModelCharacterization) -> TransformerLayerSignature:
    return transformer_layer_signature(qwen2_characterization)


@pytest.fixture
def layers(
    tiny_qwen2_checkpoint: nn.Module, tiny_qwen2_layout: ModelLayout
) -> tuple[LayerReference, ...]:
    return QwenProfilingAdapter().enumerate_transformer_layers(
        tiny_qwen2_checkpoint, tiny_qwen2_layout
    )


class TestPrimaryInputName:
    def test_decoder_layer(self, tiny_qwen2_checkpoint: nn.Module) -> None:
        layer = tiny_qwen2_checkpoint.model.layers[0]
        assert primary_input_name(layer) == "hidden_states"

    def test_mlp_uses_x(self, tiny_qwen2_checkpoint: nn.Module) -> None:
        assert primary_input_name(tiny_qwen2_checkpoint.model.layers[0].mlp) == "x"

    def test_no_named_parameter_fails_typed(self) -> None:
        class VarArgsOnly(nn.Module):
            def forward(self, *args: torch.Tensor) -> torch.Tensor:
                return args[0]

        with pytest.raises(ProfilingError) as excinfo:
            primary_input_name(VarArgsOnly())
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_MODEL


class TestModuleDeviceAndListPath:
    def test_cpu_model(self) -> None:
        assert module_device(nn.Linear(2, 2)) == torch.device("cpu")

    def test_parameterless_module_is_cpu(self) -> None:
        assert module_device(nn.Dropout()) == torch.device("cpu")

    def test_layer_list_path(self, tiny_qwen2_layout: ModelLayout) -> None:
        assert layer_list_path(tiny_qwen2_layout) == "model.layers"

    def test_layer_list_path_requires_prefix(self, tiny_qwen2_layout: ModelLayout) -> None:
        broken = tiny_qwen2_layout.model_copy(update={"block_prefix_template": "{index}"})
        with pytest.raises(ProfilingError):
            layer_list_path(broken)


class TestBuildLayerInputs:
    def test_shapes_and_keys(
        self,
        tiny_qwen2_checkpoint: nn.Module,
        tiny_qwen2_layout: ModelLayout,
        layers: tuple,
        signature: TransformerLayerSignature,
    ) -> None:
        inputs = QwenProfilingAdapter().build_layer_inputs(
            _layer_case(signature=signature), tiny_qwen2_checkpoint, layers[0], tiny_qwen2_layout
        )
        assert set(inputs) == {"hidden_states", "position_ids", "position_embeddings", "use_cache"}
        assert inputs["hidden_states"].shape == (1, 8, 64)
        assert inputs["hidden_states"].dtype == torch.float32
        assert inputs["position_ids"].shape == (1, 8)
        assert inputs["position_ids"][0].tolist() == list(range(8))
        cos, sin = inputs["position_embeddings"]
        assert cos.shape == (1, 8, 16)
        assert sin.shape == (1, 8, 16)
        assert inputs["use_cache"] is False

    def test_batch_broadcast(
        self,
        tiny_qwen2_checkpoint: nn.Module,
        tiny_qwen2_layout: ModelLayout,
        layers: tuple,
        signature: TransformerLayerSignature,
    ) -> None:
        inputs = QwenProfilingAdapter().build_layer_inputs(
            _layer_case(signature=signature, batch_size=2),
            tiny_qwen2_checkpoint,
            layers[0],
            tiny_qwen2_layout,
        )
        assert inputs["hidden_states"].shape == (2, 8, 64)
        assert inputs["position_ids"].shape == (2, 8)
        assert torch.equal(inputs["position_ids"][0], inputs["position_ids"][1])

    def test_layer_executes_with_inputs(
        self,
        tiny_qwen2_checkpoint: nn.Module,
        tiny_qwen2_layout: ModelLayout,
        layers: tuple,
        signature: TransformerLayerSignature,
    ) -> None:
        inputs = QwenProfilingAdapter().build_layer_inputs(
            _layer_case(signature=signature), tiny_qwen2_checkpoint, layers[0], tiny_qwen2_layout
        )
        with torch.no_grad():
            outputs = layers[0].layer(**inputs)
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        assert hidden.shape == (1, 8, 64)

    def test_seed_reproducible(
        self,
        tiny_qwen2_checkpoint: nn.Module,
        tiny_qwen2_layout: ModelLayout,
        layers: tuple,
        signature: TransformerLayerSignature,
    ) -> None:
        adapter = QwenProfilingAdapter()
        case = _layer_case(signature=signature)
        first = adapter.build_layer_inputs(
            case, tiny_qwen2_checkpoint, layers[0], tiny_qwen2_layout, seed=7
        )
        second = adapter.build_layer_inputs(
            case, tiny_qwen2_checkpoint, layers[0], tiny_qwen2_layout, seed=7
        )
        other = adapter.build_layer_inputs(
            case, tiny_qwen2_checkpoint, layers[0], tiny_qwen2_layout, seed=8
        )
        assert torch.equal(first["hidden_states"], second["hidden_states"])
        assert not torch.equal(first["hidden_states"], other["hidden_states"])

    def test_decode_fails_typed(
        self,
        tiny_qwen2_checkpoint: nn.Module,
        tiny_qwen2_layout: ModelLayout,
        layers: tuple,
        signature: TransformerLayerSignature,
    ) -> None:
        case = _layer_case(
            signature=signature,
            phase=InferencePhase.DECODE,
            sequence_length=1,
            context_length=8,
        )
        with pytest.raises(ProfilingError) as excinfo:
            QwenProfilingAdapter().build_layer_inputs(
                case, tiny_qwen2_checkpoint, layers[0], tiny_qwen2_layout
            )
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_PHASE
        assert ("phase", "decode") in excinfo.value.to_failure().details

    def test_dtype_mismatch_fails_typed(
        self,
        tiny_qwen2_checkpoint: nn.Module,
        tiny_qwen2_layout: ModelLayout,
        layers: tuple,
    ) -> None:
        bf16_signature = TransformerLayerSignature(
            architecture_family="qwen2",
            layer_type="decoder",
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_kv_heads=2,
            head_dim=16,
            dtype="bf16",
            quantization=None,
        )
        case = _layer_case(signature=bf16_signature, dtype="bf16")
        with pytest.raises(ProfilingError) as excinfo:
            QwenProfilingAdapter().build_layer_inputs(
                case, tiny_qwen2_checkpoint, layers[0], tiny_qwen2_layout
            )
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_MODEL

    def test_unknown_dtype_label_rejected(
        self,
        tiny_qwen2_checkpoint: nn.Module,
        tiny_qwen2_layout: ModelLayout,
        layers: tuple,
    ) -> None:
        signature = TransformerLayerSignature(
            architecture_family="qwen2",
            layer_type="decoder",
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_kv_heads=2,
            head_dim=16,
            dtype="fp7",
            quantization=None,
        )
        case = _layer_case(signature=signature, dtype="fp7")
        with pytest.raises(ValueError, match="unknown dtype label"):
            QwenProfilingAdapter().build_layer_inputs(
                case, tiny_qwen2_checkpoint, layers[0], tiny_qwen2_layout
            )

    def test_missing_rotary_fails_typed(self, tiny_qwen2_layout: ModelLayout) -> None:
        model = AutoModelForCausalLM.from_config(
            Qwen2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
            )
        ).eval()
        model.model.rotary_emb = None
        small_layout = tiny_qwen2_layout.model_copy(
            update={
                "num_blocks": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_kv_heads": 1,
            }
        )
        layer = LayerReference(index=0, module_path="model.layers.0", layer=model.model.layers[0])
        signature = TransformerLayerSignature(
            architecture_family="qwen2",
            layer_type="decoder",
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_kv_heads=1,
            head_dim=8,
            dtype="fp32",
            quantization=None,
        )
        with pytest.raises(ProfilingError) as excinfo:
            QwenProfilingAdapter().build_layer_inputs(
                _layer_case(signature=signature), model, layer, small_layout
            )
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_MODEL
        assert "rotary" in str(excinfo.value)


class TestBuildModuleInputs:
    def _modules(
        self, tiny_qwen2_checkpoint: nn.Module, tiny_qwen2_layout: ModelLayout
    ) -> dict[str, ProfileModule]:
        layers = QwenProfilingAdapter().enumerate_transformer_layers(
            tiny_qwen2_checkpoint, tiny_qwen2_layout
        )
        modules = QwenProfilingAdapter().enumerate_profile_modules(layers[0], tiny_qwen2_layout)
        return {module.name: module for module in modules}

    def test_attention_inputs(
        self,
        tiny_qwen2_checkpoint: nn.Module,
        tiny_qwen2_layout: ModelLayout,
        signature: TransformerLayerSignature,
    ) -> None:
        modules = self._modules(tiny_qwen2_checkpoint, tiny_qwen2_layout)
        inputs = QwenProfilingAdapter().build_module_inputs(
            _layer_case(signature=signature),
            tiny_qwen2_checkpoint,
            modules["self_attn"],
            tiny_qwen2_layout,
        )
        assert set(inputs) == {"hidden_states", "attention_mask", "position_embeddings"}
        assert inputs["attention_mask"] is None
        assert inputs["hidden_states"].shape == (1, 8, 64)

    def test_mlp_inputs_use_forward_signature_name(
        self,
        tiny_qwen2_checkpoint: nn.Module,
        tiny_qwen2_layout: ModelLayout,
        signature: TransformerLayerSignature,
    ) -> None:
        modules = self._modules(tiny_qwen2_checkpoint, tiny_qwen2_layout)
        inputs = QwenProfilingAdapter().build_module_inputs(
            _layer_case(signature=signature),
            tiny_qwen2_checkpoint,
            modules["mlp"],
            tiny_qwen2_layout,
        )
        assert set(inputs) == {"x"}

    def test_modules_execute_with_inputs(
        self,
        tiny_qwen2_checkpoint: nn.Module,
        tiny_qwen2_layout: ModelLayout,
        signature: TransformerLayerSignature,
    ) -> None:
        adapter = QwenProfilingAdapter()
        modules = self._modules(tiny_qwen2_checkpoint, tiny_qwen2_layout)
        case = _layer_case(signature=signature)
        for name in ("self_attn", "mlp", "input_layernorm"):
            inputs = adapter.build_module_inputs(
                case, tiny_qwen2_checkpoint, modules[name], tiny_qwen2_layout
            )
            with torch.no_grad():
                output = modules[name].module(**inputs)
            tensor = output[0] if isinstance(output, tuple) else output
            assert tensor.shape == (1, 8, 64), name
