"""ModelProfilingAdapter protocol and the standard decoder-LM base.

The adapter protocol follows spec §16.1. Characterization and
enumeration consume the Phase 0 :class:`ModelLayout` — never a fresh
inspection of HF configs for backbone structure. The HF config is read
only for facts the layout does not carry (``vocab_size``, an explicit
``head_dim`` override, the quantization method).

``build_layer_inputs``/``build_module_inputs`` (§16.1, P2D) materialize
shape-correct benchmark inputs for a case spec: v1 supports ``PREFILL``
only — decode fails with a typed ``UNSUPPORTED_PHASE`` error and is
never approximated from prefill (§24). Inputs are built on CPU with an
optional seed (reproducible experiments) and moved to the module's
actual device; the declared case dtype must match the module's observed
weight dtype, so a dtype mix-up fails before any benchmark runs.

Unknown models fail with a typed ``UNSUPPORTED_MODEL`` error (§16.1);
unexpected layer children are preserved as ``ModuleKind.OTHER`` — module
enumeration never drops structure silently (§19 in spirit: what exists
is recorded, even when unrecognized).
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from edgeshard.model.layout import ModelLayout
from edgeshard.profiling.domain.experiment import ModelCaseSpec, ProfilingErrorCategory
from edgeshard.profiling.domain.hashing import normalized_items
from edgeshard.profiling.domain.model import (
    ModelCharacterization,
    ModelReference,
    ModelStage,
    StageKind,
)
from edgeshard.profiling.domain.signature import InferencePhase, ModuleKind, ModuleSignature
from edgeshard.profiling.dtypes import dtype_label, torch_dtype
from edgeshard.profiling.errors import ProfilingError


@dataclass(frozen=True)
class LayerReference:
    """One enumerated Transformer layer of a model instance.

    ``index`` is the position within the enumerated ``ModuleList``: for a
    full model it is the global layer index; for a shard skeleton trimmed
    by Phase 0 ``keep_layer_range`` it is the skeleton-local index
    (global block = ``shard.blocks.start + index``).
    """

    index: int
    module_path: str
    layer: nn.Module


@dataclass(frozen=True)
class ProfileModule:
    """One normalized profile module inside a Transformer layer (§23)."""

    kind: ModuleKind
    name: str
    module_path: str
    module: nn.Module
    signature: ModuleSignature


class ModelProfilingAdapter(Protocol):
    """Family-specific profiling behavior (spec §16.1)."""

    #: ``layout.model_type`` handled by this adapter.
    model_type: ClassVar[str]
    #: ``config.architectures`` entries handled by this adapter.
    architectures: ClassVar[tuple[str, ...]]

    def supports(self, layout: ModelLayout) -> bool:
        """Whether this adapter profiles models with the given layout."""
        ...

    def characterize(
        self,
        layout: ModelLayout,
        config: PretrainedConfig,
        model: ModelReference,
        *,
        dtype: str,
        quantization: str | None = None,
    ) -> ModelCharacterization:
        """Static characterization without any benchmarking (§17)."""
        ...

    def enumerate_transformer_layers(
        self, model: nn.Module, layout: ModelLayout
    ) -> tuple[LayerReference, ...]:
        """The model's Transformer layers in execution order."""
        ...

    def enumerate_profile_modules(
        self, layer: LayerReference, layout: ModelLayout, *, quantization: str | None = None
    ) -> tuple[ProfileModule, ...]:
        """Normalized profile modules of one layer (§23)."""
        ...

    def build_layer_inputs(
        self,
        case: ModelCaseSpec,
        model: nn.Module,
        layer: LayerReference,
        layout: ModelLayout,
        *,
        seed: int | None = None,
    ) -> Mapping[str, Any]:
        """Shape-correct keyword inputs for one layer benchmark (§21)."""
        ...

    def build_module_inputs(
        self,
        case: ModelCaseSpec,
        model: nn.Module,
        module: ProfileModule,
        layout: ModelLayout,
        *,
        seed: int | None = None,
    ) -> Mapping[str, Any]:
        """Shape-correct keyword inputs for one module benchmark (§23)."""
        ...


def layer_list_path(layout: ModelLayout) -> str:
    """Module path of the ``ModuleList`` holding the transformer layers."""
    list_path = layout.block_prefix_template.split("{index}")[0].rstrip(".")
    if not list_path:
        raise ProfilingError(
            ProfilingErrorCategory.UNSUPPORTED_MODEL,
            f"layout block prefix template {layout.block_prefix_template!r} "
            "does not identify a layer list",
        )
    return list_path


def module_device(module: nn.Module) -> torch.device:
    """Device of the module's first parameter/buffer (CPU when it has none)."""
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def primary_input_name(module: nn.Module) -> str:
    """Keyword name of the module's primary (hidden-states) input.

    Read from the module's actual ``forward`` signature: decoder layers
    and norms take ``hidden_states``, MLPs take ``x`` — the convention
    is "the first declared parameter", never a guessed name. A module
    whose forward declares no named parameter cannot be fed and fails
    typed.
    """
    signature = inspect.signature(module.forward)
    for name, parameter in signature.parameters.items():
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        return name
    raise ProfilingError(
        ProfilingErrorCategory.UNSUPPORTED_MODEL,
        f"module {type(module).__name__} declares no named forward input; "
        "cannot build benchmark inputs",
    )


def quantization_from_config(config: PretrainedConfig) -> str | None:
    """The config's quantization method label, or ``None`` when unquantized."""
    raw: Any = getattr(config, "quantization_config", None)
    if raw is None:
        return None
    if isinstance(raw, dict):
        method = raw.get("quant_method")
    else:
        method = getattr(raw, "quant_method", None)
    if method is None:
        return None
    method_label = str(method)
    if not method_label:
        raise ProfilingError(
            ProfilingErrorCategory.UNSUPPORTED_MODEL,
            "config carries a quantization_config without a quant_method",
        )
    return method_label


def observed_module_dtype(module: nn.Module) -> str | None:
    """Label of the module's first parameter/buffer dtype, if it has one."""
    for parameter in module.parameters():
        return dtype_label(parameter.dtype)
    for buffer in module.buffers():
        return dtype_label(buffer.dtype)
    return None


class StandardDecoderProfilingAdapter:
    """Profiling adapter for HF decoder-only causal LMs with the standard
    layout (``model.embed_tokens`` / ``model.layers.{i}`` / ``model.norm``
    / ``lm_head``) — the profiling counterpart of Phase 0's
    ``StandardDecoderLMAdapter``.

    Concrete subclasses declare identity (``model_type``,
    ``architectures``, ``architecture_family``) and the layer-child
    vocabulary when it deviates from :attr:`module_kinds`.
    """

    model_type: ClassVar[str]
    architectures: ClassVar[tuple[str, ...]]
    #: Signature-level family label, e.g. ``"qwen2"``.
    architecture_family: ClassVar[str]
    #: Normalization flavor of the family's norm modules.
    norm_variant: ClassVar[str] = "rms"
    #: Layer-child names mapped to normalized module kinds; children not
    #: listed are preserved as ``ModuleKind.OTHER``.
    module_kinds: ClassVar[dict[str, ModuleKind]] = {
        "self_attn": ModuleKind.ATTENTION,
        "mlp": ModuleKind.MLP,
        "input_layernorm": ModuleKind.NORM,
        "post_attention_layernorm": ModuleKind.NORM,
    }

    def supports(self, layout: ModelLayout) -> bool:
        return layout.model_type == self.model_type

    def characterize(
        self,
        layout: ModelLayout,
        config: PretrainedConfig,
        model: ModelReference,
        *,
        dtype: str,
        quantization: str | None = None,
    ) -> ModelCharacterization:
        vocab_size = int(getattr(config, "vocab_size", 0))
        if vocab_size <= 0:
            raise ProfilingError(
                ProfilingErrorCategory.UNSUPPORTED_MODEL,
                f"config for {layout.model_type!r} carries no usable vocab_size",
                {"vocab_size": vocab_size},
            )
        return ModelCharacterization(
            model=model,
            architecture_family=self.architecture_family,
            num_layers=layout.num_blocks,
            hidden_size=layout.hidden_size,
            intermediate_size=layout.intermediate_size,
            vocab_size=vocab_size,
            num_attention_heads=layout.num_attention_heads,
            num_kv_heads=layout.num_kv_heads,
            head_dim=self.head_dim(layout, config),
            dtype=dtype,
            quantization=(
                quantization if quantization is not None else quantization_from_config(config)
            ),
            tied_word_embeddings=layout.tied_word_embeddings,
            stages=self.stage_graph(layout),
        )

    def head_dim(self, layout: ModelLayout, config: PretrainedConfig) -> int | None:
        """Per-head dimension: the config override when present, else derived."""
        override = getattr(config, "head_dim", None)
        if override:
            return int(override)
        if layout.num_attention_heads and layout.hidden_size % layout.num_attention_heads == 0:
            return layout.hidden_size // layout.num_attention_heads
        return None

    def stage_graph(self, layout: ModelLayout) -> tuple[ModelStage, ...]:
        """Homogeneous decoder stack: embed → layers → final norm → head."""
        return (
            ModelStage(kind=StageKind.EMBEDDING),
            ModelStage(kind=StageKind.TRANSFORMER_LAYER_GROUP, layer_count=layout.num_blocks),
            ModelStage(kind=StageKind.FINAL_NORM),
            ModelStage(kind=StageKind.LM_HEAD),
        )

    def enumerate_transformer_layers(
        self, model: nn.Module, layout: ModelLayout
    ) -> tuple[LayerReference, ...]:
        list_path = layer_list_path(layout)
        try:
            module_list = model.get_submodule(list_path)
        except AttributeError as exc:
            raise ProfilingError(
                ProfilingErrorCategory.UNSUPPORTED_MODEL,
                f"model structure does not match its layout: no module at {list_path!r}",
            ) from exc
        if not isinstance(module_list, nn.ModuleList):
            raise ProfilingError(
                ProfilingErrorCategory.UNSUPPORTED_MODEL,
                f"module at {list_path!r} is {type(module_list).__name__}, "
                "expected a ModuleList of transformer layers",
            )
        layers = tuple(
            LayerReference(
                index=index,
                module_path=f"{list_path}.{index}",
                layer=module_list[index],
            )
            for index in range(len(module_list))
        )
        if not layers:
            raise ProfilingError(
                ProfilingErrorCategory.UNSUPPORTED_MODEL,
                f"model exposes no transformer layers at {list_path!r}",
            )
        return layers

    def enumerate_profile_modules(
        self,
        layer: LayerReference,
        layout: ModelLayout,
        *,
        quantization: str | None = None,
    ) -> tuple[ProfileModule, ...]:
        modules: list[ProfileModule] = []
        for name, module in layer.layer.named_children():
            kind = self.module_kinds.get(name, ModuleKind.OTHER)
            dtype = observed_module_dtype(module)
            if dtype is None:
                raise ProfilingError(
                    ProfilingErrorCategory.UNSUPPORTED_MODEL,
                    f"module {layer.module_path}.{name} exposes no parameter or "
                    "buffer dtype; cannot build a module signature",
                )
            modules.append(
                ProfileModule(
                    kind=kind,
                    name=name,
                    module_path=f"{layer.module_path}.{name}",
                    module=module,
                    signature=ModuleSignature(
                        kind=kind,
                        architecture_family=self.architecture_family,
                        structural_parameters=normalized_items(
                            self.structural_parameters(kind, layout), "structural_parameters"
                        ),
                        dtype=dtype,
                        quantization=quantization,
                    ),
                )
            )
        return tuple(modules)

    def structural_parameters(
        self, kind: ModuleKind, layout: ModelLayout
    ) -> dict[str, int | float | str | bool]:
        """Normalized structural parameters per module kind (§6.2)."""
        if kind is ModuleKind.ATTENTION:
            return {
                "hidden_size": layout.hidden_size,
                "num_heads": layout.num_attention_heads,
                "num_kv_heads": layout.num_kv_heads,
            }
        if kind is ModuleKind.MLP:
            return {
                "hidden_size": layout.hidden_size,
                "intermediate_size": layout.intermediate_size,
            }
        if kind is ModuleKind.NORM:
            return {"hidden_size": layout.hidden_size, "variant": self.norm_variant}
        return {}

    def build_layer_inputs(
        self,
        case: ModelCaseSpec,
        model: nn.Module,
        layer: LayerReference,
        layout: ModelLayout,
        *,
        seed: int | None = None,
    ) -> Mapping[str, Any]:
        self._require_prefill(case)
        dtype = torch_dtype(case.dtype)  # label validity first, never guessed
        self._require_dtype_match(case.dtype, layer.layer, layer.module_path)
        device = module_device(layer.layer)
        hidden_states = self._hidden_states(case, dtype, layout, device, seed)
        position_ids = (
            torch.arange(case.sequence_length).unsqueeze(0).expand(case.batch_size, -1)
        )
        position_embeddings = self.position_embeddings(
            model, layer, layout, hidden_states, position_ids.to(device)
        )
        return {
            primary_input_name(layer.layer): hidden_states,
            "position_ids": position_ids.to(device),
            "position_embeddings": position_embeddings,
            "use_cache": False,
        }

    def build_module_inputs(
        self,
        case: ModelCaseSpec,
        model: nn.Module,
        module: ProfileModule,
        layout: ModelLayout,
        *,
        seed: int | None = None,
    ) -> Mapping[str, Any]:
        self._require_prefill(case)
        dtype = torch_dtype(case.dtype)  # label validity first, never guessed
        self._require_dtype_match(case.dtype, module.module, module.module_path)
        device = module_device(module.module)
        hidden_states = self._hidden_states(case, dtype, layout, device, seed)
        inputs: dict[str, Any] = {primary_input_name(module.module): hidden_states}
        if module.kind is ModuleKind.ATTENTION:
            # Attention modules consume rotary embeddings directly; the
            # layer reference is not needed — only the root model that
            # owns the rotary module. ``attention_mask=None`` is passed
            # explicitly: current transformers attention modules require
            # it as a positional argument, and None means "no mask" —
            # the attention implementation applies its own causality.
            layer = LayerReference(index=0, module_path=module.module_path, layer=module.module)
            position_ids = (
                torch.arange(case.sequence_length)
                .unsqueeze(0)
                .expand(case.batch_size, -1)
                .to(device)
            )
            inputs["attention_mask"] = None
            inputs["position_embeddings"] = self.position_embeddings(
                model, layer, layout, hidden_states, position_ids
            )
        return inputs

    def position_embeddings(
        self,
        model: nn.Module,
        layer: LayerReference,
        layout: ModelLayout,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotary ``(cos, sin)`` for the given positions.

        Resolves the rotary module from the installed structure: the
        model-level ``rotary_emb`` of current transformers releases
        first, then the layer/attention-level module of older ones. No
        rotary module anywhere is a typed structure failure — cos/sin are
        never fabricated.
        """
        rotary = self.resolve_rotary_emb(model, layer, layout)
        with torch.no_grad():
            cos, sin = rotary(hidden_states, position_ids)
        return cos, sin

    def resolve_rotary_emb(
        self, model: nn.Module, layer: LayerReference, layout: ModelLayout
    ) -> nn.Module:
        list_path = layer_list_path(layout)
        root, _, _ = list_path.rpartition(".")
        model_level = f"{root}.rotary_emb" if root else "rotary_emb"
        candidate = _resolved_submodule(model, model_level)
        if candidate is not None:
            return candidate
        for holder in (layer.layer, getattr(layer.layer, "self_attn", None)):
            legacy = getattr(holder, "rotary_emb", None) if holder is not None else None
            if isinstance(legacy, nn.Module):
                return legacy
        raise ProfilingError(
            ProfilingErrorCategory.UNSUPPORTED_MODEL,
            f"no rotary embedding module found at {model_level!r} or inside "
            f"{layer.module_path!r}; cannot build position embeddings",
        )

    def _hidden_states(
        self,
        case: ModelCaseSpec,
        dtype: torch.dtype,
        layout: ModelLayout,
        device: torch.device,
        seed: int | None,
    ) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        return torch.randn(
            (case.batch_size, case.sequence_length, layout.hidden_size),
            dtype=dtype,
            generator=generator,
        ).to(device)

    @staticmethod
    def _require_prefill(case: ModelCaseSpec) -> None:
        if case.phase is not InferencePhase.PREFILL:
            raise ProfilingError(
                ProfilingErrorCategory.UNSUPPORTED_PHASE,
                "decode profiling is not supported in v1: the runtime does not "
                "expose stable KV-cache decode semantics to the profiling "
                "harness (§24), and decode is never approximated from prefill",
                {"phase": case.phase.value, "context_length": case.context_length},
            )

    @staticmethod
    def _require_dtype_match(dtype: str, module: nn.Module, module_path: str) -> None:
        observed = observed_module_dtype(module)
        if observed is not None and observed != dtype:
            raise ProfilingError(
                ProfilingErrorCategory.UNSUPPORTED_MODEL,
                f"case dtype {dtype!r} does not match observed dtype {observed!r} "
                f"at {module_path!r}; load or cast the checkpoint to the "
                "measurement dtype before profiling",
                {"case_dtype": dtype, "observed_dtype": observed},
            )


def _resolved_submodule(model: nn.Module, path: str) -> nn.Module | None:
    """``model.get_submodule(path)`` when it resolves to a module, else None."""
    try:
        candidate = model.get_submodule(path)
    except AttributeError:
        return None
    return candidate if isinstance(candidate, nn.Module) else None


class ProfilingAdapterRegistry:
    """Resolves profiling adapters; unknown models fail explicitly (§16.1)."""

    def __init__(self) -> None:
        self._by_model_type: dict[str, ModelProfilingAdapter] = {}
        self._by_architecture: dict[str, ModelProfilingAdapter] = {}

    @property
    def model_types(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_model_type))

    def register(self, adapter: ModelProfilingAdapter) -> None:
        model_type: str = adapter.model_type
        if model_type in self._by_model_type:
            raise ValueError(f"profiling adapter for model_type {model_type!r} already registered")
        self._by_model_type[model_type] = adapter
        for architecture in adapter.architectures:
            if architecture in self._by_architecture:
                raise ValueError(
                    f"profiling adapter for architecture {architecture!r} already registered"
                )
            self._by_architecture[architecture] = adapter

    def resolve(
        self, model_type: str | None, architectures: tuple[str, ...] = ()
    ) -> ModelProfilingAdapter:
        if model_type is not None:
            adapter = self._by_model_type.get(model_type)
            if adapter is not None:
                return adapter
        for architecture in architectures:
            adapter = self._by_architecture.get(architecture)
            if adapter is not None:
                return adapter
        raise ProfilingError(
            ProfilingErrorCategory.UNSUPPORTED_MODEL,
            f"no profiling adapter for model_type={model_type!r}, "
            f"architectures={list(architectures)!r} "
            f"(registered model types: {list(self.model_types)})",
        )


def default_profiling_registry() -> ProfilingAdapterRegistry:
    """Registry with all built-in Phase 2 v1 profiling adapters."""
    from edgeshard.profiling.model.adapters.llama import LlamaProfilingAdapter
    from edgeshard.profiling.model.adapters.qwen import QwenProfilingAdapter

    registry = ProfilingAdapterRegistry()
    registry.register(QwenProfilingAdapter())
    registry.register(LlamaProfilingAdapter())
    return registry


def resolve_profiling_adapter(
    layout: ModelLayout,
    config: PretrainedConfig,
    registry: ProfilingAdapterRegistry | None = None,
) -> ModelProfilingAdapter:
    """Resolve the profiling adapter for a layout, verifying support.

    The Phase 0 :class:`ModelAdapter` stays responsible for producing the
    layout; this resolution never re-inspects the config for structure.
    """
    active = registry if registry is not None else default_profiling_registry()
    architectures: tuple[str, ...] = tuple(getattr(config, "architectures", None) or ())
    adapter = active.resolve(layout.model_type, architectures)
    if not adapter.supports(layout):
        raise ProfilingError(
            ProfilingErrorCategory.UNSUPPORTED_MODEL,
            f"resolved profiling adapter {type(adapter).__name__} does not support "
            f"layout model_type={layout.model_type!r}",
        )
    return adapter


__all__ = [
    "LayerReference",
    "ModelProfilingAdapter",
    "ProfileModule",
    "ProfilingAdapterRegistry",
    "StandardDecoderProfilingAdapter",
    "default_profiling_registry",
    "layer_list_path",
    "module_device",
    "observed_module_dtype",
    "primary_input_name",
    "quantization_from_config",
    "resolve_profiling_adapter",
]
