"""Static model characterization domain (Phase 2 spec §16-17).

Characterization is *static*: it must be possible without any performance
benchmarking (§17). It captures the structural facts a profiling strategy
needs to plan cases — architecture, layer/module hierarchy dimensions, and
the major stage structure — plus the ``ModelReference`` provenance used by
cases and measurements.

Models are not assumed to be homogeneous decoder stacks (§17): the stage
graph accommodates embedding/norm/head stages today and vision
encoder/projector stages for future VLM support. ``model_signature_id``
hashes the *structural* identity only — model ids and revisions are
provenance, recorded alongside but excluded, so two checkpoints with the
same structure share one signature (§7).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from edgeshard.profiling.domain.hashing import canonical_sha256


class StageKind(StrEnum):
    """Major stage of a model's execution graph (spec §17)."""

    EMBEDDING = "embedding"
    TRANSFORMER_LAYER_GROUP = "transformer_layer_group"
    FINAL_NORM = "final_norm"
    LM_HEAD = "lm_head"
    VISION_ENCODER = "vision_encoder"
    PROJECTOR = "projector"
    OTHER = "other"


@dataclass(frozen=True)
class ModelStage:
    """One node of the static stage graph (spec §17).

    ``layer_count`` is mandatory for ``TRANSFORMER_LAYER_GROUP`` (the number
    of repeated blocks it contains) and meaningless — hence forbidden — for
    other stage kinds.
    """

    kind: StageKind
    layer_count: int | None = None

    def __post_init__(self) -> None:
        if self.kind is StageKind.TRANSFORMER_LAYER_GROUP:
            if self.layer_count is None:
                raise ValueError("transformer layer group stage requires layer_count")
            if self.layer_count <= 0:
                raise ValueError(f"layer_count must be positive, got {self.layer_count}")
        elif self.layer_count is not None:
            raise ValueError(
                f"layer_count is only valid for transformer layer group stages, "
                f"not {self.kind.value!r}"
            )


@dataclass(frozen=True)
class ModelReference:
    """Provenance of a characterized model (mirrors ``ModelSource`` labels).

    ``model_id`` is the Hugging Face repository id (or local label) and
    ``revision`` the resolved immutable commit SHA when known.
    """

    model_id: str
    revision: str | None = None

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if self.revision is not None and not self.revision:
            raise ValueError("revision must not be empty when present")


@dataclass(frozen=True)
class ModelCharacterization:
    """Complete static characterization of one model snapshot (spec §17).

    Construction validates internal consistency loudly: grouped-query
    attention must divide evenly, and the transformer-layer-group stages
    must account for exactly ``num_layers`` layers.
    """

    model: ModelReference
    architecture_family: str

    num_layers: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int

    num_attention_heads: int
    num_kv_heads: int
    head_dim: int | None

    dtype: str
    quantization: str | None

    tied_word_embeddings: bool

    stages: tuple[ModelStage, ...]

    def __post_init__(self) -> None:
        if not self.architecture_family:
            raise ValueError("architecture_family must not be empty")
        for field_name in (
            "num_layers",
            "hidden_size",
            "intermediate_size",
            "vocab_size",
            "num_attention_heads",
            "num_kv_heads",
        ):
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be "
                f"divisible by num_kv_heads ({self.num_kv_heads})"
            )
        if self.head_dim is not None and self.head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {self.head_dim}")
        if not self.dtype:
            raise ValueError("dtype must not be empty")
        if self.quantization is not None and not self.quantization:
            raise ValueError("quantization must not be empty when present")
        if not self.stages:
            raise ValueError("stages must not be empty")
        grouped = sum(
            stage.layer_count or 0
            for stage in self.stages
            if stage.kind is StageKind.TRANSFORMER_LAYER_GROUP
        )
        if grouped != self.num_layers:
            raise ValueError(
                f"transformer layer group stages cover {grouped} layers but "
                f"num_layers is {self.num_layers}"
            )


def model_signature_id(characterization: ModelCharacterization) -> str:
    """Canonical SHA-256 structural identity of a model (spec §7, §17).

    Excludes ``model`` (id/revision are provenance, not structure): two
    checkpoints of different names but identical structure, dtype, and
    quantization share one signature and therefore one operator/layer
    workload surface.
    """
    return canonical_sha256(
        (
            "model_signature",
            characterization.architecture_family,
            characterization.num_layers,
            characterization.hidden_size,
            characterization.intermediate_size,
            characterization.vocab_size,
            characterization.num_attention_heads,
            characterization.num_kv_heads,
            characterization.head_dim,
            characterization.dtype,
            characterization.quantization,
            characterization.tied_word_embeddings,
            characterization.stages,
        )
    )
