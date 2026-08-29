"""Canonical execution state (spec 14).

These dataclasses are the version-independent boundary of shard execution:
wire semantics must never depend on a particular Transformers version, and
HF runtime objects (caches, model outputs, attention masks) never appear
here. Phase 0 execution is formally ``batch_size = 1``; batch fields are
retained for later extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch


class InferencePhase(StrEnum):
    """Which stage of generation a message belongs to."""

    PREFILL = "prefill"
    DECODE = "decode"


@dataclass(frozen=True)
class ExecutionContext:
    """Canonical sequence/position metadata (spec 14).

    Backends reconstruct whatever representation they need (attention masks,
    RoPE inputs) from this metadata; backend-specific masks are never
    transmitted (spec 14.1).
    """

    phase: InferencePhase
    step: int

    batch_size: int
    sequence_lengths: tuple[int, ...]

    past_length: int
    positions: torch.Tensor | None


@dataclass
class ShardState:
    """Hidden-state output of a non-final shard."""

    hidden_states: torch.Tensor
    context: ExecutionContext


@dataclass
class LogitsOutput:
    """Final-shard output: logits over the vocabulary."""

    logits: torch.Tensor
    context: ExecutionContext
