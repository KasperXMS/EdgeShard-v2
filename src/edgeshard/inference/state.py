"""Canonical execution state (spec 14).

These dataclasses are the version-independent boundary of shard execution:
wire semantics must never depend on a particular Transformers version, and
HF runtime objects (caches, model outputs, attention masks) never appear
here. Phase 0 execution is formally ``batch_size = 1``; batch fields are
retained for later extension.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import torch

# UP036 suppressed deliberately: the minimum *development* Python is 3.12,
# but Jetson/L4T container bases (spec 21.4) ship Python 3.10 and still run
# this code.
if sys.version_info >= (3, 11):  # noqa: UP036
    from enum import StrEnum
else:  # pragma: no cover - Jetson/L4T bases ship Python 3.10 (spec 21.4)
    from enum import Enum

    class StrEnum(str, Enum):  # noqa: UP042 - minimal enum.StrEnum backport
        """``enum.StrEnum`` semantics for pre-3.11 Pythons."""

        def __str__(self) -> str:
            return str(self.value)


class InferencePhase(StrEnum):
    """Which stage of generation a message belongs to."""

    PREFILL = "prefill"
    DECODE = "decode"


class LogitsMode(StrEnum):
    """Which positions the final shard projects to vocabulary logits.

    A prefill computes ``sequence_length`` positions; projecting every one
    of them through the LM head produces ``[batch, seq, vocab]`` logits,
    which grows with context length. Generation only ever consumes the last
    position, so it requests :attr:`LAST_TOKEN` and the final shard trims
    its hidden states *before* the LM head. :attr:`FULL` remains for local
    numerical validation and profiling. Decode is single-position either
    way.
    """

    FULL = "full"
    LAST_TOKEN = "last_token"


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
