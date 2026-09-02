"""Deterministic generation driver (spec 15.3).

Sampling is not a ``ShardModule`` responsibility: the driver runs greedy
decoding (``argmax``) outside the shard runtimes, consuming the pipeline's
logits and feeding sampled tokens back in. Top-p / top-k / temperature /
beam search are out of Phase 0 scope.
"""

from __future__ import annotations

import torch

from edgeshard.inference.pipeline import LocalPipeline
from edgeshard.inference.state import LogitsMode


class GenerationDriver:
    """Greedy generation over a local pipeline."""

    def __init__(self, pipeline: LocalPipeline) -> None:
        self._pipeline = pipeline

    def generate(
        self, session_id: str, input_ids: torch.Tensor, *, max_new_tokens: int
    ) -> list[int]:
        """Prefill the prompt, then sample ``max_new_tokens`` greedy tokens.

        Prefill runs with ``LAST_TOKEN`` logits: greedy sampling consumes
        only the final position, so the runtime projects just that position
        through the LM head instead of the whole context.
        """
        if max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
        output = self._pipeline.prefill(
            session_id, input_ids, logits_mode=LogitsMode.LAST_TOKEN
        )
        generated: list[int] = []
        token = int(output.logits[0, -1].argmax())
        generated.append(token)
        for _ in range(max_new_tokens - 1):
            output = self._pipeline.decode(session_id, token_id=token)
            token = int(output.logits[0, -1].argmax())
            generated.append(token)
        return generated[:max_new_tokens]
