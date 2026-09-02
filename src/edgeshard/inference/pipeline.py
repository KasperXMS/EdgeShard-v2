"""Local multi-shard pipeline with a pluggable serialized boundary (0E).

The pipeline orchestrates contiguous shards into one inference pass. The
boundary between stages is a seam (the :class:`StageTransport` port): the
inference layer names only canonical domain types, while the protocol layer
supplies the serialized implementation
(:class:`edgeshard.protocol.boundary.SerializedStageBoundary`). Composition
happens in tests and, later, the runtime server — never by making
``inference`` depend on ``protocol`` (spec 7).

Prefill input (the prompt ``input_ids``) enters stage 0 directly: the
canonical payload categories of spec 16 are Token/HiddenState/Logits, and
prompt ingestion is the driver's local concern, not an inter-shard state
transfer.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Protocol

import torch

from edgeshard.inference.shard import ShardModule
from edgeshard.inference.state import (
    ExecutionContext,
    InferencePhase,
    LogitsMode,
    LogitsOutput,
    ShardState,
)
from edgeshard.model.errors import EdgeShardError


class PipelineError(EdgeShardError):
    """Invalid pipeline composition (partition gaps or stage roles)."""


class StageTransport(Protocol):
    """One serialized hop of the boundary between pipeline stages."""

    def carry_state(
        self,
        *,
        session_id: str,
        phase: InferencePhase,
        step: int,
        source_stage: int,
        target_stage: int,
        state: ShardState,
    ) -> ShardState:
        """Carry one stage's hidden-state output to the next stage."""
        ...

    def carry_token(
        self,
        *,
        session_id: str,
        step: int,
        token_id: int,
        context: ExecutionContext,
    ) -> int:
        """Carry the sampled token from the driver to the input stage."""
        ...

    def carry_reply(
        self,
        *,
        session_id: str,
        phase: InferencePhase,
        step: int,
        source_stage: int,
        output: LogitsOutput,
    ) -> LogitsOutput:
        """Carry the final stage's logits back to the driver."""
        ...


class LocalPipeline:
    """Contiguous shards executed as one model via explicit stage hops.

    Sessions are chained across all stages (spec 18.1): creating or closing
    a session touches every shard.
    """

    def __init__(
        self,
        *,
        execution_id: str,
        stages: Sequence[ShardModule],
        transport: StageTransport,
    ) -> None:
        _validate_partition(stages)
        self._execution_id = execution_id
        self._stages = tuple(stages)
        self._transport = transport

    @property
    def execution_id(self) -> str:
        return self._execution_id

    @property
    def stage_count(self) -> int:
        return len(self._stages)

    @property
    def stages(self) -> tuple[ShardModule, ...]:
        return self._stages

    def create_session(self, session_id: str) -> None:
        """Open the session on every stage, rolling back on failure."""
        created: list[ShardModule] = []
        try:
            for shard in self._stages:
                shard.create_session(session_id)
                created.append(shard)
        except Exception:
            for shard in created:
                shard.close_session(session_id)
            raise

    def close_session(self, session_id: str) -> None:
        for shard in self._stages:
            shard.close_session(session_id)

    def prefill(
        self,
        session_id: str,
        input_ids: torch.Tensor,
        *,
        logits_mode: LogitsMode = LogitsMode.FULL,
    ) -> LogitsOutput:
        """Run the prompt through every stage; wire step for prefill is 0.

        ``logits_mode`` is a request-scoped directive for the final stage;
        middle stages pass it through untouched.
        """
        result: ShardState | LogitsOutput = self._stages[0].prefill(
            session_id, input_ids=input_ids, logits_mode=logits_mode
        )
        for target in range(1, len(self._stages)):
            if isinstance(result, LogitsOutput):
                raise PipelineError("non-final stage produced logits")
            result = self._transport.carry_state(
                session_id=session_id,
                phase=InferencePhase.PREFILL,
                step=0,
                source_stage=target - 1,
                target_stage=target,
                state=result,
            )
            result = self._stages[target].prefill(
                session_id,
                hidden_states=result.hidden_states,
                logits_mode=logits_mode,
            )
        if not isinstance(result, LogitsOutput):
            raise PipelineError("final stage produced no logits")
        return self._transport.carry_reply(
            session_id=session_id,
            phase=InferencePhase.PREFILL,
            step=0,
            source_stage=len(self._stages) - 1,
            output=result,
        )

    def decode(self, session_id: str, token_id: int) -> LogitsOutput:
        """Run one greedy step through every stage."""
        entry_session = self._stages[0].session(session_id)
        step = entry_session.step
        context = ExecutionContext(
            phase=InferencePhase.DECODE,
            step=step,
            batch_size=1,
            sequence_lengths=(entry_session.sequence_length,),
            past_length=entry_session.sequence_length,
            positions=None,
        )
        token = self._transport.carry_token(
            session_id=session_id, step=step, token_id=token_id, context=context
        )
        result: ShardState | LogitsOutput = self._stages[0].decode(
            session_id, token_id=token
        )
        for target in range(1, len(self._stages)):
            if isinstance(result, LogitsOutput):
                raise PipelineError("non-final stage produced logits")
            result = self._transport.carry_state(
                session_id=session_id,
                phase=InferencePhase.DECODE,
                step=step,
                source_stage=target - 1,
                target_stage=target,
                state=result,
            )
            result = self._stages[target].decode(
                session_id, hidden_states=result.hidden_states
            )
        if not isinstance(result, LogitsOutput):
            raise PipelineError("final stage produced no logits")
        return self._transport.carry_reply(
            session_id=session_id,
            phase=InferencePhase.DECODE,
            step=step,
            source_stage=len(self._stages) - 1,
            output=result,
        )


def _validate_partition(stages: Sequence[ShardModule]) -> None:
    """Reject partitions that do not reassemble one complete model."""
    if not stages:
        raise PipelineError("pipeline needs at least one stage")
    first, last = stages[0], stages[-1]
    num_blocks = first.layout.num_blocks
    for index, shard in enumerate(stages):
        layout = shard.layout
        if layout.num_blocks != num_blocks or layout.model_type != first.layout.model_type:
            raise PipelineError(f"stage {index} belongs to a different model")
        spec = shard.shard_spec
        if index == 0 and not spec.include_input_stage:
            raise PipelineError("first stage must include the input stage")
        if index == len(stages) - 1 and not spec.include_output_stage:
            raise PipelineError("last stage must include the output stage")
        if index != 0 and spec.include_input_stage:
            raise PipelineError("only the first stage may include the input stage")
        if index != len(stages) - 1 and spec.include_output_stage:
            raise PipelineError("only the last stage may include the output stage")
    if first.shard_spec.blocks.start != 0:
        raise PipelineError("partition must start at block 0")
    if last.shard_spec.blocks.end != num_blocks:
        raise PipelineError("partition must end at the model's last block")
    for previous, current in pairwise(stages):
        if current.shard_spec.blocks.start != previous.shard_spec.blocks.end:
            raise PipelineError(
                f"gap between blocks {previous.shard_spec.blocks.end} and "
                f"{current.shard_spec.blocks.start}"
            )
