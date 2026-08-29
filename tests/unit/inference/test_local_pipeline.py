"""L3 correctness: local serialized pipeline matches the full HF reference.

Covers all mandatory partition cases for the 4-layer tiny model (spec 25.3):
[0,4), [0,2)+[2,4), and [0,1)+[1,3)+[3,4). Every inter-stage hop crosses
the serialized boundary (domain -> protobuf -> bytes -> domain -> validate).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import LlamaForCausalLM

from edgeshard.inference.pipeline import LocalPipeline, PipelineError
from edgeshard.inference.session import SessionError
from edgeshard.inference.shard import ShardModule
from edgeshard.inference.state import LogitsOutput
from edgeshard.model.source import ModelSource
from edgeshard.model.spec import BlockRange, ShardSpec
from edgeshard.protocol.boundary import SerializedStageBoundary

TOLERANCE = {"atol": 1e-6, "rtol": 1e-5}

Partition = list[tuple[int, int]]
MANDATORY_PARTITIONS: list[Partition] = [
    [(0, 4)],
    [(0, 2), (2, 4)],
    [(0, 1), (1, 3), (3, 4)],
]


def build_stage(source: ModelSource, start: int, end: int, **stages: bool) -> ShardModule:
    spec = ShardSpec(
        model_id="tiny/llama",
        blocks=BlockRange(start, end),
        include_input_stage=stages.get("input", False),
        include_output_stage=stages.get("output", False),
    )
    return ShardModule.build(source=source, shard=spec)


def build_pipeline(
    source: ModelSource, execution_id: str, partition: Partition
) -> LocalPipeline:
    stages = [
        build_stage(
            source,
            start,
            end,
            input=(index == 0),
            output=(index == len(partition) - 1),
        )
        for index, (start, end) in enumerate(partition)
    ]
    return LocalPipeline(
        execution_id=execution_id,
        stages=stages,
        transport=SerializedStageBoundary(execution_id=execution_id),
    )


@pytest.fixture(params=MANDATORY_PARTITIONS, ids=["single", "two-way", "three-way"])
def partition(request: pytest.FixtureRequest) -> Partition:
    return request.param


@pytest.fixture
def pipeline(partition: Partition, tiny_llama_source: ModelSource) -> LocalPipeline:
    return build_pipeline(tiny_llama_source, "exec-1", partition)


@pytest.fixture
def llama_reference(tiny_llama_dir: Path) -> LlamaForCausalLM:
    return LlamaForCausalLM.from_pretrained(tiny_llama_dir).eval()


@pytest.fixture
def prompt_ids() -> torch.Tensor:
    generator = torch.Generator().manual_seed(30)
    return torch.randint(0, 128, (1, 6), generator=generator)


def test_prefill_logits_match_reference(
    pipeline: LocalPipeline, llama_reference: LlamaForCausalLM, prompt_ids: torch.Tensor
) -> None:
    pipeline.create_session("s")
    output = pipeline.prefill("s", prompt_ids)
    assert torch.allclose(output.logits, llama_reference(prompt_ids).logits, **TOLERANCE)
    assert output.context.sequence_lengths == (6,)


def test_multistep_decode_matches_reference(
    pipeline: LocalPipeline, llama_reference: LlamaForCausalLM, prompt_ids: torch.Tensor
) -> None:
    pipeline.create_session("s")
    output = pipeline.prefill("s", prompt_ids)
    ref_output = llama_reference(prompt_ids, use_cache=True)
    ref_cache = ref_output.past_key_values

    assert torch.allclose(output.logits, ref_output.logits, **TOLERANCE)
    token = int(output.logits[0, -1].argmax())
    assert token == int(ref_output.logits[0, -1].argmax())

    for _ in range(4):
        output = pipeline.decode("s", token)
        ref_output = llama_reference(
            torch.tensor([[token]]), past_key_values=ref_cache, use_cache=True
        )
        ref_cache = ref_output.past_key_values
        assert torch.allclose(output.logits, ref_output.logits, **TOLERANCE)
        token = int(output.logits[0, -1].argmax())
        assert token == int(ref_output.logits[0, -1].argmax())


def test_pipeline_sessions_are_isolated(tiny_llama_source: ModelSource) -> None:
    """Concurrent pipeline sessions must not perturb each other (KV per stage)."""
    prompt_a = torch.tensor([[1, 5, 9, 14]])
    prompt_b = torch.tensor([[42, 17]])

    solo = build_pipeline(tiny_llama_source, "exec-solo", [(0, 1), (1, 3), (3, 4)])
    solo.create_session("a")
    solo_logits: list[torch.Tensor] = []
    out = solo.prefill("a", prompt_a)
    solo_logits.append(out.logits)
    token_a = int(out.logits[0, -1].argmax())
    for _ in range(2):
        out = solo.decode("a", token_a)
        solo_logits.append(out.logits)
        token_a = int(out.logits[0, -1].argmax())

    shared = build_pipeline(tiny_llama_source, "exec-shared", [(0, 1), (1, 3), (3, 4)])
    shared.create_session("a")
    shared.create_session("b")
    out_a = shared.prefill("a", prompt_a)
    out_b = shared.prefill("b", prompt_b)
    shared_logits = [out_a.logits]
    token_a = int(out_a.logits[0, -1].argmax())
    token_b = int(out_b.logits[0, -1].argmax())
    for _ in range(2):
        out_b = shared.decode("b", token_b)
        out_a = shared.decode("a", token_a)
        shared_logits.append(out_a.logits)
        token_a = int(out_a.logits[0, -1].argmax())
        token_b = int(out_b.logits[0, -1].argmax())

    assert len(solo_logits) == len(shared_logits)
    for expected, actual in zip(solo_logits, shared_logits, strict=True):
        assert torch.equal(expected, actual)


def test_session_chaining_and_cleanup(pipeline: LocalPipeline) -> None:
    pipeline.create_session("s")
    for shard in pipeline.stages:
        assert shard.session("s").sequence_length == 0

    with pytest.raises(SessionError, match="already exists"):
        pipeline.create_session("s")

    pipeline.close_session("s")
    for shard in pipeline.stages:
        with pytest.raises(SessionError, match="no such session"):
            shard.session("s")


def test_partition_gaps_are_rejected(tiny_llama_source: ModelSource) -> None:
    stages = [
        build_stage(tiny_llama_source, 0, 1, input=True),
        build_stage(tiny_llama_source, 2, 4, output=True),
    ]
    with pytest.raises(PipelineError, match="gap"):
        LocalPipeline(
            execution_id="e",
            stages=stages,
            transport=SerializedStageBoundary(execution_id="e"),
        )


def test_partition_must_cover_the_model(tiny_llama_source: ModelSource) -> None:
    with pytest.raises(PipelineError, match="start at block 0"):
        LocalPipeline(
            execution_id="e",
            stages=[build_stage(tiny_llama_source, 1, 4, input=True, output=True)],
            transport=SerializedStageBoundary(execution_id="e"),
        )
    with pytest.raises(PipelineError, match="last block"):
        LocalPipeline(
            execution_id="e",
            stages=[build_stage(tiny_llama_source, 0, 3, input=True, output=True)],
            transport=SerializedStageBoundary(execution_id="e"),
        )
    with pytest.raises(PipelineError, match="at least one stage"):
        LocalPipeline(
            execution_id="e", stages=[], transport=SerializedStageBoundary(execution_id="e")
        )


def test_stage_roles_are_enforced(tiny_llama_source: ModelSource) -> None:
    with pytest.raises(PipelineError, match="input stage"):
        LocalPipeline(
            execution_id="e",
            stages=[build_stage(tiny_llama_source, 0, 4, output=True)],
            transport=SerializedStageBoundary(execution_id="e"),
        )
    with pytest.raises(PipelineError, match="output stage"):
        LocalPipeline(
            execution_id="e",
            stages=[build_stage(tiny_llama_source, 0, 4, input=True)],
            transport=SerializedStageBoundary(execution_id="e"),
        )
    with pytest.raises(PipelineError, match="only the first stage"):
        LocalPipeline(
            execution_id="e",
            stages=[
                build_stage(tiny_llama_source, 0, 2, input=True),
                build_stage(tiny_llama_source, 2, 4, input=True, output=True),
            ],
            transport=SerializedStageBoundary(execution_id="e"),
        )


def test_decode_without_session_fails(pipeline: LocalPipeline) -> None:
    with pytest.raises(SessionError, match="no such session"):
        pipeline.decode("missing", 1)


def test_prefill_output_is_a_reply_message(
    pipeline: LocalPipeline, prompt_ids: torch.Tensor
) -> None:
    """The final logits reach the driver through the serialized reply hop."""
    pipeline.create_session("s")
    output = pipeline.prefill("s", prompt_ids)
    assert isinstance(output, LogitsOutput)
    assert output.logits.shape[-1] == 128  # tiny llama vocab
