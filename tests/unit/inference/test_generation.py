"""GenerationDriver: deterministic greedy sequences (spec 15.3, 25.4)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import LlamaForCausalLM

from edgeshard.inference.generation import GenerationDriver
from edgeshard.inference.pipeline import LocalPipeline
from edgeshard.inference.shard import ShardModule
from edgeshard.model.source import ModelSource
from edgeshard.model.spec import BlockRange, ShardSpec
from edgeshard.protocol.boundary import SerializedStageBoundary


def build_three_stage_pipeline(source: ModelSource, execution_id: str) -> LocalPipeline:
    cuts = [(0, 1), (1, 3), (3, 4)]
    stages = [
        ShardModule.build(
            source=source,
            shard=ShardSpec(
                model_id="tiny/llama",
                blocks=BlockRange(start, end),
                include_input_stage=(index == 0),
                include_output_stage=(index == len(cuts) - 1),
            ),
        )
        for index, (start, end) in enumerate(cuts)
    ]
    return LocalPipeline(
        execution_id=execution_id,
        stages=stages,
        transport=SerializedStageBoundary(execution_id=execution_id),
    )


def reference_greedy_tokens(
    model: LlamaForCausalLM, input_ids: torch.Tensor, max_new_tokens: int
) -> list[int]:
    output = model(input_ids, use_cache=True)
    cache = output.past_key_values
    tokens = [int(output.logits[0, -1].argmax())]
    for _ in range(max_new_tokens - 1):
        output = model(
            torch.tensor([[tokens[-1]]]), past_key_values=cache, use_cache=True
        )
        cache = output.past_key_values
        tokens.append(int(output.logits[0, -1].argmax()))
    return tokens


def test_greedy_sequence_matches_reference(
    tiny_llama_source: ModelSource, tiny_llama_dir: Path
) -> None:
    pipeline = build_three_stage_pipeline(tiny_llama_source, "exec-gen")
    reference = LlamaForCausalLM.from_pretrained(tiny_llama_dir).eval()
    prompt = torch.tensor([[3, 11, 29, 47, 65, 83]])

    pipeline.create_session("s")
    driver = GenerationDriver(pipeline)
    generated = driver.generate("s", prompt, max_new_tokens=6)

    assert generated == reference_greedy_tokens(reference, prompt, 6)


def test_generate_respects_max_new_tokens(
    tiny_llama_source: ModelSource, tiny_llama_dir: Path
) -> None:
    pipeline = build_three_stage_pipeline(tiny_llama_source, "exec-gen")
    reference = LlamaForCausalLM.from_pretrained(tiny_llama_dir).eval()
    prompt = torch.tensor([[3, 11, 29]])

    pipeline.create_session("zero")
    pipeline.create_session("one")
    driver = GenerationDriver(pipeline)
    assert driver.generate("zero", prompt, max_new_tokens=0) == []
    assert driver.generate("one", prompt, max_new_tokens=1) == reference_greedy_tokens(
        reference, prompt, 1
    )


def test_negative_max_new_tokens_is_rejected(tiny_llama_source: ModelSource) -> None:
    pipeline = build_three_stage_pipeline(tiny_llama_source, "exec-gen")
    pipeline.create_session("s")
    driver = GenerationDriver(pipeline)
    with pytest.raises(ValueError, match="max_new_tokens"):
        driver.generate("s", torch.tensor([[1]]), max_new_tokens=-1)


def test_128_token_generation_matches_reference(
    tiny_llama_source: ModelSource, tiny_llama_dir: Path
) -> None:
    """Long greedy runs stay on the reference trajectory.

    The driver prefills with LAST_TOKEN logits and then decodes 128 times;
    every sampled token must match the untouched HF reference.
    """
    pipeline = build_three_stage_pipeline(tiny_llama_source, "exec-gen-128")
    reference = LlamaForCausalLM.from_pretrained(tiny_llama_dir).eval()
    prompt = torch.tensor([[3, 11, 29]])

    pipeline.create_session("s")
    driver = GenerationDriver(pipeline)
    generated = driver.generate("s", prompt, max_new_tokens=128)

    assert generated == reference_greedy_tokens(reference, prompt, 128)
