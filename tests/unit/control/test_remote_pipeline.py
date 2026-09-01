"""RemotePipeline + RemoteGenerationDriver over a real in-process runtime.

No Docker needed: one full-model shard runtime serves on loopback, and the
master-side client drives sessions, prefill/decode, and greedy generation
against the untouched HF reference.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import LlamaForCausalLM

from edgeshard.control.mock.client import RemoteGenerationDriver, RemotePipeline
from edgeshard.runtime.config import ShardRuntimeConfig
from edgeshard.runtime.shard_server import ShardRuntimeServer

EXECUTION_ID = "exec-remote"
TOLERANCE = {"atol": 1e-6, "rtol": 1e-5}
PROMPT = [3, 7, 11, 19]


def single_stage_config(tiny_llama_dir: Path) -> ShardRuntimeConfig:
    return ShardRuntimeConfig.model_validate(
        {
            "runtime": {
                "backend": "edgeshard_shard",
                "runtime_id": "stage-0",
                "execution_id": EXECUTION_ID,
            },
            "model": {"id": "tiny/llama", "path": str(tiny_llama_dir)},
            "shard": {
                "start_block": 0,
                "end_block": 4,
                "include_input_stage": True,
                "include_output_stage": True,
            },
            "pipeline": {"stage_index": 0, "stage_count": 1},
            "server": {"listen_host": "127.0.0.1", "listen_port": 0},
        }
    )


def reference_greedy(reference: LlamaForCausalLM, prompt: list[int], count: int) -> list[int]:
    """Greedy tokens from the untouched HF reference."""
    output = reference(torch.tensor([prompt]), use_cache=True)
    cache = output.past_key_values
    tokens: list[int] = []
    token = int(output.logits[0, -1].argmax())
    tokens.append(token)
    for _ in range(count - 1):
        output = reference(
            torch.tensor([[token]]), past_key_values=cache, use_cache=True
        )
        cache = output.past_key_values
        token = int(output.logits[0, -1].argmax())
        tokens.append(token)
    return tokens


async def test_remote_generation_matches_reference(tiny_llama_dir: Path) -> None:
    reference = LlamaForCausalLM.from_pretrained(tiny_llama_dir).eval()
    server = await ShardRuntimeServer.create(single_stage_config(tiny_llama_dir))
    try:
        async with RemotePipeline(
            endpoint=f"127.0.0.1:{server.port}", execution_id=EXECUTION_ID
        ) as pipeline:
            await pipeline.create_session("s")

            # Manual hops: prefill logits and step-by-step decode match the
            # reference exactly.
            output = await pipeline.prefill("s", torch.tensor([PROMPT]))
            ref_output = reference(torch.tensor([PROMPT]), use_cache=True)
            ref_cache = ref_output.past_key_values
            assert torch.allclose(output.logits, ref_output.logits, **TOLERANCE)
            token = int(output.logits[0, -1].argmax())
            assert token == int(ref_output.logits[0, -1].argmax())
            for _ in range(3):
                output = await pipeline.decode("s", token_id=token)
                ref_output = reference(
                    torch.tensor([[token]]), past_key_values=ref_cache, use_cache=True
                )
                ref_cache = ref_output.past_key_values
                assert torch.allclose(output.logits, ref_output.logits, **TOLERANCE)
                token = int(output.logits[0, -1].argmax())
                assert token == int(ref_output.logits[0, -1].argmax())
            await pipeline.close_session("s")

            # The generation driver runs the same greedy loop end to end.
            await pipeline.create_session("g")
            driver = RemoteGenerationDriver(pipeline)
            tokens = await driver.generate(
                "g", torch.tensor([PROMPT]), max_new_tokens=4
            )
            assert tokens == reference_greedy(reference, PROMPT, 4)
            await pipeline.close_session("g")
    finally:
        await server.stop()


async def test_remote_pipeline_rejects_bad_calls(tiny_llama_dir: Path) -> None:
    server = await ShardRuntimeServer.create(single_stage_config(tiny_llama_dir))
    try:
        async with RemotePipeline(
            endpoint=f"127.0.0.1:{server.port}", execution_id=EXECUTION_ID
        ) as pipeline:
            await pipeline.create_session("s")
            with pytest.raises(ValueError, match="prefilled"):
                await pipeline.decode("s", token_id=1)
            with pytest.raises(ValueError, match="batch_size=1"):
                await pipeline.prefill("s", torch.tensor([PROMPT, PROMPT]))
            driver = RemoteGenerationDriver(pipeline)
            with pytest.raises(ValueError, match="max_new_tokens"):
                await driver.generate("s", torch.tensor([PROMPT]), max_new_tokens=-1)
            await pipeline.close_session("s")
    finally:
        await server.stop()
