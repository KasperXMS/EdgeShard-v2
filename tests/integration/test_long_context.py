"""Long-context remote regression: 2048-token prefill + decode over gRPC.

Before the ``LogitsMode`` fix, the final shard answered every prefill with
full ``[batch, seq, vocab]`` logits, which grows with context length and
overflows the Phase 0 message ceiling for real vocabularies. Generation
requests ``LAST_TOKEN``: the final shard trims its hidden states before the
LM head, so prefill replies stay ``[batch, 1, vocab]`` at any length.

The cluster is deliberately an **uneven** two-stage partition of a 4-block
Qwen2 model — ``[0,1)+input / [1,4)+output`` — so long-context coverage
also exercises a boundary shard owning a single block.

The long-context model is generated locally with a 4096-position rope
range (Tier 1 forbids downloads); all comparisons are against the
untouched HF reference.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import grpc
import pytest
import torch
import yaml
from transformers import Qwen2Config, Qwen2ForCausalLM

from edgeshard.control.mock.client import RemoteGenerationDriver, RemotePipeline
from edgeshard.inference.pipeline import LocalPipeline
from edgeshard.inference.shard import ShardModule
from edgeshard.inference.state import LogitsMode
from edgeshard.model.source import ModelSource
from edgeshard.model.spec import BlockRange, ShardSpec
from edgeshard.protocol.boundary import SerializedStageBoundary
from edgeshard.protocol.pb import shard_runtime_pb2 as pb
from edgeshard.protocol.pb import shard_runtime_pb2_grpc as pb_grpc

pytestmark = pytest.mark.integration

EXECUTION_ID = "exec-long-context"
STARTUP_TIMEOUT_S = 120.0
TOLERANCE = {"atol": 1e-6, "rtol": 1e-5}

CONTEXT_TOKENS = 2048
DECODE_TOKENS = 8

#: Uneven boundary partition: the first shard owns exactly one block.
STAGE_LAYOUTS: list[dict[str, Any]] = [
    {"start_block": 0, "end_block": 1, "include_input_stage": True, "include_output_stage": False},
    {"start_block": 1, "end_block": 4, "include_input_stage": False, "include_output_stage": True},
]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until_ready(
    process: subprocess.Popen[str], endpoint: str, timeout_s: float
) -> None:
    """Poll GetRuntimeInfo (the Phase 0 readiness signal) until it answers."""
    deadline = time.monotonic() + timeout_s
    channel = grpc.insecure_channel(endpoint)
    stub = pb_grpc.ShardRuntimeStub(channel)
    try:
        while True:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise RuntimeError(
                    f"runtime at {endpoint} exited early "
                    f"(code {process.returncode}):\n{output}"
                )
            try:
                stub.GetRuntimeInfo(pb.RuntimeInfoRequest(), timeout=2.0)
                return
            except grpc.RpcError:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"runtime at {endpoint} did not become ready in {timeout_s}s"
                    ) from None
                time.sleep(0.25)
    finally:
        channel.close()


@pytest.fixture(scope="module")
def long_context_qwen2_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Tiny Qwen2 with a rope range that covers 2048 + decode positions."""
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
    )
    directory = tmp_path_factory.mktemp("long-context-qwen2")
    Qwen2ForCausalLM(config).save_pretrained(directory, safe_serialization=True)
    return directory


@pytest.fixture(scope="module")
def qwen_cluster(
    long_context_qwen2_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[list[str]]:
    """Launch the uneven two-stage pipeline as subprocesses."""
    config_dir = tmp_path_factory.mktemp("long-context-cluster")
    ports = [free_port() for _ in STAGE_LAYOUTS]
    endpoints = [f"127.0.0.1:{port}" for port in ports]
    processes: list[subprocess.Popen[str]] = []
    try:
        for index in reversed(range(len(STAGE_LAYOUTS))):
            pipeline: dict[str, Any] = {"stage_index": index, "stage_count": 2}
            if index < len(STAGE_LAYOUTS) - 1:
                pipeline["next_endpoint"] = endpoints[index + 1]
            payload = {
                "runtime": {
                    "backend": "edgeshard_shard",
                    "runtime_id": f"stage-{index}",
                    "execution_id": EXECUTION_ID,
                },
                "model": {"id": "tiny/qwen2", "path": str(long_context_qwen2_dir)},
                "shard": STAGE_LAYOUTS[index],
                "pipeline": pipeline,
                "server": {"listen_host": "127.0.0.1", "listen_port": ports[index]},
            }
            config_path = config_dir / f"stage-{index}.yaml"
            config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-m", "edgeshard", "--config", str(config_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        for process, endpoint in zip(processes, reversed(endpoints), strict=True):
            _wait_until_ready(process, endpoint, STARTUP_TIMEOUT_S)
        yield endpoints
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


@pytest.fixture(scope="module")
def qwen_reference(long_context_qwen2_dir: Path) -> Qwen2ForCausalLM:
    return Qwen2ForCausalLM.from_pretrained(long_context_qwen2_dir).eval()


def reference_greedy(
    reference: Qwen2ForCausalLM, prompt: list[int], count: int
) -> list[int]:
    output = reference(torch.tensor([prompt]), use_cache=True)
    cache = output.past_key_values
    tokens: list[int] = []
    token = int(output.logits[0, -1].argmax())
    tokens.append(token)
    for _ in range(count - 1):
        output = reference(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        token = int(output.logits[0, -1].argmax())
        tokens.append(token)
    return tokens


def prompt_tokens(count: int, *, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 128, (count,), generator=generator).tolist()


def build_local_pipeline(long_context_qwen2_dir: Path) -> LocalPipeline:
    source = ModelSource(path=long_context_qwen2_dir, model_id="tiny/qwen2")
    stages = [
        ShardModule.build(
            source=source,
            shard=ShardSpec(
                model_id="tiny/qwen2",
                blocks=BlockRange(layout["start_block"], layout["end_block"]),
                include_input_stage=layout["include_input_stage"],
                include_output_stage=layout["include_output_stage"],
            ),
        )
        for layout in STAGE_LAYOUTS
    ]
    return LocalPipeline(
        execution_id="exec-long-context-local",
        stages=stages,
        transport=SerializedStageBoundary(execution_id="exec-long-context-local"),
    )


async def test_qwen_hf_local_remote_match_32_tokens(
    qwen_cluster: list[str],
    long_context_qwen2_dir: Path,
    qwen_reference: Qwen2ForCausalLM,
) -> None:
    """Qwen equivalence chain: HF reference == local pipeline == remote."""
    prompt = prompt_tokens(32, seed=32)
    input_ids = torch.tensor([prompt])
    ref_logits = qwen_reference(input_ids).logits

    local = build_local_pipeline(long_context_qwen2_dir)
    local.create_session("s")
    local_output = local.prefill("s", input_ids)
    assert torch.allclose(local_output.logits, ref_logits, **TOLERANCE)

    async with RemotePipeline(
        endpoint=qwen_cluster[0], execution_id=EXECUTION_ID
    ) as pipeline:
        await pipeline.create_session("s")
        remote_output = await pipeline.prefill("s", input_ids)
        assert torch.allclose(remote_output.logits, ref_logits, **TOLERANCE)
        assert torch.allclose(remote_output.logits, local_output.logits, **TOLERANCE)
        await pipeline.close_session("s")


async def test_2048_context_prefill_and_decode_match_reference(
    qwen_cluster: list[str],
    long_context_qwen2_dir: Path,
    qwen_reference: Qwen2ForCausalLM,
) -> None:
    """The fix: a 2048-token remote prefill answers LAST_TOKEN logits.

    The reply is ``[1, 1, vocab]`` regardless of context length (full
    logits for a real vocabulary would overflow the 512 MiB message
    ceiling), and the single-position projection matches both the HF
    reference and the local pipeline's LAST_TOKEN prefill.
    """
    prompt = prompt_tokens(CONTEXT_TOKENS, seed=2048)
    input_ids = torch.tensor([prompt])
    ref_logits = qwen_reference(input_ids).logits[:, -1:, :]

    local = build_local_pipeline(long_context_qwen2_dir)
    local.create_session("s")
    local_output = local.prefill("s", input_ids, logits_mode=LogitsMode.LAST_TOKEN)
    assert local_output.logits.shape == (1, 1, 128)
    assert torch.allclose(local_output.logits, ref_logits, **TOLERANCE)

    async with RemotePipeline(
        endpoint=qwen_cluster[0], execution_id=EXECUTION_ID
    ) as pipeline:
        await pipeline.create_session("s")
        output = await pipeline.prefill("s", input_ids, logits_mode=LogitsMode.LAST_TOKEN)
        assert output.logits.shape == (1, 1, 128)
        assert torch.allclose(output.logits, ref_logits, **TOLERANCE)

        # Decode continues on the reference trajectory.
        ref_output = qwen_reference(input_ids, use_cache=True)
        ref_cache = ref_output.past_key_values
        token = int(output.logits[0, -1].argmax())
        assert token == int(ref_output.logits[0, -1].argmax())
        for _ in range(DECODE_TOKENS):
            output = await pipeline.decode("s", token_id=token)
            ref_output = qwen_reference(
                torch.tensor([[token]]), past_key_values=ref_cache, use_cache=True
            )
            ref_cache = ref_output.past_key_values
            assert torch.allclose(output.logits, ref_output.logits, **TOLERANCE)
            token = int(output.logits[0, -1].argmax())
            assert token == int(ref_output.logits[0, -1].argmax())
        await pipeline.close_session("s")


async def test_generation_driver_runs_2048_context_end_to_end(
    qwen_cluster: list[str], qwen_reference: Qwen2ForCausalLM
) -> None:
    """2048-token prefill + 8 greedy tokens == untouched HF reference."""
    prompt = prompt_tokens(CONTEXT_TOKENS, seed=4096)
    async with RemotePipeline(
        endpoint=qwen_cluster[0], execution_id=EXECUTION_ID
    ) as pipeline:
        await pipeline.create_session("g")
        driver = RemoteGenerationDriver(pipeline)
        tokens = await driver.generate(
            "g", torch.tensor([prompt]), max_new_tokens=DECODE_TOKENS
        )
        await pipeline.close_session("g")
    assert tokens == reference_greedy(qwen_reference, prompt, DECODE_TOKENS)


async def test_sessions_stay_isolated_on_the_remote_cluster(
    qwen_cluster: list[str], qwen_reference: Qwen2ForCausalLM
) -> None:
    """Two interleaved sessions keep independent KV state."""
    prompts = {"a": prompt_tokens(16, seed=11), "b": prompt_tokens(16, seed=22)}
    async with RemotePipeline(
        endpoint=qwen_cluster[0], execution_id=EXECUTION_ID
    ) as pipeline:
        await pipeline.create_session("a")
        await pipeline.create_session("b")
        outputs = {
            name: await pipeline.prefill(name, torch.tensor([prompt]))
            for name, prompt in prompts.items()
        }
        for name, prompt in prompts.items():
            ref_logits = qwen_reference(torch.tensor([prompt])).logits
            assert torch.allclose(outputs[name].logits, ref_logits, **TOLERANCE)

        # Interleaved decodes do not leak state across sessions.
        token_a = int(outputs["a"].logits[0, -1].argmax())
        token_b = int(outputs["b"].logits[0, -1].argmax())
        for _ in range(3):
            reply_a = await pipeline.decode("a", token_id=token_a)
            reply_b = await pipeline.decode("b", token_id=token_b)
            token_a = int(reply_a.logits[0, -1].argmax())
            token_b = int(reply_b.logits[0, -1].argmax())
        assert token_a == reference_greedy(qwen_reference, prompts["a"], 4)[-1]
        assert token_b == reference_greedy(qwen_reference, prompts["b"], 4)[-1]

        await pipeline.close_session("a")
        await pipeline.close_session("b")
