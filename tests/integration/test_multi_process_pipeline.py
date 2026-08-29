"""0F gate: three shard runtime processes form one correct pipeline.

Partition [0,1)+input / [1,3) / [3,4)+output (spec 25.3) launched as real
subprocesses via ``python -m edgeshard --config ...`` over gRPC. Sessions
chain across processes, logits match the untouched HF reference, wrong
execution IDs are rejected, and CloseSession propagates to the final stage.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import grpc
import pytest
import torch
import yaml
from transformers import LlamaForCausalLM

from edgeshard.inference.state import ExecutionContext, InferencePhase
from edgeshard.model.spec import BlockRange
from edgeshard.protocol.domain import (
    MASTER_STAGE,
    PROTOCOL_VERSION,
    HiddenStatePayload,
    LogitsPayload,
    ProtocolError,
    ShardMessage,
    ShardMessageHeader,
    TokenPayload,
)
from edgeshard.protocol.grpc_client import ShardRuntimeClient
from edgeshard.protocol.pb import shard_runtime_pb2 as pb
from edgeshard.protocol.pb import shard_runtime_pb2_grpc as pb_grpc
from edgeshard.runtime.info import runtime_info_from_wire

pytestmark = pytest.mark.integration

EXECUTION_ID = "exec-multi-process"
STARTUP_TIMEOUT_S = 120.0
TOLERANCE = {"atol": 1e-6, "rtol": 1e-5}

STAGE_LAYOUTS: list[dict[str, Any]] = [
    {"start_block": 0, "end_block": 1, "include_input_stage": True, "include_output_stage": False},
    {"start_block": 1, "end_block": 3, "include_input_stage": False, "include_output_stage": False},
    {"start_block": 3, "end_block": 4, "include_input_stage": False, "include_output_stage": True},
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
def cluster(
    tiny_llama_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[list[str]]:
    """Launch the three-stage pipeline as subprocesses (final stage first)."""
    config_dir = tmp_path_factory.mktemp("runtime-cluster")
    ports = [free_port() for _ in STAGE_LAYOUTS]
    endpoints = [f"127.0.0.1:{port}" for port in ports]
    processes: list[subprocess.Popen[str]] = []
    try:
        for index in reversed(range(len(STAGE_LAYOUTS))):
            pipeline: dict[str, Any] = {"stage_index": index, "stage_count": 3}
            if index < len(STAGE_LAYOUTS) - 1:
                pipeline["next_endpoint"] = endpoints[index + 1]
            payload = {
                "runtime": {
                    "backend": "edgeshard_shard",
                    "runtime_id": f"stage-{index}",
                    "execution_id": EXECUTION_ID,
                },
                "model": {"id": "tiny/llama", "path": str(tiny_llama_dir)},
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
def llama_reference(tiny_llama_dir: Path) -> LlamaForCausalLM:
    return LlamaForCausalLM.from_pretrained(tiny_llama_dir).eval()


def token_message(
    session_id: str,
    token_ids: list[int],
    *,
    phase: InferencePhase,
    step: int,
    past_length: int,
    execution_id: str = EXECUTION_ID,
) -> ShardMessage:
    """A master -> stage-0 token hop (prompt for prefill, one token for decode)."""
    return ShardMessage(
        header=ShardMessageHeader(
            protocol_version=PROTOCOL_VERSION,
            execution_id=execution_id,
            session_id=session_id,
            request_id=uuid.uuid4().hex,
            phase=phase,
            step=step,
            source_stage=MASTER_STAGE,
            target_stage=0,
        ),
        context=ExecutionContext(
            phase=phase,
            step=step,
            batch_size=1,
            sequence_lengths=(past_length + len(token_ids),),
            past_length=past_length,
            positions=None,
        ),
        payload=TokenPayload(token_ids=tuple(token_ids)),
    )


async def test_runtime_info_matches_cluster_layout(cluster: list[str]) -> None:
    for index, endpoint in enumerate(cluster):
        async with ShardRuntimeClient(endpoint) as client:
            wire = await client.get_runtime_info()
        info = runtime_info_from_wire(wire)
        layout = STAGE_LAYOUTS[index]
        assert info.runtime_id == f"stage-{index}"
        assert info.model_id == "tiny/llama"
        assert info.stage_index == index
        assert info.stage_count == 3
        assert info.blocks == BlockRange(layout["start_block"], layout["end_block"])
        assert info.include_input_stage == layout["include_input_stage"]
        assert info.include_output_stage == layout["include_output_stage"]
        assert info.protocol_version == PROTOCOL_VERSION


async def test_pipeline_logits_match_reference(
    cluster: list[str], llama_reference: LlamaForCausalLM
) -> None:
    prompt = [3, 7, 11, 19]
    session_id = "s-reference"

    async with ShardRuntimeClient(cluster[0]) as entry:
        await entry.create_session(EXECUTION_ID, session_id)
        with pytest.raises(ProtocolError, match="already exists"):
            await entry.create_session(EXECUTION_ID, session_id)

        reply = await entry.prefill(
            token_message(
                session_id, prompt, phase=InferencePhase.PREFILL, step=0, past_length=0
            )
        )
        assert reply.header.source_stage == 2
        assert reply.header.target_stage == MASTER_STAGE
        assert isinstance(reply.payload, LogitsPayload)

        ref_output = llama_reference(torch.tensor([prompt]), use_cache=True)
        ref_cache = ref_output.past_key_values
        assert torch.allclose(reply.payload.logits, ref_output.logits, **TOLERANCE)
        token = int(reply.payload.logits[0, -1].argmax())
        assert token == int(ref_output.logits[0, -1].argmax())

        past = len(prompt)
        for wire_step in range(1, 4):
            reply = await entry.decode(
                token_message(
                    session_id,
                    [token],
                    phase=InferencePhase.DECODE,
                    step=wire_step,
                    past_length=past,
                )
            )
            assert reply.header.step == wire_step
            assert isinstance(reply.payload, LogitsPayload)
            ref_output = llama_reference(
                torch.tensor([[token]]), past_key_values=ref_cache, use_cache=True
            )
            ref_cache = ref_output.past_key_values
            assert torch.allclose(reply.payload.logits, ref_output.logits, **TOLERANCE)
            token = int(reply.payload.logits[0, -1].argmax())
            assert token == int(ref_output.logits[0, -1].argmax())
            past += 1

        await entry.close_session(EXECUTION_ID, session_id)


async def test_wrong_execution_id_is_rejected(cluster: list[str]) -> None:
    session_id = "s-wrong-exec"
    async with ShardRuntimeClient(cluster[0]) as entry:
        with pytest.raises(ProtocolError, match="execution ID"):
            await entry.create_session("exec-other", session_id)

        await entry.create_session(EXECUTION_ID, session_id)
        bad = token_message(
            session_id,
            [1, 2],
            phase=InferencePhase.PREFILL,
            step=0,
            past_length=0,
            execution_id="exec-other",
        )
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await entry.prefill(bad)
        assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        await entry.close_session(EXECUTION_ID, session_id)


async def test_close_session_propagates_to_final_stage(cluster: list[str]) -> None:
    session_id = "s-close"
    hidden = torch.zeros(1, 1, 64)  # tiny llama hidden size

    def hidden_message(step: int) -> ShardMessage:
        return ShardMessage(
            header=ShardMessageHeader(
                protocol_version=PROTOCOL_VERSION,
                execution_id=EXECUTION_ID,
                session_id=session_id,
                request_id=uuid.uuid4().hex,
                phase=InferencePhase.PREFILL,
                step=step,
                source_stage=1,
                target_stage=2,
            ),
            context=ExecutionContext(
                phase=InferencePhase.PREFILL,
                step=step,
                batch_size=1,
                sequence_lengths=(1,),
                past_length=0,
                positions=None,
            ),
            payload=HiddenStatePayload(hidden_states=hidden),
        )

    async with ShardRuntimeClient(cluster[0]) as entry, ShardRuntimeClient(
        cluster[2]
    ) as final:
        # Created only at stage 0: the probe succeeding proves propagation.
        await entry.create_session(EXECUTION_ID, session_id)
        probe = await final.prefill(hidden_message(step=0))
        assert isinstance(probe.payload, LogitsPayload)

        await entry.close_session(EXECUTION_ID, session_id)
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await final.prefill(hidden_message(step=0))
        assert excinfo.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert "no such session" in excinfo.value.details()
