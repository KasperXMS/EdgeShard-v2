"""0I gate at the serialized-transport tier: partition changes preserve semantics.

Two different contiguous partitions of the same tiny model run as real
subprocess pipelines over gRPC (spec 25.3 shapes): both match the
untouched HF reference, and a valid partition change leaves the greedy
output unchanged (invariant 9). The container-tier twin of this gate
lives in ``tests/container/test_multi_container_pipeline.py`` and needs a
Docker daemon.
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
from transformers import LlamaForCausalLM

from edgeshard.control.mock.client import RemoteGenerationDriver, RemotePipeline
from edgeshard.protocol.pb import shard_runtime_pb2 as pb
from edgeshard.protocol.pb import shard_runtime_pb2_grpc as pb_grpc

pytestmark = pytest.mark.integration

EXECUTION_ID = "exec-partition-invariance"
STARTUP_TIMEOUT_S = 120.0
TOLERANCE = {"atol": 1e-6, "rtol": 1e-5}
PROMPT = [3, 7, 11, 19]

PARTITIONS: dict[str, list[dict[str, Any]]] = {
    "two-stage": [
        {
            "start_block": 0,
            "end_block": 2,
            "include_input_stage": True,
            "include_output_stage": False,
        },
        {
            "start_block": 2,
            "end_block": 4,
            "include_input_stage": False,
            "include_output_stage": True,
        },
    ],
    "three-stage": [
        {
            "start_block": 0,
            "end_block": 1,
            "include_input_stage": True,
            "include_output_stage": False,
        },
        {
            "start_block": 1,
            "end_block": 3,
            "include_input_stage": False,
            "include_output_stage": False,
        },
        {
            "start_block": 3,
            "end_block": 4,
            "include_input_stage": False,
            "include_output_stage": True,
        },
    ],
}


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


def _launch_partition(
    name: str,
    layouts: list[dict[str, Any]],
    tiny_llama_dir: Path,
    config_dir: Path,
) -> tuple[list[subprocess.Popen[str]], list[str]]:
    """Launch one partition as subprocesses (final stage first)."""
    ports = [free_port() for _ in layouts]
    endpoints = [f"127.0.0.1:{port}" for port in ports]
    processes: list[subprocess.Popen[str]] = []
    for index in reversed(range(len(layouts))):
        pipeline: dict[str, Any] = {"stage_index": index, "stage_count": len(layouts)}
        if index < len(layouts) - 1:
            pipeline["next_endpoint"] = endpoints[index + 1]
        payload = {
            "runtime": {
                "backend": "edgeshard_shard",
                "runtime_id": f"stage-{index}",
                "execution_id": EXECUTION_ID,
            },
            "model": {"id": "tiny/llama", "path": str(tiny_llama_dir)},
            "shard": layouts[index],
            "pipeline": pipeline,
            "server": {"listen_host": "127.0.0.1", "listen_port": ports[index]},
        }
        config_path = config_dir / f"{name}-stage-{index}.yaml"
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
    return processes, endpoints


@pytest.fixture(scope="module")
def clusters(
    tiny_llama_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[dict[str, list[str]]]:
    """Both partitions running at once; endpoints in pipeline order."""
    config_dir = tmp_path_factory.mktemp("partition-clusters")
    endpoints: dict[str, list[str]] = {}
    processes: list[subprocess.Popen[str]] = []
    try:
        for name, layouts in PARTITIONS.items():
            cluster_processes, cluster_endpoints = _launch_partition(
                name, layouts, tiny_llama_dir, config_dir
            )
            processes.extend(cluster_processes)
            endpoints[name] = cluster_endpoints
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


def reference_greedy(reference: LlamaForCausalLM, prompt: list[int], count: int) -> list[int]:
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


@pytest.mark.parametrize("name", sorted(PARTITIONS))
async def test_partition_matches_reference(
    name: str, clusters: dict[str, list[str]], llama_reference: LlamaForCausalLM
) -> None:
    async with RemotePipeline(
        endpoint=clusters[name][0], execution_id=EXECUTION_ID
    ) as pipeline:
        await pipeline.create_session("s")
        output = await pipeline.prefill("s", torch.tensor([PROMPT]))
        ref_output = llama_reference(torch.tensor([PROMPT]), use_cache=True)
        ref_cache = ref_output.past_key_values
        assert torch.allclose(output.logits, ref_output.logits, **TOLERANCE)
        token = int(output.logits[0, -1].argmax())
        assert token == int(ref_output.logits[0, -1].argmax())
        for _ in range(3):
            output = await pipeline.decode("s", token_id=token)
            ref_output = llama_reference(
                torch.tensor([[token]]), past_key_values=ref_cache, use_cache=True
            )
            ref_cache = ref_output.past_key_values
            assert torch.allclose(output.logits, ref_output.logits, **TOLERANCE)
            token = int(output.logits[0, -1].argmax())
            assert token == int(ref_output.logits[0, -1].argmax())
        await pipeline.close_session("s")


async def test_partition_change_preserves_greedy_output(
    clusters: dict[str, list[str]], llama_reference: LlamaForCausalLM
) -> None:
    """Invariant 9: a valid partition change does not change model output."""
    tokens_by_partition: dict[str, list[int]] = {}
    for name, endpoints in clusters.items():
        async with RemotePipeline(
            endpoint=endpoints[0], execution_id=EXECUTION_ID
        ) as pipeline:
            await pipeline.create_session("g")
            driver = RemoteGenerationDriver(pipeline)
            tokens_by_partition[name] = await driver.generate(
                "g", torch.tensor([PROMPT]), max_new_tokens=5
            )
            await pipeline.close_session("g")

    assert tokens_by_partition["two-stage"] == tokens_by_partition["three-stage"]
    assert tokens_by_partition["two-stage"] == reference_greedy(
        llama_reference, PROMPT, 5
    )
