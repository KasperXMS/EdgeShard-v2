"""0J gate: the Mock Master manages shards AND an independent vLLM runtime.

Complete E2E with a mixed deployment: a two-stage EdgeShard shard
pipeline plus one standalone full-model vLLM server over the same tiny
model, from one manifest, through one backend-neutral lifecycle. The
shard side matches the untouched host HF reference; the vLLM side
answers OpenAI-compatible test requests (spec 22.1). Everything is
cleaned up on shutdown.

Requirements (both, else the tests skip explicitly):
- a reachable Docker daemon (this CPU development host has none);
- ``EDGESHARD_VLLM_IMAGE`` set to a pinned official image, e.g.
  ``vllm/vllm-openai:v0.28.0`` — vLLM is GPU-bound (Tier 2), so this is
  opt-in on a GPU host.

The tiny model gets a locally built fast tokenizer added to its cache
directory: vLLM requires tokenizer files, and Tier 1 rules forbid
downloading anything.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import docker as docker_sdk
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

from edgeshard.control.mock.client import (
    RemoteGenerationDriver,
    RemotePipeline,
    VLLMClient,
)
from edgeshard.control.mock.manifest import DeploymentManifest
from edgeshard.control.mock.master import MockMaster

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_TAG = "edgeshard/hf-shard:cpu-phase0-test"
TOLERANCE = {"atol": 1e-6, "rtol": 1e-5}
PROMPT = [3, 7, 11, 19]
MAX_NEW_TOKENS = 5
VLLM_IMAGE = os.environ.get("EDGESHARD_VLLM_IMAGE")


def _docker_available() -> bool:
    try:
        docker_sdk.from_env().ping()
    except Exception:  # any docker problem means "unavailable"
        return False
    return True


pytestmark = [
    pytest.mark.container,
    pytest.mark.skipif(not _docker_available(), reason="no reachable Docker daemon"),
    pytest.mark.skipif(
        not VLLM_IMAGE,
        reason="set EDGESHARD_VLLM_IMAGE (e.g. vllm/vllm-openai:v0.28.0) on a GPU host",
    ),
]


@pytest.fixture(scope="module")
def cpu_image() -> str:
    client = docker_sdk.from_env()
    client.images.build(
        path=str(REPO_ROOT),
        dockerfile="containers/hf/Dockerfile.cpu",
        tag=IMAGE_TAG,
        rm=True,
    )
    return IMAGE_TAG


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


def _write_local_tokenizer(model_dir: Path) -> None:
    """vLLM requires tokenizer files; build one locally (no internet)."""
    vocab_size = int(
        json.loads((model_dir / "config.json").read_text(encoding="utf-8"))["vocab_size"]
    )
    vocab = {f"tok{i}": i for i in range(vocab_size)}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="tok0")),
        unk_token="tok0",
    )
    tokenizer.save_pretrained(model_dir)


@pytest.fixture
def model_cache_dir(
    tiny_llama_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    cache = tmp_path_factory.mktemp("vllm-model-cache")
    model_dir = cache / tiny_llama_dir.name
    shutil.copytree(tiny_llama_dir, model_dir)
    _write_local_tokenizer(model_dir)
    return cache


def make_manifest(
    tiny_llama_dir: Path, cpu_image: str, vllm_image: str
) -> DeploymentManifest:
    return DeploymentManifest.model_validate(
        {
            "execution_id": "exec-0j-vllm",
            "model": {
                "id": "tiny/llama",
                "path": f"/models/{tiny_llama_dir.name}",
            },
            "runtimes": [
                {
                    "id": "shard-0",
                    "backend": "edgeshard_shard",
                    "image": cpu_image,
                    "shard": {"start": 0, "end": 2, "include_input_stage": True},
                },
                {
                    "id": "shard-1",
                    "backend": "edgeshard_shard",
                    "image": cpu_image,
                    "shard": {"start": 2, "end": 4, "include_output_stage": True},
                },
                {
                    "id": "vllm-0",
                    "backend": "vllm",
                    "image": vllm_image,
                    "device": {"type": "cuda", "index": 0},
                },
            ],
            "pipeline": ["shard-0", "shard-1"],
        }
    )


async def test_mock_master_manages_shards_and_vllm(
    cpu_image: str,
    tiny_llama_dir: Path,
    llama_reference: LlamaForCausalLM,
    model_cache_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    assert VLLM_IMAGE is not None
    work_dir = tmp_path_factory.mktemp("vllm-mixed")
    master = MockMaster(
        docker_client=docker_sdk.from_env(),
        model_cache_dir=model_cache_dir,
        work_dir=work_dir,
        default_image=cpu_image,
    )
    manifest = make_manifest(tiny_llama_dir, cpu_image, VLLM_IMAGE)
    reference_tokens = reference_greedy(llama_reference, PROMPT, MAX_NEW_TOKENS)

    deployment = await master.deploy(manifest)
    container_ids = [handle.container_id for handle in deployment.handles]
    try:
        assert [handle.runtime_id for handle in deployment.handles] == [
            "shard-0",
            "shard-1",
            "vllm-0",
        ]

        # The shard pipeline still matches the untouched host reference.
        async with RemotePipeline(
            endpoint=deployment.entry_endpoint, execution_id="exec-0j-vllm"
        ) as pipeline:
            await pipeline.create_session("s")
            output = await pipeline.prefill("s", torch.tensor([PROMPT]))
            ref_output = llama_reference(torch.tensor([PROMPT]), use_cache=True)
            assert torch.allclose(output.logits, ref_output.logits, **TOLERANCE)
            token = int(output.logits[0, -1].argmax())
            assert token == int(ref_output.logits[0, -1].argmax())
            for _ in range(3):
                output = await pipeline.decode("s", token_id=token)
                token = int(output.logits[0, -1].argmax())
            await pipeline.close_session("s")

            await pipeline.create_session("g")
            driver = RemoteGenerationDriver(pipeline)
            tokens = await driver.generate(
                "g", torch.tensor([PROMPT]), max_new_tokens=MAX_NEW_TOKENS
            )
            await pipeline.close_session("g")
        assert tokens == reference_tokens

        # The vLLM runtime answers OpenAI-compatible test requests.
        vllm_endpoint = deployment.handle("vllm-0").endpoint
        async with VLLMClient(
            vllm_endpoint, model=manifest.model.path.as_posix()
        ) as client:
            assert manifest.model.path.as_posix() in await client.list_models()
            completion = await client.complete(PROMPT, max_tokens=MAX_NEW_TOKENS)
            assert completion.text
            # temperature=0 keeps vLLM test requests deterministic greedy.
            repeat = await client.complete(PROMPT, max_tokens=MAX_NEW_TOKENS)
            assert repeat.text == completion.text
    finally:
        await master.shutdown(deployment)

    # Cleanup is real: containers, network, and configs are gone.
    docker_client = docker_sdk.from_env()
    for container_id in container_ids:
        with pytest.raises(docker_sdk.errors.NotFound):
            docker_client.containers.get(container_id)
    with pytest.raises(docker_sdk.errors.NotFound):
        docker_client.networks.get(deployment.network)
    assert not (work_dir / "exec-0j-vllm").exists()
