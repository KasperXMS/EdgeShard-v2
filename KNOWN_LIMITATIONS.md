# EdgeShard v2 — Known limitations (Phase 0 freeze)

Phase 0 is a **correctness/reference layer**: it proves the shard execution
model end to end with pinned, reproducible environments. The limits below
are deliberate; later phases address them. Nothing degrades silently —
unsupported inputs fail explicitly.

## Hardware and test tiers

- Development and Tier 1 run CPU-only. Docker, CUDA, and Jetson coverage
  live in skip-gated tests (`tests/container/`, `integration` marker); on a
  host without Docker they skip with an explicit reason, never run.
- GPU and container half-precision runs are Tier 2/3 work. On the CPU
  development host, BF16 shard equivalence is verified **bitwise** against
  `from_pretrained(..., torch_dtype=bfloat16)` (Tier 1: single shard,
  two-shard split, local pipeline, and remote gRPC runtime); fp16 and CUDA
  evidence come from the GPU tiers. The default dtype is fp32.
- The vLLM E2E (`tests/container/test_vllm_runtime.py`) is double-gated:
  Docker **and** `EDGESHARD_VLLM_IMAGE` (a pinned official image, e.g.
  `vllm/vllm-openai:v0.28.0`) on a GPU host. vLLM itself is GPU-bound.
- `Dockerfile.jetson` is a pinned scaffold; its runtime validation happens
  only on target JetPack/L4T hardware (Tier 3).

## Inference semantics

- `batch_size = 1` only. Protocol structures retain batch dimensions for
  later extension, but no batched path is implemented or tested.
- Sampling is deterministic greedy only (temperature 0 for vLLM test
  requests). Sampling lives outside the shard runtimes entirely.
- Sliding-window-attention masking is delegated to the HF backbone like any
  other masking, but tests only validate prompts inside a single window.
- No streaming: each Prefill/Decode is one request/reply.

## Protocol and transport

- gRPC is the correctness/reference transport, not tuned for throughput.
- A hard 512 MiB message ceiling; no fragmentation or tensor compression.
- Spec 16.2 violations abort with explicit errors (`ProtocolError` /
  `SequencingError` / `INVALID_ARGUMENT`); there are no retries,
  reordering, or silent recovery.

## Control plane (Mock Master)

- The Mock Master is **not** a production master: no scheduling, no device
  profiling or discovery, no retries, no persistence, no health monitoring
  after deploy. It executes a provided manifest exactly.
- The partition is *provided* by the manifest, never computed.
- Deploys are all-or-nothing: any failure stops every started runtime,
  removes the network, and deletes generated configs before re-raising.
- Single Docker daemon only; runtimes of one execution share one bridge
  network.

## Models

- Supported adapters: Llama and Qwen2 families. Other HF architectures
  fail explicitly at adaptation, not approximately.
- Models must already exist in the host model cache (mounted at `/models`);
  runtime containers and drivers never download models.
- The vLLM driver's `info` reads the host model cache's `config.json`
  (never weights); it requires the model directory on the host.
- vLLM numerical equivalence with the shard pipeline is **not** asserted —
  vLLM dtypes/kernels differ. The 0J E2E asserts lifecycle management plus
  deterministic smoke completions only.

## Interfaces

- EdgeShard shard runtimes speak gRPC only (the spec 16 protocol); vLLM
  runtimes speak OpenAI-compatible HTTP only. There is no unified data
  plane across backends in Phase 0.
- Tokenizer handling is outside shard execution: prompts enter as token-id
  tensors. Container tests that need a tokenizer build one locally
  (no internet in Tier 1).
