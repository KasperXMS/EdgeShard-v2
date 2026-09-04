# EdgeShard v2 — Phase 0

Phase 0 builds the stable execution layer of EdgeShard v2:

1. Adapt supported Hugging Face decoder-only causal LMs into a canonical
   Transformer-block layout.
2. Execute a contiguous model shard **without** loading the complete model
   weights.
3. A canonical inter-shard protocol such that multiple shards compose into
   inference equivalent to the original full model.
4. Containerized runtimes driven by a Mock Master from deployment manifests
   — including independent full-model vLLM runtimes through the same
   backend-neutral lifecycle.

Status: **Phase 0 complete (milestones 0A–0K).** See `ARCHITECTURE.md` for
the mandatory rules and design decisions, and `KNOWN_LIMITATIONS.md` for
what Phase 0 deliberately does not do. The full requirements live in
`EdgeShard_v2_Phase0_ClaudeCode_Spec.md`.

## Requirements

Python 3.12 and [uv](https://docs.astral.sh/uv/). Development is CPU-only:
PyTorch resolves from the pinned CPU wheel index (`tool.uv.sources`).
Docker and NVIDIA GPUs are exercised by skip-gated tests only — nothing
requires them to develop or run Tier 1.

Dependencies are layered (Phase 1 §6): the installable base is the
lightweight control-plane core (config, RPC, container lifecycle, CLI); the
`inference` extra adds the torch/transformers model-execution stack and the
`worker` extra adds host telemetry probes (psutil, nvidia-ml-py). A plain
`uv sync` installs everything needed for development, including both
extras.

## Setup and gate

```sh
uv sync
# Full Tier 1 gate (lint + strict types + tests):
uv run ruff check . && uv run mypy src/edgeshard && uv run pytest -q
```

## Run a shard runtime

```sh
uv run edgeshard --config examples/configs/process-stage.yaml
# equivalent: uv run python -m edgeshard --config <runtime.yaml>
```

The process loads its shard, binds the gRPC endpoint, prints
`READY runtime=<id> endpoint=<host:port>`, and serves.

## Examples

- `examples/configs/` — `ShardRuntimeConfig` YAML (spec 19.1): a host
  process stage and a container stage.
- `examples/manifests/` — `DeploymentManifest` YAML (spec 22.2): two-shard,
  three-shard, and mixed shard+vLLM deployments.
- `containers/hf/` — pinned CPU / CUDA / Jetson shard Dockerfiles (see the
  README there).

All shipped examples are test-parsed, so they cannot drift from the
schemas.

## Tests

- `tests/unit/` — adapters, selective loading, shard execution, protocol,
  drivers, manifest/master (fake Docker clients + in-process HTTP/gRPC).
- `tests/integration/` — multi-process gRPC pipelines and partition
  invariance (`integration` marker).
- `tests/container/` — require a Docker daemon (`container` marker) and
  skip explicitly without one; the vLLM E2E additionally needs
  `EDGESHARD_VLLM_IMAGE` set on a GPU host.

Tests never require internet access; they use locally generated tiny models.

## Regenerating protocol buffers

Generated code is committed; regenerate only when `proto/shard_runtime.proto`
changes:

```sh
uv run python scripts/generate_proto.py
```
