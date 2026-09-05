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

Status: **Phase 0 complete (milestones 0A–0K); Phase 1 in progress.**
Phase 1 builds the Master/Worker control plane on this substrate; see
`EdgeShard v2 Phase 1 Claude Code Implementation Spec.md` for its
requirements and `ARCHITECTURE.md` for the mandatory rules and design
decisions. `KNOWN_LIMITATIONS.md` records what each phase deliberately does
not do.

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
uv run edgeshard runtime serve --config examples/configs/process-stage.yaml
# equivalent: uv run python -m edgeshard --config <runtime.yaml>
```

The legacy Phase 0 forms stay as compatibility aliases during Phase 1:
`edgeshard serve --config <runtime.yaml>` and the bare
`edgeshard --config <runtime.yaml>`.

The process loads its shard, binds the gRPC endpoint, prints
`READY runtime=<id> endpoint=<host:port>`, and serves.

## Inspect a Worker host (Phase 1)

```sh
uv run edgeshard worker inspect --config examples/configs/worker.yaml
uv run edgeshard worker inspect --config <worker.yaml> --format yaml
```

`worker inspect` needs no Master: it loads/creates the persistent
`worker_id`, discovers host capability (architecture, OS, CPU device,
memory pools, network interfaces, container runtime, and NVIDIA discrete
GPUs via NVML when a driver is present), samples telemetry, and scans the
ModelStore plus EdgeShard-managed Docker containers. Its machine-readable
output (JSON by default) is the primary local validation tool for hardware
discovery. `worker.identity_path` must be writable — point it somewhere
local on development hosts (the example uses the production location).
Hardware-specific probe validation lives in `tests/platform/` and is
selected with `pytest -m rtx` / `pytest -m jetson`.

## Examples

- `examples/configs/` — `ShardRuntimeConfig` YAML (spec 19.1): a host
  process stage and a container stage; plus a `WorkerConfig` YAML
  (Phase 1 spec §26) for `worker inspect` / `worker serve`.
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
