# EdgeShard v2 — Architecture (Phase 0)

This document states the mandatory architecture rules for Phase 0 of EdgeShard v2.
It is kept in sync with the code: when an interface changes, this document changes
in the same commit.

Phase 0 establishes the lowest stable execution layer:

1. Adapt supported Hugging Face decoder-only causal LMs into a canonical
   Transformer-block layout.
2. Build and execute a contiguous model shard **without** loading the complete
   model weights.
3. Define a canonical inter-shard execution protocol such that multiple shards
   compose into inference equivalent to the original full model.
4. Package inference runtimes in containers so dependency conflicts are isolated.

Validation chain:

```text
HF Reference Model ≈ Local Shard Pipeline ≈ Serialized Protocol Pipeline
    ≈ Mock-Master-driven Multi-container Pipeline
```

---

## Core concepts

Three concepts are intentionally distinct:

```text
Container = environment/dependency isolation unit
Runtime   = independently managed model execution service
Shard     = one possible model execution unit

Container != Runtime != Shard
```

A runtime container may host an EdgeShard Transformer-layer shard, an
independent full-model vLLM server, or future backends (llama.cpp, SGLang,
TensorRT-LLM, ...).

---

## Mandatory invariants

1. **Hugging Face isolation** — all Hugging Face architecture-specific behavior
   terminates at `ModelAdapter` (`src/edgeshard/model/adapters/`). No other
   module may branch on `model_type` / `architectures`.
2. **Shard independence** — `ShardModule` knows nothing about Master, Worker,
   Docker, host IPs, remote endpoints, schedulers, or deployment topology. It
   only performs model computation.
3. **Selective loading** — a shard is loadable without materializing the full
   model weights.
4. **Local KV ownership** — the KV cache belongs to the runtime that owns the
   corresponding Transformer blocks. KV cache is never transferred across shard
   boundaries during normal inference.
5. **Canonical shard boundary** — Hugging Face runtime objects (`DynamicCache`,
   `BaseModelOutput*`, `CausalLMOutput`, architecture-specific attention masks)
   never cross shard boundaries. Only EdgeShard canonical types and tensors do.
6. **Container neutrality** — containerization may change the environment but
   never inference semantics.
7. **Backend neutrality** — `RuntimeDriver` normalizes runtime lifecycle; model
   data planes may remain backend-specific.
8. **Mock Master boundary** — the Mock Master executes a provided deployment
   manifest. It does not schedule, profile devices, or discover partitions.
9. **Partition invariance** — for deterministic inference, changing a valid
   contiguous partition does not change model output beyond accepted numerical
   tolerance.
10. **Reproducibility** — runtime dependencies are pinned; runtime environments
    are identifiable by version/image digest.

---

## Dependency direction

Allowed:

```text
model ← inference ← protocol/runtime ← control.mock
```

- `model` may depend on generic utilities and HF-related libraries.
- `inference` may depend on `model`.
- `protocol` may depend on inference domain types; `inference` MUST NOT depend
  on concrete gRPC transport.
- `runtime` may depend on `model`/`inference`/`protocol`.
- `control.mock` may depend on runtime drivers and protocol clients.

Forbidden:

```text
model        -> runtime/control
inference    -> Docker
inference    -> Mock Master
inference    -> gRPC server implementation
protocol     -> model-specific adapter branching
```

---

## Repository layout

```text
proto/                    gRPC/protobuf definitions (0D+)
src/edgeshard/
  cli.py, __main__.py     runtime process entry point (`python -m edgeshard`)
  model/                  specs, sources, layouts, adapters/, weights/
  inference/              shard module, sessions, pipeline, generation
  protocol/               canonical domain, protobuf mapper, tensor codec, gRPC
  runtime/                config, info, shard server, drivers/
  control/mock/           Mock Master, manifests, deployment
containers/hf/            CPU / x86 NVIDIA / Jetson shard Dockerfiles, README
tests/unit|integration|container/
examples/configs|manifests/
```

No future-phase modules (`scheduler/`, `worker/`, production `master/`) exist
in Phase 0.

---

## Key design decisions

- **Model math is never reimplemented.** The shard backend reuses native
  Hugging Face / PyTorch implementations via meta-device skeletons
  (`init_empty_weights`) plus selective safetensors loading.
- **Wire format is canonical.** Protocol Buffers carry wire DTOs only; domain
  objects are Python dataclasses/Pydantic models mapped explicitly. Tensor
  bundles are encoded with safetensors (dtype/shape/BF16 correctness over
  performance in Phase 0).
- **Phase 0 gRPC is a correctness/reference transport**, with an explicit
  message-size ceiling above the default (512 MiB).
- **Models are mounted, not baked.** Containers mount the host model cache
  read-only (`/models:ro`); runtime containers never download models.
- **Skeleton-local layer indexing.** `build_skeleton` trims the meta-device HF
  skeleton to the shard's modules; skeleton layer index `i` corresponds to
  global block `shard.blocks.start + i`. Weight loading maps checkpoint names
  accordingly.
- **Selective loading is strict.** `SafetensorsWeightLoader` opens only files
  containing selected tensors, reads only selected tensors, and finalizes with
  `load_state_dict(strict=True, assign=True)` — proving the selection exactly
  covers the skeleton (nothing missing, nothing extra). Tied embeddings
  materialize the lm head from the embedding tensors so final shards stay
  independently loadable.
- **Local development is CPU-only** (Tier 1). PyTorch is resolved from the
  pinned CPU wheel index via `tool.uv.sources`; GPU stacks are pinned per
  container image.
- **Phase 0 inference is `batch_size = 1`**; protocol structures retain batch
  fields for later extension. Sampling is deterministic greedy, driven outside
  the shard runtime (`GenerationDriver`).
- **Canonical execution state lives in `inference/`** (`InferencePhase`,
  `ExecutionContext`, `ShardState`, `LogitsOutput`). Backends reconstruct
  whatever they need (attention masks, RoPE inputs) from this metadata;
  backend-specific masks never cross shard boundaries (spec 14.1).
- **Execution is split adapter hooks + canonical orchestration.** The spec 9.2
  sketch shows `execute_prefill/decode` on the adapter; that would invert the
  dependency direction (`model` → `inference`). Instead the adapter exposes
  native hooks (`new_cache`, `embed_tokens`, `forward_blocks`, `finalize`)
  containing every HF-version-specific detail — layer call signature, rotary
  invocation, cache API, causal mask reconstruction, skeleton-local
  `self_attn.layer_idx` repair — while `ShardModule` orchestrates sessions,
  positions, and canonical outputs.
- **Direct layer execution reconstructs causal masking.** Layers called
  outside the full HF model get no model-level mask, so the adapter builds a
  4D additive dense causal mask from canonical position metadata and cache
  length (`finfo(dtype).min` fill; decode uses an all-keep mask). Phase 0 test
  sequences stay within any sliding window, so windowed attention models
  behave identically; window-specific masks are later work.
- **`ShardModule` is device/dtype-parametrized.** All execution derives
  devices and dtypes from the module and its inputs (no hardcoded `cpu`),
  so the same code serves the CPU development tier and GPU containers.
- **Wire DTOs are generated, committed, and never the domain model.**
  `proto/shard_runtime.proto` is compiled by `scripts/generate_proto.py`
  into `src/edgeshard/protocol/pb/` (grpcio-tools is a dev dependency only);
  `protobuf_mapper.py` is the sole translation point between Python domain
  dataclasses and wire DTOs (spec 17.2). Generated code is excluded from
  lint and mypy error reporting; the pb2 `.pyi` stubs still type the mapper.
- **Tensor bundles use fixed canonical keys.** Cross-shard tensors travel as
  one safetensors bundle per message: payload tensors under `hidden_states`
  or `logits` (the key is also declared in the payload DTO) and the
  `positions` tensor alongside when present. dtype/shape/BF16 correctness
  over performance (spec 5.5).
- **Wire steps are 0-based; session counters are not.** The protocol step
  index is `PREFILL step=0`, `DECODE step=1, 2, ...` (spec 16.2), while the
  local `ShardSession.step` counts executed steps (1 after prefill); the
  runtime maps between the two when validating inbound messages.
- **Stage routing is strict and master-anchored.** `MASTER_STAGE = -1` is
  the driver/Mock Master pseudo-stage: stage 0 accepts only
  `source_stage == MASTER_STAGE`, stage `k` only `source_stage == k - 1`.
  Protocol version is enforced both on wire reads and by message validation;
  every spec 16.2 violation fails explicitly (`ProtocolError` /
  `SequencingError`), never silently recovers.
- **The serialized boundary is a port/adapter seam.** `inference/pipeline.py`
  defines the `StageTransport` port using inference domain types only;
  `protocol/boundary.py` implements it (`SerializedStageBoundary`: domain →
  protobuf → bytes → domain → spec 16.2 validation per hop). Composition
  happens in tests and the runtime server, keeping the spec 7 direction
  `protocol → inference` intact (the inference layer never imports
  protocol). `protocol/boundary.py` is an addition to the spec 6 layout.
- **Prompt ingestion evolved into the TokenPayload sequence.** At 0E the
  prompt entered stage 0 directly via `input_ids` (driver-local). At 0F the
  gRPC runtime receives the prompt as a master → stage-0 TokenPayload hop:
  `repeated int64 token_ids = 1` carries the full prompt on prefill and
  exactly one token on decode (the header phase disambiguates). Field 1 is
  wire-compatible with the earlier singular int64 field. Stage-to-stage hops
  still carry hidden states; the final reply carries logits.
- **Sampling lives outside the shard runtime.** `GenerationDriver`
  (`inference/generation.py`) runs deterministic greedy decoding over a
  `LocalPipeline`; shards never sample.
- **One process is one pipeline stage (the 0F composition root).**
  `runtime/shard_server.py` composes config → `ShardModule` → handler →
  gRPC server; `python -m edgeshard --config stage.yaml` (or the `edgeshard`
  console script) binds, prints `READY runtime=... endpoint=...`, and serves.
  Hidden states move stage → stage directly via `next_endpoint` forwarding —
  the Mock Master never relays activations; it only sends token hops and
  receives final logits replies.
- **Session RPCs chain through the pipeline with rollback** (spec 18.1):
  CreateSession/CloseSession propagate along `next_endpoint`, every runtime
  creates its own local KV state, and a refused downstream creation rolls
  back the local session.
- **Two explicit error styles at the gRPC boundary.** Session lifecycle RPCs
  return structured `ok`/`detail` replies (clients raise `ProtocolError` on
  failure); Prefill/Decode abort with `INVALID_ARGUMENT` carrying the spec
  16.2 violation text. Both surface failures loudly; neither degrades
  silently.
- **Readiness is GetRuntimeInfo.** Phase 0 adds no health-checking
  dependency; clients poll `GetRuntimeInfo` until the runtime answers (also
  how deployment tooling reads runtime identity/topology, spec 17.1).
- **Lifecycle normalization is a driver seam (0G).** `runtime/drivers/`
  implements the spec 20 `RuntimeDriver` Protocol
  (`start`/`wait_ready`/`info`/`stop`). `EdgeShardShardRuntimeDriver`
  launches one container per stage through the Docker SDK (spec 5.6: never
  shell out to `docker run`), validating the mounted config before launch:
  loopback binds and spec/config identity mismatches fail explicitly. The
  driver publishes the config's own `server.listen_port` and reuses the
  GetRuntimeInfo readiness probe — the same signal for process and container
  runtimes.
- **Models mount at `/models:ro`; configs mount read-only** (spec 21.1).
  Managed containers carry the spec 21.5 labels
  (`io.edgeshard.managed/execution_id/runtime_id/backend`) so orphaned
  Phase 0 containers can be found and cleaned.
- **One image per target tier.** `containers/hf/Dockerfile.cpu` installs the
  exact `uv.lock` set (`uv sync --frozen`), so pinned image + pinned config
  reproduces a runtime environment (spec 4.10); `Dockerfile.cuda` starts
  from the pinned official PyTorch CUDA runtime image — torch comes from the
  base (same version as the lock), every other dependency from a hashed
  `uv export` of the lock; `Dockerfile.jetson` is a pinned scaffold whose
  non-torch pins mirror the lock, validated on target JetPack/L4T (Tier 3).
  L4T bases ship Python 3.10, covered by a scoped `StrEnum` backport in
  `inference/state.py` (the development floor stays 3.12).
- **Container E2E is skip-gated, never silently dropped.** `tests/container/`
  requires a reachable Docker daemon and skips explicitly without one (this
  CPU development host has none; Tier 2/3 machines run them). The 0G
  equivalence gate — container runtime vs untouched host HF reference —
  rides the same driver lifecycle the Mock Master will use.
- **Deployment is manifest-driven (0H).** `control/mock/` implements the spec
  22 Mock Master: `DeploymentManifest` (manifest.py) carries execution ID,
  model, runtimes, and pipeline order — the partition is *provided*, never
  computed. Static topology (unique runtime IDs, pipeline/ runtime
  bijection, contiguous partition from block 0, input/output stage roles)
  validates at manifest parse; the partition's upper bound validates at
  deploy time against the model layout (`inspect`, config.json only — never
  weights). Generated artifacts land under `work_dir/<execution_id>/`.
- **The master configures containers with aliases, never host IPs (spec
  23).** `deployment.py` emits one runtime config per stage: every stage
  binds the same in-network port, downstream hops are
  `<next-runtime-id>:<port>` Docker-network aliases, and
  `MockMaster` launches stages through the driver on the execution's
  dedicated bridge network `edgeshard-exec-<execution_id>`, runtime IDs as
  container aliases. Only the entry runtime publishes a host port.
- **Readiness cascades down the pipeline.** A non-final stage waits for its
  downstream `GetRuntimeInfo` before serving (`shard_server.py`), so
  "entry ready" means the whole chain is ready — the Mock Master can probe
  only the (solely published) entry endpoint. Stages launch in reverse
  pipeline order so the cascade resolves bottom-up; a failed deploy stops
  every started container, removes the network, and deletes generated
  configs before re-raising. `shutdown` is idempotent.
- **The master-side generation mirror.** `RemotePipeline` (control/mock/
  client.py) is the master's view of the deployed pipeline: token hops go
  to the entry, logits replies come back, hidden states never traverse the
  master. It tracks each session's wire step/past length locally; the
  runtimes validate them (spec 16.2). `RemoteGenerationDriver` runs the
  same deterministic greedy loop as `inference/generation.py` — its async
  gRPC twin — because sampling stays outside the shard runtime.
- **Partition invariance is enforced at every transport tier (0I, the
  primary Phase 0 gate).** Invariant 9 — a valid partition change does not
  change deterministic output — is asserted by cross-partition token
  equivalence plus reference equivalence at the local serialized pipeline
  (0E, three mandatory partitions), the multi-process gRPC tier
  (`tests/integration/test_partition_invariance.py`, two clusters with
  different partitions running simultaneously), and the container tier
  (`tests/container/test_multi_container_pipeline.py`, three-stage vs
  two-stage Mock Master deployments). The container form is the gate's
  final link and is skip-gated where Docker is absent; the first two run
  on every CPU development host. Shipped examples (configs and manifests)
  are themselves test-parsed so documentation cannot drift from schemas.

---

## Test tiers

- **Tier 1 (every PR, CPU):** adapters, selective loading, shard execution,
  KV/session behavior, protobuf mapping, tensor codec, gRPC, CPU
  multi-container pipeline with locally generated tiny models, lint/types.
- **Tier 2 (x86 NVIDIA, nightly/on demand):** CUDA shard runtime, FP16/BF16,
  host/container equivalence, real small HF model smoke tests, vLLM driver.
- **Tier 3 (Jetson runner):** Jetson container build/run, GPU access,
  selective loading, prefill/decode, basic multi-runtime compatibility.

Tests must not require internet access in Tier 1; they generate tiny local
HF models (e.g. 4-layer Llama/Qwen2 with small hidden sizes).

---

## Milestones (strict order)

0A model adaptation → 0B selective weight loading → 0C local shard inference →
0D canonical shard protocol → 0E local/loopback pipeline → 0F runtime server →
0G containerized shard runtime → 0H Mock Master → 0I multi-container pipeline
(primary Phase 0 gate) → 0J vLLM runtime integration → 0K regression & freeze.
