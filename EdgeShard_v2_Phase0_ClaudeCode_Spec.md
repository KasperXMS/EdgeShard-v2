# EdgeShard v2 — Phase 0 Implementation Specification
## Containerized Inference Runtime Foundation

**Purpose:** This document is the implementation specification for Claude Code.  
**Scope:** Phase 0 only.  
**Development approach:** bottom-up; every milestone must leave the implemented module independently runnable, testable, and stable before moving upward.

---

## 1. Phase 0 Goal

Phase 0 establishes the lowest stable execution layer of EdgeShard v2.

It must solve four problems:

1. Automatically adapt supported Hugging Face decoder-only causal LMs into a canonical Transformer-block layout.
2. Build and execute a contiguous model shard without loading the complete model weights.
3. Define a canonical inter-shard execution protocol such that multiple shards compose into inference equivalent to the original full model.
4. Package inference runtimes in containers so dependency/version conflicts are isolated from the host and from other runtime backends.

Phase 0 also includes a **Mock Master**. The Mock Master is not a production control plane. It exists only to validate that independently packaged runtimes can be instantiated, connected, driven, and shut down using stable contracts.

The target validation chain is:

```text
Hugging Face Reference Model
          ≈
Local EdgeShard Shard Pipeline
          ≈
Serialized Protocol Pipeline
          ≈
Mock-Master-driven Multi-container Pipeline
```

For a deterministic test configuration, partitioning must not change model semantics.

---

## 2. Non-goals

Phase 0 MUST NOT implement:

- production Master;
- Worker Agent;
- automatic scheduling;
- device profiling;
- operator profiling;
- cost estimation;
- automatic device selection;
- persistent cluster state;
- fault-tolerant distributed execution;
- dynamic re-sharding;
- migration;
- Kubernetes;
- Ray;
- production RDMA/NCCL/UCX transport;
- Prometheus/Grafana;
- authentication/TLS;
- multi-user production serving.

If an implementation task requires one of the above to make Phase 0 work, the design boundary is probably wrong.

---

## 3. Core Architecture

```text
                         Mock Master
                              |
                     DeploymentManifest
                              |
                        RuntimeDriver
                              |
              +---------------+---------------+
              |                               |
              v                               v
    +-------------------+           +-------------------+
    | EdgeShard Runtime |           |   vLLM Runtime    |
    |    Container      |           |    Container      |
    |                   |           |                   |
    | Shard Runtime     |           | vLLM OpenAI       |
    |    Server         |           | Server            |
    +---------+---------+           +-------------------+
              |
              | EdgeShard Shard Protocol
              v
    +-------------------+
    | Next Shard Runtime|
    +-------------------+
```

The following concepts are intentionally distinct:

```text
Container = environment/dependency isolation unit
Runtime   = independently managed model execution service
Shard     = one possible model execution unit
```

Therefore:

```text
Container != Runtime != Shard
```

A runtime container may host:

- an EdgeShard-owned Transformer-layer shard;
- an independent full-model vLLM server;
- future backends such as llama.cpp, SGLang, TensorRT-LLM, etc.

---

## 4. Architecture Invariants

These rules are mandatory and should be documented in `ARCHITECTURE.md`.

### 4.1 Hugging Face isolation

All Hugging Face architecture-specific behavior MUST terminate at `ModelAdapter`.

No other module may contain architecture branches such as:

```python
if model_type == "llama":
    ...
elif model_type == "qwen2":
    ...
```

outside the adapter registry/implementations.

### 4.2 Shard independence

`ShardModule` MUST NOT know about:

- Master;
- Worker;
- Docker;
- host IPs;
- remote endpoints;
- scheduler;
- deployment topology.

It only performs model computation.

### 4.3 Selective loading

A shard MUST be loadable without first loading the complete model weights.

Building a 70B shard on a memory-constrained node MUST NOT require the node to materialize a 70B full model.

### 4.4 Local KV ownership

KV cache belongs to the runtime that owns the corresponding Transformer blocks.

KV cache MUST NOT be transferred across shard boundaries during normal inference.

### 4.5 Canonical shard boundary

Hugging Face runtime objects MUST NOT cross shard boundaries.

Examples forbidden on the wire:

- `DynamicCache`;
- `BaseModelOutput`;
- `BaseModelOutputWithPast`;
- `CausalLMOutput`;
- architecture-specific attention-mask objects.

Only EdgeShard canonical types and tensors may cross the boundary.

### 4.6 Container neutrality

Containerization may change the environment, but MUST NOT change inference semantics.

### 4.7 Backend neutrality

Runtime lifecycle is normalized by `RuntimeDriver`; model data planes are allowed to remain backend-specific.

### 4.8 Mock Master boundary

The Mock Master executes a provided deployment manifest. It MUST NOT schedule, profile devices, or discover an optimal partition.

### 4.9 Partition invariance

For deterministic inference, changing a valid contiguous partition MUST NOT change the model output beyond accepted numerical tolerance.

### 4.10 Reproducibility

Runtime dependencies MUST be pinned and runtime environments MUST be identifiable by version/image digest.

---

## 5. Development Stack

### 5.1 Language and package management

- Python: **3.12**
- Project metadata: `pyproject.toml`
- Package/environment management: **uv**
- Lock file: `uv.lock`

Do not use ad-hoc `pip install latest` behavior in runtime images.

### 5.2 Model execution

- PyTorch
- Hugging Face Transformers
- Accelerate only where useful for empty/meta initialization
- safetensors
- huggingface_hub

The custom shard backend MUST reuse native Hugging Face/PyTorch model implementations. Do not reimplement Transformer attention, MLP, RMSNorm, RoPE, or cache logic unless an adapter cannot safely reuse the upstream implementation.

### 5.3 Schemas and configuration

- Pydantic v2 for:
  - `RuntimeConfig`;
  - `RuntimeInfo`;
  - `DeploymentManifest`;
  - `ModelSpec`;
  - `ShardSpec`;
  - lifecycle/configuration objects.
- Python dataclasses for hot-path tensor-bearing inference objects.
- YAML for human-authored configuration and deployment manifests.
- PyYAML for parsing.

### 5.4 Shard RPC baseline

- `grpc.aio`
- Protocol Buffers

Phase 0 gRPC is a **correctness/reference transport**, not a final performance commitment.

### 5.5 Tensor serialization baseline

For Phase 0:

- safetensors tensor bundle inside RPC payload.

This intentionally trades performance for:
- dtype correctness;
- shape preservation;
- BF16 support;
- implementation simplicity.

Future transports may replace it without changing inference semantics.

### 5.6 Containers

- Docker
- NVIDIA Container Toolkit for NVIDIA devices
- Docker SDK for Python for container lifecycle management

Do not scatter shell calls to `docker run` through control code.

### 5.7 vLLM

Use the official `vllm/vllm-openai:<pinned-version>` container.

Do **not** add vLLM as a dependency of the EdgeShard custom HF shard runtime.

The Mock Master interacts with vLLM through its existing OpenAI-compatible HTTP API.

### 5.8 CLI and HTTP clients

- Typer for CLI
- HTTPX for HTTP client operations

### 5.9 Testing and code quality

- pytest
- pytest-asyncio
- Hypothesis
- Ruff
- mypy

---

## 6. Repository Layout

```text
edgeshard/
├── pyproject.toml
├── uv.lock
├── ARCHITECTURE.md
│
├── proto/
│   └── shard_runtime.proto
│
├── src/
│   └── edgeshard/
│       ├── model/
│       │   ├── spec.py
│       │   ├── source.py
│       │   ├── layout.py
│       │   │
│       │   ├── adapters/
│       │   │   ├── base.py
│       │   │   ├── registry.py
│       │   │   ├── llama.py
│       │   │   ├── qwen2.py
│       │   │   └── generic.py
│       │   │
│       │   └── weights/
│       │       ├── base.py
│       │       └── safetensors.py
│       │
│       ├── inference/
│       │   ├── state.py
│       │   ├── session.py
│       │   ├── shard.py
│       │   ├── builder.py
│       │   ├── pipeline.py
│       │   └── generation.py
│       │
│       ├── protocol/
│       │   ├── domain.py
│       │   ├── protobuf_mapper.py
│       │   ├── tensor_codec.py
│       │   ├── grpc_client.py
│       │   └── grpc_server.py
│       │
│       ├── runtime/
│       │   ├── config.py
│       │   ├── info.py
│       │   ├── shard_server.py
│       │   │
│       │   └── drivers/
│       │       ├── base.py
│       │       ├── edgeshard_shard.py
│       │       └── vllm.py
│       │
│       └── control/
│           └── mock/
│               ├── master.py
│               ├── manifest.py
│               ├── deployment.py
│               └── client.py
│
├── containers/
│   ├── hf/
│   │   ├── Dockerfile.cpu
│   │   ├── Dockerfile.cuda
│   │   └── Dockerfile.jetson
│   └── README.md
│
├── tests/
│   ├── unit/
│   │   ├── model/
│   │   ├── inference/
│   │   └── protocol/
│   │
│   ├── integration/
│   └── container/
│
└── examples/
    ├── configs/
    └── manifests/
```

Do not create empty future-phase modules such as `scheduler/`, `worker/`, or production `master/` during Phase 0.

---

## 7. Dependency Direction

Allowed dependency direction:

```text
model
  ↑
inference
  ↑
protocol/runtime
  ↑
mock control
```

More explicitly:

- `model` may depend on generic utilities and HF-related libraries.
- `inference` may depend on `model`.
- `protocol` may depend on inference domain types, but inference MUST NOT depend on concrete gRPC transport.
- `runtime` may depend on model/inference/protocol.
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

## 8. Model Abstraction

### 8.1 Supported model family in Phase 0

Formal support:

- decoder-only causal language models;
- Llama family;
- Qwen2/Qwen2.5 family.

Optional third architecture for robustness:
- Mistral.

Explicitly out of scope:

- encoder-decoder;
- MoE;
- VLM;
- speculative decoding;
- tensor parallelism;
- continuous batching.

### 8.2 Canonical model structure

```text
Input Stage
└── Token Embedding
        |
        v
Transformer Block 0
Transformer Block 1
...
Transformer Block N-1
        |
        v
Output Stage
├── Final Norm
└── LM Head
```

The partitioning unit is a contiguous Transformer block range.

### 8.3 BlockRange

Use Python half-open interval semantics everywhere.

```python
@dataclass(frozen=True)
class BlockRange:
    start: int
    end: int

    def __post_init__(self):
        if self.start < 0 or self.end <= self.start:
            raise ValueError("invalid block range")
```

Example:

```python
BlockRange(8, 20)
```

means blocks 8 through 19.

Never introduce inclusive ranges elsewhere.

### 8.4 ShardSpec

```python
class ShardSpec(BaseModel):
    model_id: str
    blocks: BlockRange
    include_input_stage: bool = False
    include_output_stage: bool = False
```

`ShardSpec` MUST NOT contain:
- worker ID;
- device ID;
- container ID;
- network endpoint;
- scheduling information.

---

## 9. Hugging Face Adaptation

### 9.1 Adapter discovery

Load config without loading weights:

```python
config = AutoConfig.from_pretrained(
    model_path,
    trust_remote_code=False,
)
```

Resolve adapter using:
- `config.model_type`;
- `config.architectures`.

Example:

```text
llama  -> LlamaAdapter
qwen2  -> Qwen2Adapter
```

### 9.2 ModelAdapter responsibilities

`ModelAdapter` is responsible for:

- discovering model backbone;
- locating embeddings;
- locating the ordered Transformer block container;
- locating final norm;
- locating LM head;
- describing weight prefixes;
- creating a shard-compatible model skeleton;
- adapting canonical execution state to the model's native forward API;
- adapting output back to EdgeShard canonical state.

Suggested interface:

```python
class ModelAdapter(Protocol):
    def inspect(self, source: ModelSource) -> ModelLayout:
        ...

    def build_skeleton(
        self,
        source: ModelSource,
        shard: ShardSpec,
    ) -> nn.Module:
        ...

    def execute_prefill(
        self,
        module: nn.Module,
        state: ShardState,
        session: ShardSession,
    ) -> ShardState | LogitsOutput:
        ...

    def execute_decode(
        self,
        module: nn.Module,
        state: ShardState,
        session: ShardSession,
    ) -> ShardState | LogitsOutput:
        ...
```

Exact method signatures may evolve during 0A–0C, but architecture-specific behavior must remain confined here.

### 9.3 ModelLayout

Minimum fields:

```python
class ModelLayout(BaseModel):
    model_type: str

    num_blocks: int
    hidden_size: int
    intermediate_size: int

    num_attention_heads: int
    num_kv_heads: int

    embedding_prefix: str
    block_prefix_template: str
    final_norm_prefix: str
    lm_head_prefix: str

    tied_word_embeddings: bool
```

This layout becomes the stable static model description used later by profiling/cost modeling.

### 9.4 GenericAdapter

A conservative generic fallback is allowed.

It may support a model only if it can reliably identify:

- embedding;
- ordered Transformer `ModuleList`;
- final norm;
- LM head;
- shard-compatible execution semantics.

If uncertain:

```text
raise UnsupportedArchitectureError
```

Do not silently guess.

---

## 10. Model Skeleton Construction

The custom shard backend MUST reuse Hugging Face model implementations.

Preferred strategy:

```python
config = AutoConfig.from_pretrained(...)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
```

The meta-device skeleton provides the full module hierarchy without materializing full model weights.

Then the adapter reduces/reconfigures the hierarchy so only the target shard remains materially relevant.

The system MUST NOT reimplement model math merely to make sharding easier.

---

## 11. Selective Weight Loading

### 11.1 Formal runtime path

Formal runtime loading supports safetensors checkpoints.

Reference tests may use full `from_pretrained()` loading for comparison, but production shard construction may not.

### 11.2 Index-based selection

Read:

```text
model.safetensors.index.json
```

Map:

```text
tensor name -> checkpoint file
```

For shard `[8,20)` load only the matching layer prefixes.

Conceptually:

```text
model.layers.8.*
...
model.layers.19.*
```

Plus:

```text
first shard:
    embedding tensors

final shard:
    final norm tensors
    lm_head tensors
```

### 11.3 Tied embeddings

If the checkpoint ties embeddings and output weights, the final shard must still be independently loadable.

Do not depend on cross-process Python parameter aliasing.

If required, independently materialize the relevant checkpoint tensor on the final shard.

### 11.4 Weight loader interface

```python
class WeightLoader(Protocol):
    def load_shard(
        self,
        module: nn.Module,
        source: ModelSource,
        layout: ModelLayout,
        shard: ShardSpec,
    ) -> None:
        ...
```

---

## 12. Shard Runtime Objects

Three concepts must remain separate.

### 12.1 ShardSpec

Static description of what model part is executed.

### 12.2 ShardModule

Shared model weights and computation.

```text
ShardModule
├── optional InputStage
├── Transformer blocks [start,end)
└── optional OutputStage
```

### 12.3 ShardSession

Per-generation/request state.

Minimum conceptual state:

```python
@dataclass
class ShardSession:
    session_id: str
    kv_cache: object
    sequence_length: int
    step: int
```

The exact internal cache type may be a native HF cache implementation.

---

## 13. KV Cache Policy

A shard owns KV state for the blocks it owns.

Example:

```text
Shard A [0,8)
  -> KV for blocks 0..7

Shard B [8,20)
  -> KV for blocks 8..19

Shard C [20,32)
  -> KV for blocks 20..31
```

KV caches remain local.

Do not serialize or forward KV cache in the normal shard protocol.

Internal use of native Hugging Face cache objects is allowed.

---

## 14. Canonical Execution State

Wire semantics must not depend on a particular Transformers version.

Use a canonical state similar to:

```python
@dataclass
class ShardState:
    hidden_states: torch.Tensor
    context: ExecutionContext
```

```python
@dataclass(frozen=True)
class ExecutionContext:
    phase: InferencePhase
    step: int

    batch_size: int
    sequence_lengths: tuple[int, ...]

    past_length: int
    positions: torch.Tensor | None
```

Phase 0 execution support is formally limited to:

```text
batch_size = 1
```

but protocol structures should retain batch-related fields for future extension.

### 14.1 Attention mask policy

Do not transmit backend-specific attention masks.

Transmit canonical sequence/position metadata and let each adapter reconstruct whatever representation its backend expects.

Similarly, avoid transmitting model-specific RoPE caches if they can be regenerated locally from canonical position metadata.

---

## 15. Inference Semantics

### 15.1 Prefill

```text
input_ids
   |
   v
First Shard
  embedding
  local blocks
   |
   | hidden states
   v
Middle Shard(s)
   |
   v
Final Shard
  local blocks
  final norm
  lm head
   |
   v
logits
```

Each shard initializes/updates local KV state for its own blocks.

### 15.2 Decode

```text
next token
   |
   v
First Shard + local KV
   |
   v
Middle Shard + local KV
   |
   v
Final Shard + local KV
   |
   v
logits
```

### 15.3 Sampling

Sampling is not a `ShardModule` responsibility.

Phase 0 uses a `GenerationDriver` outside the shard runtime.

For correctness tests, implement deterministic greedy decoding first:

```text
next_token = argmax(logits)
```

Top-p, top-k, temperature, and beam search are out of Phase 0 scope.

---

## 16. Protocol Domain Model

Define three payload categories:

```text
TokenPayload
HiddenStatePayload
LogitsPayload
```

Conceptual message:

```python
@dataclass
class ShardMessage:
    header: ShardMessageHeader
    context: ExecutionContext
    payload: TokenPayload | HiddenStatePayload | LogitsPayload
```

Header minimum:

```python
@dataclass(frozen=True)
class ShardMessageHeader:
    protocol_version: int

    execution_id: str
    session_id: str
    request_id: str

    phase: InferencePhase
    step: int

    source_stage: int
    target_stage: int
```

### 16.1 Identity semantics

`execution_id`:
- identifies one concrete pipeline deployment.

`session_id`:
- identifies one generation session.

`request_id`:
- identifies one API-level request/action.

These IDs are not interchangeable.

### 16.2 Sequencing

Expected:

```text
PREFILL step=0
DECODE  step=1
DECODE  step=2
...
```

A shard runtime MUST reject:

- out-of-order steps;
- duplicate invalid steps;
- wrong execution ID;
- wrong source/target stage;
- messages for a closed session.

Use explicit protocol errors instead of silent recovery.

---

## 17. gRPC Protocol

### 17.1 Service

Initial service shape:

```protobuf
service ShardRuntime {
  rpc GetRuntimeInfo(RuntimeInfoRequest)
      returns (RuntimeInfoReply);

  rpc CreateSession(CreateSessionRequest)
      returns (CreateSessionReply);

  rpc CloseSession(CloseSessionRequest)
      returns (CloseSessionReply);

  rpc Prefill(ForwardRequest)
      returns (ForwardReply);

  rpc Decode(ForwardRequest)
      returns (ForwardReply);
}
```

Use standard gRPC health checking where possible rather than inventing an additional custom health protocol.

### 17.2 Domain/wire separation

Protobuf generated objects are wire DTOs only.

They are not the inference domain model.

Use explicit mapping:

```text
dataclass/Pydantic domain model
        |
        v
protobuf mapper
        |
        v
wire
```

### 17.3 Tensor payload

Protobuf contains tensor bytes, not element-by-element repeated fields.

Conceptually:

```protobuf
message ForwardRequest {
  MessageHeader header = 1;
  ExecutionContext context = 2;
  bytes tensors = 3;
}
```

Encode tensor bundle using safetensors.

### 17.4 Message size

Explicitly configure gRPC send/receive message limits above the default, e.g. a 512 MiB Phase 0 ceiling.

This is a correctness baseline only.

Do not optimize large activation transmission in Phase 0.

---

## 18. Runtime-to-runtime Forwarding

The Mock Master MUST NOT relay hidden states between shards.

Expected pipeline:

```text
Mock Master
     |
     | token/prefill request
     v
Shard A
     |
     | hidden states
     v
Shard B
     |
     | hidden states
     v
Shard C
     |
     | final reply
     v
Mock Master
```

Each shard runtime knows:

- `stage_index`;
- `stage_count`;
- `next_endpoint` where applicable.

The first runtime is the external entry point for the EdgeShard custom pipeline.

### 18.1 Session chaining

`CreateSession` should propagate through the pipeline:

```text
Master -> A -> B -> C
```

Each runtime creates its own local KV state.

`CloseSession` similarly propagates downstream.

---

## 19. Runtime Configuration

### 19.1 EdgeShard shard runtime

Example:

```yaml
runtime:
  backend: edgeshard_shard
  runtime_id: shard-1
  execution_id: exec-001

model:
  id: Qwen/Qwen2.5-7B
  path: /models/qwen2.5-7b

shard:
  start_block: 8
  end_block: 20
  include_input_stage: false
  include_output_stage: false

pipeline:
  stage_index: 1
  stage_count: 3
  next_endpoint: shard-2:50051

device:
  type: cuda
  index: 0

inference:
  dtype: bf16

server:
  listen_host: 0.0.0.0
  listen_port: 50051
```

### 19.2 vLLM runtime

Example deployment-side spec:

```yaml
runtime:
  backend: vllm
  runtime_id: vllm-0

model:
  path: /models/qwen2.5-7b

device:
  type: cuda
  index: 0

vllm:
  max_model_len: 8192
  tensor_parallel_size: 1
```

`VLLMRuntimeDriver` translates this into official vLLM container arguments.

---

## 20. RuntimeDriver Abstraction

Lifecycle normalization belongs on the control side.

```python
class RuntimeDriver(Protocol):
    async def start(self, spec: RuntimeSpec) -> RuntimeHandle:
        ...

    async def wait_ready(self, handle: RuntimeHandle) -> None:
        ...

    async def info(self, handle: RuntimeHandle) -> RuntimeInfo:
        ...

    async def stop(self, handle: RuntimeHandle) -> None:
        ...
```

Implement:

```text
EdgeShardShardRuntimeDriver
VLLMRuntimeDriver
```

### 20.1 EdgeShardShardRuntimeDriver

Uses:
- Docker SDK;
- EdgeShard runtime image;
- gRPC health/runtime service.

### 20.2 VLLMRuntimeDriver

Uses:
- official vLLM image;
- Docker SDK;
- vLLM native CLI;
- HTTP/OpenAI-compatible readiness/API.

Do not modify the vLLM image just to make it look like an EdgeShard shard runtime.

Normalize lifecycle, not native inference protocol.

---

## 21. Container Strategy

### 21.1 Models are not baked into images

Model artifacts live on the host model cache:

```text
$EDGESHARD_MODEL_CACHE/
```

Mount read-only in containers:

```text
/models:ro
```

Runtime containers must not download models themselves in Phase 0.

### 21.2 CPU correctness image

Use a small Python 3.12 base image and CPU PyTorch.

Purpose:
- CI;
- deterministic correctness tests;
- multi-container pipeline validation without GPU dependency.

### 21.3 x86 NVIDIA shard image

Use a pinned NVIDIA-compatible PyTorch/CUDA base image.

Do not build the CUDA/PyTorch stack manually from Ubuntu unless necessary.

### 21.4 Jetson shard image

Maintain a distinct Jetson/L4T-compatible Dockerfile.

Do not assume x86 CUDA images are portable to Jetson.

Host retains:
- Linux kernel;
- NVIDIA driver;
- JetPack/L4T where applicable;
- Docker;
- NVIDIA Container Toolkit.

Application dependencies remain inside runtime container.

### 21.5 Container labels

Managed runtime containers should include labels such as:

```text
io.edgeshard.managed=true
io.edgeshard.execution_id=<id>
io.edgeshard.runtime_id=<id>
io.edgeshard.backend=<backend>
```

This allows cleanup of orphaned Phase 0 containers.

---

## 22. Mock Master

The Mock Master is a thin integration orchestrator.

### 22.1 Responsibilities

It may:

- parse `DeploymentManifest`;
- validate static topology;
- create Docker network;
- generate runtime configs;
- launch runtime containers;
- wait for readiness;
- create sessions;
- invoke the first shard;
- perform deterministic generation;
- invoke vLLM test requests;
- close sessions;
- shut down runtimes;
- clean up networks/configs.

It may not:

- discover workers;
- profile hardware;
- select devices;
- calculate a partition;
- perform optimization;
- persist cluster state;
- implement production retry/recovery.

### 22.2 DeploymentManifest

Example:

```yaml
execution_id: exec-001

model:
  id: tiny-qwen
  path: /models/tiny-qwen

runtimes:
  - id: shard-0
    backend: edgeshard_shard
    shard:
      start: 0
      end: 2
      include_input_stage: true

  - id: shard-1
    backend: edgeshard_shard
    shard:
      start: 2
      end: 4
      include_output_stage: true

pipeline:
  - shard-0
  - shard-1
```

The partition is already provided.

Future relationship:

```text
Scheduler
    |
    v
DeploymentPlan
    |
    | compile
    v
DeploymentManifest
    |
    v
Controller/Worker Runtime Layer
```

Phase 0 implements only the lower part.

---

## 23. Docker Networking for Mock Master

For each execution, create a dedicated bridge network:

```text
edgeshard-exec-<execution_id>
```

Use runtime IDs as container aliases:

```text
shard-0
shard-1
shard-2
```

Example downstream endpoint:

```text
shard-1:50051
```

The Mock Master only publishes the externally accessed entry runtime port to the host as required.

Do not hard-code host IP addresses into shard configs.

---

## 24. Model Cache Policy

Phase 0 model cache is prepared externally.

Optional utility:

```python
huggingface_hub.snapshot_download(
    repo_id=...,
    revision=...,
)
```

Prefer resolving revisions to immutable commit SHAs for reproducibility.

Runtime receives a local mounted path and treats it as read-only.

Model download/cache ownership moves to Worker in a later phase.

---

## 25. Test Strategy

Phase 0 correctness is more important than performance.

### 25.1 No internet dependency in basic CI

Tests must generate tiny local Hugging Face models.

Example:

```python
config = LlamaConfig(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
)
model = LlamaForCausalLM(config)
model.save_pretrained(
    tmp_path,
    safe_serialization=True,
)
```

Create equivalent tiny Qwen2 fixtures.

This tests the complete local safetensors workflow without external downloads.

### 25.2 Correctness levels

#### L1 — Adapter/block correctness

Compare native HF execution against adapter-driven execution for the same block/state.

#### L2 — Shard correctness

Compare:
- native HF layer slice;
- EdgeShard `ShardModule`.

#### L3 — Local pipeline correctness

Compare:
- full native HF model;
- local multi-shard EdgeShard pipeline.

#### L4 — Protocol correctness

Compare:
- direct canonical state;
- encode/decode roundtrip;
- gRPC transfer.

#### L5 — Container E2E

Compare:
- native HF reference;
- Mock-Master-driven multi-container shard pipeline.

### 25.3 Mandatory partition cases

For a 4-layer tiny model:

```text
[0,4)

[0,2) + [2,4)

[0,1) + [1,3) + [3,4)
```

For larger fixtures, include uneven partitions.

### 25.4 Mandatory inference tests

- prefill logits;
- multiple decode steps;
- deterministic greedy generated token sequence;
- session creation and cleanup;
- independent KV state across two sessions;
- protocol sequence rejection;
- incorrect execution ID rejection;
- container restart/cleanup at the integration-test level.

### 25.5 Numerical comparison

Use dtype-appropriate tolerances.

Do not require bit-for-bit equivalence across different GPU kernels where normal numerical drift is expected.

For deterministic CPU FP32 reference tests, use stricter tolerances.

---

## 26. CI Tiers

### Tier 1 — CPU, every PR

Required:

- adapters;
- selective loading;
- shard execution;
- KV/session tests;
- protobuf mapping;
- tensor codec;
- gRPC tests;
- CPU multi-container pipeline using tiny models;
- lint/type tests.

### Tier 2 — x86 NVIDIA GPU

Run on merge/nightly or available GPU runner:

- CUDA custom shard runtime;
- FP16/BF16;
- host vs container equivalence;
- real small HF model smoke tests;
- vLLM RuntimeDriver/container integration.

### Tier 3 — Jetson

Run on dedicated AGX Orin/Orin NX runner:

- Jetson container build/run;
- GPU access;
- selective loading;
- prefill/decode;
- basic multi-runtime compatibility.

---

## 27. Phase 0 Milestones

Development MUST proceed in this order.

### 0A — Model Adaptation

Deliver:
- `ModelSource`;
- `ModelLayout`;
- `ModelAdapter`;
- registry;
- Llama adapter;
- Qwen2 adapter.

Gate:
- structure inspection tests pass;
- model detection is automatic;
- no full weight loading required.

### 0B — Selective Weight Loading

Deliver:
- safetensors index reader;
- prefix selection;
- shard-only loading;
- tied embedding handling.

Gate:
- target shard can materialize without loading unrelated layer tensors;
- tests verify loaded tensor set.

### 0C — Local Shard Inference

Deliver:
- `ShardModule`;
- `ShardSession`;
- prefill;
- decode;
- local KV ownership.

Gate:
- one shard behaves like equivalent native HF slice;
- multi-step decode works.

### 0D — Canonical Shard Protocol

Deliver:
- canonical state;
- protocol IDs;
- sequencing validation;
- protobuf schema;
- tensor codec.

Gate:
- roundtrip preserves inference semantics;
- invalid sequencing is rejected.

### 0E — Local/Loopback Pipeline

Deliver:
- local pipeline driver;
- serialized boundary between stages.

Gate:
- multi-shard pipeline matches full HF reference.

### 0F — EdgeShard Runtime Server

Deliver:
- runtime config;
- gRPC shard server;
- runtime info;
- session APIs;
- downstream forwarding.

Gate:
- multiple local processes form a correct pipeline.

### 0G — Containerized Shard Runtime

Deliver:
- CPU Dockerfile;
- x86 NVIDIA Dockerfile;
- Jetson Dockerfile scaffold/implementation;
- Docker runtime driver.

Gate:
- host/container inference equivalence;
- runtime is reproducible from pinned image/config.

### 0H — Mock Master

Deliver:
- deployment manifest;
- Docker network management;
- runtime startup/shutdown;
- generation driver;
- cleanup.

Gate:
- manifest-driven deployment works without manual Docker commands.

### 0I — Multi-container Shard Pipeline

Deliver:
- complete E2E pipeline via Mock Master.

Gate:
- container pipeline matches HF reference;
- partition changes preserve semantics.

This is the primary Phase 0 gate for the EdgeShard custom backend.

### 0J — vLLM Runtime Integration

Deliver:
- `VLLMRuntimeDriver`;
- official vLLM image launch;
- readiness;
- OpenAI-compatible test request;
- shutdown/cleanup.

Gate:
- Mock Master can manage both EdgeShard shard runtimes and an independent vLLM runtime without backend-specific logic leaking into generic lifecycle code.

### 0K — Regression and Freeze

Deliver:
- complete automated suite;
- `ARCHITECTURE.md`;
- examples;
- known limitations;
- stable Phase 0 public interfaces.

Gate:
- all Tier 1 tests pass;
- available Tier 2/Tier 3 tests pass;
- no Phase 1+ functionality has leaked into Phase 0 modules.

---

## 28. Definition of Done

Phase 0 is complete only when all of the following are true.

### Model

- Llama and Qwen2-family adapters work.
- Architecture selection is automatic.
- Model layout is stable and test-covered.

### Loading

- A shard is materialized without loading the complete checkpoint.
- safetensors is the supported formal runtime format.
- tied embeddings are handled correctly.

### Inference

- prefill works;
- decode works;
- multiple sequential tokens work;
- KV cache remains local to each shard;
- independent sessions do not corrupt each other.

### Protocol

- no HF runtime object crosses the shard boundary;
- canonical state is sufficient for complete inference;
- serialization preserves semantics;
- invalid sequencing is rejected.

### Runtime

- standalone runtime server works;
- runtime-to-runtime forwarding works;
- RuntimeDriver lifecycle works;
- container environment is reproducible.

### Containers

- CPU image works;
- x86 NVIDIA image works when GPU runner is available;
- Jetson image is validated on target JetPack/L4T environment;
- weights are mounted, not baked into image.

### Mock Master

- deployment is manifest-driven;
- no scheduler exists;
- no profiling exists;
- the full pipeline can be created and cleaned up automatically.

### Equivalence

For deterministic tests:

```text
HF Reference
    ≈
Local EdgeShard Pipeline
    ≈
Serialized/gRPC Pipeline
    ≈
Multi-container Pipeline
```

---

## 29. Rules for Claude Code

When implementing this specification:

1. **Work milestone-by-milestone.**
   Do not begin milestone N+1 until milestone N tests pass.

2. **Do not implement future phases opportunistically.**
   No scheduling, profiling, Worker Agent, or production Master code in Phase 0.

3. **Prefer upstream model behavior over reimplementation.**
   Use native HF model logic wherever possible.

4. **Keep architecture-specific logic inside adapters.**

5. **Tests are part of each milestone, not a final cleanup task.**

6. **Use tiny locally generated models for default tests.**

7. **Do not introduce dependency-heavy frameworks without a demonstrated need.**

8. **Do not optimize the data plane before correctness is proven.**

9. **Preserve domain/wire/container boundaries.**

10. **When an interface must change, update the tests and `ARCHITECTURE.md` in the same change.**

11. **Avoid compatibility hacks that silently guess model behavior.**
    Unsupported models should fail explicitly.

12. **Keep changes small and reviewable.**
    Each milestone should be decomposed into narrow implementation tasks.

---

## 30. Initial Claude Code Task Sequence

Claude Code should begin with the following concrete tasks.

### Task 1 — Bootstrap

Create:
- `pyproject.toml`;
- uv dependency groups;
- package layout;
- Ruff config;
- mypy config;
- pytest config;
- initial `ARCHITECTURE.md`.

Acceptance:

```bash
uv sync
uv run ruff check .
uv run mypy src/edgeshard
uv run pytest
```

all succeed on the empty/minimal project.

### Task 2 — Domain primitives for 0A

Implement only types required by model adaptation:
- `ModelSource`;
- `BlockRange`;
- `ShardSpec`;
- `ModelLayout`;
- model/adaptation errors.

Add tests.

### Task 3 — Adapter registry

Implement:
- adapter protocol/base;
- registry;
- automatic config-based resolution.

Add tiny Llama and Qwen2 tests.

### Task 4 — Llama adapter

Implement:
- structure discovery;
- layout generation;
- meta skeleton creation.

Test against generated tiny Llama model.

### Task 5 — Qwen2 adapter

Repeat for Qwen2.

### Task 6 — Start 0B

Implement safetensors checkpoint index inspection and determine the exact tensors required for a given `ShardSpec`.

Do not yet proceed to gRPC, containers, or Mock Master.

---

## 31. Phase 0 Design Summary

The stable lower-level model should be:

```text
HF Model Repository
       |
       v
ModelAdapter
       |
       v
ModelLayout + ShardSpec
       |
       v
Selective Weight Loader
       |
       v
ShardModule + ShardSession
       |
       v
Canonical Shard Protocol
       |
       v
Shard Runtime Server
       |
       v
Runtime Container
       |
       v
RuntimeDriver
       |
       v
Mock Master
```

The entire value of Phase 0 is that everything above the runtime can later change without forcing model execution to change, while everything below the Mock Master has already been proven correct in real process/container boundaries.

Once Phase 0 is frozen, later phases should consume these contracts rather than redefine them.
