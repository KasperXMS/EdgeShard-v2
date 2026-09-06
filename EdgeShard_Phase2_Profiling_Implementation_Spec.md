# EdgeShard v2 — Phase 2 Profiling Implementation Specification

**Status:** Implementation Baseline  
**Target:** Claude Code / engineering implementation  
**Scope:** Phase 2 — Extensible Profiling & Workload Characterization  
**Repository:** EdgeShard v2

---

## 0. Purpose

Phase 2 adds a reusable profiling and workload-characterization subsystem on top of the frozen Phase 0/Phase 1 foundation.

The goal is **not** to build a complete performance predictor or scheduler. The goal is to produce reliable, low-cost, versioned empirical measurements that can later be consumed by Phase 3 resource/performance models.

The central separation is:

```text
Phase 1  ClusterSnapshot
         What resources exist?

Phase 2  ProfileSnapshot
         What has been empirically observed?

Phase 3  ResourceEstimateSnapshot
         What is expected to happen?

Phase 4  Scheduler
         Where should the workload run?
```

Phase 2 MUST NOT contain placement decisions, interpolation, learned prediction, scheduling, or deployment logic.

---

# 1. Design Principles

## 1.1 Facts remain separate from estimates and decisions

Phase 2 records only:

- static model/workload characterization;
- actual runtime measurements;
- measurement environment/provenance;
- profiling experiment lifecycle.

Phase 2 MUST NOT store fields such as:

- `predicted_latency`;
- `estimated_memory`;
- `placement_score`;
- `recommended_worker`;
- `expected_transfer_time`.

Those belong to Phase 3 or later.

## 1.2 Profiling is extensible, not hard-coded to one method

The profiling subsystem is a framework, not a single profiling algorithm.

The v1 reference strategy is:

```text
Model:
    static model characterization
    + operator microprofiling per performance class
    + sparse real Module measurements
    + sparse real TransformerLayer measurements

Network:
    endpoint characterization
    + dense cheap RTT probes
    + sparse PathClass bandwidth probes
    + optional explicit Pair calibration
```

Future strategies may include:

- operator-only profiling;
- exhaustive layer profiling;
- multi-fidelity profiling;
- adaptive profiling;
- active profiling;
- profiling-under-budget;
- learned sampling policies.

The core domain, storage, orchestration, and instrumentation layers MUST remain reusable when strategies are replaced.

## 1.3 Real model execution is used for high-fidelity measurements

For `TransformerLayer` and `Module` profiling:

- use the actual model checkpoint;
- use the actual model implementation;
- use the same backend semantics as production;
- use shape-correct synthetic inputs;
- invoke the real layer/module directly;
- use CUDA events for GPU timing.

Do NOT rebuild a synthetic Transformer for the default implementation.

Do NOT use Python `forward_hook` timing as the default benchmark mechanism.

## 1.4 Operator profiling is used for low-cost coverage

Operator profiling MUST:

- avoid loading the full model checkpoint when possible;
- operate on normalized operator signatures;
- benchmark only required/deduplicated tensor shapes;
- reuse measurements across models;
- reuse measurements across physical devices belonging to the same compatible performance class.

The scheduler may eventually require a complete model/device cost matrix, but Phase 2 MUST NOT obtain that matrix through exhaustive real-model profiling.

## 1.5 Network profiling follows the same low-cost/high-fidelity hierarchy

Network profiling uses three levels:

```text
Endpoint
    ↓
PathClass
    ↓
Pair
```

- Endpoint characterization is cheap and per-worker.
- RTT pair probing is cheap enough to be dense.
- Bandwidth profiling is expensive and MUST be sparse.
- Pair-specific bandwidth measurements are calibration/validation facts, not mandatory coverage.

---

# 2. Existing Phase 1 Contracts That Must Be Preserved

Phase 1 is treated as frozen.

The implementation MUST preserve the existing architecture:

```text
Master
    ↓ gRPC
Worker Agent
    ↓
local authority
```

The Master MUST NOT:

- SSH into workers;
- call remote Docker daemons directly;
- probe GPU state directly;
- bypass Worker Agent for profiling.

Reuse existing Phase 1 concepts where applicable:

- `WorkerIdentity`;
- `DeviceIdentity`;
- `MemoryPool`;
- `Capability`;
- `WorkerState`;
- `RuntimeInventory`;
- `ClusterSnapshot`;
- `capability_revision`;
- NVML telemetry;
- Jetson tegrastats telemetry;
- worker/master session semantics.

Profiling MUST be added alongside the existing control plane, not by replacing it.

---

# 3. Top-Level Phase 2 Architecture

```text
                         Master
                           │
                 ProfilingController
                           │
                          gRPC
                           │
                    Worker Agent
                           │
                   ProfilingRunner
                   /             \
                  /               \
        Model Profiling       Network Profiling
             │                      │
      ┌──────┼──────┐        ┌─────┼─────┐
      │      │      │        │     │     │
 Operator Module Transformer Endpoint Path Pair
               Layer                  Class
      │      │      │        │     │     │
      └──────┴──────┴────────┴─────┴─────┘
                           │
                    MeasurementRecord
                           │
                        SQLite
                           │
                     ProfileStore
                           │
                    ProfileSnapshot
                           │
                        Phase 3
```

---

# 4. Proposed Package Layout

```text
src/edgeshard/
│
├── profiling/
│   ├── domain/
│   │   ├── experiment.py
│   │   ├── measurement.py
│   │   ├── signature.py
│   │   ├── model.py
│   │   ├── network.py
│   │   └── snapshot.py
│   │
│   ├── instrumentation/
│   │   ├── timing.py
│   │   ├── memory.py
│   │   └── telemetry.py
│   │
│   ├── benchmark/
│   │   ├── harness.py
│   │   └── sampling.py
│   │
│   ├── model/
│   │   ├── adapters/
│   │   │   ├── base.py
│   │   │   ├── qwen.py
│   │   │   └── llama.py
│   │   ├── characterization.py
│   │   ├── layer_profiler.py
│   │   └── module_profiler.py
│   │
│   ├── operator/
│   │   ├── extractor.py
│   │   ├── normalizer.py
│   │   ├── registry.py
│   │   ├── workloads.py
│   │   └── profiler.py
│   │
│   ├── network/
│   │   ├── classifier.py
│   │   ├── ping.py
│   │   ├── iperf.py
│   │   └── profiler.py
│   │
│   ├── strategy/
│   │   ├── base.py
│   │   ├── default.py
│   │   └── sampling.py
│   │
│   ├── runner/
│   │   ├── runner.py
│   │   ├── session.py
│   │   └── lease.py
│   │
│   └── store/
│       ├── base.py
│       └── sqlite.py
│
├── control/
│   ├── master/
│   │   └── profiling_controller.py
│   └── worker/
│       └── profiling_runner.py
│
└── protocol/
    └── profiling/
        ├── profiling.proto
        ├── mapper.py
        ├── client.py
        └── service.py
```

Keep domain modules free from PyTorch, gRPC, Docker, NVML, tegrastats, and SQLite dependencies wherever possible.

---

# 5. Core Domain Types

## 5.1 Profiling granularity

```python
from enum import StrEnum

class ProfilingGranularity(StrEnum):
    TRANSFORMER_LAYER = "transformer_layer"
    MODULE = "module"
    OPERATOR = "operator"
```

These three levels are independent empirical fidelities.

They MUST NOT imply:

```text
TransformerLayer = sum(Module)
Module = sum(Operator)
```

Composition belongs to Phase 3.

## 5.2 Inference phase

```python
class InferencePhase(StrEnum):
    PREFILL = "prefill"
    DECODE = "decode"
```

P2 v1 MUST fully support `PREFILL`.

`DECODE` MUST exist in the schema but may remain unsupported by specific adapters until the existing runtime exposes stable KV-cache semantics.

Unsupported decode cases MUST fail explicitly, not silently fall back to prefill.

## 5.3 Module kind

Recommended normalized vocabulary:

```python
class ModuleKind(StrEnum):
    ATTENTION = "attention"
    MLP = "mlp"
    NORM = "norm"
    PROJECTION = "projection"
    EMBEDDING = "embedding"
    LM_HEAD = "lm_head"
    VISION_ENCODER = "vision_encoder"
    PROJECTOR = "projector"
    OTHER = "other"
```

This is a profiling-domain abstraction and MUST NOT mirror every `torch.nn.Module` class.

## 5.4 Operator kind

Recommended stable EdgeShard vocabulary:

```python
class OperatorKind(StrEnum):
    GEMM = "gemm"
    ATTENTION = "attention"
    NORM = "norm"
    ROTARY = "rotary"
    ELEMENTWISE = "elementwise"
    REDUCTION = "reduction"
    EMBEDDING = "embedding"
    KV_COPY = "kv_copy"
    CUSTOM = "custom"
```

ATen operators are extractor output, not long-term domain identities.

Unknown operators MUST be preserved as `CUSTOM` with raw metadata.

---

# 6. Signatures

All reusable workload identities MUST be represented by canonical signatures.

## 6.1 TransformerLayerSignature

Suggested fields:

```python
@dataclass(frozen=True)
class TransformerLayerSignature:
    architecture_family: str
    layer_type: str
    hidden_size: int
    intermediate_size: int | None
    num_attention_heads: int
    num_kv_heads: int | None
    head_dim: int | None
    dtype: str
    quantization: str | None
    special_role: str | None = None
```

Do not include physical `worker_id` or `device_id` in this signature.

## 6.2 ModuleSignature

Suggested fields:

```python
@dataclass(frozen=True)
class ModuleSignature:
    kind: ModuleKind
    architecture_family: str
    structural_parameters: Mapping[str, int | float | str | bool]
    dtype: str
    quantization: str | None
```

Examples:

```text
ATTENTION
hidden_size=3584
num_heads=28
num_kv_heads=4
head_dim=128
```

## 6.3 OperatorSignature

Use typed parameter objects where useful.

Example GEMM:

```python
@dataclass(frozen=True)
class GemmSignature:
    m: int
    n: int
    k: int
    dtype: str
    transpose_a: bool = False
    transpose_b: bool = False
```

Example attention:

```python
@dataclass(frozen=True)
class AttentionSignature:
    batch_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    q_len: int
    kv_len: int
    dtype: str
    phase: InferencePhase
```

An `OperatorSignature` wrapper may contain:

```python
kind: OperatorKind
parameters: OperatorParameterUnion
backend_family: str
```

---

# 7. Canonical Identity and Hashing

Reuse the Phase 1 approach:

```text
canonical representation
        ↓
SHA-256
```

Use this for:

- `model_signature_id`;
- `transformer_layer_signature_id`;
- `module_signature_id`;
- `operator_signature_id`;
- `network_pair_signature_id`;
- `environment_fingerprint`;
- `profiling_case_id`.

Requirements:

1. semantically identical values MUST produce identical hashes;
2. map ordering MUST NOT affect hashes;
3. enum serialization MUST be stable;
4. hashes MUST NOT include volatile fields such as timestamps;
5. hash functions MUST have explicit unit tests.

Prefer canonical JSON or the repository's existing canonical serialization utility if one already exists.

Do not introduce a second incompatible canonicalization mechanism if Phase 1 already has one suitable for reuse.

---

# 8. Experiments, Cases, and Measurements

## 8.1 ProfilingExperiment

Represents a logical profiling job.

Suggested fields:

```python
@dataclass(frozen=True)
class ProfilingExperiment:
    experiment_id: str
    strategy_id: str
    created_at: datetime
    requested_by: str | None
    case_ids: tuple[str, ...]
```

Runtime lifecycle state SHOULD live separately from the immutable definition if that matches existing Phase 1 patterns.

## 8.2 ProfilingCase

Represents exactly one benchmark configuration.

Model example:

```text
granularity = transformer_layer
model = qwen2.5-7b
model_revision = ...
layer_signature = ...
worker_id = ...
device_ids = (...)
backend = torch
dtype = bf16
phase = prefill
batch_size = 1
sequence_length = 512
```

Network example:

```text
source_worker = worker-a
destination_worker = worker-b
probe_kind = bandwidth
transport = tcp
direction = forward
duration = 3s
```

A case is a request for measurement, not a result.

## 8.3 MeasurementRecord

Represents actual observations only.

Suggested structure:

```python
@dataclass(frozen=True)
class MeasurementRecord:
    measurement_id: str
    case_id: str
    environment_fingerprint: str

    started_at: datetime
    finished_at: datetime

    sample_count: int
    samples: tuple[float, ...] | None

    mean: float
    median: float
    stddev: float
    minimum: float
    maximum: float
    p95: float | None

    metadata: Mapping[str, JsonScalar]
```

Memory/network-specific fields SHOULD be represented through typed metric objects rather than by adding many nullable top-level fields.

Prefer:

```text
MeasurementRecord
    metrics:
        latency
        allocator_memory
        physical_memory
        telemetry
```

over one giant flat record.

---

# 9. Environment Fingerprint

Each measurement MUST record enough context to judge reuse later.

Minimum fields:

```text
worker_id              provenance only
device_id              provenance only
device_performance_class
capability_revision

torch_version
cuda_version
driver_version
backend
backend_revision

model_revision
dtype
quantization

profiling_implementation_revision
```

The fingerprint SHOULD exclude volatile telemetry such as current temperature.

Volatile telemetry belongs to measurement context, not compatibility identity.

---

# 10. DevicePerformanceClass

Introduce a reusable performance compatibility abstraction separate from physical device identity.

Purpose:

```text
physical device identity
    !=
reusable performance class
```

Example:

```text
RTX4090 GPU UUID A
RTX4090 GPU UUID B
RTX4090 GPU UUID C
        ↓
compatible performance class
```

Suggested compatibility dimensions:

- vendor;
- accelerator model;
- architecture / compute capability;
- memory model;
- backend family;
- dtype;
- relevant software/runtime major versions.

Do NOT overfit the initial class key.

Keep a verification benchmark mechanism so that a new physical device may reuse an existing class after passing a small sanity suite.

---

# 11. P2B — Instrumentation

## 11.1 CUDA timing

Primary GPU timing mechanism:

```python
torch.cuda.Event(enable_timing=True)
```

Implement:

```python
class CudaEventTimer:
    ...
```

The timer MUST:

1. create start/end events;
2. record both on the relevant stream;
3. synchronize before reading elapsed time;
4. return device elapsed milliseconds;
5. never use host wall-clock time as the primary GPU latency.

Also provide:

```python
class WallClockTimer:
    ...
```

using `time.perf_counter_ns()` for CPU/network/control-path measurements and diagnostics.

---

# 12. BenchmarkHarness

Create one reusable benchmark lifecycle implementation.

Suggested interface:

```python
class BenchmarkWorkload(Protocol):
    def prepare(self) -> None: ...
    def run_once(self) -> None: ...
    def reset(self) -> None: ...
    def cleanup(self) -> None: ...


class BenchmarkHarness:
    def run(
        self,
        workload: BenchmarkWorkload,
        *,
        sampling_policy: SamplingPolicy,
        instrumentation: InstrumentationBundle,
    ) -> BenchmarkResult:
        ...
```

Lifecycle:

```text
prepare
↓
validate environment
↓
warmup
↓
reset memory peaks/counters
↓
measurement loop
↓
telemetry collection
↓
summary
↓
cleanup
```

Default v1 policy:

```text
minimum warmups: 3
minimum measured runs: 5
maximum measured runs: 20
target accumulated measurement duration: ~1 second
```

Do not hard-code "100 repetitions".

---

# 13. Sampling Policy

Create a replaceable policy interface.

Example:

```python
class SamplingPolicy(Protocol):
    def should_continue(self, state: SamplingState) -> bool:
        ...
```

Default policy can be duration-based with minimum/maximum sample counts.

This is an extension point for future:

- convergence-based sampling;
- confidence-interval stopping;
- profiling-budget algorithms.

Do NOT implement advanced policies in Phase 2 v1.

---

# 14. Memory Instrumentation

## 14.1 PyTorch allocator view

Capture:

```text
memory_allocated
memory_reserved
max_memory_allocated
max_memory_reserved
```

Reset peak memory statistics before the measurement interval.

Recommended metrics:

```text
allocator_allocated_before
allocator_reserved_before
allocator_peak_allocated
allocator_peak_reserved
allocator_allocated_after
allocator_reserved_after
```

## 14.2 Physical MemoryPool view

Reuse Phase 1 `MemoryPool`.

For RTX:

```text
physical VRAM pool
```

For Jetson:

```text
shared system-memory pool
```

Record:

```text
pool_id
used_before
used_peak
used_after
```

Do not create a new "GPU memory" domain model for Phase 2.

---

# 15. Telemetry Instrumentation

Reuse existing Phase 1:

- NVML on discrete NVIDIA GPUs;
- tegrastats on Jetson.

Capture only contextual observations in v1:

- GPU utilization;
- temperature;
- power;
- frequency/clock if already available;
- memory pressure.

Do NOT use telemetry to correct latency in Phase 2.

Provide configurable contamination checks, for example:

```text
if initial GPU utilization > threshold:
    reject or mark contaminated
```

Do not silently accept obviously busy devices as clean benchmark runs.

---

# 16. P2C — Model Characterization

## 16.1 ModelProfilingAdapter

Different model families have different layer signatures and call conventions.

Create:

```python
class ModelProfilingAdapter(Protocol):
    def supports(self, model_metadata: ...) -> bool: ...
    def characterize(self, model: ...) -> ModelCharacterization: ...
    def enumerate_transformer_layers(self, model: ...) -> Sequence[...]: ...
    def enumerate_profile_modules(self, layer: ...) -> Sequence[...]: ...
    def build_layer_inputs(self, case: ...) -> Mapping[str, Any]: ...
    def build_module_inputs(self, case: ...) -> Mapping[str, Any]: ...
```

Phase 2 v1 should support only model families already relevant to the repository.

Expected initial adapters:

```text
Qwen family
Llama family
```

Add LLaVA/VLM-specific stages only after the core language-model path works.

Unknown models MUST fail with an explicit unsupported-adapter error.

---

# 17. ModelCharacterization

Static characterization MUST be possible without performance benchmarking.

Capture:

- architecture family;
- layer count;
- hidden size;
- intermediate size;
- attention heads;
- KV heads;
- head dimension;
- dtype;
- quantization;
- major stage structure.

Example stage graph:

```text
Embedding
    ↓
Repeated TransformerLayer Group
    ↓
Final Norm
    ↓
LM Head
```

Future VLM example:

```text
Vision Encoder
    ↓
Projector
    ↓
Language Transformer Layers
    ↓
LM Head
```

Do not assume every model is composed only of homogeneous decoder blocks.

---

# 18. Operator Extraction

## 18.1 Primary extractor

Use `torch.export` where supported.

Create:

```python
class OperatorExtractor(Protocol):
    def extract(
        self,
        target: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> RawOperatorGraph:
        ...
```

Primary implementation:

```text
TorchExportExtractor
```

The raw graph SHOULD preserve:

- operator identity;
- tensor shapes;
- dtype;
- graph ordering/dependencies where available.

## 18.2 Fallback extractor

Implement:

```text
TorchProfilerExtractor
```

using representative execution with shape recording when export is unavailable.

The fallback is for structural discovery, not benchmark timing.

Do not use profiler timing as the authoritative Layer/Module latency.

---

# 19. Operator Normalization

Implement:

```python
class OperatorNormalizer:
    def normalize(self, raw_graph: RawOperatorGraph) -> NormalizedOperatorGraph:
        ...
```

Examples:

```text
aten.mm
aten.addmm
aten.linear
    ↓
GEMM
```

Attention implementations:

```text
scaled_dot_product_attention
supported fused/custom attention form
    ↓
ATTENTION
```

Requirements:

- mapping is explicit and testable;
- unknown operations are preserved as `CUSTOM`;
- no operation may disappear silently;
- normalization logic must not depend on specific model names when avoidable.

---

# 20. Operator Signature Deduplication

After model characterization:

```text
operator occurrences from all models
        ↓
canonical signatures
        ↓
deduplicate
        ↓
unique workload signatures
```

Only unique missing signatures are sent to microprofiling.

This is the main mechanism that prevents model/device profiling from becoming a Cartesian product.

---

# 21. P2D — Direct TransformerLayer Profiling

Implement:

```text
TransformerLayerProfiler
```

Behavior:

1. load actual model checkpoint through existing model infrastructure;
2. select an actual Transformer layer;
3. build shape-correct inputs through the adapter;
4. run warmups;
5. benchmark via `BenchmarkHarness`;
6. capture CUDA timing and memory;
7. persist empirical measurement;
8. clean up safely.

Default v1 workload dimensions:

```text
batch_size = 1

prefill sequence lengths:
    128
    512
    2048
```

---

# 22. Layer Position Sanity Check

For a homogeneous decoder stack, sample representative positions:

```text
early
middle
late
```

Example for 32 layers:

```text
layer 1
layer 16
layer 30
```

Measure each as a single TransformerLayer.

Compute simple diagnostics such as:

- coefficient of variation;
- maximum relative deviation.

If variation is below a configurable threshold, future default calibration may use only the middle representative layer.

If variation is high, preserve a positional class:

```text
EARLY
MIDDLE
LATE
SPECIAL
```

Do not assume layer-position equivalence globally.

---

# 23. P2D — Direct Module Profiling

Implement:

```text
ModuleProfiler
```

Initial normalized modules:

```text
Attention
MLP
```

Optional after the main path works:

```text
Norm
Projection
Embedding
LM Head
```

Use direct invocation and CUDA events.

Do not use nested forward-hook timing as the production implementation.

`torch.profiler` may be used to inspect the internal composition of a Module during diagnostics.

---

# 24. Decode Support

The domain schema MUST support:

```text
phase = DECODE
context_length
KV/cache metadata
```

Implementation rule:

- if the current Phase 0 runtime cleanly exposes KV-cache decode semantics, implement decode profiling;
- otherwise keep decode unsupported in v1 and fail clearly.

Do not fabricate decode measurements from prefill.

---

# 25. P2E — Operator Microprofiling

## 25.1 OperatorWorkloadRegistry

Create:

```python
class OperatorWorkload(Protocol):
    def prepare(self, signature: OperatorSignature, device: ...) -> None: ...
    def run_once(self) -> None: ...
    def cleanup(self) -> None: ...


class OperatorWorkloadRegistry:
    def resolve(self, kind: OperatorKind) -> OperatorWorkloadFactory:
        ...
```

Initial workloads:

```text
GEMM
ATTENTION
NORM
```

Add other operators incrementally.

---

# 26. GEMM Microbenchmark

Use the production-relevant PyTorch/backend path.

Inputs come from `GemmSignature`.

Do not benchmark arbitrary `(M, N, K)` grids.

Only benchmark:

```text
signatures actually required by ModelStore
+
explicit verification/calibration signatures
```

Sequence-dependent `M` values should come from the selected profiling sequence lengths.

---

# 27. Attention Microbenchmark

Signature dimensions SHOULD include:

```text
batch
heads
kv_heads
head_dim
q_len
kv_len
dtype
phase
backend
```

The microbenchmark MUST use the same logical backend primitive used by production.

Example:

```text
production = torch SDPA
microbenchmark = torch SDPA
```

Do not use a generic attention implementation to predict a fused production implementation.

---

# 28. Incremental Operator Profiling

Algorithm:

```text
collect normalized operator signatures
        ↓
query ProfileStore
        ↓
filter measurements compatible with target performance class/environment
        ↓
compute missing signatures
        ↓
benchmark only missing signatures
        ↓
append new measurements
```

Adding a new model MUST NOT trigger full reprofiling of existing signatures.

---

# 29. Performance-Class Verification

When a new physical accelerator claims compatibility with an existing `DevicePerformanceClass`:

1. run a very small verification suite;
2. compare against reference measurements;
3. if within configurable tolerance, allow profile reuse;
4. otherwise create/separate a new performance class or mark incompatible.

The verification suite should remain small, e.g.:

```text
one representative GEMM
one attention case
one memory-sensitive workload
```

Do not design an advanced statistical certification system in v1.

---

# 30. P2F — Network Domain

## 30.1 Endpoint

Reuse Phase 1 network facts.

Capture:

- worker;
- interface;
- address;
- MTU;
- nominal link speed;
- interface type;
- overlay type.

Phase 2 adds empirical endpoint/network measurements; it does not replace Phase 1 discovery.

---

# 31. PathClass

Initial categories:

```python
class NetworkPathClass(StrEnum):
    SAME_HOST = "same_host"
    WIRED_LAN = "wired_lan"
    WIFI_LAN = "wifi_lan"
    OVERLAY = "overlay"
    CROSS_SUBNET = "cross_subnet"
    OTHER = "other"
```

Classifier inputs:

- source/destination interface metadata;
- subnet information;
- overlay metadata;
- host identity.

Do not introduce LLDP, SNMP, switch-controller, or SDN dependencies in v1.

---

# 32. Pair

A directed pair:

```text
source worker/interface
        →
destination worker/interface
```

A→B and B→A are distinct.

---

# 33. RTT Profiling

Use system `ping` via `asyncio.create_subprocess_exec`.

Default:

```text
5–10 packets per pair
limited asynchronous concurrency
```

Collect:

- median RTT;
- p95 if enough samples;
- jitter/stddev;
- packet loss.

Dense pairwise RTT probing is acceptable because each probe is cheap.

The implementation MUST place a concurrency bound on probe fan-out.

---

# 34. Bandwidth Profiling

Use:

```text
iperf3 --json
```

Requirements:

- parse JSON output, not human text;
- support forward and reverse direction;
- enforce timeout;
- guarantee server cleanup;
- preserve raw diagnostic stderr on failure.

Default policy:

```text
one/few representative directed pair(s) per PathClass
```

Do NOT run full pairwise bandwidth profiling.

Bandwidth probes SHOULD run sequentially or with very low concurrency to avoid self-induced contention.

---

# 35. Network Measurement Regime

Phase 2 v1 measures:

```text
idle or reasonably idle
single-flow baseline
```

It does NOT model:

- shared NIC contention;
- switch uplink contention;
- Wi-Fi medium contention;
- concurrent pipeline flows;
- dynamic available bandwidth;
- congestion-aware rerouting.

Record the measurement regime so Phase 3 does not mistake baseline throughput for guaranteed concurrent bandwidth.

---

# 36. Payload Sizes

Do not sweep arbitrary payload sizes exhaustively.

Use workload-relevant sizes derived from model characterization where possible.

For pipeline hidden-state transfer:

```text
payload_bytes ≈
batch_size
× sequence_length
× hidden_size
× bytes_per_element
```

Deduplicate relevant payload-size classes.

A small representative set is acceptable in v1.

---

# 37. P2G — Worker ProfilingRunner

Add:

```text
control.worker.ProfilingRunner
```

Worker Agent structure:

```text
Worker Agent
├── LocalWorkerInspector
└── ProfilingRunner
    ├── ModelProfiler
    ├── OperatorProfiler
    └── NetworkProfiler
```

The runner is long-lived.

It SHOULD reuse:

- process state;
- CUDA initialization;
- backend initialization;
- model session state when appropriate.

Do not create/destroy a container or CUDA context per measurement case unless required by the runtime architecture.

---

# 38. Profiling Session

Introduce a session abstraction to group compatible cases.

Example:

```text
prepare session
↓
load model once
↓
run multiple layer/module cases
↓
cleanup model
↓
close session
```

This reduces checkpoint loading and initialization overhead.

Do not reuse state across cases when doing so would contaminate results.

---

# 39. Profiling Lease / Resource Reservation

A target accelerator MUST not be benchmarked intrusively while serving normal EdgeShard runtime workloads.

Introduce a lightweight lease:

```text
AVAILABLE
PROFILE_RESERVED
```

Before profiling:

- inspect `running_runtime_ids`;
- inspect current GPU utilization;
- inspect memory pressure;
- ensure the requested device is compatible.

If busy:

```text
reject/defer profiling case
```

Do not attempt background-load correction in v1.

Leases MUST be released on:

- success;
- failure;
- cancellation;
- runner shutdown.

Add tests for lease leakage.

---

# 40. Master ProfilingController

Add:

```text
control.master.ProfilingController
```

Responsibilities:

- create experiment definitions;
- expand strategy into profiling cases;
- select target workers;
- dispatch cases;
- track case/experiment lifecycle;
- cancel cases;
- collect results;
- persist measurements;
- build `ProfileSnapshot`.

The Master MUST NOT execute GPU benchmarks itself for remote workers.

---

# 41. Profiling Protocol

Use existing gRPC + Protobuf infrastructure.

Prefer a dedicated package:

```text
protocol.profiling
```

Suggested RPCs:

```text
PrepareProfilingSession
RunProfilingCase
GetProfilingCase
CancelProfilingCase
CloseProfilingSession
```

Alternative simplification is acceptable if it fits existing control conventions.

Protocol messages SHOULD carry domain DTOs, not persistence-layer objects.

Preserve existing Phase 1 session validation semantics.

A stale/invalid worker session MUST NOT be allowed to publish measurements.

---

# 42. Failure Semantics

Represent failures explicitly.

Suggested error categories:

```text
UNSUPPORTED_MODEL
UNSUPPORTED_GRANULARITY
UNSUPPORTED_PHASE
UNSUPPORTED_OPERATOR
DEVICE_BUSY
INSUFFICIENT_MEMORY
EXPORT_FAILED
PROFILER_FAILED
BENCHMARK_FAILED
NETWORK_UNREACHABLE
IPERF_UNAVAILABLE
TIMEOUT
CANCELLED
INTERNAL_ERROR
```

Do not encode failures as zero latency, empty samples, or `None` measurement values.

---

# 43. ProfileStore

Use SQLite v1.

Implement repository interface:

```python
class ProfileStore(Protocol):
    def append_measurement(...): ...
    def get_measurement(...): ...
    def query_measurements(...): ...
    def append_experiment(...): ...
    def update_experiment_state(...): ...
    def build_snapshot(...): ...
```

SQLite is an implementation detail behind this interface.

Do not let SQL leak into domain/controller code.

---

# 44. Persistence Model

Recommended tables:

```text
experiments
profiling_cases
measurements
measurement_samples

model_characterizations
transformer_layer_signatures
module_signatures
operator_signatures

network_endpoints
network_path_classes
network_pairs

environment_fingerprints
device_performance_classes
```

Use append-oriented measurement persistence.

Do not overwrite historical measurements when:

- runtime version changes;
- driver changes;
- model revision changes;
- backend changes.

Create new records under a new environment fingerprint.

---

# 45. Raw Samples

For v1:

- storing all samples is acceptable;
- sample counts are small;
- retain summaries for fast query.

Do not introduce Parquet/PyArrow solely for sample storage.

---

# 46. ProfileSnapshot

Create an immutable domain view:

```python
@dataclass(frozen=True)
class ProfileSnapshot:
    created_at: datetime
    model_characterizations: ...
    measurements: ...
    network_measurements: ...
```

The snapshot contains only empirical/static Phase 2 facts.

Phase 3 consumes this snapshot.

Phase 3 MUST NOT depend on a live `ProfilingRunner`.

---

# 47. Default Profiling Strategy

Implement:

```text
DefaultProfilingStrategy
```

## Model workflow

```text
1. Static model characterization

2. Extract normalized operator signatures

3. Deduplicate signatures

4. Microprofile missing operators
   per DevicePerformanceClass

5. Sparse real Module profiling
   initially Attention + MLP

6. Sparse real TransformerLayer profiling
   early/middle/late positional check

7. Persist all measurements independently
```

Do not produce a composed performance estimate in this strategy.

## Network workflow

```text
1. Endpoint characterization

2. PathClass classification

3. Dense cheap RTT probing

4. Sparse PathClass bandwidth profiling

5. Optional explicit Pair bandwidth profiling
```

---

# 48. Strategy Interfaces

Keep the strategy layer replaceable.

Recommended abstractions:

```text
GranularityPolicy
SamplingPolicy
MeasurementPolicy
StoppingPolicy
```

Do not over-engineer v1.

It is acceptable for `DefaultProfilingStrategy` to hard-code the initial composition of these policies as long as domain/storage/runner code does not depend on that strategy.

---

# 49. CLI

Add user-facing commands only after local APIs are stable.

Suggested shape:

```bash
edgeshard profile model inspect \
  --model <model>

edgeshard profile model run \
  --config <config> \
  --worker <worker> \
  --device <device>

edgeshard profile operator run \
  --worker <worker> \
  --missing-only

edgeshard profile network rtt \
  --all-workers

edgeshard profile network bandwidth \
  --path-class wired-lan

edgeshard profile snapshot \
  --format yaml
```

Exact CLI naming may follow existing repository conventions.

Do not allow CLI design to drive the domain model.

---

# 50. Implementation Stages

## P2A — Profiling Domain

Implement:

- enums;
- signatures;
- experiments;
- cases;
- measurements;
- environment fingerprint;
- performance class identity;
- network domain;
- `ProfileSnapshot`;
- canonical hashing.

### P2A DoD

- pure Python;
- no torch/gRPC/SQLite dependency in domain;
- strict validation;
- stable hashing tests;
- immutable snapshot tests;
- ruff clean;
- mypy strict clean.

## P2B — Instrumentation & Benchmark Harness

Implement:

- `CudaEventTimer`;
- `WallClockTimer`;
- allocator memory instrumentation;
- Phase 1 `MemoryPool` adapter;
- telemetry context sampler;
- `BenchmarkHarness`;
- default sampling policy;
- contamination checks.

### P2B DoD

Real RTX4090 and AGX Orin:

- stable CUDA timing;
- correct peak memory behavior;
- physical memory observation works;
- telemetry captured;
- warmup/repeat lifecycle correct;
- no leaked subprocesses/resources.

## P2C — Model Characterization

Implement:

- `ModelProfilingAdapter`;
- initial Qwen adapter;
- initial Llama adapter;
- static model characterization;
- transformer-layer enumeration;
- normalized module enumeration;
- `TorchExportExtractor`;
- `TorchProfilerExtractor` fallback;
- `OperatorNormalizer`;
- signature deduplication.

### P2C DoD

For real supported models:

- produce layer/module/operator hierarchy;
- extract shapes;
- normalize operators;
- preserve unknown ops;
- export path tested;
- fallback path tested;
- deterministic signatures.

## P2D — Real TransformerLayer / Module Profiling

Implement:

- `TransformerLayerProfiler`;
- `ModuleProfiler`;
- adapter-based input generation;
- prefill benchmarks;
- early/middle/late layer check;
- Attention profiling;
- MLP profiling.

### P2D DoD

On RTX and Jetson:

- actual checkpoint;
- actual layer/module execution;
- CUDA-event timing;
- allocator + MemoryPool measurement;
- empirical records persisted locally;
- results stable across repeated experiments.

Compare with legacy prototype only as a sanity reference; do not make the legacy path a dependency.

## P2E — Operator Microprofiling

Implement:

- workload registry;
- GEMM benchmark;
- Attention benchmark;
- Norm benchmark;
- missing-signature lookup;
- incremental profiling;
- performance-class verification suite.

### P2E DoD

- operator profiling runs without full model execution;
- existing signatures are reused;
- new model adds only missing shapes;
- compatible RTX4090 devices can reuse class measurements after verification;
- Jetson class handled separately.

## P2F — Network Profiling

Implement:

- endpoint characterization;
- path classifier;
- async ping runner;
- iperf3 JSON runner;
- sparse path-class bandwidth policy;
- optional explicit pair profiling;
- concurrency controls;
- server cleanup and timeouts.

### P2F DoD

On the heterogeneous cluster:

- all workers have endpoint profiles;
- complete RTT matrix exists;
- path classes generated;
- representative bandwidth measurements exist;
- no runaway iperf servers;
- failures are typed.

## P2G — Distributed Orchestration & Persistence

Implement:

- `ProfilingRunner`;
- profiling sessions;
- profiling leases;
- `ProfilingController`;
- profiling gRPC/protobuf;
- SQLite store;
- `ProfileSnapshot` construction;
- CLI integration.

### P2G DoD

Real multi-node flow:

```text
Master
↓
create experiment
↓
dispatch case
↓
Worker reserves target
↓
benchmark
↓
result RPC
↓
persist SQLite
↓
ProfileSnapshot
```

Test:

- Worker restart;
- Master restart;
- stale session;
- cancellation;
- duplicate result;
- busy GPU;
- unsupported model;
- network failure;
- timeout;
- partial experiment completion.

---

# 51. Required Test Pyramid

## Unit

Must cover:

- signature hashing;
- validation;
- operator normalization;
- path classification;
- sampling policy;
- summary statistics;
- persistence mapping;
- failure typing;
- lease lifecycle.

## Integration

Must cover:

```text
adapter
→ characterization
→ signature extraction
→ local benchmark
→ measurement
→ ProfileStore
```

and:

```text
network plan
→ ping/iperf runner
→ measurement
→ store
```

## Real Hardware

Required final validation:

```text
RTX4090 single GPU
RTX4090 multi-GPU host
AGX Orin 32GB
AGX Orin 64GB
heterogeneous multi-node cluster
```

Reuse the existing P1I environment where possible.

---

# 52. Important Correctness Rules

## 52.1 Measurements are not predictions

Never populate an empirical record from an estimator.

## 52.2 Missing is valid

If a metric cannot be measured:

```text
metric unavailable
```

not guessed.

## 52.3 Device identity is not performance-class identity

Keep provenance and reuse compatibility separate.

## 52.4 Module graph is not `nn.Module` traversal

Containers/wrappers are not automatically profiling modules.

## 52.5 Operator identity is not kernel identity

Do not expose CUDA/Triton kernel names as stable EdgeShard operator types.

## 52.6 Profiler traces are not authoritative latency

Use direct CUDA events for benchmark latency.

## 52.7 Network baseline is not guaranteed bandwidth

Record the measurement regime.

## 52.8 No resource double counting on Jetson

Always reuse Phase 1 shared `MemoryPool`.

---

# 53. Explicit Non-Goals for Phase 2 v1

Do NOT implement:

```text
complete Model × Device real profiling
exhaustive contiguous-shard profiling
complete pairwise bandwidth profiling

scheduler
PlacementPlan
deployment orchestration

latency interpolation
regression
random forest predictor
analytical resource model
learned estimator

active profiling
profiling-under-budget optimization
multi-fidelity optimization

direct CUPTI dependency
Nsight automation
custom Triton profiling kernels

network contention model
dynamic available-bandwidth prediction

production E2E request profiling
migration profiling
KV-cache migration profiling
```

These are future work or belong to Phase 3+.

---

# 54. Phase 3 Boundary

Phase 3 input:

```text
ClusterSnapshot
+
ModelCharacterization
+
ProfileSnapshot
```

Phase 3 output:

```text
ResourceEstimateSnapshot
```

Only Phase 3 may introduce:

```text
estimated latency
estimated memory
estimated transfer time
feasibility
confidence
estimate provenance
```

Possible future estimate provenance:

```text
MEASURED
CALIBRATED
TRANSFERRED
ESTIMATED
```

Phase 4 Scheduler consumes the complete estimate surface.

Therefore:

```text
Scheduler requires complete information
```

does NOT imply:

```text
Phase 2 must empirically measure every combination
```

---

# 55. Research Extension Point to Preserve

Do not implement this in Phase 2 v1, but keep interfaces compatible with:

> Profiling under budget for heterogeneous distributed LLM inference.

Future research questions include:

- optimal granularity;
- operator composition error;
- multi-fidelity profiling;
- adaptive sampling;
- uncertainty-driven measurement;
- model/device profile transferability;
- scheduler sensitivity to profile error;
- network pair selection under profiling budget.

Do not prematurely encode assumptions that would prevent these strategies.

---

# 56. Coding Instructions for Claude Code

1. **Inspect the current repository before changing files.**
2. Reuse existing Phase 1 domain utilities, identity/canonicalization, config conventions, error handling, and gRPC patterns.
3. Do not rewrite frozen Phase 0/Phase 1 components unless a minimal extension is required.
4. Prefer additive changes.
5. Keep dependency direction clean:
   ```text
   profiling.domain
       ↑
   profiling implementation
       ↑
   control/protocol
   ```
6. Keep PyTorch imports outside pure domain modules.
7. Do not introduce a new distributed task framework.
8. Do not introduce a database service.
9. Do not introduce direct CUPTI dependencies.
10. Every sub-phase must be independently testable before proceeding.
11. Keep `ruff` clean.
12. Keep `mypy --strict` clean.
13. Run the complete existing test suite after each sub-phase.
14. Do not weaken existing Phase 1 tests.
15. Preserve backwards compatibility unless the specification explicitly requires a change.
16. If a repository reality conflicts with this document, preserve the architectural intent and document the minimum required adaptation rather than forcing an incompatible abstraction.

---

# 57. Suggested Implementation Workflow for Claude Code

For each sub-phase:

```text
1. Inspect repository
2. Write/update a short implementation plan
3. Implement domain/API skeleton
4. Add unit tests first
5. Implement functionality
6. Run targeted tests
7. Run full tests
8. Run ruff
9. Run mypy --strict
10. Summarize:
    - changed files
    - architecture decisions
    - tests
    - unresolved issues
```

Do not begin the next sub-phase until the current sub-phase Definition of Done is satisfied.

---

# 58. Final Phase 2 Definition of Done

Phase 2 is complete when EdgeShard can:

1. statically characterize supported models into:
   ```text
   TransformerLayer
   → Module
   → OperatorSignature
   ```

2. directly benchmark real:
   ```text
   TransformerLayer
   Module
   ```

3. microbenchmark deduplicated:
   ```text
   OperatorSignature
   ```

4. reuse operator profiles by compatible device performance class;

5. characterize network:
   ```text
   Endpoint
   → PathClass
   → Pair
   ```

6. obtain:
   ```text
   dense cheap RTT
   sparse bandwidth measurements
   ```

7. execute profiling through:
   ```text
   Master
   → Worker Agent
   → ProfilingRunner
   ```

8. safely reserve local resources during intrusive profiling;

9. persist empirical facts in SQLite with full provenance;

10. produce an immutable:
    ```text
    ProfileSnapshot
    ```

11. pass unit, integration, RTX, Jetson, and heterogeneous multi-node validation;

12. contain **no performance estimator and no scheduler logic**.

At that point Phase 3 can begin.

---

# 59. Implementation Order

Use this order unless repository constraints make a small deviation necessary:

```text
P2A  Profiling Domain
 ↓
P2B  Instrumentation & Benchmark Harness
 ↓
P2C  Model Characterization
 ↓
P2D  Direct Layer/Module Profiling
 ↓
P2E  Operator Microprofiling
 ↓
P2F  Network Profiling
 ↓
P2G  Distributed Orchestration & Persistence
 ↓
ProfileSnapshot
 ↓
Phase 3
```

This order is intentionally designed so that each stage can be validated independently and so that profiling logic is working locally before distributed orchestration is added.
