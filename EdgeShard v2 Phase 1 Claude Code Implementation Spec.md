# EdgeShard v2 — Phase 1 Claude Code Implementation Specification

**Phase:** 1  
**Theme:** Worker + Device Capability / State  
**Repository:** `KasperXMS/EdgeShard-v2`  
**Baseline:** current `main` after completion of Phase 0 / Phase 0.5  
**Status:** implementation specification

---

# 1. Phase 1 Goal

Phase 1 introduces the production cluster control-plane foundation for EdgeShard v2.

The goal is:

> Turn heterogeneous physical machines into Master-managed Workers with stable identity, static capability discovery, dynamic device/resource state reporting, runtime/model inventory, registration, heartbeat/liveness tracking, and immutable cluster snapshots.

Phase 1 answers:

```text
Which Workers exist?

Which Devices exist on each Worker?

What are the static capabilities of each Worker and Device?

Which physical memory pools exist?

Which Devices share the same physical memory pool?

What is the current resource state?

Which EdgeShard-managed runtimes are currently running?

Which model snapshots exist in each Worker's ModelStore?

Is a Worker ONLINE / SUSPECT / OFFLINE?

What is the immutable state of the cluster at a particular instant?
```

Phase 1 does **not** answer:

```text
Which Worker should run a model?

How many Transformer blocks should be placed on a Device?

How fast will a shard run?

Can model X fit on Device Y?

What is the optimal partition?

What is the optimal placement?

How should runtimes be deployed according to a scheduler?
```

Those belong to later phases.

---

# 2. Existing Repository Baseline

The current repository already contains the Phase 0 inference substrate:

```text
src/edgeshard/
├── model/
├── inference/
├── protocol/
├── runtime/
│   ├── config.py
│   ├── info.py
│   ├── model_store.py
│   ├── shard_server.py
│   └── drivers/
│       ├── base.py
│       ├── edgeshard_shard.py
│       └── vllm.py
│
└── control/
    └── mock/
        ├── master.py
        ├── deployment.py
        ├── manifest.py
        └── client.py
```

The following Phase 0 abstractions are considered stable and must be reused:

```text
ModelAdapter
ShardModule
inference session / KV semantics
shard data-plane protocol
RuntimeDriver
RuntimeHandle
EdgeShardShardRuntimeDriver
VLLMRuntimeDriver
ModelStore
MockMaster
```

Phase 1 must extend the system above these seams rather than rewriting them.

---

# 3. Mandatory Architectural Invariants

These invariants are non-negotiable.

## 3.1 Do not change inference semantics

Phase 1 must not introduce Worker/Master awareness into:

```text
edgeshard.model
edgeshard.inference
ShardModule
ModelAdapter
KV cache
```

Forbidden examples:

```python
if worker_id:
    ...

from edgeshard.control.worker import WorkerAgent
```

inside inference/model code.

---

## 3.2 Shard data plane and cluster control plane are separate

Existing shard communication remains the model data plane:

```text
token payload
hidden states
logits
session lifecycle
```

Phase 1 introduces a separate control plane:

```text
Worker registration
heartbeat
capability reporting
cluster state
```

Do not add Worker management RPCs to `ShardRuntime`.

---

## 3.3 `control.mock` remains a Phase 0 integration harness

Do not turn `MockMaster` into the production Master.

Existing:

```text
control.mock.MockMaster
    ↓
RuntimeDriver
    ↓
local Docker
```

continues to exist for manifest-driven Phase 0 integration tests.

Production control becomes:

```text
Master
    ↕
Worker Agent
```

The two concepts are intentionally separate.

---

## 3.4 Runtime lifecycle remains owned by `RuntimeDriver`

Do not introduce a second runtime abstraction.

Continue to use:

```python
RuntimeDriver.start()
RuntimeDriver.wait_ready()
RuntimeDriver.info()
RuntimeDriver.stop()
```

Phase 1 only observes current runtime state.

Production runtime deployment commands are reserved for a later phase.

---

## 3.5 ModelStore remains worker-local

Do not redesign `ModelStore`.

Different Workers may use:

```text
/data/edgeshard-models
/mnt/ssd/edgeshard-models
```

but containers continue seeing:

```text
/models/<local_name>
```

Phase 1 adds inventory/reporting only.

No:

```text
model download
model eviction
multi-disk model placement
```

---

# 4. Target Architecture

```text
                              Master
                                │
                       gRPC control plane
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
        Worker Agent A                      Worker Agent B
         RTX4090 host                        AGX Orin host
              │                                   │
      ┌───────┼────────┐                  ┌───────┼────────┐
      │       │        │                  │       │        │
   Device   Model    Runtime           Device   Model    Runtime
   Probe    Store   Inventory          Probe    Store   Inventory
      │                │                  │                │
    NVML           Docker SDK         tegrastats       Docker SDK
      │                                   │
    RTX4090                           Jetson GPU + CPU
```

Master is the authority for:

```text
Worker registry
registration sessions
latest reported state
liveness
ClusterSnapshot generation
```

Worker Agent is the authority for:

```text
host discovery
device discovery
telemetry
ModelStore inventory
local runtime inventory
```

---

# 5. Technology Stack

Use the following stack.

```text
Language
    Python 3.12

Dependency/environment
    uv
    pyproject.toml
    uv.lock

Domain model
    frozen dataclasses
    Enum

Configuration
    Pydantic v2
    PyYAML

Concurrency
    asyncio

Control plane RPC
    grpc.aio
    Protocol Buffers

Host telemetry
    psutil

Discrete NVIDIA GPU
    nvidia-ml-py / NVML

Jetson telemetry
    psutil
    /proc
    /sys
    /etc
    persistent tegrastats subprocess

Container inspection
    Docker SDK for Python

Runtime abstraction
    existing RuntimeDriver

Model store
    existing ModelStore
    pathlib filesystem scanning

Master state
    in-memory registry/state store

CLI
    Typer

Logging
    Python stdlib logging

Process supervision
    systemd outside Python implementation

Testing
    pytest
    pytest-asyncio
    Hypothesis
```

Do not introduce:

```text
FastAPI
Flask
Redis
SQLite
PostgreSQL
etcd
Kafka
RabbitMQ
Celery
Ray
Kubernetes
Prometheus
OpenTelemetry
jetson-stats as required dependency
SSH-based worker control
```

---

# 6. Dependency Refactor

The current package installation couples control-plane code to inference dependencies.

Phase 1 should separate lightweight host control dependencies from the Hugging Face/PyTorch inference stack.

Target logical dependency structure:

```toml
[project]
dependencies = [
    "pydantic>=2.9",
    "pyyaml>=6.0.2",
    "grpcio>=1.66",
    "protobuf>=5.28",
    "docker>=7.1",
    "typer>=0.12",
    "httpx>=0.27",
]

[project.optional-dependencies]

inference = [
    "torch>=2.4",
    "transformers>=4.45",
    "accelerate>=0.34",
    "safetensors>=0.4.5",
    "huggingface-hub>=0.25",
]

worker = [
    "psutil>=7,<8",
    "nvidia-ml-py>=13,<14",
]
```

Exact locked versions must be determined by `uv lock`.

Do not break the existing Jetson container rule:

```text
NVIDIA-provided PyTorch and NumPy
must remain part of the NVIDIA platform stack.

Generic uv dependency resolution must not overwrite
the Jetson-native torch/numpy stack.
```

---

# 7. New Repository Layout

Add:

```text
src/edgeshard/
│
├── cluster/
│   ├── __init__.py
│   ├── identity.py
│   ├── capability.py
│   ├── state.py
│   ├── inventory.py
│   └── snapshot.py
│
├── control/
│   ├── mock/                     # existing, preserve
│   │
│   ├── worker/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── config.py
│   │   ├── identity.py
│   │   │
│   │   ├── discovery/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── host.py
│   │   │   ├── nvidia.py
│   │   │   └── jetson.py
│   │   │
│   │   ├── telemetry/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── host.py
│   │   │   ├── nvidia.py
│   │   │   └── jetson.py
│   │   │
│   │   ├── runtime_inventory.py
│   │   ├── model_inventory.py
│   │   └── master_client.py
│   │
│   └── master/
│       ├── __init__.py
│       ├── config.py
│       ├── service.py
│       ├── registry.py
│       ├── sessions.py
│       ├── state_store.py
│       ├── liveness.py
│       └── snapshot.py
│
└── protocol/
    └── control/
        ├── __init__.py
        ├── mapper.py
        ├── grpc_client.py
        ├── grpc_server.py
        └── pb/
```

Add:

```text
proto/worker_control.proto
```

Do not move the existing shard protobuf modules during Phase 1.

Avoid unrelated Phase 0 path churn.

---

# 8. Dependency Rules

`edgeshard.cluster` is a pure domain package.

It may import only:

```text
stdlib
dataclasses
enum
datetime
typing
```

It must not import:

```text
grpc
protobuf
docker
torch
transformers
psutil
pynvml
control
runtime
```

Allowed dependency direction:

```text
cluster
   ↑
protocol.control
   ↑
control.worker / control.master
```

Also:

```text
control.worker → runtime
control.worker → ModelStore
```

when inventory/lifecycle abstractions need them.

Forbidden:

```text
cluster → control
cluster → runtime
cluster → protocol
inference → cluster
model → cluster
```

---

# 9. Domain Model

All domain structures should be immutable unless there is a strong implementation reason otherwise.

Prefer:

```python
@dataclass(frozen=True)
```

Wire protobuf types must never become domain types.

---

# 10. Identity Types

## 10.1 WorkerIdentity

```python
@dataclass(frozen=True)
class WorkerIdentity:
    worker_id: str
    hostname: str
    agent_version: str
    protocol_version: str
```

### `worker_id`

Persistent identity of one Worker installation.

It must not derive from:

```text
hostname
IP address
MAC address
CUDA ordinal
```

On first Worker start:

```text
generate UUIDv4
persist to identity file
```

Subsequent starts reuse it.

Default production path:

```text
/var/lib/edgeshard/worker-id
```

Configurable for development/tests.

---

## 10.2 Agent instance ID

Every Worker Agent process creates:

```text
instance_id = UUIDv4
```

This is not persisted.

New process:

```text
same worker_id
new instance_id
```

---

## 10.3 Master session ID

After registration the Master returns:

```text
session_id
```

Every heartbeat includes:

```text
worker_id
instance_id
session_id
```

Old sessions must not update Worker state.

---

# 11. Device Identity

Do not use:

```text
cuda:0
```

as stable `device_id`.

Domain:

```python
@dataclass(frozen=True)
class DeviceIdentity:
    device_id: str
    kind: DeviceKind
    local_locator: str
```

Enum:

```python
class DeviceKind(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    OTHER = "other"
```

Discrete NVIDIA GPU:

```text
device_id should use a stable NVIDIA GPU UUID when available.
```

Integrated Jetson device:

derive persistent local identity from:

```text
worker_id
+
stable platform device key
```

Do not rely on enumeration order.

---

# 12. Memory Pool Abstraction

Memory must be represented as physical resource pools.

Do not model all memory as a property directly owned by a Device.

```python
class MemoryModel(str, Enum):
    DISCRETE = "discrete"
    SHARED = "shared"
```

```python
@dataclass(frozen=True)
class MemoryPoolCapability:
    memory_pool_id: str
    model: MemoryModel
    total_bytes: int
```

Dynamic:

```python
@dataclass(frozen=True)
class MemoryPoolState:
    memory_pool_id: str
    available_bytes: int | None
```

---

# 13. RTX Memory Representation

Example:

```text
CPU
 └── host-memory

RTX4090
 └── gpu-<uuid>-vram
```

Example capability:

```yaml
memory_pools:

  - memory_pool_id: host-memory
    model: shared
    total_bytes: ...

  - memory_pool_id: gpu-GPU-abc-vram
    model: discrete
    total_bytes: ...
```

---

# 14. Jetson Memory Representation

Jetson CPU and integrated GPU share one physical memory pool.

Represent:

```text
CPU ─┐
     ├── system-memory
GPU ─┘
```

Do not report separate independent 32 GB CPU and 32 GB GPU resources.

Both DeviceCapability objects should reference:

```text
memory_pool_id = "system-memory"
```

This prevents future scheduler double counting.

---

# 15. Capability Domain Types

Implement at minimum:

```python
@dataclass(frozen=True)
class OSInfo:
    name: str
    version: str | None
    kernel: str | None
```

```python
@dataclass(frozen=True)
class ContainerRuntimeCapability:
    runtime: str
    version: str | None
    nvidia_runtime_available: bool
```

```python
@dataclass(frozen=True)
class NetworkInterfaceCapability:
    interface_id: str
    name: str
    addresses: tuple[str, ...]
    mtu: int | None
```

```python
@dataclass(frozen=True)
class RuntimePlatformCapability:
    backend: str
    platform: str
    image: str | None
```

```python
@dataclass(frozen=True)
class DeviceCapability:
    identity: DeviceIdentity

    vendor: str
    model: str

    compute_capability: str | None

    memory_pool_id: str | None

    supported_dtypes: tuple[str, ...]

    driver_version: str | None

    platform_tags: tuple[str, ...]
```

```python
@dataclass(frozen=True)
class WorkerCapability:
    architecture: str
    os: OSInfo

    container_runtime: ContainerRuntimeCapability | None

    network_interfaces: tuple[NetworkInterfaceCapability, ...]

    devices: tuple[DeviceCapability, ...]

    memory_pools: tuple[MemoryPoolCapability, ...]

    runtime_platforms: tuple[RuntimePlatformCapability, ...]

    capability_revision: str
```

---

# 16. Capability Revision

Capability updates are rare compared with heartbeats.

Generate:

```text
capability_revision
```

from a deterministic canonical serialization of static capability.

Use a cryptographic hash such as SHA-256.

Heartbeat includes the current revision.

The full capability should be sent:

```text
on registration
or
when capability_revision changes
```

Do not resend the full static capability in every heartbeat.

---

# 17. Dynamic State Types

```python
class DeviceAvailability(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"
```

```python
@dataclass(frozen=True)
class DeviceState:
    device_id: str

    utilization: float | None
    temperature_c: float | None
    power_w: float | None

    availability: DeviceAvailability

    running_runtime_ids: tuple[str, ...]
```

Missing telemetry must be represented by `None`.

Example:

```text
power_w = None
```

does not imply:

```text
device unavailable
```

---

# 18. Worker State

```python
class WorkerStatus(str, Enum):
    ONLINE = "online"
    SUSPECT = "suspect"
    OFFLINE = "offline"
```

```python
@dataclass(frozen=True)
class WorkerState:
    worker_id: str
    status: WorkerStatus

    device_states: tuple[DeviceState, ...]
    memory_states: tuple[MemoryPoolState, ...]

    runtime_instances: tuple[RuntimeInstanceState, ...]
    models: tuple[ModelInventoryEntry, ...]
```

Do not put Master-local monotonic timestamps in the Worker-generated domain report.

Master tracking should live in a separate Master record.

---

# 19. Runtime Inventory

Runtime inventory reports existing EdgeShard-managed runtime containers.

Use the labels already applied to managed containers.

Do not assume Agent memory contains all runtime handles, because Worker Agent can restart while runtime containers remain alive.

Worker startup must:

```text
Docker SDK
↓
find managed EdgeShard containers
↓
read labels/state
↓
reconstruct runtime inventory
```

Domain:

```python
class RuntimeStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    UNKNOWN = "unknown"
```

```python
@dataclass(frozen=True)
class RuntimeInstanceState:
    runtime_id: str
    backend: str

    execution_id: str | None

    status: RuntimeStatus

    device_ids: tuple[str, ...]

    container_id: str | None
    endpoint: str | None

    model_local_name: str | None
```

Phase 1 runtime inventory is observational only.

Do not implement:

```text
StartRuntime RPC
StopRuntime RPC
DeployShard RPC
```

---

# 20. Model Inventory

Reuse:

```python
edgeshard.runtime.model_store.ModelStore
```

Add Worker-side scanning.

```python
class ModelAvailability(str, Enum):
    READY = "ready"
    INCOMPLETE = "incomplete"
    INVALID = "invalid"
```

```python
@dataclass(frozen=True)
class ModelInventoryEntry:
    local_name: str

    model_id: str | None
    revision: str | None

    size_bytes: int | None

    status: ModelAvailability
```

The Master must never depend on:

```text
/data/edgeshard-models
/mnt/ssd/edgeshard-models
```

or any other host path.

Only logical model identity and status are reported.

---

# 21. Host Discovery Backend

Use Python standard library and `psutil`.

Static host information:

```python
platform.machine()
platform.system()
platform.release()
platform.freedesktop_os_release()
socket.gethostname()
```

Network:

```python
psutil.net_if_addrs()
psutil.net_if_stats()
```

Memory:

```python
psutil.virtual_memory()
```

Do not introduce `pyroute2` or `netifaces` in Phase 1.

---

# 22. NVIDIA Discrete GPU Backend

Use:

```text
nvidia-ml-py
NVML
```

Do not parse `nvidia-smi`.

Create:

```text
control/worker/discovery/nvidia.py
control/worker/telemetry/nvidia.py
```

Static information should include when available:

```text
GPU UUID
GPU name
driver version
total VRAM
compute capability
```

Dynamic information:

```text
GPU utilization
memory free/used
temperature
power
```

Errors for unsupported individual metrics should not fail the entire Device probe.

---

# 23. Jetson Backend

Do not model Jetson as an ordinary discrete NVML GPU.

Create dedicated implementation:

```text
JetsonPlatformProbe
JetsonTelemetryBackend
```

Static sources may include:

```text
platform.machine()
/proc/device-tree/*
/etc/nv_tegra_release
/sys/*
```

Shared system memory comes from:

```python
psutil.virtual_memory()
```

GPU/device telemetry comes from:

```text
tegrastats
```

---

# 24. tegrastats Process Model

Do not launch a fresh `tegrastats` process on every heartbeat.

Worker Agent startup:

```text
start one long-lived tegrastats subprocess
↓
read stdout asynchronously
↓
parse samples
↓
store latest sample
```

Heartbeat simply reads the latest cached sample.

Design:

```text
TegrastatsProcess
        │
        ▼
TegrastatsParser
        │
        ▼
LatestTelemetryCache
        │
        ▼
JetsonTelemetryBackend.sample()
```

Parser must tolerate fields that differ across Jetson releases.

Unknown/missing fields produce `None`.

---

# 25. Probe Interfaces

Static and dynamic probing must be separate.

```python
class CapabilityProbe(Protocol):
    def discover(self) -> CapabilityFragment:
        ...
```

```python
class TelemetryProbe(Protocol):
    async def sample(self) -> StateFragment:
        ...
```

Do not implement one giant:

```python
probe_everything()
```

because capability collection and heartbeat state collection have different lifetimes.

---

# 26. Worker Agent Configuration

Example:

```yaml
worker:
  identity_path: /var/lib/edgeshard/worker-id

  master:
    endpoint: 192.168.1.100:51000

  heartbeat_interval_s: 5

  reconnect:
    initial_delay_s: 1
    max_delay_s: 30

model_store:
  root: /data/edgeshard-models

runtime:
  discover_managed_containers: true

tls:
  enabled: false
```

Jetson example:

```yaml
model_store:
  root: /mnt/ssd/edgeshard-models
```

Configuration should use Pydantic.

Do not use Pydantic classes as cluster domain objects.

---

# 27. Worker Agent Lifecycle

Implement:

```text
start process
    │
    ▼
load/create persistent worker_id
    │
    ▼
create instance_id
    │
    ▼
discover static capability
    │
    ▼
calculate capability_revision
    │
    ▼
discover model inventory
    │
    ▼
discover runtime inventory
    │
    ▼
RegisterWorker
    │
    ▼
receive session_id
    │
    ▼
start heartbeat loop
```

If Master connection is lost:

```text
disconnect
↓
backoff
↓
register again
↓
receive new session
↓
send fresh current state
```

Do not attempt to restore a stale registration session.

---

# 28. Control Plane Proto

Add:

```text
proto/worker_control.proto
```

Service:

```protobuf
service WorkerRegistryService {
  rpc RegisterWorker(RegisterWorkerRequest)
      returns (RegisterWorkerResponse);

  rpc Heartbeat(HeartbeatRequest)
      returns (HeartbeatResponse);

  rpc UpdateCapability(UpdateCapabilityRequest)
      returns (UpdateCapabilityResponse);
}
```

Do not add deployment RPCs in Phase 1.

---

# 29. RegisterWorker Semantics

Request logically contains:

```text
protocol_version

worker_id
instance_id

WorkerIdentity

capability_revision
full WorkerCapability

initial dynamic state
```

Master behavior:

```text
validate protocol
upsert Worker identity
invalidate old registration session
store new static capability
create new session_id
store initial state
mark Worker ONLINE
```

Response:

```text
session_id
heartbeat_interval_ms
server_protocol_version
```

---

# 30. Heartbeat Semantics

Request:

```text
worker_id
instance_id
session_id

sequence_number

worker_reported_at

capability_revision

dynamic WorkerState
```

Sequence numbers begin from a known value after registration and increase monotonically.

Master accepts heartbeat only when:

```text
worker_id exists
session_id == current session
instance_id == current instance
sequence_number > previous accepted sequence
```

Out-of-order heartbeat:

```text
ignore/reject
```

It must never overwrite newer state.

---

# 31. Worker Clock Is Not Trusted for Liveness

Worker may include wall-clock timestamps for debugging.

Master liveness must use:

```python
time.monotonic()
```

at receive time.

Maintain:

```text
last_heartbeat_monotonic
last_heartbeat_wall_clock
```

Timeout logic uses only the monotonic value.

---

# 32. Liveness Semantics

Default:

```text
heartbeat interval: 5 seconds

ONLINE:
    heartbeat received within 10 seconds

SUSPECT:
    no valid heartbeat for >10 seconds

OFFLINE:
    no valid heartbeat for >20 seconds
```

Make thresholds configurable.

A missing single heartbeat must not immediately mark the Worker offline.

Offline Workers remain in registry.

Do not delete them automatically.

---

# 33. Master Internal Architecture

```text
MasterService
│
├── WorkerRegistry
├── SessionManager
├── StateStore
├── LivenessManager
└── SnapshotBuilder
```

---

# 34. WorkerRegistry

Stores stable information:

```text
WorkerIdentity
WorkerCapability
capability_revision
```

Phase 1 implementation can be purely in-memory.

Example internal API:

```python
class WorkerRegistry:
    def upsert(...)
    def get(worker_id: str) -> WorkerRecord
    def list_workers() -> tuple[WorkerRecord, ...]
```

Do not introduce a database.

---

# 35. SessionManager

Stores:

```text
worker_id
current instance_id
current session_id
last accepted sequence
```

New registration for an existing worker invalidates the previous session immediately.

Old heartbeat must not update state.

---

# 36. StateStore

Stores the latest accepted dynamic state per Worker.

It is not a historical telemetry database.

Only:

```text
latest valid state
```

is required.

---

# 37. LivenessManager

Runs independently of Worker timestamps.

Suggested implementation:

```text
periodic asyncio task
```

that evaluates all Workers against Master-local monotonic receive timestamps.

Tests must use an injectable clock.

Do not implement timeout tests using real `sleep(20)`.

---

# 38. ClusterSnapshot

The primary Phase 1 output is:

```python
@dataclass(frozen=True)
class ClusterSnapshot:
    snapshot_id: str
    created_at: datetime
    workers: tuple[WorkerSnapshot, ...]
```

```python
@dataclass(frozen=True)
class WorkerSnapshot:
    identity: WorkerIdentity
    capability: WorkerCapability
    state: WorkerState

    session_id: str | None

    last_seen_at: datetime | None
```

Snapshot creation copies current state.

After creation:

```text
new heartbeat
```

must not mutate an existing snapshot.

---

# 39. ClusterSnapshot Contains Facts Only

Allowed:

```text
RTX4090
24 GB discrete VRAM
18 GB currently available
SM89
GPU utilization 21%

Jetson
32 GB shared memory pool
21 GB currently available
SM87
```

Forbidden:

```text
Qwen shard [0,24) will fit

estimated prefill = 44 ms

this Worker is better for block 20

recommended partition = [0,14), [14,28)
```

Those are profiling/scheduling outputs.

---

# 40. Control Proto Mapping

Follow the existing Phase 0 wire/domain separation.

Add:

```text
src/edgeshard/protocol/control/mapper.py
```

Only this boundary translates:

```text
cluster dataclass
↔
protobuf DTO
```

Never pass generated protobuf objects into:

```text
WorkerRegistry
StateStore
SnapshotBuilder
future Scheduler
```

---

# 41. Protobuf Generation

Extend:

```text
scripts/generate_proto.py
```

so it also compiles:

```text
proto/worker_control.proto
```

into:

```text
src/edgeshard/protocol/control/pb/
```

Generated files remain committed.

Update Ruff/mypy exclusions accordingly.

Do not move existing shard protobuf generation in Phase 1.

---

# 42. CLI Target

Refactor the CLI into command groups while preserving backwards compatibility.

Target:

```bash
edgeshard runtime serve --config runtime.yaml

edgeshard worker inspect --config worker.yaml

edgeshard worker serve --config worker.yaml

edgeshard master serve --config master.yaml
```

Keep existing:

```bash
edgeshard serve --config runtime.yaml
```

as a compatibility alias during Phase 1.

---

# 43. `worker inspect`

Implement this before network registration.

Example:

```bash
edgeshard worker inspect --config worker.yaml
```

Output a machine-readable snapshot of local capability/state.

Prefer JSON default or a clear option:

```bash
--format json
--format yaml
```

It must work without Master.

This command is the primary local validation tool for Phase 1 hardware discovery.

---

# 44. `worker serve`

Responsibilities:

```text
identity
static discovery
telemetry collection
runtime/model inventory
registration
heartbeat
reconnect
```

It does not:

```text
schedule
profile workloads
start production runtimes
download models
```

---

# 45. `master serve`

Responsibilities:

```text
start WorkerRegistryService
accept registrations
accept heartbeats
maintain liveness
build cluster snapshots
```

Phase 1 does not require:

```text
REST UI
web dashboard
database
scheduler
```

A debug CLI/log representation of ClusterSnapshot is sufficient.

---

# 46. Logging

Use standard library `logging`.

Every relevant record should include where applicable:

```text
component
worker_id
instance_id
session_id
sequence
```

Examples:

```text
INFO worker.registration worker_id=... session_id=...
INFO worker.heartbeat sequence=31
WARN worker.telemetry probe=nvidia metric=power unavailable
WARN master.liveness worker_id=... state=SUSPECT
INFO master.liveness worker_id=... state=OFFLINE
```

Do not add `structlog`, `loguru`, or OpenTelemetry in Phase 1.

---

# 47. Error Handling

Follow the existing EdgeShard principle:

> Fail loudly on semantic errors, but tolerate unavailable optional telemetry.

Fatal examples:

```text
invalid Worker config
cannot load/create Worker identity
invalid registration response
protocol version mismatch
duplicate invalid Device identity
internal capability inconsistency
```

Non-fatal examples:

```text
temperature unsupported
power telemetry unavailable
one optional Jetson field missing
Docker daemon temporarily unavailable for runtime inventory
```

Non-fatal telemetry failures should be logged and represented with unknown/empty state where possible.

They must not crash the entire Worker Agent unless the core Worker cannot function.

---

# 48. Phase 1 Security Scope

Initial implementation assumes:

```text
trusted LAN
or trusted overlay network such as ZeroTier
```

Support config:

```yaml
tls:
  enabled: false
```

Structure code such that TLS can be added later.

Do not implement full PKI/mTLS management in Phase 1.

---

# 49. Tests

Preserve existing:

```text
tests/unit/
tests/integration/
tests/container/
```

Add:

```text
tests/unit/cluster/
tests/unit/control/worker/
tests/unit/control/master/
tests/unit/protocol/control/

tests/integration/control/
```

Optional hardware-specific:

```text
tests/platform/
```

---

# 50. pytest Markers

Add:

```text
rtx
jetson
```

Existing test suite must not require physical NVIDIA hardware by default.

Example:

```bash
pytest
pytest -m rtx
pytest -m jetson
```

---

# 51. Required Unit Tests

## Identity

```text
worker_id created once
worker_id persists across Agent restart
instance_id changes each process start
```

## Memory pool

```text
discrete GPU has independent VRAM pool
Jetson CPU/GPU can reference same shared pool
invalid missing memory_pool reference rejected
```

## Capability revision

```text
same canonical capability → same revision
modified capability → different revision
```

## Master sessions

```text
registration creates session
second registration invalidates old session
old heartbeat rejected
```

## Heartbeat order

```text
seq=5 accepted
seq=4 rejected
seq=5 duplicate rejected
seq=6 accepted
```

## Liveness

Using fake clock:

```text
ONLINE
→ SUSPECT
→ OFFLINE
```

without real-time sleeping.

## Snapshot immutability

```text
create snapshot A
accept heartbeat B
snapshot A unchanged
snapshot B contains new state
```

---

# 52. Required Integration Tests

## Test A — Registration

```text
Master process
+
Worker process
```

Expected:

```text
Worker registers
Master sees exactly one Worker
Worker becomes ONLINE
```

---

## Test B — Multiple Workers

Use fake probes or test fixtures:

```text
RTX-like Worker
Jetson-like Worker
```

Master sees both independently.

---

## Test C — Re-registration

```text
Worker registers
Agent restarts
same worker_id
new instance_id
new session
```

Expected:

```text
Worker is still same registry entry
old session becomes invalid
```

---

## Test D — Out-of-order heartbeat

Send:

```text
1
2
4
3
```

Final state must remain state from sequence `4`.

---

## Test E — Offline

Stop Worker.

Master transitions:

```text
ONLINE
↓
SUSPECT
↓
OFFLINE
```

using shortened test-config thresholds.

---

## Test F — Recovery

Restart Worker.

Expected:

```text
same worker_id
new session
ONLINE
```

---

# 53. Physical RTX Acceptance Test

On RTX4090 Worker verify:

```text
architecture = x86_64

GPU:
    NVIDIA RTX 4090
    stable identity
    compute capability = 8.9 / SM89 equivalent
    discrete memory pool
    VRAM total correct

dynamic:
    utilization present
    available memory present
    temperature if supported
    power if supported
```

Start a GPU workload.

Expected:

```text
GPU utilization increases
available VRAM decreases
Master receives new values
```

---

# 54. Physical Jetson Acceptance Test

On AGX Orin verify:

```text
architecture = aarch64

CPU
GPU

both can reference:
    system-memory
```

Expected memory topology:

```text
CPU ─┐
     ├── shared system-memory
GPU ─┘
```

Do not expose two independent 32 GB resources.

Verify:

```text
L4T/platform information discoverable
shared RAM available state
Jetson GPU utilization where supported
temperature where supported
```

Unsupported fields may be `None`.

---

# 55. Heterogeneous Cluster Acceptance Test

Final Phase 1 physical topology:

```text
                     Master
                       │
           ┌───────────┴───────────┐
           │                       │
       RTX4090                 AGX Orin
        Worker                  Worker
```

Master-generated ClusterSnapshot must simultaneously represent:

```text
Worker A:
    x86_64
    RTX4090
    SM89
    discrete VRAM

Worker B:
    aarch64
    AGX Orin
    SM87
    shared CPU/GPU memory
```

Both must be:

```text
ONLINE
```

and independently heartbeat.

This is the main Phase 1 heterogeneous-system validation.

---

# 56. Implementation Milestones

Implement in the following order.

Do not skip ahead.

---

## Milestone P1A — Dependency and Cluster Domain

Implement:

```text
dependency extras
cluster/
```

including:

```text
identity
capability
memory pools
state
runtime inventory type
model inventory type
snapshot
```

Tests:

```text
all domain unit tests
```

Do not implement gRPC yet.

---

## Milestone P1B — Worker Local Discovery

Implement:

```text
WorkerConfig
IdentityManager
Host capability probe
psutil telemetry
network discovery
ModelInventory
RuntimeInventory
```

Add:

```bash
edgeshard worker inspect
```

Validate on normal development host.

---

## Milestone P1C — NVIDIA Discrete GPU

Implement NVML capability and telemetry backend.

Validate on RTX4090.

Do not add scheduling logic.

---

## Milestone P1D — Jetson

Implement:

```text
Jetson platform discovery
shared memory representation
long-lived tegrastats reader
Jetson telemetry parser
```

Validate on AGX Orin.

At the end of P1D the local resource abstraction must represent both RTX and Jetson correctly.

---

## Milestone P1E — Control Protocol

Implement:

```text
worker_control.proto
generated pb
domain mapper
grpc.aio client/server
RegisterWorker
UpdateCapability
Heartbeat
```

Use fake probes first.

---

## Milestone P1F — Master State Management

Implement:

```text
WorkerRegistry
SessionManager
StateStore
LivenessManager
```

Required:

```text
persistent worker identity semantics
new session on registration
stale-session rejection
heartbeat sequence validation
ONLINE/SUSPECT/OFFLINE
```

---

## Milestone P1G — Worker Agent Networking

Implement:

```text
registration loop
heartbeat loop
reconnect/backoff
capability revision handling
```

Add:

```bash
edgeshard worker serve
edgeshard master serve
```

---

## Milestone P1H — ClusterSnapshot

Implement immutable Master-side snapshot production.

Test concurrent heartbeat updates versus snapshot creation.

---

## Milestone P1I — Real Heterogeneous Validation

Run:

```text
Master
RTX4090 Worker
AGX Orin Worker
```

Validate full Phase 1 acceptance matrix.

---

# 57. Explicitly Deferred Work

Do not implement any of the following during Phase 1:

```text
device performance benchmarks
representative Transformer block benchmarks
network bandwidth/RTT profiling
model structural profiling
memory feasibility estimator
prefill/decode estimator

scheduler
device selection
dynamic partition
placement optimization
PlacementPlan

production StartRuntime command
production StopRuntime command
DeployShard
runtime migration

model download
model eviction
model replication

fault-tolerant inference
runtime migration
batching
streaming
optimized tensor transport

Redis
database persistence
Kubernetes
Prometheus/OpenTelemetry
```

If implementation appears to require one of these, reconsider the boundary before adding it.

---

# 58. Phase 2 Compatibility Requirements

Phase 1 identifiers must be stable enough for later profiles to reference:

```text
worker_id
device_id
memory_pool_id
network interface ID
```

Future Phase 2 will produce something conceptually like:

```text
DeviceProfile(device_id=...)
NetworkProfile(worker_a, worker_b, interface=...)
```

Do not design Phase 1 identifiers that change after reboot or enumeration order changes.

---

# 59. Phase 4 Scheduler Compatibility Requirement

The eventual scheduler should be able to consume:

```python
schedule(
    model_spec,
    cluster_snapshot,
    profile_snapshot,
) -> PlacementPlan
```

Therefore `ClusterSnapshot` must not depend on:

```text
Docker client
NVML handles
grpc channel
Worker Agent process
Pydantic config
protobuf object
```

It must be a pure immutable representation of cluster facts.

---

# 60. Code Quality Requirements

Every new module must comply with existing:

```text
ruff
mypy --strict
pytest
```

Avoid:

```text
Any
```

unless wrapping untyped third-party APIs.

Contain third-party API ugliness inside backend adapters.

Example:

```text
NVML-specific types
    stay in nvidia backend

Docker SDK objects
    stay in runtime inventory/backend

protobuf DTOs
    stay in protocol/control

tegrastats strings
    stay in Jetson backend
```

Do not leak them into cluster domain code.

---

# 61. Documentation Update Requirement

When Phase 1 interfaces stabilize, update:

```text
ARCHITECTURE.md
README.md
KNOWN_LIMITATIONS.md
```

`ARCHITECTURE.md` must document the new dependency direction:

```text
                cluster
             ↗           ↖
protocol.control      control.*
                         │
                    runtime drivers
```

and clearly distinguish:

```text
Mock Master
Production Master
Worker Agent
Runtime
Shard
```

---

# 62. Expected Final Repository State

Conceptually:

```text
EdgeShard v2
│
├── Inference substrate
│   ├── ModelAdapter
│   ├── ShardModule
│   ├── KV/session
│   └── shard protocol
│
├── Runtime substrate
│   ├── EdgeShard shard runtime
│   ├── vLLM runtime
│   ├── RuntimeDriver
│   └── ModelStore
│
├── Worker control
│   ├── identity
│   ├── device discovery
│   ├── memory pools
│   ├── telemetry
│   ├── runtime inventory
│   └── model inventory
│
└── Master control
    ├── registry
    ├── sessions
    ├── heartbeat
    ├── liveness
    └── ClusterSnapshot
```

The execution/data path remains separate:

```text
Master
  │
  │ control only
  ▼
Workers

Worker A runtime ── hidden state ──> Worker B runtime

Master is not in the token data path.
```

---

# 63. Phase 1 Definition of Done

Phase 1 is complete only when all items below pass.

```text
Identity
[x] persistent worker_id
[x] per-process instance_id
[x] per-registration session_id
[x] stable device_id

Capability
[x] architecture / OS
[x] container runtime
[x] network interfaces
[x] DeviceCapability
[x] RuntimePlatformCapability
[x] deterministic capability_revision

Memory
[x] explicit MemoryPool abstraction
[x] RTX discrete VRAM
[x] Jetson shared CPU/GPU memory
[x] no shared-memory double counting

Dynamic state
[x] available memory
[x] GPU utilization
[x] optional temperature
[x] optional power
[x] Device availability

Inventory
[x] existing managed runtime discovery
[x] ModelStore inventory

Worker Agent
[x] host-native process
[x] worker inspect
[x] registration
[x] heartbeat
[x] reconnect/backoff
[x] capability update

Master
[x] WorkerRegistry
[x] SessionManager
[x] StateStore
[x] stale-session rejection
[x] heartbeat sequence validation
[x] ONLINE/SUSPECT/OFFLINE
[x] immutable ClusterSnapshot

Platforms
[x] RTX4090 real-device validation
[x] AGX Orin real-device validation
[x] simultaneous heterogeneous cluster validation

Regression
[x] existing Phase 0 unit tests pass
[x] existing Phase 0 integration tests pass
[x] existing container/runtime tests remain valid
[x] control.mock retains existing semantics
```

The following must still be absent:

```text
[ ] profiler
[ ] network benchmark
[ ] memory estimator
[ ] scheduler
[ ] dynamic partition
[ ] PlacementPlan
[ ] production deployment orchestration
```

---

# 64. Claude Code Execution Rules

When implementing this specification:

1. Inspect the current repository before editing each subsystem.
2. Reuse existing abstractions rather than creating parallel ones.
3. Do not refactor unrelated Phase 0 modules merely for aesthetic cleanup.
4. Implement one milestone at a time.
5. Add unit tests together with each domain/control component.
6. Run Ruff, mypy and relevant pytest suites after each milestone.
7. Preserve existing Phase 0 behavior.
8. Do not silently broaden Phase 1 scope.
9. If an interface in this specification conflicts with an established repository invariant, preserve the invariant and document the discrepancy.
10. Prefer a small explicit implementation over a generic framework.

The overriding design principle is:

> Phase 1 builds a reliable observation and management plane for heterogeneous Workers. It must expose clean immutable cluster state to future profiling and scheduling layers without allowing those future concerns to leak downward into Worker discovery, runtime execution, or inference.