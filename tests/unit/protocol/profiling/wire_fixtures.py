"""Shared hand-built domain objects for the profiling wire tests (§41).

The mapper/transport tests need real domain trees (session facts, cases,
outcomes) but none of the torch-dependent fixtures: everything here is
built by hand with the same shapes the codec tests pin, so a wire
round-trip failure points at the protocol layer, not at a fixture.
"""

from __future__ import annotations

from datetime import UTC, datetime

from edgeshard.profiling.domain.environment import (
    EnvironmentFingerprint,
    environment_fingerprint_id,
)
from edgeshard.profiling.domain.experiment import (
    CaseOutcome,
    ModelCaseSpec,
    NetworkCaseSpec,
    ProfilingCase,
    ProfilingErrorCategory,
    ProfilingFailure,
)
from edgeshard.profiling.domain.measurement import (
    LatencyMetrics,
    MeasurementMetrics,
    MeasurementRecord,
    TimeUnit,
    summarize_samples,
)
from edgeshard.profiling.domain.model import (
    ModelCharacterization,
    ModelReference,
    ModelStage,
    StageKind,
)
from edgeshard.profiling.domain.network import ProbeKind
from edgeshard.profiling.domain.session import (
    LayerEntry,
    ModelSessionFacts,
    ModuleEntry,
    ProfilingSessionKind,
    ProfilingSessionRequest,
)
from edgeshard.profiling.domain.signature import (
    GemmSignature,
    ModuleKind,
    ModuleSignature,
    OperatorKind,
    OperatorSignature,
    ProfilingGranularity,
    TransformerLayerSignature,
)
from edgeshard.profiling.network.classifier import (
    InterfaceFacts,
    InterfaceKind,
    WorkerNetworkFacts,
)

WORKER_ID = "w-1"
INSTANCE_ID = "instance-1"
REGISTRATION_SESSION_ID = "registration-1"
PROFILING_SESSION_ID = "profiling-session-1"

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)
LATER = datetime(2026, 9, 7, 12, 5, tzinfo=UTC)

MODEL = ModelReference(model_id="tiny/llama", revision="local")

LAYER_SIG = TransformerLayerSignature(
    architecture_family="llama",
    layer_type="standard_decoder",
    hidden_size=32,
    intermediate_size=64,
    num_attention_heads=4,
    num_kv_heads=2,
    head_dim=8,
    dtype="fp32",
    quantization=None,
)
MODULE_SIG = ModuleSignature(
    kind=ModuleKind.MLP,
    architecture_family="llama",
    structural_parameters=(("hidden_size", 32), ("intermediate_size", 64)),
    dtype="fp32",
    quantization=None,
)
OPERATOR_SIG = OperatorSignature(
    kind=OperatorKind.GEMM,
    parameters=GemmSignature(m=512, n=64, k=32, dtype="bf16", transpose_a=True),
    backend_family="torch",
)

CHARACTERIZATION = ModelCharacterization(
    model=MODEL,
    architecture_family="llama",
    num_layers=2,
    hidden_size=32,
    intermediate_size=64,
    vocab_size=64,
    num_attention_heads=4,
    num_kv_heads=2,
    head_dim=8,
    dtype="fp32",
    quantization=None,
    tied_word_embeddings=False,
    stages=(
        ModelStage(kind=StageKind.EMBEDDING),
        ModelStage(kind=StageKind.TRANSFORMER_LAYER_GROUP, layer_count=2),
        ModelStage(kind=StageKind.LM_HEAD),
    ),
)

SESSION_FACTS = ModelSessionFacts(
    characterization=CHARACTERIZATION,
    layer_entries=(
        LayerEntry(0, "model.layers.0", LAYER_SIG),
        LayerEntry(1, "model.layers.1", LAYER_SIG),
    ),
    module_entries=(
        ModuleEntry("mlp", "model.layers.0.mlp", ModuleKind.MLP, 0, MODULE_SIG),
    ),
    operator_signatures=(OPERATOR_SIG,),
)

MODEL_SESSION_REQUEST = ProfilingSessionRequest(
    kind=ProfilingSessionKind.MODEL,
    device_ids=("gpu-0",),
    model=MODEL,
    dtype="fp32",
)
OPERATOR_SESSION_REQUEST = ProfilingSessionRequest(
    kind=ProfilingSessionKind.OPERATOR,
    device_ids=("gpu-0",),
)
NETWORK_SESSION_REQUEST = ProfilingSessionRequest(kind=ProfilingSessionKind.NETWORK)

NETWORK_FACTS = WorkerNetworkFacts(
    worker_id=WORKER_ID,
    hostname="host-a",
    interfaces=(
        InterfaceFacts(
            interface_id="if-eth0",
            name="eth0",
            kind=InterfaceKind.WIRED,
            overlay_type=None,
            addresses=("192.168.1.10",),
            mtu=1500,
        ),
    ),
)
PEER_NETWORK_FACTS = WorkerNetworkFacts(
    worker_id="w-2",
    hostname="host-b",
    interfaces=(
        InterfaceFacts(
            interface_id="if-eth0",
            name="eth0",
            kind=InterfaceKind.WIRED,
            overlay_type=None,
            addresses=("192.168.1.11",),
            mtu=1500,
        ),
    ),
)

MODEL_CASE = ProfilingCase.for_spec(
    WORKER_ID,
    ModelCaseSpec(
        granularity=ProfilingGranularity.OPERATOR,
        device_ids=("gpu-0",),
        dtype="fp32",
        operator_signature=OPERATOR_SIG,
    ),
)
NETWORK_CASE = ProfilingCase.for_spec(
    WORKER_ID,
    NetworkCaseSpec(
        probe_kind=ProbeKind.RTT,
        source_worker_id=WORKER_ID,
        destination_worker_id="w-2",
        packet_count=5,
    ),
)

ENVIRONMENT = EnvironmentFingerprint(
    backend="torch",
    profiling_implementation_revision="test",
    torch_version="test",
    dtype="fp32",
    worker_id=WORKER_ID,
    device_id="gpu-0",
)

RECORD = MeasurementRecord(
    measurement_id="m-1",
    case_id=MODEL_CASE.case_id,
    environment_fingerprint=environment_fingerprint_id(ENVIRONMENT),
    environment=ENVIRONMENT,
    started_at=NOW,
    finished_at=LATER,
    sample_count=3,
    samples=(1.0, 2.0, 3.0),
    metrics=MeasurementMetrics(
        latency=LatencyMetrics(
            summary=summarize_samples((1.0, 2.0, 3.0)),
            unit=TimeUnit.MILLISECONDS,
        )
    ),
)
SUCCESS_OUTCOME = CaseOutcome.from_record(RECORD)

FAILURE = ProfilingFailure(
    category=ProfilingErrorCategory.DEVICE_BUSY,
    message="device is busy",
    details=(("device_id", "gpu-0"),),
)
FAILURE_OUTCOME = CaseOutcome.from_failure(FAILURE)
