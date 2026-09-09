"""WorkerProfilingRunner lifecycle tests (Phase 2 spec §37-39, §41-42, §44).

The runner is driven exactly like the servicer drives it: domain DTOs in,
domain DTOs out. Profilers are recording fakes (the real ones are tested in
``tests/unit/profiling``), so these tests pin the *orchestration* contract:

* §41 — every RPC rides the current registration or is refused;
* §39 — leases gate preparation and every device-bound run, and are
  released on close, preparation failure and shutdown (no leakage);
* §42 — benchmark problems are typed FAILED outcomes, transport verdicts
  are refusals, and a preparation failure keeps its category;
* §44/§50 — the ledger only moves forward and a duplicate run replays the
  recorded decision instead of re-benchmarking;
* §38 — a MODEL session loads its checkpoint once and answers retries from
  the cached facts; one test drives the *real* CPU loader chain against the
  tiny-llama fixture.
"""

from __future__ import annotations

import asyncio
import dataclasses
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from edgeshard.cluster.inventory import ModelAvailability, ModelInventoryEntry
from edgeshard.cluster.state import (
    DeviceAvailability,
    DeviceState,
    MemoryPoolState,
    WorkerState,
)
from edgeshard.control.worker import profiling_model_loader as model_loader_module
from edgeshard.control.worker.agent import LocalInspection
from edgeshard.control.worker.compute_executor import (
    ComputeExecutionEnvironment,
    PreparedComputeSession,
)
from edgeshard.control.worker.profiling_leases import DeviceLeaseManager
from edgeshard.control.worker.profiling_model_loader import (
    MODEL_EXTRACTION_SEQUENCE_LENGTHS,
    LoadedModelSession,
)
from edgeshard.control.worker.profiling_runner import (
    WorkerProfilingRunner,
    default_instrumentation,
)
from edgeshard.control.worker.profiling_sessions import (
    ProfilingSessionManager,
    ProfilingSessionRecord,
    RegistrationTokens,
)
from edgeshard.profiling.benchmark.harness import InstrumentationBundle
from edgeshard.profiling.domain.environment import environment_fingerprint_id
from edgeshard.profiling.domain.experiment import (
    CaseState,
    ModelCaseSpec,
    NetworkCaseSpec,
    ProfilingCase,
    ProfilingErrorCategory,
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
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.instrumentation.timing import WallClockTimer
from edgeshard.profiling.model.adapters.base import LayerReference, ProfileModule
from edgeshard.profiling.network.classifier import (
    InterfaceFacts,
    InterfaceKind,
    WorkerNetworkFacts,
)
from edgeshard.profiling.operator.extractor import RawOperatorGraph
from edgeshard.protocol.profiling.mapper import (
    CancelProfilingCaseRequest,
    CloseProfilingSessionRequest,
    GetProfilingCaseRequest,
    PrepareIperfServerRequest,
    PrepareProfilingSessionRequest,
    ProfilingRejection,
    RunProfilingCaseRequest,
    StopIperfServerRequest,
)
from factories import (
    RTX_GPU_DEVICE_ID,
    make_jetson_capability,
    make_rtx_capability,
    make_worker_identity,
    make_worker_state,
)

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)
LATER = datetime(2026, 9, 7, 12, 5, tzinfo=UTC)

WORKER_ID = "w-1"
PEER_ID = "w-2"
INSTANCE_ID = "instance-1"
REG_SESSION = "reg-1"
CPU_DEVICE = "cpu-dev-1"
SESSION_ID = "ps-1"

TOKENS = RegistrationTokens(
    worker_id=WORKER_ID, instance_id=INSTANCE_ID, registration_session_id=REG_SESSION
)

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
    structural_parameters=(("hidden_size", 32),),
    dtype="fp32",
    quantization=None,
)
OPERATOR_SIG = OperatorSignature(
    kind=OperatorKind.GEMM,
    parameters=GemmSignature(m=8, n=8, k=8, dtype="fp32", transpose_a=False),
    backend_family="torch",
)

CHARACTERIZATION = ModelCharacterization(
    model=MODEL,
    architecture_family="llama",
    num_layers=1,
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
        ModelStage(kind=StageKind.TRANSFORMER_LAYER_GROUP, layer_count=1),
        ModelStage(kind=StageKind.LM_HEAD),
    ),
)
STUB_FACTS = ModelSessionFacts(
    characterization=CHARACTERIZATION,
    layer_entries=(LayerEntry(0, "model.layers.0", LAYER_SIG),),
    module_entries=(ModuleEntry("mlp", "model.layers.0.mlp", ModuleKind.MLP, 0, MODULE_SIG),),
    operator_signatures=(OPERATOR_SIG,),
)

NETWORK_FACTS = (
    WorkerNetworkFacts(
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
    ),
    WorkerNetworkFacts(
        worker_id=PEER_ID,
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
    ),
)

MODEL_ENTRY = ModelInventoryEntry(
    local_name="tiny-llama",
    model_id="tiny/llama",
    revision="local",
    size_bytes=2**20,
    status=ModelAvailability.READY,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class StubInspector:
    """One canned inspection; counts calls (§37 shares one inspector)."""

    def __init__(self, state: WorkerState) -> None:
        self._inspection = LocalInspection(
            identity=make_worker_identity(WORKER_ID),
            capability=make_rtx_capability(),
            state=state,
        )
        self.inspect_count = 0

    async def inspect(self) -> LocalInspection:
        self.inspect_count += 1
        return self._inspection


def make_record(case: ProfilingCase) -> MeasurementRecord:
    samples = (1.0, 2.0, 3.0)
    return MeasurementRecord(
        measurement_id=f"m-{case.case_id[:8]}",
        case_id=case.case_id,
        environment_fingerprint="fp-test",
        started_at=NOW,
        finished_at=LATER,
        sample_count=len(samples),
        samples=samples,
        metrics=MeasurementMetrics(
            latency=LatencyMetrics(
                summary=summarize_samples(samples), unit=TimeUnit.MILLISECONDS
            )
        ),
    )


class RecordingProfiler:
    """Sync stand-in for the layer/module/operator profilers."""

    def __init__(self, error: Exception | None = None, hook=None) -> None:
        self.calls: list[ProfilingCase] = []
        self.kwargs: list[dict] = []
        self.error = error
        self.hook = hook

    def profile(self, case: ProfilingCase, *args, **kwargs) -> MeasurementRecord:
        self.calls.append(case)
        self.kwargs.append(kwargs)
        if self.hook is not None:
            self.hook(case)
        if self.error is not None:
            raise self.error
        return make_record(case)


class RecordingNetworkProfiler:
    """Async stand-in for NetworkProfiler."""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[ProfilingCase] = []
        self.facts: list[object] = []
        self.error = error
        self.server_starts: list[tuple[int, str | None]] = []
        self.server_stops: list[object] = []

    async def profile(
        self,
        case,
        *,
        network_facts,
        environment_fingerprint,
        iperf_server_port=None,
    ):
        self.calls.append(case)
        self.facts.append(network_facts)
        if self.error is not None:
            raise self.error
        return make_record(case)

    async def start_iperf_server(self, *, port: int, bind_address=None):
        self.server_starts.append((port, bind_address))
        return object()

    async def stop_iperf_server(self, process) -> None:
        self.server_stops.append(process)


class StubLoader:
    """Stands in for TorchModelSessionLoader; counts loads (§38 load once)."""

    def __init__(self, facts=STUB_FACTS, error: Exception | None = None) -> None:
        self.load_count = 0
        self.facts = facts
        self.error = error

    def load(self, request: ProfilingSessionRequest, source) -> LoadedModelSession:
        self.load_count += 1
        if self.error is not None:
            raise self.error
        return LoadedModelSession(
            source=source,
            model=nn.Identity(),
            layout=None,  # unused: the fake profilers never walk the tree
            adapter=None,
            layers=(LayerReference(index=0, module_path="model.layers.0", layer=nn.Identity()),),
            modules=(
                ProfileModule(
                    kind=ModuleKind.MLP,
                    name="mlp",
                    module_path="model.layers.0.mlp",
                    module=nn.Identity(),
                    signature=MODULE_SIG,
                ),
            ),
            facts=self.facts,
        )


class StubComputeExecutor:
    def __init__(
        self, environment: ComputeExecutionEnvironment, *, delay_s: float = 0.0
    ) -> None:
        self.environment = environment
        self.delay_s = delay_s
        self.closed = False

    def prepare_session(self, session_id, worker_id, request, source, capability):
        return PreparedComputeSession(
            session_id=session_id,
            target_device_id=request.device_ids[0],
            environment=self.environment,
        )

    def profile_case(self, session, case):
        time.sleep(self.delay_s)
        return make_record(case)

    def close_session(self, session) -> None:
        session.closed = True
        self.closed = True


@dataclasses.dataclass
class Rig:
    runner: WorkerProfilingRunner
    sessions: ProfilingSessionManager
    leases: DeviceLeaseManager
    inspector: StubInspector
    layer: RecordingProfiler
    module: RecordingProfiler
    operator: RecordingProfiler
    network: RecordingNetworkProfiler
    loader: StubLoader
    tokens: list


def make_state(
    *,
    devices: tuple[DeviceState, ...] | None = None,
    models: tuple[ModelInventoryEntry, ...] = (),
) -> WorkerState:
    if devices is None:
        devices = (
            DeviceState(
                device_id=CPU_DEVICE,
                utilization=0.0,
                temperature_c=40.0,
                power_w=None,
                availability=DeviceAvailability.AVAILABLE,
                running_runtime_ids=(),
            ),
        )
    return WorkerState(
        worker_id=WORKER_ID,
        device_states=devices,
        memory_states=(),
        runtime_instances=(),
        models=models,
    )


def make_rig(
    state: WorkerState | None = None,
    *,
    loader: StubLoader | None = None,
    model_store_root: Path | None = None,
    real_loader: bool = False,
    compute_executor=None,
) -> Rig:
    tokens: list = [TOKENS]
    sessions = ProfilingSessionManager(token_source=lambda: tokens[0])
    rig_loader = loader if loader is not None else StubLoader()
    layer, module = RecordingProfiler(), RecordingProfiler()
    operator, network = RecordingProfiler(), RecordingNetworkProfiler()
    inspector = StubInspector(state if state is not None else make_state())
    leases = DeviceLeaseManager()
    kwargs: dict = {}
    if not real_loader:
        # The real-chain test leaves the factory unset so the runner uses
        # its default TorchModelSessionLoader against the tmp ModelStore.
        kwargs["model_loader_factory"] = lambda device: rig_loader
    if compute_executor is not None:
        kwargs["compute_executor"] = compute_executor
    runner = WorkerProfilingRunner(
        sessions=sessions,
        inspector=inspector,
        model_store_root=model_store_root or Path("unused-model-root"),
        leases=leases,
        layer_profiler=layer,
        module_profiler=module,
        operator_profiler=operator,
        network_profiler=network,
        device_resolver=lambda device_id, worker_id: torch.device("cpu"),
        clock=lambda: NOW,
        **kwargs,
    )
    return Rig(
        runner=runner,
        sessions=sessions,
        leases=leases,
        inspector=inspector,
        layer=layer,
        module=module,
        operator=operator,
        network=network,
        loader=rig_loader,
        tokens=tokens,
    )


def fields(**extra):
    return {
        "worker_id": WORKER_ID,
        "instance_id": INSTANCE_ID,
        "registration_session_id": REG_SESSION,
        **extra,
    }


def prepare_request(
    session_request: ProfilingSessionRequest, *, session_id: str = SESSION_ID, network_facts=()
) -> PrepareProfilingSessionRequest:
    return PrepareProfilingSessionRequest(
        **fields(profiling_session_id=session_id),
        session_request=session_request,
        network_facts=network_facts,
    )


OPERATOR_SESSION = ProfilingSessionRequest(
    kind=ProfilingSessionKind.OPERATOR, device_ids=(CPU_DEVICE,)
)
MODEL_SESSION = ProfilingSessionRequest(
    kind=ProfilingSessionKind.MODEL, device_ids=(CPU_DEVICE,), model=MODEL, dtype="fp32"
)
NETWORK_SESSION = ProfilingSessionRequest(kind=ProfilingSessionKind.NETWORK)


def operator_case(device_ids: tuple[str, ...] = (CPU_DEVICE,)) -> ProfilingCase:
    return ProfilingCase.for_spec(
        WORKER_ID,
        ModelCaseSpec(
            granularity=ProfilingGranularity.OPERATOR,
            device_ids=device_ids,
            dtype="fp32",
            operator_signature=OPERATOR_SIG,
        ),
    )


def layer_case(signature=LAYER_SIG, layer_index: int | None = 0) -> ProfilingCase:
    return ProfilingCase.for_spec(
        WORKER_ID,
        ModelCaseSpec(
            granularity=ProfilingGranularity.TRANSFORMER_LAYER,
            device_ids=(CPU_DEVICE,),
            dtype="fp32",
            model=MODEL,
            layer_signature=signature,
            layer_index=layer_index,
            batch_size=1,
            sequence_length=16,
        ),
    )


def module_case() -> ProfilingCase:
    return ProfilingCase.for_spec(
        WORKER_ID,
        ModelCaseSpec(
            granularity=ProfilingGranularity.MODULE,
            device_ids=(CPU_DEVICE,),
            dtype="fp32",
            model=MODEL,
            module_signature=MODULE_SIG,
        ),
    )


def network_case(destination: str = PEER_ID) -> ProfilingCase:
    return ProfilingCase.for_spec(
        WORKER_ID,
        NetworkCaseSpec(
            probe_kind=ProbeKind.RTT,
            source_worker_id=WORKER_ID,
            destination_worker_id=destination,
            packet_count=3,
        ),
    )


def run_request(case: ProfilingCase, session_id: str = SESSION_ID) -> RunProfilingCaseRequest:
    return RunProfilingCaseRequest(**fields(profiling_session_id=session_id), case=case)


async def prepare(rig: Rig, request=OPERATOR_SESSION, **kwargs):
    return await rig.runner.prepare_profiling_session(
        prepare_request(request, **kwargs)
    )


# ---------------------------------------------------------------------------
# §41 registration gate on every RPC
# ---------------------------------------------------------------------------


async def test_every_rpc_refuses_without_active_registration() -> None:
    rig = make_rig()
    rig.tokens[0] = None
    case = operator_case()
    prepared = await rig.runner.prepare_profiling_session(prepare_request(OPERATOR_SESSION))
    ran = await rig.runner.run_profiling_case(run_request(case))
    got = await rig.runner.get_profiling_case(
        GetProfilingCaseRequest(**fields(profiling_session_id=SESSION_ID, case_id=case.case_id))
    )
    cancelled = await rig.runner.cancel_profiling_case(
        CancelProfilingCaseRequest(
            **fields(profiling_session_id=SESSION_ID, case_id=case.case_id)
        )
    )
    closed = await rig.runner.close_profiling_session(
        CloseProfilingSessionRequest(**fields(profiling_session_id=SESSION_ID))
    )
    for response in (prepared, ran, got, cancelled, closed):
        assert response.accepted is False
        assert response.reason is ProfilingRejection.STALE_SESSION


async def test_superseded_registration_refuses() -> None:
    rig = make_rig()
    rig.tokens[0] = dataclasses.replace(TOKENS, registration_session_id="reg-2")
    response = await prepare(rig)
    assert response.accepted is False
    assert response.reason is ProfilingRejection.STALE_SESSION


# ---------------------------------------------------------------------------
# PrepareProfilingSession (§38, §39)
# ---------------------------------------------------------------------------


async def test_prepare_operator_session_leases_devices() -> None:
    rig = make_rig()
    response = await prepare(rig)
    assert response.accepted is True
    assert response.session_facts is None  # operator sessions publish no facts
    assert rig.leases.leased_device_ids == (CPU_DEVICE,)
    assert rig.inspector.inspect_count == 1  # preparation reads *fresh* state
    record = rig.sessions.peek(SESSION_ID)
    assert record is not None
    assert record.capability_revision  # taken from the inspection's capability


async def test_prepare_busy_device_is_refused_without_session() -> None:
    busy = make_state(
        devices=(
            DeviceState(
                device_id=CPU_DEVICE,
                utilization=93.0,
                temperature_c=70.0,
                power_w=None,
                availability=DeviceAvailability.AVAILABLE,
                running_runtime_ids=(),
            ),
        )
    )
    rig = make_rig(busy)
    response = await prepare(rig)
    assert response.accepted is False
    assert response.reason is ProfilingRejection.DEVICE_BUSY
    assert CPU_DEVICE in response.detail
    assert rig.sessions.peek(SESSION_ID) is None
    assert rig.leases.leased_device_ids == ()


async def test_prepare_model_session_loads_once_and_replays() -> None:
    rig = make_rig(make_state(models=(MODEL_ENTRY,)))
    first = await prepare(rig, MODEL_SESSION)
    assert first.accepted is True
    assert first.session_facts is not None
    assert dataclasses.replace(first.session_facts, environment=None) == STUB_FACTS
    assert first.session_facts.environment is not None
    assert rig.loader.load_count == 1
    # A retried prepare (lost response) replays the cached session: no
    # second checkpoint load (§37), identical facts (§6 determinism).
    second = await prepare(rig, MODEL_SESSION)
    assert second.accepted is True
    assert second.session_facts == first.session_facts
    assert rig.loader.load_count == 1


async def test_prepare_retry_of_closed_session_is_refused() -> None:
    rig = make_rig()
    assert (await prepare(rig)).accepted is True
    close = await rig.runner.close_profiling_session(
        CloseProfilingSessionRequest(**fields(profiling_session_id=SESSION_ID))
    )
    assert close.accepted is True
    retry = await prepare(rig)
    assert retry.accepted is False
    assert retry.reason is ProfilingRejection.SESSION_CLOSED


async def test_prepare_kind_mismatch_on_existing_id_is_refused() -> None:
    rig = make_rig(make_state(models=(MODEL_ENTRY,)))
    assert (await prepare(rig, MODEL_SESSION)).accepted is True
    other = await prepare(rig, OPERATOR_SESSION)
    assert other.accepted is False
    assert other.reason is ProfilingRejection.SESSION_KIND_MISMATCH
    # The prepared MODEL session is untouched.
    assert rig.sessions.peek(SESSION_ID).kind is ProfilingSessionKind.MODEL


async def test_prepare_unsupported_model_keeps_typed_failure() -> None:
    """§42: the category survives the runner; §39: the lease does not leak."""
    rig = make_rig(make_state(models=()))  # inventory lacks tiny/llama
    response = await prepare(rig, MODEL_SESSION)
    assert response.accepted is False
    assert response.reason is None
    assert response.failure is not None
    assert response.failure.category is ProfilingErrorCategory.UNSUPPORTED_MODEL
    assert "tiny/llama" in response.detail
    assert rig.leases.leased_device_ids == ()  # released on the failure path
    assert rig.sessions.peek(SESSION_ID) is None


async def test_prepare_loader_error_becomes_typed_failure() -> None:
    boom = ProfilingError(
        ProfilingErrorCategory.EXPORT_FAILED, "operator export exploded"
    )
    rig = make_rig(make_state(models=(MODEL_ENTRY,)), loader=StubLoader(error=boom))
    response = await prepare(rig, MODEL_SESSION)
    assert response.accepted is False
    assert response.failure is not None
    assert response.failure.category is ProfilingErrorCategory.EXPORT_FAILED
    assert rig.leases.leased_device_ids == ()


async def test_prepare_unexpected_error_becomes_internal_failure() -> None:
    rig = make_rig(
        make_state(models=(MODEL_ENTRY,)), loader=StubLoader(error=RuntimeError("kaboom"))
    )
    response = await prepare(rig, MODEL_SESSION)
    assert response.accepted is False
    assert response.failure is not None
    assert response.failure.category is ProfilingErrorCategory.INTERNAL_ERROR
    assert "kaboom" in response.failure.message
    assert rig.leases.leased_device_ids == ()


async def test_prepare_network_session_requires_own_facts() -> None:
    """§52.2: the executing worker's probe addresses come from the Master."""
    rig = make_rig()
    peer_only = (NETWORK_FACTS[1],)
    with pytest.raises(ValueError, match="executing worker"):
        await prepare(rig, NETWORK_SESSION, network_facts=peer_only)
    assert rig.sessions.peek(SESSION_ID) is None


async def test_prepare_network_session_stores_master_facts() -> None:
    rig = make_rig()
    response = await prepare(rig, NETWORK_SESSION, network_facts=NETWORK_FACTS)
    assert response.accepted is True
    assert rig.leases.leased_device_ids == ()  # network sessions lease nothing
    record = rig.sessions.peek(SESSION_ID)
    assert set(record.network_facts) == {WORKER_ID, PEER_ID}


# ---------------------------------------------------------------------------
# RunProfilingCase (§39, §42, §44, §50)
# ---------------------------------------------------------------------------


async def test_run_operator_case_completes() -> None:
    rig = make_rig()
    assert (await prepare(rig)).accepted is True
    case = operator_case()
    response = await rig.runner.run_profiling_case(run_request(case))
    assert response.accepted is True
    assert response.outcome is not None
    assert response.outcome.succeeded
    assert response.outcome.record.case_id == case.case_id
    assert len(rig.operator.calls) == 1
    entry = rig.sessions.case_entry(SESSION_ID, case.case_id)
    assert entry.state is CaseState.COMPLETED
    assert entry.outcome == response.outcome
    # The profiler received a real fingerprint and a resolved torch device.
    kwargs = rig.operator.kwargs[0]
    assert kwargs["environment_fingerprint"]
    assert kwargs["device"] == torch.device("cpu")
    assert kwargs["instrumentation"] is not None


async def test_second_gpu_case_execution_and_fingerprint_agree() -> None:
    state = make_state(
        devices=(
            DeviceState(
                device_id="gpu-0",
                utilization=0.0,
                temperature_c=40.0,
                power_w=None,
                availability=DeviceAvailability.AVAILABLE,
                running_runtime_ids=(),
            ),
            DeviceState(
                device_id="gpu-1",
                utilization=0.0,
                temperature_c=40.0,
                power_w=None,
                availability=DeviceAvailability.AVAILABLE,
                running_runtime_ids=(),
            ),
        )
    )
    rig = make_rig(state)
    rig.runner._device_resolver = lambda device_id, worker_id: torch.device(
        "cuda", int(device_id[-1])
    )
    rig.runner._instrumentation_factory = lambda device: InstrumentationBundle(
        timer=WallClockTimer()
    )
    session = ProfilingSessionRequest(
        kind=ProfilingSessionKind.OPERATOR, device_ids=("gpu-1",)
    )
    assert (await prepare(rig, session)).accepted
    case = operator_case(device_ids=("gpu-1",))

    response = await rig.runner.run_profiling_case(run_request(case))

    assert response.outcome.succeeded
    record = response.outcome.record
    assert record is not None and record.environment is not None
    assert rig.operator.kwargs[0]["device"] == torch.device("cuda", 1)
    assert case.spec.device_ids == ("gpu-1",)
    assert record.environment.device_id == "gpu-1"
    assert record.environment_fingerprint == environment_fingerprint_id(
        record.environment
    )


async def test_compute_fingerprint_uses_container_software_environment() -> None:
    environment = ComputeExecutionEnvironment(
        torch_version="2.13.0+cu126-container",
        cuda_version="12.6-container",
        backend_revision="sha256:runtime-image",
        target_device_id=RTX_GPU_DEVICE_ID,
        execution_device="cuda:0",
    )
    executor = StubComputeExecutor(environment)
    state = make_state(
        devices=(
            DeviceState(
                device_id=RTX_GPU_DEVICE_ID,
                utilization=0.0,
                temperature_c=40.0,
                power_w=None,
                availability=DeviceAvailability.AVAILABLE,
                running_runtime_ids=(),
            ),
        )
    )
    rig = make_rig(state, compute_executor=executor)
    session = ProfilingSessionRequest(
        kind=ProfilingSessionKind.OPERATOR,
        device_ids=(RTX_GPU_DEVICE_ID,),
    )
    assert (await prepare(rig, session)).accepted
    case = operator_case(device_ids=(RTX_GPU_DEVICE_ID,))

    response = await rig.runner.run_profiling_case(run_request(case))

    assert response.outcome is not None and response.outcome.record is not None
    fingerprint = response.outcome.record.environment
    assert fingerprint is not None
    assert fingerprint.torch_version == "2.13.0+cu126-container"
    assert fingerprint.cuda_version == "12.6-container"
    assert fingerprint.backend_revision == "sha256:runtime-image"
    assert fingerprint.worker_id == WORKER_ID
    assert fingerprint.device_id == RTX_GPU_DEVICE_ID
    assert fingerprint.driver_version == "550.90"
    assert fingerprint.device_performance_class is not None
    assert dict(fingerprint.device_performance_class.software_versions) == {
        "cuda": "12.6-container",
        "torch": "2.13.0",
    }


async def test_worker_merges_fresh_physical_telemetry_around_compute_executor() -> None:
    capability = make_rtx_capability()
    base_state = dataclasses.replace(
        make_worker_state(WORKER_ID),
        device_states=(
            dataclasses.replace(
                make_worker_state(WORKER_ID).device_states[0], utilization=0.0
            ),
        ),
        runtime_instances=(),
    )

    class FreshInspector:
        def __init__(self) -> None:
            self.fresh_calls = 0

        async def inspect(self) -> LocalInspection:
            return LocalInspection(
                identity=make_worker_identity(WORKER_ID),
                capability=capability,
                state=base_state,
            )

        def sample_fresh_state(self) -> WorkerState:
            self.fresh_calls += 1
            available = max(1, 20 * 2**30 - self.fresh_calls * 2**20)
            return dataclasses.replace(
                base_state,
                memory_states=tuple(
                    dataclasses.replace(item, available_bytes=available)
                    if item.memory_pool_id == capability.devices[1].memory_pool_id
                    else item
                    for item in base_state.memory_states
                ),
            )

    environment = ComputeExecutionEnvironment(
        torch_version="2.13.0+cu126",
        cuda_version="12.6",
        backend_revision="sha256:runtime-image",
        target_device_id=RTX_GPU_DEVICE_ID,
        execution_device="cuda:0",
    )
    executor = StubComputeExecutor(environment, delay_s=0.12)
    inspector = FreshInspector()
    runner = WorkerProfilingRunner(
        sessions=ProfilingSessionManager(token_source=lambda: TOKENS),
        inspector=inspector,
        model_store_root=Path("unused-model-root"),
        compute_executor=executor,
        clock=lambda: NOW,
    )
    session = ProfilingSessionRequest(
        kind=ProfilingSessionKind.OPERATOR,
        device_ids=(RTX_GPU_DEVICE_ID,),
    )
    prepared = await runner.prepare_profiling_session(prepare_request(session))
    assert prepared.accepted

    response = await runner.run_profiling_case(
        run_request(operator_case(device_ids=(RTX_GPU_DEVICE_ID,)))
    )

    assert response.outcome is not None and response.outcome.record is not None
    metrics = response.outcome.record.metrics
    assert metrics.physical_memory is not None
    assert metrics.physical_memory.pool_id == capability.devices[1].memory_pool_id
    assert metrics.physical_memory.used_peak is not None
    assert metrics.telemetry is not None
    assert metrics.telemetry.initial is not None
    assert metrics.telemetry.final is not None
    assert inspector.fresh_calls >= 5


def test_default_instrumentation_reuses_phase1_shared_jetson_pool() -> None:
    capability = make_jetson_capability()
    state = make_worker_state(
        WORKER_ID, device_ids=("gpu-system",), pool_ids=("system-memory",)
    )
    state_holder = [state]
    fresh_calls = 0

    def fresh_state() -> WorkerState:
        nonlocal fresh_calls
        fresh_calls += 1
        return state_holder[0]

    record = ProfilingSessionRecord(
        session_id="jetson-session",
        request=ProfilingSessionRequest(
            kind=ProfilingSessionKind.OPERATOR, device_ids=("gpu-system",)
        ),
        prepared_at=NOW,
        capability_revision=capability.capability_revision,
        capability=capability,
        worker_state=state,
        worker_state_source=fresh_state,
    )

    bundle = default_instrumentation(
        torch.device("cpu"), device_id="gpu-system", record=record
    )

    assert bundle.physical_memory is not None
    assert bundle.telemetry is not None
    assert {device.memory_pool_id for device in capability.devices} == {
        "system-memory"
    }
    assert len(capability.memory_pools) == 1
    bundle.physical_memory.open()
    state_holder[0] = dataclasses.replace(
        state,
        device_states=(dataclasses.replace(state.device_states[0], utilization=77.0),),
        memory_states=(
            MemoryPoolState(
                memory_pool_id="system-memory", available_bytes=12 * 2**30
            ),
        ),
    )
    bundle.physical_memory.poll()
    state_holder[0] = dataclasses.replace(
        state_holder[0],
        memory_states=(
            MemoryPoolState(
                memory_pool_id="system-memory", available_bytes=17 * 2**30
            ),
        ),
    )
    physical = bundle.physical_memory.close()
    bundle.telemetry.capture_initial()
    state_holder[0] = dataclasses.replace(
        state_holder[0],
        device_states=(
            dataclasses.replace(state_holder[0].device_states[0], utilization=77.0),
        ),
    )
    telemetry = bundle.telemetry.capture_final()
    assert physical.pool_id == "system-memory"
    assert physical.used_after is not None and physical.used_before is not None
    assert physical.used_after > physical.used_before
    assert physical.used_peak is not None
    assert physical.used_peak > physical.used_after
    assert fresh_calls == 5
    assert telemetry is not None
    assert telemetry.initial is not None
    assert telemetry.initial.device_id == "gpu-system"
    assert telemetry.final is not None and telemetry.final.utilization == 77.0


def test_default_instrumentation_maps_phase1_nvml_gpu_state() -> None:
    capability = make_rtx_capability()
    state = make_worker_state(WORKER_ID)
    record = ProfilingSessionRecord(
        session_id="rtx-session",
        request=ProfilingSessionRequest(
            kind=ProfilingSessionKind.OPERATOR,
            device_ids=(RTX_GPU_DEVICE_ID,),
        ),
        prepared_at=NOW,
        capability_revision=capability.capability_revision,
        capability=capability,
        worker_state=state,
        worker_state_source=lambda: state,
    )

    bundle = default_instrumentation(
        torch.device("cpu"), device_id=RTX_GPU_DEVICE_ID, record=record
    )
    assert bundle.physical_memory is not None
    assert bundle.telemetry is not None
    bundle.telemetry.capture_initial()
    telemetry = bundle.telemetry.capture_final()
    assert telemetry is not None and telemetry.initial is not None
    assert telemetry.initial.device_id == RTX_GPU_DEVICE_ID
    assert telemetry.initial.utilization == 21.0


def test_default_instrumentation_does_not_claim_cached_physical_peak() -> None:
    capability = make_rtx_capability()
    state = make_worker_state(WORKER_ID)
    record = ProfilingSessionRecord(
        session_id="cached-only",
        request=ProfilingSessionRequest(
            kind=ProfilingSessionKind.OPERATOR,
            device_ids=(RTX_GPU_DEVICE_ID,),
        ),
        prepared_at=NOW,
        capability=capability,
        worker_state=state,
    )

    bundle = default_instrumentation(
        torch.device("cpu"), device_id=RTX_GPU_DEVICE_ID, record=record
    )

    assert bundle.physical_memory is None
    assert bundle.telemetry is None


async def test_duplicate_run_replays_recorded_outcome() -> None:
    """§50: a duplicate dispatch never re-benchmarks."""
    rig = make_rig()
    assert (await prepare(rig)).accepted is True
    request = run_request(operator_case())
    first = await rig.runner.run_profiling_case(request)
    second = await rig.runner.run_profiling_case(request)
    assert second.accepted is True
    assert second.outcome == first.outcome
    assert len(rig.operator.calls) == 1


async def test_run_unknown_session_is_refused() -> None:
    rig = make_rig()
    response = await rig.runner.run_profiling_case(run_request(operator_case()))
    assert response.accepted is False
    assert response.reason is ProfilingRejection.UNKNOWN_SESSION


async def test_run_after_close_is_refused() -> None:
    rig = make_rig()
    assert (await prepare(rig)).accepted is True
    await rig.runner.close_profiling_session(
        CloseProfilingSessionRequest(**fields(profiling_session_id=SESSION_ID))
    )
    response = await rig.runner.run_profiling_case(run_request(operator_case()))
    assert response.accepted is False
    assert response.reason is ProfilingRejection.SESSION_CLOSED


async def test_run_wrong_kind_case_is_refused() -> None:
    rig = make_rig()
    assert (await prepare(rig)).accepted is True  # OPERATOR session
    response = await rig.runner.run_profiling_case(run_request(network_case()))
    assert response.accepted is False
    assert response.reason is ProfilingRejection.SESSION_KIND_MISMATCH
    # And a model-free operator case is refused by a MODEL session.
    rig2 = make_rig(make_state(models=(MODEL_ENTRY,)))
    assert (await prepare(rig2, MODEL_SESSION)).accepted is True
    response2 = await rig2.runner.run_profiling_case(run_request(operator_case()))
    assert response2.accepted is False
    assert response2.reason is ProfilingRejection.SESSION_KIND_MISMATCH


async def test_run_on_unleased_device_is_refused() -> None:
    """§39: the session's lease set is the run's authority."""
    rig = make_rig()
    assert (await prepare(rig)).accepted is True
    response = await rig.runner.run_profiling_case(
        run_request(operator_case(device_ids=("cpu-dev-2",)))
    )
    assert response.accepted is False
    assert response.reason is ProfilingRejection.DEVICE_BUSY
    assert "cpu-dev-2" in response.detail
    assert rig.operator.calls == []


async def test_profiling_error_becomes_typed_failed_outcome() -> None:
    rig = make_rig()
    rig.operator = RecordingProfiler(
        error=ProfilingError(ProfilingErrorCategory.BENCHMARK_FAILED, "benchmark exploded")
    )
    rig.runner._operator_profiler = rig.operator
    assert (await prepare(rig)).accepted is True
    case = operator_case()
    response = await rig.runner.run_profiling_case(run_request(case))
    assert response.accepted is True  # a failure is a result, not a refusal
    assert response.outcome.failure.category is ProfilingErrorCategory.BENCHMARK_FAILED
    entry = rig.sessions.case_entry(SESSION_ID, case.case_id)
    assert entry.state is CaseState.FAILED


async def test_unexpected_error_becomes_internal_failed_outcome() -> None:
    rig = make_rig()
    rig.operator = RecordingProfiler(error=RuntimeError("kaboom"))
    rig.runner._operator_profiler = rig.operator
    assert (await prepare(rig)).accepted is True
    response = await rig.runner.run_profiling_case(run_request(operator_case()))
    assert response.accepted is True
    assert response.outcome.failure.category is ProfilingErrorCategory.INTERNAL_ERROR
    assert "kaboom" in response.outcome.failure.message


# ---------------------------------------------------------------------------
# Cancellation (§41, §44, §50)
# ---------------------------------------------------------------------------


async def test_cancel_before_run_replays_cancelled_outcome() -> None:
    rig = make_rig()
    assert (await prepare(rig)).accepted is True
    case = operator_case()
    cancel_fields = fields(profiling_session_id=SESSION_ID, case_id=case.case_id)
    cancelled = await rig.runner.cancel_profiling_case(
        CancelProfilingCaseRequest(**cancel_fields)
    )
    assert cancelled.accepted is True
    assert cancelled.case_state is CaseState.CANCELLED
    # The later duplicate run replays the cancellation (§50) and never
    # benchmarks (§44: terminal history is final).
    ran = await rig.runner.run_profiling_case(run_request(case))
    assert ran.accepted is True
    assert ran.outcome.failure.category is ProfilingErrorCategory.CANCELLED
    assert rig.operator.calls == []
    got = await rig.runner.get_profiling_case(GetProfilingCaseRequest(**cancel_fields))
    assert got.case_state is CaseState.CANCELLED


async def test_cancel_during_run_discards_measurement() -> None:
    """§41: a mid-run cancellation wins; the finished record is never published."""
    rig = make_rig()
    assert (await prepare(rig)).accepted is True

    def cancel_mid_run(case: ProfilingCase) -> None:
        rig.sessions.cancel_case(SESSION_ID, case.case_id, now=LATER)

    rig.operator = RecordingProfiler(hook=cancel_mid_run)
    rig.runner._operator_profiler = rig.operator
    response = await rig.runner.run_profiling_case(run_request(operator_case()))
    assert response.accepted is True
    assert response.outcome.failure is not None
    assert response.outcome.failure.category is ProfilingErrorCategory.CANCELLED
    assert response.outcome.record is None  # the measurement was discarded
    assert len(rig.operator.calls) == 1  # it did run — and its result was dropped


async def test_cancel_completed_case_keeps_history() -> None:
    rig = make_rig()
    assert (await prepare(rig)).accepted is True
    case = operator_case()
    ran = await rig.runner.run_profiling_case(run_request(case))
    assert ran.outcome.succeeded
    cancelled = await rig.runner.cancel_profiling_case(
        CancelProfilingCaseRequest(**fields(profiling_session_id=SESSION_ID, case_id=case.case_id))
    )
    assert cancelled.accepted is True
    assert cancelled.case_state is CaseState.COMPLETED  # §44: never rewritten


async def test_cancel_unknown_session_is_refused() -> None:
    rig = make_rig()
    response = await rig.runner.cancel_profiling_case(
        CancelProfilingCaseRequest(**fields(profiling_session_id=SESSION_ID, case_id="c"))
    )
    assert response.accepted is False
    assert response.reason is ProfilingRejection.UNKNOWN_SESSION


# ---------------------------------------------------------------------------
# GetProfilingCase (§44 ledger reads)
# ---------------------------------------------------------------------------


async def test_get_unknown_case_is_refused() -> None:
    rig = make_rig()
    assert (await prepare(rig)).accepted is True
    response = await rig.runner.get_profiling_case(
        GetProfilingCaseRequest(**fields(profiling_session_id=SESSION_ID, case_id="nope"))
    )
    assert response.accepted is False
    assert response.reason is ProfilingRejection.UNKNOWN_CASE


async def test_get_tracks_case_state() -> None:
    rig = make_rig()
    assert (await prepare(rig)).accepted is True
    case = operator_case()
    get_fields = fields(profiling_session_id=SESSION_ID, case_id=case.case_id)
    await rig.runner.run_profiling_case(run_request(case))
    response = await rig.runner.get_profiling_case(GetProfilingCaseRequest(**get_fields))
    assert response.accepted is True
    assert response.case_state is CaseState.COMPLETED
    assert response.outcome is not None and response.outcome.succeeded


# ---------------------------------------------------------------------------
# MODEL-session execution through the stub checkpoint
# ---------------------------------------------------------------------------


async def test_model_layer_and_module_cases_complete() -> None:
    rig = make_rig(make_state(models=(MODEL_ENTRY,)))
    prepared = await prepare(rig, MODEL_SESSION)
    assert prepared.accepted is True
    facts = prepared.session_facts

    layer = layer_case(signature=facts.layer_entries[0].signature)
    layer_response = await rig.runner.run_profiling_case(run_request(layer))
    assert layer_response.accepted is True
    assert layer_response.outcome.succeeded
    assert len(rig.layer.calls) == 1

    module = module_case()
    module_response = await rig.runner.run_profiling_case(run_request(module))
    assert module_response.accepted is True
    assert module_response.outcome.succeeded
    assert len(rig.module.calls) == 1


async def test_layer_case_without_index_fails_typed() -> None:
    """§22/§52.2: the runner never defaults to 'some' layer."""
    rig = make_rig(make_state(models=(MODEL_ENTRY,)))
    assert (await prepare(rig, MODEL_SESSION)).accepted is True
    response = await rig.runner.run_profiling_case(
        run_request(layer_case(layer_index=None))
    )
    assert response.accepted is True
    assert (
        response.outcome.failure.category
        is ProfilingErrorCategory.UNSUPPORTED_GRANULARITY
    )
    assert rig.layer.calls == []


async def test_layer_case_out_of_range_index_fails_typed() -> None:
    rig = make_rig(make_state(models=(MODEL_ENTRY,)))
    assert (await prepare(rig, MODEL_SESSION)).accepted is True
    response = await rig.runner.run_profiling_case(run_request(layer_case(layer_index=7)))
    assert response.accepted is True
    assert (
        response.outcome.failure.category
        is ProfilingErrorCategory.UNSUPPORTED_GRANULARITY
    )


async def test_layer_case_contradicting_session_facts_fails() -> None:
    """§41: a case whose signature contradicts the published facts fails."""
    rig = make_rig(make_state(models=(MODEL_ENTRY,)))
    assert (await prepare(rig, MODEL_SESSION)).accepted is True
    wrong = dataclasses.replace(LAYER_SIG, hidden_size=999)
    response = await rig.runner.run_profiling_case(run_request(layer_case(signature=wrong)))
    assert response.accepted is True
    assert response.outcome.failure.category is ProfilingErrorCategory.INTERNAL_ERROR
    assert "contradicts" in response.outcome.failure.message
    assert rig.layer.calls == []


async def test_module_case_without_matching_module_fails_typed() -> None:
    rig = make_rig(make_state(models=(MODEL_ENTRY,)))
    assert (await prepare(rig, MODEL_SESSION)).accepted is True
    attention = ModuleSignature(
        kind=ModuleKind.ATTENTION,
        architecture_family="llama",
        structural_parameters=(("hidden_size", 32),),
        dtype="fp32",
        quantization=None,
    )
    case = ProfilingCase.for_spec(
        WORKER_ID,
        ModelCaseSpec(
            granularity=ProfilingGranularity.MODULE,
            device_ids=(CPU_DEVICE,),
            dtype="fp32",
            model=MODEL,
            module_signature=attention,
        ),
    )
    response = await rig.runner.run_profiling_case(run_request(case))
    assert response.accepted is True
    assert (
        response.outcome.failure.category
        is ProfilingErrorCategory.UNSUPPORTED_GRANULARITY
    )


# ---------------------------------------------------------------------------
# NETWORK-session execution
# ---------------------------------------------------------------------------


async def test_network_case_completes_with_master_facts() -> None:
    rig = make_rig()
    assert (
        await prepare(rig, NETWORK_SESSION, network_facts=NETWORK_FACTS)
    ).accepted is True
    response = await rig.runner.run_profiling_case(run_request(network_case()))
    assert response.accepted is True
    assert response.outcome.succeeded
    assert len(rig.network.calls) == 1
    assert set(rig.network.facts[0]) == {WORKER_ID, PEER_ID}


async def test_network_case_unknown_destination_fails_typed() -> None:
    rig = make_rig()
    assert (
        await prepare(rig, NETWORK_SESSION, network_facts=NETWORK_FACTS)
    ).accepted is True
    response = await rig.runner.run_profiling_case(
        run_request(network_case(destination="w-ghost"))
    )
    assert response.accepted is True
    assert (
        response.outcome.failure.category is ProfilingErrorCategory.NETWORK_UNREACHABLE
    )
    assert rig.network.calls == []


async def test_temporary_iperf_server_is_ready_idempotent_and_cleaned_up() -> None:
    rig = make_rig()
    request = PrepareIperfServerRequest(
        **fields(),
        server_id="case-server",
        port=45678,
        timeout_s=30.0,
        bind_address="127.0.0.1",
    )
    first = await rig.runner.prepare_iperf_server(request)
    replay = await rig.runner.prepare_iperf_server(request)

    assert first.accepted and replay.accepted
    assert first.port == replay.port == 45678
    assert rig.network.server_starts == [(45678, "127.0.0.1")]

    stopped = await rig.runner.stop_iperf_server(
        StopIperfServerRequest(**fields(), server_id="case-server")
    )
    assert stopped.accepted
    assert len(rig.network.server_stops) == 1


async def test_temporary_iperf_server_timeout_prevents_process_leak() -> None:
    rig = make_rig()
    response = await rig.runner.prepare_iperf_server(
        PrepareIperfServerRequest(
            **fields(), server_id="expiring-server", port=45679, timeout_s=0.01
        )
    )
    assert response.accepted
    await asyncio.sleep(0.03)
    assert len(rig.network.server_stops) == 1


# ---------------------------------------------------------------------------
# Close and shutdown (§38, §39: no lease outlives its session or the runner)
# ---------------------------------------------------------------------------


async def test_close_releases_leases_and_is_idempotent() -> None:
    rig = make_rig()
    assert (await prepare(rig)).accepted is True
    assert rig.leases.leased_device_ids == (CPU_DEVICE,)
    request = CloseProfilingSessionRequest(**fields(profiling_session_id=SESSION_ID))
    first = await rig.runner.close_profiling_session(request)
    second = await rig.runner.close_profiling_session(request)
    assert first.accepted is True and second.accepted is True
    assert rig.leases.leased_device_ids == ()
    assert rig.sessions.open_session_count() == 0


async def test_close_unknown_session_is_accepted() -> None:
    rig = make_rig()
    response = await rig.runner.close_profiling_session(
        CloseProfilingSessionRequest(**fields(profiling_session_id="never-prepared"))
    )
    assert response.accepted is True


async def test_shutdown_closes_everything_and_leaks_no_lease() -> None:
    rig = make_rig()
    assert (await prepare(rig, OPERATOR_SESSION, session_id="ps-a")).accepted is True
    assert (
        await prepare(
            rig, NETWORK_SESSION, session_id="ps-b", network_facts=NETWORK_FACTS
        )
    ).accepted is True
    assert rig.leases.leased_device_ids == (CPU_DEVICE,)
    await rig.runner.shutdown()
    assert rig.leases.leased_device_ids == ()
    assert rig.sessions.open_session_count() == 0


# ---------------------------------------------------------------------------
# The real CPU loader chain (§38, §51): tiny-llama from the ModelStore
# ---------------------------------------------------------------------------


@pytest.fixture
def model_store_with_tiny_llama(
    tmp_path: Path, tiny_llama_dir: Path
) -> tuple[Path, WorkerState]:
    root = tmp_path / "models"
    shutil.copytree(tiny_llama_dir, root / "tiny-llama")
    state = make_state(
        models=(
            ModelInventoryEntry(
                local_name="tiny-llama",
                model_id="tiny/llama",
                revision="main",
                size_bytes=2**20,
                status=ModelAvailability.READY,
            ),
        )
    )
    return root, state


async def test_real_model_session_prepare_and_layer_run(
    model_store_with_tiny_llama: tuple[Path, WorkerState],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prepare a MODEL session through the real TorchModelSessionLoader.

    The checkpoint is genuinely loaded, characterized, enumerated and
    exported on CPU; only the benchmark itself is faked (the real layer
    profiler is covered in tests/unit/profiling). This is the §51 chain
    'registration → prepare → facts' on a CPU-only host.
    """
    root, state = model_store_with_tiny_llama
    monkeypatch.setattr(
        AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            torch.OutOfMemoryError("whole model does not fit")
        ),
    )
    rig = make_rig(state, model_store_root=root, real_loader=True)
    request = ProfilingSessionRequest(
        kind=ProfilingSessionKind.MODEL,
        device_ids=(CPU_DEVICE,),
        model=ModelReference(model_id="tiny/llama", revision="main"),
        dtype="fp32",
        target_layer_index=2,
    )
    prepared = await prepare(rig, request)
    assert prepared.accepted is True, prepared.detail
    facts = prepared.session_facts
    assert facts.characterization.num_layers == 4  # TINY_LLAMA_CONFIG
    assert len(facts.layer_entries) == 4
    handle = rig.sessions.peek(SESSION_ID).model_handle
    assert isinstance(handle, PreparedComputeSession)
    assert isinstance(handle.opaque, LoadedModelSession)
    assert len(handle.opaque.layers) == 1
    assert facts.module_entries  # attention + mlp per layer
    assert facts.operator_signatures  # the representative export produced ops
    assert rig.leases.leased_device_ids == (CPU_DEVICE,)

    case = ProfilingCase.for_spec(
        WORKER_ID,
        ModelCaseSpec(
            granularity=ProfilingGranularity.TRANSFORMER_LAYER,
            device_ids=(CPU_DEVICE,),
            dtype="fp32",
            model=request.model,
            layer_signature=facts.layer_entries[2].signature,
            layer_index=2,
            batch_size=1,
            sequence_length=16,
        ),
    )
    ran = await rig.runner.run_profiling_case(run_request(case))
    assert ran.accepted is True
    assert ran.outcome.succeeded, ran.outcome.failure
    assert len(rig.layer.calls) == 1

    await rig.runner.shutdown()
    assert rig.leases.leased_device_ids == ()


async def test_model_loader_export_failure_uses_structural_profiler_fallback(
    model_store_with_tiny_llama: tuple[Path, WorkerState],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, state = model_store_with_tiny_llama
    export_calls = 0
    profiler_calls = 0

    def fail_export(self, target, args, kwargs):
        nonlocal export_calls
        export_calls += 1
        raise ProfilingError(
            ProfilingErrorCategory.EXPORT_FAILED, "synthetic export rejection"
        )

    def structural_only(self, target, args, kwargs):
        nonlocal profiler_calls
        profiler_calls += 1
        return RawOperatorGraph(extractor="torch_profiler", operations=())

    monkeypatch.setattr(model_loader_module.TorchExportExtractor, "extract", fail_export)
    monkeypatch.setattr(
        model_loader_module.TorchProfilerExtractor, "extract", structural_only
    )
    rig = make_rig(state, model_store_root=root, real_loader=True)
    request = ProfilingSessionRequest(
        kind=ProfilingSessionKind.MODEL,
        device_ids=(CPU_DEVICE,),
        model=ModelReference(model_id="tiny/llama", revision="main"),
        dtype="fp32",
        target_layer_index=0,
    )

    prepared = await prepare(rig, request)

    assert prepared.accepted, prepared.detail
    assert MODEL_EXTRACTION_SEQUENCE_LENGTHS == (128, 512, 2048)
    assert export_calls == profiler_calls == len(MODEL_EXTRACTION_SEQUENCE_LENGTHS)
