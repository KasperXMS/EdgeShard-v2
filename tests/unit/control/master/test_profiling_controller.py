"""Master ProfilingController tests (Phase 2 spec §40).

The controller is exercised against a real ``MasterService`` (real registry,
sessions and cluster snapshots), a real ``SqliteProfileStore``, and an
in-process fake of the Worker profiling transport — so every Master-side
P2G DoD scenario lands here: stale session, cancellation, duplicate result,
busy GPU, unsupported model, network failure, timeout, partial completion,
and Master-restart resume. The Worker-side behavior behind the transport is
pinned by ``tests/unit/control/worker/test_profiling_runner.py``; the real
gRPC wire is covered by the P2G integration tests.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import grpc
import pytest

from edgeshard.control.master.profiling_controller import (
    CaseReport,
    ExperimentReport,
    ProfilingController,
    _session_request,
)
from edgeshard.control.master.service import MasterService
from edgeshard.profiling.domain.experiment import (
    CaseOutcome,
    CaseState,
    ExperimentState,
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
from edgeshard.profiling.domain.model import ModelReference
from edgeshard.profiling.domain.network import ProbeKind
from edgeshard.profiling.domain.session import (
    ProfilingSessionKind,
    profiling_session_id,
)
from edgeshard.profiling.domain.signature import (
    GemmSignature,
    OperatorKind,
    OperatorSignature,
    ProfilingGranularity,
    TransformerLayerSignature,
)
from edgeshard.profiling.store.sqlite import SqliteProfileStore
from edgeshard.protocol.control.mapper import (
    CONTROL_PROTOCOL_VERSION,
    RegisterWorkerRequest,
)
from edgeshard.protocol.profiling.mapper import (
    CancelProfilingCaseRequest,
    CancelProfilingCaseResponse,
    CloseProfilingSessionRequest,
    CloseProfilingSessionResponse,
    GetProfilingCaseRequest,
    GetProfilingCaseResponse,
    PrepareProfilingSessionRequest,
    PrepareProfilingSessionResponse,
    ProfilingRejection,
    RunProfilingCaseRequest,
    RunProfilingCaseResponse,
)
from factories import make_rtx_capability, make_worker_identity, make_worker_state

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)

W1 = "w-1"
W2 = "w-2"
ENDPOINT_1 = "127.0.0.1:9101"
ENDPOINT_2 = "127.0.0.1:9102"

MODEL = ModelReference(model_id="tiny/llama", revision="local")
OPERATOR_SIG = OperatorSignature(
    kind=OperatorKind.GEMM,
    parameters=GemmSignature(m=64, n=64, k=64, dtype="fp32"),
    backend_family="torch",
)
OPERATOR_SIG_B = OperatorSignature(
    kind=OperatorKind.GEMM,
    parameters=GemmSignature(m=128, n=64, k=64, dtype="fp32"),
    backend_family="torch",
)
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


# ---------------------------------------------------------------------------
# Fakes and rig
# ---------------------------------------------------------------------------


class FakeClock:
    """One source for both Master clocks (the pattern from test_service)."""

    def __init__(self) -> None:
        self.now = 1_000.0
        self.wall_base = NOW

    def monotonic(self) -> float:
        return self.now

    def wall(self) -> datetime:
        return self.wall_base + timedelta(seconds=self.now - 1_000.0)

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeAioRpcError(grpc.aio.AioRpcError):
    """A transport loss with a chosen gRPC status code.

    Built through the real ``AioRpcError`` constructor so ``code()``/
    ``details()`` and ``__str__`` (the controller logs the exception) all
    behave exactly like a genuine lost RPC.
    """

    def __init__(self, code: grpc.StatusCode, details: str = "transport lost") -> None:
        super().__init__(code=code, details=details, debug_error_string="fake")


def make_record(case_id: str, measurement_id: str | None = None) -> MeasurementRecord:
    samples = (1.0, 2.0, 3.0)
    return MeasurementRecord(
        measurement_id=measurement_id or f"m-{uuid.uuid4().hex[:12]}",
        case_id=case_id,
        environment_fingerprint="fp-1",
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=1),
        sample_count=len(samples),
        samples=samples,
        metrics=MeasurementMetrics(
            latency=LatencyMetrics(
                summary=summarize_samples(samples), unit=TimeUnit.MILLISECONDS
            )
        ),
    )


class FakeTransport:
    """In-process stand-in for ``WorkerProfilingClient`` (ProfilingTransport).

    Defaults answer the way a healthy Worker would: prepare accepted, runs
    succeed with a fresh record for the requested case, cancellations land as
    CANCELLED, closes are accepted. Tests script deviations per case id.
    """

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self.prepare_requests: list[PrepareProfilingSessionRequest] = []
        self.run_requests: list[RunProfilingCaseRequest] = []
        self.get_requests: list[GetProfilingCaseRequest] = []
        self.cancel_requests: list[CancelProfilingCaseRequest] = []
        self.close_requests: list[CloseProfilingSessionRequest] = []
        self.timeouts: list[float | None] = []
        self.closed = False
        self.prepare_response: PrepareProfilingSessionResponse = (
            PrepareProfilingSessionResponse(accepted=True)
        )
        self.prepare_error: Exception | None = None
        self._run_scripts: dict[str, RunProfilingCaseResponse | Exception] = {}
        self._get_scripts: dict[str, GetProfilingCaseResponse | Exception] = {}
        self._cancel_scripts: dict[str, CancelProfilingCaseResponse | Exception] = {}

    def script_run(
        self, case_id: str, response: RunProfilingCaseResponse | Exception
    ) -> None:
        self._run_scripts[case_id] = response

    def script_get(
        self, case_id: str, response: GetProfilingCaseResponse | Exception
    ) -> None:
        self._get_scripts[case_id] = response

    def script_cancel(
        self, case_id: str, response: CancelProfilingCaseResponse | Exception
    ) -> None:
        self._cancel_scripts[case_id] = response

    async def prepare_profiling_session(
        self,
        request: PrepareProfilingSessionRequest,
        *,
        timeout: float | None = None,
    ) -> PrepareProfilingSessionResponse:
        self.prepare_requests.append(request)
        self.timeouts.append(timeout)
        if self.prepare_error is not None:
            raise self.prepare_error
        return self.prepare_response

    async def run_profiling_case(
        self, request: RunProfilingCaseRequest, *, timeout: float | None = None
    ) -> RunProfilingCaseResponse:
        self.run_requests.append(request)
        self.timeouts.append(timeout)
        scripted = self._run_scripts.get(request.case.case_id)
        if isinstance(scripted, Exception):
            raise scripted
        if scripted is not None:
            return scripted
        return RunProfilingCaseResponse(
            accepted=True,
            outcome=CaseOutcome.from_record(make_record(request.case.case_id)),
        )

    async def get_profiling_case(
        self, request: GetProfilingCaseRequest, *, timeout: float | None = None
    ) -> GetProfilingCaseResponse:
        self.get_requests.append(request)
        self.timeouts.append(timeout)
        scripted = self._get_scripts.get(request.case_id)
        if isinstance(scripted, Exception):
            raise scripted
        if scripted is not None:
            return scripted
        return GetProfilingCaseResponse(
            accepted=False,
            detail="session does not track the case",
            reason=ProfilingRejection.UNKNOWN_CASE,
        )

    async def cancel_profiling_case(
        self, request: CancelProfilingCaseRequest, *, timeout: float | None = None
    ) -> CancelProfilingCaseResponse:
        self.cancel_requests.append(request)
        self.timeouts.append(timeout)
        scripted = self._cancel_scripts.get(request.case_id)
        if isinstance(scripted, Exception):
            raise scripted
        if scripted is not None:
            return scripted
        return CancelProfilingCaseResponse(
            accepted=True, case_state=CaseState.CANCELLED
        )

    async def close_profiling_session(
        self, request: CloseProfilingSessionRequest, *, timeout: float | None = None
    ) -> CloseProfilingSessionResponse:
        self.close_requests.append(request)
        self.timeouts.append(timeout)
        return CloseProfilingSessionResponse(accepted=True)

    async def close(self) -> None:
        self.closed = True


@dataclass
class Rig:
    service: MasterService
    store: SqliteProfileStore
    controller: ProfilingController
    clock: FakeClock
    transports: dict[str, FakeTransport]

    def transport(self, endpoint: str = ENDPOINT_1) -> FakeTransport:
        return self.transports[endpoint]


def make_rig(tmp_path: Path, *, rpc_timeout: float | None = None) -> Rig:
    clock = FakeClock()
    service = MasterService(None, monotonic=clock.monotonic, wall=clock.wall)
    store = SqliteProfileStore(tmp_path / "profile.sqlite")
    transports: dict[str, FakeTransport] = {}

    def factory(endpoint: str) -> FakeTransport:
        return transports.setdefault(endpoint, FakeTransport(endpoint))

    controller = ProfilingController(
        service=service,
        store=store,
        transport_factory=factory,
        clock=clock.wall,
        rpc_timeout=rpc_timeout,
    )
    return Rig(
        service=service,
        store=store,
        controller=controller,
        clock=clock,
        transports=transports,
    )


async def register_worker(
    rig: Rig,
    worker_id: str,
    *,
    profiling_endpoint: str | None = ENDPOINT_1,
) -> tuple[str, str, str]:
    """Register a real worker; returns (worker_id, instance_id, session_id)."""
    identity = make_worker_identity(worker_id)
    request = RegisterWorkerRequest(
        protocol_version=CONTROL_PROTOCOL_VERSION,
        instance_id=str(uuid.uuid4()),
        identity=identity,
        capability=make_rtx_capability(),
        initial_state=make_worker_state(identity.worker_id),
        profiling_endpoint=profiling_endpoint,
    )
    response = await rig.service.register_worker(request)
    return identity.worker_id, request.instance_id, response.session_id


def operator_case(
    worker_id: str = W1,
    signature: OperatorSignature = OPERATOR_SIG,
    device_ids: tuple[str, ...] = ("gpu-0",),
) -> ProfilingCase:
    return ProfilingCase.for_spec(
        worker_id,
        ModelCaseSpec(
            granularity=ProfilingGranularity.OPERATOR,
            device_ids=device_ids,
            dtype="fp32",
            operator_signature=signature,
        ),
    )


def layer_case(
    worker_id: str = W1,
    *,
    dtype: str = "fp32",
    layer_index: int = 0,
    device_ids: tuple[str, ...] = ("gpu-0",),
) -> ProfilingCase:
    return ProfilingCase.for_spec(
        worker_id,
        ModelCaseSpec(
            granularity=ProfilingGranularity.TRANSFORMER_LAYER,
            device_ids=device_ids,
            dtype=dtype,
            model=MODEL,
            layer_signature=LAYER_SIG,
            layer_index=layer_index,
        ),
    )


def network_case(source: str = W1, destination: str = W2) -> ProfilingCase:
    return ProfilingCase.for_spec(
        source,
        NetworkCaseSpec(
            probe_kind=ProbeKind.RTT,
            source_worker_id=source,
            destination_worker_id=destination,
            packet_count=5,
        ),
    )


def report_for(report: ExperimentReport, case_id: str) -> CaseReport:
    for case in report.cases:
        if case.case_id == case_id:
            return case
    raise KeyError(case_id)


def failure_of(case: CaseReport) -> ProfilingFailure:
    assert case.outcome is not None and case.outcome.failure is not None
    return case.outcome.failure


# ---------------------------------------------------------------------------
# Experiment definitions (§8.1, §44)
# ---------------------------------------------------------------------------


def test_create_experiment_persists_definition_and_cases(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    cases = [operator_case(), layer_case(layer_index=1)]
    experiment = rig.controller.create_experiment(
        strategy_id="default", cases=cases, requested_by="operator"
    )
    assert experiment.case_ids == tuple(case.case_id for case in cases)
    assert experiment.created_at == NOW
    stored = rig.store.get_experiment(experiment.experiment_id)
    assert stored is not None
    assert stored.state is ExperimentState.PENDING
    assert stored.experiment == experiment
    for case in cases:
        stored_case = rig.store.get_case(case.case_id)
        assert stored_case is not None
        assert stored_case.case == case
        assert stored_case.state is CaseState.PENDING


def test_create_experiment_is_idempotent_and_dedupes(tmp_path: Path) -> None:
    """§7/§44: same strategy + case set replays the stored definition."""
    rig = make_rig(tmp_path)
    case = operator_case()
    first = rig.controller.create_experiment(strategy_id="default", cases=[case, case])
    assert first.case_ids == (case.case_id,)
    rig.clock.advance(60.0)
    second = rig.controller.create_experiment(strategy_id="default", cases=[case])
    assert second.experiment_id == first.experiment_id
    assert second.created_at == first.created_at  # history never rewritten
    assert second.requested_by == first.requested_by


async def test_run_unknown_experiment_fails_loudly(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    with pytest.raises(ValueError, match="unknown experiment"):
        await rig.controller.run_experiment("no-such-experiment")
    with pytest.raises(ValueError, match="unknown experiment"):
        await rig.controller.cancel_experiment("no-such-experiment")


# ---------------------------------------------------------------------------
# Happy paths (§40: dispatch, collect, persist)
# ---------------------------------------------------------------------------


async def test_operator_case_completes_end_to_end(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    await register_worker(rig, W1)
    case = operator_case()
    experiment = rig.controller.create_experiment(strategy_id="default", cases=[case])

    report = await rig.controller.run_experiment(experiment.experiment_id)

    assert report.state is ExperimentState.COMPLETED
    verdict = report_for(report, case.case_id)
    assert verdict.state is CaseState.COMPLETED
    assert verdict.outcome is not None and verdict.outcome.succeeded
    assert rig.store.get_case(case.case_id).state is CaseState.COMPLETED  # type: ignore[union-attr]
    assert rig.store.get_experiment(experiment.experiment_id).state is (  # type: ignore[union-attr]
        ExperimentState.COMPLETED
    )
    records = rig.store.query_measurements(case_id=case.case_id)
    assert len(records) == 1
    assert records[0].case_id == case.case_id
    transport = rig.transport()
    assert transport.closed
    assert [request.profiling_session_id for request in transport.close_requests] == [
        transport.prepare_requests[0].profiling_session_id
    ]


async def test_prepare_envelope_carries_registration_and_canonical_session(
    tmp_path: Path,
) -> None:
    """§41: dispatch tokens come from the live registration, never guessed."""
    rig = make_rig(tmp_path)
    _, instance_id, session_id = await register_worker(rig, W1)
    case = operator_case()
    experiment = rig.controller.create_experiment(strategy_id="default", cases=[case])

    await rig.controller.run_experiment(experiment.experiment_id)

    prepare = rig.transport().prepare_requests[0]
    assert prepare.worker_id == W1
    assert prepare.instance_id == instance_id
    assert prepare.registration_session_id == session_id
    assert prepare.session_request.kind is ProfilingSessionKind.OPERATOR
    assert prepare.session_request.device_ids == ("gpu-0",)
    assert prepare.profiling_session_id == profiling_session_id(
        experiment.experiment_id, W1, prepare.session_request
    )
    run = rig.transport().run_requests[0]
    assert run.profiling_session_id == prepare.profiling_session_id
    assert run.case == case


async def test_model_cases_share_one_session_and_split_on_dtype(
    tmp_path: Path,
) -> None:
    """§38: build the expensive state once; a different dtype is a new session."""
    rig = make_rig(tmp_path)
    await register_worker(rig, W1)
    fp32_a = layer_case(layer_index=0, device_ids=("gpu-0",))
    fp32_b = layer_case(layer_index=1, device_ids=("gpu-1",))
    bf16 = layer_case(layer_index=2, dtype="bf16")
    experiment = rig.controller.create_experiment(
        strategy_id="default", cases=[fp32_a, fp32_b, bf16]
    )

    report = await rig.controller.run_experiment(experiment.experiment_id)

    assert report.state is ExperimentState.COMPLETED
    prepares = rig.transport().prepare_requests
    assert len(prepares) == 2
    by_dtype = {prepare.session_request.dtype: prepare for prepare in prepares}
    assert by_dtype["fp32"].session_request.device_ids == ("gpu-0", "gpu-1")
    assert by_dtype["fp32"].session_request.model == MODEL
    assert by_dtype["bf16"].session_request.device_ids == ("gpu-0",)
    runs = rig.transport().run_requests
    session_of = {run.case.case_id: run.profiling_session_id for run in runs}
    assert session_of[fp32_a.case_id] == session_of[fp32_b.case_id]
    assert session_of[bf16.case_id] != session_of[fp32_a.case_id]
    assert len(rig.transport().close_requests) == 2


async def test_operator_and_model_cases_use_separate_sessions(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    await register_worker(rig, W1)
    op = operator_case()
    model = layer_case()
    experiment = rig.controller.create_experiment(strategy_id="default", cases=[op, model])

    report = await rig.controller.run_experiment(experiment.experiment_id)

    assert report.state is ExperimentState.COMPLETED
    kinds = {
        prepare.session_request.kind for prepare in rig.transport().prepare_requests
    }
    assert kinds == {ProfilingSessionKind.OPERATOR, ProfilingSessionKind.MODEL}
    model_prepare = next(
        prepare
        for prepare in rig.transport().prepare_requests
        if prepare.session_request.kind is ProfilingSessionKind.MODEL
    )
    assert model_prepare.session_request.model == MODEL
    assert model_prepare.session_request.dtype == "fp32"


async def test_two_workers_dispatch_concurrently_on_own_transports(
    tmp_path: Path,
) -> None:
    rig = make_rig(tmp_path)
    await register_worker(rig, W1, profiling_endpoint=ENDPOINT_1)
    await register_worker(rig, W2, profiling_endpoint=ENDPOINT_2)
    case_a = operator_case(W1, OPERATOR_SIG)
    case_b = operator_case(W2, OPERATOR_SIG)
    experiment = rig.controller.create_experiment(
        strategy_id="default", cases=[case_a, case_b]
    )

    report = await rig.controller.run_experiment(experiment.experiment_id)

    assert report.state is ExperimentState.COMPLETED
    assert set(rig.transports) == {ENDPOINT_1, ENDPOINT_2}
    assert [run.case.case_id for run in rig.transport(ENDPOINT_1).run_requests] == [
        case_a.case_id
    ]
    assert [run.case.case_id for run in rig.transport(ENDPOINT_2).run_requests] == [
        case_b.case_id
    ]
    assert all(transport.closed for transport in rig.transports.values())


async def test_rpc_timeout_is_passed_to_every_call(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, rpc_timeout=5.0)
    await register_worker(rig, W1)
    experiment = rig.controller.create_experiment(
        strategy_id="default", cases=[operator_case()]
    )
    await rig.controller.run_experiment(experiment.experiment_id)
    assert rig.transport().timeouts
    assert all(timeout == 5.0 for timeout in rig.transport().timeouts)


# ---------------------------------------------------------------------------
# Dispatch context failures (§52.2: never guess a Worker)
# ---------------------------------------------------------------------------


async def test_unregistered_worker_cases_fail_typed(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    case = operator_case("w-ghost")
    experiment = rig.controller.create_experiment(strategy_id="default", cases=[case])

    report = await rig.controller.run_experiment(experiment.experiment_id)

    assert report.state is ExperimentState.FAILED
    verdict = report_for(report, case.case_id)
    assert verdict.state is CaseState.FAILED
    failure = failure_of(verdict)
    assert failure.category is ProfilingErrorCategory.NETWORK_UNREACHABLE
    assert "not registered" in failure.message
    assert rig.transports == {}  # never dialed
    assert rig.store.get_case(case.case_id).state is CaseState.FAILED  # type: ignore[union-attr]


async def test_worker_without_profiling_endpoint_fails_typed(tmp_path: Path) -> None:
    """§41: a Phase 1 worker does not host the profiling plane."""
    rig = make_rig(tmp_path)
    await register_worker(rig, W1, profiling_endpoint=None)
    case = operator_case()
    experiment = rig.controller.create_experiment(strategy_id="default", cases=[case])

    report = await rig.controller.run_experiment(experiment.experiment_id)

    failure = failure_of(report_for(report, case.case_id))
    assert failure.category is ProfilingErrorCategory.NETWORK_UNREACHABLE
    assert "does not host" in failure.message
    assert rig.transports == {}


# ---------------------------------------------------------------------------
# Prepare failures (busy GPU, unsupported model, transport loss)
# ---------------------------------------------------------------------------


async def test_prepare_device_busy_refusal_keeps_category(tmp_path: Path) -> None:
    """P2G DoD busy GPU: the §39 lease refusal survives into the report."""
    rig = make_rig(tmp_path)
    await register_worker(rig, W1)
    case = operator_case()
    experiment = rig.controller.create_experiment(strategy_id="default", cases=[case])
    # Transports are created lazily on dispatch; seed the fake up front.
    transport = rig.transports.setdefault(ENDPOINT_1, FakeTransport(ENDPOINT_1))
    transport.prepare_response = PrepareProfilingSessionResponse(
        accepted=False,
        detail="device gpu-0 is busy: utilization 97.0% exceeds floor",
        reason=ProfilingRejection.DEVICE_BUSY,
    )

    report = await rig.controller.run_experiment(experiment.experiment_id)

    assert report.state is ExperimentState.FAILED
    verdict = report_for(report, case.case_id)
    failure = failure_of(verdict)
    assert failure.category is ProfilingErrorCategory.DEVICE_BUSY
    assert "device_busy" in failure.message
    assert transport.run_requests == []  # never benchmarked
    assert transport.close_requests == []  # nothing to clean up
    assert rig.store.query_measurements(case_id=case.case_id) == ()


async def test_prepare_typed_failure_preserved(tmp_path: Path) -> None:
    """P2G DoD unsupported model: the §42 domain failure travels intact."""
    rig = make_rig(tmp_path)
    await register_worker(rig, W1)
    case = layer_case()
    experiment = rig.controller.create_experiment(strategy_id="default", cases=[case])
    transport = rig.transports.setdefault(ENDPOINT_1, FakeTransport(ENDPOINT_1))
    typed = ProfilingFailure(
        category=ProfilingErrorCategory.UNSUPPORTED_MODEL,
        message="no READY snapshot for model tiny/llama revision local",
        details=(("model_id", "tiny/llama"),),
    )
    transport.prepare_response = PrepareProfilingSessionResponse(
        accepted=False, detail=typed.message, failure=typed
    )

    report = await rig.controller.run_experiment(experiment.experiment_id)

    failure = failure_of(report_for(report, case.case_id))
    assert failure is typed  # category AND details preserved, no string parsing
    assert transport.run_requests == []


async def test_prepare_timeout_maps_to_timeout_failure(tmp_path: Path) -> None:
    """P2G DoD timeout: DEADLINE_EXCEEDED is a typed TIMEOUT, not a crash."""
    rig = make_rig(tmp_path)
    await register_worker(rig, W1)
    case = operator_case()
    experiment = rig.controller.create_experiment(strategy_id="default", cases=[case])
    transport = rig.transports.setdefault(ENDPOINT_1, FakeTransport(ENDPOINT_1))
    transport.prepare_error = FakeAioRpcError(grpc.StatusCode.DEADLINE_EXCEEDED)

    report = await rig.controller.run_experiment(experiment.experiment_id)

    failure = failure_of(report_for(report, case.case_id))
    assert failure.category is ProfilingErrorCategory.TIMEOUT
    assert dict(failure.details)["grpc_code"] == grpc.StatusCode.DEADLINE_EXCEEDED.value


# ---------------------------------------------------------------------------
# Run failures (stale session, transport loss, protocol violations)
# ---------------------------------------------------------------------------


async def test_run_stale_session_refusal_publishes_nothing(tmp_path: Path) -> None:
    """P2G DoD stale session: §41 — its results MUST NOT be published."""
    rig = make_rig(tmp_path)
    await register_worker(rig, W1)
    case = operator_case()
    experiment = rig.controller.create_experiment(strategy_id="default", cases=[case])
    transport = rig.transports.setdefault(ENDPOINT_1, FakeTransport(ENDPOINT_1))
    transport.script_run(
        case.case_id,
        RunProfilingCaseResponse(
            accepted=False,
            detail="registration superseded",
            reason=ProfilingRejection.STALE_SESSION,
        ),
    )

    report = await rig.controller.run_experiment(experiment.experiment_id)

    assert report.state is ExperimentState.FAILED
    failure = failure_of(report_for(report, case.case_id))
    assert failure.category is ProfilingErrorCategory.NETWORK_UNREACHABLE
    assert "stale_session" in failure.message
    assert rig.store.query_measurements(case_id=case.case_id) == ()
    assert rig.store.get_case(case.case_id).state is CaseState.FAILED  # type: ignore[union-attr]
    # The session is still closed best-effort so the Worker releases leases.
    assert len(transport.close_requests) == 1


async def test_run_transport_unavailable_maps_typed(tmp_path: Path) -> None:
    """P2G DoD network failure: a lost RPC is typed NETWORK_UNREACHABLE."""
    rig = make_rig(tmp_path)
    await register_worker(rig, W1)
    case = operator_case()
    experiment = rig.controller.create_experiment(strategy_id="default", cases=[case])
    transport = rig.transports.setdefault(ENDPOINT_1, FakeTransport(ENDPOINT_1))
    transport.script_run(
        case.case_id, FakeAioRpcError(grpc.StatusCode.UNAVAILABLE, "connection reset")
    )

    report = await rig.controller.run_experiment(experiment.experiment_id)

    failure = failure_of(report_for(report, case.case_id))
    assert failure.category is ProfilingErrorCategory.NETWORK_UNREACHABLE
    assert dict(failure.details)["transport_detail"] == "connection reset"


async def test_run_timeout_maps_typed(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    await register_worker(rig, W1)
    case = operator_case()
    experiment = rig.controller.create_experiment(strategy_id="default", cases=[case])
    transport = rig.transports.setdefault(ENDPOINT_1, FakeTransport(ENDPOINT_1))
    transport.script_run(case.case_id, FakeAioRpcError(grpc.StatusCode.DEADLINE_EXCEEDED))

    report = await rig.controller.run_experiment(experiment.experiment_id)

    failure = failure_of(report_for(report, case.case_id))
    assert failure.category is ProfilingErrorCategory.TIMEOUT


async def test_unexpected_run_exception_is_internal_error(tmp_path: Path) -> None:
    """§47: a bizarre failure is typed and logged, never swallowed silently."""
    rig = make_rig(tmp_path)
    await register_worker(rig, W1)
    case = operator_case()
    experiment = rig.controller.create_experiment(strategy_id="default", cases=[case])
    transport = rig.transports.setdefault(ENDPOINT_1, FakeTransport(ENDPOINT_1))
    transport.script_run(case.case_id, RuntimeError("kaboom"))

    report = await rig.controller.run_experiment(experiment.experiment_id)

    failure = failure_of(report_for(report, case.case_id))
    assert failure.category is ProfilingErrorCategory.INTERNAL_ERROR
    assert "kaboom" in failure.message


async def test_record_case_mismatch_is_protocol_violation(tmp_path: Path) -> None:
    """§47: a result under a foreign case id is refused, not persisted."""
    rig = make_rig(tmp_path)
    await register_worker(rig, W1)
    case = operator_case()
    experiment = rig.controller.create_experiment(strategy_id="default", cases=[case])
    transport = rig.transports.setdefault(ENDPOINT_1, FakeTransport(ENDPOINT_1))
    foreign = make_record("some-other-case")
    transport.script_run(
        case.case_id,
        RunProfilingCaseResponse(accepted=True, outcome=CaseOutcome.from_record(foreign)),
    )

    report = await rig.controller.run_experiment(experiment.experiment_id)

    failure = failure_of(report_for(report, case.case_id))
    assert failure.category is ProfilingErrorCategory.INTERNAL_ERROR
    assert rig.store.query_measurements(case_id=case.case_id) == ()
    assert rig.store.get_measurement(foreign.measurement_id) is None


# ---------------------------------------------------------------------------
# Partial completion and failed outcomes (§8.1)
# ---------------------------------------------------------------------------


async def test_partial_experiment_completion(tmp_path: Path) -> None:
    """P2G DoD: one case succeeds, one fails → PARTIALLY_COMPLETED."""
    rig = make_rig(tmp_path)
    await register_worker(rig, W1)
    good = operator_case(signature=OPERATOR_SIG)
    bad = operator_case(signature=OPERATOR_SIG_B)
    experiment = rig.controller.create_experiment(
        strategy_id="default", cases=[good, bad]
    )
    transport = rig.transports.setdefault(ENDPOINT_1, FakeTransport(ENDPOINT_1))
    transport.script_run(
        bad.case_id,
        RunProfilingCaseResponse(
            accepted=True,
            outcome=CaseOutcome.from_failure(
                ProfilingFailure(
                    category=ProfilingErrorCategory.PROFILER_FAILED,
                    message="profiler crashed",
                )
            ),
        ),
    )

    report = await rig.controller.run_experiment(experiment.experiment_id)

    assert report.state is ExperimentState.PARTIALLY_COMPLETED
    assert report_for(report, good.case_id).state is CaseState.COMPLETED
    assert report_for(report, bad.case_id).state is CaseState.FAILED
    assert failure_of(report_for(report, bad.case_id)).category is (
        ProfilingErrorCategory.PROFILER_FAILED
    )
    assert rig.store.get_case(good.case_id).state is CaseState.COMPLETED  # type: ignore[union-attr]
    assert rig.store.get_case(bad.case_id).state is CaseState.FAILED  # type: ignore[union-attr]
    assert len(rig.store.query_measurements(case_id=good.case_id)) == 1
    assert rig.store.query_measurements(case_id=bad.case_id) == ()
    assert rig.store.get_experiment(experiment.experiment_id).state is (  # type: ignore[union-attr]
        ExperimentState.PARTIALLY_COMPLETED
    )


async def test_all_cases_cancelled_maps_to_cancelled_experiment(
    tmp_path: Path,
) -> None:
    rig = make_rig(tmp_path)
    await register_worker(rig, W1)
    experiment = rig.controller.create_experiment(
        strategy_id="default", cases=[operator_case()]
    )
    report = await rig.controller.cancel_experiment(experiment.experiment_id)
    assert report.state is ExperimentState.CANCELLED


# ---------------------------------------------------------------------------
# Duplicate results and resume (§50, P2G DoD: Master restart)
# ---------------------------------------------------------------------------


async def test_duplicate_measurement_is_a_replay(tmp_path: Path) -> None:
    """P2G DoD duplicate result: an already-stored record never re-benchmarks
    and never errors — the case simply completes."""
    rig = make_rig(tmp_path)
    await register_worker(rig, W1)
    case = operator_case()
    experiment = rig.controller.create_experiment(strategy_id="default", cases=[case])
    record = make_record(case.case_id)
    assert rig.store.append_measurement(record) is True
    transport = rig.transports.setdefault(ENDPOINT_1, FakeTransport(ENDPOINT_1))
    transport.script_run(
        case.case_id,
        RunProfilingCaseResponse(accepted=True, outcome=CaseOutcome.from_record(record)),
    )

    report = await rig.controller.run_experiment(experiment.experiment_id)

    assert report.state is ExperimentState.COMPLETED
    assert report_for(report, case.case_id).state is CaseState.COMPLETED
    assert len(rig.store.query_measurements(case_id=case.case_id)) == 1


async def test_resume_after_master_restart_skips_finished_cases(
    tmp_path: Path,
) -> None:
    """P2G DoD Master restart: a new controller over the same store continues
    the interrupted experiment under the *new* registration."""
    rig = make_rig(tmp_path)
    await register_worker(rig, W1)
    done = operator_case(signature=OPERATOR_SIG)
    todo = operator_case(signature=OPERATOR_SIG_B)
    experiment = rig.controller.create_experiment(
        strategy_id="default", cases=[done, todo]
    )
    # Stage the crash: the experiment was RUNNING and one case had finished.
    rig.store.update_experiment_state(experiment.experiment_id, ExperimentState.RUNNING)
    rig.store.update_case_state(done.case_id, CaseState.RUNNING)
    rig.store.update_case_state(done.case_id, CaseState.COMPLETED)
    rig.store.append_measurement(make_record(done.case_id))
    rig.store.close()

    restarted = make_rig(tmp_path)
    _, _, new_session_id = await register_worker(restarted, W1)

    report = await restarted.controller.run_experiment(experiment.experiment_id)

    assert report.state is ExperimentState.COMPLETED
    replayed = report_for(report, done.case_id)
    assert replayed.state is CaseState.COMPLETED
    assert replayed.outcome is None  # the store keeps states, not outcomes
    assert "replayed" in replayed.detail
    transport = restarted.transport()
    assert [run.case.case_id for run in transport.run_requests] == [todo.case_id]
    assert transport.prepare_requests[0].registration_session_id == new_session_id
    assert report_for(report, todo.case_id).state is CaseState.COMPLETED
    assert len(restarted.store.query_measurements(case_id=done.case_id)) == 1


async def test_terminal_experiment_never_re_dispatches(tmp_path: Path) -> None:
    """§44/§50: a finished experiment replays from the store."""
    rig = make_rig(tmp_path)
    await register_worker(rig, W1)
    case = operator_case()
    experiment = rig.controller.create_experiment(strategy_id="default", cases=[case])
    first = await rig.controller.run_experiment(experiment.experiment_id)
    assert first.state is ExperimentState.COMPLETED
    transport = rig.transport()
    runs_after_first = len(transport.run_requests)

    second = await rig.controller.run_experiment(experiment.experiment_id)

    assert second.state is ExperimentState.COMPLETED
    assert len(transport.run_requests) == runs_after_first
    assert "replayed from the store" in report_for(second, case.case_id).detail
    assert report_for(second, case.case_id).outcome is None


# ---------------------------------------------------------------------------
# Network cases (§30, §52.2: Master-resolved facts only)
# ---------------------------------------------------------------------------


async def test_network_case_carries_master_resolved_facts(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    await register_worker(rig, W1, profiling_endpoint=ENDPOINT_1)
    await register_worker(rig, W2, profiling_endpoint=ENDPOINT_2)
    case = network_case(W1, W2)
    experiment = rig.controller.create_experiment(strategy_id="default", cases=[case])

    report = await rig.controller.run_experiment(experiment.experiment_id)

    assert report.state is ExperimentState.COMPLETED
    transport = rig.transport(ENDPOINT_1)
    prepare = transport.prepare_requests[0]
    assert prepare.session_request.kind is ProfilingSessionKind.NETWORK
    assert prepare.session_request.device_ids == ()
    assert prepare.network_facts  # mandatory for network sessions (§41)
    assert {facts.worker_id for facts in prepare.network_facts} == {W1, W2}
    for facts in prepare.network_facts:
        assert facts.hostname == "worker-host"
        assert [interface.interface_id for interface in facts.interfaces] == ["nic-0"]
    assert ENDPOINT_2 not in rig.transports  # the destination is never dialed


async def test_network_case_unknown_peer_fails_without_dispatch(
    tmp_path: Path,
) -> None:
    """§52.2: no cluster facts for the peer → typed failure, never a guess."""
    rig = make_rig(tmp_path)
    await register_worker(rig, W1)
    case = network_case(W1, "w-ghost")
    experiment = rig.controller.create_experiment(strategy_id="default", cases=[case])

    report = await rig.controller.run_experiment(experiment.experiment_id)

    assert report.state is ExperimentState.FAILED
    failure = failure_of(report_for(report, case.case_id))
    assert failure.category is ProfilingErrorCategory.NETWORK_UNREACHABLE
    assert "w-ghost" in failure.message
    assert rig.transport().prepare_requests == []


# ---------------------------------------------------------------------------
# Cancellation (§40, §44)
# ---------------------------------------------------------------------------


async def test_cancel_dispatches_to_worker_and_records(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    await register_worker(rig, W1)
    case = operator_case()
    experiment = rig.controller.create_experiment(strategy_id="default", cases=[case])

    report = await rig.controller.cancel_experiment(experiment.experiment_id)

    assert report.state is ExperimentState.CANCELLED
    verdict = report_for(report, case.case_id)
    assert verdict.state is CaseState.CANCELLED
    assert failure_of(verdict).category is ProfilingErrorCategory.CANCELLED
    assert rig.store.get_case(case.case_id).state is CaseState.CANCELLED  # type: ignore[union-attr]
    transport = rig.transport()
    assert [request.case_id for request in transport.cancel_requests] == [case.case_id]
    # The cancel targets the session the case *would* run under, and the
    # session is closed so leases release (§39).
    assert transport.cancel_requests[0].profiling_session_id == (
        transport.close_requests[0].profiling_session_id
    )
    assert transport.closed


async def test_cancel_unreachable_worker_records_master_side(tmp_path: Path) -> None:
    """The Master's bookkeeping reflects its intent even when the Worker is gone."""
    rig = make_rig(tmp_path)
    case = operator_case("w-ghost")
    experiment = rig.controller.create_experiment(strategy_id="default", cases=[case])

    report = await rig.controller.cancel_experiment(experiment.experiment_id)

    assert report.state is ExperimentState.CANCELLED
    verdict = report_for(report, case.case_id)
    assert verdict.state is CaseState.CANCELLED
    assert "unreachable" in verdict.detail
    assert rig.store.get_case(case.case_id).state is CaseState.CANCELLED  # type: ignore[union-attr]


async def test_cancel_preserves_completed_history(tmp_path: Path) -> None:
    """§44: a finished case and its measurement survive cancellation."""
    rig = make_rig(tmp_path)
    await register_worker(rig, W1)
    done = operator_case(signature=OPERATOR_SIG)
    active = operator_case(signature=OPERATOR_SIG_B)
    experiment = rig.controller.create_experiment(
        strategy_id="default", cases=[done, active]
    )
    rig.store.update_case_state(done.case_id, CaseState.COMPLETED)
    rig.store.append_measurement(make_record(done.case_id))

    report = await rig.controller.cancel_experiment(experiment.experiment_id)

    assert report.state is ExperimentState.CANCELLED
    assert report_for(report, done.case_id).state is CaseState.COMPLETED
    assert report_for(report, active.case_id).state is CaseState.CANCELLED
    assert len(rig.store.query_measurements(case_id=done.case_id)) == 1
    transport = rig.transport()
    assert [request.case_id for request in transport.cancel_requests] == [
        active.case_id
    ]


async def test_cancel_race_persists_the_workers_truth(tmp_path: Path) -> None:
    """§44: when the case terminalized on the Worker first, its outcome is
    fetched and persisted instead of being overwritten with CANCELLED."""
    rig = make_rig(tmp_path)
    await register_worker(rig, W1)
    case = operator_case()
    experiment = rig.controller.create_experiment(strategy_id="default", cases=[case])
    transport = rig.transports.setdefault(ENDPOINT_1, FakeTransport(ENDPOINT_1))
    record = make_record(case.case_id)
    transport.script_cancel(
        case.case_id,
        CancelProfilingCaseResponse(accepted=True, case_state=CaseState.COMPLETED),
    )
    transport.script_get(
        case.case_id,
        GetProfilingCaseResponse(
            accepted=True,
            case_state=CaseState.COMPLETED,
            outcome=CaseOutcome.from_record(record),
        ),
    )

    report = await rig.controller.cancel_experiment(experiment.experiment_id)

    verdict = report_for(report, case.case_id)
    assert verdict.state is CaseState.COMPLETED
    assert "lost the race" in verdict.detail
    assert len(rig.store.query_measurements(case_id=case.case_id)) == 1
    # Nothing was actually cancelled, so the honest verdict is COMPLETED.
    assert report.state is ExperimentState.COMPLETED


async def test_cancel_refusal_still_records_cancellation(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    await register_worker(rig, W1)
    case = operator_case()
    experiment = rig.controller.create_experiment(strategy_id="default", cases=[case])
    transport = rig.transports.setdefault(ENDPOINT_1, FakeTransport(ENDPOINT_1))
    transport.script_cancel(
        case.case_id,
        CancelProfilingCaseResponse(
            accepted=False,
            detail="no active registration",
            reason=ProfilingRejection.STALE_SESSION,
        ),
    )

    report = await rig.controller.cancel_experiment(experiment.experiment_id)

    verdict = report_for(report, case.case_id)
    assert verdict.state is CaseState.CANCELLED
    assert "refused" in verdict.detail
    assert report.state is ExperimentState.CANCELLED


async def test_cancel_terminal_experiment_is_a_noop(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    await register_worker(rig, W1)
    case = operator_case()
    experiment = rig.controller.create_experiment(strategy_id="default", cases=[case])
    first = await rig.controller.run_experiment(experiment.experiment_id)
    assert first.state is ExperimentState.COMPLETED
    transport = rig.transport()
    cancels_before = len(transport.cancel_requests)

    report = await rig.controller.cancel_experiment(experiment.experiment_id)

    assert report.state is ExperimentState.COMPLETED
    assert len(transport.cancel_requests) == cancels_before
    assert rig.store.get_case(case.case_id).state is CaseState.COMPLETED  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# ProfileSnapshot (§46)
# ---------------------------------------------------------------------------


async def test_build_profile_snapshot_views_the_store(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    await register_worker(rig, W1)
    case = operator_case()
    experiment = rig.controller.create_experiment(strategy_id="default", cases=[case])
    await rig.controller.run_experiment(experiment.experiment_id)

    snapshot = rig.controller.build_profile_snapshot()

    assert len(snapshot.snapshot_id) == 64
    assert snapshot.created_at == NOW
    assert [record.case_id for record in snapshot.measurements] == [case.case_id]
    assert snapshot.network_measurements == ()
    assert snapshot.model_characterizations == ()

    frozen = rig.controller.build_profile_snapshot()
    assert frozen.snapshot_id == snapshot.snapshot_id  # same instant, same content
    rig.clock.advance(1.0)
    assert rig.controller.build_profile_snapshot().snapshot_id != snapshot.snapshot_id


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_rpc_timeout_must_be_positive(tmp_path: Path) -> None:
    rig = make_rig(tmp_path)
    with pytest.raises(ValueError, match="positive"):
        ProfilingController(
            service=rig.service, store=rig.store, rpc_timeout=0.0
        )


def test_session_request_rejects_model_group_without_model(tmp_path: Path) -> None:
    """The defensive guard: a non-operator spec without a model cannot plan."""
    broken = object.__new__(ModelCaseSpec)  # bypass validation for the guard test
    object.__setattr__(broken, "granularity", ProfilingGranularity.TRANSFORMER_LAYER)
    object.__setattr__(broken, "device_ids", ("gpu-0",))
    object.__setattr__(broken, "dtype", None)
    object.__setattr__(broken, "backend", "torch")
    object.__setattr__(broken, "model", None)
    case = ProfilingCase(case_id="c-1", worker_id=W1, spec=broken)
    with pytest.raises(ValueError, match="model reference and dtype"):
        _session_request([case])
