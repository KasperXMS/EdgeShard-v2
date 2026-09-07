"""Profiling-plane gRPC transport tests (Phase 2 spec §41).

An in-process ``grpc.aio`` server with a recording handler isolates the
transport layer (Worker session/lease semantics are tested against the
real runner in tests/unit/control/worker, P2G.5): domain objects must
cross the wire and back unchanged, and protocol violations must surface
as explicit gRPC errors instead of degraded responses (§47).
"""

from __future__ import annotations

import grpc
import pytest
from wire_fixtures import (
    CHARACTERIZATION,
    ENVIRONMENT,
    FAILURE,
    INSTANCE_ID,
    MODEL,
    MODEL_CASE,
    MODEL_SESSION_REQUEST,
    NETWORK_FACTS,
    NETWORK_SESSION_REQUEST,
    NOW,
    PROFILING_SESSION_ID,
    RECORD,
    REGISTRATION_SESSION_ID,
    SESSION_FACTS,
    SUCCESS_OUTCOME,
    WORKER_ID,
)

from edgeshard.profiling.codec import encode_json
from edgeshard.profiling.domain.experiment import (
    CaseState,
    CaseStatus,
    ExperimentState,
    ExperimentStatus,
    ProfilingErrorCategory,
    ProfilingExperiment,
    ProfilingRequest,
)
from edgeshard.profiling.domain.session import ProfilingSessionKind
from edgeshard.profiling.domain.snapshot import ProfileSnapshot
from edgeshard.protocol.profiling.grpc_client import (
    ProfilingAdminClient,
    WorkerProfilingClient,
    profiling_channel_options,
)
from edgeshard.protocol.profiling.grpc_server import (
    start_admin_server,
    start_profiling_server,
)
from edgeshard.protocol.profiling.mapper import (
    BuildProfileSnapshotRequest,
    BuildProfileSnapshotResponse,
    CancelExperimentRequest,
    CancelExperimentResponse,
    CancelProfilingCaseRequest,
    CancelProfilingCaseResponse,
    CloseProfilingSessionRequest,
    CloseProfilingSessionResponse,
    GetExperimentRequest,
    GetExperimentResponse,
    GetProfilingCaseRequest,
    GetProfilingCaseResponse,
    PrepareIperfServerRequest,
    PrepareIperfServerResponse,
    PrepareProfilingSessionRequest,
    PrepareProfilingSessionResponse,
    ProfilingProtocolError,
    ProfilingRejection,
    RunProfilingCaseRequest,
    RunProfilingCaseResponse,
    StartExperimentRequest,
    StartExperimentResponse,
    StopIperfServerRequest,
    StopIperfServerResponse,
)
from edgeshard.protocol.profiling.pb import profiling_pb2 as pb
from edgeshard.protocol.profiling.pb import profiling_pb2_grpc as pb_grpc

HOST = "127.0.0.1"

TOKENS = {
    "worker_id": WORKER_ID,
    "instance_id": INSTANCE_ID,
    "registration_session_id": REGISTRATION_SESSION_ID,
    "profiling_session_id": PROFILING_SESSION_ID,
}


class RecordingHandler:
    """Fake Worker profiling runner: records DTOs, returns canned responses."""

    def __init__(self) -> None:
        self.prepares: list[PrepareProfilingSessionRequest] = []
        self.runs: list[RunProfilingCaseRequest] = []
        self.gets: list[GetProfilingCaseRequest] = []
        self.cancels: list[CancelProfilingCaseRequest] = []
        self.closes: list[CloseProfilingSessionRequest] = []
        self.iperf_prepares: list[PrepareIperfServerRequest] = []
        self.iperf_stops: list[StopIperfServerRequest] = []
        self.prepare_response = PrepareProfilingSessionResponse(
            accepted=True, session_facts=SESSION_FACTS
        )
        self.run_response = RunProfilingCaseResponse(
            accepted=True, outcome=SUCCESS_OUTCOME
        )
        self.get_response = GetProfilingCaseResponse(
            accepted=True, case_state=CaseState.COMPLETED, outcome=SUCCESS_OUTCOME
        )
        self.cancel_response = CancelProfilingCaseResponse(
            accepted=True, case_state=CaseState.CANCELLED
        )
        self.close_response = CloseProfilingSessionResponse(accepted=True)
        self.iperf_prepare_response = PrepareIperfServerResponse(
            accepted=True, port=45678
        )
        self.iperf_stop_response = StopIperfServerResponse(accepted=True)
        self.error: Exception | None = None

    async def prepare_profiling_session(
        self, request: PrepareProfilingSessionRequest
    ) -> PrepareProfilingSessionResponse:
        if self.error is not None:
            raise self.error
        self.prepares.append(request)
        return self.prepare_response

    async def run_profiling_case(
        self, request: RunProfilingCaseRequest
    ) -> RunProfilingCaseResponse:
        if self.error is not None:
            raise self.error
        self.runs.append(request)
        return self.run_response

    async def get_profiling_case(
        self, request: GetProfilingCaseRequest
    ) -> GetProfilingCaseResponse:
        if self.error is not None:
            raise self.error
        self.gets.append(request)
        return self.get_response

    async def cancel_profiling_case(
        self, request: CancelProfilingCaseRequest
    ) -> CancelProfilingCaseResponse:
        if self.error is not None:
            raise self.error
        self.cancels.append(request)
        return self.cancel_response

    async def close_profiling_session(
        self, request: CloseProfilingSessionRequest
    ) -> CloseProfilingSessionResponse:
        if self.error is not None:
            raise self.error
        self.closes.append(request)
        return self.close_response

    async def prepare_iperf_server(
        self, request: PrepareIperfServerRequest
    ) -> PrepareIperfServerResponse:
        self.iperf_prepares.append(request)
        return self.iperf_prepare_response

    async def stop_iperf_server(
        self, request: StopIperfServerRequest
    ) -> StopIperfServerResponse:
        self.iperf_stops.append(request)
        return self.iperf_stop_response


@pytest.fixture
async def transport():
    handler = RecordingHandler()
    server, port = await start_profiling_server(handler, host=HOST, port=0)
    client = WorkerProfilingClient(f"{HOST}:{port}")
    try:
        yield handler, client, port
    finally:
        await client.close()
        await server.stop(grace=None)


async def test_prepare_roundtrip_over_grpc(transport) -> None:
    handler, client, _port = transport
    request = PrepareProfilingSessionRequest(
        **TOKENS, session_request=MODEL_SESSION_REQUEST
    )

    response = await client.prepare_profiling_session(request)

    assert response.accepted is True
    assert response.session_facts == SESSION_FACTS  # the full static tree survived
    (received,) = handler.prepares
    assert received == request


async def test_prepare_network_facts_roundtrip_over_grpc(transport) -> None:
    handler, client, _port = transport
    handler.prepare_response = PrepareProfilingSessionResponse(accepted=True)
    request = PrepareProfilingSessionRequest(
        **TOKENS,
        session_request=NETWORK_SESSION_REQUEST,
        network_facts=(NETWORK_FACTS,),
    )

    response = await client.prepare_profiling_session(request)

    assert response.accepted is True
    assert response.session_facts is None
    (received,) = handler.prepares
    assert received.network_facts == (NETWORK_FACTS,)


async def test_prepare_rejection_over_grpc(transport) -> None:
    handler, client, _port = transport
    handler.prepare_response = PrepareProfilingSessionResponse(
        accepted=False,
        detail="registration superseded",
        reason=ProfilingRejection.STALE_SESSION,
    )

    response = await client.prepare_profiling_session(
        PrepareProfilingSessionRequest(**TOKENS, session_request=MODEL_SESSION_REQUEST)
    )

    assert response.accepted is False
    assert response.detail == "registration superseded"
    assert response.reason is ProfilingRejection.STALE_SESSION


async def test_run_case_roundtrip_over_grpc(transport) -> None:
    handler, client, _port = transport
    request = RunProfilingCaseRequest(**TOKENS, case=MODEL_CASE)

    response = await client.run_profiling_case(request)

    assert response.accepted is True
    assert response.outcome == SUCCESS_OUTCOME
    (received,) = handler.runs
    assert received == request
    assert received.case.case_id == MODEL_CASE.case_id


async def test_run_case_device_busy_refusal_over_grpc(transport) -> None:
    """§39: a lease refusal is a typed response, not a transport error."""
    handler, client, _port = transport
    handler.run_response = RunProfilingCaseResponse(
        accepted=False, detail="gpu-0 is busy", reason=ProfilingRejection.DEVICE_BUSY
    )

    response = await client.run_profiling_case(
        RunProfilingCaseRequest(**TOKENS, case=MODEL_CASE)
    )

    assert response.accepted is False
    assert response.reason is ProfilingRejection.DEVICE_BUSY
    assert response.outcome is None


async def test_get_and_cancel_roundtrip_over_grpc(transport) -> None:
    handler, client, _port = transport

    got = await client.get_profiling_case(
        GetProfilingCaseRequest(**TOKENS, case_id=MODEL_CASE.case_id)
    )
    cancelled = await client.cancel_profiling_case(
        CancelProfilingCaseRequest(**TOKENS, case_id=MODEL_CASE.case_id)
    )

    assert got.case_state is CaseState.COMPLETED
    assert got.outcome == SUCCESS_OUTCOME
    assert cancelled.accepted is True
    assert cancelled.case_state is CaseState.CANCELLED
    assert handler.gets[0].case_id == MODEL_CASE.case_id
    assert handler.cancels[0].case_id == MODEL_CASE.case_id


async def test_close_roundtrip_over_grpc(transport) -> None:
    handler, client, _port = transport

    response = await client.close_profiling_session(CloseProfilingSessionRequest(**TOKENS))

    assert response.accepted is True
    assert handler.closes == [CloseProfilingSessionRequest(**TOKENS)]


async def test_destination_iperf_lifecycle_roundtrip_over_grpc(transport) -> None:
    handler, client, _port = transport
    token_fields = {
        key: value for key, value in TOKENS.items() if key != "profiling_session_id"
    }
    prepare = PrepareIperfServerRequest(
        **token_fields,
        server_id="server-for-case",
        timeout_s=7.5,
        bind_address="100.64.0.20",
    )
    stop = StopIperfServerRequest(**token_fields, server_id="server-for-case")

    prepared = await client.prepare_iperf_server(prepare)
    stopped = await client.stop_iperf_server(stop)

    assert prepared.port == 45678
    assert stopped.accepted
    assert handler.iperf_prepares == [prepare]
    assert handler.iperf_stops == [stop]


async def test_handler_protocol_error_aborts_rpc(transport) -> None:
    handler, client, _port = transport
    handler.error = ProfilingProtocolError("boom")

    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await client.run_profiling_case(RunProfilingCaseRequest(**TOKENS, case=MODEL_CASE))

    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


async def test_handler_value_error_aborts_rpc(transport) -> None:
    handler, client, _port = transport
    handler.error = ValueError("bad request shape")

    with pytest.raises(grpc.aio.AioRpcError) as exc_info:
        await client.run_profiling_case(RunProfilingCaseRequest(**TOKENS, case=MODEL_CASE))

    assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT


async def test_invalid_wire_request_never_reaches_handler(transport) -> None:
    """Crafted wire garbage aborts at the mapper boundary (§47)."""
    handler, _client, port = transport
    channel = grpc.aio.insecure_channel(
        f"{HOST}:{port}", options=profiling_channel_options()
    )
    stub = pb_grpc.WorkerProfilingServiceStub(channel)  # type: ignore[no-untyped-call]
    try:
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.RunProfilingCase(
                pb.RunProfilingCaseRequest(
                    worker_id=WORKER_ID,
                    instance_id=INSTANCE_ID,
                    registration_session_id=REGISTRATION_SESSION_ID,
                    profiling_session_id=PROFILING_SESSION_ID,
                    case_id="c-1",
                    case_payload="{not json",
                )
            )
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    finally:
        await channel.close()
    assert handler.runs == []


async def test_wire_case_id_mismatch_never_reaches_handler(transport) -> None:
    """The redundant envelope id must agree with the payload's canonical id (§47)."""
    handler, _client, port = transport
    channel = grpc.aio.insecure_channel(
        f"{HOST}:{port}", options=profiling_channel_options()
    )
    stub = pb_grpc.WorkerProfilingServiceStub(channel)  # type: ignore[no-untyped-call]
    try:
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.RunProfilingCase(
                pb.RunProfilingCaseRequest(
                    **TOKENS,
                    case_id="wrong-id",
                    case_payload=encode_json(MODEL_CASE),
                )
            )
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    finally:
        await channel.close()
    assert handler.runs == []


def test_client_rejects_empty_endpoint() -> None:
    with pytest.raises(ValueError, match="endpoint"):
        WorkerProfilingClient("")


async def test_client_context_manager_closes_channel() -> None:
    handler = RecordingHandler()
    server, port = await start_profiling_server(handler, host=HOST, port=0)
    try:
        async with WorkerProfilingClient(f"{HOST}:{port}") as client:
            response = await client.close_profiling_session(
                CloseProfilingSessionRequest(**TOKENS)
            )
            assert response.accepted is True
    finally:
        await server.stop(grace=None)


# ---------------------------------------------------------------------------
# ProfilingAdminService transport (§49)
# ---------------------------------------------------------------------------

ADMIN_INTENT = ProfilingRequest(
    kind=ProfilingSessionKind.MODEL,
    model=MODEL,
    dtype="fp32",
    worker_ids=(WORKER_ID,),
    device_ids=("gpu-0",),
    requested_by="operator",
)
ADMIN_EXPERIMENT = ProfilingExperiment.for_cases(
    strategy_id="default-v1", case_ids=["case-1", "case-2"], created_at=NOW
)
ADMIN_STATUS = ExperimentStatus(
    experiment=ADMIN_EXPERIMENT,
    state=ExperimentState.PARTIALLY_COMPLETED,
    cases=(
        CaseStatus("case-1", WORKER_ID, CaseState.COMPLETED),
        CaseStatus("case-2", WORKER_ID, CaseState.FAILED, failure=FAILURE),
    ),
)
ADMIN_SNAPSHOT = ProfileSnapshot(
    snapshot_id="snapshot-1",
    created_at=NOW,
    model_characterizations=(CHARACTERIZATION,),
    measurements=(RECORD,),
    network_measurements=(),
    profiling_cases=(MODEL_CASE,),
    environment_fingerprints=(ENVIRONMENT,),
)


class RecordingAdminHandler:
    """Fake MasterProfilingAdmin: records DTOs, returns canned responses."""

    def __init__(self) -> None:
        self.starts: list[StartExperimentRequest] = []
        self.gets: list[GetExperimentRequest] = []
        self.cancels: list[CancelExperimentRequest] = []
        self.snapshots: list[BuildProfileSnapshotRequest] = []
        self.start_response = StartExperimentResponse(
            accepted=True, experiment_id=ADMIN_EXPERIMENT.experiment_id
        )
        self.get_response = GetExperimentResponse(found=True, status=ADMIN_STATUS)
        self.cancel_response = CancelExperimentResponse(
            accepted=True, detail="experiment cancelled"
        )
        self.snapshot_response = BuildProfileSnapshotResponse(
            accepted=True, snapshot=ADMIN_SNAPSHOT
        )
        self.error: Exception | None = None

    async def start_experiment(
        self, request: StartExperimentRequest
    ) -> StartExperimentResponse:
        if self.error is not None:
            raise self.error
        self.starts.append(request)
        return self.start_response

    async def get_experiment(
        self, request: GetExperimentRequest
    ) -> GetExperimentResponse:
        if self.error is not None:
            raise self.error
        self.gets.append(request)
        return self.get_response

    async def cancel_experiment(
        self, request: CancelExperimentRequest
    ) -> CancelExperimentResponse:
        if self.error is not None:
            raise self.error
        self.cancels.append(request)
        return self.cancel_response

    async def build_profile_snapshot(
        self, request: BuildProfileSnapshotRequest
    ) -> BuildProfileSnapshotResponse:
        if self.error is not None:
            raise self.error
        self.snapshots.append(request)
        return self.snapshot_response


@pytest.fixture
async def admin_transport():
    handler = RecordingAdminHandler()
    server, port = await start_admin_server(handler, host=HOST, port=0)
    client = ProfilingAdminClient(f"{HOST}:{port}")
    try:
        yield handler, client, port
    finally:
        await client.close()
        await server.stop(grace=None)


async def test_admin_start_roundtrip_over_grpc(admin_transport) -> None:
    handler, client, _port = admin_transport
    request = StartExperimentRequest(request=ADMIN_INTENT)

    response = await client.start_experiment(request)

    assert response.accepted is True
    assert response.experiment_id == ADMIN_EXPERIMENT.experiment_id
    (received,) = handler.starts
    assert received == request  # the full intent tree survived the wire


async def test_admin_start_rejection_over_grpc(admin_transport) -> None:
    handler, client, _port = admin_transport
    handler.start_response = StartExperimentResponse(
        accepted=False, detail="no registered worker hosts the profiling service"
    )

    response = await client.start_experiment(StartExperimentRequest(request=ADMIN_INTENT))

    assert response.accepted is False
    assert "no registered worker" in response.detail
    assert response.experiment_id == ""


async def test_admin_get_roundtrip_over_grpc(admin_transport) -> None:
    handler, client, _port = admin_transport

    response = await client.get_experiment(
        GetExperimentRequest(experiment_id=ADMIN_EXPERIMENT.experiment_id)
    )

    assert response.found is True
    assert response.status == ADMIN_STATUS  # states AND typed failures survived
    assert response.status is not None
    failed = response.status.cases[1]
    assert failed.failure is not None
    assert failed.failure.category is ProfilingErrorCategory.DEVICE_BUSY
    (received,) = handler.gets
    assert received.experiment_id == ADMIN_EXPERIMENT.experiment_id


async def test_admin_get_not_found_over_grpc(admin_transport) -> None:
    handler, client, _port = admin_transport
    handler.get_response = GetExperimentResponse(found=False)

    response = await client.get_experiment(GetExperimentRequest(experiment_id="nope"))

    assert response.found is False
    assert response.status is None


async def test_admin_cancel_roundtrip_over_grpc(admin_transport) -> None:
    handler, client, _port = admin_transport

    response = await client.cancel_experiment(
        CancelExperimentRequest(experiment_id=ADMIN_EXPERIMENT.experiment_id)
    )

    assert response.accepted is True
    assert response.detail == "experiment cancelled"
    (received,) = handler.cancels
    assert received.experiment_id == ADMIN_EXPERIMENT.experiment_id


async def test_admin_snapshot_roundtrip_over_grpc(admin_transport) -> None:
    handler, client, _port = admin_transport

    response = await client.build_profile_snapshot(BuildProfileSnapshotRequest())

    assert response.accepted is True
    assert response.snapshot == ADMIN_SNAPSHOT  # the whole §46 view survived
    assert len(handler.snapshots) == 1


async def test_admin_handler_value_error_aborts_rpc(admin_transport) -> None:
    handler, client, _port = admin_transport
    handler.error = ValueError("domain says no")

    with pytest.raises(grpc.aio.AioRpcError) as excinfo:
        await client.start_experiment(StartExperimentRequest(request=ADMIN_INTENT))

    assert excinfo.value.code() is grpc.StatusCode.INVALID_ARGUMENT
    assert "domain says no" in excinfo.value.details()


async def test_admin_invalid_wire_request_never_reaches_handler(
    admin_transport,
) -> None:
    handler, _client, port = admin_transport
    stub = pb_grpc.ProfilingAdminServiceStub(  # type: ignore[no-untyped-call]
        grpc.aio.insecure_channel(
            f"{HOST}:{port}", options=profiling_channel_options()
        )
    )

    with pytest.raises(grpc.aio.AioRpcError) as excinfo:
        await stub.StartExperiment(pb.StartExperimentRequest(request_payload=""))

    assert excinfo.value.code() is grpc.StatusCode.INVALID_ARGUMENT
    assert handler.starts == []


def test_admin_client_rejects_empty_endpoint() -> None:
    with pytest.raises(ValueError, match="endpoint"):
        ProfilingAdminClient("")
