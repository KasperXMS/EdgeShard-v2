"""Profiling-plane domain ↔ wire mapping tests (Phase 2 spec §41).

Every domain tree the WorkerProfilingService carries — session requests,
session facts, cases, outcomes, network facts — must survive the
typed-envelope + canonical-JSON round trip exactly, and every redundant
wire copy (case_id envelope field vs payload, accepted vs reason) must be
cross-checked at the boundary (§47): malformed payloads, unknown enums and
mismatched ids fail loudly instead of degrading silently.
"""

from __future__ import annotations

import dataclasses

import pytest
from wire_fixtures import (
    CHARACTERIZATION,
    ENVIRONMENT,
    FAILURE,
    FAILURE_OUTCOME,
    INSTANCE_ID,
    MODEL,
    MODEL_CASE,
    MODEL_SESSION_REQUEST,
    NETWORK_CASE,
    NETWORK_FACTS,
    NETWORK_SESSION_REQUEST,
    NOW,
    OPERATOR_SESSION_REQUEST,
    PEER_NETWORK_FACTS,
    PROFILING_SESSION_ID,
    RECORD,
    REGISTRATION_SESSION_ID,
    SESSION_FACTS,
    SUCCESS_OUTCOME,
    WORKER_ID,
)

from edgeshard.profiling.domain.experiment import (
    CaseOutcome,
    CaseState,
    CaseStatus,
    ExperimentState,
    ExperimentStatus,
    ProfilingErrorCategory,
    ProfilingExperiment,
    ProfilingRequest,
)
from edgeshard.profiling.domain.session import ModelSessionFacts, ProfilingSessionKind
from edgeshard.profiling.domain.snapshot import ProfileSnapshot
from edgeshard.protocol.profiling import mapper
from edgeshard.protocol.profiling.mapper import (
    CancelProfilingCaseRequest,
    CancelProfilingCaseResponse,
    CloseProfilingSessionRequest,
    CloseProfilingSessionResponse,
    GetProfilingCaseRequest,
    GetProfilingCaseResponse,
    PrepareIperfServerRequest,
    PrepareProfilingSessionRequest,
    PrepareProfilingSessionResponse,
    ProfilingProtocolError,
    ProfilingRejection,
    RunProfilingCaseRequest,
    RunProfilingCaseResponse,
    StopIperfServerRequest,
)
from edgeshard.protocol.profiling.pb import profiling_pb2 as pb

TOKENS = {
    "worker_id": WORKER_ID,
    "instance_id": INSTANCE_ID,
    "registration_session_id": REGISTRATION_SESSION_ID,
    "profiling_session_id": PROFILING_SESSION_ID,
}


def make_prepare_request(
    *, session_request=MODEL_SESSION_REQUEST, network_facts=()
) -> PrepareProfilingSessionRequest:
    return PrepareProfilingSessionRequest(
        **TOKENS, session_request=session_request, network_facts=network_facts
    )


# ---------------------------------------------------------------------------
# PrepareProfilingSession
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "session_request",
    [MODEL_SESSION_REQUEST, OPERATOR_SESSION_REQUEST],
    ids=lambda r: r.kind.value,
)
def test_prepare_request_roundtrip(session_request) -> None:
    request = make_prepare_request(session_request=session_request)
    assert mapper.prepare_request_from_wire(
        mapper.prepare_request_to_wire(request)
    ) == request


def test_prepare_request_network_facts_roundtrip() -> None:
    """The Master-resolved facts batch crosses the wire intact (§52.2)."""
    request = make_prepare_request(
        session_request=NETWORK_SESSION_REQUEST,
        network_facts=(NETWORK_FACTS, PEER_NETWORK_FACTS),
    )
    wire = mapper.prepare_request_to_wire(request)
    assert wire.network_facts_payload  # absence is the empty string, never a sentinel
    restored = mapper.prepare_request_from_wire(wire)
    assert restored == request
    assert restored.network_facts == (NETWORK_FACTS, PEER_NETWORK_FACTS)


def test_prepare_request_without_facts_leaves_payload_empty() -> None:
    wire = mapper.prepare_request_to_wire(make_prepare_request())
    assert wire.network_facts_payload == ""
    assert mapper.prepare_request_from_wire(wire).network_facts == ()


def test_prepare_request_network_session_requires_facts() -> None:
    """The executing Worker never guesses destinations (§52.2)."""
    with pytest.raises(ValueError, match="network facts"):
        make_prepare_request(session_request=NETWORK_SESSION_REQUEST)


def test_prepare_request_device_session_forbids_facts() -> None:
    with pytest.raises(ValueError, match="must not carry network facts"):
        make_prepare_request(
            session_request=OPERATOR_SESSION_REQUEST, network_facts=(NETWORK_FACTS,)
        )


def test_iperf_server_lifecycle_requests_roundtrip() -> None:
    token_fields = {
        key: value for key, value in TOKENS.items() if key != "profiling_session_id"
    }
    prepare = PrepareIperfServerRequest(
        **token_fields,
        server_id="case-server",
        timeout_s=12.0,
        bind_address="100.64.0.20",
    )
    stop = StopIperfServerRequest(**token_fields, server_id="case-server")
    assert mapper.prepare_iperf_server_request_from_wire(
        mapper.prepare_iperf_server_request_to_wire(prepare)
    ) == prepare
    assert mapper.stop_iperf_server_request_from_wire(
        mapper.stop_iperf_server_request_to_wire(stop)
    ) == stop


@pytest.mark.parametrize(
    "field", sorted(TOKENS), ids=lambda name: name
)
def test_prepare_request_requires_nonempty_tokens(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        PrepareProfilingSessionRequest(
            **{**TOKENS, field: ""}, session_request=MODEL_SESSION_REQUEST
        )


def test_prepare_response_with_facts_roundtrip() -> None:
    response = PrepareProfilingSessionResponse(accepted=True, session_facts=SESSION_FACTS)
    restored = mapper.prepare_response_from_wire(
        mapper.prepare_response_to_wire(response)
    )
    assert restored == response
    assert isinstance(restored.session_facts, ModelSessionFacts)


def test_prepare_response_without_facts_roundtrip() -> None:
    """OPERATOR/NETWORK sessions answer with no facts: empty payload → None."""
    response = PrepareProfilingSessionResponse(accepted=True)
    wire = mapper.prepare_response_to_wire(response)
    assert wire.session_facts_payload == ""
    restored = mapper.prepare_response_from_wire(wire)
    assert restored == response
    assert restored.session_facts is None


def test_prepare_response_rejection_roundtrip() -> None:
    response = PrepareProfilingSessionResponse(
        accepted=False, detail="gpu-0 is busy", reason=ProfilingRejection.DEVICE_BUSY
    )
    assert mapper.prepare_response_from_wire(
        mapper.prepare_response_to_wire(response)
    ) == response


def test_prepare_response_rejected_requires_detail_and_reason() -> None:
    with pytest.raises(ValueError, match="detail"):
        PrepareProfilingSessionResponse(
            accepted=False, reason=ProfilingRejection.UNKNOWN_SESSION
        )
    with pytest.raises(ValueError, match="reason"):
        PrepareProfilingSessionResponse(accepted=False, detail="unknown session")


def test_prepare_response_rejected_forbids_facts() -> None:
    with pytest.raises(ValueError, match="session facts"):
        PrepareProfilingSessionResponse(
            accepted=False,
            detail="busy",
            reason=ProfilingRejection.DEVICE_BUSY,
            session_facts=SESSION_FACTS,
        )


def test_prepare_response_accepted_forbids_reason() -> None:
    with pytest.raises(ValueError, match="rejection reason"):
        PrepareProfilingSessionResponse(
            accepted=True, reason=ProfilingRejection.UNKNOWN_SESSION
        )


def test_prepare_response_typed_failure_roundtrip() -> None:
    """§42: a preparation failure keeps its category across the wire."""
    response = PrepareProfilingSessionResponse(
        accepted=False,
        detail="no profiling adapter for model_type 'gpt2'",
        failure=FAILURE,
    )
    wire = mapper.prepare_response_to_wire(response)
    assert wire.failure_payload  # the typed failure rides its own channel
    assert wire.reason == pb.PROFILING_REJECTION_REASON_UNSPECIFIED
    restored = mapper.prepare_response_from_wire(wire)
    assert restored == response
    assert restored.failure is not None
    assert restored.failure.category is ProfilingErrorCategory.DEVICE_BUSY
    assert restored.reason is None


def test_prepare_response_rejected_requires_reason_xor_failure() -> None:
    """A rejection carries exactly one verdict channel (§47)."""
    with pytest.raises(ValueError, match="exactly one of reason or failure"):
        PrepareProfilingSessionResponse(accepted=False, detail="boom")
    with pytest.raises(ValueError, match="exactly one of reason or failure"):
        PrepareProfilingSessionResponse(
            accepted=False,
            detail="boom",
            reason=ProfilingRejection.DEVICE_BUSY,
            failure=FAILURE,
        )


def test_prepare_response_accepted_forbids_failure() -> None:
    with pytest.raises(ValueError, match="failure"):
        PrepareProfilingSessionResponse(accepted=True, failure=FAILURE)


def test_prepare_response_wire_rejected_without_verdict_fails() -> None:
    """A crafted rejection with neither reason nor failure fails loudly."""
    wire = pb.PrepareProfilingSessionResponse(accepted=False, detail="boom")
    with pytest.raises(ProfilingProtocolError, match="unknown rejection reason"):
        mapper.prepare_response_from_wire(wire)


def test_prepare_response_wire_both_verdicts_fails() -> None:
    """A crafted rejection carrying both channels fails loudly (§47)."""
    wire = mapper.prepare_response_to_wire(
        PrepareProfilingSessionResponse(
            accepted=False,
            detail="boom",
            reason=ProfilingRejection.DEVICE_BUSY,
        )
    )
    wire.failure_payload = mapper.prepare_response_to_wire(
        PrepareProfilingSessionResponse(accepted=False, detail="boom", failure=FAILURE)
    ).failure_payload
    with pytest.raises(ValueError, match="exactly one of reason or failure"):
        mapper.prepare_response_from_wire(wire)


# ---------------------------------------------------------------------------
# RunProfilingCase
# ---------------------------------------------------------------------------


def make_run_request(case=MODEL_CASE) -> RunProfilingCaseRequest:
    return RunProfilingCaseRequest(**TOKENS, case=case)


@pytest.mark.parametrize("case", [MODEL_CASE, NETWORK_CASE], ids=["model", "network"])
def test_run_request_roundtrip(case) -> None:
    request = make_run_request(case)
    wire = mapper.run_case_request_to_wire(request)
    assert wire.case_id == case.case_id  # the envelope id mirrors the payload
    assert mapper.run_case_request_from_wire(wire) == request


def test_run_request_case_id_mismatch_rejected() -> None:
    """The redundant envelope copy must agree with the canonical id (§47)."""
    wire = mapper.run_case_request_to_wire(make_run_request())
    wire.case_id = "not-the-canonical-id"
    with pytest.raises(ProfilingProtocolError, match="case_id mismatch"):
        mapper.run_case_request_from_wire(wire)


def test_run_request_foreign_case_rejected() -> None:
    """A case assigned to another worker must never ride this envelope (§8.2)."""
    foreign = dataclasses.replace(MODEL_CASE, worker_id="w-other")
    with pytest.raises(ValueError, match="worker_id mismatch"):
        make_run_request(foreign)


def test_run_response_accepted_roundtrip() -> None:
    for outcome in (SUCCESS_OUTCOME, FAILURE_OUTCOME):
        response = RunProfilingCaseResponse(accepted=True, outcome=outcome)
        restored = mapper.run_case_response_from_wire(
            mapper.run_case_response_to_wire(response)
        )
        assert restored == response
        assert isinstance(restored.outcome, CaseOutcome)


def test_run_response_rejection_roundtrip() -> None:
    """§42: a refusal is a transport-level verdict, distinct from a failure."""
    response = RunProfilingCaseResponse(
        accepted=False, detail="gpu-0 is busy", reason=ProfilingRejection.DEVICE_BUSY
    )
    assert mapper.run_case_response_from_wire(
        mapper.run_case_response_to_wire(response)
    ) == response


def test_run_response_accepted_requires_outcome() -> None:
    with pytest.raises(ValueError, match="outcome"):
        RunProfilingCaseResponse(accepted=True)


def test_run_response_rejected_forbids_outcome() -> None:
    with pytest.raises(ValueError, match="outcome"):
        RunProfilingCaseResponse(
            accepted=False,
            detail="busy",
            reason=ProfilingRejection.DEVICE_BUSY,
            outcome=SUCCESS_OUTCOME,
        )


def test_run_response_wire_missing_outcome_rejected() -> None:
    """A crafted accepted response without payload fails at the boundary."""
    wire = pb.RunProfilingCaseResponse(accepted=True)
    with pytest.raises(ProfilingProtocolError, match="no outcome payload"):
        mapper.run_case_response_from_wire(wire)


# ---------------------------------------------------------------------------
# GetProfilingCase / CancelProfilingCase
# ---------------------------------------------------------------------------


def make_get_request(case_id: str = "case-1") -> GetProfilingCaseRequest:
    return GetProfilingCaseRequest(**TOKENS, case_id=case_id)


def test_get_request_roundtrip() -> None:
    request = make_get_request(MODEL_CASE.case_id)
    assert mapper.get_case_request_from_wire(
        mapper.get_case_request_to_wire(request)
    ) == request


def test_get_request_requires_case_id() -> None:
    with pytest.raises(ValueError, match="case_id"):
        make_get_request("")


@pytest.mark.parametrize(
    ("state", "outcome"),
    [
        (CaseState.PENDING, None),
        (CaseState.RUNNING, None),
        (CaseState.COMPLETED, SUCCESS_OUTCOME),
        (CaseState.FAILED, FAILURE_OUTCOME),
        (CaseState.CANCELLED, FAILURE_OUTCOME),
    ],
    ids=["pending", "running", "completed", "failed", "cancelled"],
)
def test_get_response_state_matrix_roundtrip(state: CaseState, outcome) -> None:
    response = GetProfilingCaseResponse(
        accepted=True, case_state=state, outcome=outcome
    )
    restored = mapper.get_case_response_from_wire(
        mapper.get_case_response_to_wire(response)
    )
    assert restored == response
    assert restored.case_state is state


@pytest.mark.parametrize("state", [CaseState.COMPLETED, CaseState.FAILED, CaseState.CANCELLED])
def test_get_response_terminal_state_requires_outcome(state: CaseState) -> None:
    """§44: a terminal case without its outcome is an incomplete ledger view."""
    with pytest.raises(ValueError, match="outcome"):
        GetProfilingCaseResponse(accepted=True, case_state=state)


@pytest.mark.parametrize("state", [CaseState.PENDING, CaseState.RUNNING])
def test_get_response_nonterminal_state_forbids_outcome(state: CaseState) -> None:
    with pytest.raises(ValueError, match="outcome"):
        GetProfilingCaseResponse(
            accepted=True, case_state=state, outcome=SUCCESS_OUTCOME
        )


def test_get_response_accepted_requires_state() -> None:
    with pytest.raises(ValueError, match="case state"):
        GetProfilingCaseResponse(accepted=True)


def test_get_response_unknown_case_rejection_roundtrip() -> None:
    response = GetProfilingCaseResponse(
        accepted=False,
        detail="session does not track this case",
        reason=ProfilingRejection.UNKNOWN_CASE,
    )
    assert mapper.get_case_response_from_wire(
        mapper.get_case_response_to_wire(response)
    ) == response


def test_get_response_wire_missing_state_rejected() -> None:
    wire = pb.GetProfilingCaseResponse(accepted=True)
    with pytest.raises(ProfilingProtocolError, match="no case state"):
        mapper.get_case_response_from_wire(wire)


def test_get_response_wire_unknown_state_rejected() -> None:
    wire = pb.GetProfilingCaseResponse(accepted=True, case_state="exploded")
    with pytest.raises(ProfilingProtocolError, match="unknown case state"):
        mapper.get_case_response_from_wire(wire)


def make_cancel_request(case_id: str = "case-1") -> CancelProfilingCaseRequest:
    return CancelProfilingCaseRequest(**TOKENS, case_id=case_id)


def test_cancel_request_roundtrip() -> None:
    request = make_cancel_request(MODEL_CASE.case_id)
    assert mapper.cancel_case_request_from_wire(
        mapper.cancel_case_request_to_wire(request)
    ) == request


def test_cancel_response_roundtrip() -> None:
    for response in (
        CancelProfilingCaseResponse(accepted=True, case_state=CaseState.CANCELLED),
        # Already terminal: cancellation must not undo history (§44).
        CancelProfilingCaseResponse(accepted=True, case_state=CaseState.COMPLETED),
        CancelProfilingCaseResponse(
            accepted=False,
            detail="unknown session",
            reason=ProfilingRejection.UNKNOWN_SESSION,
        ),
    ):
        assert mapper.cancel_case_response_from_wire(
            mapper.cancel_case_response_to_wire(response)
        ) == response


def test_cancel_response_accepted_requires_state() -> None:
    with pytest.raises(ValueError, match="case state"):
        CancelProfilingCaseResponse(accepted=True)


def test_cancel_response_rejected_forbids_state() -> None:
    with pytest.raises(ValueError, match="case state"):
        CancelProfilingCaseResponse(
            accepted=False,
            detail="stale",
            reason=ProfilingRejection.STALE_SESSION,
            case_state=CaseState.CANCELLED,
        )


# ---------------------------------------------------------------------------
# CloseProfilingSession
# ---------------------------------------------------------------------------


def test_close_request_roundtrip() -> None:
    request = CloseProfilingSessionRequest(**TOKENS)
    assert mapper.close_session_request_from_wire(
        mapper.close_session_request_to_wire(request)
    ) == request


def test_close_response_roundtrip() -> None:
    for response in (
        CloseProfilingSessionResponse(accepted=True),
        CloseProfilingSessionResponse(
            accepted=False,
            detail="registration superseded",
            reason=ProfilingRejection.STALE_SESSION,
        ),
    ):
        assert mapper.close_session_response_from_wire(
            mapper.close_session_response_to_wire(response)
        ) == response


def test_close_response_rejected_requires_detail_and_reason() -> None:
    with pytest.raises(ValueError, match="detail"):
        CloseProfilingSessionResponse(
            accepted=False, reason=ProfilingRejection.STALE_SESSION
        )
    with pytest.raises(ValueError, match="reason"):
        CloseProfilingSessionResponse(accepted=False, detail="stale")


# ---------------------------------------------------------------------------
# Boundary strictness (§47)
# ---------------------------------------------------------------------------


def test_every_rejection_reason_survives_the_wire() -> None:
    for rejection in ProfilingRejection:
        response = RunProfilingCaseResponse(
            accepted=False, detail=f"refused: {rejection.value}", reason=rejection
        )
        restored = mapper.run_case_response_from_wire(
            mapper.run_case_response_to_wire(response)
        )
        assert restored.reason is rejection


def test_unknown_wire_rejection_reason_rejected() -> None:
    wire = mapper.run_case_response_to_wire(
        RunProfilingCaseResponse(
            accepted=False,
            detail="x",
            reason=ProfilingRejection.UNKNOWN_SESSION,
        )
    )
    wire.reason = 99  # proto3 enums are open; decode must reject
    with pytest.raises(ProfilingProtocolError, match="unknown rejection reason"):
        mapper.run_case_response_from_wire(wire)


def test_missing_mandatory_payload_rejected() -> None:
    wire = pb.PrepareProfilingSessionRequest(
        worker_id=WORKER_ID,
        instance_id=INSTANCE_ID,
        registration_session_id=REGISTRATION_SESSION_ID,
        profiling_session_id=PROFILING_SESSION_ID,
        session_request_payload="",
    )
    with pytest.raises(ProfilingProtocolError, match="missing session request payload"):
        mapper.prepare_request_from_wire(wire)


def test_missing_case_payload_rejected() -> None:
    wire = pb.RunProfilingCaseRequest(**TOKENS, case_id="c")
    with pytest.raises(ProfilingProtocolError, match="missing case payload"):
        mapper.run_case_request_from_wire(wire)


def test_malformed_payload_rejected() -> None:
    wire = mapper.prepare_request_to_wire(make_prepare_request())
    wire.session_request_payload = "{not json"
    with pytest.raises(ProfilingProtocolError, match="malformed session request"):
        mapper.prepare_request_from_wire(wire)


def test_corrupt_domain_payload_rejected() -> None:
    """A payload that parses as JSON but violates the domain fails loudly."""
    wire = mapper.run_case_request_to_wire(make_run_request())
    wire.case_payload = wire.case_payload.replace('"case_id"', '"case_id_RENAMED"')
    with pytest.raises(ProfilingProtocolError, match="malformed case payload"):
        mapper.run_case_request_from_wire(wire)


def test_session_kind_mismatch_reason_available() -> None:
    """§41: running a network case in an operator session is a typed refusal."""
    response = RunProfilingCaseResponse(
        accepted=False,
        detail="session kind operator cannot run network cases",
        reason=ProfilingRejection.SESSION_KIND_MISMATCH,
    )
    restored = mapper.run_case_response_from_wire(
        mapper.run_case_response_to_wire(response)
    )
    assert restored.reason is ProfilingRejection.SESSION_KIND_MISMATCH


# ---------------------------------------------------------------------------
# ProfilingAdminService DTOs (§49): intents in, status/snapshots out
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


def test_start_request_roundtrip() -> None:
    request = mapper.StartExperimentRequest(request=ADMIN_INTENT)
    wire = mapper.start_experiment_request_to_wire(request)
    assert wire.request_payload
    assert mapper.start_experiment_request_from_wire(wire) == request


def test_start_request_missing_payload_rejected() -> None:
    with pytest.raises(ProfilingProtocolError, match="profiling request"):
        mapper.start_experiment_request_from_wire(pb.StartExperimentRequest())


def test_start_request_malformed_payload_rejected() -> None:
    with pytest.raises(ProfilingProtocolError, match="malformed"):
        mapper.start_experiment_request_from_wire(
            pb.StartExperimentRequest(request_payload="{not json")
        )


def test_start_response_roundtrip() -> None:
    accepted = mapper.StartExperimentResponse(
        accepted=True, experiment_id=ADMIN_EXPERIMENT.experiment_id
    )
    wire = mapper.start_experiment_response_to_wire(accepted)
    assert mapper.start_experiment_response_from_wire(wire) == accepted
    rejected = mapper.StartExperimentResponse(accepted=False, detail="no workers")
    assert (
        mapper.start_experiment_response_from_wire(
            mapper.start_experiment_response_to_wire(rejected)
        )
        == rejected
    )


def test_start_response_verdict_consistency() -> None:
    with pytest.raises(ValueError, match="experiment_id"):
        mapper.StartExperimentResponse(accepted=True)
    with pytest.raises(ValueError, match="detail"):
        mapper.StartExperimentResponse(accepted=False)
    with pytest.raises(ValueError, match="experiment_id"):
        mapper.StartExperimentResponse(
            accepted=False, detail="no", experiment_id="e-1"
        )


def test_get_experiment_roundtrip() -> None:
    request = mapper.GetExperimentRequest(experiment_id="e-1")
    assert (
        mapper.get_experiment_request_from_wire(
            mapper.get_experiment_request_to_wire(request)
        )
        == request
    )
    with pytest.raises(ValueError, match="experiment_id"):
        mapper.GetExperimentRequest(experiment_id="")

    found = mapper.GetExperimentResponse(found=True, status=ADMIN_STATUS)
    wire = mapper.get_experiment_response_to_wire(found)
    assert wire.status_payload
    assert mapper.get_experiment_response_from_wire(wire) == found

    missing = mapper.GetExperimentResponse(found=False)
    assert mapper.get_experiment_response_to_wire(missing).status_payload == ""
    assert mapper.get_experiment_response_from_wire(
        mapper.get_experiment_response_to_wire(missing)
    ) == missing


def test_get_experiment_response_found_requires_status() -> None:
    with pytest.raises(ValueError, match="found"):
        mapper.GetExperimentResponse(found=True)
    with pytest.raises(ValueError, match="found"):
        mapper.GetExperimentResponse(found=False, status=ADMIN_STATUS)


def test_get_experiment_response_malformed_status_rejected() -> None:
    with pytest.raises(ProfilingProtocolError, match="experiment status"):
        mapper.get_experiment_response_from_wire(
            pb.GetExperimentResponse(found=True, status_payload="{oops")
        )


def test_cancel_experiment_roundtrip() -> None:
    request = mapper.CancelExperimentRequest(experiment_id="e-1")
    assert (
        mapper.cancel_experiment_request_from_wire(
            mapper.cancel_experiment_request_to_wire(request)
        )
        == request
    )
    for response in (
        mapper.CancelExperimentResponse(accepted=True, detail="cancelled"),
        mapper.CancelExperimentResponse(accepted=True),
        mapper.CancelExperimentResponse(accepted=False, detail="unknown experiment"),
    ):
        assert (
            mapper.cancel_experiment_response_from_wire(
                mapper.cancel_experiment_response_to_wire(response)
            )
            == response
        )
    with pytest.raises(ValueError, match="detail"):
        mapper.CancelExperimentResponse(accepted=False)


def test_build_snapshot_roundtrip() -> None:
    request = mapper.BuildProfileSnapshotRequest()
    assert (
        mapper.build_snapshot_request_from_wire(
            mapper.build_snapshot_request_to_wire(request)
        )
        == request
    )

    accepted = mapper.BuildProfileSnapshotResponse(
        accepted=True, snapshot=ADMIN_SNAPSHOT
    )
    wire = mapper.build_snapshot_response_to_wire(accepted)
    assert wire.snapshot_payload
    assert mapper.build_snapshot_response_from_wire(wire) == accepted

    rejected = mapper.BuildProfileSnapshotResponse(
        accepted=False, detail="store failed"
    )
    assert mapper.build_snapshot_response_to_wire(rejected).snapshot_payload == ""
    assert mapper.build_snapshot_response_from_wire(
        mapper.build_snapshot_response_to_wire(rejected)
    ) == rejected


def test_build_snapshot_response_verdict_consistency() -> None:
    with pytest.raises(ValueError, match="snapshot"):
        mapper.BuildProfileSnapshotResponse(accepted=True)
    with pytest.raises(ValueError, match="snapshot"):
        mapper.BuildProfileSnapshotResponse(
            accepted=False, detail="x", snapshot=ADMIN_SNAPSHOT
        )
    with pytest.raises(ValueError, match="detail"):
        mapper.BuildProfileSnapshotResponse(accepted=False)
