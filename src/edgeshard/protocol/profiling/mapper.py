"""Profiling-plane domain ↔ wire mapping (Phase 2 spec §41).

This module is the only boundary that translates between the profiling
domain objects and the profiling protobuf messages; generated protobuf
objects never reach the ProfilingController, the Worker profiling runner,
or the store (mirroring the Phase 1 ``protocol.control.mapper`` discipline,
spec §40).

Encoding split (the §41 simplification pinned in ``proto/profiling.proto``):
identity/session tokens and canonical ids are real proto fields, while the
deep variant-rich domain trees (session requests, session facts, cases,
outcomes) travel as deterministic canonical-JSON payload strings produced
by :mod:`edgeshard.profiling.codec`. The mapper is where both halves are
joined — and where every redundant copy is cross-checked (§47): the
``case_id`` envelope field must equal the decoded case's canonical id, and
the case's assigned executor must equal the envelope's ``worker_id``.

Fail-loudly policy (spec §47): malformed or missing payloads, unknown enum
values, and mismatched redundant fields raise :class:`ProfilingProtocolError`
at the boundary instead of producing silently degraded domain objects.
Missing optional content is an *empty payload string* on the wire and
``None``/``()`` in the DTO — never a sentinel (§17, §52.2).

Semantic refusals are not protocol errors: an unknown/stale session or a
busy device travels as an ``accepted=False`` response with a typed
:class:`ProfilingRejection` the Master's recovery keys off (§41), exactly
as Phase 1 heartbeats carry ``RejectionReason``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from edgeshard.model.errors import EdgeShardError
from edgeshard.profiling.codec import PayloadCodecError, decode_json, encode_json
from edgeshard.profiling.domain.experiment import (
    CaseOutcome,
    CaseState,
    ExperimentStatus,
    ProfilingCase,
    ProfilingFailure,
    ProfilingRequest,
)
from edgeshard.profiling.domain.session import (
    ModelSessionFacts,
    ProfilingSessionKind,
    ProfilingSessionRequest,
)
from edgeshard.profiling.domain.snapshot import ProfileSnapshot
from edgeshard.profiling.network.classifier import WorkerNetworkFacts
from edgeshard.protocol.profiling.pb import profiling_pb2 as pb


class ProfilingProtocolError(EdgeShardError):
    """Profiling protocol violation: payload shape, enum value, or cross-check."""


class ProfilingRejection(StrEnum):
    """Why a Worker refused a profiling RPC (spec §41).

    The Master's recovery keys off this enum, never off the detail string:
    ``UNKNOWN_SESSION``/``SESSION_CLOSED`` mean "prepare (again) first";
    ``STALE_SESSION`` means this registration was superseded — the session's
    results MUST NOT be published and the Master must re-select;
    ``SESSION_KIND_MISMATCH``/``UNKNOWN_CASE`` are Master-side protocol
    violations that must fail loudly; ``DEVICE_BUSY`` is the §39 lease
    rejection (defer the case or re-select a worker).
    """

    UNKNOWN_SESSION = "unknown_session"
    STALE_SESSION = "stale_session"
    SESSION_CLOSED = "session_closed"
    SESSION_KIND_MISMATCH = "session_kind_mismatch"
    DEVICE_BUSY = "device_busy"
    UNKNOWN_CASE = "unknown_case"


_REJECTION_TO_WIRE: dict[ProfilingRejection, pb.ProfilingRejectionReason] = {
    ProfilingRejection.UNKNOWN_SESSION: pb.PROFILING_REJECTION_REASON_UNKNOWN_SESSION,
    ProfilingRejection.STALE_SESSION: pb.PROFILING_REJECTION_REASON_STALE_SESSION,
    ProfilingRejection.SESSION_CLOSED: pb.PROFILING_REJECTION_REASON_SESSION_CLOSED,
    ProfilingRejection.SESSION_KIND_MISMATCH: (
        pb.PROFILING_REJECTION_REASON_SESSION_KIND_MISMATCH
    ),
    ProfilingRejection.DEVICE_BUSY: pb.PROFILING_REJECTION_REASON_DEVICE_BUSY,
    ProfilingRejection.UNKNOWN_CASE: pb.PROFILING_REJECTION_REASON_UNKNOWN_CASE,
}
_WIRE_TO_REJECTION: dict[pb.ProfilingRejectionReason, ProfilingRejection] = {
    value: key for key, value in _REJECTION_TO_WIRE.items()
}

_TERMINAL_CASE_STATES = frozenset(
    {CaseState.COMPLETED, CaseState.FAILED, CaseState.CANCELLED}
)


@dataclass(frozen=True)
class NetworkFactsBatch:
    """Wire-shape wrapper around the Master-resolved network facts (§41).

    The codec needs a taggable top-level object — a bare tuple has no type
    tag — so the ``network_facts_payload`` carries this one-field batch.
    """

    facts: tuple[WorkerNetworkFacts, ...]


def _check_tokens(
    worker_id: str,
    instance_id: str,
    registration_session_id: str,
    profiling_session_id: str,
) -> None:
    """Every worker-plane RPC rides the Phase 1 registration tokens (§41)."""
    if not worker_id:
        raise ValueError("worker_id must not be empty")
    if not instance_id:
        raise ValueError("instance_id must not be empty")
    if not registration_session_id:
        raise ValueError("registration_session_id must not be empty")
    if not profiling_session_id:
        raise ValueError("profiling_session_id must not be empty")


def _decode_payload[T](cls: type[T], text: str, what: str) -> T:
    """Decode a mandatory canonical-JSON payload; empty or corrupt fails loudly."""
    if not text:
        raise ProfilingProtocolError(f"missing {what} payload")
    try:
        return decode_json(cls, text)
    except PayloadCodecError as exc:
        raise ProfilingProtocolError(f"malformed {what} payload: {exc}") from exc


def _decode_optional_payload[T](cls: type[T], text: str, what: str) -> T | None:
    """Decode an optional payload; the empty string is absence, never a sentinel."""
    if not text:
        return None
    try:
        return decode_json(cls, text)
    except PayloadCodecError as exc:
        raise ProfilingProtocolError(f"malformed {what} payload: {exc}") from exc


def _rejection_to_wire(rejection: ProfilingRejection | None) -> pb.ProfilingRejectionReason:
    if rejection is None:
        return pb.PROFILING_REJECTION_REASON_UNSPECIFIED
    return _REJECTION_TO_WIRE[rejection]


def _rejection_from_wire(
    wire_reason: pb.ProfilingRejectionReason, *, accepted: bool, label: str
) -> ProfilingRejection | None:
    if accepted:
        # An accepted verdict carries no reason; ignore whatever the wire says.
        return None
    rejection = _WIRE_TO_REJECTION.get(wire_reason)
    if rejection is None:
        # Fail loudly (§47): a refusal without a usable reason leaves the
        # Master unable to choose a recovery path.
        raise ProfilingProtocolError(
            f"rejected {label} carries unknown rejection reason {wire_reason!r}"
        )
    return rejection


def _case_state_to_wire(state: CaseState | None) -> str:
    return "" if state is None else state.value


def _case_state_from_wire(
    value: str, *, required: bool, label: str
) -> CaseState | None:
    if not value:
        if required:
            raise ProfilingProtocolError(f"accepted {label} carries no case state")
        return None
    try:
        return CaseState(value)
    except ValueError as exc:
        raise ProfilingProtocolError(
            f"unknown case state {value!r} in {label}"
        ) from exc


# ---------------------------------------------------------------------------
# PrepareProfilingSession (§38)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrepareProfilingSessionRequest:
    """Open one profiling session on a Worker (spec §38, §41).

    ``network_facts`` are the Master-resolved cluster address facts a
    NETWORK session probes against — mandatory for network sessions (the
    executing Worker never guesses destinations, §52.2) and forbidden for
    every other kind (a stray batch would be silently ignored otherwise).
    """

    worker_id: str
    instance_id: str
    registration_session_id: str
    profiling_session_id: str
    session_request: ProfilingSessionRequest
    network_facts: tuple[WorkerNetworkFacts, ...] = ()

    def __post_init__(self) -> None:
        _check_tokens(
            self.worker_id,
            self.instance_id,
            self.registration_session_id,
            self.profiling_session_id,
        )
        if self.session_request.kind is ProfilingSessionKind.NETWORK:
            if not self.network_facts:
                raise ValueError(
                    "network sessions require Master-resolved network facts (§52.2)"
                )
        elif self.network_facts:
            raise ValueError(
                f"{self.session_request.kind.value} sessions must not carry "
                "network facts"
            )


@dataclass(frozen=True)
class PrepareProfilingSessionResponse:
    """Preparation verdict; ``session_facts`` only for accepted MODEL sessions.

    Preparation performs static work only — no benchmark ever runs inside
    prepare (§40); OPERATOR/NETWORK sessions answer with ``None`` facts.

    A rejection is *either* a transport-level verdict (``reason``: stale
    session, busy device, …) *or* a typed domain preparation failure
    (``failure``: unsupported model, export failure, …) — never both, never
    neither. The split matters: ``reason`` keys the Master's transport
    recovery (re-select / defer), while ``failure`` is a profiling outcome
    the Master records with its §42 category intact (no string parsing).
    """

    accepted: bool
    detail: str = ""
    reason: ProfilingRejection | None = None
    session_facts: ModelSessionFacts | None = None
    failure: ProfilingFailure | None = None

    def __post_init__(self) -> None:
        if not self.accepted:
            if not self.detail:
                raise ValueError("a rejected prepare must carry a detail")
            if (self.reason is None) == (self.failure is None):
                raise ValueError(
                    "a rejected prepare must carry exactly one of reason or failure"
                )
            if self.session_facts is not None:
                raise ValueError("a rejected prepare must not carry session facts")
        else:
            if self.reason is not None:
                raise ValueError("an accepted prepare must not carry a rejection reason")
            if self.failure is not None:
                raise ValueError("an accepted prepare must not carry a failure")


def prepare_request_to_wire(
    request: PrepareProfilingSessionRequest,
) -> pb.PrepareProfilingSessionRequest:
    return pb.PrepareProfilingSessionRequest(
        worker_id=request.worker_id,
        instance_id=request.instance_id,
        registration_session_id=request.registration_session_id,
        profiling_session_id=request.profiling_session_id,
        session_request_payload=encode_json(request.session_request),
        network_facts_payload=(
            encode_json(NetworkFactsBatch(facts=request.network_facts))
            if request.network_facts
            else ""
        ),
    )


def prepare_request_from_wire(
    wire: pb.PrepareProfilingSessionRequest,
) -> PrepareProfilingSessionRequest:
    session_request = _decode_payload(
        ProfilingSessionRequest, wire.session_request_payload, "session request"
    )
    batch = _decode_optional_payload(
        NetworkFactsBatch, wire.network_facts_payload, "network facts"
    )
    return PrepareProfilingSessionRequest(
        worker_id=wire.worker_id,
        instance_id=wire.instance_id,
        registration_session_id=wire.registration_session_id,
        profiling_session_id=wire.profiling_session_id,
        session_request=session_request,
        network_facts=batch.facts if batch is not None else (),
    )


def prepare_response_to_wire(
    response: PrepareProfilingSessionResponse,
) -> pb.PrepareProfilingSessionResponse:
    return pb.PrepareProfilingSessionResponse(
        accepted=response.accepted,
        detail=response.detail,
        reason=_rejection_to_wire(response.reason),
        session_facts_payload=(
            encode_json(response.session_facts)
            if response.session_facts is not None
            else ""
        ),
        failure_payload=(
            encode_json(response.failure) if response.failure is not None else ""
        ),
    )


def prepare_response_from_wire(
    wire: pb.PrepareProfilingSessionResponse,
) -> PrepareProfilingSessionResponse:
    failure = _decode_optional_payload(
        ProfilingFailure, wire.failure_payload, "prepare failure"
    )
    # A rejected prepare carries *either* a transport reason *or* a typed
    # failure; when the failure channel is used the reason legitimately
    # stays UNSPECIFIED. Every other rejection is decoded strictly, and a
    # wire message carrying both channels is caught by the DTO's
    # exclusivity rule (§47: fail loudly, never degrade silently).
    reason: ProfilingRejection | None = None
    failure_only = (
        failure is not None
        and wire.reason == pb.PROFILING_REJECTION_REASON_UNSPECIFIED
    )
    if not wire.accepted and not failure_only:
        reason = _rejection_from_wire(wire.reason, accepted=False, label="prepare")
    return PrepareProfilingSessionResponse(
        accepted=wire.accepted,
        detail=wire.detail,
        reason=reason,
        session_facts=_decode_optional_payload(
            ModelSessionFacts, wire.session_facts_payload, "session facts"
        ),
        failure=failure,
    )


# ---------------------------------------------------------------------------
# RunProfilingCase (§39, §42)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunProfilingCaseRequest:
    """Dispatch one benchmark case to its assigned executor (spec §8.2).

    The wire ``case_id`` field is redundant with the payload's canonical id
    (it exists so a Worker can index the case before decoding); the mapper
    cross-checks both copies, plus case/envelope worker agreement (§47).
    """

    worker_id: str
    instance_id: str
    registration_session_id: str
    profiling_session_id: str
    case: ProfilingCase

    def __post_init__(self) -> None:
        _check_tokens(
            self.worker_id,
            self.instance_id,
            self.registration_session_id,
            self.profiling_session_id,
        )
        if self.case.worker_id != self.worker_id:
            raise ValueError(
                f"case/envelope worker_id mismatch: case is assigned to "
                f"{self.case.worker_id!r} but the envelope targets {self.worker_id!r}"
            )


@dataclass(frozen=True)
class RunProfilingCaseResponse:
    """Run verdict (§42: failures are results, not transport rejections).

    ``accepted=False`` means the case was *refused* (session/lease problem)
    and never ran. A case that ran and failed is ``accepted=True`` with the
    typed :class:`~edgeshard.profiling.domain.errors.ProfilingFailure`
    inside its outcome.
    """

    accepted: bool
    detail: str = ""
    reason: ProfilingRejection | None = None
    outcome: CaseOutcome | None = None

    def __post_init__(self) -> None:
        if not self.accepted:
            if not self.detail:
                raise ValueError("a rejected run must carry a detail")
            if self.reason is None:
                raise ValueError("a rejected run must carry a reason")
            if self.outcome is not None:
                raise ValueError("a rejected run must not carry an outcome")
        else:
            if self.reason is not None:
                raise ValueError("an accepted run must not carry a rejection reason")
            if self.outcome is None:
                raise ValueError("an accepted run must carry an outcome")


def run_case_request_to_wire(request: RunProfilingCaseRequest) -> pb.RunProfilingCaseRequest:
    return pb.RunProfilingCaseRequest(
        worker_id=request.worker_id,
        instance_id=request.instance_id,
        registration_session_id=request.registration_session_id,
        profiling_session_id=request.profiling_session_id,
        case_id=request.case.case_id,
        case_payload=encode_json(request.case),
    )


def run_case_request_from_wire(wire: pb.RunProfilingCaseRequest) -> RunProfilingCaseRequest:
    case = _decode_payload(ProfilingCase, wire.case_payload, "case")
    if wire.case_id != case.case_id:
        raise ProfilingProtocolError(
            f"case_id mismatch: envelope carries {wire.case_id!r} but the "
            f"payload's canonical id is {case.case_id!r}"
        )
    return RunProfilingCaseRequest(
        worker_id=wire.worker_id,
        instance_id=wire.instance_id,
        registration_session_id=wire.registration_session_id,
        profiling_session_id=wire.profiling_session_id,
        case=case,
    )


def run_case_response_to_wire(response: RunProfilingCaseResponse) -> pb.RunProfilingCaseResponse:
    return pb.RunProfilingCaseResponse(
        accepted=response.accepted,
        detail=response.detail,
        reason=_rejection_to_wire(response.reason),
        outcome_payload=(
            encode_json(response.outcome) if response.outcome is not None else ""
        ),
    )


def run_case_response_from_wire(wire: pb.RunProfilingCaseResponse) -> RunProfilingCaseResponse:
    accepted = bool(wire.accepted)
    outcome = _decode_optional_payload(CaseOutcome, wire.outcome_payload, "outcome")
    if accepted and outcome is None:
        raise ProfilingProtocolError("accepted run response carries no outcome payload")
    return RunProfilingCaseResponse(
        accepted=accepted,
        detail=wire.detail,
        reason=_rejection_from_wire(wire.reason, accepted=accepted, label="run"),
        outcome=outcome,
    )


# ---------------------------------------------------------------------------
# GetProfilingCase (§44: poll the worker-side case ledger)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GetProfilingCaseRequest:
    worker_id: str
    instance_id: str
    registration_session_id: str
    profiling_session_id: str
    case_id: str

    def __post_init__(self) -> None:
        _check_tokens(
            self.worker_id,
            self.instance_id,
            self.registration_session_id,
            self.profiling_session_id,
        )
        if not self.case_id:
            raise ValueError("case_id must not be empty")


@dataclass(frozen=True)
class GetProfilingCaseResponse:
    """Case ledger view; terminal states always carry their outcome (§44).

    A case the session does not track is the typed ``UNKNOWN_CASE``
    refusal — never an accepted response with invented state (§52.2).
    """

    accepted: bool
    detail: str = ""
    reason: ProfilingRejection | None = None
    case_state: CaseState | None = None
    outcome: CaseOutcome | None = None

    def __post_init__(self) -> None:
        if not self.accepted:
            if not self.detail:
                raise ValueError("a rejected case query must carry a detail")
            if self.reason is None:
                raise ValueError("a rejected case query must carry a reason")
            if self.case_state is not None or self.outcome is not None:
                raise ValueError("a rejected case query must not carry case data")
            return
        if self.reason is not None:
            raise ValueError("an accepted case query must not carry a rejection reason")
        if self.case_state is None:
            raise ValueError("an accepted case query must carry a case state")
        if self.case_state in _TERMINAL_CASE_STATES:
            if self.outcome is None:
                raise ValueError(
                    f"a {self.case_state.value} case must carry its outcome (§44)"
                )
        elif self.outcome is not None:
            raise ValueError(
                f"a {self.case_state.value} case must not carry an outcome yet"
            )


def get_case_request_to_wire(request: GetProfilingCaseRequest) -> pb.GetProfilingCaseRequest:
    return pb.GetProfilingCaseRequest(
        worker_id=request.worker_id,
        instance_id=request.instance_id,
        registration_session_id=request.registration_session_id,
        profiling_session_id=request.profiling_session_id,
        case_id=request.case_id,
    )


def get_case_request_from_wire(wire: pb.GetProfilingCaseRequest) -> GetProfilingCaseRequest:
    return GetProfilingCaseRequest(
        worker_id=wire.worker_id,
        instance_id=wire.instance_id,
        registration_session_id=wire.registration_session_id,
        profiling_session_id=wire.profiling_session_id,
        case_id=wire.case_id,
    )


def get_case_response_to_wire(response: GetProfilingCaseResponse) -> pb.GetProfilingCaseResponse:
    return pb.GetProfilingCaseResponse(
        accepted=response.accepted,
        detail=response.detail,
        reason=_rejection_to_wire(response.reason),
        case_state=_case_state_to_wire(response.case_state),
        outcome_payload=(
            encode_json(response.outcome) if response.outcome is not None else ""
        ),
    )


def get_case_response_from_wire(wire: pb.GetProfilingCaseResponse) -> GetProfilingCaseResponse:
    accepted = bool(wire.accepted)
    return GetProfilingCaseResponse(
        accepted=accepted,
        detail=wire.detail,
        reason=_rejection_from_wire(wire.reason, accepted=accepted, label="case query"),
        case_state=_case_state_from_wire(
            wire.case_state, required=accepted, label="case query"
        ),
        outcome=_decode_optional_payload(CaseOutcome, wire.outcome_payload, "outcome"),
    )


# ---------------------------------------------------------------------------
# CancelProfilingCase (§44: cancellation never rewrites history)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CancelProfilingCaseRequest:
    worker_id: str
    instance_id: str
    registration_session_id: str
    profiling_session_id: str
    case_id: str

    def __post_init__(self) -> None:
        _check_tokens(
            self.worker_id,
            self.instance_id,
            self.registration_session_id,
            self.profiling_session_id,
        )
        if not self.case_id:
            raise ValueError("case_id must not be empty")


@dataclass(frozen=True)
class CancelProfilingCaseResponse:
    """Cancellation verdict with the case state *after* the attempt.

    ``accepted=True`` means the case is now cancelled — or was already
    terminal, which cancellation must not undo (§44); ``case_state`` tells
    the Master which of the two happened.
    """

    accepted: bool
    detail: str = ""
    reason: ProfilingRejection | None = None
    case_state: CaseState | None = None

    def __post_init__(self) -> None:
        if not self.accepted:
            if not self.detail:
                raise ValueError("a rejected cancellation must carry a detail")
            if self.reason is None:
                raise ValueError("a rejected cancellation must carry a reason")
            if self.case_state is not None:
                raise ValueError("a rejected cancellation must not carry a case state")
        else:
            if self.reason is not None:
                raise ValueError("an accepted cancellation must not carry a rejection reason")
            if self.case_state is None:
                raise ValueError("an accepted cancellation must carry the case state")


def cancel_case_request_to_wire(
    request: CancelProfilingCaseRequest,
) -> pb.CancelProfilingCaseRequest:
    return pb.CancelProfilingCaseRequest(
        worker_id=request.worker_id,
        instance_id=request.instance_id,
        registration_session_id=request.registration_session_id,
        profiling_session_id=request.profiling_session_id,
        case_id=request.case_id,
    )


def cancel_case_request_from_wire(
    wire: pb.CancelProfilingCaseRequest,
) -> CancelProfilingCaseRequest:
    return CancelProfilingCaseRequest(
        worker_id=wire.worker_id,
        instance_id=wire.instance_id,
        registration_session_id=wire.registration_session_id,
        profiling_session_id=wire.profiling_session_id,
        case_id=wire.case_id,
    )


def cancel_case_response_to_wire(
    response: CancelProfilingCaseResponse,
) -> pb.CancelProfilingCaseResponse:
    return pb.CancelProfilingCaseResponse(
        accepted=response.accepted,
        detail=response.detail,
        reason=_rejection_to_wire(response.reason),
        case_state=_case_state_to_wire(response.case_state),
    )


def cancel_case_response_from_wire(
    wire: pb.CancelProfilingCaseResponse,
) -> CancelProfilingCaseResponse:
    accepted = bool(wire.accepted)
    return CancelProfilingCaseResponse(
        accepted=accepted,
        detail=wire.detail,
        reason=_rejection_from_wire(wire.reason, accepted=accepted, label="cancellation"),
        case_state=_case_state_from_wire(
            wire.case_state, required=accepted, label="cancellation"
        ),
    )


# ---------------------------------------------------------------------------
# CloseProfilingSession (§38: idempotent cleanup)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CloseProfilingSessionRequest:
    worker_id: str
    instance_id: str
    registration_session_id: str
    profiling_session_id: str

    def __post_init__(self) -> None:
        _check_tokens(
            self.worker_id,
            self.instance_id,
            self.registration_session_id,
            self.profiling_session_id,
        )


@dataclass(frozen=True)
class CloseProfilingSessionResponse:
    """Close verdict; closing an unknown/closed session is accepted (§38).

    Cleanup must have released the session's leases and model state either
    way, so idempotence is the contract — a refusal here is reserved for
    invalid registration tokens (stale session).
    """

    accepted: bool
    detail: str = ""
    reason: ProfilingRejection | None = None

    def __post_init__(self) -> None:
        if not self.accepted:
            if not self.detail:
                raise ValueError("a rejected close must carry a detail")
            if self.reason is None:
                raise ValueError("a rejected close must carry a reason")
        elif self.reason is not None:
            raise ValueError("an accepted close must not carry a rejection reason")


def close_session_request_to_wire(
    request: CloseProfilingSessionRequest,
) -> pb.CloseProfilingSessionRequest:
    return pb.CloseProfilingSessionRequest(
        worker_id=request.worker_id,
        instance_id=request.instance_id,
        registration_session_id=request.registration_session_id,
        profiling_session_id=request.profiling_session_id,
    )


def close_session_request_from_wire(
    wire: pb.CloseProfilingSessionRequest,
) -> CloseProfilingSessionRequest:
    return CloseProfilingSessionRequest(
        worker_id=wire.worker_id,
        instance_id=wire.instance_id,
        registration_session_id=wire.registration_session_id,
        profiling_session_id=wire.profiling_session_id,
    )


def close_session_response_to_wire(
    response: CloseProfilingSessionResponse,
) -> pb.CloseProfilingSessionResponse:
    return pb.CloseProfilingSessionResponse(
        accepted=response.accepted,
        detail=response.detail,
        reason=_rejection_to_wire(response.reason),
    )


def close_session_response_from_wire(
    wire: pb.CloseProfilingSessionResponse,
) -> CloseProfilingSessionResponse:
    accepted = bool(wire.accepted)
    return CloseProfilingSessionResponse(
        accepted=accepted,
        detail=wire.detail,
        reason=_rejection_from_wire(wire.reason, accepted=accepted, label="close"),
    )


# ---------------------------------------------------------------------------
# ProfilingAdminService (§49): the CLI submits intents, reads back status.
# No session tokens here — the admin plane terminates at the Master, which
# owns registration state; refusals are plain accepted=False + detail.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StartExperimentRequest:
    """One profiling intent submitted by an operator (spec §49)."""

    request: ProfilingRequest


@dataclass(frozen=True)
class StartExperimentResponse:
    """Acceptance verdict; ``experiment_id`` only when accepted."""

    accepted: bool
    detail: str = ""
    experiment_id: str = ""

    def __post_init__(self) -> None:
        if self.accepted:
            if not self.experiment_id:
                raise ValueError("an accepted start must carry an experiment_id")
        else:
            if not self.detail:
                raise ValueError("a rejected start must carry a detail")
            if self.experiment_id:
                raise ValueError("a rejected start must not carry an experiment_id")


@dataclass(frozen=True)
class GetExperimentRequest:
    """Status read-back for one experiment id."""

    experiment_id: str

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id must not be empty")


@dataclass(frozen=True)
class GetExperimentResponse:
    """Lookup verdict; ``status`` only when the experiment is known (§52.2)."""

    found: bool
    status: ExperimentStatus | None = None

    def __post_init__(self) -> None:
        if self.found != (self.status is not None):
            raise ValueError("found must be true exactly when status is present")


@dataclass(frozen=True)
class CancelExperimentRequest:
    """Cancellation intent for one experiment id."""

    experiment_id: str

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id must not be empty")


@dataclass(frozen=True)
class CancelExperimentResponse:
    """Cancellation verdict; a rejected cancel always explains itself."""

    accepted: bool
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.accepted and not self.detail:
            raise ValueError("a rejected cancel must carry a detail")


@dataclass(frozen=True)
class BuildProfileSnapshotRequest:
    """Snapshot intent; v1 snapshots the whole store (§46), no scoping."""


@dataclass(frozen=True)
class BuildProfileSnapshotResponse:
    """Snapshot verdict; ``snapshot`` only when accepted."""

    accepted: bool
    detail: str = ""
    snapshot: ProfileSnapshot | None = None

    def __post_init__(self) -> None:
        if self.accepted != (self.snapshot is not None):
            raise ValueError("accepted must be true exactly when snapshot is present")
        if not self.accepted and not self.detail:
            raise ValueError("a rejected snapshot build must carry a detail")


def start_experiment_request_to_wire(
    request: StartExperimentRequest,
) -> pb.StartExperimentRequest:
    return pb.StartExperimentRequest(request_payload=encode_json(request.request))


def start_experiment_request_from_wire(
    wire: pb.StartExperimentRequest,
) -> StartExperimentRequest:
    return StartExperimentRequest(
        request=_decode_payload(
            ProfilingRequest, wire.request_payload, "profiling request"
        )
    )


def start_experiment_response_to_wire(
    response: StartExperimentResponse,
) -> pb.StartExperimentResponse:
    return pb.StartExperimentResponse(
        accepted=response.accepted,
        detail=response.detail,
        experiment_id=response.experiment_id,
    )


def start_experiment_response_from_wire(
    wire: pb.StartExperimentResponse,
) -> StartExperimentResponse:
    return StartExperimentResponse(
        accepted=bool(wire.accepted),
        detail=wire.detail,
        experiment_id=wire.experiment_id,
    )


def get_experiment_request_to_wire(
    request: GetExperimentRequest,
) -> pb.GetExperimentRequest:
    return pb.GetExperimentRequest(experiment_id=request.experiment_id)


def get_experiment_request_from_wire(
    wire: pb.GetExperimentRequest,
) -> GetExperimentRequest:
    return GetExperimentRequest(experiment_id=wire.experiment_id)


def get_experiment_response_to_wire(
    response: GetExperimentResponse,
) -> pb.GetExperimentResponse:
    return pb.GetExperimentResponse(
        found=response.found,
        status_payload=(
            encode_json(response.status) if response.status is not None else ""
        ),
    )


def get_experiment_response_from_wire(
    wire: pb.GetExperimentResponse,
) -> GetExperimentResponse:
    return GetExperimentResponse(
        found=bool(wire.found),
        status=_decode_optional_payload(
            ExperimentStatus, wire.status_payload, "experiment status"
        ),
    )


def cancel_experiment_request_to_wire(
    request: CancelExperimentRequest,
) -> pb.CancelExperimentRequest:
    return pb.CancelExperimentRequest(experiment_id=request.experiment_id)


def cancel_experiment_request_from_wire(
    wire: pb.CancelExperimentRequest,
) -> CancelExperimentRequest:
    return CancelExperimentRequest(experiment_id=wire.experiment_id)


def cancel_experiment_response_to_wire(
    response: CancelExperimentResponse,
) -> pb.CancelExperimentResponse:
    return pb.CancelExperimentResponse(
        accepted=response.accepted, detail=response.detail
    )


def cancel_experiment_response_from_wire(
    wire: pb.CancelExperimentResponse,
) -> CancelExperimentResponse:
    return CancelExperimentResponse(accepted=bool(wire.accepted), detail=wire.detail)


def build_snapshot_request_to_wire(
    request: BuildProfileSnapshotRequest,
) -> pb.BuildProfileSnapshotRequest:
    return pb.BuildProfileSnapshotRequest()


def build_snapshot_request_from_wire(
    wire: pb.BuildProfileSnapshotRequest,
) -> BuildProfileSnapshotRequest:
    return BuildProfileSnapshotRequest()


def build_snapshot_response_to_wire(
    response: BuildProfileSnapshotResponse,
) -> pb.BuildProfileSnapshotResponse:
    return pb.BuildProfileSnapshotResponse(
        accepted=response.accepted,
        detail=response.detail,
        snapshot_payload=(
            encode_json(response.snapshot) if response.snapshot is not None else ""
        ),
    )


def build_snapshot_response_from_wire(
    wire: pb.BuildProfileSnapshotResponse,
) -> BuildProfileSnapshotResponse:
    return BuildProfileSnapshotResponse(
        accepted=bool(wire.accepted),
        detail=wire.detail,
        snapshot=_decode_optional_payload(
            ProfileSnapshot, wire.snapshot_payload, "profile snapshot"
        ),
    )
