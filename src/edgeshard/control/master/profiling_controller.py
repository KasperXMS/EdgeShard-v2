"""Master-side profiling orchestration (Phase 2 spec §40).

The ProfilingController is the Master's half of the profiling plane:

- create experiment definitions and persist them (§8.1, §43-44);
- expand them into per-Worker session plans (§38) and dispatch every case to
  its assigned executor (§8.2) over the Worker profiling plane (§41);
- track case/experiment lifecycle in the ProfileStore (§44: append-oriented,
  terminal states are history and never rewritten);
- persist measurement results idempotently (§50: a duplicate result is a
  replay, never a re-benchmark);
- cancel non-terminal cases on request (§40);
- build the ProfileSnapshot Phase 3 consumes (§46).

The Master NEVER executes a benchmark itself (§40): every observation in the
store arrived as a typed :class:`CaseOutcome` from the assigned Worker, and
every dispatch problem — unregistered Worker, missing profiling endpoint,
refusal, transport loss, timeout — is recorded as a typed
:class:`ProfilingFailure` with its §42 category intact, never guessed
(§52.2) and never string-parsed back into shape.

Execution model (v1): workers run concurrently, cases within one worker run
sequentially (the Worker executes one case per session synchronously, §37).
``run_experiment`` is resume-safe: cases already terminal in the store are
replayed into the report and never re-dispatched, so a Master restart
continues an interrupted experiment instead of re-benchmarking finished work.
ProfileStore access stays on the event-loop thread — SQLite writes are short
and the store is internally locked (§43).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

import grpc

from edgeshard.control.master.service import MasterService
from edgeshard.profiling.domain.environment import (
    DevicePerformanceClassMembership,
    EnvironmentFingerprint,
)
from edgeshard.profiling.domain.experiment import (
    CaseOutcome,
    CaseState,
    CaseStatus,
    ExperimentState,
    ExperimentStatus,
    ModelCaseSpec,
    NetworkCaseSpec,
    ProfilingCase,
    ProfilingErrorCategory,
    ProfilingExperiment,
    ProfilingFailure,
    profiling_case_id,
    profiling_experiment_id,
)
from edgeshard.profiling.domain.hashing import JsonScalar, canonical_sha256
from edgeshard.profiling.domain.network import NetworkEndpointProfile, ProbeKind
from edgeshard.profiling.domain.session import (
    ModelSessionFacts,
    ProfilingSessionKind,
    ProfilingSessionRequest,
    profiling_session_id,
)
from edgeshard.profiling.domain.signature import ProfilingGranularity
from edgeshard.profiling.domain.snapshot import ProfileSnapshot
from edgeshard.profiling.network.classifier import (
    ClassifiedPair,
    WorkerNetworkFacts,
    selected_ipv4_address,
    worker_network_facts,
)
from edgeshard.profiling.store.base import ProfileStore, ProfileStoreError, StoredExperiment
from edgeshard.protocol.profiling.grpc_client import WorkerProfilingClient
from edgeshard.protocol.profiling.mapper import (
    CancelProfilingCaseRequest,
    CancelProfilingCaseResponse,
    CloseProfilingSessionRequest,
    CloseProfilingSessionResponse,
    GetProfilingCaseRequest,
    GetProfilingCaseResponse,
    PrepareIperfServerRequest,
    PrepareIperfServerResponse,
    PrepareProfilingSessionRequest,
    PrepareProfilingSessionResponse,
    ProfilingRejection,
    RunProfilingCaseRequest,
    RunProfilingCaseResponse,
    StopIperfServerRequest,
    StopIperfServerResponse,
)

logger = logging.getLogger("edgeshard.control.master.profiling")

if TYPE_CHECKING:
    from edgeshard.profiling.operator.profiler import PerformanceClassVerification

_TERMINAL_CASE_STATES = frozenset(
    {CaseState.COMPLETED, CaseState.FAILED, CaseState.CANCELLED}
)
_TERMINAL_EXPERIMENT_STATES = frozenset(
    {
        ExperimentState.COMPLETED,
        ExperimentState.PARTIALLY_COMPLETED,
        ExperimentState.CANCELLED,
        ExperimentState.FAILED,
    }
)

_REJECTION_CATEGORY: dict[ProfilingRejection, ProfilingErrorCategory] = {
    # §39: the lease rejection — the case is deferrable/re-selectable and the
    # category must survive into the report unchanged.
    ProfilingRejection.DEVICE_BUSY: ProfilingErrorCategory.DEVICE_BUSY,
    # §41: the registration died mid-flight — the Worker's profiling plane is
    # unreachable under it and its results MUST NOT be published.
    ProfilingRejection.STALE_SESSION: ProfilingErrorCategory.NETWORK_UNREACHABLE,
    # Everything else right after a successful prepare (or on a case the
    # session never saw) is a Master-side protocol violation: fail loudly.
    ProfilingRejection.UNKNOWN_SESSION: ProfilingErrorCategory.INTERNAL_ERROR,
    ProfilingRejection.SESSION_CLOSED: ProfilingErrorCategory.INTERNAL_ERROR,
    ProfilingRejection.SESSION_KIND_MISMATCH: ProfilingErrorCategory.INTERNAL_ERROR,
    ProfilingRejection.UNKNOWN_CASE: ProfilingErrorCategory.INTERNAL_ERROR,
}

MODEL_INSPECTION_SCOPE = "model-inspection"
"""Scope prefix of §47-step-1 inspection sessions (not tied to an experiment).

Experiment-dispatched sessions are scoped by their ``experiment_id``; an
inspection happens *before* any experiment exists, so it gets its own scope
namespace (uniqueness per call is added by :meth:`_inspection_session_id`).
"""


class ProfilingTransport(Protocol):
    """The slice of :class:`WorkerProfilingClient` the controller speaks.

    Structural so tests can substitute an in-process fake; the real client
    (``protocol.profiling.grpc_client``) satisfies it unchanged.
    """

    async def prepare_profiling_session(
        self,
        request: PrepareProfilingSessionRequest,
        *,
        timeout: float | None = None,
    ) -> PrepareProfilingSessionResponse: ...

    async def run_profiling_case(
        self,
        request: RunProfilingCaseRequest,
        *,
        timeout: float | None = None,
    ) -> RunProfilingCaseResponse: ...

    async def get_profiling_case(
        self,
        request: GetProfilingCaseRequest,
        *,
        timeout: float | None = None,
    ) -> GetProfilingCaseResponse: ...

    async def cancel_profiling_case(
        self,
        request: CancelProfilingCaseRequest,
        *,
        timeout: float | None = None,
    ) -> CancelProfilingCaseResponse: ...

    async def close_profiling_session(
        self,
        request: CloseProfilingSessionRequest,
        *,
        timeout: float | None = None,
    ) -> CloseProfilingSessionResponse: ...

    async def prepare_iperf_server(
        self,
        request: PrepareIperfServerRequest,
        *,
        timeout: float | None = None,
    ) -> PrepareIperfServerResponse: ...

    async def stop_iperf_server(
        self,
        request: StopIperfServerRequest,
        *,
        timeout: float | None = None,
    ) -> StopIperfServerResponse: ...

    async def close(self) -> None: ...


@dataclass(frozen=True)
class CaseReport:
    """Master-side verdict of one case within an experiment run.

    ``outcome`` is present for every case that reached a terminal verdict
    this run (success record or typed failure); cases replayed from the store
    carry their state only — the store keeps states, not outcomes (§43).
    """

    case_id: str
    worker_id: str
    state: CaseState
    outcome: CaseOutcome | None = None
    detail: str = ""


@dataclass(frozen=True)
class ExperimentReport:
    """Terminal view of one experiment run: state plus every case verdict."""

    experiment_id: str
    state: ExperimentState
    cases: tuple[CaseReport, ...]


@dataclass(frozen=True)
class _DispatchTokens:
    """Everything one profiling RPC envelope needs (§41)."""

    worker_id: str
    instance_id: str
    registration_session_id: str
    endpoint: str


@dataclass(frozen=True)
class _IperfServerLease:
    transport: ProfilingTransport
    tokens: _DispatchTokens
    server_id: str
    port: int


@dataclass(frozen=True)
class _SessionPlan:
    """One Worker session and the cases that share it (§38)."""

    session_id: str
    session_request: ProfilingSessionRequest
    cases: tuple[ProfilingCase, ...]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _session_kind(case: ProfilingCase) -> ProfilingSessionKind:
    """Which session family a case belongs to (§38)."""
    if isinstance(case.spec, NetworkCaseSpec):
        return ProfilingSessionKind.NETWORK
    if case.spec.granularity is ProfilingGranularity.OPERATOR:
        return ProfilingSessionKind.OPERATOR
    return ProfilingSessionKind.MODEL


def _group_key(case: ProfilingCase) -> tuple[object, ...]:
    """Cases sharing one key share one session (§38: build state once).

    Model cases split by backend/model/dtype, target device, and selective
    layer target. Operator cases split by backend and target device. Network
    cases all share the source worker's session.
    """
    spec = case.spec
    if isinstance(spec, NetworkCaseSpec):
        return (ProfilingSessionKind.NETWORK,)
    if spec.granularity is ProfilingGranularity.OPERATOR:
        return (ProfilingSessionKind.OPERATOR, spec.backend, spec.device_ids[0])
    target_layer = (
        spec.layer_index
        if spec.granularity is ProfilingGranularity.TRANSFORMER_LAYER
        else 0
    )
    return (
        ProfilingSessionKind.MODEL,
        spec.backend,
        spec.model,
        spec.dtype,
        spec.device_ids[0],
        target_layer,
    )


def _session_request(cases: Sequence[ProfilingCase]) -> ProfilingSessionRequest:
    """The session request covering every case of one group.

    Device sessions lease exactly one target device (§39); MODEL sessions
    additionally carry the one layer shard that preparation must load.
    """
    model_specs = [case.spec for case in cases if isinstance(case.spec, ModelCaseSpec)]
    if not model_specs:
        return ProfilingSessionRequest(kind=ProfilingSessionKind.NETWORK)
    devices = tuple(sorted({d for spec in model_specs for d in spec.device_ids}))
    if len(devices) != 1:
        raise ValueError(
            "device-bound profiling sessions require exactly one target device"
        )
    first = model_specs[0]
    if first.granularity is ProfilingGranularity.OPERATOR:
        return ProfilingSessionRequest(
            kind=ProfilingSessionKind.OPERATOR,
            device_ids=devices,
            backend=first.backend,
        )
    if first.model is None or first.dtype is None:
        # Unreachable: ModelCaseSpec validation forces both for non-operator
        # granularities. Fail loudly rather than prepare a malformed session.
        raise ValueError(
            "model session cases must carry a model reference and dtype (§38)"
        )
    return ProfilingSessionRequest(
        kind=ProfilingSessionKind.MODEL,
        device_ids=devices,
        backend=first.backend,
        model=first.model,
        dtype=first.dtype,
        target_layer_index=(
            first.layer_index
            if first.granularity is ProfilingGranularity.TRANSFORMER_LAYER
            else 0
        ),
    )


def _plan_sessions(
    experiment_id: str, worker_id: str, cases: Sequence[ProfilingCase]
) -> tuple[_SessionPlan, ...]:
    """Group one worker's cases into sessions with canonical ids (§7).

    A pure function of (experiment, worker, case content): a Master restart
    re-derives identical session ids, so the Worker replays prepares
    idempotently (§50) and cancellations can find the session a case was
    dispatched under.
    """
    groups: dict[tuple[object, ...], list[ProfilingCase]] = {}
    for case in cases:
        groups.setdefault(_group_key(case), []).append(case)
    plans = []
    for group in groups.values():
        request = _session_request(group)
        plans.append(
            _SessionPlan(
                session_id=profiling_session_id(experiment_id, worker_id, request),
                session_request=request,
                cases=tuple(group),
            )
        )
    return tuple(plans)


def _rejection_failure(
    reason: ProfilingRejection | None, detail: str, what: str
) -> ProfilingFailure:
    """Typed failure for a Worker refusal — the category keys recovery (§41)."""
    items: list[tuple[str, JsonScalar]] = [("detail", detail)]
    category = ProfilingErrorCategory.INTERNAL_ERROR
    if reason is not None:
        items.append(("rejection", reason.value))
        category = _REJECTION_CATEGORY[reason]
        message = f"{what} refused by the worker: {reason.value}"
    else:
        message = f"{what} refused by the worker"
    return ProfilingFailure(
        category=category, message=message, details=tuple(items)
    )


def _transport_failure(exc: grpc.aio.AioRpcError, what: str) -> ProfilingFailure:
    """Typed failure for a lost profiling RPC (timeout vs unreachable)."""
    code = exc.code()
    category = (
        ProfilingErrorCategory.TIMEOUT
        if code is grpc.StatusCode.DEADLINE_EXCEEDED
        else ProfilingErrorCategory.NETWORK_UNREACHABLE
    )
    return ProfilingFailure(
        category=category,
        message=f"{what} transport failed: {code.name if code is not None else 'unknown'}",
        details=(
            ("grpc_code", code.name if code is not None else "unknown"),
            ("transport_detail", exc.details()),
        ),
    )


def _unexpected_failure(exc: Exception, what: str) -> ProfilingFailure:
    return ProfilingFailure(
        category=ProfilingErrorCategory.INTERNAL_ERROR,
        message=f"{what} raised {exc!r}",
    )


def _terminal_experiment_state(states: Sequence[CaseState]) -> ExperimentState:
    """The experiment verdict over its final case states (§8.1).

    Uniform verdicts map onto their exact state; any mix of terminal outcomes
    is PARTIALLY_COMPLETED — the honest label for "some measurements landed,
    some did not" (P2G DoD: partial experiment completion).
    """
    if not states:
        return ExperimentState.COMPLETED
    if all(state is CaseState.COMPLETED for state in states):
        return ExperimentState.COMPLETED
    if all(state is CaseState.CANCELLED for state in states):
        return ExperimentState.CANCELLED
    if all(state is CaseState.FAILED for state in states):
        return ExperimentState.FAILED
    if all(state in _TERMINAL_CASE_STATES for state in states):
        return ExperimentState.PARTIALLY_COMPLETED
    # Unreachable through run_experiment — every dispatched case lands in a
    # terminal report. Fail loudly rather than invent a verdict (§47).
    raise ValueError(f"non-terminal case state in a finished experiment: {states!r}")


class ProfilingController:
    """Orchestrates profiling experiments from the Master (spec §40)."""

    def __init__(
        self,
        *,
        service: MasterService,
        store: ProfileStore,
        transport_factory: Callable[[str], ProfilingTransport] = WorkerProfilingClient,
        clock: Callable[[], datetime] = _utc_now,
        rpc_timeout: float | None = None,
    ) -> None:
        if rpc_timeout is not None and rpc_timeout <= 0:
            raise ValueError(
                f"rpc_timeout must be positive when present, got {rpc_timeout}"
            )
        self._service = service
        self._store = store
        self._transport_factory = transport_factory
        self._clock = clock
        self._rpc_timeout = rpc_timeout
        self._inspection_seq = 0

    # -- experiment definitions (§8.1, §44) ----------------------------------

    def create_experiment(
        self,
        *,
        strategy_id: str,
        cases: Sequence[ProfilingCase],
        requested_by: str | None = None,
        force_new_execution: bool = False,
    ) -> ProfilingExperiment:
        """Persist one configuration or an explicit new execution (§7/§44).

        Normal creation is idempotent by stable configuration identity and
        supports restart/resume. ``force_new_execution`` retains that identity
        but allocates fresh experiment/case execution ids so a requested rerun
        appends measurements without rewriting terminal history.
        """
        configuration_case_ids = tuple(
            sorted(
                {
                    profiling_case_id(case.worker_id, case.spec)
                    for case in cases
                }
            )
        )
        configuration_id = profiling_experiment_id(
            strategy_id, configuration_case_ids
        )
        execution_cases = tuple(cases)
        experiment_id = configuration_id
        if force_new_execution:
            run_nonce = uuid4().hex
            execution_cases = tuple(
                dataclasses.replace(
                    case,
                    case_id=canonical_sha256(
                        (
                            "profiling_case_execution",
                            profiling_case_id(case.worker_id, case.spec),
                            run_nonce,
                        )
                    ),
                )
                for case in cases
            )
            experiment_id = canonical_sha256(
                ("profiling_execution", configuration_id, run_nonce)
            )

        unique: dict[str, ProfilingCase] = {}
        for case in execution_cases:
            unique.setdefault(case.case_id, case)
        experiment = ProfilingExperiment(
            experiment_id=experiment_id,
            configuration_id=configuration_id,
            strategy_id=strategy_id,
            created_at=self._clock(),
            requested_by=requested_by,
            case_ids=tuple(unique),
        )
        existing = self._store.get_experiment(experiment.experiment_id)
        if existing is not None:
            logger.info(
                "experiment %s already persisted (state=%s); replaying the "
                "stored definition (§44)",
                experiment.experiment_id,
                existing.state.value,
            )
            return existing.experiment
        for case in unique.values():
            self._store.append_case(case)
        self._store.append_experiment(experiment)
        logger.info(
            "created experiment %s (strategy=%s, cases=%d)",
            experiment.experiment_id,
            strategy_id,
            len(experiment.case_ids),
        )
        return experiment

    def load_experiment(self, experiment_id: str) -> StoredExperiment | None:
        """The stored definition + lifecycle state of one experiment (§43)."""
        return self._store.get_experiment(experiment_id)

    def experiment_status(self, experiment_id: str) -> ExperimentStatus | None:
        """Read-back view of one experiment for the admin plane (§49).

        ``None`` for an unknown id (never a guess, §52.2). Typed failures are
        read from the same durable ledger as case state, so a Master restart
        does not erase the reason a case failed or was cancelled.
        """
        stored = self._store.get_experiment(experiment_id)
        if stored is None:
            return None
        cases: list[CaseStatus] = []
        for case_id in stored.experiment.case_ids:
            stored_case = self._store.get_case(case_id)
            if stored_case is None:
                raise ValueError(
                    f"experiment {experiment_id!r} references case {case_id!r} "
                    "which is not persisted (§8.3)"
                )
            cases.append(
                CaseStatus(
                    case_id=case_id,
                    worker_id=stored_case.case.worker_id,
                    state=stored_case.state,
                    failure=stored_case.failure,
                )
            )
        return ExperimentStatus(
            experiment=stored.experiment, state=stored.state, cases=tuple(cases)
        )

    # -- planning support for the admin layer (§46-§47) ------------------------

    def profiling_worker_ids(self) -> tuple[str, ...]:
        """Every registered Worker that hosts the profiling plane (§41).

        Sorted and registration-order independent; Workers without a
        profiling endpoint are invisible to profiling intents rather than
        dispatched-to-fail.
        """
        return tuple(
            sorted(
                record.identity.worker_id
                for record in self._service.registry.list_workers()
                if record.profiling_endpoint is not None
            )
        )

    def profiling_device_ids(self, worker_id: str) -> tuple[str, ...]:
        """Device ids owned by one registered profiling Worker."""
        record = self._service.registry.find(worker_id)
        if record is None or record.profiling_endpoint is None:
            return ()
        return tuple(
            device.identity.device_id for device in record.capability.devices
        )

    def cluster_network_facts(self) -> dict[str, WorkerNetworkFacts]:
        """Public view of the Master-resolved network facts (§30, §52.2)."""
        return self._cluster_network_facts()

    def measured_operator_signature_ids_for_environment(
        self, fingerprint: EnvironmentFingerprint
    ) -> AbstractSet[str]:
        """Reuse licensed for this physical and compatible environment."""
        return self._store.measured_operator_signature_ids_for_environment(
            fingerprint
        )

    def record_performance_class_verification(
        self,
        fingerprint: EnvironmentFingerprint,
        verification: PerformanceClassVerification,
        *,
        evidence_measurement_ids: tuple[str, ...] = (),
    ) -> None:
        """Persist a verification verdict that directly gates reuse planning."""
        class_id = fingerprint.device_performance_class_id
        if (
            class_id is None
            or fingerprint.worker_id is None
            or fingerprint.device_id is None
        ):
            raise ValueError(
                "performance-class verification requires class, worker, and device ids"
            )
        self._store.store_environment_fingerprint(fingerprint)
        self._store.store_performance_class_membership(
            DevicePerformanceClassMembership(
                device_performance_class_id=class_id,
                worker_id=fingerprint.worker_id,
                device_id=fingerprint.device_id,
                verified=verification.compatible,
                verified_at=self._clock() if verification.compatible else None,
                evidence_measurement_ids=evidence_measurement_ids,
            )
        )

    def record_model_facts(self, facts: ModelSessionFacts) -> None:
        """Persist one inspection's facts into the §43 registries.

        The characterization and every layer/module/operator signature the
        Worker reported become stored facts — so the ProfileSnapshot carries
        the static model picture (§46) even before (or without) any
        measurement, and reuse queries can see the signatures. Idempotent:
        the registries are content-addressed (§9).
        """
        self._store.store_characterization(facts.characterization)
        for layer_entry in facts.layer_entries:
            self._store.store_layer_signature(layer_entry.signature)
        for module_entry in facts.module_entries:
            self._store.store_module_signature(module_entry.signature)
        for signature in facts.operator_signatures:
            self._store.store_operator_signature(signature)
        if facts.environment is not None:
            self._store.store_environment_fingerprint(facts.environment)
        logger.info(
            "recorded model facts: %s (%d layer, %d module, %d operator "
            "signature(s))",
            facts.characterization.model.model_id,
            len(facts.layer_entries),
            len(facts.module_entries),
            len(facts.operator_signatures),
        )

    def record_network_characterization(
        self,
        endpoint_profiles: Iterable[NetworkEndpointProfile],
        classified_pairs: Iterable[ClassifiedPair],
    ) -> None:
        """Persist the §47 network steps 1-2 facts (endpoints + classes)."""
        for profile in endpoint_profiles:
            self._store.store_network_endpoint(profile)
        for classified in classified_pairs:
            self._store.store_path_classification(
                classified.pair, classified.path_class
            )

    # -- model inspection (§47 step 1, over the §41 plane) ---------------------

    async def inspect_model(
        self, *, worker_id: str, request: ProfilingSessionRequest
    ) -> ModelSessionFacts | ProfilingFailure:
        """Prepare a MODEL session on one Worker purely to read its facts.

        §46/§47: planning consumes facts, and the static characterization
        happens Worker-side (§38) — the Master never loads a model. The
        session is closed on every exit path: an inspection is a read, not a
        lease the caller keeps. Every problem (unreachable Worker, refusal,
        transport loss, facts missing from an accepted prepare) comes back as
        a typed :class:`ProfilingFailure`, never an exception (§42, §52.2).
        """
        if request.kind is not ProfilingSessionKind.MODEL:
            raise ValueError(
                f"inspect_model requires a MODEL session request, got "
                f"{request.kind.value!r}"
            )
        resolved = self._resolve_tokens(worker_id)
        if isinstance(resolved, ProfilingFailure):
            return resolved
        session_id = self._inspection_session_id(resolved, request)
        transport = self._transport_factory(resolved.endpoint)
        try:
            prepared = await self._prepare(
                transport,
                resolved,
                _SessionPlan(
                    session_id=session_id, session_request=request, cases=()
                ),
                (),
            )
            if isinstance(prepared, ProfilingFailure):
                return prepared
            if prepared.session_facts is None:
                # The response DTO forbids this for accepted MODEL sessions;
                # guard anyway (§47).
                return ProfilingFailure(
                    category=ProfilingErrorCategory.INTERNAL_ERROR,
                    message=(
                        f"accepted model inspection on worker {worker_id!r} "
                        "carries no session facts (§47)"
                    ),
                )
            return prepared.session_facts
        finally:
            await self._close_session(transport, resolved, session_id)
            await transport.close()

    def _inspection_session_id(
        self, tokens: _DispatchTokens, request: ProfilingSessionRequest
    ) -> str:
        """A fresh canonical inspection session id (§7, §38).

        Scoped to the Worker's current registration epoch plus a per-process
        sequence, so repeated inspections of the same model never re-present
        a *closed* session id — the Worker refuses those by design (§38:
        closed ids never come back, prepare a new one). An inspection is a
        live fact read, not an append-oriented ledger entry, so uniqueness
        beats cross-restart replay here; experiment sessions keep their
        deterministic experiment-scoped ids (§50).
        """
        self._inspection_seq += 1
        scope = (
            f"{MODEL_INSPECTION_SCOPE}:"
            f"{tokens.registration_session_id}:{self._inspection_seq}"
        )
        return profiling_session_id(scope, tokens.worker_id, request)

    # -- execution (§40) ------------------------------------------------------

    async def run_experiment(self, experiment_id: str) -> ExperimentReport:
        """Dispatch every non-terminal case of one experiment.

        Resume-safe (P2G DoD: Master restart): cases already terminal in the
        store are replayed into the report and never re-dispatched (§50); the
        rest are grouped into sessions per worker (§38) and executed — workers
        concurrently, cases within one worker sequentially (§37). The Worker
        side is idempotent too: a re-prepared session replays, and a case the
        Worker's ledger already decided replays its recorded outcome.
        """
        stored = self._require_experiment(experiment_id)
        if stored.state in _TERMINAL_EXPERIMENT_STATES:
            logger.info(
                "experiment %s is terminal (%s); replaying stored states (§44)",
                experiment_id,
                stored.state.value,
            )
            return self._report_from_store(stored)

        reports: dict[str, CaseReport] = {}
        pending: dict[str, list[ProfilingCase]] = {}
        network_needed = False
        for case_id in stored.experiment.case_ids:
            stored_case = self._store.get_case(case_id)
            if stored_case is None:
                raise ValueError(
                    f"experiment {experiment_id!r} references case {case_id!r} "
                    "which is not persisted (§8.3)"
                )
            if stored_case.state in _TERMINAL_CASE_STATES:
                reports[case_id] = CaseReport(
                    case_id=case_id,
                    worker_id=stored_case.case.worker_id,
                    state=stored_case.state,
                    outcome=(
                        CaseOutcome.from_failure(stored_case.failure)
                        if stored_case.failure is not None
                        else None
                    ),
                    detail="terminal in the store; replayed without re-dispatch (§50)",
                )
                continue
            case = stored_case.case
            pending.setdefault(case.worker_id, []).append(case)
            network_needed = (
                network_needed
                or _session_kind(case) is ProfilingSessionKind.NETWORK
            )

        self._store.update_experiment_state(experiment_id, ExperimentState.RUNNING)

        if pending:
            facts = self._cluster_network_facts() if network_needed else {}
            results = await asyncio.gather(
                *(
                    self._run_worker(stored.experiment, worker_id, cases, facts)
                    for worker_id, cases in sorted(pending.items())
                )
            )
            for chunk in results:
                for report in chunk:
                    reports[report.case_id] = report

        case_ids = stored.experiment.case_ids
        state = _terminal_experiment_state(
            [reports[case_id].state for case_id in case_ids]
        )
        self._store.update_experiment_state(experiment_id, state)
        logger.info(
            "experiment %s finished: %s (%d case(s))",
            experiment_id,
            state.value,
            len(case_ids),
        )
        return ExperimentReport(
            experiment_id=experiment_id,
            state=state,
            cases=tuple(reports[case_id] for case_id in case_ids),
        )

    # -- cancellation (§40, §44) ----------------------------------------------

    async def cancel_experiment(self, experiment_id: str) -> ExperimentReport:
        """Cancel every non-terminal case of one experiment.

        Cancellation dispatches ``CancelProfilingCase`` so the Worker's ledger
        and leases converge, closes the sessions, then records CANCELLED
        Master-side. When the Worker cannot be reached the cancellation is
        still recorded — it is the Master's bookkeeping of its own intent,
        and §41 forbids publishing any result of a stale session anyway.
        Terminal cases keep their state and their persisted measurements:
        history is never rewritten (§44) — and when a cancellation loses the
        race against a case that already terminalized on the Worker, the
        Worker's outcome is fetched and persisted truthfully instead of being
        overwritten with CANCELLED.
        """
        stored = self._require_experiment(experiment_id)
        if stored.state in _TERMINAL_EXPERIMENT_STATES:
            return self._report_from_store(stored)

        reports: dict[str, CaseReport] = {}
        active: dict[str, list[ProfilingCase]] = {}
        all_cases: list[ProfilingCase] = []
        for case_id in stored.experiment.case_ids:
            stored_case = self._store.get_case(case_id)
            if stored_case is None:
                raise ValueError(
                    f"experiment {experiment_id!r} references case {case_id!r} "
                    "which is not persisted (§8.3)"
                )
            all_cases.append(stored_case.case)
            if stored_case.state in _TERMINAL_CASE_STATES:
                reports[case_id] = CaseReport(
                    case_id=case_id,
                    worker_id=stored_case.case.worker_id,
                    state=stored_case.state,
                    outcome=(
                        CaseOutcome.from_failure(stored_case.failure)
                        if stored_case.failure is not None
                        else None
                    ),
                    detail="terminal in the store; untouched by cancellation (§44)",
                )
            else:
                active.setdefault(stored_case.case.worker_id, []).append(
                    stored_case.case
                )

        cancelled_any = False
        if active:
            results = await asyncio.gather(
                *(
                    self._cancel_worker(experiment_id, worker_id, cases, all_cases)
                    for worker_id, cases in sorted(active.items())
                )
            )
            for chunk in results:
                for report in chunk:
                    reports[report.case_id] = report
                    cancelled_any = (
                        cancelled_any or report.state is CaseState.CANCELLED
                    )

        case_ids = stored.experiment.case_ids
        state = (
            ExperimentState.CANCELLED
            if cancelled_any
            else _terminal_experiment_state(
                [reports[case_id].state for case_id in case_ids]
            )
        )
        self._store.update_experiment_state(experiment_id, state)
        logger.info("experiment %s cancelled: %s", experiment_id, state.value)
        return ExperimentReport(
            experiment_id=experiment_id,
            state=state,
            cases=tuple(reports[case_id] for case_id in case_ids),
        )

    # -- snapshot (§46) --------------------------------------------------------

    def build_profile_snapshot(self) -> ProfileSnapshot:
        """The §46 view of everything Phase 2 has measured so far.

        The snapshot id is the canonical hash (§7) of the observation time
        plus the persisted content ids — a snapshot *is* a point-in-time
        observation, so unlike a request identity its timestamp is part of
        what it is; the payload hashes stay out (the content ids are
        themselves canonical).
        """
        created_at = self._clock()
        provisional = self._store.build_snapshot("provisional", created_at=created_at)
        snapshot_id = canonical_sha256(
            (
                "profile_snapshot",
                created_at.isoformat(),
                tuple(
                    sorted(
                        (c.model.model_id, c.model.revision or "")
                        for c in provisional.model_characterizations
                    )
                ),
                tuple(sorted(r.measurement_id for r in provisional.measurements)),
                tuple(
                    sorted(
                        r.measurement_id for r in provisional.network_measurements
                    )
                ),
            )
        )
        return dataclasses.replace(provisional, snapshot_id=snapshot_id)

    # -- per-worker execution --------------------------------------------------

    async def _run_worker(
        self,
        experiment: ProfilingExperiment,
        worker_id: str,
        cases: Sequence[ProfilingCase],
        network_facts: Mapping[str, WorkerNetworkFacts],
    ) -> tuple[CaseReport, ...]:
        resolved = self._resolve_tokens(worker_id)
        if isinstance(resolved, ProfilingFailure):
            logger.warning(
                "cannot dispatch to worker %s: %s", worker_id, resolved.message
            )
            return tuple(
                self._record_failure(case, resolved, "not dispatched")
                for case in cases
            )
        transport = self._transport_factory(resolved.endpoint)
        try:
            reports: list[CaseReport] = []
            for plan in _plan_sessions(experiment.experiment_id, worker_id, cases):
                reports.extend(
                    await self._run_session(transport, resolved, plan, network_facts)
                )
            return tuple(reports)
        finally:
            await transport.close()

    async def _run_session(
        self,
        transport: ProfilingTransport,
        tokens: _DispatchTokens,
        plan: _SessionPlan,
        network_facts: Mapping[str, WorkerNetworkFacts],
    ) -> tuple[CaseReport, ...]:
        cases = plan.cases
        facts: tuple[WorkerNetworkFacts, ...] = ()
        reports: list[CaseReport] = []
        if plan.session_request.kind is ProfilingSessionKind.NETWORK:
            cases, facts, fact_failures = self._resolve_network_facts(
                cases, network_facts
            )
            reports.extend(fact_failures)
            if not cases:
                return tuple(reports)

        try:
            prepared = await self._prepare(transport, tokens, plan, facts)
            if isinstance(prepared, ProfilingFailure):
                reports.extend(
                    self._record_failure(case, prepared, "session prepare failed")
                    for case in cases
                )
                return tuple(reports)
            for case in cases:
                reports.append(
                    await self._run_case(transport, tokens, plan.session_id, case)
                )
        finally:
            # §38: cleanup on every exit path — a failed run must not strand
            # the session's leases or model state on the Worker.
            await self._close_session(transport, tokens, plan.session_id)
        return tuple(reports)

    def _resolve_network_facts(
        self,
        cases: Sequence[ProfilingCase],
        network_facts: Mapping[str, WorkerNetworkFacts],
    ) -> tuple[tuple[ProfilingCase, ...], tuple[WorkerNetworkFacts, ...], tuple[CaseReport, ...]]:
        """Split network cases into dispatchable and fact-less (§52.2).

        A case whose source or destination is absent from the cluster
        snapshot fails typed NETWORK_UNREACHABLE *without* being dispatched:
        the executing Worker never guesses destinations, and its prepare
        requires the Master-resolved facts to be complete (§41).
        """
        dispatchable: list[ProfilingCase] = []
        failures: list[CaseReport] = []
        needed: set[str] = set()
        for case in cases:
            spec = case.spec
            if not isinstance(spec, NetworkCaseSpec):
                raise ValueError(
                    f"network session case {case.case_id!r} carries a "
                    "non-network spec (§47)"
                )
            missing = sorted(
                worker_id
                for worker_id in (spec.source_worker_id, spec.destination_worker_id)
                if worker_id not in network_facts
            )
            if missing:
                failure = ProfilingFailure(
                    category=ProfilingErrorCategory.NETWORK_UNREACHABLE,
                    message=(
                        "no cluster facts for network peer(s): "
                        + ", ".join(repr(worker_id) for worker_id in missing)
                    ),
                    details=(("missing_worker_ids", ", ".join(missing)),),
                )
                failures.append(
                    self._record_failure(case, failure, "not dispatched (§52.2)")
                )
                continue
            needed.update((spec.source_worker_id, spec.destination_worker_id))
            dispatchable.append(case)
        facts = tuple(network_facts[worker_id] for worker_id in sorted(needed))
        return tuple(dispatchable), facts, tuple(failures)

    async def _prepare(
        self,
        transport: ProfilingTransport,
        tokens: _DispatchTokens,
        plan: _SessionPlan,
        network_facts: tuple[WorkerNetworkFacts, ...],
    ) -> PrepareProfilingSessionResponse | ProfilingFailure:
        request = PrepareProfilingSessionRequest(
            worker_id=tokens.worker_id,
            instance_id=tokens.instance_id,
            registration_session_id=tokens.registration_session_id,
            profiling_session_id=plan.session_id,
            session_request=plan.session_request,
            network_facts=network_facts,
        )
        try:
            response = await transport.prepare_profiling_session(
                request, timeout=self._rpc_timeout
            )
        except grpc.aio.AioRpcError as exc:
            logger.warning(
                "prepare of session %s on worker %s lost: %s",
                plan.session_id,
                tokens.worker_id,
                exc,
            )
            return _transport_failure(exc, "PrepareProfilingSession")
        except Exception as exc:
            logger.exception(
                "prepare of session %s on worker %s raised",
                plan.session_id,
                tokens.worker_id,
            )
            return _unexpected_failure(exc, "PrepareProfilingSession")
        if response.accepted:
            return response
        if response.failure is not None:
            # Typed domain preparation failure (§42): the category (e.g.
            # UNSUPPORTED_MODEL) travels into the case reports intact.
            logger.info(
                "session %s prepare failed on worker %s: [%s] %s",
                plan.session_id,
                tokens.worker_id,
                response.failure.category.value,
                response.failure.message,
            )
            return response.failure
        logger.warning(
            "session %s prepare refused on worker %s: %s",
            plan.session_id,
            tokens.worker_id,
            response.detail,
        )
        return _rejection_failure(
            response.reason, response.detail, "PrepareProfilingSession"
        )

    async def _run_case(
        self,
        transport: ProfilingTransport,
        tokens: _DispatchTokens,
        session_id: str,
        case: ProfilingCase,
    ) -> CaseReport:
        self._store.update_case_state(case.case_id, CaseState.RUNNING)
        iperf_lease: _IperfServerLease | None = None
        if (
            isinstance(case.spec, NetworkCaseSpec)
            and case.spec.probe_kind is ProbeKind.BANDWIDTH
        ):
            prepared = await self._prepare_remote_iperf_server(case)
            if isinstance(prepared, ProfilingFailure):
                return self._record_failure(
                    case, prepared, "destination iperf3 server preparation failed"
                )
            iperf_lease = prepared
        request = RunProfilingCaseRequest(
            worker_id=tokens.worker_id,
            instance_id=tokens.instance_id,
            registration_session_id=tokens.registration_session_id,
            profiling_session_id=session_id,
            case=case,
            iperf_server_port=(
                iperf_lease.port if iperf_lease is not None else None
            ),
        )
        try:
            response = await transport.run_profiling_case(
                request, timeout=self._rpc_timeout
            )
        except grpc.aio.AioRpcError as exc:
            logger.warning(
                "run of case %s on worker %s lost: %s",
                case.case_id,
                tokens.worker_id,
                exc,
            )
            return self._record_failure(
                case, _transport_failure(exc, "RunProfilingCase"), "run lost"
            )
        except Exception as exc:
            logger.exception(
                "run of case %s on worker %s raised", case.case_id, tokens.worker_id
            )
            return self._record_failure(
                case, _unexpected_failure(exc, "RunProfilingCase"), "run raised"
            )
        finally:
            if iperf_lease is not None:
                await self._stop_remote_iperf_server(iperf_lease)
        if not response.accepted:
            failure = _rejection_failure(
                response.reason, response.detail, "RunProfilingCase"
            )
            logger.warning(
                "run of case %s refused on worker %s: %s",
                case.case_id,
                tokens.worker_id,
                response.detail,
            )
            return self._record_failure(case, failure, "run refused by the worker")
        if response.outcome is None:
            # The response DTO forbids this; guard anyway (§47).
            return self._record_failure(
                case,
                ProfilingFailure(
                    category=ProfilingErrorCategory.INTERNAL_ERROR,
                    message="accepted run response carries no outcome",
                ),
                "protocol violation",
            )
        return self._persist_outcome(case, response.outcome)

    def _persist_outcome(self, case: ProfilingCase, outcome: CaseOutcome) -> CaseReport:
        """Persist one terminal outcome (§43-44) and report it.

        ``append_measurement`` returning ``False`` is the §50 duplicate
        replay — the identical result was already stored, so the case simply
        completes. A store error degrades to a typed INTERNAL_ERROR failure:
        the Worker's ledger still holds the outcome, so a resume replays it
        instead of re-benchmarking.
        """
        record = outcome.record
        if record is not None:
            if record.case_id != case.case_id:
                return self._record_failure(
                    case,
                    ProfilingFailure(
                        category=ProfilingErrorCategory.INTERNAL_ERROR,
                        message=(
                            f"result of case {case.case_id!r} arrived under "
                            f"record case_id {record.case_id!r} (§47)"
                        ),
                    ),
                    "protocol violation; measurement not persisted",
                )
            environment = record.environment
            expected_device = (
                case.spec.device_ids[0]
                if isinstance(case.spec, ModelCaseSpec)
                else None
            )
            if (
                environment is None
                or environment.worker_id != case.worker_id
                or environment.device_id != expected_device
            ):
                return self._record_failure(
                    case,
                    ProfilingFailure(
                        category=ProfilingErrorCategory.INTERNAL_ERROR,
                        message=(
                            "measurement lacks a complete, correctly attributed "
                            "EnvironmentFingerprint"
                        ),
                        details=(
                            ("expected_device_id", expected_device),
                            ("expected_worker_id", case.worker_id),
                        ),
                    ),
                    "protocol violation; measurement not persisted",
                )
            try:
                if not self._store.append_measurement(record):
                    logger.info(
                        "measurement %s of case %s was already stored; "
                        "duplicate replay (§50)",
                        record.measurement_id,
                        case.case_id,
                    )
            except ProfileStoreError as exc:
                return self._record_failure(
                    case,
                    ProfilingFailure(
                        category=ProfilingErrorCategory.INTERNAL_ERROR,
                        message=f"persisting the measurement failed: {exc}",
                    ),
                    "store failure; the case replays on resume (§50)",
                )
            state = CaseState.COMPLETED
        else:
            state = CaseState.FAILED
        self._store.update_case_state(
            case.case_id,
            state,
            outcome.failure if outcome.record is None else None,
        )
        return CaseReport(
            case_id=case.case_id,
            worker_id=case.worker_id,
            state=state,
            outcome=outcome,
        )

    async def _prepare_remote_iperf_server(
        self, case: ProfilingCase
    ) -> _IperfServerLease | ProfilingFailure:
        spec = case.spec
        assert isinstance(spec, NetworkCaseSpec)
        resolved = self._resolve_tokens(spec.destination_worker_id)
        if isinstance(resolved, ProfilingFailure):
            return resolved
        transport = self._transport_factory(resolved.endpoint)
        destination_facts = self._cluster_network_facts().get(
            spec.destination_worker_id
        )
        bind_address = (
            selected_ipv4_address(
                destination_facts, spec.destination_interface_id
            )
            if destination_facts is not None
            else None
        )
        if bind_address is None:
            await transport.close()
            return ProfilingFailure(
                category=ProfilingErrorCategory.NETWORK_UNREACHABLE,
                message=(
                    "bandwidth destination has no unambiguous IPv4 probe path; "
                    "set destination_interface_id explicitly on multi-NIC workers"
                ),
            )
        server_id = canonical_sha256(("iperf_server", case.case_id))
        timeout_s = (spec.duration_s or 0.0) + 30.0
        request = PrepareIperfServerRequest(
            worker_id=resolved.worker_id,
            instance_id=resolved.instance_id,
            registration_session_id=resolved.registration_session_id,
            server_id=server_id,
            timeout_s=timeout_s,
            bind_address=bind_address,
        )
        try:
            response = await transport.prepare_iperf_server(
                request, timeout=self._rpc_timeout
            )
        except grpc.aio.AioRpcError as exc:
            await transport.close()
            return _transport_failure(exc, "PrepareIperfServer")
        except Exception as exc:
            await transport.close()
            return _unexpected_failure(exc, "PrepareIperfServer")
        if not response.accepted or response.port is None:
            await transport.close()
            return _rejection_failure(
                response.reason, response.detail, "PrepareIperfServer"
            )
        return _IperfServerLease(
            transport=transport,
            tokens=resolved,
            server_id=server_id,
            port=response.port,
        )

    async def _stop_remote_iperf_server(self, lease: _IperfServerLease) -> None:
        request = StopIperfServerRequest(
            worker_id=lease.tokens.worker_id,
            instance_id=lease.tokens.instance_id,
            registration_session_id=lease.tokens.registration_session_id,
            server_id=lease.server_id,
        )
        try:
            response = await lease.transport.stop_iperf_server(
                request, timeout=self._rpc_timeout
            )
            if not response.accepted:
                logger.warning(
                    "destination iperf3 server %s refused cleanup: %s",
                    lease.server_id,
                    response.detail,
                )
        except Exception:
            logger.warning(
                "destination iperf3 server %s cleanup failed",
                lease.server_id,
                exc_info=True,
            )
        finally:
            await lease.transport.close()

    async def _close_session(
        self, transport: ProfilingTransport, tokens: _DispatchTokens, session_id: str
    ) -> None:
        """Best-effort idempotent close (§38): never fails a decided case.

        A refused or lost close is logged only — the Worker releases every
        session lease on shutdown regardless (§39), and case outcomes are
        already recorded history the Master does not rewrite for a cleanup
        hiccup.
        """
        request = CloseProfilingSessionRequest(
            worker_id=tokens.worker_id,
            instance_id=tokens.instance_id,
            registration_session_id=tokens.registration_session_id,
            profiling_session_id=session_id,
        )
        try:
            response = await transport.close_profiling_session(
                request, timeout=self._rpc_timeout
            )
            if not response.accepted:
                logger.warning(
                    "close of session %s on worker %s refused: %s",
                    session_id,
                    tokens.worker_id,
                    response.detail,
                )
        except Exception:
            logger.warning(
                "close of session %s on worker %s failed",
                session_id,
                tokens.worker_id,
                exc_info=True,
            )

    # -- per-worker cancellation ------------------------------------------------

    async def _cancel_worker(
        self,
        experiment_id: str,
        worker_id: str,
        active: Sequence[ProfilingCase],
        all_cases: Sequence[ProfilingCase],
    ) -> tuple[CaseReport, ...]:
        resolved = self._resolve_tokens(worker_id)
        if isinstance(resolved, ProfilingFailure):
            logger.warning(
                "cannot cancel on worker %s: %s", worker_id, resolved.message
            )
            return tuple(
                self._record_cancelled(
                    case,
                    f"worker unreachable ({resolved.message}); recorded Master-side",
                )
                for case in active
            )
        # The plan is a pure function of the experiment's cases, so the
        # session ids match the ones the cases were dispatched under.
        plans = {
            case.case_id: plan
            for plan in _plan_sessions(experiment_id, worker_id, all_cases)
            for case in plan.cases
        }
        transport = self._transport_factory(resolved.endpoint)
        try:
            reports = [
                await self._cancel_case(
                    transport, resolved, plans[case.case_id].session_id, case
                )
                for case in active
            ]
            for session_id in {plans[case.case_id].session_id for case in active}:
                await self._close_session(transport, resolved, session_id)
            return tuple(reports)
        finally:
            await transport.close()

    async def _cancel_case(
        self,
        transport: ProfilingTransport,
        tokens: _DispatchTokens,
        session_id: str,
        case: ProfilingCase,
    ) -> CaseReport:
        request = CancelProfilingCaseRequest(
            worker_id=tokens.worker_id,
            instance_id=tokens.instance_id,
            registration_session_id=tokens.registration_session_id,
            profiling_session_id=session_id,
            case_id=case.case_id,
        )
        try:
            response = await transport.cancel_profiling_case(
                request, timeout=self._rpc_timeout
            )
        except Exception as exc:
            logger.warning(
                "cancel of case %s on worker %s failed: %r",
                case.case_id,
                tokens.worker_id,
                exc,
            )
            return self._record_cancelled(
                case, f"worker unreachable during cancellation ({exc!r})"
            )
        if not response.accepted:
            reason = response.reason.value if response.reason is not None else "?"
            return self._record_cancelled(
                case,
                f"cancellation refused ({reason}: {response.detail}); "
                "recorded Master-side",
            )
        if response.case_state is CaseState.CANCELLED:
            return self._record_cancelled(case, "cancelled on the worker")
        # §44: the Worker's ledger terminalized the case another way before
        # the cancellation landed — that outcome is history. Fetch it and
        # persist it truthfully instead of overwriting it with CANCELLED.
        state = (
            response.case_state.value if response.case_state is not None else "?"
        )
        outcome = await self._fetch_outcome(
            transport, tokens, session_id, case.case_id
        )
        if outcome is None:
            return self._record_cancelled(
                case,
                f"worker reported {state} but its outcome could not be "
                "fetched; recorded Master-side",
            )
        report = self._persist_outcome(case, outcome)
        return dataclasses.replace(
            report,
            detail=(
                "cancellation lost the race: the case was already "
                f"{report.state.value} on the worker (§44)"
            ),
        )

    async def _fetch_outcome(
        self,
        transport: ProfilingTransport,
        tokens: _DispatchTokens,
        session_id: str,
        case_id: str,
    ) -> CaseOutcome | None:
        try:
            response = await transport.get_profiling_case(
                GetProfilingCaseRequest(
                    worker_id=tokens.worker_id,
                    instance_id=tokens.instance_id,
                    registration_session_id=tokens.registration_session_id,
                    profiling_session_id=session_id,
                    case_id=case_id,
                ),
                timeout=self._rpc_timeout,
            )
        except Exception:
            logger.warning(
                "fetching the outcome of case %s on worker %s failed",
                case_id,
                tokens.worker_id,
                exc_info=True,
            )
            return None
        if not response.accepted or response.outcome is None:
            return None
        return response.outcome

    # -- shared helpers ---------------------------------------------------------

    def _resolve_tokens(self, worker_id: str) -> _DispatchTokens | ProfilingFailure:
        """The live dispatch context of one Worker, or a typed refusal.

        Registration facts are never guessed (§52.2): a Worker that is
        unknown, does not host the profiling plane (§41), or has no current
        registration session cannot receive cases, and the reason is recorded
        per case as a typed NETWORK_UNREACHABLE failure.
        """
        record = self._service.registry.find(worker_id)
        if record is None:
            return ProfilingFailure(
                category=ProfilingErrorCategory.NETWORK_UNREACHABLE,
                message=f"worker {worker_id!r} is not registered",
            )
        endpoint = record.profiling_endpoint
        if endpoint is None:
            return ProfilingFailure(
                category=ProfilingErrorCategory.NETWORK_UNREACHABLE,
                message=(
                    f"worker {worker_id!r} does not host the profiling "
                    "service (§41)"
                ),
            )
        info = self._service.sessions.current(worker_id)
        if info is None:
            return ProfilingFailure(
                category=ProfilingErrorCategory.NETWORK_UNREACHABLE,
                message=f"worker {worker_id!r} has no active registration session",
            )
        return _DispatchTokens(
            worker_id=worker_id,
            instance_id=info.instance_id,
            registration_session_id=info.session_id,
            endpoint=endpoint,
        )

    def _cluster_network_facts(self) -> dict[str, WorkerNetworkFacts]:
        """Master-resolved address facts per registered Worker (§30, §52.2).

        Derived from the immutable cluster snapshot — the executing Worker
        never scans for destinations itself.
        """
        return {
            worker.identity.worker_id: worker_network_facts(worker)
            for worker in self._service.build_snapshot().workers
        }

    def _require_experiment(self, experiment_id: str) -> StoredExperiment:
        stored = self._store.get_experiment(experiment_id)
        if stored is None:
            raise ValueError(
                f"unknown experiment {experiment_id!r} (§44: create it first)"
            )
        return stored

    def _record_failure(
        self, case: ProfilingCase, failure: ProfilingFailure, detail: str
    ) -> CaseReport:
        """One case FAILED with a typed failure, persisted (§42/§44)."""
        self._store.update_case_state(case.case_id, CaseState.FAILED, failure)
        return CaseReport(
            case_id=case.case_id,
            worker_id=case.worker_id,
            state=CaseState.FAILED,
            outcome=CaseOutcome.from_failure(failure),
            detail=detail,
        )

    def _record_cancelled(self, case: ProfilingCase, detail: str) -> CaseReport:
        failure = ProfilingFailure(
            category=ProfilingErrorCategory.CANCELLED,
            message="case cancelled by the Master (§40)",
        )
        self._store.update_case_state(case.case_id, CaseState.CANCELLED, failure)
        return CaseReport(
            case_id=case.case_id,
            worker_id=case.worker_id,
            state=CaseState.CANCELLED,
            outcome=CaseOutcome.from_failure(failure),
            detail=detail,
        )

    def _report_from_store(self, stored: StoredExperiment) -> ExperimentReport:
        """Report a terminal experiment purely from stored states (§44)."""
        reports = []
        for case_id in stored.experiment.case_ids:
            stored_case = self._store.get_case(case_id)
            if stored_case is None:
                raise ValueError(
                    f"experiment {stored.experiment.experiment_id!r} references "
                    f"case {case_id!r} which is not persisted (§8.3)"
                )
            reports.append(
                CaseReport(
                    case_id=case_id,
                    worker_id=stored_case.case.worker_id,
                    state=stored_case.state,
                    outcome=(
                        CaseOutcome.from_failure(stored_case.failure)
                        if stored_case.failure is not None
                        else None
                    ),
                    detail="replayed from the store (§44); nothing dispatched",
                )
            )
        return ExperimentReport(
            experiment_id=stored.experiment.experiment_id,
            state=stored.state,
            cases=tuple(reports),
        )
