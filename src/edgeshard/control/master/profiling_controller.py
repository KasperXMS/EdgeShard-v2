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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import grpc

from edgeshard.control.master.service import MasterService
from edgeshard.profiling.domain.experiment import (
    CaseOutcome,
    CaseState,
    ExperimentState,
    ModelCaseSpec,
    NetworkCaseSpec,
    ProfilingCase,
    ProfilingErrorCategory,
    ProfilingExperiment,
    ProfilingFailure,
)
from edgeshard.profiling.domain.hashing import JsonScalar, canonical_sha256
from edgeshard.profiling.domain.session import (
    ProfilingSessionKind,
    ProfilingSessionRequest,
    profiling_session_id,
)
from edgeshard.profiling.domain.signature import ProfilingGranularity
from edgeshard.profiling.domain.snapshot import ProfileSnapshot
from edgeshard.profiling.network.classifier import (
    WorkerNetworkFacts,
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
    PrepareProfilingSessionRequest,
    PrepareProfilingSessionResponse,
    ProfilingRejection,
    RunProfilingCaseRequest,
    RunProfilingCaseResponse,
)

logger = logging.getLogger("edgeshard.control.master.profiling")

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

    Model cases split by (backend, model, dtype) — a session loads exactly
    one checkpoint at one declared dtype; operator cases split by backend
    only (each case carries its own signature and dtype, §25); network cases
    all share the source worker's single session.
    """
    spec = case.spec
    if isinstance(spec, NetworkCaseSpec):
        return (ProfilingSessionKind.NETWORK,)
    if spec.granularity is ProfilingGranularity.OPERATOR:
        return (ProfilingSessionKind.OPERATOR, spec.backend)
    return (ProfilingSessionKind.MODEL, spec.backend, spec.model, spec.dtype)


def _session_request(cases: Sequence[ProfilingCase]) -> ProfilingSessionRequest:
    """The session request covering every case of one group.

    Leases the *union* of the group's devices so each case's ``device_ids``
    is a subset of the leased set (§39: the Worker refuses unleased devices).
    """
    model_specs = [case.spec for case in cases if isinstance(case.spec, ModelCaseSpec)]
    if not model_specs:
        return ProfilingSessionRequest(kind=ProfilingSessionKind.NETWORK)
    devices = tuple(sorted({d for spec in model_specs for d in spec.device_ids}))
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
            ("grpc_code", code.value if code is not None else -1),
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

    # -- experiment definitions (§8.1, §44) ----------------------------------

    def create_experiment(
        self,
        *,
        strategy_id: str,
        cases: Sequence[ProfilingCase],
        requested_by: str | None = None,
    ) -> ProfilingExperiment:
        """Persist a new experiment definition (idempotent, §7/§44).

        Cases deduplicate on their canonical ids, and the experiment id is
        the canonical hash of the strategy plus the case set — re-planning
        the same logical job replays the stored definition instead of forking
        the append-oriented ledger. An already-persisted experiment (terminal
        or not) is returned exactly as stored: its ``created_at`` and history
        are never rewritten.
        """
        unique: dict[str, ProfilingCase] = {}
        for case in cases:
            unique.setdefault(case.case_id, case)
        experiment = ProfilingExperiment.for_cases(
            strategy_id=strategy_id,
            case_ids=tuple(unique),
            created_at=self._clock(),
            requested_by=requested_by,
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

        prepared = await self._prepare(transport, tokens, plan, facts)
        if isinstance(prepared, ProfilingFailure):
            reports.extend(
                self._record_failure(case, prepared, "session prepare failed")
                for case in cases
            )
            return tuple(reports)
        try:
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
        request = RunProfilingCaseRequest(
            worker_id=tokens.worker_id,
            instance_id=tokens.instance_id,
            registration_session_id=tokens.registration_session_id,
            profiling_session_id=session_id,
            case=case,
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
        self._store.update_case_state(case.case_id, state)
        return CaseReport(
            case_id=case.case_id,
            worker_id=case.worker_id,
            state=state,
            outcome=outcome,
        )

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
        self._store.update_case_state(case.case_id, CaseState.FAILED)
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
        self._store.update_case_state(case.case_id, CaseState.CANCELLED)
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
                    detail="replayed from the store (§44); nothing dispatched",
                )
            )
        return ExperimentReport(
            experiment_id=stored.experiment.experiment_id,
            state=stored.state,
            cases=tuple(reports),
        )
