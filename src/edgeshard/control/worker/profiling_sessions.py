"""Worker-side profiling sessions and the case ledger (§38, §41, §44).

A session is the Worker-local lifecycle unit the Master's
``PrepareProfilingSession`` opens: it groups compatible cases so expensive
state is built once (a MODEL session loads its checkpoint once, §38), it
owns the §39 device leases for its lifetime, and it keeps the per-case
ledger the ``Get``/``Cancel`` RPCs answer from.

Two spec disciplines shape this module:

* **§41 stale-session gate** — every RPC validates the Phase 1 registration
  tokens (``worker_id`` / ``instance_id`` / ``registration_session_id``)
  against the Worker's *current* registration before any session state is
  touched, so a superseded or restarted Agent can never execute or publish
  profiling work under a dead session. The token source is injectable: the
  serve wiring binds it to the live :class:`WorkerAgent`, tests bind it to
  a fixture.
* **§44 append-oriented ledger** — case states only move forward
  (PENDING/RUNNING → one terminal state) and a terminal entry carries its
  outcome forever; a duplicate ``Run`` of an already-decided case replays
  the recorded outcome instead of re-benchmarking (§50 duplicate results),
  and ``Cancel`` never undoes history.

The module is deliberately torch-free: a prepared MODEL session's loaded
checkpoint is stored as an opaque handle plus a cleanup callable, which the
runner (:mod:`edgeshard.control.worker.profiling_runner`) supplies.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from edgeshard.profiling.domain.experiment import (
    CaseOutcome,
    CaseState,
    ProfilingErrorCategory,
    ProfilingFailure,
)
from edgeshard.profiling.domain.session import (
    ModelSessionFacts,
    ProfilingSessionKind,
    ProfilingSessionRequest,
)
from edgeshard.profiling.network.classifier import WorkerNetworkFacts
from edgeshard.protocol.profiling.mapper import ProfilingRejection

logger = logging.getLogger("worker.profiling.sessions")


@dataclass(frozen=True)
class RegistrationTokens:
    """The Phase 1 registration a profiling RPC must ride (§41)."""

    worker_id: str
    instance_id: str
    registration_session_id: str


RegistrationTokenSource = Callable[[], RegistrationTokens | None]
"""Supplies the Worker's *current* registration tokens; ``None`` while the
Agent is between registrations. Bound to the live WorkerAgent in serve."""


class SessionRefused(Exception):
    """A typed refusal the runner maps onto an ``accepted=False`` response.

    Not an :class:`EdgeShardError` crash path: refusals are the protocol's
    normal vocabulary for stale sessions, unknown sessions and kind
    mismatches (§41), and the servicer never sees them as transport errors.
    """

    def __init__(self, reason: ProfilingRejection, detail: str) -> None:
        super().__init__(f"{reason.value}: {detail}")
        self.reason = reason
        self.detail = detail


_TERMINAL_STATES = frozenset(
    {CaseState.COMPLETED, CaseState.FAILED, CaseState.CANCELLED}
)


@dataclass
class CaseLedgerEntry:
    """One case's forward-only state in a session ledger (§44)."""

    state: CaseState = CaseState.PENDING
    outcome: CaseOutcome | None = None
    decided_at: datetime | None = None

    @property
    def terminal(self) -> bool:
        return self.state in _TERMINAL_STATES


@dataclass
class ProfilingSessionRecord:
    """Live Worker-side state of one prepared session (§38).

    ``model_handle``/``model_cleanup`` are opaque to this module: the runner
    stores its loaded checkpoint session and the callable that releases it.
    ``network_facts`` is the Master-resolved destination table a NETWORK
    session probes against (§52.2 — never completed locally).
    """

    session_id: str
    request: ProfilingSessionRequest
    prepared_at: datetime
    capability_revision: str | None = None
    session_facts: ModelSessionFacts | None = None
    network_facts: Mapping[str, WorkerNetworkFacts] = field(default_factory=dict)
    model_handle: object | None = None
    model_cleanup: Callable[[], None] | None = None
    cases: dict[str, CaseLedgerEntry] = field(default_factory=dict)
    closed: bool = False

    @property
    def kind(self) -> ProfilingSessionKind:
        return self.request.kind

    @property
    def device_ids(self) -> tuple[str, ...]:
        return self.request.device_ids


class ProfilingSessionManager:
    """Session/ledger store with the §41 registration gate.

    Every entry point first validates the RPC's tokens against the current
    registration (refusing ``STALE_SESSION`` when they do not match or no
    registration is active), then performs the session operation. Refusals
    raise :class:`SessionRefused`; protocol violations that must abort the
    RPC (mutating a terminal ledger entry, closing over a broken cleanup)
    raise loudly instead (§47).
    """

    def __init__(self, *, token_source: RegistrationTokenSource) -> None:
        self._token_source = token_source
        self._sessions: dict[str, ProfilingSessionRecord] = {}

    @property
    def session_ids(self) -> tuple[str, ...]:
        return tuple(self._sessions)

    def open_session_count(self) -> int:
        return sum(1 for record in self._sessions.values() if not record.closed)

    # ------------------------------------------------------------------
    # §41 registration gate
    # ------------------------------------------------------------------

    def validate_tokens(
        self,
        *,
        worker_id: str,
        instance_id: str,
        registration_session_id: str,
    ) -> RegistrationTokens:
        """The RPC's tokens vs the Worker's current registration (§41).

        A mismatch means the caller's registration was superseded (or this
        Worker restarted): its results MUST NOT be published, so every RPC —
        including read-only ones — is refused before touching session state.
        """
        current = self._token_source()
        if current is None:
            raise SessionRefused(
                ProfilingRejection.STALE_SESSION,
                "worker has no active registration; profiling requests are refused",
            )
        if worker_id != current.worker_id:
            raise SessionRefused(
                ProfilingRejection.STALE_SESSION,
                f"request worker_id {worker_id!r} does not match this worker "
                f"({current.worker_id!r})",
            )
        if instance_id != current.instance_id:
            raise SessionRefused(
                ProfilingRejection.STALE_SESSION,
                f"request instance_id {instance_id!r} predates the current agent "
                f"instance {current.instance_id!r}",
            )
        if registration_session_id != current.registration_session_id:
            raise SessionRefused(
                ProfilingRejection.STALE_SESSION,
                f"registration session {registration_session_id!r} was superseded "
                f"by {current.registration_session_id!r}",
            )
        return current

    # ------------------------------------------------------------------
    # Session lifecycle (§38)
    # ------------------------------------------------------------------

    def peek(self, session_id: str) -> ProfilingSessionRecord | None:
        """The record for ``session_id`` regardless of state, or ``None``.

        Token-free by design: only :meth:`prepare_replay` uses it, after the
        caller has already passed :meth:`validate_tokens`.
        """
        return self._sessions.get(session_id)

    def create_session(
        self,
        session_id: str,
        request: ProfilingSessionRequest,
        *,
        now: datetime,
        capability_revision: str | None = None,
        session_facts: ModelSessionFacts | None = None,
        network_facts: Mapping[str, WorkerNetworkFacts] | None = None,
        model_handle: object | None = None,
        model_cleanup: Callable[[], None] | None = None,
    ) -> ProfilingSessionRecord:
        """Register a prepared session; a duplicate id is a caller bug (§47)."""
        if session_id in self._sessions:
            raise ValueError(f"profiling session {session_id!r} already exists")
        record = ProfilingSessionRecord(
            session_id=session_id,
            request=request,
            prepared_at=now,
            capability_revision=capability_revision,
            session_facts=session_facts,
            network_facts=dict(network_facts or {}),
            model_handle=model_handle,
            model_cleanup=model_cleanup,
        )
        self._sessions[session_id] = record
        logger.info(
            "prepared profiling session %s kind=%s devices=%d",
            session_id,
            record.kind.value,
            len(record.device_ids),
        )
        return record

    def require_open_session(self, session_id: str) -> ProfilingSessionRecord:
        """The open session or the typed refusal the RPC answers with (§41)."""
        record = self._sessions.get(session_id)
        if record is None:
            raise SessionRefused(
                ProfilingRejection.UNKNOWN_SESSION,
                f"profiling session {session_id!r} was never prepared on this worker",
            )
        if record.closed:
            raise SessionRefused(
                ProfilingRejection.SESSION_CLOSED,
                f"profiling session {session_id!r} is closed",
            )
        return record

    def close_session(self, session_id: str) -> bool:
        """Close one session and release its model; idempotent (§38).

        Returns ``True`` when this call closed an open session, ``False``
        when there was nothing to close (unknown or already closed) — the
        RPC answer is ``accepted=True`` either way. The model cleanup runs
        exactly once, on the transition.
        """
        record = self._sessions.get(session_id)
        if record is None or record.closed:
            return False
        record.closed = True
        cleanup, record.model_cleanup = record.model_cleanup, None
        handle_was_present = record.model_handle is not None
        record.model_handle = None
        if cleanup is not None:
            cleanup()
        logger.info(
            "closed profiling session %s (model released: %s)",
            session_id,
            handle_was_present,
        )
        return True

    def close_all(self) -> tuple[str, ...]:
        """Close every open session (runner shutdown); returns closed ids."""
        closed = tuple(
            session_id
            for session_id, record in self._sessions.items()
            if not record.closed
        )
        for session_id in closed:
            self.close_session(session_id)
        return closed

    # ------------------------------------------------------------------
    # Case ledger (§44)
    # ------------------------------------------------------------------

    def case_entry(
        self, session_id: str, case_id: str
    ) -> CaseLedgerEntry | None:
        """The ledger entry for one case, or ``None`` when never dispatched."""
        record = self._sessions.get(session_id)
        if record is None:
            return None
        return record.cases.get(case_id)

    def mark_running(self, session_id: str, case_id: str) -> None:
        """Record the PENDING→RUNNING start of one synchronous execution."""
        record = self.require_open_session(session_id)
        entry = record.cases.get(case_id)
        if entry is not None and entry.terminal:
            raise ValueError(
                f"case {case_id!r} is already terminal ({entry.state.value}); "
                "terminal ledger entries never reopen (§44)"
            )
        record.cases[case_id] = CaseLedgerEntry(state=CaseState.RUNNING)

    def decide_case(
        self,
        session_id: str,
        case_id: str,
        state: CaseState,
        outcome: CaseOutcome,
        *,
        now: datetime,
    ) -> CaseLedgerEntry:
        """Move one case to its terminal state, outcome attached (§44)."""
        if state not in _TERMINAL_STATES:
            raise ValueError(f"decide_case requires a terminal state, got {state!r}")
        record = self.require_open_session(session_id)
        entry = record.cases.get(case_id)
        if entry is not None and entry.terminal:
            raise ValueError(
                f"case {case_id!r} already decided ({entry.state.value}); "
                "history is never rewritten (§44)"
            )
        entry = CaseLedgerEntry(state=state, outcome=outcome, decided_at=now)
        record.cases[case_id] = entry
        return entry

    def cancel_case(
        self, session_id: str, case_id: str, *, now: datetime
    ) -> CaseLedgerEntry:
        """Cancel one case; already-terminal history is returned unchanged.

        A case cancelled before (or between) runs is decided as ``CANCELLED``
        with a typed failure outcome — §44 terminal states carry outcomes,
        and a later duplicate ``Run`` replays this decision instead of
        benchmarking (§50).
        """
        record = self.require_open_session(session_id)
        entry = record.cases.get(case_id)
        if entry is not None and entry.terminal:
            return entry  # cancellation never undoes history (§44)
        cancelled = CaseLedgerEntry(
            state=CaseState.CANCELLED,
            outcome=CaseOutcome.from_failure(
                ProfilingFailure(
                    category=ProfilingErrorCategory.CANCELLED,
                    message="case cancelled before execution",
                    details=(("case_id", case_id), ("session_id", session_id)),
                )
            ),
            decided_at=now,
        )
        record.cases[case_id] = cancelled
        logger.info("case %s cancelled in session %s", case_id, session_id)
        return cancelled


def utc_now() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "CaseLedgerEntry",
    "ProfilingSessionManager",
    "ProfilingSessionRecord",
    "RegistrationTokenSource",
    "RegistrationTokens",
    "SessionRefused",
    "utc_now",
]
