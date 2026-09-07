"""Worker session-manager and case-ledger tests (Phase 2 spec §38, §41, §44).

The manager owns two disciplines: every RPC entry point passes the §41
registration-token gate before touching session state (a superseded or
absent registration refuses STALE_SESSION — such results must never be
published), and the per-case ledger only moves forward (§44): terminal
entries carry their outcome forever, never reopen, and are never rewritten.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from edgeshard.control.worker.profiling_sessions import (
    CaseLedgerEntry,
    ProfilingSessionManager,
    RegistrationTokens,
    SessionRefused,
)
from edgeshard.profiling.domain.experiment import (
    CaseOutcome,
    CaseState,
    ProfilingErrorCategory,
    ProfilingFailure,
)
from edgeshard.profiling.domain.session import (
    ProfilingSessionKind,
    ProfilingSessionRequest,
)
from edgeshard.protocol.profiling.mapper import ProfilingRejection

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)
LATER = datetime(2026, 9, 7, 12, 5, tzinfo=UTC)

CURRENT = RegistrationTokens(
    worker_id="w-1", instance_id="instance-1", registration_session_id="reg-1"
)

OPERATOR_REQUEST = ProfilingSessionRequest(
    kind=ProfilingSessionKind.OPERATOR, device_ids=("dev-0",)
)

OUTCOME = CaseOutcome.from_failure(
    ProfilingFailure(category=ProfilingErrorCategory.BENCHMARK_FAILED, message="boom")
)


class TokenHolder:
    """The serve wiring binds tokens to the live Agent; tests bind to this."""

    def __init__(self, tokens: RegistrationTokens | None = CURRENT) -> None:
        self.tokens = tokens

    def __call__(self) -> RegistrationTokens | None:
        return self.tokens


@pytest.fixture
def holder() -> TokenHolder:
    return TokenHolder()


@pytest.fixture
def manager(holder: TokenHolder) -> ProfilingSessionManager:
    return ProfilingSessionManager(token_source=holder)


def validate(manager: ProfilingSessionManager, **overrides: str) -> RegistrationTokens:
    fields: dict[str, str] = {
        "worker_id": CURRENT.worker_id,
        "instance_id": CURRENT.instance_id,
        "registration_session_id": CURRENT.registration_session_id,
        **overrides,
    }
    return manager.validate_tokens(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# §41 registration-token gate
# ---------------------------------------------------------------------------


def test_matching_tokens_pass(manager: ProfilingSessionManager) -> None:
    assert validate(manager) == CURRENT


def test_no_active_registration_refuses_stale(
    manager: ProfilingSessionManager, holder: TokenHolder
) -> None:
    holder.tokens = None
    with pytest.raises(SessionRefused) as exc:
        validate(manager)
    assert exc.value.reason is ProfilingRejection.STALE_SESSION
    assert "no active registration" in exc.value.detail


@pytest.mark.parametrize(
    ("field", "value", "detail_match"),
    [
        ("worker_id", "w-other", "does not match this worker"),
        ("instance_id", "instance-old", "predates the current agent"),
        ("registration_session_id", "reg-0", "was superseded"),
    ],
)
def test_mismatched_tokens_refuse_stale(
    manager: ProfilingSessionManager, field: str, value: str, detail_match: str
) -> None:
    with pytest.raises(SessionRefused) as exc:
        validate(manager, **{field: value})
    assert exc.value.reason is ProfilingRejection.STALE_SESSION
    assert detail_match in exc.value.detail


def test_session_refused_message_carries_reason() -> None:
    error = SessionRefused(ProfilingRejection.UNKNOWN_SESSION, "never prepared")
    assert "UNKNOWN_SESSION" in str(error) or "unknown_session" in str(error).lower()
    assert "never prepared" in str(error)


# ---------------------------------------------------------------------------
# Session lifecycle (§38)
# ---------------------------------------------------------------------------


def test_create_and_peek(manager: ProfilingSessionManager) -> None:
    record = manager.create_session(
        "s-1", OPERATOR_REQUEST, now=NOW, capability_revision="cap-rev-1"
    )
    assert record.kind is ProfilingSessionKind.OPERATOR
    assert record.device_ids == ("dev-0",)
    assert record.prepared_at == NOW
    assert record.capability_revision == "cap-rev-1"
    assert record.closed is False
    assert manager.peek("s-1") is record
    assert manager.peek("s-missing") is None
    assert manager.session_ids == ("s-1",)
    assert manager.open_session_count() == 1


def test_duplicate_session_id_fails_loudly(manager: ProfilingSessionManager) -> None:
    manager.create_session("s-1", OPERATOR_REQUEST, now=NOW)
    with pytest.raises(ValueError, match="already exists"):
        manager.create_session("s-1", OPERATOR_REQUEST, now=LATER)


def test_require_open_session_refusals(manager: ProfilingSessionManager) -> None:
    with pytest.raises(SessionRefused) as unknown:
        manager.require_open_session("s-missing")
    assert unknown.value.reason is ProfilingRejection.UNKNOWN_SESSION

    manager.create_session("s-1", OPERATOR_REQUEST, now=NOW)
    manager.close_session("s-1")
    with pytest.raises(SessionRefused) as closed:
        manager.require_open_session("s-1")
    assert closed.value.reason is ProfilingRejection.SESSION_CLOSED


def test_close_runs_cleanup_exactly_once(manager: ProfilingSessionManager) -> None:
    calls: list[str] = []
    manager.create_session(
        "s-1",
        OPERATOR_REQUEST,
        now=NOW,
        model_handle=object(),
        model_cleanup=lambda: calls.append("cleanup"),
    )
    assert manager.close_session("s-1") is True
    assert calls == ["cleanup"]
    record = manager.peek("s-1")
    assert record is not None and record.closed
    # The model handle AND its cleanup reference are dropped (§37: the
    # checkpoint dies with the session, never with a per-case timer).
    assert record.model_handle is None
    assert record.model_cleanup is None
    # Idempotent: nothing left to close or clean.
    assert manager.close_session("s-1") is False
    assert manager.close_session("s-missing") is False
    assert calls == ["cleanup"]


def test_close_all_only_closes_open_sessions(manager: ProfilingSessionManager) -> None:
    manager.create_session("s-1", OPERATOR_REQUEST, now=NOW)
    manager.create_session("s-2", OPERATOR_REQUEST, now=NOW)
    manager.close_session("s-2")
    assert manager.close_all() == ("s-1",)
    assert manager.close_all() == ()
    assert manager.open_session_count() == 0
    # Closed records stay readable (history is never erased, §44).
    assert set(manager.session_ids) == {"s-1", "s-2"}


# ---------------------------------------------------------------------------
# §44 forward-only case ledger
# ---------------------------------------------------------------------------


def test_ledger_starts_empty(manager: ProfilingSessionManager) -> None:
    manager.create_session("s-1", OPERATOR_REQUEST, now=NOW)
    assert manager.case_entry("s-1", "case-1") is None
    assert manager.case_entry("s-missing", "case-1") is None


def test_mark_running_creates_running_entry(manager: ProfilingSessionManager) -> None:
    manager.create_session("s-1", OPERATOR_REQUEST, now=NOW)
    manager.mark_running("s-1", "case-1")
    entry = manager.case_entry("s-1", "case-1")
    assert entry is not None
    assert entry.state is CaseState.RUNNING
    assert entry.terminal is False
    assert entry.outcome is None


def test_mark_running_never_reopens_terminal_entry(
    manager: ProfilingSessionManager,
) -> None:
    manager.create_session("s-1", OPERATOR_REQUEST, now=NOW)
    manager.decide_case("s-1", "case-1", CaseState.COMPLETED, OUTCOME, now=LATER)
    with pytest.raises(ValueError, match="never reopen"):
        manager.mark_running("s-1", "case-1")


def test_decide_case_records_terminal_outcome(manager: ProfilingSessionManager) -> None:
    manager.create_session("s-1", OPERATOR_REQUEST, now=NOW)
    manager.mark_running("s-1", "case-1")
    entry = manager.decide_case(
        "s-1", "case-1", CaseState.FAILED, OUTCOME, now=LATER
    )
    assert entry.state is CaseState.FAILED
    assert entry.outcome is OUTCOME
    assert entry.decided_at == LATER
    assert entry.terminal is True


@pytest.mark.parametrize(
    "state", [CaseState.PENDING, CaseState.RUNNING], ids=["pending", "running"]
)
def test_decide_case_requires_terminal_state(
    manager: ProfilingSessionManager, state: CaseState
) -> None:
    manager.create_session("s-1", OPERATOR_REQUEST, now=NOW)
    with pytest.raises(ValueError, match="terminal state"):
        manager.decide_case("s-1", "case-1", state, OUTCOME, now=LATER)


def test_decide_case_never_rewrites_history(manager: ProfilingSessionManager) -> None:
    manager.create_session("s-1", OPERATOR_REQUEST, now=NOW)
    manager.decide_case("s-1", "case-1", CaseState.COMPLETED, OUTCOME, now=LATER)
    with pytest.raises(ValueError, match="never rewritten"):
        manager.decide_case("s-1", "case-1", CaseState.FAILED, OUTCOME, now=LATER)
    # The recorded decision survived the rejected rewrite untouched.
    entry = manager.case_entry("s-1", "case-1")
    assert entry is not None and entry.state is CaseState.COMPLETED


def test_cancel_undispatched_case_records_typed_cancellation(
    manager: ProfilingSessionManager,
) -> None:
    manager.create_session("s-1", OPERATOR_REQUEST, now=NOW)
    entry = manager.cancel_case("s-1", "case-1", now=LATER)
    assert entry.state is CaseState.CANCELLED
    assert entry.terminal is True
    assert entry.outcome is not None
    failure = entry.outcome.failure
    assert failure is not None
    assert failure.category is ProfilingErrorCategory.CANCELLED
    assert ("case_id", "case-1") in failure.details


def test_cancel_running_case_wins(manager: ProfilingSessionManager) -> None:
    """Mid-run cancellation is the §41 path the runner detects and honors."""
    manager.create_session("s-1", OPERATOR_REQUEST, now=NOW)
    manager.mark_running("s-1", "case-1")
    entry = manager.cancel_case("s-1", "case-1", now=LATER)
    assert entry.state is CaseState.CANCELLED
    # The runner's decide_case for the discarded measurement now fails.
    with pytest.raises(ValueError, match="already decided"):
        manager.decide_case("s-1", "case-1", CaseState.COMPLETED, OUTCOME, now=LATER)


@pytest.mark.parametrize(
    "state",
    [CaseState.COMPLETED, CaseState.FAILED, CaseState.CANCELLED],
    ids=["completed", "failed", "cancelled"],
)
def test_cancel_never_undoes_terminal_history(
    manager: ProfilingSessionManager, state: CaseState
) -> None:
    manager.create_session("s-1", OPERATOR_REQUEST, now=NOW)
    decided = manager.decide_case("s-1", "case-1", state, OUTCOME, now=NOW)
    cancelled = manager.cancel_case("s-1", "case-1", now=LATER)
    assert cancelled is decided  # unchanged entry, same identity


def test_ledger_operations_require_live_session(manager: ProfilingSessionManager) -> None:
    manager.create_session("s-1", OPERATOR_REQUEST, now=NOW)
    manager.close_session("s-1")
    with pytest.raises(SessionRefused):
        manager.mark_running("s-1", "case-1")
    with pytest.raises(SessionRefused):
        manager.decide_case("s-1", "case-1", CaseState.COMPLETED, OUTCOME, now=LATER)
    with pytest.raises(SessionRefused):
        manager.cancel_case("s-1", "case-1", now=LATER)


def test_case_ledger_entry_defaults() -> None:
    entry = CaseLedgerEntry()
    assert entry.state is CaseState.PENDING
    assert entry.outcome is None
    assert entry.decided_at is None
    assert entry.terminal is False
