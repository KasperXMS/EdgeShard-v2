"""Liveness evaluation (Phase 1 spec §32, §37, §31).

Status is *derived*, never stored: a Worker is ONLINE while its last
accepted heartbeat is within ``suspect_after_ms`` of the Master-local
monotonic clock, SUSPECT up to ``offline_after_ms``, and OFFLINE beyond
that (spec §32). Deriving from the StateStore's receive timestamps keeps a
single source of truth — Worker wall clocks play no role (spec §31) — and
makes the default thresholds (5 s interval / 10 s suspect / 20 s offline)
tolerate a single missed heartbeat without degradation.

``evaluate`` is the synchronous tick a periodic asyncio task runs (spec
§37); it only exists to log status *transitions* in the §46 format. Tests
drive it with an injected fake clock and never sleep in real time.

OFFLINE is a status, not an eviction: Workers remain in the registry and
the StateStore indefinitely (spec §32).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable

from edgeshard.cluster.state import WorkerStatus
from edgeshard.control.master.config import MasterConfig
from edgeshard.control.master.state_store import StateStore

logger = logging.getLogger("master.liveness")


class LivenessManager:
    """Derives ONLINE/SUSPECT/OFFLINE from Master-local monotonic timestamps."""

    def __init__(
        self,
        states: StateStore,
        config: MasterConfig,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._states = states
        self._config = config
        self._monotonic = monotonic
        self._last_logged: dict[str, WorkerStatus] = {}
        self._task: asyncio.Task[None] | None = None

    def status_of(self, worker_id: str) -> WorkerStatus:
        """Current liveness of one Worker; ``KeyError`` if it never registered."""
        return self.status_at(worker_id, self._monotonic())

    def status_at(self, worker_id: str, now: float) -> WorkerStatus:
        """Liveness of one Worker at an explicit monotonic instant.

        ``SnapshotBuilder`` passes one ``now`` for the whole snapshot, so
        every Worker in it is classified against the same instant (§38: a
        snapshot is internally consistent, not a spread of clock reads).
        """
        entry = self._states.get(worker_id)
        if entry is None:
            raise KeyError(worker_id)
        return self._classify(now - entry.received_monotonic)

    def statuses(self) -> dict[str, WorkerStatus]:
        """Liveness of every tracked Worker (snapshot/P1H and test convenience)."""
        return self.statuses_at(self._monotonic())

    def statuses_at(self, now: float) -> dict[str, WorkerStatus]:
        """Liveness of every tracked Worker at one explicit monotonic instant."""
        return {
            worker_id: self._classify(now - entry.received_monotonic)
            for worker_id, entry in self._states.items()
        }

    def _classify(self, elapsed_seconds: float) -> WorkerStatus:
        """Map age of the last accepted heartbeat to a status (spec §32).

        Boundaries follow the spec wording: ONLINE *within* suspect_after,
        SUSPECT *beyond* suspect_after, OFFLINE *beyond* offline_after.
        """
        if elapsed_seconds * 1000.0 <= self._config.suspect_after_ms:
            return WorkerStatus.ONLINE
        if elapsed_seconds * 1000.0 <= self._config.offline_after_ms:
            return WorkerStatus.SUSPECT
        return WorkerStatus.OFFLINE

    def evaluate(self) -> None:
        """One tick: recompute every status and log transitions (spec §37, §46)."""
        for worker_id, status in self.statuses().items():
            previous = self._last_logged.get(worker_id)
            if previous == status:
                continue
            self._last_logged[worker_id] = status
            if status is WorkerStatus.SUSPECT:
                logger.warning("worker_id=%s state=%s", worker_id, status.value)
            else:
                logger.info("worker_id=%s state=%s", worker_id, status.value)

    async def _run(self) -> None:
        while True:
            self.evaluate()
            await asyncio.sleep(self._config.liveness_tick_ms / 1000.0)

    async def start(self) -> None:
        """Start the periodic evaluation task (spec §37); idempotent."""
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="master-liveness")

    async def stop(self) -> None:
        """Cancel the periodic evaluation task; idempotent."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
