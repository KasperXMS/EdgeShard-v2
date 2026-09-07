"""Master-side profiling admin plane (Phase 2 spec §49).

``MasterProfilingAdmin`` is the handler behind ``ProfilingAdminService``:
the single place where an operator's *intent* (:class:`ProfilingRequest`,
submitted by the CLI) is expanded into dispatched work. The split of
responsibilities is the §46-§49 layering made executable:

- the CLI submits intents and reads back status/snapshots — it never plans;
- this module resolves the intent against live cluster facts (registered
  profiling Workers, Master-resolved network facts, Worker-side model
  inspections) and asks the :class:`ProfilingStrategy` to plan cases;
- the :class:`ProfilingController` persists and dispatches (§40) — the
  admin layer never touches a transport or the store directly except
  through the controller's public API;
- execution runs in the background (``StartExperiment`` answers as soon as
  the experiment is *created*; the CLI polls ``GetExperiment``).

Honesty rules this module enforces:

- an intent referencing an unknown Worker, a failed model inspection, or
  missing cluster facts is *rejected whole* with a detail — never silently
  narrowed (§52.2: no guessing which subset the operator meant);
- an intent that plans to zero cases (everything already measured, or a
  single Worker with no network peer) is refused with that honest reason
  rather than accepted as an empty experiment;
- a request that cannot be expanded is a protocol-level ``ValueError`` →
  the servicer aborts INVALID_ARGUMENT, mirroring the Worker plane (§47).

Typed case failures are not part of the store's ledger (it keeps states,
§43), so ``GetExperiment`` enriches stored states with the failures from
this process's cached run reports; after a Master restart a FAILED case is
reported without its failure object — absence, never a reconstruction.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Mapping
from collections.abc import Set as AbstractSet

from edgeshard.control.master.profiling_controller import (
    ExperimentReport,
    ProfilingController,
)
from edgeshard.profiling.domain.experiment import (
    CaseState,
    ExperimentStatus,
    ModelCaseSpec,
    NetworkCaseSpec,
    ProfilingCase,
    ProfilingFailure,
    ProfilingRequest,
)
from edgeshard.profiling.domain.session import (
    ProfilingSessionKind,
    ProfilingSessionRequest,
)
from edgeshard.profiling.domain.signature import ProfilingGranularity
from edgeshard.profiling.store.base import ProfileStoreError
from edgeshard.profiling.strategy.base import ProfilingStrategy
from edgeshard.profiling.strategy.default import DefaultProfilingStrategy
from edgeshard.protocol.profiling import mapper

logger = logging.getLogger("edgeshard.control.master.profiling_admin")


class _AdminRejection(Exception):
    """An intent that cannot be expanded honestly (§52.2).

    Rejections are *not* protocol errors: they travel back as
    ``accepted=False`` with the detail, so the CLI shows the operator why
    instead of surfacing a transport failure.
    """


class MasterProfilingAdmin:
    """Expands operator intents into experiments; satisfies the admin handler.

    Structural typing (``ProfilingAdminHandler``) keeps the servicer and the
    tests decoupled from this concrete class.
    """

    def __init__(
        self,
        *,
        controller: ProfilingController,
        strategy: ProfilingStrategy | None = None,
    ) -> None:
        self._controller = controller
        self._strategy = strategy if strategy is not None else DefaultProfilingStrategy()
        self._runs: dict[str, asyncio.Task[ExperimentReport]] = {}
        self._reports: dict[str, ExperimentReport] = {}

    # -- StartExperiment (§49) -------------------------------------------------

    async def start_experiment(
        self, request: mapper.StartExperimentRequest
    ) -> mapper.StartExperimentResponse:
        intent = request.request
        try:
            cases = await self._expand(intent)
        except _AdminRejection as exc:
            logger.info("start_experiment rejected: %s", exc)
            return mapper.StartExperimentResponse(accepted=False, detail=str(exc))
        if not cases:
            detail = (
                "the request plans to zero cases — everything it targets is "
                "already measured or no peer exists to probe (§28); nothing "
                "was dispatched"
            )
            logger.info("start_experiment rejected: %s", detail)
            return mapper.StartExperimentResponse(accepted=False, detail=detail)

        experiment = self._controller.create_experiment(
            strategy_id=self._strategy.strategy_id,
            cases=cases,
            requested_by=intent.requested_by,
        )
        self._launch(experiment.experiment_id)
        logger.info(
            "start_experiment accepted: %s (%d case(s), strategy=%s)",
            experiment.experiment_id,
            len(cases),
            self._strategy.strategy_id,
        )
        return mapper.StartExperimentResponse(
            accepted=True, experiment_id=experiment.experiment_id
        )

    def _launch(self, experiment_id: str) -> None:
        """Run the experiment in the background unless a run is in flight.

        ``StartExperiment`` answers at creation time; the CLI polls
        ``GetExperiment`` (§49). A duplicate start of an already-running
        experiment does not fork a second dispatcher — ``run_experiment``
        is resume-safe, and one in-flight task per experiment is enough.
        """
        existing = self._runs.get(experiment_id)
        if existing is not None and not existing.done():
            return
        task = asyncio.create_task(
            self._controller.run_experiment(experiment_id),
            name=f"profiling-experiment-{experiment_id[:12]}",
        )
        self._runs[experiment_id] = task
        task.add_done_callback(
            lambda finished: self._finish_run(experiment_id, finished)
        )

    def _finish_run(
        self, experiment_id: str, task: asyncio.Task[ExperimentReport]
    ) -> None:
        self._runs.pop(experiment_id, None)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            # The store states remain the source of truth; a crashed run
            # leaves non-terminal cases that a resume replays (§50).
            logger.error(
                "background run of experiment %s raised: %r", experiment_id, exc
            )
            return
        # Cache the terminal report: it carries the typed failures the store
        # deliberately does not keep (§43), used to enrich GetExperiment.
        self._reports[experiment_id] = task.result()

    # -- GetExperiment (§49) ----------------------------------------------------

    async def get_experiment(
        self, request: mapper.GetExperimentRequest
    ) -> mapper.GetExperimentResponse:
        status = self._controller.experiment_status(request.experiment_id)
        if status is None:
            return mapper.GetExperimentResponse(found=False)
        return mapper.GetExperimentResponse(found=True, status=self._enrich(status))

    def _enrich(self, status: ExperimentStatus) -> ExperimentStatus:
        """Attach this process's cached typed failures to stored case states.

        The store keeps case *states* only (§43); failures live in the run
        report. Without a cached report (e.g. after a Master restart) the
        enriched status simply carries no failure objects — honest absence
        (§52.2), never a reconstruction from strings.
        """
        report = self._reports.get(status.experiment.experiment_id)
        if report is None:
            return status
        failures = {
            case.case_id: case.outcome.failure
            for case in report.cases
            if case.outcome is not None and case.outcome.failure is not None
        }
        if not failures:
            return status
        negative = (CaseState.FAILED, CaseState.CANCELLED)
        enriched = tuple(
            dataclasses.replace(case, failure=failures[case.case_id])
            if case.failure is None and case.state in negative and case.case_id in failures
            else case
            for case in status.cases
        )
        return dataclasses.replace(status, cases=enriched)

    # -- CancelExperiment (§49) --------------------------------------------------

    async def cancel_experiment(
        self, request: mapper.CancelExperimentRequest
    ) -> mapper.CancelExperimentResponse:
        experiment_id = request.experiment_id
        if self._controller.load_experiment(experiment_id) is None:
            return mapper.CancelExperimentResponse(
                accepted=False, detail=f"unknown experiment {experiment_id!r}"
            )
        task = self._runs.pop(experiment_id, None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning(
                    "background run of experiment %s raised during cancellation",
                    experiment_id,
                    exc_info=True,
                )
        report = await self._controller.cancel_experiment(experiment_id)
        self._reports[experiment_id] = report
        logger.info(
            "cancel_experiment accepted: %s is now %s",
            experiment_id,
            report.state.value,
        )
        return mapper.CancelExperimentResponse(
            accepted=True,
            detail=f"experiment {experiment_id} is {report.state.value}",
        )

    # -- BuildProfileSnapshot (§46, §49) ------------------------------------------

    async def build_profile_snapshot(
        self, request: mapper.BuildProfileSnapshotRequest
    ) -> mapper.BuildProfileSnapshotResponse:
        try:
            snapshot = self._controller.build_profile_snapshot()
        except ProfileStoreError as exc:
            return mapper.BuildProfileSnapshotResponse(
                accepted=False, detail=f"snapshot build failed: {exc}"
            )
        return mapper.BuildProfileSnapshotResponse(accepted=True, snapshot=snapshot)

    # -- lifecycle ---------------------------------------------------------------

    async def shutdown(self) -> None:
        """Stop background runs without cancelling the experiments themselves.

        A stopping Master leaves non-terminal cases as they are in the store;
        ``run_experiment`` is resume-safe (§50), so the next ``master serve``
        continues an interrupted experiment instead of re-benchmarking
        finished work. Worker-side sessions converge on their own: leases
        die with the registration (§39/§41).
        """
        tasks = list(self._runs.values())
        self._runs.clear()
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning(
                    "background run raised during shutdown", exc_info=True
                )

    # -- intent expansion (§46-§47) -----------------------------------------------

    async def _expand(self, intent: ProfilingRequest) -> tuple[ProfilingCase, ...]:
        workers = self._resolve_workers(intent.worker_ids)
        if intent.kind is ProfilingSessionKind.NETWORK:
            return self._expand_network(intent, workers)
        return await self._expand_model_family(intent, workers)

    def _resolve_workers(self, requested: tuple[str, ...]) -> tuple[str, ...]:
        """The intent's executor set, or an honest rejection (§52.2).

        Empty means "every registered Worker hosting the profiling plane";
        an explicit id that is unknown — or known without a profiling
        endpoint — rejects the whole request rather than silently dropping
        the Worker the operator named.
        """
        available = set(self._controller.profiling_worker_ids())
        if not requested:
            if not available:
                raise _AdminRejection(
                    "no registered worker hosts the profiling service (§41); "
                    "start workers with profiling enabled first"
                )
            return tuple(sorted(available))
        unknown = [worker_id for worker_id in requested if worker_id not in available]
        if unknown:
            raise _AdminRejection(
                "worker(s) not registered with a profiling endpoint: "
                + ", ".join(repr(worker_id) for worker_id in unknown)
                + " (§52.2)"
            )
        return requested

    def _expand_network(
        self, intent: ProfilingRequest, workers: tuple[str, ...]
    ) -> tuple[ProfilingCase, ...]:
        all_facts = self._controller.cluster_network_facts()
        missing = [worker_id for worker_id in workers if worker_id not in all_facts]
        if missing:
            raise _AdminRejection(
                "no cluster network facts for worker(s): "
                + ", ".join(repr(worker_id) for worker_id in missing)
                + " (§52.2)"
            )
        plan = self._strategy.plan_network_cases(
            facts=(all_facts[worker_id] for worker_id in workers),
            extra_bandwidth_pairs=intent.extra_bandwidth_pairs,
            bandwidth_path_classes=intent.bandwidth_path_classes or None,
        )
        # §47 network steps 1-2 are facts in their own right: persist the
        # endpoint profiles and classifications before dispatching probes.
        self._controller.record_network_characterization(
            plan.endpoint_profiles, plan.classified_pairs
        )
        cases = plan.cases
        if intent.network_probe is not None:
            cases = tuple(
                case
                for case in cases
                if isinstance(case.spec, NetworkCaseSpec)
                and case.spec.probe_kind is intent.network_probe
            )
        return cases

    async def _expand_model_family(
        self, intent: ProfilingRequest, workers: tuple[str, ...]
    ) -> tuple[ProfilingCase, ...]:
        model = intent.model
        dtype = intent.dtype
        if model is None or dtype is None or not intent.device_ids:
            # Unreachable: ProfilingRequest validation forces all three for
            # MODEL/OPERATOR intents. Fail loudly rather than plan malformed.
            raise _AdminRejection(
                "model/operator intents require a model reference, dtype, "
                "and device ids (§38)"
            )
        # §47 step 1 happens Worker-side: one inspection session per worker
        # (a MODEL prepare loads and characterizes the checkpoint, §38), and
        # the strategy plans from the reported facts — the Master never
        # loads a model itself (§40, §46).
        inspection = ProfilingSessionRequest(
            kind=ProfilingSessionKind.MODEL,
            device_ids=intent.device_ids,
            model=model,
            dtype=dtype,
        )
        measured = self._measured_by_device(intent)
        cases: list[ProfilingCase] = []
        for worker_id in workers:
            facts = await self._controller.inspect_model(
                worker_id=worker_id, request=inspection
            )
            if isinstance(facts, ProfilingFailure):
                raise _AdminRejection(
                    f"model inspection failed on worker {worker_id!r}: "
                    f"[{facts.category.value}] {facts.message}"
                )
            self._controller.record_model_facts(facts)
            plan = self._strategy.plan_model_cases(
                worker_id=worker_id,
                facts=facts,
                device_ids=intent.device_ids,
                measured_signature_ids=measured,
            )
            cases.extend(plan.cases)
        if intent.kind is ProfilingSessionKind.OPERATOR:
            cases = [
                case
                for case in cases
                if isinstance(case.spec, ModelCaseSpec)
                and case.spec.granularity is ProfilingGranularity.OPERATOR
            ]
        return tuple(cases)

    def _measured_by_device(
        self, intent: ProfilingRequest
    ) -> Mapping[str, AbstractSet[str]] | None:
        """The §28 reuse answer per device, or None to re-measure everything.

        v1 scopes reuse to "ever measured anywhere" — device performance
        classes and environment fingerprints (§9) are persisted by the store
        but assigning them needs hardware identity the Master does not have
        yet (§48: no over-engineering; the knob exists in the store query).
        """
        if not intent.missing_only:
            return None
        measured_ids = self._controller.measured_operator_signature_ids()
        return dict.fromkeys(intent.device_ids, measured_ids)


__all__ = ["MasterProfilingAdmin"]
