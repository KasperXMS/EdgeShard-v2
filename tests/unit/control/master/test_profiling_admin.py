"""MasterProfilingAdmin tests (Phase 2 spec §49).

The admin plane is exercised over the controller rig (real MasterService,
real SqliteProfileStore, fake Worker transports): intents expand against
live cluster facts, rejections stay honest (§52.2), background runs answer
at creation time, GetExperiment enriches stored states with this process's
typed failures (§43), and shutdown leaves experiments resumable (§50).
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path

from test_profiling_controller import (
    CHARACTERIZATION,
    ENDPOINT_2,
    MODEL,
    SESSION_FACTS,
    W1,
    W2,
    Rig,
    make_rig,
    register_worker,
    transport_for,
)

from edgeshard.control.master.profiling_admin import MasterProfilingAdmin
from edgeshard.profiling.domain.experiment import (
    CaseOutcome,
    CaseState,
    ExperimentState,
    ModelCaseSpec,
    NetworkCaseSpec,
    ProfilingErrorCategory,
    ProfilingFailure,
    ProfilingRequest,
    WorkerDeviceTarget,
)
from edgeshard.profiling.domain.network import NetworkPair, ProbeKind
from edgeshard.profiling.domain.session import ProfilingSessionKind
from edgeshard.profiling.domain.signature import ProfilingGranularity
from edgeshard.protocol.profiling.mapper import (
    BuildProfileSnapshotRequest,
    CancelExperimentRequest,
    GetExperimentRequest,
    PrepareProfilingSessionResponse,
    RunProfilingCaseResponse,
    StartExperimentRequest,
)
from factories import (
    RTX_GPU_DEVICE_ID,
    finalize,
    make_rtx_capability,
    make_worker_state,
)


def model_intent(**overrides: object) -> ProfilingRequest:
    base: dict = {
        "kind": ProfilingSessionKind.MODEL,
        "model": MODEL,
        "dtype": "fp32",
        "worker_device_targets": (
            WorkerDeviceTarget(W1, RTX_GPU_DEVICE_ID),
        ),
        "requested_by": "operator",
    }
    base.update(overrides)
    return ProfilingRequest(**base)


def network_intent(**overrides: object) -> ProfilingRequest:
    base: dict = {
        "kind": ProfilingSessionKind.NETWORK,
        "worker_ids": (W1, W2),
        "requested_by": "operator",
    }
    base.update(overrides)
    return ProfilingRequest(**base)


def make_admin(rig: Rig) -> MasterProfilingAdmin:
    return MasterProfilingAdmin(controller=rig.controller)


def script_healthy_inspection(rig: Rig) -> None:
    assert SESSION_FACTS.environment is not None
    facts = dataclasses.replace(
        SESSION_FACTS,
        environment=dataclasses.replace(
            SESSION_FACTS.environment,
            device_id=RTX_GPU_DEVICE_ID,
        ),
    )
    transport_for(rig).prepare_response = PrepareProfilingSessionResponse(
        accepted=True, session_facts=facts
    )


def stored_cases(rig: Rig, experiment_id: str):
    stored_experiment = rig.store.get_experiment(experiment_id)
    assert stored_experiment is not None
    cases = []
    for case_id in stored_experiment.experiment.case_ids:
        stored = rig.store.get_case(case_id)
        assert stored is not None
        cases.append(stored.case)
    return cases


async def drain(admin: MasterProfilingAdmin, experiment_id: str):
    """Await the background run launched by StartExperiment (§49)."""
    task = admin._runs[experiment_id]
    report = await task
    await asyncio.sleep(0)  # flush the done-callback report cache
    return report


class TestStartExperimentNetwork:
    async def test_two_workers_plan_runs_and_complete(self, tmp_path: Path) -> None:
        rig = make_rig(tmp_path)
        await register_worker(rig, W1)
        await register_worker(rig, W2, profiling_endpoint=ENDPOINT_2)
        admin = make_admin(rig)

        response = await admin.start_experiment(
            StartExperimentRequest(request=network_intent())
        )

        assert response.accepted is True
        assert response.experiment_id
        report = await drain(admin, response.experiment_id)
        assert report.state is ExperimentState.COMPLETED
        cases = stored_cases(rig, response.experiment_id)
        assert cases
        assert all(isinstance(case.spec, NetworkCaseSpec) for case in cases)
        # §47 network steps 1-2 were persisted before dispatch, and the run
        # landed measurements in the store (the fake transport's records
        # carry latency-only metrics, so they route to `measurements`).
        snapshot = rig.store.build_snapshot("s-1")
        assert snapshot.measurements

    async def test_probe_filter_selects_only_rtt(self, tmp_path: Path) -> None:
        rig = make_rig(tmp_path)
        await register_worker(rig, W1)
        await register_worker(rig, W2, profiling_endpoint=ENDPOINT_2)
        admin = make_admin(rig)

        response = await admin.start_experiment(
            StartExperimentRequest(
                request=network_intent(network_probe=ProbeKind.RTT)
            )
        )

        assert response.accepted is True
        cases = stored_cases(rig, response.experiment_id)
        assert cases
        for case in cases:
            assert isinstance(case.spec, NetworkCaseSpec)
            assert case.spec.probe_kind is ProbeKind.RTT
        await drain(admin, response.experiment_id)

    async def test_explicit_interface_pair_excludes_default_bandwidth_paths(
        self, tmp_path: Path
    ) -> None:
        rig = make_rig(tmp_path)
        await register_worker(rig, W1)
        await register_worker(rig, W2, profiling_endpoint=ENDPOINT_2)
        admin = make_admin(rig)
        pair = NetworkPair(
            source_worker_id=W1,
            destination_worker_id=W2,
            source_interface_id="nic-0",
            destination_interface_id="nic-0",
        )

        response = await admin.start_experiment(
            StartExperimentRequest(
                request=network_intent(
                    network_probe=ProbeKind.BANDWIDTH,
                    extra_bandwidth_pairs=(pair,),
                )
            )
        )

        assert response.accepted
        cases = stored_cases(rig, response.experiment_id)
        assert cases
        assert all(
            isinstance(case.spec, NetworkCaseSpec)
            and case.spec.source_interface_id == "nic-0"
            and case.spec.destination_interface_id == "nic-0"
            for case in cases
        )
        await drain(admin, response.experiment_id)

    async def test_explicit_interface_pair_selects_rtt_path(
        self, tmp_path: Path
    ) -> None:
        rig = make_rig(tmp_path)
        await register_worker(rig, W1)
        await register_worker(rig, W2, profiling_endpoint=ENDPOINT_2)
        admin = make_admin(rig)
        pair = NetworkPair(W1, W2, "nic-0", "nic-0")

        response = await admin.start_experiment(
            StartExperimentRequest(
                request=network_intent(
                    network_probe=ProbeKind.RTT,
                    network_pairs=(pair,),
                )
            )
        )

        assert response.accepted
        cases = stored_cases(rig, response.experiment_id)
        assert len(cases) == 1
        spec = cases[0].spec
        assert isinstance(spec, NetworkCaseSpec)
        assert spec.probe_kind is ProbeKind.RTT
        assert spec.source_interface_id == "nic-0"
        assert spec.destination_interface_id == "nic-0"
        await drain(admin, response.experiment_id)

    async def test_single_worker_without_peer_is_zero_case_rejection(
        self, tmp_path: Path
    ) -> None:
        rig = make_rig(tmp_path)
        await register_worker(rig, W1)
        admin = make_admin(rig)

        response = await admin.start_experiment(
            StartExperimentRequest(request=network_intent(worker_ids=(W1,)))
        )

        assert response.accepted is False
        assert "zero cases" in response.detail
        assert response.experiment_id == ""


class TestStartExperimentRejections:
    async def test_unknown_worker_rejects_whole_request(self, tmp_path: Path) -> None:
        rig = make_rig(tmp_path)
        await register_worker(rig, W1)
        admin = make_admin(rig)

        response = await admin.start_experiment(
            StartExperimentRequest(request=network_intent(worker_ids=(W1, "ghost")))
        )

        assert response.accepted is False
        assert "ghost" in response.detail
        assert "52.2" in response.detail
        assert response.experiment_id == ""

    async def test_no_profiling_workers_rejects(self, tmp_path: Path) -> None:
        rig = make_rig(tmp_path)
        await register_worker(rig, W1, profiling_endpoint=None)
        admin = make_admin(rig)

        response = await admin.start_experiment(
            StartExperimentRequest(request=network_intent(worker_ids=()))
        )

        assert response.accepted is False
        assert "no registered worker" in response.detail

    async def test_failed_inspection_rejects_with_typed_detail(
        self, tmp_path: Path
    ) -> None:
        rig = make_rig(tmp_path)
        await register_worker(rig, W1)
        transport_for(rig).prepare_response = PrepareProfilingSessionResponse(
            accepted=False,
            detail="model unavailable",
            failure=ProfilingFailure(
                category=ProfilingErrorCategory.UNSUPPORTED_MODEL,
                message="no READY snapshot",
            ),
        )
        admin = make_admin(rig)

        response = await admin.start_experiment(
            StartExperimentRequest(request=model_intent())
        )

        assert response.accepted is False
        assert "model inspection failed" in response.detail
        assert "unsupported_model" in response.detail


class TestStartExperimentModelFamily:
    async def test_explicit_targets_preserve_worker_local_device_mapping(
        self, tmp_path: Path
    ) -> None:
        rig = make_rig(tmp_path)
        await register_worker(rig, W1)
        second_gpu_id = "GPU-worker-two-only"
        capability = make_rtx_capability()
        cpu, gpu = capability.devices
        host_pool, gpu_pool = capability.memory_pools
        second_pool_id = f"gpu-{second_gpu_id}-vram"
        capability = finalize(
            dataclasses.replace(
                capability,
                capability_revision="",
                devices=(
                    cpu,
                    dataclasses.replace(
                        gpu,
                        identity=dataclasses.replace(
                            gpu.identity, device_id=second_gpu_id
                        ),
                        memory_pool_id=second_pool_id,
                    ),
                ),
                memory_pools=(
                    host_pool,
                    dataclasses.replace(
                        gpu_pool, memory_pool_id=second_pool_id
                    ),
                ),
            )
        )
        await register_worker(
            rig,
            W2,
            profiling_endpoint=ENDPOINT_2,
            capability=capability,
            initial_state=make_worker_state(
                W2,
                device_ids=(second_gpu_id,),
                pool_ids=(second_pool_id,),
            ),
        )
        script_healthy_inspection(rig)
        assert SESSION_FACTS.environment is not None
        transport_for(rig, ENDPOINT_2).prepare_response = (
            PrepareProfilingSessionResponse(
                accepted=True,
                session_facts=dataclasses.replace(
                    SESSION_FACTS,
                    environment=dataclasses.replace(
                        SESSION_FACTS.environment,
                        worker_id=W2,
                        device_id=second_gpu_id,
                    ),
                ),
            )
        )
        intent = model_intent(
            kind=ProfilingSessionKind.OPERATOR,
            worker_device_targets=(
                WorkerDeviceTarget(W1, RTX_GPU_DEVICE_ID),
                WorkerDeviceTarget(W2, second_gpu_id),
            ),
        )

        admin = make_admin(rig)
        cases = await admin._expand(intent)

        mapped = {
            (case.worker_id, case.spec.device_ids[0])
            for case in cases
            if isinstance(case.spec, ModelCaseSpec)
        }
        assert mapped == {
            (W1, RTX_GPU_DEVICE_ID),
            (W2, second_gpu_id),
        }
        assert len(transport_for(rig).prepare_requests) == 1
        assert len(transport_for(rig, ENDPOINT_2).prepare_requests) == 1

        rejected = await admin.start_experiment(
            StartExperimentRequest(
                request=model_intent(
                    worker_device_targets=(
                        WorkerDeviceTarget(W1, second_gpu_id),
                    )
                )
            )
        )
        assert not rejected.accepted
        assert "does not belong to worker" in rejected.detail

    async def test_model_intent_inspects_plans_and_runs(self, tmp_path: Path) -> None:
        rig = make_rig(tmp_path)
        await register_worker(rig, W1)
        script_healthy_inspection(rig)
        admin = make_admin(rig)

        response = await admin.start_experiment(
            StartExperimentRequest(request=model_intent())
        )

        assert response.accepted is True
        report = await drain(admin, response.experiment_id)
        assert report.state is ExperimentState.COMPLETED
        # The inspection went over the wire as a MODEL prepare (§47 step 1);
        # later prepares belong to the execution session of the run itself.
        prepare = transport_for(rig).prepare_requests[0]
        assert prepare.session_request.kind is ProfilingSessionKind.MODEL
        assert prepare.session_request.model == MODEL
        assert prepare.session_request.dtype == "fp32"
        # The plan spans operator, module, and layer granularities.
        granularities = {
            case.spec.granularity
            for case in stored_cases(rig, response.experiment_id)
            if isinstance(case.spec, ModelCaseSpec)
        }
        assert ProfilingGranularity.OPERATOR in granularities
        assert ProfilingGranularity.TRANSFORMER_LAYER in granularities
        # Inspection facts were persisted (§46 planning artifact).
        snapshot = rig.store.build_snapshot("s-1")
        assert CHARACTERIZATION in snapshot.model_characterizations

    async def test_operator_intent_filters_to_operator_cases(
        self, tmp_path: Path
    ) -> None:
        rig = make_rig(tmp_path)
        await register_worker(rig, W1)
        script_healthy_inspection(rig)
        admin = make_admin(rig)

        response = await admin.start_experiment(
            StartExperimentRequest(
                request=model_intent(kind=ProfilingSessionKind.OPERATOR)
            )
        )

        assert response.accepted is True
        cases = stored_cases(rig, response.experiment_id)
        assert cases
        for case in cases:
            assert isinstance(case.spec, ModelCaseSpec)
            assert case.spec.granularity is ProfilingGranularity.OPERATOR
        await drain(admin, response.experiment_id)

    async def test_missing_only_reuses_measured_operators(self, tmp_path: Path) -> None:
        """§28: the second identical OPERATOR intent plans to zero cases."""
        rig = make_rig(tmp_path)
        await register_worker(rig, W1)
        script_healthy_inspection(rig)
        admin = make_admin(rig)
        intent = model_intent(kind=ProfilingSessionKind.OPERATOR)

        first = await admin.start_experiment(StartExperimentRequest(request=intent))
        assert first.accepted is True
        await drain(admin, first.experiment_id)

        second = await admin.start_experiment(StartExperimentRequest(request=intent))
        assert second.accepted is False
        assert "zero cases" in second.detail

    async def test_include_measured_replans_everything(self, tmp_path: Path) -> None:
        rig = make_rig(tmp_path)
        await register_worker(rig, W1)
        script_healthy_inspection(rig)
        admin = make_admin(rig)
        intent = model_intent(kind=ProfilingSessionKind.OPERATOR)

        first = await admin.start_experiment(StartExperimentRequest(request=intent))
        assert first.accepted is True
        await drain(admin, first.experiment_id)

        second = await admin.start_experiment(
            StartExperimentRequest(
                request=model_intent(
                    kind=ProfilingSessionKind.OPERATOR, missing_only=False
                )
            )
        )
        assert second.accepted is True
        # Explicit reruns retain configuration identity but get a fresh run id.
        assert second.experiment_id != first.experiment_id
        report = await drain(admin, second.experiment_id)
        assert report.state is ExperimentState.COMPLETED
        first_stored = rig.store.get_experiment(first.experiment_id)
        second_stored = rig.store.get_experiment(second.experiment_id)
        assert first_stored is not None and second_stored is not None
        assert (
            first_stored.experiment.configuration_id
            == second_stored.experiment.configuration_id
        )
        assert first_stored.experiment.case_ids != second_stored.experiment.case_ids
        assert len(rig.store.build_snapshot("rerun").measurements) == 2


class TestGetExperiment:
    async def test_unknown_experiment_is_not_found(self, tmp_path: Path) -> None:
        rig = make_rig(tmp_path)
        admin = make_admin(rig)

        response = await admin.get_experiment(GetExperimentRequest(experiment_id="nope"))

        assert response.found is False
        assert response.status is None

    async def test_failed_case_is_enriched_with_typed_failure(
        self, tmp_path: Path
    ) -> None:
        rig = make_rig(tmp_path)
        await register_worker(rig, W1)
        script_healthy_inspection(rig)
        admin = make_admin(rig)

        response = await admin.start_experiment(
            StartExperimentRequest(
                request=model_intent(kind=ProfilingSessionKind.OPERATOR)
            )
        )
        assert response.accepted is True
        # Script the failure before the background task gets to run.
        (case,) = stored_cases(rig, response.experiment_id)
        failure = ProfilingFailure(
            category=ProfilingErrorCategory.BENCHMARK_FAILED, message="boom"
        )
        transport_for(rig).script_run(
            case.case_id,
            RunProfilingCaseResponse(
                accepted=True, outcome=CaseOutcome.from_failure(failure)
            ),
        )
        report = await drain(admin, response.experiment_id)
        assert report.state is ExperimentState.FAILED

        status_response = await admin.get_experiment(
            GetExperimentRequest(experiment_id=response.experiment_id)
        )

        assert status_response.found is True
        assert status_response.status is not None
        (case_status,) = status_response.status.cases
        assert case_status.state is CaseState.FAILED
        # The typed failure remains available without the report cache.
        assert case_status.failure == failure

    async def test_without_cached_report_reads_durable_failure(
        self, tmp_path: Path
    ) -> None:
        rig = make_rig(tmp_path)
        await register_worker(rig, W1)
        script_healthy_inspection(rig)
        admin = make_admin(rig)
        response = await admin.start_experiment(
            StartExperimentRequest(
                request=model_intent(kind=ProfilingSessionKind.OPERATOR)
            )
        )
        (case,) = stored_cases(rig, response.experiment_id)
        transport_for(rig).script_run(
            case.case_id,
            RunProfilingCaseResponse(
                accepted=True,
                outcome=CaseOutcome.from_failure(
                    ProfilingFailure(
                        category=ProfilingErrorCategory.BENCHMARK_FAILED,
                        message="boom",
                    )
                ),
            ),
        )
        # Run through the controller directly: the admin has no cached report,
        # as after a Master restart.
        await rig.controller.run_experiment(response.experiment_id)

        status_response = await admin.get_experiment(
            GetExperimentRequest(experiment_id=response.experiment_id)
        )

        assert status_response.found is True
        assert status_response.status is not None
        (case_status,) = status_response.status.cases
        assert case_status.state is CaseState.FAILED
        assert case_status.failure is not None
        assert case_status.failure.category is ProfilingErrorCategory.BENCHMARK_FAILED


class TestCancelExperiment:
    async def test_unknown_experiment_is_rejected(self, tmp_path: Path) -> None:
        rig = make_rig(tmp_path)
        admin = make_admin(rig)

        response = await admin.cancel_experiment(
            CancelExperimentRequest(experiment_id="nope")
        )

        assert response.accepted is False
        assert "unknown experiment" in response.detail

    async def test_cancel_stops_background_run_and_reports_state(
        self, tmp_path: Path
    ) -> None:
        rig = make_rig(tmp_path)
        await register_worker(rig, W1)
        script_healthy_inspection(rig)
        admin = make_admin(rig)
        response = await admin.start_experiment(
            StartExperimentRequest(
                request=model_intent(kind=ProfilingSessionKind.OPERATOR)
            )
        )
        assert response.accepted is True

        cancel = await admin.cancel_experiment(
            CancelExperimentRequest(experiment_id=response.experiment_id)
        )

        assert cancel.accepted is True
        assert "cancelled" in cancel.detail
        assert response.experiment_id not in admin._runs
        experiment = rig.store.get_experiment(response.experiment_id)
        assert experiment is not None
        assert experiment.state is ExperimentState.CANCELLED


class TestBuildProfileSnapshot:
    async def test_snapshot_reflects_persisted_facts(self, tmp_path: Path) -> None:
        rig = make_rig(tmp_path)
        await register_worker(rig, W1)
        script_healthy_inspection(rig)
        admin = make_admin(rig)
        response = await admin.start_experiment(
            StartExperimentRequest(
                request=model_intent(kind=ProfilingSessionKind.OPERATOR)
            )
        )
        assert response.accepted is True
        await drain(admin, response.experiment_id)

        snapshot_response = await admin.build_profile_snapshot(
            BuildProfileSnapshotRequest()
        )

        assert snapshot_response.accepted is True
        assert snapshot_response.snapshot is not None
        assert CHARACTERIZATION in snapshot_response.snapshot.model_characterizations
        assert snapshot_response.snapshot.measurements

    async def test_empty_store_snapshot_is_still_accepted(self, tmp_path: Path) -> None:
        rig = make_rig(tmp_path)
        admin = make_admin(rig)

        snapshot_response = await admin.build_profile_snapshot(
            BuildProfileSnapshotRequest()
        )

        assert snapshot_response.accepted is True
        assert snapshot_response.snapshot is not None
        assert snapshot_response.snapshot.measurements == ()


class TestShutdown:
    async def test_shutdown_leaves_experiment_resumable(self, tmp_path: Path) -> None:
        """§50: a stopping Master cancels tasks, never stored experiments."""
        rig = make_rig(tmp_path)
        await register_worker(rig, W1)
        script_healthy_inspection(rig)
        admin = make_admin(rig)
        response = await admin.start_experiment(
            StartExperimentRequest(
                request=model_intent(kind=ProfilingSessionKind.OPERATOR)
            )
        )
        assert response.accepted is True

        await admin.shutdown()

        assert admin._runs == {}
        experiment = rig.store.get_experiment(response.experiment_id)
        assert experiment is not None
        assert experiment.state is not ExperimentState.CANCELLED
        # The next serve resumes and finishes the interrupted experiment.
        report = await rig.controller.run_experiment(response.experiment_id)
        assert report.state is ExperimentState.COMPLETED
