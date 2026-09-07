"""End-to-end profiling chains over real localhost gRPC (P2G DoD, §51).

Everything between the operator intent and the persisted ProfileSnapshot
is the real stack: a real ``MasterService`` behind a real control-plane
server, real ``WorkerAgent`` registrations, a real ``WorkerProfilingRunner``
(real torch CPU loader, profilers, benchmark harness, leases, sessions)
behind a real ``WorkerProfilingService``, the real ``ProfilingController``
transport, a real SQLite store, and the real admin plane.

Only the *local facts* are fixed: each worker reports a deterministic
inspection (derived CPU device, loopback-only interface, tiny-llama READY)
so runs are reproducible, RTT probes stay on this host, and benchmarks run
on CPU — the §51 real-hardware tier (RTX 4090 / AGX Orin, iperf3 bandwidth)
stays on the hardware-validation checklist.

DoD scenarios covered here: the model chain and the network chain (§51),
operator reuse (§28), unsupported model, busy device, Master restart
(graceful shutdown *and* a crashed run whose Worker ledger replays,
§44/§50), duplicate results (§50), Worker restart with a network failure
and partial completion, stale sessions (§41), and mid-run cancellation.
The timeout DoD is unit-covered (fake transport with scripted deadlines):
a real deadline expiry is indistinguishable from a slow host here.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from edgeshard.cluster.capability import NetworkInterfaceCapability
from edgeshard.cluster.identity import DeviceKind
from edgeshard.control.master.profiling_admin import MasterProfilingAdmin
from edgeshard.control.master.profiling_controller import (
    ExperimentReport,
    ProfilingController,
    _plan_sessions,
)
from edgeshard.control.master.service import MasterService
from edgeshard.control.worker.agent import LocalInspection
from edgeshard.control.worker.config import (
    MasterConfig as WorkerMasterEndpoint,
)
from edgeshard.control.worker.config import (
    ModelStoreSection,
    ProfilingSection,
    ReconnectConfig,
    RuntimeSection,
    WorkerConfig,
    WorkerSection,
)
from edgeshard.control.worker.identity import derive_cpu_device_id
from edgeshard.control.worker.master_client import WorkerAgent
from edgeshard.control.worker.profiling_runner import WorkerProfilingRunner
from edgeshard.control.worker.profiling_sessions import (
    ProfilingSessionManager,
    RegistrationTokens,
)
from edgeshard.profiling.domain.experiment import (
    CaseState,
    ExperimentState,
    ProfilingErrorCategory,
    ProfilingRequest,
    WorkerDeviceTarget,
)
from edgeshard.profiling.domain.model import ModelReference
from edgeshard.profiling.domain.network import ProbeKind
from edgeshard.profiling.domain.session import (
    ProfilingSessionKind,
    ProfilingSessionRequest,
)
from edgeshard.profiling.store.sqlite import SqliteProfileStore
from edgeshard.protocol.control.grpc_server import start_control_server
from edgeshard.protocol.profiling import mapper
from edgeshard.protocol.profiling.grpc_client import WorkerProfilingClient
from edgeshard.protocol.profiling.grpc_server import start_profiling_server
from factories import (
    RTX_HOST_POOL_ID,
    finalize,
    make_rtx_capability,
    make_worker_identity,
    make_worker_state,
)

HOST = "127.0.0.1"
W1 = "w-int-1"
W2 = "w-int-2"
MODEL = ModelReference("tiny/llama", "local")
RUN_TIMEOUT_S = 300.0
"""Generous bound: a cold transformers import plus ~1 s per real benchmark."""


# ---------------------------------------------------------------------------
# Deterministic local facts
# ---------------------------------------------------------------------------


def make_inspection(worker_id: str) -> LocalInspection:
    """Fixed identity/capability/state for one integration worker.

    * The CPU device carries the ``derive_cpu_device_id`` id so the real
      ``resolve_torch_device`` maps it to ``torch.device("cpu")`` (§38).
    * The only IPv4-capable interface is loopback, so path selection is
      unambiguous and real pings never leave this host.
    * The state reports an *idle* device (the real lease floor is 5%
      utilization, §39) and the tiny-llama READY inventory entry the real
      ``resolve_model_source`` resolves against.
    """
    cpu_device_id = derive_cpu_device_id(worker_id)
    identity = dataclasses.replace(
        make_worker_identity(worker_id), hostname=f"host-{worker_id}"
    )
    capability = make_rtx_capability()
    capability = dataclasses.replace(
        capability,
        network_interfaces=(
            NetworkInterfaceCapability(
                interface_id="lo-0",
                name="lo",
                addresses=("127.0.0.1",),
                mtu=65_536,
            ),
        ),
        devices=tuple(
            dataclasses.replace(
                device,
                identity=dataclasses.replace(
                    device.identity, device_id=cpu_device_id
                ),
            )
            if device.identity.kind is DeviceKind.CPU
            else device
            for device in capability.devices
        ),
    )
    capability = finalize(capability)
    state = make_worker_state(
        worker_id, device_ids=(cpu_device_id,), pool_ids=(RTX_HOST_POOL_ID,)
    )
    state = dataclasses.replace(
        state,
        device_states=tuple(
            dataclasses.replace(device, utilization=1.0)
            for device in state.device_states
        ),
    )
    return LocalInspection(identity=identity, capability=capability, state=state)


class FixedInspector:
    """Serves both loops: the Agent's Inspector and the runner's StateInspector."""

    def __init__(self, worker_id: str) -> None:
        self.worker_id = worker_id
        self.cpu_device_id = derive_cpu_device_id(worker_id)
        self.inspection = make_inspection(worker_id)
        self.inspect_calls = 0

    async def start(self) -> None:
        pass

    async def inspect(self) -> LocalInspection:
        self.inspect_calls += 1
        return self.inspection

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# The in-process cluster
# ---------------------------------------------------------------------------


class WorkerNode:
    """One ``worker serve --profiling`` process's worth of wiring, in-process.

    Replicates the §41 ordering of ``cli._worker_serve``: the profiling
    server binds first (port 0), the Agent advertises the bound endpoint at
    registration, and one shared inspector feeds both loops. Downstream of
    the wiring everything is the real measurement stack. ``stop``/``start``
    model a Worker restart: a new generation binds a new port and
    re-registers with fresh tokens.
    """

    def __init__(
        self,
        worker_id: str,
        control_endpoint: str,
        root: Path,
        tiny_llama_dir: Path,
    ) -> None:
        self.worker_id = worker_id
        self.cpu_device_id = derive_cpu_device_id(worker_id)
        self.inspector = FixedInspector(worker_id)
        self.model_store_root = root / f"models-{worker_id}"
        self.model_store_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(tiny_llama_dir, self.model_store_root / "tiny-llama")
        self._control_endpoint = control_endpoint
        self._root = root
        self.endpoint: str | None = None
        self.agent: WorkerAgent | None = None
        self.runner: WorkerProfilingRunner | None = None
        self._server: object | None = None
        self._task: asyncio.Task[None] | None = None

    def _config(self) -> WorkerConfig:
        return WorkerConfig(
            worker=WorkerSection(
                identity_path=self._root / f"identity-{self.worker_id}",
                master=WorkerMasterEndpoint(endpoint=self._control_endpoint),
                heartbeat_interval_s=99.0,
                reconnect=ReconnectConfig(initial_delay_s=0.05, max_delay_s=0.2),
            ),
            model_store=ModelStoreSection(root=self.model_store_root),
            runtime=RuntimeSection(discover_managed_containers=False),
            profiling=ProfilingSection(enabled=True, host=HOST, port=0),
        )

    async def start(self) -> None:
        assert self.agent is None, "node is already running"
        agent_holder: list[WorkerAgent] = []

        def current_tokens() -> RegistrationTokens | None:
            if not agent_holder:
                return None
            agent = agent_holder[0]
            worker_id = agent.worker_id
            registration_session_id = agent.registration_session_id
            if worker_id is None or registration_session_id is None:
                return None
            return RegistrationTokens(
                worker_id=worker_id,
                instance_id=agent.instance_id,
                registration_session_id=registration_session_id,
            )

        runner = WorkerProfilingRunner(
            sessions=ProfilingSessionManager(token_source=current_tokens),
            inspector=self.inspector,
            model_store_root=self.model_store_root,
        )
        server, port = await start_profiling_server(runner, host=HOST, port=0)
        endpoint = f"{HOST}:{port}"
        agent = WorkerAgent(
            self._config(), inspector=self.inspector, profiling_endpoint=endpoint
        )
        agent_holder.append(agent)
        self.agent = agent
        self.runner = runner
        self.endpoint = endpoint
        self._server = server
        self._task = asyncio.create_task(agent.run(), name=f"agent-{self.worker_id}")

    async def stop(self) -> None:
        """Kill this generation: no deregistration, exactly like a crash."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self.runner is not None:
            await self.runner.shutdown()
        if self._server is not None:
            await self._server.stop(grace=None)  # type: ignore[attr-defined]
        self.agent = None
        self.runner = None
        self.endpoint = None
        self._server = None
        self._task = None


class Cluster:
    """Real Master plane + real Worker nodes on localhost gRPC."""

    def __init__(self, tmp_path: Path, tiny_llama_dir: Path) -> None:
        self._tmp_path = tmp_path
        self._tiny_llama_dir = tiny_llama_dir
        self.store_path = tmp_path / "profile.sqlite"
        self.nodes: dict[str, WorkerNode] = {}
        self.service: MasterService | None = None
        self.store: SqliteProfileStore | None = None
        self.controller: ProfilingController | None = None
        self.admin: MasterProfilingAdmin | None = None
        self._control_server: object | None = None
        self.control_endpoint = ""

    async def start_master(self) -> None:
        # No `async with service`: without the liveness loop nothing
        # degrades or expires on its own — registration and heartbeats work,
        # and a stopped Worker keeps its (stale) record, exactly the state a
        # Master with a dead profiling peer dispatches against.
        self.service = MasterService()
        server, port = await start_control_server(self.service, host=HOST, port=0)
        self._control_server = server
        self.control_endpoint = f"{HOST}:{port}"
        self.store = SqliteProfileStore(self.store_path)
        self.controller = ProfilingController(service=self.service, store=self.store)
        self.admin = MasterProfilingAdmin(controller=self.controller)

    async def add_worker(self, worker_id: str) -> WorkerNode:
        assert self.service is not None and self.admin is not None
        node = WorkerNode(
            worker_id, self.control_endpoint, self._tmp_path, self._tiny_llama_dir
        )
        await node.start()
        self.nodes[worker_id] = node
        service = self.service
        await wait_for(
            lambda: service.registry.find(worker_id) is not None,
            message=f"registration of {worker_id}",
        )
        record = service.registry.find(worker_id)
        assert record is not None
        assert record.profiling_endpoint == node.endpoint
        await wait_for(
            lambda: node.agent is not None
            and node.agent.registration_session_id is not None,
            message=f"live tokens of {worker_id}",
        )
        return node

    async def restart_worker(self, worker_id: str) -> WorkerNode:
        node = self.nodes[worker_id]
        await node.stop()
        await node.start()
        assert self.service is not None
        await wait_for(
            lambda: node.agent is not None
            and node.agent.registration_session_id is not None,
            message=f"re-registration of {worker_id}",
        )
        return node

    async def stop(self) -> None:
        if self.admin is not None:
            await self.admin.shutdown()
        for node in self.nodes.values():
            if node.agent is not None:
                await node.stop()
        if self._control_server is not None:
            await self._control_server.stop(grace=None)  # type: ignore[attr-defined]


@pytest.fixture
async def cluster(tmp_path: Path, tiny_llama_dir: Path):
    rig = Cluster(tmp_path, tiny_llama_dir)
    await rig.start_master()
    try:
        yield rig
    finally:
        await rig.stop()


async def wait_for(
    predicate: Callable[[], bool],
    timeout_s: float = 60.0,
    message: str = "condition",
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise TimeoutError(f"timed out waiting for {message}")


# ---------------------------------------------------------------------------
# Intents and run helpers
# ---------------------------------------------------------------------------


def operator_intent(worker_id: str, cpu_device_id: str, **overrides: object) -> ProfilingRequest:
    base: dict = {
        "kind": ProfilingSessionKind.OPERATOR,
        "model": MODEL,
        "dtype": "fp32",
        "worker_device_targets": (
            WorkerDeviceTarget(worker_id, cpu_device_id),
        ),
        "requested_by": "integration",
    }
    base.update(overrides)
    return ProfilingRequest(**base)


def model_intent(worker_id: str, cpu_device_id: str, **overrides: object) -> ProfilingRequest:
    return operator_intent(
        worker_id, cpu_device_id, kind=ProfilingSessionKind.MODEL, **overrides
    )


def network_intent(worker_ids: tuple[str, ...] = (W1, W2)) -> ProfilingRequest:
    return ProfilingRequest(
        kind=ProfilingSessionKind.NETWORK,
        worker_ids=worker_ids,
        network_probe=ProbeKind.RTT,
        requested_by="integration",
    )


async def drain(admin: MasterProfilingAdmin, experiment_id: str) -> ExperimentReport:
    """Await the background run launched by StartExperiment (§49)."""
    task = admin._runs[experiment_id]
    report = await asyncio.wait_for(task, timeout=RUN_TIMEOUT_S)
    await asyncio.sleep(0)  # flush the done-callback report cache
    return report


def stored_cases(cluster: Cluster, experiment_id: str):
    assert cluster.store is not None
    stored_experiment = cluster.store.get_experiment(experiment_id)
    assert stored_experiment is not None
    cases = []
    for case_id in stored_experiment.experiment.case_ids:
        stored = cluster.store.get_case(case_id)
        assert stored is not None
        cases.append(stored.case)
    return cases


# ---------------------------------------------------------------------------
# §51 chain 1: adapter → characterization → signatures → benchmark → store
# ---------------------------------------------------------------------------


class TestModelChain:
    async def test_operator_chain_over_real_grpc(self, cluster: Cluster) -> None:
        """The P2G DoD flow with nothing faked below the admin plane."""
        node = await cluster.add_worker(W1)
        assert cluster.admin is not None

        response = await cluster.admin.start_experiment(
            mapper.StartExperimentRequest(
                request=operator_intent(W1, node.cpu_device_id)
            )
        )

        assert response.accepted is True, response.detail
        report = await drain(cluster.admin, response.experiment_id)
        assert report.state is ExperimentState.COMPLETED
        assert report.cases
        assert all(case.state is CaseState.COMPLETED for case in report.cases)
        # The inspection really loaded tiny-llama on the Worker (§38): the
        # characterization in the store is the checkpoint's, not a fixture's.
        assert cluster.store is not None
        snapshot = cluster.store.build_snapshot("s-model-chain")
        (characterization,) = snapshot.model_characterizations
        assert characterization.model.model_id == "tiny/llama"
        assert characterization.num_layers == 4
        assert characterization.hidden_size == 64
        # Real CPU benchmarks produced real latency observations (§51).
        assert snapshot.measurements
        for record in snapshot.measurements:
            assert record.metrics.latency is not None
            assert record.metrics.latency.summary.mean >= 0.0
            assert record.metrics.latency.summary.median >= 0.0

    async def test_incremental_reuse_over_real_wire(self, cluster: Cluster) -> None:
        """§28: measured operator signatures are reused, never re-benchmarked."""
        node = await cluster.add_worker(W1)
        assert cluster.admin is not None
        intent = operator_intent(W1, node.cpu_device_id)

        first = await cluster.admin.start_experiment(
            mapper.StartExperimentRequest(request=intent)
        )
        assert first.accepted is True, first.detail
        report = await drain(cluster.admin, first.experiment_id)
        assert report.state is ExperimentState.COMPLETED
        measured = len(report.cases)
        assert measured > 0

        second = await cluster.admin.start_experiment(
            mapper.StartExperimentRequest(request=intent)
        )
        assert second.accepted is False
        assert "zero cases" in second.detail
        assert cluster.store is not None
        before_rerun = len(cluster.store.query_measurements())

        third = await cluster.admin.start_experiment(
            mapper.StartExperimentRequest(
                request=operator_intent(W1, node.cpu_device_id, missing_only=False)
            )
        )
        assert third.accepted is True, third.detail
        assert third.experiment_id != first.experiment_id
        report3 = await drain(cluster.admin, third.experiment_id)
        assert report3.state is ExperimentState.COMPLETED
        assert len(report3.cases) == measured
        first_stored = cluster.store.get_experiment(first.experiment_id)
        third_stored = cluster.store.get_experiment(third.experiment_id)
        assert first_stored is not None and third_stored is not None
        assert (
            first_stored.experiment.configuration_id
            == third_stored.experiment.configuration_id
        )
        assert len(cluster.store.query_measurements()) == before_rerun + measured


# ---------------------------------------------------------------------------
# §51 chain 2: network plan → real ping runner → measurement → store
# ---------------------------------------------------------------------------


class TestNetworkChain:
    async def test_rtt_matrix_with_real_ping(self, cluster: Cluster) -> None:
        await cluster.add_worker(W1)
        await cluster.add_worker(W2)
        assert cluster.admin is not None and cluster.store is not None

        response = await cluster.admin.start_experiment(
            mapper.StartExperimentRequest(request=network_intent())
        )

        assert response.accepted is True, response.detail
        report = await drain(cluster.admin, response.experiment_id)
        assert report.state is ExperimentState.COMPLETED
        # Dense directed RTT matrix over two workers (§32): A→B and B→A,
        # each executed by its source worker with the real platform ping
        # against the deterministic loopback target.
        assert len(report.cases) == 2
        cases = stored_cases(cluster, response.experiment_id)
        assert {(c.spec.source_worker_id, c.spec.destination_worker_id) for c in cases} == {
            (W1, W2),
            (W2, W1),
        }
        snapshot = cluster.store.build_snapshot("s-network-chain")
        assert len(snapshot.network_measurements) == 2
        for record in snapshot.network_measurements:
            rtt = record.metrics.rtt
            assert rtt is not None
            assert rtt.packets_sent > 0
            # Loopback never drops packets; a loss here would mean the
            # probe escaped the host or the parser broke.
            assert rtt.packets_received == rtt.packets_sent


# ---------------------------------------------------------------------------
# Rejections over the real wire (§39, §42, §52.2)
# ---------------------------------------------------------------------------


class TestRejections:
    async def test_unsupported_model_rejected_over_wire(self, cluster: Cluster) -> None:
        node = await cluster.add_worker(W1)
        assert cluster.admin is not None and cluster.store is not None

        response = await cluster.admin.start_experiment(
            mapper.StartExperimentRequest(
                request=operator_intent(
                    W1, node.cpu_device_id, model=ModelReference("missing/model", "local")
                )
            )
        )

        assert response.accepted is False
        assert response.experiment_id == ""
        assert "unsupported_model" in response.detail
        # The Worker's typed failure crossed the wire and nothing was stored.
        snapshot = cluster.store.build_snapshot("s-unsupported")
        assert snapshot.model_characterizations == ()
        assert snapshot.measurements == ()

    async def test_busy_device_rejected_then_accepted(self, cluster: Cluster) -> None:
        """DoD busy-device: a held lease refuses the run, release unblocks it."""
        node = await cluster.add_worker(W1)
        assert cluster.admin is not None and cluster.service is not None
        assert node.runner is not None and node.endpoint is not None
        info = cluster.service.sessions.current(W1)
        assert info is not None

        async with WorkerProfilingClient(node.endpoint) as raw:
            prepared = await raw.prepare_profiling_session(
                mapper.PrepareProfilingSessionRequest(
                    worker_id=info.worker_id,
                    instance_id=info.instance_id,
                    registration_session_id=info.session_id,
                    profiling_session_id="manual-lease",
                    session_request=ProfilingSessionRequest(
                        kind=ProfilingSessionKind.OPERATOR,
                        device_ids=(node.cpu_device_id,),
                    ),
                )
            )
            assert prepared.accepted is True, prepared.detail
            assert node.runner.leases.leased_device_ids == (node.cpu_device_id,)

            rejected = await cluster.admin.start_experiment(
                mapper.StartExperimentRequest(
                    request=operator_intent(W1, node.cpu_device_id)
                )
            )
            assert rejected.accepted is False
            assert "device_busy" in rejected.detail

            closed = await raw.close_profiling_session(
                mapper.CloseProfilingSessionRequest(
                    worker_id=info.worker_id,
                    instance_id=info.instance_id,
                    registration_session_id=info.session_id,
                    profiling_session_id="manual-lease",
                )
            )
            assert closed.accepted is True
        assert node.runner.leases.leased_device_ids == ()

        accepted = await cluster.admin.start_experiment(
            mapper.StartExperimentRequest(request=operator_intent(W1, node.cpu_device_id))
        )
        assert accepted.accepted is True, accepted.detail
        report = await drain(cluster.admin, accepted.experiment_id)
        assert report.state is ExperimentState.COMPLETED


# ---------------------------------------------------------------------------
# Master restart (§50) — graceful shutdown and crashed-run ledger replay (§44)
# ---------------------------------------------------------------------------


class TestMasterRestart:
    async def test_graceful_shutdown_resumes_from_store(self, cluster: Cluster) -> None:
        node = await cluster.add_worker(W1)
        assert cluster.admin is not None and cluster.store is not None
        response = await cluster.admin.start_experiment(
            mapper.StartExperimentRequest(request=operator_intent(W1, node.cpu_device_id))
        )
        assert response.accepted is True, response.detail

        # Shutdown lands before the run task takes its first step (nothing
        # awaits between StartExperiment's create_task and this call), so the
        # stored experiment saw zero dispatches — a Master that died right
        # after accepting the request. The experiment survives (§50).
        await cluster.admin.shutdown()
        stored = cluster.store.get_experiment(response.experiment_id)
        assert stored is not None
        assert stored.state in (ExperimentState.PENDING, ExperimentState.RUNNING)

        # The restarted serve: fresh controller and store on the same files.
        store2 = SqliteProfileStore(cluster.store_path)
        controller2 = ProfilingController(service=cluster.service, store=store2)
        report = await asyncio.wait_for(
            controller2.run_experiment(response.experiment_id), timeout=RUN_TIMEOUT_S
        )
        assert report.state is ExperimentState.COMPLETED

        # DoD duplicate-result: re-running a terminal experiment replays the
        # store without dispatching — the measurement set is stable.
        before = len(store2.build_snapshot("s-resume-a").measurements)
        again = await controller2.run_experiment(response.experiment_id)
        assert again.state is ExperimentState.COMPLETED
        assert len(store2.build_snapshot("s-resume-b").measurements) == before
        assert node.runner is not None
        assert node.runner.leases.leased_device_ids == ()
        assert node.runner.sessions.open_session_count() == 0

    async def test_crashed_run_replays_worker_ledger(self, cluster: Cluster) -> None:
        """§44/§50: the Worker's ledger answers a re-dispatch, never re-runs."""
        node = await cluster.add_worker(W1)
        assert cluster.admin is not None and cluster.store is not None
        assert node.runner is not None and node.endpoint is not None
        response = await cluster.admin.start_experiment(
            mapper.StartExperimentRequest(request=operator_intent(W1, node.cpu_device_id))
        )
        assert response.accepted is True, response.detail
        await cluster.admin.shutdown()

        experiment_id = response.experiment_id
        cases = stored_cases(cluster, experiment_id)
        assert cases
        # Emulate the *crashed* first serve: prepare the canonical session,
        # decide one case on the Worker's ledger, then vanish without close —
        # exactly the Worker-side state a killed Master process leaves.
        (plan,) = _plan_sessions(experiment_id, W1, cases)
        info = cluster.service.sessions.current(W1)
        assert info is not None
        first = cases[0]
        async with WorkerProfilingClient(node.endpoint) as raw:
            prepared = await raw.prepare_profiling_session(
                mapper.PrepareProfilingSessionRequest(
                    worker_id=info.worker_id,
                    instance_id=info.instance_id,
                    registration_session_id=info.session_id,
                    profiling_session_id=plan.session_id,
                    session_request=plan.session_request,
                )
            )
            assert prepared.accepted is True, prepared.detail
            ran = await raw.run_profiling_case(
                mapper.RunProfilingCaseRequest(
                    worker_id=info.worker_id,
                    instance_id=info.instance_id,
                    registration_session_id=info.session_id,
                    profiling_session_id=plan.session_id,
                    case=first,
                )
            )
            assert ran.accepted is True, ran.detail
            assert ran.outcome is not None and ran.outcome.record is not None
        entry = node.runner.sessions.case_entry(plan.session_id, first.case_id)
        assert entry is not None and entry.terminal
        decided_at = entry.decided_at
        # The crashed incarnation never persisted the result Master-side.
        stored = cluster.store.get_case(first.case_id)
        assert stored is not None
        assert stored.state is not CaseState.COMPLETED

        # The restarted serve resumes: the canonical session id re-derives,
        # the open session replays its prepare, and the decided case replays
        # its recorded outcome instead of benchmarking a second time.
        store2 = SqliteProfileStore(cluster.store_path)
        controller2 = ProfilingController(service=cluster.service, store=store2)
        report = await asyncio.wait_for(
            controller2.run_experiment(experiment_id), timeout=RUN_TIMEOUT_S
        )

        assert report.state is ExperimentState.COMPLETED
        replayed = node.runner.sessions.case_entry(plan.session_id, first.case_id)
        assert replayed is not None
        assert replayed.decided_at == decided_at  # never re-benchmarked (§44)
        snapshot = store2.build_snapshot("s-crash-resume")
        assert len(snapshot.measurements) == len(cases)
        assert node.runner.leases.leased_device_ids == ()
        assert node.runner.sessions.open_session_count() == 0


# ---------------------------------------------------------------------------
# Worker restart, network failure, partial completion, stale session (DoD)
# ---------------------------------------------------------------------------


class TestWorkerRestart:
    async def test_dead_worker_partial_completion_and_stale_session(
        self, cluster: Cluster
    ) -> None:
        node_a = await cluster.add_worker(W1)
        await cluster.add_worker(W2)
        assert cluster.admin is not None and cluster.service is not None
        node_b = cluster.nodes[W2]
        assert node_b.endpoint is not None
        stale_info = cluster.service.sessions.current(W2)
        assert stale_info is not None

        # Kill worker B: no deregistration, so the Master still plans with
        # its (now stale) registration and dispatches into a dead endpoint.
        await node_b.stop()

        response = await cluster.admin.start_experiment(
            mapper.StartExperimentRequest(request=network_intent())
        )
        assert response.accepted is True, response.detail
        report = await drain(cluster.admin, response.experiment_id)

        # DoD partial completion: A→B executes on the living worker A (its
        # deterministic target is loopback, so the ping succeeds even with B
        # down); B→A is lost at the dead endpoint and fails typed.
        assert report.state is ExperimentState.PARTIALLY_COMPLETED
        by_worker = {case.worker_id: case.state for case in report.cases}
        assert by_worker[W1] is CaseState.COMPLETED
        assert by_worker[W2] is CaseState.FAILED
        status_response = await cluster.admin.get_experiment(
            mapper.GetExperimentRequest(experiment_id=response.experiment_id)
        )
        assert status_response.status is not None
        (failed,) = [c for c in status_response.status.cases if c.state is CaseState.FAILED]
        assert failed.failure is not None
        assert failed.failure.category is ProfilingErrorCategory.NETWORK_UNREACHABLE
        assert cluster.store is not None
        snapshot = cluster.store.build_snapshot("s-partial")
        assert len(snapshot.network_measurements) == 1  # only A→B survived

        # DoD stale session: the restarted Worker refuses the dead
        # registration's tokens on every RPC before touching state (§41).
        node_b2 = await cluster.restart_worker(W2)
        assert node_b2.endpoint is not None and node_b2.runner is not None
        assert cluster.controller is not None
        facts = cluster.controller.cluster_network_facts()
        async with WorkerProfilingClient(node_b2.endpoint) as raw:
            refused = await raw.prepare_profiling_session(
                mapper.PrepareProfilingSessionRequest(
                    worker_id=stale_info.worker_id,
                    instance_id=stale_info.instance_id,
                    registration_session_id=stale_info.session_id,
                    profiling_session_id="stale-probe",
                    session_request=ProfilingSessionRequest(
                        kind=ProfilingSessionKind.NETWORK
                    ),
                    network_facts=(facts[W2],),
                )
            )
        assert refused.accepted is False
        assert refused.reason is mapper.ProfilingRejection.STALE_SESSION

        # DoD resume: the FAILED case is terminal history (§44) — resuming
        # after the Worker restart replays the report without dispatching
        # anything to the fresh generation.
        resumed = await cluster.controller.run_experiment(response.experiment_id)
        assert resumed.state is ExperimentState.PARTIALLY_COMPLETED
        assert node_b2.runner.sessions.session_ids == ()
        assert node_a.runner is not None
        assert node_a.runner.leases.leased_device_ids == ()


# ---------------------------------------------------------------------------
# Cancellation mid-run (DoD)
# ---------------------------------------------------------------------------


class TestCancellation:
    async def test_cancel_mid_run_releases_worker_state(self, cluster: Cluster) -> None:
        node = await cluster.add_worker(W1)
        assert cluster.admin is not None and cluster.store is not None
        assert node.runner is not None
        response = await cluster.admin.start_experiment(
            mapper.StartExperimentRequest(request=model_intent(W1, node.cpu_device_id))
        )
        assert response.accepted is True, response.detail
        experiment_id = response.experiment_id
        # The full MODEL family (operator + module + layer cases) runs long
        # enough to cancel deterministically after the first completion.
        assert len(stored_cases(cluster, experiment_id)) > 3

        deadline = time.monotonic() + RUN_TIMEOUT_S
        while True:
            status = await cluster.admin.get_experiment(
                mapper.GetExperimentRequest(experiment_id=experiment_id)
            )
            assert status.status is not None
            completed = [c for c in status.status.cases if c.state is CaseState.COMPLETED]
            if completed:
                break
            assert time.monotonic() < deadline, "no case completed in time"
            await asyncio.sleep(0.1)

        cancel = await cluster.admin.cancel_experiment(
            mapper.CancelExperimentRequest(experiment_id=experiment_id)
        )

        assert cancel.accepted is True
        assert experiment_id not in cluster.admin._runs
        stored = cluster.store.get_experiment(experiment_id)
        assert stored is not None
        assert stored.state in (ExperimentState.CANCELLED, ExperimentState.PARTIALLY_COMPLETED)
        # A second cancellation of the now-terminal experiment replays the
        # stored report without touching anything (§44).
        assert cluster.controller is not None
        report = await cluster.controller.cancel_experiment(experiment_id)
        # History is never rewritten (§44): completed cases keep their state,
        # the rest are cancelled Master-side even though the Worker session
        # was already closed by the interrupted run.
        states = [case.state for case in report.cases]
        assert CaseState.COMPLETED in states
        assert CaseState.CANCELLED in states
        # §39: no reservation and no session outlives the cancellation.
        assert node.runner.leases.leased_device_ids == ()
        assert node.runner.sessions.open_session_count() == 0
