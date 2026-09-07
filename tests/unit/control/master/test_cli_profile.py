"""CLI `profile` command tests (Phase 2 spec §49).

The admin RPCs are stubbed in-process (monkeypatched ``ProfilingAdminClient``):
these tests pin the CLI's half of the contract — intent construction, exit
codes on rejection/transport failure, and json/yaml presentation — while the
Master-side expansion is covered by test_profiling_admin.py.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import grpc
import pytest
import yaml
from typer.testing import CliRunner

import edgeshard.cli as cli_module
from edgeshard.cli import app
from edgeshard.profiling.domain.experiment import (
    CaseState,
    CaseStatus,
    ExperimentState,
    ExperimentStatus,
    ProfilingExperiment,
)
from edgeshard.profiling.domain.network import NetworkPathClass, ProbeKind
from edgeshard.profiling.domain.session import ProfilingSessionKind
from edgeshard.profiling.domain.snapshot import ProfileSnapshot
from edgeshard.protocol.profiling import mapper

runner = CliRunner()

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)
MASTER = "127.0.0.1:51001"

EXPERIMENT = ProfilingExperiment.for_cases(
    strategy_id="default-v1", case_ids=["c-1"], created_at=NOW
)
STATUS = ExperimentStatus(
    experiment=EXPERIMENT,
    state=ExperimentState.COMPLETED,
    cases=(CaseStatus("c-1", "w-1", CaseState.COMPLETED),),
)
SNAPSHOT = ProfileSnapshot(
    snapshot_id="snapshot-1",
    created_at=NOW,
    model_characterizations=(),
    measurements=(),
    network_measurements=(),
)


@dataclass
class FakeAdminClient:
    """In-process stand-in for the gRPC admin client.

    ``responses`` maps method name → response object, or an Exception to
    raise; ``requests`` records (method, request) pairs for assertions.
    """

    endpoint: str

    responses: dict[str, Any] = field(default_factory=dict, init=False)
    requests: list[tuple[str, Any]] = field(default_factory=list, init=False)

    async def __aenter__(self) -> FakeAdminClient:
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    async def _dispatch(self, method: str, request: Any) -> Any:
        self.requests.append((method, request))
        response = self.responses[method]
        if isinstance(response, Exception):
            raise response
        return response

    async def start_experiment(self, request: Any) -> Any:
        return await self._dispatch("start", request)

    async def get_experiment(self, request: Any) -> Any:
        return await self._dispatch("get", request)

    async def cancel_experiment(self, request: Any) -> Any:
        return await self._dispatch("cancel", request)

    async def build_profile_snapshot(self, request: Any) -> Any:
        return await self._dispatch("snapshot", request)


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch):
    """Install the fake admin client; returns a fresh per-test holder."""

    class Holder:
        endpoint: ClassVar[str | None] = None
        responses: ClassVar[dict[str, Any]] = {}
        requests: ClassVar[list[tuple[str, Any]]] = []

    def factory(endpoint: str) -> Any:
        client = FakeAdminClient(endpoint)
        client.responses = Holder.responses
        client.requests = Holder.requests
        Holder.endpoint = endpoint
        return client

    monkeypatch.setattr(cli_module, "ProfilingAdminClient", factory)
    return Holder


def test_profile_help_lists_commands() -> None:
    result = runner.invoke(app, ["profile", "--help"])
    assert result.exit_code == 0
    for name in ("model", "operator", "network", "status", "cancel", "snapshot"):
        assert name in result.output


def test_profile_network_help_lists_probes() -> None:
    result = runner.invoke(app, ["profile", "network", "--help"])
    assert result.exit_code == 0
    assert "rtt" in result.output
    assert "bandwidth" in result.output


class TestStartCommands:
    def test_model_run_builds_intent_and_echoes_id(self, stub) -> None:
        stub.responses["start"] = mapper.StartExperimentResponse(
            accepted=True, experiment_id="exp-1"
        )

        result = runner.invoke(
            app,
            [
                "profile", "model", "run",
                "--master", MASTER,
                "--model", "tiny/llama",
                "--dtype", "fp32",
                "--target", "w-1@gpu-0",
                "--requested-by", "alice",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "experiment exp-1 started" in result.output
        assert stub.endpoint == MASTER
        (method, request), = stub.requests
        assert method == "start"
        intent = request.request
        assert intent.kind is ProfilingSessionKind.MODEL
        assert intent.model is not None
        assert intent.model.model_id == "tiny/llama"
        assert intent.dtype == "fp32"
        assert len(intent.worker_device_targets) == 1
        assert intent.worker_device_targets[0].worker_id == "w-1"
        assert intent.worker_device_targets[0].device_id == "gpu-0"
        assert intent.missing_only is True
        assert intent.requested_by == "alice"

    def test_operator_run_include_measured_flag(self, stub) -> None:
        stub.responses["start"] = mapper.StartExperimentResponse(
            accepted=True, experiment_id="exp-2"
        )

        result = runner.invoke(
            app,
            [
                "profile", "operator", "run",
                "--master", MASTER,
                "--model", "tiny/llama",
                "--dtype", "bf16",
                "--target", "w-1@gpu-0",
                "--include-measured",
            ],
        )

        assert result.exit_code == 0, result.output
        intent = stub.requests[0][1].request
        assert intent.kind is ProfilingSessionKind.OPERATOR
        assert intent.missing_only is False

    def test_network_rtt_intent(self, stub) -> None:
        stub.responses["start"] = mapper.StartExperimentResponse(
            accepted=True, experiment_id="exp-3"
        )

        result = runner.invoke(
            app, ["profile", "network", "rtt", "--master", MASTER]
        )

        assert result.exit_code == 0, result.output
        intent = stub.requests[0][1].request
        assert intent.kind is ProfilingSessionKind.NETWORK
        assert intent.network_probe is ProbeKind.RTT
        assert intent.worker_ids == ()

    def test_network_rtt_explicit_interface_path(self, stub) -> None:
        stub.responses["start"] = mapper.StartExperimentResponse(
            accepted=True, experiment_id="exp-rtt-path"
        )
        result = runner.invoke(
            app,
            [
                "profile", "network", "rtt",
                "--master", MASTER,
                "--pair", "w-1@lan:w-2@zt",
            ],
        )
        assert result.exit_code == 0, result.output
        (pair,) = stub.requests[0][1].request.network_pairs
        assert pair.source_interface_id == "lan"
        assert pair.destination_interface_id == "zt"

    def test_network_bandwidth_knobs(self, stub) -> None:
        stub.responses["start"] = mapper.StartExperimentResponse(
            accepted=True, experiment_id="exp-4"
        )

        result = runner.invoke(
            app,
            [
                "profile", "network", "bandwidth",
                "--master", MASTER,
                "--path-class", "wired_lan",
                "--pair", "w-1@zt0:w-2@zt1",
            ],
        )

        assert result.exit_code == 0, result.output
        intent = stub.requests[0][1].request
        assert intent.network_probe is ProbeKind.BANDWIDTH
        assert intent.bandwidth_path_classes == (NetworkPathClass.WIRED_LAN,)
        (pair,) = intent.network_pairs
        assert pair.source_worker_id == "w-1"
        assert pair.destination_worker_id == "w-2"
        assert pair.source_interface_id == "zt0"
        assert pair.destination_interface_id == "zt1"

    def test_unknown_path_class_exits_nonzero(self, stub) -> None:
        result = runner.invoke(
            app,
            [
                "profile", "network", "bandwidth",
                "--master", MASTER,
                "--path-class", "carrier-pigeon",
            ],
        )
        assert result.exit_code == 1

    def test_malformed_pair_exits_nonzero(self, stub) -> None:
        result = runner.invoke(
            app,
            ["profile", "network", "bandwidth", "--master", MASTER, "--pair", "w-1"],
        )
        assert result.exit_code == 1

    def test_rejected_experiment_exits_nonzero(self, stub) -> None:
        stub.responses["start"] = mapper.StartExperimentResponse(
            accepted=False, detail="no registered worker hosts the profiling service"
        )

        result = runner.invoke(
            app, ["profile", "network", "rtt", "--master", MASTER]
        )

        assert result.exit_code == 1

    def test_invalid_intent_exits_nonzero(self, stub) -> None:
        # An empty --worker id violates the domain contract (§52.2).
        result = runner.invoke(
            app,
            [
                "profile", "network", "rtt",
                "--master", MASTER,
                "--worker", "",
            ],
        )
        assert result.exit_code == 1

    def test_transport_failure_exits_nonzero(self, stub) -> None:
        stub.responses["start"] = grpc.aio.AioRpcError(
            code=grpc.StatusCode.UNAVAILABLE,
            details="master down",
            debug_error_string="fake",
        )

        result = runner.invoke(
            app, ["profile", "network", "rtt", "--master", MASTER]
        )

        assert result.exit_code == 1


class TestReadCommands:
    def test_status_renders_json(self, stub) -> None:
        stub.responses["get"] = mapper.GetExperimentResponse(found=True, status=STATUS)

        result = runner.invoke(
            app,
            [
                "profile", "status",
                "--master", MASTER,
                "--experiment", EXPERIMENT.experiment_id,
            ],
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["__type__"]
        assert EXPERIMENT.experiment_id in result.output
        assert "c-1" in result.output
        (method, request), = stub.requests
        assert method == "get"
        assert request.experiment_id == EXPERIMENT.experiment_id

    def test_status_yaml_format(self, stub) -> None:
        stub.responses["get"] = mapper.GetExperimentResponse(found=True, status=STATUS)

        result = runner.invoke(
            app,
            [
                "profile", "status",
                "--master", MASTER,
                "--experiment", EXPERIMENT.experiment_id,
                "--format", "yaml",
            ],
        )

        assert result.exit_code == 0, result.output
        payload = yaml.safe_load(result.output)
        assert payload["state"] == ExperimentState.COMPLETED.value

    def test_status_unknown_experiment_exits_nonzero(self, stub) -> None:
        stub.responses["get"] = mapper.GetExperimentResponse(found=False)

        result = runner.invoke(
            app,
            ["profile", "status", "--master", MASTER, "--experiment", "nope"],
        )

        assert result.exit_code == 1

    def test_cancel_accepted_echoes_detail(self, stub) -> None:
        stub.responses["cancel"] = mapper.CancelExperimentResponse(
            accepted=True, detail=f"experiment {EXPERIMENT.experiment_id} is cancelled"
        )

        result = runner.invoke(
            app,
            [
                "profile", "cancel",
                "--master", MASTER,
                "--experiment", EXPERIMENT.experiment_id,
            ],
        )

        assert result.exit_code == 0, result.output
        assert "is cancelled" in result.output

    def test_cancel_rejected_exits_nonzero(self, stub) -> None:
        stub.responses["cancel"] = mapper.CancelExperimentResponse(
            accepted=False, detail="unknown experiment 'nope'"
        )

        result = runner.invoke(
            app, ["profile", "cancel", "--master", MASTER, "--experiment", "nope"]
        )

        assert result.exit_code == 1

    def test_snapshot_renders_yaml_by_default(self, stub) -> None:
        stub.responses["snapshot"] = mapper.BuildProfileSnapshotResponse(
            accepted=True, snapshot=SNAPSHOT
        )

        result = runner.invoke(app, ["profile", "snapshot", "--master", MASTER])

        assert result.exit_code == 0, result.output
        payload = yaml.safe_load(result.output)
        assert payload["snapshot_id"] == "snapshot-1"

    def test_snapshot_rejected_exits_nonzero(self, stub) -> None:
        stub.responses["snapshot"] = mapper.BuildProfileSnapshotResponse(
            accepted=False, detail="snapshot build failed"
        )

        result = runner.invoke(app, ["profile", "snapshot", "--master", MASTER])

        assert result.exit_code == 1


def write_worker_config(tmp_path: Path) -> Path:
    payload = {
        "worker": {"identity_path": str(tmp_path / "worker-id")},
        "model_store": {"root": str(tmp_path / "models")},
        "runtime": {"discover_managed_containers": False},
    }
    path = tmp_path / "worker.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


class TestLocalModelInspect:
    def test_without_torch_exits_with_install_hint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # torch is optional (inference extra): the lazy import must fail with
        # the install hint, not a traceback.
        monkeypatch.setitem(sys.modules, "torch", None)
        config = write_worker_config(tmp_path)

        result = runner.invoke(
            app,
            [
                "profile", "model", "inspect",
                "--model", "tiny/llama",
                "--config", str(config),
            ],
        )

        assert result.exit_code == 1

    def test_unresolvable_model_exits_nonzero(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        config = write_worker_config(tmp_path)

        result = runner.invoke(
            app,
            [
                "profile", "model", "inspect",
                "--model", "missing/model",
                "--config", str(config),
            ],
        )

        assert result.exit_code == 1


def test_runtime_serve_without_torch_exits_with_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The Phase 0 runtime legitimately needs torch; on a torch-free Master
    # box the command must exit with the install hint, not a traceback.
    monkeypatch.delitem(sys.modules, "edgeshard.runtime.config", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    config = tmp_path / "runtime.yaml"
    config.write_text("{}", encoding="utf-8")

    result = runner.invoke(app, ["runtime", "serve", "--config", str(config)])

    assert result.exit_code == 1
