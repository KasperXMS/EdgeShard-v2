"""Container compute execution-plane lifecycle and attribution tests."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime
from pathlib import Path

import pytest

from edgeshard.cluster.capability import RuntimePlatformCapability
from edgeshard.cluster.identity import DeviceIdentity
from edgeshard.control.worker.compute_executor import (
    CONTAINER_COMPUTE_PORT,
    ComputeExecutionEnvironment,
    ContainerComputeProfilingExecutor,
)
from edgeshard.profiling.domain.experiment import (
    ModelCaseSpec,
    ProfilingCase,
    ProfilingErrorCategory,
)
from edgeshard.profiling.domain.measurement import (
    LatencyMetrics,
    MeasurementMetrics,
    MeasurementRecord,
    TimeUnit,
    summarize_samples,
)
from edgeshard.profiling.domain.session import ProfilingSessionKind, ProfilingSessionRequest
from edgeshard.profiling.domain.signature import (
    GemmSignature,
    OperatorKind,
    OperatorSignature,
    ProfilingGranularity,
)
from edgeshard.profiling.errors import ProfilingError
from edgeshard.runtime.model_store import ModelStore
from factories import make_rtx_capability

WORKER_ID = "worker-container"
GPU_0 = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
GPU_1 = "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
IMAGE = "edgeshard/hf-shard:cuda"
IMAGE_ID = "sha256:container-runtime-revision"


def gpu_capability():
    base = make_rtx_capability()
    template = base.devices[1]
    gpu_0 = dataclasses.replace(
        template,
        identity=DeviceIdentity(
            device_id=GPU_0,
            kind=template.identity.kind,
            local_locator="0000:01:00.0",
        ),
    )
    gpu_1 = dataclasses.replace(
        template,
        identity=DeviceIdentity(
            device_id=GPU_1,
            kind=template.identity.kind,
            local_locator="0000:02:00.0",
        ),
    )
    return dataclasses.replace(
        base,
        devices=(base.devices[0], gpu_0, gpu_1),
        runtime_platforms=(
            RuntimePlatformCapability(
                backend="edgeshard_shard", platform="cuda", image=IMAGE
            ),
        ),
    )


OPERATOR_SIGNATURE = OperatorSignature(
    kind=OperatorKind.GEMM,
    parameters=GemmSignature(m=8, n=8, k=8, dtype="fp32"),
    backend_family="torch",
)
SESSION_REQUEST = ProfilingSessionRequest(
    kind=ProfilingSessionKind.OPERATOR,
    device_ids=(GPU_1,),
)
CASE = ProfilingCase.for_spec(
    WORKER_ID,
    ModelCaseSpec(
        granularity=ProfilingGranularity.OPERATOR,
        device_ids=(GPU_1,),
        dtype="fp32",
        operator_signature=OPERATOR_SIGNATURE,
    ),
)


def measurement() -> MeasurementRecord:
    now = datetime(2026, 9, 9, tzinfo=UTC)
    return MeasurementRecord(
        measurement_id="container-measurement",
        case_id=CASE.case_id,
        environment_fingerprint="pending",
        started_at=now,
        finished_at=now,
        sample_count=1,
        samples=(1.0,),
        metrics=MeasurementMetrics(
            latency=LatencyMetrics(
                summary=summarize_samples((1.0,)), unit=TimeUnit.MILLISECONDS
            )
        ),
    )


class FakeImage:
    id = IMAGE_ID


class FakeImages:
    def __init__(self) -> None:
        self.requests: list[str] = []

    def get(self, image: str) -> FakeImage:
        self.requests.append(image)
        return FakeImage()


class FakeContainer:
    def __init__(self) -> None:
        self.id = "container-1"
        self.attrs = {
            "NetworkSettings": {
                "Ports": {
                    f"{CONTAINER_COMPUTE_PORT}/tcp": [
                        {"HostIp": "127.0.0.1", "HostPort": "49200"}
                    ]
                }
            }
        }
        self.reloaded = False
        self.stop_timeout: int | None = None
        self.removed = False

    def reload(self) -> None:
        self.reloaded = True

    def stop(self, timeout: int) -> None:
        self.stop_timeout = timeout

    def remove(self) -> None:
        self.removed = True


class FakeContainers:
    def __init__(self, container: FakeContainer) -> None:
        self.container = container
        self.calls: list[tuple[str, dict[str, object]]] = []

    def run(self, image: str, **kwargs: object) -> FakeContainer:
        self.calls.append((image, kwargs))
        return self.container


class FakeDockerClient:
    def __init__(self) -> None:
        self.container = FakeContainer()
        self.containers = FakeContainers(self.container)
        self.images = FakeImages()


class FakeServiceClient:
    def __init__(
        self,
        environment: ComputeExecutionEnvironment | None = None,
        error: ProfilingError | None = None,
    ) -> None:
        self.environment = environment or ComputeExecutionEnvironment(
            torch_version="2.13.0+cu126",
            cuda_version="12.6",
            backend_revision=IMAGE_ID,
            target_device_id=GPU_1,
            execution_device="cuda:0",
        )
        self.error = error
        self.health_calls = 0
        self.prepare_calls: list[tuple[object, ...]] = []
        self.profile_calls: list[ProfilingCase] = []
        self.session_closed = False
        self.client_closed = False

    def health(self) -> None:
        self.health_calls += 1

    def prepare(self, *args: object):
        self.prepare_calls.append(args)
        if self.error is not None:
            raise self.error
        return None, self.environment

    def profile(self, case: ProfilingCase) -> MeasurementRecord:
        self.profile_calls.append(case)
        if self.error is not None:
            raise self.error
        return measurement()

    def close_session(self) -> None:
        self.session_closed = True

    def close(self) -> None:
        self.client_closed = True


def make_executor(
    tmp_path: Path,
    docker_client: FakeDockerClient,
    service_client: FakeServiceClient,
) -> ContainerComputeProfilingExecutor:
    return ContainerComputeProfilingExecutor(
        docker_client_factory=lambda: docker_client,
        model_store=ModelStore(tmp_path / "models"),
        ready_timeout_s=0.1,
        poll_interval_s=0.001,
        client_factory=lambda endpoint, timeout: service_client,
    )


def test_container_executor_lifecycle_and_exact_gpu_target(tmp_path: Path) -> None:
    docker_client = FakeDockerClient()
    service_client = FakeServiceClient()
    executor = make_executor(tmp_path, docker_client, service_client)

    session = executor.prepare_session(
        "session-1", WORKER_ID, SESSION_REQUEST, None, gpu_capability()
    )
    result = executor.profile_case(session, CASE)

    assert result == measurement()
    assert docker_client.images.requests == [IMAGE]
    (image, kwargs), = docker_client.containers.calls
    assert image == IMAGE_ID
    assert kwargs["command"] == [
        "_compute-profile-service",
        "--host",
        "0.0.0.0",
        "--port",
        str(CONTAINER_COMPUTE_PORT),
    ]
    assert kwargs["volumes"] == {
        str(tmp_path / "models"): {"bind": "/models", "mode": "ro"}
    }
    (device_request,) = kwargs["device_requests"]
    assert device_request["DeviceIDs"] == [GPU_1]
    assert device_request["Count"] == 0
    assert kwargs["environment"] == {
        "EDGESHARD_PROFILING_TARGET_DEVICE_ID": GPU_1,
        "EDGESHARD_PROFILING_EXECUTION_DEVICE": "cuda:0",
        "EDGESHARD_PROFILING_BACKEND_REVISION": IMAGE_ID,
    }
    assert service_client.health_calls == 1
    assert service_client.profile_calls == [CASE]
    assert session.environment.torch_version == "2.13.0+cu126"

    executor.close_session(session)
    executor.close_session(session)

    assert service_client.session_closed
    assert service_client.client_closed
    assert docker_client.container.stop_timeout == 10
    assert docker_client.container.removed


def test_environment_device_mismatch_fails_and_cleans_up(tmp_path: Path) -> None:
    docker_client = FakeDockerClient()
    service_client = FakeServiceClient(
        dataclasses.replace(
            FakeServiceClient().environment,
            target_device_id=GPU_0,
        )
    )
    executor = make_executor(tmp_path, docker_client, service_client)

    with pytest.raises(ProfilingError, match="different physical target"):
        executor.prepare_session(
            "session-1", WORKER_ID, SESSION_REQUEST, None, gpu_capability()
        )

    assert service_client.client_closed
    assert docker_client.container.stop_timeout == 10
    assert docker_client.container.removed


def test_prepare_failure_propagates_typed_error_and_cleans_up(tmp_path: Path) -> None:
    docker_client = FakeDockerClient()
    service_client = FakeServiceClient(
        error=ProfilingError(
            ProfilingErrorCategory.EXPORT_FAILED,
            "container export failed",
        )
    )
    executor = make_executor(tmp_path, docker_client, service_client)

    with pytest.raises(ProfilingError) as raised:
        executor.prepare_session(
            "session-1", WORKER_ID, SESSION_REQUEST, None, gpu_capability()
        )

    assert raised.value.category is ProfilingErrorCategory.EXPORT_FAILED
    assert str(raised.value) == "container export failed"
    assert service_client.client_closed
    assert docker_client.container.removed


def test_profile_failure_propagates_without_premature_session_cleanup(
    tmp_path: Path,
) -> None:
    docker_client = FakeDockerClient()
    service_client = FakeServiceClient()
    executor = make_executor(tmp_path, docker_client, service_client)
    session = executor.prepare_session(
        "session-1", WORKER_ID, SESSION_REQUEST, None, gpu_capability()
    )
    service_client.error = ProfilingError(
        ProfilingErrorCategory.BENCHMARK_FAILED,
        "container benchmark failed",
    )

    with pytest.raises(ProfilingError) as raised:
        executor.profile_case(session, CASE)

    assert raised.value.category is ProfilingErrorCategory.BENCHMARK_FAILED
    assert not docker_client.container.removed
    executor.close_session(session)
    assert docker_client.container.removed
