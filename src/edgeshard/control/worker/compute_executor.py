"""Compute execution planes for Worker-side Phase 2 profiling.

Production GPU work runs in the operator-declared ``edgeshard_shard``
runtime image.  The host implementation remains an injectable CPU testing
seam and is also used *inside* that container by the small session service.
Physical telemetry is deliberately absent here: the Worker wraps executor
calls with its long-lived Phase 1 NVML/tegrastats instruments.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Protocol

import docker
import httpx
import torch

from edgeshard.cluster.capability import RuntimePlatformCapability, WorkerCapability
from edgeshard.cluster.identity import DeviceKind
from edgeshard.control.worker.identity import derive_jetson_gpu_device_id
from edgeshard.control.worker.profiling_model_loader import (
    LoadedModelSession,
    ModelSessionLoader,
    TorchModelSessionLoader,
)
from edgeshard.model.source import ModelSource
from edgeshard.profiling.benchmark.harness import InstrumentationBundle
from edgeshard.profiling.codec import decode_payload, encode_payload
from edgeshard.profiling.domain.experiment import (
    ModelCaseSpec,
    ProfilingCase,
    ProfilingErrorCategory,
    ProfilingFailure,
)
from edgeshard.profiling.domain.measurement import MeasurementRecord
from edgeshard.profiling.domain.session import (
    ModelSessionFacts,
    ProfilingSessionKind,
    ProfilingSessionRequest,
)
from edgeshard.profiling.domain.signature import ProfilingGranularity
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.instrumentation.memory import CudaAllocatorMemoryProbe
from edgeshard.profiling.instrumentation.timing import CudaEventTimer, WallClockTimer
from edgeshard.profiling.model.adapters.base import module_device
from edgeshard.profiling.model.layer_profiler import TransformerLayerProfiler
from edgeshard.profiling.model.module_profiler import ModuleProfiler
from edgeshard.profiling.operator.profiler import OperatorProfiler
from edgeshard.runtime.model_store import MODEL_MOUNT, ModelStore

logger = logging.getLogger("worker.profiling.compute_executor")

CONTAINER_COMPUTE_PORT = 51_101
_PENDING_FINGERPRINT = "compute-executor-pending"
_TARGET_DEVICE_ENV = "EDGESHARD_PROFILING_TARGET_DEVICE_ID"
_EXECUTION_DEVICE_ENV = "EDGESHARD_PROFILING_EXECUTION_DEVICE"
_BACKEND_REVISION_ENV = "EDGESHARD_PROFILING_BACKEND_REVISION"


@dataclass(frozen=True)
class ComputeExecutionEnvironment:
    """Software identity observed by the process that ran the benchmark."""

    torch_version: str
    cuda_version: str | None
    backend_revision: str
    target_device_id: str
    execution_device: str

    def __post_init__(self) -> None:
        if not self.torch_version:
            raise ValueError("torch_version must not be empty")
        if self.cuda_version is not None and not self.cuda_version:
            raise ValueError("cuda_version must not be empty when present")
        if not self.backend_revision:
            raise ValueError("backend_revision must not be empty")
        if not self.target_device_id:
            raise ValueError("target_device_id must not be empty")
        if not self.execution_device:
            raise ValueError("execution_device must not be empty")


@dataclass
class PreparedComputeSession:
    """Executor-owned live session plus facts safe to return to the Master."""

    session_id: str
    target_device_id: str
    environment: ComputeExecutionEnvironment
    facts: ModelSessionFacts | None = None
    opaque: object | None = None
    closed: bool = False


class ComputeProfilingExecutor(Protocol):
    """Execution boundary for all MODEL/OPERATOR preparation and cases."""

    def prepare_session(
        self,
        session_id: str,
        worker_id: str,
        request: ProfilingSessionRequest,
        source: ModelSource | None,
        capability: WorkerCapability,
    ) -> PreparedComputeSession: ...

    def profile_case(
        self, session: PreparedComputeSession, case: ProfilingCase
    ) -> MeasurementRecord: ...

    def close_session(self, session: PreparedComputeSession) -> None: ...


DeviceResolver = Callable[[str, str], torch.device]
InstrumentationFactory = Callable[[torch.device], InstrumentationBundle]
ModelLoaderFactory = Callable[[torch.device], ModelSessionLoader]
EnvironmentFactory = Callable[
    [torch.device, str, str], ComputeExecutionEnvironment
]


def _default_compute_instrumentation(device: torch.device) -> InstrumentationBundle:
    """Container/CPU instruments only; physical probes belong to the Worker."""
    if device.type == "cuda":
        return InstrumentationBundle(
            timer=CudaEventTimer(device.index),
            memory=CudaAllocatorMemoryProbe(device.index),
        )
    return InstrumentationBundle(timer=WallClockTimer())


class HostTorchComputeProfilingExecutor:
    """Existing torch execution path, retained for CPU tests and container use."""

    def __init__(
        self,
        *,
        device_resolver: DeviceResolver,
        model_loader_factory: ModelLoaderFactory | None = None,
        layer_profiler: TransformerLayerProfiler | None = None,
        module_profiler: ModuleProfiler | None = None,
        operator_profiler: OperatorProfiler | None = None,
        instrumentation_factory: InstrumentationFactory | None = None,
        backend_revision: str | None = None,
        environment_factory: EnvironmentFactory | None = None,
        seed: int | None = None,
    ) -> None:
        self._device_resolver = device_resolver
        self._model_loader_factory = model_loader_factory or (
            lambda device: TorchModelSessionLoader(device=device)
        )
        self._layer_profiler = layer_profiler or TransformerLayerProfiler()
        self._module_profiler = module_profiler or ModuleProfiler()
        self._operator_profiler = operator_profiler or OperatorProfiler()
        self._instrumentation_factory = (
            instrumentation_factory or _default_compute_instrumentation
        )
        self._backend_revision = backend_revision or importlib.metadata.version(
            "edgeshard"
        )
        self._environment_factory = environment_factory
        self._seed = seed

    def set_profilers(
        self,
        *,
        layer: TransformerLayerProfiler,
        module: ModuleProfiler,
        operator: OperatorProfiler,
    ) -> None:
        """Refresh injectable CPU-test doubles owned by the Worker runner."""
        self._layer_profiler = layer
        self._module_profiler = module
        self._operator_profiler = operator

    def prepare_session(
        self,
        session_id: str,
        worker_id: str,
        request: ProfilingSessionRequest,
        source: ModelSource | None,
        capability: WorkerCapability,
    ) -> PreparedComputeSession:
        del capability
        if request.kind is ProfilingSessionKind.NETWORK:
            raise ValueError("network sessions do not use a compute executor")
        target_device_id = request.device_ids[0]
        device = self._device_resolver(target_device_id, worker_id)
        loaded: LoadedModelSession | None = None
        if request.kind is ProfilingSessionKind.MODEL:
            if source is None:
                raise ValueError("model compute sessions require a model source")
            loader: ModelSessionLoader = self._model_loader_factory(device)
            loaded = loader.load(request, source)
        environment = (
            self._environment_factory(device, target_device_id, self._backend_revision)
            if self._environment_factory is not None
            else ComputeExecutionEnvironment(
                torch_version=str(torch.__version__),
                cuda_version=(
                    str(torch.version.cuda) if torch.version.cuda is not None else None
                ),
                backend_revision=self._backend_revision,
                target_device_id=target_device_id,
                execution_device=str(device),
            )
        )
        return PreparedComputeSession(
            session_id=session_id,
            target_device_id=target_device_id,
            environment=environment,
            facts=loaded.facts if loaded is not None else None,
            opaque=loaded,
        )

    def profile_case(
        self, session: PreparedComputeSession, case: ProfilingCase
    ) -> MeasurementRecord:
        if session.closed:
            raise ProfilingError(
                ProfilingErrorCategory.INTERNAL_ERROR,
                f"compute session {session.session_id!r} is closed",
            )
        spec = case.spec
        if not isinstance(spec, ModelCaseSpec):
            raise ValueError("compute executors accept only model/operator cases")
        if not spec.device_ids or spec.device_ids[0] != session.target_device_id:
            raise ProfilingError(
                ProfilingErrorCategory.INTERNAL_ERROR,
                f"case {case.case_id!r} targets a different device than its "
                f"compute session ({session.target_device_id!r})",
            )
        device = self._device_resolver(session.target_device_id, case.worker_id)
        instrumentation = self._instrumentation_factory(device)
        if spec.granularity is ProfilingGranularity.OPERATOR:
            return self._operator_profiler.profile(
                case,
                instrumentation=instrumentation,
                environment_fingerprint=_PENDING_FINGERPRINT,
                device=device,
            )

        loaded = session.opaque
        if not isinstance(loaded, LoadedModelSession):
            raise ProfilingError(
                ProfilingErrorCategory.INTERNAL_ERROR,
                f"model session {session.session_id!r} carries no loaded checkpoint",
            )
        facts = loaded.facts
        if spec.granularity is ProfilingGranularity.TRANSFORMER_LAYER:
            if spec.layer_index is None:
                raise ProfilingError(
                    ProfilingErrorCategory.UNSUPPORTED_GRANULARITY,
                    "transformer-layer cases require layer_index",
                    {"case_id": case.case_id},
                )
            layer = loaded.layer_at(spec.layer_index)
            declared = next(
                (
                    entry.signature
                    for entry in facts.layer_entries
                    if entry.index == layer.index
                ),
                None,
            )
            if declared is not None and spec.layer_signature != declared:
                raise ValueError(
                    f"case {case.case_id!r} contradicts its session layer signature"
                )
            self._require_module_device(layer.layer, device, case.case_id)
            return self._layer_profiler.profile(
                case,
                loaded.model,
                layer,
                loaded.layout,
                loaded.adapter,
                instrumentation=instrumentation,
                environment_fingerprint=_PENDING_FINGERPRINT,
                seed=self._seed,
            )

        assert spec.module_signature is not None
        module = loaded.module_for(spec.module_signature)
        self._require_module_device(module.module, device, case.case_id)
        return self._module_profiler.profile(
            case,
            loaded.model,
            module,
            loaded.layout,
            loaded.adapter,
            instrumentation=instrumentation,
            environment_fingerprint=_PENDING_FINGERPRINT,
            seed=self._seed,
        )

    @staticmethod
    def _require_module_device(
        module: torch.nn.Module, expected: torch.device, case_id: str
    ) -> None:
        actual = module_device(module)
        if actual != expected:
            raise ProfilingError(
                ProfilingErrorCategory.INTERNAL_ERROR,
                f"case {case_id!r} expects {expected} but loaded module is on {actual}",
                {
                    "case_id": case_id,
                    "expected_torch_device": str(expected),
                    "actual_torch_device": str(actual),
                },
            )

    def close_session(self, session: PreparedComputeSession) -> None:
        if session.closed:
            return
        session.closed = True
        loaded = session.opaque
        session.opaque = None
        if isinstance(loaded, LoadedModelSession):
            loaded.close()


class ComputeServiceClient(Protocol):
    """Client seam used by the Docker executor (fakeable without a daemon)."""

    def health(self) -> None: ...

    def prepare(
        self,
        session_id: str,
        worker_id: str,
        request: ProfilingSessionRequest,
        source: ModelSource | None,
    ) -> tuple[ModelSessionFacts | None, ComputeExecutionEnvironment]: ...

    def profile(self, case: ProfilingCase) -> MeasurementRecord: ...

    def close_session(self) -> None: ...

    def close(self) -> None: ...


class _HttpComputeServiceClient:
    def __init__(self, endpoint: str, timeout_s: float) -> None:
        self._client = httpx.Client(base_url=f"http://{endpoint}", timeout=timeout_s)

    def health(self) -> None:
        response = self._client.get("/health")
        response.raise_for_status()

    def prepare(
        self,
        session_id: str,
        worker_id: str,
        request: ProfilingSessionRequest,
        source: ModelSource | None,
    ) -> tuple[ModelSessionFacts | None, ComputeExecutionEnvironment]:
        source_payload = (
            {
                "path": str(source.path),
                "model_id": source.model_id,
                "revision": source.revision,
            }
            if source is not None
            else None
        )
        payload = self._call(
            "/prepare",
            {
                "session_id": session_id,
                "worker_id": worker_id,
                "request": encode_payload(request),
                "source": source_payload,
            },
        )
        raw_facts = payload.get("facts")
        facts = (
            decode_payload(ModelSessionFacts, raw_facts)
            if raw_facts is not None
            else None
        )
        environment = decode_payload(
            ComputeExecutionEnvironment, payload.get("environment")
        )
        return facts, environment

    def profile(self, case: ProfilingCase) -> MeasurementRecord:
        payload = self._call("/run", {"case": encode_payload(case)})
        return decode_payload(MeasurementRecord, payload.get("record"))

    def close_session(self) -> None:
        self._call("/close", {})

    def close(self) -> None:
        self._client.close()

    def _call(self, path: str, payload: Mapping[str, object]) -> dict[str, object]:
        response = self._client.post(path, json=payload)
        response.raise_for_status()
        decoded = response.json()
        if not isinstance(decoded, dict):
            raise ValueError("compute service returned a non-object response")
        if decoded.get("ok") is not True:
            failure = decode_payload(ProfilingFailure, decoded.get("failure"))
            raise ProfilingError(
                failure.category, failure.message, dict(failure.details)
            )
        return decoded


ClientFactory = Callable[[str, float], ComputeServiceClient]
DockerClientFactory = Callable[[], Any]


@dataclass
class _ContainerSessionState:
    container: Any
    client: ComputeServiceClient


class ContainerComputeProfilingExecutor:
    """Session-scoped compute execution in the declared shard runtime image."""

    def __init__(
        self,
        *,
        docker_client_factory: DockerClientFactory,
        model_store: ModelStore,
        ready_timeout_s: float = 120.0,
        request_timeout_s: float = 600.0,
        poll_interval_s: float = 0.1,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self._docker_client_factory = docker_client_factory
        self._model_store = model_store
        self._ready_timeout_s = ready_timeout_s
        self._request_timeout_s = request_timeout_s
        self._poll_interval_s = poll_interval_s
        self._client_factory = client_factory or _HttpComputeServiceClient

    def prepare_session(
        self,
        session_id: str,
        worker_id: str,
        request: ProfilingSessionRequest,
        source: ModelSource | None,
        capability: WorkerCapability,
    ) -> PreparedComputeSession:
        if request.kind is ProfilingSessionKind.NETWORK:
            raise ValueError("network sessions do not use a compute executor")
        if len(request.device_ids) != 1:
            raise ProfilingError(
                ProfilingErrorCategory.INTERNAL_ERROR,
                "container compute sessions require exactly one target device",
            )
        target_device_id = request.device_ids[0]
        device, platform = _select_runtime_platform(capability, target_device_id)
        assert platform.image is not None
        docker_client = self._docker_client_factory()
        try:
            image = docker_client.images.get(platform.image)
            image_revision = str(image.id)
        except Exception as exc:
            raise ProfilingError(
                ProfilingErrorCategory.INTERNAL_ERROR,
                f"cannot resolve edgeshard_shard image {platform.image!r}: {exc}",
                {"image": platform.image},
            ) from exc
        if not image_revision:
            raise ProfilingError(
                ProfilingErrorCategory.INTERNAL_ERROR,
                f"edgeshard_shard image {platform.image!r} has no immutable image id",
            )

        execution_device = "cpu" if device.identity.kind is DeviceKind.CPU else "cuda:0"
        environment = {
            _TARGET_DEVICE_ENV: target_device_id,
            _EXECUTION_DEVICE_ENV: execution_device,
            _BACKEND_REVISION_ENV: image_revision,
        }
        kwargs: dict[str, object] = {
            "detach": True,
            "name": (
                "edgeshard-profile-"
                f"{hashlib.sha256(session_id.encode()).hexdigest()[:16]}-"
                f"{uuid.uuid4().hex[:8]}"
            ),
            "command": [
                "_compute-profile-service",
                "--host",
                "0.0.0.0",
                "--port",
                str(CONTAINER_COMPUTE_PORT),
            ],
            "volumes": {
                str(self._model_store.model_root): {
                    "bind": MODEL_MOUNT,
                    "mode": "ro",
                }
            },
            "ports": {f"{CONTAINER_COMPUTE_PORT}/tcp": ("127.0.0.1", None)},
            "environment": environment,
            "labels": {
                "io.edgeshard.profiling": "true",
                "io.edgeshard.profiling_session_id": session_id,
                "io.edgeshard.backend": "edgeshard_shard",
            },
            "init": True,
        }
        if device.identity.kind is DeviceKind.GPU:
            kwargs["device_requests"] = [
                _target_gpu_device_request(target_device_id, worker_id)
            ]
        container: Any | None = None
        client: ComputeServiceClient | None = None
        try:
            # Launch the exact image object that supplied backend_revision;
            # a mutable tag must not change between fingerprinting and run.
            container = docker_client.containers.run(image_revision, **kwargs)
            endpoint = _published_endpoint(container)
            client = self._client_factory(endpoint, self._request_timeout_s)
            self._wait_ready(client)
            container_source = self._container_source(source)
            facts, observed = client.prepare(
                session_id, worker_id, request, container_source
            )
            self._validate_environment(
                observed,
                target_device_id,
                execution_device,
                image_revision,
            )
            return PreparedComputeSession(
                session_id=session_id,
                target_device_id=target_device_id,
                environment=observed,
                facts=facts,
                opaque=_ContainerSessionState(container=container, client=client),
            )
        except ProfilingError:
            _cleanup_container(container, client)
            raise
        except Exception as exc:
            _cleanup_container(container, client)
            raise ProfilingError(
                ProfilingErrorCategory.INTERNAL_ERROR,
                f"container compute session preparation failed: {exc}",
                {"session_id": session_id, "device_id": target_device_id},
            ) from exc

    def profile_case(
        self, session: PreparedComputeSession, case: ProfilingCase
    ) -> MeasurementRecord:
        if session.closed or not isinstance(session.opaque, _ContainerSessionState):
            raise ProfilingError(
                ProfilingErrorCategory.INTERNAL_ERROR,
                f"container compute session {session.session_id!r} is closed",
            )
        spec = case.spec
        if (
            not isinstance(spec, ModelCaseSpec)
            or not spec.device_ids
            or spec.device_ids[0] != session.target_device_id
        ):
            raise ProfilingError(
                ProfilingErrorCategory.INTERNAL_ERROR,
                f"case {case.case_id!r} does not target container device "
                f"{session.target_device_id!r}",
            )
        return session.opaque.client.profile(case)

    def close_session(self, session: PreparedComputeSession) -> None:
        if session.closed:
            return
        session.closed = True
        state = session.opaque
        session.opaque = None
        if isinstance(state, _ContainerSessionState):
            _cleanup_container(state.container, state.client, request_close=True)

    def _wait_ready(self, client: ComputeServiceClient) -> None:
        deadline = time.monotonic() + self._ready_timeout_s
        last_error: Exception | None = None
        while time.monotonic() <= deadline:
            try:
                client.health()
                return
            except Exception as exc:
                last_error = exc
                time.sleep(self._poll_interval_s)
        raise ProfilingError(
            ProfilingErrorCategory.TIMEOUT,
            f"container compute service was not ready within {self._ready_timeout_s}s: "
            f"{last_error}",
        )

    def _container_source(self, source: ModelSource | None) -> ModelSource | None:
        if source is None:
            return None
        root = self._model_store.model_root.resolve()
        snapshot = source.path.resolve()
        try:
            relative = snapshot.relative_to(root)
        except ValueError as exc:
            raise ProfilingError(
                ProfilingErrorCategory.UNSUPPORTED_MODEL,
                f"model source {snapshot} is outside ModelStore {root}",
            ) from exc
        return ModelSource(
            path=Path(MODEL_MOUNT) / relative,
            model_id=source.model_id,
            revision=source.revision,
        )

    @staticmethod
    def _validate_environment(
        observed: ComputeExecutionEnvironment,
        target_device_id: str,
        execution_device: str,
        backend_revision: str,
    ) -> None:
        if observed.target_device_id != target_device_id:
            raise ProfilingError(
                ProfilingErrorCategory.INTERNAL_ERROR,
                "container reported a different physical target device",
                {
                    "expected_device_id": target_device_id,
                    "actual_device_id": observed.target_device_id,
                },
            )
        if observed.execution_device != execution_device:
            raise ProfilingError(
                ProfilingErrorCategory.INTERNAL_ERROR,
                "container reported a different execution device",
                {
                    "expected_execution_device": execution_device,
                    "actual_execution_device": observed.execution_device,
                },
            )
        if observed.backend_revision != backend_revision:
            raise ProfilingError(
                ProfilingErrorCategory.INTERNAL_ERROR,
                "container reported a different backend image revision",
                {
                    "expected_backend_revision": backend_revision,
                    "actual_backend_revision": observed.backend_revision,
                },
            )


def _select_runtime_platform(
    capability: WorkerCapability, target_device_id: str
) -> tuple[Any, RuntimePlatformCapability]:
    device = next(
        (
            candidate
            for candidate in capability.devices
            if candidate.identity.device_id == target_device_id
        ),
        None,
    )
    if device is None:
        raise ProfilingError(
            ProfilingErrorCategory.INTERNAL_ERROR,
            f"target device {target_device_id!r} is absent from Worker capability",
        )
    desired_platform = "cpu"
    if device.identity.kind is DeviceKind.GPU:
        desired_platform = "cuda"
    matches = [
        candidate
        for candidate in capability.runtime_platforms
        if candidate.backend == "edgeshard_shard"
        and candidate.platform == desired_platform
        and candidate.image is not None
    ]
    if len(matches) != 1:
        raise ProfilingError(
            ProfilingErrorCategory.INTERNAL_ERROR,
            "Worker must declare exactly one edgeshard_shard image for "
            f"platform {desired_platform!r}; found {len(matches)}",
            {"platform": desired_platform, "device_id": target_device_id},
        )
    return device, matches[0]


def _target_gpu_device_request(device_id: str, worker_id: str) -> docker.types.DeviceRequest:
    """Expose exactly the selected physical GPU; never an implicit GPU 0."""
    if device_id == derive_jetson_gpu_device_id(worker_id):
        return docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
    selector = device_id
    if device_id.startswith("nvml-gpu-"):
        selector = device_id.removeprefix("nvml-gpu-")
        if not selector.isdigit():
            raise ProfilingError(
                ProfilingErrorCategory.INTERNAL_ERROR,
                f"fallback GPU id {device_id!r} has no Docker device selector",
            )
    return docker.types.DeviceRequest(
        device_ids=[selector], capabilities=[["gpu"]]
    )


def _published_endpoint(container: Any) -> str:
    container.reload()
    bindings = (
        container.attrs.get("NetworkSettings", {})
        .get("Ports", {})
        .get(f"{CONTAINER_COMPUTE_PORT}/tcp")
    )
    if not bindings:
        raise RuntimeError("container compute port was not published")
    return f"127.0.0.1:{int(bindings[0]['HostPort'])}"


def _cleanup_container(
    container: Any | None,
    client: ComputeServiceClient | None,
    *,
    request_close: bool = False,
) -> None:
    if client is not None:
        if request_close:
            try:
                client.close_session()
            except Exception as exc:
                logger.warning("container compute session close failed: %s", exc)
        try:
            client.close()
        except Exception as exc:
            logger.warning("profiling compute client close failed: %s", exc)
    if container is None:
        return
    try:
        container.stop(timeout=10)
    except Exception as exc:
        logger.warning("profiling container stop failed: %s", exc)
    try:
        container.remove()
    except Exception as exc:
        logger.warning("profiling container removal failed: %s", exc)


class _ContainerService:
    def __init__(self, executor: HostTorchComputeProfilingExecutor) -> None:
        self._executor = executor
        self._session: PreparedComputeSession | None = None

    def dispatch(self, path: str, payload: object) -> dict[str, object]:
        try:
            if not isinstance(payload, dict):
                raise ValueError("request payload must be an object")
            if path == "/prepare":
                return self._prepare(payload)
            if path == "/run":
                return self._run(payload)
            if path == "/close":
                if self._session is not None:
                    self._executor.close_session(self._session)
                    self._session = None
                return {"ok": True}
            raise ValueError(f"unknown compute service path {path!r}")
        except ProfilingError as exc:
            return {"ok": False, "failure": encode_payload(exc.to_failure())}
        except Exception as exc:
            logger.exception("container compute request failed")
            failure = ProfilingFailure(
                category=ProfilingErrorCategory.INTERNAL_ERROR,
                message=f"container compute request failed: {exc}",
            )
            return {"ok": False, "failure": encode_payload(failure)}

    def _prepare(self, payload: dict[str, object]) -> dict[str, object]:
        if self._session is not None:
            raise ValueError("container compute service already has a session")
        request = decode_payload(ProfilingSessionRequest, payload.get("request"))
        raw_source = payload.get("source")
        source: ModelSource | None = None
        if raw_source is not None:
            if not isinstance(raw_source, dict):
                raise ValueError("source must be an object")
            source = ModelSource(
                path=Path(str(raw_source.get("path"))),
                model_id=_optional_string(raw_source.get("model_id")),
                revision=_optional_string(raw_source.get("revision")),
            )
        session_id = _required_string(payload.get("session_id"), "session_id")
        worker_id = _required_string(payload.get("worker_id"), "worker_id")
        # Capability is used only by the Docker-side platform selector, never
        # by the host torch implementation running inside the chosen image.
        self._session = self._executor.prepare_session(
            session_id, worker_id, request, source, _unused_capability()
        )
        return {
            "ok": True,
            "facts": encode_payload(self._session.facts),
            "environment": encode_payload(self._session.environment),
        }

    def _run(self, payload: dict[str, object]) -> dict[str, object]:
        if self._session is None:
            raise ValueError("container compute service has no prepared session")
        case = decode_payload(ProfilingCase, payload.get("case"))
        record = self._executor.profile_case(self._session, case)
        return {"ok": True, "record": encode_payload(record)}


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("optional identity values must be strings")
    return value


def _unused_capability() -> WorkerCapability:
    """Minimal value for the host executor's intentionally unused argument."""
    from edgeshard.cluster.capability import OSInfo

    return WorkerCapability(
        architecture="container",
        os=OSInfo(name="container", version=None, kernel=None),
        container_runtime=None,
        network_interfaces=(),
        devices=(),
        memory_pools=(),
        runtime_platforms=(),
        capability_revision="container",
    )


def _container_environment(device: torch.device) -> ComputeExecutionEnvironment:
    target = _required_string(os.environ.get(_TARGET_DEVICE_ENV), _TARGET_DEVICE_ENV)
    backend_revision = _required_string(
        os.environ.get(_BACKEND_REVISION_ENV), _BACKEND_REVISION_ENV
    )
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ProfilingError(
                ProfilingErrorCategory.INTERNAL_ERROR,
                "target GPU is not available inside the profiling container",
                {"target_device_id": target},
            )
        if torch.cuda.device_count() != 1 or device.index != 0:
            raise ProfilingError(
                ProfilingErrorCategory.INTERNAL_ERROR,
                "GPU profiling container must expose exactly one GPU as cuda:0",
                {
                    "target_device_id": target,
                    "visible_device_count": torch.cuda.device_count(),
                    "execution_device": str(device),
                },
            )
    return ComputeExecutionEnvironment(
        torch_version=str(torch.__version__),
        cuda_version=(str(torch.version.cuda) if torch.version.cuda is not None else None),
        backend_revision=backend_revision,
        target_device_id=target,
        execution_device=str(device),
    )


def serve_container_compute(*, host: str, port: int) -> None:
    """Run the internal session service inside an ``edgeshard_shard`` image."""
    execution_device = torch.device(
        _required_string(os.environ.get(_EXECUTION_DEVICE_ENV), _EXECUTION_DEVICE_ENV)
    )

    def resolve(device_id: str, worker_id: str) -> torch.device:
        del worker_id
        target = _required_string(os.environ.get(_TARGET_DEVICE_ENV), _TARGET_DEVICE_ENV)
        if device_id != target:
            raise ProfilingError(
                ProfilingErrorCategory.INTERNAL_ERROR,
                f"container target is {target!r}, request named {device_id!r}",
            )
        return execution_device

    backend_revision = _required_string(
        os.environ.get(_BACKEND_REVISION_ENV), _BACKEND_REVISION_ENV
    )
    executor = HostTorchComputeProfilingExecutor(
        device_resolver=resolve,
        backend_revision=backend_revision,
        environment_factory=lambda device, _target, _revision: _container_environment(
            device
        ),
    )
    service = _ContainerService(executor)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_error(404)
                return
            self._write({"ok": True})

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                failure = ProfilingFailure(
                    category=ProfilingErrorCategory.INTERNAL_ERROR,
                    message=f"invalid compute request: {exc}",
                )
                self._write({"ok": False, "failure": encode_payload(failure)})
                return
            self._write(service.dispatch(self.path, payload))

        def log_message(self, message: str, *args: object) -> None:
            logger.info("container compute service: " + message, *args)

        def _write(self, payload: object) -> None:
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer((host, port), Handler)
    print(f"READY compute-profile endpoint={host}:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


__all__ = [
    "CONTAINER_COMPUTE_PORT",
    "ComputeExecutionEnvironment",
    "ComputeProfilingExecutor",
    "ContainerComputeProfilingExecutor",
    "HostTorchComputeProfilingExecutor",
    "PreparedComputeSession",
    "serve_container_compute",
]
