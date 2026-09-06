"""P2D ModuleProfiler tests (spec §23, §50 P2D DoD).

Direct invocation of real Attention/MLP modules from a tiny checkpoint,
measured through the shared harness with CPU wall-clock timing; plus the
shared execution pieces (``ModuleCallWorkload`` behavior and the
``BenchmarkResult`` → ``MeasurementRecord`` mapping) that both P2D
profilers depend on.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
import torch
import torch.nn as nn

from edgeshard.model.layout import ModelLayout
from edgeshard.profiling.benchmark.harness import BenchmarkResult, InstrumentationBundle
from edgeshard.profiling.benchmark.sampling import DurationSamplingPolicy
from edgeshard.profiling.domain.experiment import (
    ModelCaseSpec,
    ProfilingCase,
    ProfilingErrorCategory,
)
from edgeshard.profiling.domain.measurement import (
    MeasurementRecord,
    SampleSummary,
    TelemetryContextMetrics,
    TimeUnit,
)
from edgeshard.profiling.domain.model import ModelReference
from edgeshard.profiling.domain.signature import (
    InferencePhase,
    ModuleKind,
    ProfilingGranularity,
)
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.instrumentation.timing import WallClockTimer
from edgeshard.profiling.model.adapters.base import ProfileModule
from edgeshard.profiling.model.adapters.qwen import QwenProfilingAdapter
from edgeshard.profiling.model.execution import (
    ModuleCallWorkload,
    measurement_record_from_result,
    require_model_spec,
)
from edgeshard.profiling.model.module_profiler import ModuleProfiler

MODEL = ModelReference(model_id="tiny/qwen2", revision="rev-9")


def _policy() -> DurationSamplingPolicy:
    return DurationSamplingPolicy(
        min_warmups=1, min_runs=3, max_runs=3, target_duration_ms=1_000_000.0
    )


def _module_case(module: ProfileModule, *, dtype: str = "fp32") -> ProfilingCase:
    spec = ModelCaseSpec(
        granularity=ProfilingGranularity.MODULE,
        device_ids=("cpu-0",),
        dtype=dtype,
        model=MODEL,
        module_signature=module.signature,
        phase=InferencePhase.PREFILL,
        batch_size=1,
        sequence_length=8,
    )
    return ProfilingCase.for_spec("worker-test", spec)


@pytest.fixture
def modules(
    tiny_qwen2_checkpoint: nn.Module, tiny_qwen2_layout: ModelLayout
) -> dict[str, ProfileModule]:
    adapter = QwenProfilingAdapter()
    layers = adapter.enumerate_transformer_layers(tiny_qwen2_checkpoint, tiny_qwen2_layout)
    enumerated = adapter.enumerate_profile_modules(layers[0], tiny_qwen2_layout)
    return {module.name: module for module in enumerated}


class _Recording(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.calls: list[dict[str, Any]] = []

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls.append({"hidden_states": hidden_states})
        return self.linear(hidden_states)


class TestModuleCallWorkload:
    def test_run_once_invokes_module_with_inputs(self) -> None:
        module = _Recording()
        inputs = {"hidden_states": torch.randn(1, 2, 4)}
        workload = ModuleCallWorkload(module, inputs)
        workload.prepare()
        workload.run_once()
        workload.run_once()
        assert len(module.calls) == 2
        assert torch.equal(module.calls[0]["hidden_states"], inputs["hidden_states"])
        workload.reset()
        workload.cleanup()

    def test_prepare_pins_eval_mode(self) -> None:
        module = _Recording().train()
        ModuleCallWorkload(module, {"hidden_states": torch.randn(1, 2, 4)}).prepare()
        assert module.training is False

    def test_inputs_are_copied(self) -> None:
        module = _Recording()
        inputs = {"hidden_states": torch.randn(1, 2, 4)}
        workload = ModuleCallWorkload(module, inputs)
        inputs["hidden_states"] = torch.zeros(1, 2, 4)
        workload.run_once()
        assert not torch.equal(module.calls[0]["hidden_states"], torch.zeros(1, 2, 4))

    def test_requires_nn_module(self) -> None:
        with pytest.raises(TypeError, match=r"nn\.Module"):
            ModuleCallWorkload(object(), {})  # type: ignore[arg-type]


def _result(**overrides: Any) -> BenchmarkResult:
    now = datetime.now(UTC)
    defaults: dict[str, Any] = {
        "started_at": now,
        "finished_at": now,
        "warmup_runs": 2,
        "measured_runs": 3,
        "samples_ms": (1.0, 2.0, 3.0),
        "summary": SampleSummary(
            mean=2.0, median=2.0, stddev=1.0, minimum=1.0, maximum=3.0, p95=3.0
        ),
        "telemetry": TelemetryContextMetrics(),
    }
    defaults.update(overrides)
    return BenchmarkResult(**defaults)


class TestMeasurementRecordFromResult:
    def test_maps_all_metric_slots(self) -> None:
        result = _result()
        record = measurement_record_from_result(
            result,
            case_id="case-1",
            environment_fingerprint="env-1",
            metadata={"layer_index": 2, "model_revision": None},
        )
        assert record.case_id == "case-1"
        assert record.environment_fingerprint == "env-1"
        assert record.sample_count == 3
        assert record.samples == (1.0, 2.0, 3.0)
        assert record.metrics.latency is not None
        assert record.metrics.latency.unit is TimeUnit.MILLISECONDS
        assert record.metrics.latency.summary == result.summary
        assert record.metrics.telemetry is not None
        assert record.metrics.allocator_memory is None
        assert record.metrics.physical_memory is None
        assert record.metadata_mapping == {"layer_index": 2, "model_revision": None}

    def test_measurement_ids_unique(self) -> None:
        first = measurement_record_from_result(
            _result(), case_id="case-1", environment_fingerprint="env-1"
        )
        second = measurement_record_from_result(
            _result(), case_id="case-1", environment_fingerprint="env-1"
        )
        assert first.measurement_id != second.measurement_id

    def test_inconsistent_result_fails_typed(self) -> None:
        with pytest.raises(ProfilingError) as excinfo:
            measurement_record_from_result(
                _result(measured_runs=5), case_id="case-1", environment_fingerprint="env-1"
            )
        assert excinfo.value.category is ProfilingErrorCategory.INTERNAL_ERROR

    def test_no_metadata(self) -> None:
        record: MeasurementRecord = measurement_record_from_result(
            _result(), case_id="case-1", environment_fingerprint="env-1"
        )
        assert record.metadata == ()


class TestRequireModelSpec:
    def test_accepts_matching_granularity(self, modules: dict[str, ProfileModule]) -> None:
        case = _module_case(modules["mlp"])
        spec = require_model_spec(case, ProfilingGranularity.MODULE)
        assert spec.granularity is ProfilingGranularity.MODULE

    def test_rejects_mismatch(self, modules: dict[str, ProfileModule]) -> None:
        case = _module_case(modules["mlp"])
        with pytest.raises(ValueError, match="does not match"):
            require_model_spec(case, ProfilingGranularity.TRANSFORMER_LAYER)


class TestModuleProfiler:
    @pytest.mark.parametrize("name", ["self_attn", "mlp", "input_layernorm"])
    def test_end_to_end_records(
        self,
        name: str,
        modules: dict[str, ProfileModule],
        tiny_qwen2_checkpoint: nn.Module,
        tiny_qwen2_layout: ModelLayout,
    ) -> None:
        module = modules[name]
        record = ModuleProfiler(sampling_policy=_policy()).profile(
            _module_case(module),
            tiny_qwen2_checkpoint,
            module,
            tiny_qwen2_layout,
            QwenProfilingAdapter(),
            instrumentation=InstrumentationBundle(timer=WallClockTimer()),
            environment_fingerprint="env-test",
            seed=42,
        )
        assert record.sample_count == 3
        latency = record.metrics.latency
        assert latency is not None and latency.summary.mean > 0.0
        metadata = record.metadata_mapping
        assert metadata["granularity"] == "module"
        assert metadata["module_kind"] == module.kind.value
        assert metadata["module_name"] == name
        assert metadata["module_path"] == f"model.layers.0.{name}"
        assert metadata["model_id"] == "tiny/qwen2"
        assert metadata["model_revision"] == "rev-9"

    def test_attention_kind_recorded(
        self,
        modules: dict[str, ProfileModule],
        tiny_qwen2_checkpoint: nn.Module,
        tiny_qwen2_layout: ModelLayout,
    ) -> None:
        record = ModuleProfiler(sampling_policy=_policy()).profile(
            _module_case(modules["self_attn"]),
            tiny_qwen2_checkpoint,
            modules["self_attn"],
            tiny_qwen2_layout,
            QwenProfilingAdapter(),
            instrumentation=InstrumentationBundle(timer=WallClockTimer()),
            environment_fingerprint="env-test",
        )
        assert record.metadata_mapping["module_kind"] == ModuleKind.ATTENTION.value

    def test_dtype_mismatch_fails_typed(
        self,
        modules: dict[str, ProfileModule],
        tiny_qwen2_checkpoint: nn.Module,
        tiny_qwen2_layout: ModelLayout,
    ) -> None:
        with pytest.raises(ProfilingError) as excinfo:
            ModuleProfiler(sampling_policy=_policy()).profile(
                _module_case(modules["mlp"], dtype="bf16"),
                tiny_qwen2_checkpoint,
                modules["mlp"],
                tiny_qwen2_layout,
                QwenProfilingAdapter(),
                instrumentation=InstrumentationBundle(timer=WallClockTimer()),
                environment_fingerprint="env-test",
            )
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_MODEL
