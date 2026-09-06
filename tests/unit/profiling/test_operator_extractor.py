"""P2C operator-extractor tests (spec §18, §50 P2C DoD).

Both extraction paths are tested on CPU-only torch: the primary
``torch.export`` path (operator identity, static input/output facts,
list-argument flattening, typed ``EXPORT_FAILED``) and the
``torch.profiler`` fallback (aten-only events, kineto dtype mapping,
scalar filtering, typed ``PROFILER_FAILED``). Raw container validation
is pinned as well.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from edgeshard.profiling.domain.experiment import ProfilingErrorCategory
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.operator.extractor import (
    RawOperatorGraph,
    RawOperatorOccurrence,
    TorchExportExtractor,
    TorchProfilerExtractor,
)


class _Boom(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("synthetic failure")


class _Cat(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, x], dim=0)


class _KwargAdd(nn.Module):
    def forward(self, x: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        assert extra is not None
        return x + extra


def _noop(x: torch.Tensor) -> None:
    return None


class TestRawContainers:
    def test_occurrence_alignment_required(self) -> None:
        with pytest.raises(ValueError, match="aligned"):
            RawOperatorOccurrence(
                order=0, operation="aten::mm", input_shapes=((2, 2),), input_dtypes=()
            )

    def test_occurrence_rejects_bad_order_and_name(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            RawOperatorOccurrence(order=-1, operation="aten::mm", input_shapes=(), input_dtypes=())
        with pytest.raises(ValueError, match="empty"):
            RawOperatorOccurrence(order=0, operation="", input_shapes=(), input_dtypes=())

    def test_occurrence_rejects_non_positive_dimensions(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            RawOperatorOccurrence(
                order=0, operation="aten::mm", input_shapes=((2, 0),), input_dtypes=("fp32",)
            )

    def test_graph_requires_sequential_order(self) -> None:
        occurrence = RawOperatorOccurrence(
            order=2, operation="aten::mm", input_shapes=(), input_dtypes=()
        )
        with pytest.raises(ValueError, match="sequentially ordered"):
            RawOperatorGraph(extractor="unit", operations=(occurrence,))

    def test_graph_requires_extractor_name(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            RawOperatorGraph(extractor="", operations=())


class TestTorchExportExtractor:
    def test_linear_graph(self) -> None:
        graph = TorchExportExtractor().extract(nn.Linear(8, 4), (torch.randn(2, 8),), {})
        assert graph.extractor == "torch_export"
        assert len(graph.operations) == 1
        occurrence = graph.operations[0]
        assert occurrence.operation == "aten.linear.default"
        assert occurrence.input_shapes == ((2, 8), (4, 8), (4,))
        assert occurrence.input_dtypes == ("fp32", "fp32", "fp32")
        assert occurrence.output_shapes == ((2, 4),)
        assert occurrence.output_dtypes == ("fp32",)

    def test_list_arguments_flattened(self) -> None:
        graph = TorchExportExtractor().extract(_Cat(), (torch.randn(2, 8),), {})
        cat = next(o for o in graph.operations if o.operation.startswith("aten.cat"))
        assert cat.input_shapes == ((2, 8), (2, 8))

    def test_kwargs_tensors_recorded(self) -> None:
        graph = TorchExportExtractor().extract(
            _KwargAdd(), (torch.randn(2, 8),), {"extra": torch.randn(2, 8)}
        )
        add = next(o for o in graph.operations if o.operation.startswith("aten.add"))
        assert add.input_shapes == ((2, 8), (2, 8))

    def test_orders_are_sequential(self, qwen2_layer_exports: tuple[RawOperatorGraph, ...]) -> None:
        graph = qwen2_layer_exports[0]
        assert [o.order for o in graph.operations] == list(range(len(graph.operations)))

    def test_export_failure_is_typed(self) -> None:
        with pytest.raises(ProfilingError) as excinfo:
            TorchExportExtractor().extract(_Boom(), (torch.randn(2, 2),), {})
        assert excinfo.value.category is ProfilingErrorCategory.EXPORT_FAILED
        assert isinstance(excinfo.value.__cause__, RuntimeError)


class TestTorchProfilerExtractor:
    def test_linear_events(self) -> None:
        graph = TorchProfilerExtractor().extract(nn.Linear(8, 4), (torch.randn(2, 8),), {})
        assert graph.extractor == "torch_profiler"
        assert all(o.operation.startswith("aten::") for o in graph.operations)
        linear = next(o for o in graph.operations if o.operation == "aten::linear")
        assert linear.input_shapes == ((2, 8), (4, 8), (4,))
        assert linear.input_dtypes == ("fp32", "fp32", "fp32")
        # The fallback records no output facts; timings are never captured.
        assert linear.output_shapes == ()
        assert linear.output_dtypes == ()

    def test_scalar_arguments_filtered(self) -> None:
        graph = TorchProfilerExtractor().extract(nn.Linear(8, 4), (torch.randn(2, 8),), {})
        # aten::expand receives scalar size arguments in raw events; every
        # recorded shape must be a real tensor shape with positive dims.
        expand = next(o for o in graph.operations if o.operation == "aten::expand")
        assert all(len(shape) > 0 and all(d > 0 for d in shape) for shape in expand.input_shapes)

    def test_execution_failure_is_typed(self) -> None:
        with pytest.raises(ProfilingError) as excinfo:
            TorchProfilerExtractor().extract(_Boom(), (torch.randn(2, 2),), {})
        assert excinfo.value.category is ProfilingErrorCategory.PROFILER_FAILED
        assert isinstance(excinfo.value.__cause__, RuntimeError)

    def test_no_operator_events_is_typed(self) -> None:
        with pytest.raises(ProfilingError) as excinfo:
            TorchProfilerExtractor().extract(_noop, (torch.randn(2, 2),), {})
        assert excinfo.value.category is ProfilingErrorCategory.PROFILER_FAILED
        assert "no operator events" in str(excinfo.value)


class TestTinyQwen2LayerExport:
    """Real structural discovery on a transformer layer (§50 P2C DoD)."""

    def test_attention_and_projections_discovered(
        self, qwen2_layer_exports: tuple[RawOperatorGraph, ...]
    ) -> None:
        graph = qwen2_layer_exports[0]
        sdpa = [o for o in graph.operations if "scaled_dot_product_attention" in o.operation]
        assert len(sdpa) == 1
        # q (1, 4, 8, 16), k (1, 2, 8, 16), v (1, 2, 8, 16) for the tiny model.
        assert sdpa[0].input_shapes[:3] == ((1, 4, 8, 16), (1, 2, 8, 16), (1, 2, 8, 16))
        assert sdpa[0].input_dtypes[0] == "fp32"

        linears = [o for o in graph.operations if o.operation == "aten.linear.default"]
        # q/k/v/o projections plus gate/up/down MLP projections.
        assert len(linears) == 7
        gemm_like = {o.input_shapes[:2] for o in linears}
        assert ((1, 8, 64), (64, 64)) in gemm_like  # q_proj
        assert ((1, 8, 64), (32, 64)) in gemm_like  # k_proj / v_proj

    def test_identical_layers_produce_identical_operation_sequences(
        self, qwen2_layer_exports: tuple[RawOperatorGraph, ...]
    ) -> None:
        first, second = qwen2_layer_exports
        assert [(o.operation, o.input_shapes, o.input_dtypes) for o in first.operations] == [
            (o.operation, o.input_shapes, o.input_dtypes) for o in second.operations
        ]
