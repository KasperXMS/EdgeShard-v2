"""Operator extraction (Phase 2 spec §18).

Two extractors behind one :class:`OperatorExtractor` protocol:

- :class:`TorchExportExtractor` (primary, §18.1) — ``torch.export``
  graphs preserve operator identity, static tensor shapes/dtypes (via
  node ``val`` metadata), and graph ordering. Export failures are typed
  ``EXPORT_FAILED``.
- :class:`TorchProfilerExtractor` (fallback, §18.2) — representative
  execution under ``torch.profiler`` with ``record_shapes``. It is
  *structural discovery only*: profiler timings are never used as
  authoritative latency (§52.6), and per-event output facts are not
  available. Failures are typed ``PROFILER_FAILED``.

Raw graphs preserve every operator occurrence they observe; nothing is
filtered or merged here — mapping to the stable vocabulary is the
normalizer's job (§19).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

import torch
from torch.profiler import ProfilerActivity
from torch.profiler import profile as torch_profile

from edgeshard.profiling.domain.experiment import ProfilingErrorCategory
from edgeshard.profiling.dtypes import dtype_label, kineto_dtype_label
from edgeshard.profiling.errors import ProfilingError

_PROFILER_INTERNAL_PREFIXES = ("aten::profiler",)


@dataclass(frozen=True)
class RawOperatorOccurrence:
    """One observed operator occurrence, in execution/graph order (§18.1).

    ``input_shapes``/``input_dtypes`` are aligned element-wise and cover
    tensor inputs only (scalars are extractor noise for shape purposes).
    Arguments whose shapes are not static (symbolic) are omitted rather
    than guessed (§52.2). Output facts are recorded when the extractor
    can see them (export), else empty (profiler).
    """

    order: int
    operation: str
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[str, ...]
    output_shapes: tuple[tuple[int, ...], ...] = ()
    output_dtypes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.order < 0:
            raise ValueError(f"order must not be negative, got {self.order}")
        if not self.operation:
            raise ValueError("operation must not be empty")
        if len(self.input_shapes) != len(self.input_dtypes):
            raise ValueError(
                "input_shapes and input_dtypes must be aligned: "
                f"{len(self.input_shapes)} vs {len(self.input_dtypes)}"
            )
        if len(self.output_shapes) != len(self.output_dtypes):
            raise ValueError("output_shapes and output_dtypes must be aligned")
        for shape in (*self.input_shapes, *self.output_shapes):
            for dimension in shape:
                if dimension <= 0:
                    raise ValueError(
                        f"raw shapes must have positive dimensions, got {shape}"
                    )


@dataclass(frozen=True)
class RawOperatorGraph:
    """All operator occurrences observed for one target execution."""

    extractor: str
    operations: tuple[RawOperatorOccurrence, ...]

    def __post_init__(self) -> None:
        if not self.extractor:
            raise ValueError("extractor must not be empty")
        for expected, occurrence in enumerate(self.operations):
            if occurrence.order != expected:
                raise ValueError(
                    "operations must be sequentially ordered from 0: "
                    f"expected {expected}, got {occurrence.order}"
                )


class OperatorExtractor(Protocol):
    """Turns one representative execution into a raw operator graph."""

    #: Stable extractor identity recorded on every graph.
    name: ClassVar[str]

    def extract(
        self,
        target: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> RawOperatorGraph:
        """Extract the operator graph of ``target(*args, **kwargs)``."""
        ...


def _flatten_arguments(values: Any) -> Iterator[Any]:
    """Yield arguments, flattening one level of list/tuple (e.g. ``cat``)."""
    for value in values:
        if isinstance(value, (list, tuple)):
            yield from value
        else:
            yield value


def _static_tensor_facts(value: Any) -> tuple[tuple[int, ...], str] | None:
    """Shape/dtype of a (fake) tensor when the shape is fully static."""
    if not isinstance(value, torch.Tensor):
        return None
    shape = tuple(value.shape)
    if not all(isinstance(dimension, int) for dimension in shape):
        return None  # symbolic dimensions: omitted, never guessed (§52.2)
    return shape, dtype_label(value.dtype)


def _argument_facts(values: Any) -> tuple[list[tuple[int, ...]], list[str]]:
    """Aligned shape/dtype lists over the tensor arguments of a node."""
    shapes: list[tuple[int, ...]] = []
    dtypes: list[str] = []
    for value in _flatten_arguments(values):
        node_value = value.meta.get("val") if isinstance(value, torch.fx.Node) else value
        facts = _static_tensor_facts(node_value)
        if facts is not None:
            shapes.append(facts[0])
            dtypes.append(facts[1])
    return shapes, dtypes


class TorchExportExtractor:
    """Primary extractor via ``torch.export`` (spec §18.1)."""

    name: ClassVar[str] = "torch_export"

    def extract(
        self,
        target: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> RawOperatorGraph:
        try:
            with torch.no_grad():
                program = torch.export.export(target, args, dict(kwargs))
        except Exception as exc:
            raise ProfilingError(
                ProfilingErrorCategory.EXPORT_FAILED,
                f"torch.export failed for {type(target).__name__}: {exc}",
            ) from exc

        occurrences: list[RawOperatorOccurrence] = []
        for node in program.graph.nodes:
            if node.op != "call_function":
                continue
            input_shapes, input_dtypes = _argument_facts(
                (*node.args, *node.kwargs.values())
            )
            output_shapes, output_dtypes = _argument_facts((node.meta.get("val"),))
            occurrences.append(
                RawOperatorOccurrence(
                    order=len(occurrences),
                    operation=str(node.target),
                    input_shapes=tuple(input_shapes),
                    input_dtypes=tuple(input_dtypes),
                    output_shapes=tuple(output_shapes),
                    output_dtypes=tuple(output_dtypes),
                )
            )
        return RawOperatorGraph(extractor=self.name, operations=tuple(occurrences))


class TorchProfilerExtractor:
    """Fallback extractor via representative execution (spec §18.2).

    Structural discovery only: the recorded operator names, order, and
    input shapes/dtypes feed normalization; profiler *timings* are never
    authoritative latency (§52.6) and are not captured at all. Non-aten
    events (Python functions, module markers) are not operators and are
    not recorded; every ``aten::`` event is.
    """

    name: ClassVar[str] = "torch_profiler"

    def extract(
        self,
        target: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> RawOperatorGraph:
        try:
            with (
                torch.no_grad(),
                torch_profile(activities=[ProfilerActivity.CPU], record_shapes=True) as profiler,
            ):
                target(*args, **kwargs)
        except Exception as exc:
            raise ProfilingError(
                ProfilingErrorCategory.PROFILER_FAILED,
                f"profiled execution failed for {type(target).__name__}: {exc}",
            ) from exc

        occurrences: list[RawOperatorOccurrence] = []
        for event in profiler.events():
            name = event.name
            if not name.startswith("aten::") or name.startswith(_PROFILER_INTERNAL_PREFIXES):
                continue
            raw_shapes: Sequence[list[int]] = list(getattr(event, "input_shapes", None) or [])
            raw_dtypes: Sequence[Any] = list(getattr(event, "input_dtypes", None) or [])
            shapes: list[tuple[int, ...]] = []
            dtypes: list[str] = []
            for index, raw_shape in enumerate(raw_shapes):
                if not raw_shape:
                    continue  # scalar argument
                dimensions = tuple(int(dimension) for dimension in raw_shape)
                if any(dimension <= 0 for dimension in dimensions):
                    continue  # degenerate placeholder, not a workload shape
                raw_dtype = str(raw_dtypes[index]) if index < len(raw_dtypes) else ""
                shapes.append(dimensions)
                dtypes.append(kineto_dtype_label(raw_dtype))
            occurrences.append(
                RawOperatorOccurrence(
                    order=len(occurrences),
                    operation=name,
                    input_shapes=tuple(shapes),
                    input_dtypes=tuple(dtypes),
                )
            )
        if not occurrences:
            raise ProfilingError(
                ProfilingErrorCategory.PROFILER_FAILED,
                f"profiled execution of {type(target).__name__} recorded no operator events",
            )
        return RawOperatorGraph(extractor=self.name, operations=tuple(occurrences))
