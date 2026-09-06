"""Operator workload registry (spec §25.1).

Maps the stable operator vocabulary onto synthetic workload
implementations. The initial set is deliberately small — GEMM,
ATTENTION, NORM — and grows incrementally: resolving an unregistered
kind is a typed ``UNSUPPORTED_OPERATOR`` failure (§42), never a silent
skip and never a guessed substitute workload.

The registry stores *factories*, not workload instances: each benchmark
run gets a fresh workload so materialized tensors never leak across
cases and ``prepare``/``cleanup`` bracket exactly one measurement.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from edgeshard.profiling.domain.experiment import ProfilingErrorCategory
from edgeshard.profiling.domain.signature import OperatorKind
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.operator.workloads import (
    AttentionWorkload,
    GemmWorkload,
    NormWorkload,
    OperatorWorkload,
)

OperatorWorkloadFactory = Callable[[], OperatorWorkload]
"""Creates one fresh workload instance for a single benchmark run."""

_INITIAL_WORKLOADS: dict[OperatorKind, OperatorWorkloadFactory] = {
    OperatorKind.GEMM: GemmWorkload,
    OperatorKind.ATTENTION: AttentionWorkload,
    OperatorKind.NORM: NormWorkload,
}


class OperatorWorkloadRegistry:
    """Kind → workload factory registry (spec §25.1)."""

    def __init__(
        self, factories: Mapping[OperatorKind, OperatorWorkloadFactory] | None = None
    ) -> None:
        self._factories: dict[OperatorKind, OperatorWorkloadFactory] = {}
        for kind, factory in (factories or {}).items():
            self.register(kind, factory)

    def register(self, kind: OperatorKind, factory: OperatorWorkloadFactory) -> None:
        """Add or reject a duplicate binding; silent replacement is a bug."""
        if kind in self._factories:
            raise ValueError(f"duplicate operator workload registration for {kind.value!r}")
        self._factories[kind] = factory

    def resolve(self, kind: OperatorKind) -> OperatorWorkloadFactory:
        """Factory for ``kind``; unregistered kinds fail typed (§42).

        Kinds outside the registered set (custom operators preserved for
        coverage, §19; kinds awaiting incremental workloads, §25) are
        reported as ``UNSUPPORTED_OPERATOR`` — the caller decides whether
        that is fatal or a coverage note.
        """
        factory = self._factories.get(kind)
        if factory is None:
            raise ProfilingError(
                ProfilingErrorCategory.UNSUPPORTED_OPERATOR,
                f"no operator workload registered for kind {kind.value!r}; "
                f"registered kinds: {', '.join(registered.value for registered in self.kinds)}",
                {"kind": kind.value},
            )
        return factory

    def supports(self, kind: OperatorKind) -> bool:
        return kind in self._factories

    @property
    def kinds(self) -> tuple[OperatorKind, ...]:
        """Registered kinds in registration order."""
        return tuple(self._factories)


def default_operator_workload_registry() -> OperatorWorkloadRegistry:
    """Registry with the initial §25 workload set (GEMM/ATTENTION/NORM)."""
    return OperatorWorkloadRegistry(_INITIAL_WORKLOADS)


__all__ = [
    "OperatorWorkloadFactory",
    "OperatorWorkloadRegistry",
    "default_operator_workload_registry",
]
