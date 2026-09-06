"""P2E operator workload registry tests (spec §25.1).

The initial registered set is exactly GEMM/ATTENTION/NORM; unregistered
kinds resolve to a typed ``UNSUPPORTED_OPERATOR`` failure — never a
silent skip or substitute workload — and the registry stays open for
incremental extension.
"""

from __future__ import annotations

import pytest

from edgeshard.profiling.domain.experiment import ProfilingErrorCategory
from edgeshard.profiling.domain.signature import OperatorKind
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.operator.registry import (
    OperatorWorkloadRegistry,
    default_operator_workload_registry,
)
from edgeshard.profiling.operator.workloads import (
    AttentionWorkload,
    GemmWorkload,
    NormWorkload,
)


class TestDefaultRegistry:
    def test_initial_kinds(self) -> None:
        registry = default_operator_workload_registry()
        assert registry.kinds == (
            OperatorKind.GEMM,
            OperatorKind.ATTENTION,
            OperatorKind.NORM,
        )

    def test_factories_produce_matching_workloads(self) -> None:
        registry = default_operator_workload_registry()
        assert isinstance(registry.resolve(OperatorKind.GEMM)(), GemmWorkload)
        assert isinstance(registry.resolve(OperatorKind.ATTENTION)(), AttentionWorkload)
        assert isinstance(registry.resolve(OperatorKind.NORM)(), NormWorkload)

    def test_resolve_returns_fresh_instances(self) -> None:
        registry = default_operator_workload_registry()
        factory = registry.resolve(OperatorKind.GEMM)
        assert factory() is not factory()

    def test_supports(self) -> None:
        registry = default_operator_workload_registry()
        assert registry.supports(OperatorKind.GEMM) is True
        assert registry.supports(OperatorKind.EMBEDDING) is False
        assert registry.supports(OperatorKind.CUSTOM) is False

    @pytest.mark.parametrize(
        "kind",
        [
            OperatorKind.EMBEDDING,
            OperatorKind.ROTARY,
            OperatorKind.ELEMENTWISE,
            OperatorKind.REDUCTION,
            OperatorKind.KV_COPY,
            OperatorKind.CUSTOM,
        ],
    )
    def test_unregistered_kinds_fail_typed(self, kind: OperatorKind) -> None:
        """Kinds awaiting incremental workloads (§25) fail typed (§42)."""
        with pytest.raises(ProfilingError) as excinfo:
            default_operator_workload_registry().resolve(kind)
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_OPERATOR
        assert ("kind", kind.value) in excinfo.value.to_failure().details
        assert kind.value in str(excinfo.value)


class TestRegistryExtension:
    def test_register_new_kind(self) -> None:
        registry = default_operator_workload_registry()
        assert registry.supports(OperatorKind.EMBEDDING) is False
        registry.register(OperatorKind.EMBEDDING, GemmWorkload)
        assert registry.supports(OperatorKind.EMBEDDING) is True
        assert isinstance(registry.resolve(OperatorKind.EMBEDDING)(), GemmWorkload)
        assert registry.kinds[-1] is OperatorKind.EMBEDDING

    def test_duplicate_registration_rejected(self) -> None:
        registry = default_operator_workload_registry()
        with pytest.raises(ValueError, match="duplicate"):
            registry.register(OperatorKind.GEMM, GemmWorkload)

    def test_constructor_binds_factories(self) -> None:
        registry = OperatorWorkloadRegistry({OperatorKind.NORM: NormWorkload})
        assert registry.kinds == (OperatorKind.NORM,)
        with pytest.raises(ProfilingError):
            registry.resolve(OperatorKind.GEMM)

    def test_empty_registry(self) -> None:
        registry = OperatorWorkloadRegistry()
        assert registry.kinds == ()
        with pytest.raises(ProfilingError) as excinfo:
            registry.resolve(OperatorKind.GEMM)
        assert excinfo.value.category is ProfilingErrorCategory.UNSUPPORTED_OPERATOR
