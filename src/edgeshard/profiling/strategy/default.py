"""DefaultProfilingStrategy — the v1 policy composition (spec §47-§48).

Model workflow (§47), from the facts a prepared Worker session reports:

1. static model characterization — ``facts.characterization`` (performed
   Worker-side during prepare, §38/§46; the strategy never loads a model);
2-3. normalized, deduplicated operator signatures —
   ``facts.operator_signatures``;
4. operator microprofile cases for the signatures that have no compatible
   measurement yet, per device (§28: the caller supplies each device's
   measured-id set, scoped by ``DevicePerformanceClass``/environment via
   the store's reuse query);
5. sparse real Module cases: one per unique Attention/MLP module
   signature (``SPARSE_MODULE_KINDS``);
6. sparse real TransformerLayer cases: the early/middle/late
   representative positions (§22) across the default prefill lengths
   (§21), so the positional sanity check has its observations;
7. persistence — the controller's job (§40), never the strategy's.

No composed performance estimate is produced here (§47); composition is
Phase 3 reading the ``ProfileSnapshot``.

Network workflow (§47): endpoint characterization and PathClass
classification (§30-§31), the dense cheap RTT matrix over all directed
pairs (§33), the sparse per-class bandwidth selection with both flow
directions (§34), and the explicit-pair knob — delegated to the pure P2F
case builders so selection policy lives in exactly one place.

§48: the composition is hard-coded on purpose — the documented constants
below *are* the v1 GranularityPolicy/SamplingPolicy; replaceability lives
at the :class:`~edgeshard.profiling.strategy.base.ProfilingStrategy`
protocol seam, and no domain/storage/runner code depends on this module.
Signatures of kinds outside ``INITIAL_BENCHMARK_OPERATOR_KINDS`` are
deferred rather than dispatched: the strategy does not guess Worker
benchmark support (§52.2), but dispatching cases that the v1 workload
vocabulary (§25) cannot serve would only manufacture typed failures.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet

from edgeshard.profiling.domain.experiment import ModelCaseSpec, ProfilingCase
from edgeshard.profiling.domain.network import NetworkPair, NetworkPathClass
from edgeshard.profiling.domain.session import ModelSessionFacts, ModuleEntry
from edgeshard.profiling.domain.signature import (
    CustomOperatorParameters,
    ModuleKind,
    ModuleSignature,
    OperatorKind,
    OperatorSignature,
    ProfilingGranularity,
    module_signature_id,
    operator_signature_id,
)
from edgeshard.profiling.model.planning import (
    DEFAULT_PREFILL_SEQUENCE_LENGTHS,
    representative_layer_positions,
)
from edgeshard.profiling.network.classifier import (
    DEFAULT_SAME_SUBNET_PREFIX_LENGTH,
    WorkerNetworkFacts,
    classify_pairs,
    endpoint_profiles,
)
from edgeshard.profiling.network.iperf import DEFAULT_IPERF3_DURATION_S
from edgeshard.profiling.network.profiler import bandwidth_cases, rtt_matrix_cases
from edgeshard.profiling.operator.planning import (
    IncrementalPlan,
    operator_case_spec,
    plan_incremental_profiling,
)
from edgeshard.profiling.strategy.base import ModelProfilingPlan, NetworkProfilingPlan

logger = logging.getLogger("profiling.strategy.default")

DEFAULT_STRATEGY_ID = "default-v1"
"""Stable identity of this strategy; hashed into every experiment id (§7)."""

INITIAL_BENCHMARK_OPERATOR_KINDS = frozenset(
    {OperatorKind.GEMM, OperatorKind.ATTENTION, OperatorKind.NORM}
)
"""Operator kinds the v1 synthetic workload vocabulary serves (§25).

Mirrors the Worker's default ``OperatorWorkloadRegistry``; grows as
workloads are registered. Signatures of other kinds are preserved for
coverage and reported as deferred — never dispatched to fail (§19).
"""

SPARSE_MODULE_KINDS = frozenset({ModuleKind.ATTENTION, ModuleKind.MLP})
"""§47 step 5: sparse real Module profiling covers Attention + MLP first."""

DEFAULT_MODULE_SEQUENCE_LENGTH = 512
"""Prefill length of sparse Module cases (mid-range of the §21 defaults)."""


def _validated_device_ids(device_ids: Sequence[str]) -> tuple[str, ...]:
    devices = tuple(device_ids)
    if not devices:
        raise ValueError("device_ids must not be empty")
    seen: set[str] = set()
    for device_id in devices:
        if not device_id:
            raise ValueError("device_ids must not contain empty entries")
        if device_id in seen:
            raise ValueError(f"duplicate device_id {device_id!r}")
        seen.add(device_id)
    return devices


def _split_operator_signatures(
    signatures: Iterable[OperatorSignature],
) -> tuple[tuple[OperatorSignature, ...], tuple[OperatorSignature, ...]]:
    """(benchmarkable, deferred), deduplicated by canonical id (§20).

    Deferred = custom operators (§19: preserved, no synthetic workload)
    plus kinds outside the v1 benchmark vocabulary (§25). Dedup keeps the
    plan correct even if the facts were not deduplicated upstream.
    """
    unique: dict[str, OperatorSignature] = {}
    for signature in signatures:
        unique.setdefault(operator_signature_id(signature), signature)
    benchmarkable: list[OperatorSignature] = []
    deferred: list[OperatorSignature] = []
    for signature in unique.values():
        benchmarkable_kind = (
            not isinstance(signature.parameters, CustomOperatorParameters)
            and signature.kind in INITIAL_BENCHMARK_OPERATOR_KINDS
        )
        (benchmarkable if benchmarkable_kind else deferred).append(signature)
    return tuple(benchmarkable), tuple(deferred)


def _sparse_module_signatures(
    module_entries: Iterable[ModuleEntry],
) -> tuple[ModuleSignature, ...]:
    """One signature per unique sparse-kind module shape (§47 step 5).

    Homogeneous stacks enumerate the same Attention/MLP shapes once per
    layer; the module signature is the reuse identity, so the sparse
    policy keeps first-appearance order and drops duplicates.
    """
    selected: dict[str, ModuleSignature] = {}
    for entry in module_entries:
        if entry.kind not in SPARSE_MODULE_KINDS:
            continue
        selected.setdefault(module_signature_id(entry.signature), entry.signature)
    return tuple(selected.values())


class DefaultProfilingStrategy:
    """The v1 default planner (§47); satisfies ``ProfilingStrategy``."""

    def __init__(
        self,
        *,
        representatives_per_class: int = 1,
        bandwidth_duration_s: float = DEFAULT_IPERF3_DURATION_S,
        same_subnet_prefix_length: int = DEFAULT_SAME_SUBNET_PREFIX_LENGTH,
    ) -> None:
        if representatives_per_class < 1:
            raise ValueError(
                f"representatives_per_class must be positive, "
                f"got {representatives_per_class}"
            )
        if bandwidth_duration_s <= 0.0:
            raise ValueError(
                f"bandwidth_duration_s must be positive, got {bandwidth_duration_s}"
            )
        if not 1 <= same_subnet_prefix_length <= 32:
            raise ValueError(
                f"same_subnet_prefix_length must be within [1, 32], "
                f"got {same_subnet_prefix_length}"
            )
        self._representatives_per_class = representatives_per_class
        self._bandwidth_duration_s = bandwidth_duration_s
        self._same_subnet_prefix_length = same_subnet_prefix_length

    @property
    def strategy_id(self) -> str:
        return DEFAULT_STRATEGY_ID

    def plan_model_cases(
        self,
        *,
        worker_id: str,
        facts: ModelSessionFacts,
        device_ids: Sequence[str],
        measured_signature_ids: Mapping[str, AbstractSet[str]] | None = None,
    ) -> ModelProfilingPlan:
        """Plan §47 steps 4-6 as canonical cases for one worker."""
        devices = _validated_device_ids(device_ids)
        characterization = facts.characterization
        benchmarkable, deferred = _split_operator_signatures(facts.operator_signatures)
        measured = measured_signature_ids if measured_signature_ids is not None else {}
        cases: list[ProfilingCase] = []
        reuse: dict[str, IncrementalPlan] = {}

        # Step 4 — operator microprofiles missing from each compatible,
        # verified device-performance reuse scope (§28).
        for device_id in devices:
            plan = plan_incremental_profiling(
                benchmarkable, measured_ids=measured.get(device_id, frozenset())
            )
            reuse[device_id] = plan
            cases.extend(
                ProfilingCase.for_spec(
                    worker_id, operator_case_spec(signature, device_id=device_id)
                )
                for signature in plan.missing
            )

        # Step 5 — sparse real Module cases (Attention + MLP, unique shapes).
        for device_id in devices:
            for signature in _sparse_module_signatures(facts.module_entries):
                cases.append(
                    ProfilingCase.for_spec(
                        worker_id,
                        ModelCaseSpec(
                            granularity=ProfilingGranularity.MODULE,
                            device_ids=(device_id,),
                            dtype=characterization.dtype,
                            model=characterization.model,
                            module_signature=signature,
                            sequence_length=DEFAULT_MODULE_SEQUENCE_LENGTH,
                        ),
                    )
                )

        # Step 6 — sparse positional TransformerLayer cases (§22 over §21 lengths).
        entries_by_index = {entry.index: entry for entry in facts.layer_entries}
        positions = representative_layer_positions(characterization.num_layers)
        for device_id in devices:
            for _position, index in positions:
                entry = entries_by_index.get(index)
                if entry is None:
                    raise ValueError(
                        f"layer enumeration is missing index {index} required by "
                        f"the §22 positional plan (facts cover "
                        f"{sorted(entries_by_index)})"
                    )
                for sequence_length in DEFAULT_PREFILL_SEQUENCE_LENGTHS:
                    cases.append(
                        ProfilingCase.for_spec(
                            worker_id,
                            ModelCaseSpec(
                                granularity=ProfilingGranularity.TRANSFORMER_LAYER,
                                device_ids=(device_id,),
                                dtype=characterization.dtype,
                                model=characterization.model,
                                layer_signature=entry.signature,
                                layer_index=index,
                                sequence_length=sequence_length,
                            ),
                        )
                    )

        logger.info(
            "model plan for worker %s: %d cases across %d device(s); "
            "%d operator signature(s) deferred, never benchmarked silently",
            worker_id,
            len(cases),
            len(devices),
            len(deferred),
        )
        return ModelProfilingPlan(
            worker_id=worker_id,
            cases=tuple(cases),
            operator_reuse=reuse,
            deferred_operator_signatures=deferred,
        )

    def plan_network_cases(
        self,
        *,
        facts: Iterable[WorkerNetworkFacts],
        extra_bandwidth_pairs: Iterable[NetworkPair] = (),
        bandwidth_path_classes: Iterable[NetworkPathClass] | None = None,
    ) -> NetworkProfilingPlan:
        """Plan §47 network steps 1-5 over the cluster's Phase 1 facts."""
        workers = tuple(facts)
        # Steps 1-2 — endpoint characterization and PathClass classification.
        profiles = endpoint_profiles(workers)
        classified = classify_pairs(
            workers, same_subnet_prefix_length=self._same_subnet_prefix_length
        )
        # Step 3 — dense cheap RTT matrix over all directed pairs (§33).
        rtt = rtt_matrix_cases(
            workers, same_subnet_prefix_length=self._same_subnet_prefix_length
        )
        # Steps 4-5 — sparse per-class bandwidth plus the explicit-pair knob (§34).
        bandwidth = bandwidth_cases(
            workers,
            representatives_per_class=self._representatives_per_class,
            extra_pairs=extra_bandwidth_pairs,
            duration_s=self._bandwidth_duration_s,
            same_subnet_prefix_length=self._same_subnet_prefix_length,
            path_classes=bandwidth_path_classes,
        )
        logger.info(
            "network plan: %d endpoint profile(s), %d classified pair(s), "
            "%d RTT case(s), %d bandwidth case(s)",
            len(profiles),
            len(classified),
            len(rtt),
            len(bandwidth),
        )
        return NetworkProfilingPlan(
            endpoint_profiles=profiles,
            classified_pairs=classified,
            cases=rtt + bandwidth,
        )


__all__ = [
    "DEFAULT_MODULE_SEQUENCE_LENGTH",
    "DEFAULT_STRATEGY_ID",
    "INITIAL_BENCHMARK_OPERATOR_KINDS",
    "SPARSE_MODULE_KINDS",
    "DefaultProfilingStrategy",
]
