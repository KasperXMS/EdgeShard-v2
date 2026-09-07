"""Strategy interface and plan results (spec §48).

The seam that keeps the strategy layer replaceable: a
:class:`ProfilingStrategy` answers two planning questions —

* *model*: given the static facts a prepared Worker session reported
  (§47 steps 1-3 happen Worker-side; planning consumes facts, never a
  live model, §46) and the per-device reuse answer from the store (§28),
  which cases should run?
* *network*: given the cluster's Phase 1 network facts, which endpoint
  profiles, path classifications, and probe cases should run (§30-§34)?

Plans are pure data: canonical ``ProfilingCase`` values the Master
``ProfilingController`` turns into an experiment (§40). §48's
GranularityPolicy/SamplingPolicy/MeasurementPolicy/StoppingPolicy
abstractions are deliberately *not* separate interfaces in v1 — the
default strategy hard-codes the composition as documented constants, and
replaceability lives at this protocol boundary instead.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Protocol

from edgeshard.profiling.domain.experiment import ProfilingCase
from edgeshard.profiling.domain.network import (
    NetworkEndpointProfile,
    NetworkPair,
    NetworkPathClass,
)
from edgeshard.profiling.domain.session import ModelSessionFacts
from edgeshard.profiling.domain.signature import OperatorSignature
from edgeshard.profiling.network.classifier import ClassifiedPair, WorkerNetworkFacts
from edgeshard.profiling.operator.planning import IncrementalPlan


@dataclass(frozen=True)
class ModelProfilingPlan:
    """Planned §47 model-workflow cases for one worker.

    ``operator_reuse`` maps every planned device to its §28 split (which
    signatures already have compatible measurements and are skipped, which
    are missing and become cases) so callers can report reuse honestly.
    ``deferred_operator_signatures`` are preserved-but-not-benchmarked
    signatures (custom operators §19, kinds outside the v1 workload
    vocabulary §25) — visible, never silently dropped.
    """

    worker_id: str
    cases: tuple[ProfilingCase, ...]
    operator_reuse: Mapping[str, IncrementalPlan]
    deferred_operator_signatures: tuple[OperatorSignature, ...]

    def __post_init__(self) -> None:
        if not self.worker_id:
            raise ValueError("worker_id must not be empty")
        misplaced = sorted(
            {case.worker_id for case in self.cases if case.worker_id != self.worker_id}
        )
        if misplaced:
            raise ValueError(
                f"model plan cases must be assigned to worker {self.worker_id!r}; "
                f"found {misplaced}"
            )


@dataclass(frozen=True)
class NetworkProfilingPlan:
    """Planned §47 network-workflow facts and probe cases.

    ``endpoint_profiles`` (step 1) and ``classified_pairs`` (step 2) are
    returned alongside the cases so the caller can persist the static
    characterization (§43 registries) without recomputing it; ``cases``
    holds the dense RTT matrix first, then the sparse bandwidth selection
    (steps 3-5).
    """

    endpoint_profiles: tuple[NetworkEndpointProfile, ...]
    classified_pairs: tuple[ClassifiedPair, ...]
    cases: tuple[ProfilingCase, ...]


class ProfilingStrategy(Protocol):
    """Structural interface every profiling strategy satisfies (§48).

    Implementations must be pure planners: no gRPC, no store writes, no
    torch, no benchmarking — facts in, canonical cases out (§40, §46).
    """

    @property
    def strategy_id(self) -> str:
        """Stable identity that goes into every experiment id (§7)."""
        ...

    def plan_model_cases(
        self,
        *,
        worker_id: str,
        facts: ModelSessionFacts,
        device_ids: Sequence[str],
        measured_signature_ids: Mapping[str, AbstractSet[str]] | None = None,
    ) -> ModelProfilingPlan:
        """Plan the §47 model workflow for one worker's prepared facts.

        ``measured_signature_ids`` maps device id → operator signature ids
        that already have compatible measurements for that device's
        performance class/environment (§28); a device without an entry is
        planned from scratch (the conservative direction — more
        measurement, never a guessed reuse).
        """
        ...

    def plan_network_cases(
        self,
        *,
        facts: Iterable[WorkerNetworkFacts],
        extra_bandwidth_pairs: Iterable[NetworkPair] = (),
        bandwidth_path_classes: Iterable[NetworkPathClass] | None = None,
    ) -> NetworkProfilingPlan:
        """Plan the §47 network workflow over the cluster's facts.

        ``extra_bandwidth_pairs`` is the §34 explicit-pair knob;
        ``bandwidth_path_classes`` restricts the sparse class-driven
        bandwidth selection (the §49 ``--path-class`` knob) without
        touching the dense RTT matrix.
        """
        ...


__all__ = [
    "ModelProfilingPlan",
    "NetworkProfilingPlan",
    "ProfilingStrategy",
]
