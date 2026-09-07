"""ProfileStore repository interface (spec §43, §46).

The store is the persistence *boundary*: domain objects go in, domain
objects come out, and the storage engine (SQLite in v1, §43) never leaks
into the domain or the controller. Every identity the store maintains is
a canonical §7 SHA-256 id computed by the domain's own id functions — the
store invents no identity scheme of its own.

Append-oriented discipline (§44): measurements are never updated or
overwritten; a changed environment produces a *new* fingerprint id and
new records under it. Appending an already-known id is an idempotent
no-op returning ``False`` (duplicate results from retried dispatches are
expected, not errors), while the same canonical id mapping to different
content is a loud failure — that would mean the hashing contract broke.

The reuse-query side (``measured_operator_signature_ids``) implements the
P2E :class:`~edgeshard.profiling.operator.profiler.MeasuredSignatureIndex`
seam: filtering by device performance class / environment fingerprint is
a store concern (§28), planning consumes only the id set.
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from edgeshard.model.errors import EdgeShardError
from edgeshard.profiling.domain.environment import (
    DevicePerformanceClass,
    DevicePerformanceClassMembership,
    EnvironmentFingerprint,
)
from edgeshard.profiling.domain.experiment import (
    CaseState,
    ExperimentState,
    ProfilingCase,
    ProfilingExperiment,
    ProfilingFailure,
)
from edgeshard.profiling.domain.measurement import MeasurementRecord
from edgeshard.profiling.domain.model import ModelCharacterization
from edgeshard.profiling.domain.network import (
    NetworkEndpointProfile,
    NetworkPair,
    NetworkPathClass,
    ProbeKind,
)
from edgeshard.profiling.domain.signature import (
    ModuleSignature,
    OperatorSignature,
    TransformerLayerSignature,
)
from edgeshard.profiling.domain.snapshot import ProfileSnapshot


class ProfileStoreError(EdgeShardError):
    """A persistence operation failed (unknown id, conflicting payload, I/O)."""


@dataclass(frozen=True)
class StoredCase:
    """A persisted case plus its Master-side lifecycle state (§40)."""

    case: ProfilingCase
    state: CaseState
    failure: ProfilingFailure | None = None


@dataclass(frozen=True)
class StoredExperiment:
    """A persisted experiment definition plus its lifecycle state (§40)."""

    experiment: ProfilingExperiment
    state: ExperimentState


class ProfileStore(Protocol):
    """Repository interface for profiling persistence (spec §43)."""

    # -- measurements (§43-45) ---------------------------------------------

    def append_case(self, case: ProfilingCase) -> bool:
        """Persist a case definition; ``False`` when the id already exists."""

    def update_case_state(
        self,
        case_id: str,
        state: CaseState,
        failure: ProfilingFailure | None = None,
    ) -> None:
        """Set lifecycle state and preserve any typed terminal failure."""

    def get_case(self, case_id: str) -> StoredCase | None:
        """One persisted case with its state, or ``None`` when unknown."""

    def append_measurement(self, record: MeasurementRecord) -> bool:
        """Append one measurement; ``False`` when the id already exists.

        The measurement's case must already be persisted — measurements
        never orphan from their case definition (§8.3).
        """

    def get_measurement(self, measurement_id: str) -> MeasurementRecord | None:
        """One persisted measurement, or ``None`` when unknown."""

    def query_measurements(
        self,
        *,
        case_id: str | None = None,
        environment_fingerprint_id: str | None = None,
        model_id: str | None = None,
        revision: str | None = None,
        layer_signature_id: str | None = None,
        module_signature_id: str | None = None,
        operator_signature_id: str | None = None,
        network_pair_id: str | None = None,
        probe_kind: ProbeKind | None = None,
    ) -> tuple[MeasurementRecord, ...]:
        """Measurements matching every given filter (AND), oldest first."""

    # -- experiments (§40, §43) ---------------------------------------------

    def append_experiment(self, experiment: ProfilingExperiment) -> bool:
        """Persist an experiment definition; ``False`` when already known."""

    def update_experiment_state(
        self, experiment_id: str, state: ExperimentState
    ) -> None:
        """Set the lifecycle state of a persisted experiment."""

    def get_experiment(self, experiment_id: str) -> StoredExperiment | None:
        """One persisted experiment with its state, or ``None`` when unknown."""

    # -- canonical-id registries (§44) ---------------------------------------

    def store_characterization(self, characterization: ModelCharacterization) -> str:
        """Persist a characterization; returns its canonical id (§7)."""

    def store_layer_signature(self, signature: TransformerLayerSignature) -> str:
        """Persist a transformer-layer signature; returns its canonical id."""

    def store_module_signature(self, signature: ModuleSignature) -> str:
        """Persist a module signature; returns its canonical id."""

    def store_operator_signature(self, signature: OperatorSignature) -> str:
        """Persist an operator signature; returns its canonical id."""

    def store_network_endpoint(self, profile: NetworkEndpointProfile) -> tuple[str, str]:
        """Persist an endpoint profile; returns its (worker_id, interface_id)."""

    def store_path_classification(
        self, pair: NetworkPair, path_class: NetworkPathClass
    ) -> str:
        """Persist a directed pair and its classification; returns the pair id."""

    def store_environment_fingerprint(self, fingerprint: EnvironmentFingerprint) -> str:
        """Persist an environment fingerprint; returns its canonical id (§9)."""

    def store_performance_class(self, performance_class: DevicePerformanceClass) -> str:
        """Persist a device performance class; returns its canonical id (§9)."""

    # -- reuse queries (§28) and snapshot (§46) -------------------------------

    def store_performance_class_membership(
        self, membership: DevicePerformanceClassMembership
    ) -> None:
        """Persist a physical device's verified or pending class membership."""

    def measured_operator_signature_ids_for_environment(
        self, fingerprint: EnvironmentFingerprint
    ) -> AbstractSet[str]:
        """Return reuse licensed for one physical environment."""

    def build_snapshot(
        self, snapshot_id: str, *, created_at: datetime | None = None
    ) -> ProfileSnapshot:
        """Frozen view of everything empirical the store holds (§46)."""


@dataclass(frozen=True)
class OperatorSignatureIndex:
    """Scoped reuse index over a store (P2E ``MeasuredSignatureIndex`` seam).

    Planning code receives this instead of the raw store so it depends only
    on the id-set answer, never on persistence details (§28, §43).
    """

    store: ProfileStore
    environment: EnvironmentFingerprint

    def measured_signature_ids(self) -> AbstractSet[str]:
        return self.store.measured_operator_signature_ids_for_environment(
            self.environment
        )
