"""SQLite ProfileStore tests (spec §43-§46, §51 "persistence mapping").

Pinned: append-oriented idempotency (duplicate ids return ``False``, never
overwrite), the §7 conflict rule (same canonical id + different content
fails loudly), case-before-measurement ordering, query filters over the
derived index columns, the P2E reuse-index seam (scoped by fingerprint /
performance class), snapshot splitting (network vs model measurements),
and durability across reopen (Master-restart resume relies on it). Tests
may peek at raw SQLite rows — they verify the implementation's own
mapping; production code never sees SQL (§43).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from edgeshard.profiling.domain.environment import (
    DevicePerformanceClass,
    EnvironmentFingerprint,
    MemoryModel,
    device_performance_class_id,
    environment_fingerprint_id,
)
from edgeshard.profiling.domain.experiment import (
    CaseState,
    ExperimentState,
    ModelCaseSpec,
    NetworkCaseSpec,
    ProfilingCase,
    ProfilingExperiment,
)
from edgeshard.profiling.domain.measurement import (
    BandwidthMetrics,
    LatencyMetrics,
    MeasurementMetrics,
    MeasurementRecord,
    RttMetrics,
    TimeUnit,
    summarize_samples,
)
from edgeshard.profiling.domain.model import (
    ModelCharacterization,
    ModelReference,
    ModelStage,
    StageKind,
    model_signature_id,
)
from edgeshard.profiling.domain.network import (
    NetworkDirection,
    NetworkEndpointProfile,
    NetworkPair,
    NetworkPathClass,
    NetworkTransport,
    ProbeKind,
    network_pair_signature_id,
)
from edgeshard.profiling.domain.signature import (
    GemmSignature,
    ModuleKind,
    ModuleSignature,
    OperatorKind,
    OperatorSignature,
    ProfilingGranularity,
    TransformerLayerSignature,
    module_signature_id,
    operator_signature_id,
    transformer_layer_signature_id,
)
from edgeshard.profiling.operator.profiler import plan_incremental_profiling
from edgeshard.profiling.store.base import (
    OperatorSignatureIndex,
    ProfileStoreError,
)
from edgeshard.profiling.store.sqlite import SqliteProfileStore

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)
LATER = datetime(2026, 9, 7, 12, 5, tzinfo=UTC)

MODEL = ModelReference(model_id="tiny/llama", revision="local")
LAYER_SIG = TransformerLayerSignature(
    architecture_family="llama",
    layer_type="standard_decoder",
    hidden_size=32,
    intermediate_size=64,
    num_attention_heads=4,
    num_kv_heads=2,
    head_dim=8,
    dtype="fp32",
    quantization=None,
)
MODULE_SIG = ModuleSignature(
    kind=ModuleKind.MLP,
    architecture_family="llama",
    structural_parameters=(("hidden_size", 32),),
    dtype="fp32",
    quantization=None,
)
OP_SIG = OperatorSignature(
    kind=OperatorKind.GEMM,
    parameters=GemmSignature(m=8, n=8, k=8, dtype="fp32"),
    backend_family="torch",
)
OTHER_OP_SIG = OperatorSignature(
    kind=OperatorKind.GEMM,
    parameters=GemmSignature(m=16, n=16, k=16, dtype="fp32"),
    backend_family="torch",
)
CHARACTERIZATION = ModelCharacterization(
    model=MODEL,
    architecture_family="llama",
    num_layers=2,
    hidden_size=32,
    intermediate_size=64,
    vocab_size=64,
    num_attention_heads=4,
    num_kv_heads=2,
    head_dim=8,
    dtype="fp32",
    quantization=None,
    tied_word_embeddings=False,
    stages=(
        ModelStage(kind=StageKind.EMBEDDING),
        ModelStage(kind=StageKind.TRANSFORMER_LAYER_GROUP, layer_count=2),
        ModelStage(kind=StageKind.LM_HEAD),
    ),
)
PERFORMANCE_CLASS = DevicePerformanceClass(
    vendor="nvidia",
    accelerator_model="rtx4090",
    memory_model=MemoryModel.DISCRETE,
    backend_family="torch",
)


def _fingerprint(device_id: str = "gpu-0", **overrides: object) -> EnvironmentFingerprint:
    base: dict[str, object] = {
        "backend": "torch",
        "profiling_implementation_revision": "r1",
        "worker_id": "w-1",
        "device_id": device_id,
    }
    base.update(overrides)
    return EnvironmentFingerprint(**base)  # type: ignore[arg-type]


def _operator_case(signature: OperatorSignature = OP_SIG) -> ProfilingCase:
    return ProfilingCase.for_spec(
        "w-1",
        ModelCaseSpec(
            granularity=ProfilingGranularity.OPERATOR,
            device_ids=("gpu-0",),
            dtype="fp32",
            operator_signature=signature,
        ),
    )


def _layer_case(layer_index: int | None = None) -> ProfilingCase:
    return ProfilingCase.for_spec(
        "w-1",
        ModelCaseSpec(
            granularity=ProfilingGranularity.TRANSFORMER_LAYER,
            device_ids=("gpu-0",),
            dtype="fp32",
            model=MODEL,
            layer_signature=LAYER_SIG,
            layer_index=layer_index,
        ),
    )


def _module_case() -> ProfilingCase:
    return ProfilingCase.for_spec(
        "w-1",
        ModelCaseSpec(
            granularity=ProfilingGranularity.MODULE,
            device_ids=("gpu-0",),
            dtype="fp32",
            model=MODEL,
            module_signature=MODULE_SIG,
        ),
    )


def _rtt_case() -> ProfilingCase:
    return ProfilingCase.for_spec(
        "w-1",
        NetworkCaseSpec(
            probe_kind=ProbeKind.RTT,
            source_worker_id="w-1",
            destination_worker_id="w-2",
            packet_count=5,
        ),
    )


def _bandwidth_case() -> ProfilingCase:
    return ProfilingCase.for_spec(
        "w-1",
        NetworkCaseSpec(
            probe_kind=ProbeKind.BANDWIDTH,
            source_worker_id="w-1",
            destination_worker_id="w-2",
            path_class=NetworkPathClass.WIRED_LAN,
            transport=NetworkTransport.TCP,
            direction=NetworkDirection.FORWARD,
            duration_s=5.0,
        ),
    )


def _latency_record(
    case: ProfilingCase,
    measurement_id: str,
    fingerprint: str = "fp-1",
    started_at: datetime = NOW,
) -> MeasurementRecord:
    return MeasurementRecord(
        measurement_id=measurement_id,
        case_id=case.case_id,
        environment_fingerprint=fingerprint,
        started_at=started_at,
        finished_at=started_at,
        sample_count=2,
        samples=(1.0, 2.0),
        metrics=MeasurementMetrics(
            latency=LatencyMetrics(
                summary=summarize_samples((1.0, 2.0)), unit=TimeUnit.MILLISECONDS
            )
        ),
    )


def _rtt_record(case: ProfilingCase, measurement_id: str) -> MeasurementRecord:
    return MeasurementRecord(
        measurement_id=measurement_id,
        case_id=case.case_id,
        environment_fingerprint="fp-net",
        started_at=LATER,
        finished_at=LATER,
        sample_count=5,
        samples=(0.4, 0.5, 0.6, 0.7, 0.8),
        metrics=MeasurementMetrics(
            rtt=RttMetrics(
                summary=summarize_samples((0.4, 0.5, 0.6, 0.7, 0.8)),
                unit=TimeUnit.MILLISECONDS,
                packets_sent=5,
                packets_received=5,
            )
        ),
    )


def _bandwidth_record(case: ProfilingCase, measurement_id: str) -> MeasurementRecord:
    return MeasurementRecord(
        measurement_id=measurement_id,
        case_id=case.case_id,
        environment_fingerprint="fp-net",
        started_at=LATER,
        finished_at=LATER,
        sample_count=1,
        samples=(9.3e9,),
        metrics=MeasurementMetrics(
            bandwidth=BandwidthMetrics(bits_per_second=9.3e9, duration_s=5.0)
        ),
    )


def _experiment(experiment_id: str = "exp-1", *case_ids: str) -> ProfilingExperiment:
    return ProfilingExperiment(
        experiment_id=experiment_id,
        strategy_id="default-v1",
        created_at=NOW,
        requested_by=None,
        case_ids=case_ids,
    )


@pytest.fixture
def store(tmp_path: Path) -> SqliteProfileStore:
    with SqliteProfileStore(tmp_path / "profile.db") as open_store:
        yield open_store


def _rows(store: SqliteProfileStore, sql: str) -> list[sqlite3.Row]:
    """Test-only raw peek at the implementation's own tables."""
    return store._conn.execute(sql).fetchall()


class TestCases:
    def test_append_is_idempotent_and_stateful(self, store: SqliteProfileStore) -> None:
        case = _operator_case()
        assert store.append_case(case) is True
        assert store.append_case(case) is False  # duplicate dispatch result
        stored = store.get_case(case.case_id)
        assert stored is not None
        assert stored.case == case
        assert stored.state is CaseState.PENDING
        store.update_case_state(case.case_id, CaseState.RUNNING)
        assert store.get_case(case.case_id).state is CaseState.RUNNING  # type: ignore[union-attr]

    def test_unknown_case_accessors(self, store: SqliteProfileStore) -> None:
        assert store.get_case("missing") is None
        with pytest.raises(ProfileStoreError, match="unknown profiling case"):
            store.update_case_state("missing", CaseState.COMPLETED)


class TestMeasurements:
    def test_requires_known_case(self, store: SqliteProfileStore) -> None:
        record = _latency_record(_operator_case(), "m-1")
        with pytest.raises(ProfileStoreError, match="unknown case"):
            store.append_measurement(record)

    def test_append_get_roundtrip(self, store: SqliteProfileStore) -> None:
        case = _operator_case()
        store.append_case(case)
        record = _latency_record(case, "m-1")
        assert store.append_measurement(record) is True
        assert store.get_measurement("m-1") == record
        assert store.get_measurement("missing") is None

    def test_duplicate_measurement_is_false_never_overwrite(
        self, store: SqliteProfileStore
    ) -> None:
        case = _operator_case()
        store.append_case(case)
        record = _latency_record(case, "m-1")
        assert store.append_measurement(record) is True
        assert store.append_measurement(record) is False
        assert len(_rows(store, "SELECT * FROM measurements")) == 1

    def test_same_id_different_content_fails_loud(
        self, store: SqliteProfileStore
    ) -> None:
        """§7: a canonical id mapping to different content is a contract break."""
        case = _operator_case()
        store.append_case(case)
        store.append_measurement(_latency_record(case, "m-1"))
        conflicting = _latency_record(case, "m-1", fingerprint="fp-other")
        with pytest.raises(ProfileStoreError, match="different content"):
            store.append_measurement(conflicting)

    def test_samples_stored_row_per_sample(self, store: SqliteProfileStore) -> None:
        """§45: all samples are stored; counts are small, no Parquet."""
        case = _operator_case()
        store.append_case(case)
        store.append_measurement(_latency_record(case, "m-1"))
        rows = _rows(
            store,
            "SELECT sample_index, value FROM measurement_samples "
            "WHERE measurement_id = 'm-1' ORDER BY sample_index",
        )
        assert [(row["sample_index"], row["value"]) for row in rows] == [
            (0, 1.0),
            (1, 2.0),
        ]

    def test_query_filters_and_combination(self, store: SqliteProfileStore) -> None:
        op_case = _operator_case()
        layer_case = _layer_case()
        module_case = _module_case()
        rtt_case = _rtt_case()
        for case in (op_case, layer_case, module_case, rtt_case):
            store.append_case(case)
        fp_id = store.store_environment_fingerprint(_fingerprint())
        op_record = _latency_record(op_case, "m-op", fp_id)
        layer_record = _latency_record(layer_case, "m-layer", fp_id, started_at=LATER)
        module_record = _latency_record(module_case, "m-module", "fp-other")
        rtt_record = _rtt_record(rtt_case, "m-rtt")
        for record in (op_record, layer_record, module_record, rtt_record):
            store.append_measurement(record)

        assert store.query_measurements(case_id=op_case.case_id) == (op_record,)
        assert store.query_measurements(environment_fingerprint_id=fp_id) == (
            op_record,
            layer_record,
        )  # oldest first
        assert store.query_measurements(model_id=MODEL.model_id) == (
            module_record,  # NOW, then LATER — ordered by (started_at, id)
            layer_record,
        )
        assert store.query_measurements(
            model_id=MODEL.model_id, revision="other"
        ) == ()
        assert store.query_measurements(
            layer_signature_id=transformer_layer_signature_id(LAYER_SIG)
        ) == (layer_record,)
        assert store.query_measurements(
            module_signature_id=module_signature_id(MODULE_SIG)
        ) == (module_record,)
        assert store.query_measurements(
            operator_signature_id=operator_signature_id(OP_SIG)
        ) == (op_record,)
        pair_id = network_pair_signature_id(
            NetworkPair(source_worker_id="w-1", destination_worker_id="w-2")
        )
        assert store.query_measurements(network_pair_id=pair_id) == (rtt_record,)
        assert store.query_measurements(probe_kind=ProbeKind.RTT) == (rtt_record,)
        assert store.query_measurements(probe_kind=ProbeKind.BANDWIDTH) == ()
        # AND combination
        assert store.query_measurements(
            environment_fingerprint_id=fp_id,
            operator_signature_id=operator_signature_id(OP_SIG),
        ) == (op_record,)
        assert store.query_measurements() == (
            module_record,  # NOW: m-module < m-op
            op_record,
            layer_record,  # LATER: m-layer < m-rtt
            rtt_record,
        )

    def test_corrupt_payload_fails_typed(self, store: SqliteProfileStore) -> None:
        case = _operator_case()
        store.append_case(case)
        store.append_measurement(_latency_record(case, "m-1"))
        store._conn.execute(
            "UPDATE measurements SET payload = 'garbage' WHERE measurement_id = 'm-1'"
        )
        with pytest.raises(ProfileStoreError, match="corrupt"):
            store.get_measurement("m-1")


class TestExperiments:
    def test_append_update_get(self, store: SqliteProfileStore) -> None:
        experiment = _experiment("exp-1", "c-1", "c-2")
        assert store.append_experiment(experiment) is True
        assert store.append_experiment(experiment) is False
        stored = store.get_experiment("exp-1")
        assert stored is not None
        assert stored.experiment == experiment
        assert stored.state is ExperimentState.PENDING
        store.update_experiment_state("exp-1", ExperimentState.PARTIALLY_COMPLETED)
        stored = store.get_experiment("exp-1")
        assert stored is not None and stored.state is ExperimentState.PARTIALLY_COMPLETED

    def test_unknown_experiment(self, store: SqliteProfileStore) -> None:
        assert store.get_experiment("missing") is None
        with pytest.raises(ProfileStoreError, match="unknown experiment"):
            store.update_experiment_state("missing", ExperimentState.COMPLETED)


class TestRegistries:
    def test_canonical_ids_and_idempotency(self, store: SqliteProfileStore) -> None:
        assert store.store_characterization(CHARACTERIZATION) == model_signature_id(
            CHARACTERIZATION
        )
        assert store.store_layer_signature(LAYER_SIG) == (
            transformer_layer_signature_id(LAYER_SIG)
        )
        assert store.store_module_signature(MODULE_SIG) == module_signature_id(
            MODULE_SIG
        )
        assert store.store_operator_signature(OP_SIG) == operator_signature_id(OP_SIG)
        fp = _fingerprint()
        assert store.store_environment_fingerprint(fp) == environment_fingerprint_id(fp)
        assert store.store_performance_class(PERFORMANCE_CLASS) == (
            device_performance_class_id(PERFORMANCE_CLASS)
        )
        # idempotent re-store keeps one row each
        store.store_characterization(CHARACTERIZATION)
        store.store_operator_signature(OP_SIG)
        assert len(_rows(store, "SELECT * FROM model_characterizations")) == 1
        assert len(_rows(store, "SELECT * FROM operator_signatures")) == 1

    def test_same_id_different_content_fails_loud(
        self, store: SqliteProfileStore
    ) -> None:
        """Simulated §7 hash collision: the store refuses to paper over it."""
        signature_id = store.store_operator_signature(OP_SIG)
        store._conn.execute(
            "UPDATE operator_signatures SET payload = '{}' "
            "WHERE operator_signature_id = ?",
            (signature_id,),
        )
        with pytest.raises(ProfileStoreError, match="different content"):
            store.store_operator_signature(OP_SIG)

    def test_endpoint_facts_are_last_write_wins(
        self, store: SqliteProfileStore
    ) -> None:
        """Endpoints reference rediscoverable Phase 1 facts, not hashed content."""
        key = ("w-1", "if-eth0")
        assert store.store_network_endpoint(
            NetworkEndpointProfile(worker_id="w-1", interface_id="if-eth0", mtu=1500)
        ) == key
        assert store.store_network_endpoint(
            NetworkEndpointProfile(
                worker_id="w-1", interface_id="if-eth0", mtu=9000, link_speed_mbps=1e4
            )
        ) == key
        rows = _rows(store, "SELECT mtu, link_speed_mbps FROM network_endpoints")
        assert len(rows) == 1
        assert rows[0]["mtu"] == 9000
        assert rows[0]["link_speed_mbps"] == 1e4

    def test_pair_classification_evolves_identity_does_not(
        self, store: SqliteProfileStore
    ) -> None:
        """§32: reclassification must not change which pair measurements belong to."""
        pair = NetworkPair(source_worker_id="w-1", destination_worker_id="w-2")
        pair_id = store.store_path_classification(pair, NetworkPathClass.WIRED_LAN)
        assert pair_id == network_pair_signature_id(pair)
        assert store.store_path_classification(pair, NetworkPathClass.OVERLAY) == pair_id
        rows = _rows(
            store, "SELECT path_class, payload FROM network_pairs"
        )
        assert len(rows) == 1
        assert rows[0]["path_class"] == "overlay"
        classes = {
            row["path_class"] for row in _rows(store, "SELECT * FROM network_path_classes")
        }
        assert classes == {"wired_lan", "overlay"}


class TestReuseIndex:
    def test_measured_operator_signature_ids(self, store: SqliteProfileStore) -> None:
        assert store.measured_operator_signature_ids() == frozenset()
        dpc_id = store.store_performance_class(PERFORMANCE_CLASS)
        fp = _fingerprint(device_performance_class_id=dpc_id)
        fp_id = store.store_environment_fingerprint(fp)
        other_fp_id = store.store_environment_fingerprint(_fingerprint(device_id="gpu-1"))

        op_case = _operator_case(OP_SIG)
        other_case = _operator_case(OTHER_OP_SIG)
        store.append_case(op_case)
        store.append_case(other_case)
        store.append_measurement(_latency_record(op_case, "m-1", fp_id))
        store.append_measurement(_latency_record(other_case, "m-2", other_fp_id))

        op_id = operator_signature_id(OP_SIG)
        other_id = operator_signature_id(OTHER_OP_SIG)
        assert store.measured_operator_signature_ids() == frozenset({op_id, other_id})
        assert store.measured_operator_signature_ids(
            environment_fingerprint_id=fp_id
        ) == frozenset({op_id})
        assert store.measured_operator_signature_ids(
            device_performance_class_id=dpc_id
        ) == frozenset({op_id})
        assert store.measured_operator_signature_ids(
            device_performance_class_id="dpc-never-seen"
        ) == frozenset()

    def test_index_adapter_feeds_incremental_planning(
        self, store: SqliteProfileStore
    ) -> None:
        """P2E seam: the scoped index drives missing-only planning (§28)."""
        fp_id = store.store_environment_fingerprint(_fingerprint())
        op_case = _operator_case(OP_SIG)
        store.append_case(op_case)
        store.append_measurement(_latency_record(op_case, "m-1", fp_id))
        index = OperatorSignatureIndex(
            store, environment_fingerprint_id=fp_id
        )
        plan = plan_incremental_profiling(
            [OP_SIG, OTHER_OP_SIG], measured_ids=index.measured_signature_ids()
        )
        assert plan.reused == (OP_SIG,)
        assert plan.missing == (OTHER_OP_SIG,)


class TestSnapshot:
    def test_snapshot_splits_network_and_model_measurements(
        self, store: SqliteProfileStore
    ) -> None:
        store.store_characterization(CHARACTERIZATION)
        op_case = _operator_case()
        rtt_case = _rtt_case()
        bandwidth_case = _bandwidth_case()
        for case in (op_case, rtt_case, bandwidth_case):
            store.append_case(case)
        op_record = _latency_record(op_case, "m-op")
        rtt_record = _rtt_record(rtt_case, "m-rtt")
        bandwidth_record = _bandwidth_record(bandwidth_case, "m-bw")
        for record in (op_record, rtt_record, bandwidth_record):
            store.append_measurement(record)

        snapshot = store.build_snapshot("snap-1", created_at=LATER)
        assert snapshot.snapshot_id == "snap-1"
        assert snapshot.created_at == LATER
        assert snapshot.model_characterizations == (CHARACTERIZATION,)
        assert snapshot.measurements == (op_record,)
        # both network records share started_at → ordered by measurement_id
        assert snapshot.network_measurements == (bandwidth_record, rtt_record)

    def test_snapshot_default_clock(self, store: SqliteProfileStore) -> None:
        frozen = datetime(2030, 1, 1, tzinfo=UTC)
        store._clock = lambda: frozen
        snapshot = store.build_snapshot("snap-1")
        assert snapshot.created_at == frozen

    def test_empty_store_snapshot(self, store: SqliteProfileStore) -> None:
        snapshot = store.build_snapshot("snap-empty", created_at=NOW)
        assert snapshot.model_characterizations == ()
        assert snapshot.measurements == ()
        assert snapshot.network_measurements == ()


class TestDurability:
    def test_data_survives_reopen(self, tmp_path: Path) -> None:
        """Master-restart resume (§40 DoD) reads experiments/cases from disk."""
        path = tmp_path / "profile.db"
        case = _operator_case()
        with SqliteProfileStore(path) as first:
            first.append_experiment(_experiment("exp-1", case.case_id))
            first.append_case(case)
            first.append_measurement(_latency_record(case, "m-1"))
            first.update_case_state(case.case_id, CaseState.COMPLETED)
            first.update_experiment_state("exp-1", ExperimentState.RUNNING)
        with SqliteProfileStore(path) as second:
            stored_case = second.get_case(case.case_id)
            assert stored_case is not None
            assert stored_case.state is CaseState.COMPLETED
            stored_experiment = second.get_experiment("exp-1")
            assert stored_experiment is not None
            assert stored_experiment.state is ExperimentState.RUNNING
            assert second.get_measurement("m-1") is not None
            assert second.measured_operator_signature_ids() == frozenset(
                {operator_signature_id(OP_SIG)}
            )

    def test_empty_path_rejected(self) -> None:
        with pytest.raises(ValueError, match="path"):
            SqliteProfileStore("")
