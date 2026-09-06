"""SQLite v1 ProfileStore (spec §43-§46).

Layout: every §44 table carries canonical *indexed columns* (ids and the
few facts worth querying directly) plus one ``payload`` column holding the
full domain object as deterministic codec JSON. The payload is the source
of truth for reads — indexed columns are derived, so decoding a row always
rebuilds the exact domain object (with its ``__post_init__`` validation),
and no half-mapped column set can drift from the domain shape.

Discipline enforced here:

* append-oriented measurements (§44): rows are inserted, never updated;
  a changed environment means a new fingerprint id and new rows.
  Re-appending a known ``measurement_id`` is an idempotent ``False``
  (duplicate dispatch results are expected); a known canonical id mapping
  to *different* payload content fails loudly — that would break §7;
* content-hashed registries (signatures, characterizations, fingerprints,
  performance classes) deduplicate by canonical id; the two naturally
  evolving tables — ``network_endpoints`` (Phase 1 facts get rediscovered)
  and ``network_pairs.path_class`` (reclassification must not change pair
  identity, §32) — are last-write-wins upserts by design;
* every sample series is stored row-per-sample (§45, small counts, no
  Parquet) alongside the summary the record already carries;
* SQL never leaves this module (§43).
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from edgeshard.profiling.codec import PayloadCodecError, decode_json, encode_json
from edgeshard.profiling.domain.environment import (
    DevicePerformanceClass,
    EnvironmentFingerprint,
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
from edgeshard.profiling.domain.measurement import MeasurementRecord
from edgeshard.profiling.domain.model import ModelCharacterization, model_signature_id
from edgeshard.profiling.domain.network import (
    NetworkEndpointProfile,
    NetworkPair,
    NetworkPathClass,
    ProbeKind,
    network_pair_signature_id,
)
from edgeshard.profiling.domain.signature import (
    ModuleSignature,
    OperatorSignature,
    TransformerLayerSignature,
    module_signature_id,
    operator_signature_id,
    transformer_layer_signature_id,
)
from edgeshard.profiling.domain.snapshot import ProfileSnapshot
from edgeshard.profiling.store.base import (
    ProfileStoreError,
    StoredCase,
    StoredExperiment,
)

_SqlValue = str | int | float | None

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id  TEXT PRIMARY KEY,
    strategy_id    TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    requested_by   TEXT,
    state          TEXT NOT NULL,
    payload        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS profiling_cases (
    case_id  TEXT PRIMARY KEY,
    worker_id TEXT NOT NULL,
    kind     TEXT NOT NULL,
    state    TEXT NOT NULL,
    payload  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS measurements (
    measurement_id           TEXT PRIMARY KEY,
    case_id                  TEXT NOT NULL REFERENCES profiling_cases(case_id),
    environment_fingerprint_id TEXT NOT NULL,
    model_id                 TEXT,
    model_revision           TEXT,
    layer_signature_id       TEXT,
    module_signature_id      TEXT,
    operator_signature_id    TEXT,
    network_pair_id          TEXT,
    probe_kind               TEXT,
    started_at               TEXT NOT NULL,
    finished_at              TEXT NOT NULL,
    sample_count             INTEGER NOT NULL,
    payload                  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS measurement_samples (
    measurement_id TEXT NOT NULL REFERENCES measurements(measurement_id),
    sample_index   INTEGER NOT NULL,
    value          REAL NOT NULL,
    PRIMARY KEY (measurement_id, sample_index)
);
CREATE TABLE IF NOT EXISTS model_characterizations (
    model_signature_id  TEXT PRIMARY KEY,
    model_id            TEXT NOT NULL,
    revision            TEXT NOT NULL,
    architecture_family TEXT NOT NULL,
    num_layers          INTEGER NOT NULL,
    payload             TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS transformer_layer_signatures (
    layer_signature_id  TEXT PRIMARY KEY,
    architecture_family TEXT NOT NULL,
    layer_type          TEXT NOT NULL,
    dtype               TEXT NOT NULL,
    payload             TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS module_signatures (
    module_signature_id TEXT PRIMARY KEY,
    kind                TEXT NOT NULL,
    architecture_family TEXT NOT NULL,
    dtype               TEXT NOT NULL,
    payload             TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS operator_signatures (
    operator_signature_id TEXT PRIMARY KEY,
    kind                  TEXT NOT NULL,
    backend_family        TEXT NOT NULL,
    payload               TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS network_endpoints (
    worker_id       TEXT NOT NULL,
    interface_id    TEXT NOT NULL,
    mtu             INTEGER,
    link_speed_mbps REAL,
    interface_type  TEXT,
    overlay_type    TEXT,
    payload         TEXT NOT NULL,
    PRIMARY KEY (worker_id, interface_id)
);
CREATE TABLE IF NOT EXISTS network_path_classes (
    path_class TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS network_pairs (
    network_pair_id          TEXT PRIMARY KEY,
    source_worker_id         TEXT NOT NULL,
    destination_worker_id    TEXT NOT NULL,
    source_interface_id      TEXT,
    destination_interface_id TEXT,
    path_class               TEXT REFERENCES network_path_classes(path_class),
    payload                  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS environment_fingerprints (
    environment_fingerprint_id  TEXT PRIMARY KEY,
    backend                     TEXT NOT NULL,
    worker_id                   TEXT,
    device_id                   TEXT,
    device_performance_class_id TEXT,
    payload                     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS device_performance_classes (
    device_performance_class_id TEXT PRIMARY KEY,
    vendor                      TEXT NOT NULL,
    accelerator_model           TEXT NOT NULL,
    memory_model                TEXT NOT NULL,
    backend_family              TEXT NOT NULL,
    payload                     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_measurements_case ON measurements(case_id);
CREATE INDEX IF NOT EXISTS ix_measurements_fingerprint
    ON measurements(environment_fingerprint_id);
CREATE INDEX IF NOT EXISTS ix_measurements_model
    ON measurements(model_id, model_revision);
CREATE INDEX IF NOT EXISTS ix_measurements_layer ON measurements(layer_signature_id);
CREATE INDEX IF NOT EXISTS ix_measurements_module ON measurements(module_signature_id);
CREATE INDEX IF NOT EXISTS ix_measurements_operator
    ON measurements(operator_signature_id);
CREATE INDEX IF NOT EXISTS ix_measurements_pair ON measurements(network_pair_id);
"""


def _utc_now() -> datetime:
    return datetime.now(UTC)


class SqliteProfileStore:
    """SQLite v1 implementation of :class:`~edgeshard.profiling.store.base.ProfileStore`."""

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if isinstance(path, str) and not path:
            raise ValueError("database path must not be empty")
        self._path = str(path)
        self._clock = clock
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        with self._lock, self._conn:
            self._conn.executescript(SCHEMA)

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> SqliteProfileStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- measurements (§43-45) ------------------------------------------------

    def append_case(self, case: ProfilingCase) -> bool:
        kind = "network" if isinstance(case.spec, NetworkCaseSpec) else "model"
        payload_text = encode_json(case)
        with self._lock, self._conn:
            return self._insert_verified(
                "INSERT OR IGNORE INTO profiling_cases "
                "(case_id, worker_id, kind, state, payload) VALUES (?, ?, ?, ?, ?)",
                (case.case_id, case.worker_id, kind, CaseState.PENDING.value,
                 payload_text),
                select_sql="SELECT payload FROM profiling_cases WHERE case_id = ?",
                select_params=(case.case_id,),
                payload_text=payload_text,
                what=f"profiling case {case.case_id!r}",
            )

    def update_case_state(self, case_id: str, state: CaseState) -> None:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "UPDATE profiling_cases SET state = ? WHERE case_id = ?",
                (state.value, case_id),
            )
            if cursor.rowcount == 0:
                raise ProfileStoreError(f"unknown profiling case {case_id!r}")

    def get_case(self, case_id: str) -> StoredCase | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload, state FROM profiling_cases WHERE case_id = ?",
                (case_id,),
            ).fetchone()
        if row is None:
            return None
        return StoredCase(
            case=self._decode(ProfilingCase, row["payload"], "profiling case"),
            state=CaseState(row["state"]),
        )

    def append_measurement(self, record: MeasurementRecord) -> bool:
        payload_text = encode_json(record)
        with self._lock, self._conn:
            case_row = self._conn.execute(
                "SELECT payload FROM profiling_cases WHERE case_id = ?",
                (record.case_id,),
            ).fetchone()
            if case_row is None:
                raise ProfileStoreError(
                    f"measurement {record.measurement_id!r} references unknown "
                    f"case {record.case_id!r} — persist the case first (§8.3)"
                )
            case = self._decode(ProfilingCase, case_row["payload"], "profiling case")
            columns = self._measurement_columns(record, case)
            inserted = self._insert_verified(
                "INSERT OR IGNORE INTO measurements ("
                "measurement_id, case_id, environment_fingerprint_id, model_id, "
                "model_revision, layer_signature_id, module_signature_id, "
                "operator_signature_id, network_pair_id, probe_kind, started_at, "
                "finished_at, sample_count, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (*columns, payload_text),
                select_sql="SELECT payload FROM measurements WHERE measurement_id = ?",
                select_params=(record.measurement_id,),
                payload_text=payload_text,
                what=f"measurement {record.measurement_id!r}",
            )
            if inserted and record.samples is not None:
                self._conn.executemany(
                    "INSERT INTO measurement_samples "
                    "(measurement_id, sample_index, value) VALUES (?, ?, ?)",
                    [
                        (record.measurement_id, index, value)
                        for index, value in enumerate(record.samples)
                    ],
                )
            return inserted

    def get_measurement(self, measurement_id: str) -> MeasurementRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM measurements WHERE measurement_id = ?",
                (measurement_id,),
            ).fetchone()
        if row is None:
            return None
        return self._decode(MeasurementRecord, row["payload"], "measurement")

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
        filters: list[str] = []
        params: list[_SqlValue] = []
        for column, value in (
            ("case_id", case_id),
            ("environment_fingerprint_id", environment_fingerprint_id),
            ("model_id", model_id),
            ("model_revision", revision),
            ("layer_signature_id", layer_signature_id),
            ("module_signature_id", module_signature_id),
            ("operator_signature_id", operator_signature_id),
            ("network_pair_id", network_pair_id),
            ("probe_kind", probe_kind.value if probe_kind is not None else None),
        ):
            if value is not None:
                filters.append(f"{column} = ?")
                params.append(value)
        where = f" WHERE {' AND '.join(filters)}" if filters else ""
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM measurements"
                f"{where} ORDER BY started_at, measurement_id",
                params,
            ).fetchall()
        return tuple(
            self._decode(MeasurementRecord, row["payload"], "measurement")
            for row in rows
        )

    # -- experiments (§40) ------------------------------------------------------

    def append_experiment(self, experiment: ProfilingExperiment) -> bool:
        payload_text = encode_json(experiment)
        with self._lock, self._conn:
            return self._insert_verified(
                "INSERT OR IGNORE INTO experiments "
                "(experiment_id, strategy_id, created_at, requested_by, state, "
                "payload) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    experiment.experiment_id,
                    experiment.strategy_id,
                    experiment.created_at.isoformat(),
                    experiment.requested_by,
                    ExperimentState.PENDING.value,
                    payload_text,
                ),
                select_sql="SELECT payload FROM experiments WHERE experiment_id = ?",
                select_params=(experiment.experiment_id,),
                payload_text=payload_text,
                what=f"experiment {experiment.experiment_id!r}",
            )

    def update_experiment_state(
        self, experiment_id: str, state: ExperimentState
    ) -> None:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "UPDATE experiments SET state = ? WHERE experiment_id = ?",
                (state.value, experiment_id),
            )
            if cursor.rowcount == 0:
                raise ProfileStoreError(f"unknown experiment {experiment_id!r}")

    def get_experiment(self, experiment_id: str) -> StoredExperiment | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload, state FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
        if row is None:
            return None
        return StoredExperiment(
            experiment=self._decode(
                ProfilingExperiment, row["payload"], "experiment"
            ),
            state=ExperimentState(row["state"]),
        )

    # -- canonical-id registries (§44) -------------------------------------------

    def store_characterization(self, characterization: ModelCharacterization) -> str:
        signature_id = model_signature_id(characterization)
        payload_text = encode_json(characterization)
        with self._lock, self._conn:
            self._insert_verified(
                "INSERT OR IGNORE INTO model_characterizations ("
                "model_signature_id, model_id, revision, architecture_family, "
                "num_layers, payload) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    signature_id,
                    characterization.model.model_id,
                    characterization.model.revision,
                    characterization.architecture_family,
                    characterization.num_layers,
                    payload_text,
                ),
                select_sql=(
                    "SELECT payload FROM model_characterizations "
                    "WHERE model_signature_id = ?"
                ),
                select_params=(signature_id,),
                payload_text=payload_text,
                what=f"model characterization {signature_id!r}",
            )
        return signature_id

    def store_layer_signature(self, signature: TransformerLayerSignature) -> str:
        signature_id = transformer_layer_signature_id(signature)
        payload_text = encode_json(signature)
        with self._lock, self._conn:
            self._insert_verified(
                "INSERT OR IGNORE INTO transformer_layer_signatures ("
                "layer_signature_id, architecture_family, layer_type, dtype, "
                "payload) VALUES (?, ?, ?, ?, ?)",
                (
                    signature_id,
                    signature.architecture_family,
                    signature.layer_type,
                    signature.dtype,
                    payload_text,
                ),
                select_sql=(
                    "SELECT payload FROM transformer_layer_signatures "
                    "WHERE layer_signature_id = ?"
                ),
                select_params=(signature_id,),
                payload_text=payload_text,
                what=f"transformer-layer signature {signature_id!r}",
            )
        return signature_id

    def store_module_signature(self, signature: ModuleSignature) -> str:
        signature_id = module_signature_id(signature)
        payload_text = encode_json(signature)
        with self._lock, self._conn:
            self._insert_verified(
                "INSERT OR IGNORE INTO module_signatures ("
                "module_signature_id, kind, architecture_family, dtype, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    signature_id,
                    signature.kind.value,
                    signature.architecture_family,
                    signature.dtype,
                    payload_text,
                ),
                select_sql=(
                    "SELECT payload FROM module_signatures "
                    "WHERE module_signature_id = ?"
                ),
                select_params=(signature_id,),
                payload_text=payload_text,
                what=f"module signature {signature_id!r}",
            )
        return signature_id

    def store_operator_signature(self, signature: OperatorSignature) -> str:
        signature_id = operator_signature_id(signature)
        payload_text = encode_json(signature)
        with self._lock, self._conn:
            self._insert_verified(
                "INSERT OR IGNORE INTO operator_signatures ("
                "operator_signature_id, kind, backend_family, payload) "
                "VALUES (?, ?, ?, ?)",
                (
                    signature_id,
                    signature.kind.value,
                    signature.backend_family,
                    payload_text,
                ),
                select_sql=(
                    "SELECT payload FROM operator_signatures "
                    "WHERE operator_signature_id = ?"
                ),
                select_params=(signature_id,),
                payload_text=payload_text,
                what=f"operator signature {signature_id!r}",
            )
        return signature_id

    def store_network_endpoint(
        self, profile: NetworkEndpointProfile
    ) -> tuple[str, str]:
        """Endpoint facts evolve (Phase 1 rediscovers them): last write wins."""
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO network_endpoints (worker_id, interface_id, mtu, "
                "link_speed_mbps, interface_type, overlay_type, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(worker_id, interface_id) DO UPDATE SET "
                "mtu = excluded.mtu, link_speed_mbps = excluded.link_speed_mbps, "
                "interface_type = excluded.interface_type, "
                "overlay_type = excluded.overlay_type, payload = excluded.payload",
                (
                    profile.worker_id,
                    profile.interface_id,
                    profile.mtu,
                    profile.link_speed_mbps,
                    profile.interface_type,
                    profile.overlay_type,
                    encode_json(profile),
                ),
            )
        return (profile.worker_id, profile.interface_id)

    def store_path_classification(
        self, pair: NetworkPair, path_class: NetworkPathClass
    ) -> str:
        """Pair identity is content-hashed; its classification may evolve (§32)."""
        pair_id = network_pair_signature_id(pair)
        payload_text = encode_json(pair)
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR IGNORE INTO network_path_classes (path_class) VALUES (?)",
                (path_class.value,),
            )
            self._insert_verified(
                "INSERT OR IGNORE INTO network_pairs ("
                "network_pair_id, source_worker_id, destination_worker_id, "
                "source_interface_id, destination_interface_id, payload) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    pair_id,
                    pair.source_worker_id,
                    pair.destination_worker_id,
                    pair.source_interface_id,
                    pair.destination_interface_id,
                    payload_text,
                ),
                select_sql="SELECT payload FROM network_pairs WHERE network_pair_id = ?",
                select_params=(pair_id,),
                payload_text=payload_text,
                what=f"network pair {pair_id!r}",
            )
            self._conn.execute(
                "UPDATE network_pairs SET path_class = ? WHERE network_pair_id = ?",
                (path_class.value, pair_id),
            )
        return pair_id

    def store_environment_fingerprint(self, fingerprint: EnvironmentFingerprint) -> str:
        fingerprint_id = environment_fingerprint_id(fingerprint)
        payload_text = encode_json(fingerprint)
        with self._lock, self._conn:
            self._insert_verified(
                "INSERT OR IGNORE INTO environment_fingerprints ("
                "environment_fingerprint_id, backend, worker_id, device_id, "
                "device_performance_class_id, payload) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    fingerprint_id,
                    fingerprint.backend,
                    fingerprint.worker_id,
                    fingerprint.device_id,
                    fingerprint.device_performance_class_id,
                    payload_text,
                ),
                select_sql=(
                    "SELECT payload FROM environment_fingerprints "
                    "WHERE environment_fingerprint_id = ?"
                ),
                select_params=(fingerprint_id,),
                payload_text=payload_text,
                what=f"environment fingerprint {fingerprint_id!r}",
            )
        return fingerprint_id

    def store_performance_class(self, performance_class: DevicePerformanceClass) -> str:
        class_id = device_performance_class_id(performance_class)
        payload_text = encode_json(performance_class)
        with self._lock, self._conn:
            self._insert_verified(
                "INSERT OR IGNORE INTO device_performance_classes ("
                "device_performance_class_id, vendor, accelerator_model, "
                "memory_model, backend_family, payload) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    class_id,
                    performance_class.vendor,
                    performance_class.accelerator_model,
                    performance_class.memory_model.value,
                    performance_class.backend_family,
                    payload_text,
                ),
                select_sql=(
                    "SELECT payload FROM device_performance_classes "
                    "WHERE device_performance_class_id = ?"
                ),
                select_params=(class_id,),
                payload_text=payload_text,
                what=f"device performance class {class_id!r}",
            )
        return class_id

    # -- reuse queries (§28) and snapshot (§46) -------------------------------------

    def measured_operator_signature_ids(
        self,
        *,
        device_performance_class_id: str | None = None,
        environment_fingerprint_id: str | None = None,
    ) -> frozenset[str]:
        sql = (
            "SELECT DISTINCT operator_signature_id FROM measurements "
            "WHERE operator_signature_id IS NOT NULL"
        )
        params: list[_SqlValue] = []
        if environment_fingerprint_id is not None:
            sql += " AND environment_fingerprint_id = ?"
            params.append(environment_fingerprint_id)
        if device_performance_class_id is not None:
            sql += (
                " AND environment_fingerprint_id IN ("
                "SELECT environment_fingerprint_id FROM environment_fingerprints "
                "WHERE device_performance_class_id = ?)"
            )
            params.append(device_performance_class_id)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return frozenset(row["operator_signature_id"] for row in rows)

    def build_snapshot(
        self, snapshot_id: str, *, created_at: datetime | None = None
    ) -> ProfileSnapshot:
        with self._lock:
            characterization_rows = self._conn.execute(
                "SELECT payload FROM model_characterizations "
                "ORDER BY model_signature_id"
            ).fetchall()
            measurement_rows = self._conn.execute(
                "SELECT payload FROM measurements "
                "ORDER BY started_at, measurement_id"
            ).fetchall()
        characterizations = tuple(
            self._decode(ModelCharacterization, row["payload"], "characterization")
            for row in characterization_rows
        )
        measurements: list[MeasurementRecord] = []
        network_measurements: list[MeasurementRecord] = []
        for row in measurement_rows:
            record = self._decode(MeasurementRecord, row["payload"], "measurement")
            if record.metrics.rtt is not None or record.metrics.bandwidth is not None:
                network_measurements.append(record)
            else:
                measurements.append(record)
        return ProfileSnapshot(
            snapshot_id=snapshot_id,
            created_at=created_at if created_at is not None else self._clock(),
            model_characterizations=characterizations,
            measurements=tuple(measurements),
            network_measurements=tuple(network_measurements),
        )

    # -- internals ------------------------------------------------------------------

    @staticmethod
    def _measurement_columns(
        record: MeasurementRecord, case: ProfilingCase
    ) -> tuple[_SqlValue, ...]:
        """Indexed columns derived from the case spec (never guessed, §52.2)."""
        spec = case.spec
        model_id: str | None = None
        revision: str | None = None
        layer_id: str | None = None
        module_id: str | None = None
        operator_id: str | None = None
        pair_id: str | None = None
        probe: str | None = None
        if isinstance(spec, ModelCaseSpec):
            if spec.model is not None:
                model_id = spec.model.model_id
                revision = spec.model.revision
            if spec.layer_signature is not None:
                layer_id = transformer_layer_signature_id(spec.layer_signature)
            if spec.module_signature is not None:
                module_id = module_signature_id(spec.module_signature)
            if spec.operator_signature is not None:
                operator_id = operator_signature_id(spec.operator_signature)
        elif isinstance(spec, NetworkCaseSpec):
            pair_id = network_pair_signature_id(
                NetworkPair(
                    source_worker_id=spec.source_worker_id,
                    destination_worker_id=spec.destination_worker_id,
                    source_interface_id=spec.source_interface_id,
                    destination_interface_id=spec.destination_interface_id,
                )
            )
            probe = spec.probe_kind.value
        return (
            record.measurement_id,
            record.case_id,
            record.environment_fingerprint,
            model_id,
            revision,
            layer_id,
            module_id,
            operator_id,
            pair_id,
            probe,
            record.started_at.isoformat(),
            record.finished_at.isoformat(),
            record.sample_count,
        )

    def _insert_verified(
        self,
        insert_sql: str,
        params: Sequence[_SqlValue],
        *,
        select_sql: str,
        select_params: Sequence[_SqlValue],
        payload_text: str,
        what: str,
    ) -> bool:
        """INSERT OR IGNORE + §7 conflict rule: same id, same content, or fail."""
        cursor = self._conn.execute(insert_sql, params)
        if cursor.rowcount == 1:
            return True
        stored = self._conn.execute(select_sql, select_params).fetchone()
        if stored is None:  # pragma: no cover - would mean a broken INSERT OR IGNORE
            raise ProfileStoreError(f"{what}: insert was ignored but no row exists")
        if stored["payload"] != payload_text:
            raise ProfileStoreError(
                f"{what}: the same canonical id maps to different content — "
                "this violates the §7 hashing contract"
            )
        return False

    @staticmethod
    def _decode[T](cls: type[T], payload: str, what: str) -> T:
        try:
            return decode_json(cls, payload)
        except PayloadCodecError as exc:
            raise ProfileStoreError(f"stored {what} payload is corrupt: {exc}") from exc
