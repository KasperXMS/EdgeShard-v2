"""Network case profiling: RTT probes and bandwidth baselines (§33-§36).

``NetworkProfiler`` turns a ``NetworkCaseSpec`` into an empirical
:class:`MeasurementRecord` — RTT via the bounded system-ping runner,
bandwidth via the JSON-only iperf3 runner. Probes run where the flow
originates (§8.2): the case executes on the source worker and resolves both
ends through the explicitly selected interfaces. Ambiguous multi-NIC paths
are rejected instead of silently probing an arbitrary address; the iperf3
server on the destination is placed by P2G orchestration.

Records always carry the measurement regime (§35: v1 measures
``IDLE_SINGLE_FLOW`` baselines, explicitly recorded so Phase 3 never
mistakes them for guaranteed concurrent bandwidth) and the derived path
class when the case knows it (§31).

Case builders implement the probe selection policy: a dense RTT matrix
over all directed pairs (§33: dense pairwise RTT is acceptable) and a
sparse bandwidth set — deterministic representatives per ``PathClass``
plus both directions (§34) — with payload sizes derived from model
characterization dimensions per §36 (``batch * seq * hidden *
bytes_per_element``, deduplicated, never an exhaustive sweep).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from uuid import uuid4

from edgeshard.profiling.domain.experiment import (
    NetworkCaseSpec,
    ProfilingCase,
    ProfilingErrorCategory,
)
from edgeshard.profiling.domain.hashing import JsonScalar
from edgeshard.profiling.domain.measurement import (
    BandwidthMetrics,
    MeasurementMetrics,
    MeasurementRecord,
    RttMetrics,
    TimeUnit,
)
from edgeshard.profiling.domain.network import (
    NetworkDirection,
    NetworkPair,
    NetworkPathClass,
    NetworkTransport,
    ProbeKind,
)
from edgeshard.profiling.dtypes import dtype_byte_size
from edgeshard.profiling.errors import ProfilingError
from edgeshard.profiling.network.classifier import (
    DEFAULT_SAME_SUBNET_PREFIX_LENGTH,
    WorkerNetworkFacts,
    classify_pairs,
    classify_path,
    enumerate_probe_paths,
    select_bandwidth_pairs,
    selected_ipv4_address,
)
from edgeshard.profiling.network.iperf import (
    DEFAULT_IPERF3_DURATION_S,
    DEFAULT_IPERF3_PORT,
    Iperf3Probe,
    Iperf3Runner,
)
from edgeshard.profiling.network.ping import DEFAULT_PING_CONCURRENCY, PingRunner
from edgeshard.profiling.network.processes import terminate_process

logger = logging.getLogger("profiling.network.profiler")

DEFAULT_PAYLOAD_SEQUENCE_LENGTHS = (1, 512, 2048)
"""Representative pipeline payload sequence lengths (§36 policy): one
decode-step activation and two prefill-chunk scales — never a sweep."""


def require_network_spec(case: ProfilingCase) -> NetworkCaseSpec:
    """The case's network spec (mirror of ``require_model_spec``)."""
    spec = case.spec
    if not isinstance(spec, NetworkCaseSpec):
        raise ValueError(
            f"network profiling requires a network case spec, "
            f"got {type(spec).__name__}"
        )
    return spec


def hidden_state_payload_bytes(
    *, batch_size: int, sequence_length: int, hidden_size: int, dtype: str
) -> int:
    """One pipeline hidden-state payload: ``batch * seq * hidden * bytes`` (§36)."""
    for field_name, value in (
        ("batch_size", batch_size),
        ("sequence_length", sequence_length),
        ("hidden_size", hidden_size),
    ):
        if value < 1:
            raise ValueError(f"{field_name} must be positive, got {value}")
    return batch_size * sequence_length * hidden_size * dtype_byte_size(dtype)


def pipeline_payload_sizes(
    *,
    hidden_size: int,
    dtype: str,
    batch_size: int = 1,
    sequence_lengths: Iterable[int] = DEFAULT_PAYLOAD_SEQUENCE_LENGTHS,
) -> tuple[int, ...]:
    """The small deduplicated payload-size set for a characterization (§36).

    Derived from model characterization dimensions (hidden size, dtype,
    representative sequence lengths); sorted and deduplicated so equal
    characterizations yield identical size classes.
    """
    lengths = list(sequence_lengths)
    if not lengths:
        raise ValueError("sequence_lengths must not be empty")
    sizes = {
        hidden_state_payload_bytes(
            batch_size=batch_size,
            sequence_length=length,
            hidden_size=hidden_size,
            dtype=dtype,
        )
        for length in lengths
    }
    return tuple(sorted(sizes))


def rtt_matrix_cases(
    facts: Iterable[WorkerNetworkFacts],
    *,
    pairs: Iterable[NetworkPair] | None = None,
    packet_count: int | None = None,
    same_subnet_prefix_length: int = DEFAULT_SAME_SUBNET_PREFIX_LENGTH,
) -> tuple[ProfilingCase, ...]:
    """Dense RTT matrix over explicit interface paths (§32-§33).

    Every case executes on its pair's source worker (``ProfilingCase``
    enforces it); the derived ``PathClass`` is attached as case metadata.
    """
    workers = tuple(facts)
    by_id = {worker.worker_id: worker for worker in workers}
    cases = []
    selected = enumerate_probe_paths(workers) if pairs is None else tuple(pairs)
    for pair in selected:
        if pair.source_worker_id not in by_id or pair.destination_worker_id not in by_id:
            raise ValueError(
                "explicit RTT pair references unknown worker(s): "
                f"{pair.source_worker_id!r} -> {pair.destination_worker_id!r}"
            )
        spec = NetworkCaseSpec(
            probe_kind=ProbeKind.RTT,
            source_worker_id=pair.source_worker_id,
            destination_worker_id=pair.destination_worker_id,
            source_interface_id=pair.source_interface_id,
            destination_interface_id=pair.destination_interface_id,
            path_class=classify_path(
                by_id[pair.source_worker_id],
                by_id[pair.destination_worker_id],
                same_subnet_prefix_length=same_subnet_prefix_length,
                source_interface_id=pair.source_interface_id,
                destination_interface_id=pair.destination_interface_id,
            ),
            packet_count=packet_count,
        )
        cases.append(ProfilingCase.for_spec(pair.source_worker_id, spec))
    return tuple(cases)


def bandwidth_cases(
    facts: Iterable[WorkerNetworkFacts],
    *,
    representatives_per_class: int = 1,
    extra_pairs: Iterable[NetworkPair] = (),
    transport: NetworkTransport = NetworkTransport.TCP,
    duration_s: float = DEFAULT_IPERF3_DURATION_S,
    payload_bytes: int | None = None,
    include_reverse: bool = True,
    same_subnet_prefix_length: int = DEFAULT_SAME_SUBNET_PREFIX_LENGTH,
    path_classes: Iterable[NetworkPathClass] | None = None,
) -> tuple[ProfilingCase, ...]:
    """Sparse bandwidth cases: representatives per class, both directions (§34).

    iperf3 is expensive, so the selection is sparse by policy — one (or a
    few) deterministic representative pair per ``PathClass`` plus the
    optional explicit-pair knob. Forward and reverse flows are distinct
    directed cases (§32); v1 runs them sequentially (see
    :meth:`NetworkProfiler.probe_bandwidth`).

    ``path_classes`` restricts the class-driven selection (e.g. the §49
    ``network bandwidth --path-class`` knob); explicitly requested
    ``extra_pairs`` are appended regardless — the explicit knob wins over
    the class filter.
    """
    workers = tuple(facts)
    by_id = {worker.worker_id: worker for worker in workers}
    classified = classify_pairs(
        workers, same_subnet_prefix_length=same_subnet_prefix_length
    )
    if path_classes is not None:
        allowed = frozenset(path_classes)
        classified = tuple(
            entry for entry in classified if entry.path_class in allowed
        )
    selected = select_bandwidth_pairs(
        classified,
        representatives_per_class=representatives_per_class,
        extra_pairs=extra_pairs,
    )
    directions: tuple[NetworkDirection, ...] = (
        NetworkDirection.FORWARD,
        NetworkDirection.REVERSE,
    )
    if not include_reverse:
        directions = (NetworkDirection.FORWARD,)
    cases = []
    for pair in selected:
        source = by_id.get(pair.source_worker_id)
        destination = by_id.get(pair.destination_worker_id)
        if source is None or destination is None:
            raise ValueError(
                f"explicit pair references unknown worker(s): "
                f"{pair.source_worker_id!r} → {pair.destination_worker_id!r}"
            )
        path_class = classify_path(
            source,
            destination,
            same_subnet_prefix_length=same_subnet_prefix_length,
            source_interface_id=pair.source_interface_id,
            destination_interface_id=pair.destination_interface_id,
        )
        for direction in directions:
            spec = NetworkCaseSpec(
                probe_kind=ProbeKind.BANDWIDTH,
                source_worker_id=pair.source_worker_id,
                destination_worker_id=pair.destination_worker_id,
                source_interface_id=pair.source_interface_id,
                destination_interface_id=pair.destination_interface_id,
                path_class=path_class,
                transport=transport,
                direction=direction,
                duration_s=duration_s,
                payload_bytes=payload_bytes,
            )
            cases.append(ProfilingCase.for_spec(pair.source_worker_id, spec))
    return tuple(cases)


class NetworkProfiler:
    """Produces empirical network ``MeasurementRecord``s (§33-§35).

    The runners are injectable; defaults spawn the real platform binaries.
    Failures are the runners' typed ``ProfilingError``s — this layer only
    adds fact-resolution failures (unknown destination, no probeable
    IPv4), never fabricated observations (§42, §52.2).
    """

    def __init__(
        self,
        *,
        ping_runner: PingRunner | None = None,
        iperf_runner: Iperf3Runner | None = None,
    ) -> None:
        self._ping = ping_runner if ping_runner is not None else PingRunner()
        self._iperf = iperf_runner if iperf_runner is not None else Iperf3Runner()

    async def profile(
        self,
        case: ProfilingCase,
        *,
        network_facts: Mapping[str, WorkerNetworkFacts],
        environment_fingerprint: str,
        iperf_server_port: int | None = None,
    ) -> MeasurementRecord:
        """One RTT or bandwidth measurement for ``case``."""
        spec = require_network_spec(case)
        destination = network_facts.get(spec.destination_worker_id)
        if destination is None:
            raise ValueError(
                f"no network facts for destination worker "
                f"{spec.destination_worker_id!r}"
            )
        target = selected_ipv4_address(
            destination, spec.destination_interface_id
        )
        source = network_facts.get(spec.source_worker_id)
        source_address = (
            selected_ipv4_address(source, spec.source_interface_id)
            if source is not None
            else None
        )
        if source is None or source_address is None:
            raise ProfilingError(
                ProfilingErrorCategory.NETWORK_UNREACHABLE,
                f"source worker {spec.source_worker_id!r} has no unambiguous "
                "IPv4 probe path; select source_interface_id",
                {
                    "source_worker_id": spec.source_worker_id,
                    "source_interface_id": spec.source_interface_id,
                },
            )
        if target is None:
            raise ProfilingError(
                ProfilingErrorCategory.NETWORK_UNREACHABLE,
                f"destination worker {spec.destination_worker_id!r} has no "
                "unambiguous IPv4 probe path; select destination_interface_id",
                {
                    "destination_worker_id": spec.destination_worker_id,
                    "destination_interface_id": spec.destination_interface_id,
                },
            )
        started_at = datetime.now(UTC)
        metadata: dict[str, JsonScalar] = {
            "probe_kind": spec.probe_kind.value,
            "source_worker_id": spec.source_worker_id,
            "destination_worker_id": spec.destination_worker_id,
            "target": target,
            "regime": spec.regime.value,
            "path_class": spec.path_class.value if spec.path_class is not None else None,
        }
        if spec.probe_kind is ProbeKind.RTT:
            observation = await self._ping.probe(
                target,
                packet_count=spec.packet_count,
                bind_address=source_address,
            )
            metrics = MeasurementMetrics(
                rtt=RttMetrics(
                    summary=observation.summary,
                    unit=TimeUnit.MILLISECONDS,
                    packets_sent=observation.packets_sent,
                    packets_received=observation.packets_received,
                )
            )
            samples = observation.samples_ms
            sample_count = len(samples)
            metadata["timing_unit"] = TimeUnit.MILLISECONDS.value
            metadata["packet_count"] = observation.packets_sent
        else:
            # Domain guarantees transport/direction/duration for BANDWIDTH.
            assert spec.transport is not None and spec.direction is not None
            duration_s = spec.duration_s
            assert duration_s is not None
            result = await self._iperf.run_client(
                Iperf3Probe(
                    target=target,
                    transport=spec.transport,
                    direction=spec.direction,
                    duration_s=duration_s,
                    payload_bytes=spec.payload_bytes,
                    bind_address=source_address,
                    port=iperf_server_port or DEFAULT_IPERF3_PORT,
                )
            )
            metrics = MeasurementMetrics(
                bandwidth=BandwidthMetrics(
                    bits_per_second=result.bits_per_second,
                    payload_bytes=spec.payload_bytes,
                    retransmits=result.retransmits,
                    duration_s=duration_s,
                )
            )
            samples = (result.bits_per_second,)
            sample_count = 1
            metadata["transport"] = spec.transport.value
            metadata["direction"] = spec.direction.value
            metadata["duration_s"] = duration_s
            metadata["sample_unit"] = "bits_per_second"
            if spec.payload_bytes is not None:
                metadata["payload_bytes"] = spec.payload_bytes
        finished_at = datetime.now(UTC)
        record = MeasurementRecord(
            measurement_id=uuid4().hex,
            case_id=case.case_id,
            environment_fingerprint=environment_fingerprint,
            started_at=started_at,
            finished_at=finished_at,
            sample_count=sample_count,
            samples=samples,
            metrics=metrics,
            metadata=MeasurementRecord.normalize_metadata(metadata),
        )
        logger.info(
            "network %s probe %s → %s recorded (%d samples)",
            spec.probe_kind.value,
            spec.source_worker_id,
            spec.destination_worker_id,
            sample_count,
        )
        return record

    async def start_iperf_server(
        self, *, port: int, bind_address: str | None = None
    ) -> asyncio.subprocess.Process:
        return await self._iperf.start_server(
            port=port, bind_address=bind_address
        )

    @staticmethod
    async def stop_iperf_server(process: asyncio.subprocess.Process) -> None:
        await terminate_process(process)

    async def probe_rtt(
        self,
        cases: Iterable[ProfilingCase],
        *,
        network_facts: Mapping[str, WorkerNetworkFacts],
        environment_fingerprint: str,
        concurrency: int = DEFAULT_PING_CONCURRENCY,
    ) -> tuple[MeasurementRecord, ...]:
        """Bounded-concurrency RTT fan-out (§33); input order preserved."""
        if concurrency < 1:
            raise ValueError(f"concurrency must be positive, got {concurrency}")
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded(case: ProfilingCase) -> MeasurementRecord:
            async with semaphore:
                return await self.profile(
                    case,
                    network_facts=network_facts,
                    environment_fingerprint=environment_fingerprint,
                )

        return tuple(await asyncio.gather(*(bounded(case) for case in cases)))

    async def probe_bandwidth(
        self,
        cases: Iterable[ProfilingCase],
        *,
        network_facts: Mapping[str, WorkerNetworkFacts],
        environment_fingerprint: str,
    ) -> tuple[MeasurementRecord, ...]:
        """Sequential iperf3 runs (§34-§35).

        Strictly one flow at a time: concurrent flows would contaminate
        the idle single-flow baseline the regime promises (§35).
        """
        records = []
        for case in cases:
            records.append(
                await self.profile(
                    case,
                    network_facts=network_facts,
                    environment_fingerprint=environment_fingerprint,
                )
            )
        return tuple(records)


__all__ = [
    "DEFAULT_PAYLOAD_SEQUENCE_LENGTHS",
    "NetworkProfiler",
    "bandwidth_cases",
    "hidden_state_payload_bytes",
    "pipeline_payload_sizes",
    "require_network_spec",
    "rtt_matrix_cases",
]
