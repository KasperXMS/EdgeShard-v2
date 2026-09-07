"""DefaultProfilingStrategy — the §47 policy composition (P2G.7).

Pinned here: the model workflow plans exactly the missing operator
microprofiles per device (§28/§47 step 4), sparse Attention+MLP module
cases deduplicated by signature (step 5), and early/middle/late layer
positions across the default prefill lengths (step 6) — with deferred
signatures reported, never silently dropped (§19) and never dispatched
into guaranteed typed failures. The network workflow returns the endpoint
characterization and pair classification alongside the dense RTT matrix
and the sparse per-class bandwidth selection (§30-§34), including the
explicit-pair and path-class knobs. Plans are pure data of canonical
cases (§7); no composed performance estimate exists anywhere (§47).
"""

from __future__ import annotations

import pytest

from edgeshard.profiling.domain.experiment import (
    NetworkCaseSpec,
    ProfilingCase,
    profiling_case_id,
)
from edgeshard.profiling.domain.hashing import normalized_items
from edgeshard.profiling.domain.model import (
    ModelCharacterization,
    ModelReference,
    ModelStage,
    StageKind,
)
from edgeshard.profiling.domain.network import (
    NetworkDirection,
    NetworkPair,
    NetworkPathClass,
    NetworkTransport,
    ProbeKind,
)
from edgeshard.profiling.domain.session import (
    LayerEntry,
    ModelSessionFacts,
    ModuleEntry,
)
from edgeshard.profiling.domain.signature import (
    AttentionSignature,
    CustomOperatorParameters,
    EmbeddingSignature,
    GemmSignature,
    InferencePhase,
    ModuleKind,
    ModuleSignature,
    OperatorKind,
    OperatorSignature,
    ProfilingGranularity,
    TransformerLayerSignature,
    operator_signature_id,
)
from edgeshard.profiling.model.planning import (
    DEFAULT_PREFILL_SEQUENCE_LENGTHS,
    representative_layer_positions,
)
from edgeshard.profiling.network.classifier import (
    InterfaceFacts,
    WorkerNetworkFacts,
    classify_interface,
)
from edgeshard.profiling.network.iperf import DEFAULT_IPERF3_DURATION_S
from edgeshard.profiling.operator.planning import operator_case_spec
from edgeshard.profiling.strategy.base import (
    ModelProfilingPlan,
    ProfilingStrategy,
)
from edgeshard.profiling.strategy.default import (
    DEFAULT_MODULE_SEQUENCE_LENGTH,
    DEFAULT_STRATEGY_ID,
    DefaultProfilingStrategy,
)

WORKER = "w-1"
MODEL = ModelReference(model_id="tiny/llama", revision="local")

LAYER_SIG = TransformerLayerSignature(
    architecture_family="llama",
    layer_type="decoder",
    hidden_size=32,
    intermediate_size=64,
    num_attention_heads=4,
    num_kv_heads=2,
    head_dim=8,
    dtype="fp32",
    quantization=None,
)
ATTN_MODULE_SIG = ModuleSignature(
    kind=ModuleKind.ATTENTION,
    architecture_family="llama",
    structural_parameters=normalized_items({"hidden_size": 32, "num_heads": 4}, "p"),
    dtype="fp32",
    quantization=None,
)
MLP_MODULE_SIG = ModuleSignature(
    kind=ModuleKind.MLP,
    architecture_family="llama",
    structural_parameters=normalized_items({"intermediate_size": 64}, "p"),
    dtype="fp32",
    quantization=None,
)
NORM_MODULE_SIG = ModuleSignature(
    kind=ModuleKind.NORM,
    architecture_family="llama",
    structural_parameters=normalized_items({"hidden_size": 32}, "p"),
    dtype="fp32",
    quantization=None,
)

GEMM_SIG = OperatorSignature(
    kind=OperatorKind.GEMM,
    parameters=GemmSignature(m=64, n=64, k=64, dtype="fp32"),
    backend_family="torch",
)
ATTN_OP_SIG = OperatorSignature(
    kind=OperatorKind.ATTENTION,
    parameters=AttentionSignature(
        batch_size=1,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        q_len=16,
        kv_len=16,
        dtype="fp32",
        phase=InferencePhase.PREFILL,
    ),
    backend_family="torch",
)
EMBED_SIG = OperatorSignature(
    kind=OperatorKind.EMBEDDING,
    parameters=EmbeddingSignature(
        batch_size=1, sequence_length=16, vocab_size=64, embedding_dim=32, dtype="fp32"
    ),
    backend_family="torch",
)
CUSTOM_SIG = OperatorSignature(
    kind=OperatorKind.CUSTOM,
    parameters=CustomOperatorParameters(
        raw_name="my_fused_kernel", input_shapes=((4, 8),), input_dtypes=("fp32",)
    ),
    backend_family="torch",
)


def _characterization(num_layers: int = 5) -> ModelCharacterization:
    return ModelCharacterization(
        model=MODEL,
        architecture_family="llama",
        num_layers=num_layers,
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
            ModelStage(kind=StageKind.TRANSFORMER_LAYER_GROUP, layer_count=num_layers),
            ModelStage(kind=StageKind.LM_HEAD),
        ),
    )


def _facts(num_layers: int = 5, **overrides: object) -> ModelSessionFacts:
    base: dict[str, object] = {
        "characterization": _characterization(num_layers),
        "layer_entries": tuple(
            LayerEntry(index, f"model.layers.{index}", LAYER_SIG)
            for index in range(num_layers)
        ),
        "module_entries": (
            ModuleEntry(
                "self_attn", "model.layers.0.self_attn", ModuleKind.ATTENTION, 0,
                ATTN_MODULE_SIG,
            ),
            ModuleEntry("mlp", "model.layers.0.mlp", ModuleKind.MLP, 0, MLP_MODULE_SIG),
            # Homogeneous stack: the same attention shape again — the sparse
            # policy keeps one case per unique signature (§47 step 5).
            ModuleEntry(
                "self_attn", "model.layers.1.self_attn", ModuleKind.ATTENTION, 1,
                ATTN_MODULE_SIG,
            ),
            # Not in SPARSE_MODULE_KINDS: never selected in v1.
            ModuleEntry(
                "input_layernorm", "model.layers.0.input_layernorm", ModuleKind.NORM, 0,
                NORM_MODULE_SIG,
            ),
        ),
        "operator_signatures": (GEMM_SIG, ATTN_OP_SIG, EMBED_SIG, CUSTOM_SIG, GEMM_SIG),
    }
    base.update(overrides)
    return ModelSessionFacts(**base)  # type: ignore[arg-type]


def _worker_facts(
    worker_id: str, address: str, *, name: str = "eth0"
) -> WorkerNetworkFacts:
    kind, overlay = classify_interface(name)
    return WorkerNetworkFacts(
        worker_id=worker_id,
        hostname=f"host-{worker_id}",
        interfaces=(InterfaceFacts(f"nic-{worker_id}", name, kind, overlay, (address,), 1500),),
    )


def _granularities(plan: ModelProfilingPlan) -> list[object]:
    return [
        case.spec.granularity
        for case in plan.cases
        if not isinstance(case.spec, NetworkCaseSpec)
    ]


def _accepts(strategy: ProfilingStrategy) -> str:
    """Structural protocol conformance (mypy checks this call statically)."""
    return strategy.strategy_id


class TestStrategyIdentity:
    def test_strategy_id_is_stable(self) -> None:
        assert DEFAULT_STRATEGY_ID == "default-v1"
        assert DefaultProfilingStrategy().strategy_id == DEFAULT_STRATEGY_ID

    def test_satisfies_strategy_protocol(self) -> None:
        assert _accepts(DefaultProfilingStrategy()) == "default-v1"

    def test_constructor_validates_knobs(self) -> None:
        with pytest.raises(ValueError, match="representatives_per_class"):
            DefaultProfilingStrategy(representatives_per_class=0)
        with pytest.raises(ValueError, match="bandwidth_duration_s"):
            DefaultProfilingStrategy(bandwidth_duration_s=0.0)
        with pytest.raises(ValueError, match="same_subnet_prefix_length"):
            DefaultProfilingStrategy(same_subnet_prefix_length=0)
        with pytest.raises(ValueError, match="same_subnet_prefix_length"):
            DefaultProfilingStrategy(same_subnet_prefix_length=33)


class TestModelPlanning:
    def test_plan_covers_steps_four_to_six_in_order(self) -> None:
        plan = DefaultProfilingStrategy().plan_model_cases(
            worker_id=WORKER, facts=_facts(5), device_ids=("gpu-0",)
        )
        # 2 missing operators + 2 unique sparse modules + 3 positions x 3 lengths.
        assert len(plan.cases) == 13
        assert _granularities(plan) == [
            ProfilingGranularity.OPERATOR,
            ProfilingGranularity.OPERATOR,
            ProfilingGranularity.MODULE,
            ProfilingGranularity.MODULE,
            *([ProfilingGranularity.TRANSFORMER_LAYER] * 9),
        ]
        assert all(case.worker_id == WORKER for case in plan.cases)
        assert all(
            case.case_id == profiling_case_id(WORKER, case.spec) for case in plan.cases
        )

    def test_operator_cases_match_missing_signatures(self) -> None:
        """§47 step 4 dispatches exactly the §25-vocabulary signatures."""
        plan = DefaultProfilingStrategy().plan_model_cases(
            worker_id=WORKER, facts=_facts(), device_ids=("gpu-0",)
        )
        operator_ids = {
            case.case_id
            for case in plan.cases
            if case.spec.granularity is ProfilingGranularity.OPERATOR
        }
        assert operator_ids == {
            profiling_case_id(WORKER, operator_case_spec(sig, device_id="gpu-0"))
            for sig in (GEMM_SIG, ATTN_OP_SIG)
        }

    def test_measured_signatures_are_reused_per_device(self) -> None:
        """§28: what a device's class already measured is never re-benchmarked."""
        plan = DefaultProfilingStrategy().plan_model_cases(
            worker_id=WORKER,
            facts=_facts(),
            device_ids=("gpu-0", "gpu-1"),
            measured_signature_ids={"gpu-0": {operator_signature_id(GEMM_SIG)}},
        )
        assert plan.operator_reuse["gpu-0"].reused == (GEMM_SIG,)
        assert plan.operator_reuse["gpu-0"].missing == (ATTN_OP_SIG,)
        # An unlisted device is planned from scratch — the conservative direction.
        assert plan.operator_reuse["gpu-1"].reused == ()
        assert plan.operator_reuse["gpu-1"].missing == (GEMM_SIG, ATTN_OP_SIG)
        operator_cases = [
            case
            for case in plan.cases
            if case.spec.granularity is ProfilingGranularity.OPERATOR
        ]
        assert len(operator_cases) == 3
        per_device: dict[str, list[ProfilingCase]] = {}
        for case in operator_cases:
            per_device.setdefault(case.spec.device_ids[0], []).append(case)
        assert len(per_device["gpu-0"]) == 1
        assert len(per_device["gpu-1"]) == 2

    def test_deferred_signatures_are_visible_never_dispatched(self) -> None:
        """§19: custom + out-of-vocabulary kinds are reported, not benchmarked."""
        plan = DefaultProfilingStrategy().plan_model_cases(
            worker_id=WORKER, facts=_facts(), device_ids=("gpu-0",)
        )
        assert plan.deferred_operator_signatures == (EMBED_SIG, CUSTOM_SIG)
        dispatched = {
            case.spec.operator_signature
            for case in plan.cases
            if case.spec.granularity is ProfilingGranularity.OPERATOR
        }
        assert EMBED_SIG not in dispatched and CUSTOM_SIG not in dispatched

    def test_duplicate_signatures_collapse(self) -> None:
        plan = DefaultProfilingStrategy().plan_model_cases(
            worker_id=WORKER, facts=_facts(), device_ids=("gpu-0",)
        )
        gemm_ids = [
            case.case_id
            for case in plan.cases
            if case.spec.granularity is ProfilingGranularity.OPERATOR
            and case.spec.operator_signature == GEMM_SIG
        ]
        assert len(gemm_ids) == 1  # the facts listed GEMM_SIG twice

    def test_module_cases_are_sparse_and_deduped(self) -> None:
        plan = DefaultProfilingStrategy().plan_model_cases(
            worker_id=WORKER, facts=_facts(), device_ids=("gpu-0",)
        )
        module_specs = [
            case.spec
            for case in plan.cases
            if case.spec.granularity is ProfilingGranularity.MODULE
        ]
        assert [spec.module_signature for spec in module_specs] == [
            ATTN_MODULE_SIG,
            MLP_MODULE_SIG,
        ]
        for spec in module_specs:
            assert spec.model == MODEL
            assert spec.dtype == "fp32"
            assert spec.device_ids == ("gpu-0",)
            assert spec.sequence_length == DEFAULT_MODULE_SEQUENCE_LENGTH
            assert spec.phase is InferencePhase.PREFILL

    def test_layer_cases_cover_positions_and_lengths(self) -> None:
        plan = DefaultProfilingStrategy().plan_model_cases(
            worker_id=WORKER, facts=_facts(5), device_ids=("gpu-0",)
        )
        layer_specs = [
            case.spec
            for case in plan.cases
            if case.spec.granularity is ProfilingGranularity.TRANSFORMER_LAYER
        ]
        assert {(spec.layer_index, spec.sequence_length) for spec in layer_specs} == {
            (index, length)
            for _position, index in representative_layer_positions(5)
            for length in DEFAULT_PREFILL_SEQUENCE_LENGTHS
        }
        assert len(layer_specs) == 9  # layer_index keeps positional cases distinct (§22)
        for spec in layer_specs:
            assert spec.layer_signature == LAYER_SIG
            assert spec.model == MODEL and spec.dtype == "fp32"

    def test_every_device_gets_module_and_layer_cases(self) -> None:
        plan = DefaultProfilingStrategy().plan_model_cases(
            worker_id=WORKER, facts=_facts(5), device_ids=("gpu-0", "gpu-1")
        )
        devices = {
            case.spec.device_ids
            for case in plan.cases
            if case.spec.granularity is not ProfilingGranularity.OPERATOR
        }
        assert devices == {("gpu-0",), ("gpu-1",)}
        assert len(plan.cases) == 4 + 4 + 18  # operators + modules + layers

    def test_single_layer_model_collapses_positions(self) -> None:
        plan = DefaultProfilingStrategy().plan_model_cases(
            worker_id=WORKER, facts=_facts(1), device_ids=("gpu-0",)
        )
        layer_specs = [
            case.spec
            for case in plan.cases
            if case.spec.granularity is ProfilingGranularity.TRANSFORMER_LAYER
        ]
        assert len(layer_specs) == len(DEFAULT_PREFILL_SEQUENCE_LENGTHS)
        assert all(spec.layer_index == 0 for spec in layer_specs)

    def test_incomplete_layer_enumeration_fails_loudly(self) -> None:
        """§47: planning never proceeds on facts that miss a required position."""
        facts = _facts(
            3,
            layer_entries=tuple(
                LayerEntry(index, f"model.layers.{index}", LAYER_SIG)
                for index in (1, 2, 3)
            ),
        )
        with pytest.raises(ValueError, match="missing index 0"):
            DefaultProfilingStrategy().plan_model_cases(
                worker_id=WORKER, facts=facts, device_ids=("gpu-0",)
            )

    def test_device_validation(self) -> None:
        strategy = DefaultProfilingStrategy()
        with pytest.raises(ValueError, match="device_ids"):
            strategy.plan_model_cases(worker_id=WORKER, facts=_facts(), device_ids=())
        with pytest.raises(ValueError, match="empty entries"):
            strategy.plan_model_cases(worker_id=WORKER, facts=_facts(), device_ids=("",))
        with pytest.raises(ValueError, match="duplicate device_id"):
            strategy.plan_model_cases(
                worker_id=WORKER, facts=_facts(), device_ids=("gpu-0", "gpu-0")
            )

    def test_facts_without_modules_or_operators_plan_layers_only(self) -> None:
        facts = _facts(2, module_entries=(), operator_signatures=())
        plan = DefaultProfilingStrategy().plan_model_cases(
            worker_id=WORKER, facts=facts, device_ids=("gpu-0",)
        )
        assert _granularities(plan) == [ProfilingGranularity.TRANSFORMER_LAYER] * 6
        assert plan.operator_reuse["gpu-0"].missing == ()
        assert plan.deferred_operator_signatures == ()


class TestModelProfilingPlanValidation:
    def test_worker_id_must_not_be_empty(self) -> None:
        with pytest.raises(ValueError, match="worker_id"):
            ModelProfilingPlan(
                worker_id="", cases=(), operator_reuse={}, deferred_operator_signatures=()
            )

    def test_cases_must_belong_to_the_planned_worker(self) -> None:
        case = ProfilingCase.for_spec(
            "w-other", operator_case_spec(GEMM_SIG, device_id="gpu-0")
        )
        with pytest.raises(ValueError, match="assigned to worker"):
            ModelProfilingPlan(
                worker_id=WORKER,
                cases=(case,),
                operator_reuse={},
                deferred_operator_signatures=(),
            )


class TestNetworkPlanning:
    def test_endpoint_and_classification_facts_returned(self) -> None:
        """§47 network steps 1-2 are plan outputs, ready for the §43 registries."""
        plan = DefaultProfilingStrategy().plan_network_cases(
            facts=[
                _worker_facts("w-a", "192.168.1.10"),
                _worker_facts("w-b", "192.168.1.20"),
            ]
        )
        assert {profile.worker_id for profile in plan.endpoint_profiles} == {"w-a", "w-b"}
        assert all(profile.link_speed_mbps is None for profile in plan.endpoint_profiles)
        assert len(plan.classified_pairs) == 2
        assert all(
            entry.path_class is NetworkPathClass.WIRED_LAN
            for entry in plan.classified_pairs
        )

    def test_dense_rtt_then_sparse_bandwidth(self) -> None:
        plan = DefaultProfilingStrategy().plan_network_cases(
            facts=[
                _worker_facts("w-a", "192.168.1.10"),
                _worker_facts("w-b", "192.168.1.20"),
            ]
        )
        specs = [case.spec for case in plan.cases]
        assert [spec.probe_kind for spec in specs] == [
            ProbeKind.RTT,
            ProbeKind.RTT,
            ProbeKind.BANDWIDTH,
            ProbeKind.BANDWIDTH,
        ]
        rtt = specs[:2]
        assert {(spec.source_worker_id, spec.destination_worker_id) for spec in rtt} == {
            ("w-a", "w-b"),
            ("w-b", "w-a"),
        }
        bandwidth = specs[2:]
        assert [spec.direction for spec in bandwidth] == [
            NetworkDirection.FORWARD,
            NetworkDirection.REVERSE,
        ]
        for spec in bandwidth:
            assert (spec.source_worker_id, spec.destination_worker_id) == ("w-a", "w-b")
            assert spec.transport is NetworkTransport.TCP
            assert spec.duration_s == DEFAULT_IPERF3_DURATION_S
            assert spec.path_class is NetworkPathClass.WIRED_LAN
        # Network cases execute on their source worker (§8.2).
        assert all(case.worker_id == case.spec.source_worker_id for case in plan.cases)

    def test_bandwidth_is_sparse_per_class(self) -> None:
        plan = DefaultProfilingStrategy().plan_network_cases(
            facts=[
                _worker_facts("w-a", "192.168.1.10"),
                _worker_facts("w-b", "192.168.1.20"),
                _worker_facts("w-c", "192.168.1.30", name="wlan0"),
            ]
        )
        rtt = [case.spec for case in plan.cases if case.spec.probe_kind is ProbeKind.RTT]
        bandwidth = [
            case.spec for case in plan.cases if case.spec.probe_kind is ProbeKind.BANDWIDTH
        ]
        assert len(rtt) == 6  # dense: every directed pair
        # Sparse: one deterministic representative per class, both directions.
        assert len(bandwidth) == 4
        pairs = {
            (spec.source_worker_id, spec.destination_worker_id): spec.path_class
            for spec in bandwidth
        }
        assert pairs == {
            ("w-a", "w-b"): NetworkPathClass.WIRED_LAN,
            ("w-a", "w-c"): NetworkPathClass.WIFI_LAN,
        }

    def test_path_class_filter_restricts_bandwidth_only(self) -> None:
        """§49 ``network bandwidth --path-class`` knob."""
        facts = [
            _worker_facts("w-a", "192.168.1.10"),
            _worker_facts("w-b", "192.168.1.20"),
            _worker_facts("w-c", "192.168.1.30", name="wlan0"),
        ]
        plan = DefaultProfilingStrategy().plan_network_cases(
            facts=facts, bandwidth_path_classes=[NetworkPathClass.WIFI_LAN]
        )
        rtt = [case.spec for case in plan.cases if case.spec.probe_kind is ProbeKind.RTT]
        bandwidth = [
            case.spec for case in plan.cases if case.spec.probe_kind is ProbeKind.BANDWIDTH
        ]
        assert len(rtt) == 6  # the dense matrix is untouched
        assert bandwidth
        assert all(spec.path_class is NetworkPathClass.WIFI_LAN for spec in bandwidth)

    def test_explicit_pair_knob(self) -> None:
        facts = [
            _worker_facts("w-a", "192.168.1.10"),
            _worker_facts("w-b", "192.168.1.20"),
        ]
        plan = DefaultProfilingStrategy().plan_network_cases(
            facts=facts,
            extra_bandwidth_pairs=[NetworkPair("w-b", "w-a")],
        )
        bandwidth_pairs = {
            (case.spec.source_worker_id, case.spec.destination_worker_id)
            for case in plan.cases
            if case.spec.probe_kind is ProbeKind.BANDWIDTH
        }
        assert ("w-a", "w-b") in bandwidth_pairs  # class representative
        assert ("w-b", "w-a") in bandwidth_pairs  # explicit knob

    def test_single_worker_yields_empty_cases(self) -> None:
        plan = DefaultProfilingStrategy().plan_network_cases(
            facts=[_worker_facts("w-a", "10.0.0.1")]
        )
        assert len(plan.endpoint_profiles) == 1
        assert plan.classified_pairs == ()
        assert plan.cases == ()

    def test_duplicate_worker_facts_rejected(self) -> None:
        worker = _worker_facts("w-a", "10.0.0.1")
        with pytest.raises(ValueError, match="duplicate worker_id"):
            DefaultProfilingStrategy().plan_network_cases(facts=[worker, worker])

    def test_plan_is_deterministic(self) -> None:
        """§7: the same facts always plan the same canonical case ids."""
        facts = [
            _worker_facts("w-a", "192.168.1.10"),
            _worker_facts("w-b", "192.168.1.20"),
        ]
        strategy = DefaultProfilingStrategy()
        first = strategy.plan_network_cases(facts=facts)
        second = strategy.plan_network_cases(facts=facts)
        assert [case.case_id for case in first.cases] == [
            case.case_id for case in second.cases
        ]
