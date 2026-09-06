"""Control-plane domain ↔ wire mapping tests (Phase 1 spec §40, §29-31).

Every domain structure built by P1A-P1D must survive the protobuf round
trip exactly, including the None-vs-zero distinctions spec §17 depends on.
"""

from __future__ import annotations

import dataclasses
import uuid
from datetime import UTC, datetime

import pytest

from edgeshard.cluster.capability import OSInfo, compute_capability_revision
from edgeshard.cluster.state import DeviceAvailability, DeviceState, MemoryPoolState
from edgeshard.protocol.control import mapper
from edgeshard.protocol.control.mapper import (
    CONTROL_PROTOCOL_VERSION,
    ControlProtocolError,
    HeartbeatRequest,
    HeartbeatResponse,
    RegisterWorkerRequest,
    RegisterWorkerResponse,
    RejectionReason,
    UpdateCapabilityRequest,
    UpdateCapabilityResponse,
)
from edgeshard.protocol.control.pb import worker_control_pb2 as pb
from factories import (
    make_jetson_capability,
    make_rtx_capability,
    make_worker_identity,
    make_worker_state,
)

WORKER_ID = str(uuid.uuid4())
INSTANCE_ID = str(uuid.uuid4())


def test_identity_roundtrip() -> None:
    identity = make_worker_identity(WORKER_ID)
    assert mapper.identity_from_wire(mapper.identity_to_wire(identity)) == identity


def test_rtx_capability_roundtrip() -> None:
    capability = make_rtx_capability()
    assert mapper.capability_from_wire(mapper.capability_to_wire(capability)) == capability


def test_jetson_capability_roundtrip() -> None:
    capability = make_jetson_capability()
    restored = mapper.capability_from_wire(mapper.capability_to_wire(capability))
    assert restored == capability
    # The shared system-memory pool and both device references survive.
    assert len(restored.memory_pools) == 1
    assert {device.memory_pool_id for device in restored.devices} == {"system-memory"}


def test_none_scalar_fields_survive() -> None:
    """Missing telemetry is field absence on the wire, not a sentinel (§17)."""
    worker_state = dataclasses.replace(
        make_worker_state(WORKER_ID),
        device_states=(
            DeviceState(
                device_id="dev-a",
                utilization=None,
                temperature_c=None,
                power_w=None,
                availability=DeviceAvailability.UNKNOWN,
                running_runtime_ids=(),
            ),
        ),
        memory_states=(MemoryPoolState(memory_pool_id="host-memory", available_bytes=None),),
        runtime_instances=(),
        models=(),
    )
    restored = mapper.state_from_wire(mapper.state_to_wire(worker_state))
    assert restored == worker_state
    (device_state,) = restored.device_states
    assert device_state.utilization is None
    assert device_state.temperature_c is None
    assert device_state.power_w is None
    (memory_state,) = restored.memory_states
    assert memory_state.available_bytes is None


def test_zero_metrics_distinct_from_none() -> None:
    worker_state = dataclasses.replace(
        make_worker_state(WORKER_ID),
        device_states=(
            DeviceState(
                device_id="dev-a",
                utilization=0.0,
                temperature_c=0.0,
                power_w=0.0,
                availability=DeviceAvailability.AVAILABLE,
                running_runtime_ids=(),
            ),
        ),
        memory_states=(MemoryPoolState(memory_pool_id="host-memory", available_bytes=0),),
        runtime_instances=(),
        models=(),
    )
    restored = mapper.state_from_wire(mapper.state_to_wire(worker_state))
    assert restored == worker_state
    (device_state,) = restored.device_states
    assert device_state.utilization == 0.0
    assert device_state.power_w == 0.0
    (memory_state,) = restored.memory_states
    assert memory_state.available_bytes == 0


def test_empty_string_distinct_from_none() -> None:
    capability = dataclasses.replace(
        make_rtx_capability(),
        os=OSInfo(name="ubuntu", version="", kernel=None),
    )
    restored = mapper.capability_from_wire(mapper.capability_to_wire(capability))
    assert restored.os.version == ""
    assert restored.os.kernel is None


def test_telemetry_float_precision_survives() -> None:
    worker_state = dataclasses.replace(
        make_worker_state(WORKER_ID),
        device_states=(
            DeviceState(
                device_id="dev-a",
                utilization=33.3,
                temperature_c=46.5,
                power_w=5.123456789,
                availability=DeviceAvailability.AVAILABLE,
                running_runtime_ids=(),
            ),
        ),
        runtime_instances=(),
        models=(),
    )
    restored = mapper.state_from_wire(mapper.state_to_wire(worker_state))
    assert restored == worker_state


def test_state_roundtrip_with_inventory() -> None:
    state = make_worker_state(WORKER_ID)
    assert mapper.state_from_wire(mapper.state_to_wire(state)) == state


def make_register_request(
    *,
    identity=None,
    capability=None,
    state=None,
    protocol_version=CONTROL_PROTOCOL_VERSION,
    instance_id=INSTANCE_ID,
    profiling_endpoint: str | None = None,
) -> RegisterWorkerRequest:
    identity = identity or make_worker_identity(WORKER_ID)
    capability = capability or make_rtx_capability()
    state = state or make_worker_state(identity.worker_id)
    return RegisterWorkerRequest(
        protocol_version=protocol_version,
        instance_id=instance_id,
        identity=identity,
        capability=capability,
        initial_state=state,
        profiling_endpoint=profiling_endpoint,
    )


def test_register_request_roundtrip() -> None:
    request = make_register_request()
    assert mapper.register_request_from_wire(
        mapper.register_request_to_wire(request)
    ) == request


def test_register_request_rejects_protocol_version_mismatch() -> None:
    wire = mapper.register_request_to_wire(make_register_request())
    wire.protocol_version = "999"
    with pytest.raises(ControlProtocolError, match="protocol version"):
        mapper.register_request_from_wire(wire)


def test_register_request_rejects_identity_protocol_version_mismatch() -> None:
    """§47: the redundant identity copy must agree with the request version."""
    identity = dataclasses.replace(
        make_worker_identity(WORKER_ID), protocol_version="999"
    )
    with pytest.raises(ControlProtocolError, match="identity protocol_version"):
        make_register_request(identity=identity)


def test_register_request_rejects_worker_id_mismatch() -> None:
    wire = mapper.register_request_to_wire(make_register_request())
    wire.worker_id = str(uuid.uuid4())
    with pytest.raises(ControlProtocolError, match="worker_id mismatch"):
        mapper.register_request_from_wire(wire)


def test_register_request_rejects_capability_revision_mismatch() -> None:
    wire = mapper.register_request_to_wire(make_register_request())
    wire.capability_revision = "not-the-real-revision"
    with pytest.raises(ControlProtocolError, match="capability_revision mismatch"):
        mapper.register_request_from_wire(wire)


def test_register_request_rejects_state_worker_id_mismatch() -> None:
    with pytest.raises(ValueError, match="worker_id mismatch"):
        make_register_request(state=make_worker_state(str(uuid.uuid4())))


def test_register_request_rejects_future_protocol_version() -> None:
    with pytest.raises(ControlProtocolError, match="protocol version"):
        make_register_request(protocol_version="2")


def test_register_response_roundtrip() -> None:
    response = RegisterWorkerResponse(
        session_id=str(uuid.uuid4()),
        heartbeat_interval_ms=5000,
        server_protocol_version=CONTROL_PROTOCOL_VERSION,
    )
    assert mapper.register_response_from_wire(
        mapper.register_response_to_wire(response)
    ) == response


def test_register_response_rejects_wrong_server_version() -> None:
    with pytest.raises(ControlProtocolError, match="invalid registration response"):
        RegisterWorkerResponse(
            session_id="session-1", heartbeat_interval_ms=5000, server_protocol_version="2"
        )


def test_register_response_rejects_empty_session() -> None:
    with pytest.raises(ValueError, match="session_id"):
        RegisterWorkerResponse(
            session_id="",
            heartbeat_interval_ms=5000,
            server_protocol_version=CONTROL_PROTOCOL_VERSION,
        )


def test_register_response_rejects_non_positive_interval() -> None:
    with pytest.raises(ValueError, match="heartbeat_interval_ms"):
        RegisterWorkerResponse(
            session_id="session-1",
            heartbeat_interval_ms=0,
            server_protocol_version=CONTROL_PROTOCOL_VERSION,
        )


def make_heartbeat_request(
    *, sequence_number: int = 1, worker_reported_at: datetime | None = None
) -> HeartbeatRequest:
    return HeartbeatRequest(
        worker_id=WORKER_ID,
        instance_id=INSTANCE_ID,
        session_id="session-1",
        sequence_number=sequence_number,
        capability_revision=make_rtx_capability().capability_revision,
        state=make_worker_state(WORKER_ID),
        worker_reported_at=worker_reported_at,
    )


def test_heartbeat_request_roundtrip() -> None:
    request = make_heartbeat_request()
    assert mapper.heartbeat_request_from_wire(
        mapper.heartbeat_request_to_wire(request)
    ) == request


def test_heartbeat_request_timestamp_roundtrip() -> None:
    reported_at = datetime(2026, 9, 5, 12, 34, 56, 123000, tzinfo=UTC)
    request = make_heartbeat_request(worker_reported_at=reported_at)
    restored = mapper.heartbeat_request_from_wire(
        mapper.heartbeat_request_to_wire(request)
    )
    assert restored == request
    assert restored.worker_reported_at == reported_at


def test_heartbeat_request_timestamp_truncates_to_milliseconds() -> None:
    reported_at = datetime(2026, 9, 5, 12, 34, 56, 123999, tzinfo=UTC)
    request = make_heartbeat_request(worker_reported_at=reported_at)
    wire = mapper.heartbeat_request_to_wire(request)
    restored = mapper.heartbeat_request_from_wire(wire)
    assert restored.worker_reported_at is not None
    assert restored.worker_reported_at.microsecond in (123000, 124000)


def test_heartbeat_request_requires_positive_sequence() -> None:
    for bad_sequence in (0, -1):
        with pytest.raises(ValueError, match="sequence_number"):
            make_heartbeat_request(sequence_number=bad_sequence)


def test_heartbeat_request_requires_matching_state_worker() -> None:
    with pytest.raises(ValueError, match="worker_id mismatch"):
        HeartbeatRequest(
            worker_id=WORKER_ID,
            instance_id=INSTANCE_ID,
            session_id="session-1",
            sequence_number=1,
            capability_revision="rev",
            state=make_worker_state(str(uuid.uuid4())),
        )


def test_heartbeat_request_requires_aware_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        make_heartbeat_request(
            worker_reported_at=datetime(2026, 9, 5, 12, 0, 0)
        )


def test_heartbeat_response_roundtrip() -> None:
    for response in (
        HeartbeatResponse(accepted=True),
        HeartbeatResponse(
            accepted=False,
            detail="stale session",
            reason=RejectionReason.STALE_SESSION,
        ),
    ):
        assert mapper.heartbeat_response_from_wire(
            mapper.heartbeat_response_to_wire(response)
        ) == response


def test_every_rejection_reason_survives_the_wire() -> None:
    for reason in RejectionReason:
        response = HeartbeatResponse(
            accepted=False, detail=f"rejected: {reason.value}", reason=reason
        )
        restored = mapper.heartbeat_response_from_wire(
            mapper.heartbeat_response_to_wire(response)
        )
        assert restored.reason is reason


def test_rejected_heartbeat_requires_detail() -> None:
    with pytest.raises(ValueError, match="detail"):
        HeartbeatResponse(accepted=False)


def test_rejected_heartbeat_requires_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        HeartbeatResponse(accepted=False, detail="stale session")


def test_accepted_heartbeat_forbids_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        HeartbeatResponse(accepted=True, reason=RejectionReason.STALE_SESSION)


def test_unknown_wire_rejection_reason_rejected() -> None:
    wire = mapper.heartbeat_response_to_wire(
        HeartbeatResponse(
            accepted=False, detail="x", reason=RejectionReason.UNKNOWN_WORKER
        )
    )
    wire.reason = 99  # proto3 enums are open; decode must reject
    with pytest.raises(ControlProtocolError, match="rejection reason"):
        mapper.heartbeat_response_from_wire(wire)


def make_update_request(capability=None, state=None) -> UpdateCapabilityRequest:
    return UpdateCapabilityRequest(
        worker_id=WORKER_ID,
        instance_id=INSTANCE_ID,
        session_id="session-1",
        capability=capability or make_jetson_capability(),
        state=state or make_worker_state(WORKER_ID),
    )


def test_update_capability_roundtrip() -> None:
    request = make_update_request()
    assert mapper.update_capability_request_from_wire(
        mapper.update_capability_request_to_wire(request)
    ) == request


def test_update_capability_requires_state_on_wire() -> None:
    """§16: an update without its atomic state sample cannot be applied."""
    wire = mapper.update_capability_request_to_wire(make_update_request())
    wire.ClearField("state")
    with pytest.raises(ControlProtocolError, match="atomically"):
        mapper.update_capability_request_from_wire(wire)


def test_update_capability_requires_matching_state_worker() -> None:
    with pytest.raises(ValueError, match="worker_id mismatch"):
        make_update_request(state=make_worker_state(str(uuid.uuid4())))


def test_update_capability_response_roundtrip() -> None:
    for response in (
        UpdateCapabilityResponse(accepted=True),
        UpdateCapabilityResponse(
            accepted=False,
            detail="stale session",
            reason=RejectionReason.STALE_SESSION,
        ),
    ):
        assert mapper.update_capability_response_from_wire(
            mapper.update_capability_response_to_wire(response)
        ) == response


def test_rejected_capability_update_requires_detail() -> None:
    with pytest.raises(ValueError, match="detail"):
        UpdateCapabilityResponse(accepted=False)


def test_rejected_capability_update_requires_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        UpdateCapabilityResponse(accepted=False, detail="stale session")


def test_capability_revision_roundtrip_is_stable() -> None:
    capability = make_rtx_capability()
    restored = mapper.capability_from_wire(mapper.capability_to_wire(capability))
    assert compute_capability_revision(restored) == capability.capability_revision


def test_unknown_availability_enum_rejected() -> None:
    wire = pb.DeviceState(device_id="dev-a")
    wire.availability = 99  # proto3 enums are open; decode must reject
    with pytest.raises(ControlProtocolError, match="device availability"):
        mapper._device_state_from_wire(wire)


def test_unknown_memory_model_enum_rejected() -> None:
    wire = pb.MemoryPoolCapability(memory_pool_id="pool-a", total_bytes=8)
    wire.model = 42
    with pytest.raises(ControlProtocolError, match="memory model"):
        mapper._memory_pool_from_wire(wire)


# -- profiling endpoint advertisement (Phase 2 spec §41, additive) -----------


def test_register_request_profiling_endpoint_roundtrip() -> None:
    request = make_register_request(profiling_endpoint="10.0.0.5:51100")
    restored = mapper.register_request_from_wire(mapper.register_request_to_wire(request))
    assert restored == request
    assert restored.profiling_endpoint == "10.0.0.5:51100"


def test_register_request_without_profiling_endpoint_stays_absent() -> None:
    """proto3 optional: absence means "does not host profiling", never a
    sentinel — Phase 1 workers keep producing byte-identical requests (§41)."""
    request = make_register_request()
    wire = mapper.register_request_to_wire(request)
    assert not wire.HasField("profiling_endpoint")
    restored = mapper.register_request_from_wire(wire)
    assert restored.profiling_endpoint is None
    assert restored == request


def test_register_request_rejects_empty_profiling_endpoint() -> None:
    with pytest.raises(ValueError, match="profiling_endpoint"):
        make_register_request(profiling_endpoint="")
