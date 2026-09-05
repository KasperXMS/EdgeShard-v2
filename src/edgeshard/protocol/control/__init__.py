"""Control-plane protocol: wire DTOs, mapping, and the gRPC transport.

Phase 1 §7 layout: the control plane keeps its own protocol package next to
the Phase 0 shard protocol. ``mapper`` is the single cluster-dataclass ↔
protobuf-DTO boundary (spec §40); ``grpc_client``/``grpc_server`` carry the
three control RPCs (spec §28) over ``grpc.aio``. Generated protobuf code stays
in ``pb/`` and never leaks past this package.
"""
