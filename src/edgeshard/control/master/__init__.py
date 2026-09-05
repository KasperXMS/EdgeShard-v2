"""Master-side control plane (Phase 1 spec §33).

Internal architecture of the Master::

    MasterService
    │
    ├── WorkerRegistry    stable identity + capability (§34)
    ├── SessionManager    current session + heartbeat sequence (§35)
    ├── StateStore        latest accepted dynamic state (§36)
    ├── LivenessManager   monotonic-clock ONLINE/SUSPECT/OFFLINE (§32, §37)
    └── SnapshotBuilder   immutable cluster snapshots (§38, milestone P1H)

``MasterService`` is the facade implementing the
:class:`~edgeshard.protocol.control.grpc_server.WorkerRegistryHandler`
protocol, so it plugs directly into the P1E gRPC transport. Everything is
in-memory (spec §34: no database in Phase 1) and assumes a single asyncio
event loop — handler methods never await between reading and writing
component state, so updates stay atomic.
"""
