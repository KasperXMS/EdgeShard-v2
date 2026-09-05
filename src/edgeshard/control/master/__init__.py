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
protocol, so it plugs directly into the P1E gRPC transport, and exposes
:meth:`~edgeshard.control.master.service.MasterService.build_snapshot`
for the immutable ``ClusterSnapshot`` output of Phase 1 (spec §38).
Everything is in-memory (spec §34: no database in Phase 1) and assumes a
single asyncio event loop — handler methods never await between reading
and writing component state, so updates stay atomic. ``build_snapshot`` is
synchronous for the same reason: a snapshot is a true point-in-time copy
that concurrent heartbeats cannot mutate (spec §38, §51).
"""
