"""Extensible profiling and workload characterization (Phase 2).

Phase 2 records *what has been empirically observed* — nothing else:

    Phase 1  ClusterSnapshot            what resources exist
    Phase 2  ProfileSnapshot            what has been measured   <- here
    Phase 3  ResourceEstimateSnapshot   what is expected to happen
    Phase 4  Scheduler                  where the workload should run

The domain layer (``profiling.domain``) is pure Python and free of torch,
gRPC, Docker, NVML, tegrastats, and SQLite (spec §4, P2A DoD).
Instrumentation, benchmark harness, profilers, and strategies build on top
of the domain; orchestration lives in ``control``/``protocol`` (spec §56.5).

Phase 2 MUST NOT contain placement decisions, interpolation, learned
prediction, scheduling, or deployment logic (spec §0).
"""
