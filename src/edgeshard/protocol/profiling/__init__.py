"""Profiling-plane protocol (Phase 2 spec §41).

Master→Worker profiling RPCs (``WorkerProfilingService``) and the
Master-hosted admin surface (``ProfilingAdminService``). Generated protobuf
code stays in ``pb/`` and never leaks past this package; the mapper
translates between wire messages and profiling domain DTOs, mirroring the
Phase 1 control-plane discipline.
"""
