"""Worker capability/telemetry discovery probes (Phase 1 spec §25).

Static capability discovery (``CapabilityProbe``) and dynamic telemetry
(``TelemetryProbe`` in ``edgeshard.control.worker.telemetry.base``) are
separate because their lifetimes differ: capability is collected once per
Agent start, telemetry once per sample/heartbeat.
"""
