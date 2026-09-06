"""Telemetry context instrumentation (Phase 2 spec §15).

Telemetry is *context only*: it never corrects latency in Phase 2. The
harness captures an initial and a final device observation around the
measurement interval and records the outcome of an optional
contamination check — an obviously busy device must not silently pass
as a clean benchmark run.

``contaminated`` is tri-state, matching
:class:`~edgeshard.profiling.domain.measurement.TelemetryContextMetrics`:
``True``/``False`` when a check was applied, ``None`` when none was
(no instrumentation, no threshold, or the initial observation carried no
utilization).

Backends (NVML on discrete GPUs, tegrastats on Jetson — both Phase 1
probes) are adapted to :class:`TelemetryInstrumentation` on the control
side; this module deliberately does not import ``edgeshard.control``
(dependency direction, spec §56.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from edgeshard.profiling.domain.experiment import ProfilingErrorCategory
from edgeshard.profiling.domain.measurement import (
    TelemetryContextMetrics,
    TelemetrySample,
)
from edgeshard.profiling.errors import ProfilingError


@dataclass(frozen=True)
class DeviceObservation:
    """Raw contextual device observation; missing fields are ``None``."""

    device_id: str
    utilization: float | None = None
    temperature_c: float | None = None
    power_w: float | None = None
    clock_mhz: float | None = None
    memory_used_bytes: int | None = None

    def to_sample(self) -> TelemetrySample:
        """The domain sample; range validation happens there."""
        return TelemetrySample(
            device_id=self.device_id,
            utilization=self.utilization,
            temperature_c=self.temperature_c,
            power_w=self.power_w,
            clock_mhz=self.clock_mhz,
            memory_used_bytes=self.memory_used_bytes,
        )


@runtime_checkable
class TelemetryInstrumentation(Protocol):
    """One contextual observation of the profiled device (§15).

    ``None`` means the backend could not answer right now (§52.2); it
    never means "zero utilization".
    """

    def capture(self) -> DeviceObservation | None: ...


class ContaminationAction(StrEnum):
    """What to do when the initial observation exceeds the threshold."""

    MARK = "mark"
    REJECT = "reject"


class TelemetryContextCollector:
    """Initial/final telemetry context plus the contamination check.

    ``contamination_threshold_percent`` enables the check: with action
    ``MARK`` a busy device is recorded as ``contaminated=True`` and the
    benchmark proceeds; with ``REJECT`` it raises a typed
    :class:`~edgeshard.profiling.errors.ProfilingError`
    (``DEVICE_BUSY``) before any measurement runs.
    """

    def __init__(
        self,
        instrumentation: TelemetryInstrumentation | None = None,
        *,
        contamination_threshold_percent: float | None = None,
        contamination_action: ContaminationAction = ContaminationAction.MARK,
    ) -> None:
        if contamination_threshold_percent is not None and not (
            0.0 <= contamination_threshold_percent <= 100.0
        ):
            raise ValueError(
                "contamination_threshold_percent must be within [0, 100], "
                f"got {contamination_threshold_percent}"
            )
        self._instrumentation = instrumentation
        self._threshold = contamination_threshold_percent
        self._action = contamination_action
        self._initial: TelemetrySample | None = None
        self._contaminated: bool | None = None

    @property
    def contaminated(self) -> bool | None:
        """Tri-state check outcome; ``None`` = no check was applied."""
        return self._contaminated

    def capture_initial(self) -> None:
        """Capture the pre-benchmark observation and run the check (§15)."""
        observation = self._capture()
        self._initial = observation.to_sample() if observation is not None else None
        self._contaminated = None
        if (
            self._threshold is None
            or self._initial is None
            or self._initial.utilization is None
        ):
            return
        busy = self._initial.utilization > self._threshold
        self._contaminated = busy
        if busy and self._action is ContaminationAction.REJECT:
            raise ProfilingError(
                ProfilingErrorCategory.DEVICE_BUSY,
                "device utilization above contamination threshold at benchmark start",
                {
                    "device_id": self._initial.device_id,
                    "utilization": self._initial.utilization,
                    "threshold": self._threshold,
                },
            )

    def capture_final(self) -> TelemetryContextMetrics | None:
        """The interval's context metrics, or ``None`` without any sample."""
        observation = self._capture()
        final = observation.to_sample() if observation is not None else None
        if self._initial is None and final is None:
            return None
        return TelemetryContextMetrics(
            initial=self._initial,
            final=final,
            contaminated=self._contaminated,
        )

    def _capture(self) -> DeviceObservation | None:
        if self._instrumentation is None:
            return None
        return self._instrumentation.capture()
