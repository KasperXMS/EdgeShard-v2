"""Typed profiling errors (Phase 2 spec §42).

Every failure inside the profiling subsystem is raised as a
:class:`ProfilingError` carrying one of the fixed
:class:`~edgeshard.profiling.domain.experiment.ProfilingErrorCategory`
values, so the runner can translate exceptions into the domain's
:class:`~edgeshard.profiling.domain.experiment.ProfilingFailure` records
without string parsing. Failures are never encoded as zero latency,
empty samples, or ``None`` measurement values.

Derives from :class:`~edgeshard.model.errors.EdgeShardError` so existing
callers can keep catching the single project-wide base type.
"""

from __future__ import annotations

from collections.abc import Mapping

from edgeshard.model.errors import EdgeShardError
from edgeshard.profiling.domain.experiment import (
    ProfilingErrorCategory,
    ProfilingFailure,
)
from edgeshard.profiling.domain.hashing import JsonScalar, normalized_items


class ProfilingError(EdgeShardError):
    """A profiling failure with a typed category and JSON-safe details."""

    def __init__(
        self,
        category: ProfilingErrorCategory,
        message: str,
        details: Mapping[str, JsonScalar] | None = None,
    ) -> None:
        if not message:
            raise ValueError("ProfilingError message must not be empty")
        super().__init__(message)
        self._category = category
        self._details = normalized_items(details or {}, "details")

    @property
    def category(self) -> ProfilingErrorCategory:
        return self._category

    @property
    def details(self) -> tuple[tuple[str, JsonScalar], ...]:
        return self._details

    def to_failure(self) -> ProfilingFailure:
        """The domain failure record persisted for this error (spec §42)."""
        return ProfilingFailure(
            category=self._category,
            message=str(self),
            details=self._details,
        )
