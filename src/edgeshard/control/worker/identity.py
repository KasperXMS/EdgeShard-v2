"""Worker identity management (Phase 1 spec §10-11, §27).

``IdentityManager`` owns the persistent ``worker_id``: a UUIDv4 generated on
first start and written to ``identity_path``, then reused across every Agent
restart. It is never derived from hostname, IP/MAC address, or CUDA ordinal
(spec §10.1). A missing file creates the identity *atomically and
exclusively* — two Agents starting simultaneously against a fresh install
converge on one worker_id via a unique temp file plus ``os.link``; a
corrupt file fails loudly instead of silently re-identifying the Worker
(spec §47).

Device ids derived without hardware UUIDs (generic-host CPU) come from the
persistent ``worker_id`` via UUIDv5 - the same scheme spec §11 sanctions for
integrated Jetson devices - so they stay stable across reboots.
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import uuid
from pathlib import Path

from edgeshard import __version__
from edgeshard.cluster.identity import WorkerIdentity
from edgeshard.model.errors import EdgeShardError
from edgeshard.protocol.control.mapper import CONTROL_PROTOCOL_VERSION

logger = logging.getLogger("worker.identity")

_CPU_DEVICE_KEY = "host-cpu"
_JETSON_GPU_DEVICE_KEY = "tegra-gpu"


class IdentityError(EdgeShardError):
    """The persistent Worker identity could not be loaded or created."""


class IdentityManager:
    """Loads or creates the persistent ``worker_id`` (spec §27, step 2)."""

    def __init__(self, identity_path: Path) -> None:
        self._identity_path = Path(identity_path)

    @property
    def identity_path(self) -> Path:
        return self._identity_path

    def load_or_create(self) -> str:
        """The persistent worker_id; created and saved on first use."""
        try:
            raw = self._identity_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return self._create()
        except OSError as exc:
            raise IdentityError(
                f"cannot read Worker identity file {self._identity_path}: {exc}"
            ) from exc
        try:
            worker_id = str(uuid.UUID(raw))
        except ValueError as exc:
            raise IdentityError(
                f"Worker identity file {self._identity_path} does not contain "
                f"a valid UUID: {raw!r}"
            ) from exc
        logger.info("loaded worker_id=%s from %s", worker_id, self._identity_path)
        return worker_id

    def _create(self) -> str:
        """Create the identity file exclusively; adopt a concurrent winner's.

        The fresh id is written to a unique temp file in the same directory
        and then hard-linked onto the identity path. ``os.link`` fails with
        ``FileExistsError`` when another process won the race, and a linked
        name is only ever visible with its full content — so no reader can
        observe a partially written identity, and the loser simply loads
        the winner's worker_id instead of minting a second one.
        """
        worker_id = str(uuid.uuid4())
        directory = self._identity_path.parent
        tmp_path = directory / f".{self._identity_path.name}.{uuid.uuid4().hex}.tmp"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(worker_id + "\n", encoding="utf-8")
            try:
                os.link(tmp_path, self._identity_path)
            except FileExistsError:
                # Lost the creation race: the winner's file is complete.
                logger.info(
                    "identity file %s was created concurrently; loading the winner's id",
                    self._identity_path,
                )
                return self.load_or_create()
        except OSError as exc:
            raise IdentityError(
                f"cannot create Worker identity file {self._identity_path}: {exc} "
                f"(set worker.identity_path to a writable location)"
            ) from exc
        finally:
            with contextlib.suppress(OSError):
                tmp_path.unlink()
        logger.info("created worker_id=%s at %s", worker_id, self._identity_path)
        return worker_id


def derive_cpu_device_id(worker_id: str) -> str:
    """Stable identity of the host CPU device (spec §11).

    Generic hosts expose no hardware UUID for their CPU, so the id is
    derived deterministically from the persistent ``worker_id`` and is as
    stable as the identity file itself.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"edgeshard:{worker_id}:{_CPU_DEVICE_KEY}"))


def derive_jetson_gpu_device_id(worker_id: str) -> str:
    """Stable identity of the Jetson integrated GPU (spec §11).

    Like the host CPU, the integrated GPU has no hardware UUID; the id is
    derived from ``worker_id`` plus a stable platform device key.
    """
    return str(
        uuid.uuid5(uuid.NAMESPACE_DNS, f"edgeshard:{worker_id}:{_JETSON_GPU_DEVICE_KEY}")
    )


def new_instance_id() -> str:
    """A fresh per-process Agent instance id (spec §10.2)."""
    return str(uuid.uuid4())


def build_worker_identity(worker_id: str, hostname: str | None = None) -> WorkerIdentity:
    """The Worker installation identity carried in every report (spec §10.1)."""
    return WorkerIdentity(
        worker_id=worker_id,
        hostname=hostname or socket.gethostname(),
        agent_version=__version__,
        protocol_version=CONTROL_PROTOCOL_VERSION,
    )
