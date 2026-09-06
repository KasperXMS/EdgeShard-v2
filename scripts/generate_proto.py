"""Regenerate protobuf code for the committed .proto files.

Usage (repository root, managed environment):

    uv run python scripts/generate_proto.py

Compiles ``proto/shard_runtime.proto`` (Phase 0 shard data plane),
``proto/worker_control.proto`` (Phase 1 control plane, spec 41), and
``proto/profiling.proto`` (Phase 2 profiling plane, Phase 2 spec §41). The
generated files are committed so runtime environments and CI never need
grpcio-tools (spec 4.10 reproducibility); grpcio-tools is a dev dependency
only. Generated code lands in the package listed per target and each grpc
stub's flat import is rewritten to a package import.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from grpc_tools import protoc

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTO_DIR = REPO_ROOT / "proto"
SRC_DIR = REPO_ROOT / "src" / "edgeshard"


@dataclass(frozen=True)
class ProtoTarget:
    """One .proto file and where its generated package lives."""

    proto: str
    out_dir: Path
    package: str

    @property
    def grpc_module(self) -> str:
        stem = Path(self.proto).stem
        return f"{stem}_pb2_grpc.py"


TARGETS = (
    ProtoTarget(
        proto="shard_runtime.proto",
        out_dir=SRC_DIR / "protocol" / "pb",
        package="edgeshard.protocol.pb",
    ),
    ProtoTarget(
        proto="worker_control.proto",
        out_dir=SRC_DIR / "protocol" / "control" / "pb",
        package="edgeshard.protocol.control.pb",
    ),
    ProtoTarget(
        proto="profiling.proto",
        out_dir=SRC_DIR / "protocol" / "profiling" / "pb",
        package="edgeshard.protocol.profiling.pb",
    ),
)


def generate(target: ProtoTarget) -> int:
    target.out_dir.mkdir(parents=True, exist_ok=True)
    result = protoc.main(
        [
            "protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={target.out_dir}",
            f"--grpc_python_out={target.out_dir}",
            f"--pyi_out={target.out_dir}",
            target.proto,
        ]
    )
    if result != 0:
        return result

    stem = Path(target.proto).stem
    flat_import = re.compile(rf"^import {stem}_pb2 as (\w+)$", re.MULTILINE)
    grpc_path = target.out_dir / target.grpc_module
    source = grpc_path.read_text(encoding="utf-8")
    rewritten, count = flat_import.subn(
        rf"from {target.package} import {stem}_pb2 as \1", source
    )
    if count != 1:
        print(f"unexpected generated import layout in {grpc_path}", file=sys.stderr)
        return 1
    grpc_path.write_text(rewritten, encoding="utf-8", newline="\n")

    init_path = target.out_dir / "__init__.py"
    if not init_path.exists():
        init_path.write_text(
            '"""Generated protobuf code. Regenerate via scripts/generate_proto.py."""\n',
            encoding="utf-8",
            newline="\n",
        )
    print(f"generated protobuf code in {target.out_dir.relative_to(REPO_ROOT)}")
    return 0


def main() -> int:
    for target in TARGETS:
        result = generate(target)
        if result != 0:
            return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
