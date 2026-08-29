"""Regenerate protobuf code for proto/shard_runtime.proto.

Usage (repository root, managed environment):

    uv run python scripts/generate_proto.py

The generated files are committed so runtime environments and CI never need
grpcio-tools (spec 4.10 reproducibility); grpcio-tools is a dev dependency
only. Generated code lands in ``src/edgeshard/protocol/pb/`` and the grpc
stub's flat import is rewritten to a package import.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from grpc_tools import protoc

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTO_DIR = REPO_ROOT / "proto"
OUT_DIR = REPO_ROOT / "src" / "edgeshard" / "protocol" / "pb"
GRPC_MODULE = "shard_runtime_pb2_grpc.py"

_FLAT_IMPORT = re.compile(r"^import shard_runtime_pb2 as (\w+)$", re.MULTILINE)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = protoc.main(
        [
            "protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={OUT_DIR}",
            f"--grpc_python_out={OUT_DIR}",
            f"--pyi_out={OUT_DIR}",
            "shard_runtime.proto",
        ]
    )
    if result != 0:
        return result

    grpc_path = OUT_DIR / GRPC_MODULE
    source = grpc_path.read_text(encoding="utf-8")
    rewritten, count = _FLAT_IMPORT.subn(
        r"from edgeshard.protocol.pb import shard_runtime_pb2 as \1", source
    )
    if count != 1:
        print(f"unexpected generated import layout in {grpc_path}", file=sys.stderr)
        return 1
    grpc_path.write_text(rewritten, encoding="utf-8", newline="\n")

    init_path = OUT_DIR / "__init__.py"
    if not init_path.exists():
        init_path.write_text(
            '"""Generated protobuf code. Regenerate via scripts/generate_proto.py."""\n',
            encoding="utf-8",
            newline="\n",
        )
    print(f"generated protobuf code in {OUT_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
