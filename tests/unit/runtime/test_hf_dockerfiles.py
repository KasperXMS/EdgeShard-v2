"""Static gates for production Hugging Face shard image definitions."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[3]
INFERENCE_IMPORT_SMOKE = (
    "import torch, transformers, accelerate, safetensors, huggingface_hub; "
    "from edgeshard.control.worker.compute_executor import serve_container_compute"
)


def _normalized_dockerfile(name: str) -> str:
    text = (REPOSITORY_ROOT / "containers" / "hf" / name).read_text(
        encoding="utf-8"
    )
    return re.sub(r"\\[ \t]*\r?\n[ \t]*", " ", text)


@pytest.mark.parametrize(
    ("dockerfile", "excluded_packages"),
    [
        ("Dockerfile.cuda", ("torch",)),
        ("Dockerfile.jetson", ("torch", "numpy")),
    ],
)
def test_gpu_shard_images_install_inference_extra_and_run_import_smoke(
    dockerfile: str, excluded_packages: tuple[str, ...]
) -> None:
    contents = _normalized_dockerfile(dockerfile)
    export = contents.split("RUN uv export", maxsplit=1)[1].split(
        "RUN ", maxsplit=1
    )[0]

    assert "--frozen" in export
    assert "--no-dev" in export
    assert "--extra inference" in export
    assert "--no-emit-project" in export
    for package in excluded_packages:
        assert f"--no-emit-package {package}" in export
    assert "RUN python -c" in contents
    assert INFERENCE_IMPORT_SMOKE in contents
