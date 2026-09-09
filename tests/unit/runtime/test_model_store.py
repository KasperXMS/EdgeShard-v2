"""Worker-local ModelStore resolution (spec 24 seam).

Each worker owns its own model store root; deployment plans never carry
host paths. A local name therefore resolves to per-worker host paths but
always maps to the same uniform container path form, which is what keeps
runtimes and the inference layer unaware of host-layout differences.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from edgeshard.runtime.model_store import (
    DEFAULT_MODEL_ROOT,
    MODEL_METADATA_FILENAME,
    MODEL_MOUNT,
    ModelMetadata,
    ModelMetadataError,
    ModelStore,
    ModelStoreError,
    container_model_path,
    validate_local_name,
)


def test_default_model_root() -> None:
    assert ModelStore().model_root == DEFAULT_MODEL_ROOT
    assert DEFAULT_MODEL_ROOT.as_posix() == "/data/edgeshard-models"


def test_model_root_accepts_str_and_coerces_to_path(tmp_path: Path) -> None:
    store = ModelStore(model_root=str(tmp_path))
    assert store.model_root == tmp_path


def test_host_path_resolves_local_name_under_the_store_root(tmp_path: Path) -> None:
    store = ModelStore(model_root=tmp_path / "cache")
    assert store.host_path("tiny-llama") == tmp_path / "cache" / "tiny-llama"


def test_container_path_maps_to_the_uniform_mount(tmp_path: Path) -> None:
    store = ModelStore(model_root=tmp_path / "cache")
    assert store.container_path("tiny-llama") == Path(MODEL_MOUNT) / "tiny-llama"


def test_two_model_roots_resolve_different_host_paths_to_identical_container_paths(
    tmp_path: Path,
) -> None:
    """Requirement: per-worker roots differ on the host, agree in-container."""
    store_a = ModelStore(model_root=tmp_path / "worker-a")
    store_b = ModelStore(model_root=tmp_path / "worker-b")

    assert store_a.host_path("tiny-llama") == tmp_path / "worker-a" / "tiny-llama"
    assert store_b.host_path("tiny-llama") == tmp_path / "worker-b" / "tiny-llama"
    assert store_a.host_path("tiny-llama") != store_b.host_path("tiny-llama")

    assert store_a.container_path("tiny-llama") == store_b.container_path("tiny-llama")
    assert store_a.container_path("tiny-llama") == Path("/models/tiny-llama")


def test_container_model_path_is_independent_of_any_store(tmp_path: Path) -> None:
    assert container_model_path("tiny-llama") == Path(MODEL_MOUNT) / "tiny-llama"
    assert container_model_path("tiny-llama") == ModelStore(
        model_root=tmp_path / "anywhere"
    ).container_path("tiny-llama")


@pytest.mark.parametrize(
    "bad_name",
    ["", "a/b", "a\\b", ".", "..", "tiny llama", " tiny", "tiny\tllama"],
)
def test_local_name_validation_rejects_unsafe_names(bad_name: str) -> None:
    with pytest.raises(ModelStoreError):
        validate_local_name(bad_name)
    with pytest.raises(ModelStoreError):
        ModelStore().host_path(bad_name)
    with pytest.raises(ModelStoreError):
        container_model_path(bad_name)


def test_local_name_validation_accepts_snapshot_style_names() -> None:
    assert validate_local_name("tiny-llama") == "tiny-llama"
    assert validate_local_name("Qwen2.5-7B") == "Qwen2.5-7B"


def test_model_store_imports_standalone() -> None:
    # Regression guard: ``drivers/__init__`` eagerly imports the driver
    # modules, and the drivers construct stores — so model_store must never
    # import from edgeshard.runtime.drivers, or this import closes a cycle.
    module = importlib.import_module("edgeshard.runtime.model_store")
    assert module.MODEL_MOUNT == "/models"


def test_read_metadata_parses_model_id_and_optional_revision(tmp_path: Path) -> None:
    snapshot = tmp_path / "tiny-llama"
    snapshot.mkdir()
    (snapshot / MODEL_METADATA_FILENAME).write_text(
        json.dumps(
            {
                "model_id": "meta-llama/Llama-3.2-1B",
                "revision": "0123456789abcdef",
                "future_field": 1,
            }
        ),
        encoding="utf-8",
    )

    assert ModelStore(tmp_path).read_metadata("tiny-llama") == ModelMetadata(
        model_id="meta-llama/Llama-3.2-1B",
        revision="0123456789abcdef",
    )


def test_read_metadata_allows_omitted_revision(tmp_path: Path) -> None:
    snapshot = tmp_path / "tiny-llama"
    snapshot.mkdir()
    (snapshot / MODEL_METADATA_FILENAME).write_text(
        json.dumps({"model_id": "tiny/llama"}), encoding="utf-8"
    )

    assert ModelStore(tmp_path).read_metadata("tiny-llama") == ModelMetadata(
        model_id="tiny/llama"
    )


def test_read_metadata_returns_none_only_when_sidecar_is_absent(tmp_path: Path) -> None:
    (tmp_path / "tiny-llama").mkdir()
    assert ModelStore(tmp_path).read_metadata("tiny-llama") is None


@pytest.mark.parametrize(
    "payload",
    [
        "{ not json",
        "[]",
        "{}",
        '{"model_id": ""}',
        '{"model_id": 7}',
        '{"model_id": "tiny/llama", "revision": ""}',
        '{"model_id": "tiny/llama", "revision": 7}',
    ],
)
def test_read_metadata_rejects_invalid_sidecar(tmp_path: Path, payload: str) -> None:
    snapshot = tmp_path / "tiny-llama"
    snapshot.mkdir()
    (snapshot / MODEL_METADATA_FILENAME).write_text(payload, encoding="utf-8")

    with pytest.raises(ModelMetadataError):
        ModelStore(tmp_path).read_metadata("tiny-llama")
