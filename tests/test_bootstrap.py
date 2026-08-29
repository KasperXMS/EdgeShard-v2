"""Bootstrap smoke test: the package imports and the toolchain is wired up."""

import edgeshard


def test_package_importable() -> None:
    assert edgeshard.__version__ == "0.0.1"
