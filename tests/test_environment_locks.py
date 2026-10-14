import json
from pathlib import Path

import pytest

from scripts.verify_environment_locks import (
    channel_matches_locked_source,
    read_hash_requirements,
    verify_installed_conda_owned_distributions,
    verify_installed_conda_packages,
)


def test_channel_verifier_accepts_micromamba_short_channel_name() -> None:
    assert channel_matches_locked_source(
        "conda-forge",
        "https://conda.anaconda.org/conda-forge/noarch/annotated-types-0.8.0.conda",
    )


def test_channel_verifier_accepts_locked_channel_url() -> None:
    assert channel_matches_locked_source(
        "https://conda.anaconda.org/conda-forge/",
        "https://conda.anaconda.org/conda-forge/noarch/annotated-types-0.8.0.conda",
    )


def test_channel_verifier_accepts_locked_channel_subdirectory() -> None:
    expected_url = "https://conda.anaconda.org/pytorch/win-64/pytorch-2.5.1-py3.11.conda"
    assert channel_matches_locked_source(
        "https://conda.anaconda.org/pytorch/win-64/",
        expected_url,
    )
    assert channel_matches_locked_source(
        "https://conda.anaconda.org/pytorch/win-64",
        expected_url,
    )


def test_channel_verifier_rejects_mismatched_channel_and_host() -> None:
    expected_url = "https://conda.anaconda.org/conda-forge/noarch/annotated-types-0.8.0.conda"
    assert not channel_matches_locked_source("pytorch", expected_url)
    assert not channel_matches_locked_source("https://mirror.invalid/conda-forge", expected_url)
    assert not channel_matches_locked_source("https://conda.anaconda.org/pytorch/linux-64", expected_url)
    assert not channel_matches_locked_source("https://conda.anaconda.org/pytorch/win-64/extra", expected_url)


def _write_conda_lock_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "project"
    prefix = tmp_path / "prefix"
    metadata = prefix / "conda-meta"
    metadata.mkdir(parents=True)
    package_url = "https://conda.anaconda.org/conda-forge/noarch/sympy-1.13.1-pyh04b8f61_3.conda"
    digest = "0e6d987b28a6ea5c95b2dc855013dc1817a653f501c65d04f2809cb47b38dd67"
    root.mkdir()
    (root / "environment-win-explicit.lock.txt").write_text(
        f"@EXPLICIT\n{package_url}#sha256:{digest}\n",
        encoding="utf-8",
    )
    record: dict[str, object] = {
        "fn": "sympy-1.13.1-pyh04b8f61_3.conda",
        "url": package_url,
        "sha256": digest,
        "name": "sympy",
        "version": "1.13.1",
        "build": "pyh04b8f61_3",
        "channel": "https://conda.anaconda.org/conda-forge/noarch",
    }
    (metadata / "sympy-1.13.1-pyh04b8f61_3.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr("scripts.verify_environment_locks.sys.prefix", str(prefix))
    return root, record


def test_installed_conda_verifier_accepts_matching_url_hash_build_and_subdir(tmp_path, monkeypatch) -> None:
    root, _record = _write_conda_lock_fixture(tmp_path, monkeypatch)
    assert verify_installed_conda_packages(root, expected_count=1) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("url", "https://mirror.invalid/conda-forge/noarch/sympy-1.13.1-pyh04b8f61_3.conda"),
        ("sha256", "0" * 64),
        ("build", "wrong_build"),
        ("channel", "https://conda.anaconda.org/conda-forge/win-64"),
    ],
)
def test_installed_conda_verifier_rejects_record_mismatch(tmp_path, monkeypatch, field, value) -> None:
    root, record = _write_conda_lock_fixture(tmp_path, monkeypatch)
    record[field] = value
    next((Path(__import__("sys").prefix) / "conda-meta").glob("*.json")).write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_installed_conda_packages(root, expected_count=1)


def test_shiboken_distribution_must_be_owned_by_pyside_conda_record(tmp_path, monkeypatch) -> None:
    prefix = tmp_path / "prefix"
    metadata = prefix / "conda-meta"
    metadata.mkdir(parents=True)
    owner = {
        "name": "pyside6",
        "version": "6.11.2",
        "files": ["Lib/site-packages/shiboken6-6.11.2.dist-info/METADATA"],
    }
    (metadata / "pyside6-6.11.2-build.json").write_text(json.dumps(owner), encoding="utf-8")
    monkeypatch.setattr("scripts.verify_environment_locks.sys.prefix", str(prefix))
    monkeypatch.setattr("scripts.verify_environment_locks.importlib_metadata.version", lambda _name: "6.11.2 ")
    assert verify_installed_conda_owned_distributions(
        {"shiboken6": {"version": "6.11.2", "owner": "pyside6"}}
    ) == 1

    owner["files"] = ["Lib/site-packages/shiboken6/__init__.py"]
    (metadata / "pyside6-6.11.2-build.json").write_text(json.dumps(owner), encoding="utf-8")
    with pytest.raises(ValueError, match="does not own"):
        verify_installed_conda_owned_distributions(
            {"shiboken6": {"version": "6.11.2", "owner": "pyside6"}}
        )


def test_windows_lock_closes_torch_sympy_and_shiboken_dependencies() -> None:
    root = Path(__file__).resolve().parents[1]
    conda_lines = (root / "environment-win-explicit.lock.txt").read_text(encoding="utf-8").splitlines()
    assert any("mpmath-1.3.0-pyhd8ed1ab_1.conda#7d7aa3fcd6f42b76bd711182f3776a02bef09a68c5f117d66b712a6d81368692" in line for line in conda_lines)
    assert any("sympy-1.13.1-pyh04b8f61_3.conda#0e6d987b28a6ea5c95b2dc855013dc1817a653f501c65d04f2809cb47b38dd67" in line for line in conda_lines)
    pip_packages = read_hash_requirements(root / "requirements-win-pip.lock.txt")
    assert pip_packages == {"fsspec": "2026.7.0", "pymupdf": "1.28.2"}
    summary = json.loads((root / "environment-win-lock.json").read_text(encoding="utf-8"))
    assert summary["conda_owned_distributions"]["shiboken6"] == {"owner": "pyside6", "version": "6.11.2"}
