from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "experiments" / "archive" / "test_governance_v1"
MANIFEST = ARCHIVE / "archive_manifest.yaml"


def test_historical_test_capsule_is_reference_only() -> None:
    payload = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert payload["status"] == "diagnostic-only"
    assert payload["runnable"] is False
    assert payload["source_commit"] == "294aa18"
    for relative in payload["preserved_files"]:
        assert (ROOT / relative).is_file(), relative


def test_archive_is_outside_default_pytest_testpath() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'testpaths = ["tests"]' in pyproject
    assert ARCHIVE.parent.name == "archive"
