from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "research_assets" / "lightgen" / "science_adv7434"


def _validate_bundle(bundle: Path) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported reference-asset manifest schema")
    if manifest["status"] != "reference_only":
        raise ValueError("LightGen asset bundle must remain reference-only")
    if not manifest["consumer"]["name"] or not manifest["consumer"]["rejection_decision"]:
        raise ValueError("reference-asset manifest requires a named consumer and rejection decision")

    listed: set[str] = set()
    for item in manifest["files"]:
        relative = item["path"]
        if Path(relative).is_absolute() or relative in listed:
            raise ValueError(f"invalid or duplicate asset path: {relative}")
        listed.add(relative)
        path = bundle / relative
        if not path.is_file():
            raise ValueError(f"missing reference asset: {relative}")
        payload = path.read_bytes()
        if len(payload) != item["bytes"]:
            raise ValueError(f"size mismatch for reference asset: {relative}")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"SHA256 mismatch for reference asset: {relative}")

    controls = {"manifest.json", "README.md"}
    actual = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path.name not in controls
    }
    if actual != listed:
        raise ValueError(f"unlisted or missing bundle payloads: {sorted(actual ^ listed)}")


def test_lightgen_reference_asset_manifest_is_intact() -> None:
    _validate_bundle(BUNDLE)


def test_lightgen_reference_asset_manifest_rejects_tampering(tmp_path: Path) -> None:
    copy = tmp_path / "science_adv7434"
    shutil.copytree(BUNDLE, copy)
    tampered = copy / "source_data" / "Figure S1" / "Figure S1.csv"
    tampered.write_bytes(tampered.read_bytes() + b"\n# tampered\n")
    with pytest.raises(ValueError, match="(?:size|SHA256) mismatch"):
        _validate_bundle(copy)


def test_lightgen_reference_asset_manifest_rejects_unlisted_payload(tmp_path: Path) -> None:
    copy = tmp_path / "science_adv7434"
    shutil.copytree(BUNDLE, copy)
    (copy / "unexpected_checkpoint.pt").write_bytes(b"not a reference asset")
    with pytest.raises(ValueError, match="unlisted or missing"):
        _validate_bundle(copy)


def test_formal_runtime_does_not_auto_load_research_reference_assets() -> None:
    source_root = ROOT / "src" / "ldm_project"
    references = "research_assets/lightgen/science_adv7434"
    assert not any(
        references in path.read_text(encoding="utf-8")
        for path in source_root.rglob("*.py")
    )
