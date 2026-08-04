"""Read-only integrity checks for the v2.2 conditional-negative capsule."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "experiments" / "archive" / "v2_2_capacity_map_closeout"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_hash(payload: dict) -> str:
    value = {key: item for key, item in payload.items() if key != "content_hash"}
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_v22_closeout_archive_is_self_authenticating_and_non_runnable() -> None:
    index = json.loads((ARCHIVE / "archive_index.json").read_text(encoding="utf-8"))
    assert index["status"] == "closed_conditional_negative"
    assert index["confirmation_pass_count"] == "0/3"
    assert index["causal_scope"] == "E2:O0_declared_teacher_free_objectives"
    assert index["downstream_status"] == "stopped_by_o0_gate"
    assert index["runnable"] is False
    assert index["formal_promotion_authorized"] is False
    assert all((ARCHIVE / item["archive_path"]).is_file() for item in index["files"])
    assert all((ARCHIVE / item["archive_path"]).stat().st_size == item["bytes"] for item in index["files"])
    assert all(_sha256(ARCHIVE / item["archive_path"]) == item["sha256"] for item in index["files"])


def test_v22_closeout_inputs_and_evidence_hashes_are_complete() -> None:
    index = json.loads((ARCHIVE / "archive_index.json").read_text(encoding="utf-8"))
    by_source = {item["source_path"]: item for item in index["files"]}
    manifest = json.loads((ARCHIVE / "capacity_map_closeout_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((ARCHIVE / "capacity_map_closeout_summary.json").read_text(encoding="utf-8"))

    assert manifest["status"] == summary["status"] == "closed_conditional_negative"
    assert summary["confirmation_seed_count"] == 3
    assert summary["confirmation_pass_count"] == 0
    assert summary["formal_promotion_authorized"] is False
    assert summary["optimizer_created"] is False
    assert summary["cuda_used"] is False
    assert summary["evidence_governance"]["manifest_sha256"] == _content_hash(manifest)
    assert summary["prohibited_claims"] == [
        "optical capacity failure", "phase-only capacity failure", "spatial-DOF failure",
        "universal semantic-diversity capacity conflict",
    ]
    assert len(manifest["inputs"]) == 11
    assert Counter(item["action"] for item in manifest["inputs"]) == {
        "train-smoke": 6, "confirm": 3, "diagnose": 1, "qualify-shape": 1,
    }
    assert Counter(item["seed"] for item in manifest["inputs"] if item["action"] == "confirm") == {1234: 1, 4321: 1, 2234: 1}

    for item in manifest["inputs"]:
        archived = ARCHIVE / by_source[item["path"]]["archive_path"]
        evidence = json.loads(archived.read_text(encoding="utf-8"))
        assert evidence["content_hash"] == item["content_hash"] == _content_hash(evidence)
        assert evidence["action"] == item["action"]
        assert evidence["seed"] == item["seed"]

    invalid = manifest["invalid_revisions"]["primary"]
    ambiguous = manifest["ambiguous_revisions"]["primary"]
    assert len(invalid) == len(ambiguous) == 1
    invalid_evidence = json.loads((ARCHIVE / by_source[invalid[0]["path"]]["archive_path"]).read_text(encoding="utf-8"))
    assert invalid_evidence["state"] == "invalid_harness"
    assert invalid_evidence["content_hash"] == invalid[0]["content_hash"] == _content_hash(invalid_evidence)
    assert set(ambiguous[0]["content_hashes"]) == {invalid[0]["content_hash"], summary["diagnostic_inputs"]["diagnose"]}


def test_v22_closeout_binds_archived_configs_and_excludes_historical_tests() -> None:
    index = json.loads((ARCHIVE / "archive_index.json").read_text(encoding="utf-8"))
    summary = json.loads((ARCHIVE / "capacity_map_closeout_summary.json").read_text(encoding="utf-8"))
    by_role = {item["role"]: item for item in index["files"]}
    assert by_role["primary_config"]["sha256"] == summary["input_hashes"]["comparison_config"]
    assert by_role["dual_config"]["sha256"] == summary["input_hashes"]["dual_config"]
    assert summary["input_hashes"]["primary_evaluator"] == "46741ec76e45653c9dcb76e7b4f1645c9c057a47cdd9e4c1778d9debb553ec57"
    assert summary["input_hashes"]["smooth_evaluator"] == "45879962cf8a6dc51e141a2f2a9892356ede97dd45063f38d830299aa877e26c"
    archived_names = (
        "test_v56_coherent_relay.py", "test_v57_first_collapse.py", "test_v58_2_relay_reference.py",
        "test_v58_3_relay_acceptance.py", "test_v58_aperture_throughput.py", "test_v59_upstream_separation_isolation.py",
        "test_v61_offline_decodability.py", "test_v61_offline_decodability_upper_bound.py", "test_v61_robust_collapse.py",
        "test_v62_native_conditional_control_audit.py", "test_v63_code_conditioning_provenance_audit.py",
        "test_v64_semantic_band_carriage_audit.py", "test_v65_common_mode_readout_locality_audit.py",
        "test_v66_semantic_gradient_budget_audit.py",
    )
    assert not any((ROOT / "tests" / name).exists() for name in archived_names)
    assert (ROOT / "experiments" / "archive" / "pre_v56" / "tests" / "test_v56_coherent_relay.py").is_file()
    assert (ROOT / "experiments" / "archive" / "v57_v67" / "tests" / "test_v66_semantic_gradient_budget_audit.py").is_file()
