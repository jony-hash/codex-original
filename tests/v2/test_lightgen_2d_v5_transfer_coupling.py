"""Focused engineering acceptance for the V5 transfer-coupling entry."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/research/diagnose_lightgen_2d_v5_latent_alignment_to_optical_transfer_coupling_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("v5_transfer_coupling_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_real_authority_dry_run_consumes_nested_preflight_schema_without_data_forward():
    module = _module()
    result = module.run_no_data_dry_run(ROOT)

    assert result["result"] == "NO_DATA_DRY_RUN_PASSED"
    accepted = result["authority_resolution"]["accepted_preflight"]
    assert accepted["input"]["path"].endswith("lightgen_2d_complex_ols_semantic_alignment_preflight_input.v1.json")
    assert len(accepted["input"]["sha256"]) == 64
    assert result["data_boundary"] == {
        "real_train_scientific_reads": 0,
        "real_dev_scientific_reads": 0,
        "scientific_forward": 0,
        "smooth_scientific_evaluation": 0,
        "stop_boundary": "before real DEV row read or forward",
    }


def test_private_loader_replacement_is_audited_and_not_called():
    module = _module()
    source = SCRIPT.read_text(encoding="utf-8")
    assert "preflight._load_evaluators(" not in source
    assert "preflight._load_smooth(" not in source
    audit = module.run_no_data_dry_run(ROOT)["dependency_audit"]
    assert audit["status"] == "passed"
    assert any(item["symbol"] == "accepted_v5_preflight._load_evaluators" for item in audit["replaced"])
    assert any(item["symbol"] == "_load_contract" and item["status"] == "retained_and_signature_checked" for item in audit["retained"])


def test_same_production_wrapper_accepts_deterministic_synthetic_boundary():
    module = _module()
    result = module._run_synthetic(ROOT)

    assert result["result"] == "SYNTHETIC_ONLY"
    assert result["status"] == "valid_completed"
    assert result["production_wrapper"] is True
    assert result["no_scientific_data"] is True
    assert result["scientific_forward_count"] == 12
    assert result["smooth_evaluation_count"] == 8
    assert result["core_serialization"] == "passed"
    assert result["report_lineage_reload"] is True
    assert result["classification"] == "SYNTHETIC_ONLY"
