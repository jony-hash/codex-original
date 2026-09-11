from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from ldm_project.v2.evidence import file_sha256
from scripts.research import execute_n3_copula_mixed_component_swap_diagnostic_v1 as consumer


EXPECTED_REPORT_SHA256 = "fc139cb6ec9754babeb9e4ac6c03f08ef339cbdd6148362bb05b44c08daeaecc"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _report_path() -> Path:
    return _repo_root() / consumer.OUTPUT_DIR / consumer.REPORT_NAME


def _load_report() -> dict[str, object]:
    return json.loads(_report_path().read_text(encoding="utf-8"))


def test_report_is_valid_immutable_and_stays_outside_p1() -> None:
    report = _load_report()
    assert file_sha256(_report_path()) == EXPECTED_REPORT_SHA256
    assert report["schema"] == consumer.REPORT_SCHEMA
    assert report["task_id"] == consumer.TASK_ID
    assert report["gate"] == consumer.GATE
    assert report["status"] == "valid_completed"
    assert report["contract"]["sha256"] == consumer.EXPECTED_CONTRACT_SHA256
    assert report["classification"] in report["classification_vocabulary"]
    assert report["outer_gate"]["result"] == "VALID_COMPLETED"
    assert report["outer_gate"]["p1_execution_authorized"] is False
    assert report["governance"]["component_swap_diagnostic_executed"] is True
    assert report["governance"]["oracle_metrics_computed"] is True
    assert report["governance"]["sealed_confirmation_consumed"] is False
    assert report["governance"]["training_authorized"] is False
    assert report["governance"]["formal_promotion"] is False
    assert report["governance"]["generation_readiness"] == "not_ready"
    assert report["execution_boundary"]["measurement_only"] is True
    assert report["execution_boundary"]["p1_execution"] is False
    assert report["execution_boundary"]["complex_ols"] is False
    assert report["execution_boundary"]["generation"] is False


def test_every_historical_pair_and_fold_has_a_frozen_oracle_label() -> None:
    report = _load_report()
    folds = report["fold_results"]
    assert len(folds) == consumer.FOLD_COUNT
    assert report["aggregate"]["pair_label_output_count"] == 20
    assert report["aggregate"]["exactly_one_aggregate_oracle_sufficiency_label_emitted"] is True
    assert report["aggregate"]["classification"] == report["classification"]
    assert report["execution_settings"]["q95_reused_from_original_report"] is True
    assert report["execution_settings"]["q95_recomputed_from_oracle_hybrids"] is False
    assert report["execution_settings"]["rng_used"] is False

    for fold, item in enumerate(folds):
        assert item["fold"] == fold
        assert item["fit_rows"] == consumer.FIT_ROWS_PER_FOLD
        assert item["heldout_rows"] == consumer.HELDOUT_ROWS_PER_FOLD
        assert item["fold_oracle_sufficiency_label"] in consumer.contract_definition.FOLD_LABEL_VALUES
        assert len(item["pairs"]) == consumer.DATA_VS_MODEL_REPLICATES
        assert len(item["original_failure_pair_labels"]) + len(item["original_pass_control_pair_labels"]) == 4
        for pair in item["pairs"]:
            assert pair["pair_label"] in consumer.contract_definition.PAIR_LABEL_VALUES
            assert pair["E_BASE_identity_verified"] is True
            assert set(pair["metrics"]) == {"E_BASE", "E_MARG", "E_DEP"}
            assert pair["original_primary_status"] in {"ORIGINAL_PRIMARY_FAIL", "ORIGINAL_PRIMARY_PASS_CONTROL"}
            assert pair["included_in_original_failure_sufficiency_aggregate"] is (pair["original_primary_status"] == "ORIGINAL_PRIMARY_FAIL")
            for hybrid in pair["hybrids"].values():
                assert hybrid["annotations"] == list(consumer.HYBRID_ANNOTATIONS)


def test_ordinal_rank_uses_stable_exact_tie_order_and_reconstructs() -> None:
    values = torch.arange(consumer.HELDOUT_ROWS_PER_FOLD * consumer.DIMENSION, dtype=torch.float64).reshape(
        consumer.HELDOUT_ROWS_PER_FOLD,
        consumer.DIMENSION,
    )
    values[1, 0] = values[0, 0]
    bundle = consumer._ordinal_rank_bundle(values, "test-tie")
    assert bundle["identity"]["exact_tie_original_row_index_order_verified"] is True
    assert bundle["identity"]["self_reconstruction_verified"] is True
    assert bundle["ranks"][0, 0].item() == 0
    assert bundle["ranks"][1, 0].item() == 1
    assert torch.equal(bundle["sorted_values"].gather(0, bundle["ranks"]), values)


def test_static_preflight_rejects_stale_contract_before_q_load() -> None:
    repo_root = _repo_root()
    output_dir = repo_root / "outputs" / "_diagnostic_preflight_test_absent"
    assert not output_dir.exists()
    git = consumer._git_state(repo_root)
    with patch.object(consumer, "EXPECTED_CONTRACT_SHA256", "0" * 64), patch.object(
        consumer.comparison,
        "_load_development_pool",
        side_effect=AssertionError("q loader must not run during static preflight"),
    ):
        with pytest.raises(consumer.BlockedExecution, match="hash changed"):
            consumer._validate_static_preflight(repo_root, git, output_dir)
