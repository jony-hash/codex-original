"""Focused tests for the source-position-balanced empirical ceiling result."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research import execute_n3_source_position_balanced_empirical_ceiling_v1 as execution


def test_published_report_is_a_single_complete_five_fold_gate() -> None:
    path = Path(
        "outputs/20260909_v2/runs/"
        "n3_source_position_balanced_empirical_ceiling_v1_attempt_01/"
        "n3_source_position_balanced_empirical_ceiling_report.v1.json"
    )
    if not path.is_file():
        pytest.skip("report is published by the bounded implementation task")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["schema"] == execution.REPORT_SCHEMA
    assert report["result"] == "VALID_COMPLETED"
    assert report["classification"] == execution.CLASSIFICATION_NOT_SUPPORTED
    assert report["aggregate"]["compatible_fold_count"] == 0
    assert report["aggregate"]["fold_count"] == execution.FOLD_COUNT
    assert report["aggregate"]["all_five_folds_required"] is True
    assert len(report["empirical_ceiling"]["fold_results"]) == execution.FOLD_COUNT
    for fold, result in enumerate(report["empirical_ceiling"]["fold_results"]):
        assert result["fold"] == fold
        assert result["operationally_valid"] is True
        assert result["class_counts"]["heldout"] == [20] * execution.CLASS_COUNT
        assert result["class_counts"]["fit"] == [80] * execution.CLASS_COUNT
        assert set(result["raw_q_summary"]["all_scalar_guardrails"]) == set(execution.SCALAR_METRICS)
        assert result["duplicate_row_rate"]["role"].startswith("descriptive only")
    governance = report["governance"]
    assert governance["producer_family_evaluated"] is False
    assert governance["producer_selection"] is False
    assert governance["sealed_confirmation_consumed"] is False
    assert governance["training_authorized"] is False
    assert governance["formal_promotion"] is False
    assert governance["generation_readiness"] == "not_ready"
