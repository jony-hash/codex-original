"""Contract tests for the read-only Code8 coverage review."""

from __future__ import annotations

import pytest
import torch

from scripts.research.review_scaled_chen_code8_coverage import (
    CODE_DIM,
    _binned_trend,
    _panel_report,
    _spearman,
    _validate_config,
)


def test_spearman_is_deterministic_and_monotone() -> None:
    values = torch.tensor([3.0, 1.0, 2.0, 4.0])
    assert _spearman(values, values) == pytest.approx(1.0)
    assert _spearman(values, -values) == pytest.approx(-1.0)


def test_panel_report_is_descriptive_and_not_an_intrinsic_dimension_gate() -> None:
    base_codes = torch.tensor(
        [
            [-1.0, -0.5, 0.0, 0.5, 1.0, -0.25, 0.25, 0.75],
            [1.0, 0.5, -1.0, -0.5, -1.0, 0.25, -0.25, -0.75],
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0],
        ],
        dtype=torch.float64,
    )
    panel = {
        "distribution": "uniform[-1,1]^8",
        "seed": 1234,
        "base_code_count": 3,
        "code_dim": CODE_DIM,
        "same_base_codes_across_labels": True,
    }
    report = _panel_report(base_codes, panel)
    assert report["classification"] == "CODE_PANEL_DOMAIN_COVERAGE_ADEQUATE"
    assert report["authority"]["strict_covering_proof"] is False
    assert "intrinsic" not in report


def test_binned_trend_covers_all_observations() -> None:
    x = torch.arange(20, dtype=torch.float64)
    y = x.square()
    bins = _binned_trend(x, y)
    assert sum(item["count"] for item in bins) == 20


def test_config_rejects_scientific_forward_or_new_gate() -> None:
    payload = {
        "schema": "lightgen_scaled_chen_code8_coverage_review_v1",
        "task_id": "H-LOMA-GEN-S3-C8",
        "hypothesis": "CODE8_WITHIN_CLASS_MANIFOLD_COVERAGE",
        "review": {
            "device": "cpu",
            "deterministic": True,
            "controller_forward": False,
            "optical_forward": False,
            "jacobian_recompute": False,
            "decoder": False,
            "training": False,
            "checkpoint_write": False,
            "new_panel": False,
            "new_codes": False,
            "new_transform": False,
            "new_hard_gate": True,
        },
        "governance": {
            "mainline_state": "closed_at_current_evidence_boundary",
            "reconstruction_manifold_supported": False,
            "generative_manifold_supported": False,
            "generation_readiness": "not_ready",
            "training_authorized": False,
            "formal_promotion": False,
        },
        "output": {
            "directory": "outputs/20260906_v2/runs/lightgen_scaled_chen_code8_coverage_review_v1_attempt_01",
            "write_once": True,
            "json_only": True,
        },
    }
    try:
        _validate_config(payload)
    except ValueError as error:
        assert "execution boundary" in str(error)
    else:
        raise AssertionError("new hard gate was accepted")
