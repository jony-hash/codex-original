"""Focused contract tests for the real-OLS replay identity attribution."""

from __future__ import annotations

import torch

from scripts.research import execute_lightgen_2d_real_ols_replay_conditioning_identity_gap_v1 as execution


def test_class_stratified_bootstrap_is_fixed_and_paired() -> None:
    labels = torch.arange(10, dtype=torch.long).repeat_interleave(4)
    left = labels.clone()
    right = labels.roll(1)
    first = execution._class_stratified_bootstrap_difference(left, right, labels, count=31, seed=19)
    second = execution._class_stratified_bootstrap_difference(left, right, labels, count=31, seed=19)
    assert first == second
    assert first["bootstrap_count"] == 31
    assert first["bootstrap_seed"] == 19


def test_classification_relation_rejects_material_underperformance() -> None:
    bootstrap = {
        "E_S": {"bootstrap_ci_difference_left_minus_right": {"lower": -0.1, "upper": -0.01}},
        "E_I": {"bootstrap_ci_difference_left_minus_right": {"lower": -0.1, "upper": 0.1}},
        "E_C": {"bootstrap_ci_difference_left_minus_right": {"lower": -0.1, "upper": 0.1}},
    }
    relation = execution._classification_relation(bootstrap)
    assert relation["all_evaluators_95ci_contains_zero"] is False
    assert relation["any_evaluator_95ci_strictly_below_zero"] is True


def test_difference_metrics_has_frozen_equivalence_convention() -> None:
    left = torch.tensor([[1.0 + 0.0j, 2.0 + 0.0j]], dtype=torch.complex64)
    right = left.clone()
    report = execution._difference_metrics(left, right, name="z_ols")
    assert report["exact_tensor_equal"] is True
    assert report["allclose"] is True
    assert report["equivalence_convention"] == {"rtol": 1.0e-5, "atol": 1.0e-6}
