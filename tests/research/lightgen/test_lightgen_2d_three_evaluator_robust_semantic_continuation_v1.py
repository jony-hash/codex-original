from __future__ import annotations

import math

import torch

from scripts.research.lightgen.execute_lightgen_2d_three_evaluator_robust_semantic_continuation_v1 import BETA, DELTA_SIC, GAMMA, STAGE1_END, START_STEP, TOTAL_STEPS, ContinuationSampler, _class_metrics, _stage1_condition


def test_sic_contract_constants_are_frozen() -> None:
    assert math.isclose(DELTA_SIC, 1.365988374680088e-05, rel_tol=0.0, abs_tol=1.0e-18)
    assert math.isclose(BETA, 2.4509953404, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(GAMMA, 0.5330424285, rel_tol=0.0, abs_tol=1.0e-12)
    assert (START_STEP, STAGE1_END, TOTAL_STEPS) == (3000, 3300, 6000)


def test_continuation_sampler_preserves_cursor_and_wraps_only_after_authority_prefix() -> None:
    state = {
        "seed": 1234,
        "steps": START_STEP,
        "cursor": START_STEP,
        "permutations": [torch.arange(START_STEP, dtype=torch.long) for _ in range(10)],
    }
    sampler = ContinuationSampler(state, cursor=START_STEP)
    indices, labels = sampler.next_batch()
    assert indices.tolist() == [0] * 10
    assert labels.tolist() == list(range(10))
    assert sampler.cursor == START_STEP + 1


def test_class_metrics_uses_true_rows_and_predicted_columns() -> None:
    labels = torch.arange(10, dtype=torch.long).repeat_interleave(2)
    predictions = labels.clone()
    predictions[1] = 2
    result = _class_metrics(labels, predictions, label_source="test")
    assert result["rows"] == 20
    assert result["label_source"] == "test"
    assert result["confusion_matrix_rows_true_columns_predicted"][0][0] == 1
    assert result["confusion_matrix_rows_true_columns_predicted"][0][1] == 0
    assert result["confusion_matrix_rows_true_columns_predicted"][0][2] == 1


def test_stage1_condition_requires_independent_positive_delta_and_ci() -> None:
    base = {"L_S": 1.0, "L_I": 1.0, "L_C": 1.0}
    improved = {"L_S": 0.9, "L_I": 0.9, "L_C": 0.9}
    reconstruction = {"delta_step3300_minus_step3000": 0.0}
    optical = {"passed": True}
    firewall = {"protected": 0, "test": 0, "sealed": 0}
    failed, classification, checks = _stage1_condition(
        base,
        improved,
        {"delta_macro_recall": 0.01, "ci95": {"lower": -0.01, "upper": 0.03}},
        reconstruction,
        optical,
        sic_degenerate=False,
        firewall=firewall,
    )
    assert failed is False
    assert checks["L_C_3300_lt_L_C_3000"] is True
    assert checks["E_D_delta_positive"] is True
    assert checks["E_D_CI_lower_positive"] is False
    assert classification == "THREE_EVALUATOR_DEVELOPMENT_GAIN_WITHOUT_INDEPENDENT_TRANSFER"
