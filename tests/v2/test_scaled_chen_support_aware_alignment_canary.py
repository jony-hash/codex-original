"""Regression tests for the bounded T2 support-aware alignment canary."""

from __future__ import annotations

import copy

import pytest

from scripts.research import run_scaled_chen_support_aware_alignment_canary as t2


def test_t2_architecture_delta_is_power_only_and_routes_to_s1() -> None:
    assert t2.TASK_ID == "H-LOMA-GEN-T2-SAAC"
    assert t2.POWER_OBSERVER_INTERPRETATION == "normalized_power_coordinate_observer"
    assert t2.NEXT_ACTION == "PLAN_S1_OLS_SEMANTIC_REPRESENTATION_AUDIT"


def test_t2_result_requires_all_registered_gates_for_supported() -> None:
    assert t2.classify_t2_result(
        valid=True,
        primary_pass=True,
        paired_pass=True,
        mse_safe=True,
        mean_safe=True,
        rank_safe=True,
    ) == t2.T2_SUPPORT_AWARE_ALIGNMENT_SUPPORTED
    assert t2.classify_t2_result(
        valid=True,
        primary_pass=False,
        paired_pass=True,
        mse_safe=True,
        mean_safe=True,
        rank_safe=True,
    ) == t2.T2_SUPPORT_AWARE_ALIGNMENT_NOT_SUPPORTED


def test_t2_result_distinguishes_tradeoff_harmful_and_invalid() -> None:
    assert t2.classify_t2_result(
        valid=True,
        primary_pass=True,
        paired_pass=False,
        mse_safe=True,
        mean_safe=True,
        rank_safe=True,
    ) == t2.T2_SUPPORT_AWARE_ALIGNMENT_TRADEOFF
    assert t2.classify_t2_result(
        valid=True,
        primary_pass=False,
        paired_pass=False,
        mse_safe=False,
        mean_safe=True,
        rank_safe=True,
    ) == t2.T2_SUPPORT_AWARE_ALIGNMENT_HARMFUL
    assert t2.classify_t2_result(
        valid=False,
        primary_pass=False,
        paired_pass=False,
        mse_safe=False,
        mean_safe=False,
        rank_safe=False,
    ) == t2.T2_TRAINING_INVALID


def test_t2_config_rejects_bogt_alpha_and_nonzero_operator_terms() -> None:
    config = copy.deepcopy(t2.load_config(t2.CONFIG_PATH)[1])
    config["objective"]["alpha_bogt"] = 1.0e-6
    with pytest.raises(ValueError, match="alpha_BOGT"):
        t2.validate_registered_config(config)

    config = copy.deepcopy(t2.load_config(t2.CONFIG_PATH)[1])
    config["objective"]["gamma"] = 1.0e-6
    with pytest.raises(ValueError, match="gamma"):
        t2.validate_registered_config(config)


def test_t2_interpretation_is_not_semantic_generation_authority() -> None:
    interpretation = t2.scam_interpretation()
    assert interpretation["power_observer"] == "normalized_power_coordinate_observer"
    assert interpretation["complex_semantic_ols"] == "unestablished"
    assert interpretation["label_to_ols"] == "unresolved"
    assert interpretation["code8_to_ols"] in {"partial", "unresolved"}
    assert interpretation["global_power_track"] == "closed_after_this_canary"
    assert interpretation["next_action"] == "PLAN_S1_OLS_SEMANTIC_REPRESENTATION_AUDIT"
