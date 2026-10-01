from __future__ import annotations

from scripts.research.lightgen.execute_lightgen_2d_frozen_latent_generator_semantic_recovery_v1 import MIXED_TRANSFER_CLASSIFICATION, NOT_SUPPORTED_CLASSIFICATION, SUPPORTED_CLASSIFICATION, TRADEOFF_CLASSIFICATION, _classification


def _semantic(*, delta: float, lower: float) -> dict[str, object]:
    return {
        "delta_macro_recall": delta,
        "paired_bootstrap_ci": {"lower": lower, "upper": lower + 0.01},
    }


def _reconstruction(delta: float) -> dict[str, object]:
    return {"delta_step_300_minus_step_0": delta}


def _transfer(*, feature_after: float, valid: bool = True) -> dict[str, object]:
    return {
        "after": {"R_feature": feature_after},
        "baseline_P2_Q2": {"R_feature": 1.0},
        "successor_raw_validity": {
            "finite": valid,
            "nonconstant_fraction": 1.0 if valid else 0.5,
            "zero_image_count": 0 if valid else 1,
        },
    }


def test_successor_classification_is_fail_closed() -> None:
    classification, checks = _classification(
        _semantic(delta=0.02, lower=0.01),
        _reconstruction(delta=0.001),
        _transfer(feature_after=0.8),
    )
    assert classification == SUPPORTED_CLASSIFICATION
    assert checks == {
        "dev_semantic_supported": True,
        "dev_reconstruction_pass": True,
        "sampled_R_feature_decreased": True,
        "raw_valid": True,
    }

    classification, _ = _classification(
        _semantic(delta=0.02, lower=0.01),
        _reconstruction(delta=0.001),
        _transfer(feature_after=1.1),
    )
    assert classification == MIXED_TRANSFER_CLASSIFICATION

    classification, _ = _classification(
        _semantic(delta=0.02, lower=0.01),
        _reconstruction(delta=0.006),
        _transfer(feature_after=0.8),
    )
    assert classification == TRADEOFF_CLASSIFICATION

    classification, _ = _classification(
        _semantic(delta=0.0, lower=0.0),
        _reconstruction(delta=0.001),
        _transfer(feature_after=0.8),
    )
    assert classification == NOT_SUPPORTED_CLASSIFICATION

