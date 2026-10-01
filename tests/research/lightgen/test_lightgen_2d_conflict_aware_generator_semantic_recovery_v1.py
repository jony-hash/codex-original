from __future__ import annotations

import torch

from scripts.research.lightgen.execute_lightgen_2d_conflict_aware_generator_semantic_recovery_v1 import CONSTRAINT_CLASSIFICATION, MIXED_TRANSFER_CLASSIFICATION, NOT_SUPPORTED_CLASSIFICATION, SUPPORTED_CLASSIFICATION, _classification, _project_auxiliary_gradient


def test_conflicting_auxiliary_gradient_is_projected_orthogonally() -> None:
    semantic = torch.tensor([1.0, 0.0, 0.0])
    auxiliary = torch.tensor([-2.0, 3.0, 0.0])
    safe, details = _project_auxiliary_gradient(auxiliary, semantic, "g_rec")

    assert details["projected"] is True
    assert torch.allclose(safe, torch.tensor([0.0, 3.0, 0.0]))
    assert abs(float(torch.dot(safe, semantic))) <= 1.0e-8
    assert details["projection_fraction_norm"] == 2.0 / (13.0**0.5)


def test_nonconflicting_auxiliary_gradient_is_preserved() -> None:
    semantic = torch.tensor([1.0, 0.0])
    auxiliary = torch.tensor([2.0, 3.0])
    safe, details = _project_auxiliary_gradient(auxiliary, semantic, "g_op")

    assert details["projected"] is False
    assert torch.equal(safe, auxiliary)
    assert details["projection_fraction_norm"] == 0.0


def _semantic(*, delta: float, lower: float) -> dict[str, object]:
    return {"delta_macro_recall": delta, "paired_bootstrap_ci": {"lower": lower}}


def _reconstruction(delta: float) -> dict[str, object]:
    return {"delta_step_300_minus_step_0": delta}


def _transfer(*, feature_after: float, valid: bool = True) -> dict[str, object]:
    return {
        "after": {"R_feature": feature_after},
        "baseline_P2_Q2": {"R_feature": 1.0},
        "successor_raw_validity": {"finite": valid, "nonconstant_fraction": 1.0 if valid else 0.5, "zero_image_count": 0 if valid else 1},
    }


def test_classification_matches_conflict_aware_gate_rules() -> None:
    classification, checks = _classification(_semantic(delta=0.02, lower=0.01), _reconstruction(0.001), _transfer(feature_after=0.8))
    assert classification == SUPPORTED_CLASSIFICATION
    assert checks["dev_semantic_supported"] is True

    classification, _ = _classification(_semantic(delta=0.02, lower=0.01), _reconstruction(0.001), _transfer(feature_after=1.1))
    assert classification == MIXED_TRANSFER_CLASSIFICATION

    classification, _ = _classification(_semantic(delta=0.02, lower=0.01), _reconstruction(0.006), _transfer(feature_after=0.8))
    assert classification == CONSTRAINT_CLASSIFICATION

    classification, _ = _classification(_semantic(delta=0.0, lower=0.0), _reconstruction(0.001), _transfer(feature_after=0.8))
    assert classification == NOT_SUPPORTED_CLASSIFICATION
