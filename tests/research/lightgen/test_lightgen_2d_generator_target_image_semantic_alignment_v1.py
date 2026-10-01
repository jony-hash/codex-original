from __future__ import annotations

import torch

from scripts.research.lightgen import execute_lightgen_2d_generator_target_image_semantic_alignment_v1 as alignment


def test_cmos_teacher_view_is_the_fixed_28x28_contract() -> None:
    raw = torch.zeros((2, 1, 200, 200), dtype=torch.float32)
    raw[:, :, 0, 0] = 2.0
    view = alignment._cmos_teacher_view(raw)
    assert view.shape == (2, 1, 28, 28)
    assert float(view.min()) >= 0.0
    assert float(view.max()) <= 1.0


def test_alignment_loss_detaches_target_features() -> None:
    cmos = torch.randn((4, 128), dtype=torch.float32, requires_grad=True)
    target = torch.randn((4, 128), dtype=torch.float32, requires_grad=True)
    loss = alignment._normalized_feature_alignment_loss(cmos, target)
    loss.backward()
    assert cmos.grad is not None
    assert target.grad is None
    assert float(loss) >= 0.0


def test_conflicting_auxiliary_gradient_is_projected_against_alignment() -> None:
    align = torch.tensor([1.0, 0.0, 0.0])
    auxiliary = torch.tensor([-2.0, 3.0, 0.0])
    safe, details = alignment._project_auxiliary_gradient(auxiliary, align, "g_rec")
    assert details["projected"] is True
    assert torch.allclose(safe, torch.tensor([0.0, 3.0, 0.0]))
    assert abs(float(torch.dot(safe, align))) <= 1.0e-8


def _semantic(*, delta: float, lower: float) -> dict[str, object]:
    return {"delta_macro_recall": delta, "paired_bootstrap_ci": {"lower": lower}}


def _reconstruction(delta: float) -> dict[str, object]:
    return {"delta_step_300_minus_step_0": delta}


def _realization(*, final_fraction: float) -> dict[str, object]:
    return {
        "step_0": {"centroid_matched_nearest_fraction": 0.1},
        "final": {"centroid_matched_nearest_fraction": final_fraction},
        "training_teacher": {"step_0_L_align": 1.0, "step_300_L_align": 0.8},
    }


def _transfer() -> dict[str, object]:
    return {"successor_raw_validity": {"finite": True, "nonconstant_fraction": 1.0, "zero_image_count": 0}}


def test_classification_requires_independent_smooth_and_centroid_improvement() -> None:
    valid = {"all_finite": True, "all_nonconstant": True, "all_zero_image_counts_zero": True}
    classification, checks = alignment._classification(_semantic(delta=0.02, lower=0.01), _reconstruction(0.001), valid, _realization(final_fraction=0.3), _transfer())
    assert classification == alignment.SUPPORTED_CLASSIFICATION
    assert checks["smooth_centroid_alignment_moved"] is True

    classification, _ = alignment._classification(_semantic(delta=0.0, lower=0.0), _reconstruction(0.001), valid, _realization(final_fraction=0.1), _transfer())
    assert classification == alignment.TEACHER_ONLY_CLASSIFICATION

    classification, _ = alignment._classification(_semantic(delta=0.02, lower=0.01), _reconstruction(0.006), valid, _realization(final_fraction=0.3), _transfer())
    assert classification == alignment.TRADEOFF_CLASSIFICATION


def test_classification_rejects_missing_centroid_motion() -> None:
    valid = {"all_finite": True, "all_nonconstant": True, "all_zero_image_counts_zero": True}
    classification, checks = alignment._classification(_semantic(delta=0.02, lower=0.01), _reconstruction(0.001), valid, _realization(final_fraction=0.1), _transfer())
    assert classification == alignment.NOT_SUPPORTED_CLASSIFICATION
    assert checks["smooth_centroid_alignment_moved"] is False
