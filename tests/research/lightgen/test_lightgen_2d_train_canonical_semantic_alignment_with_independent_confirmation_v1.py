from __future__ import annotations

import torch

from scripts.research.lightgen import execute_lightgen_2d_train_canonical_semantic_alignment_with_independent_confirmation_v1 as alignment


def test_cmos_view_is_fixed_and_bounded() -> None:
    raw = torch.zeros((2, 1, 200, 200), dtype=torch.float32)
    raw[:, :, 0, 0] = 2.0
    view = alignment._cmos_view(raw)
    assert view.shape == (2, 1, 28, 28)
    assert float(view.min()) >= 0.0
    assert float(view.max()) <= 1.0


def test_canonical_feature_loss_detaches_target() -> None:
    cmos = torch.randn((4, 128), dtype=torch.float32, requires_grad=True)
    target = torch.randn((4, 128), dtype=torch.float32, requires_grad=True)
    loss = alignment._normalized_feature_loss(cmos, target, "test loss")
    loss.backward()
    assert cmos.grad is not None
    assert target.grad is None
    assert float(loss) >= 0.0


def test_auxiliary_conflict_is_projected_against_canonical_gradient() -> None:
    canonical = torch.tensor([1.0, 0.0, 0.0])
    auxiliary = torch.tensor([-2.0, 3.0, 0.0])
    safe, details = alignment._project_auxiliary_gradient(auxiliary, canonical, "g_rec")
    assert details["projected"] is True
    assert torch.allclose(safe, torch.tensor([0.0, 3.0, 0.0]))
    assert abs(float(torch.dot(safe, canonical))) <= 1.0e-8


def _smooth_summary(*, canonical_delta: float, feature_delta: float, centroid_delta: float) -> dict[str, object]:
    return {
        "canonical_loss_delta": canonical_delta,
        "paired_feature_distance_delta": feature_delta,
        "centroid_matched_nearest_fraction_delta": centroid_delta,
    }


def _confirmation(*, delta: float, lower: float) -> dict[str, object]:
    return {"delta_macro_recall": delta, "paired_bootstrap_ci": {"lower": lower}}


def _reconstruction(delta: float) -> dict[str, object]:
    return {"delta_step_300_minus_step_0": delta}


def _optical() -> dict[str, object]:
    return {"all_finite": True, "all_nonconstant": True, "all_zero_image_counts_zero": True}


def _transfer() -> dict[str, object]:
    return {"successor_raw_validity": {"finite": True, "nonconstant_fraction": 1.0, "zero_image_count": 0}}


def test_classification_requires_independent_confirmation_and_constraints() -> None:
    classification, checks = alignment._classification(
        _smooth_summary(canonical_delta=-0.1, feature_delta=-0.2, centroid_delta=0.1),
        _confirmation(delta=0.02, lower=0.01),
        _reconstruction(0.001),
        _optical(),
        _transfer(),
    )
    assert classification == alignment.SUPPORTED_CLASSIFICATION
    assert checks["independent_confirmation_supported"] is True

    classification, _ = alignment._classification(
        _smooth_summary(canonical_delta=-0.1, feature_delta=-0.2, centroid_delta=0.1),
        _confirmation(delta=0.02, lower=-0.01),
        _reconstruction(0.001),
        _optical(),
        _transfer(),
    )
    assert classification == alignment.SMOOTH_ONLY_CLASSIFICATION

    classification, _ = alignment._classification(
        _smooth_summary(canonical_delta=-0.1, feature_delta=-0.2, centroid_delta=0.1),
        _confirmation(delta=0.02, lower=0.01),
        _reconstruction(0.006),
        _optical(),
        _transfer(),
    )
    assert classification == alignment.TRADEOFF_CLASSIFICATION


def test_classification_records_local_trajectory_failure() -> None:
    classification, checks = alignment._classification(
        _smooth_summary(canonical_delta=0.1, feature_delta=0.2, centroid_delta=0.0),
        _confirmation(delta=0.0, lower=0.0),
        _reconstruction(-0.001),
        _optical(),
        _transfer(),
    )
    assert classification == alignment.LOCAL_FAILURE_CLASSIFICATION
    assert checks["canonical_loss_improved"] is False
