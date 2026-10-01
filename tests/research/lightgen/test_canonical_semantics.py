from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from ldm_project.v2.application.research.lightgen import canonical_semantics as canonical
from scripts.research.lightgen import execute_lightgen_2d_evaluator_robust_semantic_training_v1 as robust


def test_cmos_view_keeps_the_clamp_pool_and_28_by_28_view() -> None:
    value = torch.linspace(-1.0, 2.0, 2 * 200 * 200).reshape(2, 1, 200, 200)
    actual = canonical.cmos_view(value)
    expected = F.adaptive_avg_pool2d(value.clamp(0.0, 1.0), (28, 28))
    assert torch.equal(actual, expected)


def test_normalized_feature_loss_matches_the_existing_formula_and_detaches_target() -> None:
    generated = torch.randn(5, 128, requires_grad=True)
    target = torch.randn(5, 128, requires_grad=True)
    actual = canonical.normalized_feature_loss(generated, target, "L_I", epsilon=1.0e-12)
    expected = (
        F.normalize(generated, dim=1, eps=1.0e-12)
        - F.normalize(target.detach(), dim=1, eps=1.0e-12)
    ).square().sum(dim=1).mean()
    assert torch.equal(actual, expected)
    actual.backward()
    assert generated.grad is not None
    assert target.grad is None


def test_auxiliary_projection_matches_the_existing_negative_cosine_rule() -> None:
    semantic = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64)
    auxiliary = torch.tensor([-3.0, 0.5, 4.0], dtype=torch.float64)
    expected_coefficient = float(torch.dot(auxiliary, semantic)) / (
        float(torch.dot(semantic, semantic)) + 1.0e-12
    )
    expected = auxiliary - semantic * expected_coefficient

    actual, summary = canonical.project_auxiliary_gradient(
        auxiliary, semantic, "g_rec", epsilon=1.0e-12
    )

    assert torch.allclose(actual, expected, rtol=0.0, atol=0.0)
    assert summary["projected"] is True
    assert summary["eps_proj"] == 1.0e-12
    assert abs(summary["dot_after"]) < 1.0e-12


def test_nonconflicting_auxiliary_gradient_is_unchanged() -> None:
    semantic = torch.tensor([1.0, -2.0])
    auxiliary = torch.tensor([2.0, 1.0])
    actual, summary = canonical.project_auxiliary_gradient(auxiliary, semantic, "g_op")
    assert torch.equal(actual, auxiliary)
    assert summary["projected"] is False
    assert summary["removed_norm"] == 0.0


def test_gradient_statistics_keep_the_report_fields() -> None:
    value = torch.tensor([3.0, 4.0])
    assert canonical.gradient_stats(value, "g") == {
        "norm": 5.0,
        "rms": (12.5) ** 0.5,
        "shape": [2],
        "finite_fraction": 1.0,
        "zero_fraction": 0.0,
    }


def test_rms_normalization_preserves_the_existing_direction_formula() -> None:
    value = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    expected_rms = float(value.square().mean().sqrt())

    normalized, scale = canonical.rms_normalize(value, "v")

    assert scale == expected_rms
    assert torch.equal(normalized, value / expected_rms)
    assert canonical.rms(normalized, "q") == pytest.approx(1.0)


def test_robust_consumer_uses_shared_math_without_changing_its_report_schema() -> None:
    generated = torch.randn(3, 128, requires_grad=True)
    target = torch.randn(3, 128)
    expected_loss = (
        F.normalize(generated, dim=1, eps=robust.FEATURE_EPS)
        - F.normalize(target, dim=1, eps=robust.FEATURE_EPS)
    ).square().sum(dim=1).mean()
    actual_loss = robust._normalized_feature_loss(generated, target, "L_S")
    assert torch.equal(actual_loss, expected_loss)

    stats = robust._gradient_stats(torch.tensor([3.0, 4.0]), "g")
    assert set(stats) == {"norm", "shape", "finite_fraction", "zero_fraction"}

    view_input = torch.linspace(-1.0, 2.0, 28 * 28).reshape(1, 1, 28, 28)
    assert torch.equal(robust._cmos_view(view_input), canonical.cmos_view(view_input))


def test_invalid_feature_geometry_and_nonfinite_gradients_are_rejected() -> None:
    with pytest.raises(ValueError, match="feature shape"):
        canonical.normalized_feature_loss(torch.ones(2, 64), torch.ones(2, 64), "L_S")
    with pytest.raises(ValueError, match="nonfinite"):
        canonical.gradient_stats(torch.tensor([float("nan")]), "g_sem")
