from __future__ import annotations

import torch

from scripts.research.lightgen.execute_lightgen_2d_attribute_canonical_semantic_realization_failure_v1 import ROWS_PER_CLASS_PER_HALF, _compose_gradient, _normalize_rms, _select_diagnostic_bank


def test_diagnostic_bank_is_deterministic_disjoint_and_class_balanced() -> None:
    labels = torch.arange(60_000, dtype=torch.long).remainder(10)
    train_indices = torch.arange(55_000, dtype=torch.long)

    first = _select_diagnostic_bank(train_indices, labels)
    second = _select_diagnostic_bank(train_indices, labels)

    for name in ("a_indices", "a_labels", "b_indices", "b_labels"):
        assert torch.equal(first[name], second[name])
    assert first["a_indices"].numel() == 10 * ROWS_PER_CLASS_PER_HALF
    assert first["b_indices"].numel() == 10 * ROWS_PER_CLASS_PER_HALF
    assert torch.unique(torch.cat((first["a_indices"], first["b_indices"]))).numel() == 40
    assert torch.bincount(first["a_labels"], minlength=10).tolist() == [ROWS_PER_CLASS_PER_HALF] * 10
    assert torch.bincount(first["b_labels"], minlength=10).tolist() == [ROWS_PER_CLASS_PER_HALF] * 10


def test_rms_direction_is_unit_scale_and_descends_the_input_gradient() -> None:
    gradient = torch.tensor([[[[1.0, -2.0], [3.0, -4.0]]]])

    direction = _normalize_rms(gradient, "test gradient")

    assert torch.allclose(direction, -gradient / gradient.square().mean().sqrt())
    assert torch.isclose(direction.square().mean().sqrt(), torch.tensor(1.0))


def test_conflict_aware_gradient_reconstructs_alignment_primary_projection() -> None:
    g_teacher = torch.ones((2, 1, 2, 2))
    g_rec_raw = torch.tensor([[[[-2.0, -2.0], [1.0, 1.0]]]]).repeat(2, 1, 1, 1)
    g_op_raw = torch.tensor([[[[1.0, -1.0], [1.0, -1.0]]]]).repeat(2, 1, 1, 1)

    composed = _compose_gradient(
        g_teacher,
        g_rec_raw,
        g_op_raw,
        beta=2.0,
        gamma=0.5,
        delta_align=0.25,
    )

    assert composed["projection"]["rec"]["projected"] is True
    assert composed["projection"]["op"]["projected"] is False
    assert torch.isfinite(composed["g_total"]).all()
    assert torch.sum(composed["g_total"] * composed["g_align"]) > 0
