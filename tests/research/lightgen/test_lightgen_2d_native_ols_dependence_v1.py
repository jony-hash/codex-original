from __future__ import annotations

import pytest
import torch

from scripts.research.lightgen import diagnose_lightgen_2d_native_ols_dependence_v1 as diagnostic


def _fixed_labels() -> torch.Tensor:
    return torch.arange(10, dtype=torch.long).repeat(30)


def _coefficients() -> torch.Tensor:
    generator = torch.Generator().manual_seed(20261004)
    real = torch.randn((300, 100), generator=generator)
    imaginary = torch.randn((300, 100), generator=generator)
    return torch.complex(real, imaginary)


def test_independent_port_permutation_preserves_complex_marginals_and_roles() -> None:
    labels = _fixed_labels()
    source = _coefficients()
    mapping = diagnostic._port_permutation_indices(labels, seed=2026100401)
    shuffled = diagnostic._shuffled_coefficients(source, mapping)

    diagnostic._verify_port_marginals(source, shuffled, mapping, labels)
    assert torch.equal(shuffled, source.gather(0, mapping))
    assert all(torch.equal(mapping[:, port], mapping[:, 0]) is False for port in range(100) if port != 0)
    assert torch.equal(labels.gather(0, mapping[:, 0]), labels)
    assert torch.all((mapping[:200] < 200))
    assert torch.all((mapping[200:] >= 200))


def test_shared_whole_row_permutation_preserves_full_vectors_and_negative_control() -> None:
    labels = _fixed_labels()
    source = _coefficients()
    mapping = diagnostic._port_permutation_indices(labels, seed=2026100403, shared_row_permutation=True)
    shuffled = diagnostic._shuffled_coefficients(source, mapping)

    assert torch.equal(mapping, mapping[:, :1].expand_as(mapping))
    assert torch.equal(shuffled, source.index_select(0, mapping[:, 0]))
    assert torch.equal(labels, labels.index_select(0, mapping[:, 0]))
    assert torch.equal(torch.bincount(labels, minlength=10), torch.bincount(labels.index_select(0, mapping[:, 0]), minlength=10))


def test_power_matching_uses_measured_plane_power_and_preserves_complex_phase() -> None:
    source = _coefficients()[:3]
    target_plane_power = torch.tensor([2.0, 11.0, 5.0], dtype=torch.float64)
    shuffled_plane_power = torch.tensor([8.0, 4.0, 20.0], dtype=torch.float64)

    matched, scale = diagnostic._power_match_coefficients(source, target_plane_power, shuffled_plane_power)

    expected = torch.sqrt(target_plane_power / shuffled_plane_power).to(source.real.dtype)
    assert torch.equal(scale, expected)
    assert torch.allclose(matched / source, scale.unsqueeze(1).to(matched.dtype), rtol=1e-6, atol=1e-7)
    assert torch.allclose(scale.to(torch.float64).square() * shuffled_plane_power, target_plane_power)


def test_exact_paired_mcnemar_and_holm_adjustment_are_conservative() -> None:
    labels = torch.ones(10, dtype=torch.long)
    original = torch.ones(10, dtype=torch.long)
    shuffled = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.long)

    result = diagnostic._exact_one_sided_mcnemar(original, shuffled, labels, 1)
    adjusted = diagnostic._holm({"class1": result["one_sided_exact_p_original_better"], "class7": 0.04, "other1": 0.2, "other7": 0.3})

    assert result["original_only_correct"] == 5
    assert result["counterfactual_only_correct"] == 0
    assert result["one_sided_exact_p_original_better"] == pytest.approx(1 / 32)
    assert adjusted["class1"] == pytest.approx(4 / 32)
    assert adjusted["class7"] == pytest.approx(0.125)


def test_support_gate_requires_effect_size_baseline_floor_and_adjusted_p() -> None:
    comparison = {
        "baseline_support_floor": {
            "1": {"E_D_recall_at_least_0_50": True, "E_C_recall_at_least_0_50": True},
            "7": {"E_D_recall_at_least_0_50": True, "E_C_recall_at_least_0_50": True},
        },
        "seed_comparisons": {
            str(seed): {
                "E_D_macro_recall_drop": 0.10,
                "per_class_E_D_recall_drop": {"1": 0.20, "7": 0.30},
                "per_class_E_C_recall_drop": {"1": 0.10, "7": 0.15},
            }
            for seed in diagnostic.PORT_SHUFFLE_SEEDS
        },
    }

    assert diagnostic._endpoint_support(comparison, {"mainline:class1": 0.05, "mainline:class7": 0.05}, "mainline")
    assert not diagnostic._endpoint_support(comparison, {"mainline:class1": 0.051, "mainline:class7": 0.05}, "mainline")
    comparison["seed_comparisons"][str(diagnostic.PORT_SHUFFLE_SEEDS[1])]["per_class_E_C_recall_drop"]["7"] = 0.09
    assert not diagnostic._endpoint_support(comparison, {"mainline:class1": 0.01, "mainline:class7": 0.01}, "mainline")


def test_fixed_view_perturbations_are_bounded_and_do_not_change_geometry() -> None:
    view = torch.zeros((2, 1, 28, 28), dtype=torch.float32)
    view[:, :, 10:15, 10:15] = 1.0

    shifted = diagnostic._shift_right_one(view)
    blurred = diagnostic._gaussian_blur(view)

    assert shifted.shape == blurred.shape == view.shape
    assert torch.isfinite(shifted).all() and torch.isfinite(blurred).all()
    assert shifted.max() <= 1.0 and blurred.min() >= 0.0 and blurred.max() <= 1.0
    assert torch.equal(shifted[0, 0, 12, 13], torch.tensor(1.0))
    assert blurred[0, 0, 12, 12] < 1.0


def test_frozen_state_check_detects_parameter_or_buffer_mutation() -> None:
    module = torch.nn.Linear(2, 2)
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    snapshot = diagnostic._snapshot(module)

    diagnostic._assert_frozen(module, snapshot, "toy")
    with torch.no_grad():
        module.weight[0, 0].add_(1.0)
    with pytest.raises(diagnostic.DiagnosticError, match="changed"):
        diagnostic._assert_frozen(module, snapshot, "toy")


def test_saved_chunk_resume_checks_rows_geometry_and_evaluator_payload(tmp_path) -> None:
    expected = {"system": "mainline", "scenario": "port_shuffle", "start": 0, "stop": 2}
    labels = torch.tensor([1, 7], dtype=torch.long)
    rows = torch.tensor([17, 29], dtype=torch.long)
    payload = {
        **expected,
        "labels": labels,
        "source_row_indices": rows,
        "ols_coefficients": torch.ones((2, 100), dtype=torch.complex64),
        "raw_cmos": torch.ones((2, 1, 200, 200), dtype=torch.float32),
        "evaluator_view_28x28": torch.ones((2, 1, 28, 28), dtype=torch.float32),
        "input_plane_integrated_power": torch.ones(2, dtype=torch.float64),
        "evaluator_logits": {"E_D": torch.zeros((2, 10), dtype=torch.float32)},
        "updates": 0,
    }
    path = tmp_path / "valid.pt"
    torch.save(payload, path)

    loaded = diagnostic._load_saved_chunk(path, expected, labels, rows, ("E_D",))
    assert torch.equal(loaded["source_row_indices"], rows)
    payload["labels"] = torch.tensor([1, 1], dtype=torch.long)
    bad_path = tmp_path / "wrong_labels.pt"
    torch.save(payload, bad_path)
    with pytest.raises(diagnostic.DiagnosticError, match="labels changed"):
        diagnostic._load_saved_chunk(bad_path, expected, labels, rows, ("E_D",))
