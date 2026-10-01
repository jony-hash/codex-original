"""Contracts for the diagnostic electronic O0 mixture transport lifecycle."""

from __future__ import annotations

import torch

from ldm_project.v2.application.research.capacity.electronic_o0_mixture_transport import COMPONENT_CANDIDATES, _classification, _regularize_covariance, fit_gaussian_mixture, fit_classwise_mixtures, mixture_transport


def _synthetic_class_data(seed: int = 11) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(10).repeat_interleave(240)
    values = torch.empty((labels.numel(), 8))
    for label in range(10):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        class_values = torch.randn((240, 8), generator=generator) * 0.2
        class_values[:, 1] += label * 0.2
        class_values[:120, 0] -= 1.0
        class_values[120:, 0] += 1.0
        values[indices] = class_values
    return values, labels


def test_full_covariance_em_is_finite_and_nll_does_not_increase() -> None:
    values, labels = _synthetic_class_data()
    fit = fit_gaussian_mixture(values[labels == 3], 2, seed=8501)
    history = fit["nll_history"]
    assert fit["means"].shape == (2, 8)
    assert fit["cholesky"].shape == (2, 8, 8)
    assert torch.isfinite(fit["cholesky"]).all()
    assert all(
        after <= before + 1.0e-5
        for before, after in zip(history, history[1:])
    )


def test_covariance_shrinkage_preserves_shape_and_positive_diagonal() -> None:
    covariance = torch.eye(8) * 2.0
    regularized = _regularize_covariance(covariance)
    assert regularized.shape == (8, 8)
    assert torch.all(torch.diagonal(regularized) > 0)
    assert torch.allclose(regularized, regularized.T)


def test_classwise_selection_uses_registered_component_candidates() -> None:
    values, labels = _synthetic_class_data()
    mixture = fit_classwise_mixtures(values, labels, values, labels, seed=8501)
    assert len(mixture["components"]) == 10
    assert set(mixture["components"]).issubset(COMPONENT_CANDIDATES)
    assert all(
        len(fit["weights"]) == components
        for fit, components in zip(
            mixture["classes"], mixture["components"], strict=True
        )
    )


def test_mixture_transport_is_deterministic_and_preserves_pairing() -> None:
    values, labels = _synthetic_class_data()
    mixture = fit_classwise_mixtures(values, labels, values, labels, seed=8501)
    epsilon = torch.randn((100, 8), generator=torch.Generator().manual_seed(4))
    panel_labels = torch.arange(10).repeat_interleave(10)
    first = mixture_transport(epsilon, panel_labels, mixture)
    second = mixture_transport(epsilon, panel_labels, mixture)
    assert first.shape == (100, 8)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_mixture_transport_temperature_changes_only_covariance_spread() -> None:
    values, labels = _synthetic_class_data()
    mixture = fit_classwise_mixtures(values, labels, values, labels, seed=8501)
    epsilon = torch.randn((100, 8), generator=torch.Generator().manual_seed(5))
    panel_labels = torch.arange(10).repeat_interleave(10)
    base = mixture_transport(epsilon, panel_labels, mixture, covariance_temperature=1.0)
    wide = mixture_transport(epsilon, panel_labels, mixture, covariance_temperature=1.2)
    assert not torch.equal(base, wide)
    assert torch.isfinite(wide).all()


def test_closeout_classification_is_fail_closed() -> None:
    assert _classification({"passed": True}) == (
        "electronic_mixture_transport_admitted",
        "register_o0_g_command_bridge",
    )
    assert _classification(
        {"passed": False, "latent_fit_passed": False}
    )[0] == "mixture_latent_fit_failed"
    assert _classification({"status": "invalid_harness"})[0] == "invalid_harness"
