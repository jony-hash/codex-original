from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from ldm_project.v2.application.research.optical_manifold.optical_manifold_smf_bogt import PORT_COUNT, energy_distance, registration_panel_prior, registration_panel_seed, sample_training_prior, training_prior_seed, two_pass_energy_distance_surrogate
from scripts.research.scaled_chen.run_scaled_chen_support_aware_energy_distance_preflight import CONFIG_PATH, MAX_GRADIENT_RATIO, MIN_GRADIENT_RATIO, TARGET_GRADIENT_RATIO, analytic_beta_registration, classify_replay_ratios, load_config, paired_quantile_bootstrap_ci, validate_endpoint_bindings, validate_future_objective_contract, validate_parent_identity, validate_prior_contract


def _manual_energy_distance(z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    cross = torch.linalg.vector_norm(z[:, None, :] - u[None, :, :], dim=-1)
    zz = torch.linalg.vector_norm(z[:, None, :] - z[None, :, :], dim=-1)
    uu = torch.linalg.vector_norm(u[:, None, :] - u[None, :, :], dim=-1)
    n = z.shape[0]
    m = u.shape[0]
    return (
        2.0 * cross.mean()
        - (zz.sum() - torch.diagonal(zz).sum()) / (n * (n - 1))
        - (uu.sum() - torch.diagonal(uu).sum()) / (m * (m - 1))
    ) / PORT_COUNT**0.5


def test_energy_distance_uses_ordered_u_statistic_and_sqrt_100_normalization() -> None:
    z = torch.arange(10 * PORT_COUNT, dtype=torch.float64).reshape(10, PORT_COUNT) / 100.0
    u = z.flip(0) + 0.25
    actual = energy_distance(z, u)
    assert torch.allclose(actual, _manual_energy_distance(z, u), atol=1.0e-12, rtol=1.0e-12)


def test_energy_distance_keeps_prior_prior_term_and_accepts_finite_negative_value() -> None:
    generator = torch.Generator(device="cpu").manual_seed(0)
    z = torch.randn((10, PORT_COUNT), generator=generator)
    u = torch.randn((10, PORT_COUNT), generator=generator)
    value = energy_distance(z, u)
    assert torch.isfinite(value)
    assert float(value) < 0.0

    cross_only = 2.0 * torch.linalg.vector_norm(z[:, None, :] - u[None, :, :], dim=-1).mean()
    assert not torch.isclose(value, cross_only)


def test_energy_distance_rejects_independent_microbatches_and_invalid_contract() -> None:
    z = torch.ones((10, PORT_COUNT))
    u = torch.zeros((10, PORT_COUNT))
    with pytest.raises(ValueError, match="logical batch"):
        energy_distance(z[:1], u[:1])
    with pytest.raises(ValueError, match="logical batch"):
        two_pass_energy_distance_surrogate(z, u, microbatch_size=0)
    with pytest.raises(ValueError, match="logical batch"):
        energy_distance(z[:, : PORT_COUNT - 1], u[:, : PORT_COUNT - 1])


def test_direct_and_two_pass_energy_distance_match_scalar_latent_and_parameter_gradients() -> None:
    generator = torch.Generator(device="cpu").manual_seed(11)
    source = torch.randn((10, PORT_COUNT), generator=generator, dtype=torch.float64)
    prior = torch.randn((10, PORT_COUNT), generator=generator, dtype=torch.float64)

    z_direct = source.clone().requires_grad_()
    direct = energy_distance(z_direct, prior)
    direct_grad = torch.autograd.grad(direct, z_direct)[0]

    z_two = source.clone().requires_grad_()
    two = two_pass_energy_distance_surrogate(z_two, prior, microbatch_size=1)
    two_grad = torch.autograd.grad(two, z_two)[0]
    assert torch.allclose(direct.detach(), two.detach(), atol=1.0e-12, rtol=1.0e-12)
    assert torch.allclose(direct_grad, two_grad, atol=1.0e-10, rtol=1.0e-8)

    weight_direct = nn.Parameter(torch.tensor(1.15, dtype=torch.float64))
    parameter_direct = energy_distance(weight_direct * source, prior)
    parameter_grad_direct = torch.autograd.grad(parameter_direct, weight_direct)[0]
    weight_two = nn.Parameter(torch.tensor(1.15, dtype=torch.float64))
    parameter_two = two_pass_energy_distance_surrogate(weight_two * source, prior, microbatch_size=1)
    parameter_grad_two = torch.autograd.grad(parameter_two, weight_two)[0]
    assert torch.allclose(parameter_grad_direct, parameter_grad_two, atol=1.0e-10, rtol=1.0e-8)


def test_energy_distance_gradient_boundary_keeps_decoder_direct_gradient_zero() -> None:
    source = torch.linspace(0.2, 1.2, 10 * PORT_COUNT, dtype=torch.float64).reshape(10, PORT_COUNT)
    encoder = nn.Parameter(torch.tensor(1.1, dtype=torch.float64))
    decoder = nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
    prior = torch.ones((10, PORT_COUNT), dtype=torch.float64)
    ed = energy_distance(encoder * source, prior)
    ed_encoder, ed_decoder = torch.autograd.grad(ed, (encoder, decoder), allow_unused=True)
    assert ed_encoder is not None and torch.isfinite(ed_encoder)
    assert ed_decoder is None

    mse = ((decoder * encoder * source) - 1.0).square().mean()
    mse_encoder, mse_decoder = torch.autograd.grad(mse, (encoder, decoder))
    assert torch.abs(mse_encoder) > 0.0
    assert torch.abs(mse_decoder) > 0.0


def test_zero_distance_has_defined_finite_subgradient_without_epsilon_smoothing() -> None:
    z = torch.ones((10, PORT_COUNT), dtype=torch.float64, requires_grad=True)
    u = torch.ones((10, PORT_COUNT), dtype=torch.float64)
    value = energy_distance(z, u)
    gradient = torch.autograd.grad(value, z)[0]
    assert torch.isfinite(value)
    assert torch.isfinite(gradient).all()


def test_training_prior_stream_is_deterministic_and_separate_from_registration_and_g1() -> None:
    assert training_prior_seed(5001) == 1238 * 100000 + 5001
    assert registration_panel_seed(0) == 1238 * 100000 + 10000
    assert len({registration_panel_seed(i) for i in range(3)}) == 3
    assert not ({registration_panel_seed(i) for i in range(3)} & {training_prior_seed(i) for i in range(5001, 6001)})
    first = sample_training_prior(5001)
    second = sample_training_prior(5001)
    panel = registration_panel_prior(0)
    assert torch.equal(first, second)
    assert not torch.equal(first, panel)
    assert first.shape == (10, PORT_COUNT)


def test_beta_registration_is_one_candidate_arithmetic_mean_and_replay_boundary_is_inclusive() -> None:
    norms = [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]
    registration = analytic_beta_registration(norms)
    assert registration["beta_i"] == pytest.approx([0.05, 0.05, 0.05])
    assert registration["beta_registered"] == pytest.approx(0.05)
    assert classify_replay_ratios([MIN_GRADIENT_RATIO, TARGET_GRADIENT_RATIO, MAX_GRADIENT_RATIO]) is True
    assert classify_replay_ratios([MIN_GRADIENT_RATIO - 1.0e-12, TARGET_GRADIENT_RATIO, MAX_GRADIENT_RATIO]) is False


def test_future_objective_rejects_bogt_and_nonzero_alpha() -> None:
    config = load_config(CONFIG_PATH)[1]
    validate_future_objective_contract(config)
    bad = deepcopy(config)
    bad["future_canary"]["alpha_bogt"] = 1.0e-6
    with pytest.raises(ValueError, match="BOGT"):
        validate_future_objective_contract(bad)


def test_parent_calibration_prior_and_endpoint_bindings_fail_closed() -> None:
    config = load_config(CONFIG_PATH)[1]
    validate_parent_identity(config)
    bad_parent = deepcopy(config)
    bad_parent["parent"]["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="parent"):
        validate_parent_identity(bad_parent)

    validate_prior_contract(config)
    bad_prior = deepcopy(config)
    bad_prior["prior"]["variance"] = 0.5
    with pytest.raises(ValueError, match="prior"):
        validate_prior_contract(bad_prior)

    validate_endpoint_bindings(config)
    bad_endpoint = deepcopy(config)
    bad_endpoint["endpoints"]["parent_mse_panel_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="endpoint"):
        validate_endpoint_bindings(bad_endpoint)


def test_paired_quantile_bootstrap_is_deterministic_and_preserves_pairing() -> None:
    parent = torch.linspace(0.1, 1.0, 40, dtype=torch.float64)
    end = parent - 0.1
    first = paired_quantile_bootstrap_ci(parent, end, probability=0.5, replicates=200, seed=1236)
    second = paired_quantile_bootstrap_ci(parent, end, probability=0.5, replicates=200, seed=1236)
    assert first == second
    assert first[1] < 0.0


def test_registration_constants_are_fixed() -> None:
    assert TARGET_GRADIENT_RATIO == 0.10
    assert MIN_GRADIENT_RATIO == 0.05
    assert MAX_GRADIENT_RATIO == 0.40
