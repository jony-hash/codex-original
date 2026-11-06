"""Focused contracts for C3.4 assumption-only process-aware height training."""

import math
import copy

import pytest
import torch

from scripts.research.lightgen import hw207_mnist4_c34_process_aware_v1 as process_aware
from scripts.research.lightgen import hw207_mnist4_c34_v1 as c34


def test_height_process_forward_matches_c34_hard_fabrication_operator():
    generator = torch.Generator().manual_seed(340401)
    phase = torch.rand((100, 100), generator=generator) * (2 * math.pi)
    design_height = c34.height_from_phase(phase, 0.5).float()

    actual = process_aware.process_height(
        design_height, psf_sigma_px=0.8, minimum_feature_radius_px=1,
        height_levels=64, maximum_height_um=1.2, noise_nm=20.0,
        noise_seed=23, edge_height_bias_nm=-20.0,
        straight_through_quantization=True)
    expected = c34.fabricate_height(
        phase, 0.5, psf_sigma_px=0.8, minimum_feature_radius_px=1,
        height_levels=64, maximum_height_um=1.2, height_noise_nm=20.0,
        noise_seed=23, edge_error_nm=-20.0)

    assert torch.allclose(actual, expected, rtol=0, atol=2e-7)


def test_ste_quantization_keeps_hard_values_and_finite_nonzero_gradient():
    generator = torch.Generator().manual_seed(340402)
    height = (0.1 + 0.8 * torch.rand((100, 100), generator=generator)).requires_grad_()
    processed = process_aware.process_height(
        height, psf_sigma_px=0.8, minimum_feature_radius_px=1,
        height_levels=64, maximum_height_um=1.2, noise_nm=20.0,
        noise_seed=41, edge_height_bias_nm=20.0,
        straight_through_quantization=True)
    reference = c34.fabricate_height(
        c34.phase_from_height(height, 0.5), 0.5,
        psf_sigma_px=0.8, minimum_feature_radius_px=1,
        height_levels=64, maximum_height_um=1.2, height_noise_nm=20.0,
        noise_seed=41, edge_error_nm=20.0)
    loss = processed.square().mean()
    gradient = torch.autograd.grad(loss, height)[0]

    assert torch.allclose(processed, reference, rtol=0, atol=2e-7)
    assert torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0


def test_smooth_process_derivative_matches_independent_central_difference():
    result = process_aware._derivative_reference()
    assert abs(result["analytic_derivative"]) > 1e-8
    assert result["relative_error"] < 1e-4


def test_height_parameter_uses_bounded_wrapped_k0_branch():
    phase = torch.linspace(-math.pi, 3 * math.pi, 10_000).reshape(100, 100)
    height = process_aware._bounded_design_height(phase, 0.5)
    assert float(height.min()) >= 0
    assert float(height.max()) <= c34.WAVELENGTH_UM / 0.5 + 1e-7
    assert not height.requires_grad


def test_process_height_rejects_invalid_nonfinite_inputs():
    height = torch.zeros((100, 100))
    height[0, 0] = float("nan")
    with pytest.raises(ValueError, match="invalid physical-height"):
        process_aware.process_height(
            height, psf_sigma_px=0.8, minimum_feature_radius_px=1,
            height_levels=64, maximum_height_um=1.2)


def test_recovery_optimizer_branches_do_not_share_loaded_state_tensors():
    source = torch.nn.Parameter(torch.tensor([1.0]))
    source_optimizer = torch.optim.Adam([source], lr=0.05)
    source.grad = torch.tensor([0.3])
    source_optimizer.step()
    saved = source_optimizer.state_dict()

    left = torch.nn.Parameter(source.detach().clone())
    right = torch.nn.Parameter(source.detach().clone())
    left_optimizer = torch.optim.Adam([left], lr=0.05)
    right_optimizer = torch.optim.Adam([right], lr=0.05)
    left_optimizer.load_state_dict(copy.deepcopy(saved))
    right_optimizer.load_state_dict(copy.deepcopy(saved))
    right_moment_before = right_optimizer.state[right]["exp_avg"].clone()
    left.grad = torch.tensor([0.2])
    left_optimizer.step()

    assert torch.equal(right_optimizer.state[right]["exp_avg"], right_moment_before)


def test_cpu_quota_range_parser_counts_ranges_and_single_cpus():
    assert process_aware._parse_cpu_range_list("0-15,18,20-21") == 19


def test_cpu_quota_range_parser_rejects_reversed_range():
    with pytest.raises(ValueError, match="invalid cgroup CPU range"):
        process_aware._parse_cpu_range_list("4-2")
