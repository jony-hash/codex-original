from __future__ import annotations

import math

import torch

from scripts.research.lightgen import execute_lightgen_2d_chen_bogt_phase_completion_repair_v1 as repair
from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import WrappedConditionalPhaseModel, wrap_phase


def _fixture_phase_model() -> WrappedConditionalPhaseModel:
    coefficients = torch.zeros((101, 100), dtype=torch.complex128)
    coefficients[0] = 1.0 + 0.0j
    residual_mean = torch.full((100,), 0.2, dtype=torch.float64)
    covariance = torch.eye(100, dtype=torch.float64)
    return WrappedConditionalPhaseModel(
        coefficients=coefficients,
        residual_mean=residual_mean,
        covariance=covariance,
        cholesky=covariance,
        scale_sha256="fixture-scale",
        fit_data_role="train",
        fit_count=55_000,
        engineering_only=False,
        data_identity={"source": "unit fixture"},
    )


def test_q3_adds_residual_mean_once_and_q0_keeps_frozen_double_semantics() -> None:
    u = torch.ones((repair.N_REFERENCE, 100), dtype=torch.float64)
    model = _fixture_phase_model()
    q0 = repair._current_phase_raw(u, model, 1234)
    q3 = repair._single_residual_mean_phase_raw(u, model, 1234)

    generator = torch.Generator(device="cpu").manual_seed(1234)
    noise = torch.randn(u.shape, dtype=torch.float64, generator=generator)
    assert torch.allclose(q0, wrap_phase(noise + 0.4))
    assert torch.allclose(q3, wrap_phase(noise + 0.2))


def test_anchor_conditioned_candidates_keep_target_anchor_zero() -> None:
    target_u = torch.ones((repair.N_REFERENCE, 100), dtype=torch.float64)
    train_u = torch.ones((2_000, 100), dtype=torch.float64)
    train_u[:, 0] = 2.0
    train_phase = torch.zeros((2_000, 100), dtype=torch.float64)
    train_phase[:, 1:] = torch.linspace(-math.pi, math.pi, 2_000).reshape(-1, 1)
    train_anchors = torch.zeros((2_000,), dtype=torch.long)

    q1, q1_anchors = repair._anchor_conditioned_marginal_phase(
        target_u, train_u, train_phase, train_anchors, seed=1
    )
    q2, q2_anchors = repair._anchor_conditioned_joint_phase(
        target_u, train_phase, train_anchors, seed=2
    )
    assert torch.equal(q1_anchors, torch.zeros(repair.N_REFERENCE, dtype=torch.long))
    assert torch.equal(q2_anchors, torch.zeros(repair.N_REFERENCE, dtype=torch.long))
    assert torch.equal(q1[:, 0], torch.zeros(repair.N_REFERENCE, dtype=torch.float64))
    assert torch.equal(q2[:, 0], torch.zeros(repair.N_REFERENCE, dtype=torch.float64))


def test_paired_swap_statistic_matches_direct_energy_distance() -> None:
    candidate = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float64)
    real = torch.tensor([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], dtype=torch.float64)
    candidate_candidate = torch.cdist(candidate, candidate)
    candidate_real = torch.cdist(candidate, real)
    real_real = torch.cdist(real, real)
    swaps = torch.tensor([[False, False, False], [True, False, True]])
    actual = repair._paired_swap_statistic(swaps, candidate_candidate, candidate_real, real_real)
    expected = []
    for swap in swaps:
        left = torch.where(swap[:, None], real, candidate)
        right = torch.where(swap[:, None], candidate, real)
        expected.append(repair.baseline._energy_distance(left, right))
    assert torch.allclose(actual, torch.tensor(expected, dtype=torch.float64), atol=1.0e-12, rtol=0.0)
