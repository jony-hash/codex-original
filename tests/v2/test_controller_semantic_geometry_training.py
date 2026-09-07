"""Numerical and state-isolation contract tests for the S3 repair harness."""

from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from ldm_project.v2.application.research.controller_semantic_geometry_training import (
    AUTHORIZATION_TOKEN,
    CONTROLLER_PARAMETER_NAMES,
    IMPLEMENTATION_READY,
    build_controller_optimizer,
    capture_state_snapshot,
    class_moments,
    freeze_optical_model,
    geometry_alignment_loss,
    internal_validation_codes,
    load_config,
    optimizer_parameter_names,
    train_authorized_controller,
    validate_frozen_state,
)
from ldm_project.v2.student.identity_generator import (
    IdentityAnchoredConditionalAmplitudeGenerator,
)


def _balanced_values() -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.arange(10, dtype=torch.long).repeat_interleave(50)
    rows = torch.arange(500, dtype=torch.float32).view(500, 1)
    dims = torch.arange(100, dtype=torch.float32).view(1, 100)
    values = (rows + 0.01 * dims).to(torch.float32)
    return values, labels


def test_covariance_uses_exact_n_minus_one_and_accepts_rank_deficiency() -> None:
    values, labels = _balanced_values()
    means, covariances = class_moments(values, labels)
    selected = values[:50]
    expected_mean = selected.sum(dim=0) / 50.0
    centered = selected - expected_mean
    expected_covariance = centered.T @ centered / 49.0
    assert torch.equal(means[0], expected_mean)
    assert torch.equal(covariances[0], expected_covariance)
    assert torch.linalg.matrix_rank(covariances[0]) == 1
    result = geometry_alignment_loss(values, values.clone(), labels, labels)
    assert result.loss.dtype == torch.float32
    assert result.loss.item() == pytest.approx(0.0, abs=1.0e-7)


def test_exact_anchor_covariance_and_half_mechanism_aggregation() -> None:
    real, labels = _balanced_values()
    controller = real.clone()
    shifts = torch.arange(10, dtype=torch.float32).repeat_interleave(50).view(-1, 1)
    controller[:, 0] += shifts[:, 0]
    result = geometry_alignment_loss(real, controller, labels, labels)
    means, covariances = class_moments(real, labels)
    controller_means, controller_covariances = class_moments(controller, labels)
    anchor = (controller_means - means).square().sum(dim=1) / covariances.diagonal(dim1=1, dim2=2).sum(dim=1)
    covariance = (controller_covariances - covariances).square().sum(dim=(1, 2)) / covariances.square().sum(dim=(1, 2))
    expected = ((anchor + covariance) / 2.0).mean()
    assert torch.equal(result.anchor_terms, anchor)
    assert torch.equal(result.covariance_terms, covariance)
    assert torch.equal(result.loss, expected)
    assert torch.equal(result.class_terms, (anchor + covariance) / 2.0)


def test_zero_and_nonfinite_real_denominators_fail_closed() -> None:
    _, labels = _balanced_values()
    zeros = torch.zeros(500, 100, dtype=torch.float32)
    with pytest.raises(ValueError, match="denominator"):
        geometry_alignment_loss(zeros, zeros, labels, labels)
    nonfinite = zeros.clone()
    nonfinite[0, 0] = float("nan")
    with pytest.raises(ValueError):
        geometry_alignment_loss(nonfinite, nonfinite, labels, labels)


def test_float64_and_controller_nonfinite_values_are_rejected() -> None:
    values, labels = _balanced_values()
    with pytest.raises(ValueError, match="float32"):
        geometry_alignment_loss(values.double(), values, labels, labels)
    bad_controller = values.clone()
    bad_controller[0, 0] = float("inf")
    with pytest.raises(ValueError, match="invalid_harness"):
        geometry_alignment_loss(values, bad_controller, labels, labels)


def test_objective_source_has_no_runtime_stabilizer_or_auxiliary_loss() -> None:
    source = inspect.getsource(geometry_alignment_loss)
    for forbidden in ("clamp_min", "jitter", "whitening", "pseudoinverse", "Sinkhorn", "MMD", "SWD"):
        assert forbidden not in source
    config = load_config("configs/research/lightgen_controller_semantic_geometry_training_v1.yaml")
    assert config["objective"]["prohibited_auxiliary"]
    assert config["objective"]["global_multiplier"] == 1.0


def test_controller_optimizer_is_exact_and_fresh() -> None:
    generator = IdentityAnchoredConditionalAmplitudeGenerator()
    optimizer = build_controller_optimizer(generator)
    assert optimizer_parameter_names(optimizer, generator) == CONTROLLER_PARAMETER_NAMES
    assert optimizer.state == {}
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == 0.0002
    assert optimizer.param_groups[0]["foreach"] is True
    assert optimizer.param_groups[0]["fused"] is False


def test_optical_parameters_are_excluded_and_frozen_state_hash_is_checked() -> None:
    generator = IdentityAnchoredConditionalAmplitudeGenerator()
    optical = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    optimizer = build_controller_optimizer(generator)
    freeze_optical_model(optical)
    before = capture_state_snapshot(generator, optical)
    after = capture_state_snapshot(generator, optical)
    validate_frozen_state(before, after, generator, optical, optimizer)
    assert all(not parameter.requires_grad for parameter in optical.parameters())
    with torch.no_grad():
        next(optical.parameters()).add_(1.0)
    changed = capture_state_snapshot(generator, optical)
    with pytest.raises(ValueError, match="frozen optical"):
        validate_frozen_state(before, changed, generator, optical, optimizer)


def test_internal_validation_panel_is_separate_and_deterministic() -> None:
    first = internal_validation_codes()
    second = internal_validation_codes()
    assert first.dtype == torch.float32
    assert first.shape == (100, 8)
    assert torch.equal(first, second)
    assert bool((first >= -1.0).all()) and bool((first <= 1.0).all())


def test_training_entrypoint_requires_explicit_authorization_before_mutation() -> None:
    generator = IdentityAnchoredConditionalAmplitudeGenerator()
    optical = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    values, labels = _balanced_values()
    from ldm_project.v2.application.research.controller_semantic_geometry_training import TargetBank

    bank = TargetBank(values[:500], labels, torch.arange(500), values[:500], labels, torch.arange(500), {})
    with pytest.raises(ValueError, match="explicit one-shot training authorization"):
        train_authorized_controller(
            generator,
            optical,
            bank,
            torch.ones(100, dtype=torch.float32),
            "unused.pt",
            authorization=None,
        )


def test_training_entrypoint_requires_passed_update_preflight_before_mutation() -> None:
    generator = IdentityAnchoredConditionalAmplitudeGenerator()
    optical = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    values, labels = _balanced_values()
    from ldm_project.v2.application.research.controller_semantic_geometry_training import TargetBank

    bank = TargetBank(values[:500], labels, torch.arange(500), values[:500], labels, torch.arange(500), {})
    with pytest.raises(ValueError, match="passed 3-step training preflight evidence"):
        train_authorized_controller(
            generator,
            optical,
            bank,
            torch.ones(100, dtype=torch.float32),
            "unused.pt",
            authorization=AUTHORIZATION_TOKEN,
        )


def test_authorization_token_is_exactly_the_next_gate() -> None:
    assert AUTHORIZATION_TOKEN == "AUTHORIZE_ONE_SHOT_CONTROLLER_SEMANTIC_GEOMETRY_TRAINING_CANARY"


def test_preflight_status_requires_fresh_one_shot_reauthorization() -> None:
    assert IMPLEMENTATION_READY == "IMPLEMENTATION_READY_FOR_ONE_SHOT_REAUTHORIZATION"
