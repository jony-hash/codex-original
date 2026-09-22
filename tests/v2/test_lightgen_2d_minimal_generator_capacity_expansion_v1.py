from __future__ import annotations

import torch

from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.optics.d2nn import FiniteCropPhaseStack
from scripts.research.execute_lightgen_2d_minimal_generator_capacity_expansion_v1 import (
    LOCAL_GAIN_CLASSIFICATION,
    MIXED_TRANSFER_CLASSIFICATION,
    NOT_SUPPORTED_CLASSIFICATION,
    SUPPORTED_CLASSIFICATION,
    MinimalThreePlanePhaseStack,
    _classification,
    _expanded_optimizer_state,
)


def test_minimal_candidate_has_one_spatially_separated_split_plane() -> None:
    stack = MinimalThreePlanePhaseStack(
        active_shape=(4, 4),
        propagation_shape=(8, 8),
        spacing_m=1.5e-3,
    )

    assert tuple(stack.phase.shape) == (3, 1, 4, 4)
    assert stack.split_distance_m == 0.75e-3
    assert stack.split_distance_m * 2.0 == stack.spacing_m
    assert stack.layers == 3


def test_zero_midpoint_keeps_two_original_endpoints() -> None:
    torch.manual_seed(123)
    plane = PlaneSpec("test_generator_input", (4, 4), (2.0e-6, 2.0e-6), 532.0e-9, 0.0, "generator")
    field = ComplexField(
        torch.randn(1, 1, 4, 4, dtype=torch.complex64),
        plane,
        OperationProvenance("test_input", None, None),
    )
    two_plane = FiniteCropPhaseStack(
        layers=2,
        active_shape=(4, 4),
        propagation_shape=(8, 8),
        spacing_m=1.5e-6,
    )
    three_plane = MinimalThreePlanePhaseStack(
        active_shape=(4, 4),
        propagation_shape=(8, 8),
        spacing_m=1.5e-6,
    )
    with torch.no_grad():
        two_plane.phase.copy_(torch.randn_like(two_plane.phase) * 0.2)
        three_plane.phase[:2].copy_(two_plane.phase)
        three_plane.phase[2].zero_()

    two_output, two_records, _ = two_plane(field, "two")
    three_output, three_records, _ = three_plane(field, "three")

    assert len(two_records) == 2
    assert len(three_records) == 3
    assert two_output.tensor.shape == three_output.tensor.shape == (1, 1, 4, 4)
    assert torch.isfinite(three_output.tensor).all()
    # The split is the same ASM operator algebraically; this only guards the
    # small float32 FFT composition error in the focused production primitive.
    assert torch.allclose(two_output.tensor, three_output.tensor, rtol=2.0e-4, atol=2.0e-5)
    assert torch.allclose(two_records[0].tensor, three_records[0].tensor, rtol=0.0, atol=0.0)


def test_expanded_optimizer_state_copies_old_moments_and_zeros_new_plane() -> None:
    parameter = torch.zeros(3, 1, 100, 100)
    old_avg = torch.ones(2, 1, 100, 100)
    old_avg_sq = torch.full((2, 1, 100, 100), 2.0)
    prior = {
        "state": {0: {"step": torch.tensor(7.0), "exp_avg": old_avg, "exp_avg_sq": old_avg_sq}},
        "param_groups": [{"params": [0], "lr": 1.0e-3, "betas": (0.9, 0.999), "eps": 1.0e-8, "weight_decay": 0.0, "amsgrad": False}],
    }

    expanded = _expanded_optimizer_state(prior, parameter)
    state = expanded["state"][0]

    assert torch.equal(state["step"], torch.tensor(7.0))
    assert torch.equal(state["exp_avg"][:2], old_avg)
    assert torch.equal(state["exp_avg_sq"][:2], old_avg_sq)
    assert torch.equal(state["exp_avg"][2], torch.zeros_like(parameter[2]))
    assert torch.equal(state["exp_avg_sq"][2], torch.zeros_like(parameter[2]))


def _semantic(*, delta: float, lower: float) -> dict[str, object]:
    return {"delta_macro_recall": delta, "paired_bootstrap_ci": {"lower": lower}}


def _reconstruction(delta: float) -> dict[str, object]:
    return {"delta_step_300_minus_step_0": delta}


def _transfer(*, feature_after: float, valid: bool = True) -> dict[str, object]:
    return {
        "after": {"R_feature": feature_after},
        "baseline_P2_Q2": {"R_feature": 1.0},
        "candidate_raw_validity": {"finite": valid, "nonconstant_fraction": 1.0 if valid else 0.5, "zero_image_count": 0 if valid else 1},
    }


def test_capacity_classification_uses_only_declared_case_rules() -> None:
    assert _classification(_semantic(delta=0.02, lower=0.01), _reconstruction(0.001), _transfer(feature_after=0.8))[0] == SUPPORTED_CLASSIFICATION
    assert _classification(_semantic(delta=0.02, lower=0.01), _reconstruction(0.001), _transfer(feature_after=1.2))[0] == MIXED_TRANSFER_CLASSIFICATION
    assert _classification(_semantic(delta=0.0, lower=0.0), _reconstruction(0.001), _transfer(feature_after=0.8))[0] == LOCAL_GAIN_CLASSIFICATION
    assert _classification(_semantic(delta=0.02, lower=0.01), _reconstruction(0.006), _transfer(feature_after=0.8))[0] == "EXPANDED_GENERATOR_RECONSTRUCTION_CONSTRAINT_FAILED"
    assert NOT_SUPPORTED_CLASSIFICATION == "MINIMAL_PHASE_PLANE_CAPACITY_GAIN_NOT_SUPPORTED"
