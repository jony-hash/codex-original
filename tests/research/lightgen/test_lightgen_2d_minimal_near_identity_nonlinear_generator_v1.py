from __future__ import annotations

import torch

from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.optics.d2nn import FiniteCropPhaseStack
from scripts.research.lightgen.execute_lightgen_2d_minimal_near_identity_nonlinear_generator_v1 import LOCAL_GAIN_CLASSIFICATION, MIXED_TRANSFER_CLASSIFICATION, NearIdentityNonlinearPhaseStack, RECOVERY_CLASSIFICATION, RECONSTRAINT_CLASSIFICATION, _classification


def _field() -> ComplexField:
    plane = PlaneSpec("test_generator_input", (4, 4), (2.0e-6, 2.0e-6), 532.0e-9, 0.0, "generator")
    torch.manual_seed(123)
    values = torch.randn(1, 1, 4, 4, dtype=torch.complex64)
    return ComplexField(values, plane, OperationProvenance("test_input", None, None))


def test_eta_zero_is_a_function_preserving_two_plane_identity() -> None:
    field = _field()
    baseline = FiniteCropPhaseStack(layers=2, active_shape=(4, 4), propagation_shape=(8, 8), spacing_m=1.5e-6)
    candidate = NearIdentityNonlinearPhaseStack(
        active_shape=(4, 4),
        propagation_shape=(8, 8),
        spacing_m=1.5e-6,
        intensity_reference=1.0,
    )
    with torch.no_grad():
        baseline.phase.copy_(torch.randn_like(baseline.phase) * 0.2)
        candidate.phase.copy_(baseline.phase)
        candidate.eta.zero_()

    baseline_output, baseline_records, _ = baseline(field, "baseline")
    candidate_output, candidate_records, stats = candidate(field, "candidate")

    assert len(baseline_records) == len(candidate_records) == 2
    assert torch.allclose(candidate_output.tensor, baseline_output.tensor, rtol=0.0, atol=0.0)
    assert all(torch.allclose(left.tensor, right.tensor, rtol=0.0, atol=0.0) for left, right in zip(baseline_records, candidate_records, strict=True))
    assert torch.allclose(stats["generator_nonlinear_input"], candidate_records[0].tensor, rtol=0.0, atol=0.0)
    assert torch.allclose(stats["generator_nonlinear_output"], candidate_records[0].tensor, rtol=0.0, atol=0.0)


def test_nonzero_eta_changes_phase_without_changing_field_amplitude() -> None:
    field = _field()
    candidate = NearIdentityNonlinearPhaseStack(
        active_shape=(4, 4),
        propagation_shape=(8, 8),
        spacing_m=1.5e-6,
        intensity_reference=1.0,
    )
    with torch.no_grad():
        candidate.phase.copy_(torch.randn_like(candidate.phase) * 0.2)
        candidate.eta.fill_(0.25)

    output, _, stats = candidate(field, "candidate")
    assert torch.isfinite(output.tensor).all()
    assert torch.allclose(stats["generator_nonlinear_output"].abs(), stats["generator_nonlinear_input"].abs(), rtol=2.0e-6, atol=2.0e-6)
    assert not torch.allclose(stats["generator_nonlinear_output"], stats["generator_nonlinear_input"], rtol=0.0, atol=1.0e-7)


def _semantic(*, delta: float, lower: float) -> dict[str, object]:
    return {"delta_macro_recall": delta, "paired_bootstrap_ci": {"lower": lower}}


def _reconstruction(delta: float) -> dict[str, object]:
    return {"delta_step_300_minus_step_0": delta}


def _transfer(*, feature_after: float, valid: bool = True) -> dict[str, object]:
    return {
        "after": {"R_feature": feature_after},
        "baseline_P2_Q2": {"R_feature": 1.0},
        "candidate_raw_validity": {
            "finite": valid,
            "nonconstant_fraction": 1.0 if valid else 0.5,
            "zero_image_count": 0 if valid else 1,
        },
    }


def test_nonlinear_classification_matches_declared_case_rules() -> None:
    assert _classification(_semantic(delta=0.02, lower=0.01), _reconstruction(0.001), _transfer(feature_after=0.8))[0] == RECOVERY_CLASSIFICATION
    assert _classification(_semantic(delta=0.02, lower=0.01), _reconstruction(0.001), _transfer(feature_after=1.2))[0] == MIXED_TRANSFER_CLASSIFICATION
    assert _classification(_semantic(delta=0.0, lower=0.0), _reconstruction(0.001), _transfer(feature_after=0.8))[0] == LOCAL_GAIN_CLASSIFICATION
    assert _classification(_semantic(delta=0.02, lower=0.01), _reconstruction(0.006), _transfer(feature_after=0.8))[0] == RECONSTRAINT_CLASSIFICATION
