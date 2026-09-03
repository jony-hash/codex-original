from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.research.measure_scaled_chen_p1_calibration_raw_scale_canary import (
    C6_CANARY_BLOCKED_NUMERICAL,
    C6_CANARY_BLOCKED_REFERENCE_CONTRACT,
    C6_CANARY_PASS,
    classify_c6_gate,
    compute_dual_precision_means,
    float32_constants,
    float32_cast_errors,
    next_write_once_output,
    old_floor_counterfactual,
    raw_coordinate_sanity,
    raw_scale_statistics,
    validate_raw_means,
    zero_phase_reference_status,
)


def test_raw_mean_capture_and_dual_precision_are_per_port() -> None:
    powers = torch.stack(
        [
            torch.linspace(1.0e-12, 1.0e-10, 100, dtype=torch.float32),
            torch.linspace(2.0e-12, 2.0e-10, 100, dtype=torch.float32),
        ]
        * 500
    )
    mean_fp32, mean_fp64 = compute_dual_precision_means(powers)
    assert mean_fp32.shape == (100,)
    assert mean_fp64.shape == (100,)
    assert mean_fp32.dtype == torch.float32
    assert mean_fp64.dtype == torch.float64
    assert torch.allclose(mean_fp32.double(), mean_fp64, rtol=1.0e-6, atol=0.0)
    stats = raw_scale_statistics(mean_fp64)
    assert stats["count"] == 100
    assert stats["positive_count"] == 100
    assert stats["finite_count"] == 100


def test_eps_and_tiny_are_distinct_float32_constants() -> None:
    constants = float32_constants()
    assert constants["eps"] == pytest.approx(torch.finfo(torch.float32).eps)
    assert constants["tiny"] == pytest.approx(torch.finfo(torch.float32).tiny)
    assert constants["eps"] > constants["tiny"]


def test_old_floor_counterfactual_replays_epsilon_dominance() -> None:
    raw = torch.linspace(1.0e-12, 1.0e-10, 100, dtype=torch.float64)
    registered = torch.full_like(raw, torch.finfo(torch.float32).eps)
    replay = old_floor_counterfactual(raw, registered_scale=registered)
    assert replay["old_floor"] == pytest.approx(torch.finfo(torch.float32).eps)
    assert torch.equal(replay["old_scale"], torch.full_like(raw, replay["old_floor"]))
    assert float(replay["distortion"][50]) > 1000.0
    assert replay["registered_scale_equal_old_scale"] is True


def test_float32_cast_of_small_positive_means_is_representable() -> None:
    raw = torch.linspace(1.0e-12, 1.0e-9, 100, dtype=torch.float64)
    errors = float32_cast_errors(raw)
    assert errors["cast_positive"] is True
    assert errors["cast_finite"] is True
    assert errors["maximum_relative_error"] < 1.0e-5


def test_raw_mean_validation_rejects_nonpositive_or_nonfinite_values() -> None:
    validate_raw_means(torch.ones(100, dtype=torch.float64))
    with pytest.raises(ValueError, match="positive"):
        validate_raw_means(torch.zeros(100, dtype=torch.float64))
    nonfinite = torch.ones(100, dtype=torch.float64)
    nonfinite[1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_raw_means(nonfinite)


def test_raw_coordinate_sanity_is_a_definition_check() -> None:
    powers = torch.linspace(1.0e-12, 1.0e-9, 1000 * 100, dtype=torch.float32).reshape(1000, 100)
    _, mean_fp64 = compute_dual_precision_means(powers)
    sanity = raw_coordinate_sanity(powers, mean_fp64)
    assert sanity["finite"] is True
    assert sanity["global_mean"] == pytest.approx(1.0, rel=1.0e-6)
    assert torch.allclose(sanity["per_port_mean"], torch.ones(100, dtype=torch.float64), rtol=1.0e-6, atol=0.0)


def test_zero_phase_reference_contract_identity() -> None:
    parent_contract = {"calibrated_phase": "zero_residual_phase"}
    review = {"approved_contract": {"zero_phase_seed": 1234}}
    assert zero_phase_reference_status(parent_contract, review, model_phase_is_zero=True) == "YES"
    assert zero_phase_reference_status(parent_contract, review, model_phase_is_zero=False) == "UNDERDETERMINED"


def test_c6_gate_boundaries_are_fail_closed() -> None:
    kwargs = {
        "all_finite": True,
        "all_positive": True,
        "representable": True,
        "accumulation_ok": True,
        "floor_dominates": True,
        "old_replay_ok": True,
        "reference_status": "YES",
    }
    assert classify_c6_gate(**kwargs) == C6_CANARY_PASS
    assert classify_c6_gate(**{**kwargs, "reference_status": "UNDERDETERMINED"}) == C6_CANARY_BLOCKED_REFERENCE_CONTRACT
    assert classify_c6_gate(**{**kwargs, "representable": False}) == C6_CANARY_BLOCKED_NUMERICAL


def test_write_once_output_never_reuses_existing_attempt(tmp_path: Path) -> None:
    base = tmp_path / "lightgen_scaled_chen_p1_calibration_raw_scale_canary_v1_attempt_01"
    first = next_write_once_output(base)
    first.mkdir()
    second = next_write_once_output(base)
    assert second != first
    assert second.name.endswith("attempt_02")
