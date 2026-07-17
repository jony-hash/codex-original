from __future__ import annotations

import torch

from tools.diagnose_v672_transparent_phase import (
    candidate_passes,
    choose_candidate,
    direct_rayleigh_sommerfeld_reference,
    gradient_admission,
    memory_admission,
    run,
)


def _passing_candidate() -> dict[str, object]:
    summary = {
        "mean": 1.0e-5,
        "min": 1.0e-5,
        "max": 2.0e-5,
        "ci95": [1.0e-5, 2.0e-5],
    }
    return {
        "complex_field_relative_l2": summary,
        "amplitude_relative_l2": summary,
        "weighted_phase_mae_rad": summary,
        "intensity_corr": {"mean": 0.9999},
        "total_energy_relative_difference": summary,
        "guard_band_ratio": summary,
        "discarded_energy_ratio": 1.0e-5,
        "camera_intensity_corr": 0.9999,
        "reference_is_self": False,
        "sampling_fov_match": True,
        "direct_rs_pass": True,
        "all_plane_gates_pass": True,
    }


def test_v672_candidate_order_and_strict_gate() -> None:
    assert candidate_passes(_passing_candidate())
    assert choose_candidate({"state1536_fft3072": _passing_candidate()}) == {
        "status": "converged",
        "selected_state_grid": 1536,
        "selected_fft_grid": 3072,
    }
    failed = _passing_candidate()
    failed["guard_band_ratio"] = {"mean": 2.0e-4}
    decision = choose_candidate(
        {"state1536_fft3072": failed, "state2048_fft4096": failed}
    )
    assert decision["status"] == "grid-not-converged"
    assert decision["selected_state_grid"] is None


def test_v672_direct_rs_reference_is_small_grid_only() -> None:
    field = torch.ones(1, 1, 8, 8, dtype=torch.complex64)
    value = direct_rayleigh_sommerfeld_reference(field, 1.0e-5, 532e-9, 2e-6)
    assert value.shape == field.shape
    try:
        direct_rayleigh_sommerfeld_reference(
            torch.ones(1, 1, 65, 65, dtype=torch.complex64), 1.0e-5, 532e-9, 2e-6
        )
    except ValueError as exc:
        assert "64" in str(exc)
    else:
        raise AssertionError("large direct RS reference was not rejected")


def test_v672_empty_report_is_diagnostic_only(tmp_path) -> None:
    report = run(output=tmp_path / "v672.json", device="cpu")
    assert report["schema"] == "v67_transparent_convergence_v3"
    assert report["diagnostic_only"] is True
    assert report["status"] == "preflight-input-required"
    assert report["training_field_converged"] is False
    assert report["decision"] == "do-not-train-until-v67.2-gates-pass"


def test_v672_gradient_and_memory_gates_are_not_relaxed() -> None:
    metrics = gradient_admission(
        {
            "finite": True,
            "loss_relative_difference": 0.001,
            "encoder_gradient_cosine": 0.9995,
            "decoder_gradient_cosine": 0.9995,
            "generator_gradient_cosine": 0.9995,
            "gradient_norm_ratio": 1.01,
            "gradient_relative_l2": 0.01,
        }
    )
    assert metrics["passed"] is True
    metrics["gradient_norm_ratio"] = 1.03
    assert gradient_admission(metrics)["passed"] is False
    assert memory_admission(peak_reserved_bytes=85, available_bytes=100)["passed"] is True
    assert memory_admission(peak_reserved_bytes=86, available_bytes=100)["passed"] is False
