from __future__ import annotations

import json

import torch

from scripts.research.lightgen import (
    diagnose_lightgen_gaussian_effective_channel_noise_ablation_v1 as diagnosis,
)


CAMERA = {"full_well_e": 30000.0, "read_noise_e_rms": 1.0}


def test_poisson_gaussian_upper_clipped_moments_match_scalar_enumeration() -> None:
    for rate in (0.0, 1.0, 29900.0, 30000.0, 30100.0):
        mean, variance = diagnosis.clipped_camera_moments(
            torch.tensor([rate], dtype=torch.float64), CAMERA
        )
        reference_mean, reference_variance = diagnosis.enumerated_poisson_read_moments(
            rate, CAMERA
        )
        assert abs(float(mean[0]) - reference_mean) <= 2e-7
        assert abs(float(variance[0]) - reference_variance) <= 2e-5


def test_clipped_camera_mean_derivative_matches_independent_finite_difference() -> None:
    rate = 30000.0
    step = 0.1
    derivative = float(
        diagnosis.clipped_camera_mean_derivative(
            torch.tensor([rate], dtype=torch.float64), CAMERA
        )[0]
    )
    upper = diagnosis.enumerated_poisson_read_moments(rate + step, CAMERA)[0]
    lower = diagnosis.enumerated_poisson_read_moments(rate - step, CAMERA)[0]
    finite_difference = (upper - lower) / (2.0 * step)
    assert abs(derivative - finite_difference) <= 1e-6


def test_independent_angular_spectrum_reference_passes_analytic_controls() -> None:
    result = diagnosis.validate_independent_asm_reference(torch.device("cpu"))
    assert result["passed"] is True
    assert result["zero_distance_relative_error"] <= 1e-13
    assert result["same_grid_propagation_composition_relative_error"] <= 1e-10
    assert result["single_fft_plane_wave_relative_error"] <= 1e-10


def test_cross_grid_gate_failure_writes_diagnostic_closeout(tmp_path) -> None:
    reference = {
        "same_grid_all_probes_within_1e-4": True,
        "reference_method": "independent test reference",
        "independent_reference_controls": {"passed": True},
        "noise_moment_independent_reference": {
            "independent_scalar_enumeration": {
                "mean_max_abs_error_e": 1e-10,
                "variance_max_abs_error_e2": 1e-6,
            }
        },
        "endpoints": {
            "endpoint": {
                "same_grid": {
                    "2048": {
                        "saved_production_vs_independent_reference": {
                            "maximum_complex_field_relative_l2": 1e-6,
                            "maximum_camera_power_relative_error": 2e-6,
                        }
                    }
                },
                "independent_reference_cross_grid_difference": {
                    "maximum_complex_field_relative_l2": 2e-4,
                    "maximum_camera_power_relative_error": 3e-5,
                },
                "probe_ports_zero_based": [3, 9],
            }
        },
    }

    diagnosis.write_reference_gate_closeout(tmp_path, reference)

    report = json.loads((tmp_path / "machine_report.json").read_text(encoding="utf-8"))
    human_report = (tmp_path / "中文诊断报告.md").read_text(encoding="utf-8")
    assert report["status"] == "INVALID_HARNESS / CROSS_GRID_FINITE_WINDOW_OR_NUMERICAL_DIFFERENCE"
    assert report["scientific_measurement_completed"] is False
    assert "目标缺口" in human_report
    assert "全人口噪声测量未启动" in human_report
