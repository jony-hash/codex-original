from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from scripts.research.lightgen import (
    diagnose_lightgen_gaussian_effective_channel_dual_grid_v1 as diagnosis,
)


CAMERA = {"full_well_e": 30000.0, "read_noise_e_rms": 1.0}


def test_poisson_gaussian_upper_clipped_moments_match_scalar_enumeration() -> None:
    for rate in (0.0, 1.0, 29900.0, 30000.0, 30100.0, 32536.0):
        mean, variance = diagnosis.clipped_camera_moments(
            torch.tensor([rate], dtype=torch.float64), CAMERA
        )
        reference_mean, reference_variance = diagnosis.enumerated_poisson_read_moments(
            rate, CAMERA
        )
        assert abs(float(mean[0]) - reference_mean) <= 2e-7
        assert abs(float(variance[0]) - reference_variance) <= 2e-5
        if rate == 32536.0:
            assert reference_variance > 0.0
            assert abs(float(variance[0]) - reference_variance) / reference_variance <= 1e-5


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
    assert abs(derivative - (upper - lower) / (2.0 * step)) <= 1e-6


def test_independent_angular_spectrum_reference_passes_analytic_controls() -> None:
    result = diagnosis.validate_independent_asm_reference(torch.device("cpu"))
    assert result["passed"] is True
    assert result["zero_distance_relative_error"] <= 1e-13
    assert result["same_grid_propagation_composition_relative_error"] <= 1e-10
    assert result["single_fft_plane_wave_relative_error"] <= 1e-10


def test_pixelwise_detectability_is_not_diluted_by_unaffected_pixels() -> None:
    delta = torch.tensor([[[3.0]]], dtype=torch.float64)
    variance = torch.tensor([[[1.0]]], dtype=torch.float64)
    distance = diagnosis._weighted_pixel_distance(delta, variance)
    expanded = diagnosis._weighted_pixel_distance(
        torch.tensor([[[3.0, 0.0, 0.0]]], dtype=torch.float64),
        torch.ones((1, 1, 3), dtype=torch.float64),
    )
    assert float(distance.item()) == 3.0
    assert float(expanded.item()) == 3.0


def test_dual_grid_sensitivity_band_covers_both_grid_predictions() -> None:
    production = torch.tensor([[[3.0, 0.0]]], dtype=torch.float64)
    reference = torch.tensor([[[2.0, 0.0]]], dtype=torch.float64)
    variance = torch.ones_like(production)
    d_prod, d_ref, uncertainty, lower, upper = diagnosis._detectability_interval(
        production, variance, reference, variance
    )
    assert float(d_prod.item()) == 3.0
    assert float(d_ref.item()) == 2.0
    assert float(uncertainty.item()) == 1.0
    assert float(lower.item()) <= float(d_ref.item()) <= float(upper.item())


def test_dual_grid_ablation_producer_keeps_both_grids_and_controls(monkeypatch) -> None:
    def targets(_kind, _context, rows, device):
        batch = len(rows)
        return (
            torch.zeros((batch, 1, 32, 32), device=device),
            torch.full((batch, 1, 28, 28), 0.5, device=device),
        )

    monkeypatch.setattr(diagnosis, "targets_for_rows", targets)
    generator = torch.Generator().manual_seed(7)
    real = torch.rand((100, 32, 32), generator=generator) * 0.01
    imag = torch.rand((100, 32, 32), generator=generator) * 0.01
    basis = torch.complex(real, imag)
    coefficients = np.ones((1, 100), dtype=np.complex64)
    geometries = {"production": (32, 32, 1.0e-10), "reference": (32, 32, 1.0e-10)}
    camera = {
        "quantum_efficiency": 0.5,
        "read_noise_e_rms": 1.0,
        "dark_current_e_per_pixel_s": 0.0,
        "full_well_e": 30000.0,
        "exposure_sweep_s": [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2],
    }
    result = diagnosis.compute_ablation_dual_grid_microbatch(
        "small",
        coefficients,
        np.asarray([12]),
        np.asarray([3]),
        {"production": basis.reshape(100, -1), "reference": basis.clone().reshape(100, -1)},
        geometries,
        1.0,
        {},
        camera,
        8,
        torch.device("cpu"),
    )
    assert result["native_metrics_production"].shape == (1, len(diagnosis.METRICS))
    assert result["ablation_metrics_reference"].shape == (1, 100, 2, len(diagnosis.METRICS))
    assert result["port_detectability"].shape == (1, 2, 4, 100, 5)
    assert result["observable_port_counts"].shape == (1, 2, 4, 3, 2)
    assert bool(result["equal_power_defined"].all())
    single_active = np.zeros((1, 100), dtype=np.complex64)
    single_active[0, 7] = 1.0 + 0.0j
    undefined = diagnosis.compute_ablation_dual_grid_microbatch(
        "small",
        single_active,
        np.asarray([12]),
        np.asarray([3]),
        {"production": basis.reshape(100, -1), "reference": basis.clone().reshape(100, -1)},
        geometries,
        1.0,
        {},
        camera,
        8,
        torch.device("cpu"),
    )
    assert not bool(undefined["equal_power_defined"][0, 7])
    assert np.isnan(undefined["equal_power_factor"][0, 7])
    assert np.isnan(undefined["port_detectability"][0, 1, :, 7, :]).all()


def test_zero_variance_and_zero_signal_are_explicit() -> None:
    zero_variance = torch.zeros((1, 1, 2), dtype=torch.float64)
    zero_signal = diagnosis._weighted_pixel_distance(torch.zeros_like(zero_variance), zero_variance)
    nonzero_signal = diagnosis._weighted_pixel_distance(
        torch.tensor([[[0.0, 1.0]]], dtype=torch.float64), zero_variance
    )
    assert float(zero_signal.item()) == 0.0
    assert torch.isinf(nonzero_signal).all()


def test_reference_precision_promotion_preserves_nonzero_complex_phase() -> None:
    class MixedState(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.real = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
            self.complex = nn.Parameter(torch.tensor([1.0 + 2.0j], dtype=torch.complex64))
            self.register_buffer("complex_buffer", torch.tensor([3.0 - 4.0j], dtype=torch.complex64))
            self.register_buffer("integer_buffer", torch.tensor([7], dtype=torch.int64))

    model = MixedState()
    counts = diagnosis.promote_reference_precision(model)
    assert model.real.dtype == torch.float64
    assert model.complex.dtype == torch.complex128
    assert model.complex_buffer.dtype == torch.complex128
    assert model.integer_buffer.dtype == torch.int64
    assert model.complex.item() == 1.0 + 2.0j
    assert model.complex_buffer.item() == 3.0 - 4.0j
    assert counts["complex128"] == 2


def test_bootstrap_two_sided_probability_uses_centered_absolute_tail_count() -> None:
    values = np.arange(20, dtype=np.float64).reshape(-1, 1)
    labels = np.repeat(np.arange(10, dtype=np.int64), 2)
    result = diagnosis.bootstrap_effect(values, labels, 2000, 2026102101)
    draws = result["bootstrap_draws"]
    centered = draws - result["mean"][None, :]
    expected = (np.sum(np.abs(centered) >= np.abs(result["mean"])[None, :], axis=0) + 1) / 2001
    np.testing.assert_array_equal(result["p_two_sided_bootstrap"], expected)


def test_per_port_bootstrap_excludes_explicitly_undefined_rows() -> None:
    labels = np.repeat(np.arange(10, dtype=np.int64), 3)
    values = np.column_stack((np.arange(30, dtype=np.float64), np.arange(30, dtype=np.float64)))
    valid = np.ones_like(values, dtype=bool)
    valid[0, 1] = False
    result = diagnosis.bootstrap_effect_per_port(values, labels, valid, 32, 2026102101)
    assert result["mean"][0] == values[:, 0].mean()
    expected_second = np.mean([values[(labels == label) & valid[:, 1], 1].mean() for label in range(10)])
    assert result["mean"][1] == expected_second
    assert np.isfinite(result["ci95_low"][1]) and np.isfinite(result["ci95_high"][1])


def test_retained_root_count_requires_every_later_point_to_stay_in_tolerance() -> None:
    rows = [
        {"mean_absolute_relative_mse_error": value}
        for value in (0.2, 0.005, 0.03, 0.004, 0.003)
    ]
    assert diagnosis.minimum_stable_retained_roots(rows, 0.01) == 4
    assert diagnosis.minimum_stable_retained_roots(rows, 0.05) == 2


def test_cumulative_retention_accepts_production_numpy_metrics(tmp_path, monkeypatch) -> None:
    basis = torch.ones((diagnosis.PORTS, 1), dtype=torch.complex64)
    geometry_path = tmp_path / "bases" / "endpoint_production_2048.npz"
    geometry_path.parent.mkdir(parents=True)
    np.savez_compressed(
        geometry_path,
        camera_field_per_unit_coefficient=np.ones((diagnosis.PORTS, 1, 1), dtype=np.complex64),
        pixel_area_m2=np.asarray(1.0),
    )
    monkeypatch.setattr(diagnosis, "load_basis", lambda *_args: (basis, 1.0))
    monkeypatch.setattr(
        diagnosis,
        "targets_for_rows",
        lambda _kind, _context, rows, device: (
            torch.zeros((len(rows), 1, 1, 1), device=device),
            torch.zeros((len(rows), 1, 1, 1), device=device),
        ),
    )
    monkeypatch.setattr(
        diagnosis,
        "calculate_metrics",
        lambda *_args: {"mse28": np.asarray([0.25], dtype=np.float64)},
    )
    monkeypatch.setattr(diagnosis.chen_quickcheck.baseline, "_resize_view", lambda value: value)
    population = {
        "rows": np.asarray([17], dtype=np.int64),
        "coefficients": np.ones((1, diagnosis.PORTS), dtype=np.complex64),
        "context": {},
        "gain": 1.0,
    }

    rows = diagnosis.cumulative_retention_dual_grid(
        "small", "endpoint", population, tmp_path, np.arange(diagnosis.PORTS),
        "production", 2048, torch.device("cpu"),
    )

    assert len(rows) == diagnosis.PORTS
    assert rows[0]["native_mean_mse28"] == 0.25
    assert rows[-1]["retained_roots"] == diagnosis.PORTS


def test_machine_report_keeps_reference_fields_and_compacts_local_spectra() -> None:
    local_record = {
        "endpoint": "small_step_030000",
        "representative_count": 200,
        "rows_per_class": 20,
        "source_role": "fixed DEV",
        "global_phase_direction_max_relative_response": 1.0e-15,
        "phase_direction_control_passed": True,
        "perturbation_radii_relative_to_coefficient_norm": [0.001, 0.01, 0.1],
        "primary_relative_perturbation": 0.01,
        "thresholds": [1.0, 3.0, 5.0],
        "population_summary": {"0.001": {"0.01": {"lower_median": 4.0}}},
        "finite_difference_control": {"flagged_nonlinear_over_5pct": False},
        "interpretation": "local direction summary",
        "per_image": [{"large": "full spectra live in the dedicated artifact"}],
    }
    independent_reference = {
        "status": "SAME_GRID_REFERENCE_VALIDATED_CROSS_GRID_DIFFERENCES_RECORDED",
        "endpoints": {"small_step_030000": {"independent_reference_cross_grid_difference": {}}},
    }
    report = diagnosis.assemble_dual_grid_machine_report(
        {"statistics": {"bootstrap_resamples": 2000, "bootstrap_seed": 2026102101, "fdr_q": 0.05}},
        {
            "status": "READY",
            "input_lineage": {},
            "noise_proxy": {"quantum_efficiency": 0.5},
            "basis_same_grid_all_100_ports_within_1e-4": True,
            "direct_production_vs_basis_checks": {
                "maximum_direct_vs_basis_mse_absolute_error": 1.0e-8,
                "maximum_production_w_per_pixel_vs_complex_field_power_relative_error": 1.0e-7,
                "passed": True,
            },
        },
        independent_reference,
        {},
        {},
        {},
        {},
        {"small_step_030000": local_record},
        ["figures/example.png"],
        1.0,
        {"small_step_030000": 1.0},
    )

    assert report["preflight"]["independent_reference"] is independent_reference
    assert report["preflight"]["all_100_port_complex_camera_bases_compared_to_reference"] is True
    assert report["preflight"]["production_w_per_pixel_power_identity_max_relative_error"] == 1.0e-7
    compact = report["local_direction_spectra"]["small_step_030000"]
    assert compact["population_summary"] == local_record["population_summary"]
    assert "per_image" not in compact
    assert report["artifacts"]["per_image_local_direction_spectra"] == "noise/local_direction_spectra.json"


def test_dual_grid_report_writer_consumes_independent_reference_payload(tmp_path) -> None:
    control = {
        "population_mean": {"mse28": {"mean_delta": 0.01}},
        "reliably_sensitive_port_count": 2,
    }
    local = {
        "primary_relative_perturbation": 0.01,
        "population_summary": {
            "0.001": {
                "0.01": {
                    "3.0": {
                        "lower_median": 2.0,
                        "lower_p10": 1.0,
                        "lower_p90": 3.0,
                        "upper_median": 4.0,
                        "upper_p10": 3.0,
                        "upper_p90": 5.0,
                    }
                }
            }
        },
    }
    report = {
        "task_id": diagnosis.TASK_ID,
        "status": "CONDITIONAL_DUAL_GRID_DIAGNOSTIC_COMPLETE_REAL_NOISE_CALIBRATION_PENDING",
        "decoder_ablation": {
            f"small_step_030000/{grid}": {
                "sample_count": 5000,
                "mean_modal_participation_neff": 10.0,
                "population_mean_power_vector_neff": 8.0,
                "population_mean_port_power_spread_db": 32.0,
                "controls": {"raw_zero": control, "equal_ols_power": control},
            }
            for grid in ("production", "reference")
        },
        "covariance": {
            "small_step_030000": {
                "sample_count": 5000,
                "effective_participation": 10.0,
                "directions_95_percent": 40,
                "directions_99_percent": 70,
                "raw_complex_covariance_effective_participation": 9.0,
                "original_ungauged_complex_cross_moment_available": True,
            }
        },
        "noise_observability": {
            "small_step_030000": {
                "raw_zero": {
                    "0.001": {
                        "mean_guaranteed_observable_ports": 30.0,
                        "mean_possible_observable_ports": 40.0,
                        "mean_undefined_equal_power_ports": 0.0,
                        "per_class_threshold_3": {},
                    }
                }
            }
        },
        "cumulative_retention": {
            "minimum_stable_roots": {
                "small_step_030000/production": {"0.01": 20, "0.05": 12}
            }
        },
        "local_direction_spectra": {"small_step_030000": local},
        "preflight": {
            "status": "READY",
            "all_100_port_complex_camera_bases_compared_to_reference": True,
            "direct_replay_max_mse_absolute_error": 1.0e-8,
            "direct_replay_and_ablation_checks_passed": True,
            "production_w_per_pixel_power_identity_max_relative_error": 1.0e-7,
            "independent_reference": {
                "endpoints": {
                    "small_step_030000": {
                        "independent_reference_cross_grid_difference": {
                            "maximum_complex_field_relative_l2": 1.2e-4,
                            "maximum_camera_power_relative_error": 2.0e-5,
                        }
                    }
                }
            },
        },
    }
    diagnosis.write_json(tmp_path / "machine_report.json", report)

    diagnosis.write_dual_grid_report(tmp_path)

    generated = (tmp_path / "中文诊断报告.md").read_text(encoding="utf-8")
    assert "0.00012" in generated
    assert "## 工程完成与验证" in generated


def test_dual_grid_writer_checks_durable_per_image_row_count(tmp_path) -> None:
    rows = np.arange(20, dtype=np.int64)
    labels = np.repeat(np.arange(10, dtype=np.int64), 2)
    native = np.zeros((20, len(diagnosis.METRICS)), dtype=np.float64)
    ablation = np.zeros((20, diagnosis.PORTS, len(diagnosis.CHANNELS), len(diagnosis.METRICS)), dtype=np.float64)
    detectability = np.zeros((20, len(diagnosis.CHANNELS), 4, diagnosis.PORTS, 5), dtype=np.float64)
    observable_counts = np.zeros((20, len(diagnosis.CHANNELS), 4, 3, 2), dtype=np.int16)
    modal_power = np.ones((20, diagnosis.PORTS), dtype=np.float64)
    equal_power_defined = np.ones_like(modal_power, dtype=bool)
    equal_power_defined[0, 0] = False
    ablation[0, 0, 1, :] = np.nan
    detectability[0, 1, :, 0, :] = np.nan
    data = {
        "rows": rows,
        "labels": labels,
        "native_metrics_production": native.copy(),
        "native_metrics_reference": native.copy(),
        "ablation_metrics_production": ablation.copy(),
        "ablation_metrics_reference": ablation.copy(),
        "port_detectability": detectability,
        "observable_port_counts": observable_counts,
        "modal_power_w": modal_power,
        "equal_power_defined": equal_power_defined,
    }
    cfg = {"statistics": {"bootstrap_resamples": 4, "bootstrap_seed": 2026102101, "fdr_q": 0.05}}
    camera = {"exposure_sweep_s": [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2]}
    diagnosis.write_dual_grid_tables(cfg, {"small_test": data}, tmp_path, camera)
    for grid_role in ("production", "reference"):
        receipt = tmp_path / "ablation" / f"per_image_per_port_ablation_{grid_role}_row_count.json"
        payload = diagnosis.json.loads(receipt.read_text(encoding="utf-8"))
        assert payload["written_data_rows"] == 20 * diagnosis.PORTS * len(diagnosis.CHANNELS)
        assert payload["readback_data_rows"] == payload["expected_data_rows"]
        assert payload["passed"] is True
    per_port = (tmp_path / "ablation" / "dual_grid_per_port_reconstruction_sensitivity.csv").read_text(encoding="utf-8-sig")
    assert "undefined_sample_count" in per_port and ",1," in per_port


def test_public_phase_direction_is_removed_from_local_coordinate_basis() -> None:
    coefficients = torch.tensor([1.0 + 2.0j, 3.0 - 1.0j], dtype=torch.complex64)
    tangent = torch.stack((-coefficients.imag, coefficients.real), dim=-1).reshape(-1).double()
    tangent = tangent / torch.linalg.vector_norm(tangent)
    q, _ = torch.linalg.qr(tangent[:, None], mode="complete")
    projection = q[:, 1:]
    assert projection.shape == (4, 3)
    assert torch.linalg.vector_norm(projection.T @ tangent).item() <= 1e-12


def test_local_jacobian_preserves_public_phase_null_direction_in_float64() -> None:
    rng = np.random.default_rng(2026102101)
    coefficients = (
        rng.normal(size=diagnosis.PORTS) + 1j * rng.normal(size=diagnosis.PORTS)
    ).astype(np.complex64)
    basis = (
        rng.normal(size=(diagnosis.PORTS, 32))
        + 1j * rng.normal(size=(diagnosis.PORTS, 32))
    ).astype(np.complex64)
    camera = {
        "quantum_efficiency": 0.5,
        "read_noise_e_rms": 1.0,
        "dark_current_e_per_pixel_s": 0.0,
        "full_well_e": 30000.0,
    }
    z = torch.from_numpy(coefficients)
    tangent = torch.stack((-z.imag, z.real), dim=-1).reshape(-1).double()
    jacobian, _, _, _ = diagnosis._local_jacobian(
        z, torch.from_numpy(basis), (1, basis.shape[1], 1.0e-18), camera, 1.0e-3
    )
    relative_response = torch.linalg.vector_norm(jacobian @ tangent) / (
        torch.linalg.matrix_norm(jacobian) * torch.linalg.vector_norm(tangent)
    )
    assert jacobian.dtype == torch.float64
    assert float(relative_response) <= 1.0e-14


def test_symmetric_complex_perturbation_keeps_sub_float32_ulp_delta() -> None:
    center = torch.tensor([1.0e6 + 0.0j], dtype=torch.complex64)
    direction = torch.tensor([1.0 + 1.0j], dtype=torch.complex128)
    plus, minus = diagnosis._symmetric_complex_perturbation(center, direction, 1.0e-2)
    assert plus.dtype == torch.complex128 and minus.dtype == torch.complex128
    assert plus.item() == 1.0e6 + 1.0e-2 + 1.0e-2j
    assert minus.item() == 1.0e6 - 1.0e-2 - 1.0e-2j


def test_finite_difference_prediction_uses_the_requested_grid_jacobian() -> None:
    jacobians = {
        "production": torch.tensor([[2.0, 0.0]], dtype=torch.float64),
        "reference": torch.tensor([[0.0, 5.0]], dtype=torch.float64),
    }
    direction = torch.tensor([1.0, 1.0], dtype=torch.float64)
    production = diagnosis._finite_difference_linear_prediction(jacobians, "production", direction, 0.1)
    reference = diagnosis._finite_difference_linear_prediction(jacobians, "reference", direction, 0.1)
    assert torch.equal(production, torch.tensor([0.2], dtype=torch.float64))
    assert torch.equal(reference, torch.tensor([0.5], dtype=torch.float64))
    with pytest.raises(RuntimeError, match="Jacobian is missing"):
        diagnosis._finite_difference_linear_prediction(jacobians, "missing", direction, 0.1)
