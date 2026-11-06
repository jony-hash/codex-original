from __future__ import annotations

import pytest
import torch

from scripts.research.lightgen import diagnose_lightgen_frozen_slm_8bit_error_attribution_v1 as attribution


def test_common_phase_equivalent_coefficients_keep_zero_aligned_error() -> None:
    reference = torch.tensor([0.5 + 0.2j, -0.3 + 0.7j], dtype=torch.complex128)
    rotated = reference * torch.exp(torch.tensor(0.913j, dtype=torch.complex128))
    phase, error = attribution._phase_aligned_error(rotated, reference)
    assert phase == pytest.approx(0.913, abs=1e-14)
    assert error < 1e-14


def test_relative_norm_reports_delta_magnitude_not_distance_to_reference() -> None:
    reference = torch.tensor([3.0, 4.0], dtype=torch.float64)
    delta = torch.tensor([0.3, 0.4], dtype=torch.float64)
    assert attribution._relative_norm(delta, reference) == pytest.approx(0.1)
    assert attribution._relative(delta, reference) == pytest.approx(0.9)


def test_input_perturbation_report_summary_uses_fixed_condition_population() -> None:
    records = []
    for condition in attribution.CONDITIONS:
        for _ in range(200):
            records.append({
                "condition": condition,
                "amplitude_delta_mean": 0.01,
                "amplitude_delta_l2_relative_to_original": 0.1,
                "amplitude_delta_linf": 0.02,
                "intensity_delta_mean": -0.03,
                "intensity_delta_l2_relative_to_original": 0.2,
                "intensity_delta_linf": 0.04,
                "zeroed_positive_fraction_original_positive": 0.05,
            })
    summary = attribution._input_perturbation_summary(records)
    assert summary[attribution.CONDITIONS[0]]["amplitude_bias_mean"] == pytest.approx(0.01)
    assert summary[attribution.CONDITIONS[1]]["intensity_bias_mean"] == pytest.approx(-0.03)
    assert summary[attribution.CONDITIONS[1]]["zeroed_original_positive_fraction"] == {
        "median": pytest.approx(0.05), "p95": pytest.approx(0.05), "max": pytest.approx(0.05),
    }
    with pytest.raises(RuntimeError, match="input summary denominator changed"):
        attribution._input_perturbation_summary(records[:-1])


def test_zeroed_and_rest_response_needs_coherent_cross_term() -> None:
    zero_part = torch.tensor([1.0 + 0j, 0j], dtype=torch.complex128)
    rest_part = torch.tensor([1.0 + 0j, 2j], dtype=torch.complex128)
    total = zero_part + rest_part
    result = attribution._response_energy_decomposition(zero_part, rest_part, total)
    component_energy = result["zero_energy"] + result["rest_energy"]
    assert result["total_energy"] == pytest.approx(component_energy + result["cross_energy"])
    assert result["cross_energy"] == pytest.approx(2.0)
    assert result["total_energy"] != pytest.approx(component_energy)
    assert result["closure_relative_error"] == pytest.approx(0.0)


def test_intensity_perturbation_cannot_substitute_for_amplitude_perturbation() -> None:
    matrix = torch.tensor([[1.0 + 0j, 0.5 + 0j], [0.2 + 0j, -1.0 + 0j]],
                          dtype=torch.complex128)
    original = torch.tensor([0.2, 0.7], dtype=torch.float64)
    quantized = torch.tensor([0.25, 0.65], dtype=torch.float64)
    observed = matrix @ (quantized - original).to(torch.complex128)
    correct = matrix @ (quantized - original).to(torch.complex128)
    wrong = matrix @ (quantized.square() - original.square()).to(torch.complex128)
    assert attribution._relative(correct, observed) == pytest.approx(0.0, abs=1e-14)
    assert attribution._relative(wrong, observed) > 1e-2


def test_cmos_intensity_is_integrated_once_not_squared() -> None:
    intensity = torch.tensor([0.2, 0.4, 0.8], dtype=torch.float64)
    pixel_area = 2e-12
    physical_power = float(intensity.sum()) * pixel_area
    incorrect_power = float(intensity.square().sum()) * pixel_area
    assert incorrect_power != pytest.approx(physical_power, rel=1e-3)


def test_population_rejects_missing_and_duplicate_sample_identity() -> None:
    population = []
    for method in attribution.METHODS:
        for label in range(10):
            for _ in range(10):
                index = len(population)
                population.append({
                    "sample_key": f"{method}_{index:05d}",
                    "method": method,
                    "label": label,
                })
    attribution._validate_population(population, population.copy())
    with pytest.raises(RuntimeError, match="missing or duplicate"):
        attribution._validate_population(population, population[:-1])
    duplicated = population.copy()
    duplicated[-1] = duplicated[0]
    with pytest.raises(RuntimeError, match="missing or duplicate"):
        attribution._validate_population(population, duplicated)


def test_response_delta_closure_uses_continuous_coefficient_norm() -> None:
    continuous = torch.tensor([3.0 + 4.0j], dtype=torch.complex128)
    observed_delta = torch.tensor([0.01 + 0.02j], dtype=torch.complex128)
    predicted_delta = observed_delta + torch.tensor([1e-8 + 0j], dtype=torch.complex128)
    error = attribution._relative_residual(predicted_delta, observed_delta, continuous)
    assert error == pytest.approx(2e-9, rel=1e-8)


def test_production_matrix_terms_use_amplitude_and_distinct_intensity_delta() -> None:
    matrix = torch.tensor([[1.0 + 0j, 2.0 + 0j, -0.5 + 0j],
                           [0.0 + 1j, 0.5 + 0j, 1.0 - 0.5j]],
                          dtype=torch.complex128)
    amplitude = torch.tensor([[0.2, 0.7, 0.4], [0.5, 0.3, 0.8]], dtype=torch.float64)
    a8 = torch.tensor([[0.25, 0.65, 0.45], [0.55, 0.35, 0.75]], dtype=torch.float64)
    a8_gray = torch.tensor([[64, 166, 115], [140, 89, 191]], dtype=torch.uint8)
    intensity = amplitude.square()
    a8_intensity = a8.square()
    indices = [0, 1]
    predicted_continuous = (matrix @ amplitude.T.to(torch.complex128)).T
    terms = attribution._matrix_terms({
        "matrix": matrix,
        "amplitude": amplitude,
        "amplitude_a8": a8,
        "amplitude_i8": a8,
        "gray_a": a8_gray,
        "gray_i": a8_gray,
        "intensity_continuous": intensity,
        "intensity_a8": a8_intensity,
        "intensity_i8": a8_intensity,
    }, attribution.CONDITIONS[0], indices, predicted_continuous)
    expected = (matrix @ a8.T.to(torch.complex128)).T
    assert torch.allclose(terms["pred_quantized"], expected, rtol=0.0, atol=1e-14)
    wrong = (matrix @ (a8_intensity - intensity).T.to(torch.complex128)).T
    assert torch.allclose(terms["pred_wrong_intensity"], wrong, rtol=0.0, atol=1e-14)
    assert not torch.allclose(terms["pred_wrong_intensity"], terms["pred_delta"])


def test_saved_output_consumer_rejects_wrong_identity(tmp_path) -> None:
    identity = {
        "sample_key": "G1_00000", "method": "G1", "cohort_row": 0,
        "label": 0, "proposal_id": 3, "component_id": 1,
    }
    inputs = {"replay_dir": tmp_path, "replay_identity": {"run_id": "fixed-run"}}
    output_path = tmp_path / "samples" / attribution.CONDITIONS[0] / "2048" / "G1_00000.pt"
    output_path.parent.mkdir(parents=True)
    valid = {
        "schema": "lightgen_frozen_slm_8bit_sample_output.v1",
        "task_id": attribution.REPLAY_TASK_ID,
        "run_id": "fixed-run",
        "condition": attribution.CONDITIONS[0],
        "grid": 2048,
        "sample_identity": identity,
        "actual_coefficients": torch.ones(100, dtype=torch.complex128),
        "raw_cmos_intensity_w_per_m2": torch.ones(200, 200, dtype=torch.float64),
        "power_trace_w": {field: 1e-6 for field in attribution.POWER_FIELDS},
    }
    torch.save(valid, output_path)
    loaded = attribution._load_output(inputs, identity, attribution.CONDITIONS[0], 2048)
    assert loaded["sample_identity"] == identity
    assert torch.equal(loaded["actual_coefficients"], valid["actual_coefficients"])
    tampered = dict(valid)
    tampered["sample_identity"] = {**identity, "label": 1}
    torch.save(tampered, output_path)
    with pytest.raises(RuntimeError, match="saved sample identity changed"):
        attribution._load_output(inputs, identity, attribution.CONDITIONS[0], 2048)


def test_saved_power_table_identity_includes_grid_and_rejects_duplicates(tmp_path) -> None:
    rows = [
        {"condition": "A8", "sample_key": "G1_00000", "grid": 2048, "power": 1.0},
        {"condition": "A8", "sample_key": "G1_00000", "grid": 4096, "power": 2.0},
    ]
    path = tmp_path / "power.csv"
    attribution._write_csv(path, rows)
    attribution._verify_csv(path, rows)
    attribution._write_csv(path, [rows[0], rows[0]])
    with pytest.raises(RuntimeError, match="duplicate rows"):
        attribution._verify_csv(path, [rows[0], rows[0]])
