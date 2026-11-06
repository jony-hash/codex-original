from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.research.lightgen import diagnose_lightgen_terminal_projection_alignment_v1 as study
from scripts.research.lightgen import execute_lightgen_g1_slm_projected_mapping_v2 as projected
from scripts.research.lightgen import measure_lightgen_train_generation_ols_support_v1 as support


def test_registered_config_fixes_endpoint_population_budget_and_semantic_gate() -> None:
    config = study._read_config(study.DEFAULT_CONFIG)

    assert config["measurement"]["endpoint"] == "E1G1"
    assert config["population"]["rows"] == 2_000
    assert config["output"]["optimizer_updates"] == 0
    assert config["measurement"]["bootstrap_seed"] == 202611021070


def test_terminal_train_reference_and_calibration_rows_are_disjoint_and_complete() -> None:
    positions = np.asarray([101 + 17 * offset for offset in range(200)], dtype=np.int64)

    reference, calibration = study._split_terminal_train_rows(positions)

    assert len(reference) == len(calibration) == 100
    assert not np.intersect1d(reference, calibration).size
    assert set(np.concatenate((reference, calibration)).tolist()) == set(positions.tolist())
    with pytest.raises(ValueError, match="200 unique rows"):
        study._split_terminal_train_rows(np.zeros(200, dtype=np.int64))


def test_numpy_right_inverse_projection_matches_independent_linear_system() -> None:
    matrix = np.asarray([[1.0, 0.0, 1.0, 0.0],
                         [0.0, 1.0, 0.0, 1.0]], dtype=np.float64)
    residual = np.asarray([[0.2, -0.4]], dtype=np.float64)

    correction = projected._numpy_minimum_norm_correction(matrix, residual)

    np.testing.assert_allclose(matrix @ correction.T, residual.T, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(correction, [[0.1, -0.2, 0.1, -0.2]], rtol=0.0, atol=1e-14)


def test_response_right_inverse_rejects_wrong_response_row_binding() -> None:
    matrix = torch.tensor([[1.0, 0.0, 1.0, 0.0],
                           [0.0, 1.0, 0.0, 1.0]], dtype=torch.float64)
    inverse = torch.tensor([[0.5, 0.0], [0.0, 0.5],
                            [0.5, 0.0], [0.0, 0.5]], dtype=torch.float64)

    assert study._validate_right_inverse(matrix, inverse, 1e-14) == 0.0
    with pytest.raises(RuntimeError, match="identity residual"):
        study._validate_right_inverse(matrix.flip(0), inverse, 1e-14)


def test_real_slm_commands_are_promoted_to_complex_for_response_prediction() -> None:
    commands = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]],
                              [[[0.6, 0.7], [0.8, 0.9]]]], dtype=torch.float64)
    response = torch.tensor([[1.0 + 2.0j, 0.5 - 0.2j, 0.0 + 0.3j, 1.2 + 0.0j],
                             [0.1 - 0.5j, 0.7 + 1.0j, 0.4 + 0.0j, -0.2 + 0.8j]],
                            dtype=torch.complex128)

    predicted = study._predict_coefficients(commands, response)
    expected = commands.reshape(2, 4).to(torch.complex128) @ response.T

    assert predicted.dtype == torch.complex128
    torch.testing.assert_close(predicted, expected)


def test_relative_l2_retains_imaginary_component_for_complex_coefficients() -> None:
    expected = torch.tensor([1.0 + 0.0j], dtype=torch.complex128)
    actual = torch.tensor([1.0 + 2.0j], dtype=torch.complex128)

    assert study._relative_l2(actual, expected) == pytest.approx(2.0)


def test_complex_distance_matches_independent_numpy_and_detects_relative_phase() -> None:
    query = ((1.0 + np.linspace(0.0, 0.5, 100))
             * np.exp(1j * np.linspace(-0.7, 0.9, 100)))[None, :]
    reference = query.copy()
    scale = np.linspace(0.5, 3.0, 100, dtype=np.float64)
    production = support._nearest_numpy(query, reference, scale, np.asarray([19]))
    independent = study._independent_nearest_numpy(query, reference, scale)
    np.testing.assert_allclose(production["distance"], independent["distance"], atol=1e-14)
    np.testing.assert_allclose(production["amplitude_distance"],
                               independent["amplitude_distance"], atol=1e-14)
    np.testing.assert_allclose(production["distance"], [0.0], atol=1e-14)

    public_phase = np.exp(1j * 1.17)
    scaled = support._nearest_numpy(2.3 * public_phase * query,
                                    0.7 * public_phase * reference, scale, np.asarray([19]))
    np.testing.assert_allclose(scaled["distance"], production["distance"], atol=2e-8)
    phase_changed = reference.copy()
    phase_changed[0, 1] *= np.exp(0.6j)
    changed = support._nearest_numpy(query, phase_changed, scale, np.asarray([19]))
    assert changed["distance"][0] > 0.0
    np.testing.assert_allclose(changed["amplitude_distance"], [0.0], atol=1e-14)


def test_distribution_bootstrap_resamples_calibration_and_preserves_paired_endpoints() -> None:
    baseline = {label: np.ones(200, dtype=np.float64) * (0.4 + label / 100)
                for label in range(10)}
    candidate = {label: baseline[label] + 0.1 for label in range(10)}
    calibration = {label: np.linspace(0.0, 1.0, 100, dtype=np.float64) + label / 1000
                   for label in range(10)}

    result = study._bootstrap_distribution(baseline, candidate, calibration)
    independent = study._independent_bootstrap_distribution(baseline, candidate, calibration)

    assert result["resamples"] == independent["resamples"]
    for key in ("baseline_G_ci95", "candidate_G_ci95", "delta_G_ci95"):
        np.testing.assert_allclose(result[key], independent[key], atol=1e-14, rtol=0.0)
    assert result["baseline_G_ci95"][1] > result["baseline_G_ci95"][0]
    np.testing.assert_allclose(result["delta_G_ci95"], [0.1, 0.1], atol=1e-14)


def test_semantic_decision_consumes_balanced_predictions_and_all_three_rules() -> None:
    labels = torch.arange(10).repeat_interleave(200)
    always_wrong = torch.full_like(labels, -1).tolist()
    correct = labels.tolist()
    evaluators = {
        "ED": {"baseline_predictions": always_wrong, "candidate_predictions": correct},
        "EC": {"baseline_predictions": correct, "candidate_predictions": correct},
    }
    diversity = {str(label): 0.9 for label in range(10)}

    supported = study._semantic_decision(labels, evaluators, diversity)

    assert supported["decision"] == "SUPPORTED_ON_CONSUMED_DIAGNOSTIC_POPULATION"
    assert supported["semantic_rule"] == {
        "ed_paired_ci95_lower_gt_zero": True,
        "ec_drop_at_most_2pp": True,
        "all_classes_diversity_retention_at_least_80pct": True,
    }
    diversity["9"] = 0.799
    rejected = study._semantic_decision(labels, evaluators, diversity)
    assert rejected["decision"] == "SEMANTIC_BENEFIT_NOT_SUPPORTED"
    diversity["9"] = 0.9
    evaluators["EC"]["candidate_predictions"] = always_wrong
    rejected = study._semantic_decision(labels, evaluators, diversity)
    assert rejected["semantic_rule"]["ec_drop_at_most_2pp"] is False


def test_final_shard_consumer_rejects_wrong_command_target_or_population_row() -> None:
    index = torch.tensor([1_437], dtype=torch.long)
    command = torch.full((1, 1, 200, 200), 0.4)
    expected_label = torch.tensor([7], dtype=torch.long)
    expected_proposal = torch.tensor([904], dtype=torch.long)
    expected_target = torch.ones((1, 100), dtype=torch.complex128)
    labels = torch.arange(10).repeat_interleave(200)
    assets = {"generation": {"labels": labels,
                              "proposal_ids": torch.zeros(2_000, dtype=torch.long),
                              "saved_prior_targets": torch.zeros(2_000, 100,
                                                                  dtype=torch.complex128)}}
    assets["generation"]["proposal_ids"][index] = expected_proposal
    assets["generation"]["saved_prior_targets"][index] = expected_target
    command_matrix = torch.zeros((2_000, 1, 200, 200))
    command_matrix[index] = command
    shard = {
        "task_id": study.TASK_ID,
        "population_role": "consumed_train_diagnostic",
        "condition": "P1_E1G1",
        "population_indices": index,
        "labels": expected_label,
        "proposal_ids": expected_proposal,
        "prior_targets": expected_target,
        "actual_slm_commands_forwarded": command_matrix.index_select(0, index),
        "projection_source": "R1_terminal_E1G1_plus_P1",
        "coefficients": torch.ones((1, 100), dtype=torch.complex128),
        "raw_cmos": torch.ones((1, 1, 200, 200), dtype=torch.float64),
        "modal_power": torch.ones(1, dtype=torch.float64),
    }

    study._validate_shard(shard, index, command_matrix, assets)
    wrong_command = dict(shard, actual_slm_commands_forwarded=command + 0.01)
    with pytest.raises(RuntimeError, match="wrong inputs"):
        study._validate_shard(wrong_command, index, command_matrix, assets)
    wrong_label = dict(shard, labels=torch.tensor([6]))
    with pytest.raises(RuntimeError, match="wrong inputs"):
        study._validate_shard(wrong_label, index, command_matrix, assets)
    missing = dict(shard, population_indices=torch.tensor([38]))
    with pytest.raises(RuntimeError, match="wrong inputs"):
        study._validate_shard(missing, index, command_matrix, assets)
