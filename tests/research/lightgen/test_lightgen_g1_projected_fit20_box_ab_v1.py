from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from scripts.research.lightgen import execute_lightgen_g1_projected_fit20_box_ab_v1 as fit20
from scripts.research.lightgen import execute_lightgen_g1_slm_projected_mapping_v2 as projected


def _synthetic_consumer_case(tmp_path):
    amplitude = torch.zeros((20, 1, 200, 200), dtype=torch.float32)
    amplitude[:, 0, 0, 0] = torch.linspace(0.1, 0.29, 20)
    targets = torch.zeros((20, 100), dtype=torch.complex128)
    targets[:, 0] = amplitude.reshape(20, -1)[:, 0].to(torch.complex128)
    population = {
        "schema": "lightgen_g1_projected_fit20_population.v1",
        "task_id": fit20.TASK_ID,
        "role": "known_reachable_fit20",
        "coefficients": targets,
        "labels": torch.arange(10).repeat_interleave(2),
        "proposal_ids": torch.arange(20) + 100,
        "source_validation_indices": torch.arange(20) * 200,
    }
    response = torch.zeros((100, 40_000), dtype=torch.complex128)
    response[0, 0] = 1.0
    outputs = {}
    direct = {}
    for grid in fit20.GRIDS:
        outputs[grid] = {
            "coefficients": targets.clone(),
            "raw_cmos": torch.ones((20, 1, 200, 200), dtype=torch.float64),
        }
        direct[grid] = torch.ones((20, 1, 200, 200), dtype=torch.float64)
    rows = []
    for index in range(20):
        row = projected._consume_production_row(
            population,
            index,
            targets[index],
            amplitude[index:index + 1],
            {grid: {key: value[index:index + 1] for key, value in outputs[grid].items()}
             for grid in fit20.GRIDS},
            {grid: direct[grid][index:index + 1] for grid in fit20.GRIDS},
            response,
            "continuous_amplitude_float32",
        )
        row["source_validation_index"] = int(population["source_validation_indices"][index])
        row["row_passed"] = True
        rows.append(row)
    payload = {
        "schema": "lightgen_g1_fit20_observation.v1",
        "task_id": fit20.TASK_ID,
        "arm": "A",
        "completed_updates": 500,
        "population_role": "known_reachable_fit20",
        "command_encoding": "continuous_amplitude_float32",
        "target_coefficients": targets,
        "labels": population["labels"],
        "proposal_ids": population["proposal_ids"],
        "source_validation_indices": population["source_validation_indices"],
        "amplitudes": amplitude,
        "outputs": outputs,
        "direct_raw_cmos": direct,
        "rows": rows,
        "summary": projected._summarize_rows(rows),
    }
    inputs = {"response": response}
    path = tmp_path / "measurement.pt"
    torch.save(payload, path)
    return path, payload, population, inputs


def test_preclip_box_loss_matches_numpy_and_has_both_bound_derivatives() -> None:
    value = torch.full((1, 40_000), 0.5, dtype=torch.float64)
    value[0, :4] = torch.tensor([-0.25, 0.0, 1.0, 1.25], dtype=torch.float64)
    value.requires_grad_(True)

    loss = fit20._box_loss(value)
    gradient, = torch.autograd.grad(loss, value)
    array = value.detach().numpy()
    expected_loss = np.mean(np.maximum(-array, 0.0) ** 2 + np.maximum(array - 1.0, 0.0) ** 2)
    expected_gradient = 2.0 * (array - np.clip(array, 0.0, 1.0)) / 40_000

    assert float(loss) == pytest.approx(float(expected_loss), abs=1e-16)
    assert np.allclose(gradient.numpy(), expected_gradient, rtol=0.0, atol=1e-16)
    assert float(gradient[0, 0]) < 0.0 and float(gradient[0, 3]) > 0.0
    assert float(gradient[0, 1]) == 0.0 and float(gradient[0, 2]) == 0.0


def test_arm_b_total_loss_consumes_box_term_while_arm_a_does_not() -> None:
    coefficient_a = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    box_a = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    loss_a = fit20._combined_loss("A", coefficient_a, box_a)
    grad_box_a, = torch.autograd.grad(loss_a, box_a, allow_unused=True)

    coefficient_b = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    box_b = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    loss_b = fit20._combined_loss("B", coefficient_b, box_b)
    grad_box_b, = torch.autograd.grad(loss_b, box_b)

    assert float(loss_a) == pytest.approx(0.4)
    assert grad_box_a is None
    assert float(loss_b) == pytest.approx(1.4)
    assert float(grad_box_b) == pytest.approx(5.0)


def test_fit20_schedule_is_reproducible_and_visits_both_samples_for_each_class() -> None:
    population = {"role": "known_reachable_fit20"}
    first = fit20._build_schedule(population, 2026102781)
    second = fit20._build_schedule(population, 2026102781)

    assert torch.equal(first, second)
    assert tuple(first.shape) == (250, 2, 10)
    for cycle in range(250):
        for label in range(10):
            assert set(first[cycle, :, label].tolist()) == {2 * label, 2 * label + 1}


def test_final_consumer_recomputes_valid_outputs_and_rejects_bad_target_encoding_and_intensity_path(
    tmp_path,
) -> None:
    path, payload, population, inputs = _synthetic_consumer_case(tmp_path)
    accepted = fit20._consume_measurement_file(path, "A", population, inputs)
    assert accepted["summary"]["all_rows_passed"] is True

    wrong_target = copy.deepcopy(payload)
    wrong_target["target_coefficients"][0, 0] += 1.0
    torch.save(wrong_target, path)
    with pytest.raises(RuntimeError, match="wrong identity, target, or role"):
        fit20._consume_measurement_file(path, "A", population, inputs)

    wrong_encoding = copy.deepcopy(payload)
    wrong_encoding["command_encoding"] = "intensity_float32"
    torch.save(wrong_encoding, path)
    with pytest.raises(RuntimeError, match="wrong identity, target, or role"):
        fit20._consume_measurement_file(path, "A", population, inputs)

    wrong_intensity = copy.deepcopy(payload)
    wrong_intensity["outputs"][2048]["coefficients"][0, 0] += 0.02
    torch.save(wrong_intensity, path)
    with pytest.raises(ValueError, match="amplitude-to-coefficient response"):
        fit20._consume_measurement_file(path, "A", population, inputs)


def test_history_and_terminal_checkpoint_consumer_reject_wrong_schedule_or_adam_steps(tmp_path) -> None:
    schedule = fit20._build_schedule({"role": "known_reachable_fit20"}, 2026102781)
    valid_update = {"completed_update": 1, "sample_indices": schedule.reshape(500, 10)[0].tolist(),
                    "arm": "B", "coefficient_loss": 0.4, "box_loss": 0.2,
                    "total_loss": 1.4, "clipped_fraction_mean": 0.1,
                    "gradient_norms": {"network.0.weight": 1.0, "network.0.bias": 1.0,
                                       "network.2.weight": 1.0, "network.2.bias": 1.0}}
    history_path = tmp_path / "history.json"
    fit20._atomic_json(history_path, {"task_id": fit20.TASK_ID, "run_id": "run-x", "arm": "B",
                                      "state": "ACTIVE", "completed_new_updates": 1,
                                      "updates": [valid_update]})
    assert fit20._read_history(history_path, "B", "run-x", schedule) == [valid_update]

    bad_history = copy.deepcopy(valid_update)
    bad_history["sample_indices"] = list(reversed(bad_history["sample_indices"]))
    fit20._atomic_json(history_path, {"task_id": fit20.TASK_ID, "run_id": "run-x", "arm": "B",
                                      "state": "ACTIVE", "completed_new_updates": 1,
                                      "updates": [bad_history]})
    with pytest.raises(RuntimeError, match="fixed schedule"):
        fit20._read_history(history_path, "B", "run-x", schedule)

    checkpoint_dir = tmp_path / "arms" / "A" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "checkpoint_step_000500.pt"
    valid_checkpoint = {"schema": "lightgen_g1_projected_fit20_box_ab_checkpoint.v1",
                       "task_id": fit20.TASK_ID, "arm": "A", "completed_updates": 500,
                       "history_count": 500, "optimizer_deepcopied": True,
                       "optimizer_state": {"state": {str(i): {"step": torch.tensor(500.)}
                                                        for i in range(4)}}}
    torch.save(valid_checkpoint, checkpoint_path)
    assert fit20._arm_terminal_checkpoint(tmp_path, "A")["completed_updates"] == 500

    valid_checkpoint["optimizer_state"]["state"]["3"]["step"] = torch.tensor(499.)
    torch.save(valid_checkpoint, checkpoint_path)
    with pytest.raises(RuntimeError, match="Adam step-500"):
        fit20._arm_terminal_checkpoint(tmp_path, "A")


def test_fixed_matrix_consumer_rejects_a_right_inverse_with_nullspace_contamination(
    tmp_path, monkeypatch,
) -> None:
    for module in (fit20, projected):
        monkeypatch.setattr(module, "PORT_COUNT", 2)
        monkeypatch.setattr(module, "REAL_DIM", 4)
        monkeypatch.setattr(module, "PIXEL_COUNT", 5)
    response = torch.zeros((2, 5), dtype=torch.complex128)
    response[0, 0] = 1.0
    response[0, 1] = 1.0j
    response[1, 2] = 1.0
    response[1, 3] = 1.0j
    scale = torch.ones(2, dtype=torch.float64)
    matrix = projected._normalized_response(response, scale)
    inverse = torch.linalg.pinv(matrix)
    inputs = {"response": response, "matrix": matrix, "inverse": inverse, "scale": scale}
    path = tmp_path / "fixed_response_and_projector.pt"

    def save(candidate: torch.Tensor) -> None:
        torch.save({
            "schema": "lightgen_g1_fit20_fixed_response_projector.v1",
            "task_id": fit20.TASK_ID,
            "response_matrix": response,
            "normalized_real_matrix": matrix,
            "full_right_inverse": candidate,
            "scale": scale,
            "real_imag_order": "per_port_interleaved_real_imag",
        }, path)

    save(inverse)
    fit20._load_matrix(tmp_path, inputs, torch.device("cpu"))

    wrong = inverse.clone()
    wrong[4, 0] += 1.0
    assert torch.allclose(matrix @ wrong, torch.eye(4, dtype=torch.float64), atol=1e-12)
    save(wrong)
    with pytest.raises(RuntimeError, match="Moore-Penrose"):
        fit20._load_matrix(tmp_path, inputs, torch.device("cpu"))


def test_transfer_consumer_rejects_a_truncated_payload(tmp_path) -> None:
    evidence = tmp_path / "attempt"
    evidence.mkdir()
    payload = evidence / "terminal.pt"
    payload.write_bytes(b"12345678")
    fit20._atomic_json(evidence / "transfer_manifest.json", {
        "task_id": fit20.TASK_ID,
        "files": [{"path": "terminal.pt", "bytes": 8}],
        "file_count": 1,
        "bytes_total": 8,
    })
    assert fit20._verify_transfer_manifest(evidence)["all_current_file_sizes_match"] is True

    payload.write_bytes(b"1234")
    with pytest.raises(ValueError, match="wrong byte size"):
        fit20._verify_transfer_manifest(evidence)
