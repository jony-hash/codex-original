from __future__ import annotations

import torch
import pytest
import json
import copy
import math
import numpy as np

from ldm_project.v2.application.research.lightgen.confirmation_evaluators import (
    PixelMLPConfirmationEvaluatorV1,
    ThirdConfirmationEvaluator,
)
from scripts.research.lightgen import execute_lightgen_g1_semantic_exposure200_v1 as study


def test_schedule_provides_exactly_750_exposures_and_one_sample_per_class() -> None:
    schedule = study._build_schedule(2026102971)
    assert schedule.shape == (15_000, 10)
    assert torch.equal(torch.bincount(schedule.flatten(), minlength=200),
                       torch.full((200,), 750))
    for batch in schedule:
        assert torch.equal(batch.div(20, rounding_mode="floor"), torch.arange(10))


def test_summary_computes_numpy_rms_without_mutating_or_rescaling_errors() -> None:
    rows = [{"label": label, "row_passed": offset == 0,
             **{key: 3.0 + offset for key in (*study.METRIC_KEYS, "clipped_fraction")}}
            for label in range(10) for offset in range(2)]
    summary = study._summary(rows)
    assert summary["passed"] == 10
    assert summary["statistics"]["coefficient_2048"]["rms"] == pytest.approx(3.5355339059)
    assert summary["statistics"]["coefficient_2048"]["max"] == 4.0


def test_frozen_state_comparison_preserves_complex_imaginary_differences() -> None:
    before = {"field": torch.tensor([1.0 + 2.0j], dtype=torch.complex128)}
    unchanged = {"field": before["field"].clone()}
    changed_imaginary = {"field": torch.tensor([1.0 + 2.25j], dtype=torch.complex128)}

    assert study._compare_parameter_states(before, unchanged) == 0.0
    assert study._compare_parameter_states(before, changed_imaginary) == pytest.approx(0.25)


def test_box_loss_has_independent_value_gradient_and_finite_difference_checks() -> None:
    result = study._box_loss_qualification(torch.device("cpu"))
    assert result["independent_numpy_gradient_max_abs_error"] == 0.0
    assert result["gradient_descent_direction_passed"] is True
    assert max(item["absolute_error"] for item in result["finite_difference_checks"].values()) <= 1.0e-11


def test_public_phase_schedule_deduplicates_the_two_anchor_phases() -> None:
    phases = study._fallback_phase_schedule(0.0, 0.0)
    assert len(phases) == 32
    assert len({round(value, 12) for value in phases}) == len(phases)
    assert study._fallback_phase_schedule(0.123, -0.456)[:2] == pytest.approx([0.123, -0.456])


def test_fista_fixed_phase_solution_and_explicit_state_resume() -> None:
    response = torch.eye(2, dtype=torch.complex128)
    phase = 0.37
    amplitude = torch.tensor([0.23, 0.78], dtype=torch.float64)
    target = amplitude.to(torch.complex128) * torch.exp(torch.tensor(-1j * phase,
                                                                     dtype=torch.complex128))
    checkpoints = []
    first = study._fista_fixed_phase_solve(
        response, target, phase, max_steps=50, checkpoint=checkpoints.append,
        checkpoint_interval=25)
    assert first["relative_error"] <= 1.0e-6
    assert torch.allclose(first["amplitude"], amplitude, atol=1.0e-12, rtol=0.0)
    assert checkpoints and checkpoints[0]["iteration"] == 25
    resumed = study._fista_fixed_phase_solve(
        response, target, phase, max_steps=50, initial_state=checkpoints[0])
    assert resumed["relative_error"] <= 1.0e-6
    assert torch.allclose(resumed["amplitude"], first["amplitude"], atol=1.0e-12, rtol=0.0)


def test_fista_third_to_fourth_step_state_is_exact_after_independent_reload(tmp_path) -> None:
    response = torch.tensor([[1.0 + 0.3j, 0.2 - 0.1j],
                             [0.1 + 0.2j, -0.4 + 0.7j]], dtype=torch.complex128)
    target = torch.tensor([0.3 - 0.4j, 0.7 + 0.2j], dtype=torch.complex128)
    phase = 0.31
    step3 = []
    study._fista_fixed_phase_solve(response, target, phase, max_steps=3,
                                   checkpoint=step3.append, checkpoint_interval=3)
    assert step3 and step3[0]["iteration"] == 3
    checkpoint = tmp_path / "solver_step3.pt"
    torch.save(step3[0], checkpoint)
    restored_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    direct4 = []
    restored4 = []
    direct = study._fista_fixed_phase_solve(
        response, target, phase, max_steps=4, initial_state=step3[0],
        checkpoint=direct4.append, checkpoint_interval=4)
    replay = study._fista_fixed_phase_solve(
        response, target, phase, max_steps=4, initial_state=restored_state,
        checkpoint=restored4.append, checkpoint_interval=4)
    same, delta = study.projected._tree_max_delta(direct4[0], restored4[0])
    assert same and delta <= 1.0e-12
    assert direct["iterations"] == replay["iterations"] == 4
    assert abs(direct["relative_error"] - replay["relative_error"]) <= 1.0e-12


def test_dual_box_error_bounds_are_below_the_primal_box_residual() -> None:
    response = torch.eye(2, dtype=torch.complex128)
    target = torch.tensor([1.4 + 0.2j, -0.3 + 0.4j], dtype=torch.complex128)
    amplitude = torch.tensor([1.0, 0.0], dtype=torch.float64)
    bounds = study._dual_box_error_bounds(response, target, amplitude)
    assert 0.0 <= bounds["relative_dual_lower_bound"] <= bounds["relative_upper_bound"]
    assert bounds["relative_gap"] == pytest.approx(
        bounds["relative_upper_bound"] - bounds["relative_dual_lower_bound"])


def test_phase_interval_sinusoid_minimum_is_exact_across_periodic_seam() -> None:
    value = complex(math.cos(0.37), math.sin(0.37))
    left, right = math.pi - 0.37 - 0.08, math.pi - 0.37 + 0.06
    actual = study._sinusoid_minimum(value, left, right)
    assert actual == pytest.approx(-1.0, abs=1.0e-14)
    seam = study._sinusoid_minimum(value, 2.0 * math.pi - 0.1, 2.0 * math.pi + 0.1)
    sample = np.linspace(2.0 * math.pi - 0.1, 2.0 * math.pi + 0.1, 10001)
    reference = float(np.min(np.real(np.exp(1j * sample) * value)))
    assert seam == pytest.approx(reference, abs=1.0e-10)


def _narrow_phase_search_fixture() -> tuple[torch.Tensor, torch.Tensor, dict]:
    response = torch.ones((1, 1), dtype=torch.complex128)
    target = torch.tensor([0.5 * np.exp(-1j * math.pi / 32)], dtype=torch.complex128)
    anchors = [2.0 * math.pi * index / 32 for index in range(32)]

    def evaluate(candidate: dict) -> dict:
        phase = float(candidate["phase_rad"])
        command = candidate["amplitude"].to(torch.float64)
        residual = command.to(torch.complex128) @ response.T
        desired = torch.exp(torch.tensor(1j * phase, dtype=torch.complex128)) * target
        error = float(torch.linalg.vector_norm(residual.reshape(-1) - desired)
                      / torch.linalg.vector_norm(target))
        passed = error <= study.GATE_LIMIT
        metrics = {key: error for key in study.METRIC_KEYS}
        return {"coefficient_relative_error_2048": error,
                "all_six_metrics_passed": passed,
                "record": {"metrics": metrics} if passed else None}

    result = study._adaptive_public_phase_search(
        response, target, anchors, torch.tensor([0.5]), evaluate_candidate=evaluate,
        lipschitz=1.0)
    return response, target, result


def test_adaptive_phase_search_finds_a_narrow_phase_missed_by_old_mesh() -> None:
    response, target, result = _narrow_phase_search_fixture()
    assert result["status"] == "witness_found"
    assert result["completed_solves"] <= study.CONTINUOUS_PHASE_SOLVE_LIMIT
    assert result["attempts"]
    assert min(row["coefficient_relative_error_2048"] for row in result["attempts"]) >= 0.0
    # The old mesh has no point within the registered 1e-4 coefficient threshold.
    mesh_errors = []
    for phase in (2.0 * math.pi * index / 32 for index in range(32)):
        desired = torch.exp(torch.tensor(1j * phase, dtype=torch.complex128)) * target
        mesh_errors.append(float(torch.linalg.vector_norm(desired - 0.5 * response[0])
                                 / torch.linalg.vector_norm(target)))
    assert min(mesh_errors) > study.GATE_LIMIT
    verified = study._verify_phase_interval_certificate(response, target, result)
    assert verified["periodic_interval_coverage_verified"] is True


def test_phase_witness_consumer_binds_pass_to_exact_command_and_full_optics_metrics() -> None:
    _response, _target, certificate = _narrow_phase_search_fixture()
    attempt = next(row for row in certificate["attempts"] if row["all_six_metrics_passed"])
    row = {"all_six_metrics_passed": True,
           "amplitude": attempt["command"].reshape(1, 1, 1, -1),
           "phase_rad": attempt["phase_rad"], "metrics": attempt["metrics"]}
    study._verify_phase_witness_binding(row, certificate)
    row["amplitude"] = row["amplitude"].clone()
    row["amplitude"].view(-1)[0] += 0.01
    with pytest.raises(RuntimeError, match="not bound to the saved measured instruction"):
        study._verify_phase_witness_binding(row, certificate)


def test_phase_interval_bound_excludes_an_over_range_box_target() -> None:
    response = torch.ones((1, 1), dtype=torch.complex128)
    target = torch.tensor([1.2 + 0.0j], dtype=torch.complex128)
    seed = torch.tensor([0.5], dtype=torch.float32)
    directions = [study._direction_for_phase(response, target, phase, seed.double())
                  for phase in (2.0 * math.pi * index / 32 for index in range(32))]
    directions = [value for value in directions if value is not None]
    intervals = study._initial_phase_intervals(
        [2.0 * math.pi * index / 32 for index in range(32)], response, target, directions)
    assert len(intervals) == 32
    assert all(row["lower_bound"] > study.GATE_LIMIT for row in intervals)
    assert all(row["status"] == "excluded" for row in intervals)


def _phase_recovery_fixture(tmp_path):
    reference = tmp_path / "reference"
    recovery = reference / "phase_solver_recovery_01"
    recovery.mkdir(parents=True)
    response = torch.tensor([[1.0 + 0.2j, 0.3 - 0.1j],
                             [0.2 - 0.4j, 0.8 + 0.1j]], dtype=torch.complex128)
    target = torch.tensor([0.3 + 0.4j, 0.2 - 0.5j], dtype=torch.complex128)
    phase = 0.27
    lipschitz = 2.0
    state3 = []
    study._fista_fixed_phase_solve(response, target, phase, max_steps=3,
                                   lipschitz=lipschitz, checkpoint=state3.append,
                                   checkpoint_interval=3)
    assert state3[0]["iteration"] == 3
    torch.save({"schema": f"{study.TASK_ID}.phase_solver_recovery.v1",
                "task_id": study.TASK_ID, "target_index": 4, "proposal_id": 14,
                "target_coefficients": target, "population_ids": torch.arange(10, 15),
                "phase_rad": phase, "lipschitz": lipschitz,
                "solver_state": state3[0], "rng_state": {"cpu": torch.get_rng_state()}},
               recovery / "checkpoint_step_000003.pt")
    step4 = []
    result = study._fista_fixed_phase_solve(response, target, phase, max_steps=4,
                                            lipschitz=lipschitz, initial_state=state3[0],
                                            checkpoint=step4.append, checkpoint_interval=4)
    payload = {"schema": f"{study.TASK_ID}.phase_solver_recovery_step4.v1",
               "target_index": 4, "proposal_id": 14, "phase_rad": phase,
               "cursor": {"target_index": 4, "phase_rad": phase, "iteration": 4},
               "solver_state": step4[0], **{key: result[key] for key in
                  ("relative_error", "relative_upper_bound", "relative_dual_lower_bound", "relative_gap")}}
    torch.save(payload, recovery / "direct_step_000004.pt")
    torch.save(copy.deepcopy(payload), recovery / "independent_step_000004.pt")
    summary = {"task_id": study.TASK_ID, "status": "PHASE_SOLVER_RECOVERY_PASSED",
               "target_index": 4, "proposal_id": 14, "checkpoint_iteration": 3,
               "direct_iteration": 4, "independent_iteration": 4,
               "direct_vs_independent_state_max_abs_delta": 0.0,
               "direct_vs_independent_loss_abs_delta": 0.0,
               "cursor_exact": True, "rng_unchanged_during_save": True,
               "solver_calls_executed": 5, "adam_calls_executed": 0,
               "predecessor_positive_rows_reconsumed": 4}
    (reference / "phase_solver_recovery_summary.json").write_text(
        json.dumps(summary), encoding="utf-8")
    (recovery / "phase_solver_recovery_summary.json").write_text(
        json.dumps(summary), encoding="utf-8")
    population = {"train": {"coefficients": torch.stack([target] * 5),
                            "proposal_ids": torch.arange(10, 15)}}
    return reference, recovery, population, response


def test_phase_solver_recovery_consumer_recomputes_step_four_and_rejects_corruption(
    tmp_path,
) -> None:
    reference, recovery, population, response = _phase_recovery_fixture(tmp_path)
    verified = study._verify_phase_solver_recovery(reference, population, response)
    assert verified["status"] == "PHASE_SOLVER_RECOVERY_VERIFIED"
    assert verified["third_to_fourth_step_recomputed"] is True

    independent_path = recovery / "independent_step_000004.pt"
    independent = torch.load(independent_path, map_location="cpu", weights_only=False)
    independent["solver_state"]["amplitude"][0] += 0.1
    torch.save(independent, independent_path)
    with pytest.raises(RuntimeError, match="independent fourth-step state differs"):
        study._verify_phase_solver_recovery(reference, population, response)


def test_qualification_transfer_consumer_checks_receipt_and_current_file_sizes(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path
    runs_dir = project_root / "runs"
    monkeypatch.setattr(study, "ROOT", project_root)
    monkeypatch.setattr(study, "RUNS_DIR", runs_dir)
    evidence = runs_dir / "autodl_backup_reference"
    evidence.mkdir(parents=True)
    required = {
        "reference_qualification_summary.json", "training_witnesses.pt",
        "training_witness_failure.json", "reference_qualification_failure.json",
        "reference_solver_checkpoint.pt", "reference_progress.json", "progress.json",
        "resource_telemetry.csv", "predecessor_witness_replay_revision_01.pt",
        "phase_solver_recovery_summary.json",
        "phase_solver_recovery_01/checkpoint_step_000003.pt",
        "phase_solver_recovery_01/direct_step_000004.pt",
        "phase_solver_recovery_01/independent_step_000004.pt",
        "phase_solver_recovery_01/phase_solver_recovery_summary.json",
        "phase_solver_recovery_01/run_config.yaml",
        "engineering_positive_controls/revision_01/positive_control_replay.pt",
        "engineering_positive_controls/revision_01/positive_control_summary.json",
        *(f"witness_rows/revision_0001/row_{index:04d}.pt" for index in range(5)),
    }
    rows = []
    for relative in sorted(required):
        path = evidence / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))
        size = path.stat().st_size
        rows.append({"path": relative, "status": "COPIED", "scp_exit_code": 0,
                     "remote_size_bytes": size, "local_size_bytes": size})
    receipt = {"status": "COMPLETE", "remote_alias": "autodl-monthly",
               "remote_run_dir": "/root/autodl-tmp/codex-original/"
               + runs_dir.relative_to(project_root).as_posix() + "/"
               + evidence.name[len("autodl_backup_"):],
               "destination": str(evidence.resolve()), "files": rows}
    (evidence / "transfer_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")

    result = study._verify_qualification_transfer(evidence, evidence)
    assert result["files_verified"] == len(required)
    assert result["remote_and_local_sizes_match"] is True

    truncated = evidence / "training_witnesses.pt"
    truncated.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="transfer size or status"):
        study._verify_qualification_transfer(evidence, evidence)


def test_fixed_phase_complex_gradient_matches_independent_finite_difference() -> None:
    response_np = np.array([[1.0 + 0.4j, 0.3 - 0.2j],
                            [0.2 + 0.1j, -0.5 + 0.7j]], dtype=np.complex128)
    target_np = np.array([0.4 - 0.2j, 0.1 + 0.8j], dtype=np.complex128)
    amplitude_np = np.array([0.4, 0.6], dtype=np.float64)
    phase = 0.23
    response = torch.from_numpy(response_np)
    target = torch.from_numpy(target_np)
    amplitude = torch.from_numpy(amplitude_np)
    objective, gradient = study._fixed_phase_objective_gradient(response, target, phase, amplitude)

    def numpy_objective(values: np.ndarray) -> float:
        desired = np.exp(1j * phase) * target_np
        residual = values @ response_np.T - desired
        return float(0.5 * np.sum(np.abs(residual) ** 2))

    finite_difference = np.empty_like(amplitude_np)
    epsilon = 1.0e-6
    for index in range(len(amplitude_np)):
        plus = amplitude_np.copy()
        minus = amplitude_np.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        finite_difference[index] = (numpy_objective(plus) - numpy_objective(minus)) / (2 * epsilon)
    assert float(objective) == pytest.approx(numpy_objective(amplitude_np), abs=1.0e-14)
    assert np.allclose(gradient.numpy(), finite_difference, atol=1.0e-9, rtol=1.0e-9)
    descended = amplitude - 1.0e-4 * gradient
    next_value, _ = study._fixed_phase_objective_gradient(response, target, phase, descended)
    assert float(next_value) < float(objective)


def test_continuous_phase_certificate_rejects_coverage_and_dual_corruption() -> None:
    response, target, certificate = _narrow_phase_search_fixture()
    gap = copy.deepcopy(certificate)
    gap["intervals"].pop()
    with pytest.raises(RuntimeError, match="interval coverage"):
        study._verify_phase_interval_certificate(response, target, gap)
    wrong_direction = copy.deepcopy(certificate)
    wrong_direction["directions"][0]["direction"] *= 0.5
    with pytest.raises(RuntimeError, match="dual direction"):
        study._verify_phase_interval_certificate(response, target, wrong_direction)


def test_ten_positive_control_rows_use_local_single_row_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []

    def error_row(_outputs, targets, _direct, index):
        calls.append((len(targets), index))
        assert len(targets) == 1
        return {key: 0.0 for key in study.METRIC_KEYS}

    monkeypatch.setattr(study.projected, "_error_row", error_row)
    targets = torch.zeros((10, 100), dtype=torch.complex128)
    actual = {grid: {"coefficients": targets.clone(),
                     "raw_cmos": torch.ones((10, 1, 200, 200), dtype=torch.float64)}
              for grid in study.GRIDS}
    direct = {grid: {"raw_cmos": torch.ones((10, 1, 200, 200), dtype=torch.float64)}
              for grid in study.GRIDS}
    rows = study._positive_control_error_rows(actual, direct, targets)
    assert len(rows) == 10
    assert calls == [(1, 0)] * 10


def test_predecessor_replay_consumer_recomputes_five_rows_and_rejects_cmos_swap(
    tmp_path,
) -> None:
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    response = torch.zeros((100, 40_000), dtype=torch.complex128)
    response[0, 0] = 1.0
    target = torch.zeros((5, 100), dtype=torch.complex128)
    target[:, 0] = 1.0
    rows = []
    for index in range(5):
        command = torch.zeros((1, 1, 200, 200), dtype=torch.float32)
        if index < 4:
            command[0, 0, 0, 0] = 1.0
        coefficient = target[index:index + 1].clone() if index < 4 else torch.zeros((1, 100),
                                                                                   dtype=torch.complex128)
        raw = torch.ones((1, 1, 200, 200), dtype=torch.float64)
        outputs = {grid: {"coefficients": coefficient.clone(), "raw_cmos": raw.clone()}
                   for grid in study.GRIDS}
        if index == 4:
            outputs[4096]["coefficients"] = target[index:index + 1].clone()
        metrics = study.projected._error_row(
            outputs, target[index:index + 1], {grid: raw.clone() for grid in study.GRIDS}, 0)
        rows.append({"index": index, "label": index, "proposal_id": 100 + index,
                     "source_index": index, "target_coefficients": target[index:index + 1].clone(),
                     "amplitude": command, "outputs": outputs,
                     "direct_raw_cmos": {grid: raw.clone() for grid in study.GRIDS},
                     "metrics": metrics, "all_six_metrics_passed": index < 4,
                     "matrix_response_relative_error": 0.0,
                     "full_optics_replay_relative_errors": {
                         f"{grid}_{key}": 0.0 for grid in study.GRIDS
                         for key in ("coefficients", "raw_cmos", "direct_raw_cmos")}})
    payload = {"schema": f"{study.TASK_ID}.predecessor_replay.v1", "task_id": study.TASK_ID,
               "population_ids": torch.arange(100, 105), "positive_rows": rows[:4],
               "failed_row": rows[4], "full_optics_replayed": True, "scientific_updates": 0}
    path = reference_dir / "predecessor_witness_replay_revision_01.pt"
    torch.save(payload, path)
    population = {"train": {"coefficients": target, "labels": torch.arange(5),
                            "proposal_ids": torch.arange(100, 105),
                            "source_indices": torch.arange(5)}}
    predecessor_source = {"witnesses": [{"amplitude": row["amplitude"]} for row in rows],
                          "failed_indices": [{"index": 4}]}
    assert study._verify_predecessor_replay_revisions(
        reference_dir, {"response_matrix": response,
                        "prior_training_witnesses": predecessor_source}, population) == 1
    payload["positive_rows"][0]["outputs"][2048]["raw_cmos"].mul_(2.0)
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="metrics differ from independent NumPy"):
        study._verify_predecessor_replay_revisions(
            reference_dir, {"response_matrix": response,
                            "prior_training_witnesses": predecessor_source}, population)


def test_reference_consumer_recomputes_the_response_of_the_saved_command(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(study, "TRAIN_ROWS", 1)
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    target = torch.zeros((1, 100), dtype=torch.complex128)
    target[0, 0] = 1.0 + 0.0j
    command = torch.zeros((1, 1, 200, 200), dtype=torch.float32)
    command[0, 0, 0, 0] = 0.5
    response = torch.zeros((100, 40_000), dtype=torch.complex128)
    response[0, 0] = 2.0 + 0.0j
    raw = torch.ones((1, 1, 200, 200), dtype=torch.float64)
    outputs = {grid: {"coefficients": target.clone(), "raw_cmos": raw.clone()}
               for grid in study.GRIDS}
    direct = {grid: raw.clone() for grid in study.GRIDS}
    metrics = study.projected._error_row(outputs, target, direct, 0)
    row = {"index": 0, "label": 0, "proposal_id": 17, "source_index": 0,
           "amplitude": command, "outputs": outputs, "direct_raw_cmos": direct,
           "metrics": metrics, "all_six_metrics_passed": True}
    torch.save({"schema": f"{study.TASK_ID}.witnesses.v1", "task_id": study.TASK_ID,
                "population_ids": torch.tensor([17]), "witnesses": [row], "failed_indices": []},
               reference_dir / "training_witnesses.pt")
    (reference_dir / "reference_qualification_summary.json").write_text(
        json.dumps({"status": "ALL_200_TRAINING_WITNESSES_PASSED", "count": 1,
                    "all_six_metrics_qualified": True, "solver_1e-6_is_early_stop_only": True}),
        encoding="utf-8")
    inputs = {"response_matrix": response}
    population = {"train": {"coefficients": target, "proposal_ids": torch.tensor([17]),
                            "labels": torch.tensor([0]), "source_indices": torch.tensor([0])}}
    class OpticalStub(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("sentinel", torch.zeros(1))

    monkeypatch.setattr(study, "_make_runtime_model", lambda _inputs, grid, _device: OpticalStub())

    def forward_amplitude(_model, candidate: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"coefficients": study._forward_matrix(candidate, response),
                "raw_cmos": raw.clone()}

    monkeypatch.setattr(study.projected, "_forward_amplitude", forward_amplitude)
    monkeypatch.setattr(study.projected, "_forward_target",
                        lambda _model, _target: {"raw_cmos": raw.clone()})
    consumed, _ = study._consume_training_witnesses(
        reference_dir, inputs, population, device=torch.device("cpu"))
    assert consumed[0]["matrix_response_relative_error"] == 0.0
    population["train"]["coefficients"][0, 1] = 1.0 + 0.0j
    with pytest.raises(RuntimeError, match="six-metric reconsumption"):
        study._consume_training_witnesses(reference_dir, inputs, population,
                                          device=torch.device("cpu"))
    saved = torch.load(reference_dir / "training_witnesses.pt", map_location="cpu", weights_only=False)
    saved["witnesses"][0]["amplitude"].fill_(0.4)
    torch.save(saved, reference_dir / "training_witnesses.pt")
    with pytest.raises(RuntimeError, match="does not replay from its command"):
        study._consume_training_witnesses(reference_dir, inputs, population,
                                           device=torch.device("cpu"))


def test_reference_consumer_rejects_cmos_output_that_does_not_replay_from_command(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(study, "TRAIN_ROWS", 1)
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    target = torch.zeros((1, 100), dtype=torch.complex128)
    target[0, 0] = 1.0 + 0.0j
    command = torch.zeros((1, 1, 200, 200), dtype=torch.float32)
    command[0, 0, 0, 0] = 0.5
    response = torch.zeros((100, 40_000), dtype=torch.complex128)
    response[0, 0] = 2.0 + 0.0j
    raw = torch.ones((1, 1, 200, 200), dtype=torch.float64)
    outputs = {grid: {"coefficients": target.clone(), "raw_cmos": raw.clone()}
               for grid in study.GRIDS}
    direct = {grid: raw.clone() for grid in study.GRIDS}
    metrics = study.projected._error_row(outputs, target, direct, 0)
    row = {"index": 0, "label": 0, "proposal_id": 17, "source_index": 0,
           "amplitude": command, "outputs": outputs, "direct_raw_cmos": direct,
           "metrics": metrics, "all_six_metrics_passed": True}
    torch.save({"schema": f"{study.TASK_ID}.witnesses.v1", "task_id": study.TASK_ID,
                "population_ids": torch.tensor([17]), "witnesses": [row], "failed_indices": []},
               reference_dir / "training_witnesses.pt")
    (reference_dir / "reference_qualification_summary.json").write_text(
        json.dumps({"status": "ALL_200_TRAINING_WITNESSES_PASSED", "count": 1,
                    "all_six_metrics_qualified": True, "solver_1e-6_is_early_stop_only": True}),
        encoding="utf-8")

    class OpticalStub(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("sentinel", torch.zeros(1))

    monkeypatch.setattr(study, "_make_runtime_model", lambda _inputs, _grid, _device: OpticalStub())
    monkeypatch.setattr(study.projected, "_forward_amplitude",
                        lambda _model, _command: {"coefficients": target.clone(),
                                                  "raw_cmos": torch.full_like(raw, 2.0)})
    monkeypatch.setattr(study.projected, "_forward_target",
                        lambda _model, _target: {"raw_cmos": raw.clone()})
    inputs = {"response_matrix": response}
    population = {"train": {"coefficients": target, "proposal_ids": torch.tensor([17]),
                            "labels": torch.tensor([0]), "source_indices": torch.tensor([0])}}
    with pytest.raises(RuntimeError, match="saved full-optics output does not replay"):
        study._consume_training_witnesses(reference_dir, inputs, population,
                                          device=torch.device("cpu"))


def test_incomplete_training_reference_population_cannot_enter_ready_consumer(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(study, "TRAIN_ROWS", 2)
    reference_dir = tmp_path / "incomplete_reference"
    reference_dir.mkdir()
    (reference_dir / "reference_qualification_summary.json").write_text(
        json.dumps({"status": "ALL_200_TRAINING_WITNESSES_PASSED", "count": 1,
                    "all_six_metrics_qualified": True, "solver_1e-6_is_early_stop_only": True}),
        encoding="utf-8")
    torch.save({"schema": f"{study.TASK_ID}.witnesses.v1", "task_id": study.TASK_ID,
                "population_ids": torch.tensor([17, 18]), "witnesses": [{}], "failed_indices": []},
               reference_dir / "training_witnesses.pt")
    population = {"train": {"proposal_ids": torch.tensor([17, 18])}}
    with pytest.raises(RuntimeError, match="summary or population is incomplete"):
        study._consume_training_witnesses(reference_dir, {}, population, device=torch.device("cpu"))


def test_semantic_summary_consumes_ec_logits_from_tuple_output() -> None:
    labels = torch.arange(10).repeat_interleave(2)
    raw = torch.arange(20, dtype=torch.float64).reshape(20, 1, 1, 1).expand(20, 1, 200, 200) + 1.0
    inputs = {"camera_gain": 1.0e11,
              "evaluator_ed": PixelMLPConfirmationEvaluatorV1().eval(),
              "evaluator_ec": ThirdConfirmationEvaluator().eval()}
    result, readouts = study._semantic_summary(
        inputs, {"labels": labels}, {"mapped_candidate": raw})
    assert result["conditions"]["mapped_candidate"]["evaluators"]["ED"]["accuracy"] >= 0.0
    assert result["conditions"]["mapped_candidate"]["evaluators"]["EC"]["accuracy"] >= 0.0
    assert readouts["mapped_candidate"].shape == (20, 1, 28, 28)
    assert result["human_readability"] == "not_measured"


def test_semantic_summary_reports_fixed_sensor_conversion_and_paired_changes() -> None:
    labels = torch.arange(10).repeat_interleave(2)
    base = torch.arange(20, dtype=torch.float64).reshape(20, 1, 1, 1) + 1.0
    direct = base.expand(20, 1, 200, 200).clone()
    candidate = direct + 1.0
    inputs = {"camera_gain": 1.0e11,
              "evaluator_ed": PixelMLPConfirmationEvaluatorV1().eval(),
              "evaluator_ec": ThirdConfirmationEvaluator().eval()}
    result, _ = study._semantic_summary(
        inputs, {"labels": labels}, {"direct_decoder": direct, "candidate": candidate})
    direct_sensor = result["conditions"]["direct_decoder"]["sensor_statistics"]
    candidate_sensor = result["conditions"]["candidate"]["sensor_statistics"]
    assert direct_sensor["raw_w_per_m2_mean"] == pytest.approx(10.5)
    assert candidate_sensor["raw_w_per_m2_mean"] == pytest.approx(11.5)
    assert direct_sensor["fraction_above_one"] > 0.0
    assert result["paired_semantic_changes"]["candidate"]["ED"]["overall"]["count"] == 20
    assert result["paired_semantic_changes"]["candidate"]["EC"]["overall"]["prediction_changed"] >= 0


def _mock_measurement_path(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict, dict, torch.nn.Module, list[tuple[int, ...]]]:
    calls: list[tuple[int, ...]] = []
    monkeypatch.setattr(study.projected, "_normalized_features",
                        lambda coefficients, _scale, _device: torch.zeros((len(coefficients), 200)))

    def map_one(_mapper, features, _target, _matrix, _inverse, _scale, *, include_preclip=False):
        calls.append(tuple(features.shape))
        command = torch.full((len(features), 1, 200, 200), 0.4, dtype=torch.float32)
        details = {"clipped_fraction": torch.zeros(len(features), dtype=torch.float64)}
        return command, details

    def forward(_model, command):
        return {"coefficients": torch.zeros((len(command), 100), dtype=torch.complex128),
                "raw_cmos": torch.zeros((len(command), 1, 200, 200), dtype=torch.float64)}

    monkeypatch.setattr(study.projected, "_map_amplitude", map_one)
    monkeypatch.setattr(study, "_make_runtime_model", lambda _inputs, grid, _device: grid)
    monkeypatch.setattr(study.projected, "_forward_amplitude", forward)
    monkeypatch.setattr(study.projected, "_forward_target",
                        lambda _model, target: {"raw_cmos": torch.zeros((len(target), 1, 200, 200),
                                                                            dtype=torch.float64)})
    monkeypatch.setattr(study.projected, "_error_row",
                        lambda *_args: {key: 0.0 for key in study.METRIC_KEYS})
    inputs = {"scale": torch.ones(100), "matrix": torch.empty(0), "inverse": torch.empty(0)}
    population = {"role": "fixed_train200", "coefficients": torch.zeros((1, 100), dtype=torch.complex128),
                  "labels": torch.tensor([0]), "proposal_ids": torch.tensor([5]),
                  "source_indices": torch.tensor([0])}
    return inputs, population, torch.nn.Identity(), calls


def test_measurement_generates_each_terminal_slm_command_from_one_target(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, population, mapper, calls = _mock_measurement_path(monkeypatch)
    monkeypatch.setattr(study, "ATTEMPT_DIR", tmp_path / "attempt")
    _, outputs = study._measure_cohort(inputs, population, mapper, torch.device("cpu"),
                                       save_outputs=True, tag="train")
    assert calls == [(1, 200)]
    assert outputs[2048]["coefficients"].shape == (1, 100)
    assert (tmp_path / "attempt" / "measurements" / "train" / "shard_0000.pt").is_file()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the production witness replay is CUDA-only")
def test_training_witness_replay_moves_solver_command_and_uses_local_row_index(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:0")
    target = torch.ones((1, 100), dtype=torch.complex128)

    class OpticalStub(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("sentinel", torch.zeros(1))

    class MapperStub(torch.nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return torch.zeros((len(features), 1, 200, 200), device=features.device)

    monkeypatch.setattr(study, "TRAIN_ROWS", 1)
    monkeypatch.setattr(study, "_mapper", lambda _inputs, _device: MapperStub())
    monkeypatch.setattr(study, "_make_runtime_model", lambda _inputs, _grid, _device: OpticalStub())
    monkeypatch.setattr(study.projected, "_normalized_features",
                        lambda _target, _scale, _device: torch.zeros((1, 200), device=device))
    monkeypatch.setattr(study.projected, "_project_double_from_raw",
                        lambda *_args, **_kwargs: (None, {"phase": torch.zeros(1, device=device)}))
    monkeypatch.setattr(study.feasibility, "_fixed_phase_solve",
                        lambda *_args, **_kwargs: {
                            "amplitude": torch.full((40_000,), 0.5, dtype=torch.float64),
                            "relative_error": 0.0, "iterations": 1, "phase_rad": 0.0,
                        })

    def forward_amplitude(_model, amplitude: torch.Tensor) -> dict[str, torch.Tensor]:
        assert amplitude.device == device
        return {"coefficients": target.to(device),
                "raw_cmos": torch.ones((1, 1, 200, 200), dtype=torch.float64, device=device)}

    monkeypatch.setattr(study.projected, "_forward_amplitude", forward_amplitude)
    monkeypatch.setattr(study.projected, "_forward_target",
                        lambda _model, _target: {
                            "raw_cmos": torch.ones((1, 1, 200, 200), dtype=torch.float64, device=device)})
    population = {"train": {"coefficients": target, "proposal_ids": torch.tensor([77])}}
    population["train"]["source_indices"] = torch.tensor([0])
    inputs = {"response_matrix": torch.empty((0, 0), dtype=torch.complex128),
              "matrix": torch.empty((0, 0), dtype=torch.float64),
              "inverse": torch.empty((0, 0), dtype=torch.float64),
              "scale": torch.ones(100, dtype=torch.float64)}
    result = study._solver_qualification({}, inputs, population, device)
    assert result["count"] == 1
    assert result["all_six_metrics_qualified"] is True


@pytest.mark.parametrize("corruption", ("target", "command"))
def test_measurement_resume_rejects_saved_target_or_command_replacement(
    tmp_path, monkeypatch: pytest.MonkeyPatch, corruption: str,
) -> None:
    inputs, population, mapper, _calls = _mock_measurement_path(monkeypatch)
    attempt = tmp_path / "attempt"
    monkeypatch.setattr(study, "ATTEMPT_DIR", attempt)
    study._measure_cohort(inputs, population, mapper, torch.device("cpu"),
                          save_outputs=True, tag="train")
    path = attempt / "measurements" / "train" / "shard_0000.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if corruption == "target":
        saved["target_coefficients"][0, 0] = 1.0 + 0.0j
    else:
        saved["commands"].fill_(0.16)
    torch.save(saved, path)
    with pytest.raises(RuntimeError, match="single-sample mapper"):
        study._measure_cohort(inputs, population, mapper, torch.device("cpu"),
                              save_outputs=True, tag="train")


def test_transfer_consumer_rejects_a_truncated_file(tmp_path) -> None:
    payload = tmp_path / "evidence.bin"
    payload.write_bytes(b"12345678")
    run_id = "run-1"
    (tmp_path / "final_decision.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    manifest = {"schema": f"{study.TASK_ID}.transfer.v1", "task_id": study.TASK_ID,
                "run_id": run_id, "files": [{"path": "evidence.bin", "bytes": 8}],
                "file_count": 1, "bytes_total": 8}
    manifest_path = tmp_path / "transfer_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_size = manifest_path.stat().st_size
    receipt = {
        "status": "COMPLETE", "remote_alias": "autodl-monthly",
        "remote_run_dir": "/root/autodl-tmp/codex-original/outputs/20261029_"
        "lightgen_g1_semantic_exposure200_v1/runs/attempt_01",
        "destination": str(tmp_path.resolve()),
        "files": [
            {"path": "evidence.bin", "status": "COPIED", "scp_exit_code": 0,
             "remote_size_bytes": 8, "local_size_bytes": 8},
            {"path": "transfer_manifest.json", "status": "COPIED", "scp_exit_code": 0,
             "remote_size_bytes": manifest_size, "local_size_bytes": manifest_size},
        ],
    }
    (tmp_path / "transfer_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert study._verify_transfer_sizes(tmp_path, require_transfer_receipt=True)["files_verified"] == 2
    payload.write_bytes(b"1234")
    with pytest.raises(RuntimeError, match="actual file size mismatch"):
        study._verify_transfer_sizes(tmp_path, require_transfer_receipt=True)


def test_transfer_consumer_rejects_path_traversal(tmp_path) -> None:
    manifest = {"schema": f"{study.TASK_ID}.transfer.v1", "task_id": study.TASK_ID,
                "run_id": "run-1", "files": [{"path": "../outside.bin", "bytes": 1}],
                "file_count": 1, "bytes_total": 1}
    (tmp_path / "transfer_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "final_decision.json").write_text(json.dumps({"run_id": "run-1"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unsafe or duplicate relative path"):
        study._verify_transfer_sizes(tmp_path)
