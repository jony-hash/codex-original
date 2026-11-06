from __future__ import annotations

import pytest
import torch

from scripts.research.lightgen import execute_chen_lightgen_400_q12_trainability_v1 as runner


def test_q12_status_requires_every_measurement_and_uses_executed_residuals() -> None:
    assert runner._status({"a": 1.0e-5}, {"a", "b"}, runner.FIELD_TOLERANCE) == "INVALID_HARNESS"
    assert runner._status({"a": 1.0e-3, "b": 1.0e-5}, {"a", "b"},
                          runner.FIELD_TOLERANCE) == "NUMERICAL_NOT_QUALIFIED"
    assert runner._status({"a": 1.0e-5, "b": 2.0e-5}, {"a", "b"},
                          runner.FIELD_TOLERANCE) == "NUMERICALLY_QUALIFIED"
    assert runner._status({"a": float("nan")}, {"a"},
                          runner.FIELD_TOLERANCE) == "INVALID_HARNESS"


def test_node_field_comparison_uses_one_global_phase_for_all_planes() -> None:
    reference = tuple(torch.arange(48, dtype=torch.float64).reshape(1, 4, 3, 4).to(torch.complex128)
                      + (plane + 1j) for plane in range(7))
    shared_phase = torch.exp(torch.tensor(0.71j, dtype=torch.complex128))
    aligned = runner._compare_node_fields(tuple(value * shared_phase for value in reference), reference)
    assert aligned["max_node_field_relative_l2"] < 1.0e-14

    per_plane = list(reference)
    per_plane[-1] = per_plane[-1] * torch.exp(torch.tensor(0.2j, dtype=torch.complex128))
    mixed = runner._compare_node_fields(tuple(per_plane), reference)
    assert mixed["max_node_field_relative_l2"] > 1.0e-3
    assert len(mixed["per_plane_node_field_relative_l2"]) == 7


def test_q12_entry_rejects_scientific_run_mode() -> None:
    with pytest.raises(SystemExit):
        runner._parse_args(["--mode", "run", "--propagation-backend", "rs_linear"])


def test_q12_resume_requires_explicit_checkpoint() -> None:
    explicit_paths = [
        "--source-checkpoint", "source.pt", "--source-report", "report.json",
        "--source-start-state", "start.pt", "--source-endpoint-checkpoint", "end.pt",
        "--fixed-population", "population.pt", "--config", "config.yaml",
        "--data", "training.pt", "--evaluator", "evaluator.pt",
        "--qualification-report", "closeout.md",
        "--qualification-cache", "cache", "--output-dir", "preflight_01",
    ]
    with pytest.raises(runner.HarnessCheckError):
        runner._parse_args(["--mode", "resume", "--propagation-backend", "rs_linear", *explicit_paths])


def test_q12_preflight_checkpoint_roundtrips_completed_stages(tmp_path) -> None:
    path = tmp_path / "preflight_state.pt"
    inputs = {"source_checkpoint": "explicit.pt"}
    completed = {"independent_direct_rs"}
    runner._write_checkpoint(path, inputs, completed, "independent_direct_rs")
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["task_id"] == runner.TASK_ID
    assert state["explicit_inputs"] == inputs
    assert state["completed_stages"] == ["independent_direct_rs"]
    assert state["stage"] == "independent_direct_rs"


def test_population_validation_reads_saved_rows_and_canonical_labels():
    labels = torch.arange(10).repeat_interleave(30)
    fit_rows = torch.cat([torch.arange(label * 30, label * 30 + 10) for label in range(10)])
    confirm_rows = torch.cat([torch.arange(label * 30 + 10, (label + 1) * 30) for label in range(10)])
    population = {"fit_rows": fit_rows, "confirm_rows": confirm_rows}
    runner._validate_population_labels(population, labels)

    wrong_population = {"fit_rows": fit_rows, "confirm_rows": confirm_rows.clone()}
    wrong_population["confirm_rows"][[0, 20]] = wrong_population["confirm_rows"][[20, 0]]
    with pytest.raises(runner.HarnessCheckError, match="class order"):
        runner._validate_population_labels(wrong_population, labels)


def test_single_pixel_reference_reports_gauss_convergence():
    reference = {order: runner._single_pixel_rs_integral(order, (0.0, 200.0), 0.0)
                for order in (24, 32, 48)}
    error = abs(reference[32] - reference[48]) / abs(reference[48])
    assert error < runner.FIELD_TOLERANCE


def test_represented_phase_delta_reports_actual_float32_displacement():
    phases = {
        "encoder_phase": torch.zeros((2, 1, 3, 3), dtype=torch.float32),
        "decoder_phase": torch.zeros((4, 1, 3, 3), dtype=torch.float32),
    }
    directions = [torch.zeros((3, 3), dtype=torch.float64) for _ in range(6)]
    directions[0].fill_(1.0 / 3.0)
    perturbed, projection, delta_l2 = runner._apply_represented_phase_delta(
        phases, directions, 1.25e-3, 1.0, torch.device("cpu")
    )
    assert torch.equal(phases["encoder_phase"], torch.zeros_like(phases["encoder_phase"]))
    assert torch.count_nonzero(perturbed["encoder_phase"][0, 0]) == 9
    assert projection > 0.0
    assert delta_l2 > 0.0
    assert abs(delta_l2 - 1.25e-3) > 1.0e-12


@pytest.mark.parametrize("rejected_order", [None, 16])
def test_cache_replay_requires_q12_and_q16_same_sample(monkeypatch, tmp_path, rejected_order):
    cache_run_dir = tmp_path / "source_run"
    cache_dir = cache_run_dir / "qualification_raw"
    cache_dir.mkdir(parents=True)
    rows = torch.arange(10, dtype=torch.long)
    for state in runner.STATES:
        for label, row_tensor in enumerate(rows):
            row = int(row_tensor)
            for order in (runner.ORDER, runner.REFERENCE_ORDER):
                if order == runner.REFERENCE_ORDER and label >= 2:
                    continue
                path = runner.prior._raw_observation_path(cache_run_dir, "reference", state, row, order)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

    loaded_from = []
    monkeypatch.setattr(
        runner.prior, "_load_full_chain_raw",
        lambda run_dir, _stage, state, row, label, order: (
            loaded_from.append(run_dir) or {
                "state": state, "row": row, "label": label, "order": order,
                "result": {"order": order},
            }
        ),
    )
    replayed_orders = []

    def replay(phases, row, label, pixels, order, device):
        replayed_orders.append(order)
        return {"state": phases, "row": row, "label": label, "result": {"order": order}}, {}

    def residual(candidate, _reference):
        order = candidate["result"]["order"]
        field_error = 2.0e-4 if order == rejected_order else 0.0
        return {"per_plane_field_relative_l2": {"plane_1": field_error},
                "ols_complex_relative_l2": 0.0,
                "camera_roi_w_per_pixel_relative_l2": 0.0}

    monkeypatch.setattr(runner.prior, "_full_chain_comparison", replay)
    monkeypatch.setattr(runner.prior, "_full_chain_residual", residual)
    context = {
        "paths": {"qualification_cache": cache_dir},
        "population": {"confirm_rows_by_class": rows[:, None]},
        "state_phases": {state: {} for state in runner.STATES},
        "pixels": torch.zeros((10, 1, 28, 28)),
    }
    result = runner._validate_cache_replay(context, tmp_path / "out", torch.device("cpu"))
    assert result["replayed_orders"] == [runner.ORDER, runner.REFERENCE_ORDER]
    assert result["actual_cache_counts"] == {"12": 20, "16": 4}
    assert loaded_from and all(path == cache_run_dir for path in loaded_from)
    if rejected_order is None:
        assert result["cache_accepted"]
    else:
        assert not result["cache_accepted"]
    assert replayed_orders == [runner.ORDER, runner.REFERENCE_ORDER]


def test_raw_cache_import_reads_explicit_qualification_raw_directory(monkeypatch, tmp_path):
    cache_run_dir = tmp_path / "source_run"
    cache_raw_dir = cache_run_dir / "qualification_raw"
    cache_raw_dir.mkdir(parents=True)
    cached_path = runner.prior._raw_observation_path(
        cache_run_dir, "reference", runner.STATES[0], 51301, runner.ORDER
    )
    cached_path.touch()
    loaded_dirs = []

    def load_raw(run_dir, stage, state, row, label, order):
        loaded_dirs.append(run_dir)
        return {"state": state, "row": row, "label": label, "order": order,
                "result": {"finite_pixel_plane_values": [], "ols_coefficients": torch.empty(0),
                           "camera_roi_w_per_pixel": torch.empty(0),
                           "physical_w_per_pixel": torch.empty(0),
                           "plane_powers_w": torch.empty(0), "input_power_w": torch.empty(0)}}

    saved = []

    def save_raw(out_dir, stage, state, comparison, order):
        saved.append((out_dir, stage, state, comparison["row"], order))

    monkeypatch.setattr(runner.prior, "_load_full_chain_raw", load_raw)
    monkeypatch.setattr(runner.prior, "_save_full_chain_raw", save_raw)
    result = runner._raw_full_chain(
        tmp_path / "destination", "candidate", runner.STATES[0], 51301, 0, runner.ORDER,
        None, torch.zeros((1, 1, 28, 28)), torch.device("cpu"), cache_raw_dir,
    )

    assert loaded_dirs == [cache_run_dir, tmp_path / "destination"]
    assert saved == [(tmp_path / "destination", "candidate", runner.STATES[0], 51301,
                      runner.ORDER)]
    assert result["row"] == 51301


def test_qualification_cache_requires_raw_directory(tmp_path):
    wrong_dir = tmp_path / "source_run"
    wrong_dir.mkdir()
    with pytest.raises(runner.HarnessCheckError, match="qualification_raw"):
        runner._qualification_cache_run_dir(wrong_dir)


def test_full_chain_cache_source_counts_use_raw_directory(monkeypatch, tmp_path):
    cache_run_dir = tmp_path / "source_run"
    cache_raw_dir = cache_run_dir / "qualification_raw"
    cache_raw_dir.mkdir(parents=True)
    output_dir = tmp_path / "output_run"
    rows = torch.arange(10, dtype=torch.long)
    for state in runner.STATES:
        for label, row_tensor in enumerate(rows):
            row = int(row_tensor)
            path = runner.prior._raw_observation_path(
                cache_run_dir, "reference", state, row, runner.ORDER
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            if label < 2:
                reference_path = runner.prior._raw_observation_path(
                    cache_run_dir, "reference", state, row, runner.REFERENCE_ORDER
                )
                reference_path.touch()

    local_candidate = runner.prior._raw_observation_path(
        output_dir, "candidate", runner.STATES[0], int(rows[0]), runner.ORDER
    )
    local_candidate.parent.mkdir(parents=True, exist_ok=True)
    local_candidate.touch()

    monkeypatch.setattr(
        runner, "_raw_full_chain",
        lambda _out, stage, state, row, label, order, *_args: {
            "state": state, "row": row, "label": label,
            "result": {"stage": stage, "order": order},
            "raw_path": str(runner.prior._raw_observation_path(
                output_dir, stage, state, row, order
            )),
        },
    )
    monkeypatch.setattr(
        runner.prior, "_full_chain_residual",
        lambda _candidate, _reference: {
            "per_plane_field_relative_l2": {f"plane_{i}": 0.0 for i in range(7)},
            "ols_complex_relative_l2": 0.0,
            "camera_roi_w_per_pixel_relative_l2": 0.0,
        },
    )
    context = {
        "paths": {"qualification_cache": cache_raw_dir},
        "population": {"confirm_rows_by_class": rows[:, None]},
        "state_phases": {state: {} for state in runner.STATES},
        "pixels": torch.zeros((10, 1, 28, 28)),
    }

    result = runner._fit_endpoint_raws(context, output_dir, torch.device("cpu"), True)

    assert result["expected_generated_forwards"] == 16
    assert result["generated_forwards"] == 16
    assert result["source_cache_observations_reused"] == 23
    assert result["local_resume_observations_reused"] == 1


def _fd_fixture(state: str, row: int, label: int, direction: str) -> dict:
    checks = {}
    for path_name in ("q12", "q16"):
        checks[path_name] = []
        for epsilon in runner.FD_STEPS:
            derivative = 1.0
            checks[path_name].append({
                "nominal_epsilon_rad": epsilon,
                "plus_loss": 1.0 + derivative * epsilon,
                "minus_loss": 1.0 - derivative * epsilon,
                "plus_represented_directional_rad": epsilon,
                "minus_represented_directional_rad": -epsilon,
                "plus_represented_phase_delta_l2_rad": epsilon,
                "minus_represented_phase_delta_l2_rad": epsilon,
                "effective_central_step_rad": 2.0 * epsilon,
                "central_difference": derivative,
                "autodiff_directional_derivative": derivative,
                "degenerate": False,
                "same_sign": True,
                "relative_error": 0.0,
            })
    return {
        "schema": "chen_lightgen_400_q12_fd_direction.v1",
        "state": state,
        "row": row,
        "label": label,
        "direction": direction,
        "steps_rad": list(runner.FD_STEPS),
        "checks": checks,
    }


def test_finite_difference_resume_reuses_completed_direction_files(monkeypatch, tmp_path):
    state = runner.STATES[0]
    row = 51301
    label = 0
    first_direction = "layer_1"
    saved_path = tmp_path / f"finite_difference_{state}_row{row}_{first_direction}.json"
    runner._save_json(saved_path, _fd_fixture(state, row, label, first_direction), new=True)
    forwards = []

    def fake_loss(_context, _phases, _row, _order, directions, epsilon, sign, _device):
        forwards.append((epsilon, sign))
        derivative = sum(float(value.sum()) for value in directions)
        return 1.0 + derivative * sign * epsilon, sign * epsilon, epsilon

    monkeypatch.setattr(runner, "_loss_at_phases", fake_loss)
    phases = {
        "encoder_phase": torch.zeros((2, 1, 1, 1), dtype=torch.float32),
        "decoder_phase": torch.zeros((4, 1, 1, 1), dtype=torch.float32),
    }
    gradients = [torch.ones((1, 1), dtype=torch.float64) for _ in range(6)]
    result = runner._finite_difference_check(
        {"state_phases": {state: phases}}, state, row, label, gradients, gradients,
        torch.device("cpu"), {}, tmp_path,
    )

    assert result["status"] == "NUMERICALLY_QUALIFIED"
    assert result["completed_forwards"] == 48
    assert result["reused_completed_forwards"] == 8
    assert result["executed_forwards_this_resume"] == 40
    assert len(forwards) == 40
    assert saved_path.read_text(encoding="utf-8")


def test_finite_difference_progress_uses_global_direction_count(monkeypatch, tmp_path):
    first_state = runner.STATES[0]
    first_row = 51301
    for layer in range(1, 7):
        direction = f"layer_{layer}"
        runner._save_json(
            tmp_path / f"finite_difference_{first_state}_row{first_row}_{direction}.json",
            _fd_fixture(first_state, first_row, 0, direction),
            new=True,
        )

    def fake_loss(_context, _phases, _row, _order, directions, epsilon, sign, _device):
        derivative = sum(float(value.sum()) for value in directions)
        return 1.0 + derivative * sign * epsilon, sign * epsilon, epsilon

    monkeypatch.setattr(runner, "_loss_at_phases", fake_loss)
    state = runner.STATES[0]
    row = 45650
    phases = {
        "encoder_phase": torch.zeros((2, 1, 1, 1), dtype=torch.float32),
        "decoder_phase": torch.zeros((4, 1, 1, 1), dtype=torch.float32),
    }
    gradients = [torch.ones((1, 1), dtype=torch.float64) for _ in range(6)]
    progress = {}

    result = runner._finite_difference_check(
        {"state_phases": {state: phases}}, state, row, 1, gradients, gradients,
        torch.device("cpu"), progress, tmp_path,
    )

    assert result["status"] == "NUMERICALLY_QUALIFIED"
    assert progress["completed_perturbation_forwards"] == 56
    assert progress["expected_perturbation_forwards"] == 72
    assert progress["probe_completed_perturbation_forwards"] == 8
    assert progress["probe_expected_perturbation_forwards"] == 8


def test_finite_difference_resume_rejects_mismatched_saved_direction(monkeypatch, tmp_path):
    state = runner.STATES[0]
    row = 51301
    path = tmp_path / f"finite_difference_{state}_row{row}_layer_1.json"
    runner._save_json(path, _fd_fixture(state, row + 1, 0, "layer_1"), new=True)
    monkeypatch.setattr(
        runner, "_loss_at_phases",
        lambda *_args: pytest.fail("a mismatched saved direction must be rejected before any forward"),
    )
    phases = {
        "encoder_phase": torch.zeros((2, 1, 1, 1), dtype=torch.float32),
        "decoder_phase": torch.zeros((4, 1, 1, 1), dtype=torch.float32),
    }
    gradients = [torch.ones((1, 1), dtype=torch.float64) for _ in range(6)]

    with pytest.raises(runner.HarnessCheckError, match="identity differs"):
        runner._finite_difference_check(
            {"state_phases": {state: phases}}, state, row, 0, gradients, gradients,
            torch.device("cpu"), {}, tmp_path,
        )
