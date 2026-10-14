from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from scripts.research.lightgen import execute_lightgen_2d_generator_cloud_foreground_dev_multiseed_v1 as task


def _population_fixture() -> dict[str, torch.Tensor]:
    labels = torch.arange(60_000, dtype=torch.long) % 10
    train = torch.arange(55_000, dtype=torch.long)
    dev = torch.arange(55_000, 60_000, dtype=torch.long)
    old_fit = torch.arange(200, dtype=torch.long)
    old = torch.cat((torch.arange(300, dtype=torch.long), torch.arange(5_000, 5_200, dtype=torch.long)))
    ancestor = torch.arange(3_300, dtype=torch.long)
    excluded = set(ancestor.tolist()) | set(old.tolist())
    added_by_class = [train[(labels[train] == digit) & torch.tensor([int(row) not in excluded for row in train.tolist()], dtype=torch.bool)][:180] for digit in range(10)]
    added = torch.stack(added_by_class, dim=1).reshape(-1)
    fit = torch.cat((old_fit, added))
    dev_rows = torch.stack([dev[labels[dev] == digit][:100] for digit in range(10)], dim=1).reshape(-1)
    return {
        "labels": labels,
        "train": train,
        "dev": dev,
        "old_fit": old_fit,
        "old": old,
        "ancestor": ancestor,
        "added": added,
        "fit": fit,
        "dev_rows": dev_rows,
    }


def _validate_population(data: dict[str, torch.Tensor], *, fit: torch.Tensor | None = None, added: torch.Tensor | None = None) -> None:
    task._validate_selected_rows(
        data["labels"], data["train"], data["dev"], data["old_fit"],
        data["added"] if added is None else added,
        data["fit"] if fit is None else fit, data["dev_rows"], data["ancestor"], data["old"],
    )


def test_legacy_windows_source_paths_normalize_before_exact_population_validation(tmp_path: Path) -> None:
    root = tmp_path / "codex-original"
    root.mkdir()
    saved = {
        "source_generation_dir": "D:/codex-original/outputs/generation/runs/attempt_01",
        "source_checkpoint": "D:/codex-original/outputs/training/runs/attempt_03/arm_a_step4700.pt",
        "source_a4700_readout": "D:/codex-original/outputs/training/runs/attempt_03/readouts_A4700.pt",
        "source_input_paths": {
            "training_data": "D:/codex-original/data/mnist/train.pt",
            "step1200": "D:/codex-original/outputs/canary/runs/attempt_01/step1200.pt",
        },
        "source_row_indices": [12, 34],
        "labels": [2, 4],
    }
    reconstructed = {
        **saved,
        "source_generation_dir": (root / "outputs/generation/runs/attempt_01").as_posix(),
        "source_checkpoint": (root / "outputs/training/runs/attempt_03/arm_a_step4700.pt").as_posix(),
        "source_a4700_readout": (root / "outputs/training/runs/attempt_03/readouts_A4700.pt").as_posix(),
        "source_input_paths": {
            "training_data": (root / "data/mnist/train.pt").as_posix(),
            "step1200": (root / "outputs/canary/runs/attempt_01/step1200.pt").as_posix(),
        },
    }
    with task._legacy_path_translation(root):
        normalized = task.artifact_source._validate_reconstructed_population(root, saved, reconstructed)
    assert normalized == reconstructed

    mismatched = {**reconstructed, "labels": [2, 5]}
    with task._legacy_path_translation(root):
        with pytest.raises(task.artifact_source.EnergyTargetError, match="reconstructed stroke population differs"):
            task.artifact_source._validate_reconstructed_population(root, saved, mismatched)

    outside_root = {**saved, "source_generation_dir": "D:/other-project/outputs/generation"}
    with task._legacy_path_translation(root):
        with pytest.raises(task.CloudForegroundError, match="outside codex-original"):
            task.artifact_source._normalize_population_paths(root, outside_root)


def test_zero_lambda_comparisons_use_float_roundoff_tolerance() -> None:
    reference = torch.tensor([0.5, -0.25], dtype=torch.float32)
    one_ulp = reference.clone()
    one_ulp[0] = torch.nextafter(one_ulp[0], torch.tensor(float("inf"), dtype=one_ulp.dtype))
    comparison = task._float_tensor_comparison(reference, one_ulp)
    assert comparison["passed"] is True
    assert comparison["max_abs_difference"] <= comparison["tolerance"]

    divergent = reference.clone()
    divergent[0] += 1.0e-4
    assert task._float_tensor_comparison(reference, divergent)["passed"] is False

    expected_optimizer = {
        "state": {0: {"step": torch.tensor(4_701.0), "exp_avg": reference, "exp_avg_sq": reference.square()}},
        "param_groups": [{"lr": 1.0e-3, "betas": (0.9, 0.999), "params": [0]}],
    }
    roundoff_optimizer = {
        "state": {0: {"step": torch.tensor(4_701.0), "exp_avg": one_ulp, "exp_avg_sq": reference.square()}},
        "param_groups": [{"lr": 1.0e-3, "betas": (0.9, 0.999), "params": [0]}],
    }
    assert task._adam_state_comparison(expected_optimizer, roundoff_optimizer)["passed"] is True

    wrong_step = {
        "state": {0: {"step": torch.tensor(4_702.0), "exp_avg": one_ulp, "exp_avg_sq": reference.square()}},
        "param_groups": [{"lr": 1.0e-3, "betas": (0.9, 0.999), "params": [0]}],
    }
    assert task._adam_state_comparison(expected_optimizer, wrong_step)["passed"] is False


def test_recovery_state_must_fit_independent_same_device_repeatability_envelope() -> None:
    reference = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)
    repeated = reference.clone()
    repeated[0] += 2.0e-6
    within_envelope = reference.clone()
    within_envelope[0] += 1.0e-6
    phase_comparison = task._tensor_repeatability_comparison(reference, repeated, within_envelope)
    assert phase_comparison["passed"] is True
    assert phase_comparison["resume_variation"]["max_abs_difference"] <= phase_comparison["continuous_repeat_variation"]["max_abs_difference"]
    assert phase_comparison["resume_variation"]["rms_abs_difference"] <= phase_comparison["continuous_repeat_variation"]["rms_abs_difference"]

    outside_envelope = reference.clone()
    outside_envelope[0] += 3.0e-6
    assert task._tensor_repeatability_comparison(reference, repeated, outside_envelope)["passed"] is False

    reference_optimizer = {
        "state": {0: {"step": torch.tensor(5.0), "exp_avg": reference, "exp_avg_sq": reference.square()}},
        "param_groups": [{"lr": 1.0e-3, "betas": (0.9, 0.999), "params": [0]}],
    }
    repeated_optimizer = {
        "state": {0: {"step": torch.tensor(5.0), "exp_avg": repeated, "exp_avg_sq": reference.square()}},
        "param_groups": [{"lr": 1.0e-3, "betas": (0.9, 0.999), "params": [0]}],
    }
    resumed_optimizer = {
        "state": {0: {"step": torch.tensor(5.0), "exp_avg": within_envelope, "exp_avg_sq": reference.square()}},
        "param_groups": [{"lr": 1.0e-3, "betas": (0.9, 0.999), "params": [0]}],
    }
    optimizer_comparison = task._adam_state_repeatability_comparison(reference_optimizer, repeated_optimizer, resumed_optimizer)
    assert optimizer_comparison["passed"] is True
    wrong_step_optimizer = {
        "state": {0: {"step": torch.tensor(6.0), "exp_avg": within_envelope, "exp_avg_sq": reference.square()}},
        "param_groups": [{"lr": 1.0e-3, "betas": (0.9, 0.999), "params": [0]}],
    }
    assert task._adam_state_repeatability_comparison(reference_optimizer, repeated_optimizer, wrong_step_optimizer)["passed"] is False


def test_persisted_repeatability_envelope_is_consumed_without_widening() -> None:
    values = [torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1.0])]
    envelope = task._continuous_tensor_envelope(values)
    within = task._tensor_repeatability_envelope(values, values[0].clone(), fixed_envelope=envelope)
    assert within["passed"] is True

    widened = {**envelope, "max_abs_difference": 1.0}
    rejected = task._tensor_repeatability_envelope(values, values[0].clone(), fixed_envelope=widened)
    assert rejected["passed"] is False
    assert rejected["reason"] == "persisted_repeatability_envelope_does_not_match_continuous_repeats"


def test_task_wide_engineering_budget_reserves_before_update_and_never_resets(tmp_path: Path) -> None:
    budget_path = tmp_path / "engineering_update_budget.json"
    assert task._initialize_engineering_budget(budget_path)["reservations"] == []
    value = task._read_engineering_budget(budget_path)
    value["limit"] = 42
    value["reservations"] = [
        {"index": index, "status": "optimizer_step_completed", "context": {"phase": "historical"}}
        for index in range(1, 4)
    ]
    task._write_json(budget_path, value, replace=True)
    extended = task._extend_engineering_budget(budget_path)
    assert extended["limit"] == 45
    assert len(extended["reservations"]) == 3
    assert all(row["origin"] == "historical_before_budget_extension" for row in extended["reservations"])
    context = {"phase": "test", "seed": 7, "task_local_update": 1}
    attempt_dir = task._root() / task.DEFAULT_ENGINEERING_OUTPUT / "engineering_attempt_02"
    inputs = {"engineering_budget_file": budget_path, "engineering_update_context": context, "engineering_attempt_dir": attempt_dir}

    first = task._reserve_engineering_update(inputs)
    assert first == 4
    after_reservation = task._read_engineering_budget(budget_path)
    assert len(after_reservation["reservations"]) == 4
    assert after_reservation["reservations"][3]["status"] == "reserved_before_optimizer_step"
    assert after_reservation["reservations"][3]["context"] == {
        **context,
        "engineering_attempt_dir": attempt_dir.relative_to(task._root()).as_posix(),
    }
    task._complete_engineering_update(inputs, first)

    for _ in range(41):
        task._reserve_engineering_update(inputs)
    final_budget = task._read_engineering_budget(budget_path)
    assert len(final_budget["reservations"]) == 45
    assert sum(row["origin"] == "historical_before_budget_extension" for row in final_budget["reservations"]) == 3
    with pytest.raises(task.CloudForegroundError, match="budget is exhausted"):
        task._reserve_engineering_update(inputs)


def test_progress_keeps_task_local_checkpoint_cursor_and_existing_path(tmp_path: Path) -> None:
    target = tmp_path / "run"
    checkpoint = target / "checkpoints" / f"seed_{task.TRAIN_SEEDS[0]}_arm_a_step4800.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"verified checkpoint marker")

    fields = task._checkpoint_progress_state(target, checkpoint, 100, "registered_100_update_boundary")

    assert fields == {
        "last_checkpoint": "checkpoints/seed_2026101201_arm_a_step4800.pt",
        "last_checkpoint_update": 100,
        "last_checkpoint_adam_step": 4_800,
        "last_checkpoint_reason": "registered_100_update_boundary",
    }
    assert task._checkpoint_progress_state(target, None, 0, None)["last_checkpoint_adam_step"] == task.SOURCE_STEP


def test_engineering_stage_readback_is_immutable(tmp_path: Path) -> None:
    attempt = tmp_path / "engineering_attempt_01"
    payload = {"status": "COMPLETE", "passed": True, "rows": 20}
    path = task._write_engineering_stage(attempt, "saved_evidence_pipeline", payload)

    assert task._json(path) == {
        "task_id": task.TASK_ID,
        "phase": "saved_evidence_pipeline",
        "evidence_identity": {"input_population": "input_population.json", "batch_orders": "batch_orders.json", "run_config": "engineering_run_config.json"},
        **payload,
    }
    assert task._write_engineering_stage(attempt, "saved_evidence_pipeline", payload) == path
    with pytest.raises(task.CloudForegroundError, match="refusing to overwrite different evidence"):
        task._write_engineering_stage(attempt, "saved_evidence_pipeline", {**payload, "rows": 19})


def test_engineering_failure_updates_attempt_and_task_progress_without_resetting_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    relative_root = Path("outputs/engineering_progress_test")
    monkeypatch.setattr(task, "DEFAULT_ENGINEERING_OUTPUT", relative_root)
    monkeypatch.setattr(task, "_root", lambda: tmp_path)
    output_root = tmp_path / relative_root
    attempt = output_root / "engineering_attempt_01"
    attempt.mkdir(parents=True)
    (attempt / "progress.json").write_text('{"phase":"trajectory_complete"}', encoding="utf-8")
    budget_path = output_root / "engineering_update_budget.json"
    budget = task._initialize_engineering_budget(budget_path)
    budget["limit"] = 42
    budget["reservations"] = [
        {"index": index, "status": "optimizer_step_completed", "context": {"phase": "historical"}}
        for index in range(1, 4)
    ]
    task._write_json(budget_path, budget, replace=True)
    task._extend_engineering_budget(budget_path)
    budget_inputs = {
        "engineering_budget_file": budget_path,
        "engineering_update_context": {"phase": "trajectory_complete"},
        "engineering_attempt_dir": attempt,
    }
    for _ in range(3):
        task._reserve_engineering_update(budget_inputs)

    args = SimpleNamespace(engineering_resume_evidence_dir=None, engineering_attempt_dir=relative_root / "engineering_attempt_01")
    task._record_engineering_failure(tmp_path, args, RuntimeError("injected diagnostic failure"))

    attempt_progress = task._json(attempt / "progress.json")
    root_progress = task._json(output_root / "progress.json")
    assert attempt_progress["status"] == root_progress["status"] == "INVALID_HARNESS"
    assert attempt_progress["phase"] == root_progress["phase"] == "trajectory_complete"
    assert attempt_progress["engineering_updates_reserved"] == root_progress["engineering_updates_reserved"] == 6
    assert attempt_progress["historical_engineering_updates"] == root_progress["historical_engineering_updates"] == 3
    assert attempt_progress["new_engineering_updates_reserved"] == root_progress["new_engineering_updates_reserved"] == 3
    reservations = task._read_engineering_budget(output_root / "engineering_update_budget.json")["reservations"]
    assert len(reservations) == 6
    assert all(row["status"] == "optimizer_step_completed" for row in reservations[:3])
    assert all(row["status"] == "reserved_before_optimizer_step" for row in reservations[3:])

    task._record_engineering_failure(tmp_path, args, RuntimeError("second diagnostic failure"))
    assert (attempt / "engineering_failure_01.json").is_file()
    assert task._json(attempt / "invalid_harness.json")["failure"] == "second diagnostic failure"


def test_engineering_resume_consumes_valid_partial_stages_and_rejects_config_mismatch(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs" / "engineering"
    attempt = output_root / "engineering_attempt_01"
    attempt.mkdir(parents=True)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (attempt / "plan.md").write_text("plan", encoding="utf-8")
    (attempt / "input_probe.json").write_text(task.json.dumps({"evidence_dir": evidence.resolve().as_posix()}), encoding="utf-8")
    population = {"source_checkpoint": "source.pt"}
    (attempt / "input_population.json").write_text(task.json.dumps(population), encoding="utf-8")
    orders = {str(seed): [[index for index in range(10)]] for seed in task.TRAIN_SEEDS}
    (attempt / "batch_orders.json").write_text(task.json.dumps(orders), encoding="utf-8")
    config = {
        "task_id": task.TASK_ID,
        "source_checkpoint": "source.pt",
        "source_checkpoint_step": task.SOURCE_STEP,
        "source_adam_step": task.SOURCE_STEP,
        "requested_microbatch_size": 1,
        "trajectory_microbatch_sizes": [1, 2, 5, 10, 1, 1],
        "logical_batch_size": task.BATCH_SIZE,
        "propagation_grid": 2048,
        "optical_dtype": "float32",
        "qfg_dtype": "float64",
        "fixed_lambda": task.FIXED_LAMBDA,
        "generator_only_adam": True,
    }
    (attempt / "engineering_run_config.json").write_text(task.json.dumps(config), encoding="utf-8")
    (attempt / "preflight.json").write_text(task.json.dumps({"task_id": task.TASK_ID, "status": "INVALID_HARNESS", "checks": {}}), encoding="utf-8")
    inputs = {"population": population, "batch_orders": {int(seed): value for seed, value in orders.items()}}

    result = task._validate_engineering_resume_attempt(attempt, output_root, evidence, inputs, SimpleNamespace(microbatch_size=1), output_root / "engineering_update_budget.json")
    assert result["status"] == "PARTIAL"
    assert result["completed_stages"] == []

    config["fixed_lambda"] = 0.0
    (attempt / "engineering_run_config.json").write_text(task.json.dumps(config), encoding="utf-8")
    with pytest.raises(task.CloudForegroundError, match="configuration differs"):
        task._validate_engineering_resume_attempt(attempt, output_root, evidence, inputs, SimpleNamespace(microbatch_size=1), output_root / "engineering_update_budget.json")


def test_partial_engineering_update_requires_completed_budget_and_reloadable_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(task, "_root", lambda: tmp_path)
    target = tmp_path / "outputs" / "engineering" / "engineering_attempt_01"
    target.mkdir(parents=True)
    budget_path = tmp_path / "engineering_update_budget.json"
    value = {
        "schema": "autodl_engineering_adam_budget_v1",
        "limit": 45,
        "budget_changes": [{"previous_limit": 42, "new_limit": 45, "new_diagnostic_allowance": 42}],
        "reservations": [
            {
                "index": index,
                "origin": "historical_before_budget_extension",
                "status": "optimizer_step_completed",
                "context": {"phase": "six_seed_arm_three_step_updates", "seed": task.TRAIN_SEEDS[0], "arm": "A", "task_local_update": 1, "microbatch_size": 1},
            }
            for index in range(1, 4)
        ],
    }
    value["reservations"].append({
        "index": 4,
        "origin": "current_engineering_diagnostic",
        "status": "optimizer_step_completed",
        "context": {"phase": "six_seed_arm_three_step_updates", "seed": task.TRAIN_SEEDS[0], "arm": "A", "task_local_update": 1, "microbatch_size": 1, "engineering_attempt_dir": target.relative_to(tmp_path).as_posix()},
    })
    task._write_json(budget_path, value)
    seed = task.TRAIN_SEEDS[0]
    batch_order = [[index for index in range(10)]]
    key = f"seed_{seed}_arm_a"
    checkpoint_root = task._engineering_checkpoint_root(target, "trajectory", key)
    checkpoint_path = task._checkpoint_path(checkpoint_root, seed, "A", 1)
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"serialized checkpoint fixture")
    record = {
        "task_local_update": 1,
        "seed": seed,
        "arm": "A",
        "fit_positions": list(range(10)),
        "labels": list(range(10)),
        "engineering_first_update_verification": {"passed": True},
    }
    payload = {
        "task_local_updates": 1,
        "training_history": [record],
        "seed": seed,
        "arm": "A",
        "optimizer_step": task.SOURCE_STEP + 1,
        "next_batch_cursor": 1,
        "batch_order": batch_order,
        "trainable_parameter_state": {"generator.phase": torch.zeros(1)},
        "optimizer_state": {},
    }
    monkeypatch.setattr(task, "_read_checkpoint_file", lambda *_args, **_kwargs: payload)
    inputs = {"engineering_budget_file": budget_path, "batch_orders": {seed: batch_order}, "engineering_attempt_dir": target}

    loaded, stages, _last = task._engineering_update_prefix(target, inputs, "trajectory", key, seed, "A", 1)
    assert loaded is payload
    assert len(stages) == 1 and stages[0]["checkpoint_readback_passed"] is True
    assert task._read_engineering_stage(target, task._engineering_update_stage_name("trajectory", key, 1)) is not None

    missing_target = tmp_path / "engineering_attempt_02"
    # A reservation from another attempt cannot masquerade as this attempt's incomplete update.
    assert task._engineering_update_prefix(missing_target, inputs, "trajectory", key, seed, "A", 1) == (None, [], None)
    missing_target.mkdir()
    value = task._read_engineering_budget(budget_path)
    value["reservations"].append({
        "index": len(value["reservations"]) + 1,
        "origin": "current_engineering_diagnostic",
        "status": "reserved_before_optimizer_step",
        "context": {"phase": "six_seed_arm_three_step_updates", "seed": seed, "arm": "A", "task_local_update": 1, "microbatch_size": 1, "engineering_attempt_dir": missing_target.relative_to(tmp_path).as_posix()},
    })
    task._write_json(budget_path, value, replace=True)
    with pytest.raises(task.CloudForegroundError, match="reservation has no reusable checkpoint or stage"):
        task._engineering_update_prefix(missing_target, inputs, "trajectory", key, seed, "A", 1)


def test_closed_form_adam_reference_matches_one_real_optimizer_update() -> None:
    parameter = torch.nn.Parameter(torch.tensor([0.25, -0.5, 1.25], dtype=torch.float32))
    optimizer = torch.optim.Adam([parameter], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    optimizer.state[parameter] = {
        "step": torch.tensor(float(task.SOURCE_STEP)),
        "exp_avg": torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32),
        "exp_avg_sq": torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32),
    }
    optimizer_before = task.capacity._cpu_tree(optimizer.state_dict())
    gradient = torch.tensor([0.03, -0.01, 0.02], dtype=torch.float32)
    expected_parameter, expected_state = task._closed_form_adam_step(parameter.detach().clone(), optimizer_before, gradient)
    parameter.grad = gradient.clone()
    optimizer.step()

    assert torch.allclose(parameter.detach(), expected_parameter, rtol=2e-6, atol=2e-7)
    actual_state = optimizer.state[parameter]
    assert actual_state["step"].item() == task.SOURCE_STEP + 1
    assert torch.allclose(actual_state["exp_avg"], expected_state["exp_avg"], rtol=2e-6, atol=2e-7)
    assert torch.allclose(actual_state["exp_avg_sq"], expected_state["exp_avg_sq"], rtol=2e-6, atol=2e-7)


def test_production_reference_uses_qfg_gradient_not_reconstruction_gradient(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = torch.tensor([1.0, -2.0], dtype=torch.float32)
    reconstruction = torch.tensor([50.0, 60.0], dtype=torch.float32)
    qfg = torch.tensor([0.25, -0.5], dtype=torch.float32)

    def components(*_args: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        return baseline, reconstruction, qfg, {}

    monkeypatch.setattr(task.gradient, "_single_microbatch_batch_gradients", components)
    inputs = {
        "fit_images": torch.zeros((10, 1, 28, 28), dtype=torch.float32),
        "fit_labels": torch.arange(10, dtype=torch.long),
    }

    actual = task._production_reference_batch_gradient(
        SimpleNamespace(device=torch.device("cpu")), {}, {}, inputs, list(range(10)), 0.5
    )
    expected = baseline + task.semantic.BETA * 0.5 * qfg

    assert torch.equal(actual, expected)


def test_ready_receipt_consumes_the_production_zero_lambda_control_result() -> None:
    check_names = (
        "source_model_and_optimizer",
        "cloud_local_replay",
        "cached_ols_source_recheck",
        "qfg_independent_value_gradient",
        "merged_production_gradient_and_speed",
        "six_seed_arm_three_step_updates",
        "explicit_checkpoint_recovery",
        "cuda_and_disk",
    )
    checks: dict[str, dict[str, Any]] = {name: {"executed": True, "passed": True} for name in check_names}
    checks["merged_production_gradient_and_speed"]["Encoder_autograd_enabled"] = True
    checks["six_seed_arm_three_step_updates"]["all_zero_lambda_controls_matched_first_A_within_float_roundoff"] = True
    inputs = {
        "population": {"source_checkpoint": "outputs/source/arm_a_step4700.pt"},
        "fit_rows": torch.arange(task.FIT_ROWS),
        "dev_rows": torch.arange(task.DEV_ROWS),
    }

    _control, receipt = task._make_receipt(inputs, {"checks": checks})
    assert receipt.to_dict()["checks"]["CORRECTNESS_INVARIANTS"]["zero_lambda_negative_control"] is True


def test_conditional_sample_rows_are_keyed_by_class_and_within_class_draw(tmp_path: Path) -> None:
    seed = task.SAMPLE_SEEDS[0]
    bank = {"z": torch.empty((len(task.SAMPLE_SEEDS), 10, task.SAMPLES_PER_CLASS, 100))}
    generated = task._sample_rows_for_seed(seed, bank)
    rows = [
        {
            "scenario": "A4700",
            "sample_seed": seed,
            "draw_index": row["draw_index"],
            "requested_class": row["requested_class"],
            "e_s_prediction": 0,
            "e_i_prediction": 0,
            "e_c_prediction": 0,
            "e_d_prediction": 0,
            "raw_cmos_integral": 1.0,
            "clipped_fraction": 0.0,
            "generator_input_power": 1.0,
        }
        for row in generated
    ]
    csv_path = tmp_path / "paired_rows.csv"
    task._write_csv(csv_path, rows, task.SAMPLE_FIELDS)
    loaded = task._read_csv(csv_path)

    by_class_draw = task._index_sample_rows(loaded, "A4700", seed)

    assert len(by_class_draw) == 300
    assert by_class_draw[(0, 0)]["requested_class"] == "0"
    assert by_class_draw[(9, task.SAMPLES_PER_CLASS - 1)]["requested_class"] == "9"

    duplicated_and_missing = loaded.copy()
    duplicated_and_missing[-1] = {**duplicated_and_missing[-1], "requested_class": "8", "draw_index": str(task.SAMPLES_PER_CLASS - 1)}
    with pytest.raises(task.CloudForegroundError, match="missing or duplicate draws"):
        task._index_sample_rows(duplicated_and_missing, "A4700", seed)


def test_bounded_conditional_diagnostic_uses_two_draw_keys_per_class() -> None:
    seed = task.SAMPLE_SEEDS[0]
    bank = {"z": torch.zeros((len(task.SAMPLE_SEEDS), 10, task.SAMPLES_PER_CLASS, 100), dtype=torch.complex64)}
    generated = task._sample_rows_for_seed(seed, bank, draws_per_class=2)
    assert len(generated) == 20
    assert generated[0]["position"] == 0 and generated[0]["requested_class"] == 0 and generated[0]["draw_index"] == 0
    assert generated[-1]["position"] == 19 and generated[-1]["requested_class"] == 9 and generated[-1]["draw_index"] == 1
    rows = [
        {
            "scenario": "A4700",
            "sample_seed": seed,
            "draw_index": row["draw_index"],
            "requested_class": row["requested_class"],
            "e_s_prediction": 0,
            "e_i_prediction": 0,
            "e_c_prediction": 0,
            "e_d_prediction": 0,
            "raw_cmos_integral": 1.0,
            "clipped_fraction": 0.0,
            "generator_input_power": 1.0,
        }
        for row in generated
    ]
    assert len(task._index_sample_rows(rows, "A4700", seed, draws_per_class=2)) == 20
    with pytest.raises(task.CloudForegroundError, match="incomplete"):
        task._index_sample_rows(rows[:-1], "A4700", seed, draws_per_class=2)


def test_bounded_dev_readout_uses_production_save_and_raw_recheck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    labels = torch.tensor([digit for digit in range(10) for _ in range(2)], dtype=torch.long)
    target_raw = torch.zeros((1, 1, 200, 200), dtype=torch.float32)
    target_raw[:, :, 20:80, 30:90] = 0.7
    raw = torch.full((1, 1, 200, 200), 0.2, dtype=torch.float32)

    class Runtime:
        def forward_from_image(self, image: torch.Tensor) -> Any:
            return SimpleNamespace(
                raw_intensity=raw.clone(),
                image_input=SimpleNamespace(slm_intensity=target_raw.clone()),
            )

    consumer = SimpleNamespace(device=torch.device("cpu"), runtime=Runtime())
    inputs = {
        "dev_images": torch.zeros((20, 1, 28, 28), dtype=torch.float32),
        "dev_rows": torch.arange(20, dtype=torch.long),
        "dev_labels": labels,
        "sources": {"source": {"pixel_area_m2": 1.0e-8}},
    }
    monkeypatch.setattr(
        task,
        "_evaluator_logits",
        lambda _evaluators, view: {name: torch.zeros((view.shape[0], 10)) for name in task.EVALUATOR_NAMES},
    )
    positions = list(range(20))
    saved = task._save_dev_readout(tmp_path, inputs, {}, consumer, "A4700", positions=positions)
    rechecked = task._recheck_dev_scenario(tmp_path, inputs, "A4700", positions)
    assert len(saved) == len(rechecked) == 20
    assert [sum(int(row["label"]) == digit for row in rechecked) for digit in range(10)] == [2] * 10
    assert (tmp_path / "readouts" / "A4700" / "chunk_0000.pt").is_file()


def test_fixed_fit_and_dev_populations_are_disjoint_balanced_and_exclusion_safe() -> None:
    data = _population_fixture()
    _validate_population(data)
    assert data["fit"].numel() == task.FIT_ROWS
    assert data["dev_rows"].numel() == task.DEV_ROWS
    assert torch.bincount(data["labels"][data["fit"]], minlength=10).tolist() == [200] * 10
    assert torch.bincount(data["labels"][data["dev_rows"]], minlength=10).tolist() == [100] * 10
    assert not set(data["added"].tolist()) & (set(data["ancestor"].tolist()) | set(data["old"].tolist()))


def test_fixed_population_rejects_out_of_range_overlap_and_dev_as_training_input() -> None:
    data = _population_fixture()
    out_of_range = data["added"].clone()
    out_of_range[0] = 60_000
    with pytest.raises(task.CloudForegroundError, match="out-of-range"):
        _validate_population(data, added=out_of_range)

    ancestor_overlap = data["added"].clone()
    ancestor_overlap[0] = data["ancestor"][0]
    with pytest.raises(task.CloudForegroundError, match="overlaps a named exclusion"):
        _validate_population(data, added=ancestor_overlap)

    old_only_overlap = data["added"].clone()
    old_only_overlap[0] = torch.tensor(5_000)
    with pytest.raises(task.CloudForegroundError, match="overlaps a named exclusion"):
        _validate_population(data, added=old_only_overlap)

    dev_as_fit = data["fit"].clone()
    dev_as_fit[-1] = data["dev_rows"][0]
    with pytest.raises(task.CloudForegroundError, match="DEV row entered"):
        _validate_population(data, fit=dev_as_fit)


def test_seed_schedules_are_reproducible_paired_and_one_row_per_class() -> None:
    data = _population_fixture()
    fit_labels = data["labels"][data["fit"]]
    first = task._make_schedule(fit_labels, task.TRAIN_SEEDS[0])
    repeated = task._make_schedule(fit_labels, task.TRAIN_SEEDS[0])
    second = task._make_schedule(fit_labels, task.TRAIN_SEEDS[1])
    assert len(first) == task.UPDATES_PER_ARM
    assert first == repeated
    assert first != second
    assert all(fit_labels[batch].tolist() == list(range(10)) for batch in first)


def test_canonical_semantic_direction_changes_when_encoder_gradients_are_dropped() -> None:
    encoder = torch.nn.Parameter(torch.tensor([0.2, -0.4], dtype=torch.float64))
    generator = torch.nn.Parameter(torch.tensor([0.1, 0.3], dtype=torch.float64))
    coefficients = {
        "E_S": (torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])),
        "E_I": (torch.tensor([0.0, 2.0]), torch.tensor([0.0, 1.0])),
        "E_C": (torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])),
    }
    gradients: dict[str, dict[str, torch.Tensor]] = {}
    for name, (encoder_coeff, generator_coeff) in coefficients.items():
        loss = (encoder * encoder_coeff.to(torch.float64)).sum() + (generator * generator_coeff.to(torch.float64)).sum()
        encoder_grad, generator_grad = torch.autograd.grad(loss, (encoder, generator))
        gradients[name] = {"encoder.phase": encoder_grad, "generator.phase": generator_grad}

    full, full_reason = task._canonical_semantic_direction(gradients)
    without_encoder = {
        name: {key: value.clone() for key, value in fields.items()}
        for name, fields in gradients.items()
    }
    for fields in without_encoder.values():
        fields["encoder.phase"].zero_()
    omitted, omitted_reason = task._canonical_semantic_direction(without_encoder)

    assert full_reason is None and full is not None
    assert omitted_reason is None and omitted is not None
    assert full["encoder.phase"].abs().sum() > 0
    assert not torch.allclose(full["generator.phase"], omitted["generator.phase"])


def test_qfg_numpy_analytic_and_mass_normalized_finite_difference_agree() -> None:
    target = torch.zeros((1, 1, 200, 200), dtype=torch.float64)
    target[:, :, 20:80, 30:90] = 0.7
    raw = torch.linspace(0.08, 0.32, 40_000, dtype=torch.float64).reshape(1, 1, 200, 200).requires_grad_()
    value = task.gradient._qfg_components(raw, target)["q_fg"].sum()
    autograd_image = torch.autograd.grad(value, raw)[0]
    analytic_image = task.gradient._qfg_gradient_reference(raw.detach(), target)
    numpy_value = task._numpy_qfg(raw.detach(), target)
    production_value = task.gradient._qfg_components(raw.detach(), target)["q_fg"].cpu().numpy()
    assert np.allclose(numpy_value, production_value, rtol=1e-12, atol=1e-14)
    assert torch.allclose(autograd_image, analytic_image, rtol=1e-11, atol=1e-13)

    epsilon = 1e-3
    plus = raw.detach().clone()
    minus = raw.detach().clone()
    plus[0, 0, 40, 40] += epsilon
    minus[0, 0, 40, 40] -= epsilon
    finite_difference = (task._numpy_qfg(plus, target)[0] - task._numpy_qfg(minus, target)[0]) / (2.0 * epsilon)
    assert finite_difference == pytest.approx(float(autograd_image[0, 0, 40, 40]), rel=2e-3, abs=2e-10)


def test_cached_ols_is_recomputed_against_the_named_encoder() -> None:
    class Runtime:
        def forward_from_image(self, image: torch.Tensor) -> Any:
            value = image.mean().reshape(1)
            coefficients = torch.complex(value[:, None].expand(-1, 100), (2.0 * value)[:, None].expand(-1, 100))
            target_slm = (value + 0.1)[:, None, None, None].expand(-1, 1, 200, 200)
            return SimpleNamespace(z_ols=SimpleNamespace(coefficients=coefficients), image_input=SimpleNamespace(slm_intensity=target_slm))

    images = torch.arange(10 * 28 * 28, dtype=torch.float32).reshape(10, 1, 28, 28) / 1000.0
    values = images.mean(dim=(1, 2, 3))
    coefficients = torch.complex(values[:, None].expand(-1, 100), (2.0 * values)[:, None].expand(-1, 100))
    targets = (values + 0.1)[:, None, None, None].expand(-1, 1, 200, 200).clone()
    target_shapes = task.frozen._shape_vectors(targets, 28)
    cache = {"coefficients": coefficients, "target_shape_vectors": target_shapes, "cache_was_existing": False}
    consumer = SimpleNamespace(device=torch.device("cpu"), runtime=Runtime())
    inputs = {"fit_images": images}
    assert "target_slm_intensity" not in cache
    assert target_shapes.shape == (len(images), 28 * 28)
    assert task._verify_cached_ols(consumer, inputs, cache)["rows_recomputed"] == task.BATCH_SIZE

    corrupted = {**cache, "coefficients": coefficients.clone()}
    corrupted["coefficients"][0, 0] += 0.01 + 0.0j
    with pytest.raises(task.CloudForegroundError, match="cached OLS coefficients differ"):
        task._verify_cached_ols(consumer, inputs, corrupted)


def test_source_step_and_recovery_batch_order_mismatches_are_rejected(tmp_path: Path) -> None:
    bad_source = {"task_id": task.artifact_source.TASK_ID, "arm": "A_unscaled_target", "completed_updates": task.SOURCE_STEP, "optimizer_step": task.SOURCE_STEP - 1}
    with pytest.raises(task.CloudForegroundError, match="model/Adam state is not at step"):
        task._validate_source_checkpoint(bad_source, {})

    target = tmp_path / "registered"
    target.mkdir()
    source_path = tmp_path / "source.pt"
    inputs = {"sources": {"source": {"checkpoint_path": source_path}}, "population": {"source_row_indices": [3, 4]}}
    seed = task.TRAIN_SEEDS[0]
    expected_order = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
    path = task._checkpoint_path(target, seed, "A", 1)
    path.parent.mkdir(parents=True)
    path.touch()
    payload = {
        "schema": task.CHECKPOINT_SCHEMA,
        "task_id": task.TASK_ID,
        "seed": seed,
        "arm": "A",
        "task_local_updates": 1,
        "source_step": task.SOURCE_STEP,
        "completed_updates": task.SOURCE_STEP + 1,
        "optimizer_step": task.SOURCE_STEP + 1,
        "lambda": task.FIXED_LAMBDA,
        "numeric_mode": task.NUMERIC_MODE,
        "source_checkpoint": source_path.as_posix(),
        "source_row_indices": [3, 4],
        "batch_order": [[9, 8, 7, 6, 5, 4, 3, 2, 1, 0]],
    }
    with pytest.raises(task.CloudForegroundError, match="batch order differs"):
        task._validate_checkpoint(payload, path, target, inputs, {seed: expected_order})


def test_resume_must_name_the_latest_checkpoint_for_its_trajectory(tmp_path: Path) -> None:
    target = tmp_path / "registered"
    checkpoint_dir = target / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    seed, arm = task.TRAIN_SEEDS[0], "A"
    older = checkpoint_dir / f"seed_{seed}_arm_a_step4800.pt"
    latest = checkpoint_dir / f"seed_{seed}_arm_a_step4900.pt"
    older.touch()
    latest.touch()
    payload = {"seed": seed, "arm": arm}
    with pytest.raises(task.CloudForegroundError, match="older than a saved checkpoint"):
        task._validate_explicit_resume_is_latest(target, older, payload)
    task._validate_explicit_resume_is_latest(target, latest, payload)


def test_recovery_checkpoint_consumer_reads_and_validates_serialized_file(tmp_path: Path) -> None:
    target = tmp_path / "registered"
    seed = task.TRAIN_SEEDS[0]
    source_path = target / "source.pt"
    batch_order = [list(range(10))]
    inputs = {"sources": {"source": {"checkpoint_path": source_path}}, "population": {"source_row_indices": [3, 4]}}
    path = task._checkpoint_path(target, seed, "A", 1)
    path.parent.mkdir(parents=True)
    phase_state = {
        "encoder.phase": torch.zeros((2, 1, 100, 100), dtype=torch.float32),
        "generator.phase": torch.zeros((340_000,), dtype=torch.float32),
    }
    moments = torch.zeros((340_000,), dtype=torch.float32)
    payload = {
        "schema": task.CHECKPOINT_SCHEMA,
        "task_id": task.TASK_ID,
        "seed": seed,
        "arm": "A",
        "task_local_updates": 1,
        "source_step": task.SOURCE_STEP,
        "completed_updates": task.SOURCE_STEP + 1,
        "optimizer_step": task.SOURCE_STEP + 1,
        "epoch": 0,
        "next_batch_cursor": 1,
        "batch_order": batch_order,
        "lambda": task.FIXED_LAMBDA,
        "numeric_mode": task.NUMERIC_MODE,
        "source_checkpoint": source_path.as_posix(),
        "source_row_indices": [3, 4],
        "trainable_parameter_state": phase_state,
        "optimizer_state": {
            "param_groups": [{"params": [0], "lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8}],
            "state": {0: {"step": torch.tensor(float(task.SOURCE_STEP + 1)), "exp_avg": moments, "exp_avg_sq": moments.clone()}},
        },
        "frozen_runtime_state": {},
        "rng_state": {"python": [1, 2, 3]},
        "training_history": [{"task_local_update": 1, "seed": seed, "arm": "A"}],
    }
    torch.save(payload, path)

    assert task._read_checkpoint_file(path, target, inputs, {seed: batch_order})["task_local_updates"] == 1
    corrupted = dict(payload)
    corrupted["optimizer_state"] = task.capacity._cpu_tree(payload["optimizer_state"])
    corrupted["optimizer_state"]["state"][0]["step"] = torch.tensor(float(task.SOURCE_STEP + 2))
    torch.save(corrupted, path)
    with pytest.raises(task.CloudForegroundError, match="resume Adam state step or geometry differs"):
        task._read_checkpoint_file(path, target, inputs, {seed: batch_order})


def test_checkpoint_policy_keeps_legacy_resume_behavior_without_registered_policy(tmp_path: Path) -> None:
    target = tmp_path / "legacy"
    target.mkdir()
    assert task._checkpoint_policy(target) == {"interval": 100, "keep_recent": None, "source": "legacy_default"}
    (target / "fixed_parameters.json").write_text(
        '{"checkpoint_policy":{"interval":500,"keep_recent":2}}', encoding="utf-8"
    )
    assert task._checkpoint_policy(target) == {"interval": 500, "keep_recent": 2, "source": "registered"}


def test_verified_checkpoint_rotation_keeps_two_recent_and_terminal(tmp_path: Path) -> None:
    target = tmp_path / "run"
    checkpoint_dir = target / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    seed, arm = task.TRAIN_SEEDS[0], "A"
    updates = (100, 500, 1_000, 1_500, task.UPDATES_PER_ARM)
    paths = [task._checkpoint_path(target, seed, arm, local) for local in updates]
    for path in paths:
        path.touch()

    task._rotate_verified_checkpoints(
        target, seed, arm, {"interval": 500, "keep_recent": 2, "source": "registered"}
    )

    assert {path.name for path in checkpoint_dir.iterdir()} == {
        task._checkpoint_path(target, seed, arm, 1_000).name,
        task._checkpoint_path(target, seed, arm, 1_500).name,
        task._checkpoint_path(target, seed, arm, task.UPDATES_PER_ARM).name,
    }


def test_incomplete_saved_dev_readout_is_rejected_before_scientific_use(tmp_path: Path) -> None:
    scenario = "seed_2026101201_A"
    scenario_dir = tmp_path / "readouts" / scenario
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "paired_rows.csv").write_text(",".join(task.DEV_FIELDS) + "\n", encoding="utf-8")
    with pytest.raises(task.CloudForegroundError, match="does not contain 1,000 rows"):
        task._recheck_dev_scenario(tmp_path, {}, scenario)


def test_input_probe_lists_missing_files_and_deduplicates_resolved_paths(tmp_path: Path) -> None:
    present = tmp_path / "weights.pt"
    present.write_bytes(b"checkpoint")
    missing = tmp_path / "missing.pt"

    inventory, missing_paths = task._input_file_inventory([present, present.parent / "." / present.name, missing])

    assert len(inventory) == 2
    assert inventory[0] == {"path": present.resolve().as_posix(), "exists": True, "size_bytes": len(b"checkpoint")}
    assert missing_paths == [missing.resolve().as_posix()]


def test_saved_dev_readout_rejects_mismatched_column_names(tmp_path: Path) -> None:
    scenario = "seed_2026101201_A"
    scenario_dir = tmp_path / "readouts" / scenario
    scenario_dir.mkdir(parents=True)
    fields = [field for field in task.DEV_FIELDS if field != "q_fg"]
    (scenario_dir / "paired_rows.csv").write_text(",".join(fields) + "\n", encoding="utf-8")

    with pytest.raises(task.CloudForegroundError, match="missing required fields"):
        task._recheck_dev_scenario(tmp_path, {}, scenario)


def test_conditional_readout_rejects_mismatched_column_names() -> None:
    rows = [{"requested_class": label, "draw_index": 0} for label in range(10)]

    with pytest.raises(task.CloudForegroundError, match="fields are incomplete"):
        task._index_sample_rows(rows, "A4700", task.SAMPLE_SEEDS[0], draws_per_class=1)
