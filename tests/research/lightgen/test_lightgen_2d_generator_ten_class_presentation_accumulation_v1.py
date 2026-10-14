from __future__ import annotations

import torch

from scripts.research.lightgen import execute_lightgen_2d_generator_evaluator_ablation_comparison_v1 as ablation
from scripts.research.lightgen import execute_lightgen_2d_generator_ten_class_presentation_accumulation_v1 as task


def _balanced_labels(rows_per_class: int = 30) -> torch.Tensor:
    return torch.arange(10).repeat(rows_per_class)


def _metrics(correct: list[int], *, raw_mse: float = 0.2) -> dict:
    return {
        "evaluators": {"E_D": {"by_role": {"new_confirmation": {"per_class_correct": correct, "correct_total": sum(correct), "macro_recall": sum(correct) / 300.0}}}},
        "image_metrics_by_role": {"new_confirmation": {"raw_mse": raw_mse, "raw_fraction_above_one": 0.0, "raw_zero_image": 0, "raw_constant_image": 0, "by_class": {str(c): {"raw_fraction_above_one": 0.0} for c in range(10)}}},
        "all_finite": True,
    }


def test_new_confirmation_is_fixed_balanced_train_population_excluding_prior_rows() -> None:
    labels = torch.arange(10).repeat(6_000)
    prior = torch.arange(600)
    train = torch.arange(60_000)
    first = task._new_indices_for_test(labels, prior, train)
    second = task._new_indices_for_test(labels, prior, train)
    assert torch.equal(first, second)
    assert first.numel() == 300 and torch.unique(first).numel() == 300
    assert not set(first.tolist()).intersection(prior.tolist())
    assert torch.bincount(labels[first], minlength=10).tolist() == [30] * 10


def test_new_confirmation_rejects_indices_outside_canonical_train() -> None:
    labels = torch.arange(10).repeat(6_000)
    prior = torch.arange(1, 601)
    train = torch.arange(60_000)
    train[600] = 59_999
    try:
        task._new_indices_for_test(labels, prior, train)
    except task.TenClassAccumulationError:
        return
    raise AssertionError("new confirmation accepted invalid TRAIN membership")


def test_production_float64_row_scoring_reports_margin_and_ce() -> None:
    labels = _balanced_labels()
    logits = torch.zeros((300, 10), dtype=torch.float32)
    logits.scatter_(1, labels[:, None], 2.0)
    readouts = {
        "labels": labels,
        "target_slm_intensity": torch.zeros((300, 1, 200, 200)),
        "raw_cmos": torch.full((300, 1, 200, 200), 0.25),
        "evaluator_view": torch.full((300, 1, 28, 28), 0.25),
        **{f"{name.lower()}_logits": logits.clone() for name in task.EVALUATOR_NAMES},
    }
    population = {
        "role_positions": {
            "fit": [0, 200],
            "historic": [200, 230],
            "old_confirmation": [230, 260],
            "new_confirmation": [0, 300],
        }
    }
    summary, rows = task._row_and_image_metrics(readouts, population)
    assert summary["evaluators"]["E_D"]["by_role"]["new_confirmation"]["per_class_correct"] == [30] * 10
    image_summary = summary["image_metrics_by_role"]["new_confirmation"]
    assert image_summary["raw_zero_image"] == 0
    assert image_summary["raw_constant_image"] == 300
    expected_margin = 2.0
    torch.testing.assert_close(rows["e_d_margin_float64"], torch.full((300,), expected_margin, dtype=torch.float64))
    torch.testing.assert_close(rows["e_d_ce_float64"], torch.full((300,), float(torch.logsumexp(torch.tensor([2.0] + [0.0] * 9, dtype=torch.float64), dim=0) - 2.0), dtype=torch.float64))
    assert summary["image_metrics_by_role"]["new_confirmation"]["rows"] == 300
    assert task._final_decision(summary, summary, valid=True)["status"] == "VALID_FAILED"


def test_final_decision_requires_macro_gain_and_all_ten_class_floor() -> None:
    start = _metrics([22, 20, 22, 22, 22, 20, 20, 18, 24, 20])
    supported = _metrics([24, 23, 23, 23, 23, 22, 22, 22, 22, 21])
    decision = task._final_decision(start, supported, valid=True)
    assert decision["status"] == "VALID_COMPLETED"
    assert decision["classification"] == "TEN_CLASS_PRESENTATION_ACCUMULATION_SUPPORTED"
    assert all(decision["conditions"].values())

    under_floor = _metrics([24, 23, 23, 23, 23, 22, 22, 22, 22, 14])
    decision = task._final_decision(start, under_floor, valid=True)
    assert decision["classification"] == "MIXED_OR_INSUFFICIENT"
    assert not decision["conditions"]["each_class_at_least_15_of_30"]


def test_final_decision_refuses_category_loss_even_when_macro_passes() -> None:
    start = _metrics([22, 22, 22, 22, 22, 20, 20, 18, 24, 20])
    final = _metrics([24, 20, 25, 25, 23, 22, 22, 22, 22, 21])
    result = task._final_decision(start, final, valid=True)
    assert result["classification"] == "MIXED_OR_INSUFFICIENT"
    assert not result["conditions"]["classes_1_and_7_do_not_decline"]


def test_invalid_measurement_has_no_scientific_classification() -> None:
    result = task._final_decision(_metrics([20] * 10), _metrics([30] * 10), valid=False)
    assert result["status"] == "INVALID_HARNESS"
    assert result["classification"] == "INVALID_HARNESS"
    assert result["training_recommendation"] == "NONE"
    assert result["next_action"] == "NONE"


def test_training_and_adam_boundaries_are_fixed() -> None:
    assert task.SOURCE_STEP == 1700
    assert task.FINAL_STEP == 3000
    assert task.ADDED_UPDATES == 1300
    assert task.BATCH_COUNT == 20
    assert task.ADDED_UPDATES // task.BATCH_COUNT == 65
    assert task._checkpoint_name(1700) == "generator_step1700.pt"
    assert task._checkpoint_name(3000) == "generator_step3000.pt"


def test_preflight_repeatability_uses_declared_numeric_tolerance() -> None:
    left = {"trajectory": {"phase": torch.tensor([1e-12, -2e-12], dtype=torch.float64), "step": 1703}}
    within_tolerance = {"trajectory": {"phase": torch.tensor([1e-12 + task.ATOL / 2, -2e-12], dtype=torch.float64), "step": 1703}}
    outside_tolerance = {"trajectory": {"phase": torch.tensor([1e-12 + task.ATOL * 20, -2e-12], dtype=torch.float64), "step": 1703}}

    passed, detail = task._aa_repeatability(left, within_tolerance)
    assert not torch.equal(left["trajectory"]["phase"], within_tolerance["trajectory"]["phase"])
    assert passed
    assert detail["children"][0]["children"][0]["rtol"] == task.RTOL
    assert detail["children"][0]["children"][0]["atol"] == task.ATOL

    failed, failure = task._aa_repeatability(left, outside_tolerance)
    assert not failed
    assert failure["max_abs_difference"] > task.ATOL


def test_ready_consumer_recomputes_saved_target_positive_control(tmp_path) -> None:
    labels = _balanced_labels()
    indices = torch.arange(10_000, 10_300)
    logits = torch.zeros((300, 10), dtype=torch.float32)
    logits.scatter_(1, labels[:, None], 2.0)
    evaluators = {
        name: {"per_class_correct": [30] * 10, "macro_recall": 1.0, "passed": True}
        for name in task.EVALUATOR_NAMES
    }
    payload = {
        "source_row_indices": indices.clone(),
        "labels": labels.clone(),
        **{f"{name.lower()}_logits": logits.clone() for name in task.EVALUATOR_NAMES},
    }
    target = tmp_path
    torch.save(payload, target / "target_image_evaluator_positive_controls.pt")
    population = {"role_positions": {"new_confirmation": [0, 300]}}
    inputs = {"source_indices": indices, "labels": labels}
    preflight = {"checks": {"new_confirmation_target_positive_control": {"executed": True, "passed": True, "evaluators": evaluators}}}

    result = task._validate_saved_target_positive_control(target, inputs, population, preflight)
    assert result["passed"]
    assert result["evaluators"] == evaluators

    payload["source_row_indices"][0] += 1
    torch.save(payload, target / "target_image_evaluator_positive_controls.pt")
    try:
        task._validate_saved_target_positive_control(target, inputs, population, preflight)
    except task.TenClassAccumulationError:
        return
    raise AssertionError("READY consumer accepted target-control logits with a mismatched row mapping")


def test_source_adam_state_is_copied_and_next_update_is_1701() -> None:
    parameter = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=1.0e-3, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0)
    source = {
        "param_groups": [{"params": [0], "lr": 1.0e-3, "betas": (0.9, 0.999), "eps": 1.0e-8, "weight_decay": 0.0}],
        "state": {0: {"step": torch.tensor(1700.0), "exp_avg": torch.full((3,), 0.2, dtype=torch.float64), "exp_avg_sq": torch.full((3,), 0.4, dtype=torch.float64)}},
    }
    source_before = {key: value.clone() for key, value in source["state"][0].items()}
    ablation._load_adam_state(optimizer, parameter, source, expected_step=1700)
    parameter.grad = torch.full_like(parameter, 0.1)
    optimizer.step()
    assert int(optimizer.state[parameter]["step"].item()) == 1701
    torch.testing.assert_close(source["state"][0]["step"], source_before["step"])
    torch.testing.assert_close(source["state"][0]["exp_avg"], source_before["exp_avg"])
    torch.testing.assert_close(source["state"][0]["exp_avg_sq"], source_before["exp_avg_sq"])
    step = 1701
    first = 0.9 * 0.2 + 0.1 * 0.1
    second = 0.999 * 0.4 + 0.001 * 0.01
    expected = -1.0e-3 * (first / (1.0 - 0.9**step)) / ((second / (1.0 - 0.999**step)) ** 0.5 + 1.0e-8)
    torch.testing.assert_close(parameter.detach(), torch.full_like(parameter, expected), rtol=1e-12, atol=1e-12)


def test_resume_consumer_accepts_only_real_step_boundary_and_matching_history(tmp_path) -> None:
    population = {"source_checkpoint": "parent-A1700.pt", "roles": ["fit", "historic", "old_confirmation", "new_confirmation"]}
    source = {
        "checkpoint_path": population["source_checkpoint"],
        "architecture_layout": "fixed-expanded-layout",
        "training_history": [{"update": i} for i in range(1, 1701)],
        "trainable_parameter_state": {"encoder.phase": torch.zeros((2, 1, 100, 100)), "generator.phase": torch.zeros(340_000)},
        "frozen_runtime_state": {},
        "references": {},
        "weights": {},
    }
    history = list(source["training_history"])
    for update in range(1701, 1801):
        batch = (update - 1) % task.BATCH_COUNT
        positions = list(range(batch * task.BATCH_SIZE, (batch + 1) * task.BATCH_SIZE))
        history.append({"update": update, "batch_index": batch, "fit_positions": positions, "training_evaluators": ["E_S", "E_I", "E_C"]})
    checkpoint = {
        "schema": task.CHECKPOINT_SCHEMA,
        "task_id": task.TASK_ID,
        "source_task_id": task.PARENT_TASK_ID,
        "arm": task.ARM,
        "source_checkpoint": population["source_checkpoint"],
        "source_step": 1700,
        "additional_updates": 100,
        "completed_updates": 1800,
        "optimizer_step": 1800,
        "optimizer_parameter_names": ["generator.phase"],
        "architecture_layout": source["architecture_layout"],
        "population": population,
        "trainable_parameter_state": source["trainable_parameter_state"],
        "optimizer_state": {"param_groups": [{"params": [0], "lr": 1.0e-3, "betas": (0.9, 0.999), "eps": 1.0e-8, "weight_decay": 0.0}], "state": {0: {"step": torch.tensor(1800.0), "exp_avg": torch.zeros(340_000), "exp_avg_sq": torch.zeros(340_000)}}},
        "frozen_runtime_state": {},
        "references": {},
        "weights": {},
        "training_history": history,
        "next_batch_index": 1800 % 20,
        "next_batch_cursor": 1800 % 20,
        "numeric_mode": task.NUMERIC_MODE,
        "rng_state": {"python": [], "numpy": [], "torch": torch.zeros(1)},
    }
    checkpoint_path = tmp_path / task._checkpoint_name(1800)
    torch.save(checkpoint, checkpoint_path)
    accepted = task._read_checkpoint(checkpoint_path, tmp_path, population, source)
    assert accepted["optimizer_step"] == 1800

    checkpoint["training_history"][1700]["fit_positions"] = list(range(10, 20))
    torch.save(checkpoint, checkpoint_path)
    bad_history_rejected = False
    try:
        task._read_checkpoint(checkpoint_path, tmp_path, population, source)
    except task.TenClassAccumulationError:
        bad_history_rejected = True
    assert bad_history_rejected, "resume accepted a mismatched next-batch history"

    checkpoint["training_history"][1700]["fit_positions"] = list(range(0, 10))
    torch.save(checkpoint, checkpoint_path)
    (tmp_path / task._checkpoint_name(1900)).write_bytes(b"later durable checkpoint marker")
    baseline = task._read_checkpoint(checkpoint_path, tmp_path, population, source, measurement_baseline=True)
    assert baseline["completed_updates"] == 1800
    stale_resume_rejected = False
    try:
        task._read_checkpoint(checkpoint_path, tmp_path, population, source)
    except task.TenClassAccumulationError:
        stale_resume_rejected = True
    assert stale_resume_rejected, "resume accepted an older checkpoint despite later saved progress"
