from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
import pytest
from torch import nn

from scripts.research.lightgen import execute_lightgen_2d_generator_capacity_accumulation_comparison_v1 as task
from scripts.research.lightgen import execute_lightgen_2d_generator_large_capacity_quick_comparison_v1 as capacity


class _Phase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.phase = nn.Parameter(torch.zeros((2, 1, 100, 100)))


class _Runtime(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Phase()
        self.generator = _Phase()
        self.register_buffer("frozen_buffer", torch.tensor([3.0]))


class _Payload:
    def to_payload(self):
        return {"frozen": torch.tensor([1.0])}


class _Consumer:
    def __init__(self) -> None:
        self.runtime = _Runtime()
        self.device = torch.device("cpu")
        self.references = _Payload()
        self.weights = _Payload()


def _adam(step_count: int = 300):
    parameter = nn.Parameter(torch.tensor([0.2, -0.3], dtype=torch.float64))
    optimizer = torch.optim.Adam(
        [parameter],
        lr=capacity.GENERATOR_LEARNING_RATE,
        betas=capacity.GENERATOR_BETAS,
        eps=capacity.GENERATOR_EPS,
        weight_decay=capacity.GENERATOR_WEIGHT_DECAY,
    )
    for step in range(step_count):
        optimizer.zero_grad(set_to_none=True)
        parameter.grad = torch.tensor([0.1 + step * 1.0e-4, -0.2 + step * 2.0e-4], dtype=parameter.dtype)
        optimizer.step()
    return parameter, optimizer


def _summary(correct: list[int], macro: float, raw_mse: float = 0.25, clipping: float = 0.0):
    return {
        "evaluators": {
            "E_D": {
                "by_role": {
                    "confirmation": {
                        "per_class_correct": correct,
                        "macro_recall": macro,
                    }
                }
            }
        },
        "image_metrics_by_role": {
            "confirmation": {
                "overall": {"raw_mse": raw_mse, "raw_fraction_above_one": clipping},
                "1": {"raw_fraction_above_one": clipping},
                "7": {"raw_fraction_above_one": clipping},
            }
        },
        "all_finite": True,
        "has_zero_image": False,
        "has_constant_image": False,
    }


def test_adam_state_continues_from_300_and_matches_independent_bias_corrected_step() -> None:
    source_parameter, source_optimizer = _adam(300)
    saved = source_optimizer.state_dict()
    restored_parameter = nn.Parameter(source_parameter.detach().clone())
    restored_optimizer = torch.optim.Adam(
        [restored_parameter],
        lr=capacity.GENERATOR_LEARNING_RATE,
        betas=capacity.GENERATOR_BETAS,
        eps=capacity.GENERATOR_EPS,
        weight_decay=capacity.GENERATOR_WEIGHT_DECAY,
    )
    task._load_adam_state(restored_optimizer, restored_parameter, saved, expected_step=300)
    assert int(float(saved["state"][0]["step"].item())) == 300

    m0 = saved["state"][0]["exp_avg"].clone()
    v0 = saved["state"][0]["exp_avg_sq"].clone()
    initial = restored_parameter.detach().clone()
    gradient = torch.tensor([0.4, -0.15], dtype=torch.float64)
    beta1, beta2 = capacity.GENERATOR_BETAS
    m1 = beta1 * m0 + (1.0 - beta1) * gradient
    v1 = beta2 * v0 + (1.0 - beta2) * gradient.square()
    expected = initial - capacity.GENERATOR_LEARNING_RATE * (m1 / (1.0 - beta1**301)) / (
        torch.sqrt(v1 / (1.0 - beta2**301)) + capacity.GENERATOR_EPS
    )
    restored_optimizer.zero_grad(set_to_none=True)
    restored_parameter.grad = gradient.clone()
    restored_optimizer.step()
    assert int(float(restored_optimizer.state[restored_parameter]["step"].item())) == 301
    assert int(float(saved["state"][0]["step"].item())) == 300
    torch.testing.assert_close(restored_parameter.detach(), expected, rtol=1e-12, atol=1e-12)


def test_adam_continuation_rejects_reset_or_wrong_step_state() -> None:
    source_parameter, source_optimizer = _adam(300)
    saved = source_optimizer.state_dict()
    parameter = nn.Parameter(source_parameter.detach().clone())
    optimizer = torch.optim.Adam(
        [parameter],
        lr=capacity.GENERATOR_LEARNING_RATE,
        betas=capacity.GENERATOR_BETAS,
        eps=capacity.GENERATOR_EPS,
        weight_decay=capacity.GENERATOR_WEIGHT_DECAY,
    )
    reset = {"state": {}, "param_groups": saved["param_groups"]}
    with pytest.raises(task.AccumulationRunError, match="state is absent"):
        task._load_adam_state(optimizer, parameter, reset, expected_step=300)

    wrong_step = {"state": {0: dict(saved["state"][0])}, "param_groups": [dict(saved["param_groups"][0])]}
    wrong_step["state"][0]["step"] = torch.tensor(299.0)
    with pytest.raises(task.AccumulationRunError, match="expected 300"):
        task._load_adam_state(optimizer, parameter, wrong_step, expected_step=300)


def test_next_batch_after_300_replays_original_20_batch_schedule_and_fails_closed() -> None:
    assert [task._batch_for_update(step) for step in (300, 301, 318, 319, 320)] == [0, 1, 18, 19, 0]
    with pytest.raises(task.AccumulationRunError, match="301..1000"):
        task._batch_for_update(1000)


def test_source_history_row_mapping_rejects_confirmation_and_wrong_fit_rows() -> None:
    labels = torch.arange(10).repeat_interleave(20).to(torch.long)
    images = torch.zeros((600, 1, 28, 28))
    source_indices = torch.arange(1000, 1600, dtype=torch.long)
    inputs = {"images": images, "labels": labels, "source_indices": source_indices}
    record = {
        "update": 1,
        "arm": "A_small",
        "batch_index": 0,
        "fit_positions": list(range(10)),
        "source_row_indices": source_indices[:10].tolist(),
        "labels": labels[:10].tolist(),
    }
    task._validate_history_rows([record], inputs, "A_small")
    wrong = dict(record)
    wrong["fit_positions"] = list(range(590, 600))
    with pytest.raises(task.AccumulationRunError, match="batch mapping"):
        task._validate_history_rows([wrong], inputs, "A_small")


def test_large_arm_microbatch_has_independent_cpu_reference() -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = capacity._verify_large_microbatch_gradient_identity()
    finally:
        torch.set_num_threads(previous_threads)
    assert result["passed"] is True
    assert result["wrong_broadcast_negative_control_rejected"] is True


def test_cumulative_gate_uses_shared_zero_start_and_rejects_class8_tradeoff() -> None:
    start = [20, 0, 15, 8, 20, 8, 10, 0, 24, 5]
    a1000 = [21, 1, 15, 8, 20, 8, 10, 1, 24, 5]
    b1000 = [22, 4, 15, 8, 20, 8, 10, 4, 24, 5]
    summaries = {
        "common_start": _summary(start, 0.40, 0.25),
        "A_step1000": _summary(a1000, 0.42, 0.25),
        "B_step1000": _summary(b1000, 0.47, 0.25),
    }
    supported = task._final_decision(summaries, measurement_valid=True)
    assert supported["classification"] == "CUMULATIVE_CAPACITY_GAIN_SUPPORTED"
    assert supported["conditions"]["class_1_and_7_each_gain_at_least_3_over_A1000_and_common_start"]

    tradeoff = dict(summaries)
    tradeoff["B_step1000"] = _summary([22, 4, 11, 8, 20, 8, 10, 4, 20, 5], 0.47, 0.25)
    rejected = task._final_decision(tradeoff, measurement_valid=True)
    assert rejected["classification"] == "MIXED_OR_INSUFFICIENT"
    assert not rejected["conditions"]["no_class_loses_more_than_3_vs_A1000_or_common_start"]


def test_cumulative_gate_does_not_confuse_a300_with_the_common_zero_baseline() -> None:
    summaries = {
        "common_start": _summary([20, 2, 15, 8, 20, 8, 10, 2, 24, 5], 0.40),
        "A_step1000": _summary([20, 0, 15, 8, 20, 8, 10, 0, 24, 5], 0.39),
        "B_step1000": _summary([20, 3, 15, 8, 20, 8, 10, 3, 24, 5], 0.43),
    }
    result = task._final_decision(summaries, measurement_valid=True)
    assert result["classification"] == "MIXED_OR_INSUFFICIENT"
    assert result["focus_class_correct_deltas"]["1"] == {"vs_A_step1000": 3, "vs_common_zero_start": 1}


def test_resume_checkpoint_order_keeps_a_before_b_and_rejects_stale_selection() -> None:
    assert task._checkpoint_order("A_small", 300) == task._checkpoint_order("B_large", 300)
    assert task._checkpoint_order("A_small", 1000) < task._checkpoint_order("B_large", 400)
    assert task._checkpoint_order("B_large", 400) < task._checkpoint_order("B_large", 500)


def test_ready_staging_directory_can_open_same_scientific_attempt(tmp_path: Path) -> None:
    identity = task.ResearchRunIdentity(
        task_id=task.TASK_ID,
        run_id=f"{task.TASK_ID}-attempt-01",
        trajectory_id="fixed-small-large-generator-cumulative-1000-updates",
        semantic_authority_id=task.SOURCE_TASK_ID,
        data_role="canonical TRAIN fixed fit, historic and confirmation roles",
        config_identity={},
        model_identity={},
        optimizer_identity={},
        sampler_identity={},
        execution_source_semantics_id=task.TASK_ID,
        scientific_attempt_id=f"{task.TASK_ID}:attempt-01",
    )
    control = task.ResearchRunControl(identity=identity, authorization_id=f"{task.TASK_ID}:authorization:01")
    receipt = control.make_ready_receipt(
        correctness={"passed": True},
        resource={"passed": True},
        implementation={"passed": True},
        selected_engine="cuda",
    )
    control.mark_preflight_ready(receipt)
    control.start_execution()
    (tmp_path / "input_population.json").write_text("{}", encoding="utf-8")
    opened = control.open_scientific_attempt(
        tmp_path,
        f"{task.TASK_ID}:attempt-01",
        receipt,
        resume=True,
    )
    assert opened.scientific_attempt_id == f"{task.TASK_ID}:attempt-01"
    assert control.state == "SCIENTIFIC_ACTIVE"


def test_completed_training_returns_checkpoint_payload_for_endpoint_measurement(monkeypatch, tmp_path: Path) -> None:
    consumer = _Consumer()
    parameters = capacity._parameters(consumer)
    generator_parameter = parameters["generator.phase"]
    optimizer = torch.optim.Adam(
        [generator_parameter],
        lr=capacity.GENERATOR_LEARNING_RATE,
        betas=capacity.GENERATOR_BETAS,
        eps=capacity.GENERATOR_EPS,
        weight_decay=capacity.GENERATOR_WEIGHT_DECAY,
    )
    optimizer.state[generator_parameter] = {
        "step": torch.tensor(1000.0),
        "exp_avg": torch.zeros_like(generator_parameter),
        "exp_avg_sq": torch.zeros_like(generator_parameter),
    }
    history = [{"update": step, "arm": "A_small"} for step in range(1, 1001)]
    start_from = {"arm": "A_small", "completed_updates": 1000, "training_history": history}
    source_checkpoint = {"training_history": history, "architecture_layout": "two_original_100x100_phase_planes"}
    population = {"source_checkpoint_paths": {"A_small_step0300": "a.pt"}}
    checkpoint = {"arm": "A_small", "completed_updates": 1000, "training_history": history}
    torch.save(checkpoint, tmp_path / "arm_a_step1000.pt")

    monkeypatch.setattr(task, "_load_arm", lambda *_args, **_kwargs: (consumer, optimizer, parameters))
    monkeypatch.setattr(task, "_restore_checkpoint", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(capacity, "_frozen_runtime_state", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(capacity, "_assert_state_equal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    result = task._train_arm(
        "A_small",
        {},
        {},
        source_checkpoint,
        {},
        tmp_path,
        population,
        start_from=start_from,
        started=0.0,
    )
    assert result["arm"] == "A_small"
    assert result["completed_updates"] == 1000


def test_task_checkpoint_round_trips_cumulative_adam_and_source_history(monkeypatch) -> None:
    with tempfile.TemporaryDirectory(prefix="lgaccum-cp-") as temp_name:
        _check_task_checkpoint_roundtrip(Path(temp_name), monkeypatch)


def _check_task_checkpoint_roundtrip(tmp_path: Path, monkeypatch) -> None:
    saved_rng = {"torch_cpu": torch.tensor([1, 2], dtype=torch.uint8), "torch_cuda": [], "python": (3, 4, 5)}
    monkeypatch.setattr(capacity, "_rng_state", lambda: saved_rng)
    restored_rng = {}
    monkeypatch.setattr(task.capacity, "_restore_rng_state", lambda value: restored_rng.update(value))
    consumer = _Consumer()
    parameters = capacity._parameters(consumer)
    phase = parameters["generator.phase"]
    optimizer = torch.optim.Adam(
        [phase],
        lr=capacity.GENERATOR_LEARNING_RATE,
        betas=capacity.GENERATOR_BETAS,
        eps=capacity.GENERATOR_EPS,
        weight_decay=capacity.GENERATOR_WEIGHT_DECAY,
    )
    optimizer.state[phase] = {
        "step": torch.tensor(300.0),
        "exp_avg": torch.ones_like(phase) * 0.01,
        "exp_avg_sq": torch.ones_like(phase) * 0.02,
    }
    history = [{"update": step, "arm": "A_small"} for step in range(1, 301)]
    source_state = {
        "schema": capacity.CHECKPOINT_SCHEMA,
        "task_id": task.SOURCE_TASK_ID,
        "arm": "A_small",
        "completed_updates": 300,
        "optimizer_step": 300,
        "source_checkpoint_step": capacity.SOURCE_STEP,
        "source_optimizer_steps": {"encoder.phase": 2200, "generator.phase": 2500},
        "optimizer_parameter_names": ["generator.phase"],
        "architecture_layout": "two_original_100x100_phase_planes",
        "next_batch_index": 0,
        "next_batch_cursor": 0,
        "numeric_mode": dict(task.NUMERIC_MODE),
        "population": {"source": "quick-task"},
        "trainable_parameter_state": capacity._snapshot_parameters(parameters),
        "optimizer_state": capacity._cpu_tree(optimizer.state_dict()),
        "frozen_runtime_state": capacity._frozen_runtime_state(consumer),
        "training_history": history,
        "references": {"frozen": torch.tensor([1.0])},
        "weights": {"frozen": torch.tensor([1.0])},
        "rng_state": capacity._rng_state(),
    }
    population = {
        "source_checkpoint_paths": {
            "A_small_step0300": "D:/parent/arm_a_step0300.pt",
            "B_large_step0300": "D:/parent/arm_b_step0300.pt",
        }
    }
    payload = task._checkpoint_payload(consumer, optimizer, "A_small", 300, history, population, source_state)
    path = tmp_path / "arm_a_step0300.pt"
    torch.save(payload, path)

    source_parameters = source_state["trainable_parameter_state"]
    source_optimizer = source_state["optimizer_state"]
    source = dict(source_state)
    source["population"] = dict(source_state["population"])
    loaded = task._read_checkpoint(path, tmp_path, population, {"A_small": source, "B_large": source})
    assert loaded["completed_updates"] == 300
    assert loaded["additional_updates"] == 0
    assert loaded["optimizer_step"] == 300
    assert _tree_equal(loaded["training_history"], history)
    assert _tree_equal(loaded["trainable_parameter_state"], source_parameters)
    assert _tree_equal(loaded["optimizer_state"], source_optimizer)
    restored_consumer = _Consumer()
    restored_parameters = capacity._parameters(restored_consumer)
    restored_optimizer = torch.optim.Adam(
        [restored_parameters["generator.phase"]],
        lr=capacity.GENERATOR_LEARNING_RATE,
        betas=capacity.GENERATOR_BETAS,
        eps=capacity.GENERATOR_EPS,
        weight_decay=capacity.GENERATOR_WEIGHT_DECAY,
    )
    task._restore_checkpoint(loaded, restored_consumer, restored_optimizer, "A_small")
    assert _tree_equal(capacity._snapshot_parameters(restored_parameters), source_parameters)
    assert int(float(restored_optimizer.state[restored_parameters["generator.phase"]]["step"].item())) == 300
    assert _tree_equal(restored_rng, saved_rng)


def _tree_equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def test_completed_resume_returns_only_the_final_b1000_report() -> None:
    with tempfile.TemporaryDirectory(prefix="lgaccum-report-") as temp_name:
        _check_completed_resume_report(Path(temp_name))


def _check_completed_resume_report(tmp_path: Path) -> None:
    report = {"task_id": task.TASK_ID, "status": "VALID_FAILED", "classification": "NOT_SUPPORTED_WITHIN_1000_UPDATES"}
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    final_checkpoint = {"arm": "B_large", "completed_updates": 1000}
    assert task._completed_resume_report(tmp_path, final_checkpoint) == report
    with pytest.raises(task.AccumulationRunError, match="final B checkpoint"):
        task._completed_resume_report(tmp_path, {"arm": "A_small", "completed_updates": 1000})
