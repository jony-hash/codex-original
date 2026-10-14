from __future__ import annotations

import json
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import pytest
from torch import nn

from scripts.research.lightgen import execute_lightgen_2d_generator_evaluator_ablation_comparison_v1 as task
from scripts.research.lightgen import execute_lightgen_2d_generator_large_capacity_quick_comparison_v1 as capacity
from scripts.research.lightgen import diagnose_lightgen_2d_class17_mapping_v2 as diagnostic


def _gradients_from_real_ce() -> tuple[dict[str, dict[str, torch.Tensor]], torch.Tensor, torch.Tensor, torch.Tensor]:
    encoder = torch.tensor([0.13], dtype=torch.float64, requires_grad=True)
    generator = torch.tensor([0.27], dtype=torch.float64, requires_grad=True)
    labels = torch.arange(10, dtype=torch.long)
    base = torch.linspace(-0.4, 0.5, 10, dtype=torch.float64)
    row_scale = torch.linspace(0.4, 1.3, 10, dtype=torch.float64).unsqueeze(1)
    slopes = {
        "E_S": torch.linspace(-0.7, 0.9, 10, dtype=torch.float64),
        "E_I": torch.linspace(0.8, -0.5, 10, dtype=torch.float64),
        "E_C": torch.linspace(-0.2, 0.65, 10, dtype=torch.float64),
    }
    encoder_slopes = {
        "E_S": torch.linspace(0.4, -0.7, 10, dtype=torch.float64),
        "E_I": torch.linspace(-0.3, 0.8, 10, dtype=torch.float64),
        "E_C": torch.linspace(0.9, 0.1, 10, dtype=torch.float64),
    }
    gradients: dict[str, dict[str, torch.Tensor]] = {}
    for name, slope in slopes.items():
        logits = generator * row_scale * slope.unsqueeze(0) + encoder * row_scale * encoder_slopes[name].unsqueeze(0) + base.unsqueeze(0)
        loss = F.cross_entropy(logits, labels)
        grad_encoder, grad_generator = torch.autograd.grad(loss, (encoder, generator), retain_graph=True)
        gradients[name] = {"encoder.phase": grad_encoder, "generator.phase": grad_generator}
    return gradients, encoder, generator, labels


def _summary(correct: list[int], raw_mse: float = 0.25, clipping: float = 0.0) -> dict:
    return {
        "evaluators": {
            "E_D": {
                "by_role": {
                    "confirmation": {
                        "per_class_correct": correct,
                        "macro_recall": sum(correct) / 300.0,
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


def _summary_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    predicted = logits.to(torch.float64).argmax(dim=1)
    correct = [int(((predicted == labels) & (labels == class_id)).sum().item()) for class_id in range(10)]
    return _summary(correct)


def test_two_evaluator_composition_uses_real_ce_gradients_and_excludes_es() -> None:
    gradients, _encoder, _generator, _labels = _gradients_from_real_ce()
    three, unavailable = task._canonical_semantic_gradient(gradients, task.TRAINING_EVALUATORS["A_three_evaluator"])
    assert unavailable is None and three is not None
    reference, unavailable = diagnostic._safe_canonical_gradient(
        {key: gradients[name] for name, key in zip(("E_S", "E_I", "E_C"), ("L_S", "L_I", "L_C"), strict=True)}
    )
    assert unavailable is None and reference is not None
    for name in ("encoder.phase", "generator.phase"):
        torch.testing.assert_close(three[name], reference[name], rtol=2e-12, atol=2e-12)

    two, unavailable = task._canonical_semantic_gradient(gradients, task.TRAINING_EVALUATORS["B_two_evaluator"])
    assert unavailable is None and two is not None
    independent = task._independent_semantic_reference(gradients, ("E_I", "E_C"))
    for name in ("encoder.phase", "generator.phase"):
        torch.testing.assert_close(two[name], independent[name], rtol=2e-12, atol=2e-12)

    altered_es = dict(gradients)
    altered_es["E_S"] = {key: torch.full_like(value, 1.0e6) for key, value in gradients["E_S"].items()}
    two_without_es = task._canonical_semantic_gradient(altered_es, ("E_I", "E_C"))[0]
    for name in ("encoder.phase", "generator.phase"):
        torch.testing.assert_close(two[name], two_without_es[name], rtol=0.0, atol=0.0)
    assert not torch.allclose(three["generator.phase"], two["generator.phase"])


class _ToyEvaluator(nn.Module):
    def __init__(self, shift: float) -> None:
        super().__init__()
        self.register_buffer("slopes", torch.linspace(-0.8, 1.0, 10) + shift)
        self.register_buffer("bias", torch.linspace(0.2, -0.3, 10))

    def forward(self, view: torch.Tensor):
        signal = view.mean(dim=(1, 2, 3))
        logits = signal.unsqueeze(1) * self.slopes + self.bias
        return torch.zeros((view.shape[0], 128), device=view.device), logits


def _toy_microbatch_consumer(encoder: nn.Parameter, generator: nn.Parameter, target: torch.Tensor):
    class Runtime:
        def __init__(self) -> None:
            self.generator = object()

        def forward_from_image(self, images: torch.Tensor):
            batch_size = images.shape[0]
            ports = torch.linspace(-0.5, 0.7, 100, device=images.device).view(1, 100)
            row_mean = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
            coefficients = torch.complex(encoder * (ports + row_mean), encoder * (ports.flip(1) - row_mean))
            raw_level = 0.5 + 0.02 * encoder + 0.03 * generator
            raw = raw_level.expand(batch_size, 1, 200, 200)

            def field(multiplier: float):
                real = (0.5 + 0.01 * encoder + 0.02 * generator) * multiplier
                imag = 0.005 * generator * multiplier
                tensor = torch.complex(real, imag).expand(batch_size, 1, 2, 2)
                return SimpleNamespace(
                    tensor=tensor,
                    plane=SimpleNamespace(pixel_pitch_m=(1.0, 1.0)),
                )

            return SimpleNamespace(
                raw_intensity=raw,
                z_ols=SimpleNamespace(coefficients=coefficients),
                image_input=SimpleNamespace(slm_intensity=images),
                encoder_records=(field(0.8), field(0.9)),
                generator_input_field=field(1.0),
                generator_records=(field(1.1), field(1.2)),
            )

    runtime = Runtime()
    references = SimpleNamespace(
        scale=torch.linspace(0.5, 1.5, 100, dtype=torch.float64, device=target.device),
        power_reference=torch.linspace(0.8, 1.2, 5, dtype=torch.float64, device=target.device),
    )
    return SimpleNamespace(runtime=runtime, references=references)


def test_production_microbatch_gradient_path_uses_logical_batch_ols_reference(monkeypatch) -> None:
    source = capacity._dist_loss_and_ols_gradient
    calls = []

    def checked_reference(coefficients: torch.Tensor, scale: torch.Tensor):
        calls.append(tuple(coefficients.shape))
        return source(coefficients, scale)

    monkeypatch.setattr(capacity, "_dist_loss_and_ols_gradient", checked_reference)
    labels = torch.arange(10, dtype=torch.long)
    images = torch.linspace(0.1, 0.9, 10).view(10, 1, 1, 1).expand(10, 1, 200, 200).clone()
    evaluators = {name: _ToyEvaluator(offset) for name, offset in zip(task.EVALUATOR_NAMES, (0.0, 0.1, -0.2, 0.3), strict=True)}

    for arm in task.ARM_NAMES:
        encoder = nn.Parameter(torch.tensor(0.12))
        generator = nn.Parameter(torch.tensor(-0.07))
        parameters = {"encoder.phase": encoder, "generator.phase": generator}
        consumer = _toy_microbatch_consumer(encoder, generator, images)
        powers, losses, gradients, combined = task._large_microbatch_loss_gradients(
            consumer, parameters, evaluators, images, labels, arm
        )
        assert powers.shape == (10, 5)
        assert set(gradients["semantic"]) == set(task.TRAINING_EVALUATORS[arm])
        assert "E_S_ce" in losses
        assert torch.count_nonzero(gradients["L_dist"]["generator.phase"]) == 0
        assert all(torch.isfinite(value).all() for value in combined.values())
        assert all(parameter.grad is None for parameter in parameters.values())

    assert calls == [(10, 100), (10, 100)]


def test_real_ce_gradient_update_changes_ed_score_and_reaches_final_gate() -> None:
    gradients, _encoder, generator, labels = _gradients_from_real_ce()
    direction, unavailable = task._canonical_semantic_gradient(gradients, ("E_I", "E_C"))
    assert unavailable is None and direction is not None
    before = generator.detach().clone()
    optimizer = torch.optim.SGD([generator], lr=1.0e-3)
    optimizer.zero_grad(set_to_none=True)
    generator.grad = direction["generator.phase"].detach().clone()
    optimizer.step()

    confirmation_labels = torch.arange(10, dtype=torch.long).repeat(30)
    ed_base_logits = torch.zeros((300, 10), dtype=torch.float64)
    ed_base_logits[torch.arange(300), confirmation_labels] = 1.0
    ed_weights = torch.linspace(-0.6, 0.7, 10, dtype=torch.float64)
    logits_before = ed_base_logits + before * ed_weights.unsqueeze(0)
    logits_after = ed_base_logits + generator.detach() * ed_weights.unsqueeze(0)
    score_change = logits_after[0, confirmation_labels[0]] - logits_before[0, confirmation_labels[0]]
    assert torch.isfinite(score_change)
    assert score_change != 0
    assert not torch.equal(before, generator.detach())

    result = task._final_decision(
        {
            "B_step1000": _summary_from_logits(logits_before, confirmation_labels),
            "A_step1700": _summary_from_logits(logits_before, confirmation_labels),
            "B_step1700": _summary_from_logits(logits_after, confirmation_labels),
        },
        measurement_valid=True,
    )
    assert result["classification"] == "NOT_SUPPORTED_WITHIN_700_UPDATES"
    assert result["class_correct"]["B_step1700"] == result["class_correct"]["B_step1000"]


def test_adam_continues_from_1000_matches_independent_step_and_does_not_alias_source() -> None:
    source_parameter = nn.Parameter(torch.tensor([0.2, -0.3], dtype=torch.float64))
    source_optimizer = torch.optim.Adam(
        [source_parameter],
        lr=capacity.GENERATOR_LEARNING_RATE,
        betas=capacity.GENERATOR_BETAS,
        eps=capacity.GENERATOR_EPS,
        weight_decay=capacity.GENERATOR_WEIGHT_DECAY,
    )
    for index in range(1000):
        source_optimizer.zero_grad(set_to_none=True)
        source_parameter.grad = torch.tensor([0.1 + index * 1e-5, -0.2 + index * 2e-5], dtype=torch.float64)
        source_optimizer.step()
    saved = task._cpu_tree(source_optimizer.state_dict())
    old_step = saved["state"][0]["step"].clone()
    first_moment = saved["state"][0]["exp_avg"].clone()
    second_moment = saved["state"][0]["exp_avg_sq"].clone()

    parameter = nn.Parameter(source_parameter.detach().clone())
    optimizer = torch.optim.Adam(
        [parameter],
        lr=capacity.GENERATOR_LEARNING_RATE,
        betas=capacity.GENERATOR_BETAS,
        eps=capacity.GENERATOR_EPS,
        weight_decay=capacity.GENERATOR_WEIGHT_DECAY,
    )
    task._load_adam_state(optimizer, parameter, saved, expected_step=1000)
    gradient = torch.tensor([0.4, -0.15], dtype=torch.float64)
    beta1, beta2 = capacity.GENERATOR_BETAS
    m1 = beta1 * first_moment + (1.0 - beta1) * gradient
    v1 = beta2 * second_moment + (1.0 - beta2) * gradient.square()
    expected = parameter.detach() - capacity.GENERATOR_LEARNING_RATE * (m1 / (1.0 - beta1**1001)) / (
        torch.sqrt(v1 / (1.0 - beta2**1001)) + capacity.GENERATOR_EPS
    )
    optimizer.zero_grad(set_to_none=True)
    parameter.grad = gradient.clone()
    optimizer.step()
    assert int(float(optimizer.state[parameter]["step"].item())) == 1001
    torch.testing.assert_close(parameter.detach(), expected, rtol=1e-11, atol=1e-11)
    assert torch.equal(saved["state"][0]["step"], old_step)
    assert int(float(source_optimizer.state[source_parameter]["step"].item())) == 1000


def test_fixed_schedule_updates_only_twenty_fit_batches_and_stops_at_1700() -> None:
    assert [task._batch_for_update(step) for step in (1000, 1001, 1018, 1019, 1020, 1699)] == [0, 1, 18, 19, 0, 19]
    with pytest.raises(task.EvaluatorAblationRunError, match="1001..1700"):
        task._batch_for_update(1700)
    images = torch.arange(600, dtype=torch.float32).reshape(600, 1, 1, 1).expand(-1, 1, 28, 28).clone()
    labels = torch.arange(10).repeat(60)
    for update in (1000, 1001, 1699):
        batch, _batch_labels, positions = capacity.training_batch(images, labels, task._batch_for_update(update))
        assert positions.tolist() == list(range(task._batch_for_update(update) * 10, task._batch_for_update(update) * 10 + 10))
        assert bool((positions < task.FIT_ROWS).all())
        assert torch.equal(batch, images[positions])


def test_history_row_mapping_rejects_confirmation_positions() -> None:
    labels = torch.arange(10).repeat_interleave(60).to(torch.long)
    source_indices = torch.arange(5000, 5600, dtype=torch.long)
    inputs = {"labels": labels, "source_indices": source_indices}
    record = {
        "update": 1,
        "arm": "A_three_evaluator",
        "batch_index": 0,
        "fit_positions": list(range(10)),
        "source_row_indices": source_indices[:10].tolist(),
        "labels": labels[:10].tolist(),
    }
    task._validate_history_rows([record], inputs, "A_three_evaluator")
    changed = dict(record)
    changed["fit_positions"] = list(range(590, 600))
    with pytest.raises(task.EvaluatorAblationRunError, match="batch mapping"):
        task._validate_history_rows([changed], inputs, "A_three_evaluator")


def test_final_decision_distinguishes_gain_tradeoff_and_no_support() -> None:
    start = [15, 5, 14, 12, 20, 13, 12, 8, 23, 5]
    a1700 = [15, 8, 14, 12, 20, 13, 12, 10, 23, 5]
    supported = [18, 11, 19, 17, 23, 18, 16, 13, 20, 7]
    summaries = {"B_step1000": _summary(start), "A_step1700": _summary(a1700), "B_step1700": _summary(supported)}
    result = task._final_decision(summaries, measurement_valid=True)
    assert result["classification"] == "EVALUATOR_ABLATION_GAIN_SUPPORTED"
    assert all(result["conditions"].values())

    class8_tradeoff = dict(summaries)
    class8_tradeoff["B_step1700"] = _summary([18, 11, 19, 17, 23, 18, 16, 13, 19, 7])
    tradeoff = task._final_decision(class8_tradeoff, measurement_valid=True)
    assert tradeoff["classification"] == "MIXED_OR_INSUFFICIENT"
    assert tradeoff["conditions"]["class_8_at_least_20_of_30"] is False

    no_gain = dict(summaries)
    no_gain["B_step1700"] = _summary(start)
    rejected = task._final_decision(no_gain, measurement_valid=True)
    assert rejected["classification"] == "NOT_SUPPORTED_WITHIN_700_UPDATES"

    invalid = task._final_decision(summaries, measurement_valid=False)
    assert invalid["classification"] == "INVALID_HARNESS"
    assert invalid["training_recommendation"] is None


def test_preflight_ready_consumer_rejects_unexecuted_or_failed_check(tmp_path) -> None:
    population = {"task_id": task.TASK_ID, "rows": [1, 2]}
    (tmp_path / "input_population.json").write_text(json.dumps(population), encoding="utf-8")
    saved = {
        "task_id": task.TASK_ID,
        "status": "READY",
        "receipt": {},
        "preflight": {"status": "READY", "checks": {"fake": {"executed": True, "passed": False}}},
    }
    (tmp_path / "preflight.json").write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(task.EvaluatorAblationRunError, match="executed passing checks"):
        task._read_ready(tmp_path, population)


def test_source_checkpoint_step_and_checkpoint_names_match_1000_to_1700_contract() -> None:
    assert task.PARENT_UPDATES == 1000
    assert task.ADDED_UPDATES == 700
    assert task.TOTAL_UPDATES == 1700
    assert task._checkpoint_name("A_three_evaluator", 1000) == "arm_a_step1000.pt"
    assert task._checkpoint_name("B_two_evaluator", 1700) == "arm_b_step1700.pt"


def test_explicit_resume_checkpoint_round_trips_and_rejects_later_arm_progress(tmp_path) -> None:
    parameter = nn.Parameter(torch.tensor([0.2, -0.3], dtype=torch.float64))
    optimizer = torch.optim.Adam(
        [parameter],
        lr=capacity.GENERATOR_LEARNING_RATE,
        betas=capacity.GENERATOR_BETAS,
        eps=capacity.GENERATOR_EPS,
        weight_decay=capacity.GENERATOR_WEIGHT_DECAY,
    )
    optimizer.state[parameter] = {
        "step": torch.tensor(1000.0),
        "exp_avg": torch.tensor([0.01, -0.02], dtype=torch.float64),
        "exp_avg_sq": torch.tensor([0.03, 0.04], dtype=torch.float64),
    }
    history = [{"update": step} for step in range(1, 1001)]
    population = {"source_checkpoint_paths": {"shared_B_step1000": "D:/parent/arm_b_step1000.pt"}}
    parameters = {"encoder.phase": torch.tensor([0.1]), "generator.phase": parameter.detach().clone()}
    frozen = {"buffer": torch.tensor([1.0])}
    source = {
        "training_history": history,
        "trainable_parameter_state": parameters,
        "frozen_runtime_state": frozen,
    }
    checkpoint = {
        "schema": task.CHECKPOINT_SCHEMA,
        "task_id": task.TASK_ID,
        "arm": "A_three_evaluator",
        "parent_updates": 1000,
        "additional_updates": 0,
        "completed_updates": 1000,
        "next_batch_index": 0,
        "next_batch_cursor": 0,
        "population": population,
        "source_capacity_checkpoint": population["source_checkpoint_paths"]["shared_B_step1000"],
        "optimizer_step": 1000,
        "optimizer_parameter_names": ["generator.phase"],
        "numeric_mode": dict(task.NUMERIC_MODE),
        "training_history": history,
        "trainable_parameter_state": parameters,
        "optimizer_state": task._cpu_tree(optimizer.state_dict()),
        "rng_state": {"torch_cpu": torch.tensor([1]), "torch_cuda": [], "python": (1, 2, 3)},
        "frozen_runtime_state": frozen,
    }
    path = tmp_path / "arm_a_step1000.pt"
    torch.save(checkpoint, path)
    loaded = task._read_checkpoint(path, tmp_path, population, {arm: source for arm in task.ARM_NAMES})
    assert loaded["completed_updates"] == 1000
    assert loaded["training_history"] == history

    torch.save({}, tmp_path / "arm_a_step1100.pt")
    with pytest.raises(task.EvaluatorAblationRunError, match="predates saved task progress"):
        task._read_checkpoint(path, tmp_path, population, {arm: source for arm in task.ARM_NAMES})


def test_completed_resume_returns_final_b1700_report_without_reopening_training(tmp_path) -> None:
    report = {"task_id": task.TASK_ID, "status": "VALID_FAILED", "classification": "MIXED_OR_INSUFFICIENT"}
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    terminal = {"arm": "B_two_evaluator", "completed_updates": 1700}
    assert task._completed_resume_report(tmp_path, terminal) == report
    with pytest.raises(task.EvaluatorAblationRunError, match="final B checkpoint"):
        task._completed_resume_report(tmp_path, {"arm": "A_three_evaluator", "completed_updates": 1700})
