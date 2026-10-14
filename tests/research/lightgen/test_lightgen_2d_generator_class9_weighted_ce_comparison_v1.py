from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from scripts.research.lightgen import execute_lightgen_2d_generator_class9_weighted_ce_comparison_v1 as task
from scripts.research.lightgen import execute_lightgen_2d_generator_evaluator_ablation_comparison_v1 as ablation
from scripts.research.lightgen import execute_lightgen_2d_generator_large_capacity_quick_comparison_v1 as capacity


def _new_rows() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = torch.arange(10).repeat(6_000)
    prior = torch.arange(900)
    train = torch.arange(60_000)
    return labels, prior, train


def test_new_confirmation_is_seeded_interleaved_and_excludes_all_900_source_rows() -> None:
    labels, prior, train = _new_rows()
    first = task._new_indices_for_test(labels, prior, train)
    second = task._new_indices_for_test(labels, prior, train)
    assert torch.equal(first, second)
    assert first.numel() == 300 and torch.unique(first).numel() == 300
    assert not set(first.tolist()).intersection(prior.tolist())
    assert labels[first].tolist() == list(range(10)) * 30


def test_new_confirmation_rejects_a_source_row_outside_canonical_train() -> None:
    labels, prior, train = _new_rows()
    train = train[train != prior[0]]
    with pytest.raises(task.Class9WeightedRunError):
        task._new_indices_for_test(labels, prior, train)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_weighted_ce_microbatches_use_one_full_batch_denominator_and_match_analytic_vjp(dtype: torch.dtype) -> None:
    torch.manual_seed(2026100810)
    labels = torch.arange(10)
    weights = task._class_weights(labels, task.ARM_NAMES[1]).to(dtype=dtype)
    assert weights.tolist() == [1.0] * 9 + [2.0]
    denominator = weights.sum()
    assert float(denominator) == 11.0
    logits = torch.randn((10, 10), dtype=dtype, requires_grad=True)
    accumulated = torch.zeros_like(logits)
    value = torch.zeros((), dtype=dtype)
    for start in range(0, 10, 2):
        piece = task._weighted_ce_piece(logits[start : start + 2], labels[start : start + 2], weights[start : start + 2], denominator)
        accumulated += torch.autograd.grad(piece, logits, retain_graph=True)[0]
        value += piece.detach()

    expected_value = F.cross_entropy(logits, labels, weight=weights, reduction="mean")
    probabilities = torch.softmax(logits.detach(), dim=1)
    expected_gradient = (probabilities - F.one_hot(labels, num_classes=10).to(dtype)) * weights[:, None] / denominator
    torch.testing.assert_close(value, expected_value, rtol=2e-5, atol=2e-7)
    torch.testing.assert_close(accumulated, expected_gradient, rtol=2e-5, atol=2e-7)

    # The faulty per-microbatch weighted mean changes the actual gradient.
    bad = torch.zeros_like(logits)
    for start in range(0, 10, 2):
        local_weights = weights[start : start + 2]
        local_loss = (F.cross_entropy(logits[start : start + 2], labels[start : start + 2], reduction="none") * local_weights).sum() / local_weights.sum()
        bad += torch.autograd.grad(local_loss, logits, retain_graph=True)[0]
    assert not torch.allclose(bad, expected_gradient, rtol=2e-5, atol=2e-7)


def test_class9_weighting_does_not_accidentally_weight_class8_or_reconstruction() -> None:
    labels = torch.arange(10)
    a = task._class_weights(labels, task.ARM_NAMES[0])
    b = task._class_weights(labels, task.ARM_NAMES[1])
    assert torch.equal(a, torch.ones(10))
    assert b[8].item() == 1.0 and b[9].item() == 2.0
    assert task.ARM_NAMES == ("A_original", "B_class9_weight2")
    assert task.FINAL_STEP == 3700 and task.ADDED_UPDATES == 700


def test_weighted_logit_gradient_matches_centered_finite_difference_and_sign() -> None:
    labels = torch.arange(10)
    weights = task._class_weights(labels, task.ARM_NAMES[1]).to(torch.float64)
    logits = torch.linspace(-0.4, 0.5, 100, dtype=torch.float64).reshape(10, 10).requires_grad_()
    loss = task._weighted_ce_piece(logits, labels, weights, 11.0)
    gradient = torch.autograd.grad(loss, logits)[0]
    direction = torch.linspace(-0.3, 0.2, 100, dtype=torch.float64).reshape(10, 10)
    epsilon = 1e-6
    plus = task._weighted_ce_piece(logits.detach() + epsilon * direction, labels, weights, 11.0)
    minus = task._weighted_ce_piece(logits.detach() - epsilon * direction, labels, weights, 11.0)
    finite_difference = (plus - minus) / (2 * epsilon)
    predicted = (gradient * direction).sum()
    torch.testing.assert_close(finite_difference, predicted, rtol=2e-7, atol=2e-9)
    # Increasing the true-class logits must lower the weighted CE.
    true_class_direction = F.one_hot(labels, num_classes=10).to(torch.float64)
    assert float((gradient * true_class_direction).sum()) < 0.0


def _readout(labels: torch.Tensor, logits: torch.Tensor, *, shift: float) -> dict[str, torch.Tensor]:
    rows = labels.numel()
    target = torch.linspace(0.0, 1.0, 200 * 200).reshape(1, 1, 200, 200).repeat(rows, 1, 1, 1)
    raw = target * 0.4 + shift
    view = F.adaptive_avg_pool2d(raw.clamp(0, 1), (28, 28))
    return {
        "labels": labels,
        "target_slm_intensity": target,
        "raw_cmos": raw,
        "evaluator_view": view,
        **{f"{name.lower()}_logits": logits.clone() for name in task.EVALUATOR_NAMES},
    }


def test_production_weighted_loss_normalization_adam_scoring_and_decision_chain() -> None:
    torch.manual_seed(9)
    labels = torch.arange(10)
    encoder = torch.nn.Parameter(torch.tensor([0.17], dtype=torch.float64))
    generator = torch.nn.Parameter(torch.tensor([0.31, -0.22], dtype=torch.float64))
    features = torch.stack((torch.linspace(-1, 1, 10, dtype=torch.float64), torch.cos(torch.arange(10, dtype=torch.float64))), dim=1)
    class_axes = torch.stack((torch.linspace(-1, 1, 10, dtype=torch.float64), torch.cos(torch.arange(10, dtype=torch.float64)), torch.sin(torch.arange(10, dtype=torch.float64))))
    row_weights = task._class_weights(labels, task.ARM_NAMES[1]).to(torch.float64)
    semantic_gradients = {}
    evaluator_scales = {"E_S": 0.8, "E_I": 1.2, "E_C": 1.6}
    for evaluator, scale in evaluator_scales.items():
        accumulated = {"encoder.phase": torch.zeros_like(encoder), "generator.phase": torch.zeros_like(generator)}
        for start in range(0, 10, 2):
            x = features[start : start + 2]
            logits = (
                scale * x[:, :1] * generator[0] * class_axes[0]
                + scale * x[:, 1:] * generator[1] * class_axes[1]
                + encoder[0] * class_axes[2]
            )
            piece = task._weighted_ce_piece(logits, labels[start : start + 2], row_weights[start : start + 2], 11.0)
            grads = torch.autograd.grad(piece, (encoder, generator), retain_graph=False)
            accumulated["encoder.phase"] += grads[0]
            accumulated["generator.phase"] += grads[1]
        semantic_gradients[evaluator] = accumulated

    composed, unavailable = ablation._canonical_semantic_gradient(semantic_gradients, ("E_S", "E_I", "E_C"))
    assert unavailable is None and composed is not None
    direction = capacity.semantic.KAPPA * composed["generator.phase"]
    optimizer = torch.optim.Adam([generator], lr=1.0e-3, betas=capacity.GENERATOR_BETAS, eps=capacity.GENERATOR_EPS, weight_decay=0.0)
    before = generator.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    generator.grad = direction.detach().clone()
    optimizer.step()
    assert not torch.equal(generator.detach(), before)

    scoring_labels = torch.arange(10).repeat(3)
    def class_logits(phase: torch.Tensor) -> torch.Tensor:
        row = torch.arange(30, dtype=torch.float64)
        axes = torch.arange(10, dtype=torch.float64)
        values = phase[0] * torch.sin(row[:, None] + axes[None, :]) + phase[1] * torch.cos(row[:, None] - axes[None, :])
        values[:, 9] += 0.2 * phase[0]
        return values.to(torch.float32)

    baseline_logits = class_logits(before)
    final_logits = class_logits(generator.detach())
    start_readout = _readout(scoring_labels, baseline_logits, shift=0.0)
    arm_a_readout = _readout(scoring_labels, baseline_logits, shift=0.0)
    arm_b_readout = _readout(scoring_labels, final_logits, shift=0.001)
    population = {"role_positions": {"new_confirmation": [0, 30]}}
    start_metrics, _ = task._row_and_image_metrics(start_readout, population)
    arm_a_metrics, _ = task._row_and_image_metrics(arm_a_readout, population)
    arm_b_metrics, _ = task._row_and_image_metrics(arm_b_readout, population)
    decision = task._final_decision(start_metrics, arm_a_metrics, arm_b_metrics, valid=True)
    expected_b = arm_b_metrics["evaluators"]["E_D"]["by_role"]["new_confirmation"]["per_class_correct"]
    assert decision["new_confirmation"]["per_class_correct"]["B_step3700"] == expected_b
    assert decision["classification"] != "CLASS9_WEIGHTED_CE_GAIN_SUPPORTED"
    assert decision["next_action"] == "NONE"


def test_final_decision_uses_class_id_9_and_keeps_invalid_results_unclassified() -> None:
    def summary(counts: list[int], raw_mse: float = 0.1) -> dict:
        return {
            "evaluators": {"E_D": {"by_role": {"new_confirmation": {"per_class_correct": counts, "correct_total": sum(counts), "macro_recall": sum(counts) / 300.0}}}},
            "image_metrics_by_role": {"new_confirmation": {"raw_mse": raw_mse, "raw_fraction_above_one": 0.0, "raw_zero_image": 0, "raw_constant_image": 0, "by_class": {str(c): {"raw_fraction_above_one": 0.0} for c in range(10)}}},
            "all_finite": True,
        }

    source = summary([22] * 10)
    arm_a = summary([23] * 10)
    supported_counts = [24] * 10
    supported_counts[9] = 26
    supported_b = summary(supported_counts)
    supported = task._final_decision(source, arm_a, supported_b, valid=True)
    assert supported["classification"] == "CLASS9_WEIGHTED_CE_GAIN_SUPPORTED"
    assert supported["new_confirmation"]["per_class_correct"]["B_step3700"][9] == 26

    mixed_b = summary(supported_counts, raw_mse=0.2)
    mixed = task._final_decision(source, arm_a, mixed_b, valid=True)
    assert mixed["classification"] == "MIXED_OR_INSUFFICIENT"
    assert mixed["conditions"]["class9_at_least_15_and_plus_3_vs_both"]
    assert not mixed["conditions"]["raw_mse_clipping_and_image_integrity"]

    invalid = task._final_decision(source, arm_a, supported_b, valid=False)
    assert invalid["status"] == "INVALID_HARNESS"
    assert invalid["classification"] == "INVALID_HARNESS"
    assert invalid["training_recommendation"] == "NONE"
    assert invalid["next_action"] == "NONE"


def test_source_adam_deep_copy_preserves_3000_step_state_while_next_step_advances() -> None:
    parameter = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    optimizer = torch.optim.Adam([parameter], lr=1.0e-3, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0)
    source_state = {
        "param_groups": [{"params": [0], "lr": 1.0e-3, "betas": (0.9, 0.999), "eps": 1.0e-8, "weight_decay": 0.0}],
        "state": {0: {"step": torch.tensor(3000.0), "exp_avg": torch.full((3,), 0.2, dtype=torch.float64), "exp_avg_sq": torch.full((3,), 0.4, dtype=torch.float64)}},
    }
    before = task._cpu_tree(source_state)
    ablation._load_adam_state(optimizer, parameter, source_state, expected_step=3000)
    parameter.grad = torch.full_like(parameter, 0.1)
    optimizer.step()
    assert int(float(optimizer.state[parameter]["step"].item())) == 3001
    assert task._tree_equal(source_state, before)


def test_explicit_resume_checkpoint_rejects_wrong_class_weight_policy(tmp_path) -> None:
    rows = list(range(1200))
    labels = [index % 10 for index in rows]
    population = {"source_checkpoint": "parent.pt", "source_row_indices": rows, "labels": labels}
    fit_history = []
    for update in range(1, 3001):
        batch = (update - 1) % 20
        positions = list(range(batch * 10, (batch + 1) * 10))
        fit_history.append({"update": update, "batch_index": batch, "fit_positions": positions, "source_row_indices": rows[:200][batch * 10 : (batch + 1) * 10], "labels": labels[:200][batch * 10 : (batch + 1) * 10]})
    source = {
        "checkpoint_path": "parent.pt",
        "architecture_layout": "two_original_100x100_then_two_400x400_phase_planes_flattened",
        "training_history": fit_history,
        "trainable_parameter_state": {"encoder.phase": torch.zeros((2, 1, 100, 100)), "generator.phase": torch.zeros(340_000)},
        "frozen_runtime_state": {},
        "references": {},
        "weights": {},
        "rng_state": {"python": [], "numpy": [], "torch": torch.zeros(1)},
    }
    history = list(fit_history)
    for update in range(3001, 3101):
        batch = (update - 1) % 20
        positions = list(range(batch * 10, (batch + 1) * 10))
        history.append({"update": update, "arm": task.ARM_NAMES[1], "batch_index": batch, "fit_positions": positions, "source_row_indices": rows[batch * 10 : (batch + 1) * 10], "labels": labels[batch * 10 : (batch + 1) * 10], "training_evaluators": ["E_S", "E_I", "E_C"], "semantic_weight_policy": "class9_x2", "class_weights_by_label": [1.0] * 9 + [2.0], "weighted_ce_denominator": 11.0})
    checkpoint = {
        "schema": task.CHECKPOINT_SCHEMA,
        "task_id": task.TASK_ID,
        "source_task_id": task.PARENT_TASK_ID,
        "arm": task.ARM_NAMES[1],
        "semantic_weight_policy": "class9_x2",
        "source_checkpoint": "parent.pt",
        "source_step": 3000,
        "additional_updates": 100,
        "completed_updates": 3100,
        "optimizer_step": 3100,
        "optimizer_parameter_names": ["generator.phase"],
        "architecture_layout": source["architecture_layout"],
        "population": population,
        "trainable_parameter_state": source["trainable_parameter_state"],
        "optimizer_state": {"param_groups": [{"params": [0], "lr": 1.0e-3, "betas": (0.9, 0.999), "eps": 1.0e-8, "weight_decay": 0.0}], "state": {0: {"step": torch.tensor(3100.0), "exp_avg": torch.zeros(340_000), "exp_avg_sq": torch.zeros(340_000)}}},
        "frozen_runtime_state": {},
        "references": {},
        "weights": {},
        "training_history": history,
        "source_history_count": 3000,
        "source_rng_state": source["rng_state"],
        "rng_state": source["rng_state"],
        "next_batch_index": 0,
        "next_batch_cursor": 0,
        "numeric_mode": task.NUMERIC_MODE,
    }
    (tmp_path / task._checkpoint_name(task.ARM_NAMES[0], task.FINAL_STEP)).write_bytes(b"A final boundary")
    path = tmp_path / task._checkpoint_name(task.ARM_NAMES[1], 3100)
    torch.save(checkpoint, path)
    loaded = task._read_checkpoint(path, tmp_path, population, source)
    assert loaded["completed_updates"] == 3100

    checkpoint["training_history"][3000]["class_weights_by_label"] = [1.0] * 8 + [2.0, 1.0]
    torch.save(checkpoint, path)
    with pytest.raises(task.Class9WeightedRunError):
        task._read_checkpoint(path, tmp_path, population, source)
