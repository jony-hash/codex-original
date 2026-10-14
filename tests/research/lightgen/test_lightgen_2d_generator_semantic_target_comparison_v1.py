from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from scripts.research.lightgen import execute_lightgen_2d_generator_semantic_target_comparison_v1 as task


def _three_evaluators() -> dict[str, nn.Module]:
    torch.manual_seed(17)

    class TinyEvaluator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Linear(28 * 28, 128)
            self.classifier = nn.Linear(128, 10)
            for parameter in self.parameters():
                parameter.requires_grad_(False)

        def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            features = torch.tanh(self.embedding(images.flatten(1)))
            return features, self.classifier(features)

    return {name: TinyEvaluator() for name in ("E_S", "E_I", "E_C")}


def _role_labels() -> torch.Tensor:
    return torch.cat((torch.arange(10, dtype=torch.long).repeat(20), torch.arange(10, dtype=torch.long).repeat(10), torch.arange(10, dtype=torch.long).repeat_interleave(30)))


def _predictions_with_correct_per_class(labels: torch.Tensor, correct_counts: list[int]) -> torch.Tensor:
    predictions = labels.clone()
    confirmation_start = task.FIT_ROWS + task.HISTORIC_ROWS
    confirmation_labels = labels[confirmation_start:]
    for class_id, count in enumerate(correct_counts):
        positions = torch.nonzero(confirmation_labels == class_id, as_tuple=False).flatten()
        wrong_positions = positions[count:] + confirmation_start
        predictions[wrong_positions] = (class_id + 1) % 10
    return predictions


def _image_statistics(raw_mse: float = 0.1) -> dict[str, object]:
    by_role = {}
    for role in ("fit", "historic_holdout", "confirmation"):
        by_role[role] = {
            "overall": {"raw_mse": raw_mse, "raw_fraction_above_one": 0.0, "raw_zero_image": 0, "raw_constant_image": 0},
            **{str(class_id): {"raw_mse": raw_mse, "raw_fraction_above_one": 0.0, "raw_zero_image": 0, "raw_constant_image": 0} for class_id in range(10)},
        }
    return {"by_role": by_role}


def _decision_inputs() -> tuple[torch.Tensor, dict[str, dict[str, object]], dict[str, dict[str, dict[str, object]]], dict[str, dict[str, object]], dict[str, object]]:
    labels = _role_labels()
    target_predictions = labels.clone()
    target = task._class_metrics(labels, torch.nn.functional.one_hot(target_predictions, num_classes=10).float())
    baseline = _predictions_with_correct_per_class(labels, [18] * 10)
    improved = _predictions_with_correct_per_class(labels, [20, 24, 20, 20, 20, 20, 20, 24, 20, 20])
    endpoints = {
        "step2200": {"E_D": task._class_metrics(labels, torch.nn.functional.one_hot(baseline, num_classes=10).float())},
        task.ARM_NAMES[0]: {"E_D": task._class_metrics(labels, torch.nn.functional.one_hot(baseline, num_classes=10).float())},
        task.ARM_NAMES[1]: {"E_D": task._class_metrics(labels, torch.nn.functional.one_hot(improved, num_classes=10).float())},
    }
    a_correct = baseline == labels
    b_correct = improved == labels
    confirmation = slice(task.FIT_ROWS + task.HISTORIC_ROWS, task.TOTAL_ROWS)
    pair = {
        str(class_id): task._exact_mcnemar_better(
            a_correct[confirmation][labels[confirmation] == class_id],
            b_correct[confirmation][labels[confirmation] == class_id],
        )
        for class_id in task.FOCUS_CLASSES
    }
    paired = {"raw": pair, "holm_adjusted_p": task._holm({key: value["one_sided_exact_p"] for key, value in pair.items()})}
    stats = {"step2200": _image_statistics(), task.ARM_NAMES[0]: _image_statistics(), task.ARM_NAMES[1]: _image_statistics()}
    return labels, {name: target for name in task.EVALUATOR_NAMES}, endpoints, stats, paired


def test_confirmation_population_is_seeded_train_only_disjoint_and_interleaved() -> None:
    labels = torch.arange(60_000, dtype=torch.long).remainder(10)
    old = torch.arange(300, dtype=torch.long)
    train = torch.arange(60_000, dtype=torch.long)

    first = task._confirmation_indices(labels, old, train)
    second = task._confirmation_indices(labels, old, train)

    assert torch.equal(first, second)
    assert first.shape == (300,)
    assert torch.unique(first).numel() == 300
    assert not set(first.tolist()).intersection(old.tolist())
    assert torch.bincount(labels[first], minlength=10).tolist() == [30] * 10
    assert torch.equal(labels[first].reshape(30, 10), torch.arange(10).expand(30, 10))


def test_confirmation_population_rejects_non_train_candidates() -> None:
    labels = torch.arange(60_000, dtype=torch.long).remainder(10)
    old = torch.arange(300, dtype=torch.long)
    train = torch.nonzero(labels != 9, as_tuple=False).flatten()
    with pytest.raises(task.SemanticTargetRunError, match="fewer than 30"):
        task._confirmation_indices(labels, old, train)


def test_training_batch_can_only_select_original_fit_rows() -> None:
    labels = torch.cat((torch.arange(10).repeat(20), torch.arange(10).repeat(10), torch.arange(10).repeat(30))).long()
    images = torch.arange(task.TOTAL_ROWS).reshape(-1, 1).float()
    batch, batch_labels, positions = task.training_batch(images, labels, 7)

    assert positions.tolist() == list(range(70, 80))
    assert batch[:, 0].tolist() == positions.float().tolist()
    assert torch.bincount(batch_labels, minlength=10).tolist() == [1] * 10
    with pytest.raises(task.SemanticTargetRunError, match="canonical TRAIN"):
        task.training_batch(images, labels, 0, data_role="confirmation")
    with pytest.raises(task.SemanticTargetRunError, match="20-batch"):
        task.training_batch(images, labels, 20)


def test_real_class_cross_entropy_has_label_sensitive_frozen_evaluator_gradients() -> None:
    evaluators = _three_evaluators()
    raw = torch.rand(10, 1, 28, 28, requires_grad=True)
    targets = torch.rand(10, 1, 28, 28)
    labels = torch.arange(10)
    features, correct_ce = task._semantic_losses(raw, targets, labels, evaluators)
    wrong_labels = labels.roll(1)
    _features_again, wrong_ce = task._semantic_losses(raw, targets, wrong_labels, evaluators)
    correct_gradient = torch.autograd.grad(sum(correct_ce.values()), raw, retain_graph=True)[0]
    wrong_gradient = torch.autograd.grad(sum(wrong_ce.values()), raw)[0]

    assert set(features) == {"L_S", "L_I", "L_C"}
    assert set(correct_ce) == {"L_S", "L_I", "L_C"}
    assert all(torch.isfinite(value) for value in correct_ce.values())
    assert not torch.allclose(correct_gradient, wrong_gradient)
    assert all(parameter.grad is None for model in evaluators.values() for parameter in model.parameters())


def test_canonical_joint_normalization_and_structural_zero_gradient() -> None:
    encoder = nn.Parameter(torch.tensor([1.0, 2.0]))
    generator = nn.Parameter(torch.tensor([3.0, 4.0]))
    zero_generator = task._gradient(encoder.square().sum(), {"encoder.phase": encoder, "generator.phase": generator}, retain_graph=False)
    assert torch.equal(zero_generator["generator.phase"], torch.zeros_like(generator))

    gradients = {
        name: {"encoder.phase": torch.tensor([scale, 0.0]), "generator.phase": torch.tensor([0.0, scale])}
        for name, scale in (("L_S", 1.0), ("L_I", 2.0), ("L_C", 4.0))
    }
    combined, unavailable = task.diagnostic._safe_canonical_gradient(gradients)
    assert unavailable is None
    assert combined is not None
    expected = task.DELTA_SIC
    assert torch.isclose(combined["encoder.phase"].square().mean().sqrt(), torch.tensor(expected), rtol=1e-5)


def test_generator_only_adam_does_not_update_encoder_or_its_state() -> None:
    encoder = nn.Parameter(torch.tensor([1.0, -2.0]))
    generator = nn.Parameter(torch.tensor([0.5, 3.0]))
    optimizer = torch.optim.Adam([encoder, generator], lr=1e-2)
    (encoder.square().sum() + generator.square().sum()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    parameters = {"encoder.phase": encoder, "generator.phase": generator}
    encoder_before = encoder.detach().clone()
    encoder_state_before = task._cpu_tree(optimizer.state[encoder])
    generator_state_before = float(optimizer.state[generator]["step"].item())

    task.balanced.apply_generator_only_adam_update(optimizer, parameters, torch.ones_like(generator))

    assert torch.equal(encoder, encoder_before)
    assert task._tree_equal(encoder_state_before, task._cpu_tree(optimizer.state[encoder]))
    assert float(optimizer.state[generator]["step"].item()) == generator_state_before + 1
    assert encoder.grad is None and generator.grad is None


def test_confirmation_gate_supports_preregistered_gain_and_rejects_identical_outputs() -> None:
    labels, target, endpoints, statistics, paired = _decision_inputs()
    supported = task._decision(target, {name: value["E_D"] for name, value in endpoints.items()}, statistics, paired)
    assert supported["classification"] == "SEMANTIC_TARGET_TRANSFER_SUPPORTED"
    assert supported["measurement_qualified"] is True

    same_prediction = endpoints[task.ARM_NAMES[0]]["E_D"]["predictions"]
    same_logits = torch.nn.functional.one_hot(same_prediction, num_classes=10).float()
    identical_endpoints = {
        name: {"E_D": task._class_metrics(labels, same_logits)}
        for name in ("step2200", *task.ARM_NAMES)
    }
    confirmation = slice(task.FIT_ROWS + task.HISTORIC_ROWS, task.TOTAL_ROWS)
    same_correct = same_prediction[confirmation] == labels[confirmation]
    pair = {
        str(class_id): task._exact_mcnemar_better(
            same_correct[labels[confirmation] == class_id],
            same_correct[labels[confirmation] == class_id],
        )
        for class_id in task.FOCUS_CLASSES
    }
    no_change = {"raw": pair, "holm_adjusted_p": task._holm({key: value["one_sided_exact_p"] for key, value in pair.items()})}
    rejected = task._decision(target, {name: value["E_D"] for name, value in identical_endpoints.items()}, statistics, no_change)
    assert rejected["classification"] == "NOT_SUPPORTED_WITHIN_SCOPE"
    assert rejected["training_recommendation"] == "NONE"


def test_checkpoint_resume_is_explicit_and_terminal_read_does_not_run_updates(tmp_path) -> None:
    population = {"rows": [1, 2], "roles": ["fit", "confirmation"]}
    payload = {
        "schema": task.CHECKPOINT_SCHEMA,
        "task_id": task.TASK_ID,
        "arm": task.ARM_NAMES[0],
        "completed_updates": 100,
        "next_batch_index": 0,
        "source_checkpoint_step": 2200,
        "population": population,
    }
    checkpoint = tmp_path / task._checkpoint_name(task.ARM_NAMES[0], 100)
    torch.save(payload, checkpoint)
    loaded = task._read_checkpoint(checkpoint, tmp_path, population)
    assert loaded["completed_updates"] == 100

    terminal_payload = {**payload, "arm": task.ARM_NAMES[1], "completed_updates": task.UPDATES_PER_ARM, "next_batch_index": 0}
    terminal_path = tmp_path / task._checkpoint_name(task.ARM_NAMES[1], task.UPDATES_PER_ARM)
    torch.save(terminal_payload, terminal_path)
    (tmp_path / "report.json").write_text(json.dumps({"task_id": task.TASK_ID, "status": "VALID_FAILED", "next_action": "NONE"}), encoding="utf-8")
    report = task._completed_resume_result(tmp_path, terminal_payload)
    assert report is not None and report["status"] == "VALID_FAILED"


def test_repeated_prediction_qualification_is_class_wise() -> None:
    labels = _role_labels()
    perfect_logits = torch.nn.functional.one_hot(labels, num_classes=10).float()
    metrics = task._class_metrics(labels, perfect_logits)
    assert metrics["by_role"]["confirmation"]["macro_recall"] == 1.0
    assert metrics["by_role"]["confirmation"]["per_class_correct"] == [30] * 10
    assert metrics["by_role"]["confirmation"]["predicted_class_counts"] == [30] * 10
