from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from scripts.research.lightgen import execute_lightgen_2d_generator_class9_weighted_ce_comparison_v1 as class9
from scripts.research.lightgen import execute_lightgen_2d_generator_evaluator_ablation_comparison_v1 as ablation
from scripts.research.lightgen import execute_lightgen_2d_generator_large_capacity_quick_comparison_v1 as capacity
from scripts.research.lightgen import execute_lightgen_2d_generator_training_population_expansion_comparison_v1 as task


def test_population_sampling_is_seeded_balanced_interleaved_and_disjoint() -> None:
    labels = torch.arange(60_000, dtype=torch.long) % 10
    train = torch.arange(60_000, dtype=torch.long)
    prior = torch.arange(1_200, dtype=torch.long)

    expanded = task._select_rows(labels, train, prior, 180, task.EXPANSION_SEED)
    repeated = task._select_rows(labels, train, prior, 180, task.EXPANSION_SEED)
    confirmation = task._select_rows(labels, train, torch.cat((prior, expanded)), 30, task.CONFIRMATION_SEED)

    assert torch.equal(expanded, repeated)
    assert labels[expanded].tolist() == list(range(10)) * 180
    assert labels[confirmation].tolist() == list(range(10)) * 30
    assert torch.unique(torch.cat((prior, expanded, confirmation))).numel() == 3_300
    assert not set(confirmation.tolist()).intersection(torch.cat((prior, expanded)).tolist())


def test_population_role_producer_and_image_page_mapping_cover_all_3300_rows() -> None:
    positions, counts, roles = task._role_partition()
    population = {"role_positions": positions, "role_counts": counts, "roles": roles}
    task._validate_role_partition(population)
    pages = task._role_page_positions(population)
    mapped = [position for role_pages in pages.values() for page in role_pages for position in page]
    assert len(mapped) == task.TOTAL_ROWS
    assert sorted(mapped) == list(range(task.TOTAL_ROWS))
    assert positions["class9_confirmation"] == [900, 1_200]
    assert counts["class9_confirmation"] == 300
    assert len(pages["class9_confirmation"]) == 10
    assert sum(counts.values()) == task.TOTAL_ROWS

    faulty = dict(population)
    faulty["role_positions"] = {key: value for key, value in positions.items() if key != "class9_confirmation"}
    with pytest.raises(task.PopulationExpansionError):
        task._validate_role_partition(faulty)


def test_training_batch_mapping_uses_200_batches_for_B_and_excludes_confirmation() -> None:
    rows = torch.arange(task.TOTAL_ROWS, dtype=torch.long)
    labels = rows % 10
    expanded_positions = torch.arange(task.SOURCE_ROWS, task.SOURCE_ROWS + task.EXPANDED_ROWS)
    fit_measure_positions = torch.cat((torch.arange(task.BASE_FIT_ROWS), expanded_positions))
    train_inputs = {
        "images": torch.zeros((2_000, 1, 28, 28)),
        "labels": labels.index_select(0, fit_measure_positions),
        "source_indices": rows.index_select(0, fit_measure_positions),
        "fit_measure_positions": fit_measure_positions,
    }

    assert task._batch_indices(task.ARM_A, 999) == (19, 190)
    assert task._batch_indices(task.ARM_B, 199) == (199, 1_990)
    assert task._batch_indices(task.ARM_B, 200) == (0, 0)
    assert task._batch_indices(task.ARM_B, 1_000) == (0, 0)
    _, fit_positions, source_rows, labels_last = task._training_batch(train_inputs, task.ARM_B, 199)
    assert fit_positions.tolist() == list(range(2_990, 3_000))
    assert source_rows.tolist() == list(range(2_990, 3_000))
    assert labels_last.tolist() == list(range(10))
    assert all(row < task.SOURCE_ROWS + task.EXPANDED_ROWS for row in source_rows.tolist())
    with pytest.raises(task.PopulationExpansionError):
        task._training_batch(train_inputs, task.ARM_B, task.B_BATCH_COUNT)


@pytest.mark.parametrize("dtype,epsilon,tolerance", ((torch.float32, 1e-3, 2e-4), (torch.float64, 1e-6, 2e-8)))
def test_uniform_ce_microbatch_producer_matches_full_batch_value_gradient_and_finite_difference(
    dtype: torch.dtype, epsilon: float, tolerance: float
) -> None:
    torch.manual_seed(2026100811)
    labels = torch.arange(10, dtype=torch.long)
    logits = torch.randn((10, 10), dtype=dtype, requires_grad=True)
    direction = torch.linspace(-0.5, 0.7, 100, dtype=dtype).reshape(10, 10)
    accumulated_gradient = torch.zeros_like(logits)
    accumulated_value = torch.zeros((), dtype=dtype)
    for start in range(0, 10, 2):
        piece = task._uniform_ce_piece(logits[start : start + 2], labels[start : start + 2], logical_batch_size=10)
        accumulated_value = accumulated_value + piece.detach()
        accumulated_gradient += torch.autograd.grad(piece, logits, retain_graph=True)[0]

    expected_value = F.cross_entropy(logits, labels, reduction="mean")
    expected_gradient = (torch.softmax(logits.detach(), dim=1) - F.one_hot(labels, num_classes=10).to(dtype)) / 10
    torch.testing.assert_close(accumulated_value, expected_value, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(accumulated_gradient, expected_gradient, rtol=tolerance, atol=tolerance)

    plus = sum(task._uniform_ce_piece((logits.detach() + epsilon * direction)[start : start + 2], labels[start : start + 2], logical_batch_size=10) for start in range(0, 10, 2))
    minus = sum(task._uniform_ce_piece((logits.detach() - epsilon * direction)[start : start + 2], labels[start : start + 2], logical_batch_size=10) for start in range(0, 10, 2))
    finite_difference = (plus - minus) / (2 * epsilon)
    predicted = (accumulated_gradient * direction).sum()
    torch.testing.assert_close(finite_difference, predicted, rtol=tolerance * 4, atol=tolerance * 4)

    wrong_local_denominator = torch.zeros_like(logits)
    for start in range(0, 10, 2):
        bad_piece = F.cross_entropy(logits[start : start + 2], labels[start : start + 2], reduction="mean")
        wrong_local_denominator += torch.autograd.grad(bad_piece, logits, retain_graph=True)[0]
    assert not torch.allclose(wrong_local_denominator, expected_gradient, rtol=tolerance, atol=tolerance)
    with pytest.raises(task.PopulationExpansionError):
        task._uniform_ce_piece(logits[:2], labels[:2], logical_batch_size=2)


def test_inherited_adam_loader_does_not_mutate_the_source_step_tensor() -> None:
    source_parameter = torch.nn.Parameter(torch.tensor([0.2, -0.1]))
    source_optimizer = torch.optim.Adam([source_parameter], lr=capacity.GENERATOR_LEARNING_RATE, betas=capacity.GENERATOR_BETAS, eps=capacity.GENERATOR_EPS, weight_decay=0.0)
    source_optimizer.state[source_parameter] = {
        "step": torch.tensor(float(task.SOURCE_STEP)),
        "exp_avg": torch.tensor([0.3, -0.2]),
        "exp_avg_sq": torch.tensor([0.5, 0.7]),
    }
    source_state = source_optimizer.state_dict()
    source_step = source_state["state"][0]["step"].clone()
    parameter = torch.nn.Parameter(torch.tensor([0.2, -0.1]))
    optimizer = torch.optim.Adam([parameter], lr=capacity.GENERATOR_LEARNING_RATE, betas=capacity.GENERATOR_BETAS, eps=capacity.GENERATOR_EPS, weight_decay=0.0)

    ablation._load_adam_state(optimizer, parameter, source_state, expected_step=task.SOURCE_STEP)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    assert torch.equal(source_state["state"][0]["step"], source_step)
    assert int(float(optimizer.state[parameter]["step"].item())) == task.SOURCE_STEP + 1


def _scored_summary(counts: list[int], *, raw_scale: float = 1.0) -> dict:
    labels = torch.arange(10, dtype=torch.long).repeat_interleave(30)
    logits = torch.full((300, 10), -1.0)
    seen = [0] * 10
    for index, label in enumerate(labels.tolist()):
        class_id = int(label)
        if seen[class_id] < counts[class_id]:
            logits[index, class_id] = 1.0
        else:
            logits[index, (class_id + 1) % 10] = 1.0
        seen[class_id] += 1
    target = torch.zeros((300, 1, 2, 2), dtype=torch.float32)
    target[:, 0, 0, 0] = 1.0
    raw = target * (0.8 * raw_scale) + 0.05
    view = F.adaptive_avg_pool2d(raw.clamp(0, 1), (28, 28))
    readout = {
        "labels": labels,
        "source_indices": torch.arange(300),
        "target_slm_intensity": target,
        "raw_cmos": raw,
        "evaluator_view": view,
        **{f"{name.lower()}_logits": logits.clone() for name in capacity.EVALUATOR_NAMES},
    }
    population = {"role_positions": {"coverage_confirmation": [0, 300]}}
    summary, _ = class9._row_and_image_metrics(readout, population)
    return summary


def test_scored_readouts_flow_through_final_population_decision_and_reject_weak_class() -> None:
    source = _scored_summary([21] * 10)
    arm_a = _scored_summary([21] * 10)
    supported = _scored_summary([24] * 10)
    decision = task._decision(source, arm_a, supported, valid=True)
    assert decision["classification"] == "TRAINING_POPULATION_EXPANSION_GAIN_SUPPORTED"
    assert decision["conditions"]["B_at_least_225_and_5_points_above_A_and_not_below_start"]

    weak_class_five = [26] * 10
    weak_class_five[5] = 11
    mixed = task._decision(source, arm_a, _scored_summary(weak_class_five), valid=True)
    assert mixed["new_confirmation"]["correct_total"]["B4700"] > 225
    assert not mixed["conditions"]["every_class_at_least_15"]
    assert mixed["classification"] == "MIXED_OR_INSUFFICIENT"
    invalid = task._decision(source, arm_a, supported, valid=False)
    assert invalid["status"] == "INVALID_HARNESS"
    assert invalid["classification"] == "INVALID_HARNESS"


def test_ready_consumer_accepts_expanded_metrics_after_replaying_the_source_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = torch.arange(3_300, dtype=torch.long) % 10
    source_indices = torch.arange(3_300, dtype=torch.long)
    logits = torch.full((3_300, 10), -1.0)
    logits[torch.arange(3_300), labels] = 1.0
    readout = {
        "source_indices": source_indices,
        "labels": labels,
        "z": torch.zeros((3_300, 1), dtype=torch.complex64),
        "raw_cmos": torch.zeros((3_300, 1)),
        "evaluator_view": torch.zeros((3_300, 1)),
        "target_slm_intensity": torch.zeros((3_300, 1)),
        "power_by_plane": torch.zeros((3_300, 1)),
        "new_phase_plane_power": torch.zeros((3_300, 1)),
        **{f"{name.lower()}_logits": logits.clone() for name in capacity.EVALUATOR_NAMES},
    }
    source_readout = {name: value[: task.SOURCE_ROWS].clone() for name, value in readout.items()}
    target_control = {
        "source_row_indices": source_indices[-task.CONFIRMATION_ROWS :].clone(),
        "labels": labels[-task.CONFIRMATION_ROWS :].clone(),
        **{name: value[-task.CONFIRMATION_ROWS :].clone() for name, value in readout.items() if name.endswith("_logits")},
    }
    expected_per_class = [task.CONFIRMATION_ROWS // 10] * 10
    preflight = {
        "checks": {
            "new_confirmation_target_positive_control": {
                "summary": {
                    name: {"per_class_correct": expected_per_class, "macro_recall": 1.0}
                    for name in capacity.EVALUATOR_NAMES
                }
            }
        },
        "source_replay": task._source_replay(readout, source_readout),
    }
    population = {
        "role_positions": {
            "source_rows": [0, task.SOURCE_ROWS],
            "expanded_training": [task.SOURCE_ROWS, task.SOURCE_ROWS + task.EXPANDED_ROWS],
            "coverage_confirmation": [task.SOURCE_ROWS + task.EXPANDED_ROWS, task.TOTAL_ROWS],
        }
    }
    measure_inputs = {"source_indices": source_indices, "labels": labels}
    expanded_summary = {"rows": task.TOTAL_ROWS}
    parent_summary = {"rows": task.SOURCE_ROWS}
    assert expanded_summary != parent_summary

    with tempfile.TemporaryDirectory(prefix="lg_population_ready_consumer_") as temporary:
        target = Path(temporary)
        torch.save(target_control, target / "target_image_evaluator_positive_controls.pt")
        torch.save(readout, target / "readouts_A_step3700.pt")
        (target / "metrics_A_step3700.json").write_text('{"rows": 3300}', encoding="utf-8")
        monkeypatch.setattr(task.class9, "_row_and_image_metrics", lambda saved, _population: ({"rows": saved["source_indices"].numel()}, {}))

        replay, summary = task._consume_ready_artifacts(
            target, measure_inputs, population, source_readout, preflight
        )

    assert replay["passed"] is True
    assert summary == expanded_summary


def test_interrupt_checkpoint_parser_requires_task_local_population_cursor() -> None:
    labels = (torch.arange(task.TOTAL_ROWS, dtype=torch.long) % 10).tolist()
    measure_rows = list(range(task.TOTAL_ROWS))
    fit_mapping = list(range(200)) + list(range(1_200, 3_000))
    record = {
        "update": task.SOURCE_STEP + 1,
        "task_local_update": 1,
        "arm": task.ARM_B,
        "batch_index": 0,
        "training_positions": list(range(10)),
        "fit_positions": fit_mapping[:10],
        "source_row_indices": [measure_rows[position] for position in fit_mapping[:10]],
        "labels": [labels[position] for position in fit_mapping[:10]],
        "training_evaluators": ["E_S", "E_I", "E_C"],
        "semantic_weight_policy": "uniform_x1",
        "ce_denominator": 10.0,
    }
    history = [{} for _ in range(task.SOURCE_STEP)] + [record]
    source = {
        "checkpoint_path": "D:/source/arm_a_step3700.pt",
        "training_history": history[:-1],
        "trainable_parameter_state": {"encoder.phase": torch.zeros((2, 1, 100, 100)), "generator.phase": torch.zeros(340_000)},
        "frozen_runtime_state": {},
        "references": {},
        "weights": {},
        "rng_state": {"cpu": torch.tensor([1], dtype=torch.uint8), "python": (1,), "numpy": (1,)},
        "architecture_layout": "two_original_100x100_then_two_400x400_phase_planes_flattened",
    }
    population = {
        "source_row_indices": measure_rows,
        "labels": labels,
        "fit_mapping_A_measure_positions": list(range(200)),
        "fit_mapping_B_measure_positions": fit_mapping,
    }
    optimizer_state = {
        "state": {0: {"step": torch.tensor(float(task.SOURCE_STEP + 1)), "exp_avg": torch.zeros(340_000), "exp_avg_sq": torch.zeros(340_000)}},
        "param_groups": [{"params": [0], "lr": capacity.GENERATOR_LEARNING_RATE, "betas": capacity.GENERATOR_BETAS, "eps": capacity.GENERATOR_EPS, "weight_decay": 0.0}],
    }
    checkpoint = {
        "schema": task.CHECKPOINT_SCHEMA,
        "task_id": task.TASK_ID,
        "source_task_id": task.PARENT_TASK_ID,
        "arm": task.ARM_B,
        "source_checkpoint": source["checkpoint_path"],
        "checkpoint_reason": "interrupt_boundary",
        "completed_updates": task.SOURCE_STEP + 1,
        "task_local_updates": 1,
        "additional_updates": 1,
        "optimizer_step": task.SOURCE_STEP + 1,
        "task_batch_count": task.B_BATCH_COUNT,
        "next_batch_index": 1,
        "next_batch_cursor": 1,
        "numeric_mode": task.NUMERIC_MODE,
        "architecture_layout": source["architecture_layout"],
        "population": population,
        "training_history": history,
        "source_history_count": task.SOURCE_STEP,
        "source_rng_state": source["rng_state"],
        "rng_state": source["rng_state"],
        "trainable_parameter_state": source["trainable_parameter_state"],
        "optimizer_state": optimizer_state,
        "frozen_runtime_state": {},
        "references": {},
        "weights": {},
    }

    with tempfile.TemporaryDirectory(prefix="lg_population_checkpoint_test_") as temporary:
        directory = Path(temporary)
        path = directory / task._checkpoint_name(task.ARM_B, task.SOURCE_STEP + 1)
        torch.save(checkpoint, path)
        (directory / task._checkpoint_name(task.ARM_A, task.FINAL_STEP)).write_bytes(b"completed A arm")
        recovered = task._read_checkpoint(path, directory, population, source)
        assert recovered["next_batch_index"] == 1

        invalid = dict(checkpoint)
        invalid["next_batch_index"] = 100
        invalid_dir = directory / "invalid"
        invalid_dir.mkdir()
        invalid_path = invalid_dir / task._checkpoint_name(task.ARM_B, task.SOURCE_STEP + 1)
        torch.save(invalid, invalid_path)
        (invalid_dir / task._checkpoint_name(task.ARM_A, task.FINAL_STEP)).write_bytes(b"completed A arm")
        with pytest.raises(task.PopulationExpansionError):
            task._read_checkpoint(invalid_path, invalid_dir, population, source)
