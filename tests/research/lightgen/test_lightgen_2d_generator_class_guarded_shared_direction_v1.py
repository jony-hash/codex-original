from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from ldm_project.v2.application.research.shared.research_run_control import ResearchRunIdentity
from scripts.research.lightgen import diagnose_lightgen_2d_generator_class_guarded_shared_direction_v1 as task


class _ToyRuntime(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.generator = torch.nn.Module()
        self.generator.phase = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))

    def forward_from_image(self, images: torch.Tensor):
        code = images[:, :1, :1, :1].reshape(-1).to(torch.float64)
        phase = self.generator.phase
        top_value = code - phase[0]
        bottom_value = code + 0.01 * phase[1]
        raw = torch.zeros((images.shape[0], 1, 28, 28), dtype=torch.float64, device=images.device)
        raw[:, :, :14, :] = top_value.view(-1, 1, 1, 1)
        raw[:, :, 14:, :] = bottom_value.view(-1, 1, 1, 1)
        return SimpleNamespace(raw_intensity=raw)


class _ToyConsumer:
    def __init__(self) -> None:
        self.runtime = _ToyRuntime()
        self.device = torch.device("cpu")

    def forward_loss(self, images: torch.Tensor, *, data_role: str):
        assert data_role == task.direction.bogt.TRAIN_ROLE
        return SimpleNamespace(result=self.runtime.forward_from_image(images))


class _ToyEvaluator(torch.nn.Module):
    def __init__(self, name: str, slope: float = 1.0) -> None:
        super().__init__()
        self.name = name
        self.slope = slope

    def forward(self, view: torch.Tensor):
        region = view[:, :, :14, :] if self.name != "E_D" else view[:, :, 14:, :]
        signal = region.mean(dim=(1, 2, 3)).to(torch.float64)
        class_id = (signal / 0.08 - 0.5).round().to(torch.long).clamp(0, 9)
        classes = torch.arange(10, dtype=torch.float64, device=view.device).view(1, 10)
        logits = -(classes - class_id[:, None].to(torch.float64)).square()
        code = class_id.to(torch.float64) * 0.08 + 0.04
        if self.name == "E_D":
            logits[:, 0] = 0.0
            logits[torch.arange(view.shape[0]), class_id] = 1.0
            focus = (class_id == 1) | (class_id == 7)
            logits[focus, class_id[focus]] = -1.0 + 200.0 * (signal[focus] - code[focus])
        else:
            logits[torch.arange(view.shape[0]), class_id] += self.slope * (signal - code)
        features = torch.zeros((view.shape[0], 128), dtype=torch.float64, device=view.device)
        return logits if self.name == "E_D" else (features, logits)


def _population() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    labels = torch.arange(10).repeat(60)
    code = labels.to(torch.float64) * 0.08 + 0.04
    images = code.view(-1, 1, 1, 1).expand(-1, 1, 28, 28).clone()
    rows = torch.arange(600) + 1000
    roles = ["fit"] * 200 + ["historic_holdout"] * 100 + ["confirmation"] * 300
    return images, labels, rows, roles


def _mixed_batch_reference_gradients(consumer: _ToyConsumer, evaluators: dict[str, torch.nn.Module], images: torch.Tensor, labels: torch.Tensor) -> list[dict[str, object]]:
    blocks = []
    parameter = consumer.runtime.generator.phase
    for batch_index in range(20):
        start = batch_index * 10
        batch_images = images[start:start + 10]
        batch_labels = labels[start:start + 10]
        breakdown = consumer.forward_loss(batch_images, data_role=task.direction.bogt.TRAIN_ROLE)
        view = task.canonical.cmos_view(breakdown.result.raw_intensity)
        vectors = {}
        for name in task.GUARD_EVALUATORS:
            loss = F.cross_entropy(evaluators[name](view)[1], batch_labels)
            gradient = torch.autograd.grad(loss, parameter, retain_graph=name != task.GUARD_EVALUATORS[-1])[0]
            vectors[f"ce_component_gradient_descent_{name}"] = -gradient.detach().cpu()
        blocks.append({"candidate_vectors_generator": vectors})
    return blocks


def _actual_rows(consumer: _ToyConsumer, evaluators: dict[str, torch.nn.Module], images: torch.Tensor, labels: torch.Tensor, source_rows: torch.Tensor, roles: list[str]) -> list[dict[str, object]]:
    rows = []
    with torch.inference_mode(False), torch.no_grad():
        for position in range(images.shape[0]):
            raw = consumer.runtime.forward_from_image(images[position:position + 1]).raw_intensity
            view = task.canonical.cmos_view(raw)
            logits = evaluators["E_D"](view)
            margin = task.direction._ed_margin_components(logits[0], int(labels[position]))
            true_prediction = int(logits.argmax())
            target = raw + 0.1
            mse = float((raw - target).square().mean())
            rows.append({
                "position": position,
                "source_row_index": int(source_rows[position]),
                "label": int(labels[position]),
                "role": roles[position],
                "ed_margin": float(margin["margin"]),
                "ed_prediction": true_prediction,
                "raw_mse": mse,
                "raw_foreground_mse": mse,
                "raw_background_mse": mse,
                "view_mse": mse,
                "view_foreground_mse": mse,
                "view_background_mse": mse,
                "raw_fraction_above_one": float((raw > 1.0).to(torch.float64).mean()),
                "raw_zero_image": bool(torch.equal(raw, torch.zeros_like(raw))),
                "raw_constant_image": bool((raw == raw.flatten()[0]).all()),
            })
    return rows


def _class_mean_ce(consumer: _ToyConsumer, evaluators: dict[str, torch.nn.Module], images: torch.Tensor, labels: torch.Tensor) -> dict[str, dict[int, float]]:
    result = {name: {} for name in task.GUARD_EVALUATORS}
    with torch.inference_mode(False), torch.no_grad():
        for name in task.GUARD_EVALUATORS:
            for class_id in range(10):
                positions = torch.nonzero(labels[:200] == class_id, as_tuple=False).reshape(-1)
                breakdown = consumer.forward_loss(images[:200].index_select(0, positions), data_role=task.direction.bogt.TRAIN_ROLE)
                view = task.canonical.cmos_view(breakdown.result.raw_intensity)
                result[name][class_id] = float(F.cross_entropy(evaluators[name](view)[1], labels.index_select(0, positions)))
    return result


def test_real_ce_gradient_projection_finite_response_reaches_final_decision() -> None:
    images, labels, source_rows, roles = _population()
    consumer = _ToyConsumer()
    evaluators = {
        "E_S": _ToyEvaluator("E_S", slope=1.0),
        "E_I": _ToyEvaluator("E_I", slope=2.0),
        "E_C": _ToyEvaluator("E_C", slope=3.0),
        "E_D": _ToyEvaluator("E_D"),
    }
    gradients = task._class_ce_gradients(
        consumer,
        evaluators,
        consumer.runtime.generator.phase,
        images[:200],
        labels[:200],
    )
    task._attach_source_rows(gradients, source_rows[:200])
    task._validate_gradient_population(gradients, labels[:200], source_rows[:200])
    blocks = _mixed_batch_reference_gradients(consumer, evaluators, images[:200], labels[:200])
    differences = task._validate_class_gradient_means(gradients, blocks)
    assert set(differences) == set(task.GUARD_EVALUATORS)
    assert max(item["relative_l2_error"] for item in differences.values()) < 1e-6

    source = torch.tensor([0.001, 0.001], dtype=torch.float64)
    projection = task._project_guarded_direction(source, gradients["gradients"])
    candidate = projection["candidate_equal_rms_direction"]
    assert candidate is not None
    assert abs(float(projection["unit_guard_dot_projected"]["E_S:class1"])) < 1e-8
    assert abs(float(projection["unit_guard_dot_projected"]["E_I:class7"])) < 1e-8

    baseline_phase = consumer.runtime.generator.phase.detach().clone()
    baseline_rows = _actual_rows(consumer, evaluators, images, labels, source_rows, roles)
    baseline_ce = _class_mean_ce(consumer, evaluators, images, labels)
    task.finite._apply_displacement(consumer.runtime.generator.phase, baseline_phase, candidate)
    candidate_rows = _actual_rows(consumer, evaluators, images, labels, source_rows, roles)
    candidate_ce = _class_mean_ce(consumer, evaluators, images, labels)
    deltas = [candidate_ce[name][class_id] - baseline_ce[name][class_id] for name, class_id in task.GUARD_NAMES]
    assert max(deltas) <= task.ETA_FLOOR

    decision = task._decision(baseline_rows, candidate_rows, baseline_ce, candidate_ce, task.ETA_FLOOR, task.ETA_FLOOR)
    assert decision["classification"] == "CLASS_GUARDED_SHARED_LOCAL_RESPONSE_SUPPORTED"
    assert decision["confirmation_focus_response"]["1"]["beneficial_rows"] == 30
    assert decision["confirmation_focus_response"]["7"]["beneficial_rows"] == 30
    assert decision["training_recommendation"] == "NONE"
    assert decision["next_action"] == "NONE"


def test_projection_keeps_conflicting_evaluator_guards_separate_and_matches_active_set_reference() -> None:
    source = torch.tensor([1.0, 1.0], dtype=torch.float64)
    gradients = {
        "E_S": {class_id: torch.zeros(2) for class_id in range(10)},
        "E_I": {class_id: torch.zeros(2) for class_id in range(10)},
        "E_C": {class_id: torch.zeros(2) for class_id in range(10)},
    }
    gradients["E_S"][0] = torch.tensor([1.0, 0.0])
    gradients["E_I"][0] = torch.tensor([-1.0, 0.0])
    result = task._project_guarded_direction(source, gradients)
    assert torch.allclose(result["projected_minimum_distance_vector"], torch.tensor([0.0, 1.0], dtype=torch.float64), atol=1e-8, rtol=0.0)
    assert torch.equal((gradients["E_S"][0] + gradients["E_I"][0]) / 2, torch.zeros(2))

    cpu_check = task._cpu_reference_checks()
    assert cpu_check["passed"] is True
    assert cpu_check["active_subset_projection"]["maximum_absolute_difference"] <= 1e-8

    phase_shape_source = torch.ones((2, 1, 2, 2), dtype=torch.float64)
    phase_shape_guards = {name: {class_id: torch.zeros_like(phase_shape_source) for class_id in range(10)} for name in task.GUARD_EVALUATORS}
    phase_shape_guards["E_S"][0] = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]], dtype=torch.float64)
    shaped = task._project_guarded_direction(phase_shape_source, phase_shape_guards)
    assert shaped["candidate_equal_rms_direction"].shape == phase_shape_source.shape


def test_saved_mixed_batch_reference_rejects_a_wrong_class_mean_denominator() -> None:
    images, labels, _, _ = _population()
    consumer = _ToyConsumer()
    evaluators = {"E_S": _ToyEvaluator("E_S"), "E_I": _ToyEvaluator("E_I", 2.0), "E_C": _ToyEvaluator("E_C", 3.0), "E_D": _ToyEvaluator("E_D")}
    gradients = task._class_ce_gradients(consumer, evaluators, consumer.runtime.generator.phase, images[:200], labels[:200])
    blocks = _mixed_batch_reference_gradients(consumer, evaluators, images[:200], labels[:200])
    blocks[0]["candidate_vectors_generator"]["ce_component_gradient_descent_E_S"] *= 0.5
    with pytest.raises(task.ClassGuardedDirectionError, match="classwise CE gradient mean"):
        task._validate_class_gradient_means(gradients, blocks)


def test_saved_guard_gradient_row_map_rejects_wrong_class_mask() -> None:
    images, labels, source_rows, _ = _population()
    consumer = _ToyConsumer()
    evaluators = {"E_S": _ToyEvaluator("E_S"), "E_I": _ToyEvaluator("E_I", 2.0), "E_C": _ToyEvaluator("E_C", 3.0), "E_D": _ToyEvaluator("E_D")}
    gradients = task._class_ce_gradients(consumer, evaluators, consumer.runtime.generator.phase, images[:200], labels[:200])
    task._attach_source_rows(gradients, source_rows[:200])
    gradients["fit_row_map"]["E_S"]["1"]["fit_positions"][0] = 0
    with pytest.raises(task.ClassGuardedDirectionError, match="wrong fit-row mask"):
        task._validate_gradient_population(gradients, labels[:200], source_rows[:200])


def test_zero_projection_is_valid_unavailable_candidate_and_wrong_sign_is_rejected() -> None:
    source = torch.tensor([1.0, 0.0], dtype=torch.float64)
    gradients = {
        "E_S": {class_id: torch.zeros(2) for class_id in range(10)},
        "E_I": {class_id: torch.zeros(2) for class_id in range(10)},
        "E_C": {class_id: torch.zeros(2) for class_id in range(10)},
    }
    gradients["E_S"][0] = torch.tensor([1.0, 0.0])
    result = task._project_guarded_direction(source, gradients)
    assert result["candidate_available"] is False
    assert result["candidate_equal_rms_direction"] is None
    assert result["projected_to_source_l2_ratio"] <= task.ZERO_DIRECTION_RELATIVE
    with pytest.raises(task.ClassGuardedDirectionError):
        task._project_guarded_direction(torch.tensor([float("nan"), 0.0]), gradients)


def test_one_class_mean_margin_guard_catches_twenty_three_harmful_rows() -> None:
    baseline = []
    candidate = []
    for position in range(300):
        class_id = position % 10
        row = {
            "source_row_index": position,
            "label": class_id,
            "role": "confirmation",
            "ed_margin": 1.0 if class_id not in (1, 7) else -1.0,
            "ed_prediction": class_id if class_id not in (1, 7) else 0,
            "raw_mse": 0.01,
            "raw_foreground_mse": 0.01,
            "raw_background_mse": 0.01,
            "view_mse": 0.01,
            "view_foreground_mse": 0.01,
            "view_background_mse": 0.01,
            "raw_fraction_above_one": 0.0,
            "raw_zero_image": False,
            "raw_constant_image": False,
        }
        baseline.append(dict(row))
        changed = dict(row)
        if class_id == 1:
            changed["ed_margin"] += 0.01
        if class_id == 7:
            changed["ed_margin"] += 0.01
        if class_id == 2 and position < 230:
            changed["ed_margin"] -= 0.02
        candidate.append(changed)
    fit_ce = {name: {class_id: 1.0 for class_id in range(10)} for name in task.GUARD_EVALUATORS}
    decision = task._decision(baseline, candidate, fit_ce, fit_ce, 1e-4, 1e-4)
    assert decision["confirmation_other_class_margin_changes"]["2"]["harmful_rows"] == 23
    assert decision["confirmation_other_class_margin_changes"]["2"]["mean_margin_delta"] < -1e-4
    assert decision["classification"] == "CLASS_GUARDED_RESPONSE_WITH_TRADEOFF"


def test_unavailable_candidate_is_not_promoted_by_hand_supplied_focus_metrics() -> None:
    result = task._decision([], [], {}, {}, task.ETA_FLOOR, task.ETA_FLOOR, candidate_available=False)
    assert result["classification"] == "MIXED_OR_INSUFFICIENT"
    assert result["training_recommendation"] == "NONE"


def test_completed_manifest_returns_terminal_report_without_cuda_or_forward(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    target = root / "outputs" / "attempt_01"
    target.mkdir(parents=True)
    labels = torch.arange(10).repeat(60).tolist()
    roles = ["fit"] * 200 + ["historic_holdout"] * 100 + ["confirmation"] * 300
    source_rows = list(range(600))
    identity = ResearchRunIdentity(
        task_id=task.TASK_ID,
        run_id=f"{task.TASK_ID}-attempt-01",
        trajectory_id="frozen-test",
        semantic_authority_id=task.TASK_ID,
        data_role="fixed TRAIN test population",
        config_identity={"propagation_grid": 2048},
        model_identity={"source": "toy"},
        optimizer_identity={"updates": 0},
        sampler_identity={"rows": source_rows, "labels": labels, "roles": roles},
        execution_source_semantics_id=task.TASK_ID,
        scientific_attempt_id=f"{task.TASK_ID}:attempt-01",
    )
    population = {
        "task_id": task.TASK_ID,
        "shared_direction_directory": (root / "shared").as_posix(),
        "parent_ce_directory": (root / "parent").as_posix(),
        "corrected_direction_directory": (root / "direction").as_posix(),
        "finite_response_directory": (root / "finite").as_posix(),
        "checkpoint": (root / "parent" / "arm_b_step0300.pt").as_posix(),
        "source_row_indices": source_rows,
        "labels": labels,
        "roles": roles,
        "role_counts": {"fit": 200, "historic_holdout": 100, "confirmation": 300},
        "row_order": "fixed",
        "direction_rows": "fit",
        "evaluator_guard_constraints": [f"{name}:class{class_id}" for name, class_id in task.GUARD_NAMES],
        "confirmation_used_for_gradient_or_projection": False,
        "e_d_used_for_direction_construction": False,
        "source_direction": "pure CE",
        "candidate_scale": "equal RMS",
        "conditions": ["baseline"],
        "margin_definition": task.MARGIN_DEFINITION,
        "output_directory": target.as_posix(),
    }
    expected = [task._chunk_name("baseline", index) for index in range(task.CHUNK_COUNT)]
    for name in expected:
        (target / name).touch()
    manifest = {
        "schema": task.MANIFEST_SCHEMA,
        "task_id": task.TASK_ID,
        "measurement_semantics": "independent E_S/E_I/E_C per-class CE guards",
        "identity": identity.to_dict(),
        "population": population,
        "measured_conditions": ["baseline"],
        "completed_chunks": expected,
        "status": "COMPLETED",
    }
    (target / "resume_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report = {"schema": task.REPORT_SCHEMA, "task_id": task.TASK_ID, "status": "VALID_COMPLETED", "classification": "MIXED_OR_INSUFFICIENT", "research_run_identity": identity.to_dict()}
    (target / "report.json").write_text(json.dumps(report), encoding="utf-8")

    inputs = {
        "shared_dir": root / "shared",
        "artifact_dir": root / "parent",
        "direction_dir": root / "direction",
        "finite_dir": root / "finite",
        "paths": {"checkpoint_B_class_ce": root / "parent" / "arm_b_step0300.pt"},
        "source_indices_600": torch.tensor(source_rows),
        "labels_600": torch.tensor(labels),
        "roles_600": roles,
        "directions": {"equal_rms_directions": {"S_GD": torch.ones(2)}, "reference_rms": 1.0},
        "payloads": {"B_class_ce": {"trainable_parameter_state": {"encoder.phase": torch.zeros(2), "generator.phase": torch.zeros(2)}, "optimizer_step_by_parameter": {"encoder.phase": 2200, "generator.phase": 2500}}},
    }
    monkeypatch.setattr(task, "_root", lambda: root)
    monkeypatch.setattr(task, "_load_inputs", lambda *_: inputs)
    monkeypatch.setattr(task, "_population_record", lambda *_: population)
    monkeypatch.setattr(task, "_identity", lambda *_: identity)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(AssertionError("completed resume attempted CUDA")))
    result = task._run(argparse.Namespace(
        shared_direction_dir=Path("shared"),
        output_dir=Path("outputs/attempt_01"),
        preflight_only=False,
        resume_manifest=Path("outputs/attempt_01/resume_manifest.json"),
    ))
    assert result == report
