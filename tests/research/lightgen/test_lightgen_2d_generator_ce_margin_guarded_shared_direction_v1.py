from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from ldm_project.v2.application.research.shared.research_run_control import ResearchRunIdentity
from scripts.research.lightgen import diagnose_lightgen_2d_generator_ce_margin_guarded_shared_direction_v1 as task
from scripts.research.lightgen import diagnose_lightgen_2d_generator_class_guarded_shared_direction_v1 as class_guard


class _ToyRuntime(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.generator = torch.nn.Module()
        self.generator.phase = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        axis = torch.arange(200)
        texture = ((axis[:, None] + axis[None, :]) % 2).to(torch.float64) * 2.0 - 1.0
        self.register_buffer("texture", texture * 0.001)

    def forward_from_image(self, images: torch.Tensor):
        labels = images[:, 0, 0, 0].round().to(torch.long)
        codes = labels.to(torch.float64) * 0.08 + 0.04
        direction = torch.where(labels < 5, 1.0, -1.0).to(torch.float64)
        offset = torch.where(labels < 5, 0.01, -0.01).to(torch.float64)
        signal = codes + offset + direction * self.generator.phase[0] + self.generator.phase[1] * 0.0
        raw = signal[:, None, None, None] + self.texture.to(images.device)[None, None, :, :]
        return SimpleNamespace(raw_intensity=raw)


class _ToyConsumer:
    def __init__(self) -> None:
        self.runtime = _ToyRuntime()
        self.device = torch.device("cpu")

    def forward_loss(self, images: torch.Tensor, *, data_role: str):
        assert data_role == task.direction.bogt.TRAIN_ROLE
        return SimpleNamespace(result=self.runtime.forward_from_image(images))


class _ToyEvaluator(torch.nn.Module):
    def __init__(self, name: str, scale: float) -> None:
        super().__init__()
        self.name = name
        self.scale = scale

    def forward(self, view: torch.Tensor):
        signal = view.mean(dim=(1, 2, 3)).to(torch.float64)
        coordinate = signal / 0.08 - 0.5
        classes = torch.arange(10, dtype=torch.float64, device=view.device).view(1, 10)
        logits = (-(classes - coordinate[:, None]).square() - 0.01 * classes) * self.scale
        if self.name == "E_D":
            return logits
        features = torch.zeros((view.shape[0], 128), dtype=torch.float64, device=view.device)
        return features, logits


class _TiedEvaluator(torch.nn.Module):
    def forward(self, view: torch.Tensor):
        logits = torch.zeros((view.shape[0], 10), dtype=torch.float64, device=view.device)
        features = torch.zeros((view.shape[0], 128), dtype=torch.float64, device=view.device)
        return features, logits


def _population() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    labels = torch.arange(10).repeat(60)
    images = labels.to(torch.float64).view(-1, 1, 1, 1)
    source_rows = torch.arange(600, dtype=torch.long) + 1000
    roles = ["fit"] * 200 + ["historic_holdout"] * 100 + ["confirmation"] * 300
    return images, labels, source_rows, roles


def _measured_rows(consumer: _ToyConsumer, evaluators: dict[str, torch.nn.Module], images: torch.Tensor, labels: torch.Tensor, source_rows: torch.Tensor, roles: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with torch.inference_mode(False), torch.no_grad():
        for start in range(0, len(images), 10):
            batch = images[start:start + 10]
            batch_labels = labels[start:start + 10]
            raw = consumer.runtime.forward_from_image(batch).raw_intensity
            view = task.canonical.cmos_view(raw)
            target_codes = batch_labels.to(torch.float64) * 0.08 + 0.04
            target = target_codes[:, None, None, None] + consumer.runtime.texture[None, None, :, :]
            raw_mse = (raw - target).square().mean(dim=(1, 2, 3))
            logits_by_evaluator: dict[str, torch.Tensor] = {}
            margins_by_evaluator: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
            for name, evaluator in evaluators.items():
                output = evaluator(view)
                logits = output if torch.is_tensor(output) else output[1]
                scores = task._score_logits(logits, batch_labels)
                margin, competitor, tied = scores["margin"], scores["actual_competitor"], scores["actual_tied"]
                logits_by_evaluator[name] = logits
                margins_by_evaluator[name] = (margin, competitor, tied)
            for local in range(len(batch)):
                position = start + local
                ed_margin, ed_competitor, ed_tie = margins_by_evaluator["E_D"]
                row: dict[str, object] = {
                    "position": position,
                    "source_row_index": int(source_rows[position]),
                    "label": int(labels[position]),
                    "role": roles[position],
                    "ed_margin": float(ed_margin[local]),
                    "ed_prediction": int(logits_by_evaluator["E_D"][local].argmax()),
                    "ed_best_wrong_class": int(ed_competitor[local]),
                    "ed_best_wrong_tied_classes": [int(ed_competitor[local])] if bool(ed_tie[local]) else [],
                    "raw_mse": float(raw_mse[local]),
                    "raw_foreground_mse": float(raw_mse[local]),
                    "raw_background_mse": float(raw_mse[local]),
                    "view_mse": float(raw_mse[local]),
                    "view_foreground_mse": float(raw_mse[local]),
                    "view_background_mse": float(raw_mse[local]),
                    "raw_fraction_above_one": float((raw[local] > 1.0).to(torch.float64).mean()),
                    "raw_zero_image": bool(torch.equal(raw[local], torch.zeros_like(raw[local]))),
                    "raw_constant_image": bool((raw[local] == raw[local].flatten()[0]).all()),
                }
                for name, logits in logits_by_evaluator.items():
                    values = logits[local]
                    prediction = int(values.argmax())
                    margin, competitor, _, _ = task._batch_margin(values[None, :], torch.tensor([row["label"]]))
                    row[f"{name}_prediction"] = prediction
                    row[f"{name}_correct"] = int(prediction == row["label"])
                    score = task._score_logits(values[None, :], torch.tensor([row["label"]]))
                    row[f"{name}_ce"] = float(score["ce"][0])
                    row[f"{name}_margin"] = float(score["margin"][0])
                    row[f"{name}_max_wrong_class"] = int(competitor[0])
                rows.append(row)
    return rows


def test_independent_cpu_references_check_values_derivatives_signs_and_projection() -> None:
    result = task._cpu_references()
    assert result["passed"] is True
    counterexample = result["ce_decrease_margin_decrease"]
    assert counterexample["ce_derivative"] < 0.0
    assert counterexample["margin_derivative"] == pytest.approx(-0.01, abs=1e-12)
    assert counterexample["margin_central_difference"] == pytest.approx(counterexample["margin_derivative"], abs=1e-8)
    assert result["common_logit_shift_margin_derivative"] == 0.0
    assert result["exact_tie_preserved"] is True
    assert result["small_projection_reference"]["candidate"] == pytest.approx([0.0, -0.3], abs=1e-10)


def test_production_score_path_uses_float64_and_paired_reduction_defeats_float32_mean_cancellation() -> None:
    assert isinstance(task._numeric_settings()["torch_version"], str)
    logits = torch.tensor([[0.0, -1.0, -2.0], [0.0, -1.0, -2.0]], dtype=torch.float32)
    labels = torch.tensor([1, 1])
    scores = task._score_logits(logits, labels)
    assert scores["ce"].dtype == torch.float64
    assert scores["margin"].dtype == torch.float64
    independent = F.cross_entropy(logits.to(torch.float64), labels, reduction="none")
    torch.testing.assert_close(scores["ce"], independent, rtol=0.0, atol=0.0)

    base = torch.tensor(16.0, dtype=torch.float32)
    next_value = torch.nextafter(base, torch.tensor(float("inf"), dtype=torch.float32))
    plus = base.repeat(1024)
    minus = base.repeat(1024)
    plus[:512] = next_value
    epsilon = 0.125
    legacy_slope = float((plus.mean() - minus.mean()) / (2.0 * epsilon))
    paired_slope, row_slopes = task._paired_finite_difference(plus, minus, epsilon)
    assert legacy_slope == 0.0
    assert paired_slope > 0.0
    assert row_slopes.dtype == torch.float64
    assert paired_slope == pytest.approx(float((next_value - base).to(torch.float64)) / (4.0 * epsilon), abs=1e-15)


def test_cpu_reference_rejects_a_detached_maximum_wrong_logit(monkeypatch: pytest.MonkeyPatch) -> None:
    production = task._batch_margin

    def detached_wrong(logits: torch.Tensor, labels: torch.Tensor, fixed_competitor: torch.Tensor | None = None):
        margin, competitor, tied, true_logits = production(logits, labels, fixed_competitor)
        rows = torch.arange(logits.shape[0], device=logits.device)
        faulty_margin = true_logits - logits[rows, competitor].detach()
        return faulty_margin, competitor, tied, true_logits

    monkeypatch.setattr(task, "_batch_margin", detached_wrong)
    with pytest.raises(task.CEMarginDirectionError, match="CE-decrease/margin-decrease counterexample"):
        task._cpu_references()


def test_production_margin_gradient_includes_wrong_logit_and_validates_row_mapping() -> None:
    images, labels, source_rows, _ = _population()
    consumer = _ToyConsumer()
    evaluators = {name: _ToyEvaluator(name, scale) for name, scale in (("E_S", 1.0), ("E_I", 2.0), ("E_C", 3.0))}
    gradients = task._mean_margin_gradients(consumer, evaluators, consumer.runtime.generator.phase, images[:200], labels[:200], source_rows[:200])
    task._validate_margin_population(gradients, labels[:200], source_rows[:200])
    for evaluator in task.GUARD_EVALUATORS:
        for class_id in (1, 7, 8, 9):
            assert gradients["unique_max_wrong_component_checks"][evaluator][str(class_id)]["passed"] is True
            assert len(gradients["fit_row_map"][evaluator][str(class_id)]["max_wrong_classes_in_fit_order"]) == 20
    bad = {**gradients, "fit_row_map": {name: {key: dict(value) for key, value in mapping.items()} for name, mapping in gradients["fit_row_map"].items()}}
    bad["fit_row_map"]["E_S"]["1"]["fit_positions"][0] = 0
    with pytest.raises(task.CEMarginDirectionError, match="wrong fit-row mask"):
        task._validate_margin_population(bad, labels[:200], source_rows[:200])


def test_exact_margin_ties_are_accounted_for_without_selecting_a_surrogate_gradient() -> None:
    images, labels, source_rows, _ = _population()
    consumer = _ToyConsumer()
    evaluators = {name: _TiedEvaluator() for name in task.GUARD_EVALUATORS}
    bundle = task._mean_margin_gradients(consumer, evaluators, consumer.runtime.generator.phase, images[:200], labels[:200], source_rows[:200])
    task._validate_margin_population(bundle, labels[:200], source_rows[:200])
    assert bundle["candidate_unavailable_for_ties"] is True
    assert len(bundle["exact_ties"]) == 600
    ce = {name: {class_id: torch.zeros(2) for class_id in range(10)} for name in task.GUARD_EVALUATORS}
    projection = task._project(torch.ones(2), ce, bundle["gradients"], 1.0)
    assert projection["candidate_available"] is False
    assert projection["exact_tie_exit"] is True


def test_projection_preserves_evaluator_constraints_independently_and_exact_ties_exit_cleanly() -> None:
    ce = {name: {class_id: torch.zeros(2) for class_id in range(10)} for name in task.GUARD_EVALUATORS}
    margin = {name: {class_id: torch.zeros(2) for class_id in range(10)} for name in task.GUARD_EVALUATORS}
    ce["E_S"][0] = torch.tensor([1.0, 0.0])
    ce["E_I"][0] = torch.tensor([-1.0, 0.0])
    source = torch.tensor([1.0, 1.0], dtype=torch.float64)
    projection = task._project(source, ce, margin, 1.0)
    assert projection["candidate_available"] is True
    assert torch.allclose(projection["projected_minimum_distance_vector"], torch.tensor([0.0, 1.0], dtype=torch.float64), atol=1e-8)
    assert max(projection["unit_constraint_products_after_equal_rms"].values()) <= projection["scaled_feasibility_tolerance"]
    tie_margin = {name: dict(values) for name, values in margin.items()}
    del tie_margin["E_C"][7]
    tied_projection = task._project(source, ce, tie_margin, 1.0)
    assert tied_projection["candidate_available"] is False
    assert tied_projection["exact_tie_exit"] is True
    assert tied_projection["missing_margin_gradient_classes"] == {"E_C": [7]}

    zero_ce = {name: {class_id: torch.zeros(2) for class_id in range(10)} for name in task.GUARD_EVALUATORS}
    margin_only = {name: {class_id: torch.zeros(2) for class_id in range(10)} for name in task.GUARD_EVALUATORS}
    margin_only["E_S"][0] = torch.tensor([1.0, 0.0])
    positive_margin_source = torch.tensor([1.0, 1.0], dtype=torch.float64)
    margin_projection = task._project(positive_margin_source, zero_ce, margin_only, 1.0)
    assert margin_projection["candidate_available"] is True
    assert torch.allclose(margin_projection["projected_minimum_distance_vector"], positive_margin_source)
    assert torch.dot(margin_only["E_S"][0].to(torch.float64), margin_projection["candidate_equal_rms_direction"].reshape(-1)) >= 0.0


def test_independent_margin_guard_rejects_a_projection_sign_flip(monkeypatch: pytest.MonkeyPatch) -> None:
    zero_ce = {name: {class_id: torch.zeros(2) for class_id in range(10)} for name in task.GUARD_EVALUATORS}
    margin = {name: {class_id: torch.zeros(2) for class_id in range(10)} for name in task.GUARD_EVALUATORS}
    margin["E_S"][0] = torch.tensor([1.0, 0.0])
    production = task._constraint_rows

    def flipped_margin_rows(ce_gradients, margin_gradients):
        rows, names, norms, zero = production(ce_gradients, margin_gradients)
        rows = rows.clone()
        for index, name in enumerate(names):
            if name.endswith(":margin"):
                rows[index] *= -1.0
        return rows, names, norms, zero

    monkeypatch.setattr(task, "_constraint_rows", flipped_margin_rows)
    with pytest.raises(task.CEMarginDirectionError, match="independently checked margin guard"):
        task._project(torch.tensor([-1.0, 1.0], dtype=torch.float64), zero_ce, margin, 1.0)


def test_each_phase_intervention_starts_from_the_same_frozen_value() -> None:
    phase = torch.nn.Parameter(torch.tensor([0.2, -0.4], dtype=torch.float64))
    baseline = phase.detach().clone()
    delta = torch.tensor([0.01, 0.03], dtype=torch.float64)
    plus = task.shared._phase_set(phase, baseline, delta)
    minus = task.shared._phase_set(phase, baseline, -delta)
    torch.testing.assert_close(plus, delta, rtol=0.0, atol=1e-16)
    torch.testing.assert_close(minus, -delta, rtol=0.0, atol=1e-16)
    torch.testing.assert_close(phase.detach(), baseline - delta, rtol=0.0, atol=0.0)


def test_completed_resume_returns_saved_report_without_preflight_or_forward(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    target = root / "outputs" / "attempt_01"
    target.mkdir(parents=True)
    identity = ResearchRunIdentity(
        task_id=task.TASK_ID,
        run_id=f"{task.TASK_ID}-attempt-01",
        trajectory_id="frozen-test",
        semantic_authority_id=task.TASK_ID,
        data_role="fixed TRAIN test population",
        config_identity={"updates": 0},
        model_identity={"source": "toy"},
        optimizer_identity={"updates": 0},
        sampler_identity={"rows": list(range(600))},
        execution_source_semantics_id=task.TASK_ID,
        scientific_attempt_id=f"{task.TASK_ID}:attempt-01",
    )
    population = {"task_id": task.TASK_ID, "rows": list(range(600))}
    conditions = ["baseline"]
    chunks = [task._condition_name(condition, index) for condition in conditions for index in range(task.CHUNK_COUNT)]
    for name in chunks:
        (target / name).write_bytes(b"completed")
    report = {"schema": task.REPORT_SCHEMA, "task_id": task.TASK_ID, "status": "VALID_COMPLETED", "research_run_identity": identity.to_dict()}
    (target / "report.json").write_text(json.dumps(report), encoding="utf-8")
    manifest = {
        "schema": task.MANIFEST_SCHEMA,
        "task_id": task.TASK_ID,
        "measurement_semantics": "v2 separate E_S/E_I/E_C class CE and margin constraints",
        "finite_difference_reduction": "rowwise paired plus-minus differences followed by float64 mean",
        "required_finite_difference_count": 48,
        "selected_forward_mode": "current_default",
        "preflight": {"numerical_scoring": {"selected_forward_mode": "current_default"}},
        "identity": identity.to_dict(),
        "population": population,
        "status": "COMPLETED",
        "measured_conditions": conditions,
        "completed_chunks": chunks,
    }
    manifest_path = target / "resume_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    inputs = {"class_guard_dir": target}
    monkeypatch.setattr(task, "_root", lambda: root)
    monkeypatch.setattr(task, "_load_inputs", lambda _root, _directory: inputs)
    monkeypatch.setattr(task, "_population_record", lambda _inputs, _output: population)
    monkeypatch.setattr(task, "_identity", lambda _inputs, _population, _output: identity)
    monkeypatch.setattr(task, "_preflight", lambda *args: pytest.fail("completed resume must not repeat preflight"))

    result = task._run(argparse.Namespace(
        class_guard_dir=Path("unused"),
        output_dir=Path("outputs/attempt_01"),
        preflight_only=False,
        resume_manifest=Path("outputs/attempt_01/resume_manifest.json"),
    ))
    assert result == report


def test_full_logits_to_gradient_projection_finite_response_and_decision() -> None:
    images, labels, source_rows, roles = _population()
    consumer = _ToyConsumer()
    evaluators = {name: _ToyEvaluator(name, scale) for name, scale in (("E_S", 1.0), ("E_I", 2.0), ("E_C", 3.0), ("E_D", 1.5))}
    fit_images, fit_labels = images[:200], labels[:200]
    ce_bundle = class_guard._class_ce_gradients(consumer, evaluators, consumer.runtime.generator.phase, fit_images, fit_labels)
    margin_bundle = task._mean_margin_gradients(consumer, evaluators, consumer.runtime.generator.phase, fit_images, fit_labels, source_rows[:200])
    task._validate_margin_population(margin_bundle, fit_labels, source_rows[:200])
    positions = torch.nonzero(fit_labels == 3, as_tuple=False).reshape(-1)
    full_batch = consumer.forward_loss(fit_images.index_select(0, positions), data_role=task.direction.bogt.TRAIN_ROLE)
    full_view = task.canonical.cmos_view(full_batch.result.raw_intensity)
    full_logits = evaluators["E_S"](full_view)[1]
    independent_ce = F.cross_entropy(full_logits, fit_labels.index_select(0, positions))
    independent_ce_gradient = torch.autograd.grad(independent_ce, consumer.runtime.generator.phase)[0]
    assert torch.allclose(ce_bundle["gradients"]["E_S"][3].to(torch.float64), independent_ce_gradient, rtol=1e-6, atol=1e-8)
    source = torch.tensor([-0.001, 0.0001], dtype=torch.float64)
    projection = task._project(source, ce_bundle["gradients"], margin_bundle["gradients"], task._rms(source))
    candidate = projection["candidate_equal_rms_direction"]
    assert projection["candidate_available"] is True
    assert candidate is not None
    for evaluator in task.GUARD_EVALUATORS:
        for class_id in range(10):
            assert torch.dot(ce_bundle["gradients"][evaluator][class_id].to(torch.float64).reshape(-1), candidate.reshape(-1)) <= 1e-9
            assert torch.dot(margin_bundle["gradients"][evaluator][class_id].to(torch.float64).reshape(-1), candidate.reshape(-1)) >= -1e-9
    baseline_phase = consumer.runtime.generator.phase.detach().clone()
    finite_difference = task._finite_differences(
        consumer,
        evaluators,
        {"images_600": images, "labels_600": labels, "source_indices_600": source_rows},
        consumer.runtime.generator.phase,
        baseline_phase,
        candidate,
        ce_bundle["gradients"],
        margin_bundle,
    )
    assert finite_difference["passed"] is True
    assert finite_difference["method"] == "float64 logits scoring; rowwise plus-minus pairing then float64 mean"
    assert finite_difference["executed_check_count"] == 48
    assert len(finite_difference["checks"]) == 48
    assert all("nonlinear_branch_changes_plus" in item for item in finite_difference["checks"].values())
    assert set(finite_difference["diagnostic_snapshots"]["baseline"]) == {"1", "7", "8", "9"}
    assert torch.equal(consumer.runtime.generator.phase.detach(), baseline_phase)
    assert consumer.runtime.generator.phase.grad is None
    assert all(parameter.grad is None for evaluator in evaluators.values() for parameter in evaluator.parameters())

    baseline_rows = _measured_rows(consumer, evaluators, images, labels, source_rows, roles)
    base_matrix = task._metric_matrix(baseline_rows)
    with torch.no_grad():
        consumer.runtime.generator.phase.copy_(baseline_phase + candidate)
    candidate_rows = _measured_rows(consumer, evaluators, images, labels, source_rows, roles)
    candidate_matrix = task._metric_matrix(candidate_rows)
    with torch.no_grad():
        consumer.runtime.generator.phase.copy_(baseline_phase)
    decision = task._decision(baseline_rows, candidate_rows, base_matrix, candidate_matrix, 1e-4, 1e-4, 1e-4)
    assert decision["classification"] == "CE_MARGIN_GUARDED_SHARED_LOCAL_RESPONSE_SUPPORTED"
    assert decision["confirmation_focus_response"]["1"]["beneficial_rows"] == 30
    assert decision["confirmation_focus_response"]["7"]["beneficial_rows"] == 30
    assert decision["training_recommendation"] == "NONE"
    assert decision["next_action"] == "NONE"

    tradeoff_rows = [dict(row) for row in candidate_rows]
    baseline_by_source = {int(row["source_row_index"]): row for row in baseline_rows}
    class_two_positions = [i for i, row in enumerate(tradeoff_rows) if row["role"] == "confirmation" and row["label"] == 2]
    for position in class_two_positions:
        source_index = int(tradeoff_rows[position]["source_row_index"])
        offset = -0.002 if position in class_two_positions[:23] else 0.0
        tradeoff_rows[position]["ed_margin"] = float(baseline_by_source[source_index]["ed_margin"]) + offset
    tradeoff = task._decision(baseline_rows, tradeoff_rows, base_matrix, candidate_matrix, 1e-4, 1e-4, 1e-4)
    assert tradeoff["confirmation_other_class_margin_changes"]["2"]["harmful_rows"] == 23
    assert tradeoff["confirmation_other_class_margin_changes"]["2"]["mean_margin_delta"] < -1e-4
    assert tradeoff["classification"] == "CE_MARGIN_GUARDED_RESPONSE_WITH_TRADEOFF"


def test_failed_preflight_saves_all_numeric_checks_and_row_level_diagnostics(tmp_path) -> None:
    target = tmp_path / "attempt"
    target.mkdir()
    checks = {f"check_{index}": {"passed": index != 7, "observed_slope": float(index)} for index in range(48)}
    preflight = {
        "status": "INVALID_HARNESS",
        "finite_difference_checks": {"executed_check_count": 48, "checks": checks},
        "_diagnostic_snapshots": {"baseline": {"raw": torch.ones((2, 1, 2, 2))}},
    }
    task._save_preflight_failure(target, {"rows": list(range(600))}, preflight, {"tensor": torch.ones(1)}, {"candidate_equal_rms_direction": torch.zeros(1), "actual_float32_displacements": {}})
    summary = json.loads((target / "preflight_diagnostics.json").read_text(encoding="utf-8"))
    evidence = torch.load(target / "preflight_numeric_evidence.pt", map_location="cpu", weights_only=True)
    assert len(summary["finite_difference_checks"]["checks"]) == 48
    assert sum(not item["passed"] for item in summary["finite_difference_checks"]["checks"].values()) == 1
    assert evidence["diagnostic_snapshots"]["baseline"]["raw"].shape == (2, 1, 2, 2)


def test_ready_receipt_rejects_missing_original_finite_difference_check() -> None:
    class _ReceiptConsumer:
        def __init__(self) -> None:
            self.correctness = None

        def make_ready_receipt(self, *, correctness, resource, implementation, **_kwargs):
            self.correctness = correctness
            return SimpleNamespace(status="READY" if correctness["passed"] else "BLOCKED")

    checks = {f"check_{index}": {"passed": True} for index in range(48)}
    preflight = {
        "status": "READY",
        "cpu_independent_references": {"passed": True},
        "target_positive_controls_qualified": True,
        "real_encoder_vs_fixed_ols": {"fit": {}, "confirmation": {}},
        "saved_ce_gradient_reference": {"passed": True},
        "margin_gradient_production": {"passed": True, "all_fit_rows_accounted_for": True},
        "projection": {"passed": True},
        "finite_difference_checks": {
            "method": "float64 logits scoring; rowwise plus-minus pairing then float64 mean",
            "passed": True,
            "executed_check_count": 48,
            "checks": checks,
        },
        "numerical_scoring": {
            "production_forward_dtype_unchanged": "float32",
            "score_dtype": "float64",
            "finite_difference_reduction": "rowwise paired plus-minus differences followed by float64 mean",
            "required_original_check_set_satisfied": True,
            "selected_forward_mode": "current_default",
            "strict_mode_adoption_check": {"passed": False},
        },
        "model_and_scientific_buffers_unchanged": True,
        "source_adam_unchanged": True,
        "evaluator_state_unchanged": True,
        "parameter_grad_buffers_empty": True,
        "optimizer_updates": 0,
        "storage_lifecycle": {
            "immutable_write_read_passed": True,
            "interrupted_manifest_recovery_passed": True,
            "completed_resume_without_forward_passed": True,
        },
        "cuda_before": {"total_bytes": 1, "free_bytes": 1},
        "cuda_after": {"total_bytes": 1, "free_bytes": 1},
        "disk_observation": {"passed": True},
        "runtime_dll": {"status": "already_present"},
    }
    consumer = _ReceiptConsumer()
    assert task._ready_receipt(consumer, preflight).status == "READY"
    del checks["check_47"]
    preflight["finite_difference_checks"]["executed_check_count"] = 47
    assert task._ready_receipt(consumer, preflight).status == "BLOCKED"
