from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from ldm_project.v2.application.research.shared.research_run_control import CORRECTNESS_INVARIANTS, ResearchRunControl, ResearchRunIdentity, RunControlError
from scripts.research.lightgen import diagnose_lightgen_2d_generator_semantic_direction_adam_v1 as task


def _population() -> dict[str, object]:
    fit_labels = torch.arange(10).repeat(20)
    historic_labels = torch.arange(10).repeat(10)
    confirmation_labels = torch.arange(10).repeat_interleave(30)
    labels = torch.cat((fit_labels, historic_labels, confirmation_labels))
    rows = torch.arange(600)
    return {
        "source_row_indices": rows.tolist(),
        "labels": labels.tolist(),
        "roles": ["fit"] * 200 + ["historic_holdout"] * 100 + ["confirmation"] * 300,
        "original_fit_rows": 200,
        "historic_holdout_rows": 100,
        "new_confirmation_rows": 300,
        "focus_classes": [1, 7],
        "confirmation_indices": rows[300:].tolist(),
    }


def _identity() -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=task.TASK_ID,
        run_id=f"{task.TASK_ID}-attempt-01",
        trajectory_id="frozen-test",
        semantic_authority_id="parent-test",
        data_role="fixed TRAIN fit rows only",
        config_identity={"grid": 2048},
        model_identity={"phase_shape": [2, 1, 100, 100]},
        optimizer_identity={"source_adam": "read-only"},
        sampler_identity={"indices": list(range(200))},
        execution_source_semantics_id=task.TASK_ID,
        scientific_attempt_id=f"{task.TASK_ID}:attempt-01",
    )


def _decision_rows(setup) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    for endpoint in task.ENDPOINTS:
        rows = []
        for class_id in (*task.FOCUS_CLASSES, *task.CONTROL_CLASSES):
            for row_index in range(20):
                rows.append({"label": class_id, "directions": setup(endpoint, class_id, row_index)})
        result[endpoint] = rows
    return result


def _statuses(**overrides: str) -> dict[str, dict[str, str]]:
    names = (
        "ce_semantic_gradient_descent",
        "ce_gradient_descent",
        "ce_adam_saved",
        "ce_adam_zero_history",
        "ce_projected_adam_saved",
    )
    return {name: {"status": overrides.get(name, "BOUNDARY")} for name in names}


def test_fixed_population_and_fit_batch_exclude_readout_roles() -> None:
    population = _population()
    rows, labels = task._validate_population_vectors(population)
    assert rows.shape == labels.shape == (600,)
    assert task._fit_batch_slice(19) == (190, 200)
    assert int(labels[task._fit_batch_slice(19)[0]]) == 0
    assert int(labels[task._fit_batch_slice(19)[1] - 1]) == 9
    population["roles"][205] = "fit"  # type: ignore[index]
    with pytest.raises(task.DirectionDiagnosticError, match="data roles changed"):
        task._validate_population_vectors(population)
    with pytest.raises(task.DirectionDiagnosticError, match="fixed TRAIN fit population"):
        task._fit_batch_slice(20)


def test_preflight_population_consumer_uses_the_saved_roles_and_source_rows() -> None:
    population = _population()
    inputs = {
        "population": population,
        "images": torch.zeros((200, 1, 28, 28)),
        "labels": torch.tensor(population["labels"][:200]),
        "source_indices": torch.tensor(population["source_row_indices"][:200]),
    }
    assert task._fixed_fit_population_matches(inputs) is True
    population["roles"][199] = "historic_holdout"
    assert task._fixed_fit_population_matches(inputs) is False


def test_canonical_semantic_direction_reuses_three_evaluator_combination() -> None:
    gradients = {
        name: {
            "encoder.phase": torch.tensor([float(index + 1), 0.5]),
            "generator.phase": torch.tensor([0.25, float(index + 2)]),
        }
        for index, name in enumerate(task.EVALUATORS)
    }
    result, metadata = task._canonical_semantic(gradients)
    expected, unavailable = task.base._safe_canonical_gradient({f"L_{name[-1]}": gradients[name] for name in task.EVALUATORS})
    assert unavailable is None
    assert metadata["status"] == "AVAILABLE"
    assert result is not None and expected is not None
    for name in ("encoder.phase", "generator.phase"):
        torch.testing.assert_close(result[name], expected[name], rtol=0, atol=0)
    assert metadata["evaluator_normalization"]["E_S"]["joint_l2_norm"] > 0


def test_generator_auxiliary_projection_uses_canonical_projection() -> None:
    semantic = torch.tensor([1.0, 0.0, 0.0])
    auxiliary = torch.tensor([-2.0, 3.0, 1.0])
    projected, record = task._project_auxiliary(auxiliary, semantic)
    assert record["available"] is True and record["projected"] is True
    assert abs(float(torch.dot(projected, semantic))) < 1e-7
    unchanged, unchanged_record = task._project_auxiliary(torch.tensor([2.0, 1.0, 0.0]), semantic)
    assert unchanged_record["projected"] is False
    torch.testing.assert_close(unchanged, torch.tensor([2.0, 1.0, 0.0]))
    unavailable, unavailable_record = task._project_auxiliary(auxiliary, torch.zeros_like(semantic))
    assert unavailable_record["available"] is False
    assert torch.count_nonzero(unavailable) == 0


def test_readonly_adam_candidates_match_independent_cpu_optimizer() -> None:
    result = task._adam_formula_cpu_reference_check()
    assert result["passed"] is True
    assert set(result["maximum_absolute_difference"]) == {"saved", "zero_first", "zero_all"}
    state = {
        "state": {1: {"step": torch.tensor(5.0), "exp_avg": torch.tensor([0.2, -0.1]), "exp_avg_sq": torch.tensor([0.4, 0.3])}},
        "param_groups": [{"betas": (0.9, 0.999), "lr": 1e-3, "eps": 1e-8}],
    }
    before = task._cpu_tree(state)
    candidates = {variant: task._adam_candidate(torch.tensor([0.3, 0.15]), state, variant=variant) for variant in ("saved", "zero_first", "zero_all")}
    assert all(torch.isfinite(value).all() for value in candidates.values())
    assert task._tree_equal(before, state)


def test_direction_sign_boundary_zero_and_tie_are_explicit() -> None:
    h = torch.tensor([1.0, 0.0])
    assert task._direction_record(h, torch.tensor([1.0, 0.0]))["status"] == "BENEFICIAL"
    assert task._direction_record(h, torch.tensor([-1.0, 0.0]))["status"] == "HARMFUL"
    assert task._direction_record(h, torch.tensor([0.01, 1.0]))["status"] == "BOUNDARY"
    assert task._direction_record(torch.zeros_like(h), h)["status"] == "UNDECIDABLE_ZERO_GRADIENT"
    assert task._direction_record(h, h, tied_maximum=True)["status"] == "UNDECIDABLE_TIED_MAXIMUM"


def test_margin_production_expression_matches_independent_derivative_and_finite_difference() -> None:
    result = task._margin_derivative_cpu_reference_check()
    assert result["passed"] is True
    reversal = result["cases"]["sign_reversal"]
    assert reversal["observed_derivative"] == -1.0
    assert reversal["central_finite_difference"] == pytest.approx(-1.0, abs=1e-9)
    common_shift = result["cases"]["common_logit_shift"]
    assert common_shift["observed_derivative"] == 0.0
    assert result["cases"]["exact_tie"]["best_wrong_tied_classes"] == [1, 2]

    phi = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    logits = torch.stack((phi, 2 * phi + 1, phi * 0 - 5))
    wrong = logits.detach().clone()
    wrong[0] = -torch.inf
    selected = int(wrong.argmax())
    broken_detached_gradient = torch.autograd.grad(logits[0] - logits[selected].detach(), phi)[0]
    assert float(broken_detached_gradient) == 1.0
    assert float(broken_detached_gradient) != reversal["central_finite_difference"]


class _ToyDirectionConsumer:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.encoder = torch.nn.Parameter(torch.tensor([0.0]))
        self.generator = torch.nn.Parameter(torch.tensor([0.0]))

    def forward_loss(self, images: torch.Tensor, *, data_role: str):
        assert data_role == task.bogt.TRAIN_ROLE
        batch = images.shape[0]
        signal = images.mean(dim=(1, 2, 3)) - 0.5 + self.generator + 0.01 * self.encoder
        raw = (torch.full((batch, 1, 200, 200), 0.5) + signal[:, None, None, None]).contiguous()
        target = torch.nn.functional.interpolate(images, size=(200, 200), mode="bilinear", align_corners=False)
        optical = SimpleNamespace(
            raw_intensity=raw,
            image_input=SimpleNamespace(slm_intensity=target),
            z_ols=SimpleNamespace(coefficients=torch.zeros(batch, 100, dtype=torch.complex64)),
            encoder_records=(SimpleNamespace(tensor=torch.ones(batch, 1, 1, dtype=torch.complex64), plane=SimpleNamespace(pixel_pitch_m=(1.0, 1.0))),) * 2,
            generator_input_field=SimpleNamespace(tensor=torch.ones(batch, 1, 1, dtype=torch.complex64), plane=SimpleNamespace(pixel_pitch_m=(1.0, 1.0))),
            generator_records=(SimpleNamespace(tensor=torch.ones(batch, 1, 1, dtype=torch.complex64), plane=SimpleNamespace(pixel_pitch_m=(1.0, 1.0))),) * 2,
        )
        return SimpleNamespace(result=optical, L_dist=self.encoder.square() + 0.1, L_op=self.generator.square() + 0.1)


class _ToyDirectionEvaluator(torch.nn.Module):
    def __init__(self, *, return_features: bool, tied: bool = False) -> None:
        super().__init__()
        self.return_features = return_features
        self.tied = tied

    def forward(self, values: torch.Tensor):
        signal = values.mean(dim=(1, 2, 3))
        logits = torch.full((values.shape[0], 10), -5.0, dtype=values.dtype, device=values.device)
        logits = logits.scatter(1, torch.ones((values.shape[0], 1), dtype=torch.long, device=values.device), signal[:, None])
        wrong_logits = 2 * signal + 1
        logits = logits.scatter(1, torch.full((values.shape[0], 1), 2, dtype=torch.long, device=values.device), wrong_logits[:, None])
        if self.tied:
            tied_logits = torch.where(signal > 0.8, wrong_logits, torch.full_like(wrong_logits, -5.0))
            logits = logits.scatter(1, torch.full((values.shape[0], 1), 3, dtype=torch.long, device=values.device), tied_logits[:, None])
        if not self.return_features:
            return logits
        features = torch.cat((signal[:, None], torch.full((values.shape[0], 127), 0.25, device=values.device)), dim=1)
        return features, logits


def _toy_optimizer_state() -> dict[str, object]:
    return {
        "state": {
            0: {"step": torch.tensor(2200.0), "exp_avg": torch.zeros(1), "exp_avg_sq": torch.ones(1)},
            1: {"step": torch.tensor(2200.0), "exp_avg": torch.zeros(1), "exp_avg_sq": torch.ones(1)},
        },
        "param_groups": [{"betas": (0.9, 0.999), "lr": 1e-3, "eps": 1e-8}],
    }


def _record_toy_batch(batch_index: int, *, tied: bool = False) -> dict[str, object]:
    consumer = _ToyDirectionConsumer()
    parameters = {"encoder.phase": consumer.encoder, "generator.phase": consumer.generator}
    evaluators = {
        name: _ToyDirectionEvaluator(return_features=name != "E_D", tied=tied and name == "E_D")
        for name in task.ALL_EVALUATORS
    }
    for evaluator in evaluators.values():
        for parameter in evaluator.parameters():
            parameter.requires_grad_(False)
    images = torch.full((200, 1, 28, 28), 0.5)
    if tied:
        images[batch_index * task.BATCH_SIZE + 1] = 0.9
    labels = torch.arange(10).repeat(20)
    rows = torch.arange(600)
    return task._record_batch(consumer, parameters, evaluators, _toy_optimizer_state(), images, labels, rows, batch_index, "CPU_PRODUCTION_PATH")


def test_record_batch_produces_correct_margin_gradient_and_decision_rows() -> None:
    rows_by_endpoint: dict[str, list[dict[str, object]]] = {}
    selected_row = None
    selected_batch = None
    for endpoint in task.ENDPOINTS:
        endpoint_rows = []
        for batch_index in range(task.BATCH_COUNT):
            block = _record_toy_batch(batch_index)
            endpoint_rows.extend({**row, "endpoint": endpoint} for row in block["rows"])
            if endpoint == "B_class_ce" and batch_index == 0:
                selected_batch = block
                selected_row = next(row for row in block["rows"] if row["label"] == 1)
            assert block["margin_gradient_fit_positions"] == [row["fit_position"] for row in block["rows"]]
        rows_by_endpoint[endpoint] = endpoint_rows

    assert selected_batch is not None and selected_row is not None
    assert selected_batch["margin_derivative_reference"] is None
    assert selected_row["best_wrong_class"] == 2
    assert selected_row["ed_margin"] == pytest.approx(-1.5)
    assert selected_row["margin_gradient_generator_l2"] == pytest.approx(1.0)
    assert selected_batch["margin_gradients_generator"].shape == (5, 1)
    assert len(rows_by_endpoint["B_class_ce"]) == 100

    h = selected_batch["margin_gradients_generator"][1].to(torch.float64)
    candidate = selected_batch["candidate_vectors_generator"]["ce_semantic_gradient_descent"].reshape(-1).to(torch.float64)
    expected_dot = float(torch.dot(h, candidate))
    assert selected_row["directions"]["ce_semantic_gradient_descent"]["dot"] == pytest.approx(expected_dot)
    decision = task._decision(rows_by_endpoint)
    assert decision["endpoint_class_summaries"]["B_class_ce"]["1"]["ce_semantic_gradient_descent"]["rows"] == 20


def test_record_batch_marks_tied_margin_undecidable_without_placeholder_gradient() -> None:
    block = _record_toy_batch(0, tied=True)
    row = next(value for value in block["rows"] if value["label"] == 1)
    assert row["best_wrong_exact_tie"] is True
    assert row["best_wrong_tied_classes"] == [2, 3]
    assert row["margin_gradient_generator_l2"] is None
    assert row["directions"]["ce_semantic_gradient_descent"]["status"] == "UNDECIDABLE_TIED_MAXIMUM"
    assert 1 not in block["margin_gradient_fit_positions"]
    assert block["margin_gradients_generator"].shape[0] == 4


def test_preflight_candidate_replay_is_observed_but_saved_readouts_remain_gated() -> None:
    logits = {name: torch.zeros(2, 10) for name in task.EVALUATORS}
    readouts = {
        "source_indices": torch.tensor([10, 11]),
        "labels": torch.tensor([0, 1]),
        "z": torch.zeros(2, 100, dtype=torch.complex64),
        "raw_cmos": torch.zeros(2, 1, 200, 200),
        "target_slm_intensity": torch.zeros(2, 1, 200, 200),
        "power_by_plane": torch.zeros(2, 5),
        "evaluator_logits": logits,
        "e_d_logits": torch.zeros(2, 10),
    }
    expected = {"candidate_vectors_generator": {"adam_zero_history": torch.zeros(4)}, "readouts": readouts}
    actual_readouts = {key: ({name: value.clone() for name, value in value.items()} if key == "evaluator_logits" else value.clone()) for key, value in readouts.items()}
    actual = {"candidate_vectors_generator": {"adam_zero_history": torch.full((4,), 1e-6)}, "readouts": actual_readouts}
    result = task._compare_preflight_batch(actual, expected, "step2200")
    assert result["candidate_comparison_is_observation_only"] is True
    assert result["candidate_vector_within_tolerance"]["adam_zero_history"] is False
    assert result["readout_replay_passed"] is True
    actual_readouts["e_d_logits"][0, 0] = 1e-3
    with pytest.raises(task.DirectionDiagnosticError, match="readout replay differs"):
        task._compare_preflight_batch(actual, expected, "step2200")


def test_optimizer_candidate_requires_sixteen_paired_rows_for_both_focus_classes() -> None:
    def setup(endpoint: str, class_id: int, index: int) -> dict[str, dict[str, str]]:
        status = _statuses(
            ce_semantic_gradient_descent="BENEFICIAL",
            ce_gradient_descent="BENEFICIAL",
            ce_adam_saved="HARMFUL" if index < 16 and class_id in task.FOCUS_CLASSES else "BOUNDARY",
            ce_adam_zero_history="BENEFICIAL",
            ce_projected_adam_saved="BENEFICIAL",
        )
        return status if endpoint == "B_class_ce" else _statuses()

    decision = task._decision(_decision_rows(setup))
    assert decision["classification"] == "OPTIMIZER_HISTORY_DIRECTION_CANDIDATE"
    assert decision["paired_row_counts"]["optimizer_reversal"] == {"1": 16, "7": 16}

    def below_threshold(endpoint: str, class_id: int, index: int) -> dict[str, dict[str, str]]:
        status = _statuses(
            ce_semantic_gradient_descent="BENEFICIAL",
            ce_gradient_descent="BENEFICIAL",
            ce_adam_saved="HARMFUL" if index < 15 and class_id in task.FOCUS_CLASSES else "BOUNDARY",
            ce_adam_zero_history="BENEFICIAL",
            ce_projected_adam_saved="BENEFICIAL",
        )
        return status if endpoint == "B_class_ce" else _statuses()

    assert task._decision(_decision_rows(below_threshold))["classification"] == "MIXED_OR_INSUFFICIENT"


def test_auxiliary_interference_requires_semantic_to_total_reversal_and_projection() -> None:
    def setup(endpoint: str, class_id: int, index: int) -> dict[str, dict[str, str]]:
        if endpoint != "B_class_ce":
            return _statuses()
        focus = class_id in task.FOCUS_CLASSES
        return _statuses(
            ce_semantic_gradient_descent="BENEFICIAL" if focus else "BOUNDARY",
            ce_gradient_descent="HARMFUL" if focus and index < 16 else "BOUNDARY",
            ce_adam_saved="HARMFUL" if focus and index < 16 else "BOUNDARY",
            ce_adam_zero_history="BOUNDARY",
            ce_projected_adam_saved="BENEFICIAL",
        )

    decision = task._decision(_decision_rows(setup))
    assert decision["classification"] == "AUXILIARY_INTERFERENCE_DIRECTION_CANDIDATE"
    assert decision["paired_row_counts"]["auxiliary_interference"] == {"1": 16, "7": 16}


def test_no_optimizer_reversal_means_no_training_candidate() -> None:
    def setup(endpoint: str, class_id: int, index: int) -> dict[str, dict[str, str]]:
        del endpoint, class_id, index
        return _statuses(**{name: "BENEFICIAL" for name in (
            "ce_semantic_gradient_descent",
            "ce_gradient_descent",
            "ce_adam_saved",
            "ce_adam_zero_history",
            "ce_projected_adam_saved",
        )})

    decision = task._decision(_decision_rows(setup))
    assert decision["classification"] == "MIXED_OR_INSUFFICIENT"
    assert decision["training_recommendation"] == "NONE"
    assert decision["next_action"] == "NONE"


def test_flattened_direction_rows_keep_endpoint_for_report_table() -> None:
    flattened = task._class_rows({"B_class_ce": [{"rows": [{"fit_position": 7, "label": 1}]}]})
    assert flattened["B_class_ce"][0] == {"fit_position": 7, "label": 1, "endpoint": "B_class_ce"}


def test_orphan_block_recovery_checks_fixed_batch_rows(tmp_path) -> None:
    batch_index = 3
    rows = torch.arange(200)
    labels = torch.arange(10).repeat(20)
    expected_rows = rows[30:40]
    block = {"readouts": {"source_indices": expected_rows.clone(), "labels": labels[30:40].clone()}}
    path = tmp_path / task._block_name("B_class_ce", batch_index)
    block["margin_derivative_definition"] = task.MARGIN_DEFINITION
    torch.save({"schema": task.BLOCK_SCHEMA, "task_id": task.TASK_ID, "margin_derivative_definition": task.MARGIN_DEFINITION, "endpoint": "B_class_ce", "batch_index": batch_index, "block": block}, path)
    inputs = {"source_indices": rows, "labels": labels}
    assert torch.equal(task._read_orphan_block(path, "B_class_ce", batch_index, inputs)["readouts"]["source_indices"], expected_rows)
    block["readouts"]["source_indices"][0] = 999
    torch.save({"schema": task.BLOCK_SCHEMA, "task_id": task.TASK_ID, "margin_derivative_definition": task.MARGIN_DEFINITION, "endpoint": "B_class_ce", "batch_index": batch_index, "block": block}, path)
    with pytest.raises(task.DirectionDiagnosticError, match="source rows differ"):
        task._read_orphan_block(path, "B_class_ce", batch_index, inputs)


def test_resume_identity_and_terminal_recovery_do_not_reopen_measurement(tmp_path) -> None:
    identity = _identity()
    population = {"source_row_indices": list(range(200)), "labels": list(range(200))}
    target = tmp_path
    manifest_path = target / "resume_manifest.json"
    manifest = {"schema": task.MANIFEST_SCHEMA, "task_id": task.TASK_ID, "margin_derivative_definition": task.MARGIN_DEFINITION, "identity": identity.to_dict(), "population": population, "status": "RUNNING", "completed_blocks": []}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    validated = task._validate_resume_manifest(manifest_path, target, population, identity)
    report = {"schema": task.REPORT_SCHEMA, "task_id": task.TASK_ID, "margin_derivative_definition": task.MARGIN_DEFINITION, "status": "VALID_COMPLETED", "research_run_identity": identity.to_dict(), "training_updates": 0, "next_action": "NONE"}
    expected_blocks = [task._block_name(endpoint, index) for endpoint in task.ENDPOINTS for index in range(task.BATCH_COUNT)]
    for name in expected_blocks:
        (target / name).touch()
    validated["completed_blocks"] = expected_blocks
    manifest_path.write_text(json.dumps(validated), encoding="utf-8")
    (target / "report.json").write_text(json.dumps(report), encoding="utf-8")
    recovered = task._terminal_report_if_present(target, validated, identity)
    assert recovered == report
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["status"] == "COMPLETED"
    assert task._terminal_report_if_present(target, validated, identity) == report


def test_resume_rejects_v1_derivative_artifacts(tmp_path) -> None:
    identity = _identity()
    population = {"source_row_indices": list(range(200)), "labels": list(range(200))}
    manifest_path = tmp_path / "resume_manifest.json"
    manifest_path.write_text(json.dumps({"schema": "m0_generator_semantic_direction_adam_manifest.v1", "task_id": task.TASK_ID, "population": population, "status": "RUNNING"}), encoding="utf-8")
    with pytest.raises(task.DirectionDiagnosticError, match="schema/derivative semantics mismatch"):
        task._validate_resume_manifest(manifest_path, tmp_path, population, identity)


def _ready_preflight(*, margin_cpu_passed: bool = True) -> dict[str, object]:
    target_metrics = {
        name: {"macro_recall": 1.0, "per_class_correct": [20] * 10}
        for name in task.ALL_EVALUATORS
    }
    gpu_references = {
        endpoint: {
            "passed": True,
            "rows": [{"label": class_id, "passed": True} for class_id in task.ROW_CLASSES],
            "undecidable_tied_rows": [],
        }
        for endpoint in task.ENDPOINTS
    }
    representative = {
        endpoint: {
            "no_grad_then_autograd_equal": True,
            "parent_readout_replay": {"passed": True},
            "runtime_state_unchanged": True,
            "references_and_weights_unchanged": True,
            "source_adam_unchanged": True,
            "parameter_grad_buffers_empty": True,
            "margin_derivative_reference": gpu_references[endpoint],
        }
        for endpoint in task.ENDPOINTS
    }
    return {
        "margin_derivative_cpu_reference": {"passed": margin_cpu_passed},
        "gpu_margin_derivative_references": gpu_references,
        "target_positive_control_qualified": True,
        "target_positive_control": target_metrics,
        "negative_label_control": {"different_generator_and_encoder_gradient": True},
        "representative_autograd_batches": representative,
        "fixed_fit_rows_and_roles": True,
        "parameter_grad_buffers_empty": True,
        "evaluator_weights_frozen": True,
        "source_adam_tensor_state_modified": False,
        "cuda_runtime": {"status": "prepended"},
        "cuda_before": {"global_used_fraction": 0.5},
        "cuda_after": {"global_used_fraction": 0.5},
        "disk_observation": {"free_bytes": 2_000_000_000, "estimated_output_bytes": 100, "required_remaining_bytes": 1_500_000_000},
    }


def test_ready_receipt_requires_executed_margin_reference_results() -> None:
    control = ResearchRunControl(identity=_identity(), authorization_id="test-authorization")
    receipt = task._ready_receipt(control, _ready_preflight())
    control.mark_preflight_ready(receipt)
    assert receipt.checks[CORRECTNESS_INVARIANTS]["margin_derivative_cpu_reference"] is True
    assert receipt.checks[CORRECTNESS_INVARIANTS]["margin_derivative_gpu_references"] is True

    blocked_control = ResearchRunControl(identity=_identity(), authorization_id="test-authorization")
    with pytest.raises(RunControlError, match="check group did not pass"):
        task._ready_receipt(blocked_control, _ready_preflight(margin_cpu_passed=False))
