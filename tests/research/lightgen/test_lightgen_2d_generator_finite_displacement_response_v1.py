from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from ldm_project.v2.contracts.planes import ComplexField, IntensityFrame, OperationProvenance, PlaneSpec
from ldm_project.v2.application.research.shared.research_run_control import ResearchRunIdentity
from scripts.research.lightgen import diagnose_lightgen_2d_generator_finite_displacement_response_v1 as task


def _candidate_block() -> dict[str, object]:
    rows = [
        {
            "fit_position": label,
            "within_batch": label,
            "source_row_index": 100 + label,
            "label": label,
            "best_wrong_exact_tie": False,
        }
        for label in task.DISPLAY_CLASSES
    ]
    return {
        "candidate_vectors_generator": {
            "ce_adam_saved": torch.tensor([0.1]),
            "ce_adam_zero_history": torch.tensor([0.3]),
            "ce_projected_adam_saved": torch.tensor([-0.05]),
            "ce_semantic_gradient_descent": torch.tensor([0.02]),
            "ce_component_gradient_descent_E_S": torch.tensor([-0.02]),
        },
        "rows": rows,
        "margin_gradient_fit_positions": list(task.DISPLAY_CLASSES),
        "margin_gradients_generator": torch.full((len(task.DISPLAY_CLASSES), 1), 0.01),
    }


def test_equal_rms_candidates_and_native_reset_keep_distinct_step_sizes() -> None:
    plan = task._condition_plan(_candidate_block(), 0)
    saved = plan[task._condition_name("adam_saved", 1.0, 1)]
    reset = plan[task._condition_name("adam_zero_history", 1.0, 1)]
    native = plan[task._condition_name("adam_zero_history", 1.0, 1, native=True)]
    assert saved["available"] is True and reset["available"] is True and native["available"] is True
    assert float(saved["delta"].abs().mean()) == pytest.approx(0.1)
    assert float(reset["delta"].abs().mean()) == pytest.approx(0.1)
    assert float(native["delta"].abs().mean()) == pytest.approx(0.3)
    assert saved["candidate_rms"] == pytest.approx(reset["candidate_rms"])
    assert native["candidate_rms"] == pytest.approx(0.3)
    assert task._condition_name("ed_oracle_class1", 1 / 8, 1) not in plan
    preflight_plan = task._condition_plan(_candidate_block(), 0, include_oracle_preflight_scales=True)
    assert preflight_plan[task._condition_name("ed_oracle_class1", 1 / 8, 1)]["available"] is True
    assert task._condition_name("adam_saved", 1 / 8, 1) not in preflight_plan
    assert sum("delta" in value for value in plan.values()) == 44
    assert sum("delta" in value for value in preflight_plan.values()) == 52
    assert "delta" not in plan["adam_saved"]  # A cached direction vector is not itself a measurement condition.


def test_each_phase_intervention_starts_from_the_exact_same_baseline() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
    baseline = parameter.detach().clone()
    delta = torch.tensor([0.5, 1.0])
    task._apply_displacement(parameter, baseline, delta)
    first = parameter.detach().clone()
    task._apply_displacement(parameter, baseline, delta)
    torch.testing.assert_close(parameter, first, rtol=0, atol=0)
    task._apply_displacement(parameter, baseline, delta, sign=-1)
    torch.testing.assert_close(parameter, baseline - delta, rtol=0, atol=0)
    task._restore_phase(parameter, baseline)
    torch.testing.assert_close(parameter, baseline, rtol=0, atol=0)


def test_finite_response_sign_matches_an_independent_linear_toy_and_rejects_wrong_sign() -> None:
    baseline = 0.4
    sensitivity = 2.0
    delta = 0.125
    plus = {"margins": [{"margin": baseline + sensitivity * delta, "tied_maximum": False, "best_wrong_class": 2}]}
    minus = {"margins": [{"margin": baseline - sensitivity * delta, "tied_maximum": False, "best_wrong_class": 2}]}
    correct = task._finite_difference_check(plus, minus, 0, sensitivity * delta, delta)
    wrong_sign = task._finite_difference_check(plus, minus, 0, -sensitivity * delta, delta)
    assert correct["passed"] is True
    assert correct["observed_symmetric_response"] == pytest.approx(sensitivity * delta)
    assert wrong_sign["passed"] is False


def test_exact_maximum_switch_makes_the_local_reference_unavailable() -> None:
    positive = {"margins": [{"margin": 0.2, "tied_maximum": False, "best_wrong_class": 2}]}
    negative = {"margins": [{"margin": 0.1, "tied_maximum": False, "best_wrong_class": 3}]}
    result = task._finite_difference_check(positive, negative, 0, 0.05, 0.1)
    assert result["same_maximum_wrong_class_both_sides"] is False
    assert result["passed"] is False


def test_image_review_panel_conversion_removes_batch_and_channel_axes() -> None:
    raw = torch.linspace(0.0, 1.0, 200 * 200).reshape(1, 200, 200)
    panel = task._panel_view(raw)
    assert panel.shape == (28, 28)
    assert torch.isfinite(panel).all()


class _ToyRuntime(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.generator_phase = torch.nn.Parameter(torch.zeros(1))
        self.encoder_phase = torch.nn.Parameter(torch.zeros(()))
        self.raw_plane = PlaneSpec("toy_cmos", (200, 200), (1e-6, 1e-6), 532e-9, 0.02, "camera")
        self.field_plane = PlaneSpec("toy_field", (2, 2), (1e-6, 1e-6), 532e-9, 0.01, "generator")
        self.provenance = OperationProvenance("toy", None, None)

    def forward_from_ols(self, z: torch.Tensor):
        labels = z[:, 0].real
        top = (0.2 + labels * 0.05).view(-1, 1, 1, 1).expand(-1, 1, 100, 200)
        bottom = torch.zeros((z.shape[0], 1, 100, 200), dtype=top.dtype, device=top.device)
        raw = torch.cat((top, bottom), dim=2) + self.generator_phase[0] * 0.01
        field = ComplexField(
            torch.ones((z.shape[0], 1, 2, 2), dtype=torch.complex64, device=z.device),
            self.field_plane,
            self.provenance,
        )
        camera = IntensityFrame(raw, self.raw_plane, self.provenance)
        return SimpleNamespace(
            raw_intensity=raw,
            raw_camera=camera,
            generator_input_field=field,
            generator_records=(field,),
            generator_output_field=field,
        )


class _ToyEvaluator(torch.nn.Module):
    def __init__(self, *, e_d: bool) -> None:
        super().__init__()
        self.e_d = e_d

    def forward(self, view: torch.Tensor):
        mean = view.mean(dim=(1, 2, 3))
        inferred_label = ((mean - 0.1) / 0.025).round().long().clamp(0, 9)
        logits = torch.full((view.shape[0], 10), -1.0, dtype=view.dtype, device=view.device)
        logits.scatter_(1, inferred_label[:, None], mean[:, None])
        wrong_label = (inferred_label + 1).remainder(10)
        logits.scatter_(1, wrong_label[:, None], 0.0)
        if self.e_d:
            return logits
        features = torch.zeros((view.shape[0], 128), dtype=view.dtype, device=view.device)
        return features, logits


def _integration_inputs():
    labels = torch.arange(10, dtype=torch.long)
    code = (0.2 + labels.to(torch.float32) * 0.05).view(10, 1, 1, 1)
    image = torch.zeros((10, 1, 28, 28))
    image[:, :, :14, :] = code
    target_raw = torch.zeros((10, 1, 200, 200))
    target_raw[:, :, :100, :] = (0.2 + labels.to(torch.float32) * 0.05).view(10, 1, 1, 1)
    z = torch.zeros((10, 100), dtype=torch.complex64)
    z[:, 0] = labels.to(torch.float32).to(torch.complex64)
    return labels, image, target_raw, z


def test_production_finite_response_readout_linear_prediction_and_decision_consume_same_batch() -> None:
    runtime = _ToyRuntime()
    consumer = SimpleNamespace(runtime=runtime, device=torch.device("cpu"))
    parameters = {"generator.phase": runtime.generator_phase, "encoder.phase": runtime.encoder_phase}
    evaluators = {name: _ToyEvaluator(e_d=name == "E_D") for name in task.EVALUATORS}
    labels, images, target_raw, z = _integration_inputs()
    source_readout = task._readout(consumer, evaluators, z)
    source_block = _candidate_block()
    source_block["readouts"] = {
        **source_readout,
        "z": z,
        "raw_cmos": source_readout["raw_cmos"],
        "target_slm_intensity": target_raw,
        "evaluator_logits": {name: source_readout["evaluator_logits"][name] for name in ("E_S", "E_I", "E_C")},
        "e_d_logits": source_readout["evaluator_logits"]["E_D"],
    }
    inputs = {"labels": labels, "images": images, "source_indices": torch.arange(100, 110)}
    runtime_before = task._snapshot(runtime)
    recorded = task._record_conditions(consumer, parameters, evaluators, inputs, source_block, 0, eta=1e-4)

    condition_name = task._condition_name("adam_saved", 1.0, 1)
    condition = recorded["conditions"][condition_name]
    assert len(condition["margins"]) == 10
    assert condition["margins"][1]["response_status"] == "BENEFICIAL"
    assert condition["delta_margin_linear_prediction"][1] == pytest.approx(0.001, abs=1e-8)
    assert condition["margins"][1]["delta_from_baseline"] == pytest.approx(0.001, abs=1e-6)
    assert recorded["phase_intervention_count"] == sum("margins" in value for value in recorded["conditions"].values())
    assert recorded["optical_forward_calls"] == recorded["phase_intervention_count"] + 1
    assert task._tree_equal(task._snapshot(runtime), runtime_before)
    assert all(parameter.grad is None for parameter in parameters.values())

    decision_rows = []
    for local, label in enumerate(labels.tolist()):
        decision_rows.append({
            "label": label,
            "conditions": {
                name: {"response_status": value["margins"][local]["response_status"]}
                for name, value in recorded["conditions"].items()
                if "margins" in value
            },
        })
    decision = task._decision(decision_rows, eta=1e-4)
    assert decision["classification"] == "MIXED_OR_INSUFFICIENT"  # one observed batch cannot satisfy 16/20
    assert decision["training_recommendation"] == "NONE"
    torch.testing.assert_close(runtime.generator_phase.detach(), torch.tensor([0.0]), rtol=0, atol=0)


def test_response_decision_requires_sixteen_of_twenty_actual_rows() -> None:
    saved = task._condition_name("adam_saved", 1.0, 1)
    reset = task._condition_name("adam_zero_history", 1.0, 1, native=True)
    oracle1 = task._condition_name("ed_oracle_class1", 1.0, 1)
    oracle7 = task._condition_name("ed_oracle_class7", 1.0, 1)
    rows = []
    for class_id in (*task.FOCUS_CLASSES, *task.CONTROL_CLASSES):
        for index in range(20):
            status = "BENEFICIAL" if index < (15 if class_id == 1 else 16) else "BOUNDARY"
            rows.append({"label": class_id, "conditions": {
                saved: {"response_status": status},
                reset: {"response_status": status},
                oracle1: {"response_status": "BENEFICIAL" if class_id == 1 and index < 16 else "BOUNDARY"},
                oracle7: {"response_status": "BENEFICIAL" if class_id == 7 and index < 16 else "BOUNDARY"},
            }})
    result = task._decision(rows, eta=1e-4)
    assert result["classification"] == "TRAINING_DISPLACEMENT_MISMATCH_WITH_LOCAL_RESPONSE"
    assert result["candidate_class_counts"]["adam_saved"]["1"]["BENEFICIAL"] == 15
    assert result["candidate_class_counts"]["adam_saved"]["7"]["BENEFICIAL"] == 16
    summaries = task._all_condition_summaries(rows, eta=1e-4)
    assert summaries[saved]["1"]["BENEFICIAL"] == 15


def test_finite_response_block_rejects_wrong_row_mapping_and_old_schema() -> None:
    inputs = {"source_indices": torch.arange(10), "labels": torch.arange(10)}
    block = {"source_row_indices": torch.arange(10), "labels": torch.arange(10)}
    payload = {
        "schema": task.BLOCK_SCHEMA,
        "task_id": task.TASK_ID,
        "margin_derivative_definition": task.MARGIN_DEFINITION,
        "batch_index": 0,
        "block": block,
    }
    task._validate_source_block(payload, 0, inputs)
    payload["block"] = {**block, "source_row_indices": torch.arange(10).roll(1)}
    with pytest.raises(task.FiniteResponseError, match="source rows changed"):
        task._validate_source_block(payload, 0, inputs)
    payload["schema"] = "m0_generator_finite_displacement_response_block.v0"
    with pytest.raises(task.FiniteResponseError, match="schema, task or batch mismatch"):
        task._validate_source_block(payload, 0, inputs)


def test_task_artifact_writers_roundtrip_and_refuse_overwrite(tmp_path) -> None:
    json_path = tmp_path / "preflight.json"
    json_payload = {"status": "READY", "optimizer_updates": 0}
    task._write_json_new(json_path, json_payload, "test preflight")
    assert json.loads(json_path.read_text(encoding="utf-8")) == json_payload
    with pytest.raises(task.FiniteResponseError, match="refusing to overwrite"):
        task._write_json_new(json_path, {"status": "INVALID"}, "test preflight")
    assert json.loads(json_path.read_text(encoding="utf-8")) == json_payload

    torch_path = tmp_path / "block.pt"
    torch_payload = {"schema": task.BLOCK_SCHEMA, "rows": torch.arange(10)}
    task._write_torch_new(torch_path, torch_payload, "test block")
    restored = torch.load(torch_path, map_location="cpu", weights_only=True)
    assert torch.equal(restored["rows"], torch.arange(10))
    with pytest.raises(task.FiniteResponseError, match="refusing to overwrite"):
        task._write_torch_new(torch_path, {"rows": torch.zeros(1)}, "test block")
    assert torch.equal(torch.load(torch_path, map_location="cpu", weights_only=True)["rows"], torch.arange(10))


def test_resource_telemetry_uses_this_task_identity_and_keeps_allocator_fields(tmp_path, monkeypatch) -> None:
    resource = {"global_used_fraction": 0.5, "pytorch_allocated_bytes": 100, "pytorch_reserved_bytes": 200}
    monkeypatch.setattr(task.direction, "_resource_snapshot", lambda device: dict(resource))
    path = tmp_path / "resource.jsonl"
    sample = task._sample_resource_telemetry(path, torch.device("cpu"), "test", progress={"completed_batches": 4}, started=0.0)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert sample["task_id"] == task.TASK_ID
    assert saved["task_id"] == task.TASK_ID
    assert saved["resource"]["pytorch_allocated_bytes"] == 100
    assert saved["resource"]["pytorch_reserved_bytes"] == 200


def test_completed_manifest_restart_returns_saved_report_without_measurement(tmp_path, monkeypatch) -> None:
    identity = ResearchRunIdentity(
        task_id=task.TASK_ID,
        run_id="M0_GENERATOR_FINITE_DISPLACEMENT_RESPONSE_V1-attempt-01",
        trajectory_id="frozen-test",
        semantic_authority_id="direction-test",
        data_role="fixed TRAIN fit rows",
        config_identity={"lambda": [0.25, 1.0, 4.0]},
        model_identity={"generator_step": 2500},
        optimizer_identity={"updates": 0},
        sampler_identity={"rows": list(range(200))},
        execution_source_semantics_id=task.TASK_ID,
        scientific_attempt_id=f"{task.TASK_ID}:attempt-01",
    )
    target = tmp_path / "completed"
    target.mkdir()
    completed_blocks = []
    for batch_index in range(task.BATCH_COUNT):
        name = task._block_name(batch_index)
        (target / name).write_bytes(b"preserved complete block")
        completed_blocks.append(name)
    saved_report = {"schema": task.REPORT_SCHEMA, "task_id": task.TASK_ID, "status": "VALID_COMPLETED", "research_run_identity": identity.to_dict(), "classification": "MIXED_OR_INSUFFICIENT"}
    (target / "report.json").write_text(json.dumps(saved_report), encoding="utf-8")
    manifest = {"status": "COMPLETED", "completed_blocks": completed_blocks}
    monkeypatch.setattr(task, "_readout", lambda *args, **kwargs: pytest.fail("completed recovery performed a new optical forward"))
    assert task._terminal_report(target, manifest, identity) == saved_report
