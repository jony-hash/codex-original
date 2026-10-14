from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from ldm_project.v2.contracts.planes import ComplexField, IntensityFrame, OperationProvenance, PlaneSpec
from scripts.research.lightgen import diagnose_lightgen_2d_generator_shared_direction_transfer_v1 as task


def _optimizer_state(first: float = 0.13, second: float = 0.2) -> dict[str, object]:
    return {
        "state": {
            1: {
                "step": torch.tensor(2500.0),
                "exp_avg": torch.tensor([first], dtype=torch.float32),
                "exp_avg_sq": torch.tensor([second], dtype=torch.float32),
            }
        },
        "param_groups": [{"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8}],
    }


def _direction_blocks(*, zero: bool = False, alternating: bool = False) -> list[dict[str, object]]:
    result = []
    for index in range(20):
        value = 0.0 if zero else (0.03 if index < 15 else -0.01) if alternating else 0.01 + index * 0.002
        descent = torch.tensor([value], dtype=torch.float32)
        auxiliary = torch.tensor([0.0 if zero else (-0.004 + index * 0.0003)], dtype=torch.float32)
        result.append({
            "candidate_vectors_generator": {
                "ce_semantic_gradient_descent": descent,
                "ce_bogt_auxiliary_gradient_descent": auxiliary,
                "ce_gradient_descent": descent + auxiliary,
            }
        })
    return result


def _pytorch_adam_candidate(gradient: torch.Tensor, saved: dict[str, object]) -> torch.Tensor:
    group = saved["param_groups"][0]
    state = saved["state"][1]
    parameter = torch.nn.Parameter(torch.zeros_like(gradient))
    optimizer = torch.optim.Adam([parameter], lr=group["lr"], betas=group["betas"], eps=group["eps"], foreach=False)
    optimizer.state[parameter] = {
        "step": state["step"].clone(),
        "exp_avg": state["exp_avg"].clone(),
        "exp_avg_sq": state["exp_avg_sq"].clone(),
    }
    parameter.grad = gradient.clone()
    optimizer.step()
    return parameter.detach().clone()


def test_shared_adam_is_applied_once_after_meaning_gradient_and_decomposition_is_consumed() -> None:
    blocks = _direction_blocks(alternating=True)
    optimizer = _optimizer_state(first=0.0, second=0.0)
    result = task._build_shared_directions(blocks, optimizer)
    semantic_mean = torch.stack([block["candidate_vectors_generator"]["ce_semantic_gradient_descent"] for block in blocks]).mean(0)
    full_mean = torch.stack([block["candidate_vectors_generator"]["ce_gradient_descent"] for block in blocks]).mean(0)
    expected_s_adam = _pytorch_adam_candidate(-semantic_mean, optimizer)
    expected_f_adam = _pytorch_adam_candidate(-full_mean, optimizer)
    torch.testing.assert_close(result["batch_mean_descent_directions"]["S_GD"], semantic_mean)
    torch.testing.assert_close(result["batch_mean_descent_directions"]["F_GD"], full_mean)
    torch.testing.assert_close(result["batch_mean_descent_directions"]["S_ADAM"], expected_s_adam)
    torch.testing.assert_close(result["batch_mean_descent_directions"]["F_ADAM"], expected_f_adam)
    assert result["adam_applied_once_to_mean_gradient"] is True
    assert result["mean_of_batch_adam_candidates_used"] is False
    assert result["mean_decomposition_max_abs_residual"] < 1e-7

    per_batch_adam = torch.stack([_pytorch_adam_candidate(-block["candidate_vectors_generator"]["ce_semantic_gradient_descent"], optimizer) for block in blocks]).mean(0)
    assert not torch.allclose(expected_s_adam, per_batch_adam, rtol=1e-3, atol=1e-6)
    rms_values = [task._rms(value) for value in result["equal_rms_directions"].values()]
    assert max(rms_values) - min(rms_values) < 1e-8
    assert task._rms(result["equal_rms_directions"]["F_ADAM"]) == pytest.approx(result["reference_rms"])


def test_bad_decomposition_fails_and_zero_directions_remain_unavailable() -> None:
    bad = _direction_blocks()
    bad[4]["candidate_vectors_generator"]["ce_bogt_auxiliary_gradient_descent"] += 0.2
    with pytest.raises(task.SharedDirectionError, match="do not sum"):
        task._build_shared_directions(bad, _optimizer_state())

    zero = task._build_shared_directions(_direction_blocks(zero=True), _optimizer_state(first=0.0, second=0.0))
    conditions = task._condition_vectors(zero)
    assert len(conditions) == 10
    assert all(spec["available"] is False for spec in conditions.values())
    assert all("planned_delta" not in spec and spec["reason"].startswith("UNDECIDABLE") for spec in conditions.values())


def test_float32_phase_application_always_restarts_from_the_same_base() -> None:
    phase = torch.nn.Parameter(torch.tensor([0.25, -0.5], dtype=torch.float32))
    base = phase.detach().clone()
    planned = torch.tensor([1.25e-4, -3.75e-4], dtype=torch.float32)
    first = task._phase_set(phase, base, planned)
    second = task._phase_set(phase, base, planned)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(phase.detach(), base + planned, rtol=0, atol=0)
    assert not torch.equal(phase.detach(), base + planned + planned)


class _ToyRuntime(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.generator = torch.nn.Module()
        self.generator.phase = torch.nn.Parameter(torch.zeros(1))
        self.raw_plane = PlaneSpec("toy_cmos", (200, 200), (1e-6, 1e-6), 532e-9, 0.02, "camera")
        self.field_plane = PlaneSpec("toy_field", (2, 2), (1e-6, 1e-6), 532e-9, 0.01, "generator")
        self.provenance = OperationProvenance("toy", None, None)

    def forward_from_ols(self, z: torch.Tensor):
        labels = z[:, 0].real.to(torch.long)
        focus = ((labels == 1) | (labels == 7)).to(torch.float32)
        coordinate = labels.to(torch.float32) - 1.0 + self.generator.phase[0] * 1000.0 * focus
        intensity = 0.2 + coordinate * 0.025
        top = intensity.view(-1, 1, 1, 1).expand(-1, 1, 100, 200)
        bottom = torch.zeros_like(top)
        raw = torch.cat((top, bottom), dim=2)
        field = ComplexField(torch.ones((z.shape[0], 1, 2, 2), dtype=torch.complex64, device=z.device), self.field_plane, self.provenance)
        return SimpleNamespace(
            raw_intensity=raw,
            raw_camera=IntensityFrame(raw, self.raw_plane, self.provenance),
            generator_input_field=field,
            generator_records=(field,),
            generator_output_field=field,
        )


class _ToyEvaluator(torch.nn.Module):
    def __init__(self, *, e_d: bool) -> None:
        super().__init__()
        self.e_d = e_d

    def forward(self, view: torch.Tensor):
        coordinate = (view.mean(dim=(1, 2, 3)) * 2.0 - 0.2) / 0.025
        classes = torch.arange(10, dtype=view.dtype, device=view.device).view(1, 10)
        logits = -(classes - coordinate[:, None]).square()
        if self.e_d:
            return logits
        features = torch.zeros((view.shape[0], 128), dtype=view.dtype, device=view.device)
        return features, logits


def _toy_inputs() -> dict[str, object]:
    labels = torch.cat((torch.arange(10).repeat(20), torch.arange(10).repeat(10), torch.arange(10).repeat(30))).long()
    rows = torch.arange(600)
    z = torch.zeros((600, 100), dtype=torch.complex64)
    z[:, 0] = labels.to(torch.float32).to(torch.complex64)
    raw = torch.zeros((600, 1, 200, 200))
    images = torch.zeros((600, 1, 28, 28))
    for index, label in enumerate(labels.tolist()):
        value = 0.2 + label * 0.025
        raw[index, :, :100, :] = value
        images[index, :, :14, :] = value
    return {
        "source_indices_600": rows,
        "labels_600": labels,
        "roles_600": [task._role_for_position(index) for index in range(600)],
        "images_600": images,
        "readouts": {"B_class_ce": {"z": z, "target_slm_intensity": raw}},
    }


def _decision_rows(block: dict[str, object], start: int, condition: str) -> list[dict[str, object]]:
    rows = []
    for offset in range(task.CHUNK_SIZE):
        position = start + offset
        margin = block["margins"][offset]
        metric = block["image_metrics"][offset]
        rows.append({
            "position": position,
            "source_row_index": int(block["source_row_indices"][offset]),
            "label": int(block["labels"][offset]),
            "role": block["roles"][offset],
            "ed_margin": float(margin["margin"]),
            "ed_prediction": int(block["evaluator_logits"]["E_D"][offset].argmax()),
            "raw_mse": metric["raw_mse"],
            "raw_foreground_mse": metric["raw_foreground_mse"],
            "raw_background_mse": metric["raw_background_mse"],
            "view_mse": metric["view_mse"],
            "view_foreground_mse": metric["view_foreground_mse"],
            "view_background_mse": metric["view_background_mse"],
            "raw_fraction_above_one": metric["raw_fraction_above_one"],
            "raw_zero_image": metric["raw_zero_image"],
            "raw_constant_image": metric["raw_constant_image"],
            "condition": condition,
            **({"delta_from_baseline": margin["delta_from_baseline"], "response_status": margin["response_status"]} if "delta_from_baseline" in margin else {}),
        })
    return rows


def _integrated_toy_states(*, reverse: bool = False):
    blocks = _direction_blocks()
    directions = task._build_shared_directions(blocks, _optimizer_state(first=-0.13, second=0.2))
    conditions = task._condition_vectors(directions)
    runtime = _ToyRuntime()
    consumer = SimpleNamespace(runtime=runtime, device=torch.device("cpu"))
    evaluators = {name: _ToyEvaluator(e_d=name == "E_D") for name in task.EVALUATORS}
    inputs = _toy_inputs()
    baseline_phase = runtime.generator.phase.detach().clone()
    baseline_by_chunk: dict[int, dict[str, object]] = {}
    baseline_rows = []
    for chunk in range(3, 6):
        start = chunk * task.CHUNK_SIZE
        block = task._measure_condition_chunk(consumer, evaluators, inputs, chunk, "baseline", None, None, task.ETA_FLOOR, {}, None)
        baseline_by_chunk[chunk] = block
        baseline_rows.extend(_decision_rows(block, start, "baseline"))

    state_rows: dict[str, list[dict[str, object]]] = {}
    for name in (f"{direction}_equal_rms_plus1" for direction in task.EQUAL_RMS_NAMES):
        spec = conditions[name]
        applied = task._phase_set(runtime.generator.phase, baseline_phase, -spec["planned_delta"] if reverse else spec["planned_delta"])
        actual = []
        for chunk in range(3, 6):
            start = chunk * task.CHUNK_SIZE
            block = task._measure_condition_chunk(consumer, evaluators, inputs, chunk, name, spec["planned_delta"], applied, task.ETA_FLOOR, {}, baseline_by_chunk[chunk])
            actual.extend(_decision_rows(block, start, name))
        state_rows[name] = actual
        task.finite._restore_phase(runtime.generator.phase, baseline_phase)
    native = conditions[f"{task.NATIVE_NAME}_plus1"]
    applied = task._phase_set(runtime.generator.phase, baseline_phase, -native["planned_delta"] if reverse else native["planned_delta"])
    native_rows = []
    for chunk in range(3, 6):
        start = chunk * task.CHUNK_SIZE
        block = task._measure_condition_chunk(consumer, evaluators, inputs, chunk, f"{task.NATIVE_NAME}_plus1", native["planned_delta"], applied, task.ETA_FLOOR, {}, baseline_by_chunk[chunk])
        native_rows.extend(_decision_rows(block, start, f"{task.NATIVE_NAME}_plus1"))
    state_rows[f"{task.NATIVE_NAME}_plus1"] = native_rows
    task.finite._restore_phase(runtime.generator.phase, baseline_phase)
    return baseline_rows, state_rows, inputs


def test_production_readout_finite_response_flows_into_the_final_confirmation_decision() -> None:
    baseline, states, _ = _integrated_toy_states()
    decision = task._decision(baseline, states, task.ETA_FLOOR)
    assert decision["classification"] == "SHARED_LOCAL_RESPONSE_SUPPORTED_WITHIN_SCOPE"
    assert set(decision["supported_conditions"]) == set(states)
    assert decision["condition_metrics"]["S_GD_equal_rms_plus1"]["per_class"]["1"]["beneficial_rows"] == 30
    assert decision["condition_metrics"]["S_GD_equal_rms_plus1"]["per_class"]["7"]["beneficial_rows"] == 30
    assert decision["training_recommendation"] == "NONE"
    assert decision["next_action"] == "NONE"


def test_wrong_sign_finite_response_is_not_promoted_by_the_confirmation_gate() -> None:
    baseline, states, _ = _integrated_toy_states(reverse=True)
    decision = task._decision(baseline, states, task.ETA_FLOOR)
    assert decision["classification"] == "MIXED_OR_INSUFFICIENT"
    assert decision["supported_conditions"] == []
    assert decision["condition_metrics"]["S_GD_equal_rms_plus1"]["per_class"]["1"]["harmful_rows"] == 30


def test_shared_chunk_rejects_wrong_parent_row_or_role_mapping() -> None:
    block = {
        "row_start": 0,
        "row_stop": task.CHUNK_SIZE,
        "kind": "baseline",
        "source_row_indices": torch.arange(task.CHUNK_SIZE),
        "labels": torch.arange(task.CHUNK_SIZE),
        "roles": ["fit"] * task.CHUNK_SIZE,
    }
    payload = {"schema": task.BLOCK_SCHEMA, "task_id": task.TASK_ID, "condition": "baseline", "chunk_index": 0, "block": block}
    inputs = {"source_indices_600": torch.arange(600), "labels_600": torch.arange(600), "roles_600": ["fit"] * 200 + ["historic_holdout"] * 100 + ["confirmation"] * 300}
    task._validate_chunk(payload, "baseline", 0, inputs)
    payload["block"] = {**block, "source_row_indices": torch.arange(task.CHUNK_SIZE).roll(1)}
    with pytest.raises(task.SharedDirectionError, match="row mapping changed"):
        task._validate_chunk(payload, "baseline", 0, inputs)


def test_completed_resume_returns_saved_terminal_report_before_cuda_or_forward(tmp_path, monkeypatch) -> None:
    root = tmp_path / "repo"
    target = root / "outputs" / "completed"
    target.mkdir(parents=True)
    labels = torch.arange(10).repeat(60).tolist()
    roles = ["fit"] * 200 + ["historic_holdout"] * 100 + ["confirmation"] * 300
    vectors = {name: torch.ones(2, dtype=torch.float32) for name in task.EQUAL_RMS_NAMES}
    direction_bundle = {
        "equal_rms_directions": vectors,
        "batch_mean_descent_directions": {"S_ADAM": torch.ones(2)},
        "source_rms": {"S_ADAM": 1.0},
    }
    inputs = {
        "artifact_dir": root / "parent",
        "direction_dir": root / "direction",
        "finite_dir": root / "finite",
        "source_indices_600": torch.arange(600),
        "labels_600": torch.tensor(labels),
        "roles_600": roles,
        "directions": direction_bundle,
        "payloads": {"B_class_ce": {
            "trainable_parameter_state": {"encoder.phase": torch.zeros(2), "generator.phase": torch.zeros(2)},
            "optimizer_step_by_parameter": {"encoder.phase": 2200, "generator.phase": 2500},
        }},
        "paths": {
            "checkpoint_B_class_ce": root / "parent" / "arm_b_step0300.pt",
            "readout_B_class_ce": root / "parent" / "readouts_B_class_ce.pt",
            "training_data": root / "training.pt",
        },
    }
    population = task._population_record(inputs, target)
    identity = task._identity(inputs, population, target)
    measured = ["baseline", "baseline_repeat", *task._condition_names()]
    expected = [task._chunk_name(condition, index) for condition in measured for index in range(task.CHUNK_COUNT)]
    for name in expected:
        (target / name).touch()
    manifest = {
        "schema": task.MANIFEST_SCHEMA,
        "task_id": task.TASK_ID,
        "margin_derivative_definition": task.MARGIN_DEFINITION,
        "identity": identity.to_dict(),
        "population": population,
        "completed_chunks": expected,
        "measured_conditions": measured,
        "status": "COMPLETED",
    }
    (target / "resume_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    report = {"schema": task.REPORT_SCHEMA, "task_id": task.TASK_ID, "status": "VALID_COMPLETED", "classification": "MIXED_OR_INSUFFICIENT", "research_run_identity": identity.to_dict()}
    (target / "report.json").write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(task, "_root", lambda: root)
    monkeypatch.setattr(task, "_load_inputs", lambda *args: inputs)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(AssertionError("terminal restore attempted CUDA")))
    result = task._run(argparse.Namespace(
        artifact_dir=Path("parent"),
        direction_dir=Path("direction"),
        finite_response_dir=Path("finite"),
        output_dir=Path("outputs/completed"),
        preflight_only=False,
        resume_manifest=Path("outputs/completed/resume_manifest.json"),
    ))
    assert result == report
