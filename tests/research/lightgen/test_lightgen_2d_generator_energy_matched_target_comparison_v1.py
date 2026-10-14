from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from scripts.research.lightgen import execute_lightgen_2d_generator_energy_matched_target_comparison_v1 as task


def _synthetic_scale_inputs() -> dict[str, object]:
    labels = torch.arange(500, dtype=torch.long) % 10
    rows = torch.arange(500, dtype=torch.long) + 2_000
    target = torch.full((500, 1, 200, 200), 0.25, dtype=torch.float32)
    ratios = torch.linspace(0.4, 0.8, 500, dtype=torch.float64)
    raw = target * ratios.to(torch.float32).view(500, 1, 1, 1)
    readout = {
        "source_indices": rows.clone(),
        "labels": labels.clone(),
        "target_slm_intensity": target,
        "raw_cmos": raw,
        "plane_metadata": {
            "input_slm_field": {"plane": {"pixel_pitch_m": [2.5e-6, 2.5e-6]}},
            "camera_plane": {"pixel_pitch_m": [2.5e-6, 2.5e-6]},
        },
    }
    return {"readout": readout, "rows": rows, "labels": labels}


def test_energy_coefficients_use_only_fit_rows_and_physical_integrals() -> None:
    inputs = _synthetic_scale_inputs()
    scales, evidence = task._derive_energy_scales(inputs)
    expected = torch.linspace(0.4, 0.8, 500, dtype=torch.float64)[:200]
    assert torch.allclose(scales, expected, rtol=2e-7, atol=2e-8)
    assert evidence["fit_only"] is True
    assert evidence["source_rows"] == inputs["rows"][:200].tolist()
    assert evidence["labels"] == inputs["labels"][:200].tolist()
    assert evidence["target_pixel_pitch_m"] == evidence["camera_pixel_pitch_m"] == [2.5e-6, 2.5e-6]
    assert len(evidence["scale_coefficients"]) == 200

    changed_confirmation = _synthetic_scale_inputs()
    changed_confirmation["readout"]["target_slm_intensity"][200:] *= 3.0
    changed_confirmation["readout"]["raw_cmos"][200:] *= 0.25
    confirm_scales, _ = task._derive_energy_scales(changed_confirmation)
    assert torch.equal(confirm_scales, scales)


@pytest.mark.parametrize("mismatch", ["rows", "labels"])
def test_energy_coefficients_reject_wrong_fit_population_binding(mismatch: str) -> None:
    inputs = _synthetic_scale_inputs()
    if mismatch == "rows":
        inputs["readout"]["source_indices"][0] += 1
    else:
        inputs["readout"]["labels"][0] = (inputs["readout"]["labels"][0] + 1) % 10
    with pytest.raises(task.EnergyTargetError, match="readout"):
        task._derive_energy_scales(inputs)


def test_batch_target_mapping_keeps_A_original_and_B_fixed_scaled_rows() -> None:
    targets = torch.arange(200 * 200, dtype=torch.float32).view(200, 1, 200, 1).expand(-1, -1, -1, 200).contiguous()
    scales = torch.linspace(0.4, 0.7, 200, dtype=torch.float64)
    inputs = {"fit_targets_original": targets, "fit_targets_scaled": (targets.to(torch.float64) * scales.view(200, 1, 1, 1)).to(torch.float32), "scales": scales}
    a = task._targets_for_batch(inputs, task.ARM_A, 7, torch.device("cpu"))
    b = task._targets_for_batch(inputs, task.ARM_B, 7, torch.device("cpu"))
    start = 7 * task.BATCH_SIZE
    assert torch.equal(a, targets[start : start + 10])
    assert torch.equal(b, inputs["fit_targets_scaled"][start : start + 10])
    assert not torch.equal(a, b)
    assert not a.requires_grad and not b.requires_grad


def _readout_for_q() -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    labels = torch.arange(10, dtype=torch.long)
    target_view = torch.zeros((10, 1, 28, 28), dtype=torch.float32)
    for class_id in range(10):
        target_view[class_id, 0, class_id : class_id + 3, class_id : class_id + 3] = 0.7
    target_raw = F.interpolate(target_view, size=(200, 200), mode="bilinear", align_corners=False)
    generated_view = 0.5 * task.diagnostic.canonical.cmos_view(target_raw)
    raw = F.interpolate(generated_view, size=(200, 200), mode="bilinear", align_corners=False)
    readout = {
        "raw_cmos": raw,
        "target_slm_intensity": target_raw,
        "evaluator_view": generated_view,
        "labels": labels,
        "plane_metadata": {"camera_plane": {"pixel_pitch_m": [2.5e-6, 2.5e-6]}},
    }
    inputs = {"population": {"role_positions": {"confirmation": [0, 10]}}}
    return readout, inputs


def test_brightness_normalized_Q_is_scale_invariant_but_energy_is_not() -> None:
    readout, inputs = _readout_for_q()
    energy, rows = task._energy_image_metrics(readout, inputs)
    confirmation = energy["metrics_by_role"]["confirmation"]
    assert confirmation["Q_undefined_rows"] == 0
    assert confirmation["mean_Q"] == pytest.approx(0.0, abs=1e-12)
    assert all(float(value) == pytest.approx(0.0, abs=1e-12) for value in rows["brightness_normalized_stroke_error_Q"])
    assert confirmation["mean_cmos_integrated_intensity"] > 0.0


def test_zero_cmos_output_leaves_Q_undefined_instead_of_adding_a_denominator() -> None:
    readout, inputs = _readout_for_q()
    readout["raw_cmos"].zero_()
    readout["evaluator_view"].zero_()
    energy, rows = task._energy_image_metrics(readout, inputs)
    confirmation = energy["metrics_by_role"]["confirmation"]
    assert confirmation["mean_Q"] is None
    assert confirmation["Q_undefined_rows"] == 10
    assert torch.isnan(rows["brightness_normalized_stroke_error_Q"]).all()
    assert int(rows["raw_zero_image"].sum()) == 10


@pytest.mark.parametrize(("dirname", "expected"), [("attempt_01", "01"), ("attempt_02", "02"), ("attempt_12", "12")])
def test_run_identity_is_bound_to_explicit_attempt_directory(dirname: str, expected: str) -> None:
    inputs = {
        "rows": torch.arange(500),
        "population": {"source_checkpoint": "source.pt", "source_parameter_shapes": {"generator.phase": [340_000]}},
    }
    identity = task._make_identity(inputs, Path("outputs/task/runs") / dirname)
    assert identity.run_id == f"{task.TASK_ID}-attempt-{expected}"
    assert identity.scientific_attempt_id == f"{task.TASK_ID}:attempt-{expected}"
    assert task._attempt_number(Path("outputs/task/runs") / dirname) == int(expected)


@pytest.mark.parametrize("dirname", ["attempt_latest", "run_attempt_02", "attempt_"])
def test_attempt_identity_rejects_non_numeric_attempt_directory(dirname: str) -> None:
    with pytest.raises(task.EnergyTargetError, match="attempt_<number>"):
        task._attempt_number(Path("outputs/task/runs") / dirname)


def test_gradient_bundle_comparison_uses_declared_float_replay_tolerance() -> None:
    reference = {"phase": torch.tensor([1.0, -0.25], dtype=torch.float32)}
    within_tolerance = {"phase": torch.tensor([1.0 + 1.0e-6, -0.25], dtype=torch.float32)}
    outside_tolerance = {"phase": torch.tensor([1.0 + 1.0e-3, -0.25], dtype=torch.float32)}
    assert task._gradient_bundle_comparison(reference, within_tolerance)["passed"] is True
    failed = task._gradient_bundle_comparison(reference, outside_tolerance)
    assert failed["passed"] is False
    assert failed["comparisons"]["phase"]["max_absolute_difference"] > 1.0e-4


def _summary(*, overall_q: float, focused_q: float, other_q: float, energy: float, correct: int = 20) -> dict[str, object]:
    class_stats = {
        str(c): {
            "mean_Q": focused_q if c in (2, 4) else other_q,
            "mean_cmos_integrated_intensity": energy,
            "mean_raw_clip_fraction": 0.0,
            "zero_images": 0,
            "constant_images": 0,
        }
        for c in range(10)
    }
    counts = [correct] * 10
    return {
        "all_finite": True,
        "energy_metrics": {
            "all_rows_finite": True,
            "all_rows_zero_images": 0,
            "all_rows_constant_images": 0,
            "metrics_by_role": {
                "confirmation": {
                    "mean_Q": overall_q,
                    "Q_undefined_rows": 0,
                    "mean_cmos_integrated_intensity": energy,
                    "mean_raw_clip_fraction": 0.0,
                    "zero_images": 0,
                    "constant_images": 0,
                    "by_class": class_stats,
                }
            },
        },
        "evaluators": {
            "E_D": {
                "by_role": {
                    "confirmation": {
                        "per_class_correct": counts,
                        "correct_total": sum(counts),
                        "macro_recall": correct / 30,
                    }
                }
            }
        },
    }


def test_final_gate_uses_production_Q_and_rejects_darkening() -> None:
    start = _summary(overall_q=1.0, focused_q=1.0, other_q=1.0, energy=1.0)
    arm_a = _summary(overall_q=0.98, focused_q=0.98, other_q=1.0, energy=1.0)
    arm_b = _summary(overall_q=0.88, focused_q=0.88, other_q=1.05, energy=0.94)
    decision = task._final_decision(start, arm_a, arm_b, valid=True)
    assert decision["classification"] == "MIXED_OR_INSUFFICIENT"
    assert decision["conditions"]["overall_and_class_2_4_Q_at_least_10pct_lower_vs_A_and_start"] is True
    assert decision["conditions"]["mean_cmos_integrated_energy_at_least_95pct_vs_A_and_start"] is False


def test_final_gate_supports_only_when_all_declared_thresholds_pass() -> None:
    start = _summary(overall_q=1.0, focused_q=1.0, other_q=1.0, energy=1.0)
    arm_a = _summary(overall_q=0.98, focused_q=0.98, other_q=1.0, energy=1.0)
    arm_b = _summary(overall_q=0.88, focused_q=0.88, other_q=1.05, energy=0.95)
    decision = task._final_decision(start, arm_a, arm_b, valid=True)
    assert decision["classification"] == "ENERGY_MATCHED_TARGET_STROKE_METRIC_GAIN_SUPPORTED"
    assert decision["all_conditions_passed"] is True


def test_zero_or_constant_output_cannot_become_supported() -> None:
    start = _summary(overall_q=1.0, focused_q=1.0, other_q=1.0, energy=1.0)
    arm_a = _summary(overall_q=0.98, focused_q=0.98, other_q=1.0, energy=1.0)
    arm_b = _summary(overall_q=0.88, focused_q=0.88, other_q=1.0, energy=1.0)
    arm_b["energy_metrics"]["all_rows_zero_images"] = 1
    decision = task._final_decision(start, arm_a, arm_b, valid=True)
    assert decision["classification"] == "MIXED_OR_INSUFFICIENT"
    assert decision["conditions"]["finite_nonzero_nonconstant_outputs"] is False


def test_terminal_report_does_not_authorize_more_training() -> None:
    assert task.ADDED_UPDATES == 2_000
    assert task.FINAL_STEP == 6_700
    assert task._checkpoint_path(Path("D:/out"), task.ARM_B, 2_000).name == "arm_b_step6700.pt"
    assert task._checkpoint_path(Path("D:/out"), task.ARM_A, 100).name == "arm_a_step4800.pt"


def _checkpoint_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    target = tmp_path / "attempt_01"
    target.mkdir()
    population = {"task_id": task.TASK_ID, "source_checkpoint": "D:/source/arm_a_step4700.pt", "energy_scale_coefficients": [0.5] * 200}
    scales = torch.full((200,), 0.5, dtype=torch.float64)
    source_rng = {"torch_cpu": torch.arange(4, dtype=torch.uint8)}
    frozen = {"ols.operator.transport": torch.eye(2)}
    moments = torch.zeros(340_000, dtype=torch.float32)
    payload = {
        "schema": task.CHECKPOINT_SCHEMA,
        "task_id": task.TASK_ID,
        "arm": task.ARM_B,
        "source_step": task.SOURCE_STEP,
        "completed_updates": task.SOURCE_STEP,
        "task_local_updates": 0,
        "optimizer_step": task.SOURCE_STEP,
        "next_batch_index": 0,
        "next_batch_cursor": 0,
        "architecture_layout": "two_original_100x100_then_two_400x400_phase_planes_flattened",
        "numeric_mode": task.NUMERIC_MODE,
        "objective_definition": "raw_MSE_plus_0.1_SSIM_for_both_arms; B_fixed_fit_energy_matched_y",
        "optimizer_state": {
            "param_groups": [{"params": [0], "lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0}],
            "state": {0: {"step": torch.tensor(float(task.SOURCE_STEP)), "exp_avg": moments.clone(), "exp_avg_sq": moments.clone()}},
        },
        "trainable_parameter_state": {"encoder.phase": torch.zeros((2, 1, 100, 100)), "generator.phase": torch.zeros(340_000)},
        "frozen_runtime_state": frozen,
        "references": {"scale": torch.ones(100)},
        "weights": {"beta": 1.0},
        "source_checkpoint": population["source_checkpoint"],
        "population": population,
        "target_scale_coefficients": scales,
        "rng_state": source_rng,
        "source_rng_state": source_rng,
        "task_training_history": [],
    }
    path = task._checkpoint_path(target, task.ARM_B, 0)
    torch.save(payload, path)
    inputs = {
        "population": population,
        "scales": scales,
        "source": {"rng_state": source_rng, "references": payload["references"], "weights": payload["weights"], "frozen_runtime_state": frozen},
    }
    return target, path, inputs


def test_explicit_resume_checkpoint_validates_source_step_rows_scales_and_cursor(tmp_path: Path) -> None:
    target, path, inputs = _checkpoint_fixture(tmp_path)
    loaded = task._read_checkpoint(path, target, inputs)
    assert loaded["task_local_updates"] == 0
    assert loaded["optimizer_step"] == 4_700
    assert loaded["next_batch_cursor"] == 0

    changed_inputs = dict(inputs)
    changed_inputs["scales"] = torch.full((200,), 0.6, dtype=torch.float64)
    with pytest.raises(task.EnergyTargetError, match="target coefficients"):
        task._read_checkpoint(path, target, changed_inputs)

    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["next_batch_cursor"] = 1
    malformed = task._checkpoint_path(target, task.ARM_B, 0)
    torch.save(payload, malformed)
    with pytest.raises(task.EnergyTargetError, match="next batch"):
        task._read_checkpoint(malformed, target, inputs)
