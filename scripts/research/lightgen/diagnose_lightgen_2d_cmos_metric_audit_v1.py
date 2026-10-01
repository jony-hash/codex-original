"""Read-only CMOS metric audit for four frozen LightGen checkpoints.

The executor first validates the frozen data/checkpoint identities, replays the
existing 50-image subset, and checks the batch-10 CUDA envelope. Only after a
READY receipt does it create the diagnostic output attempt and replay 5,000
class-balanced DEV samples per checkpoint. No optimizer, backward pass, or
checkpoint write is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ldm_project.v2.application.research.lightgen import lightgen_2d_chen_bogt_complex_ols_training as optical
from ldm_project.v2.application.research.shared.research_run_control import CORRECTNESS_INVARIANTS, IMPLEMENTATION_SELECTION, RESOURCE_ENVELOPE, PREFLIGHT_READY, PreflightReceipt, ResearchRunControl, ResearchRunIdentity, capture_execution_source_manifest
from ldm_project.v2.evidence import file_sha256, safe_torch_load, tensor_sha256
from scripts.research.lightgen import execute_lightgen_2d_evaluator_robust_semantic_training_v1 as robust
from scripts.research.lightgen import execute_lightgen_2d_three_evaluator_robust_semantic_continuation_v1 as sic
from scripts.research.lightgen import visualize_lightgen_2d_cmos_eight_node_step_comparison_v1 as visual


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_RELATIVE = Path("outputs/20260924_cmos_metric_audit_v1/runs/attempt_02")
PRIOR_OUTPUT = Path("outputs/20260923_cmos_eight_node_step_comparison_v1/runs/attempt_01")
SELECTION_MANIFEST = Path("outputs/20260921_step6000_conditional_generation_readiness_v1/runs/attempt_06/fixed_selection_manifest.json")
STEPS = (1000, 3100, 6000, 9000)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 271828
FOREGROUND_THRESHOLD = 0.2
FORWARD_BATCH_SIZE = 10
PREFLIGHT_BATCH_SIZE = 2
AUTHORIZATION_ID = "USER_AUTHORIZED_GROUP_MEETING_CMOS_METRIC_AUDIT_20260924"
TASK_ID = "DIAGNOSE_LIGHTGEN_2D_CMOS_METRIC_AUDIT_V1"
SEMANTIC_AUTHORITY_ID = "DESCRIPTIVE_FIXED_DEV_CMOS_METRIC_AUDIT"
RUN_ID = "lightgen-cmos-metric-audit-20260924"
TRAJECTORY_ID = "lightgen-cmos-metric-audit-four-checkpoints"

NODE_KEYS = (
    "slm_output",
    "encoder_input",
    "encoder_layer_1",
    "pre_ols",
    "post_ols",
    "decoder_layer_1",
    "decoder_layer_2",
    "cmos_field",
)
NODE_LABELS = (
    "SLM output",
    "Encoder input relay",
    "Encoder layer 1",
    "Encoder layer 2 / pre-OLS",
    "OLS output / generator input",
    "Decoder layer 1",
    "Decoder layer 2",
    "CMOS incident field",
)
MODEL_NAMES = ("E_S", "E_I", "E_C", "E_D")
METRIC_NAMES = (
    "L_rec_mse",
    "pearson_r",
    "foreground_dice",
    "foreground_miss_rate",
    "background_false_positive_rate",
    "intensity_sum_ratio",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), f"no rows to write: {path.name}")
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), f"no rows to append: {path.name}")
    exists = path.exists()
    with path.open("a" if exists else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _selected_checkpoints(root: Path) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    manifest_path = root / PRIOR_OUTPUT / "checkpoint_manifest.json"
    manifest = _read_json(manifest_path)
    _require(manifest.get("frozen_max_step") == 9300, "prior frozen checkpoint upper bound changed")
    _require(manifest.get("candidate_count") == 27, "prior candidate manifest is incomplete")
    _require(all(step in manifest.get("candidate_steps", []) for step in STEPS), "one or more requested steps are absent from the frozen manifest")
    records = {int(item["step"]): item for item in manifest.get("candidates", [])}
    _require(set(STEPS).issubset(records), "manifest records are missing requested steps")
    validated: list[dict[str, Any]] = []
    payloads: dict[int, dict[str, Any]] = {}
    for step in STEPS:
        record = records[step]
        checkpoint = root / record["checkpoint_path"]
        report = root / record["report_path"]
        _require(checkpoint.is_file() and report.is_file(), f"checkpoint/report missing at step{step}")
        _require(file_sha256(checkpoint) == record["checkpoint_sha256"], f"checkpoint SHA mismatch at step{step}")
        _require(file_sha256(report) == record["report_sha256"], f"producer report SHA mismatch at step{step}")
        report_data = _read_json(report)
        _require(report_data.get("result") == "VALID_COMPLETED", f"producer report is not valid completed at step{step}")
        _require(report_data.get("classification") == record["run_classification"], f"producer classification changed at step{step}")
        _require(report_data.get("outer_gate_passed") is True, f"checkpoint producer outer gate is not passed at step{step}")
        payload = safe_torch_load(checkpoint, expected_sha256=record["checkpoint_sha256"], map_location="cpu")
        _require(isinstance(payload, Mapping), f"checkpoint payload malformed at step{step}")
        _require(payload.get("schema") == record["checkpoint_schema"], f"checkpoint schema changed at step{step}")
        _require(payload.get("completed_updates") == step, f"checkpoint cursor does not match filename at step{step}")
        _require(payload.get("scientific_artifact") is True and payload.get("operational_artifact") is False, f"step{step} is not a scientific checkpoint")
        _require(payload.get("trainable_parameter_names") == ["generator.phase"], f"trainable boundary changed at step{step}")
        for state_name in ("encoder_state", "generator_state"):
            state = payload.get(state_name)
            _require(torch.is_tensor(state) and state.shape == (2, 1, 100, 100) and state.dtype == torch.float32, f"invalid {state_name} at step{step}")
            _require(bool(torch.isfinite(state).all()), f"nonfinite {state_name} at step{step}")
        validated.append({key: record[key] for key in ("step", "run_name", "run_dir", "checkpoint_path", "checkpoint_sha256", "checkpoint_schema", "report_path", "report_sha256", "run_classification", "outer_gate_passed")})
        payloads[step] = dict(payload)
    return validated, payloads


def _load_fixed_inputs(root: Path, authorities: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], dict[str, Any], torch.Tensor]:
    selection_path = root / SELECTION_MANIFEST
    selection_manifest = _read_json(selection_path)
    selection = selection_manifest.get("real_dev")
    _require(isinstance(selection, Mapping), "accepted fixed 50-image selection is missing")
    source_indices = [int(value) for value in selection["canonical_source_indices"]]
    positions = torch.tensor(selection["positions_in_sorted_dev"], dtype=torch.long)
    labels_50 = [int(value) for value in selection["labels"]]
    _require(len(source_indices) == 50 and len(positions) == 50 and len(labels_50) == 50, "fixed readability subset must contain 50 rows")
    source_tensor = torch.tensor(source_indices, dtype=torch.long)
    expected_indices = authorities["development_indices"].index_select(0, positions)
    _require(torch.equal(source_tensor, expected_indices), "fixed 50-image selection no longer matches accepted sorted DEV positions")
    selected_labels = authorities["labels"].index_select(0, source_tensor)
    _require(selected_labels.tolist() == labels_50, "fixed 50-image labels changed")
    _require(torch.bincount(selected_labels, minlength=10).tolist() == [5] * 10, "fixed 50-image sample is no longer class-balanced")
    selected_images = authorities["images"].index_select(0, source_tensor).contiguous()
    _require(selected_images.shape == (50, 1, 28, 28), "fixed 50-image geometry changed")

    dev_indices = authorities["development_indices"].clone().long().contiguous()
    dev_labels = authorities["labels"].index_select(0, dev_indices).contiguous()
    dev_images = authorities["images"].index_select(0, dev_indices).contiguous()
    _require(dev_indices.numel() == 5000 and dev_images.shape == (5000, 1, 28, 28), "canonical DEV selection changed geometry or row count")
    _require(torch.bincount(dev_labels, minlength=10).tolist() == [500] * 10, "canonical DEV is not 500 examples per class")
    _require(torch.equal(dev_indices, torch.sort(dev_indices).values), "canonical DEV indices are no longer in sorted source order")
    _require(bool(torch.isfinite(dev_images).all()), "DEV images contain nonfinite values")

    supplement = _read_json(root / PRIOR_OUTPUT / "step6000_trace_supplement.json")
    node_indices = [int(value) for value in supplement["display_canonical_source_indices"]]
    node_labels = torch.tensor(supplement["display_labels"], dtype=torch.long)
    _require(len(node_indices) == 20 and node_labels.tolist() == [digit for digit in range(10) for _ in range(2)], "fixed eight-node examples changed")
    node_tensor = torch.tensor(node_indices, dtype=torch.long)
    expected_node_images = authorities["images"].index_select(0, node_tensor).contiguous()
    _require(torch.equal(expected_node_images, selected_images[torch.tensor([source_indices.index(index) for index in node_indices])]), "eight-node sample identities differ from the fixed 50-image selection")
    _require(torch.equal(authorities["labels"].index_select(0, node_tensor), node_labels), "eight-node labels changed")
    return dev_images, dev_labels, dev_indices, source_indices, selection_manifest, node_tensor


def _metric_arrays(raw: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    _require(raw.shape == target.shape and raw.ndim == 4 and tuple(raw.shape[1:]) == (1, 200, 200), "CMOS/target geometry must be paired raw 200x200 intensity")
    _require(bool(torch.isfinite(raw).all()) and bool(torch.isfinite(target).all()), "CMOS or target intensity is nonfinite")
    _require(bool((raw >= 0).all()) and bool((target >= 0).all()), "intensity contains a negative value")
    r = raw.to(torch.float64).flatten(1)
    t = target.to(torch.float64).flatten(1)
    mse = (r - t).square().mean(dim=1)
    rc = r - r.mean(dim=1, keepdim=True)
    tc = t - t.mean(dim=1, keepdim=True)
    denom = torch.sqrt(rc.square().sum(dim=1) * tc.square().sum(dim=1))
    _require(bool((denom > 0).all()), "Pearson correlation is undefined for a constant image")
    pearson = (rc * tc).sum(dim=1) / denom
    fg = t >= FOREGROUND_THRESHOLD
    pred_fg = r >= FOREGROUND_THRESHOLD
    fg_count = fg.sum(dim=1)
    bg_count = (~fg).sum(dim=1)
    _require(bool((fg_count > 0).all()) and bool((bg_count > 0).all()), "fixed threshold yields empty target foreground/background")
    inter = (fg & pred_fg).sum(dim=1)
    dice = (2.0 * inter) / (fg_count + pred_fg.sum(dim=1)).clamp_min(1)
    miss = (fg & ~pred_fg).sum(dim=1) / fg_count
    false_positive = ((~fg) & pred_fg).sum(dim=1) / bg_count
    target_sum = t.sum(dim=1)
    _require(bool((target_sum > 0).all()), "target intensity sum is zero")
    intensity_ratio = r.sum(dim=1) / target_sum
    return {
        "L_rec_mse": mse,
        "pearson_r": pearson,
        "foreground_dice": dice,
        "foreground_miss_rate": miss,
        "background_false_positive_rate": false_positive,
        "intensity_sum_ratio": intensity_ratio,
    }


def _logits(model: Any, value: torch.Tensor) -> torch.Tensor:
    output = model(value)
    logits = output[-1] if isinstance(output, (tuple, list)) else output
    _require(torch.is_tensor(logits) and logits.ndim == 2 and logits.shape[1] == 10, "frozen evaluator output is not ten-class logits")
    return logits


def _replay_fixed_50(root: Path, consumer: Any, payloads: Mapping[int, Mapping[str, Any]], images50: torch.Tensor) -> dict[str, Any]:
    artifact = safe_torch_load(root / PRIOR_OUTPUT / "candidate_cmos_raw.pt", map_location="cpu")
    _require(isinstance(artifact, Mapping) and artifact.get("schema") == "cmos_step_comparison_candidate_raw.v1", "prior raw CMOS replay artifact changed")
    _require(tuple(artifact.get("steps", [])) == tuple(visual.CANDIDATE_STEPS), "prior raw CMOS candidate order changed")
    raw_stack = artifact.get("raw_intensity")
    _require(torch.is_tensor(raw_stack) and raw_stack.shape == (27, 50, 1, 200, 200), "prior raw CMOS replay shape changed")
    old_position = {int(step): index for index, step in enumerate(artifact["steps"])}
    report: dict[str, Any] = {}
    for step in STEPS:
        payload = payloads[step]
        consumer.load_trainable_state_dict({"encoder.phase": payload["encoder_state"], "generator.phase": payload["generator_state"]})
        before = visual._state_digest(consumer.runtime.state_dict())
        outputs: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(images50), PREFLIGHT_BATCH_SIZE):
                result = consumer.runtime.forward_from_image(images50[start : start + PREFLIGHT_BATCH_SIZE].to(consumer.device))
                raw = result.raw_intensity.detach().cpu().contiguous()
                _require(raw.shape[1:] == (1, 200, 200), f"preflight raw geometry changed at step{step}")
                _require(bool(torch.isfinite(raw).all()), f"preflight raw contains nonfinite values at step{step}")
                outputs.append(raw)
        after = visual._state_digest(consumer.runtime.state_dict())
        replay = torch.cat(outputs, dim=0)
        reference = raw_stack[old_position[step]]
        _require(torch.allclose(replay, reference, atol=1e-6, rtol=1e-6), f"step{step} no longer reproduces the existing 50-image CMOS replay")
        _require(before == after, f"step{step} changed runtime state during 50-image preflight")
        report[str(step)] = {"rows": 50, "allclose_atol_rtol": 1e-6, "matches_prior_raw_replay": True, "runtime_state_before_sha256": before, "runtime_state_after_sha256": after}
    return {"artifact_sha256": file_sha256(root / PRIOR_OUTPUT / "candidate_cmos_raw.pt"), "step_replays": report}


def _bootstrap_summary(values: np.ndarray, labels: np.ndarray) -> tuple[dict[int, tuple[float, float, float]], dict[tuple[int, int], tuple[float, float, float]]]:
    _require(values.shape == (len(STEPS), len(labels)), "bootstrap input shape changed")
    groups = [np.flatnonzero(labels == digit) for digit in range(10)]
    _require(all(len(group) == 500 for group in groups), "bootstrap input is not class-balanced")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = np.empty((BOOTSTRAP_REPLICATES, len(STEPS)), dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        boot[replicate] = values[:, sampled].mean(axis=1)
    point = values.mean(axis=1)
    step_intervals = {i: (float(point[i]), float(np.quantile(boot[:, i], 0.025)), float(np.quantile(boot[:, i], 0.975))) for i in range(len(STEPS))}
    paired = {
        (i, j): (
            float(point[i] - point[j]),
            float(np.quantile(boot[:, i] - boot[:, j], 0.025)),
            float(np.quantile(boot[:, i] - boot[:, j], 0.975)),
        )
        for i in range(len(STEPS))
        for j in range(i + 1, len(STEPS))
    }
    return step_intervals, paired


def _summarize_metrics(
    metric_data: Mapping[str, np.ndarray],
    predictions: Mapping[int, Mapping[str, np.ndarray]],
    labels: np.ndarray,
    steps: Sequence[int],
    ai_scores: Mapping[int, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    metric_cis = {name: _bootstrap_summary(values, labels) for name, values in metric_data.items()}
    model_values = {
        name: np.stack([(predictions[step][name] == labels).astype(np.float64) for step in steps])
        for name in MODEL_NAMES
    }
    model_cis = {name: _bootstrap_summary(values, labels) for name, values in model_values.items()}
    for step_i, step in enumerate(steps):
        base: dict[str, Any] = {"step": step, "n": int(len(labels)), "per_class_n": 500}
        for metric_name, (ci, _paired) in metric_cis.items():
            estimate, low, high = ci[step_i]
            base.update({f"{metric_name}_mean": estimate, f"{metric_name}_ci95_low": low, f"{metric_name}_ci95_high": high})
        for model_name, model_predictions in predictions[step].items():
            correctness = (model_predictions == labels).astype(np.float64)
            ci, _paired = model_cis[model_name]
            estimate, low, high = ci[step_i]
            base.update({f"{model_name}_accuracy": float(correctness.mean()), f"{model_name}_ci95_low": low, f"{model_name}_ci95_high": high})
            if model_name == "E_D":
                base.update({"E_D_accuracy_mean": float(correctness.mean()), "E_D_accuracy_ci95_low": low, "E_D_accuracy_ci95_high": high})
            confusion = np.zeros((10, 10), dtype=np.int64)
            np.add.at(confusion, (labels, model_predictions), 1)
            for true_label in range(10):
                support = int(confusion[true_label].sum())
                recall = float(confusion[true_label, true_label] / support) if support else math.nan
                class_rows.append({"step": step, "evaluator": model_name, "digit": true_label, "support": support, "recall": recall})
                for predicted_label in range(10):
                    confusion_rows.append({"step": step, "evaluator": model_name, "true_digit": true_label, "predicted_digit": predicted_label, "count": int(confusion[true_label, predicted_label])})
        base.update({
            "single_AI_correct_clear_50": ai_scores[step]["correct_clear"],
            "single_AI_correct_50": ai_scores[step]["correct"],
            "single_AI_clear_50": ai_scores[step]["clear"],
            "single_AI_readability_rank_27": ai_scores[step]["readability_rank"],
        })
        summary_rows.append(base)
    for metric_name, (_step_cis, pair_cis) in metric_cis.items():
        for (i, j), (delta, low, high) in pair_cis.items():
            paired_rows.append({"metric": metric_name, "step_a": steps[i], "step_b": steps[j], "paired_mean_difference_a_minus_b": delta, "ci95_low": low, "ci95_high": high, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED})
    for model_name, (_step_cis, pair_cis) in model_cis.items():
        for (i, j), (delta, low, high) in pair_cis.items():
            paired_rows.append({"metric": f"{model_name}_accuracy", "step_a": steps[i], "step_b": steps[j], "paired_mean_difference_a_minus_b": delta, "ci95_low": low, "ci95_high": high, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED})
    return summary_rows, paired_rows, class_rows, confusion_rows


def _compute_node_records(root: Path, consumer: Any, payloads: Mapping[int, Mapping[str, Any]], node_indices: torch.Tensor, authorities: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    supplement = _read_json(root / PRIOR_OUTPUT / "step6000_trace_supplement.json")
    reference_indices = [int(value) for value in supplement["display_canonical_source_indices"]]
    _require(node_indices.tolist() == reference_indices, "eight-node source order differs from the accepted fixed manifest")
    images = authorities["images"].index_select(0, node_indices).contiguous()
    node_labels = authorities["labels"].index_select(0, node_indices).tolist()
    _require(len(images) == 20, "node comparison must use the fixed 20 rows")
    node_rows: list[dict[str, Any]] = []
    ols_rows: list[dict[str, Any]] = []
    state_hashes: dict[str, Any] = {}
    for step in STEPS:
        payload = payloads[step]
        consumer.load_trainable_state_dict({"encoder.phase": payload["encoder_state"], "generator.phase": payload["generator_state"]})
        runtime = consumer.runtime
        input_relay = runtime.input_relay
        output_relay = runtime.output_relay
        relay_fields: dict[str, Any] = {"input": None, "output": None}

        class CaptureRelay:
            def __init__(self, key: str, relay: Any) -> None:
                self.key = key
                self.relay = relay

            def __call__(self, field: Any) -> Any:
                value = self.relay(field)
                relay_fields[self.key] = value
                return value

            def __getattr__(self, name: str) -> Any:
                return getattr(self.relay, name)

        runtime.input_relay = CaptureRelay("input", input_relay)
        runtime.output_relay = CaptureRelay("output", output_relay)
        before = visual._state_digest(runtime.state_dict())
        try:
            with torch.inference_mode():
                result = runtime.forward_from_image(images.to(consumer.device))
        finally:
            runtime.input_relay = input_relay
            runtime.output_relay = output_relay
        after = visual._state_digest(runtime.state_dict())
        _require(before == after, f"step{step} changed runtime state during node capture")
        _require(result.image_input is not None and relay_fields["input"] is not None and relay_fields["output"] is not None, f"step{step} omitted an optical plane")
        _require(torch.allclose(relay_fields["output"].tensor.abs().square(), result.raw_intensity, atol=1e-6, rtol=1e-6), f"step{step} CMOS square-law readout differs from raw intensity")
        coeff = result.z_ols.coefficients
        _require(torch.is_complex(coeff) and coeff.shape == (20, 100), f"step{step} OLS coefficient count/order changed")
        _require(torch.equal(coeff, coeff.reshape(20, 10, 10).reshape(20, 100)), f"step{step} OLS row-major 10x10 layout does not round-trip")
        fields = (
            result.image_input.slm_field,
            relay_fields["input"],
            result.encoder_records[0],
            result.encoder_records[1],
            result.generator_input_field,
            result.generator_records[0],
            result.generator_records[1],
            relay_fields["output"],
        )
        _require(len(fields) == 8 and len(result.encoder_records) == 2 and len(result.generator_records) == 2, f"step{step} optical node count changed")
        power_by_row: list[np.ndarray] = []
        for node_i, (field, key, label) in enumerate(zip(fields, NODE_KEYS, NODE_LABELS, strict=True)):
            tensor = field.tensor.detach()
            _require(torch.is_complex(tensor) and bool(torch.isfinite(tensor.real).all()) and bool(torch.isfinite(tensor.imag).all()), f"nonfinite complex field at {key}, step{step}")
            power = optical.physical_plane_power(field).detach().cpu().numpy()
            intensity_weights = tensor.abs().square()
            weight_sum = intensity_weights.flatten(1).sum(dim=1).clamp_min(1e-30)
            phasor = (tensor * tensor.abs()).flatten(1).sum(dim=1) / weight_sum
            phase_resultant = phasor.abs().detach().cpu().numpy()
            power_by_row.append(power)
            plane = field.plane
            for sample_i, (source_index, label_value) in enumerate(zip(node_indices.tolist(), node_labels, strict=True)):
                row: dict[str, Any] = {
                    "step": step,
                    "source_index": source_index,
                    "true_digit": int(label_value),
                    "node_index": node_i + 1,
                    "node_key": key,
                    "node_name": label,
                    "height": int(tensor.shape[-2]),
                    "width": int(tensor.shape[-1]),
                    "pixel_pitch_y_m": float(plane.pixel_pitch_m[0]),
                    "pixel_pitch_x_m": float(plane.pixel_pitch_m[1]),
                    "physical_power": float(power[sample_i]),
                    "intensity_weighted_phase_resultant": float(phase_resultant[sample_i]),
                    "phase_statistic_role": "descriptive_only_no_phase_reference",
                }
                if node_i > 0:
                    row["power_ratio_to_previous_node"] = float(power[sample_i] / max(power_by_row[node_i - 1][sample_i], 1e-300))
                else:
                    row["power_ratio_to_previous_node"] = ""
                node_rows.append(row)
        coeff_cpu = coeff.detach().cpu()
        for sample_i, (source_index, label_value) in enumerate(zip(node_indices.tolist(), node_labels, strict=True)):
            for port in range(100):
                value = coeff_cpu[sample_i, port]
                ols_rows.append({
                    "step": step,
                    "source_index": source_index,
                    "true_digit": int(label_value),
                    "port_flat_row_major": port,
                    "row_10x10": port // 10,
                    "column_10x10": port % 10,
                    "coefficient_real": float(value.real),
                    "coefficient_imag": float(value.imag),
                    "coefficient_amplitude": float(value.abs()),
                    "coefficient_phase_rad": float(torch.angle(value).remainder(2.0 * math.pi)),
                })
        state_hashes[str(step)] = {"before": before, "after": after, "unchanged": before == after}
    return node_rows, ols_rows, state_hashes


def _figure_style() -> None:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })


def _load_panel_alignment_helper() -> Any:
    path = Path.home() / ".codex" / "skills" / "nature-figure" / "scripts" / "audit_panel_alignment.py"
    _require(path.is_file(), f"scientific-figure panel alignment helper is missing: {path}")
    spec = importlib.util.spec_from_file_location("nature_figure_audit_panel_alignment", path)
    _require(spec is not None and spec.loader is not None, "cannot load panel alignment helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render_summary(output_dir: Path, summary_rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    _figure_style()
    alignment = _load_panel_alignment_helper()
    x = np.arange(len(STEPS))
    labels = [f"{step}" for step in STEPS]
    colors = ["#c6d7e8", "#8eacc9", "#557da5", "#244f78"]
    panels = [
        ("E_D_accuracy", "E_D accuracy", "Accuracy (machine proxy)", (0.0, 1.0)),
        ("L_rec_mse", "Raw reconstruction MSE", "MSE (raw 200×200)", None),
        ("pearson_r", "Paired pixel correlation", "Pearson r", (-1.0, 1.0)),
        ("foreground_dice", "Fixed-threshold foreground overlap", "Dice (threshold 0.2)", (0.0, 1.0)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.4), constrained_layout=True)
    axes_flat = axes.flat
    for panel_i, (axis, (metric, title, ylabel, ylim)) in enumerate(zip(axes_flat, panels, strict=True)):
        values = np.array([float(row[f"{metric}_mean"]) for row in summary_rows])
        low = np.array([float(row[f"{metric}_ci95_low"]) for row in summary_rows])
        high = np.array([float(row[f"{metric}_ci95_high"]) for row in summary_rows])
        for index in range(len(STEPS)):
            axis.errorbar(index, values[index], yerr=[[values[index] - low[index]], [high[index] - values[index]]], fmt="o", color=colors[index], markeredgecolor="#182634", markeredgewidth=0.6, capsize=4, markersize=7, linewidth=1.2)
        axis.set_xticks(x, labels)
        axis.set_xlabel("Checkpoint step")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", pad=10)
        axis.grid(axis="y", color="#cbd3da", linewidth=0.6, alpha=0.7)
        if ylim is not None:
            axis.set_ylim(*ylim)
        if metric == "E_D_accuracy":
            axis.axhline(0.1, color="#7d858c", linestyle="--", linewidth=0.9, zorder=0)
            axis.text(0.02, 0.12, "10% chance", transform=axis.transAxes, color="#687078", fontsize=8)
        axis.text(-0.10, 1.07, chr(ord("a") + panel_i), transform=axis.transAxes, fontsize=13, fontweight="bold", va="bottom", ha="left")
    fig.suptitle("CMOS evaluation on the same balanced DEV set", fontsize=15, fontweight="bold")
    fig.text(0.5, -0.012, "n = 5,000 (500/digit); points and 95% intervals use paired, digit-stratified bootstrap (2,000 resamples). E_D is a frozen machine-recognition proxy.", ha="center", fontsize=8.5)
    base = output_dir / "four_step_metric_summary"
    fig.canvas.draw()
    alignment.require_matplotlib_panel_alignment(
        fig,
        json_out=str(base.with_suffix(".alignment.json")),
        overlay_svg=str(base.with_suffix(".alignment.svg")),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return {"png": base.with_suffix(".png").name, "pdf": base.with_suffix(".pdf").name, "svg": base.with_suffix(".svg").name, "alignment_json": base.with_suffix(".alignment.json").name, "alignment_overlay": base.with_suffix(".alignment.svg").name}


def _predict_clean_reference(models: Mapping[str, Any], images: torch.Tensor, labels: torch.Tensor, device: torch.device) -> dict[str, dict[str, Any]]:
    outputs: dict[str, list[torch.Tensor]] = {name: [] for name in models}
    with torch.inference_mode():
        for start in range(0, len(images), FORWARD_BATCH_SIZE):
            batch = images[start : start + FORWARD_BATCH_SIZE].to(device)
            view = robust._cmos_view(batch)
            for name, model in models.items():
                outputs[name].append(_logits(model, view).argmax(dim=1).cpu())
    y = labels.cpu().numpy()
    result: dict[str, dict[str, Any]] = {}
    for name, batches in outputs.items():
        prediction = torch.cat(batches).numpy()
        result[name] = {"accuracy": float((prediction == y).mean()), "n": int(len(y)), "per_class_recall": [float((prediction[y == digit] == digit).mean()) for digit in range(10)]}
    return result


def _run(root: Path, output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"requested diagnostic output already exists: {output_dir}")
    cuda_runtime = robust._prepare_cuda_runtime()
    _require(torch.cuda.is_available() and torch.cuda.device_count() > 0, "full replay requires CUDA")
    device = torch.device("cuda:0")
    started = time.perf_counter()
    memory_free_before, memory_total = torch.cuda.mem_get_info(device)
    _require(memory_free_before > 1_500_000_000, "CUDA free memory is below the 1.5 GB diagnostic envelope")

    checkpoints, payloads = _selected_checkpoints(root)
    authorities = sic._load_authorities(root)
    dev_images, dev_labels, dev_indices, sample50_indices, selection_manifest, node_indices = _load_fixed_inputs(root, authorities)
    consumer, models = visual._load_models(root, authorities, device)
    _require(consumer.device == device, "LightGen consumer was not loaded on the selected CUDA device")
    for name, model in models.items():
        model.eval()
        _require(all(not parameter.requires_grad for parameter in model.parameters()), f"{name} parameters are not frozen")

    relevant_sources = (
        Path(__file__).resolve().relative_to(root).as_posix(),
        Path("scripts/research/lightgen/visualize_lightgen_2d_cmos_eight_node_step_comparison_v1.py"),
        Path("scripts/research/lightgen/execute_lightgen_2d_evaluator_robust_semantic_training_v1.py"),
        Path("scripts/research/lightgen/execute_lightgen_2d_three_evaluator_robust_semantic_continuation_v1.py"),
        Path(robust.successor.__file__).resolve().relative_to(root).as_posix(),
        Path("outputs/20260923_cmos_eight_node_step_comparison_v1/runs/attempt_01/checkpoint_manifest.json"),
        Path("outputs/20260923_cmos_eight_node_step_comparison_v1/runs/attempt_01/candidate_cmos_raw.pt"),
        SELECTION_MANIFEST,
    ) + tuple(Path(item["checkpoint_path"]) for item in checkpoints) + tuple(Path(item["report_path"]) for item in checkpoints)
    config_identity = {
        "steps": list(STEPS),
        "n_dev": 5000,
        "per_class": 500,
        "foreground_threshold": FOREGROUND_THRESHOLD,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "batch_size": FORWARD_BATCH_SIZE,
        "optimizer": None,
        "training": False,
    }
    source_manifest = capture_execution_source_manifest(
        root,
        task_id=TASK_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        relevant_paths=relevant_sources,
        config_identity=config_identity,
    )
    identity = ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=RUN_ID,
        trajectory_id=TRAJECTORY_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        data_role="DEV",
        config_identity=config_identity,
        model_identity={str(item["step"]): item["checkpoint_sha256"] for item in checkpoints},
        optimizer_identity={"optimizer": "none", "backward": False},
        sampler_identity={"role": "fixed canonical DEV", "development_indices_sha256": tensor_sha256(dev_indices), "class_counts": [500] * 10},
        execution_source_semantics_id=source_manifest.execution_source_semantics_id,
        source_manifest_id=source_manifest.source_manifest_id,
    )
    run_control = ResearchRunControl(identity=identity, authorization_id=AUTHORIZATION_ID)

    # Bounded preflight: fixed subset replay, identical data/checkpoint lineage,
    # inference-batch resource check, and frozen state checks. No output run is
    # created until the READY receipt is accepted.
    fixed_images = authorities["images"].index_select(0, torch.tensor(sample50_indices, dtype=torch.long)).contiguous()
    preflight_started = time.perf_counter()
    replay_check = _replay_fixed_50(root, consumer, payloads, fixed_images)
    model_states_before = {name: visual._state_digest(model.state_dict()) for name, model in models.items()}
    consumer.load_trainable_state_dict({"encoder.phase": payloads[STEPS[0]]["encoder_state"], "generator.phase": payloads[STEPS[0]]["generator_state"]})
    batch_before = visual._state_digest(consumer.runtime.state_dict())
    with torch.inference_mode():
        batch_result = consumer.runtime.forward_from_image(dev_images[:FORWARD_BATCH_SIZE].to(device))
    _require(batch_result.raw_intensity.shape == (FORWARD_BATCH_SIZE, 1, 200, 200), "batch-10 CUDA preflight geometry changed")
    _require(bool(torch.isfinite(batch_result.raw_intensity).all()), "batch-10 CUDA preflight produced nonfinite intensity")
    _require(batch_result.image_input is not None and batch_result.image_input.slm_intensity.shape == batch_result.raw_intensity.shape, "batch-10 CMOS target geometry changed")
    batch_after = visual._state_digest(consumer.runtime.state_dict())
    _require(batch_before == batch_after, "batch-10 CUDA preflight mutated the model runtime")
    for name, model in models.items():
        _require(model_states_before[name] == visual._state_digest(model.state_dict()), f"preflight changed frozen evaluator {name}")
    torch.cuda.synchronize(device)
    preflight_seconds = time.perf_counter() - preflight_started
    memory_free_after, _ = torch.cuda.mem_get_info(device)
    allocated_peak = int(torch.cuda.max_memory_allocated(device))
    receipt = PreflightReceipt(
        source_semantics_id=source_manifest.execution_source_semantics_id,
        authorization_id=AUTHORIZATION_ID,
        checks={
            CORRECTNESS_INVARIANTS: {
                "status": "PASS",
                "four_checkpoint_hashes_and_reports_validated": True,
                "canonical_DEV_exactly_500_per_class": True,
                "50_image_same_step_replays_match_existing_raw_CMOS": True,
                "batch10_raw_target_geometry_and_finiteness": True,
                "model_and_evaluator_state_unchanged": True,
            },
            RESOURCE_ENVELOPE: {
                "status": "PASS",
                "device": str(device),
                "cuda_free_bytes_before": int(memory_free_before),
                "cuda_free_bytes_after_preflight": int(memory_free_after),
                "cuda_total_bytes": int(memory_total),
                "batch_size": FORWARD_BATCH_SIZE,
                "peak_allocated_bytes": allocated_peak,
                "preflight_seconds": preflight_seconds,
                "cuda_runtime_dll_search": cuda_runtime,
            },
            IMPLEMENTATION_SELECTION: {
                "status": "PASS",
                "selected_engine": "CUDA inference batch 10",
                "steps": list(STEPS),
                "metric_protocol": "fixed paired raw 200x200 CMOS metrics; no per-image stretch",
                "optimizer_or_training": False,
            },
        },
        selected_engine="cuda:0",
        resource={"device": str(device), "free_bytes_before": int(memory_free_before), "free_bytes_after_preflight": int(memory_free_after), "total_bytes": int(memory_total), "batch_size": FORWARD_BATCH_SIZE, "peak_allocated_bytes": allocated_peak, "cuda_runtime_dll_search": cuda_runtime},
    )
    _require(receipt.status == PREFLIGHT_READY, "metric audit preflight did not produce READY")
    run_control.mark_preflight_ready(receipt)
    run_control.open_scientific_attempt(output_dir, attempt_id=f"{RUN_ID}:attempt_02", receipt=receipt)
    _write_json(output_dir / "run_identity.json", {"identity": run_control.identity.to_dict(), "source_manifest": source_manifest.to_dict(), "preflight_receipt": receipt.to_dict()})
    _write_json(output_dir / "checkpoint_selection.json", {"source_manifest": str((root / PRIOR_OUTPUT / "checkpoint_manifest.json").relative_to(root)), "source_manifest_sha256": file_sha256(root / PRIOR_OUTPUT / "checkpoint_manifest.json"), "steps": checkpoints, "candidate_bound": 9300})
    _write_json(output_dir / "sample_manifest.json", {
        "schema": "lightgen_cmos_metric_audit_fixed_dev.v1",
        "selection_manifest": SELECTION_MANIFEST.as_posix(),
        "selection_manifest_sha256": file_sha256(root / SELECTION_MANIFEST),
        "n": int(dev_indices.numel()),
        "per_class": 500,
        "canonical_source_indices": dev_indices.tolist(),
        "labels": dev_labels.tolist(),
        "development_indices_sha256": tensor_sha256(dev_indices),
        "fixed_50_cmos_source_indices": sample50_indices,
        "fixed_50_sample_manifest_sha256": file_sha256(root / PRIOR_OUTPUT / "sample_manifest.json"),
        "eight_node_source_indices": node_indices.tolist(),
        "roles": {"all_full_metrics": "DEV", "protected": 0, "test": 0, "sealed": 0},
    })

    clean_reference = _predict_clean_reference(models, dev_images, dev_labels, device)
    _require(all(0.0 <= result["accuracy"] <= 1.0 for result in clean_reference.values()), "clean DEV positive-control accuracy is out of range")
    clean_state_after = {name: visual._state_digest(model.state_dict()) for name, model in models.items()}
    _require(clean_state_after == model_states_before, "clean DEV positive control changed frozen evaluator state")
    _write_json(output_dir / "clean_dev_positive_control.json", {"role": "same frozen CMOS view applied to the real 28x28 DEV images", "evaluators": clean_reference})

    n = int(dev_indices.numel())
    metric_data: dict[str, np.ndarray] = {name: np.full((len(STEPS), n), np.nan, dtype=np.float64) for name in METRIC_NAMES}
    mismatch_data: dict[str, np.ndarray] = {name: np.full((len(STEPS), n), np.nan, dtype=np.float64) for name in METRIC_NAMES}
    predictions: dict[int, dict[str, np.ndarray]] = {step: {name: np.full(n, -1, dtype=np.int64) for name in MODEL_NAMES} for step in STEPS}
    per_sample_path = output_dir / "per_sample_metrics_20000.csv"
    per_sample_row_count = 0
    step_timing: dict[str, Any] = {}
    source_to_position = {int(source): position for position, source in enumerate(dev_indices.tolist())}
    labels_np = dev_labels.numpy()
    for step_i, step in enumerate(STEPS):
        row_by_position: list[dict[str, Any] | None] = [None] * n
        payload = payloads[step]
        consumer.load_trainable_state_dict({"encoder.phase": payload["encoder_state"], "generator.phase": payload["generator_state"]})
        consumer.runtime.eval()
        before = visual._state_digest(consumer.runtime.state_dict())
        model_before = {name: visual._state_digest(model.state_dict()) for name, model in models.items()}
        per_step_started = time.perf_counter()
        done = 0
        with torch.inference_mode():
            for digit in range(10):
                class_indices = dev_indices[dev_labels == digit]
                _require(class_indices.numel() == 500, f"step{step} digit {digit} lost class balance")
                for start in range(0, 500, FORWARD_BATCH_SIZE):
                    indices = class_indices[start : start + FORWARD_BATCH_SIZE]
                    batch = authorities["images"].index_select(0, indices).to(device)
                    result = consumer.runtime.forward_from_image(batch)
                    raw = result.raw_intensity
                    target = result.image_input.slm_intensity
                    _require(raw.shape == (len(indices), 1, 200, 200) and target.shape == raw.shape, f"step{step} raw/target geometry changed")
                    _require(bool(torch.isfinite(raw).all()) and bool(torch.isfinite(target).all()), f"step{step} raw/target intensity is nonfinite")
                    paired = _metric_arrays(raw, target)
                    mismatched_target = torch.roll(target, shifts=1, dims=0)
                    mismatched = _metric_arrays(raw, mismatched_target)
                    view = robust._cmos_view(raw)
                    batch_predictions: dict[str, torch.Tensor] = {}
                    for name, model in models.items():
                        prediction = _logits(model, view).argmax(dim=1)
                        batch_predictions[name] = prediction
                    batch_pos = np.array([source_to_position[int(value)] for value in indices.tolist()], dtype=np.int64)
                    current_label = authorities["labels"].index_select(0, indices).numpy()
                    for name, values in paired.items():
                        metric_data[name][step_i, batch_pos] = values.detach().cpu().numpy()
                    for name, values in mismatched.items():
                        mismatch_data[name][step_i, batch_pos] = values.detach().cpu().numpy()
                    for name, prediction in batch_predictions.items():
                        predictions[step][name][batch_pos] = prediction.cpu().numpy()
                    metric_names = list(paired)
                    paired_np = {name: values.detach().cpu().numpy() for name, values in paired.items()}
                    mismatch_np = {name: values.detach().cpu().numpy() for name, values in mismatched.items()}
                    for local, source in enumerate(indices.tolist()):
                        position = source_to_position[int(source)]
                        row: dict[str, Any] = {"step": step, "source_index": int(source), "true_digit": int(current_label[local])}
                        for name in metric_names:
                            row[name] = float(paired_np[name][local])
                            row[f"same_digit_mismatched_{name}"] = float(mismatch_np[name][local])
                        for name, prediction in batch_predictions.items():
                            guessed = int(prediction[local].item())
                            row[f"{name}_prediction"] = guessed
                            row[f"{name}_correct"] = guessed == int(current_label[local])
                        row["E_D_accuracy"] = row["E_D_correct"]
                        row_by_position[position] = row
                    done += int(len(indices))
                    if done % 500 == 0:
                        print(json.dumps({"step": step, "rows": done, "total": 5000}, ensure_ascii=False), flush=True)
        after = visual._state_digest(consumer.runtime.state_dict())
        model_after = {name: visual._state_digest(model.state_dict()) for name, model in models.items()}
        _require(before == after, f"step{step} changed runtime state across 5,000-image inference")
        _require(model_before == model_after == model_states_before, f"step{step} changed one or more frozen evaluators")
        _require(all(np.isfinite(value[step_i]).all() for value in metric_data.values()), f"step{step} omitted or produced nonfinite metrics")
        _require(all(np.isfinite(value[step_i]).all() for value in mismatch_data.values()), f"step{step} omitted or produced nonfinite mismatch-control metrics")
        _require(all((array >= 0).all() and (array <= 9).all() for array in predictions[step].values()), f"step{step} evaluator predictions are incomplete")
        _require(all(row is not None for row in row_by_position), f"step{step} per-sample DEV rows are incomplete")
        step_rows = [row for row in row_by_position if row is not None]
        _append_csv(per_sample_path, step_rows)
        per_sample_row_count += len(step_rows)
        torch.cuda.synchronize(device)
        step_timing[str(step)] = {"elapsed_seconds": time.perf_counter() - per_step_started, "rows": done, "runtime_state_before_sha256": before, "runtime_state_after_sha256": after, "evaluators_unchanged": True}

    _require(per_sample_row_count == len(STEPS) * n and per_sample_path.is_file(), "per-sample DEV output is incomplete")
    ai_ranking_path = root / PRIOR_OUTPUT / "readability_ranking_27_candidates.csv"
    with ai_ranking_path.open("r", newline="", encoding="utf-8-sig") as stream:
        ai_rows = list(csv.DictReader(stream))
    ai_scores = {int(row["step"]): {"correct_clear": int(row["correct_clear"]), "correct": int(row["correct"]), "clear": int(row["clear"]), "readability_rank": int(row["readability_rank"])} for row in ai_rows if int(row["step"]) in STEPS}
    _require(set(ai_scores) == set(STEPS), "existing single-AI 50-image readability record is incomplete")
    summary_rows, pair_rows, class_rows, confusion_rows = _summarize_metrics(metric_data, predictions, labels_np, STEPS, ai_scores)
    _write_csv(output_dir / "step_metric_summary.csv", summary_rows)
    _write_csv(output_dir / "paired_step_differences.csv", pair_rows)
    _write_csv(output_dir / "per_class_evaluator_recall.csv", class_rows)
    _write_csv(output_dir / "confusion_matrices.csv", confusion_rows)

    node_rows, ols_rows, node_state_hashes = _compute_node_records(root, consumer, payloads, node_indices, authorities)
    _write_csv(output_dir / "eight_node_physical_metrics_n20.csv", node_rows)
    _write_csv(output_dir / "ols_100_port_coefficients_n20.csv", ols_rows)
    fig_files = _render_summary(output_dir, summary_rows)

    metric_names = list(metric_data)
    mismatch_checks: dict[str, Any] = {}
    for name in ("L_rec_mse", "pearson_r", "foreground_dice"):
        paired_values = metric_data[name]
        wrong_values = mismatch_data[name]
        if name == "L_rec_mse":
            direction_pass = bool((wrong_values.mean(axis=1) > paired_values.mean(axis=1)).all())
        else:
            direction_pass = bool((wrong_values.mean(axis=1) < paired_values.mean(axis=1)).all())
        mismatch_checks[name] = {"paired_mean_by_step": paired_values.mean(axis=1).tolist(), "same_digit_wrong_pair_mean_by_step": wrong_values.mean(axis=1).tolist(), "expected_direction_detected_for_every_step": direction_pass}
    negative_control_status = "PASS" if all(record["expected_direction_detected_for_every_step"] for record in mismatch_checks.values()) else "DID_NOT_SEPARATE_ALL_METRICS"

    sample_manifest = _read_json(root / PRIOR_OUTPUT / "sample_manifest.json")
    old_metadata = _read_json(root / PRIOR_OUTPUT / "visualization_metadata.json")
    node_summary: dict[str, Any] = {}
    for step in STEPS:
        step_summary: dict[str, Any] = {}
        for key in NODE_KEYS:
            entries = [row for row in node_rows if row["step"] == step and row["node_key"] == key]
            powers = np.array([float(row["physical_power"]) for row in entries], dtype=np.float64)
            phase_r = np.array([float(row["intensity_weighted_phase_resultant"]) for row in entries], dtype=np.float64)
            step_summary[key] = {
                "node_name": entries[0]["node_name"],
                "physical_grid": [entries[0]["height"], entries[0]["width"]],
                "pixel_pitch_m": [entries[0]["pixel_pitch_y_m"], entries[0]["pixel_pitch_x_m"]],
                "n": len(entries),
                "median_physical_power": float(np.median(powers)),
                "power_q25": float(np.quantile(powers, 0.25)),
                "power_q75": float(np.quantile(powers, 0.75)),
                "median_intensity_weighted_phase_resultant": float(np.median(phase_r)),
                "phase_resultant_role": "descriptive only; no target phase field exists",
            }
        node_summary[str(step)] = step_summary
    _write_json(output_dir / "eight_node_summary.json", {"step_node_summary": node_summary, "interpretation": "plane power uses dy*dx*sum(|E|^2); phase resultant is descriptive and does not rank image quality"})

    elapsed = time.perf_counter() - started
    run_report = {
        "schema": "lightgen_cmos_metric_audit_report.v1",
        "result": "VALID_COMPLETED_DESCRIPTIVE_DIAGNOSTIC",
        "classification": "GROUP_MEETING_STAGE_COMPARISON_ONLY",
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "scientific_attempt_id": run_control.identity.scientific_attempt_id,
        "steps": list(STEPS),
        "full_dev_rows_per_step": 5000,
        "per_digit": 500,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "raw_cmos_shape": [5000, 1, 200, 200],
        "raw_per_image_stretch": False,
        "foreground_threshold": FOREGROUND_THRESHOLD,
        "paired_mismatch_control": mismatch_checks,
        "paired_mismatch_control_status": negative_control_status,
        "clean_dev_positive_control": clean_reference,
        "50_image_prior_replay": replay_check,
        "checkpoint_identities": checkpoints,
        "selection_manifest_sha256": file_sha256(root / SELECTION_MANIFEST),
        "fixed_50_sample_manifest": {"canonical_source_indices": sample50_indices, "sha256": file_sha256(root / PRIOR_OUTPUT / "sample_manifest.json")},
        "eight_node_source_indices": node_indices.tolist(),
        "eight_node_state_hashes": node_state_hashes,
        "node_count": 8,
        "OLS_coefficients": {"count": 100, "ordering": "row-major 10x10", "round_trip_check": True},
        "camera_square_law": "|cmos_field|^2 == raw_intensity within 1e-6",
        "runtime_state_unchanged_for_each_step": True,
        "evaluator_parameters_frozen": True,
        "training_or_optimizer_or_backward_executed": False,
        "fid": {"computed": False, "reason": "distribution-only ImageNet feature statistic is not paired task accuracy; sample/domain/preprocessing mismatch makes it unsuitable for this checkpoint ranking"},
        "existing_single_AI_readability": "descriptive only; one AI reviewer on fixed 50-image subset; not human validation",
        "hardware_measured": False,
        "elapsed_seconds": elapsed,
        "per_step_timing": step_timing,
        "outputs": {**fig_files, "per_sample": "per_sample_metrics_20000.csv", "summary": "step_metric_summary.csv", "paired_differences": "paired_step_differences.csv", "confusions": "confusion_matrices.csv", "node_metrics": "eight_node_physical_metrics_n20.csv", "node_summary": "eight_node_summary.json", "ols": "ols_100_port_coefficients_n20.csv"},
    }
    _write_json(output_dir / "run_report.json", run_report)
    _write_json(output_dir / "figure_source_and_qa.json", {
        "figure": fig_files,
        "source_data": {"summary": "step_metric_summary.csv", "paired_differences": "paired_step_differences.csv", "full_per_sample": "per_sample_metrics_20000.csv"},
        "prior_cmos_comparison": (PRIOR_OUTPUT / "finalist_cmos_comparison.png").as_posix(),
        "prior_step1000_nodes": [(PRIOR_OUTPUT / f"step001000_{kind}.png").as_posix() for kind in ("amplitude", "phase")],
        "prior_step3100_nodes": [(PRIOR_OUTPUT / f"step003100_{kind}.png").as_posix() for kind in ("amplitude", "phase")],
        "prior_step6000_nodes": [(PRIOR_OUTPUT / f"step006000_{kind}.png").as_posix() for kind in ("amplitude", "phase")],
        "prior_step9000_nodes": [(PRIOR_OUTPUT / f"step009000_{kind}.png").as_posix() for kind in ("amplitude", "phase")],
        "previous_metadata_sha256": file_sha256(root / PRIOR_OUTPUT / "visualization_metadata.json"),
        "old_blind_sample_manifest_sha256": file_sha256(root / PRIOR_OUTPUT / "sample_manifest.json"),
    })
    (output_dir / "README.md").write_text(
        "# CMOS 与八节点指标审计\n\n"
        "## 用途\n\n"
        "本产物用于组会阶段比较 step1000、step3100、step6000、step9000。所有检查点在同一固定、按类平衡的 DEV 集上只读重放，每类500张、共5000张。指标支持模拟CMOS的机器识别和输入图对应性判断，不构成人眼可读性、人群共识、硬件测量、验收或检查点晋升。\n\n"
        "## 指标\n\n"
        "E_D准确率和逐类召回率是主要机器识别代理；E_S、E_I、E_C为辅助冻结评估器。E_D不参与反向传播，但曾用于阶段确认。真实DEV图像以相同CMOS视图作正对照。逐图输入对应指标在原始200×200强度上计算：MSE、Pearson相关、固定阈值0.2的前景Dice、前景漏检率、背景误显率、总强度比。每张图再与同一数字类别的另一张目标图错配，作为保留语义类别但破坏笔画配对的负对照。所有步骤使用相同曝光，禁止逐图拉伸。\n\n"
        "八节点功率按各光场的物理像素间距计算 `dy*dx*sum(|E|^2)`，不能直接比较不同网格的像素求和。相位采用强度加权圆统计量仅作描述，因为没有独立复场真值；它不进入最佳step判断。OLS复系数以行优先10×10顺序列出。\n\n"
        "误差区间使用逐类分层配对bootstrap（B=2000，seed=271828）；重采样单位是同一DEV图像，四个检查点共用重采样索引。区间描述该固定样本上的抽样波动，不表示训练随机种子或硬件重复实验。\n\n"
        "此前50张图的AI评分为单个AI观察者的盲读记录，仅与本次5000张自动指标并列提供。FID未计算：它是分布级图像特征距离，不能判断样本是否对应正确输入；ImageNet特征、灰度CMOS预处理与小样本估计也使其不适合作为本次排序依据。\n\n"
        f"同类错配负对照状态：{negative_control_status}。若某项指标未按预期区分正确配对与同类错配，报告保留该指标并标出这一局限，不将其用作单独判优依据。\n\n"
        "## 配对图表与光路参考\n\n"
        "CMOS并排图见 `../../../20260923_cmos_eight_node_step_comparison_v1/runs/attempt_01/finalist_cmos_comparison.png`；四个step的振幅/相位图见对应历史输出中的 `step001000_*`、`step003100_*`、`step006000_*`、`step009000_*`。本次新绘制的四步指标图保存在 `four_step_metric_summary.png` 和PDF/SVG版本。\n\n"
        "## 复核\n\n"
        "输出包含检查点及报告SHA、固定DEV索引及50张子集身份、重放一致性、CUDA预检收据、逐模型状态摘要、逐样本指标、混淆计数、八节点物理网格、相机平方律一致性和OLS端口表。无训练、反向传播、优化器更新或检查点改写。\n",
        encoding="utf-8",
    )
    run_control.complete()
    _write_json(output_dir / "run_status.json", {"state": "COMPLETED", "preflight_status": run_control.preflight_status, "result": run_report["result"], "elapsed_seconds": elapsed})
    return {"result": run_report["result"], "output_dir": output_dir.relative_to(root).as_posix(), "steps": list(STEPS), "seconds": elapsed, "preflight_status": run_control.preflight_status, "figures": fig_files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / OUTPUT_RELATIVE)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir = output_dir.resolve()
    result = _run(ROOT, output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
