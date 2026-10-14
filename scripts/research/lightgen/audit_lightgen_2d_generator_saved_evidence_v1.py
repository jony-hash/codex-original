"""Pair saved LightGen generator readouts on CPU without new optical work."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from scripts.research.lightgen import (
    execute_lightgen_2d_generator_class9_weighted_ce_comparison_v1 as class9,
)
from scripts.research.lightgen import (
    execute_lightgen_2d_generator_energy_matched_target_comparison_v1 as energy_task,
)


TASK_ID = "M0_GENERATOR_SAVED_EVIDENCE_CPU_REVIEW_V1"
SOURCE_TASK_ID = "M0_GENERATOR_ENERGY_MATCHED_TARGET_COMPARISON_V1"
SOURCE_ATTEMPT = Path(
    "outputs/20261009_m0_generator_energy_matched_target_comparison_v1/runs/attempt_03"
)
DEFAULT_OUTPUT = Path(
    "outputs/20261009_m0_generator_saved_evidence_cpu_review_v1/runs/attempt_01"
)
ENDPOINTS = {
    "A4700": "readouts_start.pt",
    "A6700": "readouts_A6700.pt",
    "B6700": "readouts_B6700.pt",
}
ROW_METRIC_FILES = {"A6700": "row_metrics_A6700.pt", "B6700": "row_metrics_B6700.pt"}
EVALUATORS = ("E_S", "E_I", "E_C", "E_D")
PAIRS = (("A4700", "A6700"), ("A4700", "B6700"), ("A6700", "B6700"))
FIT_ROWS = 200
CONFIRM_ROWS = 300
TOTAL_ROWS = FIT_ROWS + CONFIRM_ROWS
UPDATES_PER_ARM = 2_000
BATCH_SIZE = 10
BATCH_COUNT = FIT_ROWS // BATCH_SIZE
FLOAT_RTOL = 2.0e-6
FLOAT_ATOL = 5.0e-8
SIGN_RTOL = 2.0e-5
SIGN_ATOL = 2.0e-7
ENERGY_RELATIVE_SIGN_TOL = 2.0e-5


class SavedEvidenceAuditError(RuntimeError):
    """The explicit saved evidence is incomplete, inconsistent, or invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SavedEvidenceAuditError(message)


def _load_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required evidence file is missing: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise SavedEvidenceAuditError(f"evidence JSON is malformed: {path.name}: {error}") from error
    _require(isinstance(value, dict), f"evidence JSON must contain an object: {path.name}")
    return value


def _load_torch(path: Path, *, mmap: bool = False) -> Any:
    _require(path.is_file(), f"required tensor evidence file is missing: {path.name}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        raise SavedEvidenceAuditError(f"tensor evidence cannot be loaded: {path.name}: {error}") from error


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _verify_attempt(population: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    _require(population.get("task_id") == SOURCE_TASK_ID, "population belongs to a different source task")
    _require(report.get("task_id") == SOURCE_TASK_ID, "report belongs to a different source task")
    _require(report.get("status") == "VALID_COMPLETED", "source report is not a valid completed attempt")
    _require(report.get("classification") == "MIXED_OR_INSUFFICIENT", "source report classification differs from the named attempt")
    _require(report.get("training_recommendation") == "NONE", "source report contains an unexpected training recommendation")
    _require(report.get("next_action") == "NONE", "source report contains an unexpected automatic next action")
    _require(report.get("preflight", {}).get("status") == "READY", "source attempt lacks its completed production preflight")
    _require(report.get("source_A4700_replay", {}).get("passed") is True, "source A4700 readout replay did not pass")
    _require(report.get("source_step") == 4_700, "source step is not the explicit A4700 endpoint")
    _require(report.get("updates_per_arm") == UPDATES_PER_ARM, "source update count differs from the declared attempt")
    _require(report.get("final_generator_adam_step") == 6_700, "source final Generator step differs from the declared attempt")
    _require(report.get("stop") is True, "source report does not record a stop")
    _require(report.get("population") == population, "report and population manifests differ")
    roles = population.get("roles")
    labels = population.get("labels")
    rows = population.get("source_row_indices")
    _require(isinstance(roles, list) and len(roles) == TOTAL_ROWS, "population role list must have 500 rows")
    _require(isinstance(labels, list) and len(labels) == TOTAL_ROWS, "population labels must have 500 rows")
    _require(isinstance(rows, list) and len(rows) == TOTAL_ROWS, "population source rows must have 500 rows")
    _require(len(set(map(int, rows))) == TOTAL_ROWS, "population source rows contain duplicates")
    _require(all(int(label) in range(10) for label in labels), "population contains a label outside 0..9")
    _require(roles == ["fit"] * FIT_ROWS + ["confirmation"] * CONFIRM_ROWS, "fit and confirmation roles are not in the declared positions")
    _require(population.get("role_positions") == {"fit": [0, FIT_ROWS], "confirmation": [FIT_ROWS, TOTAL_ROWS]}, "population role intervals differ from the declared split")
    _require(population.get("fit_rows") == FIT_ROWS and population.get("confirmation_rows") == CONFIRM_ROWS, "population role counts differ from the declared split")
    _require(population.get("canonical_train_not_claimed_unseen") is True, "population provenance does not preserve the TRAIN evidence boundary")
    for role, start, stop, per_class in (("fit", 0, FIT_ROWS, 20), ("confirmation", FIT_ROWS, TOTAL_ROWS, 30)):
        role_labels = list(map(int, labels[start:stop]))
        _require(role_labels == [digit for _ in range(per_class) for digit in range(10)], f"{role} labels do not contain the declared balanced class order")
    _require(population.get("source_generator_adam_step") == 4_700, "population source Adam step differs from A4700")
    _require(population.get("final_generator_adam_step") == 6_700, "population final Adam step differs from the named attempt")
    _require(population.get("updates_per_arm") == UPDATES_PER_ARM, "population update count differs from the named attempt")
    _require(population.get("energy_scale_coefficients") and len(population["energy_scale_coefficients"]) == FIT_ROWS, "fixed fit-row coefficients are missing or incomplete")
    _require(population.get("energy_scale_source_rows") == rows[:FIT_ROWS], "fixed coefficient source rows differ from the fit population")
    _require(population.get("energy_scale_labels") == labels[:FIT_ROWS], "fixed coefficient labels differ from the fit population")
    checks["source_report_identity_and_terminal_decision"] = True
    checks["population_split_labels_and_train_boundary"] = True
    checks["fixed_target_coefficients_are_fit_only"] = True
    return checks


def _all_finite(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value).all()), f"saved {name} contains nonfinite values")


def _verify_readouts(
    population: Mapping[str, Any], readouts: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    expected_rows = torch.as_tensor(population["source_row_indices"], dtype=torch.long)
    expected_labels = torch.as_tensor(population["labels"], dtype=torch.long)
    base = readouts["A4700"]
    tensor_fields = sorted(key for key, value in base.items() if torch.is_tensor(value))
    metadata_fields = sorted(key for key, value in base.items() if not torch.is_tensor(value))
    required_tensors = {
        "z",
        "raw_cmos",
        "evaluator_view",
        "target_slm_intensity",
        "source_indices",
        "labels",
        "power_by_plane",
        "new_phase_plane_power",
        *(f"{evaluator.lower()}_logits" for evaluator in EVALUATORS),
    }
    _require(required_tensors.issubset(tensor_fields), "saved readout is missing a required measurement tensor")
    _require("plane_metadata" in metadata_fields, "saved readout lacks physical plane metadata")
    checks: dict[str, Any] = {"endpoint_rows": {}, "shared_population_fields": True, "shared_target_and_coordinates": True}
    for name, readout in readouts.items():
        _require(sorted(key for key, value in readout.items() if torch.is_tensor(value)) == tensor_fields, f"{name} tensor field set differs from A4700")
        _require(sorted(key for key, value in readout.items() if not torch.is_tensor(value)) == metadata_fields, f"{name} metadata field set differs from A4700")
        for key in metadata_fields:
            _require(readout[key] == base[key], f"{name} readout metadata differs from A4700: {key}")
        for key in tensor_fields:
            value = readout[key]
            _require(value.device.type == "cpu", f"{name} {key} was not loaded on CPU")
            _require(value.shape[0] == TOTAL_ROWS, f"{name} {key} does not contain 500 rows")
            _all_finite(value, f"{name}.{key}")
        _require(torch.equal(readout["source_indices"].to(torch.long), expected_rows), f"{name} source row indices differ from the input population")
        _require(torch.equal(readout["labels"].to(torch.long), expected_labels), f"{name} labels differ from the input population")
        _require(tuple(readout["z"].shape) == (TOTAL_ROWS, 100) and readout["z"].is_complex(), f"{name} OLS coefficients have an unexpected shape or type")
        _require(tuple(readout["raw_cmos"].shape) == (TOTAL_ROWS, 1, 200, 200), f"{name} raw CMOS shape differs")
        _require(tuple(readout["target_slm_intensity"].shape) == (TOTAL_ROWS, 1, 200, 200), f"{name} SLM target shape differs")
        _require(tuple(readout["evaluator_view"].shape) == (TOTAL_ROWS, 1, 28, 28), f"{name} production CMOS view shape differs")
        for evaluator in EVALUATORS:
            _require(tuple(readout[f"{evaluator.lower()}_logits"].shape) == (TOTAL_ROWS, 10), f"{name} {evaluator} logits shape differs")
        checks["endpoint_rows"][name] = {"rows": TOTAL_ROWS, "source_rows_match": True, "labels_match": True, "all_saved_tensors_finite": True}
    for name in ("A6700", "B6700"):
        _require(torch.equal(readouts[name]["target_slm_intensity"], base["target_slm_intensity"]), f"{name} target pixels differ from A4700")
        _require(torch.equal(readouts[name]["z"], base["z"]), f"{name} OLS coefficients differ from A4700")
    return checks


def _validate_history(
    source_dir: Path, population: Mapping[str, Any]
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    fit_rows = list(map(int, population["source_row_indices"][:FIT_ROWS]))
    fit_labels = list(map(int, population["labels"][:FIT_ROWS]))
    scales = list(map(float, population["energy_scale_coefficients"]))
    numeric_fields = (
        "raw_mse_training_target",
        "ssim_loss_training_target",
        "reconstruction_objective_training_target",
        "l_dist",
        "l_op",
        "phase_gradient_rms",
        "generator_displacement_rms",
        "original_100x100_displacement_rms",
        "added_400x400_displacement_rms",
        "original_100x100_gradient_norm",
        "added_400x400_gradient_norm",
        "ssim_gradient_norm",
    )
    for arm, filename in (("A_unscaled_target", "training_history_arm_a.json"), ("B_energy_matched_target", "training_history_arm_b.json")):
        payload = _load_json(source_dir / filename)
        _require(payload.get("task_id") == SOURCE_TASK_ID, f"{filename} task identity differs")
        _require(payload.get("source_step") == 4_700, f"{filename} source step differs")
        _require(payload.get("additional_updates") == UPDATES_PER_ARM, f"{filename} update count differs")
        _require(payload.get("arm") == arm, f"{filename} arm identity differs")
        records = payload.get("records")
        _require(isinstance(records, list) and len(records) == UPDATES_PER_ARM, f"{filename} must contain 2000 records")
        epochs: list[dict[str, Any]] = []
        for epoch in range(UPDATES_PER_ARM // BATCH_COUNT):
            block = records[epoch * BATCH_COUNT : (epoch + 1) * BATCH_COUNT]
            _require(len(block) == BATCH_COUNT, f"{filename} epoch {epoch + 1} is incomplete")
            for batch_index, record in enumerate(block):
                update = epoch * BATCH_COUNT + batch_index + 1
                start = batch_index * BATCH_SIZE
                expected_rule = "original_y" if arm == "A_unscaled_target" else "fixed_A4700_energy_ratio_times_y"
                _require(record.get("task_local_update") == update, f"{filename} task-local update sequence is discontinuous")
                _require(record.get("optimizer_step") == 4_700 + update, f"{filename} optimizer step sequence is discontinuous")
                _require(record.get("batch_index") == batch_index, f"{filename} batch order differs at update {update}")
                _require(record.get("arm") == arm and record.get("target_rule") == expected_rule, f"{filename} update {update} arm/target rule differs")
                _require(record.get("source_row_indices") == fit_rows[start : start + BATCH_SIZE], f"{filename} update {update} source rows differ")
                _require(record.get("labels") == fit_labels[start : start + BATCH_SIZE], f"{filename} update {update} labels differ")
                _require(record.get("fit_positions") == list(range(start, start + BATCH_SIZE)), f"{filename} update {update} fit positions differ")
                _require(len(record.get("target_scale_coefficients", [])) == BATCH_SIZE, f"{filename} update {update} coefficient count differs")
                _require(all(math.isclose(float(a), scales[start + i], rel_tol=FLOAT_RTOL, abs_tol=FLOAT_ATOL) for i, a in enumerate(record["target_scale_coefficients"])), f"{filename} update {update} coefficients differ")
                _require(record.get("all_finite") is True, f"{filename} update {update} is not marked finite")
                _require(all(math.isfinite(float(record[key])) for key in numeric_fields), f"{filename} update {update} contains a nonfinite metric")
                _require(set(record.get("ce_by_evaluator", {})) == {"E_S", "E_I", "E_C"}, f"{filename} update {update} semantic CE fields differ")
                _require(all(math.isfinite(float(value)) for value in record["ce_by_evaluator"].values()), f"{filename} update {update} contains a nonfinite CE")
                _require(len(record.get("five_plane_power_mean", [])) == 5, f"{filename} update {update} plane power count differs")
            epoch_summary: dict[str, Any] = {"epoch": epoch + 1, "optimizer_steps": [4_701 + epoch * BATCH_COUNT, 4_700 + (epoch + 1) * BATCH_COUNT]}
            for key in numeric_fields:
                values = [float(record[key]) for record in block]
                epoch_summary[key] = {"mean": sum(values) / BATCH_COUNT, "min": min(values), "max": max(values)}
            epoch_summary["ce_by_evaluator"] = {
                evaluator: {
                    "mean": sum(float(record["ce_by_evaluator"][evaluator]) for record in block) / BATCH_COUNT
                }
                for evaluator in ("E_S", "E_I", "E_C")
            }
            epoch_summary["five_plane_power_mean"] = [
                sum(float(record["five_plane_power_mean"][plane]) for record in block) / BATCH_COUNT
                for plane in range(5)
            ]
            epochs.append(epoch_summary)
        summaries[arm] = {
            "updates": len(records),
            "epochs": epochs,
            "objective_comparison_note": "A and B reconstruction values use different paired targets and are summarized separately.",
        }
    return {"passed": True, "arms": summaries}


def _derive_readout_metrics(
    readout: Mapping[str, Any], population: Mapping[str, Any], endpoint: str
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    _, row_values = class9._row_and_image_metrics(readout, population)
    energy, energy_rows = energy_task._energy_image_metrics(readout, {"population": population})
    combined_rows = {**row_values, **energy_rows}
    independent = _independent_metric_rows(readout)
    for key, value in independent.items():
        current = combined_rows[key]
        if current.dtype == torch.bool:
            _require(torch.equal(current, value), f"independent CPU image metric differs from the saved-metric producer: {key}")
        else:
            _require(bool(torch.allclose(current.to(torch.float64), value.to(torch.float64), rtol=FLOAT_RTOL, atol=FLOAT_ATOL)), f"independent CPU image metric differs from the saved-metric producer: {key}")
    combined_rows.update(_classification_rows(readout))
    combined_summary = energy_task._metrics_from_saved(readout, {"population": population}, endpoint)
    return combined_summary, combined_rows


def _classification_rows(readout: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    labels = readout["labels"].to(torch.long)
    truth_column = labels.view(-1, 1)
    result: dict[str, torch.Tensor] = {}
    for evaluator in EVALUATORS:
        prefix = evaluator.lower()
        logits = readout[f"{prefix}_logits"].to(torch.float64)
        true_logit = logits.gather(1, truth_column).squeeze(1)
        wrong_logits = logits.scatter(1, truth_column, -torch.inf)
        max_wrong = wrong_logits.max(dim=1, keepdim=True).values
        result[f"{prefix}_predictions"] = logits.argmax(dim=1)
        result[f"{prefix}_ce_float64"] = torch.logsumexp(logits, dim=1) - true_logit
        result[f"{prefix}_margin_float64"] = true_logit - max_wrong.squeeze(1)
        result[f"{prefix}_max_wrong_logit_tie_count"] = (wrong_logits == max_wrong).sum(dim=1)
    return result


def _normalized_q(generated_view: torch.Tensor, target_view: torch.Tensor) -> torch.Tensor:
    _require(generated_view.shape == target_view.shape and generated_view.ndim == 4, "Q views must have matching NCHW shapes")
    generated = generated_view.to(torch.float64).flatten(1)
    target = target_view.to(torch.float64).flatten(1)
    generated_mass = generated.sum(1)
    target_mass = target.sum(1)
    _require(bool((generated_mass > 0).all() and (target_mass > 0).all()), "Q is undefined for a zero-mass generated image or target")
    normalized_generated = generated / generated_mass[:, None]
    normalized_target = target / target_mass[:, None]
    denominator = normalized_target.square().sum(1)
    _require(bool((denominator > 0).all()), "Q target normalization has a zero denominator")
    return (normalized_generated - normalized_target).square().sum(1) / denominator


def _integrated_energy(raw_cmos: torch.Tensor, pixel_area_m2: float) -> torch.Tensor:
    _require(math.isfinite(pixel_area_m2) and pixel_area_m2 > 0.0, "camera pixel area must be finite and positive")
    return raw_cmos.to(torch.float64).flatten(1).sum(1) * pixel_area_m2


def _independent_metric_rows(readout: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    raw = readout["raw_cmos"].to(torch.float64)
    target_raw = readout["target_slm_intensity"].to(torch.float64)
    generated_view = readout["evaluator_view"].to(torch.float64)
    target_view = energy_task.diagnostic.canonical.cmos_view(readout["target_slm_intensity"]).to(torch.float64)
    error = (raw - target_raw).square()
    foreground = target_raw > 0.1
    background = ~foreground
    foreground_count = foreground.flatten(1).sum(1).clamp_min(1)
    background_count = background.flatten(1).sum(1).clamp_min(1)
    pixel_area, _ = energy_task._pixel_area(readout, "camera_plane")
    return {
        "brightness_normalized_stroke_error_Q": _normalized_q(generated_view, target_view),
        "cmos_integrated_intensity_joule_proxy": _integrated_energy(raw, pixel_area),
        "raw_clip_fraction_outside_0_1": ((raw < 0) | (raw > 1)).flatten(1).to(torch.float64).mean(1),
        "raw_zero_image": raw.flatten(1).abs().sum(1) == 0,
        "raw_constant_image": raw.flatten(1).std(1, unbiased=False) == 0,
        "original_target_raw_mse": error.flatten(1).mean(1),
        "original_target_raw_foreground_mse": (error * foreground).flatten(1).sum(1) / foreground_count,
        "original_target_raw_background_mse": (error * background).flatten(1).sum(1) / background_count,
        "original_target_ssim_28x28": energy_task.stroke.ssim_per_image(generated_view, target_view).flatten(),
    }


def _close_number(actual: Any, expected: Any, path: str, differences: list[dict[str, Any]]) -> None:
    _require(isinstance(actual, (float, int)) and isinstance(expected, (float, int)), f"report metric is not numeric at {path}")
    _require(math.isfinite(float(actual)) and math.isfinite(float(expected)), f"report metric is nonfinite at {path}")
    if not math.isclose(float(actual), float(expected), rel_tol=FLOAT_RTOL, abs_tol=FLOAT_ATOL):
        differences.append({"path": path, "stored": float(actual), "recomputed": float(expected)})


def _compare_tree(actual: Any, expected: Any, path: str, differences: list[dict[str, Any]]) -> None:
    if isinstance(expected, Mapping):
        _require(isinstance(actual, Mapping) and set(actual) == set(expected), f"saved report fields differ at {path}")
        for key in expected:
            _compare_tree(actual[key], expected[key], f"{path}.{key}", differences)
    elif isinstance(expected, list):
        _require(isinstance(actual, list) and len(actual) == len(expected), f"saved report list shape differs at {path}")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _compare_tree(left, right, f"{path}[{index}]", differences)
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        _require(actual == expected, f"saved report value differs at {path}")
    elif isinstance(expected, (float, int)):
        _close_number(actual, expected, path, differences)
    else:
        raise SavedEvidenceAuditError(f"unsupported report value at {path}: {type(expected).__name__}")


def _verify_stored_metrics(
    source_dir: Path,
    population: Mapping[str, Any],
    report: Mapping[str, Any],
    readouts: Mapping[str, Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
    row_values: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    differences: list[dict[str, Any]] = []
    reports = report.get("metrics")
    _require(isinstance(reports, Mapping) and set(reports) == set(ENDPOINTS), "source report endpoint set differs")
    for endpoint in ENDPOINTS:
        stored_endpoint = reports[endpoint]
        _compare_tree(stored_endpoint, summaries[endpoint], f"metrics.{endpoint}", differences)
        if endpoint in ROW_METRIC_FILES:
            saved_rows = _load_torch(source_dir / ROW_METRIC_FILES[endpoint], mmap=True)
            _require(isinstance(saved_rows, Mapping), f"{endpoint} saved row metrics are not a mapping")
            current_rows = row_values[endpoint]
            _require(set(saved_rows) == set(row_values[endpoint]) - set(TIE_KEYS), f"saved {endpoint} row metric field set differs")
            for stored_key, left in saved_rows.items():
                right = current_rows[stored_key]
                _require(torch.is_tensor(left) and left.shape == right.shape, f"saved {endpoint} row metric shape differs: {stored_key}")
                if left.dtype == torch.bool:
                    _require(torch.equal(left, right), f"saved {endpoint} row metric differs: {stored_key}")
                else:
                    _all_finite(left, f"{endpoint}.{stored_key}")
                    _require(bool(torch.allclose(left.to(torch.float64), right.to(torch.float64), rtol=FLOAT_RTOL, atol=FLOAT_ATOL)), f"saved {endpoint} row metric differs: {stored_key}")
    if differences:
        raise SavedEvidenceAuditError(f"source report summary differs from CPU replay at {differences[0]['path']}")
    decision = energy_task._final_decision(summaries["A4700"], summaries["A6700"], summaries["B6700"], valid=True)
    _compare_tree(report.get("decision"), decision, "decision", differences)
    if differences:
        raise SavedEvidenceAuditError(f"parent scientific decision does not replay from saved readouts at {differences[0]['path']}")
    _require(population["role_positions"] == {"fit": [0, FIT_ROWS], "confirmation": [FIT_ROWS, TOTAL_ROWS]}, "population changed during replay")
    return {
        "passed": True,
        "report_endpoint_summaries_replayed": True,
        "saved_A_B_row_metrics_replayed": True,
        "parent_decision_replayed_through_production_consumer": True,
        "float_tolerance": {"rtol": FLOAT_RTOL, "atol": FLOAT_ATOL},
    }


METRIC_KEYS = {
    "e_d_margin_float64": "e_d_margin_float64",
    "e_d_ce_float64": "e_d_ce_float64",
    "e_d_predictions": "e_d_predictions",
    "e_s_margin_float64": "e_s_margin_float64",
    "e_s_ce_float64": "e_s_ce_float64",
    "e_s_predictions": "e_s_predictions",
    "e_i_margin_float64": "e_i_margin_float64",
    "e_i_ce_float64": "e_i_ce_float64",
    "e_i_predictions": "e_i_predictions",
    "e_c_margin_float64": "e_c_margin_float64",
    "e_c_ce_float64": "e_c_ce_float64",
    "e_c_predictions": "e_c_predictions",
    "brightness_normalized_stroke_error_Q": "brightness_normalized_stroke_error_Q",
    "raw_mse": "original_target_raw_mse",
    "raw_foreground_mse": "original_target_raw_foreground_mse",
    "raw_background_mse": "original_target_raw_background_mse",
    "ssim": "original_target_ssim_28x28",
    "integrated_energy": "cmos_integrated_intensity_joule_proxy",
    "raw_clip_fraction": "raw_clip_fraction_outside_0_1",
    "raw_zero_image": "raw_zero_image",
    "raw_constant_image": "raw_constant_image",
}
TIE_KEYS = tuple(f"{evaluator.lower()}_max_wrong_logit_tie_count" for evaluator in EVALUATORS)
SIGN_METRICS_LOWER_BETTER = {
    "brightness_normalized_stroke_error_Q",
    "raw_mse",
    "raw_foreground_mse",
    "raw_background_mse",
    "e_d_ce_float64",
    "e_s_ce_float64",
    "e_i_ce_float64",
    "e_c_ce_float64",
    "raw_clip_fraction",
}
SIGN_METRICS_HIGHER_BETTER = {
    "e_d_margin_float64",
    "e_s_margin_float64",
    "e_i_margin_float64",
    "e_c_margin_float64",
    "ssim",
}


def _sign(delta: float, before: float, after: float) -> str:
    tolerance = SIGN_ATOL + SIGN_RTOL * max(abs(before), abs(after))
    return "increased" if delta > tolerance else "decreased" if delta < -tolerance else "stable"


def _metric_arrays(rows: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for output_name, source_name in METRIC_KEYS.items():
        value = rows[source_name].detach().cpu()
        result[output_name] = value
    for tie_key in TIE_KEYS:
        result[tie_key] = rows[tie_key].detach().cpu()
    return result


def _metric_direction(metric: str, delta: float, before: float, after: float) -> str:
    if metric == "integrated_energy":
        relative_delta = delta / before if before != 0 else math.inf
        return "improved" if relative_delta > ENERGY_RELATIVE_SIGN_TOL else "worsened" if relative_delta < -ENERGY_RELATIVE_SIGN_TOL else "stable"
    direction = _sign(delta, before, after)
    if metric in SIGN_METRICS_LOWER_BETTER:
        return "improved" if direction == "decreased" else "worsened" if direction == "increased" else "stable"
    if metric in SIGN_METRICS_HIGHER_BETTER:
        return "improved" if direction == "increased" else "worsened" if direction == "decreased" else "stable"
    raise SavedEvidenceAuditError(f"no direction is defined for metric {metric}")


def _quantile(values: Sequence[float], fraction: float) -> float:
    return float(torch.quantile(torch.as_tensor(values, dtype=torch.float64), fraction))


def _distribution(values: Sequence[float], metric: str, before: Sequence[float] | None = None) -> dict[str, Any]:
    clean = [float(value) for value in values]
    _require(bool(clean) and all(math.isfinite(value) for value in clean), f"{metric} distribution is empty or nonfinite")
    if before is None:
        counts = None
    else:
        _require(len(before) == len(clean), f"{metric} paired vectors differ in length")
        directions = [_metric_direction(metric, value, float(old), float(new)) for value, old, new in zip(clean, before, [float(a + d) for a, d in zip(before, clean, strict=True)], strict=True)]
        counts = {direction: directions.count(direction) for direction in ("improved", "worsened", "stable")}
    return {
        "rows": len(clean),
        "mean": sum(clean) / len(clean),
        "median": _quantile(clean, 0.5),
        "q25": _quantile(clean, 0.25),
        "q75": _quantile(clean, 0.75),
        "change_counts": counts,
    }


def _subset_indices(labels: torch.Tensor, roles: Sequence[str], role: str, class_id: int | None) -> list[int]:
    return [
        index
        for index, (label, row_role) in enumerate(zip(labels.tolist(), roles, strict=True))
        if row_role == role and (class_id is None or int(label) == class_id)
    ]


def _stats_for_indices(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    indices: Sequence[int],
) -> dict[str, Any]:
    result: dict[str, Any] = {"rows": len(indices)}
    for metric in METRIC_KEYS:
        left = before[metric][list(indices)]
        right = after[metric][list(indices)]
        if metric.endswith("_predictions") or metric in {"raw_zero_image", "raw_constant_image"}:
            continue
        left_values = [float(value) for value in left.tolist()]
        delta_values = [float(value) for value in (right.to(torch.float64) - left.to(torch.float64)).tolist()]
        result[metric] = {
            "before": _distribution(left_values, metric),
            "after": _distribution([float(value) for value in right.tolist()], metric),
            "delta": _distribution(delta_values, metric, left_values),
        }
    for evaluator in EVALUATORS:
        key = f"{evaluator.lower()}_predictions"
        prior = before[key]
        later = after[key]
        # Prediction transitions are interpreted against each row's true label by the caller.
        result[key] = {"rows": len(indices), "changed": int((prior[list(indices)] != later[list(indices)]).sum())}
    return result


def _correctness_transitions(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    indices: Sequence[int],
    evaluator: str = "E_D",
) -> dict[str, Any]:
    key = f"{evaluator.lower()}_predictions"
    selected = list(indices)
    truth = labels[selected]
    was_correct = before[key][selected] == truth
    now_correct = after[key][selected] == truth
    return {
        "rows": len(selected),
        "wrong_to_correct": int((~was_correct & now_correct).sum()),
        "correct_to_wrong": int((was_correct & ~now_correct).sum()),
        "correct_to_correct": int((was_correct & now_correct).sum()),
        "wrong_to_wrong": int((~was_correct & ~now_correct).sum()),
    }


def _cross_counts(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
    indices: Sequence[int],
    score_metric: str,
    shape_metric: str,
) -> dict[str, Any]:
    score_before = before[score_metric].to(torch.float64)
    score_after = after[score_metric].to(torch.float64)
    shape_before = before[shape_metric].to(torch.float64)
    shape_after = after[shape_metric].to(torch.float64)
    counts = {
        "score_improved_shape_improved": 0,
        "score_improved_shape_worsened": 0,
        "score_improved_shape_stable": 0,
        "score_worsened_shape_improved": 0,
        "score_worsened_shape_worsened": 0,
        "score_worsened_shape_stable": 0,
        "score_stable_shape_improved": 0,
        "score_stable_shape_worsened": 0,
        "score_stable_shape_stable": 0,
        "shape_improved_energy_declined": 0,
        "shape_improved_energy_stable_or_increased": 0,
    }
    energy_before = before["integrated_energy"].to(torch.float64)
    energy_after = after["integrated_energy"].to(torch.float64)
    for index in indices:
        score_direction = _metric_direction(score_metric, float(score_after[index] - score_before[index]), float(score_before[index]), float(score_after[index]))
        shape_direction = _metric_direction(shape_metric, float(shape_after[index] - shape_before[index]), float(shape_before[index]), float(shape_after[index]))
        counts[f"score_{score_direction}_shape_{shape_direction}"] += 1
        if shape_direction == "improved":
            energy_direction = _metric_direction("integrated_energy", float(energy_after[index] - energy_before[index]), float(energy_before[index]), float(energy_after[index]))
            counts["shape_improved_energy_declined" if energy_direction == "worsened" else "shape_improved_energy_stable_or_increased"] += 1
    return {"rows": len(indices), "counts": counts}


def _pair_analysis(
    before_name: str,
    after_name: str,
    arrays: Mapping[str, Mapping[str, torch.Tensor]],
    labels: torch.Tensor,
    roles: Sequence[str],
) -> dict[str, Any]:
    before, after = arrays[before_name], arrays[after_name]
    result: dict[str, Any] = {"before": before_name, "after": after_name, "direction": "after_minus_before", "by_role": {}, "by_class": {}}
    for role in ("fit", "confirmation"):
        indices = _subset_indices(labels, roles, role, None)
        result["by_role"][role] = {
            "rows": len(indices),
            "metrics": _stats_for_indices(before, after, indices),
            "E_D_correctness_transitions": _correctness_transitions(before, after, labels, indices),
            "margin_shape_energy_cross_counts": {
                shape_metric: _cross_counts(before, after, indices, "e_d_margin_float64", shape_metric)
                for shape_metric in ("brightness_normalized_stroke_error_Q", "raw_mse", "ssim")
            },
        }
        for class_id in range(10):
            class_indices = _subset_indices(labels, roles, role, class_id)
            result["by_class"].setdefault(str(class_id), {})[role] = {
                "rows": len(class_indices),
                "metrics": _stats_for_indices(before, after, class_indices),
                "E_D_correctness_transitions": _correctness_transitions(before, after, labels, class_indices),
                "margin_Q_energy_cross_counts": _cross_counts(before, after, class_indices, "e_d_margin_float64", "brightness_normalized_stroke_error_Q"),
            }
    return result


def _training_history_summary(history: Mapping[str, Any]) -> dict[str, Any]:
    return {"passed": history["passed"], "arms": history["arms"], "note": "Training history is summarized by matching 20-batch epochs. Different A/B reconstruction targets make their objective magnitudes descriptive only."}


def _interpret(pairs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    primary = pairs["A4700_to_B6700"]["by_role"]["confirmation"]
    q_cross = primary["margin_shape_energy_cross_counts"]["brightness_normalized_stroke_error_Q"]["counts"]
    score_improved = q_cross["score_improved_shape_improved"] + q_cross["score_improved_shape_worsened"] + q_cross["score_improved_shape_stable"]
    tradeoff = q_cross["score_improved_shape_worsened"]
    joint = q_cross["score_improved_shape_improved"]
    correct = primary["E_D_correctness_transitions"]
    if score_improved > 0 and tradeoff > joint:
        classification = "LOCALIZED_CLASSIFIER_MARGIN_AND_STROKE_TRADEOFF_SIGNAL"
        recommendation = "分类分数差提高而Q恶化的配对样本多于二者共同改善的样本；建议把后续冻结方向测量限定在对应类别和指标。"
    elif joint > tradeoff:
        classification = "JOINT_SCORE_AND_STROKE_IMPROVEMENT_WITH_POSSIBLE_ENERGY_TRADEOFF"
        recommendation = "分类分数差与Q在配对样本上以共同改善为主，广泛的分类—笔画反向竞争解释减弱；下一问题可聚焦归一化形态与实际输出能量的取舍。"
    else:
        classification = "MIXED_OR_NO_DOMINANT_ROW_LEVEL_PATTERN"
        recommendation = "现有配对读数没有显示单一主导关系；不据此改变全局目标，后续问题应收窄到关系最清楚的类别和指标。"
    return {
        "classification": classification,
        "primary_comparison": "A4700_to_B6700 confirmation rows",
        "E_D_margin_improved_rows": score_improved,
        "among_margin_improved_Q_improved_rows": joint,
        "among_margin_improved_Q_worsened_rows": tradeoff,
        "E_D_correctness_transitions": correct,
        "interpretation": recommendation,
        "rule_note": "只按配对计数描述关系；不构成模型采用门槛、因果结论或新增科学Gate。",
        "training_recommendation": "NONE",
        "next_action": "NONE",
    }


CSV_BASE_FIELDS = ["population_position", "source_row_index", "role", "true_class"]
CSV_METRIC_FIELDS = list(METRIC_KEYS)


def _write_paired_rows(
    path: Path,
    readouts: Mapping[str, Mapping[str, Any]],
    arrays: Mapping[str, Mapping[str, torch.Tensor]],
    roles: Sequence[str],
) -> None:
    fields = list(CSV_BASE_FIELDS)
    for endpoint in ENDPOINTS:
        fields.extend(f"{endpoint}_{metric}" for metric in CSV_METRIC_FIELDS)
        fields.extend(f"{endpoint}_{tie_key}" for tie_key in TIE_KEYS)
    for before, after in PAIRS:
        pair = f"{before}_to_{after}"
        fields.append(f"{pair}_e_d_correctness_transition")
        fields.extend(f"{pair}_delta_{metric}" for metric in CSV_METRIC_FIELDS if not metric.endswith("_predictions") and metric not in {"raw_zero_image", "raw_constant_image"})
        fields.append(f"{pair}_e_d_margin_direction")
        for shape_metric in ("brightness_normalized_stroke_error_Q", "raw_mse", "ssim"):
            fields.append(f"{pair}_e_d_margin_vs_{shape_metric}_pattern")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        base = readouts["A4700"]
        label_tensor = base["labels"].to(torch.long)
        for index in range(TOTAL_ROWS):
            record: dict[str, Any] = {
                "population_position": index,
                "source_row_index": int(base["source_indices"][index]),
                "role": roles[index],
                "true_class": int(label_tensor[index]),
            }
            for endpoint in ENDPOINTS:
                for metric in CSV_METRIC_FIELDS:
                    value = arrays[endpoint][metric][index]
                    record[f"{endpoint}_{metric}"] = int(value) if metric.endswith("_predictions") or metric in {"raw_zero_image", "raw_constant_image"} else float(value)
                for tie_key in TIE_KEYS:
                    record[f"{endpoint}_{tie_key}"] = int(arrays[endpoint][tie_key][index])
            for before_name, after_name in PAIRS:
                before, after = arrays[before_name], arrays[after_name]
                prefix = f"{before_name}_to_{after_name}"
                prior_correct = int(before["e_d_predictions"][index]) == int(label_tensor[index])
                later_correct = int(after["e_d_predictions"][index]) == int(label_tensor[index])
                transition = "wrong_to_correct" if not prior_correct and later_correct else "correct_to_wrong" if prior_correct and not later_correct else "correct_to_correct" if prior_correct else "wrong_to_wrong"
                record[f"{prefix}_e_d_correctness_transition"] = transition
                for metric in CSV_METRIC_FIELDS:
                    if metric.endswith("_predictions") or metric in {"raw_zero_image", "raw_constant_image"}:
                        continue
                    record[f"{prefix}_delta_{metric}"] = float(after[metric][index]) - float(before[metric][index])
                margin_delta = float(after["e_d_margin_float64"][index]) - float(before["e_d_margin_float64"][index])
                record[f"{prefix}_e_d_margin_direction"] = _metric_direction("e_d_margin_float64", margin_delta, float(before["e_d_margin_float64"][index]), float(after["e_d_margin_float64"][index]))
                for shape_metric in ("brightness_normalized_stroke_error_Q", "raw_mse", "ssim"):
                    shape_delta = float(after[shape_metric][index]) - float(before[shape_metric][index])
                    record[f"{prefix}_e_d_margin_vs_{shape_metric}_pattern"] = f"{record[f'{prefix}_e_d_margin_direction']}_{_metric_direction(shape_metric, shape_delta, float(before[shape_metric][index]), float(after[shape_metric][index]))}"
            writer.writerow(record)


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    decision = summary["decision"]
    confirm_rows = []
    for pair_name, pair in summary["comparisons"].items():
        data = pair["by_role"]["confirmation"]
        transition = data["E_D_correctness_transitions"]
        q_counts = data["margin_shape_energy_cross_counts"]["brightness_normalized_stroke_error_Q"]["counts"]
        q_stats = data["metrics"]["brightness_normalized_stroke_error_Q"]["delta"]
        energy = data["metrics"]["integrated_energy"]["delta"]
        q_energy = data["margin_shape_energy_cross_counts"]["brightness_normalized_stroke_error_Q"]["counts"]
        q_improved = sum(q_energy[f"score_{direction}_shape_improved"] for direction in ("improved", "worsened", "stable"))
        confirm_rows.append(
            f"| {pair_name} | {transition['wrong_to_correct']} / {transition['correct_to_wrong']} | "
            f"{q_counts['score_improved_shape_improved']} / {q_counts['score_improved_shape_worsened']} | "
            f"{q_stats['median']:.6g} | {q_energy['shape_improved_energy_declined']} / {q_improved} | "
            f"{energy['median']:.6g} |"
        )
    class_lines = []
    primary = summary["comparisons"]["A4700_to_B6700"]["by_class"]
    for class_id in range(10):
        data = primary[str(class_id)]["confirmation"]
        transition = data["E_D_correctness_transitions"]
        cross = data["margin_Q_energy_cross_counts"]["counts"]
        class_lines.append(
            f"| {class_id} | {transition['wrong_to_correct']} / {transition['correct_to_wrong']} | "
            f"{cross['score_improved_shape_improved']} / {cross['score_improved_shape_worsened']} | "
            f"{data['metrics']['brightness_normalized_stroke_error_Q']['delta']['median']:.6g} |"
        )
    return "\n".join(
        (
            f"# {TASK_ID}",
            "",
            "## 结果与状态",
            "保存证据CPU复盘有效完成。模型更新、光学前向、评价器前向、梯度、采样和人工评分均为0。",
            "",
            "## 科学认识",
            f"判读：**{decision['classification']}**。{decision['interpretation']}",
            "",
            "确认人口配对变化；分类列为 E_D 错→对 / 对→错，形态列为 E_D 分数差上升且Q改善 / 恶化。能量中位差以焦耳代理单位计。",
            "",
            "| 比较 | E_D正确性转移 | 分数差与Q：改善 / 恶化 | Q中位差 | Q改善且能量下降 / Q改善行数 | 能量中位差 |",
            "|---|---:|---:|---:|---:|---:|",
            *confirm_rows,
            "",
            "A4700→B6700确认人口按类结果（E_D分数差上升且Q改善 / 恶化）：",
            "",
            "| 类别 | E_D错→对 / 对→错 | 分数差与Q：改善 / 恶化 | Q中位差 |",
            "|---:|---:|---:|---:|",
            *class_lines,
            "",
            "类别2、4用于重点形态定位，类别5、6用于观察分类收益来源；拟合人口200行结果见 `summary.json`，不与300行确认人口合并。E_S、E_I、E_C分开统计。",
            "",
            "## 目标缺口",
            "该复盘只说明固定端点和TRAIN人口的配对读数关系。它不能建立梯度因果关系、人眼可读性、独立泛化或采样生成收益。",
            "",
            "## 工程完成与验证",
            "保存张量身份、人口分区、物理坐标、指标重算、逐行指标和父报告决策均已校验。A/B训练历史按相同的20批次轮次核对并汇总；两个目标臂的重建损失分别呈现，不直接比较其绝对值。",
            "",
            "## 决策影响与下一步",
            f"建议：{decision['interpretation']}",
            "该建议用于判断是否另行设计冻结方向测量；不采用模型，不授权训练或其他后继实验。`training_recommendation=NONE`，`next_action=NONE`。",
            "",
            f"判读规则：{decision['rule_note']}",
            "",
        )
    )


def _run_audit(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    _require(source_dir.is_dir(), f"source attempt directory is missing: {source_dir}")
    _require(not output_dir.exists(), f"refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    validation: dict[str, Any] = {
        "schema": "m0_generator_saved_evidence_cpu_review_validation.v1",
        "task_id": TASK_ID,
        "source_directory": str(source_dir),
        "output_directory": str(output_dir),
        "execution": {"device": "cpu", "model_updates": 0, "optical_forwards": 0, "evaluator_forwards": 0, "gradients": 0, "sampling": 0, "human_ratings": 0},
        "checks": {},
    }
    try:
        population = _load_json(source_dir / "input_population.json")
        source_report = _load_json(source_dir / "report.json")
        validation["checks"].update(_verify_attempt(population, source_report))
        readouts = {endpoint: _load_torch(source_dir / filename, mmap=True) for endpoint, filename in ENDPOINTS.items()}
        validation["checks"].update(_verify_readouts(population, readouts))
        summaries: dict[str, dict[str, Any]] = {}
        rows: dict[str, dict[str, torch.Tensor]] = {}
        for endpoint, readout in readouts.items():
            summaries[endpoint], rows[endpoint] = _derive_readout_metrics(readout, population, endpoint)
        validation["checks"]["endpoint_metrics_recomputed"] = {name: True for name in ENDPOINTS}
        validation["checks"]["physical_coordinate_metadata"] = {
            "camera_pixel_area_m2": energy_task._pixel_area(readouts["A4700"], "camera_plane")[0],
            "target_pixel_area_m2": energy_task._pixel_area(readouts["A4700"], "input_slm_field")[0],
        }
        validation["checks"]["report_and_saved_row_metrics_replay"] = _verify_stored_metrics(source_dir, population, source_report, readouts, summaries, rows)
        history = _validate_history(source_dir, population)
        validation["checks"]["training_history_alignment"] = {"passed": history["passed"], "records_per_arm": UPDATES_PER_ARM, "batches_per_epoch": BATCH_COUNT, "epochs_per_arm": UPDATES_PER_ARM // BATCH_COUNT}
        arrays = {endpoint: _metric_arrays(rows[endpoint]) for endpoint in ENDPOINTS}
        labels = readouts["A4700"]["labels"].to(torch.long).cpu()
        roles = list(population["roles"])
        comparisons = {
            f"{before}_to_{after}": _pair_analysis(before, after, arrays, labels, roles)
            for before, after in PAIRS
        }
        decision = _interpret(comparisons)
        paired_path = output_dir / "paired_rows.csv"
        _write_paired_rows(paired_path, readouts, arrays, roles)
        result = {
            "schema": "m0_generator_saved_evidence_cpu_review_summary.v1",
            "task_id": TASK_ID,
            "status": "EVIDENCE_REVIEW_COMPLETED",
            "source_task_id": SOURCE_TASK_ID,
            "source_classification": source_report["classification"],
            "population": {"total_rows": TOTAL_ROWS, "fit_rows": FIT_ROWS, "confirmation_rows": CONFIRM_ROWS, "per_class_fit": 20, "per_class_confirmation": 30, "all_rows_are_canonical_train": True},
            "comparisons": comparisons,
            "training_history": _training_history_summary(history),
            "paired_rows_csv": paired_path.name,
            "decision": decision,
            "scientific_scope": "Paired endpoint associations on the saved canonical TRAIN population; no causal gradient or readability claim.",
        }
        _write_json(output_dir / "summary.json", result)
        validation["status"] = "VALIDATED"
        validation["checks"]["complete_paired_population_and_class_coverage"] = True
        _write_json(output_dir / "validation.json", validation)
        (output_dir / "report.md").write_text(_summary_markdown(result), encoding="utf-8")
        return result
    except (SavedEvidenceAuditError, energy_task.EnergyTargetError, OSError, json.JSONDecodeError) as error:
        validation["status"] = "INVALID_EVIDENCE"
        validation["stop_reason"] = f"{type(error).__name__}: {error}"
        _write_json(output_dir / "validation.json", validation)
        (output_dir / "report.md").write_text(
            "\n".join((f"# {TASK_ID}", "", "## 结果与状态", "测量无效；没有科学结论。", "", "## 科学认识", "未判定。", "", "## 目标缺口", f"证据核验停止：{validation['stop_reason']}", "", "## 工程完成与验证", "未生成配对指标或汇总。", "", "## 决策影响与下一步", "修复最小证据问题后，在新的空 attempt 目录重跑；不自动启动科学后继任务。", "")),
            encoding="utf-8",
        )
        return {"status": "INVALID_EVIDENCE", "stop_reason": validation["stop_reason"]}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path, help="Explicit completed attempt_03 evidence directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="New versioned outputs/YYYYMMDD_<task>/runs/attempt_<n> directory")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    root = Path(__file__).resolve().parents[3]
    source_dir = _resolve(root, args.artifact_dir)
    output_dir = _resolve(root, args.output_dir)
    outputs = (root / "outputs").resolve()
    _require(outputs in output_dir.parents, "output must be a child of outputs/")
    relative = output_dir.relative_to(outputs)
    _require(len(relative.parts) == 3 and relative.parts[1] == "runs" and relative.parts[0].startswith("20261009_"), "output must be a versioned outputs/YYYYMMDD_<task>/runs/attempt_<n> path")
    _require(relative.parts[0] == "20261009_m0_generator_saved_evidence_cpu_review_v1", "output task directory differs from this audit task")
    _require(relative.parts[2].startswith("attempt_") and relative.parts[2][8:].isdigit(), "output attempt must have an explicit numeric suffix")
    _require(not any(part in {"final_pt", "_wsl_restore_metadata_v4", ".tmp"} for part in relative.parts), "protected, restored, or temporary output namespace is forbidden")
    result = _run_audit(source_dir, output_dir)
    print(json.dumps({"status": result.get("status"), "decision": result.get("decision"), "output_directory": str(output_dir)}, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
    if result.get("status") == "INVALID_EVIDENCE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
