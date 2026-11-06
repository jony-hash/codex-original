"""Audit saved class-9 visibility evidence and prepare a blinded review cohort."""

from __future__ import annotations

import argparse
import base64
import csv
from dataclasses import dataclass
import html
import io
import json
import math
import os
from pathlib import Path
import time
import uuid
from typing import Any

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_SMALL_CLASS9_SAVED_DEFECT_DIAGNOSIS_V1"
ATTRIBUTION_TASK_ID = "LIGHTGEN_SMALL_CMOS_VISIBILITY_ATTRIBUTION_V1"
AB_TASK_ID = "LIGHTGEN_SMALL_GAUSSIAN_STRUCTURAL_CONTINUATION_AB_V1"
CELLS = ("AA", "BA", "AB", "BB")
CELL_INPUT = {"AA": "A", "BA": "A", "AB": "B", "BB": "B"}
CELL_DECODER = {"AA": "A", "BA": "B", "AB": "A", "BB": "B"}
CONDITIONS = (
    "AA_g0p85", "AA_g1p00", "AA_g1p15", "BA_g1p00",
    "AB_g1p00", "BB_g0p85", "BB_g1p00", "BB_g1p15",
)
SAMPLES_PER_CLASS = 1_000
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 2026102102
SELECTION_SEED = 2026102103
SHUFFLE_SEED = 2026102104
REPLAY_RTOL = 2.0e-6
REPLAY_ATOL = 5.0e-8
RATING_LABELS = {"unreadable", *map(str, range(10))}
EXPECTED_MODEL = {
    "encoder": [2, 100, 100],
    "ols": "gaussian_s12_v1",
    "ols_geometry": {
        "port_pitch_m": 20.0e-6,
        "field_1e_radius_m": 3.5e-6,
        "transport": "identity",
    },
    "decoder": [2, 100, 100],
}
STRATA = (
    ("requested9_ed4", 9, 4, 5),
    ("requested9_ed8", 9, 8, 5),
    ("requested9_ed9", 9, 9, 3),
    ("requested4_unfiltered", 4, None, 1),
    ("requested8_unfiltered", 8, None, 1),
)
ARTIFACT_FILES = (
    "preflight.json", "per_class9_condition_metrics.csv", "transition_tables.json",
    "coefficient_associations.json", "spatial_decoder_changes.json", "bootstrap.json",
    "native_readout_check.json", "blind_review_manifest.json", "blind_review.html",
    "report.json", "report.html", "unblinded_comparison.html", "closeout_report.md",
    "validation_receipt.json",
)


class EvidenceError(RuntimeError):
    """An input or saved measurement failed its declared contract."""


@dataclass
class Inputs:
    attribution_dir: Path
    ab_evidence_dir: Path
    attribution_report: dict[str, Any]
    ab_report: dict[str, Any]
    source_cohorts: dict[str, dict[str, Any]]
    checkpoints: dict[str, dict[str, Any]]
    coefficient_rows: dict[str, np.ndarray]
    blocks: dict[tuple[str, int], dict[str, Any]]
    conditions: dict[str, dict[str, np.ndarray]]
    cohort_images: dict[tuple[str, int], np.ndarray]
    proposal_ids: np.ndarray
    camera_gain: float
    preflight: dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required JSON evidence is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), f"JSON evidence must be an object: {path}")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialize evidence value of type {type(value).__name__}")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _finite_tensor(value: Any, name: str, shape: tuple[int, ...], *, complex_value: bool = False) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape,
             f"{name} has an unexpected shape")
    _require(value.is_complex() is complex_value, f"{name} has an unexpected numeric kind")
    if value.is_complex():
        finite = torch.isfinite(value.real).all() & torch.isfinite(value.imag).all()
    else:
        finite = torch.isfinite(value).all()
    _require(bool(finite), f"{name} contains nonfinite values")
    return value


def _validate_checkpoint(checkpoint: dict[str, Any], branch: str, ab_run_id: str,
                         source_run_id: str) -> None:
    checkpoint_branch = "control" if branch == "A" else "treatment"
    _require(checkpoint.get("task_id") == AB_TASK_ID
             and checkpoint.get("branch") == checkpoint_branch,
             f"{branch} checkpoint task or branch identity differs")
    _require(checkpoint.get("source_checkpoint_step") == 30_000
             and checkpoint.get("completed_updates") == 5_000
             and checkpoint.get("completed_global_step") == 35_000,
             f"{branch} checkpoint is not the explicit 35,000-step endpoint")
    _require(checkpoint.get("source_run_id") == source_run_id
             and checkpoint.get("source_checkpoint_step") == 30_000
             and checkpoint.get("run_id") == ab_run_id
             and checkpoint.get("model_identity") == EXPECTED_MODEL,
             f"{branch} endpoint source or Gaussian model identity differs")
    gain = checkpoint.get("camera_gain")
    _require(isinstance(gain, torch.Tensor) and gain.shape == ()
             and torch.isfinite(gain).item() and float(gain) > 0.0,
             f"{branch} frozen camera gain is missing or invalid")


def _validate_cohort(cohort: dict[str, Any], branch: str, run_id: str) -> None:
    expected = {
        "images": ((10_000, 1, 28, 28), False),
        "requested_labels": ((10_000,), False),
        "predicted_labels": ((10_000,), False),
        "proposal_ids": ((10_000,), False),
        "ols_coefficients": ((10_000, 100), True),
        "proxy_clipping_fraction": ((10_000,), False),
    }
    for name, (shape, complex_value) in expected.items():
        _finite_tensor(cohort.get(name), f"{branch} cohort {name}", shape,
                       complex_value=complex_value)
    branch_name = "control" if branch == "A" else "treatment"
    _require(cohort.get("task_id") == AB_TASK_ID and cohort.get("branch") == branch_name
             and cohort.get("run_id") == run_id and cohort.get("global_step") == 35_000,
             f"{branch} cohort task, branch, run or step identity differs")
    expected_labels = torch.arange(10).repeat_interleave(SAMPLES_PER_CLASS)
    expected_proposals = torch.arange(SAMPLES_PER_CLASS).repeat(10)
    _require(torch.equal(cohort["requested_labels"].long(), expected_labels),
             f"{branch} cohort has missing or reordered requested labels")
    _require(torch.equal(cohort["proposal_ids"].long(), expected_proposals),
             f"{branch} cohort proposal IDs are missing, duplicated or reordered")
    _require(cohort["ols_coefficients"].dtype == torch.complex64
             and cohort["images"].dtype == torch.float32,
             f"{branch} cohort dtype differs from its saved production representation")


def _condition_names(cell: str) -> tuple[str, ...]:
    return ((f"{cell}_g0p85", f"{cell}_g1p00", f"{cell}_g1p15")
            if cell in ("AA", "BB") else (f"{cell}_g1p00",))


def _validate_block(block: Any, cell: str, label: int, expected_run_id: str) -> None:
    _require(isinstance(block, dict), f"{cell}/class_{label:02d} block is not an object")
    identity = block.get("identity")
    _require(isinstance(identity, dict), f"{cell}/class_{label:02d} block lacks identity")
    _require(identity.get("schema") == "lightgen_small_cmos_visibility_block.v1"
             and identity.get("task_id") == ATTRIBUTION_TASK_ID
             and identity.get("cell") == cell
             and identity.get("input_source") == CELL_INPUT[cell]
             and identity.get("decoder") == CELL_DECODER[cell]
             and identity.get("class_id") == label
             and identity.get("sample_count") == SAMPLES_PER_CLASS
             and identity.get("input_run_id") == expected_run_id
             and identity.get("decoder_run_id") == expected_run_id,
             f"{cell}/class_{label:02d} block row or branch identity differs")
    requested = _finite_tensor(block.get("requested_labels"), f"{cell}/class_{label:02d} requested labels",
                               (SAMPLES_PER_CLASS,)).long()
    _require(torch.equal(requested, torch.full((SAMPLES_PER_CLASS,), label)),
             f"{cell}/class_{label:02d} requested labels are missing or misaligned")
    proposals = _finite_tensor(block.get("proposal_ids"), f"{cell}/class_{label:02d} proposal IDs",
                               (SAMPLES_PER_CLASS,)).long()
    _require(torch.equal(proposals, torch.arange(SAMPLES_PER_CLASS)),
             f"{cell}/class_{label:02d} proposal IDs are missing, duplicated or reordered")
    conditions = block.get("conditions")
    _require(isinstance(conditions, dict) and set(conditions) == set(_condition_names(cell)),
             f"{cell}/class_{label:02d} conditions are incomplete or unexpected")
    for name, values in conditions.items():
        _require(isinstance(values, dict), f"{name} values are not a mapping")
        _finite_tensor(values.get("images"), f"{name} images", (SAMPLES_PER_CLASS, 1, 28, 28))
        _finite_tensor(values.get("logits"), f"{name} logits", (SAMPLES_PER_CLASS, 10))
        _finite_tensor(values.get("predicted_labels"), f"{name} predicted labels", (SAMPLES_PER_CLASS,))
        _finite_tensor(values.get("identity_margin"), f"{name} identity margin", (SAMPLES_PER_CLASS,))
        for field in ("view_mean", "camera_total_power_w", "peak_w_per_pixel", "scaled_peak",
                      "clipping_area_fraction", "clipping_region_power_share", "excess_removed_power_share"):
            tensor = values.get(field)
            _require(isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == (SAMPLES_PER_CLASS,),
                     f"{name} {field} has an unexpected shape")
            if field in ("clipping_region_power_share", "excess_removed_power_share"):
                _require(bool((torch.isfinite(tensor) | torch.isnan(tensor)).all()),
                         f"{name} {field} has an invalid nonfinite value")
            else:
                _require(bool(torch.isfinite(tensor).all()), f"{name} {field} contains nonfinite values")
    if label == 9:
        _finite_tensor(block.get("native_raw_examples"), f"{cell} native raw examples",
                       (10, 1, 200, 200))


def _numpy_adaptive_average_pool(image: np.ndarray, output_size: int = 28) -> np.ndarray:
    _require(image.shape == (200, 200), "NumPy readout reference requires a 200x200 image")
    result = np.empty((output_size, output_size), dtype=np.float64)
    for y in range(output_size):
        y0 = math.floor(y * 200 / output_size)
        y1 = math.ceil((y + 1) * 200 / output_size)
        for x in range(output_size):
            x0 = math.floor(x * 200 / output_size)
            x1 = math.ceil((x + 1) * 200 / output_size)
            result[y, x] = float(image[y0:y1, x0:x1].mean(dtype=np.float64))
    return result


def _numpy_production_readout(raw: np.ndarray, gain: float) -> np.ndarray:
    _require(raw.shape == (200, 200), "native camera array must be 200x200")
    scaled = raw.astype(np.float32, copy=False) * np.float32(gain)
    clipped = np.clip(scaled, np.float32(0.0), np.float32(1.0))
    return _numpy_adaptive_average_pool(clipped)


def _read_block_conditions(block: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for name, values in block["conditions"].items():
        result[name] = {
            field: values[field].detach().cpu().numpy()
            for field in ("images", "logits", "predicted_labels", "identity_margin", "view_mean",
                          "camera_total_power_w", "peak_w_per_pixel", "scaled_peak",
                          "clipping_area_fraction", "clipping_region_power_share", "excess_removed_power_share")
        }
    return result


def _recompute_predictions(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _require(logits.ndim == 2 and logits.shape == (len(labels), 10), "logits do not align to requested labels")
    predictions = np.argmax(logits, axis=1)
    rows = np.arange(len(labels))
    competitors = logits.copy()
    competitors[rows, labels] = -np.inf
    margins = logits[rows, labels] - np.max(competitors, axis=1)
    return predictions.astype(np.int64), margins.astype(np.float64), predictions == labels


def _load_inputs(attribution_dir: Path, ab_evidence_dir: Path) -> Inputs:
    attr = _read_json(attribution_dir / "final_report.json")
    attr_receipt = _read_json(attribution_dir / "local_verification_receipt.json")
    _require(attr.get("task_id") == ATTRIBUTION_TASK_ID and attr.get("status") == "COMPLETE_DIAGNOSTIC"
             and attr.get("measurement_complete") is True,
             "attribution package is not a complete fixed-evidence diagnostic")
    _require(attr_receipt.get("status") == "PASSED"
             and attr_receipt.get("transfer_status") == "COMPLETE"
             and attr_receipt.get("validated_blocks") == 40,
             "attribution package local verification is incomplete")
    attr_inputs = attr.get("inputs", {})
    _require(attr_inputs.get("model_identity") == EXPECTED_MODEL
             and attr_inputs.get("proposal_count_per_cell") == 10_000,
             "attribution package model or population identity differs")
    camera_gain = attr_inputs.get("camera_gain")
    _require(isinstance(camera_gain, (int, float)) and math.isfinite(camera_gain) and camera_gain > 0,
             "attribution package frozen camera gain is invalid")

    ab_report = _read_json(ab_evidence_dir / "report.json")
    run_identity = _read_json(ab_evidence_dir / "run_identity.json")
    _require(ab_report.get("task_id") == AB_TASK_ID and ab_report.get("measurement_complete") is True
             and ab_report.get("status") == "VALID_FAILED",
             "A/B continuation report is incomplete or its historical status changed")
    run_id = ab_report.get("run_id")
    source_run_id = ab_report.get("source", {}).get("run_id")
    _require(isinstance(run_id, str) and isinstance(source_run_id, str)
             and run_identity.get("run_id") == run_id
             and run_identity.get("source_run_id") == source_run_id,
             "A/B run identity does not match the explicit continuation report")

    source_cohorts: dict[str, dict[str, Any]] = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    cohort_images: dict[tuple[str, int], np.ndarray] = {}
    coefficient_rows: dict[str, np.ndarray] = {}
    for branch, branch_dir_name in (("A", "control"), ("B", "treatment")):
        branch_dir = ab_evidence_dir / branch_dir_name
        checkpoint_path = branch_dir / "checkpoint_step_005000.pt"
        cohort_path = branch_dir / "generated_cohort.pt"
        _require(checkpoint_path.is_file() and cohort_path.is_file(),
                 f"explicit {branch} checkpoint or generated cohort is missing")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cohort = torch.load(cohort_path, map_location="cpu", weights_only=False)
        _require(isinstance(checkpoint, dict) and isinstance(cohort, dict),
                 f"{branch} checkpoint and cohort must be mappings")
        _validate_checkpoint(checkpoint, branch, run_id, source_run_id)
        _validate_cohort(cohort, branch, run_id)
        _require(math.isclose(float(checkpoint["camera_gain"]), float(camera_gain), rel_tol=0.0, abs_tol=0.0),
                 f"{branch} checkpoint camera gain differs from the frozen attribution gain")
        source_cohorts[branch] = cohort
        checkpoints[branch] = checkpoint
        coefficient_rows[branch] = cohort["ols_coefficients"][9 * SAMPLES_PER_CLASS:10 * SAMPLES_PER_CLASS].numpy().copy()
        cohort_images[(branch, 9)] = cohort["images"][9 * SAMPLES_PER_CLASS:10 * SAMPLES_PER_CLASS].numpy().copy()
    for key in ("requested_labels", "proposal_ids"):
        _require(torch.equal(source_cohorts["A"][key], source_cohorts["B"][key]),
                 f"A/B fixed cohorts do not share {key}")
    _require(torch.equal(checkpoints["A"]["camera_gain"], checkpoints["B"]["camera_gain"]),
             "A/B checkpoints do not share the original frozen camera gain")
    proposal_ids = source_cohorts["A"]["proposal_ids"][9 * SAMPLES_PER_CLASS:].numpy().copy()

    blocks: dict[tuple[str, int], dict[str, Any]] = {}
    conditions: dict[str, dict[str, np.ndarray]] = {}
    for cell in CELLS:
        expected_run_id = attr_inputs["A"]["run_id"]
        _require(attr_inputs["B"]["run_id"] == expected_run_id,
                 "attribution endpoint run IDs differ across fixed combinations")
        for label in (9, 4, 8):
            path = attribution_dir / "blocks" / cell / f"class_{label:02d}.pt"
            _require(path.is_file(), f"required saved {cell}/class_{label:02d} block is missing")
            block = torch.load(path, map_location="cpu", weights_only=False)
            _validate_block(block, cell, label, expected_run_id)
            blocks[(cell, label)] = block
            if label == 9:
                conditions.update(_read_block_conditions(block))
                source = CELL_INPUT[cell]
                cohort_images[(cell, 9)] = block["conditions"][f"{cell}_g1p00"]["images"].numpy().copy()
                _require(np.array_equal(block["proposal_ids"].numpy(), proposal_ids),
                         f"{cell} saved class-9 proposal IDs do not align to the A/B cohort")
                if cell in ("AA", "BB"):
                    _require(np.allclose(cohort_images[(cell, 9)], cohort_images[(source, 9)],
                                         rtol=REPLAY_RTOL, atol=REPLAY_ATOL),
                             f"{cell} original gain view differs from its saved generated cohort")
    _require(set(conditions) == set(CONDITIONS), "the eight class-9 condition records are incomplete")
    _validate_metrics_csv(attribution_dir / "per_image_metrics.csv", conditions, proposal_ids)
    _validate_report_summaries(attr, conditions)
    native_readout = _validate_native_readout(blocks, float(camera_gain))

    preflight = {
        "status": "PASSED",
        "attribution_task_id": attr["task_id"],
        "attribution_status": attr["status"],
        "ab_task_id": ab_report["task_id"],
        "ab_status_preserved": ab_report["status"],
        "A_global_step": checkpoints["A"]["completed_global_step"],
        "B_global_step": checkpoints["B"]["completed_global_step"],
        "shared_source_run_id": source_run_id,
        "shared_source_step": checkpoints["A"]["source_checkpoint_step"],
        "model_identity": EXPECTED_MODEL,
        "camera_gain": float(camera_gain),
        "class9_proposals": int(len(proposal_ids)),
        "condition_rows_recomputed": len(CONDITIONS) * SAMPLES_PER_CLASS,
        "csv_rows_checked": len(CONDITIONS) * SAMPLES_PER_CLASS,
        "report_summaries_checked": len(CONDITIONS),
        "native_readout": native_readout,
        "executed_model_construction": 0,
        "executed_optical_forward": 0,
        "executed_encoder_forward": 0,
        "executed_evaluator_forward": 0,
        "executed_sampling": 0,
        "executed_gradient": 0,
        "executed_optimizer": 0,
        "new_train_dev_test_reads": 0,
    }
    return Inputs(
        attribution_dir=attribution_dir,
        ab_evidence_dir=ab_evidence_dir,
        attribution_report=attr,
        ab_report=ab_report,
        source_cohorts=source_cohorts,
        checkpoints=checkpoints,
        coefficient_rows=coefficient_rows,
        blocks=blocks,
        conditions=conditions,
        cohort_images=cohort_images,
        proposal_ids=proposal_ids,
        camera_gain=float(camera_gain),
        preflight=preflight,
    )


def _validate_metrics_csv(path: Path, conditions: dict[str, dict[str, np.ndarray]],
                          proposal_ids: np.ndarray) -> None:
    _require(path.is_file(), f"the original attribution per-image CSV is missing: {path}")
    expected_fields = {
        "condition_id", "cell", "input_source", "decoder", "class", "proposal_id",
        "predicted_label", "correct", "identity_margin", "view_mean", "camera_total_power_w",
        "peak_w_per_pixel", "scaled_peak", "clipping_area_fraction",
        "clipping_region_power_share", "excess_removed_power_share",
    }
    seen: set[tuple[str, int]] = set()
    checked = 0
    numeric_fields = (
        "identity_margin", "view_mean", "camera_total_power_w", "peak_w_per_pixel", "scaled_peak",
        "clipping_area_fraction", "clipping_region_power_share", "excess_removed_power_share",
    )
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(reader.fieldnames is not None and expected_fields.issubset(reader.fieldnames),
                 "original per-image CSV is missing required fields")
        for row in reader:
            if row.get("class") != "9":
                continue
            condition_id = row.get("condition_id", "")
            _require(condition_id in conditions, f"original per-image CSV contains unexpected class-9 condition {condition_id}")
            try:
                proposal = int(row["proposal_id"])
            except (TypeError, ValueError) as error:
                raise EvidenceError("original per-image CSV has an invalid proposal ID") from error
            key = (condition_id, proposal)
            _require(key not in seen, f"original per-image CSV repeats row {key}")
            seen.add(key)
            index = proposal
            _require(0 <= index < SAMPLES_PER_CLASS and int(proposal_ids[index]) == proposal,
                     f"original per-image CSV proposal is out of range: {key}")
            values = conditions[condition_id]
            predictions, margins, correctness = _recompute_predictions(
                values["logits"], np.full(SAMPLES_PER_CLASS, 9, dtype=np.int64)
            )
            _require(int(row["predicted_label"]) == int(predictions[index]),
                     f"saved CSV prediction differs from recomputed logits at {key}")
            _require(row["correct"].strip().lower() == str(bool(correctness[index])).lower(),
                     f"saved CSV correctness differs from recomputed logits at {key}")
            expected = {
                "identity_margin": margins[index],
                **{field: values[field][index] for field in numeric_fields if field != "identity_margin"},
            }
            for field, expected_value in expected.items():
                parsed = row[field].strip()
                if not math.isfinite(float(expected_value)):
                    _require(parsed == "" or parsed.lower() == "nan",
                             f"undefined saved {field} should remain null-like at {key}")
                    continue
                _require(math.isclose(float(parsed), float(expected_value), rel_tol=2e-6, abs_tol=5e-10),
                         f"saved CSV {field} differs from the production block at {key}")
            checked += 1
    _require(len(seen) == len(CONDITIONS) * SAMPLES_PER_CLASS
             and checked == len(CONDITIONS) * SAMPLES_PER_CLASS,
             "original per-image CSV is missing one or more class-9 condition rows")


def _validate_report_summaries(report: dict[str, Any], conditions: dict[str, dict[str, np.ndarray]]) -> None:
    summaries = report.get("results", {}).get("conditions", {})
    _require(isinstance(summaries, dict), "original attribution report has no condition summaries")
    for name in CONDITIONS:
        summary = summaries.get(name)
        _require(isinstance(summary, dict), f"original report lacks condition {name}")
        values = conditions[name]
        predictions, margins, correct = _recompute_predictions(
            values["logits"], np.full(SAMPLES_PER_CLASS, 9, dtype=np.int64)
        )
        recall = float(correct.mean())
        _require(math.isclose(float(summary["recall_by_class"][9]), recall, rel_tol=0.0, abs_tol=1e-12),
                 f"original report class-9 recall differs from recomputed logits for {name}")
        _require(math.isclose(float(summary["identity_margin_by_class"][9]),
                              float(margins.mean()), rel_tol=2e-8, abs_tol=2e-9),
                 f"original report class-9 identity margin differs from recomputed logits for {name}")
        clipping = float(np.mean(values["clipping_area_fraction"]))
        _require(math.isclose(float(summary["clipping_area_fraction_by_class"][9]),
                              clipping, rel_tol=2e-8, abs_tol=2e-10),
                 f"original report class-9 clipping differs from saved block for {name}")
        _require(np.array_equal(predictions, values["predicted_labels"].astype(np.int64)),
                 f"saved producer predictions differ from logits for {name}")


def _validate_native_readout(blocks: dict[tuple[str, int], dict[str, Any]], gain: float) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for cell in CELLS:
        block = blocks[(cell, 9)]
        raw = block["native_raw_examples"].detach().cpu().numpy()
        saved = block["conditions"][f"{cell}_g1p00"]["images"][:10, 0].detach().cpu().numpy()
        differences = []
        all_within_tolerance = True
        for index in range(10):
            recomputed = _numpy_production_readout(raw[index, 0], gain)
            delta = np.abs(recomputed - saved[index])
            differences.append(float(np.max(delta)))
            allowed = REPLAY_ATOL + REPLAY_RTOL * np.abs(saved[index])
            all_within_tolerance = all_within_tolerance and bool(np.all(delta <= allowed))
        maximum = max(differences)
        _require(all_within_tolerance,
                 f"independent native-to-view NumPy readout failed for {cell}: max={maximum}")
        checks[cell] = {
            "samples": 10,
            "max_abs_28x28_difference": maximum,
            "gain": gain,
            "operation_order": "global_gain_then_clamp_0_1_then_adaptive_average_pool_28",
            "tolerance": {"rtol": REPLAY_RTOL, "atol": REPLAY_ATOL},
        }
    return {"status": "PASSED", "cells": checks, "samples_checked": 40,
            "source": "saved native_raw_examples only; no unavailable native rows reconstructed"}


def _conditions_summary(conditions: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in CONDITIONS:
        values = conditions[name]
        predicted, margins, correct = _recompute_predictions(
            values["logits"], np.full(SAMPLES_PER_CLASS, 9, dtype=np.int64)
        )
        counts = np.bincount(predicted, minlength=10)
        result[name] = {
            "samples": SAMPLES_PER_CLASS,
            "prediction_counts": counts.tolist(),
            "correct9": int(counts[9]),
            "recall9": float(correct.mean()),
            "identity_margin_mean": float(margins.mean()),
            "view_mean": float(values["view_mean"].mean()),
            "camera_total_power_w_mean": float(values["camera_total_power_w"].mean()),
            "clipping_area_fraction_mean": float(values["clipping_area_fraction"].mean()),
            "clipping_region_power_share_undefined_count": int(np.count_nonzero(~np.isfinite(values["clipping_region_power_share"]))),
            "excess_removed_power_share_undefined_count": int(np.count_nonzero(~np.isfinite(values["excess_removed_power_share"]))),
        }
    return result


def _category(prediction: int) -> str:
    if prediction == 9:
        return "correct9"
    if prediction == 4:
        return "9_to_4"
    if prediction == 8:
        return "9_to_8"
    return "other_wrong"


def _transition(left: dict[str, np.ndarray], right: dict[str, np.ndarray], label: int = 9) -> dict[str, Any]:
    left_pred = left["predicted_labels"].astype(np.int64)
    right_pred = right["predicted_labels"].astype(np.int64)
    left_correct = left_pred == label
    right_correct = right_pred == label
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (left_pred, right_pred), 1)
    return {
        "samples": len(left_pred),
        "from_to_prediction_matrix": matrix.tolist(),
        "from_categories": {key: int(np.count_nonzero([_category(int(x)) == key for x in left_pred]))
                            for key in ("correct9", "9_to_4", "9_to_8", "other_wrong")},
        "to_categories": {key: int(np.count_nonzero([_category(int(x)) == key for x in right_pred]))
                          for key in ("correct9", "9_to_4", "9_to_8", "other_wrong")},
        "recovered_errors": int(np.count_nonzero(~left_correct & right_correct)),
        "new_errors": int(np.count_nonzero(left_correct & ~right_correct)),
        "persistently_wrong": int(np.count_nonzero(~left_correct & ~right_correct)),
        "persistently_wrong_fraction_of_initial_errors": (
            float(np.count_nonzero(~left_correct & ~right_correct) / np.count_nonzero(~left_correct))
            if np.any(~left_correct) else None
        ),
    }


def _transition_tables(conditions: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    pairs = (
        ("decoder_B_minus_A_fixed_zA", "AA_g1p00", "BA_g1p00"),
        ("decoder_B_minus_A_fixed_zB", "AB_g1p00", "BB_g1p00"),
        ("input_B_minus_A_fixed_decoderA", "AA_g1p00", "AB_g1p00"),
        ("input_B_minus_A_fixed_decoderB", "BA_g1p00", "BB_g1p00"),
        ("AA_gain_0p85_vs_1p00", "AA_g1p00", "AA_g0p85"),
        ("AA_gain_1p15_vs_1p00", "AA_g1p00", "AA_g1p15"),
        ("BB_gain_0p85_vs_1p00", "BB_g1p00", "BB_g0p85"),
        ("BB_gain_1p15_vs_1p00", "BB_g1p00", "BB_g1p15"),
    )
    return {name: {"from": left, "to": right, **_transition(conditions[left], conditions[right])}
            for name, left, right in pairs}


def _spatial_features(image: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(image, dtype=np.float64).reshape(28, 28)
    total = float(values.sum())
    if total <= 0.0:
        return {"total": total, "centroid_x": None, "centroid_y": None, "spread_rms": None,
                "unit_mass": None}
    mass = values / total
    ys, xs = np.mgrid[0:28, 0:28]
    cx = float(np.sum(mass * xs))
    cy = float(np.sum(mass * ys))
    spread = float(np.sqrt(np.sum(mass * ((xs - cx) ** 2 + (ys - cy) ** 2))))
    return {"total": total, "centroid_x": cx, "centroid_y": cy, "spread_rms": spread,
            "unit_mass": mass}


def _spatial_decoder_changes(conditions: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    contexts = (("same_zA_decoder_B_minus_A", "AA_g1p00", "BA_g1p00"),
                ("same_zB_decoder_B_minus_A", "AB_g1p00", "BB_g1p00"))
    output: dict[str, Any] = {}
    for name, left_name, right_name in contexts:
        left, right = conditions[left_name], conditions[right_name]
        metrics: dict[str, list[float]] = {
            "view_mean_delta": [], "clipping_area_delta": [], "unit_mass_total_variation": [],
            "centroid_dx": [], "centroid_dy": [], "centroid_displacement": [], "spread_rms_delta": [],
        }
        undefined = 0
        for index in range(SAMPLES_PER_CLASS):
            a = _spatial_features(left["images"][index])
            b = _spatial_features(right["images"][index])
            metrics["view_mean_delta"].append(float(right["view_mean"][index] - left["view_mean"][index]))
            metrics["clipping_area_delta"].append(float(right["clipping_area_fraction"][index] - left["clipping_area_fraction"][index]))
            if a["unit_mass"] is None or b["unit_mass"] is None:
                undefined += 1
                continue
            dx = float(b["centroid_x"] - a["centroid_x"])
            dy = float(b["centroid_y"] - a["centroid_y"])
            metrics["unit_mass_total_variation"].append(float(0.5 * np.abs(b["unit_mass"] - a["unit_mass"]).sum()))
            metrics["centroid_dx"].append(dx)
            metrics["centroid_dy"].append(dy)
            metrics["centroid_displacement"].append(math.hypot(dx, dy))
            metrics["spread_rms_delta"].append(float(b["spread_rms"] - a["spread_rms"]))
        output[name] = {
            "pairs": SAMPLES_PER_CLASS,
            "undefined_unit_mass_pairs": undefined,
            "normalization_scope": "descriptive spatial statistics only; images shown and replayed without normalization",
            "mean": {key: float(np.mean(values)) if values else None for key, values in metrics.items()},
            "median": {key: float(np.median(values)) if values else None for key, values in metrics.items()},
        }
    return output


def _effective_channels(coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    power = np.abs(coefficients.astype(np.complex128)) ** 2
    total = power.sum(axis=1)
    squared_sum = np.square(power).sum(axis=1)
    effective = np.divide(total ** 2, squared_sum, out=np.full_like(total, np.nan), where=squared_sum > 0)
    return total, effective


def _phase_invariant_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.complex128)
    b = b.astype(np.complex128)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    numerator = np.abs(np.sum(np.conjugate(a) * b, axis=1))
    return np.divide(numerator, denom, out=np.full_like(numerator, np.nan), where=denom > 0)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < 3:
        return None
    ar = _rankdata(a[valid])
    br = _rankdata(b[valid])
    if np.std(ar) == 0 or np.std(br) == 0:
        return None
    return float(np.corrcoef(ar, br)[0, 1])


def _coefficient_associations(coefficient_rows: dict[str, np.ndarray],
                              conditions: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    total_a, neff_a = _effective_channels(coefficient_rows["A"])
    total_b, neff_b = _effective_channels(coefficient_rows["B"])
    overlap = _phase_invariant_overlap(coefficient_rows["A"], coefficient_rows["B"])
    groups: dict[str, Any] = {}
    for branch, total, effective, name in (("A", total_a, neff_a, "AA_g1p00"),
                                           ("B", total_b, neff_b, "BB_g1p00")):
        predicted = conditions[name]["predicted_labels"].astype(np.int64)
        outcome = np.array([_category(int(value)) for value in predicted], dtype=object)
        grouped = {}
        for category in ("correct9", "9_to_4", "9_to_8", "other_wrong"):
            mask = outcome == category
            grouped[category] = {
                "count": int(mask.sum()),
                "total_modal_power_mean": float(total[mask].mean()) if mask.any() else None,
                "effective_channels_mean": float(effective[mask].mean()) if mask.any() else None,
            }
        groups[branch] = {
            "canonical_cell": name,
            "by_error_category": grouped,
            "spearman": {
                "total_modal_power_vs_correct9": _spearman(total, (predicted == 9).astype(float)),
                "effective_channels_vs_correct9": _spearman(effective, (predicted == 9).astype(float)),
                "total_modal_power_vs_prediction4": _spearman(total, (predicted == 4).astype(float)),
                "total_modal_power_vs_prediction8": _spearman(total, (predicted == 8).astype(float)),
                "effective_channels_vs_prediction4": _spearman(effective, (predicted == 4).astype(float)),
                "effective_channels_vs_prediction8": _spearman(effective, (predicted == 8).astype(float)),
            },
        }
    aa = conditions["AA_g1p00"]["predicted_labels"].astype(np.int64) == 9
    bb = conditions["BB_g1p00"]["predicted_labels"].astype(np.int64) == 9
    return {
        "sample_count": SAMPLES_PER_CLASS,
        "definitions": {
            "total_modal_power": "sum_j |z_j|^2",
            "effective_channels": "(sum_j |z_j|^2)^2 / sum_j |z_j|^4",
            "phase_invariant_complex_overlap": "|sum_j conj(zA_j) zB_j| / (||zA|| ||zB||)",
        },
        "A": {"total_modal_power_mean": float(total_a.mean()), "effective_channels_mean": float(neff_a.mean()),
              "zero_power_count": int(np.count_nonzero(total_a == 0))},
        "B": {"total_modal_power_mean": float(total_b.mean()), "effective_channels_mean": float(neff_b.mean()),
              "zero_power_count": int(np.count_nonzero(total_b == 0))},
        "A_B_phase_invariant_overlap": {
            "mean": float(np.nanmean(overlap)) if np.isfinite(overlap).any() else None,
            "median": float(np.nanmedian(overlap)) if np.isfinite(overlap).any() else None,
            "undefined_zero_norm_count": int(np.count_nonzero(~np.isfinite(overlap))),
            "spearman_with_AA_to_BB_correctness_change": _spearman(overlap, bb.astype(float) - aa.astype(float)),
            "interpretation": "descriptive association only; coefficients come from different Encoder-plus-prior distributions",
        },
        "associations": groups,
    }


def _bootstrap(conditions: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    resamples = rng.integers(0, SAMPLES_PER_CLASS, size=(BOOTSTRAP_RESAMPLES, SAMPLES_PER_CLASS), dtype=np.int32)
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for name in CONDITIONS:
        item = conditions[name]
        predicted = item["predicted_labels"].astype(np.int64)
        arrays[name] = {
            "recall9": (predicted == 9).astype(np.float64),
            "margin": item["identity_margin"].astype(np.float64),
            "view_mean": item["view_mean"].astype(np.float64),
            "clipping": item["clipping_area_fraction"].astype(np.float64),
        }
    pairs = {
        "decoder_B_minus_A_fixed_zA": ("AA_g1p00", "BA_g1p00"),
        "decoder_B_minus_A_fixed_zB": ("AB_g1p00", "BB_g1p00"),
        "input_B_minus_A_fixed_decoderA": ("AA_g1p00", "AB_g1p00"),
        "input_B_minus_A_fixed_decoderB": ("BA_g1p00", "BB_g1p00"),
        "AA_gain_0p85_minus_1p00": ("AA_g1p00", "AA_g0p85"),
        "AA_gain_1p15_minus_1p00": ("AA_g1p00", "AA_g1p15"),
        "BB_gain_0p85_minus_1p00": ("BB_g1p00", "BB_g0p85"),
        "BB_gain_1p15_minus_1p00": ("BB_g1p00", "BB_g1p15"),
    }
    result: dict[str, Any] = {}
    metrics = ("recall9", "margin", "view_mean", "clipping")
    for name, (left, right) in pairs.items():
        result[name] = {}
        for metric in metrics:
            difference = arrays[right][metric] - arrays[left][metric]
            sampled = difference[resamples].mean(axis=1)
            result[name][metric] = {
                "estimate": float(difference.mean()),
                "ci_95": [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
            }
    interaction = ((arrays["BB_g1p00"]["recall9"] - arrays["AB_g1p00"]["recall9"])
                   - (arrays["BA_g1p00"]["recall9"] - arrays["AA_g1p00"]["recall9"]))
    sampled_interaction = interaction[resamples].mean(axis=1)
    result["decoder_input_interaction_recall9"] = {
        "estimate": float(interaction.mean()),
        "ci_95": [float(np.quantile(sampled_interaction, 0.025)),
                  float(np.quantile(sampled_interaction, 0.975))],
    }
    return {
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "unit": "fixed requested-class-9 proposal IDs; same sampled IDs consumed by every condition",
        "interval": "percentile 95% bootstrap interval, descriptive for these saved cohorts only",
        "contrasts": result,
    }


def _png_data_uri(image: np.ndarray, upscale: int = 12) -> str:
    values = np.asarray(image, dtype=np.float32).reshape(28, 28)
    pixels = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="L").resize((28 * upscale, 28 * upscale), Image.Resampling.NEAREST).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _selection_entries(conditions: dict[str, dict[str, np.ndarray]], blocks: dict[tuple[str, int], dict[str, Any]],
                       proposal_ids: np.ndarray) -> list[dict[str, Any]]:
    rng = np.random.default_rng(SELECTION_SEED)
    chosen_keys: set[tuple[int, int]] = set()
    selected: list[dict[str, Any]] = []
    for cell in CELLS:
        for stratum, requested_label, ed_prediction, count in STRATA:
            if requested_label == 9:
                predictions = conditions[f"{cell}_g1p00"]["predicted_labels"].astype(np.int64)
                candidates = np.flatnonzero(predictions == ed_prediction)
            else:
                candidates = np.arange(SAMPLES_PER_CLASS)
            available = [int(index) for index in candidates
                         if (requested_label, int(proposal_ids[index])) not in chosen_keys]
            _require(len(available) >= count,
                     f"fixed {cell}/{stratum} blind-review stratum has only {len(available)} eligible rows")
            sampled = rng.choice(np.asarray(available, dtype=np.int64), size=count, replace=False)
            for index in sampled.tolist():
                proposal = int(proposal_ids[index])
                key = (requested_label, proposal)
                _require(key not in chosen_keys, f"blind-review sampling repeated class/proposal {key}")
                chosen_keys.add(key)
                selected.append({
                    "selection_index": len(selected),
                    "cell": cell,
                    "condition_id": f"{cell}_g1p00",
                    "requested_label": requested_label,
                    "proposal_id": proposal,
                    "cohort_index": int(index),
                    "ed_prediction": int(conditions[f"{cell}_g1p00"]["predicted_labels"][index]),
                    "stratum": stratum,
                })
    _require(len(selected) == 60 and len(chosen_keys) == 60, "blind-review manifest does not contain 60 unique cards")
    order = np.random.default_rng(SHUFFLE_SEED).permutation(len(selected))
    for display_index, selected_index in enumerate(order.tolist(), 1):
        selected[selected_index]["blind_id"] = f"card_{display_index:03d}"
        selected[selected_index]["display_order"] = display_index
    return sorted(selected, key=lambda item: item["display_order"])


def _public_cards(entries: list[dict[str, Any]], blocks: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, str]]:
    cards: list[dict[str, str]] = []
    for entry in entries:
        block = blocks[(entry["cell"], entry["requested_label"])]
        image = block["conditions"][entry["condition_id"]]["images"][entry["cohort_index"], 0].numpy()
        cards.append({"blind_id": entry["blind_id"], "image": _png_data_uri(image)})
    return cards


def _review_html(review_id: str, cards: list[dict[str, str]], *, task_id: str = TASK_ID,
                 download_name: str = "lightgen_class9_blind_ratings.json") -> str:
    cards_json = json.dumps(cards, ensure_ascii=False, separators=(",", ":"))
    labels = [str(number) for number in range(10)] + ["unreadable"]
    labels_json = json.dumps(labels, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>盲评手写数字</title>
<style>
body{{font-family:system-ui,'Microsoft YaHei',sans-serif;background:#f2f3f5;color:#20242a;margin:0}}
main{{max-width:720px;margin:24px auto;padding:20px;background:white;border-radius:14px;box-shadow:0 3px 18px #0001}}
h1{{font-size:22px;margin:0 0 8px}}.hint{{color:#5d6670}}#image{{display:block;width:420px;height:420px;max-width:82vw;max-height:82vw;object-fit:contain;image-rendering:pixelated;margin:18px auto;background:white;border:1px solid #ddd}}
.controls{{display:flex;flex-wrap:wrap;justify-content:center;gap:8px;margin:16px 0}}button{{font:inherit;padding:9px 14px;border:1px solid #bac2cc;background:#fff;border-radius:8px;cursor:pointer}}button.active{{background:#254e8a;color:white;border-color:#254e8a}}.nav{{display:flex;justify-content:space-between;align-items:center;gap:12px}}#progress{{font-variant-numeric:tabular-nums}}.actions{{display:flex;gap:10px;justify-content:flex-end;margin-top:16px}}#status{{min-height:1.5em;color:#456}}.blindid{{text-align:center;color:#89919a;font-size:12px}}
</style><main><h1>手写数字盲评</h1>
<p class="hint">请判断图像中是否能辨认出一个数字。选择 0–9，或“不可辨认”。页面不显示来源、目标类别或模型判断。</p>
<div class="nav"><button id="prev">上一张</button><span id="progress"></span><button id="next">下一张</button></div>
<div class="blindid" id="blindid"></div><img id="image" alt="待判读的手写数字">
<div id="choices" class="controls"></div>
<p class="hint">可以先浏览再补标签。每张卡片均无预选答案；选择会尝试保存在本浏览器。完成后导出JSON留存。</p>
<div class="actions"><button id="save">保存进度</button><button id="export">导出完整评分 JSON</button></div><p id="status" aria-live="polite"></p>
</main>
<script>
const reviewId={json.dumps(review_id)},taskId={json.dumps(task_id)},cards={cards_json},labels={labels_json};
const storageKey='lightgen-blind-review-'+reviewId;let ratings={{}};let index=0;
try{{ratings=JSON.parse(localStorage.getItem(storageKey)||'{{}}')}}catch{{ratings={{}}}}
const image=document.getElementById('image'),choices=document.getElementById('choices');
function render(){{const card=cards[index];document.getElementById('progress').textContent=`${{index+1}} / ${{cards.length}}`;document.getElementById('blindid').textContent=card.blind_id;image.src=card.image;choices.replaceChildren();for(const label of labels){{const button=document.createElement('button');button.textContent=label==='unreadable'?'不可辨认':label;if(ratings[card.blind_id]===label)button.classList.add('active');button.onclick=()=>{{ratings[card.blind_id]=label;save(false);render()}};choices.append(button)}}}}
function save(show){{let saved=true;try{{localStorage.setItem(storageKey,JSON.stringify(ratings))}}catch{{saved=false}}if(show)document.getElementById('status').textContent=saved?'进度已保存在本浏览器。':'浏览器未允许本地保存；完成后请导出评分JSON留存。';return saved}}
document.getElementById('prev').onclick=()=>{{index=Math.max(0,index-1);render()}};document.getElementById('next').onclick=()=>{{index=Math.min(cards.length-1,index+1);render()}};
document.getElementById('save').onclick=()=>save(true);
document.getElementById('export').onclick=()=>{{const missing=cards.filter(card=>!labels.includes(ratings[card.blind_id]));if(missing.length){{document.getElementById('status').textContent=`还缺 ${{missing.length}} 张，请完成后再导出。`;return}}const payload={{schema:'lightgen_blind_ratings.v1',task_id:taskId,review_id:reviewId,ratings:cards.map(card=>({{blind_id:card.blind_id,rating:ratings[card.blind_id]}}))}};const blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download={json.dumps(download_name)};a.click();URL.revokeObjectURL(a.href);document.getElementById('status').textContent=`${{cards.length}} 条标签已导出。`}};
render();
</script></html>"""


def _write_metrics_csv(path: Path, conditions: dict[str, dict[str, np.ndarray]], proposal_ids: np.ndarray) -> None:
    fields = ("condition_id", "cell", "input_source", "decoder", "class", "proposal_id", "predicted_label",
              "correct", "identity_margin", "view_mean", "camera_total_power_w", "peak_w_per_pixel",
              "scaled_peak", "clipping_area_fraction", "clipping_region_power_share", "excess_removed_power_share")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name in CONDITIONS:
            cell = name[:2]
            values = conditions[name]
            predicted, margins, correct = _recompute_predictions(
                values["logits"], np.full(SAMPLES_PER_CLASS, 9, dtype=np.int64)
            )
            for index in range(SAMPLES_PER_CLASS):
                row: dict[str, Any] = {
                    "condition_id": name, "cell": cell, "input_source": CELL_INPUT[cell],
                    "decoder": CELL_DECODER[cell], "class": 9, "proposal_id": int(proposal_ids[index]),
                    "predicted_label": int(predicted[index]), "correct": bool(correct[index]),
                }
                row["identity_margin"] = float(margins[index])
                for field in fields[9:]:
                    if field == "identity_margin":
                        continue
                    array_field = field
                    row[field] = float(values[array_field][index])
                writer.writerow({key: ("" if isinstance(value, float) and not math.isfinite(value) else value)
                                 for key, value in row.items()})
    os.replace(temporary, path)


def _blind_manifest(entries: list[dict[str, Any]], review_id: str,
                    conditions: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    counts = {}
    for cell in CELLS:
        counts[cell] = {}
        predicted = conditions[f"{cell}_g1p00"]["predicted_labels"].astype(np.int64)
        for stratum, label, ed_prediction, _ in STRATA:
            counts[cell][stratum] = int(np.count_nonzero(predicted == ed_prediction)) if label == 9 else SAMPLES_PER_CLASS
    return {
        "schema": "lightgen_class9_blind_review_manifest.v1",
        "task_id": TASK_ID,
        "review_id": review_id,
        "cards": 60,
        "selection_seed": SELECTION_SEED,
        "shuffle_seed": SHUFFLE_SEED,
        "gain_multiplier": 1.0,
        "stratum_candidate_counts": counts,
        "entries": entries,
        "blindness": "UI hides cell, requested label, proposal ID, stratum and E_D prediction; mapping is stored separately for final unblinding.",
    }


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total == 0:
        return None
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    half = z * math.sqrt((p * (1 - p) / total) + (z2 / (4 * total * total))) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _validate_ratings(payload: Any, manifest: dict[str, Any]) -> dict[str, str]:
    _require(isinstance(payload, dict) and payload.get("schema") == "lightgen_blind_ratings.v1"
             and payload.get("task_id") == TASK_ID,
             "ratings file schema or task ID is invalid")
    _require(payload.get("review_id") == manifest["review_id"],
             "ratings file belongs to a different blinded review")
    raw = payload.get("ratings")
    _require(isinstance(raw, list) and len(raw) == 60, "ratings file must contain exactly 60 cards")
    known = {entry["blind_id"] for entry in manifest["entries"]}
    ratings: dict[str, str] = {}
    for row in raw:
        _require(isinstance(row, dict) and set(row) == {"blind_id", "rating"},
                 "each rating row must contain only blind_id and rating")
        card = row["blind_id"]
        rating = row["rating"]
        _require(isinstance(card, str) and card in known, f"ratings contain an unknown review ID: {card}")
        _require(card not in ratings, f"ratings repeat blind card {card}")
        _require(isinstance(rating, str) and rating in RATING_LABELS,
                 f"rating for {card} must be 0-9 or unreadable")
        ratings[card] = rating
    _require(set(ratings) == known, "ratings file has missing cards")
    return ratings


def _human_diagnosis(manifest: dict[str, Any], ratings: dict[str, str]) -> dict[str, Any]:
    entry_by_id = {entry["blind_id"]: entry for entry in manifest["entries"]}
    categories = {"readable9": 0, "readable_other": 0, "unreadable": 0}
    def category(rating: str) -> str:
        if rating == "9":
            return "readable9"
        if rating == "unreadable":
            return "unreadable"
        return "readable_other"
    for card, rating in ratings.items():
        entry = entry_by_id[card]
        if entry["requested_label"] == 9:
            categories[category(rating)] += 1
    primary: dict[str, list[str]] = {"AA": [], "BB": []}
    for card, rating in ratings.items():
        entry = entry_by_id[card]
        if entry["cell"] in primary and entry["requested_label"] == 9 and entry["ed_prediction"] in (4, 8):
            primary[entry["cell"]].append(category(rating))
    _require(len(primary["AA"]) == 10 and len(primary["BB"]) == 10,
             "primary AA/BB human diagnostic population must contain 20 requested-class-9 cards")
    total_counts = {name: sum(value == name for cell in primary.values() for value in cell)
                    for name in ("readable9", "readable_other", "unreadable")}
    total_majority = [name for name, count in total_counts.items() if count > 10]
    cell_majorities = {}
    for cell, values in primary.items():
        counts = {name: values.count(name) for name in ("readable9", "readable_other", "unreadable")}
        majority = [name for name, count in counts.items() if count > len(values) / 2]
        cell_majorities[cell] = {"counts": counts, "majority": majority[0] if len(majority) == 1 else None}
    pooled_majority = total_majority[0] if len(total_majority) == 1 else None
    lower = _wilson(total_counts[pooled_majority], 20)[0] if pooled_majority else None
    same_cell_majority = (pooled_majority is not None
                          and cell_majorities["AA"]["majority"] == pooled_majority
                          and cell_majorities["BB"]["majority"] == pooled_majority)
    decision_type = pooled_majority if same_cell_majority and lower is not None and lower > 0.5 else None
    recommendation = {
        "readable_other": "优先设计固定预算的身份/笔画保真训练对照；训练目标不得使用确认评价器E_D。",
        "readable9": "优先核验独立身份评价器，不依据E_D错误增加损失。",
        "unreadable": "优先设计固定增益下的可视强度与空间对比质量改动。",
        None: "证据混合或不足；下一项只扩大独立评阅，不启动自动训练或增益选择。",
    }[decision_type]
    strata = {}
    for stratum, _, _, _ in STRATA:
        selected = [entry for entry in manifest["entries"] if entry["stratum"] == stratum]
        if stratum.endswith("unfiltered"):
            strata[stratum] = {"cards": len(selected), "human_labels_descriptive_only": True}
            continue
        counts = {name: 0 for name in ("readable9", "readable_other", "unreadable")}
        for entry in selected:
            counts[category(ratings[entry["blind_id"]])] += 1
        total = sum(counts.values())
        strata[stratum] = {"cards": total, "counts": counts,
                           "wilson_95": {name: _wilson(count, total) for name, count in counts.items()}}
    return {
        "rating_population": "requested class 9 only; requested 4/8 distractors are not scored as correct/incorrect",
        "requested_class9_all_52": {"counts": categories, "cards": 52},
        "primary_20_AA_BB_ED_errors": {
            "counts": total_counts,
            "wilson_95": {name: _wilson(count, 20) for name, count in total_counts.items()},
            "per_cell": cell_majorities,
            "dominant_category": pooled_majority,
            "dominant_category_wilson_lower": lower,
            "rule_satisfied": decision_type is not None,
            "decision_category": decision_type,
        },
        "strata": strata,
        "diagnostic_routing": recommendation,
        "individual_labels": {card: ratings[card] for card in sorted(ratings)},
    }


def _build_unblinded_html(report: dict[str, Any], manifest: dict[str, Any],
                          ratings: dict[str, str] | None, blocks: dict[tuple[str, int], dict[str, Any]]) -> str:
    if ratings is None:
        return "<!doctype html><html lang='zh'><meta charset='utf-8'><title>揭盲对照待完成</title><h1>人工标签待完成</h1><p>请先完成 blind_review.html 并用导出的JSON续接同一任务。当前不生成身份诊断结论。</p></html>"
    rows = []
    for entry in manifest["entries"]:
        block = blocks[(entry["cell"], entry["requested_label"])]
        image = block["conditions"][entry["condition_id"]]["images"][entry["cohort_index"], 0].numpy()
        encoded = _png_data_uri(image, upscale=6)
        rows.append(
            "<tr>" + f"<td>{entry['blind_id']}</td><td><img width='84' height='84' src='{encoded}'></td>"
            + f"<td>{entry['cell']}</td><td>{entry['requested_label']}</td><td>{entry['proposal_id']}</td>"
            + f"<td>{entry['ed_prediction']}</td><td>{html.escape(ratings[entry['blind_id']])}</td>"
            + f"<td>{html.escape(entry['stratum'])}</td></tr>"
        )
    return "\n".join([
        "<!doctype html><html lang='zh'><meta charset='utf-8'><title>类别9揭盲对照</title>",
        "<style>body{font-family:system-ui,'Microsoft YaHei',sans-serif;margin:24px;color:#20242a}table{border-collapse:collapse}td,th{border:1px solid #ccd;padding:6px;text-align:center}img{image-rendering:pixelated}</style>",
        "<h1>类别9盲评揭盲对照</h1><p>请求类别4/8卡片只作盲化混入项，不参与人眼正确率判断。</p>",
        f"<p>{html.escape(report['decision']['recommendation'])}</p>",
        "<table><thead><tr><th>评阅ID</th><th>视图</th><th>组合</th><th>请求类</th><th>proposal</th><th>E_D</th><th>人工标签</th><th>分层</th></tr></thead><tbody>",
        *rows, "</tbody></table></html>",
    ])


def _build_report(inputs: Inputs, manifest: dict[str, Any], started: float,
                  ratings: dict[str, str] | None) -> dict[str, Any]:
    automatic = {
        "class9_conditions": _conditions_summary(inputs.conditions),
        "transitions": _transition_tables(inputs.conditions),
        "coefficient_associations": _coefficient_associations(inputs.coefficient_rows, inputs.conditions),
        "same_coefficient_decoder_changes": _spatial_decoder_changes(inputs.conditions),
        "bootstrap": _bootstrap(inputs.conditions),
        "native_readout_check": inputs.preflight["native_readout"],
        "proposal_ids": {"count": len(inputs.proposal_ids), "min": int(inputs.proposal_ids.min()),
                         "max": int(inputs.proposal_ids.max()), "A_B_paired": True},
    }
    report: dict[str, Any] = {
        "schema": "lightgen_small_class9_saved_defect_diagnosis.v1",
        "task_id": TASK_ID,
        "status": "AWAITING_HUMAN_RATINGS" if ratings is None else "COMPLETE_DIAGNOSTIC",
        "measurement_complete": ratings is not None,
        "system_quality_gate": "NOT_REEVALUATED; original A/B VALID_FAILED remains unchanged",
        "scientific_question": "For class-9 9-to-4/8 errors, separate visible identity confusion, unreadability and E_D disagreement using fixed saved evidence and human labels.",
        "inputs": {
            "attribution_dir": str(inputs.attribution_dir),
            "ab_evidence_dir": str(inputs.ab_evidence_dir),
            "A_global_step": 35_000, "B_global_step": 35_000,
            "shared_source_run_id": inputs.checkpoints["A"]["source_run_id"],
            "shared_source_step": 30_000,
            "camera_gain": inputs.camera_gain,
            "model_identity": EXPECTED_MODEL,
            "class9_proposal_ids": [int(x) for x in inputs.proposal_ids],
            "A_B_coefficients_are_distinct_encoder_plus_prior_distributions": True,
        },
        "preflight": inputs.preflight,
        "automatic_diagnosis": automatic,
        "human_review": {"status": "PENDING" if ratings is None else "COMPLETE",
                         "review_id": manifest["review_id"], "cards": 60,
                         "mapping_file": "blind_review_manifest.json",
                         "ratings_file": None if ratings is None else "human_ratings.json"},
        "decision": {"code": "PENDING_HUMAN_RATINGS" if ratings is None else "PENDING_ROUTING",
                     "recommendation": "等待60张盲评标签；机器结果不能替代人工判读。" if ratings is None else "人工标签已回读，按primary20决策规则路由。"},
        "limitations": [
            "All-population analysis uses saved 28x28 production views and class-9 rows only.",
            "Native 200x200 replay is limited to the ten saved examples per cell; no missing native images were reconstructed.",
            "A/B coefficient vectors represent distinct Encoder-plus-prior distributions; associations are descriptive, not causal.",
            "Requested class-4/8 review cards are blinding distractors without per-image identity truth and are not graded.",
            "This diagnosis does not change historical A/B VALID_FAILED statuses or establish real CMOS or human-population accuracy.",
        ],
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "model_constructions": 0, "optical_forwards": 0, "encoder_forwards": 0,
            "evaluator_forwards": 0, "new_sampling": 0, "gradients": 0, "optimizer_steps": 0,
            "new_train_dev_test_reads": 0,
        },
    }
    if ratings is not None:
        human = _human_diagnosis(manifest, ratings)
        report["human_diagnosis"] = human
        report["decision"] = {
            "code": human["primary_20_AA_BB_ED_errors"]["decision_category"] or "MIXED_OR_UNRESOLVED",
            "recommendation": human["diagnostic_routing"],
            "basis": "Predeclared AA/BB primary-20 majority and Wilson-lower-bound rule; request-class-4/8 distractors are not scored.",
        }
    return report


def _render_report_html(report: dict[str, Any]) -> str:
    status = html.escape(report["status"])
    decision = html.escape(report["decision"]["recommendation"])
    class9 = report["automatic_diagnosis"]["class9_conditions"]
    rows = []
    for name, item in class9.items():
        rows.append(f"<tr><td>{name}</td><td>{item['correct9']}/1000</td><td>{item['recall9']:.3f}</td>"
                    f"<td>{item['view_mean']:.5g}</td><td>{item['clipping_area_fraction_mean']:.5g}</td></tr>")
    human_html = "<p>盲评标签尚未提供；完整身份结论待评阅完成。</p>"
    if "human_diagnosis" in report:
        primary = report["human_diagnosis"]["primary_20_AA_BB_ED_errors"]
        human_html = (f"<p>主要20例：可辨为9 {primary['counts']['readable9']}，可辨为其他数字 "
                      f"{primary['counts']['readable_other']}，不可辨认 {primary['counts']['unreadable']}。</p>"
                      f"<p>{html.escape(report['decision']['basis'])}</p>")
    return "\n".join([
        "<!doctype html><html lang='zh'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>类别9保存证据诊断</title><style>body{font-family:system-ui,'Microsoft YaHei',sans-serif;max-width:1050px;margin:28px auto;padding:0 18px;color:#20242a}table{border-collapse:collapse}td,th{border:1px solid #ccd;padding:7px 12px}code{background:#f2f3f5;padding:2px 4px}</style>",
        "<h1>Gaussian小架构：类别9保存证据诊断</h1>",
        f"<p><b>状态：</b>{status}。A/B历史 <code>VALID_FAILED</code> 保持不变；本轮不重评系统质量Gate。</p>",
        f"<h2>类别9自动重算</h2><table><thead><tr><th>组合/增益</th><th>预测为9</th><th>召回</th><th>平均灰度</th><th>裁剪面积</th></tr></thead><tbody>{''.join(rows)}</tbody></table>",
        "<p>自动分析重算了8,000条类别9读出，并核对原逐图CSV、报告汇总和四组合首10条原生读出。</p>",
        f"<h2>人工判读</h2>{human_html}",
        f"<h2>当前建议</h2><p>{decision}</p>",
        "<p><a href='blind_review.html'>打开60张盲评页面</a>　<a href='unblinded_comparison.html'>揭盲对照页</a></p>",
        "<p>说明：全人口只用保存的28×28视图；200×200仅限已保存的每组合前10张。A/B复数系数来自不同Encoder与先验输入，相关性不作因果解释。</p>",
        "</html>",
    ])


def _closeout(report: dict[str, Any]) -> str:
    if report["status"] == "AWAITING_HUMAN_RATINGS":
        decision_text = "盲评尚未提交；此时不选择身份混淆、E_D不一致或不可辨认中的主导解释。人工标签返回后按已固定的primary-20规则决定一项后继建议。"
        result_text = "自动保存证据诊断完成，状态为 `AWAITING_HUMAN_RATINGS`；整体诊断仍等待60张人工标签。"
        science_text = "固定的8,000条类别9读出已独立重算并完成CSV/报告核对；这些代理结果不能代替人眼判读。"
        gap_text = "尚缺少60张盲评人工标签。真实CMOS、人群可读性和A/B历史系统质量仍未被本轮改变。"
        next_text = "请在 `blind_review.html` 完成全部卡片并导出JSON，再用同一目录和 `--resume --ratings-file` 续接；不重新抽样。此操作是完成本轮诊断，不授权训练。"
    else:
        diagnosis = report["human_diagnosis"]["primary_20_AA_BB_ED_errors"]
        result_text = "固定保存证据和60张盲评均已完成；系统质量Gate没有重评，A/B历史 `VALID_FAILED` 保持原判定。"
        science_text = (f"主要20例人工标签为：可辨为9 {diagnosis['counts']['readable9']}，可辨为其他数字 "
                        f"{diagnosis['counts']['readable_other']}，不可辨认 {diagnosis['counts']['unreadable']}。")
        gap_text = "结论只适用于本次保存人口和60张评阅卡，不外推到完整生成总体、真实CMOS或其他读者。"
        next_text = report["decision"]["recommendation"] + " 这是建议，不会自动启动训练、增益修改或模型采用。"
        decision_text = report["decision"]["basis"]
    return "\n".join([
        "# Gaussian小架构类别9保存证据定位与盲评",
        "",
        f"结果与状态：{result_text}",
        "",
        f"科学认识：{science_text}",
        "",
        f"目标缺口：{gap_text}",
        "",
        f"工程完成与验证：CPU读取固定A/B检查点、生成队列和归因分块；模型构造、光学/Encoder/E_D前向、采样、梯度、优化器及TRAIN/DEV/TEST新读取均为0。{report['preflight']['condition_rows_recomputed']}条类别9读出与原CSV和报告汇总核对，40张保存原生图完成独立NumPy读出核验。",
        "",
        f"决策影响与下一步：{decision_text} {next_text}",
        "",
        f"实际总耗时：{report['execution']['elapsed_seconds']:.3f}秒。",
    ])


def _write_csv(path: Path, conditions: dict[str, dict[str, np.ndarray]], proposal_ids: np.ndarray) -> None:
    _write_metrics_csv(path, conditions, proposal_ids)


def _expected_output_files(output_dir: Path) -> None:
    missing = [name for name in ARTIFACT_FILES if not (output_dir / name).is_file()]
    _require(not missing, f"existing task output is incomplete; refusing resume: missing {missing}")
    report = _read_json(output_dir / "report.json")
    _require(report.get("task_id") == TASK_ID, "existing output belongs to another task")
    receipt = _read_json(output_dir / "validation_receipt.json")
    _require(receipt.get("task_id") == TASK_ID and receipt.get("automatic_diagnostic_complete") is True,
             "existing output lacks a complete automatic diagnostic receipt")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    attribution_dir = _resolve(args.attribution_dir)
    ab_evidence_dir = _resolve(args.ab_evidence_dir)
    output_dir = _resolve(args.output_dir)
    inputs = _load_inputs(attribution_dir, ab_evidence_dir)
    if args.preflight_only:
        result = {"task_id": TASK_ID, "status": "PREFLIGHT_READY", "preflight": inputs.preflight}
        print(json.dumps(_json_safe(result), ensure_ascii=False), flush=True)
        return result

    existing = output_dir.exists()
    if existing and not args.resume:
        raise EvidenceError("output directory already exists; pass --resume to read its fixed manifest")
    if not existing and args.resume:
        raise EvidenceError("--resume requires an existing output directory with a saved blind manifest")
    if existing:
        _expected_output_files(output_dir)
        saved_preflight = _read_json(output_dir / "preflight.json")
        _require(saved_preflight.get("attribution_dir") == str(attribution_dir)
                 and saved_preflight.get("ab_evidence_dir") == str(ab_evidence_dir),
                 "resume input paths differ from the existing task output")
        manifest = _read_json(output_dir / "blind_review_manifest.json")
        _require(manifest.get("task_id") == TASK_ID and manifest.get("cards") == 60
                 and isinstance(manifest.get("review_id"), str)
                 and len(manifest.get("entries", [])) == 60,
                 "saved blind-review manifest is incomplete or invalid")
        deterministic_entries = _selection_entries(inputs.conditions, inputs.blocks, inputs.proposal_ids)
        saved_projection = [{k: e[k] for k in ("selection_index", "cell", "condition_id", "requested_label",
                                                  "proposal_id", "cohort_index", "ed_prediction", "stratum", "blind_id", "display_order")}
                            for e in manifest["entries"]]
        expected_projection = [{k: e[k] for k in ("selection_index", "cell", "condition_id", "requested_label",
                                                    "proposal_id", "cohort_index", "ed_prediction", "stratum", "blind_id", "display_order")}
                               for e in deterministic_entries]
        _require(saved_projection == expected_projection,
                 "saved blind-review selection does not match the deterministic fixed sample")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        entries = _selection_entries(inputs.conditions, inputs.blocks, inputs.proposal_ids)
        manifest = _blind_manifest(entries, uuid.uuid4().hex, inputs.conditions)
        _atomic_json(output_dir / "preflight.json", {
            **inputs.preflight,
            "task_id": TASK_ID,
            "attribution_dir": str(attribution_dir),
            "ab_evidence_dir": str(ab_evidence_dir),
        })

    if existing:
        prior_report = _read_json(output_dir / "report.json")
        if prior_report.get("status") == "COMPLETE_DIAGNOSTIC":
            if args.ratings_file is None:
                return prior_report
            prior_ratings = _read_json(output_dir / "human_ratings.json")
            new_ratings = _read_json(_resolve(args.ratings_file))
            _validate_ratings(new_ratings, manifest)
            _require(prior_ratings == new_ratings, "completed human ratings cannot be silently replaced")
            return prior_report

    cards = _public_cards(manifest["entries"], inputs.blocks)
    if not existing:
        _atomic_json(output_dir / "blind_review_manifest.json", manifest)
        (output_dir / "blind_review.html").write_text(_review_html(manifest["review_id"], cards), encoding="utf-8")
        _write_csv(output_dir / "per_class9_condition_metrics.csv", inputs.conditions, inputs.proposal_ids)
        _atomic_json(output_dir / "transition_tables.json", _transition_tables(inputs.conditions))
        _atomic_json(output_dir / "coefficient_associations.json",
                     _coefficient_associations(inputs.coefficient_rows, inputs.conditions))
        _atomic_json(output_dir / "spatial_decoder_changes.json", _spatial_decoder_changes(inputs.conditions))
        _atomic_json(output_dir / "bootstrap.json", _bootstrap(inputs.conditions))
        _atomic_json(output_dir / "native_readout_check.json", inputs.preflight["native_readout"])

    ratings: dict[str, str] | None = None
    ratings_payload: dict[str, Any] | None = None
    if args.ratings_file is not None:
        ratings_path = _resolve(args.ratings_file)
        ratings_payload = _read_json(ratings_path)
        ratings = _validate_ratings(ratings_payload, manifest)
    report = _build_report(inputs, manifest, started, ratings)
    if ratings is not None and ratings_payload is not None:
        _atomic_json(output_dir / "human_ratings.json", ratings_payload)
    _atomic_json(output_dir / "report.json", report)
    (output_dir / "report.html").write_text(_render_report_html(report), encoding="utf-8")
    (output_dir / "unblinded_comparison.html").write_text(
        _build_unblinded_html(report, manifest, ratings, inputs.blocks), encoding="utf-8"
    )
    (output_dir / "closeout_report.md").write_text(_closeout(report), encoding="utf-8")
    elapsed = time.perf_counter() - started
    report["execution"]["elapsed_seconds"] = elapsed
    _atomic_json(output_dir / "report.json", report)
    (output_dir / "report.html").write_text(_render_report_html(report), encoding="utf-8")
    (output_dir / "closeout_report.md").write_text(_closeout(report), encoding="utf-8")
    checks = [
        "A/B checkpoint step, shared source, Gaussian model identity and frozen gain checked",
        "fixed A/B class and proposal rows checked across cohorts and saved attribution blocks",
        "8,000 class-9 logits, margins, correctness and CSV rows independently recomputed",
        "original report class-9 summaries checked against saved block consumers",
        "40 saved native examples checked with independent NumPy gain-clamp-pool readout",
        "2,000 shared-proposal bootstrap resamples and deterministic 60-card manifest consumed",
        "no model construction, forward, sampling, gradient, optimizer or new dataset read executed",
    ]
    _atomic_json(output_dir / "validation_receipt.json", {
        "task_id": TASK_ID,
        "status": "PASSED",
        "automatic_diagnostic_complete": True,
        "human_ratings_complete": ratings is not None,
        "final_diagnosis_complete": ratings is not None,
        "checked": checks,
        "human_label_input": None if ratings is None else str(_resolve(args.ratings_file)),
    })
    _require(all((output_dir / name).is_file() for name in ARTIFACT_FILES),
             "one or more required output artifacts were not saved")
    output_bytes = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
    _require(output_bytes <= 64 * 1024 * 1024, "formal output exceeds the approved 64 MiB budget")
    print(json.dumps({"task_id": TASK_ID, "status": report["status"],
                      "output_dir": str(output_dir), "output_bytes": output_bytes,
                      "elapsed_seconds": report["execution"]["elapsed_seconds"],
                      "decision": report["decision"]}, ensure_ascii=False), flush=True)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution-dir", type=Path, required=True)
    parser.add_argument("--ab-evidence-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ratings-file", type=Path)
    args = parser.parse_args()
    if args.preflight_only and (args.resume or args.ratings_file is not None):
        parser.error("--preflight-only cannot be combined with --resume or --ratings-file")
    if args.ratings_file is not None and not args.resume:
        parser.error("--ratings-file is accepted only when resuming the saved blind-review task")
    return args


def main() -> None:
    torch.set_grad_enabled(False)
    try:
        run(_parse_args())
    except EvidenceError as error:
        raise SystemExit(f"INVALID_HARNESS: {error}") from error


if __name__ == "__main__":
    main()
