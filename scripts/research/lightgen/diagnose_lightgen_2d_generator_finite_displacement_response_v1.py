"""Measure finite Generator displacements from a frozen LightGen endpoint."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image, ImageDraw

from ldm_project.v2.application.research.lightgen import canonical_semantics as canonical
from ldm_project.v2.application.research.lightgen import lightgen_2d_chen_bogt_complex_ols_training as bogt
from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
    check_resume_compatibility,
)
from scripts.research.lightgen import diagnose_lightgen_2d_generator_semantic_direction_adam_v1 as direction
from scripts.research.lightgen import diagnose_lightgen_2d_class17_mapping_v2 as base


TASK_ID = "M0_GENERATOR_FINITE_DISPLACEMENT_RESPONSE_V1"
REPORT_SCHEMA = "m0_generator_finite_displacement_response_report.v1"
PREFLIGHT_SCHEMA = "m0_generator_finite_displacement_response_preflight.v1"
BLOCK_SCHEMA = "m0_generator_finite_displacement_response_block.v1"
MANIFEST_SCHEMA = "m0_generator_finite_displacement_response_manifest.v1"
MARGIN_DEFINITION = direction.MARGIN_DEFINITION
FIT_ROWS = 200
BATCH_SIZE = 10
BATCH_COUNT = 20
FOCUS_CLASSES = (1, 7)
CONTROL_CLASSES = (0, 4, 8)
DISPLAY_CLASSES = (*FOCUS_CLASSES, *CONTROL_CLASSES)
EVALUATORS = ("E_S", "E_I", "E_C", "E_D")
LAMBDA_SCALES = (0.25, 1.0, 4.0)
ORACLE_PREFLIGHT_SCALES = (1 / 8, 1 / 16)
SIGNS = (-1, 1)
NORMALIZED_DIRECTIONS = {
    "adam_saved": "ce_adam_saved",
    "adam_zero_history": "ce_adam_zero_history",
    "adam_projected": "ce_projected_adam_saved",
    "semantic_ce": "ce_semantic_gradient_descent",
    "e_s_ce": "ce_component_gradient_descent_E_S",
    "ed_oracle_class1": None,
    "ed_oracle_class7": None,
}
STABLE_ROWS = 16
FLOAT_RTOL = 2e-5
FLOAT_ATOL = 2e-7
FINITE_RESPONSE_FLOOR = 1e-4
MIN_OUTPUT_RESERVE = 1_500_000_000
ESTIMATED_OUTPUT_BYTES = 2_000_000_000
DEFAULT_DIAGNOSTIC = Path(
    "outputs/20261004_m0_generator_semantic_direction_adam_diagnostic_v1/runs/attempt_04_margin_gradient_repair"
)
DEFAULT_OUTPUT = Path(
    "outputs/20261006_m0_generator_finite_displacement_response_v1/runs/attempt_01"
)


class FiniteResponseError(RuntimeError):
    """Invalid input or failed finite-response runtime invariant."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FiniteResponseError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _tree_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def _snapshot(module: torch.nn.Module) -> dict[str, Any]:
    return {
        "state_dict": {key: value.detach().cpu().clone() for key, value in module.state_dict().items()},
        "parameters": {key: value.detach().cpu().clone() for key, value in module.named_parameters()},
        "buffers": {key: value.detach().cpu().clone() for key, value in module.named_buffers()},
    }


def _write_json_new(path: Path, payload: Mapping[str, Any], label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"refusing to overwrite {label}: {path.name}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2))
            handle.write("\n")
        os.rename(temporary, path)
        restored = json.loads(path.read_text(encoding="utf-8"))
        _require(restored == dict(payload), f"write/read lifecycle verification failed for {path.name}")
    finally:
        temporary.unlink(missing_ok=True)


def _write_torch_new(path: Path, payload: Mapping[str, Any], label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"refusing to overwrite {label}: {path.name}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    try:
        with temporary.open("xb") as handle:
            torch.save(dict(payload), handle)
        os.rename(temporary, path)
        restored = torch.load(path, map_location="cpu", weights_only=True)
        _require(isinstance(restored, Mapping), f"write/read lifecycle payload is invalid for {path.name}")
        _require(set(restored) == set(payload), f"write/read lifecycle keys changed for {path.name}")
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_replace(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _load_inputs(root: Path, diagnostic_dir: Path) -> dict[str, Any]:
    population_path = diagnostic_dir / "input_population.json"
    report_path = diagnostic_dir / "report.json"
    _require(population_path.is_file() and report_path.is_file(), "direction diagnostic input lacks its population or final report")
    population = json.loads(population_path.read_text(encoding="utf-8"))
    parent_report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(
        parent_report.get("schema") == direction.REPORT_SCHEMA
        and parent_report.get("task_id") == direction.TASK_ID
        and parent_report.get("status") == "VALID_COMPLETED"
        and parent_report.get("classification") == "MIXED_OR_INSUFFICIENT"
        and parent_report.get("margin_derivative_definition") == MARGIN_DEFINITION,
        "finite response requires the corrected, valid v2 direction report",
    )
    _require(population.get("task_id") == direction.TASK_ID and population.get("fit_rows") == FIT_ROWS, "direction population task or fit size changed")
    review_dir = _resolve(root, population["review_dir"])
    paths, parent_population, parent_training_report, review = direction._path_inputs(
        root, _resolve(root, population["artifact_dir"]), review_dir
    )
    inputs = direction._load_inputs(root, paths, parent_population, _resolve(root, population["artifact_dir"]))
    _require(inputs["population"] == parent_population, "direction diagnostic and parent population differ")
    _require(direction._fixed_fit_population_matches(inputs), "fixed fit rows or data roles differ from the completed diagnostic")
    _require(parent_training_report.get("classification") == "NOT_SUPPORTED_WITHIN_SCOPE", "parent CE training result changed")
    _require(review.get("training_recommendation") == "NONE" and review.get("next_action") == "NONE", "zero-update review contains a follow-on authorization")
    _require(len(inputs["images"]) == FIT_ROWS and inputs["images"].shape == (FIT_ROWS, 1, 28, 28), "finite response may materialize only the 200 fit rows")
    return {
        **inputs,
        "population": population,
        "direction_report": parent_report,
        "diagnostic_dir": diagnostic_dir,
        "paths": paths,
        "parent_population": parent_population,
    }


def _load_direction_blocks(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    diagnostic_dir: Path = inputs["diagnostic_dir"]
    for batch_index in range(BATCH_COUNT):
        path = diagnostic_dir / direction._block_name("B_class_ce", batch_index)
        _require(path.is_file(), f"corrected B endpoint direction block is missing: {path.name}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        _require(
            payload.get("schema") == direction.BLOCK_SCHEMA
            and payload.get("task_id") == direction.TASK_ID
            and payload.get("margin_derivative_definition") == MARGIN_DEFINITION
            and payload.get("endpoint") == "B_class_ce"
            and payload.get("batch_index") == batch_index,
            f"direction block is not corrected v2 data: {path.name}",
        )
        block = payload.get("block")
        _require(isinstance(block, Mapping) and block.get("margin_derivative_definition") == MARGIN_DEFINITION, f"direction block contents are invalid: {path.name}")
        start = batch_index * BATCH_SIZE
        stop = start + BATCH_SIZE
        saved = inputs["readouts"]["B_class_ce"]
        expected_rows = inputs["source_indices"][start:stop]
        expected_labels = inputs["labels"][start:stop]
        _require(torch.equal(block["readouts"]["source_indices"], expected_rows), f"direction block row mapping changed: {path.name}")
        _require(torch.equal(block["readouts"]["labels"], expected_labels), f"direction block labels changed: {path.name}")
        _require(torch.equal(saved["source_indices"][start:stop], expected_rows), "parent readout rows differ from canonical fit rows")
        _require(torch.equal(saved["labels"][start:stop], expected_labels), "parent readout labels differ from canonical fit labels")
        _require(block["readouts"]["z"].shape == (BATCH_SIZE, 100), "cached OLS block has unexpected shape")
        _require(torch.isfinite(block["readouts"]["z"].real).all().item() and torch.isfinite(block["readouts"]["z"].imag).all().item(), "cached OLS block contains nonfinite values")
        for key in ("z", "raw_cmos", "target_slm_intensity"):
            _require(torch.allclose(block["readouts"][key], saved[key][start:stop], rtol=FLOAT_RTOL, atol=FLOAT_ATOL), f"corrected direction block {key} differs from the completed B endpoint readout")
        for name in ("E_S", "E_I", "E_C"):
            saved_key = f"{name.lower()}_logits"
            _require(torch.allclose(block["readouts"]["evaluator_logits"][name], saved[saved_key][start:stop], rtol=FLOAT_RTOL, atol=FLOAT_ATOL), f"corrected direction block {name} logits differ from the completed B endpoint readout")
        _require(torch.allclose(block["readouts"]["e_d_logits"], saved["e_d_logits"][start:stop], rtol=FLOAT_RTOL, atol=FLOAT_ATOL), "corrected direction block E_D logits differ from the completed B endpoint readout")
        rows = block["rows"]
        _require(len(rows) == 5 and [int(row["label"]) for row in rows] == [0, 1, 4, 7, 8], "direction block focus/control rows changed")
        expected_offsets = [offset for offset, label in enumerate(expected_labels.tolist()) if label in DISPLAY_CLASSES]
        rows_by_offset = {int(row["within_batch"]): row for row in rows}
        _require(set(rows_by_offset) == set(expected_offsets), f"direction block focus/control row offsets changed: {path.name}")
        for offset, row in rows_by_offset.items():
            _require(int(row["fit_position"]) == start + offset and int(row["source_row_index"]) == int(expected_rows[offset]) and int(row["label"]) == int(expected_labels[offset]), f"direction block gradient row mapping changed: {path.name}")
        fit_positions = block["margin_gradient_fit_positions"]
        _require(len(fit_positions) == len(set(int(value) for value in fit_positions)), "direction block has duplicate fit positions for E_D gradients")
        gradient_map = {int(position): block["margin_gradients_generator"][index] for index, position in enumerate(fit_positions)}
        expected_positions = {start + offset for offset in expected_offsets if not rows_by_offset[offset]["best_wrong_exact_tie"]}
        _require(set(gradient_map) == expected_positions, "E_D margin gradients do not match unique-maximum focus/control rows")
        result.append(dict(block))
    return result


def _equal_rms(vector: torch.Tensor, target_rms: float) -> tuple[torch.Tensor | None, dict[str, Any]]:
    if not torch.is_tensor(vector) or not torch.isfinite(vector).all():
        return None, {"available": False, "reason": "missing_or_nonfinite_direction"}
    rms = float(vector.detach().to(torch.float64).square().mean().sqrt())
    if rms == 0.0 or target_rms == 0.0:
        return None, {"available": False, "reason": "zero_direction_or_reference_rms", "source_rms": rms, "reference_rms": target_rms}
    scaled = vector * (target_rms / rms)
    return scaled, {"available": True, "source_rms": rms, "reference_rms": target_rms, "scaled_rms": float(scaled.detach().to(torch.float64).square().mean().sqrt())}


def _condition_name(candidate: str, scale: float, sign: int, *, native: bool = False) -> str:
    sign_name = "plus" if sign > 0 else "minus"
    scale_name = str(scale).replace(".", "p")
    suffix = "native" if native else "equal_rms"
    return f"{candidate}_{suffix}_lambda_{scale_name}_{sign_name}"


def _condition_plan(
    block: Mapping[str, Any],
    batch_index: int,
    *,
    include_oracle_preflight_scales: bool = False,
) -> dict[str, dict[str, Any]]:
    vectors: Mapping[str, torch.Tensor] = block["candidate_vectors_generator"]
    saved = vectors["ce_adam_saved"]
    target_rms = float(saved.detach().to(torch.float64).square().mean().sqrt())
    _require(target_rms > 0.0 and math.isfinite(target_rms), f"batch {batch_index} saved Adam candidate has invalid RMS")
    row_by_label = {int(row["label"]): row for row in block["rows"]}
    gradient_by_position = {
        int(position): block["margin_gradients_generator"][index]
        for index, position in enumerate(block["margin_gradient_fit_positions"])
    }
    result: dict[str, dict[str, Any]] = {}
    for candidate, source_key in NORMALIZED_DIRECTIONS.items():
        if source_key is None:
            label = int(candidate[-1])
            row = row_by_label[label]
            if row.get("best_wrong_exact_tie") is True:
                result[candidate] = {"available": False, "reason": "UNDECIDABLE_TIED_MAXIMUM", "source_fit_position": int(row["fit_position"])}
                continue
            gradient = gradient_by_position.get(int(row["fit_position"]))
            if gradient is None:
                result[candidate] = {"available": False, "reason": "UNDECIDABLE_MISSING_MARGIN_GRADIENT", "source_fit_position": int(row["fit_position"])}
                continue
            source = gradient.reshape_as(saved)
        else:
            source = vectors[source_key]
        scaled, metadata = _equal_rms(source, target_rms)
        if scaled is None:
            result[candidate] = {**metadata, "source_fit_position": int(row_by_label[int(candidate[-1])]["fit_position"])} if source_key is None else metadata
            continue
        result[candidate] = {**metadata, "candidate": candidate, "reference_rms": target_rms}
        scales = LAMBDA_SCALES
        if source_key is None and include_oracle_preflight_scales:
            scales = (*LAMBDA_SCALES, *ORACLE_PREFLIGHT_SCALES)
        for scale in scales:
            for sign in SIGNS:
                name = _condition_name(candidate, scale, sign)
                result[name] = {
                    "available": True,
                    "candidate": candidate,
                    "scale": scale,
                    "sign": sign,
                    "native": False,
                    "delta": scaled.mul(scale * sign).detach().cpu(),
                    "candidate_rms": target_rms,
                    "unscaled_candidate_rms": metadata["source_rms"],
                }
    native_reset = vectors["ce_adam_zero_history"]
    for sign in SIGNS:
        name = _condition_name("adam_zero_history", 1.0, sign, native=True)
        result[name] = {
            "available": bool(torch.isfinite(native_reset).all().item() and native_reset.norm().item() > 0.0),
            "candidate": "adam_zero_history",
            "scale": 1.0,
            "sign": sign,
            "native": True,
            "delta": native_reset.mul(sign).detach().cpu(),
            "candidate_rms": float(native_reset.detach().to(torch.float64).square().mean().sqrt()),
            "unscaled_candidate_rms": float(native_reset.detach().to(torch.float64).square().mean().sqrt()),
        }
    return result


def _apply_displacement(parameter: torch.nn.Parameter, baseline: torch.Tensor, delta: torch.Tensor, sign: int = 1) -> None:
    """Set a candidate from the same saved phase every time; never accumulate."""
    _require(parameter.shape == baseline.shape == delta.shape, "Generator phase and displacement shapes differ")
    with torch.no_grad():
        parameter.copy_(baseline)
        parameter.add_(delta.to(device=parameter.device, dtype=parameter.dtype), alpha=sign)


def _restore_phase(parameter: torch.nn.Parameter, baseline: torch.Tensor) -> None:
    with torch.no_grad():
        parameter.copy_(baseline)
    _require(torch.equal(parameter.detach().cpu(), baseline.detach().cpu()), "Generator phase did not restore exactly")


def _evaluator_logits(evaluators: Mapping[str, torch.nn.Module], raw: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    view = canonical.cmos_view(raw)
    logits: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name in EVALUATORS:
            output = evaluators[name](view)
            logits[name] = output[1] if name != "E_D" else output
    return view, logits


def _readout(consumer: Any, evaluators: Mapping[str, torch.nn.Module], z: torch.Tensor) -> dict[str, Any]:
    _require(z.shape == (BATCH_SIZE, 100) and torch.is_complex(z), "finite response requires one fixed complex OLS batch of shape [10,100]")
    with torch.no_grad():
        result = consumer.runtime.forward_from_ols(z.to(consumer.device))
        raw = result.raw_intensity.detach().cpu()
        view, logits = _evaluator_logits(evaluators, result.raw_intensity)
        _require(raw.shape == (BATCH_SIZE, 1, 200, 200) and torch.isfinite(raw).all().item(), "finite response raw CMOS is malformed or nonfinite")
        _require(view.shape == (BATCH_SIZE, 1, 28, 28) and torch.isfinite(view).all().item(), "finite response evaluator view is malformed or nonfinite")
        for name, values in logits.items():
            _require(values.shape == (BATCH_SIZE, 10) and torch.isfinite(values).all().item(), f"finite response {name} logits are malformed or nonfinite")
        planes = []
        fields = [("generator_input", result.generator_input_field), *[(f"generator_record_{i}", field) for i, field in enumerate(result.generator_records)], ("generator_output", result.generator_output_field)]
        for name, field in fields:
            plane = field.plane
            planes.append({
                "name": name,
                "plane_id": plane.plane_id,
                "shape": list(plane.shape),
                "pixel_pitch_m": list(plane.pixel_pitch_m),
                "wavelength_m": plane.wavelength_m,
                "z_m": plane.z_m,
                "coordinate_frame": plane.coordinate_frame,
                "power_w": bogt.physical_plane_power(field).detach().cpu(),
            })
        camera_plane = result.raw_camera.plane
        camera_area = float(camera_plane.pixel_pitch_m[0] * camera_plane.pixel_pitch_m[1])
        camera_power = result.raw_intensity.to(torch.float64).flatten(1).sum(1) * camera_area
        planes.append({
            "name": "raw_cmos",
            "plane_id": camera_plane.plane_id,
            "shape": list(camera_plane.shape),
            "pixel_pitch_m": list(camera_plane.pixel_pitch_m),
            "wavelength_m": camera_plane.wavelength_m,
            "z_m": camera_plane.z_m,
            "coordinate_frame": camera_plane.coordinate_frame,
            "integrated_intensity_w": camera_power.detach().cpu(),
        })
    return {"raw_cmos": raw, "evaluator_view": view.detach().cpu(), "evaluator_logits": {name: value.detach().cpu() for name, value in logits.items()}, "available_plane_records": planes}


def _margins(logits: torch.Tensor, labels: torch.Tensor) -> list[dict[str, Any]]:
    result = []
    for row_index, label in enumerate(labels.tolist()):
        item = direction._ed_margin_components(logits[row_index].detach().to(device="cpu"), int(label))
        result.append({
            "true_logit": float(logits[row_index, int(label)].detach().cpu()),
            "best_wrong_logit": float(item["best_wrong_logit"]),
            "margin": float(item["margin"]),
            "best_wrong_class": int(item["best_wrong_class"]),
            "best_wrong_tied_classes": list(item["best_wrong_tied_classes"]),
            "tied_maximum": bool(item["tied_maximum"]),
        })
    return result


def _positive_control_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    prediction = logits.argmax(1).cpu()
    labels = labels.cpu()
    correct = [int(((labels == class_id) & (prediction == labels)).sum()) for class_id in range(10)]
    counts = torch.bincount(labels, minlength=10)
    recall = [correct[index] / int(counts[index]) for index in range(10)]
    return {"rows": int(labels.numel()), "correct": correct, "rows_per_class": counts.tolist(), "per_class_recall": recall, "macro_recall": sum(recall) / 10.0, "prediction_counts": torch.bincount(prediction, minlength=10).tolist()}


def _sample_resource_telemetry(
    path: Path,
    device: torch.device,
    phase: str,
    *,
    progress: Mapping[str, Any] | None = None,
    started: float | None = None,
) -> dict[str, Any]:
    sample = {
        "task_id": TASK_ID,
        "phase": phase,
        "progress": dict(progress or {}),
        "elapsed_seconds_observation_only": None if started is None else time.perf_counter() - started,
        "resource": direction._resource_snapshot(device),
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    _require(sample["resource"]["global_used_fraction"] <= 0.90, "whole-device CUDA use exceeded the safety line during finite-response measurement")
    return sample


def _qualify_target_controls(controls: Mapping[str, Mapping[str, Any]]) -> bool:
    return all(
        record["macro_recall"] >= 0.90
        and all(record["per_class_recall"][class_id] >= 0.80 for class_id in FOCUS_CLASSES)
        for record in controls.values()
    )


def _record_conditions(
    consumer: Any,
    parameters: Mapping[str, torch.nn.Parameter],
    evaluators: Mapping[str, torch.nn.Module],
    inputs: Mapping[str, Any],
    source_block: Mapping[str, Any],
    batch_index: int,
    eta: float,
) -> dict[str, Any]:
    start = batch_index * BATCH_SIZE
    stop = start + BATCH_SIZE
    z = source_block["readouts"]["z"]
    labels = inputs["labels"][start:stop]
    rows = inputs["source_indices"][start:stop]
    images = inputs["images"][start:stop]
    target_raw = source_block["readouts"]["target_slm_intensity"]
    generator = parameters["generator.phase"]
    baseline_phase = generator.detach().cpu().clone()
    baseline = _readout(consumer, evaluators, z)
    optical_forward_calls = 1
    phase_intervention_count = 0
    baseline["margins"] = _margins(baseline["evaluator_logits"]["E_D"], labels)
    _require(torch.allclose(baseline["raw_cmos"], source_block["readouts"]["raw_cmos"], rtol=FLOAT_RTOL, atol=FLOAT_ATOL), f"B endpoint batch {batch_index} direct OLS baseline differs from saved production readout")
    for name in ("E_S", "E_I", "E_C", "E_D"):
        saved_logits = source_block["readouts"]["e_d_logits"] if name == "E_D" else source_block["readouts"]["evaluator_logits"][name]
        _require(torch.allclose(baseline["evaluator_logits"][name], saved_logits, rtol=FLOAT_RTOL, atol=FLOAT_ATOL), f"B endpoint batch {batch_index} direct OLS {name} logits differ from the corrected direction readout")
    gradient_by_position = {
        int(position): source_block["margin_gradients_generator"][index].detach().to(device="cpu", dtype=torch.float64)
        for index, position in enumerate(source_block["margin_gradient_fit_positions"])
    }
    plan = _condition_plan(source_block, batch_index)
    conditions: dict[str, Any] = {}
    try:
        for condition_name, spec in plan.items():
            if "delta" not in spec:
                conditions[condition_name] = {key: value for key, value in spec.items() if key != "delta"}
                continue
            delta = spec["delta"]
            try:
                _apply_displacement(generator, baseline_phase, delta)
                phase_intervention_count += 1
                displaced = _readout(consumer, evaluators, z)
                optical_forward_calls += 1
            finally:
                _restore_phase(generator, baseline_phase)
            displaced_margins = _margins(displaced["evaluator_logits"]["E_D"], labels)
            for row_index, margin in enumerate(displaced_margins):
                margin["delta_from_baseline"] = margin["margin"] - baseline["margins"][row_index]["margin"]
                margin["best_wrong_class_changed"] = margin["best_wrong_class"] != baseline["margins"][row_index]["best_wrong_class"]
                margin["response_status"] = _response_status(margin["delta_from_baseline"], eta)
            image_metrics = _row_image_metrics(displaced["raw_cmos"], target_raw, images)
            linear_predictions = []
            for row_index, label in enumerate(labels.tolist()):
                gradient = gradient_by_position.get(start + row_index) if int(label) in DISPLAY_CLASSES else None
                linear_predictions.append(None if gradient is None else float(torch.dot(gradient, delta.reshape(-1).to(torch.float64))))
            displacement_rms = float(delta.detach().to(torch.float64).square().mean().sqrt())
            conditions[condition_name] = {
                **{key: value for key, value in spec.items() if key != "delta"},
                "phase_displacement_rms": displacement_rms,
                "margins": displaced_margins,
                "delta_margin_linear_prediction": linear_predictions,
                "image_metrics": image_metrics,
                **displaced,
            }
        _restore_phase(generator, baseline_phase)
    finally:
        _restore_phase(generator, baseline_phase)
    _require(all(parameter.grad is None for parameter in parameters.values()), "finite response forward wrote a phase parameter gradient")
    return {
        "batch_index": batch_index,
        "source_row_indices": rows.clone(),
        "labels": labels.clone(),
        "ols_coefficients": z.clone(),
        "target_slm_intensity": target_raw.clone(),
        "target_images_28x28": images.clone(),
        "baseline": {**baseline, "image_metrics": _row_image_metrics(baseline["raw_cmos"], target_raw, images)},
        "conditions": conditions,
        "eta": eta,
        "phase_intervention_count": phase_intervention_count,
        "optical_forward_calls": optical_forward_calls,
    }


def _response_status(delta: float, eta: float) -> str:
    if delta > eta:
        return "BENEFICIAL"
    if delta < -eta:
        return "HARMFUL"
    return "BOUNDARY"


def _row_image_metrics(raw: torch.Tensor, target_raw: torch.Tensor, images: torch.Tensor) -> list[dict[str, Any]]:
    target_view = canonical.cmos_view(images)
    generated_view = canonical.cmos_view(raw)
    records = []
    for index in range(raw.shape[0]):
        raw_mask = target_raw[index] > 0.1
        view_mask = target_view[index] > 0.1

        def mse_pair(generated: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
            expanded = mask.expand_as(generated)
            fg_count = int(expanded.sum().item())
            bg_count = int((~expanded).sum().item())
            _require(fg_count > 0 and bg_count > 0, "target foreground or background mask is empty for a finite-response row")
            squared = (generated - target).square()
            fg = float(squared[expanded].mean()) if fg_count else float("nan")
            bg = float(squared[~expanded].mean()) if bg_count else float("nan")
            return fg, bg

        raw_fg, raw_bg = mse_pair(raw[index], target_raw[index], raw_mask)
        view_fg, view_bg = mse_pair(generated_view[index], target_view[index], view_mask)
        flat = raw[index].flatten()
        records.append({
            "raw_mse": float((raw[index] - target_raw[index]).square().mean()),
            "raw_foreground_mse": raw_fg,
            "raw_background_mse": raw_bg,
            "view_mse": float((generated_view[index] - target_view[index]).square().mean()),
            "view_foreground_mse": view_fg,
            "view_background_mse": view_bg,
            "raw_fraction_above_one": float((raw[index] > 1.0).to(torch.float64).mean()),
            "raw_minimum": float(flat.min()),
            "raw_maximum": float(flat.max()),
            "raw_zero_image": bool(flat.abs().sum() == 0.0),
            "raw_constant_image": bool(flat.std(unbiased=False) == 0.0),
        })
    return records


def _panel_view(value: torch.Tensor) -> torch.Tensor:
    _require(value.ndim == 3 and value.shape[0] == 1, "image-review panel requires one grayscale image")
    return canonical.cmos_view(value.unsqueeze(0))[0, 0]


def _finite_difference_check(positive: Mapping[str, Any], negative: Mapping[str, Any], row_index: int, predicted: float, scale: float) -> dict[str, Any]:
    plus_margin = positive["margins"][row_index]
    minus_margin = negative["margins"][row_index]
    unique = not plus_margin["tied_maximum"] and not minus_margin["tied_maximum"]
    same_wrong = plus_margin["best_wrong_class"] == minus_margin["best_wrong_class"]
    observed = 0.5 * (plus_margin["margin"] - minus_margin["margin"])
    tolerance = 0.10 * abs(predicted) + 1e-4
    return {
        "scale": scale,
        "predicted_symmetric_response": predicted,
        "observed_symmetric_response": observed,
        "absolute_error": abs(observed - predicted),
        "tolerance": tolerance,
        "unique_maximum_both_sides": unique,
        "same_maximum_wrong_class_both_sides": same_wrong,
        "passed": unique and same_wrong and abs(observed - predicted) <= tolerance,
    }


def _preflight(inputs: Mapping[str, Any], target: Path, started: float) -> dict[str, Any]:
    device = torch.device("cuda:0")
    _require(torch.cuda.is_available(), "CUDA is unavailable for the production finite-response preflight")
    runtime_probe = direction._runtime_dll_probe()
    before = direction._resource_snapshot(device)
    disk_root = target
    while not disk_root.exists():
        disk_root = disk_root.parent
    disk = shutil.disk_usage(disk_root)
    disk_observation = {
        "checked_path": disk_root.as_posix(),
        "free_bytes": int(disk.free),
        "estimated_output_bytes": ESTIMATED_OUTPUT_BYTES,
        "required_remaining_bytes": MIN_OUTPUT_RESERVE,
    }
    _require(disk_observation["free_bytes"] >= ESTIMATED_OUTPUT_BYTES + MIN_OUTPUT_RESERVE, "output volume cannot cover expected blocks and the required reserve")
    _require(before["global_used_fraction"] <= 0.90, "whole-device CUDA safety line exceeded before preflight")
    payload = inputs["payloads"]["B_class_ce"]
    model_parameters = payload["trainable_parameter_state"]
    consumer, parameters = direction._new_consumer(inputs, "B_class_ce", device)
    evaluators = base.load_evaluators(inputs["paths"], device)
    model_before = _snapshot(consumer.runtime)
    evaluator_before = {name: _snapshot(model) for name, model in evaluators.items()}
    optimizer_before = direction._cpu_tree(inputs["optimizers"]["B_class_ce"])
    images = inputs["images"]
    z_saved = inputs["readouts"]["B_class_ce"]["z"][:BATCH_SIZE]
    _require(torch.equal(model_parameters["generator.phase"], inputs["parameters"]["B_class_ce"]["generator.phase"]), "loaded B Generator state differs from validated checkpoint")
    with torch.no_grad():
        full = consumer.runtime.forward_from_image(images[:BATCH_SIZE].to(device))
        direct = consumer.runtime.forward_from_ols(z_saved.to(device))
        full_view, full_logits = _evaluator_logits(evaluators, full.raw_intensity)
        direct_view, direct_logits = _evaluator_logits(evaluators, direct.raw_intensity)
    _require(torch.allclose(full.z_ols.coefficients.detach().cpu(), z_saved, rtol=FLOAT_RTOL, atol=FLOAT_ATOL), "real Encoder OLS differs from cached fit OLS")
    _require(torch.allclose(full.raw_intensity.detach().cpu(), direct.raw_intensity.detach().cpu(), rtol=FLOAT_RTOL, atol=FLOAT_ATOL), "full Encoder path and direct OLS decoder disagree")
    for name in EVALUATORS:
        _require(torch.allclose(full_logits[name].detach().cpu(), direct_logits[name].detach().cpu(), rtol=FLOAT_RTOL, atol=FLOAT_ATOL), f"full and cached OLS evaluator logits differ for {name}")

    target_view = canonical.cmos_view(images.to(device))
    controls: dict[str, Any] = {}
    with torch.no_grad():
        for name in EVALUATORS:
            output = evaluators[name](target_view)
            logits = output[1] if name != "E_D" else output
            controls[name] = _positive_control_metrics(logits.detach().cpu(), inputs["labels"])
    controls_qualified = _qualify_target_controls(controls)
    _require(controls_qualified, "one or more frozen evaluators failed the target-image positive control")

    source_block = _load_direction_blocks(inputs)[0]
    target_labels = inputs["labels"][:BATCH_SIZE]
    baseline1 = _readout(consumer, evaluators, z_saved)
    baseline2 = _readout(consumer, evaluators, z_saved)
    repeat_difference = float((baseline1["evaluator_logits"]["E_D"] - baseline2["evaluator_logits"]["E_D"]).abs().max())
    margin_repeat_difference = max(
        abs(left["margin"] - right["margin"])
        for left, right in zip(_margins(baseline1["evaluator_logits"]["E_D"], target_labels), _margins(baseline2["evaluator_logits"]["E_D"], target_labels), strict=True)
    )
    baseline_readout_difference = float((baseline1["raw_cmos"] - baseline2["raw_cmos"]).abs().max())
    _require(torch.allclose(baseline1["raw_cmos"], baseline2["raw_cmos"], rtol=FLOAT_RTOL, atol=FLOAT_ATOL), "repeated frozen OLS baseline is not reproducible")
    eta = max(FINITE_RESPONSE_FLOOR, 5.0 * margin_repeat_difference)
    formal_plan = _condition_plan(source_block, 0)
    formal_condition_count = sum(1 for spec in formal_plan.values() if "delta" in spec)
    preflight_plan = _condition_plan(source_block, 0, include_oracle_preflight_scales=True)
    preflight_condition_count = sum(1 for spec in preflight_plan.values() if "delta" in spec)
    expected_formal_condition_count = len(NORMALIZED_DIRECTIONS) * len(LAMBDA_SCALES) * len(SIGNS) + len(SIGNS)
    _require(formal_condition_count <= expected_formal_condition_count, "formal finite-response plan contains unregistered phase interventions")
    _require(preflight_condition_count == formal_condition_count + 8, "preflight finite-difference scales are not limited to the two declared E_D controls")
    plan = preflight_plan
    preflight_phase_interventions = 0
    preflight_optical_forward_calls = 4  # Encoder, cached OLS, and two frozen baselines.
    row_by_label = {int(row["label"]): row for row in source_block["rows"]}
    finite_checks: dict[str, list[dict[str, Any]]] = {}
    for class_id in FOCUS_CLASSES:
        candidate = f"ed_oracle_class{class_id}"
        positive = plan.get(_condition_name(candidate, 1 / 8, 1))
        negative = plan.get(_condition_name(candidate, 1 / 8, -1))
        positive_small = plan.get(_condition_name(candidate, 1 / 16, 1))
        negative_small = plan.get(_condition_name(candidate, 1 / 16, -1))
        _require(all(item is not None and item.get("available") for item in (positive, negative, positive_small, negative_small)), f"class {class_id} E_D oracle finite-difference vectors are unavailable")
        source_row = row_by_label[class_id]
        local_index = int(source_row["within_batch"])
        margin_gradient = source_block["margin_gradients_generator"][
            source_block["margin_gradient_fit_positions"].index(int(source_row["fit_position"]))
        ].to(torch.float64)
        checks = []
        for scale in (1 / 8, 1 / 16):
            plus_name = _condition_name(candidate, scale, 1)
            minus_name = _condition_name(candidate, scale, -1)
            plus_spec, minus_spec = plan[plus_name], plan[minus_name]
            try:
                with torch.no_grad():
                    _apply_displacement(parameters["generator.phase"], model_before["parameters"]["generator.phase"], plus_spec["delta"])
                    preflight_phase_interventions += 1
                    plus = _readout(consumer, evaluators, z_saved)
                    preflight_optical_forward_calls += 1
                    _restore_phase(parameters["generator.phase"], model_before["parameters"]["generator.phase"])
                    _apply_displacement(parameters["generator.phase"], model_before["parameters"]["generator.phase"], minus_spec["delta"])
                    preflight_phase_interventions += 1
                    minus = _readout(consumer, evaluators, z_saved)
                    preflight_optical_forward_calls += 1
            finally:
                _restore_phase(parameters["generator.phase"], model_before["parameters"]["generator.phase"])
            plus["margins"] = _margins(plus["evaluator_logits"]["E_D"], target_labels)
            minus["margins"] = _margins(minus["evaluator_logits"]["E_D"], target_labels)
            predicted = float(torch.sum(margin_gradient * plus_spec["delta"].reshape(-1).to(torch.float64)))
            check = _finite_difference_check(plus, minus, local_index, predicted, scale)
            _require(check["passed"], f"E_D margin central difference failed for class {class_id} at lambda={scale}")
            checks.append(check)
        finite_checks[candidate] = checks

    _require(_tree_equal(model_before, _snapshot(consumer.runtime)), "CUDA preflight changed the saved B model state")
    _require(_tree_equal(optimizer_before, direction._cpu_tree(inputs["optimizers"]["B_class_ce"])), "CUDA preflight changed the source Adam state")
    _require(all(parameter.grad is None for parameter in parameters.values()), "CUDA preflight populated a phase .grad buffer")
    for name, model in evaluators.items():
        _require(_tree_equal(evaluator_before[name], _snapshot(model)), f"CUDA preflight changed frozen evaluator {name}")
        _require(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()), f"evaluator {name} crossed its frozen boundary")
    after = direction._resource_snapshot(device)
    _require(after["global_used_fraction"] <= 0.90, "whole-device CUDA safety line exceeded after preflight")
    return {
        "status": "READY",
        "task_id": TASK_ID,
        "margin_derivative_definition": MARGIN_DEFINITION,
        "cuda_runtime": runtime_probe,
        "cuda_before": before,
        "cuda_after": after,
        "disk_observation": disk_observation,
        "elapsed_seconds_observation_only": time.perf_counter() - started,
        "fixed_fit_rows_and_roles": True,
        "real_encoder_matches_cached_ols_decoder": True,
        "target_positive_controls": controls,
        "target_positive_controls_qualified": controls_qualified,
        "baseline_repeat": {"max_abs_raw_cmos_difference": baseline_readout_difference, "max_abs_e_d_logit_difference": repeat_difference, "max_abs_e_d_margin_difference": margin_repeat_difference, "eta": eta},
        "ed_oracle_finite_difference_checks": finite_checks,
        "model_state_unchanged": True,
        "source_adam_unchanged": True,
        "evaluator_states_unchanged": True,
        "parameter_grad_buffers_empty": True,
        "optimizer_updates": 0,
        "measurement_preflight_displacements": preflight_phase_interventions,
        "preflight_optical_forward_batches": preflight_optical_forward_calls,
        "preflight_optical_forward_rows": preflight_optical_forward_calls * BATCH_SIZE,
        "formal_phase_conditions_per_batch": formal_condition_count,
        "preflight_phase_conditions_per_batch": preflight_condition_count,
    }


def _row_statuses(rows: list[dict[str, Any]], condition: str, eta: float) -> dict[str, Any]:
    counts = {key: 0 for key in ("BENEFICIAL", "HARMFUL", "BOUNDARY", "UNDECIDABLE")}
    for row in rows:
        value = row.get("conditions", {}).get(condition)
        if not isinstance(value, Mapping) or value.get("available") is False:
            counts["UNDECIDABLE"] += 1
            continue
        counts[value.get("response_status", "UNDECIDABLE")] += 1
    return {**counts, "rows": len(rows), "stable_beneficial": counts["BENEFICIAL"] >= STABLE_ROWS, "stable_harmful": counts["HARMFUL"] >= STABLE_ROWS}


def _decision(rows: list[dict[str, Any]], eta: float) -> dict[str, Any]:
    candidate_keys = {
        "adam_saved": _condition_name("adam_saved", 1.0, 1),
        "adam_zero_history_native": _condition_name("adam_zero_history", 1.0, 1, native=True),
        "ed_oracle_class1": _condition_name("ed_oracle_class1", 1.0, 1),
        "ed_oracle_class7": _condition_name("ed_oracle_class7", 1.0, 1),
    }
    counts: dict[str, dict[str, Any]] = {}
    for name, key in candidate_keys.items():
        counts[name] = {str(class_id): _row_statuses([row for row in rows if row["label"] == class_id], key, eta) for class_id in (*FOCUS_CLASSES, *CONTROL_CLASSES)}

    def controls_safe(name: str) -> bool:
        return all(not counts[name][str(class_id)]["stable_harmful"] for class_id in CONTROL_CLASSES)

    native_supported = any(
        all(counts[name][str(class_id)]["stable_beneficial"] for class_id in FOCUS_CLASSES) and controls_safe(name)
        for name in ("adam_saved", "adam_zero_history_native")
    )
    oracle_supported = all(counts[f"ed_oracle_class{class_id}"][str(class_id)]["stable_beneficial"] for class_id in FOCUS_CLASSES)
    if native_supported:
        classification = "LOCAL_TRAINING_DISPLACEMENT_RESPONSE_PRESENT"
    elif oracle_supported:
        classification = "TRAINING_DISPLACEMENT_MISMATCH_WITH_LOCAL_RESPONSE"
    else:
        classification = "MIXED_OR_INSUFFICIENT"
    selected = None
    if native_supported:
        selected = "adam_saved" if all(counts["adam_saved"][str(c)]["stable_beneficial"] for c in FOCUS_CLASSES) and controls_safe("adam_saved") else "adam_zero_history_native"
    return {
        "classification": classification,
        "candidate_class_counts": counts,
        "native_direction_supported": native_supported,
        "ed_oracle_local_response_supported": oracle_supported,
        "highest_priority_native_candidate": selected,
        "stable_rows_required": STABLE_ROWS,
        "response_threshold_eta": eta,
        "training_recommendation": "NONE",
        "next_action": "NONE",
        "claim_boundary": "batch-conditional finite phase interventions on the original fit rows; no shared checkpoint accuracy, human readability, generalization, or architecture-capacity claim",
    }


def _all_condition_summaries(rows: list[dict[str, Any]], eta: float) -> dict[str, Any]:
    conditions = sorted({condition for row in rows for condition in row.get("conditions", {})})
    return {
        condition: {
            str(class_id): _row_statuses([row for row in rows if row["label"] == class_id], condition, eta)
            for class_id in (*FOCUS_CLASSES, *CONTROL_CLASSES)
        }
        for condition in conditions
    }


def _block_name(batch_index: int) -> str:
    return f"block_batch{batch_index:02d}.pt"


def _validate_source_block(payload: Mapping[str, Any], batch_index: int, inputs: Mapping[str, Any]) -> None:
    _require(
        payload.get("schema") == BLOCK_SCHEMA
        and payload.get("task_id") == TASK_ID
        and payload.get("margin_derivative_definition") == MARGIN_DEFINITION
        and payload.get("batch_index") == batch_index,
        "finite-response block schema, task or batch mismatch",
    )
    block = payload.get("block")
    start = batch_index * BATCH_SIZE
    stop = start + BATCH_SIZE
    _require(torch.equal(block["source_row_indices"], inputs["source_indices"][start:stop]), "saved finite-response block source rows changed")
    _require(torch.equal(block["labels"], inputs["labels"][start:stop]), "saved finite-response block labels changed")


def _write_block(target: Path, batch_index: int, block: Mapping[str, Any]) -> str:
    name = _block_name(batch_index)
    _write_torch_new(target / name, {
        "schema": BLOCK_SCHEMA,
        "task_id": TASK_ID,
        "margin_derivative_definition": MARGIN_DEFINITION,
        "batch_index": batch_index,
        "block": dict(block),
    }, "immutable finite Generator displacement response block")
    return name


def _load_block(path: Path, batch_index: int, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _validate_source_block(payload, batch_index, inputs)
    return payload["block"]


def _population_record(inputs: Mapping[str, Any], diagnostic_dir: Path) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "diagnostic_dir": diagnostic_dir.as_posix(),
        "source_checkpoint": inputs["paths"]["checkpoint_B_class_ce"].as_posix(),
        "source_readout": inputs["paths"]["readout_B_class_ce"].as_posix(),
        "corrected_direction_report": (diagnostic_dir / "report.json").as_posix(),
        "source_row_indices": inputs["source_indices"].tolist(),
        "labels": inputs["labels"].tolist(),
        "roles": ["fit"] * FIT_ROWS,
        "fit_rows": FIT_ROWS,
        "batch_size": BATCH_SIZE,
        "batch_count": BATCH_COUNT,
        "focus_classes": list(FOCUS_CLASSES),
        "control_classes": list(CONTROL_CLASSES),
        "evaluators": list(EVALUATORS),
        "margin_derivative_definition": MARGIN_DEFINITION,
    }


def _identity(inputs: Mapping[str, Any], population: Mapping[str, Any], output: Path) -> ResearchRunIdentity:
    payload = inputs["payloads"]["B_class_ce"]
    attempt = output.name.replace("_", "-")
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=f"{TASK_ID}-{attempt}",
        trajectory_id="frozen-B-class-CE-step2500-generator-finite-displacement",
        semantic_authority_id=direction.TASK_ID,
        data_role="200 original canonical TRAIN fit rows in 20 fixed batches; no holdout or confirmation forward",
        config_identity={
            "propagation_grid": 2048,
            "source_checkpoint_step": 2200,
            "encoder_adam_step": 2200,
            "generator_adam_step": 2500,
            "fit_rows": FIT_ROWS,
            "batch_size": BATCH_SIZE,
            "lambda_scales": list(LAMBDA_SCALES),
            "directions": list(NORMALIZED_DIRECTIONS),
            "margin_derivative_definition": MARGIN_DEFINITION,
        },
        model_identity={
            "parameter_shapes": {name: list(value.shape) for name, value in payload["trainable_parameter_state"].items()},
            "source_arm": "B_class_ce",
            "completed_updates": 300,
        },
        optimizer_identity={"class": "saved Adam tensors loaded and read-only; no optimizer constructed or stepped", "source_steps": payload["optimizer_step_by_parameter"]},
        sampler_identity={"source_indices": population["source_row_indices"], "labels": population["labels"], "roles": "fixed fit-only original rows"},
        execution_source_semantics_id=TASK_ID,
        scientific_attempt_id=f"{TASK_ID}:{attempt}",
    )


def _ready_receipt(control: ResearchRunControl, preflight: Mapping[str, Any]) -> PreflightReceipt:
    correctness_passed = (
        preflight.get("status") == "READY"
        and preflight.get("fixed_fit_rows_and_roles") is True
        and preflight.get("real_encoder_matches_cached_ols_decoder") is True
        and preflight.get("target_positive_controls_qualified") is True
        and preflight.get("parameter_grad_buffers_empty") is True
        and preflight.get("model_state_unchanged") is True
        and preflight.get("source_adam_unchanged") is True
        and preflight.get("evaluator_states_unchanged") is True
        and set(preflight.get("target_positive_controls", {})) == set(EVALUATORS)
        and set(preflight.get("ed_oracle_finite_difference_checks", {})) == {f"ed_oracle_class{class_id}" for class_id in FOCUS_CLASSES}
        and all(len(checks) == 2 and all(item.get("passed") is True for item in checks) for checks in preflight.get("ed_oracle_finite_difference_checks", {}).values())
        and preflight.get("baseline_repeat", {}).get("eta", 0) >= FINITE_RESPONSE_FLOOR
        and preflight.get("formal_phase_conditions_per_batch", 0) <= len(NORMALIZED_DIRECTIONS) * len(LAMBDA_SCALES) * len(SIGNS) + len(SIGNS)
        and preflight.get("preflight_phase_conditions_per_batch") == preflight.get("formal_phase_conditions_per_batch", -8) + 8
    )
    snapshots = (preflight.get("cuda_before", {}), preflight.get("cuda_after", {}))
    hardware_passed = all(float(item.get("global_used_fraction", 1.0)) <= 0.90 for item in snapshots)
    disk = preflight.get("disk_observation", {})
    disk_passed = int(disk.get("free_bytes", 0)) >= int(disk.get("estimated_output_bytes", 0)) + int(disk.get("required_remaining_bytes", MIN_OUTPUT_RESERVE))
    runtime_passed = preflight.get("cuda_runtime", {}).get("status") in {"already_present", "prepended", "not_applicable"}
    resources_passed = hardware_passed and disk_passed and runtime_passed
    implementation_passed = correctness_passed and preflight.get("optimizer_updates") == 0 and preflight.get("measurement_preflight_displacements") == 8 and preflight.get("preflight_optical_forward_batches") == 12
    return control.make_ready_receipt(
        correctness={"passed": correctness_passed, "status": "PASS" if correctness_passed else "BLOCKED", "real_encoder_vs_cached_ols": preflight.get("real_encoder_matches_cached_ols_decoder") is True, "four_evaluator_target_controls": preflight.get("target_positive_controls_qualified") is True, "centered_E_D_references": correctness_passed, "frozen_state": preflight.get("model_state_unchanged") is True and preflight.get("source_adam_unchanged") is True and preflight.get("evaluator_states_unchanged") is True},
        resource={"passed": resources_passed, "status": "PASS" if resources_passed else "BLOCKED", "cuda_runtime": runtime_passed, "whole_device_safety_line": 0.90, "whole_device_safety_passed": hardware_passed, "disk_passed": disk_passed, "disk": disk, "observations": {"before": preflight.get("cuda_before"), "after": preflight.get("cuda_after")}},
        implementation={"passed": implementation_passed, "status": "PASS" if implementation_passed else "BLOCKED", "selected_engine": "cuda_full_2048_frozen_generator_finite_displacement", "phase_restore_exact": correctness_passed, "optimizer_updates_zero": preflight.get("optimizer_updates") == 0},
        selected_engine="cuda_full_2048_frozen_generator_finite_displacement",
        resource_envelope={"cuda": preflight.get("cuda_after"), "disk": disk},
    )


def _flatten_metrics(target: Path, population: Mapping[str, Any], preflight: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scalar_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    phase_displacement_conditions = 0
    formal_forward_rows = 0
    expected_inputs = {"source_indices": torch.tensor(population["source_row_indices"]), "labels": torch.tensor(population["labels"])}
    candidates = [(name, name, scale) for name in NORMALIZED_DIRECTIONS for scale in LAMBDA_SCALES]
    candidates.append(("adam_zero_history_native", "adam_zero_history", 1.0))
    eta = float(preflight["baseline_repeat"]["eta"])
    for batch_index in range(BATCH_COUNT):
        block = _load_block(target / _block_name(batch_index), batch_index, expected_inputs)
        phase_conditions = {name: value for name, value in block["conditions"].items() if isinstance(value, Mapping) and "margins" in value}
        _require(block["phase_intervention_count"] == len(phase_conditions), "saved phase intervention count differs from completed readouts")
        _require(block["optical_forward_calls"] == 1 + len(phase_conditions), "saved optical forward count differs from completed baseline and interventions")
        phase_displacement_conditions += int(block["phase_intervention_count"])
        formal_forward_rows += BATCH_SIZE * int(block["optical_forward_calls"])
        start = batch_index * BATCH_SIZE
        for local in range(BATCH_SIZE):
            label = int(block["labels"][local])
            scalar: dict[str, Any] = {
                "fit_position": start + local,
                "within_batch": local,
                "batch_index": batch_index,
                "source_row_index": int(block["source_row_indices"][local]),
                "label": label,
            }
            scalar.update({"baseline_ed_margin": block["baseline"]["margins"][local]["margin"], "baseline_ed_prediction": int(block["baseline"]["evaluator_logits"]["E_D"][local].argmax())})
            baseline_image = block["baseline"]["image_metrics"][local]
            for name, key in (
                ("raw_mse", "raw_mse"),
                ("raw_foreground_mse", "raw_foreground_mse"),
                ("raw_background_mse", "raw_background_mse"),
                ("view_mse", "view_mse"),
                ("view_foreground_mse", "view_foreground_mse"),
                ("view_background_mse", "view_background_mse"),
                ("raw_clipping_fraction", "raw_fraction_above_one"),
                ("raw_minimum", "raw_minimum"),
                ("raw_maximum", "raw_maximum"),
                ("raw_zero_image", "raw_zero_image"),
                ("raw_constant_image", "raw_constant_image"),
            ):
                scalar[f"baseline_{name}"] = baseline_image[key]
            for evaluator in EVALUATORS:
                scalar[f"baseline_{evaluator}_prediction"] = int(block["baseline"]["evaluator_logits"][evaluator][local].argmax())
            decision_conditions: dict[str, Any] = {}
            for table_name, candidate, scale in candidates:
                for sign in SIGNS:
                    condition = _condition_name(candidate, scale, sign, native=table_name == "adam_zero_history_native")
                    item = block["conditions"].get(condition)
                    direction_name = "plus" if sign > 0 else "minus"
                    key = f"{table_name}_lambda_{scale:g}_{direction_name}"
                    if not isinstance(item, Mapping) or item.get("available") is False:
                        scalar[f"{key}_delta_margin"] = None
                        scalar[f"{key}_linear_prediction"] = None
                        scalar[f"{key}_linear_abs_error"] = None
                        scalar[f"{key}_status"] = "UNDECIDABLE"
                        scalar[f"{key}_wrong_class_changed"] = None
                        scalar[f"{key}_phase_rms"] = None
                        for image_key in ("raw_mse", "raw_fg_mse", "raw_bg_mse", "view_mse", "view_fg_mse", "view_bg_mse", "raw_clipping_fraction", "raw_minimum", "raw_maximum", "raw_zero_image", "raw_constant_image"):
                            scalar[f"{key}_{image_key}"] = None
                        for evaluator in EVALUATORS:
                            scalar[f"{key}_{evaluator}_prediction"] = None
                        decision_conditions[condition] = {"response_status": "UNDECIDABLE"}
                        continue
                    margin = item["margins"][local]
                    scalar[f"{key}_delta_margin"] = margin["delta_from_baseline"]
                    linear_prediction = item["delta_margin_linear_prediction"][local]
                    scalar[f"{key}_linear_prediction"] = linear_prediction
                    scalar[f"{key}_linear_abs_error"] = None if linear_prediction is None else abs(margin["delta_from_baseline"] - linear_prediction)
                    scalar[f"{key}_status"] = margin.get("response_status", _response_status(margin["delta_from_baseline"], eta))
                    scalar[f"{key}_wrong_class_changed"] = margin["best_wrong_class_changed"]
                    scalar[f"{key}_phase_rms"] = item["phase_displacement_rms"]
                    image = item["image_metrics"][local]
                    for image_key, source_key in (
                        ("raw_mse", "raw_mse"),
                        ("raw_fg_mse", "raw_foreground_mse"),
                        ("raw_bg_mse", "raw_background_mse"),
                        ("view_mse", "view_mse"),
                        ("view_fg_mse", "view_foreground_mse"),
                        ("view_bg_mse", "view_background_mse"),
                        ("raw_clipping_fraction", "raw_fraction_above_one"),
                        ("raw_minimum", "raw_minimum"),
                        ("raw_maximum", "raw_maximum"),
                        ("raw_zero_image", "raw_zero_image"),
                        ("raw_constant_image", "raw_constant_image"),
                    ):
                        scalar[f"{key}_{image_key}"] = image[source_key]
                    decision_conditions[condition] = {"response_status": scalar[f"{key}_status"]}
                    for evaluator in EVALUATORS:
                        scalar[f"{key}_{evaluator}_prediction"] = int(item["evaluator_logits"][evaluator][local].argmax())
            decision_rows.append({"label": label, "conditions": decision_conditions})
            scalar_rows.append(scalar)
    decision = _decision(decision_rows, eta)
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "status": "VALID_COMPLETED",
        "classification": decision["classification"],
        "research_run_identity": json.loads((target / "resume_manifest.json").read_text(encoding="utf-8"))["identity"],
        "scientific_updates": 0,
        "phase_displacement_conditions": phase_displacement_conditions,
        "formal_measurement_forward_rows": formal_forward_rows,
        "optical_forward_rows_including_preflight": formal_forward_rows + int(preflight.get("preflight_optical_forward_rows", 0)),
        "margin_derivative_definition": MARGIN_DEFINITION,
        "finite_difference_threshold": {"formula": "max(1e-4, 5*maximum repeated E_D margin difference)", "eta": eta},
        "scope": {"endpoint": "B_class_ce", "encoder_adam_step": 2200, "generator_adam_step": 2500, "fit_rows": FIT_ROWS, "batch_size": BATCH_SIZE, "batch_count": BATCH_COUNT, "training_updates": 0, "e_d_gradient_use": "saved finite-response local controls only; not used for training candidate construction", "observation_or_confirmation_images_opened": False, "batch_conditional_statistics": True},
        "preflight": json.loads((target / "preflight.json").read_text(encoding="utf-8"))["preflight"],
        "decision": decision,
        "finite_response_by_condition_and_class": _all_condition_summaries(decision_rows, eta),
        "training_recommendation": "NONE",
        "next_action": "NONE",
        "row_metrics": "row_finite_response_metrics.csv",
        "image_review": _write_image_review(target),
        "elapsed_seconds_observation_only": None,
    }
    return report, scalar_rows


def _write_image_review(target: Path) -> dict[str, Any]:
    population = json.loads((target / "input_population.json").read_text(encoding="utf-8"))
    width, scale = 10, 2
    panel = 28 * scale
    cell_width = panel * 3 + 28
    cell_height = panel + 44
    display_rows = (FIT_ROWS * len(DISPLAY_CLASSES) // 10 + width - 1) // width
    canvas = Image.new("RGB", (width * cell_width, display_rows * cell_height), color=(248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    shown: list[int] = []
    display_index = 0
    for batch_index in range(BATCH_COUNT):
        block = torch.load(target / _block_name(batch_index), map_location="cpu", weights_only=True)["block"]
        baseline = block["baseline"]["raw_cmos"]
        candidate_key = _condition_name("adam_saved", 1.0, 1)
        candidate = block["conditions"].get(candidate_key)
        candidate_raw = candidate["raw_cmos"] if isinstance(candidate, Mapping) and candidate.get("available") is not False else baseline
        for local in range(BATCH_SIZE):
            position = batch_index * BATCH_SIZE + local
            if int(population["labels"][position]) not in DISPLAY_CLASSES:
                continue
            x = (display_index % width) * cell_width
            y = (display_index // width) * cell_height
            label = int(population["labels"][position])
            source_index = int(population["source_row_indices"][position])
            draw.text((x, y), f"class {label} | fit {position} | source {source_index}", fill=(0, 0, 0))
            views = (
                _panel_view(block["target_images_28x28"][local]),
                _panel_view(baseline[local]),
                _panel_view(candidate_raw[local]),
            )
            for panel_index, view in enumerate(views):
                pixels = view.detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
                image = Image.fromarray(pixels, mode="L").resize((panel, panel), Image.Resampling.NEAREST).convert("RGB")
                draw.text((x + panel_index * (panel + 4), y + 13), ("target", "baseline", "saved Adam +1")[panel_index], fill=(40, 40, 40))
                canvas.paste(image, (x + panel_index * (panel + 4), y + 32))
            shown.append(source_index)
            display_index += 1
    path = target / "finite_response_focus_and_controls.png"
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=True)
    image_bytes = buffer.getvalue()
    if path.exists():
        _require(path.read_bytes() == image_bytes, "existing image review differs from immutable measurement blocks")
    else:
        path.write_bytes(image_bytes)
    return {"path": path.name, "rows_displayed": display_index, "source_row_indices": shown, "classes": list(DISPLAY_CLASSES), "panel_order": ["target", "baseline", "saved Adam lambda +1"], "view": "canonical clamp[0,1] then adaptive average pool to 28x28", "fixed_range": [0.0, 1.0], "per_image_rescaling": False}


def _write_metrics_csv(target: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(rows[0])
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    path = target / "row_finite_response_metrics.csv"
    content = buffer.getvalue().encode("utf-8")
    if path.exists():
        _require(path.read_bytes() == content, "saved finite response table differs from immutable blocks")
    else:
        path.write_bytes(content)


def _finalize(target: Path, preflight: Mapping[str, Any], started: float) -> dict[str, Any]:
    population = json.loads((target / "input_population.json").read_text(encoding="utf-8"))
    report, scalar_rows = _flatten_metrics(target, population, preflight)
    _write_metrics_csv(target, scalar_rows)
    report["elapsed_seconds_observation_only"] = time.perf_counter() - started
    report["evidence_artifacts"] = {
        "population": "input_population.json",
        "preflight": "preflight.json",
        "manifest": "resume_manifest.json",
        "blocks": [_block_name(index) for index in range(BATCH_COUNT)],
        "row_metrics": "row_finite_response_metrics.csv",
        "image_review": "finite_response_focus_and_controls.png",
    }
    return report


def _identity_and_population(inputs: Mapping[str, Any], diagnostic_dir: Path, target: Path) -> tuple[ResearchRunIdentity, dict[str, Any]]:
    population = _population_record(inputs, diagnostic_dir)
    return _identity(inputs, population, target), population


def _validate_manifest(path: Path, target: Path, identity: ResearchRunIdentity, population: Mapping[str, Any]) -> dict[str, Any]:
    _require(path.resolve() == (target / "resume_manifest.json").resolve() and path.is_file(), "resume requires the explicit task-owned resume_manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == MANIFEST_SCHEMA and manifest.get("task_id") == TASK_ID and manifest.get("margin_derivative_definition") == MARGIN_DEFINITION, "resume manifest schema or task semantics mismatch")
    _require(manifest.get("population") == dict(population), "resume fit rows, labels or roles changed")
    check_resume_compatibility(identity, manifest["identity"])
    _require(manifest.get("status") in {"RUNNING", "INTERRUPTED", "COMPLETED"}, "resume manifest status is invalid")
    return manifest


def _terminal_report(target: Path, manifest: Mapping[str, Any], identity: ResearchRunIdentity) -> dict[str, Any] | None:
    report_path = target / "report.json"
    if manifest.get("status") == "COMPLETED":
        _require(report_path.is_file(), "completed manifest lacks report.json")
    elif not report_path.is_file():
        return None
    expected_blocks = {_block_name(index) for index in range(BATCH_COUNT)}
    _require(expected_blocks.issubset(set(manifest.get("completed_blocks", []))) and all((target / name).is_file() for name in expected_blocks), "terminal report precedes completed immutable blocks")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report.get("schema") == REPORT_SCHEMA and report.get("task_id") == TASK_ID and report.get("status") == "VALID_COMPLETED" and report.get("research_run_identity") == identity.to_dict(), "saved terminal report is invalid")
    if manifest.get("status") != "COMPLETED":
        manifest = dict(manifest)
        manifest["status"] = "COMPLETED"
        _write_json_replace(target / "resume_manifest.json", manifest)
    return report


def _manifest_block(path: Path, batch_index: int, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _validate_source_block(payload, batch_index, inputs)
    return payload["block"]


def _fixed_population_inputs(inputs: Mapping[str, Any], diagnostic_dir: Path) -> dict[str, Any]:
    rows = inputs["source_indices"].tolist()
    labels = inputs["labels"].tolist()
    return {
        "task_id": TASK_ID,
        "diagnostic_dir": diagnostic_dir.as_posix(),
        "source_checkpoint": inputs["paths"]["checkpoint_B_class_ce"].as_posix(),
        "source_readout": inputs["paths"]["readout_B_class_ce"].as_posix(),
        "corrected_direction_report": (diagnostic_dir / "report.json").as_posix(),
        "source_row_indices": rows,
        "labels": labels,
        "roles": ["fit"] * FIT_ROWS,
        "fit_rows": FIT_ROWS,
        "batch_size": BATCH_SIZE,
        "batch_count": BATCH_COUNT,
        "focus_classes": list(FOCUS_CLASSES),
        "control_classes": list(CONTROL_CLASSES),
        "evaluators": list(EVALUATORS),
        "margin_derivative_definition": MARGIN_DEFINITION,
    }


def _run_measurement(arguments: argparse.Namespace) -> dict[str, Any]:
    root = _root()
    diagnostic_dir = _resolve(root, arguments.diagnostic_dir)
    target = _resolve(root, arguments.output_dir)
    output_root = (root / "outputs").resolve()
    _require(target != output_root and output_root in target.parents, "output must be a named child of outputs/")
    _require(not (arguments.preflight_only and arguments.resume_manifest), "preflight-only cannot resume")
    _require(arguments.resume_manifest is not None or not target.exists(), "output exists; use its explicit resume manifest")
    inputs = _load_inputs(root, diagnostic_dir)
    direction_blocks = _load_direction_blocks(inputs)
    population = _fixed_population_inputs(inputs, diagnostic_dir)
    identity = _identity(inputs, population, target)
    existing = None
    if arguments.resume_manifest is not None:
        _require(target.is_dir(), "resume output directory is missing")
        existing = _validate_manifest(_resolve(root, arguments.resume_manifest), target, identity, population)
        terminal = _terminal_report(target, existing, identity)
        if terminal is not None:
            return terminal
    started = time.perf_counter()
    if arguments.preflight_only:
        _require(not target.exists(), "standalone preflight output must be a new explicit directory")
        target.mkdir(parents=True, exist_ok=False)
    elif existing is None:
        target.parent.mkdir(parents=True, exist_ok=True)

    if existing is None:
        preflight = _preflight(inputs, target, started)
        run_control = ResearchRunControl(identity=identity, authorization_id=f"{TASK_ID}:authorization:01")
        receipt = _ready_receipt(run_control, preflight)
        if receipt.status != "READY":
            failure = {"task_id": TASK_ID, "status": "INVALID_HARNESS", "classification": None, "training_updates": 0, "reason": "finite response preflight receipt is blocked", "next_action": "NONE", "preflight": preflight}
            if not target.exists():
                target.mkdir(parents=True, exist_ok=False)
            _write_json_new(target / "failure.json", failure, "blocked finite displacement preflight")
            return failure
        run_control.mark_preflight_ready(receipt)
        if arguments.preflight_only:
            _write_json_new(target / "input_population.json", population, "fixed finite response fit population")
            _write_json_new(target / "preflight.json", {"task_id": TASK_ID, "status": "READY", "receipt": receipt.to_dict(), "preflight": preflight}, "finite response production preflight")
            return {"task_id": TASK_ID, "status": "READY", "output_dir": target.as_posix(), "scientific_updates": 0, "next_action": "NONE"}
        run_control.start_execution()
        run_control.open_scientific_attempt(target, identity.scientific_attempt_id, receipt)
        _write_json_new(target / "input_population.json", population, "fixed canonical TRAIN fit rows and explicit B endpoint")
        _write_json_new(target / "preflight.json", {"task_id": TASK_ID, "status": "READY", "receipt": receipt.to_dict(), "preflight": preflight}, "READY zero-update finite response preflight")
        manifest = {"schema": MANIFEST_SCHEMA, "task_id": TASK_ID, "margin_derivative_definition": MARGIN_DEFINITION, "identity": identity.to_dict(), "population": population, "receipt": receipt.to_dict(), "preflight": preflight, "completed_blocks": [], "status": "RUNNING"}
        _write_json_replace(target / "resume_manifest.json", manifest)
    else:
        saved_preflight = json.loads((target / "preflight.json").read_text(encoding="utf-8"))
        _require(saved_preflight.get("status") == "READY", "resume lacks a READY preflight")
        receipt = PreflightReceipt.from_mapping(saved_preflight["receipt"])
        preflight = existing["preflight"]
        run_control = ResearchRunControl(identity=identity, authorization_id=receipt.authorization_id)
        run_control.mark_preflight_ready(receipt)
        run_control.start_execution()
        run_control.open_scientific_attempt(target, identity.scientific_attempt_id, receipt, resume=True)
        manifest = dict(existing)
        manifest["status"] = "RUNNING"
        _write_json_replace(target / "resume_manifest.json", manifest)

    device = torch.device("cuda:0")
    runtime_probe = direction._runtime_dll_probe()
    _require(runtime_probe.get("status") in {"already_present", "prepended", "not_applicable"}, "CUDA runtime DLL readiness changed")
    evaluators = base.load_evaluators(inputs["paths"], device)
    evaluator_before = {name: _snapshot(model) for name, model in evaluators.items()}
    consumer, parameters = direction._new_consumer(inputs, "B_class_ce", device)
    runtime_before = _snapshot(consumer.runtime)
    optimizer_before = direction._cpu_tree(inputs["optimizers"]["B_class_ce"])
    completed = set(manifest.get("completed_blocks", []))
    try:
        for batch_index, source_block in enumerate(direction_blocks):
            name = _block_name(batch_index)
            block_path = target / name
            if name in completed:
                block = _manifest_block(block_path, batch_index, inputs)
            elif block_path.exists():
                block = _manifest_block(block_path, batch_index, inputs)
                completed.add(name)
                manifest["completed_blocks"] = sorted(completed)
                _write_json_replace(target / "resume_manifest.json", manifest)
            else:
                current_report = json.loads((target / "preflight.json").read_text(encoding="utf-8"))["preflight"]
                eta = float(current_report["baseline_repeat"]["eta"])
                block = _record_conditions(consumer, parameters, evaluators, inputs, source_block, batch_index, eta)
                _require(_tree_equal(runtime_before, _snapshot(consumer.runtime)), f"batch {batch_index} failed to restore B runtime state")
                _require(_tree_equal(optimizer_before, direction._cpu_tree(inputs["optimizers"]["B_class_ce"])), "finite response changed saved Adam state")
                name = _write_block(target, batch_index, block)
                completed.add(name)
                manifest["completed_blocks"] = sorted(completed)
                _write_json_replace(target / "resume_manifest.json", manifest)
            del block
            if batch_index % 4 == 3:
                _sample_resource_telemetry(target / "resource_telemetry_finite_response.jsonl", device, "finite_response_progress", progress={"completed_batches": batch_index + 1}, started=started)
        _require(_tree_equal(runtime_before, _snapshot(consumer.runtime)), "full finite response changed B model parameters or scientific buffers")
        _require(_tree_equal(optimizer_before, direction._cpu_tree(inputs["optimizers"]["B_class_ce"])), "full finite response changed saved Adam")
        _require(all(parameter.grad is None for parameter in parameters.values()), "finite response left phase gradients populated")
        for name, evaluator in evaluators.items():
            _require(_tree_equal(evaluator_before[name], _snapshot(evaluator)), f"finite response changed evaluator {name}")
            _require(all(parameter.grad is None and not parameter.requires_grad for parameter in evaluator.parameters()), f"evaluator {name} was not frozen")
    except KeyboardInterrupt:
        manifest["status"] = "INTERRUPTED"
        _write_json_replace(target / "resume_manifest.json", manifest)
        run_control.interrupted()
        raise
    report = _finalize(target, preflight, started)
    _write_json_new(target / "report.json", report, "finite Generator displacement response decision")
    manifest["status"] = "COMPLETED"
    _write_json_replace(target / "resume_manifest.json", manifest)
    run_control.complete()
    _sample_resource_telemetry(target / "resource_telemetry_finite_response.jsonl", device, "finite_response_complete", progress={"classification": report["classification"]}, started=started)
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    try:
        result = _run_measurement(arguments)
    except (FiniteResponseError, direction.DirectionDiagnosticError, base.DiagnosticError, OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        target = _resolve(_root(), arguments.output_dir)
        failure = {"task_id": TASK_ID, "status": "INVALID_HARNESS", "classification": None, "training_updates": 0, "reason": f"{type(error).__name__}: {error}", "next_action": "NONE"}
        try:
            output_root = (_root() / "outputs").resolve()
            if target != output_root and output_root in target.parents and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.mkdir(exist_ok=False)
            if target.is_dir() and not (target / "report.json").exists() and not (target / "failure.json").exists():
                _write_json_new(target / "failure.json", failure, "finite response engineering failure; no scientific classification")
        except OSError:
            pass
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result.get("status") == "INVALID_HARNESS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
