"""Measure common frozen Generator displacements across a fixed TRAIN population."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image, ImageDraw

from ldm_project.v2.application.research.lightgen import canonical_semantics as canonical
from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
    check_resume_compatibility,
)
from scripts.research.lightgen import diagnose_lightgen_2d_class17_mapping_v2 as base
from scripts.research.lightgen import diagnose_lightgen_2d_generator_finite_displacement_response_v1 as finite
from scripts.research.lightgen import diagnose_lightgen_2d_generator_semantic_direction_adam_v1 as direction


TASK_ID = "M0_GENERATOR_SHARED_DIRECTION_TRANSFER_V1"
REPORT_SCHEMA = "m0_generator_shared_direction_transfer_report.v1"
MANIFEST_SCHEMA = "m0_generator_shared_direction_transfer_manifest.v1"
BLOCK_SCHEMA = "m0_generator_shared_direction_transfer_block.v1"
MARGIN_DEFINITION = direction.MARGIN_DEFINITION
FLOAT_RTOL = 2e-5
FLOAT_ATOL = 2e-7
FIT_ROWS = 200
HISTORIC_ROWS = 100
CONFIRMATION_ROWS = 300
TOTAL_ROWS = FIT_ROWS + HISTORIC_ROWS + CONFIRMATION_ROWS
BATCH_SIZE = 10
CHUNK_SIZE = 100
CHUNK_COUNT = TOTAL_ROWS // CHUNK_SIZE
EVALUATORS = ("E_S", "E_I", "E_C", "E_D")
FOCUS_CLASSES = (1, 7)
CONTROL_CLASSES = (0, 4, 8)
DISPLAY_CLASSES = (*FOCUS_CLASSES, *CONTROL_CLASSES)
EQUAL_RMS_NAMES = ("S_GD", "F_GD", "S_ADAM", "F_ADAM")
NATIVE_NAME = "S_ADAM_native"
SIGNS = (-1, 1)
ETA_FLOOR = 1e-4
STABLE_ROWS = 24
ESTIMATED_OUTPUT_BYTES = 2_500_000_000
DEFAULT_ARTIFACT = Path("outputs/20261004_m0_generator_semantic_target_comparison_v1/runs/attempt_01")
DEFAULT_DIRECTION = Path("outputs/20261004_m0_generator_semantic_direction_adam_diagnostic_v1/runs/attempt_04_margin_gradient_repair")
DEFAULT_FINITE = Path("outputs/20261006_m0_generator_finite_displacement_response_v1/runs/attempt_03")
DEFAULT_OUTPUT = Path("outputs/20261006_m0_generator_shared_direction_transfer_v1/runs/attempt_01")


class SharedDirectionError(RuntimeError):
    """Input, preflight, or frozen-measurement invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SharedDirectionError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _tree_equal(left: Any, right: Any) -> bool:
    return finite._tree_equal(left, right)


def _resource_observation(device: torch.device) -> dict[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    used_bytes = total_bytes - free_bytes
    return {
        "device": str(device),
        "total_bytes": int(total_bytes),
        "free_bytes": int(free_bytes),
        "global_used_bytes": int(used_bytes),
        "global_used_fraction": used_bytes / total_bytes if total_bytes else None,
        "pytorch_allocated_bytes": int(allocated),
        "pytorch_reserved_bytes": int(reserved),
    }


def _sample_resource_telemetry(path: Path, device: torch.device, phase: str, *, progress: Mapping[str, Any], started: float) -> dict[str, Any]:
    sample = {
        "task_id": TASK_ID,
        "phase": phase,
        "progress": dict(progress),
        "elapsed_seconds_observation_only": time.perf_counter() - started,
        "resource": _resource_observation(device),
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    return sample


def _mean_vector(vectors: Sequence[torch.Tensor], name: str) -> torch.Tensor:
    _require(bool(vectors), f"{name} has no batch vectors")
    first = vectors[0]
    _require(all(value.shape == first.shape and value.dtype == first.dtype for value in vectors), f"{name} batch vector geometry or dtype changed")
    _require(all(bool(torch.isfinite(value).all()) for value in vectors), f"{name} contains a nonfinite batch vector")
    return torch.stack([value.detach().to(device="cpu", dtype=torch.float64) for value in vectors]).mean(0).to(first.dtype)


def _rms(value: torch.Tensor) -> float:
    return float(value.detach().to(device="cpu", dtype=torch.float64).square().mean().sqrt())


def _build_shared_directions(blocks: Sequence[Mapping[str, Any]], optimizer_state: Mapping[str, Any]) -> dict[str, Any]:
    """Average the persisted gradients first, then form each Adam candidate once."""
    _require(len(blocks) == 20, "shared directions require the original twenty fit batches")
    semantic_steps: list[torch.Tensor] = []
    full_steps: list[torch.Tensor] = []
    auxiliary_steps: list[torch.Tensor] = []
    residual_max = 0.0
    for index, block in enumerate(blocks):
        vectors = block["candidate_vectors_generator"]
        semantic = vectors["ce_semantic_gradient_descent"]
        full = vectors["ce_gradient_descent"]
        auxiliary = vectors["ce_bogt_auxiliary_gradient_descent"]
        _require(semantic.shape == full.shape == auxiliary.shape, f"batch {index} direction components have different shapes")
        residual = float((full.to(torch.float64) - semantic.to(torch.float64) - auxiliary.to(torch.float64)).abs().max())
        residual_max = max(residual_max, residual)
        _require(torch.allclose(full, semantic + auxiliary, rtol=2e-5, atol=1e-10), f"batch {index} persisted BOGT and semantic directions do not sum to the full direction")
        semantic_steps.append(semantic)
        full_steps.append(full)
        auxiliary_steps.append(auxiliary)

    semantic_descent = _mean_vector(semantic_steps, "semantic descent direction")
    full_descent = _mean_vector(full_steps, "full descent direction")
    auxiliary_descent = _mean_vector(auxiliary_steps, "BOGT auxiliary descent direction")
    _require(torch.allclose(full_descent, semantic_descent + auxiliary_descent, rtol=2e-5, atol=1e-10), "mean full direction does not decompose into mean semantic and BOGT directions")
    generator_gradient_semantic = -semantic_descent
    generator_gradient_full = -full_descent
    directions = {
        "S_GD": semantic_descent,
        "F_GD": full_descent,
        "S_ADAM": direction._adam_candidate(generator_gradient_semantic, optimizer_state, variant="saved"),
        "F_ADAM": direction._adam_candidate(generator_gradient_full, optimizer_state, variant="saved"),
    }
    reference_rms = _rms(directions["F_ADAM"])
    _require(math.isfinite(reference_rms), "F-Adam shared candidate has invalid RMS")
    equal_rms: dict[str, torch.Tensor | None] = {}
    source_rms: dict[str, float] = {}
    for name, vector in directions.items():
        current = _rms(vector)
        _require(math.isfinite(current), f"{name} shared candidate has invalid RMS")
        source_rms[name] = current
        equal_rms[name] = None if current == 0.0 or reference_rms == 0.0 else (vector.to(torch.float64) * (reference_rms / current)).to(vector.dtype)
    return {
        "semantic_gradient_mean": generator_gradient_semantic,
        "full_gradient_mean": generator_gradient_full,
        "auxiliary_gradient_mean": -auxiliary_descent,
        "batch_mean_descent_directions": directions,
        "equal_rms_directions": equal_rms,
        "source_rms": source_rms,
        "reference_rms": reference_rms,
        "batch_decomposition_max_abs_residual": residual_max,
        "mean_decomposition_max_abs_residual": float((full_descent.to(torch.float64) - semantic_descent.to(torch.float64) - auxiliary_descent.to(torch.float64)).abs().max()),
        "adam_applied_once_to_mean_gradient": True,
        "mean_of_batch_adam_candidates_used": False,
        "source_batch_count": len(blocks),
    }


def _condition_vectors(directions: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in EQUAL_RMS_NAMES:
        vector = directions["equal_rms_directions"][name]
        if vector is None:
            for sign in SIGNS:
                key = f"{name}_equal_rms_{'plus' if sign > 0 else 'minus'}1"
                result[key] = {"candidate": name, "sign": sign, "native_step": False, "available": False, "reason": "UNDECIDABLE_ZERO_DIRECTION_OR_REFERENCE_RMS"}
            continue
        for sign in SIGNS:
            key = f"{name}_equal_rms_{'plus' if sign > 0 else 'minus'}1"
            result[key] = {"candidate": name, "sign": sign, "native_step": False, "available": True, "planned_delta": vector.mul(sign).detach().cpu()}
    source = directions["batch_mean_descent_directions"]["S_ADAM"]
    for sign in SIGNS:
        key = f"{NATIVE_NAME}_{'plus' if sign > 0 else 'minus'}1"
        result[key] = ({"candidate": NATIVE_NAME, "sign": sign, "native_step": True, "available": False, "reason": "UNDECIDABLE_ZERO_DIRECTION"} if directions["source_rms"]["S_ADAM"] == 0.0 else {"candidate": NATIVE_NAME, "sign": sign, "native_step": True, "available": True, "planned_delta": source.mul(sign).detach().cpu()})
    _require(len(result) == 10, "shared displacement set must contain the ten declared signed conditions")
    return result


def _load_inputs(root: Path, artifact_dir: Path, direction_dir: Path, finite_dir: Path) -> dict[str, Any]:
    for path, name in ((artifact_dir, "parent CE"), (direction_dir, "corrected direction"), (finite_dir, "finite response")):
        _require(path.is_dir(), f"explicit {name} input directory is missing: {path}")
    parent_population = json.loads((artifact_dir / "input_population.json").read_text(encoding="utf-8"))
    parent_report = json.loads((artifact_dir / "report.json").read_text(encoding="utf-8"))
    _require(parent_report.get("task_id") == "M0_GENERATOR_SEMANTIC_TARGET_COMPARISON_V1" and parent_report.get("status") == "VALID_FAILED" and parent_report.get("classification") == "NOT_SUPPORTED_WITHIN_SCOPE", "parent CE experiment no longer has its declared terminal result")
    _require(len(parent_population.get("source_row_indices", [])) == TOTAL_ROWS and len(parent_population.get("labels", [])) == TOTAL_ROWS, "parent population must preserve all six hundred rows")
    _require(parent_population.get("original_fit_rows") == FIT_ROWS and parent_population.get("historic_holdout_rows") == HISTORIC_ROWS and parent_population.get("new_confirmation_rows") == CONFIRMATION_ROWS, "parent population roles or sizes changed")
    _require(parent_population.get("roles") == ["fit"] * FIT_ROWS + ["historic_holdout"] * HISTORIC_ROWS + ["confirmation"] * CONFIRMATION_ROWS, "parent population row roles changed")
    _require(torch.bincount(torch.tensor(parent_population["labels"]), minlength=10).tolist() == [60] * 10, "fixed population is not sixty rows per class")
    _require(torch.bincount(torch.tensor(parent_population["labels"][FIT_ROWS:FIT_ROWS + HISTORIC_ROWS]), minlength=10).tolist() == [10] * 10, "historic comparison role is not ten rows per class")
    _require(torch.bincount(torch.tensor(parent_population["labels"][FIT_ROWS + HISTORIC_ROWS:]), minlength=10).tolist() == [30] * 10, "confirmation role is not thirty rows per class")

    direction_population = json.loads((direction_dir / "input_population.json").read_text(encoding="utf-8"))
    direction_report = json.loads((direction_dir / "report.json").read_text(encoding="utf-8"))
    _require(_resolve(root, direction_population.get("artifact_dir", "")) == artifact_dir, "direction diagnostic does not consume the explicit parent CE directory")
    _require(direction_report.get("task_id") == direction.TASK_ID and direction_report.get("status") == "VALID_COMPLETED" and direction_report.get("classification") == "MIXED_OR_INSUFFICIENT" and direction_report.get("margin_derivative_definition") == MARGIN_DEFINITION, "corrected v2 direction diagnostic is not a valid declared input")

    finite_population = json.loads((finite_dir / "input_population.json").read_text(encoding="utf-8"))
    finite_report = json.loads((finite_dir / "report.json").read_text(encoding="utf-8"))
    _require(_resolve(root, finite_population.get("diagnostic_dir", "")) == direction_dir, "finite-response input does not consume the explicit corrected direction directory")
    _require(finite_report.get("task_id") == finite.TASK_ID and finite_report.get("status") == "VALID_COMPLETED" and finite_report.get("classification") == "TRAINING_DISPLACEMENT_MISMATCH_WITH_LOCAL_RESPONSE", "finite-response input is not the declared valid attempt_03 result")
    _require(finite_population.get("margin_derivative_definition") == MARGIN_DEFINITION, "finite-response artifact uses a different margin derivative contract")

    inputs = finite._load_inputs(root, direction_dir)
    _require(inputs["parent_population"] == parent_population, "direction diagnostic and CE parent populations differ")
    rows = torch.tensor(parent_population["source_row_indices"], dtype=torch.long)
    labels = torch.tensor(parent_population["labels"], dtype=torch.long)
    _require(torch.equal(rows[:FIT_ROWS], inputs["source_indices"]) and torch.equal(labels[:FIT_ROWS], inputs["labels"]), "fit rows in the three parent artifacts are not identical")
    _require(inputs["paths"]["checkpoint_B_class_ce"].resolve() == (artifact_dir / "arm_b_step0300.pt").resolve(), "explicit B endpoint checkpoint path changed")
    _require(inputs["paths"]["readout_B_class_ce"].resolve() == (artifact_dir / "readouts_B_class_ce.pt").resolve(), "explicit B endpoint readout path changed")

    train_payload = torch.load(inputs["paths"]["training_data"], map_location="cpu", weights_only=True, mmap=True)
    source_images, source_labels = base._read_image_labels(train_payload)
    _require(torch.equal(source_labels.index_select(0, rows), labels), "saved six-hundred row labels differ from the canonical source data")
    train_indices, _ = base.stratified_mnist_indices(source_labels, seed=1234, validation_per_label=500)
    train_mask = torch.zeros(60_000, dtype=torch.bool)
    train_mask[train_indices] = True
    _require(bool(train_mask.index_select(0, rows).all()), "one or more fixed rows are outside canonical TRAIN")
    images = source_images.index_select(0, rows).unsqueeze(1).to(torch.float32).div_(255.0).contiguous()
    _require(tuple(images.shape) == (TOTAL_ROWS, 1, 28, 28) and bool(torch.isfinite(images).all()), "materialized target images are malformed or nonfinite")
    del source_images, train_payload
    inputs["images"] = images[:FIT_ROWS]

    saved = inputs["readouts"]["B_class_ce"]
    _require(torch.equal(saved["source_indices"], rows) and torch.equal(saved["labels"], labels), "saved B readout row order or labels differ from the fixed population")
    for field, expected in (("z", (TOTAL_ROWS, 100)), ("raw_cmos", (TOTAL_ROWS, 1, 200, 200)), ("target_slm_intensity", (TOTAL_ROWS, 1, 200, 200)), ("e_d_logits", (TOTAL_ROWS, 10))):
        value = saved.get(field)
        _require(torch.is_tensor(value) and tuple(value.shape) == expected and bool(torch.isfinite(value.real if torch.is_complex(value) else value).all()), f"saved B endpoint {field} is malformed or nonfinite")

    direction_blocks = finite._load_direction_blocks(inputs)
    for index in range(20):
        saved_block = finite._load_block(finite_dir / finite._block_name(index), index, inputs)
        start = index * BATCH_SIZE
        _require(torch.equal(saved_block["source_row_indices"], rows[start:start + BATCH_SIZE]), f"finite-response block {index} row mapping changed")
        _require(torch.equal(saved_block["labels"], labels[start:start + BATCH_SIZE]), f"finite-response block {index} labels changed")
        del saved_block
    optimizer = inputs["optimizers"]["B_class_ce"]
    payload = inputs["payloads"]["B_class_ce"]
    _require(payload["optimizer_step_by_parameter"] == {"encoder.phase": 2200, "generator.phase": 2500}, "B endpoint Adam step counts are not 2200/2500")
    directions = _build_shared_directions(direction_blocks, optimizer)
    phase = inputs["parameters"]["B_class_ce"]["generator.phase"]
    _require(directions["batch_mean_descent_directions"]["F_ADAM"].shape == phase.shape, "shared Generator directions differ from the checkpoint phase shape")
    return {
        **inputs,
        "artifact_dir": artifact_dir,
        "direction_dir": direction_dir,
        "finite_dir": finite_dir,
        "parent_population": parent_population,
        "parent_report": parent_report,
        "direction_report": direction_report,
        "finite_report": finite_report,
        "source_indices_600": rows,
        "labels_600": labels,
        "roles_600": parent_population["roles"],
        "images_600": images,
        "direction_blocks": direction_blocks,
        "directions": directions,
    }


def _population_record(inputs: Mapping[str, Any], output: Path) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "parent_ce_directory": inputs["artifact_dir"].as_posix(),
        "corrected_direction_directory": inputs["direction_dir"].as_posix(),
        "finite_response_directory": inputs["finite_dir"].as_posix(),
        "checkpoint": inputs["paths"]["checkpoint_B_class_ce"].as_posix(),
        "readout": inputs["paths"]["readout_B_class_ce"].as_posix(),
        "training_data": inputs["paths"]["training_data"].as_posix(),
        "source_row_indices": inputs["source_indices_600"].tolist(),
        "labels": inputs["labels_600"].tolist(),
        "roles": list(inputs["roles_600"]),
        "role_counts": {"fit": FIT_ROWS, "historic_holdout": HISTORIC_ROWS, "confirmation": CONFIRMATION_ROWS},
        "row_order": "parent attempt_01 original order",
        "fit_batch_size": BATCH_SIZE,
        "fit_batch_count": 20,
        "confirmation_used_for_direction_or_selection": False,
        "directions": ["S_GD", "F_GD", "S_ADAM", "F_ADAM", NATIVE_NAME],
        "lambda_values": [-1, 1],
        "equal_rms_reference": "RMS(F_ADAM computed once from the mean full gradient)",
        "evaluators": list(EVALUATORS),
        "margin_derivative_definition": MARGIN_DEFINITION,
        "output_directory": output.as_posix(),
    }


def _identity(inputs: Mapping[str, Any], population: Mapping[str, Any], output: Path) -> ResearchRunIdentity:
    payload = inputs["payloads"]["B_class_ce"]
    attempt = output.name.replace("_", "-")
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=f"{TASK_ID}-{attempt}",
        trajectory_id="frozen-B-class-CE-step2500-common-generator-direction-transfer",
        semantic_authority_id=direction.TASK_ID,
        data_role="fixed 600 canonical TRAIN rows; first 200 construct directions, next 100 historical comparison, last 300 confirmation only",
        config_identity={
            "propagation_grid": 2048,
            "source_checkpoint_step": 2200,
            "encoder_adam_step": 2200,
            "generator_adam_step": 2500,
            "fit_rows": FIT_ROWS,
            "historic_rows": HISTORIC_ROWS,
            "confirmation_rows": CONFIRMATION_ROWS,
            "chunk_size": CHUNK_SIZE,
            "common_conditions": list(_condition_vectors(inputs["directions"])),
            "margin_derivative_definition": MARGIN_DEFINITION,
        },
        model_identity={"parameter_shapes": {name: list(value.shape) for name, value in payload["trainable_parameter_state"].items()}, "source_arm": "B_class_ce", "completed_updates": 300},
        optimizer_identity={"class": "saved Adam tensors read only; no optimizer constructed or stepped", "source_steps": payload["optimizer_step_by_parameter"]},
        sampler_identity={"source_indices": population["source_row_indices"], "labels": population["labels"], "roles": population["roles"]},
        execution_source_semantics_id=TASK_ID,
        scientific_attempt_id=f"{TASK_ID}:{attempt}",
    )


def _role_for_position(position: int) -> str:
    if position < FIT_ROWS:
        return "fit"
    if position < FIT_ROWS + HISTORIC_ROWS:
        return "historic_holdout"
    return "confirmation"


def _phase_set(parameter: torch.nn.Parameter, baseline: torch.Tensor, planned_delta: torch.Tensor) -> torch.Tensor:
    finite._apply_displacement(parameter, baseline, planned_delta)
    return parameter.detach().cpu().clone() - baseline.detach().cpu()


def _positive_control_counts(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    return finite._positive_control_metrics(logits, labels)


def _target_positive_controls(evaluators: Mapping[str, torch.nn.Module], images: torch.Tensor, labels: torch.Tensor, device: torch.device) -> dict[str, Any]:
    pieces: dict[str, list[torch.Tensor]] = {name: [] for name in EVALUATORS}
    with torch.no_grad():
        for start in range(0, TOTAL_ROWS, BATCH_SIZE):
            view = images[start:start + BATCH_SIZE].to(device)
            for name in EVALUATORS:
                result = evaluators[name](view)
                logits = result[1] if name != "E_D" else result
                pieces[name].append(logits.detach().cpu())
    return {name: _positive_control_counts(torch.cat(pieces[name]), labels) for name in EVALUATORS}


def _read_fixed_batch(consumer: Any, evaluators: Mapping[str, torch.nn.Module], inputs: Mapping[str, Any], start: int) -> dict[str, Any]:
    stop = start + BATCH_SIZE
    z = inputs["readouts"]["B_class_ce"]["z"][start:stop]
    readout = finite._readout(consumer, evaluators, z)
    labels = inputs["labels_600"][start:stop]
    target_raw = inputs["readouts"]["B_class_ce"]["target_slm_intensity"][start:stop]
    images = inputs["images_600"][start:stop]
    margins = finite._margins(readout["evaluator_logits"]["E_D"], labels)
    image_metrics = finite._row_image_metrics(readout["raw_cmos"], target_raw, images)
    readout.update({
        "source_row_indices": inputs["source_indices_600"][start:stop].clone(),
        "labels": labels.clone(),
        "roles": [_role_for_position(position) for position in range(start, stop)],
        "ols_coefficients": z.clone(),
        "target_slm_intensity": target_raw.clone(),
        "target_images_28x28": images.clone(),
        "margins": margins,
        "image_metrics": image_metrics,
    })
    return readout


def _concat_batch_records(records: Sequence[Mapping[str, Any]], start: int, kind: str) -> dict[str, Any]:
    _require(len(records) == CHUNK_SIZE // BATCH_SIZE, "one hundred row chunk does not contain ten fixed batches")
    tensor_keys = ("source_row_indices", "labels", "ols_coefficients", "target_slm_intensity", "target_images_28x28", "raw_cmos", "evaluator_view")
    result: dict[str, Any] = {key: torch.cat([record[key] for record in records]) for key in tensor_keys}
    result["row_start"] = start
    result["row_stop"] = start + CHUNK_SIZE
    result["kind"] = kind
    result["roles"] = [role for record in records for role in record["roles"]]
    result["margins"] = [item for record in records for item in record["margins"]]
    result["image_metrics"] = [item for record in records for item in record["image_metrics"]]
    result["available_plane_records"] = [record["available_plane_records"] for record in records]
    result["evaluator_logits"] = {name: torch.cat([record["evaluator_logits"][name] for record in records]) for name in EVALUATORS}
    return result


def _chunk_name(condition: str, chunk_index: int) -> str:
    return f"readout_{condition}_rows{chunk_index * CHUNK_SIZE:03d}_{(chunk_index + 1) * CHUNK_SIZE:03d}.pt"


def _validate_chunk(payload: Mapping[str, Any], condition: str, chunk_index: int, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    _require(payload.get("schema") == BLOCK_SCHEMA and payload.get("task_id") == TASK_ID and payload.get("condition") == condition and payload.get("chunk_index") == chunk_index, "saved shared-direction chunk schema or identity mismatch")
    block = payload.get("block")
    start = chunk_index * CHUNK_SIZE
    stop = start + CHUNK_SIZE
    _require(block.get("row_start") == start and block.get("row_stop") == stop and block.get("kind") == condition, "saved shared-direction chunk row boundary or condition changed")
    _require(torch.equal(block["source_row_indices"], inputs["source_indices_600"][start:stop]), "saved shared-direction chunk row mapping changed")
    _require(torch.equal(block["labels"], inputs["labels_600"][start:stop]) and block["roles"] == list(inputs["roles_600"][start:stop]), "saved shared-direction chunk labels or roles changed")
    return block


def _load_chunk(path: Path, condition: str, chunk_index: int, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    return _validate_chunk(torch.load(path, map_location="cpu", weights_only=True), condition, chunk_index, inputs)


def _save_chunk(target: Path, condition: str, chunk_index: int, block: Mapping[str, Any]) -> str:
    name = _chunk_name(condition, chunk_index)
    finite._write_torch_new(target / name, {"schema": BLOCK_SCHEMA, "task_id": TASK_ID, "condition": condition, "chunk_index": chunk_index, "block": dict(block)}, "immutable shared Generator direction readout chunk")
    return name


def _measure_condition_chunk(
    consumer: Any,
    evaluators: Mapping[str, torch.nn.Module],
    inputs: Mapping[str, Any],
    chunk_index: int,
    condition: str,
    planned_delta: torch.Tensor | None,
    actual_delta: torch.Tensor | None,
    eta: float,
    gradient_by_position: Mapping[int, torch.Tensor],
    baseline_block: Mapping[str, Any] | None,
) -> dict[str, Any]:
    start = chunk_index * CHUNK_SIZE
    records: list[dict[str, Any]] = []
    for batch_start in range(start, start + CHUNK_SIZE, BATCH_SIZE):
        record = _read_fixed_batch(consumer, evaluators, inputs, batch_start)
        if baseline_block is not None:
            offset = batch_start - start
            base_margins = baseline_block["margins"][offset:offset + BATCH_SIZE]
            for row_index, margin in enumerate(record["margins"]):
                base_margin = base_margins[row_index]
                delta_margin = margin["margin"] - base_margin["margin"]
                margin["delta_from_baseline"] = delta_margin
                margin["response_status"] = finite._response_status(delta_margin, eta)
                margin["best_wrong_class_changed_from_baseline"] = margin["best_wrong_class"] != base_margin["best_wrong_class"]
        records.append(record)
    block = _concat_batch_records(records, start, condition)
    if baseline_block is not None:
        linear: list[float | None] = []
        for position in range(start, start + CHUNK_SIZE):
            gradient = gradient_by_position.get(position)
            linear.append(None if gradient is None or actual_delta is None else float(torch.dot(gradient, actual_delta.to(torch.float64).reshape(-1))))
        block["delta_margin_linear_prediction"] = linear
        block["planned_displacement_rms"] = None if planned_delta is None else _rms(planned_delta)
        block["actual_float32_displacement_rms"] = None if actual_delta is None else _rms(actual_delta)
    return block


def _reference_gradient_map(blocks: Sequence[Mapping[str, Any]]) -> dict[int, torch.Tensor]:
    result: dict[int, torch.Tensor] = {}
    for block in blocks:
        for index, position in enumerate(block["margin_gradient_fit_positions"]):
            _require(int(position) not in result, "duplicate saved E_D gradient fit position")
            result[int(position)] = block["margin_gradients_generator"][index].detach().cpu().to(torch.float64).reshape(-1)
    expected = {
        int(row["fit_position"])
        for block in blocks
        for row in block["rows"]
        if not row["best_wrong_exact_tie"]
    }
    _require(set(result).issubset(expected), "saved E_D gradients include rows outside classes 1/7 and 0/4/8")
    return result


def _verify_baseline_block(block: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, float]:
    start, stop = int(block["row_start"]), int(block["row_stop"])
    saved = inputs["readouts"]["B_class_ce"]
    errors: dict[str, float] = {}
    for field in ("raw_cmos", "ols_coefficients"):
        actual = block[field]
        expected = saved["raw_cmos" if field == "raw_cmos" else "z"][start:stop]
        _require(torch.allclose(actual, expected, rtol=FLOAT_RTOL, atol=FLOAT_ATOL), f"B endpoint full-population baseline differs from saved {field}")
        errors[field] = float((actual - expected).abs().max())
    for name in EVALUATORS:
        expected_key = {"E_S": "e_s_logits", "E_I": "e_i_logits", "E_C": "e_c_logits", "E_D": "e_d_logits"}[name]
        actual = block["evaluator_logits"][name]
        expected = saved[expected_key][start:stop]
        _require(torch.allclose(actual, expected, rtol=FLOAT_RTOL, atol=FLOAT_ATOL), f"B endpoint baseline {name} logits differ from saved readout")
        errors[f"{name}_logits"] = float((actual - expected).abs().max())
    return errors


def _margin_gradient_check(
    consumer: Any,
    evaluators: Mapping[str, torch.nn.Module],
    inputs: Mapping[str, Any],
    parameters: Mapping[str, torch.nn.Parameter],
    baseline_phase: torch.Tensor,
    condition_vectors: Mapping[str, Mapping[str, Any]],
    directions: Mapping[str, Any],
    source_block: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    checks: dict[str, list[dict[str, Any]]] = {}
    first = _read_fixed_batch(consumer, evaluators, inputs, 0)
    for name in EQUAL_RMS_NAMES:
        if directions["equal_rms_directions"][name] is None:
            continue
        base_direction = directions["equal_rms_directions"][name]
        for label in FOCUS_CLASSES:
            pos = next(index for index, value in enumerate(inputs["labels_600"][:BATCH_SIZE].tolist()) if value == label)
            fit_position = pos
            gradient_index = source_block["margin_gradient_fit_positions"].index(fit_position)
            gradient = source_block["margin_gradients_generator"][gradient_index].detach().cpu().to(torch.float64).reshape(-1)
            source_margin = first["margins"][pos]
            _require(not source_margin["tied_maximum"], f"fit class {label} baseline has tied maximum-wrong logits")
            pair_checks: list[dict[str, Any]] = []
            for epsilon in (1 / 8, 1 / 16):
                delta = base_direction.mul(epsilon).detach().cpu()
                try:
                    plus_applied = _phase_set(parameters["generator.phase"], baseline_phase, delta)
                    plus = _read_fixed_batch(consumer, evaluators, inputs, 0)
                    plus_phase = parameters["generator.phase"].detach().cpu().clone()
                    minus_applied = _phase_set(parameters["generator.phase"], baseline_phase, -delta)
                    minus = _read_fixed_batch(consumer, evaluators, inputs, 0)
                    minus_phase = parameters["generator.phase"].detach().cpu().clone()
                finally:
                    finite._restore_phase(parameters["generator.phase"], baseline_phase)
                plus_margin = plus["margins"][pos]
                minus_margin = minus["margins"][pos]
                same_wrong = plus_margin["best_wrong_class"] == minus_margin["best_wrong_class"] == source_margin["best_wrong_class"]
                unique = not plus_margin["tied_maximum"] and not minus_margin["tied_maximum"]
                observed_slope = (plus_margin["margin"] - minus_margin["margin"]) / (2.0 * epsilon)
                actual_phase_delta = plus_phase - minus_phase
                reference_slope = float(torch.dot(gradient, actual_phase_delta.to(torch.float64).reshape(-1)) / (2.0 * epsilon))
                tolerance = 0.10 * abs(reference_slope) + 1e-4
                record = {
                    "epsilon": epsilon,
                    "observed_central_slope": observed_slope,
                    "saved_gradient_slope_using_actual_phase_difference": reference_slope,
                    "absolute_error": abs(observed_slope - reference_slope),
                    "tolerance": tolerance,
                    "positive_phase_rms": _rms(plus_applied),
                    "negative_phase_rms": _rms(minus_applied),
                    "maximum_wrong_class_unique": unique,
                    "maximum_wrong_class_unchanged": same_wrong,
                    "passed": unique and same_wrong and abs(observed_slope - reference_slope) <= tolerance,
                }
                pair_checks.append(record)
            checks[f"{name}_class{label}"] = pair_checks
    return checks


def _preflight(inputs: Mapping[str, Any], output: Path, started: float) -> tuple[dict[str, Any], dict[str, Any]]:
    device = torch.device("cuda:0")
    _require(torch.cuda.is_available(), "CUDA is unavailable for the common-state production preflight")
    runtime_probe = direction._runtime_dll_probe()
    _require(runtime_probe.get("status") in {"already_present", "prepended", "not_applicable"}, "CUDA runtime DLL readiness failed")
    cuda_before = _resource_observation(device)
    disk_root = output
    while not disk_root.exists():
        disk_root = disk_root.parent
    disk = shutil.disk_usage(disk_root)
    disk_observation = {"checked_path": disk_root.as_posix(), "free_bytes_observed": int(disk.free), "estimated_output_bytes_observation": ESTIMATED_OUTPUT_BYTES}

    consumer, parameters = direction._new_consumer(inputs, "B_class_ce", device)
    for parameter in parameters.values():
        parameter.requires_grad_(False)
    evaluators = base.load_evaluators(inputs["paths"], device)
    for model in evaluators.values():
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    runtime_before = finite._snapshot(consumer.runtime)
    evaluator_before = {name: finite._snapshot(model) for name, model in evaluators.items()}
    optimizer_before = direction._cpu_tree(inputs["optimizers"]["B_class_ce"])
    frozen_inputs_before = direction._consumer_frozen_inputs_snapshot(consumer)
    baseline_phase = parameters["generator.phase"].detach().cpu().clone()
    condition_vectors = _condition_vectors(inputs["directions"])
    actual_deltas: dict[str, torch.Tensor] = {}
    for name, spec in condition_vectors.items():
        if not spec["available"]:
            continue
        actual_deltas[name] = _phase_set(parameters["generator.phase"], baseline_phase, spec["planned_delta"])
        finite._restore_phase(parameters["generator.phase"], baseline_phase)

    target_controls = _target_positive_controls(evaluators, inputs["images_600"], inputs["labels_600"], device)
    controls_qualified = finite._qualify_target_controls(target_controls)
    _require(controls_qualified, "four frozen evaluators failed the fixed-TRAIN target positive control")

    full_path_checks: dict[str, Any] = {}
    with torch.no_grad():
        for role, position in (("fit", 0), ("confirmation", FIT_ROWS + HISTORIC_ROWS)):
            stop = position + BATCH_SIZE
            result = consumer.runtime.forward_from_image(inputs["images_600"][position:stop].to(device))
            expected_z = inputs["readouts"]["B_class_ce"]["z"][position:stop].to(device)
            direct = consumer.runtime.forward_from_ols(expected_z)
            _require(torch.allclose(result.z_ols.coefficients.detach(), expected_z, rtol=FLOAT_RTOL, atol=FLOAT_ATOL), f"{role} real Encoder OLS differs from the saved B endpoint")
            _require(torch.allclose(result.raw_intensity, direct.raw_intensity, rtol=FLOAT_RTOL, atol=FLOAT_ATOL), f"{role} Encoder path differs from its fixed-OLS decoder path")
            full_path_checks[role] = {"source_rows": inputs["source_indices_600"][position:stop].tolist(), "encoder_ols_matches_saved": True, "encoder_and_fixed_ols_cmos_match": True}

    source_block = inputs["direction_blocks"][0]
    finite_checks = _margin_gradient_check(consumer, evaluators, inputs, parameters, baseline_phase, condition_vectors, inputs["directions"], source_block)
    all_finite_checks_passed = all(record["passed"] for records in finite_checks.values() for record in records)
    _require(all_finite_checks_passed, "one or more common-direction central differences failed the saved E_D gradient reference")

    repeated1 = _read_fixed_batch(consumer, evaluators, inputs, 0)
    repeated2 = _read_fixed_batch(consumer, evaluators, inputs, 0)
    _require(torch.allclose(repeated1["raw_cmos"], repeated2["raw_cmos"], rtol=FLOAT_RTOL, atol=FLOAT_ATOL), "preflight repeated baseline is not reproducible")
    _require(all(parameter.grad is None and not parameter.requires_grad for parameter in parameters.values()), "preflight left runtime parameter gradients or trainable flags")
    _require(_tree_equal(runtime_before, finite._snapshot(consumer.runtime)), "preflight changed frozen model parameters or scientific buffers")
    _require(_tree_equal(frozen_inputs_before, direction._consumer_frozen_inputs_snapshot(consumer)), "preflight changed frozen optical references or weights")
    _require(_tree_equal(optimizer_before, direction._cpu_tree(inputs["optimizers"]["B_class_ce"])), "preflight changed saved Adam state")
    for name, model in evaluators.items():
        _require(_tree_equal(evaluator_before[name], finite._snapshot(model)), f"preflight changed evaluator {name}")
        _require(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()), f"evaluator {name} was not frozen")
    cuda_after = _resource_observation(device)
    result = {
        "schema": "m0_generator_shared_direction_transfer_preflight.v1",
        "task_id": TASK_ID,
        "status": "READY",
        "runtime_dll": runtime_probe,
        "cuda_before": cuda_before,
        "cuda_after": cuda_after,
        "disk_observation": disk_observation,
        "target_positive_controls": target_controls,
        "target_positive_controls_qualified": controls_qualified,
        "real_encoder_vs_cached_ols": full_path_checks,
        "finite_difference_gradient_checks": finite_checks,
        "finite_difference_checks_passed": all_finite_checks_passed,
        "baseline_repeat_batch_passed": True,
        "source_adam_unchanged": True,
        "model_state_unchanged": True,
        "evaluator_state_unchanged": True,
        "parameter_grad_buffers_empty": True,
        "optimizer_updates": 0,
        "condition_vectors": {
            name: ({"available": False, "reason": spec["reason"], "candidate": spec["candidate"], "sign": spec["sign"], "native_step": spec["native_step"]} if not spec["available"] else {"available": True, "planned_rms": _rms(spec["planned_delta"]), "actual_float32_rms": _rms(actual_deltas[name]), "candidate": spec["candidate"], "sign": spec["sign"], "native_step": spec["native_step"]})
            for name, spec in condition_vectors.items()
        },
        "elapsed_seconds_observation_only": time.perf_counter() - started,
    }
    replay = {"conditions": condition_vectors, "actual_float32_deltas": actual_deltas}
    return result, replay


def _ready_receipt(control: ResearchRunControl, preflight: Mapping[str, Any]) -> PreflightReceipt:
    correctness = (
        preflight.get("status") == "READY"
        and preflight.get("target_positive_controls_qualified") is True
        and set(preflight.get("target_positive_controls", {})) == set(EVALUATORS)
        and set(preflight.get("real_encoder_vs_cached_ols", {})) == {"fit", "confirmation"}
        and all(item.get("encoder_ols_matches_saved") and item.get("encoder_and_fixed_ols_cmos_match") for item in preflight.get("real_encoder_vs_cached_ols", {}).values())
        and preflight.get("finite_difference_checks_passed") is True
        and set(preflight.get("finite_difference_gradient_checks", {})) == {f"{name}_class{label}" for name in EQUAL_RMS_NAMES for label in FOCUS_CLASSES if preflight.get("condition_vectors", {}).get(f"{name}_equal_rms_plus1", {}).get("available")}
        and all(len(checks) == 2 and all(check.get("passed") is True for check in checks) for checks in preflight.get("finite_difference_gradient_checks", {}).values())
        and preflight.get("baseline_repeat_batch_passed") is True
        and preflight.get("model_state_unchanged") is True
        and preflight.get("source_adam_unchanged") is True
        and preflight.get("evaluator_state_unchanged") is True
        and preflight.get("parameter_grad_buffers_empty") is True
        and preflight.get("optimizer_updates") == 0
        and preflight.get("storage_lifecycle", {}).get("immutable_write_read_passed") is True
        and preflight.get("storage_lifecycle", {}).get("interrupted_manifest_recovery_passed") is True
        and preflight.get("storage_lifecycle", {}).get("completed_resume_without_forward_passed") is True
    )
    hardware = all(
        int(preflight.get(key, {}).get("total_bytes", 0)) > 0
        and 0 <= int(preflight.get(key, {}).get("free_bytes", -1)) <= int(preflight.get(key, {}).get("total_bytes", 0))
        for key in ("cuda_before", "cuda_after")
    )
    disk = preflight.get("disk_observation", {})
    disk_passed = int(disk.get("free_bytes_observed", -1)) >= 0
    runtime_passed = preflight.get("runtime_dll", {}).get("status") in {"already_present", "prepended", "not_applicable"}
    resources = hardware and disk_passed and runtime_passed
    implementation = correctness and preflight.get("optimizer_updates") == 0 and len(preflight.get("condition_vectors", {})) == 10
    return control.make_ready_receipt(
        correctness={"passed": correctness, "status": "PASS" if correctness else "BLOCKED", "evaluator_positive_controls": preflight.get("target_positive_controls_qualified") is True, "full_encoder_and_cached_ols": correctness, "central_finite_differences": preflight.get("finite_difference_checks_passed") is True, "frozen_state": preflight.get("model_state_unchanged") is True and preflight.get("source_adam_unchanged") is True and preflight.get("evaluator_state_unchanged") is True},
        resource={"passed": resources, "status": "PASS" if resources else "BLOCKED", "whole_device_memory_is_observational_only": True, "device_telemetry_valid": hardware, "disk_observation_available": disk_passed, "disk": disk, "cuda_before": preflight.get("cuda_before"), "cuda_after": preflight.get("cuda_after"), "cuda_runtime": runtime_passed},
        implementation={"passed": implementation, "status": "PASS" if implementation else "BLOCKED", "selected_engine": "cuda_2048_frozen_shared_generator_displacement", "optimizer_updates_zero": preflight.get("optimizer_updates") == 0, "condition_count": len(preflight.get("condition_vectors", {}))},
        selected_engine="cuda_2048_frozen_shared_generator_displacement",
        resource_envelope={"cuda_before": preflight.get("cuda_before"), "cuda_after": preflight.get("cuda_after"), "disk": disk},
    )


def _manifest_seed(identity: ResearchRunIdentity, population: Mapping[str, Any], receipt: PreflightReceipt, preflight: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": MANIFEST_SCHEMA, "task_id": TASK_ID, "margin_derivative_definition": MARGIN_DEFINITION, "identity": identity.to_dict(), "population": dict(population), "receipt": receipt.to_dict(), "preflight": dict(preflight), "completed_chunks": [], "status": "RUNNING"}


def _validate_manifest(path: Path, target: Path, identity: ResearchRunIdentity, population: Mapping[str, Any]) -> dict[str, Any]:
    _require(path.resolve() == (target / "resume_manifest.json").resolve() and path.is_file(), "resume requires this task's explicit resume_manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == MANIFEST_SCHEMA and manifest.get("task_id") == TASK_ID and manifest.get("margin_derivative_definition") == MARGIN_DEFINITION, "resume manifest task or measurement semantics mismatch")
    _require(manifest.get("population") == dict(population), "resume population rows, labels, roles or input paths changed")
    check_resume_compatibility(identity, manifest["identity"])
    _require(manifest.get("status") in {"RUNNING", "INTERRUPTED", "COMPLETED"}, "resume manifest status is invalid")
    return manifest


def _terminal_report(target: Path, manifest: Mapping[str, Any], identity: ResearchRunIdentity) -> dict[str, Any] | None:
    report_path = target / "report.json"
    if manifest.get("status") == "COMPLETED":
        _require(report_path.is_file(), "completed shared-direction manifest lacks its final report")
    elif not report_path.is_file():
        return None
    expected = {_chunk_name(condition, index) for condition in manifest.get("measured_conditions", []) for index in range(CHUNK_COUNT)}
    _require(expected.issubset(set(manifest.get("completed_chunks", []))) and all((target / name).is_file() for name in expected), "terminal report lacks one or more immutable readout chunks")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report.get("schema") == REPORT_SCHEMA and report.get("task_id") == TASK_ID and report.get("status") == "VALID_COMPLETED" and report.get("research_run_identity") == identity.to_dict(), "saved terminal report is invalid")
    if manifest.get("status") != "COMPLETED":
        updated = dict(manifest)
        updated["status"] = "COMPLETED"
        finite._write_json_replace(target / "resume_manifest.json", updated)
    return report


def _preflight_artifact_lifecycle(
    identity: ResearchRunIdentity,
    population: Mapping[str, Any],
    inputs: Mapping[str, Any],
    preflight: Mapping[str, Any],
    scratch_parent: Path,
) -> dict[str, bool]:
    """Exercise immutable block IO, interrupted resume validation, and terminal no-forward recovery."""
    with tempfile.TemporaryDirectory(prefix="lightgen_shared_direction_preflight_", dir=scratch_parent) as scratch_name:
        scratch = Path(scratch_name)
        block = {
            "row_start": 0,
            "row_stop": CHUNK_SIZE,
            "kind": "baseline",
            "source_row_indices": inputs["source_indices_600"][:CHUNK_SIZE].clone(),
            "labels": inputs["labels_600"][:CHUNK_SIZE].clone(),
            "roles": list(inputs["roles_600"][:CHUNK_SIZE]),
        }
        block_name = _chunk_name("baseline", 0)
        finite._write_torch_new(scratch / block_name, {"schema": BLOCK_SCHEMA, "task_id": TASK_ID, "condition": "baseline", "chunk_index": 0, "block": block}, "preflight immutable readout chunk")
        loaded = _load_chunk(scratch / block_name, "baseline", 0, inputs)
        _require(torch.equal(loaded["source_row_indices"], block["source_row_indices"]) and torch.equal(loaded["labels"], block["labels"]), "preflight block readback differs from the written row mapping")

        measured = ["baseline", "baseline_repeat", *[name for name, item in preflight["condition_vectors"].items() if item["available"]]]
        expected_files = [_chunk_name(condition, index) for condition in measured for index in range(CHUNK_COUNT)]
        partial = {
            "schema": MANIFEST_SCHEMA,
            "task_id": TASK_ID,
            "margin_derivative_definition": MARGIN_DEFINITION,
            "identity": identity.to_dict(),
            "population": dict(population),
            "completed_chunks": [block_name],
            "measured_conditions": measured,
            "status": "INTERRUPTED",
        }
        manifest_path = scratch / "resume_manifest.json"
        finite._write_json_new(manifest_path, partial, "preflight interrupted manifest")
        resumed = _validate_manifest(manifest_path, scratch, identity, population)
        _require(resumed["status"] == "INTERRUPTED" and resumed["completed_chunks"] == [block_name], "preflight did not preserve the explicit interrupted cursor")
        _load_chunk(scratch / block_name, "baseline", 0, inputs)

        for name in expected_files:
            path = scratch / name
            if not path.exists():
                path.touch()
        partial["status"] = "COMPLETED"
        partial["completed_chunks"] = expected_files
        finite._write_json_replace(manifest_path, partial)
        terminal = {
            "schema": REPORT_SCHEMA,
            "task_id": TASK_ID,
            "status": "VALID_COMPLETED",
            "research_run_identity": identity.to_dict(),
            "classification": "MIXED_OR_INSUFFICIENT",
        }
        finite._write_json_new(scratch / "report.json", terminal, "preflight terminal resume report")
        restored = _terminal_report(scratch, partial, identity)
        _require(restored == terminal, "completed recovery did not return the saved terminal report")
    return {"immutable_write_read_passed": True, "interrupted_manifest_recovery_passed": True, "completed_resume_without_forward_passed": True}


def _condition_names() -> tuple[str, ...]:
    names = [
        f"{name}_equal_rms_{'plus' if sign > 0 else 'minus'}1"
        for name in EQUAL_RMS_NAMES
        for sign in SIGNS
    ]
    names.extend(f"{NATIVE_NAME}_{'plus' if sign > 0 else 'minus'}1" for sign in SIGNS)
    return tuple(names)


def _load_readout_chunk(target: Path, condition: str, chunk_index: int, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    return _load_chunk(target / _chunk_name(condition, chunk_index), condition, chunk_index, inputs)


def _save_readout_chunk(target: Path, manifest: dict[str, Any], condition: str, chunk_index: int, block: Mapping[str, Any], inputs: Mapping[str, Any]) -> None:
    name = _chunk_name(condition, chunk_index)
    path = target / name
    if path.exists():
        _load_chunk(path, condition, chunk_index, inputs)
    else:
        _save_chunk(target, condition, chunk_index, block)
    completed = set(manifest.get("completed_chunks", []))
    completed.add(name)
    manifest["completed_chunks"] = sorted(completed)
    finite._write_json_replace(target / "resume_manifest.json", manifest)


def _load_gradient_map(inputs: Mapping[str, Any]) -> dict[int, torch.Tensor]:
    return _reference_gradient_map(inputs["direction_blocks"])


def _chunks_measurement(
    target: Path,
    manifest: dict[str, Any],
    condition: str,
    inputs: Mapping[str, Any],
    consumer: Any,
    evaluators: Mapping[str, torch.nn.Module],
    *,
    planned_delta: torch.Tensor | None,
    actual_delta: torch.Tensor | None,
    eta: float,
    gradient_map: Mapping[int, torch.Tensor],
    baseline: bool = False,
) -> None:
    for chunk_index in range(CHUNK_COUNT):
        name = _chunk_name(condition, chunk_index)
        path = target / name
        if name in manifest["completed_chunks"]:
            _load_chunk(path, condition, chunk_index, inputs)
            continue
        if path.exists():
            block = _load_chunk(path, condition, chunk_index, inputs)
            del block
            manifest["completed_chunks"] = sorted(set(manifest["completed_chunks"]) | {name})
            finite._write_json_replace(target / "resume_manifest.json", manifest)
            continue
        base_block = _load_readout_chunk(target, "baseline", chunk_index, inputs) if not baseline and condition != "baseline_repeat" else None
        block = _measure_condition_chunk(consumer, evaluators, inputs, chunk_index, condition, planned_delta, actual_delta, eta, gradient_map, base_block)
        if baseline:
            error = _verify_baseline_block(block, inputs)
            block["parent_readout_max_abs_errors"] = error
        _save_readout_chunk(target, manifest, condition, chunk_index, block, inputs)
        del block, base_block
    _require(all(_chunk_name(condition, index) in manifest["completed_chunks"] for index in range(CHUNK_COUNT)), f"condition {condition} did not complete all fixed row chunks")


def _baseline_repeat_eta(target: Path, inputs: Mapping[str, Any]) -> dict[str, float]:
    max_margin_difference = 0.0
    max_cmos_difference = 0.0
    max_logit_difference = 0.0
    for chunk_index in range(CHUNK_COUNT):
        base_block = _load_readout_chunk(target, "baseline", chunk_index, inputs)
        repeated = _load_readout_chunk(target, "baseline_repeat", chunk_index, inputs)
        max_cmos_difference = max(max_cmos_difference, float((base_block["raw_cmos"] - repeated["raw_cmos"]).abs().max()))
        for name in EVALUATORS:
            max_logit_difference = max(max_logit_difference, float((base_block["evaluator_logits"][name] - repeated["evaluator_logits"][name]).abs().max()))
        for left, right in zip(base_block["margins"], repeated["margins"], strict=True):
            max_margin_difference = max(max_margin_difference, abs(float(left["margin"]) - float(right["margin"])))
    return {"max_abs_cmos_difference": max_cmos_difference, "max_abs_evaluator_logit_difference": max_logit_difference, "max_abs_e_d_margin_difference": max_margin_difference, "eta": max(ETA_FLOOR, 5.0 * max_margin_difference)}


def _collect_state_rows(target: Path, condition: str, inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk_index in range(CHUNK_COUNT):
        block = _load_readout_chunk(target, condition, chunk_index, inputs)
        for offset in range(CHUNK_SIZE):
            position = int(block["row_start"]) + offset
            margin = block["margins"][offset]
            metric = block["image_metrics"][offset]
            row = {
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
            }
            if condition not in {"baseline", "baseline_repeat"}:
                row["delta_from_baseline"] = float(margin["delta_from_baseline"])
                row["response_status"] = margin["response_status"]
                row["best_wrong_class_changed_from_baseline"] = margin["best_wrong_class_changed_from_baseline"]
                block_prediction = block.get("delta_margin_linear_prediction", [])
                row["fit_linear_prediction"] = block_prediction[offset] if block_prediction else None
            rows.append(row)
    return rows


def _class_metrics(rows: Sequence[Mapping[str, Any]], *, baseline_rows: Sequence[Mapping[str, Any]] | None = None, eta: float | None = None) -> dict[str, Any]:
    labels = torch.tensor([int(row["label"]) for row in rows])
    prediction = torch.tensor([int(row["ed_prediction"]) for row in rows])
    correct = [int(((labels == class_id) & (prediction == class_id)).sum()) for class_id in range(10)]
    per_class_total = [int((labels == class_id).sum()) for class_id in range(10)]
    recall = [correct[class_id] / per_class_total[class_id] if per_class_total[class_id] else None for class_id in range(10)]
    result: dict[str, Any] = {
        "rows": len(rows),
        "correct": correct,
        "rows_per_class": per_class_total,
        "per_class_recall": recall,
        "macro_recall": sum(value for value in recall if value is not None) / sum(value is not None for value in recall),
        "prediction_counts": torch.bincount(prediction, minlength=10).tolist(),
        "raw_mse_mean": sum(float(row["raw_mse"]) for row in rows) / len(rows),
        "raw_foreground_mse_mean": sum(float(row["raw_foreground_mse"]) for row in rows) / len(rows),
        "raw_background_mse_mean": sum(float(row["raw_background_mse"]) for row in rows) / len(rows),
        "view_mse_mean": sum(float(row["view_mse"]) for row in rows) / len(rows),
        "view_foreground_mse_mean": sum(float(row["view_foreground_mse"]) for row in rows) / len(rows),
        "view_background_mse_mean": sum(float(row["view_background_mse"]) for row in rows) / len(rows),
        "raw_fraction_above_one_mean": sum(float(row["raw_fraction_above_one"]) for row in rows) / len(rows),
        "raw_zero_images": sum(bool(row["raw_zero_image"]) for row in rows),
        "raw_constant_images": sum(bool(row["raw_constant_image"]) for row in rows),
    }
    result["per_class"] = {
        str(class_id): {
            "correct": correct[class_id],
            "rows": per_class_total[class_id],
            "recall": recall[class_id],
            "raw_mse_mean": sum(float(row["raw_mse"]) for row in rows if int(row["label"]) == class_id) / per_class_total[class_id],
            "raw_foreground_mse_mean": sum(float(row["raw_foreground_mse"]) for row in rows if int(row["label"]) == class_id) / per_class_total[class_id],
            "raw_background_mse_mean": sum(float(row["raw_background_mse"]) for row in rows if int(row["label"]) == class_id) / per_class_total[class_id],
            "raw_fraction_above_one_mean": sum(float(row["raw_fraction_above_one"]) for row in rows if int(row["label"]) == class_id) / per_class_total[class_id],
            "zero_images": sum(bool(row["raw_zero_image"]) for row in rows if int(row["label"]) == class_id),
            "constant_images": sum(bool(row["raw_constant_image"]) for row in rows if int(row["label"]) == class_id),
        }
        for class_id in range(10)
    }
    if baseline_rows is not None:
        baseline_by_source = {int(row["source_row_index"]): row for row in baseline_rows}
        deltas = [float(row["ed_margin"]) - float(baseline_by_source[int(row["source_row_index"])]["ed_margin"]) for row in rows]
        result["margin_delta_mean"] = sum(deltas) / len(deltas)
        result["margin_delta_median"] = float(torch.tensor(deltas, dtype=torch.float64).median())
        result["beneficial_rows"] = sum(value > float(eta) for value in deltas) if eta is not None else None
        result["harmful_rows"] = sum(value < -float(eta) for value in deltas) if eta is not None else None
        result["boundary_rows"] = len(deltas) - result["beneficial_rows"] - result["harmful_rows"] if eta is not None else None
        for class_id in range(10):
            class_deltas = [
                float(row["ed_margin"]) - float(baseline_by_source[int(row["source_row_index"])]["ed_margin"])
                for row in rows
                if int(row["label"]) == class_id
            ]
            result["per_class"][str(class_id)].update({
                "beneficial_rows": sum(value > float(eta) for value in class_deltas) if eta is not None else None,
                "harmful_rows": sum(value < -float(eta) for value in class_deltas) if eta is not None else None,
                "boundary_rows": sum(abs(value) <= float(eta) for value in class_deltas) if eta is not None else None,
            })
    return result


def _decision(confirmation_baseline: Sequence[Mapping[str, Any]], confirmation_states: Mapping[str, Sequence[Mapping[str, Any]]], eta: float) -> dict[str, Any]:
    base_metrics = _class_metrics(confirmation_baseline)
    state_metrics = {name: _class_metrics(rows, baseline_rows=confirmation_baseline, eta=eta) for name, rows in confirmation_states.items()}
    conditions: dict[str, Any] = {}
    focus_response_names: list[str] = []
    supported_names: list[str] = []
    for name, metrics in state_metrics.items():
        response = all(metrics["per_class"][str(class_id)]["beneficial_rows"] >= STABLE_ROWS for class_id in FOCUS_CLASSES)
        risky_classes = [class_id for class_id in range(10) if class_id not in FOCUS_CLASSES and metrics["per_class"][str(class_id)]["harmful_rows"] >= STABLE_ROWS]
        class_tradeoff = any(base_metrics["correct"][class_id] - metrics["correct"][class_id] > 1 for class_id in range(10))
        no_accuracy_regression = not class_tradeoff and metrics["macro_recall"] >= base_metrics["macro_recall"]
        mse_guard = metrics["raw_mse_mean"] <= base_metrics["raw_mse_mean"] * 1.05
        clipping_guard = metrics["raw_fraction_above_one_mean"] <= 0.01 and all(metrics["per_class"][str(class_id)]["raw_fraction_above_one_mean"] <= 0.01 for class_id in FOCUS_CLASSES)
        image_guard = metrics["raw_zero_images"] == 0 and metrics["raw_constant_images"] == 0
        no_stable_harm = not risky_classes
        passed = response and no_stable_harm and no_accuracy_regression and mse_guard and clipping_guard and image_guard
        if response:
            focus_response_names.append(name)
        if passed:
            supported_names.append(name)
        conditions[name] = {
            "focus_1_7_response": response,
            "stable_harmful_other_classes": risky_classes,
            "accuracy_guard_passed": no_accuracy_regression,
            "raw_mse_guard_passed": mse_guard,
            "clipping_guard_passed": clipping_guard,
            "zero_and_constant_image_guard_passed": image_guard,
            "supported": passed,
        }
    if supported_names:
        classification = "SHARED_LOCAL_RESPONSE_SUPPORTED_WITHIN_SCOPE"
    elif focus_response_names:
        classification = "SHARED_RESPONSE_WITH_CLASS_OR_IMAGE_TRADEOFF"
    else:
        classification = "MIXED_OR_INSUFFICIENT"
    return {
        "classification": classification,
        "baseline_confirmation": base_metrics,
        "conditions": conditions,
        "supported_conditions": supported_names,
        "focus_response_conditions": focus_response_names,
        "condition_metrics": state_metrics,
        "minimum_focus_beneficial_rows": STABLE_ROWS,
        "eta": eta,
        "training_recommendation": "NONE",
        "next_action": "NONE",
        "claim_boundary": "one common frozen phase displacement applied to each entire population; local finite response only, not accumulated training, human readability, generalization, or architecture-capacity evidence",
    }


def _pairwise_effects(state_rows: Mapping[str, Sequence[Mapping[str, Any]]], baseline_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pairs = {
        "BOGT_F_GD_minus_S_GD": ("F_GD_equal_rms_plus1", "S_GD_equal_rms_plus1"),
        "Adam_S_ADAM_minus_S_GD": ("S_ADAM_equal_rms_plus1", "S_GD_equal_rms_plus1"),
        "Adam_F_ADAM_minus_F_GD": ("F_ADAM_equal_rms_plus1", "F_GD_equal_rms_plus1"),
    }
    result: dict[str, Any] = {}
    for pair_name, (left_name, right_name) in pairs.items():
        if left_name not in state_rows or right_name not in state_rows:
            result[pair_name] = {"available": False, "reason": "one_or_both_shared_directions_are_undecidable"}
            continue
        left = {int(row["source_row_index"]): row for row in state_rows[left_name]}
        right = {int(row["source_row_index"]): row for row in state_rows[right_name]}
        values = []
        for row in baseline_rows:
            key = int(row["source_row_index"])
            values.append({"label": int(row["label"]), "role": row["role"], "delta_difference": float(left[key]["ed_margin"] - right[key]["ed_margin"])})
        result[pair_name] = {
            role: {
                str(class_id): {
                    "rows": sum(item["role"] == role and item["label"] == class_id for item in values),
                    "mean_delta_difference": sum(item["delta_difference"] for item in values if item["role"] == role and item["label"] == class_id) / sum(item["role"] == role and item["label"] == class_id for item in values),
                    "beneficial_share": sum(item["delta_difference"] > 0 for item in values if item["role"] == role and item["label"] == class_id) / sum(item["role"] == role and item["label"] == class_id for item in values),
                }
                for class_id in range(10)
            }
            for role in ("fit", "historic_holdout", "confirmation")
        }
    return result


def _write_csv_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), "cannot write an empty shared-direction row table")
    buffer = io.StringIO(newline="")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue().encode("utf-8")
    if path.exists():
        _require(path.read_bytes() == content, f"existing {path.name} differs from immutable readout chunks")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
        os.rename(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_review_pages(target: Path, inputs: Mapping[str, Any], measured_conditions: Sequence[str]) -> list[dict[str, Any]]:
    expected_panels = ("S_GD_equal_rms_plus1", "F_GD_equal_rms_plus1", "S_ADAM_equal_rms_plus1", "F_ADAM_equal_rms_plus1")
    panels = ("target", "baseline", *(name for name in expected_panels if name in measured_conditions))
    result: list[dict[str, Any]] = []
    panel_size = 84
    gap = 10
    row_height = 104
    for class_id in DISPLAY_CLASSES:
        selected = [position for position, label in enumerate(inputs["labels_600"].tolist()) if int(label) == class_id]
        _require(len(selected) == 60, f"image review class {class_id} must show all sixty fixed population rows")
        canvas = Image.new("RGB", (330 + len(panels) * panel_size + (len(panels) - 1) * gap, 38 + row_height * len(selected)), "white")
        draw = ImageDraw.Draw(canvas)
        for panel_index, name in enumerate(panels):
            heading = {"target": "target", "baseline": "baseline", "S_GD_equal_rms_plus1": "S-GD +1", "F_GD_equal_rms_plus1": "F-GD +1", "S_ADAM_equal_rms_plus1": "S-Adam +1", "F_ADAM_equal_rms_plus1": "F-Adam +1"}[name]
            draw.text((260 + panel_index * (panel_size + gap), 12), heading, fill=(20, 20, 20))
        row_order = 0
        for chunk_index in range(CHUNK_COUNT):
            positions = [position for position in selected if position // CHUNK_SIZE == chunk_index]
            if not positions:
                continue
            blocks = {name: _load_readout_chunk(target, name, chunk_index, inputs) for name in panels[1:]}
            for position in positions:
                y = 38 + row_order * row_height
                role = _role_for_position(position)
                row_block = blocks["baseline"]
                local = position % CHUNK_SIZE
                prediction = int(row_block["evaluator_logits"]["E_D"][local].argmax())
                draw.text((5, y + 8), f"{role} | row {int(inputs['source_indices_600'][position])} | ED {prediction}", fill=(20, 20, 20))
                target_view = canonical.cmos_view(inputs["images_600"][position:position + 1])[0, 0]
                image_values = [target_view, canonical.cmos_view(row_block["raw_cmos"][local:local + 1])[0, 0]]
                for condition in panels[2:]:
                    block = blocks[condition]
                    image_values.append(canonical.cmos_view(block["raw_cmos"][local:local + 1])[0, 0])
                for panel_index, image_tensor in enumerate(image_values):
                    pixels = image_tensor.detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
                    image = Image.fromarray(pixels, mode="L").resize((panel_size, panel_size), Image.Resampling.NEAREST).convert("RGB")
                    canvas.paste(image, (260 + panel_index * (panel_size + gap), y))
                row_order += 1
            del blocks
        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG", optimize=True)
        filename = f"class_{class_id}_all_roles_shared_direction_review.png"
        path = target / filename
        content = buffer.getvalue()
        if path.exists():
            _require(path.read_bytes() == content, f"existing image review {filename} differs from saved readouts")
        else:
            path.write_bytes(content)
        result.append({"path": filename, "class": class_id, "rows_displayed": len(selected), "roles": ["fit", "historic_holdout", "confirmation"], "panel_order": list(panels), "view": "target 28x28 or raw CMOS clamp[0,1] then adaptive average pool to 28x28", "fixed_range": [0.0, 1.0], "per_image_rescaling": False})
    return result


def _finalize(target: Path, inputs: Mapping[str, Any], identity: ResearchRunIdentity, preflight: Mapping[str, Any], started: float) -> dict[str, Any]:
    condition_names = [name for name, item in preflight["condition_vectors"].items() if item["available"]]
    state_names = ("baseline", *condition_names)
    rows_by_state = {name: _collect_state_rows(target, name, inputs) for name in state_names}
    baseline = rows_by_state["baseline"]
    eta = float(preflight["baseline_repeat"]["eta"])
    roles = ("fit", "historic_holdout", "confirmation")
    summaries: dict[str, Any] = {}
    for name, rows in rows_by_state.items():
        summaries[name] = {role: _class_metrics([row for row in rows if row["role"] == role], baseline_rows=[row for row in baseline if row["role"] == role] if name != "baseline" else None, eta=eta if name != "baseline" else None) for role in roles}
    confirmation_baseline = [row for row in baseline if row["role"] == "confirmation"]
    confirmation_states = {name: [row for row in rows_by_state[name] if row["role"] == "confirmation"] for name in condition_names if name.endswith("_plus1")}
    decision = _decision(confirmation_baseline, confirmation_states, eta)
    pairwise = _pairwise_effects(rows_by_state, baseline)

    table_rows: list[dict[str, Any]] = []
    for state, rows in rows_by_state.items():
        for row in rows:
            table_rows.append({"state": state, **row})
    _write_csv_once(target / "row_metrics.csv", table_rows)

    review = _write_review_pages(target, inputs, condition_names)
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "status": "VALID_COMPLETED",
        "classification": decision["classification"],
        "research_run_identity": identity.to_dict(),
        "scientific_updates": 0,
        "optimizer_updates": 0,
        "phase_state_interventions": len(condition_names),
        "reported_measurement_rows": TOTAL_ROWS * (1 + len(condition_names)),
        "baseline_repeat_rows_separate": TOTAL_ROWS,
        "population_roles": {"fit": FIT_ROWS, "historic_holdout": HISTORIC_ROWS, "confirmation": CONFIRMATION_ROWS},
        "parent_results_preserved": {
            "parent_ce_status": inputs["parent_report"]["status"],
            "parent_ce_classification": inputs["parent_report"]["classification"],
            "direction_diagnostic_status": inputs["direction_report"]["status"],
            "direction_diagnostic_classification": inputs["direction_report"]["classification"],
            "finite_response_status": inputs["finite_report"]["status"],
            "finite_response_classification": inputs["finite_report"]["classification"],
        },
        "shared_direction_construction": {
            "definition": "equal-weight mean of the twenty persisted B endpoint batch directions after each batch canonical normalization",
            "semantic_gradient_then_descent": "S-GD=-mean(gS_b); S-Adam=read-only saved Adam applied once to mean(gS_b)",
            "full_gradient_then_descent": "F-GD=-mean(gS_b+gA_b); F-Adam=read-only saved Adam applied once to mean(gS_b+gA_b)",
            "mean_of_per_batch_adam_candidates_used": False,
            "batch_decomposition_max_abs_residual": inputs["directions"]["batch_decomposition_max_abs_residual"],
            "mean_decomposition_max_abs_residual": inputs["directions"]["mean_decomposition_max_abs_residual"],
            "source_rms": inputs["directions"]["source_rms"],
            "reference_rms": inputs["directions"]["reference_rms"],
            "direction_vector_names": list(inputs["directions"]["batch_mean_descent_directions"]),
        },
        "condition_availability": preflight["condition_vectors"],
        "preflight": dict(preflight),
        "baseline_repeat": dict(preflight["baseline_repeat"]),
        "state_summaries_by_role": summaries,
        "confirmation_decision": decision,
        "paired_direction_effects_by_role_and_class": pairwise,
        "image_review": review,
        "evidence_artifacts": {
            "population": "input_population.json",
            "shared_directions": "shared_directions.pt",
            "displacements": "actual_displacements.pt",
            "preflight": "preflight.json",
            "manifest": "resume_manifest.json",
            "readout_chunks": [_chunk_name(condition, index) for condition in ("baseline", "baseline_repeat", *condition_names) for index in range(CHUNK_COUNT)],
            "row_metrics": "row_metrics.csv",
            "image_review": [item["path"] for item in review],
        },
        "training_recommendation": "NONE",
        "next_action": "NONE",
        "elapsed_seconds_observation_only": time.perf_counter() - started,
    }
    return report


def _record_snapshot_checks(consumer: Any, evaluators: Mapping[str, torch.nn.Module], parameters: Mapping[str, torch.nn.Parameter], runtime_before: Mapping[str, Any], evaluator_before: Mapping[str, Mapping[str, Any]], frozen_inputs_before: Mapping[str, Any], optimizer_before: Mapping[str, Any], inputs: Mapping[str, Any]) -> None:
    _require(_tree_equal(runtime_before, finite._snapshot(consumer.runtime)), "measurement changed B endpoint model parameters or scientific buffers")
    _require(_tree_equal(frozen_inputs_before, direction._consumer_frozen_inputs_snapshot(consumer)), "measurement changed frozen BOGT references or weights")
    _require(_tree_equal(optimizer_before, direction._cpu_tree(inputs["optimizers"]["B_class_ce"])), "measurement changed the source Adam state")
    _require(all(parameter.grad is None and not parameter.requires_grad for parameter in parameters.values()), "measurement populated parameter gradients or re-enabled a trainable phase")
    for name, model in evaluators.items():
        _require(_tree_equal(evaluator_before[name], finite._snapshot(model)), f"measurement changed evaluator {name}")
        _require(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()), f"evaluator {name} gradient isolation changed")


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    root = _root()
    artifact_dir = _resolve(root, arguments.artifact_dir)
    direction_dir = _resolve(root, arguments.direction_dir)
    finite_dir = _resolve(root, arguments.finite_response_dir)
    target = _resolve(root, arguments.output_dir)
    outputs = (root / "outputs").resolve()
    _require(target != outputs and outputs in target.parents, "output must be a named child under outputs/")
    _require(not (arguments.preflight_only and arguments.resume_manifest), "preflight-only cannot resume")
    _require(arguments.resume_manifest is not None or not target.exists(), "output exists; pass its explicit resume manifest to continue")

    inputs = _load_inputs(root, artifact_dir, direction_dir, finite_dir)
    conditions = _condition_vectors(inputs["directions"])
    population = _population_record(inputs, target)
    identity = _identity(inputs, population, target)
    existing = None
    if arguments.resume_manifest is not None:
        _require(target.is_dir(), "resume output directory is missing")
        existing = _validate_manifest(_resolve(root, arguments.resume_manifest), target, identity, population)
        terminal = _terminal_report(target, existing, identity)
        if terminal is not None:
            return terminal

    started = time.perf_counter()
    replay: dict[str, Any] | None = None
    if existing is None:
        try:
            preflight, replay = _preflight(inputs, target, started)
        except (SharedDirectionError, finite.FiniteResponseError, direction.DirectionDiagnosticError, base.DiagnosticError, OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
            failure = {"task_id": TASK_ID, "status": "INVALID_HARNESS", "classification": None, "reason": f"{type(error).__name__}: {error}", "optimizer_updates": 0, "next_action": "NONE"}
            target.mkdir(parents=True, exist_ok=False)
            finite._write_json_new(target / "failure.json", failure, "shared-direction preflight failure")
            if arguments.preflight_only:
                return failure
            return failure
        run_control = ResearchRunControl(identity=identity, authorization_id=f"{TASK_ID}:authorization:01")
        lifecycle_parent = Path(preflight["disk_observation"]["checked_path"])
        preflight = {**preflight, "storage_lifecycle": _preflight_artifact_lifecycle(identity, population, inputs, preflight, lifecycle_parent)}
        receipt = _ready_receipt(run_control, preflight)
        if receipt.status != "READY":
            failure = {"task_id": TASK_ID, "status": "INVALID_HARNESS", "classification": None, "reason": "shared-direction READY receipt was blocked", "optimizer_updates": 0, "next_action": "NONE", "preflight": preflight}
            target.mkdir(parents=True, exist_ok=False)
            finite._write_json_new(target / "failure.json", failure, "blocked shared-direction readiness")
            return failure
        run_control.mark_preflight_ready(receipt)
        if arguments.preflight_only:
            target.mkdir(parents=True, exist_ok=False)
            finite._write_json_new(target / "input_population.json", population, "shared-direction fixed population")
            finite._write_torch_new(target / "shared_directions.pt", inputs["directions"], "shared mean directions")
            finite._write_torch_new(target / "preflight_replay.pt", replay, "preflight vectors and actual float32 displacements")
            finite._write_json_new(target / "preflight.json", {"task_id": TASK_ID, "status": "READY", "receipt": receipt.to_dict(), "preflight": preflight}, "shared-direction preflight")
            return {"task_id": TASK_ID, "status": "READY", "output_dir": target.as_posix(), "optimizer_updates": 0, "next_action": "NONE"}
        run_control.start_execution()
        run_control.open_scientific_attempt(target, identity.scientific_attempt_id, receipt)
        finite._write_json_new(target / "input_population.json", population, "shared-direction fixed population")
        finite._write_torch_new(target / "shared_directions.pt", inputs["directions"], "shared mean directions")
        finite._write_torch_new(target / "preflight_replay.pt", replay, "preflight vectors and actual float32 displacements")
        finite._write_json_new(target / "preflight.json", {"task_id": TASK_ID, "status": "READY", "receipt": receipt.to_dict(), "preflight": preflight}, "shared-direction CUDA preflight")
        manifest = _manifest_seed(identity, population, receipt, preflight)
        manifest["measured_conditions"] = ["baseline", "baseline_repeat", *[name for name, spec in conditions.items() if spec["available"]]]
        finite._write_json_replace(target / "resume_manifest.json", manifest)
    else:
        preflight_file = json.loads((target / "preflight.json").read_text(encoding="utf-8"))
        _require(preflight_file.get("task_id") == TASK_ID and preflight_file.get("status") == "READY", "resume lacks the same task's READY preflight")
        receipt = PreflightReceipt.from_mapping(preflight_file["receipt"])
        preflight = existing["preflight"]
        run_control = ResearchRunControl(identity=identity, authorization_id=receipt.authorization_id)
        run_control.mark_preflight_ready(receipt)
        run_control.start_execution()
        run_control.open_scientific_attempt(target, identity.scientific_attempt_id, receipt, resume=True)
        manifest = dict(existing)
        manifest["status"] = "RUNNING"
        finite._write_json_replace(target / "resume_manifest.json", manifest)

    device = torch.device("cuda:0")
    _require(torch.cuda.is_available(), "CUDA became unavailable before the frozen shared readout")
    runtime_probe = direction._runtime_dll_probe()
    _require(runtime_probe.get("status") in {"already_present", "prepended", "not_applicable"}, "CUDA runtime DLL readiness changed before measurement")
    consumer, parameters = direction._new_consumer(inputs, "B_class_ce", device)
    for parameter in parameters.values():
        parameter.requires_grad_(False)
    evaluators = base.load_evaluators(inputs["paths"], device)
    for model in evaluators.values():
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    runtime_before = finite._snapshot(consumer.runtime)
    evaluator_before = {name: finite._snapshot(model) for name, model in evaluators.items()}
    frozen_inputs_before = direction._consumer_frozen_inputs_snapshot(consumer)
    optimizer_before = direction._cpu_tree(inputs["optimizers"]["B_class_ce"])
    baseline_phase = parameters["generator.phase"].detach().cpu().clone()
    conditions = _condition_vectors(inputs["directions"])
    actual_deltas = {name: _phase_set(parameters["generator.phase"], baseline_phase, spec["planned_delta"]) for name, spec in conditions.items() if spec["available"]}
    finite._restore_phase(parameters["generator.phase"], baseline_phase)
    if existing is None:
        finite._write_torch_new(target / "actual_displacements.pt", {"planned": {name: item["planned_delta"] for name, item in conditions.items() if item["available"]}, "actual_float32": actual_deltas, "condition_metadata": {name: {key: value for key, value in item.items() if key != "planned_delta"} for name, item in conditions.items()}}, "planned and actual shared phase displacements")
    else:
        saved_deltas = torch.load(target / "actual_displacements.pt", map_location="cpu", weights_only=True)
        _require(set(saved_deltas["actual_float32"]) == set(actual_deltas), "resume displacement set changed")
        for name in actual_deltas:
            _require(torch.equal(saved_deltas["actual_float32"][name], actual_deltas[name]), f"resume float32 applied displacement changed for {name}")

    manifest["preflight"] = preflight
    gradient_map = _load_gradient_map(inputs)
    try:
        # Establish and repeat the unchanged common B endpoint across all 600 rows.
        _require(torch.equal(parameters["generator.phase"].detach().cpu(), baseline_phase), "B endpoint phase is not at the exact frozen baseline")
        _chunks_measurement(target, manifest, "baseline", inputs, consumer, evaluators, planned_delta=None, actual_delta=None, eta=ETA_FLOOR, gradient_map=gradient_map, baseline=True)
        finite._restore_phase(parameters["generator.phase"], baseline_phase)
        _chunks_measurement(target, manifest, "baseline_repeat", inputs, consumer, evaluators, planned_delta=None, actual_delta=None, eta=ETA_FLOOR, gradient_map=gradient_map)
        baseline_repeat = _baseline_repeat_eta(target, inputs)
        preflight = {**preflight, "baseline_repeat": baseline_repeat}
        manifest["preflight"] = preflight
        finite._write_json_replace(target / "resume_manifest.json", manifest)
        finite._write_json_replace(target / "preflight.json", {"task_id": TASK_ID, "status": "READY", "receipt": receipt.to_dict(), "preflight": preflight})

        for name, spec in conditions.items():
            if not spec["available"]:
                continue
            finite._restore_phase(parameters["generator.phase"], baseline_phase)
            applied = _phase_set(parameters["generator.phase"], baseline_phase, spec["planned_delta"])
            _require(torch.equal(applied, actual_deltas[name]), f"applied common phase displacement differs for {name}")
            _chunks_measurement(target, manifest, name, inputs, consumer, evaluators, planned_delta=spec["planned_delta"], actual_delta=applied, eta=baseline_repeat["eta"], gradient_map=gradient_map)
            finite._restore_phase(parameters["generator.phase"], baseline_phase)
            _record_snapshot_checks(consumer, evaluators, parameters, runtime_before, evaluator_before, frozen_inputs_before, optimizer_before, inputs)
            _sample_resource_telemetry(target / "resource_telemetry.jsonl", device, "shared_direction_condition_complete", progress={"condition": name, "completed_chunks": len(manifest["completed_chunks"])}, started=started)
        _require(all(_chunk_name(condition, index) in manifest["completed_chunks"] for condition in manifest["measured_conditions"] for index in range(CHUNK_COUNT)), "one or more fixed 100-row readout chunks are missing")
        finite._restore_phase(parameters["generator.phase"], baseline_phase)
        _record_snapshot_checks(consumer, evaluators, parameters, runtime_before, evaluator_before, frozen_inputs_before, optimizer_before, inputs)
    except KeyboardInterrupt:
        manifest["status"] = "INTERRUPTED"
        finite._write_json_replace(target / "resume_manifest.json", manifest)
        run_control.interrupted()
        raise
    except Exception:
        finite._restore_phase(parameters["generator.phase"], baseline_phase)
        manifest["status"] = "INVALID_HARNESS"
        finite._write_json_replace(target / "resume_manifest.json", manifest)
        run_control.invalid_harness()
        raise

    try:
        report = _finalize(target, inputs, identity, preflight, started)
    except Exception:
        manifest["status"] = "INVALID_HARNESS"
        finite._write_json_replace(target / "resume_manifest.json", manifest)
        run_control.invalid_harness()
        raise
    finite._write_json_new(target / "report.json", report, "shared Generator direction transfer decision")
    manifest["status"] = "COMPLETED"
    finite._write_json_replace(target / "resume_manifest.json", manifest)
    run_control.complete()
    _sample_resource_telemetry(target / "resource_telemetry.jsonl", device, "shared_direction_complete", progress={"classification": report["classification"]}, started=started)
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--direction-dir", type=Path, default=DEFAULT_DIRECTION)
    parser.add_argument("--finite-response-dir", type=Path, default=DEFAULT_FINITE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    try:
        result = _run(arguments)
    except (SharedDirectionError, finite.FiniteResponseError, direction.DirectionDiagnosticError, base.DiagnosticError, OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        target = _resolve(_root(), arguments.output_dir)
        failure = {"task_id": TASK_ID, "status": "INVALID_HARNESS", "classification": None, "reason": f"{type(error).__name__}: {error}", "optimizer_updates": 0, "next_action": "NONE"}
        try:
            output_root = (_root() / "outputs").resolve()
            if target != output_root and output_root in target.parents:
                target.mkdir(parents=True, exist_ok=True)
                if not (target / "report.json").exists() and not (target / "failure.json").exists():
                    finite._write_json_new(target / "failure.json", failure, "shared direction invalid harness")
                manifest_path = target / "resume_manifest.json"
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest.get("status") not in {"COMPLETED", "INVALID_HARNESS"}:
                        manifest["status"] = "INVALID_HARNESS"
                        finite._write_json_replace(manifest_path, manifest)
        except OSError:
            pass
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result.get("status") == "INVALID_HARNESS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
