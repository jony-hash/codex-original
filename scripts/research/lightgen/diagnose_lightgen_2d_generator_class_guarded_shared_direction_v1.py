"""Construct and freeze-test a class-guarded common Generator displacement."""

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
import torch.nn.functional as F
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
from scripts.research.lightgen import diagnose_lightgen_2d_generator_shared_direction_transfer_v1 as shared


TASK_ID = "M0_GENERATOR_CLASS_GUARDED_SHARED_DIRECTION_V1"
REPORT_SCHEMA = "m0_generator_class_guarded_shared_direction_report.v1"
MANIFEST_SCHEMA = "m0_generator_class_guarded_shared_direction_manifest.v1"
BLOCK_SCHEMA = "m0_generator_class_guarded_shared_direction_block.v1"
GRADIENT_SCHEMA = "m0_generator_class_guarded_gradients.v1"
PROJECTION_SCHEMA = "m0_generator_class_guarded_projection.v1"
FIT_ROWS = 200
HISTORIC_ROWS = 100
CONFIRMATION_ROWS = 300
TOTAL_ROWS = 600
CHUNK_SIZE = 100
CHUNK_COUNT = TOTAL_ROWS // CHUNK_SIZE
EVALUATORS = ("E_S", "E_I", "E_C", "E_D")
GUARD_EVALUATORS = ("E_S", "E_I", "E_C")
FOCUS_CLASSES = (1, 7)
CONTROL_CLASSES = (0, 4, 8)
DISPLAY_CLASSES = (*FOCUS_CLASSES, *CONTROL_CLASSES)
GUARD_NAMES = tuple((name, class_id) for name in GUARD_EVALUATORS for class_id in range(10))
MARGIN_DEFINITION = direction.MARGIN_DEFINITION
FLOAT_RTOL = 2e-5
FLOAT_ATOL = 2e-7
ETA_FLOOR = 1e-4
STABLE_ROWS = 24
KKT_TOLERANCE = 1e-8
ZERO_DIRECTION_RELATIVE = 1e-8
MIXED_CE_GRADIENT_RELATIVE_TOLERANCE = 2e-5
ESTIMATED_OUTPUT_BYTES = 1_000_000_000
STORAGE_RESERVE_BYTES = int(1.5 * 1024**3)
DEFAULT_SHARED = Path("outputs/20261006_m0_generator_shared_direction_transfer_v1/runs/attempt_03")
DEFAULT_OUTPUT = Path("outputs/20261006_m0_generator_class_guarded_shared_direction_v1/runs/attempt_01")


class ClassGuardedDirectionError(RuntimeError):
    """Input, projection, preflight, or frozen-measurement invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ClassGuardedDirectionError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _rms(value: torch.Tensor) -> float:
    return float(value.detach().to(device="cpu", dtype=torch.float64).square().mean().sqrt())


def _resource_observation(device: torch.device) -> dict[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "device": str(device),
        "total_bytes": int(total_bytes),
        "free_bytes": int(free_bytes),
        "global_used_fraction_observation": (total_bytes - free_bytes) / total_bytes if total_bytes else None,
        "pytorch_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "pytorch_reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def _load_inputs(root: Path, shared_dir: Path) -> dict[str, Any]:
    _require(shared_dir.is_dir(), f"explicit shared-direction directory is missing: {shared_dir}")
    population = json.loads((shared_dir / "input_population.json").read_text(encoding="utf-8"))
    report = json.loads((shared_dir / "report.json").read_text(encoding="utf-8"))
    _require(
        report.get("task_id") == shared.TASK_ID
        and report.get("status") == "VALID_COMPLETED"
        and report.get("classification") == "SHARED_RESPONSE_WITH_CLASS_OR_IMAGE_TRADEOFF",
        "the named shared-direction input is not the declared valid attempt_03 result",
    )
    parent_dir = _resolve(root, population["parent_ce_directory"])
    direction_dir = _resolve(root, population["corrected_direction_directory"])
    finite_dir = _resolve(root, population["finite_response_directory"])
    inputs = shared._load_inputs(root, parent_dir, direction_dir, finite_dir)
    _require(
        inputs["source_indices_600"].tolist() == population["source_row_indices"]
        and inputs["labels_600"].tolist() == population["labels"]
        and inputs["roles_600"] == population["roles"],
        "shared-direction population differs from the explicitly recorded population",
    )
    _require(inputs["directions"]["equal_rms_directions"]["S_GD"].shape == inputs["parameters"]["B_class_ce"]["generator.phase"].shape, "source S-GD displacement shape differs from the frozen Generator phase")
    source_rms = _rms(inputs["directions"]["equal_rms_directions"]["S_GD"])
    _require(math.isfinite(source_rms) and source_rms > 0.0 and abs(source_rms - float(inputs["directions"]["reference_rms"])) <= 1e-10, "source S-GD direction is zero or differs from its recorded common RMS")
    inputs["shared_dir"] = shared_dir
    inputs["shared_report"] = report
    return inputs


def _population_record(inputs: Mapping[str, Any], output: Path) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "shared_direction_directory": inputs["shared_dir"].as_posix(),
        "parent_ce_directory": inputs["artifact_dir"].as_posix(),
        "corrected_direction_directory": inputs["direction_dir"].as_posix(),
        "finite_response_directory": inputs["finite_dir"].as_posix(),
        "checkpoint": inputs["paths"]["checkpoint_B_class_ce"].as_posix(),
        "source_row_indices": inputs["source_indices_600"].tolist(),
        "labels": inputs["labels_600"].tolist(),
        "roles": list(inputs["roles_600"]),
        "role_counts": {"fit": FIT_ROWS, "historic_holdout": HISTORIC_ROWS, "confirmation": CONFIRMATION_ROWS},
        "row_order": "parent attempt_01 original order",
        "direction_rows": "first 200 fit rows only",
        "evaluator_guard_constraints": [f"{name}:class{class_id}" for name, class_id in GUARD_NAMES],
        "confirmation_used_for_gradient_or_projection": False,
        "e_d_used_for_direction_construction": False,
        "source_direction": "previous valid shared S-GD equal-RMS positive direction",
        "candidate_scale": "projected vector rescaled to the existing common reference RMS",
        "conditions": ["baseline", "baseline_repeat", "guarded_plus1", "guarded_minus1"],
        "margin_definition": MARGIN_DEFINITION,
        "output_directory": output.as_posix(),
    }


def _identity(inputs: Mapping[str, Any], population: Mapping[str, Any], output: Path) -> ResearchRunIdentity:
    payload = inputs["payloads"]["B_class_ce"]
    attempt = output.name.replace("_", "-")
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=f"{TASK_ID}-{attempt}",
        trajectory_id="frozen-B-class-CE-step2500-class-guarded-common-generator-displacement",
        semantic_authority_id=TASK_ID,
        data_role="fixed 600 canonical TRAIN rows; first 200 fit only, last 300 confirmation only",
        config_identity={
            "propagation_grid": 2048,
            "encoder_adam_step": 2200,
            "generator_adam_step": 2500,
            "fit_rows": FIT_ROWS,
            "historic_rows": HISTORIC_ROWS,
            "confirmation_rows": CONFIRMATION_ROWS,
            "guard_constraints": [f"{name}:class{class_id}" for name, class_id in GUARD_NAMES],
            "projection": "float64 nearest-point projection with unit-L2 nonzero guard rows and KKT tolerance 1e-8",
            "candidate_rms": float(inputs["directions"]["reference_rms"]),
            "margin_definition": MARGIN_DEFINITION,
        },
        model_identity={
            "parameter_shapes": {name: list(value.shape) for name, value in payload["trainable_parameter_state"].items()},
            "source_arm": "B_class_ce",
            "completed_updates": 300,
        },
        optimizer_identity={"class": "saved Adam state read only; no optimizer constructed or stepped", "source_steps": payload["optimizer_step_by_parameter"]},
        sampler_identity={"source_indices": population["source_row_indices"], "labels": population["labels"], "roles": population["roles"]},
        execution_source_semantics_id=TASK_ID,
        scientific_attempt_id=f"{TASK_ID}:{attempt}",
    )


def _class_ce_gradients(
    consumer: Any,
    evaluators: Mapping[str, torch.nn.Module],
    generator_phase: torch.nn.Parameter,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    """Produce each evaluator/class mean CE gradient from the full image path."""
    _require(images.shape[0] == FIT_ROWS and labels.shape == (FIT_ROWS,), "guard gradients must consume exactly the original 200 fit rows")
    gradients: dict[str, dict[int, torch.Tensor]] = {name: {} for name in GUARD_EVALUATORS}
    row_map: dict[str, dict[str, Any]] = {}
    for evaluator_name in GUARD_EVALUATORS:
        row_map[evaluator_name] = {}
        for class_id in range(10):
            positions = torch.nonzero(labels == class_id, as_tuple=False).reshape(-1)
            _require(positions.numel() == 20, f"{evaluator_name} class {class_id} must have exactly twenty fit rows")
            batch_gradients: list[torch.Tensor] = []
            for batch_positions in (positions[:10], positions[10:]):
                image_batch = images.index_select(0, batch_positions).to(consumer.device)
                label_batch = labels.index_select(0, batch_positions).to(consumer.device)
                with torch.inference_mode(False):
                    breakdown = consumer.forward_loss(image_batch, data_role=direction.bogt.TRAIN_ROLE)
                    view = canonical.cmos_view(breakdown.result.raw_intensity)
                    output = evaluators[evaluator_name](view)
                    logits = output[1]
                    loss = F.cross_entropy(logits, label_batch, reduction="mean")
                    gradient = torch.autograd.grad(loss, generator_phase, retain_graph=False, create_graph=False)[0]
                _require(tuple(gradient.shape) == tuple(generator_phase.shape) and bool(torch.isfinite(gradient).all()), f"{evaluator_name} class {class_id} produced an invalid Generator guard gradient")
                batch_gradients.append(gradient.detach().to(device="cpu", dtype=torch.float32))
                del breakdown, view, output, logits, loss, gradient, image_batch, label_batch
            gradients[evaluator_name][class_id] = torch.stack(batch_gradients).mean(0)
            row_map[evaluator_name][str(class_id)] = {
                "fit_positions": positions.tolist(),
                "source_row_indices": [],
            }
    return {"schema": GRADIENT_SCHEMA, "gradient_definition": "mean CE over twenty fit rows; equal mean of two ordered ten-row batch gradients", "gradients": gradients, "fit_row_map": row_map, "batch_count_per_constraint": 2}


def _attach_source_rows(bundle: dict[str, Any], source_indices: torch.Tensor) -> None:
    for evaluator_name in GUARD_EVALUATORS:
        for class_id in range(10):
            positions = bundle["fit_row_map"][evaluator_name][str(class_id)]["fit_positions"]
            bundle["fit_row_map"][evaluator_name][str(class_id)]["source_row_indices"] = source_indices[positions].tolist()


def _validate_gradient_population(bundle: Mapping[str, Any], labels: torch.Tensor, source_indices: torch.Tensor) -> None:
    _require(bundle.get("schema") == GRADIENT_SCHEMA, "saved class guard gradients use unknown semantics")
    _require(labels.shape == source_indices.shape == (FIT_ROWS,), "guard-gradient population metadata must cover exactly two hundred fit rows")
    for evaluator_name in GUARD_EVALUATORS:
        _require(set(bundle["fit_row_map"].get(evaluator_name, {})) == {str(class_id) for class_id in range(10)}, f"{evaluator_name} class gradient row mapping is incomplete")
        for class_id in range(10):
            expected_positions = torch.nonzero(labels == class_id, as_tuple=False).reshape(-1)
            saved = bundle["fit_row_map"][evaluator_name][str(class_id)]
            _require(saved.get("fit_positions") == expected_positions.tolist(), f"{evaluator_name} class {class_id} gradient uses the wrong fit-row mask")
            _require(saved.get("source_row_indices") == source_indices[expected_positions].tolist(), f"{evaluator_name} class {class_id} gradient source rows changed")


def _validate_class_gradient_means(bundle: Mapping[str, Any], source_blocks: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    _require(len(source_blocks) == 20, "raw mixed-batch CE reference requires twenty saved fit batches")
    differences: dict[str, dict[str, float]] = {}
    for name in GUARD_EVALUATORS:
        class_mean = torch.stack([bundle["gradients"][name][class_id].to(torch.float64) for class_id in range(10)]).mean(0)
        saved_batch_gradients = []
        key = f"ce_component_gradient_descent_{name}"
        for block in source_blocks:
            _require(key in block["candidate_vectors_generator"], f"saved raw mixed-batch gradient is missing {name}")
            saved_batch_gradients.append(-block["candidate_vectors_generator"][key].to(torch.float64))
        mixed_mean = torch.stack(saved_batch_gradients).mean(0)
        absolute_l2_error = float((class_mean - mixed_mean).norm())
        reference_l2_norm = float(mixed_mean.norm())
        relative_l2_error = absolute_l2_error / reference_l2_norm if reference_l2_norm > 0.0 else absolute_l2_error
        allowed_l2_error = MIXED_CE_GRADIENT_RELATIVE_TOLERANCE * reference_l2_norm + 1e-12
        _require(absolute_l2_error <= allowed_l2_error, f"{name} classwise CE gradient mean differs from the saved mixed-batch reference: relative_l2={relative_l2_error:.9g}, absolute_l2={absolute_l2_error:.9g}, allowed_l2={allowed_l2_error:.9g}")
        differences[name] = {
            "absolute_l2_error": absolute_l2_error,
            "relative_l2_error": relative_l2_error,
            "reference_l2_norm": reference_l2_norm,
            "allowed_l2_error": allowed_l2_error,
        }
    return differences


def _project_guarded_direction(source: torch.Tensor, guard_gradients: Mapping[str, Mapping[int, torch.Tensor]]) -> dict[str, Any]:
    """Solve min ||v-source||^2/2 subject to every nonzero q dot v <= 0."""
    source64 = source.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    _require(source64.numel() > 0 and bool(torch.isfinite(source64).all()), "source shared direction is malformed")
    names: list[str] = []
    raw_norms: dict[str, float] = {}
    rows: list[torch.Tensor] = []
    for evaluator_name in GUARD_EVALUATORS:
        for class_id in range(10):
            key = f"{evaluator_name}:class{class_id}"
            value = guard_gradients[evaluator_name][class_id].detach().to(device="cpu", dtype=torch.float64).reshape(-1)
            _require(value.shape == source64.shape and bool(torch.isfinite(value).all()), f"guard {key} shape or finiteness differs from source direction")
            norm = float(value.norm())
            raw_norms[key] = norm
            if norm == 0.0:
                continue
            names.append(key)
            rows.append(value / norm)
    if rows:
        matrix = torch.stack(rows)
        gram = matrix @ matrix.T
        linear = matrix @ source64
        multipliers = torch.zeros(matrix.shape[0], dtype=torch.float64)
        for sweep in range(1, 1_000_001):
            for index in range(matrix.shape[0]):
                updated = max(0.0, float(multipliers[index] + linear[index] - torch.dot(gram[index], multipliers)))
                multipliers[index] = updated
            projected = source64 - matrix.T @ multipliers
            products = matrix @ projected
            primal_residual = float(torch.clamp(products, min=0.0).max())
            complementarity_residual = float((multipliers * products).abs().max())
            stationarity_residual = float((projected - source64 + matrix.T @ multipliers).abs().max())
            if max(primal_residual, complementarity_residual, stationarity_residual) <= KKT_TOLERANCE:
                break
        else:
            raise ClassGuardedDirectionError("class-guard projection did not meet its KKT residual tolerance")
        multiplier_map = {name: float(value) for name, value in zip(names, multipliers.tolist(), strict=True)}
        constraint_products = {name: float(value) for name, value in zip(names, products.tolist(), strict=True)}
    else:
        projected = source64.clone()
        sweep = 0
        primal_residual = complementarity_residual = stationarity_residual = 0.0
        multiplier_map = {}
        constraint_products = {}
    source_norm = float(source64.norm())
    projected_norm = float(projected.norm())
    retained = projected_norm / source_norm if source_norm else 0.0
    scale = _rms(source)
    projected_rms = _rms(projected)
    unavailable = source_norm == 0.0 or retained <= ZERO_DIRECTION_RELATIVE or projected_rms == 0.0
    candidate = None if unavailable else (projected * (scale / projected_rms)).to(dtype=source.dtype).reshape_as(source)
    projected_shaped = projected.reshape_as(source).cpu()
    return {
        "schema": PROJECTION_SCHEMA,
        "solver": "deterministic cyclic nonnegative dual coordinate descent",
        "source_direction": source.detach().cpu().clone(),
        "projected_minimum_distance_vector": projected_shaped,
        "candidate_equal_rms_direction": candidate,
        "reference_rms": scale,
        "projected_vector_rms": projected_rms,
        "projected_to_source_l2_ratio": retained,
        "direction_cosine": None if source_norm == 0.0 or projected_norm == 0.0 else float(torch.dot(source64, projected) / (source_norm * projected_norm)),
        "zero_gradient_constraints": [key for key, norm in raw_norms.items() if norm == 0.0],
        "guard_gradient_norms": raw_norms,
        "active_constraint_multipliers": multiplier_map,
        "unit_guard_dot_projected": constraint_products,
        "solver_sweeps": sweep,
        "kkt_residuals": {
            "primal_feasibility": primal_residual,
            "complementarity": complementarity_residual,
            "stationarity": stationarity_residual,
            "maximum": max(primal_residual, complementarity_residual, stationarity_residual),
            "tolerance": KKT_TOLERANCE,
        },
        "candidate_available": candidate is not None,
        "candidate_unavailable_reason": "projected vector is within the declared numerical zero range" if unavailable else None,
    }


def _projection_active_subset_reference(source: torch.Tensor, constraints: torch.Tensor) -> torch.Tensor:
    """Independent exhaustive active-set reference for small CPU test problems."""
    from itertools import combinations

    source64 = source.detach().to(torch.float64).reshape(-1)
    rows = constraints.detach().to(torch.float64).reshape(-1, source64.numel())
    best: tuple[float, torch.Tensor] | None = None
    for count in range(rows.shape[0] + 1):
        for active in combinations(range(rows.shape[0]), count):
            if active:
                matrix = rows[list(active)]
                gram = matrix @ matrix.T
                rhs = matrix @ source64
                multipliers = torch.linalg.lstsq(gram, rhs).solution
                if bool((multipliers < -1e-10).any()):
                    continue
                candidate = source64 - matrix.T @ multipliers.clamp_min(0.0)
            else:
                multipliers = torch.empty(0, dtype=torch.float64)
                candidate = source64.clone()
            products = rows @ candidate
            if bool((products > 1e-9).any()):
                continue
            if active and float((multipliers * products[list(active)]).abs().max()) > 1e-8:
                continue
            objective = float((candidate - source64).square().sum())
            if best is None or objective < best[0]:
                best = (objective, candidate)
    _require(best is not None, "independent active-subset reference found no feasible projection")
    return best[1].reshape_as(source)


def _cpu_reference_checks() -> dict[str, Any]:
    source = torch.tensor([1.0, 1.0], dtype=torch.float64)
    guards = {
        "E_S": {0: torch.tensor([1.0, 0.0]), **{c: torch.zeros(2) for c in range(1, 10)}},
        "E_I": {0: torch.tensor([-1.0, 0.0]), **{c: torch.zeros(2) for c in range(1, 10)}},
        "E_C": {c: torch.zeros(2) for c in range(10)},
    }
    actual = _project_guarded_direction(source, guards)
    _require(actual["candidate_available"] and torch.allclose(actual["projected_minimum_distance_vector"], torch.tensor([0.0, 1.0], dtype=torch.float64), atol=1e-8, rtol=0.0), "opposing evaluator constraints were averaged or the projection sign is wrong")
    averaged = (guards["E_S"][0] + guards["E_I"][0]) / 2.0
    _require(torch.equal(averaged, torch.zeros_like(averaged)) and not torch.allclose(actual["projected_minimum_distance_vector"], source), "opposing-evaluator counterexample does not falsify a mean-gradient implementation")

    source3 = torch.tensor([0.8, 0.3], dtype=torch.float64)
    rows = torch.tensor([[1.0, 0.0], [0.6, 0.8], [-0.6, 0.8]], dtype=torch.float64)
    active_guards = {"E_S": {0: rows[0], **{c: torch.zeros(2) for c in range(1, 10)}}, "E_I": {0: rows[1], **{c: torch.zeros(2) for c in range(1, 10)}}, "E_C": {0: rows[2], **{c: torch.zeros(2) for c in range(1, 10)}}}
    solved = _project_guarded_direction(source3, active_guards)
    reference = _projection_active_subset_reference(source3, rows)
    _require(torch.allclose(solved["projected_minimum_distance_vector"], reference, atol=1e-8, rtol=1e-8), "production dual projection differs from independent exhaustive active-set solution")

    phi = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    logits = torch.stack((-phi[0] + 1.0, phi[0], torch.tensor(-2.0, dtype=torch.float64))).reshape(1, 3)
    loss = F.cross_entropy(logits, torch.tensor([0]))
    derivative = torch.autograd.grad(loss, phi)[0]
    epsilon = 1e-5
    with torch.no_grad():
        plus = F.cross_entropy(torch.tensor([[1.0 - epsilon, epsilon, -2.0]], dtype=torch.float64), torch.tensor([0]))
        minus = F.cross_entropy(torch.tensor([[1.0 + epsilon, -epsilon, -2.0]], dtype=torch.float64), torch.tensor([0]))
    finite_difference = float((plus - minus) / (2.0 * epsilon))
    _require(abs(float(derivative) - finite_difference) < 1e-8 and float(derivative) > 0.0, "CE guard gradient value, derivative, or sign differs from independent central difference")
    return {
        "passed": True,
        "opposing_evaluator_constraints": {
            "averaged_guard_is_zero": True,
            "retained_projection": actual["projected_minimum_distance_vector"].tolist(),
            "independent_constraints_remained_active": True,
        },
        "active_subset_projection": {
            "production_solution": solved["projected_minimum_distance_vector"].tolist(),
            "independent_exhaustive_solution": reference.tolist(),
            "maximum_absolute_difference": float((solved["projected_minimum_distance_vector"] - reference).abs().max()),
        },
        "cross_entropy_gradient": {
            "autograd_derivative": float(derivative),
            "central_finite_difference": finite_difference,
            "epsilon": epsilon,
        },
    }


def _measure_fit_mean_ce(
    consumer: Any,
    evaluators: Mapping[str, torch.nn.Module],
    images: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, dict[int, float]]:
    values: dict[str, dict[int, float]] = {name: {} for name in GUARD_EVALUATORS}
    for name in GUARD_EVALUATORS:
        for class_id in (1, 7, 8):
            positions = torch.nonzero(labels == class_id, as_tuple=False).reshape(-1)
            batch_losses: list[torch.Tensor] = []
            for start in range(0, positions.numel(), 10):
                selected = positions[start:start + 10]
                with torch.inference_mode(False), torch.no_grad():
                    result = consumer.runtime.forward_from_image(images.index_select(0, selected).to(consumer.device))
                    view = canonical.cmos_view(result.raw_intensity)
                    logits = evaluators[name](view)[1]
                    loss = F.cross_entropy(logits, labels.index_select(0, selected).to(consumer.device), reduction="mean")
                batch_losses.append(loss.detach().cpu() * (selected.numel() / positions.numel()))
                del result, view, logits, loss
            values[name][class_id] = float(torch.stack(batch_losses).sum())
    return values


def _finite_difference_checks(
    consumer: Any,
    evaluators: Mapping[str, torch.nn.Module],
    parameters: Mapping[str, torch.nn.Parameter],
    inputs: Mapping[str, Any],
    gradients: Mapping[str, Any],
    projection: Mapping[str, Any],
    baseline_phase: torch.Tensor,
) -> dict[str, Any]:
    candidate = projection["candidate_equal_rms_direction"]
    if candidate is None:
        return {"passed": True, "not_applicable": True, "reason": projection["candidate_unavailable_reason"], "checks": {}}
    phase = parameters["generator.phase"]
    records: dict[str, Any] = {}
    for class_id in (1, 7, 8):
        mean_gradient = torch.stack([gradients["gradients"][name][class_id].to(torch.float64) for name in GUARD_EVALUATORS]).mean(0).reshape(-1)
        for epsilon in (1.0 / 8.0, 1.0 / 16.0):
            plus_applied = shared._phase_set(phase, baseline_phase, candidate * epsilon)
            plus_values = _measure_fit_mean_ce(consumer, evaluators, inputs["images_600"][:FIT_ROWS], inputs["labels_600"][:FIT_ROWS])
            plus_loss = sum(plus_values[name][class_id] for name in GUARD_EVALUATORS) / len(GUARD_EVALUATORS)
            plus_phase = phase.detach().cpu().clone()
            finite._restore_phase(phase, baseline_phase)
            minus_applied = shared._phase_set(phase, baseline_phase, -candidate * epsilon)
            minus_values = _measure_fit_mean_ce(consumer, evaluators, inputs["images_600"][:FIT_ROWS], inputs["labels_600"][:FIT_ROWS])
            minus_loss = sum(minus_values[name][class_id] for name in GUARD_EVALUATORS) / len(GUARD_EVALUATORS)
            minus_phase = phase.detach().cpu().clone()
            finite._restore_phase(phase, baseline_phase)
            actual_rate = (plus_phase.to(torch.float64) - minus_phase.to(torch.float64)).reshape(-1) / (2.0 * epsilon)
            predicted_slope = float(torch.dot(mean_gradient, actual_rate))
            observed_slope = (plus_loss - minus_loss) / (2.0 * epsilon)
            tolerance = 0.10 * abs(predicted_slope) + 1e-4
            key = f"class{class_id}/epsilon{epsilon:g}"
            records[key] = {
                "class_id": class_id,
                "epsilon": epsilon,
                "observed_central_difference_slope": observed_slope,
                "predicted_slope_from_actual_phase_difference": predicted_slope,
                "absolute_error": abs(observed_slope - predicted_slope),
                "tolerance": tolerance,
                "passed": abs(observed_slope - predicted_slope) <= tolerance,
                "actual_plus_displacement_rms": _rms(plus_applied),
                "actual_minus_displacement_rms": _rms(minus_applied),
                "evaluators": list(GUARD_EVALUATORS),
            }
    return {"passed": all(item["passed"] for item in records.values()), "not_applicable": False, "checks": records}


def _target_controls(evaluators: Mapping[str, torch.nn.Module], inputs: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return shared._target_positive_controls(evaluators, inputs["images_600"], inputs["labels_600"], device)


def _preflight_artifact_lifecycle(
    identity: ResearchRunIdentity,
    population: Mapping[str, Any],
    preflight: Mapping[str, Any],
    scratch_parent: Path,
) -> dict[str, bool]:
    with tempfile.TemporaryDirectory(prefix="lightgen_class_guard_preflight_", dir=scratch_parent) as scratch_name:
        scratch = Path(scratch_name)
        sample = {"schema": BLOCK_SCHEMA, "task_id": TASK_ID, "condition": "baseline", "chunk_index": 0, "block": {"row_start": 0, "row_stop": CHUNK_SIZE, "kind": "baseline", "source_row_indices": torch.arange(CHUNK_SIZE), "labels": torch.zeros(CHUNK_SIZE, dtype=torch.long), "roles": ["fit"] * CHUNK_SIZE}}
        finite._write_torch_new(scratch / "immutable.pt", sample, "preflight immutable output")
        saved = torch.load(scratch / "immutable.pt", map_location="cpu", weights_only=True)
        _require(
            saved["schema"] == sample["schema"]
            and saved["task_id"] == sample["task_id"]
            and torch.equal(saved["block"]["source_row_indices"], sample["block"]["source_row_indices"])
            and torch.equal(saved["block"]["labels"], sample["block"]["labels"]),
            "preflight immutable write/read did not preserve evidence",
        )
        measured = ["baseline", "baseline_repeat", *(["guarded_plus1", "guarded_minus1"] if preflight["projection"]["candidate_available"] else [])]
        expected = [f"readout_{name}_rows{index * CHUNK_SIZE:03d}_{(index + 1) * CHUNK_SIZE:03d}.pt" for name in measured for index in range(CHUNK_COUNT)]
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "task_id": TASK_ID,
            "measurement_semantics": "independent E_S/E_I/E_C per-class CE guards; projected S-GD; full raw-logit classwise mean CE gradients",
            "identity": identity.to_dict(),
            "population": dict(population),
            "completed_chunks": ["readout_baseline_rows000_100.pt"],
            "measured_conditions": measured,
            "status": "INTERRUPTED",
        }
        manifest_path = scratch / "resume_manifest.json"
        finite._write_json_new(manifest_path, manifest, "preflight resumable manifest")
        resumed = _validate_manifest(manifest_path, scratch, identity, population)
        _require(resumed["status"] == "INTERRUPTED" and resumed["completed_chunks"] == manifest["completed_chunks"], "preflight did not preserve the explicit interrupted cursor")
        for name in expected:
            (scratch / name).touch(exist_ok=True)
        manifest["completed_chunks"] = expected
        manifest["status"] = "COMPLETED"
        finite._write_json_replace(manifest_path, manifest)
        terminal = {"schema": REPORT_SCHEMA, "task_id": TASK_ID, "status": "VALID_COMPLETED", "classification": "MIXED_OR_INSUFFICIENT", "research_run_identity": identity.to_dict()}
        finite._write_json_new(scratch / "report.json", terminal, "preflight terminal report")
        _require(_terminal_report(scratch, manifest, identity) == terminal, "completed resume did not return its report without additional measurement")
    return {"immutable_write_read_passed": True, "interrupted_manifest_recovery_passed": True, "completed_resume_without_forward_passed": True}


def _preflight(inputs: Mapping[str, Any], output: Path, identity: ResearchRunIdentity, population: Mapping[str, Any], started: float) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    device = torch.device("cuda:0")
    _require(torch.cuda.is_available(), "CUDA is unavailable for the class-guarded frozen preflight")
    runtime_probe = direction._runtime_dll_probe()
    _require(runtime_probe.get("status") in {"already_present", "prepended", "not_applicable"}, "CUDA runtime DLL readiness failed")
    cpu_references = _cpu_reference_checks()
    cuda_before = _resource_observation(device)
    disk_root = output
    while not disk_root.exists():
        disk_root = disk_root.parent
    disk = shutil.disk_usage(disk_root)
    disk_observation = {
        "checked_path": disk_root.as_posix(),
        "free_bytes_observed": int(disk.free),
        "estimated_output_bytes_observation": ESTIMATED_OUTPUT_BYTES,
        "reserve_bytes": STORAGE_RESERVE_BYTES,
        "required_bytes": ESTIMATED_OUTPUT_BYTES + STORAGE_RESERVE_BYTES,
        "passed": disk.free >= ESTIMATED_OUTPUT_BYTES + STORAGE_RESERVE_BYTES,
    }
    _require(disk_observation["passed"], "output volume lacks the estimated artifact size plus the declared 1.5 GiB reserve")

    consumer, parameters = direction._new_consumer(inputs, "B_class_ce", device)
    parameters["encoder.phase"].requires_grad_(False)
    parameters["generator.phase"].requires_grad_(False)
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

    controls = _target_controls(evaluators, inputs, device)
    controls_qualified = finite._qualify_target_controls(controls)
    _require(controls_qualified, "one or more frozen evaluator target positive controls failed")
    path_checks: dict[str, Any] = {}
    with torch.inference_mode(False), torch.no_grad():
        for role, position in (("fit", 0), ("confirmation", FIT_ROWS + HISTORIC_ROWS)):
            image = inputs["images_600"][position:position + 10].to(device)
            result = consumer.runtime.forward_from_image(image)
            expected_z = inputs["readouts"]["B_class_ce"]["z"][position:position + 10].to(device)
            decoded = consumer.runtime.forward_from_ols(expected_z)
            _require(torch.allclose(result.z_ols.coefficients, expected_z, rtol=FLOAT_RTOL, atol=FLOAT_ATOL), f"{role} Encoder OLS does not reproduce the saved B readout")
            _require(torch.allclose(result.raw_intensity, decoded.raw_intensity, rtol=FLOAT_RTOL, atol=FLOAT_ATOL), f"{role} Encoder path differs from fixed-OLS production decoding")
            path_checks[role] = {"source_rows": inputs["source_indices_600"][position:position + 10].tolist(), "encoder_ols_matches_saved": True, "encoder_and_fixed_ols_cmos_match": True}

    parameters["generator.phase"].requires_grad_(True)
    gradients = _class_ce_gradients(consumer, evaluators, parameters["generator.phase"], inputs["images_600"][:FIT_ROWS], inputs["labels_600"][:FIT_ROWS])
    _attach_source_rows(gradients, inputs["source_indices_600"][:FIT_ROWS])
    _validate_gradient_population(gradients, inputs["labels_600"][:FIT_ROWS], inputs["source_indices_600"][:FIT_ROWS])
    mixed_gradient_differences = _validate_class_gradient_means(gradients, inputs["direction_blocks"])
    projection = _project_guarded_direction(inputs["directions"]["equal_rms_directions"]["S_GD"], gradients["gradients"])
    finite_differences = _finite_difference_checks(consumer, evaluators, parameters, inputs, gradients, projection, baseline_phase)
    _require(finite_differences["passed"], "candidate failed one or more two-scale CE finite-difference checks")
    actual_displacements: dict[str, torch.Tensor] = {}
    if projection["candidate_equal_rms_direction"] is not None:
        for name, planned in (
            ("guarded_plus1", projection["candidate_equal_rms_direction"]),
            ("guarded_minus1", -projection["candidate_equal_rms_direction"]),
        ):
            actual_displacements[name] = shared._phase_set(parameters["generator.phase"], baseline_phase, planned)
            finite._restore_phase(parameters["generator.phase"], baseline_phase)
    projection = {**projection, "actual_float32_displacements": actual_displacements}
    _require(parameters["generator.phase"].grad is None and parameters["encoder.phase"].grad is None, "read-only CE gradients accumulated into model parameter .grad")
    parameters["generator.phase"].requires_grad_(False)

    model_unchanged = finite._tree_equal(runtime_before, finite._snapshot(consumer.runtime))
    frozen_inputs_unchanged = finite._tree_equal(frozen_inputs_before, direction._consumer_frozen_inputs_snapshot(consumer))
    optimizer_unchanged = finite._tree_equal(optimizer_before, direction._cpu_tree(inputs["optimizers"]["B_class_ce"]))
    evaluator_unchanged = all(finite._tree_equal(evaluator_before[name], finite._snapshot(model)) for name, model in evaluators.items())
    _require(model_unchanged and frozen_inputs_unchanged and optimizer_unchanged and evaluator_unchanged, "preflight changed frozen scientific or evaluator state")
    _require(all(parameter.grad is None and not parameter.requires_grad for parameter in parameters.values()), "preflight did not leave all optical parameters frozen and grad-free")
    _require(all(parameter.grad is None and not parameter.requires_grad for model in evaluators.values() for parameter in model.parameters()), "preflight did not leave evaluator parameters frozen and grad-free")
    scratch_parent = output
    while not scratch_parent.exists():
        scratch_parent = scratch_parent.parent
    storage_lifecycle = _preflight_artifact_lifecycle(identity, population, {"projection": projection}, scratch_parent)
    cuda_after = _resource_observation(device)
    preflight = {
        "schema": "m0_generator_class_guarded_shared_direction_preflight.v1",
        "task_id": TASK_ID,
        "status": "READY",
        "runtime_dll": runtime_probe,
        "cpu_independent_references": cpu_references,
        "cuda_before": cuda_before,
        "cuda_after": cuda_after,
        "disk_observation": disk_observation,
        "target_positive_controls": controls,
        "target_positive_controls_qualified": controls_qualified,
        "real_encoder_vs_fixed_ols": path_checks,
        "class_gradient_production": {
            "passed": True,
            "constraints": len(GUARD_NAMES),
            "backward_calls": len(GUARD_NAMES) * 2,
            "definition": gradients["gradient_definition"],
            "mean_by_class_matches_saved_mixed_batch_mean": mixed_gradient_differences,
        },
        "projection": {
            "passed": projection["kkt_residuals"]["maximum"] <= KKT_TOLERANCE,
            "candidate_available": projection["candidate_available"],
            "solver_sweeps": projection["solver_sweeps"],
            "kkt_residuals": projection["kkt_residuals"],
            "zero_gradient_constraints": projection["zero_gradient_constraints"],
            "projected_to_source_l2_ratio": projection["projected_to_source_l2_ratio"],
            "reference_rms": projection["reference_rms"],
            "candidate_rms": None if projection["candidate_equal_rms_direction"] is None else _rms(projection["candidate_equal_rms_direction"]),
        },
        "finite_difference_checks": finite_differences,
        "source_adam_unchanged": optimizer_unchanged,
        "model_and_scientific_buffers_unchanged": model_unchanged and frozen_inputs_unchanged,
        "evaluator_state_unchanged": evaluator_unchanged,
        "parameter_grad_buffers_empty": True,
        "optimizer_updates": 0,
        "storage_lifecycle": storage_lifecycle,
        "elapsed_seconds_observation_only": time.perf_counter() - started,
    }
    _require(preflight["projection"]["passed"], "projection KKT residual exceeded the declared tolerance")
    return preflight, gradients, projection


def _ready_receipt(control: ResearchRunControl, preflight: Mapping[str, Any]) -> PreflightReceipt:
    controls = preflight.get("target_positive_controls", {})
    path_checks = preflight.get("real_encoder_vs_fixed_ols", {})
    lifecycle = preflight.get("storage_lifecycle", {})
    correctness = (
        preflight.get("status") == "READY"
        and preflight.get("cpu_independent_references", {}).get("passed") is True
        and preflight.get("target_positive_controls_qualified") is True
        and set(controls) == set(EVALUATORS)
        and set(path_checks) == {"fit", "confirmation"}
        and all(item.get("encoder_ols_matches_saved") and item.get("encoder_and_fixed_ols_cmos_match") for item in path_checks.values())
        and preflight.get("class_gradient_production", {}).get("passed") is True
        and len(preflight.get("class_gradient_production", {}).get("mean_by_class_matches_saved_mixed_batch_mean", {})) == len(GUARD_EVALUATORS)
        and preflight.get("projection", {}).get("passed") is True
        and preflight.get("finite_difference_checks", {}).get("passed") is True
        and preflight.get("model_and_scientific_buffers_unchanged") is True
        and preflight.get("source_adam_unchanged") is True
        and preflight.get("evaluator_state_unchanged") is True
        and preflight.get("parameter_grad_buffers_empty") is True
        and preflight.get("optimizer_updates") == 0
        and all(lifecycle.get(key) is True for key in ("immutable_write_read_passed", "interrupted_manifest_recovery_passed", "completed_resume_without_forward_passed"))
    )
    disk = preflight.get("disk_observation", {})
    hardware_ok = all(int(preflight.get(key, {}).get("total_bytes", 0)) > 0 and 0 <= int(preflight.get(key, {}).get("free_bytes", -1)) <= int(preflight.get(key, {}).get("total_bytes", 0)) for key in ("cuda_before", "cuda_after"))
    runtime_ok = preflight.get("runtime_dll", {}).get("status") in {"already_present", "prepended", "not_applicable"}
    resources = hardware_ok and disk.get("passed") is True and runtime_ok
    implementation = correctness and preflight.get("optimizer_updates") == 0
    return control.make_ready_receipt(
        correctness={"passed": correctness, "status": "PASS" if correctness else "BLOCKED", "independent_reference": preflight.get("cpu_independent_references", {}).get("passed") is True, "class_gradient_producer": preflight.get("class_gradient_production", {}).get("passed") is True, "projection_and_finite_difference": preflight.get("projection", {}).get("passed") is True and preflight.get("finite_difference_checks", {}).get("passed") is True, "frozen_state": preflight.get("model_and_scientific_buffers_unchanged") is True and preflight.get("source_adam_unchanged") is True and preflight.get("evaluator_state_unchanged") is True},
        resource={"passed": resources, "status": "PASS" if resources else "BLOCKED", "device_telemetry_valid": hardware_ok, "whole_device_memory_observation_only": True, "storage": disk, "runtime_dll": runtime_ok},
        implementation={"passed": implementation, "status": "PASS" if implementation else "BLOCKED", "selected_engine": "cuda_2048_frozen_class_guarded_displacement", "optimizer_updates_zero": preflight.get("optimizer_updates") == 0, "guard_count": len(GUARD_NAMES)},
        selected_engine="cuda_2048_frozen_class_guarded_displacement",
        resource_envelope={"cuda_before": preflight.get("cuda_before"), "cuda_after": preflight.get("cuda_after"), "storage": disk},
    )


def _manifest_seed(identity: ResearchRunIdentity, population: Mapping[str, Any], receipt: PreflightReceipt, preflight: Mapping[str, Any], projection: Mapping[str, Any]) -> dict[str, Any]:
    conditions = ["baseline", "baseline_repeat"]
    if projection["candidate_available"]:
        conditions.extend(("guarded_plus1", "guarded_minus1"))
    return {
        "schema": MANIFEST_SCHEMA,
        "task_id": TASK_ID,
        "measurement_semantics": "independent E_S/E_I/E_C per-class CE guards; projected S-GD; full raw-logit classwise mean CE gradients",
        "identity": identity.to_dict(),
        "population": dict(population),
        "receipt": receipt.to_dict(),
        "preflight": dict(preflight),
        "measured_conditions": conditions,
        "completed_chunks": [],
        "status": "READY",
    }


def _validate_manifest(path: Path, target: Path, identity: ResearchRunIdentity, population: Mapping[str, Any]) -> dict[str, Any]:
    _require(path.resolve() == (target / "resume_manifest.json").resolve() and path.is_file(), "resume requires this output's explicit resume_manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == MANIFEST_SCHEMA and manifest.get("task_id") == TASK_ID and manifest.get("measurement_semantics", "").startswith("independent E_S/E_I/E_C"), "resume manifest has different task or measurement semantics")
    _require(manifest.get("population") == dict(population), "resume population, inputs, rows, labels, or roles changed")
    check_resume_compatibility(identity, manifest.get("identity", {}))
    _require(manifest.get("status") in {"READY", "RUNNING", "INTERRUPTED", "COMPLETED"}, "resume manifest status is not resumable")
    return manifest


def _terminal_report(target: Path, manifest: Mapping[str, Any], identity: ResearchRunIdentity) -> dict[str, Any] | None:
    report_path = target / "report.json"
    if manifest.get("status") == "COMPLETED":
        _require(report_path.is_file(), "completed manifest lacks the final report")
    elif not report_path.is_file():
        return None
    expected = {_chunk_name(condition, index) for condition in manifest.get("measured_conditions", []) for index in range(CHUNK_COUNT)}
    _require(expected.issubset(set(manifest.get("completed_chunks", []))) and all((target / name).is_file() for name in expected), "terminal report lacks completed immutable readout chunks")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report.get("schema") == REPORT_SCHEMA and report.get("task_id") == TASK_ID and report.get("status") == "VALID_COMPLETED" and report.get("research_run_identity") == identity.to_dict(), "saved terminal report is invalid")
    if manifest.get("status") != "COMPLETED":
        updated = dict(manifest)
        updated["status"] = "COMPLETED"
        finite._write_json_replace(target / "resume_manifest.json", updated)
    return report


def _chunk_name(condition: str, index: int) -> str:
    start = index * CHUNK_SIZE
    return f"readout_{condition}_rows{start:03d}_{start + CHUNK_SIZE:03d}.pt"


def _validate_chunk(payload: Mapping[str, Any], condition: str, index: int, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    _require(payload.get("schema") == BLOCK_SCHEMA and payload.get("task_id") == TASK_ID and payload.get("condition") == condition and payload.get("chunk_index") == index, "readout block task, condition, or schema mismatch")
    block = payload["block"]
    start, stop = index * CHUNK_SIZE, (index + 1) * CHUNK_SIZE
    _require(block.get("row_start") == start and block.get("row_stop") == stop and block.get("kind") == condition, "readout block row boundaries or kind changed")
    _require(torch.equal(block["source_row_indices"], inputs["source_indices_600"][start:stop]), "readout block source row mapping changed")
    _require(torch.equal(block["labels"], inputs["labels_600"][start:stop]) and block["roles"] == inputs["roles_600"][start:stop], "readout block labels or roles changed")
    for field in ("raw_cmos", "evaluator_view", "ols_coefficients"):
        _require(bool(torch.isfinite(block[field]).all()), f"readout block {field} contains nonfinite values")
    _require(len(block.get("margins", [])) == CHUNK_SIZE and len(block.get("image_metrics", [])) == CHUNK_SIZE, "readout block lost per-row margin or image metrics")
    for evaluator in EVALUATORS:
        logits = block.get("evaluator_logits", {}).get(evaluator)
        _require(torch.is_tensor(logits) and tuple(logits.shape) == (CHUNK_SIZE, 10) and bool(torch.isfinite(logits).all()), f"readout block {evaluator} logits are malformed or nonfinite")
    return block


def _load_chunk(target: Path, condition: str, index: int, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    return _validate_chunk(torch.load(target / _chunk_name(condition, index), map_location="cpu", weights_only=True), condition, index, inputs)


def _save_chunk(target: Path, condition: str, index: int, block: Mapping[str, Any]) -> str:
    name = _chunk_name(condition, index)
    finite._write_torch_new(target / name, {"schema": BLOCK_SCHEMA, "task_id": TASK_ID, "condition": condition, "chunk_index": index, "block": dict(block)}, "immutable class-guarded readout block")
    return name


def _measure_chunks(
    target: Path,
    manifest: dict[str, Any],
    condition: str,
    inputs: Mapping[str, Any],
    consumer: Any,
    evaluators: Mapping[str, torch.nn.Module],
    *,
    planned_delta: torch.Tensor | None = None,
    actual_delta: torch.Tensor | None = None,
    baseline: bool = False,
) -> None:
    for index in range(CHUNK_COUNT):
        name = _chunk_name(condition, index)
        path = target / name
        if name in manifest["completed_chunks"]:
            _load_chunk(target, condition, index, inputs)
            continue
        if path.exists():
            _load_chunk(target, condition, index, inputs)
            manifest["completed_chunks"] = sorted(set(manifest["completed_chunks"]) | {name})
            finite._write_json_replace(target / "resume_manifest.json", manifest)
            continue
        baseline_block = None if baseline or condition == "baseline_repeat" else _load_chunk(target, "baseline", index, inputs)
        block = shared._measure_condition_chunk(
            consumer,
            evaluators,
            inputs,
            index,
            condition,
            planned_delta,
            actual_delta,
            ETA_FLOOR,
            {},
            baseline_block,
        )
        if baseline:
            block["parent_readout_max_abs_errors"] = shared._verify_baseline_block(block, inputs)
        _save_chunk(target, condition, index, block)
        manifest["completed_chunks"] = sorted(set(manifest["completed_chunks"]) | {name})
        finite._write_json_replace(target / "resume_manifest.json", manifest)
        del block, baseline_block
    _require(all(_chunk_name(condition, index) in manifest["completed_chunks"] for index in range(CHUNK_COUNT)), f"{condition} did not complete all six 100-row chunks")


def _collect_rows(target: Path, condition: str, inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(CHUNK_COUNT):
        block = _load_chunk(target, condition, index, inputs)
        for offset in range(CHUNK_SIZE):
            position = index * CHUNK_SIZE + offset
            margin = block["margins"][offset]
            image = block["image_metrics"][offset]
            row = {
                "position": position,
                "source_row_index": int(block["source_row_indices"][offset]),
                "label": int(block["labels"][offset]),
                "role": block["roles"][offset],
                "ed_margin": float(margin["margin"]),
                "ed_prediction": int(block["evaluator_logits"]["E_D"][offset].argmax()),
                "raw_mse": float(image["raw_mse"]),
                "raw_foreground_mse": float(image["raw_foreground_mse"]),
                "raw_background_mse": float(image["raw_background_mse"]),
                "view_mse": float(image["view_mse"]),
                "view_foreground_mse": float(image["view_foreground_mse"]),
                "view_background_mse": float(image["view_background_mse"]),
                "raw_fraction_above_one": float(image["raw_fraction_above_one"]),
                "raw_zero_image": bool(image["raw_zero_image"]),
                "raw_constant_image": bool(image["raw_constant_image"]),
                "condition": condition,
            }
            for evaluator in EVALUATORS:
                logits = block["evaluator_logits"][evaluator][offset]
                label = row["label"]
                row[f"{evaluator}_prediction"] = int(logits.argmax())
                row[f"{evaluator}_correct"] = int(row[f"{evaluator}_prediction"] == label)
                row[f"{evaluator}_ce"] = float(F.cross_entropy(logits.reshape(1, -1), torch.tensor([label]), reduction="mean"))
            rows.append(row)
    return rows


def _wilson(correct: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    z = 1.959963984540054
    p = correct / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _role_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in ("fit", "historic_holdout", "confirmation"):
        selected = [row for row in rows if row["role"] == role]
        by_evaluator: dict[str, Any] = {}
        for evaluator in EVALUATORS:
            per_class = {}
            for class_id in range(10):
                class_rows = [row for row in selected if row["label"] == class_id]
                correct = sum(int(row[f"{evaluator}_correct"]) for row in class_rows)
                low, high = _wilson(correct, len(class_rows))
                predictions = [sum(int(row[f"{evaluator}_prediction"]) == predicted for row in class_rows) for predicted in range(10)]
                per_class[str(class_id)] = {"rows": len(class_rows), "correct": correct, "recall": correct / len(class_rows) if class_rows else None, "wilson_95": [low, high], "prediction_counts": predictions, "mean_ce": sum(float(row[f"{evaluator}_ce"]) for row in class_rows) / len(class_rows) if class_rows else None}
            recalls = [item["recall"] for item in per_class.values() if item["recall"] is not None]
            by_evaluator[evaluator] = {"per_class": per_class, "macro_recall": sum(recalls) / len(recalls) if recalls else None, "correct": sum(int(row[f"{evaluator}_correct"]) for row in selected), "rows": len(selected)}
        result[role] = {
            "evaluators": by_evaluator,
            "raw_mse_mean": sum(float(row["raw_mse"]) for row in selected) / len(selected) if selected else None,
            "raw_fraction_above_one_mean": sum(float(row["raw_fraction_above_one"]) for row in selected) / len(selected) if selected else None,
            "zero_images": sum(bool(row["raw_zero_image"]) for row in selected),
            "constant_images": sum(bool(row["raw_constant_image"]) for row in selected),
            "by_class_image_errors": {
                str(class_id): {
                    "raw_foreground_mse_mean": sum(float(row["raw_foreground_mse"]) for row in selected if row["label"] == class_id) / sum(row["label"] == class_id for row in selected) if any(row["label"] == class_id for row in selected) else None,
                    "raw_background_mse_mean": sum(float(row["raw_background_mse"]) for row in selected if row["label"] == class_id) / sum(row["label"] == class_id for row in selected) if any(row["label"] == class_id for row in selected) else None,
                    "view_foreground_mse_mean": sum(float(row["view_foreground_mse"]) for row in selected if row["label"] == class_id) / sum(row["label"] == class_id for row in selected) if any(row["label"] == class_id for row in selected) else None,
                    "view_background_mse_mean": sum(float(row["view_background_mse"]) for row in selected if row["label"] == class_id) / sum(row["label"] == class_id for row in selected) if any(row["label"] == class_id for row in selected) else None,
                    "raw_fraction_above_one_mean": sum(float(row["raw_fraction_above_one"]) for row in selected if row["label"] == class_id) / sum(row["label"] == class_id for row in selected) if any(row["label"] == class_id for row in selected) else None,
                }
                for class_id in range(10)
            },
        }
    return result


def _class_mean(rows: Sequence[Mapping[str, Any]], key: str, class_id: int) -> float:
    values = [float(row[key]) for row in rows if int(row["label"]) == class_id]
    _require(bool(values), f"no rows for class {class_id} in {key}")
    return sum(values) / len(values)


def _decision(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    fit_ce_baseline: Mapping[str, Mapping[int, float]],
    fit_ce_candidate: Mapping[str, Mapping[int, float]],
    eta_margin: float,
    eta_ce: float,
    *,
    candidate_available: bool = True,
) -> dict[str, Any]:
    confirm_base = [row for row in baseline if row["role"] == "confirmation"]
    confirm_candidate = [row for row in candidate if row["role"] == "confirmation"]
    if not candidate_available:
        return {
            "classification": "MIXED_OR_INSUFFICIENT",
            "candidate_available": False,
            "confirmation_focus_response": {},
            "confirmation_other_class_margin_changes": {},
            "stable_harmful_other_classes": [],
            "other_class_mean_margin_failures": [],
            "fit_ce_guard_deltas": {},
            "fit_ce_guard_failures": {},
            "accuracy_correct_declines_by_class": {},
            "accuracy_guard_passed": None,
            "raw_mse_guard_passed": None,
            "clipping_guard_passed": None,
            "no_zero_or_constant_confirmation_images": None,
            "ce_tolerance": eta_ce,
            "margin_tolerance": eta_margin,
            "candidate_unavailable_reason": "the guarded projection is within the declared numerical zero range",
            "training_recommendation": "NONE",
            "next_action": "NONE",
            "claim_boundary": "a zero projection is a valid insufficient-evidence exit; it does not establish absence of other feasible directions or architecture capacity",
        }
    _require(len(confirm_base) == len(confirm_candidate) == CONFIRMATION_ROWS, "final class-guard decision requires the complete 300-row confirmation population")
    by_source = {int(row["source_row_index"]): row for row in confirm_base}
    deltas = {int(row["source_row_index"]): float(row["ed_margin"]) - float(by_source[int(row["source_row_index"])]["ed_margin"]) for row in confirm_candidate}
    focus = {
        str(class_id): {
            "rows": sum(row["label"] == class_id for row in confirm_candidate),
            "beneficial_rows": sum(row["label"] == class_id and deltas[int(row["source_row_index"])] > eta_margin for row in confirm_candidate),
            "harmful_rows": sum(row["label"] == class_id and deltas[int(row["source_row_index"])] < -eta_margin for row in confirm_candidate),
            "mean_margin_delta": sum(deltas[int(row["source_row_index"])] for row in confirm_candidate if row["label"] == class_id) / 30.0,
        }
        for class_id in FOCUS_CLASSES
    }
    other_classes: dict[str, Any] = {}
    stable_harm: list[int] = []
    mean_margin_failures: list[int] = []
    for class_id in range(10):
        class_rows = [row for row in confirm_candidate if row["label"] == class_id]
        values = [deltas[int(row["source_row_index"])] for row in class_rows]
        harmful = sum(value < -eta_margin for value in values)
        mean_delta = sum(values) / len(values)
        if class_id not in FOCUS_CLASSES:
            if harmful >= STABLE_ROWS:
                stable_harm.append(class_id)
            if mean_delta < -eta_margin:
                mean_margin_failures.append(class_id)
        other_classes[str(class_id)] = {"rows": len(class_rows), "beneficial_rows": sum(value > eta_margin for value in values), "harmful_rows": harmful, "boundary_rows": sum(abs(value) <= eta_margin for value in values), "mean_margin_delta": mean_delta}

    base_metrics = shared._class_metrics(confirm_base)
    candidate_metrics = shared._class_metrics(confirm_candidate)
    accuracy_declines = {str(class_id): base_metrics["correct"][class_id] - candidate_metrics["correct"][class_id] for class_id in range(10)}
    accuracy_passed = all(value <= 1 for value in accuracy_declines.values()) and candidate_metrics["macro_recall"] >= base_metrics["macro_recall"]
    base_raw_mse = sum(float(row["raw_mse"]) for row in confirm_base) / len(confirm_base)
    candidate_raw_mse = sum(float(row["raw_mse"]) for row in confirm_candidate) / len(confirm_candidate)
    raw_clip = sum(float(row["raw_fraction_above_one"]) for row in confirm_candidate) / len(confirm_candidate)
    class_clip = {str(class_id): sum(float(row["raw_fraction_above_one"]) for row in confirm_candidate if row["label"] == class_id) / 30.0 for class_id in range(10)}
    image_valid = all(math.isfinite(float(row["raw_mse"])) and not row["raw_zero_image"] and not row["raw_constant_image"] for row in confirm_candidate)
    ce_deltas = {
        f"{name}:class{class_id}": float(fit_ce_candidate[name][class_id]) - float(fit_ce_baseline[name][class_id])
        for name in GUARD_EVALUATORS
        for class_id in range(10)
    }
    ce_failures = {key: value for key, value in ce_deltas.items() if value > eta_ce}
    focus_response = all(focus[str(class_id)]["beneficial_rows"] >= STABLE_ROWS for class_id in FOCUS_CLASSES)
    no_class_harm = not stable_harm and not mean_margin_failures
    mse_passed = candidate_raw_mse <= base_raw_mse * 1.05
    clipping_passed = raw_clip <= 0.01 and all(class_clip[str(class_id)] <= 0.01 for class_id in FOCUS_CLASSES)
    all_guards = candidate_available and not ce_failures and focus_response and no_class_harm and accuracy_passed and mse_passed and clipping_passed and image_valid
    if all_guards:
        classification = "CLASS_GUARDED_SHARED_LOCAL_RESPONSE_SUPPORTED"
    elif focus_response:
        classification = "CLASS_GUARDED_RESPONSE_WITH_TRADEOFF"
    else:
        classification = "MIXED_OR_INSUFFICIENT"
    return {
        "classification": classification,
        "candidate_available": candidate_available,
        "confirmation_focus_response": focus,
        "confirmation_other_class_margin_changes": other_classes,
        "stable_harmful_other_classes": stable_harm,
        "other_class_mean_margin_failures": mean_margin_failures,
        "fit_ce_guard_deltas": ce_deltas,
        "fit_ce_guard_failures": ce_failures,
        "accuracy_correct_declines_by_class": accuracy_declines,
        "accuracy_guard_passed": accuracy_passed,
        "baseline_macro_recall": base_metrics["macro_recall"],
        "candidate_macro_recall": candidate_metrics["macro_recall"],
        "baseline_confirmation_raw_mse_mean": base_raw_mse,
        "candidate_confirmation_raw_mse_mean": candidate_raw_mse,
        "raw_mse_guard_passed": mse_passed,
        "candidate_confirmation_raw_fraction_above_one_mean": raw_clip,
        "candidate_confirmation_raw_fraction_above_one_by_class": class_clip,
        "clipping_guard_passed": clipping_passed,
        "no_zero_or_constant_confirmation_images": image_valid,
        "ce_tolerance": eta_ce,
        "margin_tolerance": eta_margin,
        "minimum_focus_beneficial_rows": STABLE_ROWS,
        "training_recommendation": "NONE",
        "next_action": "NONE",
        "claim_boundary": "one equal-RMS frozen common phase displacement; local response only, not accumulated training, human readability, generalization, or architecture-capacity evidence",
    }


def _write_csv_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), "cannot write an empty row-metrics table")
    buffer = io.StringIO(newline="")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue().encode("utf-8")
    if path.exists():
        _require(path.read_bytes() == content, f"existing {path.name} differs from immutable measured rows")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.pending")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
        os.rename(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_review_pages(target: Path, inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    panels = ("target", "baseline", "pure_ce_plus", "guarded_plus")
    panel_names = {"target": "target TRAIN", "baseline": "B endpoint", "pure_ce_plus": "pure CE +1", "guarded_plus": "guarded +1"}
    panel_size, row_height, gap = 84, 104, 10
    for class_id in DISPLAY_CLASSES:
        positions = [position for position, label in enumerate(inputs["labels_600"].tolist()) if int(label) == class_id]
        _require(len(positions) == 60, f"review page class {class_id} must display all sixty rows")
        canvas = Image.new("RGB", (310 + len(panels) * panel_size + (len(panels) - 1) * gap, 44 + 60 * row_height), "white")
        draw = ImageDraw.Draw(canvas)
        for index, panel in enumerate(panels):
            draw.text((220 + index * (panel_size + gap), 12), panel_names[panel], fill=(20, 20, 20))
        row_number = 0
        for chunk_index in range(CHUNK_COUNT):
            selected = [position for position in positions if position // CHUNK_SIZE == chunk_index]
            if not selected:
                continue
            baseline = _load_chunk(target, "baseline", chunk_index, inputs)
            guarded = _load_chunk(target, "guarded_plus1", chunk_index, inputs) if (target / _chunk_name("guarded_plus1", chunk_index)).exists() else None
            previous = shared._load_readout_chunk(inputs["shared_dir"], "S_GD_equal_rms_plus1", chunk_index, inputs)
            for position in selected:
                local = position % CHUNK_SIZE
                y = 44 + row_number * row_height
                role = inputs["roles_600"][position]
                prediction = int(baseline["evaluator_logits"]["E_D"][local].argmax())
                draw.text((8, y + 8), f"row {position} / source {int(inputs['source_indices_600'][position])}", fill=(20, 20, 20))
                draw.text((8, y + 30), f"class {class_id} / {role} / E_D {prediction}", fill=(20, 20, 20))
                arrays: dict[str, torch.Tensor] = {
                    "target": inputs["images_600"][position],
                    "baseline": baseline["evaluator_view"][local],
                    "pure_ce_plus": previous["evaluator_view"][local],
                }
                if guarded is not None:
                    arrays["guarded_plus"] = guarded["evaluator_view"][local]
                for panel_index, panel in enumerate(panels):
                    value = arrays.get(panel)
                    if value is None:
                        continue
                    gray = (value.squeeze().detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).round().astype("uint8")
                    image = Image.fromarray(gray, mode="L").resize((panel_size, panel_size), Image.Resampling.NEAREST).convert("RGB")
                    canvas.paste(image, (210 + panel_index * (panel_size + gap), y + 8))
                row_number += 1
        path = target / f"review_class_{class_id}.png"
        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        content = buffer.getvalue()
        if path.exists():
            _require(path.read_bytes() == content, f"existing {path.name} differs from immutable image review")
        else:
            with path.open("xb") as handle:
                handle.write(content)
        pages.append({"class_id": class_id, "rows": 60, "path": path.name, "panels": list(panels), "fixed_view": "28x28, clamped [0,1]"})
    return pages


def _finalize(target: Path, inputs: Mapping[str, Any], identity: ResearchRunIdentity, preflight: Mapping[str, Any], projection: Mapping[str, Any], started: float) -> dict[str, Any]:
    baseline = _collect_rows(target, "baseline", inputs)
    repeated = _collect_rows(target, "baseline_repeat", inputs)
    repeat_by_source = {int(row["source_row_index"]): row for row in repeated}
    max_margin_repeat = max(abs(float(row["ed_margin"]) - float(repeat_by_source[int(row["source_row_index"])]["ed_margin"])) for row in baseline)
    eta_margin = max(ETA_FLOOR, 5.0 * max_margin_repeat)
    fit_ce_baseline: dict[str, dict[int, float]] = {name: {} for name in GUARD_EVALUATORS}
    fit_ce_repeat: dict[str, dict[int, float]] = {name: {} for name in GUARD_EVALUATORS}
    for name in GUARD_EVALUATORS:
        for class_id in range(10):
            selected = [row for row in baseline if row["role"] == "fit" and row["label"] == class_id]
            selected_repeat = [row for row in repeated if row["role"] == "fit" and row["label"] == class_id]
            fit_ce_baseline[name][class_id] = sum(float(row[f"{name}_ce"]) for row in selected) / len(selected)
            fit_ce_repeat[name][class_id] = sum(float(row[f"{name}_ce"]) for row in selected_repeat) / len(selected_repeat)
    max_ce_repeat = max(abs(fit_ce_baseline[name][class_id] - fit_ce_repeat[name][class_id]) for name, class_id in GUARD_NAMES)
    eta_ce = max(ETA_FLOOR, 5.0 * max_ce_repeat)
    rows_by_state = {"baseline": baseline, "baseline_repeat": repeated}
    candidate_rows: list[dict[str, Any]] = []
    if projection["candidate_available"]:
        candidate_rows = _collect_rows(target, "guarded_plus1", inputs)
        rows_by_state["guarded_plus1"] = candidate_rows
        rows_by_state["guarded_minus1"] = _collect_rows(target, "guarded_minus1", inputs)
    fit_ce_candidate = {name: {} for name in GUARD_EVALUATORS}
    for name in GUARD_EVALUATORS:
        for class_id in range(10):
            selected = [row for row in candidate_rows if row["role"] == "fit" and row["label"] == class_id]
            fit_ce_candidate[name][class_id] = sum(float(row[f"{name}_ce"]) for row in selected) / len(selected) if selected else fit_ce_baseline[name][class_id]
    decision = _decision(baseline, candidate_rows, fit_ce_baseline, fit_ce_candidate, eta_margin, eta_ce, candidate_available=bool(projection["candidate_available"]))
    role_summaries = {state: _role_summary(rows) for state, rows in rows_by_state.items()}
    prior_summary = inputs["shared_report"]["state_summaries_by_role"]["S_GD_equal_rms_plus1"]
    table_rows = [{"state": state, **row} for state, rows in rows_by_state.items() for row in rows]
    _write_csv_once(target / "row_metrics.csv", table_rows)
    review = _write_review_pages(target, inputs)
    return {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "status": "VALID_COMPLETED",
        "classification": decision["classification"],
        "research_run_identity": identity.to_dict(),
        "scientific_updates": 0,
        "optimizer_updates": 0,
        "phase_interventions": 2 if projection["candidate_available"] else 0,
        "formal_measurement_rows": TOTAL_ROWS * (2 + (2 if projection["candidate_available"] else 0)),
        "reference_readout_rows_not_remeasured": TOTAL_ROWS,
        "population_roles": {"fit": FIT_ROWS, "historic_holdout": HISTORIC_ROWS, "confirmation": CONFIRMATION_ROWS},
        "data_scope": "all six hundred fixed rows are canonical TRAIN; confirmation rows were not used in gradient or projection",
        "parent_result_preserved": {
            "shared_direction_status": inputs["shared_report"]["status"],
            "shared_direction_classification": inputs["shared_report"]["classification"],
            "parent_ce_status": inputs["parent_report"]["status"],
            "parent_ce_classification": inputs["parent_report"]["classification"],
        },
        "guard_gradient_construction": {
            "definition": "q[e,c] = mean over twenty fit rows of CE(E_e(view(raw_cmos)), c), measured through full image-to-OLS-to-Generator path",
            "evaluators": list(GUARD_EVALUATORS),
            "constraints": [f"{name}:class{class_id}" for name, class_id in GUARD_NAMES],
            "class_mean_raw_ce_matches_saved_mixed_batch_gradient": preflight["class_gradient_production"]["mean_by_class_matches_saved_mixed_batch_mean"],
            "e_d_used_for_direction_construction": False,
        },
        "projection": {
            key: value
            for key, value in projection.items()
            if key not in {"source_direction", "projected_minimum_distance_vector", "candidate_equal_rms_direction", "actual_float32_displacements"}
        },
        "preflight": dict(preflight),
        "baseline_repeat": {
            "max_abs_e_d_margin_difference": max_margin_repeat,
            "eta_margin": eta_margin,
            "max_abs_fit_class_ce_difference": max_ce_repeat,
            "eta_ce": eta_ce,
        },
        "state_summaries_by_role": role_summaries,
        "previous_pure_ce_positive_reference_by_role": prior_summary,
        "fit_mean_ce_by_evaluator_class": {
            name: {str(class_id): {"baseline": fit_ce_baseline[name][class_id], "guarded_plus": fit_ce_candidate[name][class_id], "delta": fit_ce_candidate[name][class_id] - fit_ce_baseline[name][class_id]} for class_id in range(10)}
            for name in GUARD_EVALUATORS
        },
        "confirmation_decision": decision,
        "image_review": review,
        "evidence_artifacts": {
            "population": "input_population.json",
            "class_gradients": "class_guard_gradients.pt",
            "projection": "class_guard_projection.pt",
            "actual_displacements": "actual_displacements.pt",
            "preflight": "preflight.json",
            "manifest": "resume_manifest.json",
            "readout_chunks": [_chunk_name(condition, index) for condition in (["baseline", "baseline_repeat", "guarded_plus1", "guarded_minus1"] if projection["candidate_available"] else ["baseline", "baseline_repeat"]) for index in range(CHUNK_COUNT)],
            "row_metrics": "row_metrics.csv",
            "image_review": [item["path"] for item in review],
        },
        "training_recommendation": "NONE",
        "next_action": "NONE",
        "elapsed_seconds_observation_only": time.perf_counter() - started,
    }


def _record_snapshot_checks(consumer: Any, evaluators: Mapping[str, torch.nn.Module], parameters: Mapping[str, torch.nn.Parameter], runtime_before: Mapping[str, Any], evaluator_before: Mapping[str, Mapping[str, Any]], frozen_inputs_before: Mapping[str, Any], optimizer_before: Mapping[str, Any], inputs: Mapping[str, Any]) -> None:
    _require(finite._tree_equal(runtime_before, finite._snapshot(consumer.runtime)), "measurement changed frozen model parameters or scientific buffers")
    _require(finite._tree_equal(frozen_inputs_before, direction._consumer_frozen_inputs_snapshot(consumer)), "measurement changed frozen optical references or BOGT weights")
    _require(finite._tree_equal(optimizer_before, direction._cpu_tree(inputs["optimizers"]["B_class_ce"])), "measurement changed the saved source Adam state")
    for name, model in evaluators.items():
        _require(finite._tree_equal(evaluator_before[name], finite._snapshot(model)), f"measurement changed evaluator {name}")
        _require(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()), f"evaluator {name} is not frozen and grad-free")
    _require(all(parameter.grad is None and not parameter.requires_grad for parameter in parameters.values()), "optical phase parameters are not frozen and grad-free after measurement")


def _save_preflight(target: Path, population: Mapping[str, Any], identity: ResearchRunIdentity, preflight: Mapping[str, Any], receipt: PreflightReceipt, gradients: Mapping[str, Any], projection: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    finite._write_json_new(target / "input_population.json", population, "fixed class-guarded population")
    finite._write_torch_new(target / "class_guard_gradients.pt", gradients, "thirty fixed class CE gradient constraints")
    finite._write_torch_new(target / "class_guard_projection.pt", projection, "class-guard nearest-point projection")
    finite._write_torch_new(
        target / "actual_displacements.pt",
        {
            "planned": {
                "guarded_plus1": projection["candidate_equal_rms_direction"],
                "guarded_minus1": None if projection["candidate_equal_rms_direction"] is None else -projection["candidate_equal_rms_direction"],
            },
            "actual_float32": projection.get("actual_float32_displacements", {}),
        },
        "planned and actual float32 class-guard phase displacements",
    )
    finite._write_json_new(target / "preflight.json", {"task_id": TASK_ID, "status": "READY", "receipt": receipt.to_dict(), "preflight": dict(preflight)}, "executed class-guard preflight")
    finite._write_json_new(target / "resume_manifest.json", dict(manifest), "class-guard resume manifest")


def _load_saved_preflight(target: Path, inputs: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], PreflightReceipt]:
    payload = json.loads((target / "preflight.json").read_text(encoding="utf-8"))
    _require(payload.get("task_id") == TASK_ID and payload.get("status") == "READY", "resume lacks this task's executed READY preflight")
    preflight = payload["preflight"]
    gradients = torch.load(target / "class_guard_gradients.pt", map_location="cpu", weights_only=True)
    projection = torch.load(target / "class_guard_projection.pt", map_location="cpu", weights_only=True)
    _require(gradients.get("schema") == GRADIENT_SCHEMA and set(gradients.get("gradients", {})) == set(GUARD_EVALUATORS), "saved guard gradients are missing or use different semantics")
    _validate_gradient_population(gradients, inputs["labels_600"][:FIT_ROWS], inputs["source_indices_600"][:FIT_ROWS])
    _require(projection.get("schema") == PROJECTION_SCHEMA and projection.get("candidate_available") is (projection.get("candidate_equal_rms_direction") is not None), "saved projection semantics are malformed")
    _require(_validate_class_gradient_means(gradients, inputs["direction_blocks"]) == preflight["class_gradient_production"]["mean_by_class_matches_saved_mixed_batch_mean"], "saved guard gradients no longer match their named mixed-batch references")
    recomputed = _project_guarded_direction(inputs["directions"]["equal_rms_directions"]["S_GD"], gradients["gradients"])
    _require(torch.allclose(recomputed["projected_minimum_distance_vector"], projection["projected_minimum_distance_vector"], rtol=1e-10, atol=1e-12), "saved minimum-distance projection differs from its guard gradients")
    recomputed_candidate = recomputed["candidate_equal_rms_direction"]
    saved_candidate = projection["candidate_equal_rms_direction"]
    _require(
        (recomputed_candidate is None and saved_candidate is None)
        or (torch.is_tensor(recomputed_candidate) and torch.is_tensor(saved_candidate) and torch.allclose(recomputed_candidate, saved_candidate, rtol=1e-7, atol=1e-12)),
        "saved equal-RMS candidate differs from its minimum-distance projection",
    )
    receipt = PreflightReceipt.from_mapping(payload["receipt"])
    return preflight, gradients, projection, receipt


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    root = _root()
    shared_dir = _resolve(root, arguments.shared_direction_dir)
    target = _resolve(root, arguments.output_dir)
    outputs = (root / "outputs").resolve()
    _require(target != outputs and outputs in target.parents, "output must be a named child of outputs/")
    _require(not (arguments.preflight_only and arguments.resume_manifest), "preflight-only cannot resume")
    _require(arguments.resume_manifest is not None or not target.exists(), "output already exists; pass its explicit resume manifest or use a new attempt directory")
    inputs = _load_inputs(root, shared_dir)
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
    if existing is None:
        preflight, gradients, projection = _preflight(inputs, target, identity, population, started)
        run_control = ResearchRunControl(identity=identity, authorization_id=f"{TASK_ID}:authorization:01")
        lifecycle = _preflight_artifact_lifecycle(identity, population, preflight, target.parent)
        preflight = {**preflight, "storage_lifecycle": lifecycle}
        receipt = _ready_receipt(run_control, preflight)
        if receipt.status != "READY":
            raise ClassGuardedDirectionError("executed class-guard preflight did not produce a READY receipt")
        run_control.mark_preflight_ready(receipt)
        manifest = _manifest_seed(identity, population, receipt, preflight, projection)
        if arguments.preflight_only:
            target.mkdir(parents=True, exist_ok=False)
            _save_preflight(target, population, identity, preflight, receipt, gradients, projection, manifest)
            return {"task_id": TASK_ID, "status": "READY", "output_dir": target.as_posix(), "optimizer_updates": 0, "next_action": "NONE"}
        run_control.start_execution()
        run_control.open_scientific_attempt(target, identity.scientific_attempt_id, receipt)
        _save_preflight(target, population, identity, preflight, receipt, gradients, projection, manifest)
    else:
        preflight, gradients, projection, receipt = _load_saved_preflight(target, inputs)
        _require(existing.get("preflight") == preflight and existing.get("receipt") == receipt.to_dict(), "resume manifest differs from the saved READY preflight and receipt")
        run_control = ResearchRunControl(identity=identity, authorization_id=receipt.authorization_id)
        run_control.mark_preflight_ready(receipt)
        run_control.start_execution()
        run_control.open_scientific_attempt(target, identity.scientific_attempt_id, receipt, resume=True)
        manifest = dict(existing)
        manifest["status"] = "RUNNING"
        finite._write_json_replace(target / "resume_manifest.json", manifest)

    device = torch.device("cuda:0")
    _require(torch.cuda.is_available(), "CUDA became unavailable before frozen readout")
    runtime_probe = direction._runtime_dll_probe()
    _require(runtime_probe.get("status") in {"already_present", "prepended", "not_applicable"}, "CUDA runtime DLL readiness changed before frozen readout")
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
    candidate = projection["candidate_equal_rms_direction"]
    actual_deltas: dict[str, torch.Tensor] = {}
    if candidate is not None:
        for name, planned in (("guarded_plus1", candidate), ("guarded_minus1", -candidate)):
            actual_deltas[name] = shared._phase_set(parameters["generator.phase"], baseline_phase, planned)
            finite._restore_phase(parameters["generator.phase"], baseline_phase)
    displacement_path = target / "actual_displacements.pt"
    saved = torch.load(displacement_path, map_location="cpu", weights_only=True)
    expected_planned = {"guarded_plus1": candidate, "guarded_minus1": None if candidate is None else -candidate}
    _require(
        set(saved["planned"]) == set(expected_planned)
        and all(
            (saved["planned"][name] is None and expected_planned[name] is None)
            or (torch.is_tensor(saved["planned"][name]) and torch.equal(saved["planned"][name], expected_planned[name]))
            for name in expected_planned
        ),
        "saved planned displacements differ from the READY projection",
    )
    _require(
        set(saved["actual_float32"]) == set(actual_deltas)
        and all(torch.equal(saved["actual_float32"][name], value) for name, value in actual_deltas.items()),
        "measured or resumed float32 phase displacements differ from the READY preflight",
    )

    try:
        _require(torch.equal(parameters["generator.phase"].detach().cpu(), baseline_phase), "baseline Generator phase differs from the frozen B endpoint")
        _measure_chunks(target, manifest, "baseline", inputs, consumer, evaluators, baseline=True)
        finite._restore_phase(parameters["generator.phase"], baseline_phase)
        _measure_chunks(target, manifest, "baseline_repeat", inputs, consumer, evaluators)
        baseline = _collect_rows(target, "baseline", inputs)
        repeated = _collect_rows(target, "baseline_repeat", inputs)
        repeat_by_source = {int(row["source_row_index"]): row for row in repeated}
        eta_margin = max(ETA_FLOOR, 5.0 * max(abs(float(row["ed_margin"]) - float(repeat_by_source[int(row["source_row_index"])]["ed_margin"])) for row in baseline))
        fit_base = _role_summary(baseline)["fit"]
        fit_repeat = _role_summary(repeated)["fit"]
        ce_repeat_error = max(
            abs(float(fit_base["evaluators"][name]["per_class"][str(class_id)]["mean_ce"]) - float(fit_repeat["evaluators"][name]["per_class"][str(class_id)]["mean_ce"]))
            for name, class_id in GUARD_NAMES
        )
        preflight = {**preflight, "baseline_repeat": {"eta_margin": eta_margin, "max_fit_class_ce_repeat_error": ce_repeat_error, "eta_ce": max(ETA_FLOOR, 5.0 * ce_repeat_error)}}
        manifest["preflight"] = preflight
        finite._write_json_replace(target / "resume_manifest.json", manifest)
        finite._write_json_replace(target / "preflight.json", {"task_id": TASK_ID, "status": "READY", "receipt": receipt.to_dict(), "preflight": preflight})
        for name, planned in (("guarded_plus1", candidate), ("guarded_minus1", None if candidate is None else -candidate)):
            if planned is None:
                continue
            finite._restore_phase(parameters["generator.phase"], baseline_phase)
            applied = shared._phase_set(parameters["generator.phase"], baseline_phase, planned)
            _require(torch.equal(applied, actual_deltas[name]), f"measured float32 phase change differs from saved {name} replay")
            _measure_chunks(target, manifest, name, inputs, consumer, evaluators, planned_delta=planned, actual_delta=applied)
            finite._restore_phase(parameters["generator.phase"], baseline_phase)
            _record_snapshot_checks(consumer, evaluators, parameters, runtime_before, evaluator_before, frozen_inputs_before, optimizer_before, inputs)
        finite._restore_phase(parameters["generator.phase"], baseline_phase)
        _record_snapshot_checks(consumer, evaluators, parameters, runtime_before, evaluator_before, frozen_inputs_before, optimizer_before, inputs)
        expected = {_chunk_name(name, index) for name in manifest["measured_conditions"] for index in range(CHUNK_COUNT)}
        _require(expected.issubset(set(manifest["completed_chunks"])), "one or more measured readout chunks are missing")
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
        report = _finalize(target, inputs, identity, preflight, projection, started)
    except Exception:
        manifest["status"] = "INVALID_HARNESS"
        finite._write_json_replace(target / "resume_manifest.json", manifest)
        run_control.invalid_harness()
        raise
    finite._write_json_new(target / "report.json", report, "class-guarded common displacement decision")
    manifest["status"] = "COMPLETED"
    finite._write_json_replace(target / "resume_manifest.json", manifest)
    run_control.complete()
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-direction-dir", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-manifest", type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    try:
        result = _run(arguments)
    except (ClassGuardedDirectionError, shared.SharedDirectionError, finite.FiniteResponseError, direction.DirectionDiagnosticError, base.DiagnosticError, OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        target = _resolve(_root(), arguments.output_dir)
        failure = {"task_id": TASK_ID, "status": "INVALID_HARNESS", "classification": None, "reason": f"{type(error).__name__}: {error}", "optimizer_updates": 0, "next_action": "NONE"}
        try:
            output_root = (_root() / "outputs").resolve()
            if target != output_root and output_root in target.parents:
                target.mkdir(parents=True, exist_ok=True)
                if not (target / "report.json").exists() and not (target / "failure.json").exists():
                    finite._write_json_new(target / "failure.json", failure, "class-guard invalid harness")
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
