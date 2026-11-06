from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
    RUN_COMPLETED,
    RUN_SCIENTIFIC_ACTIVE,
    RUN_VALID_FAILED,
)
from scripts.research.lightgen import diagnose_lightgen_frozen_slm_ols_reachability_v1 as reach
from scripts.research.lightgen import execute_lightgen_g1_fullpop_feasibility_diagnosis_v1 as feasibility
from scripts.research.lightgen import execute_lightgen_g1_projected_fit20_box_ab_v1 as fit20
from scripts.research.lightgen import execute_lightgen_g1_slm_input_mapping_pilot_v1 as pilot
from scripts.research.lightgen import execute_lightgen_g1_slm_projected_mapping_v2 as projected


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_G1_SEMANTIC_EXPOSURE200_V1"
SCHEMA = "lightgen_g1_semantic_exposure200.v1"
AUTHORITY = "USER_AUTHORIZED_LIGHTGEN_G1_SEMANTIC_EXPOSURE200_20261029"
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/g1_semantic_exposure200_v1.yaml"
RUNS_DIR = ROOT / "outputs/20261029_lightgen_g1_semantic_exposure200_v1/runs"
PREFLIGHT_DIR = RUNS_DIR / "preflight_01"
ATTEMPT_DIR = RUNS_DIR / "attempt_01"
REFERENCE_DIR = RUNS_DIR / "reference_qualification_01"
ENGINE_DIR = RUNS_DIR / "engine_qualification_01"
CLASS_COUNT = 10
PORT_COUNT = 100
PIXEL_COUNT = 40_000
TRAIN_PER_CLASS = 20
DEV_PER_CLASS = 20
TRAIN_ROWS = CLASS_COUNT * TRAIN_PER_CLASS
DEV_ROWS = CLASS_COUNT * DEV_PER_CLASS
UPDATE_BUDGET = 15_000
EXPOSURES_PER_TARGET = 750
BOX_LAMBDA = 5.0
GRIDS = (2048, 4096)
METRIC_KEYS = fit20.METRIC_KEYS
GATE_LIMIT = fit20.GATE_LIMIT
RECOVERY_TOLERANCE = 5.0e-8
CONTINUOUS_PHASE_SOLVE_LIMIT = 64
CONTINUOUS_PHASE_INTERVAL_MIN_WIDTH = 1.0e-10
CONTINUOUS_PHASE_NUMERIC_GUARD = 1.0e-8
DUAL_BOUND_NUMERIC_TOLERANCE = 1.0e-10


class TrainingWitnessQualificationError(RuntimeError):
    """A fixed target lacks the registered pretraining reachability witness."""


def _require(condition: bool, message: str, error: type[Exception] = RuntimeError) -> None:
    if not condition:
        raise error(message)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                     default=str), encoding="utf-8")
    os.replace(temporary, path)


def _save_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(config, dict) and config.get("schema") == SCHEMA
             and config.get("task_id") == TASK_ID
             and config.get("semantic_authority_id") == AUTHORITY
             and config.get("evidence_tier") == "diagnostic_only",
             "configuration is not the fixed diagnostic-only G1 exposure task", ValueError)
    training = config["training"]
    _require(training == {
        "new_updates": UPDATE_BUDGET, "logical_batch_size": 10,
        "samples_per_class_per_batch": 1, "epochs": EXPOSURES_PER_TARGET,
        "optimizer": "Adam", "learning_rate": 1.0e-4,
        "betas": [0.9, 0.999], "epsilon": 1.0e-8, "weight_decay": 0.0,
        "box_lambda": BOX_LAMBDA, "rng_seed": 2026102970,
        "schedule_seed": 2026102971, "checkpoint_interval_steps": 500,
        "progress_interval_steps": 100, "observation_steps": [0, 5000, 10000, 15000],
        "select_best_checkpoint": False, "amp": False, "tf32": False,
    }, "fixed population, optimizer, schedule, or update budget changed", ValueError)
    _require(config["population"] == {
        "train_role": "fixed_train200", "train_per_class": TRAIN_PER_CLASS,
        "train_source_offset_per_class": 0, "dev_role": "historical_development200",
        "dev_per_class": DEV_PER_CLASS, "dev_source_offset_per_class": 4,
        "classes": CLASS_COUNT, "exposures_per_target": EXPOSURES_PER_TARGET,
    }, "fixed training/development population definition changed", ValueError)
    _require(config["measurement"]["relative_error_max"] == GATE_LIMIT
             and config["measurement"]["grids"] == [2048, 4096]
             and config["measurement"]["coefficient_alignment"] == "one_global_phase_per_sample"
             and config["measurement"]["native_cmos_unit"] == "raw_w_per_m2"
             and config["measurement"]["cmos_normalization"] is False
             and config["measurement"]["cmos_power_matching"] is False
             and config["measurement"]["cmos_gain_adjustment"] is False
             and config["measurement"]["semantic_metrics"] == "frozen_ED_EC_auxiliary_only"
             and config["output"]["experiment_root"] == "outputs/20261029_lightgen_g1_semantic_exposure200_v1/runs"
             and config["output"]["attempt_root"].endswith("/attempt_01")
             and config["output"]["audit_dir"].endswith("/audit_01")
             and config["output"]["preflight_root"].endswith("/preflight_repair_03")
             and config["output"]["windows_backup_root"].endswith("/autodl_backup_attempt_01")
             and config["output"]["reference_qualification_root"].endswith("/reference_qualification_01")
             and config["output"]["reference_continuous_phase_root"].endswith(
                 "/reference_continuous_phase_01")
             and config["output"]["preflight_repair_root"].endswith("/preflight_repair_03")
             and config["output"]["qualification_verification_root"].endswith(
                 "/qualification_verification_01")
             and config["output"]["engine_qualification_root"].endswith("/engine_qualification_01")
             and config["output"]["audit_extension_root"].endswith("/audit_extension_01")
             and config["source"]["prior_training_witnesses"].endswith(
                 "/autodl_backup_preflight_repair_02/training_witnesses.pt"),
             "the fixed six-metric measurement contract changed", ValueError)
    return config


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["source"]
    mapping_config = projected._load_config(_resolve(source["projected_mapping_config"]))
    inputs = projected._load_inputs(mapping_config)
    checkpoint_path = _resolve(source["v2_terminal_checkpoint"])
    _require(checkpoint_path.is_file(), f"explicit V2 terminal checkpoint is missing: {checkpoint_path}",
             FileNotFoundError)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    steps = [int(torch.as_tensor(value["step"]).item())
             for value in checkpoint.get("optimizer_state", {}).get("state", {}).values()
             if isinstance(value, Mapping) and "step" in value]
    _require(checkpoint.get("task_id") == projected.TASK_ID
             and checkpoint.get("completed_updates") == 5000
             and checkpoint.get("checkpoint_step") == 5000
             and checkpoint.get("source_mapper_updates") == 5000
             and checkpoint.get("history_count") == 5000
             and len(steps) == 4 and steps == [5000] * 4
             and checkpoint.get("optimizer_deepcopied") is True
             and set(checkpoint.get("mapper_state", {})) == {
                 "network.0.weight", "network.0.bias", "network.2.weight", "network.2.bias"},
             "explicit starting mapper is not the closed V2 terminal checkpoint")
    projector_path = _resolve(source["v2_fixed_response_projector_evidence"])
    projector = torch.load(projector_path, map_location="cpu", weights_only=False)
    response = inputs["response_matrix"].detach().cpu().to(torch.complex128)
    scale = inputs["scale"].detach().cpu().double()
    matrix = projected._normalized_response(response, scale)
    _require(torch.equal(projector.get("response_matrix"), response)
             and torch.equal(projector.get("scale"), scale)
             and projector.get("real_imag_order") == "per_port_interleaved_real_imag",
             "fixed projector evidence does not match the loaded C45k response and scale")
    saved_matrix = projector.get("normalized_real_matrix")
    inverse = projector.get("full_right_inverse")
    _require(torch.is_tensor(saved_matrix) and torch.is_tensor(inverse)
             and tuple(saved_matrix.shape) == (200, PIXEL_COUNT)
             and tuple(inverse.shape) == (PIXEL_COUNT, 200)
             and torch.max(torch.abs(saved_matrix - matrix)) <= 1.0e-14,
             "saved fixed matrix differs from independently normalized response")
    identity = torch.eye(200, dtype=torch.float64)
    gram = saved_matrix.double() @ saved_matrix.double().T
    mp_inverse = saved_matrix.double().T @ torch.linalg.solve(gram, identity)
    matrix_check = {
        "max_abs_AP_minus_I": float((saved_matrix.double() @ inverse.double() - identity).abs().max()),
        "max_abs_inverse_minus_moore_penrose": float((inverse.double() - mp_inverse).abs().max()),
    }
    _require(matrix_check["max_abs_AP_minus_I"] <= 1.0e-10
             and matrix_check["max_abs_inverse_minus_moore_penrose"] <= 1.0e-10,
             "fixed right inverse fails independent Moore-Penrose verification")
    witness_path = _resolve(source["positive_control_witnesses"])
    _require(witness_path.is_file(), f"fixed positive-control witnesses are missing: {witness_path}",
             FileNotFoundError)
    positive_control = torch.load(witness_path, map_location="cpu", weights_only=False)
    _require(positive_control.get("task_id") == "LIGHTGEN_G1_PROJECTED_BOX_FULLPOP_V1"
             and positive_control.get("population_role") == "positive_control_only"
             and tuple(positive_control.get("amplitudes", torch.empty(0)).shape) == (20, 1, 200, 200)
             and len(positive_control.get("rows", [])) == 20
             and all(row.get("row_passed") for row in positive_control["rows"]),
             "source positive-control witness is incomplete or has a different role")
    prior_witness_path = _resolve(source["prior_training_witnesses"])
    _require(prior_witness_path.is_file(),
             f"explicit predecessor training-witness evidence is missing: {prior_witness_path}",
             FileNotFoundError)
    prior_witnesses = torch.load(prior_witness_path, map_location="cpu", weights_only=False)
    _require(prior_witnesses.get("schema") == f"{TASK_ID}.witnesses.v1"
             and prior_witnesses.get("task_id") == TASK_ID
             and len(prior_witnesses.get("witnesses", [])) == 5,
             "predecessor witness source has an invalid task identity or five-row denominator")
    inputs.update({
        "source_mapper_path": checkpoint_path, "source_mapper": checkpoint,
        "v2_terminal_checkpoint": checkpoint, "response": response,
        "matrix": saved_matrix.double(), "inverse": inverse.double(),
        "projector_evidence_path": projector_path, "projector_check": matrix_check,
        "positive_control_witness_path": witness_path,
        "positive_control_witness": positive_control,
        "prior_training_witness_path": prior_witness_path,
        "prior_training_witnesses": prior_witnesses,
    })
    return inputs


def _load_population(config: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
    path = _resolve(config["source"]["source_population"])
    saved = torch.load(path, map_location="cpu", weights_only=False)
    _require(saved.get("schema") == "lightgen_g1_slm_mapping_population.v1"
             and saved.get("task_id") == pilot.TASK_ID,
             "source coefficient population is not the fixed original G1 population")
    cohorts: dict[str, dict[str, torch.Tensor]] = {}
    for name, per_class, offset in (("train", TRAIN_PER_CLASS, 0),
                                    ("validation", DEV_PER_CLASS, 4)):
        base = saved[name]
        indices = torch.cat([torch.arange(label * (1000 if name == "train" else 200) + offset,
                                          label * (1000 if name == "train" else 200) + offset + per_class)
                             for label in range(CLASS_COUNT)])
        cohorts[name] = {
            "coefficients": base["coefficients"][indices].to(torch.complex128).contiguous(),
            "labels": base["labels"][indices].clone(),
            "proposal_ids": base["proposal_ids"][indices].clone(),
            "source_indices": indices,
        }
        expected_labels = torch.arange(CLASS_COUNT).repeat_interleave(per_class)
        class_proposal_identity = torch.stack((cohorts[name]["labels"],
                                               cohorts[name]["proposal_ids"]), dim=1)
        _require(torch.equal(cohorts[name]["labels"], expected_labels)
                 and cohorts[name]["coefficients"].shape == (CLASS_COUNT * per_class, PORT_COUNT)
                 and torch.isfinite(cohorts[name]["coefficients"].real).all()
                 and torch.isfinite(cohorts[name]["coefficients"].imag).all()
                 and torch.unique(class_proposal_identity, dim=0).shape[0] == CLASS_COUNT * per_class,
                 f"fixed {name} cohort has malformed identities or coefficients")
    exact_overlap = (cohorts["train"]["coefficients"][:, None, :]
                     == cohorts["validation"]["coefficients"][None, :, :]).all(dim=2)
    _require(not exact_overlap.any().item(),
             "fixed training and development populations contain an identical complex target")
    cohorts["train"]["role"] = "fixed_train200"
    cohorts["validation"]["role"] = "historical_development200"
    return cohorts


def _build_schedule(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    per_class = [torch.arange(label * TRAIN_PER_CLASS, (label + 1) * TRAIN_PER_CLASS)
                 for label in range(CLASS_COUNT)]
    rows = []
    for _ in range(EXPOSURES_PER_TARGET):
        permutations = [indices[torch.randperm(TRAIN_PER_CLASS, generator=generator)]
                        for indices in per_class]
        for position in range(TRAIN_PER_CLASS):
            rows.append(torch.stack([permutations[label][position] for label in range(CLASS_COUNT)]))
    schedule = torch.stack(rows).long()
    _require(tuple(schedule.shape) == (UPDATE_BUDGET, CLASS_COUNT), "constructed sample schedule has wrong shape")
    counts = torch.bincount(schedule.reshape(-1), minlength=TRAIN_ROWS)
    _require(torch.equal(counts, torch.full((TRAIN_ROWS,), EXPOSURES_PER_TARGET)),
             "constructed schedule does not provide exactly 750 exposures per target")
    return schedule


def _run_identity(config: Mapping[str, Any], run_id: str) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID, run_id=run_id, trajectory_id=f"{run_id}:g1_semantic_exposure200",
        semantic_authority_id=AUTHORITY,
        data_role="FIXED_G1_TRAIN200_AND_HISTORICAL_DEVELOPMENT200",
        config_identity=dict(config),
        model_identity={"optics": "C_seed_2026101501_step_45000_frozen",
                        "mapper_start": "projected_mapping_v2_terminal_step_5000",
                        "mapper": "Linear(200,256)-SiLU-Linear(256,40000)-Sigmoid+fixed_projection",
                        "objective": "full_optics_Lc_plus_5Lbox"},
        optimizer_identity={"kind": "Adam", "learning_rate": 1.0e-4,
                            "betas": [0.9, 0.999], "epsilon": 1.0e-8,
                            "weight_decay": 0.0, "new_updates": UPDATE_BUDGET,
                            "source_optimizer_inherited": False},
        sampler_identity={"train_indices_per_class": list(range(TRAIN_PER_CLASS)),
                          "development_indices_per_class": list(range(4, 24)),
                          "exposures_per_target": EXPOSURES_PER_TARGET,
                          "schedule_seed": 2026102971, "rng_seed": 2026102970},
        execution_source_semantics_id=json.dumps({"task": TASK_ID, "config": config},
                                                 sort_keys=True, ensure_ascii=False),
    )


def _rng_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {"cpu": torch.get_rng_state().clone()}
    if torch.cuda.is_available():
        result["cuda"] = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return result


def _rng_equal(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return torch.equal(a["cpu"], b["cpu"]) and (
        ("cuda" not in a and "cuda" not in b)
        or ("cuda" in a and "cuda" in b and len(a["cuda"]) == len(b["cuda"])
            and all(torch.equal(x, y) for x, y in zip(a["cuda"], b["cuda"], strict=True)))
    )


def _copy_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in state.items()}


def _mapper(inputs: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    return projected._new_mapper(device, inputs["source_mapper"]["mapper_state"])


def _optimizer(mapper: torch.nn.Module, config: Mapping[str, Any]) -> torch.optim.Adam:
    training = config["training"]
    return torch.optim.Adam(mapper.parameters(), lr=training["learning_rate"],
                            betas=tuple(training["betas"]), eps=training["epsilon"],
                            weight_decay=training["weight_decay"])


def _phase_loss(actual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return pilot._phase_invariant_loss(actual.to(torch.complex128), target.to(torch.complex128))


def _phase_invariant_relative_error(actual: torch.Tensor, target: torch.Tensor) -> float:
    actual = actual.reshape(-1).to(torch.complex128)
    target = target.reshape(-1).to(device=actual.device, dtype=torch.complex128)
    target_norm = torch.linalg.vector_norm(target)
    _require(torch.isfinite(target_norm).item() and float(target_norm) > 0.0
             and torch.isfinite(actual.real).all().item()
             and torch.isfinite(actual.imag).all().item(),
             "phase-invariant coefficient metric received invalid values")
    overlap = torch.vdot(target, actual)
    phase = torch.angle(overlap) if torch.abs(overlap) > 0 else torch.zeros((), device=actual.device)
    aligned = target * torch.exp(1j * phase.to(dtype=torch.float64))
    return float(torch.linalg.vector_norm(actual - aligned) / target_norm)


def _dual_box_error_bounds(response: torch.Tensor, target: torch.Tensor,
                           amplitude: torch.Tensor) -> dict[str, float]:
    """Independent convex lower bound for a fixed-phase box least-squares point."""
    _require(response.dtype == torch.complex128 and target.dtype == torch.complex128
             and amplitude.dtype == torch.float64 and response.ndim == 2
             and target.ndim == 1 and amplitude.shape == (response.shape[1],),
             "dual box bound requires the fixed complex128 response and float64 command")
    predicted = amplitude.to(torch.complex128) @ response.T
    residual = target - predicted
    residual_norm = torch.linalg.vector_norm(residual)
    target_norm = torch.linalg.vector_norm(target)
    _require(torch.isfinite(target_norm).item() and target_norm.item() > 0.0
             and torch.isfinite(residual_norm).item(), "dual box bound received invalid values")
    upper = float(residual_norm / target_norm)
    if residual_norm.item() == 0.0:
        return {"relative_upper_bound": 0.0, "relative_dual_lower_bound": 0.0,
                "relative_gap": 0.0}
    direction = residual / residual_norm
    support = torch.clamp((direction.conj().reshape(1, -1) @ response).real.reshape(-1),
                          min=0.0).sum()
    lower = torch.clamp((torch.vdot(direction, target).real - support) / target_norm,
                        min=0.0)
    lower_value = float(lower)
    _require(lower_value <= upper + DUAL_BOUND_NUMERIC_TOLERANCE,
             "dual box lower bound exceeds its feasible primal upper bound")
    return {"relative_upper_bound": upper,
            "relative_dual_lower_bound": lower_value,
            "relative_gap": upper - lower_value}


def _phase_unit_interval(value: float) -> float:
    return float(value) % (2.0 * math.pi)


def _direction_for_phase(response: torch.Tensor, target: torch.Tensor, phase: float,
                         amplitude: torch.Tensor) -> torch.Tensor | None:
    desired = target * torch.exp(torch.tensor(1j * phase, dtype=torch.complex128,
                                               device=response.device))
    predicted = amplitude.to(device=response.device, dtype=torch.float64).to(
        torch.complex128).reshape(1, -1) @ response.T
    residual = desired - predicted.reshape(-1)
    norm = torch.linalg.vector_norm(residual)
    if not torch.isfinite(norm).item():
        raise RuntimeError("phase-dual direction has a nonfinite residual")
    if float(norm) <= 1.0e-30:
        return None
    direction = residual / norm
    _require(abs(float(torch.linalg.vector_norm(direction)) - 1.0) <= 1.0e-12,
             "phase-dual direction is not a unit vector")
    return direction.detach().cpu().clone()


def _fixed_phase_objective_gradient(response: torch.Tensor, target: torch.Tensor, phase: float,
                                    amplitude: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    desired = target * torch.exp(torch.tensor(1j * phase, dtype=torch.complex128,
                                               device=response.device))
    predicted = amplitude.to(device=response.device, dtype=torch.float64).to(
        torch.complex128).reshape(1, -1) @ response.T
    residual = predicted.reshape(-1) - desired
    objective = 0.5 * torch.sum(residual.abs().square())
    gradient = (residual.reshape(1, -1) @ response.conj()).real.reshape(-1)
    return objective, gradient


def _sinusoid_minimum(value: complex, left: float, right: float) -> float:
    """Exact minimum of Re(exp(i*phi)*value) on a closed phase interval."""
    _require(math.isfinite(left) and math.isfinite(right) and left <= right
             and math.isfinite(value.real) and math.isfinite(value.imag),
             "phase interval or sinusoid is invalid", ValueError)
    magnitude = abs(value)
    if magnitude == 0.0:
        return 0.0
    if right - left >= 2.0 * math.pi:
        return -magnitude
    angle = math.atan2(value.imag, value.real)
    minimum_phase = math.pi - angle
    first = math.ceil((left - minimum_phase) / (2.0 * math.pi) - 1.0e-14)
    critical = minimum_phase + first * 2.0 * math.pi
    candidates = [((complex(math.cos(left), math.sin(left)) * value).real),
                  ((complex(math.cos(right), math.sin(right)) * value).real)]
    if critical <= right + 1.0e-14:
        candidates.append(-magnitude)
    return min(candidates)


def _phase_interval_dual_lower_bound(response: torch.Tensor, target: torch.Tensor,
                                     left: float, right: float,
                                     directions: Sequence[torch.Tensor], *,
                                     numeric_guard: float = CONTINUOUS_PHASE_NUMERIC_GUARD
                                     ) -> float:
    """Conservative fixed-phase box-residual lower bound over a whole interval."""
    _require(response.dtype == torch.complex128 and target.dtype == torch.complex128
             and response.ndim == 2 and target.ndim == 1
             and response.shape[0] == target.numel()
             and left <= right and numeric_guard >= 0.0,
             "phase interval bound inputs are invalid")
    target_norm = float(torch.linalg.vector_norm(target))
    _require(math.isfinite(target_norm) and target_norm > 0.0,
             "phase interval bound target norm is invalid")
    lower_bounds = []
    for direction_cpu in directions:
        direction = direction_cpu.to(device=response.device, dtype=torch.complex128).reshape(-1)
        _require(direction.shape == target.shape and torch.isfinite(direction.real).all().item()
                 and torch.isfinite(direction.imag).all().item()
                 and abs(float(torch.linalg.vector_norm(direction)) - 1.0) <= 1.0e-10,
                 "phase interval dual direction is malformed")
        signal = complex(torch.vdot(direction, target).detach().cpu().item())
        support = torch.clamp((direction.conj().reshape(1, -1) @ response).real.reshape(-1),
                              min=0.0).sum()
        interval_signal_min = _sinusoid_minimum(signal, left, right)
        raw_lower = (interval_signal_min - float(support)) / target_norm
        lower_bounds.append(max(0.0, raw_lower - numeric_guard))
    return max(lower_bounds, default=0.0)


def _initial_phase_intervals(anchors: Sequence[float], response: torch.Tensor,
                             target: torch.Tensor,
                             directions: Sequence[torch.Tensor]) -> list[dict[str, Any]]:
    normalized = sorted(_phase_unit_interval(value) for value in anchors)
    unique: list[float] = []
    for phase in normalized:
        if not unique or phase - unique[-1] > 1.0e-12:
            unique.append(phase)
    if len(unique) > 1 and unique[0] + 2.0 * math.pi - unique[-1] <= 1.0e-12:
        unique.pop()
    _require(len(unique) >= 2, "continuous phase anchors do not partition a full cycle")
    intervals = []
    for index, left in enumerate(unique):
        right = unique[index + 1] if index + 1 < len(unique) else unique[0] + 2.0 * math.pi
        lower = _phase_interval_dual_lower_bound(response, target, left, right, directions)
        intervals.append({"left_rad": left, "right_rad": right,
                          "lower_bound": lower,
                          "status": "excluded" if lower > GATE_LIMIT else "pending",
                          "depth": 0})
    return intervals


def _adaptive_public_phase_search(response: torch.Tensor, target: torch.Tensor,
                                  anchors: Sequence[float], seed_amplitude: torch.Tensor, *,
                                  evaluate_candidate: Any,
                                  initial_state: Mapping[str, Any] | None = None,
                                  checkpoint: Any = None,
                                  lipschitz: float | None = None) -> dict[str, Any]:
    """Bisect unresolved phase intervals using verified dual lower bounds."""
    _require(response.dtype == torch.complex128 and target.dtype == torch.complex128
             and target.ndim == 1 and response.shape[0] == target.numel()
             and seed_amplitude.numel() == response.shape[1],
             "adaptive public-phase search inputs are incompatible")
    normalized_anchors = [_phase_unit_interval(float(value)) for value in anchors]
    seed = seed_amplitude.detach().to(device="cpu", dtype=torch.float32).reshape(-1).clone()
    if initial_state is None:
        direction_records = []
        for phase in normalized_anchors:
            direction = _direction_for_phase(response, target, phase, seed.to(torch.float64))
            if direction is not None:
                direction_records.append({"phase_rad": phase, "source": "seed_command",
                                          "command": seed.clone(), "direction": direction})
        directions = [row["direction"] for row in direction_records]
        state: dict[str, Any] = {
            "schema": f"{TASK_ID}.continuous_phase_search.v1",
            "phase_anchors": normalized_anchors,
            "seed_amplitude": seed,
            "directions": direction_records,
            "intervals": _initial_phase_intervals(normalized_anchors, response, target, directions),
            "attempts": [], "completed_solves": 0, "completed_iterations": 0,
            "active": None, "status": "searching",
            "best_candidate_error_2048": 1.0e300,
            "best_candidate": None, "best_full_record": None}
    else:
        state = copy.deepcopy(dict(initial_state))
        _require(state.get("schema") == f"{TASK_ID}.continuous_phase_search.v1"
                 and state.get("phase_anchors") == normalized_anchors
                 and torch.equal(state.get("seed_amplitude"), seed)
                 and 0 <= int(state.get("completed_solves", -1)) <= CONTINUOUS_PHASE_SOLVE_LIMIT
                 and int(state.get("completed_iterations", -1)) >= 0
                 and len(state.get("attempts", [])) == int(state["completed_solves"])
                 and isinstance(state.get("intervals"), list)
                 and isinstance(state.get("directions"), list)
                 and "best_candidate_error_2048" in state,
                 "continuous phase checkpoint has an invalid search cursor")

    spectral_lipschitz = (feasibility._spectral_lipschitz(response)
                          if lipschitz is None else float(lipschitz))
    _require(math.isfinite(spectral_lipschitz) and spectral_lipschitz > 0.0,
             "adaptive phase search requires a finite cached spectral constant")

    def refresh_bounds() -> None:
        direction_values = [row["direction"] for row in state["directions"]]
        for interval in state["intervals"]:
            left, right = float(interval["left_rad"]), float(interval["right_rad"])
            lower = _phase_interval_dual_lower_bound(response, target, left, right,
                                                     direction_values)
            interval["lower_bound"] = lower
            if lower > GATE_LIMIT:
                interval["status"] = "excluded"
            elif right - left <= CONTINUOUS_PHASE_INTERVAL_MIN_WIDTH:
                interval["status"] = "unresolved_resolution"
            else:
                interval["status"] = "pending"

    refresh_bounds()
    while True:
        if state["status"] == "witness_found":
            break
        pending = [(index, row) for index, row in enumerate(state["intervals"])
                   if row["status"] == "pending"]
        if not pending:
            if all(row["status"] == "excluded" for row in state["intervals"]):
                state["status"] = "whole_phase_circle_excluded"
            else:
                state["status"] = "phase_interval_unresolved"
            break
        if int(state["completed_solves"]) >= CONTINUOUS_PHASE_SOLVE_LIMIT:
            state["status"] = "new_phase_budget_exhausted"
            break
        interval_index, interval = min(
            pending, key=lambda item: (float(item[1]["lower_bound"]),
                                       -(float(item[1]["right_rad"])
                                         - float(item[1]["left_rad"])),
                                       float(item[1]["left_rad"])))
        left, right = float(interval["left_rad"]), float(interval["right_rad"])
        phase = _phase_unit_interval(0.5 * (left + right))
        active = state.get("active")
        if active is not None:
            _require(abs(float(active["left_rad"]) - left) <= 1.0e-14
                     and abs(float(active["right_rad"]) - right) <= 1.0e-14
                     and abs(math.atan2(math.sin(float(active["phase_rad"]) - phase),
                                        math.cos(float(active["phase_rad"]) - phase))) <= 1.0e-14,
                     "active continuous phase solve no longer matches the queue head")
            solver_state = active.get("solver_state")
        else:
            solver_state = None
            state["active"] = {"left_rad": left, "right_rad": right,
                               "phase_rad": phase, "solver_state": None}
            if checkpoint is not None:
                checkpoint(copy.deepcopy(state))

        def solver_checkpoint(saved_solver: Mapping[str, Any]) -> None:
            state["active"]["solver_state"] = copy.deepcopy(dict(saved_solver))
            if checkpoint is not None:
                checkpoint(copy.deepcopy(state))

        solved = _fista_fixed_phase_solve(
            response, target, phase, max_steps=5000, lipschitz=spectral_lipschitz,
            initial_state=solver_state, checkpoint=solver_checkpoint,
            checkpoint_interval=500)
        command = solved["amplitude"].detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        evaluated = evaluate_candidate({**solved, "amplitude": command.clone()})
        error_2048 = float(evaluated["coefficient_relative_error_2048"])
        _require(math.isfinite(error_2048) and error_2048 >= 0.0,
                 "actual 2048 production coefficient check is invalid")
        direction = _direction_for_phase(response, target, phase, command.to(torch.float64))
        desired = target * torch.exp(torch.tensor(1j * phase, dtype=torch.complex128,
                                                   device=response.device))
        matrix_prediction = command.to(device=response.device, dtype=torch.float64).to(
            torch.complex128).reshape(1, -1) @ response.T
        fixed_error = float(torch.linalg.vector_norm(desired - matrix_prediction.reshape(-1))
                            / torch.linalg.vector_norm(target))
        direction_index = None
        if direction is not None:
            direction_index = len(state["directions"])
            state["directions"].append({"phase_rad": phase, "source": "candidate_command",
                                        "command": command.clone(), "direction": direction})
        attempt = {"phase_rad": phase, "parent_left_rad": left, "parent_right_rad": right,
                   "iterations": int(solved["iterations"]),
                   "solver_relative_error": float(solved["relative_error"]),
                   "relative_upper_bound": float(solved["relative_upper_bound"]),
                   "relative_dual_lower_bound": float(solved["relative_dual_lower_bound"]),
                   "relative_gap": float(solved["relative_gap"]),
                   "fixed_phase_relative_error_from_command": fixed_error,
                   "coefficient_relative_error_2048": error_2048,
                   "all_six_metrics_passed": bool(evaluated.get("all_six_metrics_passed", False)),
                   "direction_index": direction_index, "command": command.clone()}
        if evaluated.get("all_six_metrics_passed"):
            attempt["metrics"] = dict(evaluated["record"]["metrics"])
        state["attempts"].append(attempt)
        if error_2048 < float(state["best_candidate_error_2048"]):
            state["best_candidate_error_2048"] = error_2048
            state["best_candidate"] = {
                "amplitude": command.clone(), "phase_rad": phase,
                "iterations": int(solved["iterations"]),
                "solver_relative_error": float(solved["relative_error"]),
                "coefficient_relative_error_2048": error_2048}
            state["best_full_record"] = (copy.deepcopy(evaluated["record"])
                                          if evaluated.get("record") is not None else None)
        state["completed_solves"] = int(state["completed_solves"]) + 1
        state["completed_iterations"] = (int(state["completed_iterations"])
                                         + int(solved["iterations"]))
        state["active"] = None
        state["intervals"].pop(interval_index)
        midpoint = 0.5 * (left + right)
        child_intervals = [
            {"left_rad": left, "right_rad": midpoint, "lower_bound": 0.0,
             "status": "pending", "depth": int(interval.get("depth", 0)) + 1},
            {"left_rad": midpoint, "right_rad": right, "lower_bound": 0.0,
             "status": "pending", "depth": int(interval.get("depth", 0)) + 1},
        ]
        state["intervals"][interval_index:interval_index] = child_intervals
        refresh_bounds()
        if evaluated.get("all_six_metrics_passed"):
            state["status"] = "witness_found"
            state["witness_record"] = copy.deepcopy(evaluated["record"])
        if checkpoint is not None:
            checkpoint(copy.deepcopy(state))
        if state["status"] == "witness_found":
            break
    return state


def _fallback_phase_schedule(v2_phase: float, prior_best_phase: float) -> list[float]:
    """V2 and prior phases followed by a fixed 32-point public-phase mesh."""
    candidates = [v2_phase, prior_best_phase,
                  *(2.0 * math.pi * index / 32 for index in range(32))]
    unique: list[float] = []
    for value in candidates:
        phase = (float(value) + math.pi) % (2.0 * math.pi) - math.pi
        if all(abs(math.atan2(math.sin(phase - old), math.cos(phase - old))) > 1.0e-12
               for old in unique):
            unique.append(phase)
    _require(2 <= len(unique) <= 34, "fallback phase schedule has an invalid number of phases")
    return unique


def _fista_fixed_phase_solve(response: torch.Tensor, target: torch.Tensor, phase: float, *,
                             max_steps: int = 5000,
                             lipschitz: float | None = None,
                             initial_state: Mapping[str, Any] | None = None,
                             checkpoint: Any = None,
                             checkpoint_interval: int = 500) -> dict[str, Any]:
    """Solve one convex box least-squares phase with a resumable float64 FISTA state."""
    _require(response.dtype == torch.complex128 and target.dtype == torch.complex128
             and response.ndim == 2 and target.ndim == 1
             and response.shape[0] == target.numel(),
             "FISTA phase solver requires a dimensionally matched complex128 response")
    _require(0 < max_steps <= 5000, "FISTA phase budget exceeds the fixed bound", ValueError)
    pixels = response.shape[1]
    lipschitz = (feasibility._spectral_lipschitz(response)
                 if lipschitz is None else float(lipschitz))
    _require(math.isfinite(lipschitz) and lipschitz > 0.0,
             "FISTA requires a finite positive cached spectral constant")
    target_norm = torch.linalg.vector_norm(target)
    _require(torch.isfinite(target_norm).item() and target_norm.item() > 0.0,
             "FISTA target must have a finite positive norm")
    desired = target * torch.exp(torch.tensor(1j * phase, dtype=torch.complex128,
                                               device=response.device))
    if initial_state is None:
        amplitude = torch.full((pixels,), 0.5, dtype=torch.float64, device=response.device)
        extrapolated = amplitude.clone()
        momentum = 1.0
        iteration = 0
        best_amplitude = amplitude.clone()
        best_error = float("inf")
    else:
        _require(float(initial_state["phase_rad"]) == float(phase),
                 "FISTA checkpoint phase does not match the requested phase")
        amplitude = initial_state["amplitude"].to(device=response.device, dtype=torch.float64)
        extrapolated = initial_state["extrapolated"].to(device=response.device, dtype=torch.float64)
        best_amplitude = initial_state["best_amplitude"].to(device=response.device,
                                                            dtype=torch.float64)
        momentum = float(initial_state["momentum"])
        iteration = int(initial_state["iteration"])
        best_error = float(initial_state["best_error"])
        _require(tuple(amplitude.shape) == (pixels,) and tuple(extrapolated.shape) == (pixels,)
                 and tuple(best_amplitude.shape) == (pixels,) and 0 <= iteration <= max_steps
                 and math.isfinite(momentum) and momentum >= 1.0
                 and math.isfinite(best_error) and best_error >= 0.0
                 and torch.isfinite(amplitude).all().item()
                 and torch.isfinite(extrapolated).all().item()
                 and torch.isfinite(best_amplitude).all().item()
                 and 0.0 <= float(amplitude.min()) <= float(amplitude.max()) <= 1.0
                 and 0.0 <= float(best_amplitude.min()) <= float(best_amplitude.max()) <= 1.0,
                 "FISTA checkpoint state is malformed")
    while iteration < max_steps:
        _, gradient = _fixed_phase_objective_gradient(response, target, phase, extrapolated)
        updated = torch.clamp(extrapolated - gradient / lipschitz, 0.0, 1.0)
        next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
        extrapolated = updated + ((momentum - 1.0) / next_momentum) * (updated - amplitude)
        amplitude, momentum = updated, next_momentum
        iteration += 1
        bounds = None
        if iteration % 25 == 0 or iteration == max_steps:
            bounds = _dual_box_error_bounds(response, desired, amplitude)
            if bounds["relative_upper_bound"] < best_error:
                best_error = bounds["relative_upper_bound"]
                best_amplitude = amplitude.detach().clone()
        if checkpoint is not None and iteration % checkpoint_interval == 0:
            best_bounds = _dual_box_error_bounds(response, desired, best_amplitude)
            checkpoint({"phase_rad": float(phase), "iteration": iteration,
                        "amplitude": amplitude.detach().cpu().clone(),
                        "extrapolated": extrapolated.detach().cpu().clone(),
                        "momentum": momentum,
                        "best_amplitude": best_amplitude.detach().cpu().clone(),
                        "best_error": best_error, **best_bounds})
        if bounds is not None and (bounds["relative_upper_bound"] <= 1.0e-6
                                   or bounds["relative_gap"] <= 1.0e-8):
            break
    final_bounds = _dual_box_error_bounds(response, desired, best_amplitude)
    return {"amplitude": best_amplitude.detach().cpu().clone(),
            "phase_rad": float(phase), "relative_error": final_bounds["relative_upper_bound"],
            "iterations": iteration, **final_bounds}


def _forward_matrix(amplitude: torch.Tensor, response: torch.Tensor) -> torch.Tensor:
    flat = amplitude.reshape(amplitude.shape[0], PIXEL_COUNT).double()
    return flat.to(torch.complex128) @ response.to(device=flat.device, dtype=torch.complex128).T


def _update(mapper: torch.nn.Module, optimizer: torch.optim.Optimizer, model: Any,
            inputs: Mapping[str, Any], coefficients: torch.Tensor, device: torch.device,
            engine: str, *, audit: bool = False) -> dict[str, Any]:
    target = coefficients.to(device=device, dtype=torch.complex128)
    features = projected._normalized_features(coefficients, inputs["scale"], device)
    old = [parameter.detach().clone() for parameter in mapper.parameters()] if audit else []
    optimizer.zero_grad(set_to_none=True)
    amplitude, projection = projected._map_amplitude(
        mapper, features, target, inputs["matrix"].to(device),
        inputs["inverse"].to(device), inputs["scale"].to(device), include_preclip=True,
    )
    if engine == "full_optics":
        actual = projected._forward_amplitude(model, amplitude)["coefficients"]
    elif engine == "response_matrix":
        actual = _forward_matrix(amplitude, inputs["response_matrix"])
    else:
        raise ValueError(f"unknown fixed training engine: {engine}")
    coefficient_loss = _phase_loss(actual, target).mean()
    box_loss = fit20._box_loss(projection["preclip_amplitude"].reshape(-1, PIXEL_COUNT))
    total = coefficient_loss + BOX_LAMBDA * box_loss
    _require(torch.isfinite(total).item() and torch.isfinite(actual.real).all().item()
             and torch.isfinite(actual.imag).all().item(), "training producer returned nonfinite values")
    total.backward()
    gradient_norms = {}
    for name, parameter in mapper.named_parameters():
        _require(parameter.grad is not None and torch.isfinite(parameter.grad).all().item(),
                 f"mapper gradient missing or nonfinite: {name}")
        gradient_norms[name] = float(torch.linalg.vector_norm(parameter.grad.detach().double()))
    optimizer.step()
    if audit:
        _require(all(not torch.equal(before, after.detach()) for before, after in
                     zip(old, mapper.parameters(), strict=True)),
                 "a mapper parameter tensor did not update")
    return {"loss": float(total.detach()), "coefficient_loss": float(coefficient_loss.detach()),
            "box_loss": float(box_loss.detach()),
            "coefficient_relative_error_mean": float(torch.sqrt(_phase_loss(actual, target).detach()).mean()),
            "clipped_fraction": float(projection["clipped_fraction"].mean().detach()),
            "gradient_norms": gradient_norms, "sample_indices": None}


def _reconsume_predecessor_witnesses(config: Mapping[str, Any], inputs: Mapping[str, Any],
                                     population: Mapping[str, Any],
                                     device: torch.device
                                     ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = _resolve(config["source"]["prior_training_witnesses"])
    _require(path.is_file() and Path(inputs["prior_training_witness_path"]).resolve() == path.resolve(),
             f"fixed predecessor witness evidence is missing or differs from loaded source: {path}",
             FileNotFoundError)
    saved = inputs["prior_training_witnesses"]
    ids = population["train"]["proposal_ids"]
    _require(saved.get("schema") == f"{TASK_ID}.witnesses.v1"
             and saved.get("task_id") == TASK_ID
             and torch.equal(saved.get("population_ids"), ids)
             and len(saved.get("witnesses", [])) == 5
             and saved.get("failed_indices")
             and int(saved["failed_indices"][0]["index"]) == 4,
             "predecessor witness evidence does not match the fixed population and stop row")
    response = inputs["response_matrix"].to(torch.complex128)
    optical = {grid: _make_runtime_model(inputs, grid, device) for grid in GRIDS}
    optical_before = _state_snapshot(optical)

    def replay(index: int, row: Mapping[str, Any], *, expected_pass: bool,
               claimed_metrics: Mapping[str, Any]) -> dict[str, Any]:
        target_cpu = population["train"]["coefficients"][index:index + 1]
        target = target_cpu.to(device=device, dtype=torch.complex128)
        command = row["amplitude"].to(device=device, dtype=torch.float32)
        _require(tuple(command.shape) == (1, 1, 200, 200)
                 and torch.isfinite(command).all().item()
                 and float(command.min()) >= 0.0 and float(command.max()) <= 1.0,
                 f"predecessor witness command is malformed at row {index}")
        with torch.no_grad():
            actual = {grid: projected._forward_amplitude(optical[grid], command) for grid in GRIDS}
            direct = {grid: projected._forward_target(optical[grid], target) for grid in GRIDS}
        matrix_prediction = _forward_matrix(command, response.to(device=device)).reshape(-1)
        actual_coefficients = actual[2048]["coefficients"].reshape(-1).to(torch.complex128)
        response_error = float(torch.linalg.vector_norm(matrix_prediction - actual_coefficients)
                               / torch.linalg.vector_norm(actual_coefficients).clamp_min(1.0e-30))
        _require(response_error <= 1.0e-6,
                 f"predecessor witness command fails frozen matrix response at row {index}")
        replay_errors: dict[str, float] = {}
        for grid in GRIDS:
            for key in ("coefficients", "raw_cmos"):
                saved = row["outputs"][grid][key].to(device=device)
                current = actual[grid][key]
                dtype = torch.complex128 if current.is_complex() or saved.is_complex() else torch.float64
                saved, current = saved.to(dtype), current.to(dtype)
                error = float(torch.linalg.vector_norm(current - saved)
                              / torch.linalg.vector_norm(saved).clamp_min(1.0e-30))
                replay_errors[f"{grid}_{key}"] = error
                _require(error <= RECOVERY_TOLERANCE,
                         f"predecessor full-optics output does not replay at row {index}, {grid}_{key}")
            saved_direct = row["direct_raw_cmos"][grid].to(device=device, dtype=torch.float64)
            current_direct = direct[grid]["raw_cmos"].to(torch.float64)
            error = float(torch.linalg.vector_norm(current_direct - saved_direct)
                          / torch.linalg.vector_norm(saved_direct).clamp_min(1.0e-30))
            replay_errors[f"{grid}_direct_raw_cmos"] = error
            _require(error <= RECOVERY_TOLERANCE,
                     f"predecessor direct-Decoder output does not replay at row {index}, grid {grid}")
        metrics = projected._error_row(
            actual, target, {grid: direct[grid]["raw_cmos"] for grid in GRIDS}, 0)
        numpy_outputs = {grid: {key: value.detach().cpu().numpy()
                                for key, value in actual[grid].items()}
                         for grid in GRIDS}
        numpy_direct = {grid: direct[grid]["raw_cmos"].detach().cpu().numpy()
                        for grid in GRIDS}
        numpy_metrics = _numpy_error_row(numpy_outputs, target_cpu[0].numpy(), numpy_direct)
        _require(all(abs(metrics[key] - numpy_metrics[key]) <= 1.0e-14
                     and abs(metrics[key] - float(claimed_metrics[key])) <= 1.0e-14
                     for key in METRIC_KEYS)
                 and all(metrics[key] <= GATE_LIMIT for key in METRIC_KEYS) == expected_pass,
                 f"predecessor full-optics replay metrics differ at row {index}")
        return {**row, "source_index": int(population["train"]["source_indices"][index]),
                "target_coefficients": target_cpu.detach().cpu().clone(),
                "amplitude": command.detach().cpu().clone(),
                "outputs": {grid: {key: value.detach().cpu().clone()
                                   for key, value in actual[grid].items()}
                            for grid in GRIDS},
                "direct_raw_cmos": {grid: direct[grid]["raw_cmos"].detach().cpu().clone()
                                    for grid in GRIDS},
                "metrics": metrics, "all_six_metrics_passed": expected_pass,
                "predecessor_reconsumed": True,
                "matrix_response_relative_error": response_error,
                "full_optics_replay_relative_errors": replay_errors}

    verified: list[dict[str, Any]] = []
    for index, row in enumerate(saved["witnesses"][:4]):
        _require(int(row.get("index", -1)) == index
                 and int(row.get("label", -1)) == int(population["train"]["labels"][index])
                 and int(row.get("proposal_id", -1)) == int(ids[index])
                 and bool(row.get("all_six_metrics_passed")),
                 f"predecessor positive witness identity is invalid at row {index}")
        verified.append(replay(index, row, expected_pass=True, claimed_metrics=row["metrics"]))
    failed = saved["failed_indices"][0]
    row4 = saved["witnesses"][4]
    _require(int(failed.get("index", -1)) == 4
             and int(failed.get("label", -1)) == int(population["train"]["labels"][4])
             and int(failed.get("proposal_id", -1)) == int(ids[4])
             and int(row4.get("index", -1)) == 4
             and int(row4.get("proposal_id", -1)) == int(ids[4]),
             "predecessor failed-row record does not identify the fifth fixed training target")
    failure_replay = replay(4, row4, expected_pass=False, claimed_metrics=failed["metrics"])
    _require(all(_compare_parameter_states(optical_before[str(grid)], optical[grid].state_dict()) == 0.0
                 for grid in GRIDS),
             "optical state changed during predecessor full-optics replay")
    return verified, {**failure_replay,
                      "historical_claimed_pass": bool(row4.get("all_six_metrics_passed")),
                      "predecessor_failure_reconsumed": True}


def _continuous_phase_recovery_replay(config: Mapping[str, Any], inputs: Mapping[str, Any],
                                      population: Mapping[str, Any], checkpoint_path: Path,
                                      output_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require(checkpoint.get("schema") == f"{TASK_ID}.phase_solver_recovery.v1"
             and checkpoint.get("task_id") == TASK_ID
             and int(checkpoint.get("target_index", -1)) == 4
             and int(checkpoint.get("proposal_id", -1))
             == int(population["train"]["proposal_ids"][4])
             and torch.equal(checkpoint.get("target_coefficients"),
                             population["train"]["coefficients"][4].cpu()),
             "phase solver recovery checkpoint does not match fixed target five")
    state3 = checkpoint.get("solver_state")
    phase = float(checkpoint.get("phase_rad", float("nan")))
    _require(isinstance(state3, Mapping) and int(state3.get("iteration", -1)) == 3
             and float(state3.get("phase_rad", float("nan"))) == phase,
             "phase solver recovery checkpoint is not at its registered third iteration")
    response = inputs["response_matrix"].to(device=device, dtype=torch.complex128)
    target = population["train"]["coefficients"][4].to(device=device, dtype=torch.complex128)
    states = []
    result = _fista_fixed_phase_solve(
        response, target, phase, max_steps=4,
        lipschitz=float(checkpoint["lipschitz"]), initial_state=state3,
        checkpoint=states.append, checkpoint_interval=4)
    _require(states and int(states[-1]["iteration"]) == 4,
             "independent phase recovery replay did not execute exactly the fourth iteration")
    payload = {"schema": f"{TASK_ID}.phase_solver_recovery_step4.v1",
               "target_index": 4, "proposal_id": int(population["train"]["proposal_ids"][4]),
               "phase_rad": phase, "cursor": {"target_index": 4, "phase_rad": phase,
                                                "iteration": 4},
               "solver_state": states[-1], "relative_error": result["relative_error"],
               "relative_upper_bound": result["relative_upper_bound"],
               "relative_dual_lower_bound": result["relative_dual_lower_bound"],
               "relative_gap": result["relative_gap"]}
    _save_torch(output_path, payload)
    return {"status": "PHASE_SOLVER_STEP4_REPLAY_SAVED", "output_path": str(output_path),
            "iteration": 4, "relative_error": result["relative_error"]}


def _continuous_phase_recovery_preflight(config_path: Path, config: Mapping[str, Any],
                                         inputs: Mapping[str, Any],
                                         population: Mapping[str, Any], device: torch.device,
                                         reference_dir: Path,
                                         predecessor: tuple[list[dict[str, Any]], dict[str, Any]]
                                         ) -> dict[str, Any]:
    reference_dir = reference_dir.resolve()
    _require(reference_dir.is_relative_to(RUNS_DIR.resolve()),
             "phase solver recovery must stay inside this experiment's runs directory", ValueError)
    recovery_roots = sorted(reference_dir.glob("phase_solver_recovery*"))
    recovery_dir = reference_dir / f"phase_solver_recovery_{len(recovery_roots) + 1:02d}"
    _require(not recovery_dir.exists(),
             "phase solver recovery revision path already exists", FileExistsError)
    predecessor_witnesses, failure = predecessor
    _require(len(predecessor_witnesses) == 4 and int(failure["index"]) == 4,
             "phase solver recovery requires the four reconsumed positives and fixed fifth target")
    phase = float(failure["phase_rad"])
    response = inputs["response_matrix"].to(device=device, dtype=torch.complex128)
    target = population["train"]["coefficients"][4].to(device=device, dtype=torch.complex128)
    lipschitz = float(feasibility._spectral_lipschitz(response))
    initial_states = []
    torch_rng_before = _rng_snapshot()
    first = _fista_fixed_phase_solve(
        response, target, phase, max_steps=3, lipschitz=lipschitz,
        checkpoint=initial_states.append, checkpoint_interval=3)
    _require(initial_states and int(initial_states[-1]["iteration"]) == 3,
             "phase solver recovery did not save the third iteration")
    recovery_dir.mkdir(parents=True)
    checkpoint_path = recovery_dir / "checkpoint_step_000003.pt"
    checkpoint_payload = {"schema": f"{TASK_ID}.phase_solver_recovery.v1",
                          "task_id": TASK_ID, "target_index": 4,
                          "proposal_id": int(population["train"]["proposal_ids"][4]),
                          "target_coefficients": target.detach().cpu().clone(),
                          "population_ids": population["train"]["proposal_ids"].clone(),
                          "phase_rad": phase, "lipschitz": lipschitz,
                          "solver_state": copy.deepcopy(initial_states[-1]),
                          "rng_state": torch_rng_before}
    _save_torch(checkpoint_path, checkpoint_payload)
    _require(_rng_equal(torch_rng_before, _rng_snapshot()),
             "phase solver checkpoint serialization changed process RNG")
    direct_states = []
    direct = _fista_fixed_phase_solve(
        response, target, phase, max_steps=4, lipschitz=lipschitz,
        initial_state=initial_states[-1], checkpoint=direct_states.append,
        checkpoint_interval=4)
    _require(direct_states and int(direct_states[-1]["iteration"]) == 4,
             "original process did not execute the registered fourth FISTA iteration")
    direct_payload = {"schema": f"{TASK_ID}.phase_solver_recovery_step4.v1",
                      "target_index": 4,
                      "proposal_id": int(population["train"]["proposal_ids"][4]),
                      "phase_rad": phase,
                      "cursor": {"target_index": 4, "phase_rad": phase, "iteration": 4},
                      "solver_state": direct_states[-1],
                      "relative_error": direct["relative_error"],
                      "relative_upper_bound": direct["relative_upper_bound"],
                      "relative_dual_lower_bound": direct["relative_dual_lower_bound"],
                      "relative_gap": direct["relative_gap"]}
    direct_path = recovery_dir / "direct_step_000004.pt"
    _save_torch(direct_path, direct_payload)
    _require(_rng_equal(torch_rng_before, _rng_snapshot()),
             "direct recovery checkpoint serialization changed process RNG")
    independent_path = recovery_dir / "independent_step_000004.pt"
    command = [sys.executable, "-m", "scripts.research.lightgen.execute_lightgen_g1_semantic_exposure200_v1",
               "--mode", "continuous-phase-recovery-replay", "--config", str(config_path),
               "--checkpoint", str(checkpoint_path), "--recovery-output", str(independent_path)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    _require(completed.returncode == 0,
             "independent phase solver recovery process failed: "
             + completed.stderr[-4000:])
    independent = torch.load(independent_path, map_location="cpu", weights_only=False)
    direct_saved = torch.load(direct_path, map_location="cpu", weights_only=False)
    same_tree, state_delta = projected._tree_max_delta(
        direct_saved["solver_state"], independent["solver_state"])
    loss_delta = abs(float(direct_saved["relative_error"])
                     - float(independent["relative_error"]))
    cursor_equal = direct_saved["cursor"] == independent["cursor"]
    _require(same_tree and state_delta <= 1.0e-12 and loss_delta <= 1.0e-12
             and cursor_equal and _rng_equal(torch_rng_before, _rng_snapshot()),
             "phase solver direct and independent restored fourth-step states diverged")
    summary = {"task_id": TASK_ID, "status": "PHASE_SOLVER_RECOVERY_PASSED",
               "target_index": 4, "proposal_id": int(population["train"]["proposal_ids"][4]),
               "phase_rad": phase, "checkpoint_iteration": 3,
               "direct_iteration": 4, "independent_iteration": 4,
               "third_iteration_relative_error": float(first["relative_error"]),
               "direct_vs_independent_state_max_abs_delta": state_delta,
               "direct_vs_independent_loss_abs_delta": loss_delta,
               "cursor_exact": cursor_equal, "rng_unchanged_during_save": True,
               "solver_calls_executed": 5, "adam_calls_executed": 0,
               "predecessor_positive_rows_reconsumed": 4,
               "evidence_dir": str(recovery_dir),
               "files": [path.name for path in (checkpoint_path, direct_path, independent_path)]}
    _write_json(recovery_dir / "phase_solver_recovery_summary.json", summary)
    _write_json(reference_dir / "phase_solver_recovery_summary.json", summary)
    shutil.copy2(config_path, recovery_dir / "run_config.yaml")
    return summary


def _solver_qualification(config: Mapping[str, Any], inputs: Mapping[str, Any],
                          population: Mapping[str, Any], device: torch.device, *,
                          output_dir: Path | None = None,
                          resume_checkpoint: Path | None = None,
                          continuous_phase: bool = False,
                          preconsumed_predecessor: tuple[list[dict[str, Any]], dict[str, Any]] | None = None
                          ) -> dict[str, Any]:
    targets = population["train"]["coefficients"]
    response = inputs["response_matrix"].to(device=device, dtype=torch.complex128)
    continuous_lipschitz = (feasibility._spectral_lipschitz(response)
                             if continuous_phase else None)
    matrix, inverse = inputs["matrix"].to(device), inputs["inverse"].to(device)
    scale = inputs["scale"].to(device)
    mapper = _mapper(inputs, device).eval()
    optical_models = {grid: _make_runtime_model(inputs, grid, device) for grid in GRIDS}
    optical_state = _state_snapshot(optical_models)
    if output_dir is None and "source" in config:
        output_dir = _resolve(config["output"]["reference_qualification_root"])
    if output_dir is not None:
        output_dir = output_dir.resolve()
        _require(output_dir.is_relative_to(RUNS_DIR.resolve()),
                 "reference qualification output must stay inside this experiment's runs directory",
                 ValueError)
    witness_root = None if output_dir is None else output_dir / "witness_rows"
    witness_dir: Path | None = None
    if output_dir is not None and resume_checkpoint is None:
        if continuous_phase and output_dir.exists():
            _require((output_dir / "phase_solver_recovery_summary.json").is_file()
                     and not witness_root.exists(),
                     "continuous-phase output exists without its completed solver recovery preflight",
                     FileExistsError)
        else:
            _require(not output_dir.exists(), f"reference qualification already exists: {output_dir}",
                     FileExistsError)
            output_dir.mkdir(parents=True)
        witness_root.mkdir()
        witness_dir = witness_root / "revision_0001"
        witness_dir.mkdir()
    elif output_dir is not None:
        _require(output_dir.is_dir() and witness_root.is_dir(),
                 "reference resume requires the existing qualification directory", FileNotFoundError)
        if continuous_phase:
            recovery_summary = output_dir / "phase_solver_recovery_summary.json"
            _require(recovery_summary.is_file()
                     and _json(recovery_summary).get("status") == "PHASE_SOLVER_RECOVERY_PASSED",
                     "continuous-phase resume lacks its accepted solver recovery evidence",
                     FileNotFoundError)

    witnesses: list[dict[str, Any]] = []
    predecessor_failure: Mapping[str, Any] | None = None
    total_iterations = 0
    if resume_checkpoint is not None:
        _require(output_dir is not None and resume_checkpoint.is_file(),
                 "reference resume requires an explicit checkpoint and output directory", ValueError)
        saved = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        ids = population["train"]["proposal_ids"]
        _require(saved.get("schema") == f"{TASK_ID}.reference_solver_checkpoint.v1"
                 and saved.get("task_id") == TASK_ID
                 and torch.equal(saved.get("population_ids"), ids),
                 "explicit reference checkpoint belongs to a different task or population")
        start_index = int(saved["target_index"])
        completed = int(saved["completed_targets"])
        _require(0 <= completed <= start_index <= TRAIN_ROWS,
                 "reference checkpoint cursor is outside the fixed training population")
        previous_row_store = Path(str(saved.get("row_store", "")))
        _require(not previous_row_store.is_absolute() and ".." not in previous_row_store.parts,
                 "reference checkpoint row store path is unsafe")
        previous_witness_dir = (output_dir / previous_row_store).resolve()
        _require(previous_witness_dir.is_relative_to(witness_root.resolve())
                 and previous_witness_dir.is_dir(),
                 "explicit reference checkpoint does not name an existing committed-row store")
        revisions = sorted(path for path in witness_root.glob("revision_*") if path.is_dir())
        next_revision = len(revisions) + 1
        witness_dir = witness_root / f"revision_{next_revision:04d}"
        _require(not witness_dir.exists(), f"reference resume revision already exists: {witness_dir}",
                 FileExistsError)
        witness_dir.mkdir()
        for index in range(completed):
            row_path = previous_witness_dir / f"row_{index:04d}.pt"
            _require(row_path.is_file(), f"completed reference row is missing: {row_path}", FileNotFoundError)
            witnesses.append(torch.load(row_path, map_location="cpu", weights_only=False))
            shutil.copy2(row_path, witness_dir / row_path.name)
        _require(len(witnesses) == completed,
                 "reference checkpoint completed-row prefix is incomplete")
        current = saved
        total_iterations = int(saved.get("completed_solver_iterations", saved["total_iterations"]))
        rng_state = saved.get("rng_state")
        _require(isinstance(rng_state, Mapping) and torch.is_tensor(rng_state.get("cpu")),
                 "reference checkpoint is missing the saved RNG state")
        torch.set_rng_state(rng_state["cpu"])
        if "cuda" in rng_state:
            _require(device.type == "cuda" and torch.cuda.is_available(),
                     "reference checkpoint requires its saved CUDA RNG state")
            torch.cuda.set_rng_state_all(rng_state["cuda"])
        if start_index == 4:
            _, predecessor_failure = _reconsume_predecessor_witnesses(
                config, inputs, population, device)
    else:
        start_index = 0
        current = None
        if "source" in config:
            initial_witnesses, predecessor_failure = (
                preconsumed_predecessor if preconsumed_predecessor is not None else
                _reconsume_predecessor_witnesses(config, inputs, population, device))
            witnesses.extend(initial_witnesses)
            start_index = len(initial_witnesses)
            if witness_dir is not None:
                for row in initial_witnesses:
                    _save_torch(witness_dir / f"row_{int(row['index']):04d}.pt", row)
            current = {"target_index": start_index, "completed_targets": start_index,
                       "stage": "target_boundary", "phase_cursor": 0,
                       "phase_attempts": [], "total_iterations": 0}

    def write_checkpoint(payload: Mapping[str, Any]) -> None:
        if output_dir is None:
            return
        checkpoint = {"schema": f"{TASK_ID}.reference_solver_checkpoint.v1",
                      "task_id": TASK_ID,
                      "population_ids": population["train"]["proposal_ids"].clone(),
                      **copy.deepcopy(dict(payload))}
        checkpoint.setdefault("completed_solver_iterations", int(checkpoint.get("total_iterations", 0)))
        checkpoint.setdefault("total_work_observed", int(checkpoint["completed_solver_iterations"]))
        if witness_dir is not None:
            checkpoint["row_store"] = witness_dir.relative_to(output_dir).as_posix()
        checkpoint.setdefault("rng_state", _rng_snapshot())
        before_rng = _rng_snapshot()
        checkpoint_path = output_dir / "reference_solver_checkpoint.pt"
        _save_torch(checkpoint_path, checkpoint)
        _require(_rng_equal(before_rng, _rng_snapshot()),
                 "reference solver checkpoint serialization changed process RNG")
        reopened = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        same_tree, checkpoint_delta = projected._tree_max_delta(checkpoint, reopened)
        _require(same_tree and checkpoint_delta <= RECOVERY_TOLERANCE
                 and int(reopened.get("target_index", -1)) == int(checkpoint["target_index"])
                 and int(reopened.get("completed_targets", -1)) == int(checkpoint["completed_targets"])
                 and reopened.get("stage") == checkpoint["stage"]
                 and int(reopened.get("completed_solver_iterations", -1))
                 == int(checkpoint["completed_solver_iterations"]),
                 "saved continuous-phase solver checkpoint failed reload verification")
        continuous_state = reopened.get("continuous_state", {})
        active = continuous_state.get("active") or {}
        active_solver = active.get("solver_state") or {}
        generic_solver = reopened.get("solver_state", {})
        phase = (active.get("phase_rad") if active else
                 generic_solver.get("phase_rad"))
        local_iteration = int(active_solver.get("iteration", generic_solver.get("iteration", 0)))
        _write_json(output_dir / "progress.json", {
            "task_id": TASK_ID, "status": "REFERENCE_QUALIFICATION_RUNNING",
            "stage": checkpoint["stage"],
            "current_seed": "fixed_train200",
            "current_arm": "continuous_public_phase_qualification",
            "current_task_local_update": int(checkpoint.get("total_work_observed", 0)),
            "current_optimizer_step": 0,
            "completed_targets": int(checkpoint["completed_targets"]),
            "current_target_index": int(checkpoint["target_index"]),
            "current_phase_rad": phase,
            "current_phase_iteration": local_iteration,
            "checkpoint_reload_max_abs_delta": checkpoint_delta,
        })

    if output_dir is not None and resume_checkpoint is None:
        write_checkpoint(current or {"target_index": start_index, "completed_targets": start_index,
                                     "stage": "target_boundary", "phase_cursor": 0,
                                     "phase_attempts": [], "total_iterations": total_iterations})

    def measure_candidate(index: int, target: torch.Tensor, candidate: Mapping[str, Any],
                         branch: str, iterations: int, phase_v2: float,
                          diagnostics: Mapping[str, Any] | None = None,
                          production_2048: Mapping[str, torch.Tensor] | None = None,
                          models: Mapping[int, Any] = optical_models) -> dict[str, Any]:
        amplitude = candidate["amplitude"].to(device=device, dtype=torch.float32).reshape(1, 1, 200, 200)
        production, direct = {}, {}
        for grid in GRIDS:
            with torch.no_grad():
                production[grid] = (production_2048 if grid == 2048 and production_2048 is not None
                                    else projected._forward_amplitude(models[grid], amplitude))
                direct[grid] = projected._forward_target(models[grid], target)
        direct_cmos = {grid: direct[grid]["raw_cmos"] for grid in GRIDS}
        metrics = projected._error_row(production, target, direct_cmos, 0)
        passed = all(metrics[key] <= GATE_LIMIT for key in METRIC_KEYS)
        return {"index": index, "label": index // TRAIN_PER_CLASS,
                "proposal_id": int(population["train"]["proposal_ids"][index]),
                "source_index": int(population["train"]["source_indices"][index]),
                "branch": branch, "phase_rad": float(candidate["phase_rad"]),
                 "target_coefficients": target.detach().cpu().clone(),
                "v2_phase_rad": float(phase_v2),
                "solver_relative_error": float(candidate["relative_error"]),
                "solver_stop_criterion_reached": float(candidate["relative_error"]) <= 1.0e-6,
                "solver_iterations": int(iterations), "solver_diagnostics": dict(diagnostics or {}),
                "amplitude": amplitude.cpu(), "metrics": metrics,
                "all_six_metrics_passed": passed,
                "outputs": {grid: {key: value.detach().cpu() for key, value in
                                   (("coefficients", production[grid]["coefficients"]),
                                    ("raw_cmos", production[grid]["raw_cmos"]))}
                            for grid in GRIDS},
                "direct_raw_cmos": {grid: direct[grid]["raw_cmos"].detach().cpu()
                                    for grid in GRIDS}}

    def checkpoint_state(index: int, stage: str, *, completed_targets: int,
                         phase_cursor: int = 0, phase_schedule: Sequence[float] = (),
                         solver_state: Mapping[str, Any] | None = None,
                         continuous_state: Mapping[str, Any] | None = None,
                         phase_attempts: Sequence[Mapping[str, Any]] = (),
                         best_record: Mapping[str, Any] | None = None,
                         v2_phase: float = 0.0, target_iterations: int = 0,
                         total_done: int = 0,
                         completed_solver_iterations: int | None = None,
                         parallel_eight_start_cycles: int = 0) -> None:
        completed_solver_iterations = (total_iterations if completed_solver_iterations is None
                                       else completed_solver_iterations)
        write_checkpoint({"target_index": index, "completed_targets": completed_targets,
                          "stage": stage, "phase_cursor": phase_cursor,
                          "phase_schedule": list(phase_schedule),
                          "solver_state": copy.deepcopy(dict(solver_state or {})),
                          "continuous_state": copy.deepcopy(dict(continuous_state or {})),
                          "phase_attempts": copy.deepcopy(list(phase_attempts)),
                          "best_record": copy.deepcopy(dict(best_record or {})),
                          "v2_phase_rad": float(v2_phase),
                          "target_iterations": int(target_iterations),
                          "parallel_eight_start_cycles": int(parallel_eight_start_cycles),
                          "total_iterations": int(completed_solver_iterations),
                          "completed_solver_iterations": int(completed_solver_iterations),
                          "total_work_observed": int(total_done), "rng_state": _rng_snapshot()})

    def commit_record(record: dict[str, Any], phase_attempts: Sequence[Mapping[str, Any]],
                      total_done: int) -> None:
        witnesses.append(record)
        if witness_dir is not None:
            _save_torch(witness_dir / f"row_{int(record['index']):04d}.pt", record)
        if output_dir is not None:
            _write_json(output_dir / "reference_progress.json", {
                "task_id": TASK_ID, "completed_targets": len(witnesses),
                "total_training_targets": TRAIN_ROWS,
                "solver_iterations_counting_parallel_starts": total_done,
                "last_target": {key: record[key] for key in
                                ("index", "label", "proposal_id", "branch", "phase_rad",
                                 "solver_relative_error", "metrics")}})
            checkpoint_state(int(record["index"]) + 1, "target_boundary",
                             completed_targets=len(witnesses), total_done=total_done,
                             completed_solver_iterations=total_done)

    for index in range(start_index, TRAIN_ROWS):
        target = targets[index:index + 1].to(device=device, dtype=torch.complex128)
        feature = projected._normalized_features(targets[index:index + 1], inputs["scale"], device)
        with torch.no_grad():
            raw = mapper(feature)
            _, details = projected._project_double_from_raw(
                raw, target, matrix, inverse, scale, include_preclip=False)
        v2_phase = float(details["phase"][0])
        saved_current = current if (current is not None and index == int(current["target_index"])) else None
        stage = "target_start" if saved_current is None else str(saved_current["stage"])
        phase_attempts = list((saved_current or {}).get("phase_attempts", []))
        target_iterations = int((saved_current or {}).get("target_iterations", 0))
        parallel_eight_start_cycles = int((saved_current or {}).get("parallel_eight_start_cycles", 0))
        initial_record = None
        best_record = (copy.deepcopy(saved_current.get("best_record"))
                       if saved_current and saved_current.get("best_record") else None)
        phase_schedule = list((saved_current or {}).get("phase_schedule", []))
        phase_cursor = int((saved_current or {}).get("phase_cursor", 0))
        phase_state = (saved_current or {}).get("solver_state") or None
        continuous_state = ((saved_current or {}).get("continuous_state") or None)
        if stage == "target_boundary":
            stage = "target_start"

        if stage in {"target_start", "fixed_v2", "free_8_phase"}:
            if index == 4 and predecessor_failure is not None:
                prior_best = {"amplitude": predecessor_failure["amplitude"].reshape(-1).double(),
                              "phase_rad": float(predecessor_failure["phase_rad"]),
                              "relative_error": float(predecessor_failure["solver_relative_error"])}
                initial_record = measure_candidate(
                    index, target, prior_best, "predecessor_saved_failure", 0, v2_phase,
                    {"historical_solver_iterations": int(predecessor_failure.get("solver_iterations", 0)),
                     "historical_failure_reconsumed": True})
                best_record = initial_record
                phase_schedule = _fallback_phase_schedule(v2_phase, float(prior_best["phase_rad"]))
                phase_cursor = 0
                stage = "continuous_phase" if continuous_phase else "target_ready"
            else:
                if stage == "free_8_phase":
                    _require(best_record is not None,
                             "free-phase solver recovery is missing its measured fixed-phase candidate")
                    prior_best = {"amplitude": best_record["amplitude"].reshape(-1).double(),
                                  "phase_rad": float(best_record["phase_rad"]),
                                  "relative_error": float(best_record["solver_relative_error"])}
                    initial_record = best_record
                else:
                    if stage == "fixed_v2":
                        fixed_state = phase_state
                    else:
                        fixed_state = None
                    def fixed_checkpoint(state: dict[str, Any]) -> None:
                        checkpoint_state(index, "fixed_v2", completed_targets=len(witnesses),
                                         solver_state=state, v2_phase=v2_phase,
                                         total_done=total_iterations + target_iterations + int(state["iteration"]))
                    fixed = feasibility._fixed_phase_solve(
                        response, target[0], v2_phase, max_steps=5000,
                        initial_state=fixed_state, checkpoint=fixed_checkpoint,
                        checkpoint_interval=500)
                    target_iterations += int(fixed["iterations"])
                    prior_best = fixed
                    initial_record = measure_candidate(index, target, fixed, "fixed_v2_phase",
                                                       target_iterations, v2_phase)
                if not initial_record["all_six_metrics_passed"]:
                    free_state = phase_state if stage == "free_8_phase" else None
                    def free_checkpoint(state: dict[str, Any]) -> None:
                        checkpoint_state(index, "free_8_phase", completed_targets=len(witnesses),
                                         solver_state=state, best_record=initial_record,
                                         v2_phase=v2_phase, target_iterations=target_iterations,
                                         total_done=total_iterations + target_iterations +
                                         int(state["iteration"]) * 8,
                                         parallel_eight_start_cycles=parallel_eight_start_cycles)
                    free = reach._bounded_phase_solve(
                        response, target[0], starts=8, max_steps=5000, stop_error=1.0e-6,
                        initial_state=free_state, checkpoint=free_checkpoint,
                        checkpoint_interval=500)
                    parallel_eight_start_cycles += int(free["iterations"])
                    target_iterations += int(free["iterations"]) * 8
                    if float(free["relative_error"]) < float(prior_best["relative_error"]):
                        prior_best = free
                    free_record = measure_candidate(index, target, free, "free_public_phase",
                                                    target_iterations, v2_phase,
                                                    {"parallel_starts": 8,
                                                     "cycles": int(free["iterations"])})
                    if free_record["solver_relative_error"] < initial_record["solver_relative_error"]:
                        initial_record = free_record
                best_record = initial_record
                phase_schedule = _fallback_phase_schedule(v2_phase, float(prior_best["phase_rad"]))
                phase_cursor = 0
                best_record = initial_record
                stage = "target_ready"

        if stage == "target_ready":
            _require(best_record is not None, "reference target lost its initial measured candidate")
            if best_record["all_six_metrics_passed"]:
                commit_record(best_record, phase_attempts, total_iterations + target_iterations)
                total_iterations += target_iterations
                current = None
                continue
            phase_schedule = phase_schedule or _fallback_phase_schedule(
                v2_phase, float(best_record["phase_rad"]))
            stage = "continuous_phase" if continuous_phase else "fallback_fista"

        if stage in {"fallback_fista", "fallback_next_phase"}:
            if not phase_schedule:
                phase_schedule = _fallback_phase_schedule(v2_phase, float(best_record["phase_rad"]))
            record = None
            for phase_index in range(phase_cursor, len(phase_schedule)):
                phase = float(phase_schedule[phase_index])
                start_state = phase_state if stage == "fallback_fista" and phase_index == phase_cursor else None

                def checkpoint(state: dict[str, Any]) -> None:
                    checkpoint_state(index, "fallback_fista", completed_targets=len(witnesses),
                                     phase_cursor=phase_index, phase_schedule=phase_schedule,
                                     solver_state=state, phase_attempts=phase_attempts,
                                     best_record=best_record, v2_phase=v2_phase,
                                     target_iterations=target_iterations,
                                      total_done=total_iterations + target_iterations + int(state["iteration"]),
                                      parallel_eight_start_cycles=parallel_eight_start_cycles)

                result = _fista_fixed_phase_solve(
                    response, target[0], phase, max_steps=5000,
                    lipschitz=continuous_lipschitz, initial_state=start_state,
                    checkpoint=checkpoint, checkpoint_interval=500)
                phase_iterations = int(result["iterations"])
                target_iterations += phase_iterations
                phase_attempts.append({key: result[key] for key in
                                       ("phase_rad", "relative_error", "relative_upper_bound",
                                        "relative_dual_lower_bound", "relative_gap", "iterations")})
                candidate_record = measure_candidate(index, target, result,
                                                     "fista_fixed_public_phase", target_iterations,
                                                     v2_phase, {"dual_bounds": {
                                                         key: result[key] for key in
                                                         ("relative_upper_bound", "relative_dual_lower_bound",
                                                          "relative_gap")},
                                                         "phase_count": len(phase_schedule),
                                                         "phase_index": phase_index})
                if (best_record is None
                        or candidate_record["solver_relative_error"] < best_record["solver_relative_error"]):
                    best_record = candidate_record
                if candidate_record["all_six_metrics_passed"]:
                    record = candidate_record
                    break
                phase_cursor = phase_index + 1
                stage = "fallback_next_phase"
                phase_state = None
                checkpoint_state(index, stage, completed_targets=len(witnesses),
                                 phase_cursor=phase_cursor, phase_schedule=phase_schedule,
                                 phase_attempts=phase_attempts, best_record=best_record,
                                 v2_phase=v2_phase, target_iterations=target_iterations,
                                 total_done=total_iterations + target_iterations,
                                 parallel_eight_start_cycles=parallel_eight_start_cycles)
            record = record or best_record
        else:
            record = best_record or initial_record

        continuous_result = None
        if (continuous_phase and stage == "continuous_phase" and record is not None
                and not record["all_six_metrics_passed"]):
            anchors = phase_schedule or _fallback_phase_schedule(
                v2_phase, float(record["phase_rad"]))
            seed_record = best_record or record
            seed_amplitude = seed_record["amplitude"].detach().to(
                device="cpu", dtype=torch.float32).reshape(-1)
            current_best_error = float(seed_record["metrics"]["coefficient_2048"])

            def evaluate_adaptive_candidate(candidate: Mapping[str, Any],
                                            models: Mapping[int, Any] = optical_models
                                            ) -> dict[str, Any]:
                command = candidate["amplitude"].to(device=device, dtype=torch.float32).reshape(
                    1, 1, 200, 200)
                with torch.no_grad():
                    production_2048 = projected._forward_amplitude(models[2048], command)
                coefficient_error = _phase_invariant_relative_error(
                    production_2048["coefficients"], target)
                result: dict[str, Any] = {"coefficient_relative_error_2048": coefficient_error,
                                          "all_six_metrics_passed": False}
                if coefficient_error <= GATE_LIMIT:
                    full_record = measure_candidate(
                        index, target, candidate, "continuous_public_phase_fista",
                        int(candidate["iterations"]), v2_phase,
                        {"continuous_phase_candidate": True,
                         "matrix_solver_relative_error": float(candidate["relative_error"])},
                        production_2048=production_2048)
                    result.update({"record": full_record,
                                   "all_six_metrics_passed": full_record["all_six_metrics_passed"]})
                return result

            def adaptive_checkpoint(saved_search: Mapping[str, Any]) -> None:
                completed_in_search = int(saved_search.get("completed_iterations", 0))
                active_search = saved_search.get("active")
                active_iteration = (0 if not active_search or not active_search.get("solver_state")
                                    else int(active_search["solver_state"].get("iteration", 0)))
                checkpoint_state(
                    index, "continuous_phase", completed_targets=len(witnesses),
                    phase_schedule=anchors, continuous_state=saved_search,
                    phase_attempts=phase_attempts, best_record=best_record,
                    v2_phase=v2_phase, target_iterations=target_iterations,
                    total_done=(total_iterations + target_iterations
                                + completed_in_search + active_iteration),
                    parallel_eight_start_cycles=parallel_eight_start_cycles)

            continuous_result = _adaptive_public_phase_search(
                response, target[0], anchors, seed_amplitude,
                evaluate_candidate=evaluate_adaptive_candidate,
                initial_state=continuous_state, checkpoint=adaptive_checkpoint,
                lipschitz=continuous_lipschitz)
            target_iterations += int(continuous_result["completed_iterations"])
            if continuous_result["status"] == "witness_found":
                record = copy.deepcopy(continuous_result["witness_record"])
            else:
                best_candidate_error = float(continuous_result.get(
                    "best_candidate_error_2048", float("inf")))
                if best_candidate_error < current_best_error:
                    best_candidate = continuous_result["best_candidate"]
                    record = continuous_result.get("best_full_record")
                    if record is None:
                        best = {"amplitude": best_candidate["amplitude"],
                                "phase_rad": best_candidate["phase_rad"],
                                "relative_error": best_candidate["solver_relative_error"]}
                        record = measure_candidate(
                            index, target, best, "continuous_public_phase_best_candidate",
                            int(best_candidate["iterations"]), v2_phase,
                            {"continuous_phase_candidate": True,
                             "matrix_solver_relative_error": best_candidate["solver_relative_error"]})
            record["phase_interval_certificate"] = {
                key: value for key, value in continuous_result.items()
                if key not in {"witness_record", "best_full_record"}}

        _require(record is not None, f"reference solver produced no candidate for target {index}")
        record["solver_budget_accounting"] = {
            "sequential_fixed_phase_iterations": max(
                0, target_iterations - parallel_eight_start_cycles * 8
                - sum(int(item["iterations"]) for item in phase_attempts)),
            "parallel_eight_start_cycles": parallel_eight_start_cycles,
            "parallel_eight_start_iterations_counted": parallel_eight_start_cycles * 8,
            "fista_single_start_iterations": sum(int(item["iterations"]) for item in phase_attempts),
            "iterations_counted_parallel_starts": target_iterations,
            "fista_phase_attempts": len(phase_attempts),
            "fallback_phase_limit": 34, "iterations_per_phase_limit": 5000,
            "continuous_phase_new_solves": (0 if continuous_result is None else
                                             int(continuous_result["completed_solves"])),
            "continuous_phase_iteration_count": (0 if continuous_result is None else
                                                 int(continuous_result["completed_iterations"])),
            "continuous_phase_solve_limit": CONTINUOUS_PHASE_SOLVE_LIMIT}
        commit_record(record, phase_attempts, total_iterations + target_iterations)
        total_iterations += target_iterations
        current = None
        if not record["all_six_metrics_passed"]:
            failure = {"index": index, "label": record["label"],
                       "proposal_id": record["proposal_id"],
                       "source_index": record["source_index"],
                       "solver_stop_criterion_reached": record["solver_stop_criterion_reached"],
                       "solver_relative_error": record["solver_relative_error"],
                       "phase_rad": record["phase_rad"], "v2_phase_rad": v2_phase,
                       "metrics": record["metrics"], "phase_attempts": phase_attempts,
                       "continuous_phase_status": (None if continuous_result is None else
                                                    continuous_result["status"]),
                       "continuous_phase_solves": (0 if continuous_result is None else
                                                    int(continuous_result["completed_solves"])),
                       "solver_budget": {"max_phases": 34, "max_steps_per_phase": 5000,
                                         "continuous_new_phases": CONTINUOUS_PHASE_SOLVE_LIMIT,
                                         "continuous_steps_per_phase": 5000}}
            failure_path = None if output_dir is None else output_dir / "training_witness_failure.json"
            if failure_path is not None:
                _write_json(failure_path, failure)
                _save_torch(output_dir / "training_witnesses.pt", {
                    "schema": f"{TASK_ID}.witnesses.v1", "task_id": TASK_ID,
                    "population_ids": population["train"]["proposal_ids"].clone(),
                    "witnesses": witnesses, "failed_indices": [failure]})
                _write_json(output_dir / "reference_qualification_summary.json", {
                    "task_id": TASK_ID, "status": "TRAINING_REFERENCE_INSUFFICIENT",
                    "checked_training_targets": index + 1, "passed_witnesses": index,
                    "remaining_unchecked_targets": TRAIN_ROWS - index - 1,
                    "science_updates": 0, "adam_calls": 0,
                    "total_solver_iterations_counting_parallel_starts": total_iterations,
                    "failed_target": failure})
                _write_json(output_dir / "progress.json", {
                    "task_id": TASK_ID, "status": "TRAINING_REFERENCE_INSUFFICIENT",
                    "current_seed": "fixed_train200",
                    "current_arm": "continuous_public_phase_qualification",
                    "current_task_local_update": total_iterations,
                    "current_optimizer_step": 0,
                    "completed_targets": index + 1, "current_target_index": index})
            raise TrainingWitnessQualificationError(
                f"fixed training target {index} lacks a full-optics six-metric witness within its fixed budget")
    _require(all(_compare_parameter_states(optical_state[str(grid)], optical_models[grid].state_dict()) == 0.0
                 for grid in GRIDS), "frozen optics changed during training-witness qualification")
    del mapper, optical_models
    _require(len(witnesses) == TRAIN_ROWS and all(row["all_six_metrics_passed"] for row in witnesses),
             "not every fixed training target has a full-optics witness")
    summary = {"task_id": TASK_ID, "status": "ALL_200_TRAINING_WITNESSES_PASSED",
               "count": len(witnesses), "all_six_metrics_qualified": True,
               "checked_training_targets": len(witnesses),
               "passed_witnesses": len(witnesses), "remaining_unchecked_targets": 0,
               "qualification_criterion": "all_six_full_optics_metrics_at_or_below_1e-4",
               "solver_1e-6_is_early_stop_only": True,
                "used_as_training_labels": False,
                "solver_iterations_counting_parallel_starts": total_iterations,
               "six_metric_population_passed": len(witnesses)}
    if output_dir is not None:
        _save_torch(output_dir / "training_witnesses.pt", {
            "schema": f"{TASK_ID}.witnesses.v1", "task_id": TASK_ID,
            "population_ids": population["train"]["proposal_ids"].clone(),
            "witnesses": witnesses, "failed_indices": []})
        _write_json(output_dir / "reference_qualification_summary.json", summary)
        _write_json(output_dir / "progress.json", {
            "task_id": TASK_ID, "status": "ALL_200_TRAINING_WITNESSES_PASSED",
            "current_seed": "fixed_train200",
            "current_arm": "continuous_public_phase_qualification",
            "current_task_local_update": total_iterations,
            "current_optimizer_step": 0,
            "completed_targets": TRAIN_ROWS, "current_target_index": TRAIN_ROWS})
    return {**summary, "solver_iterations_total": total_iterations}


def _consume_training_witnesses(reference_dir: Path, inputs: Mapping[str, Any],
                                population: Mapping[str, Any], *,
                                device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_path = reference_dir / "reference_qualification_summary.json"
    evidence_path = reference_dir / "training_witnesses.pt"
    _require(summary_path.is_file() and evidence_path.is_file(),
             "complete training reference qualification evidence is required", FileNotFoundError)
    summary = _json(summary_path)
    saved = torch.load(evidence_path, map_location="cpu", weights_only=False)
    expected_ids = population["train"]["proposal_ids"]
    _require(summary.get("status") == "ALL_200_TRAINING_WITNESSES_PASSED"
             and int(summary.get("count", -1)) == TRAIN_ROWS
             and bool(summary.get("all_six_metrics_qualified"))
             and bool(summary.get("solver_1e-6_is_early_stop_only"))
             and saved.get("schema") == f"{TASK_ID}.witnesses.v1"
             and saved.get("task_id") == TASK_ID
             and torch.equal(saved.get("population_ids"), expected_ids)
             and len(saved.get("witnesses", [])) == TRAIN_ROWS
             and saved.get("failed_indices") == [],
             "reference qualification summary or population is incomplete")
    response = inputs["response_matrix"].to(device=device, dtype=torch.complex128)
    optical = {grid: _make_runtime_model(inputs, grid, device) for grid in GRIDS}
    optical_before = _state_snapshot(optical)

    def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
        dtype = (torch.complex128 if actual.is_complex() or expected.is_complex()
                 else torch.float64)
        actual = actual.to(device=device, dtype=dtype)
        expected = expected.to(device=device, dtype=dtype)
        return float(torch.linalg.vector_norm(actual - expected)
                     / torch.linalg.vector_norm(expected).clamp_min(1.0e-30))

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(saved["witnesses"]):
        target = population["train"]["coefficients"][index:index + 1]
        _require(int(row.get("index", -1)) == index
                 and int(row.get("label", -1)) == int(population["train"]["labels"][index])
                 and int(row.get("proposal_id", -1)) == int(expected_ids[index])
                 and int(row.get("source_index", -1)) == int(population["train"]["source_indices"][index]),
                 f"reference target identity differs at row {index}")
        command = row["amplitude"]
        _require(torch.is_tensor(command) and tuple(command.shape) == (1, 1, 200, 200)
                 and torch.isfinite(command).all().item()
                 and float(command.min()) >= 0.0 and float(command.max()) <= 1.0,
                 f"reference SLM command is malformed at row {index}")
        command = command.to(device=device, dtype=torch.float32)
        target_device = target.to(device=device, dtype=torch.complex128)
        with torch.no_grad():
            actual = {grid: projected._forward_amplitude(optical[grid], command)
                      for grid in GRIDS}
            direct = {grid: projected._forward_target(optical[grid], target_device)
                      for grid in GRIDS}
        predicted = _forward_matrix(command, response).reshape(-1)
        actual_coefficients = actual[2048]["coefficients"].reshape(-1).to(torch.complex128)
        response_error = float(torch.linalg.vector_norm(predicted - actual_coefficients)
                               / torch.linalg.vector_norm(actual_coefficients).clamp_min(1.0e-30))
        _require(response_error <= 1.0e-6,
                 f"reference command fails frozen-response reconsumption at row {index}")
        output_replay_errors = {}
        for grid in GRIDS:
            for key in ("coefficients", "raw_cmos"):
                error = relative_l2(actual[grid][key], row["outputs"][grid][key])
                output_replay_errors[f"{grid}_{key}"] = error
                _require(error <= RECOVERY_TOLERANCE,
                         f"saved full-optics output does not replay from its command at row {index}: "
                         f"grid={grid}, output={key}, relative_l2={error}")
            error = relative_l2(direct[grid]["raw_cmos"], row["direct_raw_cmos"][grid])
            output_replay_errors[f"{grid}_direct_raw_cmos"] = error
            _require(error <= RECOVERY_TOLERANCE,
                     f"saved direct-Decoder CMOS reference does not replay from its target at row {index}: "
                     f"grid={grid}, relative_l2={error}")
        metrics = projected._error_row(actual, target_device,
                                       {grid: direct[grid]["raw_cmos"] for grid in GRIDS}, 0)
        _require(bool(row.get("all_six_metrics_passed"))
                 and all(metrics[key] <= GATE_LIMIT for key in METRIC_KEYS)
                 and all(abs(float(row["metrics"][key]) - metrics[key]) <= 1.0e-14
                         for key in METRIC_KEYS),
                 f"reference witness fails independent six-metric reconsumption at row {index}")
        certificate = row.get("phase_interval_certificate")
        certificate_result = (None if certificate is None else
                              _verify_phase_interval_certificate(
                                  response, target_device[0], certificate))
        if certificate is not None:
            _verify_phase_witness_binding(row, certificate)
        rows.append({**row, "metrics": metrics, "matrix_response_relative_error": response_error,
                     "full_optics_replay_relative_errors": output_replay_errors,
                     "phase_interval_certificate_verification": certificate_result,
                     "consumer_recomputed": True})
    _require(len(rows) == TRAIN_ROWS, "training reference consumer has an incomplete row denominator")
    _require(all(_compare_parameter_states(optical_before[str(grid)], optical[grid].state_dict()) == 0.0
                 for grid in GRIDS), "frozen optical parameters changed during reference consumption")
    return rows, summary


def _positive_control_error_rows(actual: Mapping[int, Mapping[str, torch.Tensor]],
                                 direct: Mapping[int, Mapping[str, torch.Tensor]],
                                 targets: torch.Tensor) -> list[dict[str, float]]:
    _require(targets.ndim == 2 and len(targets) == 10
             and all(len(actual[grid]["coefficients"]) == 10
                     and len(actual[grid]["raw_cmos"]) == 10
                     and len(direct[grid]["raw_cmos"]) == 10 for grid in GRIDS),
             "ten-sample positive-control batch is incomplete")
    return [projected._error_row(
        {grid: {key: value[index:index + 1] for key, value in actual[grid].items()}
         for grid in GRIDS},
        targets[index:index + 1],
        {grid: direct[grid]["raw_cmos"][index:index + 1] for grid in GRIDS}, 0)
        for index in range(10)]


def _numpy_sinusoid_minimum(value: complex, left: float, right: float) -> float:
    magnitude = float(abs(value))
    if magnitude == 0.0:
        return 0.0
    if right - left >= 2.0 * math.pi:
        return -magnitude
    angle = float(np.angle(value))
    base = math.pi - angle
    multiple = math.ceil((left - base) / (2.0 * math.pi) - 1.0e-14)
    critical = base + multiple * 2.0 * math.pi
    candidates = [float(np.real(np.exp(1j * left) * value)),
                  float(np.real(np.exp(1j * right) * value))]
    if critical <= right + 1.0e-14:
        candidates.append(-magnitude)
    return min(candidates)


def _verify_phase_interval_certificate(response: torch.Tensor, target: torch.Tensor,
                                       certificate: Mapping[str, Any]) -> dict[str, Any]:
    """Independently verify interval coverage, dual vectors, support bounds and budget."""
    _require(certificate.get("schema") == f"{TASK_ID}.continuous_phase_search.v1",
             "phase certificate schema is invalid")
    matrix = np.asarray(response.detach().cpu(), dtype=np.complex128)
    coefficients = np.asarray(target.detach().cpu(), dtype=np.complex128).reshape(-1)
    target_norm = float(np.linalg.norm(coefficients))
    _require(matrix.ndim == 2 and matrix.shape[0] == coefficients.size
             and np.isfinite(matrix.real).all() and np.isfinite(matrix.imag).all()
             and math.isfinite(target_norm) and target_norm > 0.0,
             "phase certificate source matrix or target is invalid")
    anchors = [float(value) for value in certificate.get("phase_anchors", [])]
    _require(len(anchors) >= 2 and all(math.isfinite(value) for value in anchors),
             "phase certificate anchors are incomplete")
    normalized = sorted(value % (2.0 * math.pi) for value in anchors)
    unique = []
    for value in normalized:
        if not unique or value - unique[-1] > 1.0e-12:
            unique.append(value)
    if len(unique) > 1 and unique[0] + 2.0 * math.pi - unique[-1] <= 1.0e-12:
        unique.pop()
    _require(len(unique) >= 2, "phase certificate anchors do not cover a full cycle")
    seed = np.asarray(certificate.get("seed_amplitude"), dtype=np.float32).reshape(-1)
    _require(seed.size == matrix.shape[1] and np.isfinite(seed).all()
             and float(seed.min()) >= 0.0 and float(seed.max()) <= 1.0,
             "phase certificate seed command is invalid")
    direction_rows = certificate.get("directions", [])
    attempts = certificate.get("attempts", [])
    intervals = certificate.get("intervals", [])
    _require(isinstance(direction_rows, list) and isinstance(attempts, list)
             and isinstance(intervals, list)
             and len(attempts) <= CONTINUOUS_PHASE_SOLVE_LIMIT
             and int(certificate.get("completed_solves", -1)) == len(attempts)
             and int(certificate.get("completed_iterations", -1))
             == sum(int(row.get("iterations", -1)) for row in attempts),
             "phase certificate solve accounting is inconsistent")
    verified_directions: list[np.ndarray] = []
    for index, row in enumerate(direction_rows):
        phase = float(row.get("phase_rad", float("nan")))
        command = np.asarray(row.get("command"), dtype=np.float32).reshape(-1)
        direction = np.asarray(row.get("direction"), dtype=np.complex128).reshape(-1)
        _require(math.isfinite(phase) and command.size == matrix.shape[1]
                 and direction.size == coefficients.size
                 and np.isfinite(command).all() and np.isfinite(direction.real).all()
                 and np.isfinite(direction.imag).all() and float(command.min()) >= 0.0
                 and float(command.max()) <= 1.0
                 and abs(float(np.linalg.norm(direction)) - 1.0) <= 1.0e-10,
                 f"phase certificate dual direction {index} is malformed")
        expected_command = seed if row.get("source") == "seed_command" else command
        if row.get("source") == "seed_command":
            _require(np.array_equal(command, seed),
                     "seed phase-dual direction is not bound to the seed command")
        elif row.get("source") != "candidate_command":
            raise RuntimeError("phase certificate dual source is unknown")
        desired = np.exp(1j * phase) * coefficients
        residual = desired - expected_command.astype(np.float64) @ matrix.T
        residual_norm = float(np.linalg.norm(residual))
        _require(residual_norm > 1.0e-30,
                 "phase certificate stores a direction for a zero residual")
        expected_direction = residual / residual_norm
        _require(float(np.linalg.norm(direction - expected_direction)) <= 1.0e-8,
                 f"phase certificate dual direction {index} does not match its command")
        verified_directions.append(direction)

    for index, attempt in enumerate(attempts):
        left = float(attempt.get("parent_left_rad", float("nan")))
        right = float(attempt.get("parent_right_rad", float("nan")))
        phase = float(attempt.get("phase_rad", float("nan")))
        iterations = int(attempt.get("iterations", -1))
        command = np.asarray(attempt.get("command"), dtype=np.float32).reshape(-1)
        _require(math.isfinite(left) and math.isfinite(right) and left < right
                 and math.isfinite(phase)
                 and abs(math.atan2(math.sin(phase - ((0.5 * (left + right)) % (2 * math.pi))),
                                    math.cos(phase - ((0.5 * (left + right)) % (2 * math.pi))))) <= 1.0e-12
                 and 0 < iterations <= 5000 and command.size == matrix.shape[1]
                 and np.isfinite(command).all() and float(command.min()) >= 0.0
                 and float(command.max()) <= 1.0
                 and math.isfinite(float(attempt.get("coefficient_relative_error_2048", float("nan"))))
                 and float(attempt["coefficient_relative_error_2048"]) >= 0.0,
                 f"phase certificate attempt {index} is malformed")
        predicted = command.astype(np.float64) @ matrix.T
        desired = np.exp(1j * phase) * coefficients
        fixed_error = float(np.linalg.norm(desired - predicted) / target_norm)
        _require(math.isfinite(float(attempt.get(
                     "fixed_phase_relative_error_from_command", float("nan"))))
                 and abs(fixed_error - float(attempt["fixed_phase_relative_error_from_command"]))
                 <= 1.0e-10,
                 f"phase certificate attempt {index} fixed-phase primal error differs from NumPy")
        predicted_phase_error = _numpy_phase_error(predicted, coefficients)
        reported_phase_error = float(attempt.get("coefficient_relative_error_2048", float("nan")))
        _require(math.isfinite(reported_phase_error)
                 and abs(predicted_phase_error - reported_phase_error) <= 1.0e-6,
                 f"phase certificate attempt {index} coefficient error differs from its matrix command")
        passed = attempt.get("all_six_metrics_passed")
        _require(isinstance(passed, bool),
                 f"phase certificate attempt {index} lacks its six-metric decision")
        attempt_metrics = attempt.get("metrics")
        if attempt_metrics is not None:
            _require(set(attempt_metrics) == set(METRIC_KEYS)
                     and all(math.isfinite(float(attempt_metrics[key]))
                             and float(attempt_metrics[key]) >= 0.0
                             for key in METRIC_KEYS),
                     f"phase certificate attempt {index} has malformed full-optics metrics")
            _require(passed == all(float(attempt_metrics[key]) <= GATE_LIMIT
                                    for key in METRIC_KEYS),
                     f"phase certificate attempt {index} decision differs from its full-optics metrics")
        else:
            _require(not passed,
                     f"phase certificate attempt {index} claims a pass without full-optics metrics")
        direction_index = attempt.get("direction_index")
        if direction_index is not None:
            direction_index = int(direction_index)
            _require(0 <= direction_index < len(direction_rows)
                     and direction_rows[direction_index].get("source") == "candidate_command"
                     and np.array_equal(np.asarray(direction_rows[direction_index]["command"],
                                                   dtype=np.float32).reshape(-1), command),
                     f"phase certificate attempt {index} is not bound to its dual direction")
            direction = verified_directions[direction_index]
            support = float(np.maximum(np.real(np.conjugate(direction) @ matrix), 0.0).sum())
            fixed_lower = max(0.0, float(np.real(np.vdot(direction, desired)) - support)
                              / target_norm)
            _require(fixed_lower <= fixed_error + DUAL_BOUND_NUMERIC_TOLERANCE,
                     f"phase certificate attempt {index} dual lower bound exceeds its primal error")
        else:
            _require(fixed_error <= 1.0e-14,
                     f"phase certificate attempt {index} omits a nonzero residual direction")

    first = unique[0]
    ordered = sorted(intervals, key=lambda row: float(row.get("left_rad", float("nan"))))
    _require(ordered and abs(float(ordered[0]["left_rad"]) - first) <= 1.0e-12,
             "phase certificate interval coverage starts at the wrong point")
    expected_end = first + 2.0 * math.pi
    for index, interval in enumerate(ordered):
        left = float(interval.get("left_rad", float("nan")))
        right = float(interval.get("right_rad", float("nan")))
        _require(math.isfinite(left) and math.isfinite(right) and left < right
                 and abs(left - (float(ordered[index - 1]["right_rad"]) if index else first)) <= 1.0e-12,
                 "phase certificate interval coverage has a gap or overlap")
        if index + 1 < len(ordered):
            _require(abs(right - float(ordered[index + 1]["left_rad"])) <= 1.0e-12,
                     "phase certificate interval coverage has a gap or overlap")
        bound_values = []
        for direction in verified_directions:
            signal = np.vdot(direction, coefficients)
            support = float(np.maximum(np.real(np.conjugate(direction) @ matrix), 0.0).sum())
            raw = (_numpy_sinusoid_minimum(complex(signal), left, right) - support) / target_norm
            bound_values.append(max(0.0, raw - CONTINUOUS_PHASE_NUMERIC_GUARD))
        independent_lower = max(bound_values, default=0.0)
        saved_lower = float(interval.get("lower_bound", float("nan")))
        _require(math.isfinite(saved_lower) and abs(saved_lower - independent_lower) <= 1.0e-10,
                 f"phase interval {index} lower bound differs from independent NumPy recomputation")
        expected_status = ("excluded" if independent_lower > GATE_LIMIT else
                           "unresolved_resolution" if right - left <= CONTINUOUS_PHASE_INTERVAL_MIN_WIDTH
                           else "pending")
        _require(interval.get("status") == expected_status,
                 f"phase interval {index} status is inconsistent with its verified lower bound")
    _require(abs(float(ordered[-1]["right_rad"]) - expected_end) <= 1.0e-12,
             "phase certificate interval coverage does not close the periodic seam")
    status = str(certificate.get("status"))
    if status == "whole_phase_circle_excluded":
        _require(all(row["status"] == "excluded" for row in ordered),
                 "whole-circle exclusion lacks complete interval coverage")
    elif status == "phase_interval_unresolved":
        _require(not any(row["status"] == "pending" for row in ordered)
                 and any(row["status"] == "unresolved_resolution" for row in ordered),
                 "unresolved phase status does not match its remaining intervals")
    elif status == "new_phase_budget_exhausted":
        _require(len(attempts) == CONTINUOUS_PHASE_SOLVE_LIMIT,
                 "continuous phase budget-exhausted status has the wrong denominator")
    elif status == "witness_found":
        _require(sum(bool(row.get("all_six_metrics_passed")) for row in attempts) == 1,
                 "phase witness status lacks an accepted candidate")
    else:
        raise RuntimeError("phase certificate has an unknown terminal status")
    return {"status": status, "phase_intervals": len(ordered),
            "dual_directions": len(verified_directions), "new_phase_solves": len(attempts),
            "independent_numpy_bounds_verified": True,
            "periodic_interval_coverage_verified": True}


def _verify_phase_witness_binding(row: Mapping[str, Any],
                                  certificate: Mapping[str, Any]) -> None:
    """Bind a continuous-phase pass to the exact measured command and six outputs."""
    passing_attempts = [attempt for attempt in certificate.get("attempts", [])
                        if bool(attempt.get("all_six_metrics_passed"))]
    row_passed = bool(row.get("all_six_metrics_passed"))
    if row_passed:
        _require(certificate.get("status") == "witness_found" and len(passing_attempts) == 1,
                 "passing witness row is not backed by exactly one accepted continuous-phase attempt")
        attempt = passing_attempts[0]
        measured_command = np.asarray(row.get("amplitude"), dtype=np.float32).reshape(-1)
        attempted_command = np.asarray(attempt.get("command"), dtype=np.float32).reshape(-1)
        phase_delta = abs(math.atan2(math.sin(float(row.get("phase_rad", float("nan")))
                                     - float(attempt.get("phase_rad", float("nan")))),
                                    math.cos(float(row.get("phase_rad", float("nan")))
                                             - float(attempt.get("phase_rad", float("nan"))))))
        _require(measured_command.shape == attempted_command.shape
                 and np.array_equal(measured_command, attempted_command)
                 and phase_delta <= 1.0e-12
                 and set(attempt.get("metrics", {})) == set(METRIC_KEYS)
                 and all(abs(float(attempt["metrics"][key]) - float(row["metrics"][key])) <= 1.0e-14
                         for key in METRIC_KEYS),
                 "continuous-phase passing attempt is not bound to the saved measured instruction and outputs")
    else:
        _require(certificate.get("status") != "witness_found" and not passing_attempts,
                 "failed witness row conflicts with a passing continuous-phase attempt")


def _numpy_phase_error(actual: Any, target: Any) -> float:
    actual_array = np.asarray(actual, dtype=np.complex128).reshape(-1)
    target_array = np.asarray(target, dtype=np.complex128).reshape(-1)
    denominator = float(np.linalg.norm(target_array))
    _require(actual_array.shape == target_array.shape and denominator > 0.0
             and np.isfinite(actual_array.real).all() and np.isfinite(actual_array.imag).all()
             and np.isfinite(target_array.real).all() and np.isfinite(target_array.imag).all(),
             "independent NumPy complex metric input is invalid")
    phase = float(np.angle(np.vdot(target_array, actual_array)))
    return float(np.linalg.norm(actual_array - np.exp(1j * phase) * target_array) / denominator)


def _numpy_raw_error(actual: Any, target: Any) -> float:
    actual_array = np.asarray(actual, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    denominator = float(np.linalg.norm(target_array.reshape(-1)))
    _require(actual_array.shape == target_array.shape and denominator > 0.0
             and np.isfinite(actual_array).all() and np.isfinite(target_array).all()
             and float(actual_array.min()) >= 0.0 and float(target_array.min()) >= 0.0,
             "independent NumPy raw CMOS metric input is invalid")
    return float(np.linalg.norm((actual_array - target_array).reshape(-1)) / denominator)


def _numpy_error_row(outputs: Mapping[int, Mapping[str, Any]], target: Any,
                     direct_raw: Mapping[int, Any]) -> dict[str, float]:
    return {
        "coefficient_2048": _numpy_phase_error(outputs[2048]["coefficients"], target),
        "coefficient_4096": _numpy_phase_error(outputs[4096]["coefficients"], target),
        "native_cmos_2048": _numpy_raw_error(outputs[2048]["raw_cmos"], direct_raw[2048]),
        "native_cmos_4096": _numpy_raw_error(outputs[4096]["raw_cmos"], direct_raw[4096]),
        "cross_grid_coefficient": _numpy_phase_error(
            outputs[2048]["coefficients"], outputs[4096]["coefficients"]),
        "cross_grid_native_cmos": _numpy_raw_error(
            outputs[2048]["raw_cmos"], outputs[4096]["raw_cmos"]),
    }


def _verify_positive_control_revisions(reference_dir: Path, inputs: Mapping[str, Any]) -> int:
    root = reference_dir / "engineering_positive_controls"
    revisions = sorted(path for path in root.glob("revision_*") if path.is_dir())
    _require(revisions, "continuous phase qualification lacks the real positive-control replay",
             FileNotFoundError)
    source = inputs["positive_control_witness"]
    indices = torch.arange(0, 20, 2)
    expected_targets = source["target_coefficients"][indices]
    expected_commands = source["amplitudes"][indices]
    response = inputs["response_matrix"].to(torch.complex128).cpu().numpy()
    for revision in revisions:
        payload_path = revision / "positive_control_replay.pt"
        summary_path = revision / "positive_control_summary.json"
        _require(payload_path.is_file() and summary_path.is_file(),
                 f"positive-control revision is incomplete: {revision}", FileNotFoundError)
        summary = _json(summary_path)
        payload = torch.load(payload_path, map_location="cpu", weights_only=False)
        _require(summary.get("task_id") == TASK_ID
                 and summary.get("status") == "TEN_POSITIVE_CONTROLS_PASSED"
                 and summary.get("count") == 10
                 and summary.get("scientific_adam_updates") == 0
                 and summary.get("engineering_adam_calls") == 0
                 and summary.get("optics_and_evaluators_frozen") is True
                 and payload.get("schema") == f"{TASK_ID}.positive_control_replay.v1"
                 and payload.get("task_id") == TASK_ID
                 and torch.equal(payload.get("source_indices"), indices)
                 and torch.equal(payload.get("target_coefficients"), expected_targets)
                 and torch.equal(payload.get("amplitudes"), expected_commands),
                 f"positive-control revision identity differs from frozen source: {revision}")
        _require(summary.get("six_metric_rows") == payload.get("metrics"),
                 f"positive-control revision summary differs from saved production outputs: {revision}")
        for index in range(10):
            command = payload["amplitudes"][index].numpy().astype(np.float64).reshape(-1)
            actual = np.asarray(payload["outputs"][2048]["coefficients"][index],
                                dtype=np.complex128).reshape(-1)
            predicted = command @ response.T
            response_error = float(np.linalg.norm(predicted - actual)
                                   / max(float(np.linalg.norm(actual)), 1.0e-30))
            _require(response_error <= 1.0e-6,
                     f"positive-control matrix response differs at sample {index}")
            row_outputs = {grid: {key: payload["outputs"][grid][key][index]
                                  for key in ("coefficients", "raw_cmos")}
                           for grid in GRIDS}
            row_direct = {grid: payload["direct_raw_cmos"][grid][index] for grid in GRIDS}
            metric = _numpy_error_row(row_outputs, payload["target_coefficients"][index], row_direct)
            _require(all(metric[key] <= GATE_LIMIT for key in METRIC_KEYS)
                     and all(abs(metric[key] - float(payload["metrics"][index][key])) <= 1.0e-14
                             for key in METRIC_KEYS),
                     f"positive-control six-metric row differs from independent NumPy at sample {index}")
        _require(float(payload["matrix_response_relative_error_max"]) <= 1.0e-6
                 and float(summary["response_relative_error_max"]) <= 1.0e-6,
                 "positive-control response summary does not satisfy its production tolerance")
    return len(revisions)


def _verify_predecessor_replay_revisions(reference_dir: Path, inputs: Mapping[str, Any],
                                         population: Mapping[str, Any]) -> int:
    paths = sorted(reference_dir.glob("predecessor_witness_replay_revision_*.pt"))
    _require(paths, "qualification lacks the CUDA replay of its four predecessor positives and failed fifth row",
             FileNotFoundError)
    response = inputs["response_matrix"].to(torch.complex128).cpu().numpy()
    source = population["train"]
    predecessor_source = inputs["prior_training_witnesses"]
    _require(len(predecessor_source.get("witnesses", [])) == 5
             and int(predecessor_source["failed_indices"][0].get("index", -1)) == 4,
             "loaded predecessor witness source has the wrong fixed stop row")
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        positive_rows = payload.get("positive_rows", [])
        failure_row = payload.get("failed_row", {})
        rows = [*positive_rows, failure_row]
        expected_ids = source["proposal_ids"][:5]
        _require(payload.get("schema") == f"{TASK_ID}.predecessor_replay.v1"
                 and payload.get("task_id") == TASK_ID
                 and torch.equal(payload.get("population_ids"), expected_ids)
                 and payload.get("full_optics_replayed") is True
                 and payload.get("scientific_updates") == 0
                 and len(positive_rows) == 4 and len(rows) == 5,
                 f"predecessor replay identity or denominator is invalid: {path}")
        for index, row in enumerate(rows):
            expected_target = source["coefficients"][index:index + 1]
            expected_command = predecessor_source["witnesses"][index]["amplitude"].numpy().astype(
                np.float32)
            expected_pass = index < 4
            target = np.asarray(row.get("target_coefficients"), dtype=np.complex128)
            command = np.asarray(row.get("amplitude"), dtype=np.float32)
            _require(int(row.get("index", -1)) == index
                     and int(row.get("label", -1)) == int(source["labels"][index])
                     and int(row.get("proposal_id", -1)) == int(source["proposal_ids"][index])
                     and int(row.get("source_index", -1)) == int(source["source_indices"][index])
                     and np.array_equal(target, expected_target.numpy())
                     and np.array_equal(command, expected_command)
                     and command.shape == (1, 1, 200, 200) and np.isfinite(command).all()
                     and float(command.min()) >= 0.0 and float(command.max()) <= 1.0
                     and bool(row.get("all_six_metrics_passed")) is expected_pass,
                     f"predecessor replay row identity or command is invalid at index {index}")
            actual_coefficients = np.asarray(row["outputs"][2048]["coefficients"],
                                             dtype=np.complex128).reshape(-1)
            predicted = command.reshape(-1).astype(np.float64) @ response.T
            response_error = float(np.linalg.norm(predicted - actual_coefficients)
                                   / max(float(np.linalg.norm(actual_coefficients)), 1.0e-30))
            replay_errors = row.get("full_optics_replay_relative_errors", {})
            _require(response_error <= 1.0e-6
                     and float(row.get("matrix_response_relative_error", float("inf")))
                     <= 1.0e-6
                     and set(replay_errors) == {
                         f"{grid}_{key}" for grid in GRIDS for key in
                         ("coefficients", "raw_cmos", "direct_raw_cmos")}
                     and all(math.isfinite(float(value))
                             and float(value) <= RECOVERY_TOLERANCE
                             for value in replay_errors.values()),
                     f"predecessor replay response/output binding is invalid at index {index}")
            metrics = _numpy_error_row(row["outputs"], expected_target[0],
                                       row["direct_raw_cmos"])
            _require(set(row.get("metrics", {})) == set(METRIC_KEYS)
                     and all(abs(metrics[key] - float(row["metrics"][key])) <= 1.0e-14
                             for key in METRIC_KEYS)
                     and all(metrics[key] <= GATE_LIMIT for key in METRIC_KEYS) is expected_pass,
                     f"predecessor replay metrics differ from independent NumPy at index {index}")
    return len(paths)


def _verify_phase_solver_recovery(reference_dir: Path,
                                  population: Mapping[str, Any],
                                  response: torch.Tensor) -> dict[str, Any]:
    summary_path = reference_dir / "phase_solver_recovery_summary.json"
    recovery_dirs = sorted(path for path in reference_dir.glob("phase_solver_recovery_[0-9][0-9]")
                           if path.is_dir())
    _require(summary_path.is_file() and len(recovery_dirs) == 1,
             "qualification requires one complete saved phase-solver recovery", FileNotFoundError)
    recovery_dir = recovery_dirs[0]
    summary = _json(summary_path)
    files = {
        "checkpoint": recovery_dir / "checkpoint_step_000003.pt",
        "direct": recovery_dir / "direct_step_000004.pt",
        "independent": recovery_dir / "independent_step_000004.pt",
    }
    _require(all(path.is_file() for path in files.values()),
             "phase-solver recovery raw files are incomplete", FileNotFoundError)
    checkpoint = torch.load(files["checkpoint"], map_location="cpu", weights_only=False)
    direct = torch.load(files["direct"], map_location="cpu", weights_only=False)
    independent = torch.load(files["independent"], map_location="cpu", weights_only=False)
    target = population["train"]["coefficients"][4].to(torch.complex128).cpu()
    proposal_id = int(population["train"]["proposal_ids"][4])
    population_ids = population["train"]["proposal_ids"].cpu()
    phase = float(checkpoint.get("phase_rad", float("nan")))
    state3 = checkpoint.get("solver_state")
    rng_state = checkpoint.get("rng_state")
    _require(checkpoint.get("schema") == f"{TASK_ID}.phase_solver_recovery.v1"
             and checkpoint.get("task_id") == TASK_ID
             and int(checkpoint.get("target_index", -1)) == 4
             and int(checkpoint.get("proposal_id", -1)) == proposal_id
             and torch.equal(checkpoint.get("target_coefficients"), target)
             and torch.equal(checkpoint.get("population_ids"), population_ids)
             and isinstance(state3, Mapping) and int(state3.get("iteration", -1)) == 3
             and float(state3.get("phase_rad", float("nan"))) == phase
             and math.isfinite(phase)
             and math.isfinite(float(checkpoint.get("lipschitz", float("nan"))))
             and float(checkpoint["lipschitz"]) > 0.0
             and isinstance(rng_state, Mapping) and torch.is_tensor(rng_state.get("cpu"))
             and ("cuda" not in rng_state or isinstance(rng_state["cuda"], list)
                  and all(torch.is_tensor(item) for item in rng_state["cuda"])),
             "phase-solver recovery checkpoint is not bound to fixed target five and iteration three")

    def check_step4(payload: Mapping[str, Any], role: str) -> None:
        expected_cursor = {"target_index": 4, "phase_rad": phase, "iteration": 4}
        _require(payload.get("schema") == f"{TASK_ID}.phase_solver_recovery_step4.v1"
                 and int(payload.get("target_index", -1)) == 4
                 and int(payload.get("proposal_id", -1)) == proposal_id
                 and float(payload.get("phase_rad", float("nan"))) == phase
                 and payload.get("cursor") == expected_cursor
                 and isinstance(payload.get("solver_state"), Mapping)
                 and int(payload["solver_state"].get("iteration", -1)) == 4
                 and float(payload["solver_state"].get("phase_rad", float("nan"))) == phase,
                 f"saved {role} phase-solver fourth-step cursor is invalid")

    check_step4(direct, "direct")
    check_step4(independent, "independent")
    rng_before = _rng_snapshot()
    replay_states = []
    replay = _fista_fixed_phase_solve(
        response.to(dtype=torch.complex128, device="cpu"), target, phase, max_steps=4,
        lipschitz=float(checkpoint["lipschitz"]), initial_state=state3,
        checkpoint=replay_states.append, checkpoint_interval=4)
    _require(replay_states and _rng_equal(rng_before, _rng_snapshot()),
             "independent recovery consumption changed RNG or omitted its fourth-step state")
    for role, payload in (("direct", direct), ("independent", independent)):
        same_state, state_delta = projected._tree_max_delta(replay_states[-1], payload["solver_state"])
        _require(same_state and state_delta <= 1.0e-12,
                 f"saved {role} fourth-step state differs from recomputed FISTA recovery")
        for key in ("relative_error", "relative_upper_bound", "relative_dual_lower_bound", "relative_gap"):
            _require(abs(float(payload.get(key, float("nan"))) - float(replay[key])) <= 1.0e-12,
                     f"saved {role} fourth-step {key} differs from recomputed recovery")
    same_direct_independent, direct_independent_delta = projected._tree_max_delta(
        direct["solver_state"], independent["solver_state"])
    _require(summary.get("task_id") == TASK_ID
             and summary.get("status") == "PHASE_SOLVER_RECOVERY_PASSED"
             and int(summary.get("target_index", -1)) == 4
             and int(summary.get("proposal_id", -1)) == proposal_id
             and int(summary.get("checkpoint_iteration", -1)) == 3
             and int(summary.get("direct_iteration", -1)) == 4
             and int(summary.get("independent_iteration", -1)) == 4
             and int(summary.get("solver_calls_executed", -1)) == 5
             and int(summary.get("adam_calls_executed", -1)) == 0
             and int(summary.get("predecessor_positive_rows_reconsumed", -1)) == 4
             and summary.get("cursor_exact") is True
             and summary.get("rng_unchanged_during_save") is True
             and same_direct_independent and direct_independent_delta <= 1.0e-12
             and abs(float(summary.get("direct_vs_independent_state_max_abs_delta", float("nan")))
                     - direct_independent_delta) <= 1.0e-12
             and abs(float(summary.get("direct_vs_independent_loss_abs_delta", float("nan"))))
             <= 1.0e-12,
             "phase-solver recovery summary disagrees with consumed raw fourth-step evidence")
    return {"status": "PHASE_SOLVER_RECOVERY_VERIFIED", "target_index": 4,
            "third_to_fourth_step_recomputed": True,
            "direct_and_independent_state_max_abs_delta": direct_independent_delta,
            "rng_unchanged_during_consumer_replay": True,
            "solver_calls_in_original_preflight": 5, "adam_calls": 0}


def _verify_qualification_transfer(evidence_dir: Path, reference_dir: Path) -> dict[str, Any]:
    evidence_dir, reference_dir = evidence_dir.resolve(), reference_dir.resolve()
    _require(evidence_dir == reference_dir,
             "qualification must be consumed from the exact transferred evidence directory", ValueError)
    receipt_path = evidence_dir / "transfer_receipt.json"
    _require(receipt_path.is_file(), "qualification evidence transfer receipt is missing", FileNotFoundError)
    receipt = _json(receipt_path)
    relative_runs = RUNS_DIR.resolve().relative_to(ROOT.resolve())
    relative_parts = list(reference_dir.relative_to(RUNS_DIR.resolve()).parts)
    _require(relative_parts and relative_parts[-1].startswith("autodl_backup_"),
             "qualification evidence is not in its explicit AutoDL backup directory", ValueError)
    relative_parts[-1] = relative_parts[-1][len("autodl_backup_"):]
    expected_remote = ("/root/autodl-tmp/codex-original/"
                       + Path(relative_runs, *relative_parts).as_posix())
    _require(receipt.get("status") == "COMPLETE"
             and receipt.get("remote_alias") == "autodl-monthly"
             and str(receipt.get("remote_run_dir", "")).rstrip("/") == expected_remote
             and Path(str(receipt.get("destination", ""))).resolve() == evidence_dir,
             "qualification transfer receipt does not name the explicit AutoDL run and local backup")
    files = receipt.get("files")
    _require(isinstance(files, list), "qualification transfer receipt lacks file rows")
    received: dict[str, Mapping[str, Any]] = {}
    for row in files:
        relative = Path(str(row.get("path", "")))
        name = relative.as_posix()
        _require(not relative.is_absolute() and ".." not in relative.parts and name not in received,
                 "qualification transfer receipt contains unsafe or duplicate paths")
        path = evidence_dir / relative
        actual_size = path.stat().st_size if path.is_file() else -1
        _require(row.get("status") == "COPIED" and int(row.get("scp_exit_code", -1)) == 0
                 and int(row.get("remote_size_bytes", -1)) == actual_size
                 and int(row.get("local_size_bytes", -1)) == actual_size,
                 f"qualification transfer size or status does not match current local file: {name}")
        received[name] = row
    required = {
        "reference_qualification_summary.json", "training_witnesses.pt",
        "training_witness_failure.json", "reference_qualification_failure.json",
        "reference_solver_checkpoint.pt", "reference_progress.json", "progress.json",
        "resource_telemetry.csv", "predecessor_witness_replay_revision_01.pt",
        "phase_solver_recovery_summary.json",
        "phase_solver_recovery_01/checkpoint_step_000003.pt",
        "phase_solver_recovery_01/direct_step_000004.pt",
        "phase_solver_recovery_01/independent_step_000004.pt",
        "phase_solver_recovery_01/phase_solver_recovery_summary.json",
        "phase_solver_recovery_01/run_config.yaml",
        "engineering_positive_controls/revision_01/positive_control_replay.pt",
        "engineering_positive_controls/revision_01/positive_control_summary.json",
        *(f"witness_rows/revision_0001/row_{index:04d}.pt" for index in range(5)),
    }
    _require(set(received) == required,
             "qualification transfer receipt does not match the exact required evidence set")
    return {"status": "QUALIFICATION_TRANSFER_VERIFIED", "files_verified": len(received),
            "bytes_verified": sum(int(row["local_size_bytes"]) for row in received.values()),
            "remote_and_local_sizes_match": True}


def _verify_saved_qualification(reference_dir: Path, inputs: Mapping[str, Any],
                                population: Mapping[str, Any], output_dir: Path,
                                transfer_dir: Path) -> dict[str, Any]:
    reference_dir = reference_dir.resolve()
    output_dir = output_dir.resolve()
    _require(reference_dir.is_dir() and reference_dir.is_relative_to(RUNS_DIR.resolve())
             and output_dir.is_relative_to(RUNS_DIR.resolve())
             and not output_dir.exists(),
             "qualification verification requires explicit existing evidence and a new output directory",
             ValueError)
    summary_path = reference_dir / "reference_qualification_summary.json"
    witness_path = reference_dir / "training_witnesses.pt"
    _require(summary_path.is_file() and witness_path.is_file(),
             "saved reference qualification summary and witness population are required",
             FileNotFoundError)
    summary = _json(summary_path)
    saved = torch.load(witness_path, map_location="cpu", weights_only=False)
    expected_ids = population["train"]["proposal_ids"]
    rows = saved.get("witnesses", [])
    failed = saved.get("failed_indices", [])
    _require(saved.get("schema") == f"{TASK_ID}.witnesses.v1"
             and saved.get("task_id") == TASK_ID
             and torch.equal(saved.get("population_ids"), expected_ids)
             and isinstance(rows, list) and isinstance(failed, list)
             and len(rows) <= TRAIN_ROWS,
             "saved qualification population identity or denominator is invalid")
    response = inputs["response_matrix"].to(torch.complex128).cpu().numpy()
    recovery_result = _verify_phase_solver_recovery(
        reference_dir, population, torch.as_tensor(response, dtype=torch.complex128))
    transfer_result = _verify_qualification_transfer(transfer_dir, reference_dir)
    positive_control_revisions = _verify_positive_control_revisions(reference_dir, inputs)
    predecessor_replay_revisions = _verify_predecessor_replay_revisions(
        reference_dir, inputs, population)
    coefficients = population["train"]["coefficients"].cpu().numpy()
    ids = population["train"]["proposal_ids"].tolist()
    labels = population["train"]["labels"].tolist()
    source_indices = population["train"]["source_indices"].tolist()
    verified_rows = []
    for index, row in enumerate(rows):
        _require(int(row.get("index", -1)) == index
                 and int(row.get("label", -1)) == int(labels[index])
                 and int(row.get("proposal_id", -1)) == int(ids[index])
                 and int(row.get("source_index", -1)) == int(source_indices[index]),
                 f"saved qualification row identity differs at index {index}")
        target = coefficients[index]
        saved_target = np.asarray(row.get("target_coefficients"), dtype=np.complex128).reshape(-1)
        _require(saved_target.shape == target.shape
                 and np.array_equal(saved_target, np.asarray(target, dtype=np.complex128).reshape(-1)),
                 f"saved qualification target coefficients differ from the fixed source at index {index}")
        command = np.asarray(row.get("amplitude"), dtype=np.float32)
        _require(command.shape == (1, 1, 200, 200) and np.isfinite(command).all()
                 and float(command.min()) >= 0.0 and float(command.max()) <= 1.0,
                 f"saved qualification command is invalid at index {index}")
        predicted = command.reshape(-1).astype(np.float64) @ response.T
        actual_2048 = np.asarray(row["outputs"][2048]["coefficients"], dtype=np.complex128).reshape(-1)
        response_error = float(np.linalg.norm(predicted - actual_2048)
                               / max(float(np.linalg.norm(actual_2048)), 1.0e-30))
        _require(response_error <= 1.0e-6,
                 f"saved qualification command fails response recheck at index {index}")
        metric_row = _numpy_error_row(row["outputs"], target,
                                      row["direct_raw_cmos"])
        _require(all(abs(float(row["metrics"][key]) - metric_row[key]) <= 1.0e-14
                     for key in METRIC_KEYS),
                 f"saved qualification metrics differ from independent NumPy at index {index}")
        passed = all(metric_row[key] <= GATE_LIMIT for key in METRIC_KEYS)
        _require(bool(row.get("all_six_metrics_passed")) == passed,
                 f"saved qualification row decision differs from raw measurements at index {index}")
        certificate = row.get("phase_interval_certificate")
        certificate_result = None
        if certificate is not None:
            certificate_result = _verify_phase_interval_certificate(
                torch.as_tensor(response, dtype=torch.complex128),
                torch.as_tensor(target, dtype=torch.complex128), certificate)
            _verify_phase_witness_binding(row, certificate)
        verified_rows.append({"index": index, "label": int(labels[index]),
                              "all_six_metrics_passed": passed,
                              "matrix_response_relative_error": response_error,
                              "metrics": metric_row,
                              "phase_certificate": certificate_result})
    _require(len(failed) <= 1,
             "saved training qualification contains more than one stopping failure")
    for failure in failed:
        failure_index = int(failure.get("index", -1))
        _require(bool(rows) and failure_index == len(rows) - 1
                 and 0 <= failure_index < TRAIN_ROWS
                 and not bool(rows[-1].get("all_six_metrics_passed")),
                 "saved failed-target record does not match the incomplete witness prefix")
        _require(int(failure.get("proposal_id", -1)) == int(ids[failure_index])
                 and int(failure.get("label", -1)) == int(labels[failure_index]),
                 "saved failed-target identity differs from the fixed population")
    complete = (len(rows) == TRAIN_ROWS and not failed
                and all(row["all_six_metrics_passed"] for row in verified_rows)
                and summary.get("status") == "ALL_200_TRAINING_WITNESSES_PASSED"
                and int(summary.get("count", -1)) == TRAIN_ROWS
                and int(summary.get("checked_training_targets", -1)) == TRAIN_ROWS
                and int(summary.get("passed_witnesses", -1)) == TRAIN_ROWS
                and int(summary.get("remaining_unchecked_targets", -1)) == 0
                and bool(summary.get("all_six_metrics_qualified")))
    incomplete = (bool(failed) and len(rows) < TRAIN_ROWS
                  and summary.get("status") == "TRAINING_REFERENCE_INSUFFICIENT"
                  and int(summary.get("checked_training_targets", -1)) == len(rows)
                  and int(summary.get("passed_witnesses", -1)) == len(rows) - 1
                  and int(summary.get("remaining_unchecked_targets", -1)) == TRAIN_ROWS - len(rows)
                  and int(failed[0].get("index", -1)) == len(rows) - 1
                  and failed[0].get("metrics") == rows[-1].get("metrics")
                  and rows[-1].get("phase_interval_certificate") is not None)
    _require(complete or incomplete,
             "saved qualification summary does not match a valid complete or insufficient population")
    report = {"task_id": TASK_ID,
              "status": "COMPLETE_EVIDENCE_VERIFIED" if complete else "REFERENCE_INSUFFICIENT_VERIFIED",
              "training_rows_verified": len(verified_rows), "training_rows_required": TRAIN_ROWS,
              "passed_rows": sum(row["all_six_metrics_passed"] for row in verified_rows),
              "training_ready": False,
              "preflight_eligible": bool(complete),
              "independent_numpy_recalculation": True,
              "phase_solver_recovery": recovery_result,
              "transfer": transfer_result,
              "positive_control_revisions_verified": positive_control_revisions,
              "predecessor_replay_revisions_verified": predecessor_replay_revisions,
              "phase_certificates_verified": sum(row["phase_certificate"] is not None
                                                  for row in verified_rows),
              "rows": verified_rows}
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "qualification_verification.json", report)
    return report


def _compare_parameter_states(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> float:
    _require(set(a) == set(b), "mapper state keys differ")
    deltas = []
    for key in a:
        left, right = a[key].detach().cpu(), b[key].detach().cpu()
        _require(left.shape == right.shape and left.dtype == right.dtype,
                 f"state tensor shape or dtype differs: {key}")
        deltas.append(float((left - right).abs().max()))
    return max(deltas)


def _engine_qualification(config: Mapping[str, Any], inputs: Mapping[str, Any],
                          population: Mapping[str, Any], schedule: torch.Tensor,
                          witnesses: Sequence[Mapping[str, Any]],
                          device: torch.device) -> dict[str, Any]:
    models = {grid: pilot._new_optical_model(inputs["checkpoint"], grid, device)
              for grid in (2048,)}
    optical_before = _state_snapshot(models)
    torch.manual_seed(config["training"]["rng_seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["training"]["rng_seed"])
    full = _mapper(inputs, device)
    torch.manual_seed(config["training"]["rng_seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["training"]["rng_seed"])
    matrix = _mapper(inputs, device)
    full_opt = _optimizer(full, config)
    matrix_opt = _optimizer(matrix, config)
    updates = []
    recovery_states: dict[str, dict[str, Any]] = {}
    branch_histories: dict[str, list[dict[str, Any]]] = {
        "full_optics": [], "response_matrix": []}
    for index in range(4):
        coefficients = population["train"]["coefficients"][schedule[index]]
        full_result = _update(full, full_opt, models[2048], inputs,
                              coefficients, device, "full_optics", audit=index == 0)
        matrix_result = _update(matrix, matrix_opt, models[2048], inputs,
                                coefficients, device, "response_matrix", audit=index == 0)
        state_delta = _compare_parameter_states(full.state_dict(), matrix.state_dict())
        _, optimizer_delta = fit20._tree_max_delta(full_opt.state_dict(), matrix_opt.state_dict())
        updates.append({"step": index + 1, "schedule_indices": schedule[index].tolist(),
            "loss_abs_delta": abs(full_result["loss"] - matrix_result["loss"]),
            "coefficient_loss_abs_delta": abs(full_result["coefficient_loss"] - matrix_result["coefficient_loss"]),
            "box_loss_abs_delta": abs(full_result["box_loss"] - matrix_result["box_loss"]),
            "mapper_max_abs_delta": state_delta, "adam_max_abs_delta": optimizer_delta})
        for engine_name, model, optimizer, result in (
                ("full_optics", full, full_opt, full_result),
                ("response_matrix", matrix, matrix_opt, matrix_result)):
            # Keep immutable per-branch loss history for the selected path's
            # uninterrupted step-four versus restored replay comparison.
            branch_histories[engine_name].append({
                "step": index + 1, "schedule_indices": schedule[index].tolist(),
                "loss": result["loss"], "coefficient_loss": result["coefficient_loss"],
                "box_loss": result["box_loss"]})
            if index == 2:
                recovery_states[f"{engine_name}_step3"] = {
                    "mapper_state": _copy_state(model.state_dict()),
                    "optimizer_state": copy.deepcopy(optimizer.state_dict()),
                    "rng_state": _rng_snapshot(), "history": copy.deepcopy(branch_histories[engine_name]),
                    "schedule_cursor": 3, "batch": schedule[3].clone(),
                }
            if index == 3:
                recovery_states[f"{engine_name}_direct_step4"] = {
                    "mapper_state": _copy_state(model.state_dict()),
                    "optimizer_state": copy.deepcopy(optimizer.state_dict()),
                    "rng_state": _rng_snapshot(), "history": copy.deepcopy(branch_histories[engine_name]),
                    "loss": result["loss"], "batch": schedule[3].clone(),
                }
    # Independent response equality includes nonuniform and boundary commands.
    response = inputs["response_matrix"].to(device=device, dtype=torch.complex128)
    probe = torch.cat((torch.zeros((1, 1, 200, 200), device=device),
                       torch.ones((1, 1, 200, 200), device=device),
                       torch.linspace(0.0, 1.0, PIXEL_COUNT, device=device).reshape(1, 1, 200, 200),
                       torch.zeros((1, 1, 200, 200), device=device)), dim=0)
    probe[3, 0, 0, 0] = 1.0
    response_errors = []
    command_batches = [probe]
    witness_commands = [item["amplitude"][0].to(device=device, dtype=torch.float32)
                        for item in witnesses]
    command_batches.extend(torch.stack(witness_commands[i:i + 10])
                           for i in range(0, len(witness_commands), 10))
    with torch.no_grad():
        for commands in command_batches:
            actual = projected._forward_amplitude(models[2048], commands)["coefficients"]
            predicted = _forward_matrix(commands, response)
            response_errors.append(float(torch.linalg.vector_norm(predicted - actual, dim=1).div(
                torch.linalg.vector_norm(actual, dim=1).clamp_min(1.0e-30)).max()))
    response_error = max(response_errors)
    _require(all(_compare_parameter_states(optical_before[str(grid)], models[grid].state_dict()) == 0.0
                 for grid in models),
             "frozen optical state changed during matrix/full-optics update qualification")
    update_equivalence = all(row["mapper_max_abs_delta"] <= RECOVERY_TOLERANCE
                             and row["adam_max_abs_delta"] <= RECOVERY_TOLERANCE
                             and row["loss_abs_delta"] <= RECOVERY_TOLERANCE
                             for row in updates)
    matrix_qualified = response_error <= 1.0e-6 and update_equivalence
    selected_engine = "response_matrix" if matrix_qualified else "full_optics"
    del full, matrix, full_opt, matrix_opt, models
    return {"response_relative_error_max": response_error, "four_step_updates": updates,
            "matrix_qualification_passed": matrix_qualified, "selected_engine": selected_engine,
            "full_optics_endpoint_required": True,
            "_recovery_states": recovery_states}


def _qualify_engine(config_path: Path, config: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    _require(output_dir.is_relative_to(RUNS_DIR.resolve()),
             "engine qualification output must stay inside this experiment's runs directory", ValueError)
    _require(not output_dir.exists(), f"engine qualification already exists: {output_dir}", FileExistsError)
    _require(torch.cuda.is_available(), "engine qualification requires the production CUDA device")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inputs = _load_inputs(config)
    population = _load_population(config, inputs)
    schedule = _build_schedule(config["training"]["schedule_seed"])
    source = inputs["positive_control_witness"]
    positive = [{"amplitude": source["amplitudes"][index:index + 1]}
                for index in range(0, 20, 2)]
    box = _box_loss_qualification(torch.device("cuda:0"))
    matrix_result = _engine_qualification(config, inputs, population, schedule, positive,
                                          torch.device("cuda:0"))
    recovery_states = matrix_result.pop("_recovery_states")
    selected_engine = str(matrix_result["selected_engine"])
    step3 = recovery_states[f"{selected_engine}_step3"]
    direct4 = recovery_states[f"{selected_engine}_direct_step4"]
    output_dir.mkdir(parents=True)
    recovery_dir = output_dir / "engineering_recovery"
    recovery_dir.mkdir()
    third = {"schema": f"{TASK_ID}.checkpoint.v1", "task_id": TASK_ID,
             "completed_updates": 3, "checkpoint_step": 3,
             "source_mapper_updates": 5000, "source_lineage_mapper_updates": 10000,
             "lineage_mapper_updates": 10003, **step3,
             "engine": selected_engine, "schedule": schedule.clone(),
             "history_count": 3, "optimizer_state_deepcopied": True}
    direct4_saved = {**direct4, "completed_updates": 4, "engine": selected_engine,
                     "schedule": schedule.clone(), "history_count": 4}
    _save_torch(recovery_dir / "checkpoint_step_000003.pt", third)
    _save_torch(recovery_dir / "direct_step_000004.pt", direct4_saved)
    replay_path = recovery_dir / "independent_step_000004.pt"
    command = [sys.executable, "-m", "scripts.research.lightgen.execute_lightgen_g1_semantic_exposure200_v1",
               "--mode", "recovery-replay", "--config", str(config_path.resolve()),
               "--checkpoint", str((recovery_dir / "checkpoint_step_000003.pt").resolve()),
               "--recovery-output", str(replay_path.resolve())]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    _require(completed.returncode == 0 and replay_path.is_file(),
             "independent process could not restore the three-step qualification checkpoint: "
             f"stdout={completed.stdout[-2000:]} stderr={completed.stderr[-4000:]}")
    replay = torch.load(replay_path, map_location="cpu", weights_only=False)
    model_delta = _compare_parameter_states(direct4_saved["mapper_state"], replay["mapper_state"])
    _, adam_delta = fit20._tree_max_delta(direct4_saved["optimizer_state"], replay["optimizer_state"])
    _require(model_delta <= RECOVERY_TOLERANCE and adam_delta <= RECOVERY_TOLERANCE
             and abs(float(direct4_saved["loss"]) - float(replay["loss"])) <= RECOVERY_TOLERANCE
             and direct4_saved["history"] == replay["history"]
             and torch.equal(replay["batch"], schedule[3])
             and _rng_equal(direct4_saved["rng_state"], replay["rng_state"]),
             "engineering fourth update differs after independent process recovery")
    recovery = {"successful_engineering_adam_calls": 9,
                "full_optics_vs_matrix_four_step_comparison": matrix_result["four_step_updates"],
                "direct_vs_restored_step4_model_max_abs_delta": model_delta,
                "direct_vs_restored_adam_max_abs_delta": adam_delta,
                "direct_vs_restored_loss_abs_delta": abs(float(direct4_saved["loss"])
                                                          - float(replay["loss"])),
                "schedule_and_history_exact": True, "independent_process": True,
                "selected_engine": selected_engine}
    _save_torch(recovery_dir / "engineering_recovery.pt", {
        "schema": f"{TASK_ID}.recovery.v1", "task_id": TASK_ID,
        "third_step_checkpoint": third, "direct_step4": direct4_saved,
        "replayed_step4": replay, "summary": recovery})
    summary = {"task_id": TASK_ID, "status": "ENGINE_QUALIFICATION_COMPLETE",
               "matrix_qualification": {key: value for key, value in matrix_result.items()},
               "recovery": recovery, "box_loss_qualification": box,
               "positive_control_count": len(positive), "positive_controls_are_training_labels": False,
               "scientific_adam_updates": 0, "engineering_adam_calls": 9,
               "training_ready": False}
    _write_json(output_dir / "engine_qualification_summary.json", summary)
    shutil.copy2(config_path, output_dir / "run_config.yaml")
    shutil.copy2(Path(__file__).resolve(), output_dir / "runner_source.py")
    return summary


def _consume_engine_qualification(engine_dir: Path, inputs: Mapping[str, Any]
                                 ) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = engine_dir / "engine_qualification_summary.json"
    recovery_path = engine_dir / "engineering_recovery" / "engineering_recovery.pt"
    _require(summary_path.is_file() and recovery_path.is_file(),
             "executed matrix/full-optics and recovery evidence is required", FileNotFoundError)
    summary = _json(summary_path)
    recovery = torch.load(recovery_path, map_location="cpu", weights_only=False)
    _require(summary.get("task_id") == TASK_ID
             and summary.get("status") == "ENGINE_QUALIFICATION_COMPLETE"
             and summary.get("scientific_adam_updates") == 0
             and summary.get("engineering_adam_calls") == 9
             and summary.get("training_ready") is False
             and recovery.get("schema") == f"{TASK_ID}.recovery.v1"
             and recovery.get("task_id") == TASK_ID,
             "engine qualification status or recovery evidence identity is invalid")
    selected = summary["matrix_qualification"]["selected_engine"]
    _require(selected in {"response_matrix", "full_optics"}
             and recovery["summary"]["selected_engine"] == selected,
             "engine qualification selected-path record is inconsistent")
    direct = recovery["direct_step4"]
    replay = recovery["replayed_step4"]
    third = recovery["third_step_checkpoint"]
    model_delta = _compare_parameter_states(direct["mapper_state"], replay["mapper_state"])
    _, adam_delta = fit20._tree_max_delta(direct["optimizer_state"], replay["optimizer_state"])
    _require(model_delta <= RECOVERY_TOLERANCE and adam_delta <= RECOVERY_TOLERANCE
             and abs(float(direct["loss"]) - float(replay["loss"])) <= RECOVERY_TOLERANCE
             and direct["history"] == replay["history"]
             and torch.equal(direct["batch"], replay["batch"])
             and torch.equal(third["schedule"], direct["schedule"])
             and torch.equal(third["batch"], third["schedule"][3])
             and torch.equal(direct["batch"], direct["schedule"][3])
             and direct["history"][:3] == third["history"]
             and torch.equal(direct["rng_state"]["cpu"], replay["rng_state"]["cpu"])
             and ("cuda" not in direct["rng_state"] or _rng_equal(direct["rng_state"], replay["rng_state"]))
             and int(recovery["third_step_checkpoint"]["completed_updates"]) == 3
             and len(recovery["third_step_checkpoint"]["history"]) == 3
             and int(direct["completed_updates"]) == 4,
             "independent engine recovery states fail direct consumer comparison")
    source_state = inputs["source_mapper"]["mapper_state"]
    _require(set(third["mapper_state"]) == set(source_state)
             and all(not torch.equal(third["mapper_state"][key], source_state[key])
                     for key in source_state),
             "three real qualification updates did not change every mapper parameter tensor")
    parameter_shapes = [tuple(value.shape) for value in source_state.values()]
    for state_name, expected_step in ((third["optimizer_state"], 3),
                                      (direct["optimizer_state"], 4),
                                      (replay["optimizer_state"], 4)):
        optimizer_states = state_name.get("state", {})
        _require(len(optimizer_states) == 4,
                 "engineering recovery Adam state does not cover all four mapper parameters")
        ordered_states = [optimizer_states[index] for index in range(4)]
        steps = [int(torch.as_tensor(value.get("step")).item()) for value in ordered_states]
        _require(steps == [expected_step] * 4,
                 "engineering recovery Adam step counts differ from the executed update count")
        _require(all(tuple(value["exp_avg"].shape) == shape
                     and tuple(value["exp_avg_sq"].shape) == shape
                     and torch.isfinite(value["exp_avg"]).all().item()
                     and torch.isfinite(value["exp_avg_sq"]).all().item()
                     for value, shape in zip(ordered_states, parameter_shapes, strict=True)),
                 "engineering recovery Adam state contains nonfinite moments")
    matrix = summary["matrix_qualification"]
    _require(len(matrix["four_step_updates"]) == 4
             and bool(summary["recovery"]["schedule_and_history_exact"]),
             "matrix/full-optics qualification is incomplete")
    if selected == "response_matrix":
        _require(bool(matrix["matrix_qualification_passed"])
                 and float(matrix["response_relative_error_max"]) <= 1.0e-6
                 and all(float(row["loss_abs_delta"]) <= RECOVERY_TOLERANCE
                         and float(row["mapper_max_abs_delta"]) <= RECOVERY_TOLERANCE
                         and float(row["adam_max_abs_delta"]) <= RECOVERY_TOLERANCE
                         for row in matrix["four_step_updates"]),
                 "matrix path selected without equivalent executed updates")
    _require(model_delta <= RECOVERY_TOLERANCE and adam_delta <= RECOVERY_TOLERANCE,
             "engine recovery direct-consumer state comparison exceeded tolerance")
    return summary, recovery


def _checkpoint_payload(mapper: torch.nn.Module, optimizer: torch.optim.Optimizer,
                        schedule: torch.Tensor, history: Sequence[Mapping[str, Any]],
                        completed: int, inputs: Mapping[str, Any], engine: str,
                        run_id: str) -> dict[str, Any]:
    return {"schema": f"{TASK_ID}.checkpoint.v1", "task_id": TASK_ID,
            "run_id": run_id,
            "completed_updates": completed, "checkpoint_step": completed,
            "source_mapper_updates": 5000, "source_lineage_mapper_updates": 10000,
            "lineage_mapper_updates": 10000 + completed,
            "mapper_state": _copy_state(mapper.state_dict()),
            "optimizer_state": copy.deepcopy(optimizer.state_dict()),
            "rng_state": _rng_snapshot(), "schedule_cursor": completed,
            "schedule": schedule.clone(), "history": copy.deepcopy(list(history)),
            "history_count": len(history), "engine": engine,
            "source_checkpoint": str(inputs["source_mapper_path"]),
            "optimizer_state_deepcopied": True}


def _save_checkpoint(path: Path, mapper: torch.nn.Module, optimizer: torch.optim.Optimizer,
                     schedule: torch.Tensor, history: Sequence[Mapping[str, Any]],
                     completed: int, inputs: Mapping[str, Any], engine: str,
                     run_id: str) -> None:
    before = _rng_snapshot()
    payload = _checkpoint_payload(mapper, optimizer, schedule, history, completed, inputs, engine,
                                  run_id)
    _require(_rng_equal(before, _rng_snapshot()), "checkpoint serialization changed process RNG")
    _save_torch(path, payload)
    reopened = torch.load(path, map_location="cpu", weights_only=False)
    _require(reopened.get("task_id") == TASK_ID and reopened.get("completed_updates") == completed
             and reopened.get("history_count") == completed
             and torch.equal(reopened.get("schedule"), schedule)
             and _compare_parameter_states(reopened["mapper_state"], mapper.state_dict()) == 0.0,
             "saved checkpoint failed reload verification")


def _restore(path: Path, mapper: torch.nn.Module, optimizer: torch.optim.Optimizer,
             schedule: torch.Tensor, engine: str) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    _require(saved.get("schema") == f"{TASK_ID}.checkpoint.v1" and saved.get("task_id") == TASK_ID
             and saved.get("engine") == engine and torch.equal(saved.get("schedule"), schedule),
             "explicit resume checkpoint does not match the fixed exposure task")
    completed = int(saved["completed_updates"])
    _require(0 <= completed < UPDATE_BUDGET and len(saved["history"]) == completed
             and int(saved["schedule_cursor"]) == completed,
             "resume checkpoint is terminal or has inconsistent history/schedule cursor")
    mapper.load_state_dict(saved["mapper_state"], strict=True)
    optimizer.load_state_dict(saved["optimizer_state"])
    if "cuda" in saved["rng_state"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(saved["rng_state"]["cuda"])
    torch.set_rng_state(saved["rng_state"]["cpu"])
    return completed, list(saved["history"]), saved


def _make_runtime_model(inputs: Mapping[str, Any], grid: int, device: torch.device) -> Any:
    model = pilot._new_optical_model(inputs["checkpoint"], grid, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _measure_cohort(inputs: Mapping[str, Any], population: Mapping[str, torch.Tensor],
                    mapper: torch.nn.Module, device: torch.device, *, save_outputs: bool,
                    tag: str) -> tuple[list[dict[str, Any]], dict[int, dict[str, torch.Tensor]]]:
    rows: list[dict[str, Any]] = []
    all_outputs = {grid: {"coefficients": [], "raw_cmos": [], "direct_raw_cmos": [], "commands": []}
                   for grid in GRIDS}
    model_by_grid = {grid: _make_runtime_model(inputs, grid, device) for grid in GRIDS}
    for start in range(0, len(population["labels"]), 10):
        stop = min(start + 10, len(population["labels"]))
        shard_path = ATTEMPT_DIR / "measurements" / tag / f"shard_{start // 10:04d}.pt"
        coeff = population["coefficients"][start:stop]
        target_device = coeff.to(device=device, dtype=torch.complex128)
        features = projected._normalized_features(coeff, inputs["scale"], device)
        command_rows, clipped_rows = [], []
        with torch.no_grad():
            for local in range(stop - start):
                one_command, detail = projected._map_amplitude(
                    mapper, features[local:local + 1], target_device[local:local + 1],
                    inputs["matrix"].to(device), inputs["inverse"].to(device),
                    inputs["scale"].to(device), include_preclip=True)
                command_rows.append(one_command)
                clipped_rows.append(detail["clipped_fraction"].reshape(-1))
            command = torch.cat(command_rows, dim=0)
            detail_fraction = torch.cat(clipped_rows).cpu()
            raw_outputs = {grid: projected._forward_amplitude(model_by_grid[grid], command)
                           for grid in GRIDS}
            outputs = {grid: {"coefficients": raw_outputs[grid]["coefficients"],
                              "raw_cmos": raw_outputs[grid]["raw_cmos"]}
                       for grid in GRIDS}
            direct = {grid: projected._forward_target(model_by_grid[grid], target_device)["raw_cmos"]
                      for grid in GRIDS}
        _require(torch.isfinite(command).all().item()
                 and all(torch.isfinite(outputs[grid]["coefficients"].real).all().item()
                         and torch.isfinite(outputs[grid]["coefficients"].imag).all().item()
                         and torch.isfinite(outputs[grid]["raw_cmos"]).all().item()
                         and torch.isfinite(direct[grid]).all().item()
                         for grid in GRIDS),
                 f"{tag} production output contains nonfinite command, coefficient, or CMOS values")
        if save_outputs and shard_path.is_file():
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            _require(shard.get("schema") == f"{TASK_ID}.measurement_shard.v1"
                     and shard.get("task_id") == TASK_ID
                     and shard.get("cohort_role") == population["role"]
                     and int(shard.get("start", -1)) == start and int(shard.get("stop", -1)) == stop
                     and torch.equal(shard.get("indices"), population["source_indices"][start:stop])
                     and torch.equal(shard.get("proposal_ids"), population["proposal_ids"][start:stop])
                     and torch.equal(shard.get("target_coefficients"), coeff)
                     and torch.equal(shard.get("commands"), command.cpu()),
                     f"existing {tag} shard is not bound to the fixed population and single-sample mapper")
            for grid in GRIDS:
                for key in ("coefficients", "raw_cmos"):
                    saved = shard["outputs"][grid][key].to(device=device)
                    _require(torch.allclose(saved, outputs[grid][key], rtol=1.0e-12, atol=1.0e-14),
                             f"existing {tag} {grid} {key} differs from independent production replay")
                saved_direct = shard["direct_raw_cmos"][grid].to(device=device)
                _require(torch.allclose(saved_direct, direct[grid], rtol=1.0e-12, atol=1.0e-14),
                         f"existing {tag} {grid} direct reference differs from production replay")
        elif save_outputs:
            shard = {"schema": f"{TASK_ID}.measurement_shard.v1", "task_id": TASK_ID,
                     "cohort_role": population["role"], "start": start, "stop": stop,
                     "indices": population["source_indices"][start:stop].clone(),
                     "labels": population["labels"][start:stop].clone(),
                     "proposal_ids": population["proposal_ids"][start:stop].clone(),
                     "target_coefficients": coeff.clone(), "commands": command.cpu(),
                     "clipped_fraction": detail_fraction,
                     "direct_raw_cmos": {grid: direct[grid].cpu() for grid in GRIDS},
                     "outputs": {grid: {key: value.cpu() for key, value in outputs[grid].items()}
                                 for grid in GRIDS}}
            _save_torch(shard_path, shard)
        target = coeff.to(device=device, dtype=torch.complex128)
        for local in range(stop - start):
            index = start + local
            source_index = int(population["source_indices"][index])
            one = {grid: {key: value[local:local + 1] for key, value in outputs[grid].items()}
                   for grid in GRIDS}
            direct_one = {grid: direct[grid][local:local + 1] for grid in GRIDS}
            metric = projected._error_row(one, target[local:local + 1], direct_one, source_index)
            row = {"index": source_index, "cohort_row_index": index,
                   "label": int(population["labels"][index]),
                   "proposal_id": int(population["proposal_ids"][index]),
                   "coefficient_2048": metric["coefficient_2048"],
                   "coefficient_4096": metric["coefficient_4096"],
                   "native_cmos_2048": metric["native_cmos_2048"],
                   "native_cmos_4096": metric["native_cmos_4096"],
                   "cross_grid_coefficient": metric["cross_grid_coefficient"],
                   "cross_grid_native_cmos": metric["cross_grid_native_cmos"],
                   "row_passed": all(metric[key] <= GATE_LIMIT for key in METRIC_KEYS),
                   "clipped_fraction": float(detail_fraction[local])}
            rows.append(row)
            for grid in GRIDS:
                all_outputs[grid]["coefficients"].append(outputs[grid]["coefficients"][local].cpu())
                all_outputs[grid]["raw_cmos"].append(outputs[grid]["raw_cmos"][local].cpu())
                all_outputs[grid]["direct_raw_cmos"].append(direct[grid][local].cpu())
                all_outputs[grid]["commands"].append(command[local].cpu())
    return rows, {grid: {key: torch.stack(value) for key, value in fields.items()}
                  for grid, fields in all_outputs.items()}


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def stats(selected: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
        values = np.asarray([float(row[key]) for row in selected], dtype=np.float64)
        return {"mean": float(values.mean()), "rms": float(np.sqrt(np.mean(np.square(values)))),
                "median": float(np.median(values)), "max": float(values.max())}

    by_class = {}
    for label in range(CLASS_COUNT):
        selected = [row for row in rows if int(row["label"]) == label]
        by_class[str(label)] = {key: stats(selected, key) for key in (*METRIC_KEYS, "clipped_fraction")}
    statistics = {key: stats(rows, key) for key in (*METRIC_KEYS, "clipped_fraction")}
    return {"count": len(rows), "passed": sum(bool(row["row_passed"]) for row in rows),
            "all_six_metrics_required": list(METRIC_KEYS), "statistics": statistics,
            "by_class": by_class}


def _observe_2048(inputs: Mapping[str, Any], population: Mapping[str, Any],
                  mapper: torch.nn.Module, device: torch.device, step: int) -> dict[str, Any]:
    model = _make_runtime_model(inputs, 2048, device)
    errors = []
    commands, actual_coefficients = [], []
    with torch.no_grad():
        for start in range(0, len(population["labels"]), 10):
            stop = min(start + 10, len(population["labels"]))
            coeff = population["coefficients"][start:stop]
            target = coeff.to(device=device, dtype=torch.complex128)
            features = projected._normalized_features(coeff, inputs["scale"], device)
            command, _ = projected._map_amplitude(
                mapper, features, target, inputs["matrix"].to(device),
                inputs["inverse"].to(device), inputs["scale"].to(device))
            actual = projected._forward_amplitude(model, command)["coefficients"]
            error = _phase_loss(actual, target).sqrt().detach().cpu()
            errors.extend(float(value) for value in error)
            commands.append(command.detach().cpu())
            actual_coefficients.append(actual.detach().cpu())
    result = {"step": step, "cohort_rows": len(errors),
              "coefficient_relative_error_mean": float(np.mean(errors)),
              "coefficient_relative_error_rms": float(np.sqrt(np.mean(np.square(errors)))),
              "coefficient_relative_error_median": float(np.median(errors)),
              "coefficient_relative_error_max": float(np.max(errors)),
              "role": population["role"], "full_optics_2048": True}
    stem = f"step_{step:05d}_{population['role']}_coefficient_2048"
    _write_json(ATTEMPT_DIR / "observations" / f"{stem}.json", result)
    _save_torch(ATTEMPT_DIR / "observations" / f"{stem}.pt", {
        "task_id": TASK_ID, "step": step, "role": population["role"],
        "source_indices": population["source_indices"].clone(),
        "labels": population["labels"].clone(),
        "proposal_ids": population["proposal_ids"].clone(),
        "target_coefficients": population["coefficients"].clone(),
        "commands": torch.cat(commands),
        "actual_coefficients_2048": torch.cat(actual_coefficients),
        "summary": result,
    })
    return result


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot save empty measurement rows")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _write_final_report(attempt_dir: Path, decision: Mapping[str, Any],
                        summaries: Mapping[str, Any], semantic: Mapping[str, Any]) -> None:
    state = decision["state"]
    train = summaries["train"]
    dev = summaries["validation"]
    semantic_lines = []
    for condition in ("direct_decoder", "frozen_v2", "mapped_candidate"):
        if condition not in semantic.get("conditions", {}):
            continue
        record = semantic["conditions"][condition]
        semantic_lines.append(
            f"- {condition}: ED={record['evaluators']['ED']['accuracy']:.4f}, "
            f"EC={record['evaluators']['EC']['accuracy']:.4f}, "
            f"ED/EC一致率={record['ED_EC_prediction_agreement']:.4f}, "
            f"裁剪后均值={record['sensor_statistics']['clipped_readout_mean']:.6g}。"
        )
    text = f"""# G1语义生成主线：200目标固定暴露试训

## 结果与状态

任务`{TASK_ID}`从原V2终点新增恰好{decision['new_updates']:,}步，固定光学更新为0，证据等级为`diagnostic-only`。训练人口六项误差通过{train['passed']}/{train['count']}，固定开发人口通过{dev['passed']}/{dev['count']}；最终状态为`{state}`。开发人口是已被历史研究消费的固定开发证据，不是独立留出。

## 科学认识

本任务使用`Lc + 5Lbox`，只更新映射器。历史2,000条配对语义辅助结果：
{chr(10).join(semantic_lines)}

冻结评价器差异只作辅助诊断，不等同于人工可读性或新颖性。终态的固定开发人口语义指标与直接Decoder、冻结V2起点配对保存；不进入数值Gate。

## 目标缺口

唯一数值Gate要求固定训练200条及开发200条的全部六项误差均不高于`{GATE_LIMIT:.0e}`。通过只支持这400条固定目标上的连续数值制备，不证明新人口泛化、语义可读性、全部G1分布或实物光路可用。

## 工程完成与验证

起点为V2 5,000步终点，新建空Adam；本任务新增{decision['new_updates']:,}步，训练日程提供每目标{EXPOSURES_PER_TARGET}次暴露，累计映射器参数谱系为{decision['lineage_mapper_updates']:,}步。矩阵/完整光路选路、训练目标见证、真实更新、恢复及固定正控记录在`preflight_evidence/`。终态测量按单样本生成SLM指令并经2048/4096完整光路；Windows消费者从保存数组重算六项误差和人口分母。

## 决策影响与下一步

{('当前人口全部通过；建议另行设计新独立留出与语义生成评价。' if decision['all_400_rows_passed'] else '完整有效终态测量未满足全人口六项门槛；当前固定结构、损失和15,000步预算未达到目标，不自动续训。')}后续的全新留出、量化、G4、人工评分和实物验证均需另行规划与授权。历史失败结论及`PROJECT_GEOMETRY_UNMEASURED`保持。
"""
    (attempt_dir / "final_report.md").write_text(text, encoding="utf-8")


def _write_transfer_manifest(attempt_dir: Path, run_id: str) -> dict[str, Any]:
    excluded = {"transfer_manifest.json", "transfer_receipt.json", "windows_reconsumption.json"}
    files = [{"path": path.relative_to(attempt_dir).as_posix(), "bytes": path.stat().st_size}
             for path in sorted(attempt_dir.rglob("*"))
             if path.is_file() and path.name not in excluded and ".tmp." not in path.name]
    payload = {"schema": f"{TASK_ID}.transfer.v1", "task_id": TASK_ID, "run_id": run_id,
               "files": files, "file_count": len(files),
               "bytes_total": sum(item["bytes"] for item in files)}
    _write_json(attempt_dir / "transfer_manifest.json", payload)
    return payload


def _snapshot_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _semantic_summary(inputs: Mapping[str, Any], population: Mapping[str, torch.Tensor],
                      raw: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    labels = population["labels"].long().cpu()
    area = 6.25e-12
    gain = float(inputs["camera_gain"])
    evaluators = {"ED": inputs["evaluator_ed"].cpu().eval(),
                  "EC": inputs["evaluator_ec"].cpu().eval()}
    states = {key: _snapshot_state(model) for key, model in evaluators.items()}
    logits_by_condition: dict[str, dict[str, list[torch.Tensor]]] = {
        condition: {key: [] for key in evaluators} for condition in raw}
    readouts: dict[str, list[torch.Tensor]] = {key: [] for key in logits_by_condition}
    sensor_accumulators: dict[str, dict[str, float]] = {}
    for start in range(0, len(labels), 10):
        stop = min(start + 10, len(labels))
        for condition in logits_by_condition:
            intensity = raw[condition][start:stop].to(torch.float64)
            calibrated = intensity * area * gain
            clipped = calibrated.clamp(0.0, 1.0)
            readout = pilot._readout_for_evaluator(intensity, area, gain)
            readouts[condition].append(readout.cpu())
            values = {
                "pixel_count": float(calibrated.numel()),
                "raw_sum": float(intensity.sum()), "raw_square_sum": float(intensity.square().sum()),
                "raw_max": float(intensity.max()),
                "calibrated_sum": float(calibrated.sum()),
                "calibrated_square_sum": float(calibrated.square().sum()),
                "calibrated_max": float(calibrated.max()),
                "below_zero_count": float((calibrated < 0.0).sum()),
                "above_one_count": float((calibrated > 1.0).sum()),
                "clipped_sum": float(clipped.sum()), "clipped_square_sum": float(clipped.square().sum()),
                "zero_count": float((clipped == 0.0).sum()), "one_count": float((clipped == 1.0).sum()),
                "readout_sum": float(readout.double().sum()),
            }
            accumulator = sensor_accumulators.setdefault(condition, {})
            for key in ("pixel_count", "raw_sum", "raw_square_sum", "below_zero_count",
                        "above_one_count", "calibrated_sum", "calibrated_square_sum", "clipped_sum",
                        "clipped_square_sum", "zero_count", "one_count", "readout_sum"):
                accumulator[key] = accumulator.get(key, 0.0) + values[key]
            accumulator["raw_max"] = max(accumulator.get("raw_max", -math.inf), values["raw_max"])
            accumulator["calibrated_max"] = max(accumulator.get("calibrated_max", -math.inf),
                                                 values["calibrated_max"])
            for family, model in evaluators.items():
                with torch.no_grad():
                    prediction = model(readout)
                    logits = prediction[-1] if isinstance(prediction, tuple) else prediction
                    _require(logits.ndim == 2 and tuple(logits.shape) == (stop - start, CLASS_COUNT)
                             and torch.isfinite(logits).all().item(),
                             f"frozen {family} evaluator returned malformed logits")
                    logits_by_condition[condition][family].append(logits.detach().cpu())
    result: dict[str, Any] = {"camera_pixel_area_m2": area, "camera_gain_w_inverse": gain,
                              "evidence_role": "historical_consumed_holdout_descriptive_only",
                              "human_readability": "not_measured", "novelty": "not_measured",
                              "conditions": {}, "same_class_diversity": {}}
    condition_images = {}
    for condition, families in logits_by_condition.items():
        condition_images[condition] = torch.cat(readouts[condition])
        sensor = sensor_accumulators[condition]
        pixel_count = sensor["pixel_count"]
        image_count = float(len(labels) * 28 * 28)
        sensor_stats = {
            "raw_w_per_m2_mean": sensor["raw_sum"] / pixel_count,
            "raw_w_per_m2_rms": math.sqrt(sensor["raw_square_sum"] / pixel_count),
            "raw_w_per_m2_max": sensor["raw_max"],
            "fixed_gain_uncut_mean": sensor["calibrated_sum"] / pixel_count,
            "fixed_gain_uncut_rms": math.sqrt(sensor["calibrated_square_sum"] / pixel_count),
            "fixed_gain_uncut_max": sensor["calibrated_max"],
            "clipped_readout_mean": sensor["clipped_sum"] / pixel_count,
            "clipped_readout_rms": math.sqrt(sensor["clipped_square_sum"] / pixel_count),
            "fraction_below_zero": sensor["below_zero_count"] / pixel_count,
            "fraction_above_one": sensor["above_one_count"] / pixel_count,
            "fraction_at_zero_after_clip": sensor["zero_count"] / pixel_count,
            "fraction_at_one_after_clip": sensor["one_count"] / pixel_count,
            "mean_pooled_28x28_readout": sensor["readout_sum"] / image_count,
            "readout_mapping": "raw_W_per_m2_x_pixel_area_x_frozen_gain_then_clip_0_1",
        }
        scores = {}
        for family, batches in families.items():
            logits = torch.cat(batches)
            pred = logits.argmax(dim=1)
            confusion = torch.bincount(labels * CLASS_COUNT + pred, minlength=CLASS_COUNT**2).reshape(
                CLASS_COUNT, CLASS_COUNT)
            recall = confusion.diag().double() / confusion.sum(dim=1).clamp_min(1)
            scores[family] = {"accuracy": float((pred == labels).double().mean()),
                              "macro_recall": float(recall.mean()),
                              "per_class_recall": [float(x) for x in recall],
                              "confusion_matrix": confusion.tolist(),
                              "predicted_labels": pred.tolist()}
        ed, ec = scores["ED"]["predicted_labels"], scores["EC"]["predicted_labels"]
        result["conditions"][condition] = {"evaluators": scores, "sensor_statistics": sensor_stats,
                                           "ED_EC_prediction_agreement": float(np.mean(
                                               np.asarray(ed) == np.asarray(ec)))}
        flat = condition_images[condition].flatten(1)
        class_diversity = {}
        duplicate_count = 0
        for label in range(CLASS_COUNT):
            x = flat[labels == label].double()
            norms = x.square().sum(1)
            distance_sq = (norms[:, None] + norms[None, :] - 2.0 * (x @ x.T)).clamp_min(0.0)
            upper = torch.triu_indices(x.shape[0], x.shape[0], offset=1)
            distances = distance_sq[upper[0], upper[1]].sqrt() / math.sqrt(x.shape[1])
            unique = torch.unique(x, dim=0).shape[0]
            duplicate_count += x.shape[0] - unique
            class_diversity[str(label)] = {"mean_pairwise_normalized_l2": float(distances.mean()),
                                           "median_pairwise_normalized_l2": float(distances.median()),
                                           "unique_readout_count": int(unique)}
        result["same_class_diversity"][condition] = {
            "pairwise_distance_definition": "mean_euclidean_distance_over_sqrt_784_on_28x28_fixed_gain_readout",
            "by_class": class_diversity, "exact_duplicate_count": duplicate_count}
    if "direct_decoder" in result["conditions"]:
        labels_np = labels.numpy()

        def paired_change(reference: str, condition: str, family: str) -> dict[str, Any]:
            reference_predictions = np.asarray(
                result["conditions"][reference]["evaluators"][family]["predicted_labels"])
            candidate_predictions = np.asarray(
                result["conditions"][condition]["evaluators"][family]["predicted_labels"])
            reference_correct = reference_predictions == labels_np
            candidate_correct = candidate_predictions == labels_np

            def counts(mask: np.ndarray) -> dict[str, int]:
                return {
                    "reference_correct_candidate_correct": int(np.sum(mask & reference_correct & candidate_correct)),
                    "reference_correct_candidate_wrong": int(np.sum(mask & reference_correct & ~candidate_correct)),
                    "reference_wrong_candidate_correct": int(np.sum(mask & ~reference_correct & candidate_correct)),
                    "reference_wrong_candidate_wrong": int(np.sum(mask & ~reference_correct & ~candidate_correct)),
                    "prediction_changed": int(np.sum(mask & (reference_predictions != candidate_predictions))),
                    "count": int(mask.sum()),
                }

            per_class = {str(label): counts(labels_np == label) for label in range(CLASS_COUNT)}
            overall = counts(np.ones_like(labels_np, dtype=bool))
            overall["net_accuracy_change"] = float(candidate_correct.mean() - reference_correct.mean())
            return {"reference": reference, "candidate": condition, "evaluator": family,
                    "overall": overall, "by_class": per_class}

        comparisons = {}
        for condition in result["conditions"]:
            if condition == "direct_decoder":
                continue
            comparisons[condition] = {
                family: paired_change("direct_decoder", condition, family)
                for family in ("ED", "EC")}
        if "frozen_v2" in result["conditions"] and "box_b5000" in result["conditions"]:
            comparisons["box_b5000_vs_frozen_v2"] = {
                family: paired_change("frozen_v2", "box_b5000", family)
                for family in ("ED", "EC")}
        result["paired_semantic_changes"] = comparisons
    _require(all(_compare_parameter_states(states[key], model.state_dict()) == 0.0
                 for key, model in evaluators.items()), "frozen semantic evaluator state changed")
    return result, condition_images


def _save_contact_sheet(path: Path, images: Mapping[str, torch.Tensor], labels: torch.Tensor) -> None:
    conditions = tuple(images)
    scale = 7
    label_column = 180
    canvas = Image.new("RGB", (len(conditions) * 28 * scale + label_column,
                                20 * 28 * scale), "white")
    draw = ImageDraw.Draw(canvas)
    for col, condition in enumerate(conditions):
        draw.text((col * 28 * scale, 4), condition, fill="black")
    for label in range(CLASS_COUNT):
        indices = [i for i, value in enumerate(labels.tolist()) if value == label][:2]
        for sample_offset, index in enumerate(indices):
            row = label * 2 + sample_offset
            for col, condition in enumerate(conditions):
                gray = torch.round(images[condition][index, 0].clamp(0, 1) * 255).byte().numpy()
                image = Image.fromarray(gray).resize((28 * scale, 28 * scale), Image.Resampling.NEAREST).convert("RGB")
                canvas.paste(image, (col * 28 * scale, row * 28 * scale))
                draw.text((len(conditions) * 28 * scale + 8, row * 28 * scale + 8),
                          f"class {label} sample {sample_offset}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _audit_saved(config: Mapping[str, Any], inputs: Mapping[str, Any],
                 output_dir: Path | None = None) -> dict[str, Any]:
    output = (_resolve(config["output"]["audit_dir"]) if output_dir is None
              else output_dir.resolve())
    _require(output.is_relative_to(RUNS_DIR.resolve()),
             "semantic audit output must stay inside this experiment's runs directory", ValueError)
    _require(not output.exists(), f"refusing to overwrite saved semantic audit: {output}", FileExistsError)
    source_dir = _resolve(config["source"]["fullpop_evidence_dir"])
    source_population = torch.load(source_dir / "fixed_population_and_schedule.pt",
                                   map_location="cpu", weights_only=False)
    holdout = source_population["holdout"]
    _require(tuple(holdout["coefficients"].shape) == (2000, PORT_COUNT)
             and torch.equal(holdout["labels"], torch.arange(CLASS_COUNT).repeat_interleave(200)),
             "historical holdout identities are malformed")
    conditions = ("direct_decoder", "frozen_v2", "box_b5000")
    raw = {condition: [] for condition in conditions}
    labels = []
    ids = []
    measurement_dir = source_dir / "measurements" / "holdout"
    shards = sorted(measurement_dir.glob("holdout_*.pt"))
    _require(len(shards) == 200, "historical saved holdout must contain exactly 200 shards")
    for shard_index, path in enumerate(shards):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        start, stop = shard_index * 10, shard_index * 10 + 10
        _require(torch.equal(payload["population_indices"], torch.arange(start, stop))
                 and torch.equal(payload["target_coefficients"], holdout["coefficients"][start:stop])
                 and torch.equal(payload["labels"], holdout["labels"][start:stop])
                 and torch.equal(payload["proposal_ids"], holdout["proposal_ids"][start:stop]),
                 f"historical semantic sample identity mismatch in {path.name}")
        labels.append(payload["labels"])
        ids.append(payload["proposal_ids"])
        raw["direct_decoder"].append(payload["direct_raw_cmos"][2048].double())
        raw["frozen_v2"].append(payload["baseline_outputs"][2048]["raw_cmos"].double())
        raw["box_b5000"].append(payload["candidate_outputs"][2048]["raw_cmos"].double())
    population = {"labels": torch.cat(labels), "proposal_ids": torch.cat(ids)}
    raw_arrays = {name: torch.cat(shards) for name, shards in raw.items()}
    semantic, images = _semantic_summary(inputs, population, raw_arrays)
    output.mkdir(parents=True)
    _write_json(output / "semantic_audit.json", semantic)
    _save_torch(output / "semantic_readouts.pt", {"labels": population["labels"],
                                                   "proposal_ids": population["proposal_ids"],
                                                   "readouts": images})
    _save_contact_sheet(output / "fixed_class_examples.png", images, population["labels"])
    return {"status": "SAVED_OUTPUTS_AUDITED", "samples": len(population["labels"]),
            "output_dir": str(output), "conditions": list(semantic["conditions"])}


def _semantic_attribution(config: Mapping[str, Any], output_dir: Path | None = None) -> dict[str, Any]:
    output = (_resolve(config["output"]["audit_extension_root"]) if output_dir is None
              else output_dir.resolve())
    _require(output.is_relative_to(RUNS_DIR.resolve()),
             "semantic attribution output must stay inside this experiment's runs directory", ValueError)
    _require(not output.exists(), f"refusing to overwrite semantic attribution: {output}", FileExistsError)
    source_dir = _resolve(config["source"]["fullpop_evidence_dir"])
    audit_dir = RUNS_DIR / "audit_repair_01"
    semantic_path = audit_dir / "semantic_audit.json"
    readout_path = audit_dir / "semantic_readouts.pt"
    candidate_path = source_dir / "candidate_holdout_rows.csv"
    baseline_path = source_dir / "frozen_v2_baseline_holdout_rows.csv"
    _require(all(path.is_file() for path in (semantic_path, readout_path, candidate_path, baseline_path)),
             "saved semantic predictions or paired numeric errors are missing", FileNotFoundError)
    semantic = _json(semantic_path)
    saved_readouts = torch.load(readout_path, map_location="cpu", weights_only=False)
    expected_labels = saved_readouts["labels"].long()
    expected_ids = saved_readouts["proposal_ids"].long()
    _require(expected_labels.shape == (2000,) and expected_ids.shape == (2000,)
             and torch.equal(expected_labels, torch.arange(CLASS_COUNT).repeat_interleave(200)),
             "historical semantic readout identities are malformed")

    def load_rows(path: Path) -> list[dict[str, str]]:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        _require(len(rows) == 2000, f"saved paired error table has the wrong row count: {path.name}")
        return sorted(rows, key=lambda row: int(row["population_index"]))

    baseline_rows, candidate_rows = load_rows(baseline_path), load_rows(candidate_path)
    for index, (baseline, candidate) in enumerate(zip(baseline_rows, candidate_rows, strict=True)):
        _require(int(baseline["population_index"]) == index
                 and int(candidate["population_index"]) == index
                 and int(baseline["label"]) == int(expected_labels[index])
                 and int(candidate["label"]) == int(expected_labels[index])
                 and int(baseline["proposal_id"]) == int(expected_ids[index])
                 and int(candidate["proposal_id"]) == int(expected_ids[index]),
                 f"paired semantic/error table identity differs at row {index}")
    predictions = {
        condition: {family: np.asarray(semantic["conditions"][condition]["evaluators"][family][
            "predicted_labels"], dtype=np.int64)
                    for family in ("ED", "EC")}
        for condition in ("direct_decoder", "frozen_v2", "box_b5000")}
    labels = expected_labels.numpy().astype(np.int64, copy=False)
    for condition, families in predictions.items():
        _require(all(values.shape == (2000,) and np.all((values >= 0) & (values < CLASS_COUNT))
                     for values in families.values()),
                 f"saved {condition} prediction arrays are incomplete")

    def transition(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> dict[str, int | float]:
        ref_correct, cand_correct = reference == labels, candidate == labels
        return {"count": int(mask.sum()),
                "reference_correct_candidate_correct": int(np.sum(mask & ref_correct & cand_correct)),
                "reference_correct_candidate_wrong": int(np.sum(mask & ref_correct & ~cand_correct)),
                "reference_wrong_candidate_correct": int(np.sum(mask & ~ref_correct & cand_correct)),
                "reference_wrong_candidate_wrong": int(np.sum(mask & ~ref_correct & ~cand_correct)),
                "wrong_prediction_preserved": int(np.sum(mask & ~ref_correct & ~cand_correct
                                                          & (reference == candidate))),
                "prediction_changed": int(np.sum(mask & (reference != candidate))),
                "net_accuracy_change": (float(np.mean(cand_correct[mask]) - np.mean(ref_correct[mask]))
                                        if np.any(mask) else 0.0)}

    paired = {}
    for comparator, base_condition, new_condition in (
            ("direct_decoder_to_frozen_v2", "direct_decoder", "frozen_v2"),
            ("direct_decoder_to_box_b5000", "direct_decoder", "box_b5000"),
            ("frozen_v2_to_box_b5000", "frozen_v2", "box_b5000")):
        paired[comparator] = {}
        for family in ("ED", "EC"):
            base, new = predictions[base_condition][family], predictions[new_condition][family]
            paired[comparator][family] = {
                "overall": transition(base, new, np.ones(2000, dtype=bool)),
                "by_class": {str(label): transition(base, new, labels == label)
                             for label in range(CLASS_COUNT)}}

    coefficient_errors = np.asarray([
        max(float(row["coefficient_2048"]), float(row["coefficient_4096"]))
        for row in candidate_rows], dtype=np.float64)
    cmos_errors = np.asarray([
        max(float(row["native_cmos_2048"]), float(row["native_cmos_4096"]))
        for row in candidate_rows], dtype=np.float64)
    _require(np.isfinite(coefficient_errors).all() and np.isfinite(cmos_errors).all(),
             "paired coefficient or CMOS errors contain nonfinite values")

    def quartile_groups(values: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
        order = np.argsort(values, kind="stable")
        groups = np.empty(values.shape, dtype=np.int64)
        groups[order] = np.minimum(np.arange(len(values)) * 4 // len(values), 3)
        boundaries = [{"min": float(values[groups == group].min()),
                       "max": float(values[groups == group].max()),
                       "q25": float(np.quantile(values, 0.25)),
                       "q50": float(np.quantile(values, 0.50)),
                       "q75": float(np.quantile(values, 0.75))}
                      for group in range(4)]
        return groups, boundaries

    coefficient_groups, coefficient_bounds = quartile_groups(coefficient_errors)
    cmos_groups, cmos_bounds = quartile_groups(cmos_errors)
    quartile_rows = []
    for metric, groups, bounds in (("candidate_max_coefficient_error", coefficient_groups, coefficient_bounds),
                                   ("candidate_max_native_cmos_error", cmos_groups, cmos_bounds)):
        for group in range(4):
            mask = groups == group
            for family in ("ED", "EC"):
                ref, new = predictions["frozen_v2"][family], predictions["box_b5000"][family]
                outcome = transition(ref, new, mask)
                quartile_rows.append({"metric": metric, "quartile": group + 1,
                                      "label": "all",
                                      "count": outcome["count"],
                                      "error_min": bounds[group]["min"],
                                      "error_max": bounds[group]["max"],
                                      "prediction_changed": outcome["prediction_changed"],
                                      "reference_correct_candidate_wrong": outcome[
                                          "reference_correct_candidate_wrong"],
                                      "reference_wrong_candidate_correct": outcome[
                                          "reference_wrong_candidate_correct"],
                                      "wrong_prediction_preserved": outcome["wrong_prediction_preserved"],
                                      "net_accuracy_change": outcome["net_accuracy_change"],
                                      "evaluator": family})
                for label in range(CLASS_COUNT):
                    class_mask = mask & (labels == label)
                    class_outcome = transition(ref, new, class_mask)
                    quartile_rows.append({"metric": metric, "quartile": group + 1,
                                          "label": label,
                                          "count": class_outcome["count"],
                                          "error_min": bounds[group]["min"],
                                          "error_max": bounds[group]["max"],
                                          "prediction_changed": class_outcome["prediction_changed"],
                                          "reference_correct_candidate_wrong": class_outcome[
                                              "reference_correct_candidate_wrong"],
                                          "reference_wrong_candidate_correct": class_outcome[
                                              "reference_wrong_candidate_correct"],
                                          "wrong_prediction_preserved": class_outcome[
                                              "wrong_prediction_preserved"],
                                          "net_accuracy_change": class_outcome["net_accuracy_change"],
                                          "evaluator": family})
    per_sample = []
    for index in range(2000):
        record = {"population_index": index, "label": int(labels[index]),
                  "proposal_id": int(expected_ids[index]),
                  "candidate_max_coefficient_error": float(coefficient_errors[index]),
                  "candidate_max_native_cmos_error": float(cmos_errors[index]),
                  "coefficient_error_quartile": int(coefficient_groups[index] + 1),
                  "native_cmos_error_quartile": int(cmos_groups[index] + 1)}
        for family in ("ED", "EC"):
            record[f"direct_{family}_prediction"] = int(predictions["direct_decoder"][family][index])
            record[f"v2_{family}_prediction"] = int(predictions["frozen_v2"][family][index])
            record[f"b5000_{family}_prediction"] = int(predictions["box_b5000"][family][index])
        per_sample.append(record)
    result = {"task_id": TASK_ID, "status": "SEMANTIC_ATTRIBUTION_COMPLETE",
              "population_role": "historical_consumed_holdout_descriptive_only",
              "count": 2000, "predictions_recomputed": False,
              "prediction_source": str(semantic_path),
              "numeric_error_sources": {"frozen_v2": str(baseline_path),
                                        "box_b5000": str(candidate_path)},
              "error_quartile_definition": "stable_rank_quartiles_of_candidate_max_grid_error",
              "coefficient_quartile_bounds": coefficient_bounds,
              "native_cmos_quartile_bounds": cmos_bounds,
              "pairwise_transitions": paired,
              "causality_claim": False, "LPIPS": "not_computed", "human_readability": "not_measured"}
    output.mkdir(parents=True)
    _write_json(output / "semantic_attribution_summary.json", result)
    _atomic_csv(output / "semantic_attribution_by_sample.csv", per_sample)
    _atomic_csv(output / "semantic_error_quartiles.csv", quartile_rows)
    return result


def _state_snapshot(models: Mapping[int, Any]) -> dict[str, Any]:
    return {str(grid): {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            for grid, model in models.items()}


def _box_loss_qualification(device: torch.device) -> dict[str, Any]:
    contract = fit20._box_contract_check(device)
    values = torch.full((1, PIXEL_COUNT), 0.5, dtype=torch.float64, device=device)
    values[0, :5] = torch.tensor([-0.25, 0.0, 0.4, 1.0, 1.25], device=device)
    values.requires_grad_(True)
    loss = fit20._box_loss(values)
    gradient, = torch.autograd.grad(loss, values)
    expected = 2.0 * (values.detach() - values.detach().clamp(0.0, 1.0)) / PIXEL_COUNT
    _require(float((gradient - expected).abs().max()) <= 1.0e-16,
             "box-loss analytic derivative differs from the bound formula")
    finite_differences = {}
    h = 1.0e-6
    for position in (0, 2, 4):
        plus, minus = values.detach().clone(), values.detach().clone()
        plus[0, position] += h
        minus[0, position] -= h
        numeric = (float(fit20._box_loss(plus)) - float(fit20._box_loss(minus))) / (2.0 * h)
        finite_differences[str(position)] = {
            "numeric": numeric, "analytic": float(expected[0, position]),
            "absolute_error": abs(numeric - float(expected[0, position]))}
    _require(max(item["absolute_error"] for item in finite_differences.values()) <= 1.0e-11,
             "box-loss finite-difference derivative exceeds 1e-11")
    shifted = values.detach().clone()
    shifted[0, 0] -= h * gradient[0, 0]
    shifted[0, 4] -= h * gradient[0, 4]
    _require(float(fit20._box_loss(shifted)) < float(loss.detach()),
             "box-loss gradient does not descend at both amplitude bounds")
    return {**contract, "finite_difference_checks": finite_differences,
            "gradient_descent_direction_passed": True, "box_weight": BOX_LAMBDA}


def _recovery_replay(config: Mapping[str, Any], inputs: Mapping[str, Any],
                     population: Mapping[str, Any], checkpoint_path: Path,
                     output_path: Path) -> dict[str, Any]:
    schedule = _build_schedule(config["training"]["schedule_seed"])
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require(saved.get("task_id") == TASK_ID and saved.get("completed_updates") == 3
             and saved.get("schedule_cursor") == 3 and len(saved.get("history", [])) == 3,
             "recovery replay checkpoint is not the exact three-step state")
    engine = str(saved["engine"])
    device = torch.device("cuda:0")
    mapper = _mapper(inputs, device)
    optimizer = _optimizer(mapper, config)
    mapper.load_state_dict(saved["mapper_state"], strict=True)
    optimizer.load_state_dict(saved["optimizer_state"])
    model = _make_runtime_model(inputs, 2048, device)
    torch.set_rng_state(saved["rng_state"]["cpu"])
    if "cuda" in saved["rng_state"]:
        torch.cuda.set_rng_state_all(saved["rng_state"]["cuda"])
    batch = schedule[3].clone()
    _require(torch.equal(batch, saved["batch"]), "independent recovery selected a different fourth batch")
    result = _update(mapper, optimizer, model, inputs,
                     population["train"]["coefficients"][batch], device, engine)
    payload = {"task_id": TASK_ID, "completed_updates": 4, "engine": engine,
               "loss": result["loss"], "history": [*saved["history"], {
                   "step": 4, "schedule_indices": batch.tolist(), "loss": result["loss"],
                   "coefficient_loss": result["coefficient_loss"], "box_loss": result["box_loss"]}],
               "mapper_state": _copy_state(mapper.state_dict()),
               "optimizer_state": copy.deepcopy(optimizer.state_dict()),
               "rng_state": _rng_snapshot(), "batch": batch}
    before_save_rng = _rng_snapshot()
    _save_torch(output_path, payload)
    _require(_rng_equal(before_save_rng, _rng_snapshot()),
             "independent fourth-step serialization changed process RNG")
    return {"status": "RECOVERY_STEP4_SAVED", "completed_updates": 4,
            "engine": engine, "loss": result["loss"], "output_path": str(output_path)}


def _run_ten_positive_controls(inputs: Mapping[str, Any],
                              device: torch.device) -> dict[str, Any]:
    source = inputs["positive_control_witness"]
    indices = torch.arange(0, 20, 2)
    targets = source["target_coefficients"][indices].to(device=device, dtype=torch.complex128)
    commands = source["amplitudes"][indices].to(device=device, dtype=torch.float32)
    optical = {grid: _make_runtime_model(inputs, grid, device) for grid in GRIDS}
    optical_before = _state_snapshot(optical)
    evaluators_before = {name: _snapshot_state(inputs[f"evaluator_{name.lower()}"])
                         for name in ("ED", "EC")}
    with torch.no_grad():
        actual = {grid: projected._forward_amplitude(optical[grid], commands) for grid in GRIDS}
        direct = {grid: projected._forward_target(optical[grid], targets) for grid in GRIDS}
    response_prediction = _forward_matrix(commands, inputs["response_matrix"].to(device))
    actual_2048 = actual[2048]["coefficients"]
    response_errors = torch.linalg.vector_norm(response_prediction - actual_2048, dim=1).div(
        torch.linalg.vector_norm(actual_2048, dim=1).clamp_min(1.0e-30))
    response_error = float(response_errors.max())
    row_metrics = _positive_control_error_rows(actual, direct, targets)
    _require(response_error <= 1.0e-6
             and all(all(metric[key] <= GATE_LIMIT for key in METRIC_KEYS) for metric in row_metrics),
             "production ten-sample positive control failed the response or six-metric check")
    _require(all(_compare_parameter_states(optical_before[str(grid)], optical[grid].state_dict()) == 0.0
                 for grid in GRIDS)
             and all(_compare_parameter_states(evaluators_before[name],
                                               inputs[f"evaluator_{name.lower()}"].state_dict()) == 0.0
                     for name in ("ED", "EC")),
             "optical or evaluator state changed during positive-control replay")
    payload = {"schema": f"{TASK_ID}.positive_control_replay.v1", "task_id": TASK_ID,
               "source_indices": indices, "target_coefficients": targets.detach().cpu(),
               "amplitudes": commands.detach().cpu(), "metrics": row_metrics,
               "outputs": {grid: {key: value.detach().cpu() for key, value in actual[grid].items()}
                           for grid in GRIDS},
               "direct_raw_cmos": {grid: direct[grid]["raw_cmos"].detach().cpu()
                                   for grid in GRIDS},
               "matrix_response_relative_error_max": response_error}
    return {"response_relative_error_max": response_error, "metrics": row_metrics,
            "optics_and_evaluators_frozen": True, "payload": payload}


def _preflight(config_path: Path, config: Mapping[str, Any], *,
               reference_dir: Path | None = None,
               engine_dir: Path | None = None) -> dict[str, Any]:
    reference_dir = (_resolve(config["output"]["reference_qualification_root"])
                     if reference_dir is None else reference_dir.resolve())
    engine_dir = (_resolve(config["output"]["engine_qualification_root"])
                  if engine_dir is None else engine_dir.resolve())
    for path in (reference_dir, engine_dir, PREFLIGHT_DIR):
        _require(path.is_relative_to(RUNS_DIR.resolve()),
                 "preflight source/output must stay inside this experiment's runs directory", ValueError)
    _require(not PREFLIGHT_DIR.exists(), f"preflight path already exists: {PREFLIGHT_DIR}", FileExistsError)
    _require(torch.cuda.is_available(), "G1 exposure preflight requires CUDA")
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    inputs = _load_inputs(config)
    population = _load_population(config, inputs)
    training_witnesses, witness = _consume_training_witnesses(
        reference_dir, inputs, population, device=device)
    engine, engine_recovery = _consume_engine_qualification(engine_dir, inputs)
    schedule = _build_schedule(config["training"]["schedule_seed"])
    PREFLIGHT_DIR.mkdir(parents=True)
    run_id = uuid.uuid4().hex
    identity = _run_identity(config, run_id)
    _write_json(PREFLIGHT_DIR / "preflight_identity.json", {
        "task_id": TASK_ID, "run_id": run_id, "run_identity": identity.to_dict()})
    _save_torch(PREFLIGHT_DIR / "fixed_population_and_schedule.pt", {
        "schema": f"{TASK_ID}.population.v1", "task_id": TASK_ID,
        "training": population["train"], "development": population["validation"],
        "schedule": schedule})
    _save_torch(PREFLIGHT_DIR / "mapper_initialization.pt", {
        "schema": f"{TASK_ID}.initialization.v1", "task_id": TASK_ID,
        "source_checkpoint": str(inputs["source_mapper_path"]),
        "mapper_state": _copy_state(inputs["source_mapper"]["mapper_state"]),
        "new_adam_state": {}})
    projector = inputs["projector_check"]
    box_check = _box_loss_qualification(device)
    matrix_summary = engine["matrix_qualification"]
    selected_engine = str(matrix_summary["selected_engine"])
    recovery = engine["recovery"]
    initial_mapper = _mapper(inputs, device)
    initial_optimizer = _optimizer(initial_mapper, config)
    _require(len(initial_optimizer.state) == 0
             and _compare_parameter_states(initial_mapper.state_dict(),
                                           inputs["source_mapper"]["mapper_state"]) == 0.0,
             "formal preflight mapper start or empty Adam state differs from the registered V2 endpoint")
    del initial_mapper, initial_optimizer
    _save_torch(PREFLIGHT_DIR / "training_witnesses.pt", {
        "schema": f"{TASK_ID}.witnesses.v1", "task_id": TASK_ID,
        "population_ids": population["train"]["proposal_ids"].clone(),
        "witnesses": training_witnesses, "failed_indices": []})
    shutil.copy2(reference_dir / "reference_qualification_summary.json",
                 PREFLIGHT_DIR / "reference_qualification_summary.json")
    shutil.copy2(engine_dir / "engine_qualification_summary.json",
                 PREFLIGHT_DIR / "engine_qualification_summary.json")
    shutil.copytree(engine_dir / "engineering_recovery", PREFLIGHT_DIR / "engineering_recovery")
    _save_torch(PREFLIGHT_DIR / "engineering_recovery.pt", engine_recovery)
    shutil.copy2(engine_dir / "engineering_recovery" / "engineering_recovery.pt",
                 PREFLIGHT_DIR / "engineering_recovery_source.pt")
    source = inputs["positive_control_witness"]
    witness_indices = torch.arange(0, 20, 2)
    targets = source["target_coefficients"][witness_indices].to(device=device, dtype=torch.complex128)
    commands = source["amplitudes"][witness_indices].to(device=device, dtype=torch.float32)
    optical = {grid: _make_runtime_model(inputs, grid, device) for grid in GRIDS}
    before = _state_snapshot(optical)
    evaluator_state = {name: _snapshot_state(inputs[f"evaluator_{name.lower()}"])
                       for name in ("ED", "EC")}
    with torch.no_grad():
        actual = {grid: projected._forward_amplitude(optical[grid], commands) for grid in GRIDS}
        direct = {grid: projected._forward_target(optical[grid], targets) for grid in GRIDS}
    response_error = float(torch.linalg.vector_norm(
        _forward_matrix(commands, inputs["response_matrix"].to(device))
        - actual[2048]["coefficients"], dim=1).div(
            torch.linalg.vector_norm(actual[2048]["coefficients"], dim=1).clamp_min(1.0e-30)).max())
    if selected_engine == "response_matrix":
        _require(response_error <= 1.0e-6,
                 "selected matrix training path fails the ten-control production response check")
    _require(all(_compare_parameter_states(before[str(grid)], optical[grid].state_dict()) == 0.0
                 for grid in GRIDS)
             and all(_compare_parameter_states(evaluator_state[name],
                                               inputs[f"evaluator_{name.lower()}"].state_dict()) == 0.0
                     for name in ("ED", "EC")),
             "frozen optical/evaluator state changed during preflight")
    row_metrics = _positive_control_error_rows(actual, direct, targets)
    _require(all(all(metric[key] <= GATE_LIMIT for key in METRIC_KEYS) for metric in row_metrics),
             "production ten-sample positive control failed the six-metric gate")
    resource = {"passed": True, "status": "PASS", "cuda_device": torch.cuda.get_device_name(device),
                "free_memory_bytes_observation": int(torch.cuda.mem_get_info(device)[0]),
                "peak_allocated_bytes_observation": int(torch.cuda.max_memory_allocated(device)),
                "fixed_free_memory_quota": False}
    correctness = {"passed": True, "status": "PASS", "source_v2_terminal_verified": True,
                   "fixed_200_training_targets_witnessed": len(training_witnesses) == TRAIN_ROWS,
                   "population_schedule_and_roles_verified": True,
                   "six_metric_positive_control_passed": True,
                   "all_mapper_parameters_updated_and_optics_frozen": True,
                   "box_loss_value_gradient_and_direction_checked": True,
                   "positive_control_response_error_max": response_error,
                   "selected_engine": selected_engine}
    implementation = {"passed": True, "status": "PASS",
                      **engine["matrix_qualification"], "recovery_verified": True,
                      "recovery_tolerance": RECOVERY_TOLERANCE}
    control = ResearchRunControl(identity=identity, authorization_id=AUTHORITY)
    receipt = control.make_ready_receipt(
        correctness=correctness, resource=resource, implementation=implementation,
        selected_engine=selected_engine, resource_envelope=resource)
    control.mark_preflight_ready(receipt)
    _write_json(PREFLIGHT_DIR / "preflight_receipt.json", receipt.to_dict())
    summary = {"task_id": TASK_ID, "run_id": run_id, "status": "READY",
               "engine": selected_engine, "training_witnesses": witness,
               "matrix_qualification": engine["matrix_qualification"],
               "engineering_recovery": recovery, "positive_control_metrics": row_metrics,
               "box_loss_qualification": box_check, "projector": projector,
               "qualification_sources": {"reference": str(reference_dir), "engine": str(engine_dir)},
               "resource_observations": resource, "scientific_updates": 0}
    _write_json(PREFLIGHT_DIR / "preflight_summary.json", summary)
    shutil.copy2(config_path, PREFLIGHT_DIR / "run_config.yaml")
    shutil.copy2(Path(__file__).resolve(), PREFLIGHT_DIR / "runner_source.py")
    return {"status": "READY", "run_id": run_id, "engine": selected_engine}


def _get_ready(config: Mapping[str, Any], inputs: Mapping[str, Any],
               population: Mapping[str, Any]) -> tuple[dict[str, Any], PreflightReceipt]:
    identity_payload = _json(PREFLIGHT_DIR / "preflight_identity.json")
    identity = ResearchRunIdentity.from_mapping(identity_payload["run_identity"])
    receipt = PreflightReceipt.from_mapping(_json(PREFLIGHT_DIR / "preflight_receipt.json"))
    summary = _json(PREFLIGHT_DIR / "preflight_summary.json")
    required_evidence = (
        PREFLIGHT_DIR / "fixed_population_and_schedule.pt",
        PREFLIGHT_DIR / "mapper_initialization.pt",
        PREFLIGHT_DIR / "training_witnesses.pt",
        PREFLIGHT_DIR / "engineering_recovery.pt",
        PREFLIGHT_DIR / "run_config.yaml",
        PREFLIGHT_DIR / "runner_source.py",
    )
    _require(summary.get("status") == "READY" and identity_payload.get("task_id") == TASK_ID
             and not (PREFLIGHT_DIR / "preflight_failure.json").exists()
             and all(path.is_file() for path in required_evidence)
             and receipt.authorization_id == AUTHORITY and receipt.status == "READY"
             and receipt.source_semantics_id == identity.execution_source_semantics_id
             and summary.get("run_id") == identity.run_id
             and summary.get("engine") in {"response_matrix", "full_optics"}
             and receipt.selected_engine == summary.get("engine"),
             "formal execute requires the complete consumed READY receipt")
    _require(torch.cuda.is_available(), "G1 READY evidence consumption requires CUDA")
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference, _ = _consume_training_witnesses(PREFLIGHT_DIR, inputs, population,
                                               device=device)
    engine, _ = _consume_engine_qualification(PREFLIGHT_DIR, inputs)
    _require(len(reference) == TRAIN_ROWS
             and engine["matrix_qualification"]["selected_engine"] == receipt.selected_engine
             and summary.get("training_witnesses", {}).get("count") == TRAIN_ROWS
             and summary.get("qualification_sources", {}).get("reference")
             and summary.get("qualification_sources", {}).get("engine"),
             "READY consumer could not reconsume the executed reference and engine qualification evidence")
    del reference
    return identity_payload, receipt


def _open_attempt(config: Mapping[str, Any], resume: bool,
                  identity_payload: Mapping[str, Any], receipt: PreflightReceipt) -> ResearchRunControl:
    identity = ResearchRunIdentity.from_mapping(identity_payload["run_identity"])
    control = ResearchRunControl(identity=identity, authorization_id=AUTHORITY)
    control.mark_preflight_ready(receipt)
    control.open_scientific_attempt(ATTEMPT_DIR, f"{identity.run_id}:attempt_01", receipt,
                                    resume=resume)
    return control


def _train(config: Mapping[str, Any], inputs: Mapping[str, Any], population: Mapping[str, Any],
           checkpoint_path: Path | None) -> dict[str, Any]:
    identity_payload, receipt = _get_ready(config, inputs, population)
    run_id = identity_payload["run_id"]
    device = torch.device("cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    schedule = _build_schedule(config["training"]["schedule_seed"])
    checkpoints = ATTEMPT_DIR / "checkpoints"
    mapper = _mapper(inputs, device)
    optimizer = _optimizer(mapper, config)
    engine = str(receipt.selected_engine)
    if checkpoint_path is None:
        _require(not ATTEMPT_DIR.exists(), "refusing to overwrite an existing attempt", FileExistsError)
        _open_attempt(config, resume=False, identity_payload=identity_payload, receipt=receipt)
        start, history = 0, []
        torch.manual_seed(config["training"]["rng_seed"])
        torch.cuda.manual_seed_all(config["training"]["rng_seed"])
        ATTEMPT_DIR.mkdir(parents=True, exist_ok=True)
        checkpoints.mkdir(parents=True, exist_ok=True)
        shutil.copytree(PREFLIGHT_DIR, ATTEMPT_DIR / "preflight_evidence")
        _save_torch(ATTEMPT_DIR / "fixed_population_and_schedule.pt", {
            "training": population["train"], "development": population["validation"],
            "schedule": schedule})
        _save_torch(ATTEMPT_DIR / "mapper_initialization.pt", {
            "source_checkpoint": str(inputs["source_mapper_path"]),
            "mapper_state": _copy_state(mapper.state_dict()), "new_adam_state": {}})
        shutil.copy2(PREFLIGHT_DIR / "preflight_receipt.json", ATTEMPT_DIR / "preflight_receipt.json")
        shutil.copy2(Path(__file__).resolve(), ATTEMPT_DIR / "runner_source.py")
        shutil.copy2(DEFAULT_CONFIG, ATTEMPT_DIR / "run_config.yaml")
        torch.manual_seed(config["training"]["rng_seed"])
        torch.cuda.manual_seed_all(config["training"]["rng_seed"])
        _save_checkpoint(checkpoints / "checkpoint_step_00000.pt", mapper, optimizer,
                         schedule, history, 0, inputs, engine, run_id)
    else:
        _require(ATTEMPT_DIR.is_dir() and checkpoint_path.is_file()
                 and checkpoint_path.is_relative_to((ATTEMPT_DIR / "checkpoints").resolve()),
                 "resume requires an explicit checkpoint inside the same attempt", ValueError)
        task_state_path = ATTEMPT_DIR / "task_state.json"
        task_state = _json(task_state_path) if task_state_path.is_file() else {}
        _require(task_state.get("state") not in {RUN_COMPLETED, RUN_VALID_FAILED, "CLOSED"},
                 "a closed scientific attempt cannot be reopened")
        _open_attempt(config, resume=True, identity_payload=identity_payload, receipt=receipt)
        saved_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        engine = str(saved_payload.get("engine"))
        _require(engine in {"response_matrix", "full_optics"} and engine == receipt.selected_engine,
                 "resume checkpoint has an unknown fixed compute engine")
        start, history, saved = _restore(checkpoint_path, mapper, optimizer, schedule, engine)
        _require(len(history) == start and saved.get("source_lineage_mapper_updates") == 10000
                 and saved.get("lineage_mapper_updates") == 10000 + start,
                 "explicit resume state has inconsistent lineage or history")
        _atomic_csv(ATTEMPT_DIR / "training_history.csv", history) if history else None
    if start == 0:
        _observe_2048(inputs, population["train"], mapper, device, 0)
        _observe_2048(inputs, population["validation"], mapper, device, 0)
    elif start in {5000, 10000, UPDATE_BUDGET}:
        for cohort in (population["train"], population["validation"]):
            observation = ATTEMPT_DIR / "observations" / (
                f"step_{start:05d}_{cohort['role']}_coefficient_2048.json")
            if not observation.is_file():
                _observe_2048(inputs, cohort, mapper, device, start)
    history_flushed = start
    model = _make_runtime_model(inputs, 2048, device)
    if checkpoint_path is None:
        torch.manual_seed(config["training"]["rng_seed"])
        torch.cuda.manual_seed_all(config["training"]["rng_seed"])
    else:
        torch.set_rng_state(saved["rng_state"]["cpu"])
        if "cuda" in saved["rng_state"]:
            torch.cuda.set_rng_state_all(saved["rng_state"]["cuda"])
    fixed = {0, 5000, 10000, UPDATE_BUDGET}
    checkpoints.mkdir(parents=True, exist_ok=True)
    for step in range(start, UPDATE_BUDGET):
        indices = schedule[step]
        result = _update(mapper, optimizer, model, inputs,
                         population["train"]["coefficients"][indices], device,
                         engine, audit=step == start)
        result.update({"completed_update": step + 1, "sample_indices": indices.tolist(),
                       "labels": list(range(CLASS_COUNT)), "loss_is_pre_update": True,
                       "target_exposure_count": step // 20 + 1})
        history.append(result)
        if len(history) % 100 == 0:
            _append_csv(ATTEMPT_DIR / "training_history.csv", history[history_flushed:])
            history_flushed = len(history)
            _write_json(ATTEMPT_DIR / "progress.json", {
                "task_id": TASK_ID, "run_id": run_id, "completed_updates": len(history),
                "authorized_updates": UPDATE_BUDGET, "engine": engine,
                "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        completed = step + 1
        if completed % 500 == 0 or completed in fixed:
            latest_path = checkpoints / "latest.pt"
            previous_path = checkpoints / "previous.pt"
            if latest_path.is_file():
                shutil.copy2(latest_path, previous_path)
            _save_checkpoint(checkpoints / "latest.pt", mapper, optimizer, schedule,
                             history, completed, inputs, engine, run_id)
            if completed in fixed:
                shutil.copy2(latest_path, checkpoints / f"checkpoint_step_{completed:05d}.pt")
            if history_flushed < len(history):
                _append_csv(ATTEMPT_DIR / "training_history.csv", history[history_flushed:])
                history_flushed = len(history)
            if completed in {5000, 10000}:
                _observe_2048(inputs, population["train"], mapper, device, completed)
                _observe_2048(inputs, population["validation"], mapper, device, completed)
            _write_json(ATTEMPT_DIR / "task_state.json", {
                "task_id": TASK_ID, "run_id": run_id, "state": RUN_SCIENTIFIC_ACTIVE,
                "completed_updates": completed, "authorized_updates": UPDATE_BUDGET,
                "source_mapper_updates": 5000, "source_lineage_mapper_updates": 10000,
                "lineage_mapper_updates": 10000 + completed,
                "adam_steps": completed, "engine": engine})
    _save_checkpoint(checkpoints / "terminal_checkpoint.pt", mapper, optimizer, schedule,
                     history, UPDATE_BUDGET, inputs, engine, run_id)
    shutil.copy2(checkpoints / "terminal_checkpoint.pt", checkpoints / "checkpoint_step_15000.pt")
    return {"task_id": TASK_ID, "run_id": run_id, "completed_updates": len(history),
            "lineage_mapper_updates": 10000 + len(history), "attempt_dir": str(ATTEMPT_DIR)}


def _measure(config: Mapping[str, Any], inputs: Mapping[str, Any],
             population: Mapping[str, Any]) -> dict[str, Any]:
    _require(ATTEMPT_DIR.is_dir() and not (ATTEMPT_DIR / "final_decision.json").is_file(),
             "measurement can only complete an open attempt; closed runs use verify-saved")
    checkpoint = torch.load(ATTEMPT_DIR / "checkpoints" / "terminal_checkpoint.pt",
                            map_location="cpu", weights_only=False)
    _require(checkpoint.get("completed_updates") == UPDATE_BUDGET
             and checkpoint.get("task_id") == TASK_ID
             and checkpoint.get("history_count") == UPDATE_BUDGET
             and checkpoint.get("source_lineage_mapper_updates") == 10000
             and checkpoint.get("lineage_mapper_updates") == 10000 + UPDATE_BUDGET
             and checkpoint.get("run_id"), "measurement requires the exact terminal checkpoint")
    device = torch.device("cuda:0")
    mapper = _mapper(inputs, device)
    mapper.load_state_dict(checkpoint["mapper_state"], strict=True)
    mapper.eval()
    source_delta = _compare_parameter_states(checkpoint["mapper_state"], mapper.state_dict())
    _require(source_delta == 0.0, "loaded terminal mapper differs from its checkpoint state")
    _save_torch(ATTEMPT_DIR / "measurement_mapper_binding.pt", {
        "schema": f"{TASK_ID}.measurement_mapper_binding.v1", "task_id": TASK_ID,
        "run_id": checkpoint["run_id"], "checkpoint_step": UPDATE_BUDGET,
        "loaded_mapper_state": _copy_state(mapper.state_dict()),
        "selected_engine": checkpoint["engine"],
        "single_sample_mapper_producer": True})
    rows = {}
    outputs = {}
    for role in ("train", "validation"):
        role_population = population[role]
        role_rows, measured = _measure_cohort(inputs, role_population, mapper, device,
                                              save_outputs=True, tag=role)
        rows[role] = role_rows
        outputs[role] = measured
        _atomic_csv(ATTEMPT_DIR / f"{role}_terminal_rows.csv", role_rows)
        _write_json(ATTEMPT_DIR / f"{role}_terminal_summary.json", _summary(role_rows))
        observation_errors = np.asarray([row["coefficient_2048"] for row in role_rows], dtype=np.float64)
        observation = {
            "step": UPDATE_BUDGET, "cohort_rows": len(role_rows),
            "coefficient_relative_error_mean": float(observation_errors.mean()),
            "coefficient_relative_error_rms": float(np.sqrt(np.mean(np.square(observation_errors)))),
            "coefficient_relative_error_median": float(np.median(observation_errors)),
            "coefficient_relative_error_max": float(observation_errors.max()),
            "role": role_population["role"], "full_optics_2048": True,
        }
        stem = f"step_{UPDATE_BUDGET:05d}_{role_population['role']}_coefficient_2048"
        _write_json(ATTEMPT_DIR / "observations" / f"{stem}.json", observation)
        _save_torch(ATTEMPT_DIR / "observations" / f"{stem}.pt", {
            "task_id": TASK_ID, "step": UPDATE_BUDGET, "role": role_population["role"],
            "source_indices": role_population["source_indices"].clone(),
            "labels": role_population["labels"].clone(),
            "proposal_ids": role_population["proposal_ids"].clone(),
            "target_coefficients": role_population["coefficients"].clone(),
            "commands": measured[2048]["commands"],
            "actual_coefficients_2048": measured[2048]["coefficients"],
            "summary": observation,
        })
    # Measure step zero on the fixed development cohort for paired semantic attribution.
    initial = _mapper(inputs, device)
    initialization = torch.load(ATTEMPT_DIR / "mapper_initialization.pt", map_location="cpu",
                                weights_only=False)
    initial.load_state_dict(initialization["mapper_state"], strict=True)
    _, step0_outputs = _measure_cohort(inputs, population["validation"], initial, device,
                                       save_outputs=True, tag="validation_step00000")
    terminal_outputs = outputs["validation"]
    semantic_raw = {"direct_decoder": step0_outputs[2048]["direct_raw_cmos"],
                    "frozen_v2": step0_outputs[2048]["raw_cmos"],
                    "mapped_candidate": terminal_outputs[2048]["raw_cmos"]}
    sem, images = _semantic_summary(inputs, population["validation"], semantic_raw)
    _write_json(ATTEMPT_DIR / "development_semantic_auxiliary.json", sem)
    _save_torch(ATTEMPT_DIR / "development_semantic_readouts.pt", {
        "labels": population["validation"]["labels"], "images": images})
    _save_contact_sheet(ATTEMPT_DIR / "development_semantic_examples.png", images,
                        population["validation"]["labels"])
    summaries = {role: _summary(value) for role, value in rows.items()}
    passed = all(summaries[role]["passed"] == len(rows[role]) for role in rows)
    status = "VALID_COMPLETED" if passed else "VALID_FAILED"
    _write_json(ATTEMPT_DIR / "final_decision.json", {
        "task_id": TASK_ID, "run_id": checkpoint["run_id"], "state": status,
        "evidence_tier": "diagnostic_only", "new_updates": UPDATE_BUDGET,
        "fixed_joint_gate": {role: summaries[role] for role in summaries},
        "all_400_rows_passed": passed,
        "semantic_metrics_are_auxiliary": True,
        "human_readability_measured": False,
        "selected_engine": checkpoint["engine"],
        "mapper_terminal_state_bound": source_delta == 0.0,
        "independent_holdout": False,
    })
    _write_json(ATTEMPT_DIR / "task_state.json", {
        "task_id": TASK_ID, "run_id": checkpoint["run_id"], "state": RUN_COMPLETED if passed else RUN_VALID_FAILED,
        "completed_updates": UPDATE_BUDGET, "authorized_updates": UPDATE_BUDGET,
        "phase": "CLOSED", "terminal_measurement_complete": True})
    _write_final_report(ATTEMPT_DIR, {
        "state": status, "new_updates": UPDATE_BUDGET,
        "lineage_mapper_updates": checkpoint["lineage_mapper_updates"],
        "all_400_rows_passed": passed,
    }, summaries, sem)
    _write_transfer_manifest(ATTEMPT_DIR, checkpoint["run_id"])
    return {"state": status, "all_400_rows_passed": passed, "summaries": summaries}


def _verify_saved(evidence_dir: Path, config: Mapping[str, Any], inputs: Mapping[str, Any],
                  population: Mapping[str, Any]) -> dict[str, Any]:
    allowed_roots = {ATTEMPT_DIR.resolve(), _resolve(config["output"]["windows_backup_root"]).resolve()}
    _require(evidence_dir.resolve() in allowed_roots,
             "saved evidence verifier requires the explicit cloud attempt or its Windows backup", ValueError)
    decision = _json(evidence_dir / "final_decision.json")
    checkpoint = torch.load(evidence_dir / "checkpoints" / "terminal_checkpoint.pt",
                            map_location="cpu", weights_only=False)
    history = checkpoint.get("history", [])
    schedule = _build_schedule(config["training"]["schedule_seed"])
    initialization = torch.load(evidence_dir / "mapper_initialization.pt", map_location="cpu",
                                weights_only=False)
    binding = torch.load(evidence_dir / "measurement_mapper_binding.pt", map_location="cpu",
                         weights_only=False)
    optimizer_state = checkpoint.get("optimizer_state", {})
    adam_steps = [int(torch.as_tensor(state["step"]).item())
                  for state in optimizer_state.get("state", {}).values()
                  if isinstance(state, Mapping) and "step" in state]
    _require(checkpoint.get("task_id") == TASK_ID and checkpoint.get("completed_updates") == UPDATE_BUDGET
             and checkpoint.get("run_id") == decision.get("run_id")
             and len(history) == UPDATE_BUDGET
             and checkpoint.get("history_count") == UPDATE_BUDGET
             and checkpoint.get("source_lineage_mapper_updates") == 10000
             and checkpoint.get("lineage_mapper_updates") == 10000 + UPDATE_BUDGET
             and decision.get("new_updates") == UPDATE_BUDGET
             and checkpoint.get("engine") == decision.get("selected_engine")
             and torch.equal(checkpoint.get("schedule"), schedule)
             and checkpoint.get("schedule_cursor") == UPDATE_BUDGET
             and len(adam_steps) == 4 and adam_steps == [UPDATE_BUDGET] * 4
             and initialization.get("source_checkpoint") == str(inputs["source_mapper_path"])
             and initialization.get("new_adam_state") == {}
             and _compare_parameter_states(initialization["mapper_state"],
                                           inputs["source_mapper"]["mapper_state"]) == 0.0
             and binding.get("schema") == f"{TASK_ID}.measurement_mapper_binding.v1"
             and binding.get("task_id") == TASK_ID and binding.get("run_id") == checkpoint.get("run_id")
             and binding.get("checkpoint_step") == UPDATE_BUDGET
             and binding.get("selected_engine") == checkpoint.get("engine")
             and _compare_parameter_states(binding["loaded_mapper_state"], checkpoint["mapper_state"]) == 0.0,
             "terminal checkpoint and decision do not bind the fixed 15,000-step task")
    exposure_counts = torch.bincount(schedule.reshape(-1), minlength=TRAIN_ROWS)
    _require(torch.equal(exposure_counts, torch.full((TRAIN_ROWS,), EXPOSURES_PER_TARGET))
             and all(row.get("completed_update") == index + 1
                     and row.get("sample_indices") == schedule[index].tolist()
                     and row.get("labels") == list(range(CLASS_COUNT))
                     and row.get("loss_is_pre_update") is True
                     and row.get("target_exposure_count") == index // 20 + 1
                     for index, row in enumerate(history)),
             "terminal training history does not follow the fixed 750-exposure schedule")
    preflight_dir = evidence_dir / "preflight_evidence"
    preflight_summary = _json(preflight_dir / "preflight_summary.json")
    preflight_identity = _json(preflight_dir / "preflight_identity.json")
    preflight_receipt = PreflightReceipt.from_mapping(
        _json(preflight_dir / "preflight_receipt.json"))
    witness_payload = torch.load(preflight_dir / "training_witnesses.pt",
                                 map_location="cpu", weights_only=False)
    recovery_payload = torch.load(preflight_dir / "engineering_recovery.pt",
                                  map_location="cpu", weights_only=False)
    witnesses = witness_payload.get("witnesses", [])
    _require(preflight_summary.get("status") == "READY"
             and preflight_summary.get("run_id") == checkpoint.get("run_id")
             and preflight_identity.get("task_id") == TASK_ID
             and preflight_identity.get("run_id") == checkpoint.get("run_id")
             and preflight_receipt.status == "READY"
             and preflight_receipt.authorization_id == AUTHORITY
             and preflight_receipt.selected_engine == checkpoint.get("engine")
             and witness_payload.get("task_id") == TASK_ID
             and torch.equal(witness_payload.get("population_ids"), population["train"]["proposal_ids"])
             and len(witnesses) == TRAIN_ROWS
             and all(item.get("index") == index
                     and item.get("proposal_id") == int(population["train"]["proposal_ids"][index])
                     and item.get("all_six_metrics_passed") is True
                     and all(float(item["metrics"][key]) <= GATE_LIMIT for key in METRIC_KEYS)
                     and tuple(item.get("amplitude", torch.empty(0)).shape) == (1, 1, 200, 200)
                     for index, item in enumerate(witnesses))
             and len(preflight_summary.get("positive_control_metrics", [])) == 10
             and all(all(float(row[key]) <= GATE_LIMIT for key in METRIC_KEYS)
                     for row in preflight_summary["positive_control_metrics"])
             and preflight_summary.get("matrix_qualification", {}).get("selected_engine")
                 == checkpoint.get("engine"),
             "copied preflight receipt, fixed training witnesses, or positive controls are incomplete")
    recovery = recovery_payload.get("summary", {})
    direct4, replay4 = recovery_payload.get("direct_step4", {}), recovery_payload.get("replayed_step4", {})
    recovery_model_delta = _compare_parameter_states(direct4.get("mapper_state", {}),
                                                     replay4.get("mapper_state", {}))
    _, recovery_adam_delta = fit20._tree_max_delta(direct4.get("optimizer_state", {}),
                                                   replay4.get("optimizer_state", {}))
    _require(recovery_payload.get("task_id") == TASK_ID
             and direct4.get("batch") is not None
             and torch.equal(direct4["batch"], schedule[3])
             and torch.equal(replay4.get("batch"), schedule[3])
             and direct4.get("history") == replay4.get("history")
             and _rng_equal(direct4.get("rng_state", {}), replay4.get("rng_state", {}))
             and recovery_model_delta <= RECOVERY_TOLERANCE
             and recovery_adam_delta <= RECOVERY_TOLERANCE
             and abs(float(direct4.get("loss", math.inf)) - float(replay4.get("loss", -math.inf)))
                 <= RECOVERY_TOLERANCE
             and recovery.get("selected_engine") == checkpoint.get("engine")
             and recovery.get("successful_engineering_adam_calls") == 9
             and preflight_summary.get("box_loss_qualification", {}).get("gradient_descent_direction_passed")
                 is True,
             "copied independent fourth-step recovery or executed box-loss checks are inconsistent")
    groups = optimizer_state.get("param_groups", [])
    _require(len(groups) == 1 and groups[0].get("lr") == 1.0e-4
             and tuple(groups[0].get("betas", ())) == (0.9, 0.999)
             and groups[0].get("eps") == 1.0e-8 and groups[0].get("weight_decay") == 0.0,
             "terminal Adam hyperparameters differ from the fixed plan")
    cpu_mapper = projected._new_mapper(torch.device("cpu"), checkpoint["mapper_state"]).eval()
    cpu_matrix = inputs["matrix"].detach().cpu().double()
    cpu_inverse = inputs["inverse"].detach().cpu().double()
    cpu_scale = inputs["scale"].detach().cpu().double()
    response_matrix = inputs["response_matrix"].detach().cpu().to(torch.complex128)
    command_replay_max_abs = 0.0
    response_replay_max_relative = 0.0

    def compare_payload(expected: Any, actual: Any, location: str) -> None:
        if isinstance(expected, Mapping):
            _require(isinstance(actual, Mapping) and set(expected) == set(actual),
                     f"saved summary keys differ at {location}")
            for key, value in expected.items():
                compare_payload(value, actual[key], f"{location}.{key}")
        elif isinstance(expected, list):
            _require(isinstance(actual, list) and len(expected) == len(actual),
                     f"saved summary list differs at {location}")
            for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
                compare_payload(left, right, f"{location}[{index}]")
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            _require(isinstance(actual, (int, float)) and abs(float(expected) - float(actual)) <= 1.0e-14,
                     f"saved summary numeric value differs at {location}")
        else:
            _require(expected == actual, f"saved summary value differs at {location}")

    recomputed = {}
    for role in ("train", "validation"):
        rows = []
        shards = sorted((evidence_dir / "measurements" / role).glob("shard_*.pt"))
        expected = TRAIN_ROWS if role == "train" else DEV_ROWS
        _require(len(shards) == expected // 10, f"{role} measurement shards are incomplete")
        for shard_index, path in enumerate(shards):
            shard = torch.load(path, map_location="cpu", weights_only=False)
            base = population[role]
            expected_start = shard_index * 10
            expected_stop = expected_start + 10
            _require(shard.get("task_id") == TASK_ID and shard.get("cohort_role") == base["role"]
                     and shard.get("start") == expected_start and shard.get("stop") == expected_stop
                     and torch.equal(shard.get("labels"), base["labels"][expected_start:expected_stop])
                     and torch.equal(shard.get("indices"), base["source_indices"][shard["start"]:shard["stop"]])
                     and torch.equal(shard.get("proposal_ids"), base["proposal_ids"][shard["start"]:shard["stop"]])
                     and torch.equal(shard.get("target_coefficients"),
                                     base["coefficients"][shard["start"]:shard["stop"]])
                     and tuple(shard.get("commands", torch.empty(0)).shape) == (10, 1, 200, 200)
                     and torch.isfinite(shard["commands"]).all().item()
                     and float(shard["commands"].min()) >= 0.0 and float(shard["commands"].max()) <= 1.0,
                     f"{role} measurement shard does not match its fixed source population")
            for grid in GRIDS:
                output = shard.get("outputs", {}).get(grid, {})
                coefficients = output.get("coefficients")
                raw_cmos = output.get("raw_cmos")
                reference = shard.get("direct_raw_cmos", {}).get(grid)
                _require(torch.is_tensor(coefficients) and tuple(coefficients.shape) == (10, PORT_COUNT)
                         and torch.isfinite(coefficients.real).all().item()
                         and torch.isfinite(coefficients.imag).all().item()
                         and torch.is_tensor(raw_cmos) and raw_cmos.ndim == 4
                         and tuple(raw_cmos.shape[:2]) == (10, 1)
                         and torch.isfinite(raw_cmos).all().item()
                         and torch.is_tensor(reference) and tuple(reference.shape) == tuple(raw_cmos.shape)
                         and torch.isfinite(reference).all().item(),
                         f"{role} {grid} optical or direct-reference arrays are incomplete/nonfinite")
            targets = shard["target_coefficients"].to(torch.complex128)
            for local in range(10):
                features = projected._normalized_features(targets[local:local + 1], cpu_scale,
                                                          torch.device("cpu"))
                with torch.no_grad():
                    replayed, _ = projected._map_amplitude(
                        cpu_mapper, features, targets[local:local + 1], cpu_matrix, cpu_inverse,
                        cpu_scale)
                command_delta = float((replayed[0].cpu() - shard["commands"][local]).abs().max())
                command_replay_max_abs = max(command_replay_max_abs, command_delta)
                _require(command_delta <= 1.0e-5,
                         f"{role} saved command is not produced by the terminal mapper at source index "
                         f"{int(shard['indices'][local])}")
            predicted = _forward_matrix(shard["commands"], response_matrix)
            actual_2048 = shard["outputs"][2048]["coefficients"].to(torch.complex128)
            response_delta = torch.linalg.vector_norm(predicted - actual_2048, dim=1) / torch.linalg.vector_norm(
                actual_2048, dim=1).clamp_min(1.0e-30)
            response_replay_max_relative = max(response_replay_max_relative, float(response_delta.max()))
            _require(float(response_delta.max()) <= 1.0e-6,
                     f"{role} saved 2048 coefficients do not match the frozen response for the saved commands")
            for local in range(shard["stop"] - shard["start"]):
                outputs = {grid: {key: value[local:local + 1] for key, value in shard["outputs"][grid].items()}
                           for grid in GRIDS}
                direct = {grid: shard["direct_raw_cmos"][grid][local:local + 1] for grid in GRIDS}
                target = shard["target_coefficients"][local:local + 1].to(torch.complex128)
                metrics = projected._error_row(outputs, target, direct,
                                               int(shard["indices"][local]))
                rows.append({"index": int(shard["indices"][local]),
                             "label": int(shard["labels"][local]),
                             "proposal_id": int(shard["proposal_ids"][local]),
                             **metrics, "row_passed": all(metrics[key] <= GATE_LIMIT for key in METRIC_KEYS)})
        summary = _summary(rows)
        current = _json(evidence_dir / f"{role}_terminal_summary.json")
        compare_payload(summary, current, role)
        recomputed[role] = summary
    _require(all(recomputed[role]["passed"] == len(population[role]["labels"])
                 for role in recomputed)
             == bool(decision.get("all_400_rows_passed")),
             "terminal decision disagrees with independently recomputed raw measurements")
    for role in ("train", "validation"):
        compare_payload(recomputed[role], decision["fixed_joint_gate"][role],
                        f"decision.fixed_joint_gate.{role}")
    is_windows_backup = evidence_dir.resolve() == _resolve(config["output"]["windows_backup_root"]).resolve()
    transfer = _verify_transfer_sizes(evidence_dir, require_transfer_receipt=is_windows_backup)
    result = {"task_id": TASK_ID, "status": decision["state"], "recomputed": recomputed,
            "transfer": transfer, "checkpoint_bound": True,
            "training_witness_count": len(witnesses),
            "recovery_verified": True,
            "cpu_single_sample_command_replay_max_abs": command_replay_max_abs,
            "response_matrix_to_saved_2048_max_relative": response_replay_max_relative}
    _write_json(evidence_dir / "windows_reconsumption.json", result)
    return result


def _verify_transfer_sizes(attempt_dir: Path, *, require_transfer_receipt: bool = False) -> dict[str, Any]:
    manifest_path = attempt_dir / "transfer_manifest.json"
    _require(manifest_path.is_file(), "exact transfer size manifest is missing", FileNotFoundError)
    manifest = _json(manifest_path)
    _require(manifest.get("schema") == f"{TASK_ID}.transfer.v1"
             and manifest.get("task_id") == TASK_ID and isinstance(manifest.get("files"), list),
             "transfer manifest identity is invalid")
    expected: dict[str, int] = {}
    for item in manifest["files"]:
        relative = Path(str(item.get("path", "")))
        key = relative.as_posix()
        _require(not relative.is_absolute() and ".." not in relative.parts and key not in expected,
                 "transfer manifest has an unsafe or duplicate relative path")
        expected[key] = int(item["bytes"])
    _require(manifest.get("file_count") == len(expected)
             and manifest.get("bytes_total") == sum(expected.values())
             and manifest.get("run_id") == _json(attempt_dir / "final_decision.json").get("run_id"),
             "transfer manifest denominator or run identity differs")
    for relative, size in expected.items():
        path = attempt_dir / Path(relative)
        _require(path.is_file() and path.resolve().is_relative_to(attempt_dir.resolve())
                 and path.stat().st_size == size,
                 f"transfer actual file size mismatch: {relative}")
    manifest_bytes = manifest_path.stat().st_size
    expected["transfer_manifest.json"] = manifest_bytes
    receipt_path = attempt_dir / "transfer_receipt.json"
    if not receipt_path.exists():
        _require(not require_transfer_receipt,
                 "Windows backup requires the exact AutoDL transfer receipt", FileNotFoundError)
        return {"status": "REMOTE_MANIFEST_VERIFIED", "files_verified": len(expected) - 1,
                "bytes_total": sum(expected.values()) - manifest_bytes}
    receipt = _json(receipt_path)
    _require(receipt.get("status") == "COMPLETE" and receipt.get("remote_alias") == "autodl-monthly"
             and Path(str(receipt.get("destination", ""))).resolve() == attempt_dir.resolve()
             and str(receipt.get("remote_run_dir", "")).rstrip("/").endswith(
                 "/outputs/20261029_lightgen_g1_semantic_exposure200_v1/runs/attempt_01"),
             "AutoDL transfer receipt is incomplete or names a different run")
    rows = receipt.get("files", [])
    received = {str(item.get("path")): item for item in rows}
    _require(len(received) == len(rows) and set(received) == set(expected),
             "AutoDL transfer receipt paths differ from the exact run manifest")
    for relative, size in expected.items():
        item = received[relative]
        path = attempt_dir / Path(relative)
        _require(item.get("status") == "COPIED" and item.get("scp_exit_code") == 0
                 and int(item.get("remote_size_bytes", -1)) == size
                 and int(item.get("local_size_bytes", -1)) == size
                 and path.is_file() and path.stat().st_size == size,
                 f"transfer receipt or current file byte count mismatch: {relative}")
    _require(not require_transfer_receipt or receipt.get("status") == "COMPLETE",
             "Windows transfer receipt does not show a complete copy")
    return {"status": "VERIFIED", "files_verified": len(expected),
            "bytes_total": sum(expected.values())}


def main(argv: Sequence[str] | None = None) -> int:
    global PREFLIGHT_DIR
    parser = argparse.ArgumentParser(description="Fixed-population G1 semantic exposure study")
    parser.add_argument("--mode", required=True,
                        choices=("audit-saved", "audit-extension", "qualify-engine",
                                 "qualify-reference", "resume-reference", "preflight",
                                 "qualify-continuous-phase", "resume-continuous-phase",
                                 "verify-qualification",
                                 "continuous-phase-recovery-replay",
                                 "execute", "resume", "measure", "verify-saved",
                                 "recovery-replay"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--preflight-dir", type=Path)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--engine-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--recovery-output", type=Path)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = _load_config(config_path)
    configured_preflight = _resolve(config["output"]["preflight_root"]).resolve()
    PREFLIGHT_DIR = configured_preflight if args.preflight_dir is None else args.preflight_dir.resolve()
    _require(PREFLIGHT_DIR.is_relative_to(RUNS_DIR.resolve()),
             "preflight evidence must stay inside this experiment's runs directory", ValueError)
    reference_dir = (_resolve(config["output"]["reference_qualification_root"])
                     if args.reference_dir is None else args.reference_dir.resolve())
    engine_dir = (_resolve(config["output"]["engine_qualification_root"])
                  if args.engine_dir is None else args.engine_dir.resolve())
    if args.mode == "audit-extension":
        result = _semantic_attribution(config, args.audit_dir or args.output_dir)
    elif args.mode == "qualify-engine":
        output_dir = (_resolve(config["output"]["engine_qualification_root"])
                      if args.output_dir is None else args.output_dir.resolve())
        result = _qualify_engine(config_path, config, output_dir)
    elif args.mode == "continuous-phase-recovery-replay":
        _require(args.checkpoint is not None and args.recovery_output is not None,
                 "continuous-phase-recovery-replay requires explicit checkpoint and output paths",
                 ValueError)
        checkpoint = args.checkpoint.resolve()
        recovery_output = args.recovery_output.resolve()
        _require(checkpoint.is_relative_to(RUNS_DIR.resolve())
                 and recovery_output.is_relative_to(RUNS_DIR.resolve())
                 and not recovery_output.exists(),
                 "continuous phase recovery inputs/outputs must stay in runs and output must be new",
                 ValueError)
        _require(torch.cuda.is_available(), "continuous phase solver recovery requires CUDA")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        inputs = _load_inputs(config)
        population = _load_population(config, inputs)
        result = _continuous_phase_recovery_replay(
            config, inputs, population, checkpoint, recovery_output, torch.device("cuda:0"))
    elif args.mode in {"qualify-continuous-phase", "resume-continuous-phase"}:
        output_dir = (_resolve(config["output"]["reference_continuous_phase_root"])
                      if args.reference_dir is None else args.reference_dir.resolve())
        _require(output_dir.is_relative_to(RUNS_DIR.resolve()),
                 "continuous phase qualification output must stay inside this experiment's runs directory",
                 ValueError)
        resume_checkpoint = None
        if args.mode == "qualify-continuous-phase":
            _require(args.checkpoint is None,
                     "qualify-continuous-phase starts a new qualification and does not accept a checkpoint",
                     ValueError)
            _require(not (output_dir / "reference_qualification_summary.json").is_file()
                     and not (output_dir / "witness_rows").exists(),
                     "terminal continuous phase qualification cannot be reopened", FileExistsError)
        else:
            _require(args.checkpoint is not None,
                     "resume-continuous-phase requires an explicit solver checkpoint", ValueError)
            resume_checkpoint = args.checkpoint.resolve()
            _require(resume_checkpoint.is_relative_to(output_dir.resolve()),
                     "continuous phase checkpoint must be inside the named qualification directory",
                     ValueError)
            _require(not (output_dir / "reference_qualification_summary.json").is_file(),
                     "completed continuous phase qualification is terminal")
        _require(torch.cuda.is_available(), "continuous phase qualification requires production CUDA")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        inputs = _load_inputs(config)
        population = _load_population(config, inputs)
        try:
            if args.mode == "qualify-continuous-phase":
                output_dir.mkdir(parents=True, exist_ok=True)
                positive_root = output_dir / "engineering_positive_controls"
                positive_root.mkdir(exist_ok=True)
                revision = len([path for path in positive_root.glob("revision_*") if path.is_dir()]) + 1
                positive_dir = positive_root / f"revision_{revision:02d}"
                _require(not positive_dir.exists(),
                         "positive-control qualification revision already exists", FileExistsError)
                positive_dir.mkdir()
                positive = _run_ten_positive_controls(inputs, torch.device("cuda:0"))
                _save_torch(positive_dir / "positive_control_replay.pt", positive["payload"])
                _write_json(positive_dir / "positive_control_summary.json", {
                    "task_id": TASK_ID, "status": "TEN_POSITIVE_CONTROLS_PASSED",
                    "count": 10, "response_relative_error_max": positive["response_relative_error_max"],
                    "six_metric_rows": positive["metrics"],
                    "optics_and_evaluators_frozen": positive["optics_and_evaluators_frozen"],
                    "scientific_adam_updates": 0, "engineering_adam_calls": 0})
                _require(_verify_positive_control_revisions(output_dir, inputs) >= 1,
                         "saved production positive-control replay was not consumed")
                predecessor = _reconsume_predecessor_witnesses(
                    config, inputs, population, torch.device("cuda:0"))
                predecessor_path = output_dir / f"predecessor_witness_replay_revision_{revision:02d}.pt"
                _require(not predecessor_path.exists(),
                         "predecessor witness replay revision already exists", FileExistsError)
                _save_torch(predecessor_path, {
                    "schema": f"{TASK_ID}.predecessor_replay.v1", "task_id": TASK_ID,
                    "population_ids": population["train"]["proposal_ids"][:5].clone(),
                    "positive_rows": predecessor[0], "failed_row": predecessor[1],
                    "full_optics_replayed": True, "scientific_updates": 0})
                recovery_summary_path = output_dir / "phase_solver_recovery_summary.json"
                if recovery_summary_path.is_file():
                    _require(_json(recovery_summary_path).get("status")
                             == "PHASE_SOLVER_RECOVERY_PASSED",
                             "existing phase solver recovery evidence did not pass")
                else:
                    _continuous_phase_recovery_preflight(
                        config_path, config, inputs, population, torch.device("cuda:0"),
                        output_dir, predecessor)
            result = _solver_qualification(config, inputs, population, torch.device("cuda:0"),
                                           output_dir=output_dir,
                                           resume_checkpoint=resume_checkpoint,
                                           continuous_phase=True,
                                           preconsumed_predecessor=(predecessor if args.mode ==
                                                                    "qualify-continuous-phase" else None))
        except TrainingWitnessQualificationError as exc:
            if output_dir.is_dir():
                _write_json(output_dir / "reference_qualification_failure.json", {
                    "task_id": TASK_ID, "status": "TRAINING_REFERENCE_INSUFFICIENT",
                    "error_type": type(exc).__name__, "error": str(exc),
                    "scientific_updates": 0, "optimizer_calls": 0})
                progress_path = output_dir / "progress.json"
                progress = _json(progress_path) if progress_path.is_file() else {}
                progress.update({"task_id": TASK_ID, "status": "TRAINING_REFERENCE_INSUFFICIENT",
                                 "error": str(exc)})
                _write_json(progress_path, progress)
            raise
        except Exception as exc:
            if output_dir.is_dir():
                _write_json(output_dir / "reference_qualification_failure.json", {
                    "task_id": TASK_ID, "status": "INVALID_HARNESS",
                    "error_type": type(exc).__name__, "error": str(exc),
                    "scientific_updates": 0, "optimizer_calls": 0})
                progress_path = output_dir / "progress.json"
                progress = _json(progress_path) if progress_path.is_file() else {}
                progress.update({"task_id": TASK_ID, "status": "INVALID_HARNESS",
                                 "error": str(exc)})
                _write_json(progress_path, progress)
            raise
    elif args.mode in {"qualify-reference", "resume-reference"}:
        output_dir = (_resolve(config["output"]["reference_qualification_root"])
                      if args.reference_dir is None else args.reference_dir.resolve())
        _require(output_dir.is_relative_to(RUNS_DIR.resolve()),
                 "reference qualification output must stay inside this experiment's runs directory",
                 ValueError)
        resume_checkpoint = None
        if args.mode == "qualify-reference":
            _require(args.checkpoint is None,
                     "qualify-reference starts a new qualification and does not accept a checkpoint", ValueError)
        else:
            _require(args.checkpoint is not None,
                     "resume-reference requires an explicit solver checkpoint", ValueError)
            resume_checkpoint = args.checkpoint.resolve()
            _require(resume_checkpoint.is_relative_to(output_dir.resolve()),
                     "reference resume checkpoint must be inside the named qualification directory", ValueError)
            summary_path = output_dir / "reference_qualification_summary.json"
            _require(not summary_path.is_file(),
                     "a completed reference qualification, whether passed or insufficient, is terminal")
        _require(torch.cuda.is_available(), "reference qualification requires production CUDA")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        inputs = _load_inputs(config)
        population = _load_population(config, inputs)
        try:
            result = _solver_qualification(config, inputs, population, torch.device("cuda:0"),
                                           output_dir=output_dir,
                                           resume_checkpoint=resume_checkpoint)
        except TrainingWitnessQualificationError as exc:
            if output_dir.is_dir():
                _write_json(output_dir / "reference_qualification_failure.json", {
                    "task_id": TASK_ID, "status": "TRAINING_REFERENCE_INSUFFICIENT",
                    "error_type": type(exc).__name__, "error": str(exc),
                    "scientific_updates": 0, "optimizer_calls": 0})
            raise
        except Exception as exc:
            if output_dir.is_dir():
                _write_json(output_dir / "reference_qualification_failure.json", {
                    "task_id": TASK_ID, "status": "INVALID_HARNESS",
                    "error_type": type(exc).__name__, "error": str(exc),
                    "scientific_updates": 0, "optimizer_calls": 0})
            raise
    elif args.mode == "audit-saved":
        inputs = _load_inputs(config)
        result = _audit_saved(config, inputs, args.audit_dir)
    elif args.mode == "verify-qualification":
        _require(args.reference_dir is not None and args.evidence_dir is not None
                 and args.output_dir is not None,
                 "verify-qualification requires explicit --reference-dir, --evidence-dir and --output-dir",
                 ValueError)
        inputs = _load_inputs(config)
        population = _load_population(config, inputs)
        result = _verify_saved_qualification(args.reference_dir.resolve(), inputs, population,
                                             args.output_dir.resolve(), args.evidence_dir.resolve())
    elif args.mode == "preflight":
        try:
            result = _preflight(config_path, config, reference_dir=reference_dir, engine_dir=engine_dir)
        except Exception as exc:
            if PREFLIGHT_DIR.is_dir():
                status = ("TRAINING_REFERENCE_INSUFFICIENT"
                          if isinstance(exc, TrainingWitnessQualificationError)
                          else "INVALID_HARNESS")
                _write_json(PREFLIGHT_DIR / "preflight_failure.json", {
                    "task_id": TASK_ID, "status": status,
                    "error_type": type(exc).__name__, "error": str(exc),
                    "scientific_updates": 0})
            raise
    elif args.mode == "recovery-replay":
        inputs = _load_inputs(config)
        _require(args.checkpoint is not None and args.recovery_output is not None,
                 "recovery-replay requires explicit checkpoint and output paths", ValueError)
        population = _load_population(config, inputs)
        result = _recovery_replay(config, inputs, population, args.checkpoint.resolve(),
                                  args.recovery_output.resolve())
    else:
        inputs = _load_inputs(config)
        population = _load_population(config, inputs)
        if args.mode == "execute":
            _require(args.checkpoint is None, "execute does not accept a checkpoint", ValueError)
            result = _train(config, inputs, population, None)
        elif args.mode == "resume":
            _require(args.checkpoint is not None, "resume requires an explicit checkpoint", ValueError)
            result = _train(config, inputs, population, args.checkpoint.resolve())
        elif args.mode == "measure":
            result = _measure(config, inputs, population)
        else:
            _require(args.evidence_dir is not None, "verify-saved requires --evidence-dir", ValueError)
            result = _verify_saved(args.evidence_dir.resolve(), config, inputs, population)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
