"""Rebuild a physical-phase G1 prior and run its fixed diagnostic trial."""

from __future__ import annotations

import argparse
import copy
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
)
from scripts.research.lightgen import execute_lightgen_g1_semantic_exposure200_v1 as exposure
from scripts.research.lightgen import execute_lightgen_g1_projected_fit20_box_ab_v1 as fit20
from scripts.research.lightgen import execute_lightgen_g1_slm_input_mapping_pilot_v1 as pilot
from scripts.research.lightgen import execute_lightgen_g1_slm_projected_mapping_v2 as projected
from scripts.research.lightgen import execute_lightgen_matched_architecture_v1 as source_run


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_G1_FIXED_FRAME_SEMANTIC_V1"
AUTHORITY = "USER_AUTHORIZED_LIGHTGEN_G1_FIXED_FRAME_SEMANTIC_20261029"
SCHEMA = "lightgen_g1_fixed_frame_semantic.v1"
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/g1_fixed_frame_semantic_v1.yaml"
RUNS_DIR = ROOT / "outputs/20261029_lightgen_g1_fixed_frame_semantic_v1/runs"
PRIOR_DIR = RUNS_DIR / "prior_reconstruction_repair_01"
QUALIFICATION_DIR = RUNS_DIR / "qualification_repair_01"
PREFLIGHT_DIR = RUNS_DIR / "preflight_01"
ATTEMPT_DIR = RUNS_DIR / "attempt_01"
CLASS_COUNT = 10
PORT_COUNT = 100
PIXEL_COUNT = 40_000
REAL_DIM = 200
TRAIN_PER_CLASS = 20
DEV_PER_CLASS = 20
HOLDOUT_PER_CLASS = 200
TRAIN_ROWS = CLASS_COUNT * TRAIN_PER_CLASS
DEV_ROWS = CLASS_COUNT * DEV_PER_CLASS
HOLDOUT_ROWS = CLASS_COUNT * HOLDOUT_PER_CLASS
UPDATE_BUDGET = 15_000
EXPOSURES = 750
BOX_LAMBDA = 5.0
ALPHAS = (1.0, 0.875, 0.75, 0.625, 0.5)
GRIDS = (2048, 4096)
METRIC_KEYS = exposure.METRIC_KEYS
GATE_LIMIT = 1.0e-4
RECOVERY_TOLERANCE = 5.0e-8
SOURCE_MAPPER_UPDATES = 10_000


def _require(condition: bool, message: str, error: type[Exception] = RuntimeError) -> None:
    if not condition:
        raise error(message)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _registered_source_path_matches(saved_path: str | Path,
                                    configured_path: str | Path) -> bool:
    saved = str(saved_path).replace("\\", "/")
    configured = str(configured_path).replace("\\", "/")
    local_root = ROOT.as_posix().rstrip("/") + "/"
    remote_root = "/root/autodl-tmp/codex-original/"
    for prefix in (local_root, remote_root):
        if configured.startswith(prefix):
            configured = configured[len(prefix):]
            break
    for prefix in (local_root, remote_root):
        if saved.startswith(prefix):
            return saved[len(prefix):] == configured
    return False


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _save_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    torch.save(value, temp)
    os.replace(temp, path)


def _load_config(path: Path) -> dict[str, Any]:
    _require(path.resolve() == DEFAULT_CONFIG.resolve(),
             "fixed-frame study uses its registered config", ValueError)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict) and value.get("schema") == SCHEMA
             and value.get("task_id") == TASK_ID
             and value.get("semantic_authority_id") == AUTHORITY
             and value.get("evidence_tier") == "diagnostic_only",
             "fixed-frame task identity or evidence tier changed", ValueError)
    _require(tuple(value["prior"]["alphas"]) == ALPHAS
             and value["prior"]["per_sample_gauge_rotation"] is False
             and value["prior"]["representation"] ==
             "raw_physical_phase_normalized_by_frozen_modal_scale_interleaved_real_imag",
             "registered raw-phase prior definition changed", ValueError)
    _require(value["prior"]["covariance"] == "sample"
             and value["prior"]["ridge"] == "1.0e-6_mean_covariance_diagonal",
             "raw-phase covariance or ridge rule changed", ValueError)
    _require(value["source"] == {
        "g1_source_config": "configs/research/lightgen/g1_semantic_exposure200_v1.yaml",
        "train_coefficient_cache":
            "outputs/20261021_lightgen_small_conditional_mixture_prior_ab_v1/runs/autodl_backup_attempt_01/train_coefficient_cache.pt",
        "mnist_training_file": "data/MNIST/processed/training.pt",
        "v2_terminal_checkpoint":
            "outputs/20261027_lightgen_g1_slm_projected_mapping_v2/runs/autodl_backup_attempt_01/terminal_checkpoint.pt",
        "fixed_response_projector":
            "outputs/20261027_lightgen_g1_slm_projected_mapping_v2/runs/autodl_backup_attempt_01/preflight_evidence/fixed_response_and_projector.pt",
    }, "fixed source paths changed", ValueError)
    _require(value["population"] == {
        "classes": 10, "train_per_class": 20, "development_per_class": 20,
        "holdout_per_class": 200, "train_seed_base": 202610290010,
        "development_seed_base": 202610290030, "holdout_seed_base": 202610290050,
        "max_proposals_per_class": 1_000_000, "proposal_batch_size": 256,
        "rejection": "original_total_modal_power_only",
    }, "fixed-frame population contract changed", ValueError)
    training = value["training"]
    _require(training["new_updates"] == UPDATE_BUDGET
             and training["logical_batch_size"] == 10
             and training["samples_per_class_per_batch"] == 1
             and training["exposures_per_target"] == EXPOSURES
             and training["learning_rate"] == 1.0e-4
             and training["betas"] == [0.9, 0.999]
             and training["epsilon"] == 1.0e-8 and training["weight_decay"] == 0.0
             and training["box_lambda"] == BOX_LAMBDA
             and training["rng_seed"] == 202610290070
             and training["schedule_seed"] == 202610290071
             and training["engine"] == "response_matrix"
             and training["optimizer"] == "Adam"
             and training["mapper_precision"] == "float32"
             and training["fixed_projection_precision"] == "float64"
             and training["optical_precision"] == "float64_complex128"
             and training["checkpoint_interval_steps"] == 500
             and training["progress_interval_steps"] == 100
             and training["observation_steps"] == [0, 5000, 10000, 15000]
             and training["amp"] is False and training["tf32"] is False,
             "fixed-frame training contract changed", ValueError)
    measurement = value["measurement"]
    _require(measurement["grids"] == [2048, 4096]
             and measurement["relative_error_max"] == GATE_LIMIT
             and measurement["coefficient_alignment"] == "one_global_phase_per_sample"
             and measurement["native_cmos_unit"] == "raw_w_per_m2"
             and measurement["semantic_readout"] == "fixed_gain_clamp_adaptive_average_pool_28"
             and measurement["cmos_normalization"] is False
             and measurement["cmos_power_matching"] is False
             and measurement["cmos_gain_adjustment"] is False
             and measurement["semantic_accuracy_drop_max_absolute"] == 0.02
             and measurement["class_diversity_retention_min"] == 0.8,
             "fixed-frame final measurement definition changed", ValueError)
    expected_output = {
        "experiment_root": RUNS_DIR.relative_to(ROOT).as_posix(),
        "prior_root": PRIOR_DIR.relative_to(ROOT).as_posix(),
        "qualification_root": QUALIFICATION_DIR.relative_to(ROOT).as_posix(),
        "preflight_root": PREFLIGHT_DIR.relative_to(ROOT).as_posix(),
        "attempt_root": ATTEMPT_DIR.relative_to(ROOT).as_posix(),
        "windows_backup_root": (RUNS_DIR / "autodl_backup_attempt_01").relative_to(ROOT).as_posix(),
    }
    _require(value["output"] == expected_output, "fixed-frame output paths changed", ValueError)
    return value


def _load_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    old_config_path = _resolve(config["source"]["g1_source_config"])
    old_config = exposure._load_config(old_config_path)
    inputs = exposure._load_inputs(old_config)
    source = config["source"]
    mapper_path = _resolve(source["v2_terminal_checkpoint"])
    _require(inputs["source_mapper_path"].resolve() == mapper_path.resolve(),
             "loaded mapper path differs from the explicit V2 terminal checkpoint", ValueError)
    mapper = inputs["source_mapper"]
    source_adam_steps = _adam_steps(mapper.get("optimizer_state", {}))
    _require(mapper.get("task_id") == projected.TASK_ID
             and mapper.get("source_mapper_checkpoint") == "original_G1_mapper_step_5000"
             and mapper.get("completed_updates") == 5000
             and mapper.get("checkpoint_step") == 5000
             and mapper.get("source_mapper_updates") == 5000
             and mapper.get("history_count") == 5000
             and mapper.get("optimizer_deepcopied") is True
             and source_adam_steps == [5000] * 4
             and SOURCE_MAPPER_UPDATES == mapper.get("source_mapper_updates") + mapper.get("completed_updates")
             and set(mapper.get("mapper_state", {})) == {
                 "network.0.weight", "network.0.bias", "network.2.weight", "network.2.bias"},
             "starting mapper is not the 10,000-update V2 terminal lineage")
    cache_path = _resolve(source["train_coefficient_cache"])
    data_path = _resolve(source["mnist_training_file"])
    _require(cache_path.is_file() and data_path.is_file(),
             "explicit TRAIN cache or MNIST training source is missing", FileNotFoundError)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    _require(cache.get("schema") == "lightgen_small_conditional_mixture_train_features.v1"
             and cache.get("task_id") == "LIGHTGEN_SMALL_CONDITIONAL_MIXTURE_PRIOR_AB_V1"
             and cache.get("source_checkpoint_step") == 45000
             and tuple(cache.get("rows", torch.empty(0)).shape) == (55_000,)
             and tuple(cache.get("labels", torch.empty(0)).shape) == (55_000,)
             and cache.get("encoder_forward_images") == 55_000,
             "fixed source cache is not the registered 55,000-row TRAIN cache")
    pixels, labels, train_indices, dev_indices = source_run._load_data(data_path)
    _require(torch.equal(cache["rows"].long(), train_indices)
             and torch.equal(cache["labels"].long(), labels.index_select(0, train_indices))
             and torch.equal(torch.bincount(cache["labels"].long(), minlength=10),
                             torch.bincount(labels[train_indices], minlength=10)),
             "cached TRAIN identities/labels do not match the original fixed split")
    projector_path = _resolve(source["fixed_response_projector"])
    projector = torch.load(projector_path, map_location="cpu", weights_only=False)
    _require(torch.equal(projector.get("response_matrix"), inputs["response_matrix"])
             and torch.equal(projector.get("scale"), inputs["scale"])
             and projector.get("real_imag_order") == "per_port_interleaved_real_imag"
             and torch.equal(projector.get("full_right_inverse"), inputs["inverse"]),
             "fixed response/projector differs from the closed G1 source")
    inputs.update({"train_cache": cache, "train_cache_path": cache_path,
                   "training_data_path": data_path, "pixels": pixels, "labels": labels,
                   "train_indices": train_indices, "dev_indices": dev_indices,
                   "projector_path": projector_path})
    return inputs


def _schedule(config: Mapping[str, Any]) -> torch.Tensor:
    schedule = exposure._build_schedule(int(config["training"]["schedule_seed"]))
    _require(tuple(schedule.shape) == (UPDATE_BUDGET, 10),
             "fixed-frame schedule has the wrong update or batch count")
    counts = torch.bincount(schedule.flatten(), minlength=TRAIN_ROWS)
    _require(torch.equal(counts, torch.full((TRAIN_ROWS,), EXPOSURES)),
             "fixed-frame schedule does not expose every training target exactly 750 times")
    return schedule


def _new_mapper(inputs: Mapping[str, Any], device: torch.device,
                state: Mapping[str, torch.Tensor] | None = None) -> torch.nn.Module:
    source_state = inputs["source_mapper"]["mapper_state"] if state is None else state
    mapper = projected._new_mapper(device, source_state)
    mapper.train()
    return mapper


def _optimizer(mapper: torch.nn.Module, config: Mapping[str, Any]) -> torch.optim.Adam:
    training = config["training"]
    return torch.optim.Adam(mapper.parameters(), lr=float(training["learning_rate"]),
                            betas=tuple(training["betas"]), eps=float(training["epsilon"]),
                            weight_decay=float(training["weight_decay"]))


def _frozen_input_snapshot(inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: inputs[key].detach().cpu().clone()
        for key in ("response_matrix", "matrix", "inverse", "scale")
    } | {
        "camera_gain": float(inputs["camera_gain"]),
        "prior": copy.deepcopy(inputs["prior"]),
        "source_mapper": exposure._copy_state(inputs["source_mapper"]["mapper_state"]),
        "evaluator_ed": exposure._copy_state(inputs["evaluator_ed"].state_dict()),
        "evaluator_ec": exposure._copy_state(inputs["evaluator_ec"].state_dict()),
    }


def _assert_frozen_inputs(inputs: Mapping[str, Any], snapshot: Mapping[str, Any]) -> None:
    for key in ("response_matrix", "matrix", "inverse", "scale"):
        _require(torch.equal(inputs[key].detach().cpu(), snapshot[key]),
                 f"frozen optical projection input changed during mapper updates: {key}")
    _require(float(inputs["camera_gain"]) == snapshot["camera_gain"]
             and _compare_states(inputs["source_mapper"]["mapper_state"],
                                 snapshot["source_mapper"]) == 0.0
             and _compare_states(inputs["evaluator_ed"].state_dict(), snapshot["evaluator_ed"]) == 0.0
             and _compare_states(inputs["evaluator_ec"].state_dict(), snapshot["evaluator_ec"]) == 0.0,
             "camera gain, V2 source mapper, or frozen semantic evaluator changed during training")
    _, prior_delta = fit20._tree_max_delta(inputs["prior"], snapshot["prior"])
    _require(prior_delta == 0.0,
             "fitted G1 prior changed during mapper updates")


def _identity(config: Mapping[str, Any], run_id: str) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID, run_id=run_id,
        trajectory_id=f"{run_id}:fixed_frame_semantic",
        semantic_authority_id=AUTHORITY,
        data_role="RAW_PHASE_G1_TRAIN200_DEV200_HOLDOUT2000",
        config_identity=dict(config),
        model_identity={"source_mapper": str(config["source"]["v2_terminal_checkpoint"]),
                        "optics": "frozen_C45k", "prior": "raw_physical_phase_gaussian",
                        "objective": "full_production_Lc_plus_5Lbox"},
        optimizer_identity={"kind": "Adam", "learning_rate": 1.0e-4,
                            "new_updates": UPDATE_BUDGET, "source_optimizer_inherited": False},
        sampler_identity={"train_rows": TRAIN_ROWS, "development_rows": DEV_ROWS,
                          "holdout_rows": HOLDOUT_ROWS, "exposures_per_target": EXPOSURES,
                          "schedule_seed": int(config["training"]["schedule_seed"]),
                          "rng_seed": int(config["training"]["rng_seed"])},
        execution_source_semantics_id=json.dumps(
            {"task_id": TASK_ID, "config": config}, sort_keys=True, ensure_ascii=False)
    )


def _rng_snapshot() -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.get_rng_state().clone()}
    if torch.cuda.is_available():
        state["cuda"] = [value.clone() for value in torch.cuda.get_rng_state_all()]
    return state


def _rng_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (torch.equal(left["cpu"], right["cpu"])
            and ("cuda" not in left and "cuda" not in right
                 or "cuda" in left and "cuda" in right
                 and len(left["cuda"]) == len(right["cuda"])
                 and all(torch.equal(a, b) for a, b in zip(left["cuda"], right["cuda"], strict=True))))


def _compare_states(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    return exposure._compare_parameter_states(left, right)


def _nested_equal_with_tolerance(left: Any, right: Any, tolerance: float) -> bool:
    if isinstance(left, Mapping):
        return (isinstance(right, Mapping) and set(left) == set(right)
                and all(_nested_equal_with_tolerance(left[key], right[key], tolerance) for key in left))
    if isinstance(left, (list, tuple)):
        return (isinstance(right, (list, tuple)) and len(left) == len(right)
                and all(_nested_equal_with_tolerance(a, b, tolerance)
                        for a, b in zip(left, right, strict=True)))
    if isinstance(left, bool) or left is None or isinstance(left, str):
        return left == right
    if isinstance(left, (int, float)):
        return isinstance(right, (int, float)) and abs(float(left) - float(right)) <= tolerance
    return left == right


def _adam_steps(optimizer_state: Mapping[str, Any]) -> list[int]:
    return sorted(int(torch.as_tensor(item["step"]).item())
                  for item in optimizer_state.get("state", {}).values()
                  if isinstance(item, Mapping) and "step" in item)


def _check_box_loss(device: torch.device) -> dict[str, Any]:
    pixels = torch.full((1, PIXEL_COUNT), 0.5, dtype=torch.float64, device=device)
    pixels[0, :5] = torch.tensor([-0.2, 0.0, 0.35, 1.0, 1.2], dtype=torch.float64,
                                 device=device)
    pixels.requires_grad_(True)
    value = fit20._box_loss(pixels)
    gradient, = torch.autograd.grad(value, pixels)
    expected = 2.0 * (pixels.detach() - pixels.detach().clamp(0.0, 1.0)) / PIXEL_COUNT
    _require(float((gradient - expected).abs().max()) <= 1.0e-16,
             "production box-loss derivative differs from the independent analytic reference")
    actual_value = float(value.detach())
    numpy_values = pixels.detach().cpu().numpy()
    independent = float((np.maximum(-numpy_values, 0.0) ** 2
                         + np.maximum(numpy_values - 1.0, 0.0) ** 2).mean())
    _require(abs(actual_value - independent) <= 1.0e-18,
             "production box loss differs from independent NumPy calculation")
    _require(float(fit20._box_loss(pixels.detach() - 100.0 * gradient)) < actual_value,
        "box-loss gradient direction did not lower the production auxiliary loss")
    return {"passed": True, "production_value": actual_value,
            "independent_numpy_value": independent, "analytic_gradient_max_abs_error": 0.0,
            "gradient_descent_direction_passed": True, "box_lambda": BOX_LAMBDA}


def _set_numerics() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _camera_area(inputs: Mapping[str, Any], device: torch.device) -> float:
    model = exposure._make_runtime_model(inputs, 2048, device)
    dy, dx = model.runtime.camera_plane.pixel_pitch_m
    area = float(dy) * float(dx)
    del model
    return area


def _raw_features(coefficients: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    normalized = coefficients.to(torch.complex128) / scale.double().sqrt().unsqueeze(0)
    return torch.view_as_real(normalized).reshape(coefficients.shape[0], REAL_DIM)


def _fit_raw_gaussians(coefficients: torch.Tensor, labels: torch.Tensor,
                       scale: torch.Tensor) -> dict[int, dict[str, torch.Tensor]]:
    fitted: dict[int, dict[str, torch.Tensor]] = {}
    for label in range(CLASS_COUNT):
        selected = coefficients[labels == label]
        _require(selected.ndim == 2 and selected.shape[1] == PORT_COUNT and selected.shape[0] > REAL_DIM,
                 f"class-{label} raw TRAIN population is incomplete")
        features = _raw_features(selected, scale)
        mean = features.mean(0)
        centered = features - mean
        covariance = centered.T @ centered / float(selected.shape[0] - 1)
        ridge = covariance.diag().mean() * 1.0e-6
        regularized = covariance + ridge * torch.eye(REAL_DIM, dtype=torch.float64)
        cholesky, info = torch.linalg.cholesky_ex(regularized)
        _require(int(info) == 0 and torch.isfinite(cholesky).all().item(),
                 f"raw physical-phase class-{label} covariance is not positive definite")
        fitted[label] = {"mean": mean.contiguous(), "covariance": covariance.contiguous(),
                         "ridge": ridge.clone(), "cholesky": cholesky.contiguous(),
                         "sample_count": torch.tensor(selected.shape[0], dtype=torch.long)}
    return fitted


def _sample_class(fit: Mapping[str, torch.Tensor], scale: torch.Tensor, *, alpha: float,
                  label: int, seed: int, count: int, config: Mapping[str, Any]
                  ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    adjusted = {"mean": fit["mean"].double(), "cholesky": fit["cholesky"].double() * alpha}
    return pilot._sample_g1_class(
        adjusted, scale, label=label, seed=seed, count=count,
        proposal_limit=int(config["population"]["max_proposals_per_class"]),
        proposal_batch_size=int(config["population"]["proposal_batch_size"]),
    )


def _cohort(fit: Mapping[int, Mapping[str, torch.Tensor]], scale: torch.Tensor, *,
            alpha: float, role: str, count: int, seed_base: int,
            config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    coefficients, labels, proposal_ids, accounting = [], [], [], {}
    for label in range(CLASS_COUNT):
        values, ids, record = _sample_class(
            fit[label], scale, alpha=alpha, label=label, seed=seed_base + label,
            count=count, config=config)
        coefficients.append(values.to(torch.complex128))
        labels.append(torch.full((count,), label, dtype=torch.long))
        proposal_ids.append(ids)
        accounting[str(label)] = record
    cohort = {"role": role, "coefficients": torch.cat(coefficients).contiguous(),
              "labels": torch.cat(labels), "proposal_ids": torch.cat(proposal_ids),
              "source_indices": torch.arange(CLASS_COUNT * count, dtype=torch.long),
              "seeds": [seed_base + label for label in range(CLASS_COUNT)],
              "samples_per_class": count}
    _require(torch.equal(cohort["labels"], torch.arange(CLASS_COUNT).repeat_interleave(count))
             and bool(torch.all(cohort["coefficients"].abs().square().sum(dim=1)
                                <= pilot.base.MAX_MODAL_POWER_W)),
             f"{role} sampling changed class order or original power rejection")
    return cohort, accounting


def _old_g1_cohort(inputs: Mapping[str, Any], scale: torch.Tensor, *, count: int,
                   seed_base: int, role: str, config: Mapping[str, Any]) -> dict[str, Any]:
    coefficients, labels, proposal_ids = [], [], []
    accounting = {}
    for label in range(CLASS_COUNT):
        values, ids, record = pilot._sample_g1_class(
            inputs["prior"][label], scale, label=label, seed=seed_base + label,
            count=count, proposal_limit=int(config["population"]["max_proposals_per_class"]),
            proposal_batch_size=int(config["population"]["proposal_batch_size"]))
        coefficients.append(values.to(torch.complex128))
        labels.append(torch.full((count,), label, dtype=torch.long))
        proposal_ids.append(ids)
        accounting[str(label)] = record
    return {"role": role, "coefficients": torch.cat(coefficients),
            "labels": torch.cat(labels), "proposal_ids": torch.cat(proposal_ids),
            "source_indices": torch.arange(CLASS_COUNT * count),
            "seeds": [seed_base + label for label in range(CLASS_COUNT)],
            "proposal_accounting": accounting}


def _summary_from_readouts(inputs: Mapping[str, Any], cohort: Mapping[str, Any],
                           readouts: torch.Tensor) -> dict[str, Any]:
    labels = cohort["labels"].long().cpu()
    readouts = readouts.detach().cpu().float()
    _require(tuple(readouts.shape) == (len(labels), 1, 28, 28)
             and torch.isfinite(readouts).all().item()
             and float(readouts.min()) >= 0.0 and float(readouts.max()) <= 1.0,
             "saved semantic readouts have an invalid shape or range")
    evaluators = {"ED": copy.deepcopy(inputs["evaluator_ed"]).cpu().eval(),
                  "EC": copy.deepcopy(inputs["evaluator_ec"]).cpu().eval()}
    states = {key: exposure._snapshot_state(model) for key, model in evaluators.items()}
    scores: dict[str, Any] = {}
    for family, model in evaluators.items():
        logits_batches = []
        with torch.no_grad():
            for start in range(0, len(labels), 10):
                prediction = model(readouts[start:start + 10])
                logits = prediction[-1] if isinstance(prediction, tuple) else prediction
                _require(logits.ndim == 2 and logits.shape == (min(10, len(labels) - start), CLASS_COUNT)
                         and torch.isfinite(logits).all().item(),
                         f"frozen {family} evaluator returned malformed logits")
                logits_batches.append(logits.detach().cpu())
        predictions = torch.cat(logits_batches).argmax(dim=1)
        confusion = torch.bincount(labels * CLASS_COUNT + predictions,
                                   minlength=CLASS_COUNT ** 2).reshape(CLASS_COUNT, CLASS_COUNT)
        recall = confusion.diag().double() / confusion.sum(dim=1).clamp_min(1)
        scores[family] = {"accuracy": float((predictions == labels).double().mean()),
                          "macro_recall": float(recall.mean()),
                          "per_class_recall": [float(value) for value in recall],
                          "confusion_matrix": confusion.tolist(),
                          "predicted_labels": predictions.tolist()}
    for family, model in evaluators.items():
        _require(exposure._compare_parameter_states(states[family], model.state_dict()) == 0.0,
                 f"semantic proxy evaluation modified frozen {family} parameters")
    diversity: dict[str, Any] = {}
    flat = readouts.flatten(1).double()
    duplicate_count = 0
    for label in range(CLASS_COUNT):
        values = flat[labels == label]
        norms = values.square().sum(1)
        distance_sq = (norms[:, None] + norms[None, :] - 2.0 * (values @ values.T)).clamp_min(0.0)
        upper = torch.triu_indices(values.shape[0], values.shape[0], offset=1)
        distances = distance_sq[upper[0], upper[1]].sqrt() / math.sqrt(values.shape[1])
        unique = torch.unique(values, dim=0).shape[0]
        duplicate_count += values.shape[0] - unique
        diversity[str(label)] = {"mean_pairwise_normalized_l2": float(distances.mean()),
                                 "median_pairwise_normalized_l2": float(distances.median()),
                                 "unique_readout_count": int(unique)}
    prediction_agreement = float(np.mean(
        np.asarray(scores["ED"]["predicted_labels"])
        == np.asarray(scores["EC"]["predicted_labels"])))
    return {"camera_pixel_area_m2": 6.25e-12,
            "camera_gain_w_inverse": float(inputs["camera_gain"]),
            "evidence_role": "fixed_development_population_semantic_proxy",
            "human_readability": "not_measured", "novelty": "not_measured",
            "conditions": {"direct_decoder": {"evaluators": scores,
                                                  "ED_EC_prediction_agreement": prediction_agreement}},
            "same_class_diversity": {"direct_decoder": {
                "pairwise_distance_definition": "mean_euclidean_distance_over_sqrt_784_on_28x28_fixed_gain_readout",
                "by_class": diversity, "exact_duplicate_count": int(duplicate_count)}}}


def _direct_decoder_semantics(inputs: Mapping[str, Any], cohort: Mapping[str, Any],
                              device: torch.device
                              ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    model = exposure._make_runtime_model(inputs, 2048, device)
    dy, dx = model.runtime.camera_plane.pixel_pitch_m
    area = float(dy) * float(dx)
    gain = float(inputs["camera_gain"])
    sensor = {key: 0.0 for key in (
        "pixel_count", "raw_sum", "raw_square_sum", "raw_max", "calibrated_sum",
        "calibrated_square_sum", "calibrated_max", "below_zero_count", "above_one_count",
        "clipped_sum", "clipped_square_sum", "zero_count", "one_count", "readout_sum")}
    readout_batches = []
    coefficients = cohort["coefficients"]
    with torch.no_grad():
        for start in range(0, len(coefficients), 10):
            target = coefficients[start:start + 10].to(device=device, dtype=torch.complex128)
            raw = projected._forward_target(model, target)["raw_cmos"].detach().cpu().double()
            calibrated = raw * area * gain
            clipped = calibrated.clamp(0.0, 1.0)
            readout = pilot._readout_for_evaluator(raw, area, gain)
            sensor["pixel_count"] += float(raw.numel())
            sensor["raw_sum"] += float(raw.sum())
            sensor["raw_square_sum"] += float(raw.square().sum())
            sensor["raw_max"] = max(sensor["raw_max"], float(raw.max()))
            sensor["calibrated_sum"] += float(calibrated.sum())
            sensor["calibrated_square_sum"] += float(calibrated.square().sum())
            sensor["calibrated_max"] = max(sensor["calibrated_max"], float(calibrated.max()))
            sensor["below_zero_count"] += float((calibrated < 0.0).sum())
            sensor["above_one_count"] += float((calibrated > 1.0).sum())
            sensor["clipped_sum"] += float(clipped.sum())
            sensor["clipped_square_sum"] += float(clipped.square().sum())
            sensor["zero_count"] += float((clipped == 0.0).sum())
            sensor["one_count"] += float((clipped == 1.0).sum())
            sensor["readout_sum"] += float(readout.double().sum())
            readout_batches.append(readout.cpu())
    readout_values = torch.cat(readout_batches)
    summary = _summary_from_readouts(inputs, cohort, readout_values)
    pixel_count = sensor["pixel_count"]
    image_count = float(len(coefficients) * 28 * 28)
    summary["conditions"]["direct_decoder"]["sensor_statistics"] = {
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
        "readout_mapping": "raw_W_per_m2_x_pixel_area_x_frozen_gain_then_clip_0_1"}
    summary["camera_pixel_area_m2"] = area
    return summary, {"direct_decoder": readout_values}


def _candidate_semantics_from_readouts(inputs: Mapping[str, Any], cohort: Mapping[str, Any],
                                       readouts: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    summary = _summary_from_readouts(inputs, cohort, readouts["direct_decoder"])
    return {"conditions": {"direct_decoder": {
                "evaluators": summary["conditions"]["direct_decoder"]["evaluators"],
                "ED_EC_prediction_agreement": summary["conditions"]["direct_decoder"][
                    "ED_EC_prediction_agreement"]}},
            "same_class_diversity": summary["same_class_diversity"]}


def _semantic_candidate_passes(summary: Mapping[str, Any], baseline: Mapping[str, Any],
                               config: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    condition = summary["conditions"]["direct_decoder"]["evaluators"]
    reference = baseline["conditions"]["direct_decoder"]["evaluators"]
    max_drop = float(config["measurement"]["semantic_accuracy_drop_max_absolute"])
    diversity_min = float(config["measurement"]["class_diversity_retention_min"])
    accuracies = {
        family: {"candidate": float(condition[family]["accuracy"]),
                 "reference": float(reference[family]["accuracy"]),
                 "drop": float(reference[family]["accuracy"] - condition[family]["accuracy"])}
        for family in ("ED", "EC")}
    candidate_diversity = summary["same_class_diversity"]["direct_decoder"]["by_class"]
    reference_diversity = baseline["same_class_diversity"]["direct_decoder"]["by_class"]
    ratios = {str(label): float(candidate_diversity[str(label)]["mean_pairwise_normalized_l2"])
              / max(float(reference_diversity[str(label)]["mean_pairwise_normalized_l2"]), 1.0e-30)
              for label in range(CLASS_COUNT)}
    passed = (all(row["drop"] <= max_drop for row in accuracies.values())
              and all(value >= diversity_min for value in ratios.values()))
    return passed, {"passed": passed, "accuracy": accuracies,
                    "class_diversity_retention_ratio": ratios,
                    "accuracy_drop_limit": max_drop,
                    "class_diversity_retention_min": diversity_min}


def _prepare_prior(config: Mapping[str, Any]) -> dict[str, Any]:
    _require(not PRIOR_DIR.exists(), f"prior output already exists: {PRIOR_DIR}", FileExistsError)
    _require(torch.cuda.is_available(), "TRAIN re-encoding and prior fit require AutoDL CUDA")
    _set_numerics()
    device = torch.device("cuda:0")
    inputs = _load_inputs(config)
    rows = inputs["train_indices"]
    labels = inputs["labels"]
    pixels = inputs["pixels"]
    model = exposure._make_runtime_model(inputs, 2048, device)
    camera_area = float(np.prod(model.runtime.camera_plane.pixel_pitch_m))
    _require(abs(camera_area - 6.25e-12) <= 1.0e-22,
             f"frozen camera pixel area differs from 6.25e-12 m^2: {camera_area:.17g}")
    response = inputs["response_matrix"].to(device=device, dtype=torch.complex128)
    selected = torch.cat([rows[labels[rows] == label][:10] for label in range(CLASS_COUNT)])
    measured_commands, measured_coefficients, measured_matrix_coefficients = [], [], []
    with torch.no_grad():
        for batch_rows in selected.split(10):
            image = source_run._image_batch(pixels, batch_rows, "small", device)
            optical = model.forward_from_image(image)
            command = model.runtime.input_adapter(image).slm_amplitude
            predicted = exposure._forward_matrix(command, response)
            actual = optical["ols_coefficients"].to(torch.complex128)
            delta = torch.linalg.vector_norm(predicted - actual, dim=1) / torch.linalg.vector_norm(
                actual, dim=1).clamp_min(1.0e-30)
            _require(float(delta.max()) <= 1.0e-6,
                     "100-row matrix preflight did not match the real production forward")
            measured_commands.append(command.detach().cpu())
            measured_coefficients.append(actual.detach().cpu())
            measured_matrix_coefficients.append(predicted.detach().cpu())
    replay_error = float(torch.cat([
        torch.linalg.vector_norm(
            command.reshape(command.shape[0], -1).double().to(torch.complex128) @ response.cpu().T
            - actual, dim=1) / torch.linalg.vector_norm(actual, dim=1).clamp_min(1.0e-30)
        for command, actual in zip(measured_commands, measured_coefficients, strict=True)
    ]).max())
    _require(replay_error <= 1.0e-6,
             f"100-image physical SLM amplitude response differs from frozen B: {replay_error:.3g}")

    class_counts = torch.bincount(labels[rows], minlength=CLASS_COUNT)
    amplitude_sums = torch.zeros((CLASS_COUNT, PIXEL_COUNT), dtype=torch.float64)
    coefficient_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, rows.numel(), 10):
            batch_rows = rows[start:start + 10]
            image = source_run._image_batch(pixels, batch_rows, "small", device)
            command = model.runtime.input_adapter(image).slm_amplitude
            coefficients = exposure._forward_matrix(command, response)
            batch_labels = labels[batch_rows]
            for label in range(CLASS_COUNT):
                mask = batch_labels == label
                if bool(mask.any()):
                    amplitude_sums[label] += command[mask.to(device)].reshape(-1, PIXEL_COUNT).double().sum(0).cpu()
            coefficient_parts.append(coefficients.cpu())
            label_parts.append(batch_labels.clone())
    raw_coefficients = torch.cat(coefficient_parts).to(torch.complex128).contiguous()
    train_labels = torch.cat(label_parts).long()
    _require(torch.equal(train_labels, inputs["train_cache"]["labels"].long())
             and torch.equal(train_labels, labels[rows]),
             "55,000 physical-phase coefficient rows drifted from the original TRAIN cache")
    mean_amplitudes = amplitude_sums / class_counts.double()[:, None]
    class_means = []
    for label in range(CLASS_COUNT):
        class_coefficients = raw_coefficients[train_labels == label]
        class_means.append(class_coefficients.mean(0))
        response_mean = mean_amplitudes[label].to(device=device).to(torch.complex128) @ response.T
        mean_error = float(torch.linalg.vector_norm(
            response_mean.cpu() - class_means[-1]) / torch.linalg.vector_norm(class_means[-1]).clamp_min(1.0e-30))
        _require(mean_error <= 1.0e-12,
                 f"class-{label} mean amplitude does not reproduce mean raw coefficient: {mean_error:.3g}")
    prior = _fit_raw_gaussians(raw_coefficients, train_labels, inputs["scale"])
    PRIOR_DIR.mkdir(parents=True)
    source_rows = torch.cat([batch for batch in selected.split(10)])
    _save_torch(PRIOR_DIR / "source_matrix_replay_100.pt", {
        "schema": f"{TASK_ID}.source_matrix_replay_100.v1", "task_id": TASK_ID,
        "source_train_rows": source_rows.clone(), "labels": labels[source_rows].clone(),
        "amplitude_commands": torch.cat(measured_commands),
        "production_optical_coefficients": torch.cat(measured_coefficients),
        "frozen_matrix_coefficients": torch.cat(measured_matrix_coefficients),
        "relative_error_per_row": torch.cat([
            torch.linalg.vector_norm(matrix_coeff - optical_coeff, dim=1)
            / torch.linalg.vector_norm(optical_coeff, dim=1).clamp_min(1.0e-30)
            for matrix_coeff, optical_coeff in zip(measured_matrix_coefficients,
                                                   measured_coefficients, strict=True)]),
        "maximum_relative_error": replay_error,
        "production_response": "C45k image adapter to SLM amplitude to Encoder/OLS"})
    _save_torch(PRIOR_DIR / "raw_physical_phase_train_coefficients.pt", {
        "schema": f"{TASK_ID}.raw_train_coefficients.v1", "task_id": TASK_ID,
        "cache_path": str(inputs["train_cache_path"]), "rows": inputs["train_cache"]["rows"].clone(),
        "labels": train_labels, "coefficients": raw_coefficients,
        "representation": "C45k raw physical phase OLS coefficients; no per-sample gauge rotation",
        "matrix_reencoding_max_relative_error_on_100": replay_error})
    _save_torch(PRIOR_DIR / "physical_phase_gaussian_prior.pt", {
        "schema": f"{TASK_ID}.prior.v1", "task_id": TASK_ID,
        "population_count": int(rows.numel()), "class_counts": class_counts,
        "scale": inputs["scale"].clone(), "mean_amplitudes": mean_amplitudes,
        "mean_coefficients": torch.stack(class_means), "fit": prior,
        "ridge_rule": "1e-6 times mean sample covariance diagonal",
        "feature_order": "port-interleaved real,imag of raw coefficient divided by sqrt(frozen scale)"})

    grid_models = {grid: exposure._make_runtime_model(inputs, grid, device) for grid in GRIDS}
    mean_outputs = {grid: {"coefficients": [], "raw_cmos": [], "direct_raw_cmos": []} for grid in GRIDS}
    with torch.no_grad():
        for label in range(CLASS_COUNT):
            command = mean_amplitudes[label].reshape(1, 1, 200, 200).to(device=device, dtype=torch.float32)
            target = class_means[label].reshape(1, PORT_COUNT).to(device=device, dtype=torch.complex128)
            for grid in GRIDS:
                actual = projected._forward_amplitude(grid_models[grid], command)
                direct = projected._forward_target(grid_models[grid], target)
                mean_outputs[grid]["coefficients"].append(actual["coefficients"].cpu())
                mean_outputs[grid]["raw_cmos"].append(actual["raw_cmos"].cpu())
                mean_outputs[grid]["direct_raw_cmos"].append(direct["raw_cmos"].cpu())
    mean_metrics = []
    for label in range(CLASS_COUNT):
        outputs = {grid: {key: torch.cat(mean_outputs[grid][key])[label:label + 1]
                          for key in ("coefficients", "raw_cmos")} for grid in GRIDS}
        direct = {grid: torch.cat(mean_outputs[grid]["direct_raw_cmos"])[label:label + 1]
                  for grid in GRIDS}
        metrics = projected._error_row(outputs, class_means[label:label + 1], direct, 0)
        mean_metrics.append({"label": label, **metrics})
    _save_torch(PRIOR_DIR / "mean_amplitude_full_optics_replay.pt", {
        "schema": f"{TASK_ID}.mean_amplitude_replay.v1", "task_id": TASK_ID,
        "commands": mean_amplitudes.float().reshape(10, 1, 200, 200),
        "mean_coefficients": torch.stack(class_means), "metrics": mean_metrics,
        "outputs": {grid: {key: torch.cat(values) for key, values in fields.items()}
                    for grid, fields in mean_outputs.items()},
        "camera_pixel_area_m2": camera_area, "camera_gain_w_inverse": inputs["camera_gain"],
        "mean_amplitude_identity_max_relative_error": replay_error,
        "mean_intensity_sqrt_used": False})

    old_dev = _old_g1_cohort(
        inputs, inputs["scale"], count=DEV_PER_CLASS,
        seed_base=int(config["population"]["development_seed_base"]),
        role="old_g1_direct_decoder_reference_dev200", config=config)
    baseline, baseline_images = _direct_decoder_semantics(inputs, old_dev, device)
    _save_torch(PRIOR_DIR / "old_g1_semantic_reference_dev200.pt", {
        "schema": f"{TASK_ID}.old_g1_semantic_reference.v1", "task_id": TASK_ID,
        "population": old_dev,
        "semantic_summary": baseline, "readouts": baseline_images})

    candidate_summaries: list[dict[str, Any]] = []
    old_ids = old_dev["proposal_ids"].reshape(CLASS_COUNT, DEV_PER_CLASS)
    for alpha in ALPHAS:
        train_cohort, train_accounting = _cohort(
            prior, inputs["scale"], alpha=alpha, role="fixed_train200", count=TRAIN_PER_CLASS,
            seed_base=int(config["population"]["train_seed_base"]), config=config)
        dev_cohort, dev_accounting = _cohort(
            prior, inputs["scale"], alpha=alpha, role="fixed_development200", count=DEV_PER_CLASS,
            seed_base=int(config["population"]["development_seed_base"]), config=config)
        semantic, readouts = _direct_decoder_semantics(inputs, dev_cohort, device)
        semantic_pass, semantic_gate = _semantic_candidate_passes(semantic, baseline, config)
        new_ids = dev_cohort["proposal_ids"].reshape(CLASS_COUNT, DEV_PER_CLASS)
        paired_counts = {str(label): int(torch.isin(new_ids[label], old_ids[label]).sum())
                         for label in range(CLASS_COUNT)}
        candidate_dir = PRIOR_DIR / f"alpha_{alpha:.3f}"
        candidate_dir.mkdir()
        _save_torch(candidate_dir / "candidate_population.pt", {
            "schema": f"{TASK_ID}.candidate_population.v1", "task_id": TASK_ID,
            "alpha": alpha, "train": train_cohort, "development": dev_cohort,
            "sampling_accounting": {"train": train_accounting, "development": dev_accounting}})
        _save_torch(candidate_dir / "development_semantic_outputs.pt", {
            "schema": f"{TASK_ID}.development_semantics.v1", "task_id": TASK_ID,
            "population": dev_cohort,
            "readouts": readouts, "semantic_summary": semantic,
            "paired_proposal_counts_by_class": paired_counts,
            "raw_cmos_full_population_retained": False,
            "selection_measurement": "full_2048_frozen_decoder; readout images and sensor statistics retained"})
        _write_json(candidate_dir / "semantic_gate.json", semantic_gate)
        candidate_summaries.append({"alpha": alpha, "semantic_gate": semantic_gate,
                                    "semantic_summary": semantic,
                                    "paired_proposal_counts_by_class": paired_counts,
                                    "train_rows": int(train_cohort["labels"].numel()),
                                    "development_rows": int(dev_cohort["labels"].numel())})
    summary = {"task_id": TASK_ID, "status": "PRIOR_RECONSTRUCTION_COMPLETE",
               "train_rows": 55_000, "official_test_reads": 0,
               "matrix_response_100_image_max_relative_error": replay_error,
               "fixed_camera_pixel_area_m2": 6.25e-12,
               "camera_gain_w_inverse": inputs["camera_gain"],
               "raw_phase_gaussians": 10, "alphas": candidate_summaries,
               "scientific_updates": 0, "mapper_updates": 0, "optical_updates": 0,
               "physical_phase_gauge_hypothesis": "candidate explanation tested, not assumed proven"}
    _write_json(PRIOR_DIR / "prior_reconstruction_summary.json", summary)
    _write_json(PRIOR_DIR / "run_config.json", config)
    return summary


def _load_prior_artifacts(prior_dir: Path = PRIOR_DIR
                          ) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, Any]]:
    prior_path = prior_dir / "physical_phase_gaussian_prior.pt"
    summary_path = prior_dir / "prior_reconstruction_summary.json"
    _require(prior_path.is_file() and summary_path.is_file(),
             "completed raw-phase prior reconstruction is required", FileNotFoundError)
    payload = torch.load(prior_path, map_location="cpu", weights_only=False)
    summary = _json(summary_path)
    _require(payload.get("task_id") == TASK_ID and summary.get("task_id") == TASK_ID
             and summary.get("status") == "PRIOR_RECONSTRUCTION_COMPLETE"
             and set(payload.get("fit", {})) == set(range(CLASS_COUNT)),
             "prior reconstruction evidence identity or class population is invalid")
    return payload["fit"], payload


def _candidate_population_path(alpha: float, prior_dir: Path = PRIOR_DIR) -> Path:
    return prior_dir / f"alpha_{alpha:.3f}" / "candidate_population.pt"


def _load_candidate_population(alpha: float, prior_dir: Path = PRIOR_DIR) -> dict[str, Any]:
    path = _candidate_population_path(alpha, prior_dir)
    _require(path.is_file(), f"fixed candidate population is missing: {path}", FileNotFoundError)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(payload.get("task_id") == TASK_ID and payload.get("alpha") == alpha
             and tuple(payload["train"]["coefficients"].shape) == (TRAIN_ROWS, PORT_COUNT)
             and tuple(payload["development"]["coefficients"].shape) == (DEV_ROWS, PORT_COUNT),
             f"saved alpha={alpha} population identity or size changed")
    return payload


def _model_for_grid(inputs: Mapping[str, Any], grid: int, device: torch.device) -> Any:
    return exposure._make_runtime_model(inputs, grid, device)


def _measure_command(inputs: Mapping[str, Any], command: torch.Tensor,
                     target: torch.Tensor, models: Mapping[int, Any],
                     ) -> tuple[dict[str, Any], dict[int, dict[str, torch.Tensor]]]:
    device = next(models[2048].parameters()).device
    command = command.reshape(1, 1, 200, 200).to(device=device, dtype=torch.float32)
    target = target.reshape(1, PORT_COUNT).to(device=device, dtype=torch.complex128)
    outputs: dict[int, dict[str, torch.Tensor]] = {}
    direct: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for grid in GRIDS:
            outputs[grid] = projected._forward_amplitude(models[grid], command)
            direct[grid] = projected._forward_target(models[grid], target)["raw_cmos"]
    metrics = _metric_row(outputs, target, direct)
    return metrics, {grid: {**outputs[grid], "direct_raw_cmos": direct[grid]} for grid in GRIDS}


def _save_qualification_row(directory: Path, index: int, row: Mapping[str, Any],
                            outputs: Mapping[int, Mapping[str, torch.Tensor]]) -> None:
    row_dir = directory / "witness_rows"
    _save_torch(row_dir / f"row_{index:04d}.pt", {
        "schema": f"{TASK_ID}.witness_row.v1", "task_id": TASK_ID,
        "index": index, "label": int(row["label"]), "proposal_id": int(row["proposal_id"]),
        "target_coefficients": row["target_coefficients"].clone(),
        "amplitude_command": row["amplitude_command"].clone(),
        "solver_source": row["solver_source"], "solver_relative_error": row["solver_relative_error"],
        "solver_iterations": row["solver_iterations"], "phase_rad": row["phase_rad"],
        "metrics": dict(row["metrics"]),
        "all_six_metrics_passed": bool(row["all_six_metrics_passed"]),
        "outputs": {grid: {key: value.detach().cpu().clone() for key, value in fields.items()}
                    for grid, fields in outputs.items()}})


def _initial_fixed_phase_state(response: torch.Tensor, target: torch.Tensor,
                               seed_amplitude: torch.Tensor) -> dict[str, Any]:
    amplitude = seed_amplitude.detach().to(device=response.device, dtype=torch.float64).reshape(-1).clone()
    bounds = exposure._dual_box_error_bounds(response, target, amplitude)
    return {"phase_rad": 0.0, "iteration": 0, "amplitude": amplitude,
            "extrapolated": amplitude.clone(), "momentum": 1.0,
            "best_amplitude": amplitude.clone(),
            "best_error": float(bounds["relative_upper_bound"])}


def _verify_qualification_output(population: Mapping[str, Any],
                                 selected_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(len(selected_records) == TRAIN_ROWS,
             "selected prior lacks one full-optics witness per fixed training target")
    _require(all(bool(row["all_six_metrics_passed"])
                 and all(float(row["metrics"][key]) <= GATE_LIMIT for key in METRIC_KEYS)
                 for row in selected_records),
             "selected prior includes a training target without all six full-optics witnesses")
    for index, row in enumerate(selected_records):
        _require(int(row["index"]) == index
                 and int(row["label"]) == int(population["train"]["labels"][index])
                 and int(row["proposal_id"]) == int(population["train"]["proposal_ids"][index])
                 and torch.equal(row["target_coefficients"], population["train"]["coefficients"][index]),
                 f"training witness does not bind to the fixed coefficient at row {index}")
    return {"count": len(selected_records), "passed": len(selected_records),
            "all_six_full_optics_metrics_passed": True,
            "threshold": GATE_LIMIT, "metric_keys": list(METRIC_KEYS)}


def _qualify(config: Mapping[str, Any], *, resume_checkpoint: Path | None = None) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "prior candidate qualification requires AutoDL CUDA")
    _set_numerics()
    device = torch.device("cuda:0")
    inputs = _load_inputs(config)
    prior, prior_payload = _load_prior_artifacts()
    _require(PRIOR_DIR.is_relative_to(RUNS_DIR) and QUALIFICATION_DIR.is_relative_to(RUNS_DIR),
             "qualification paths must remain inside this study", ValueError)
    summary_path = QUALIFICATION_DIR / "qualification_summary.json"
    _require(not summary_path.is_file(), "terminal qualification cannot be reopened", FileExistsError)
    if resume_checkpoint is None:
        _require(not QUALIFICATION_DIR.exists(),
                 "incomplete qualification requires its explicit solver checkpoint", FileExistsError)
        QUALIFICATION_DIR.mkdir(parents=True)
        completed_candidates: list[dict[str, Any]] = []
        active_alpha_index = 0
        row_start = 0
        solver_state = None
    else:
        checkpoint_path = resume_checkpoint.resolve()
        _require(checkpoint_path.is_file() and checkpoint_path.is_relative_to(QUALIFICATION_DIR.resolve()),
                 "qualification recovery requires an explicit checkpoint in this task directory",
                 ValueError)
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _require(saved.get("task_id") == TASK_ID and saved.get("schema") ==
                 f"{TASK_ID}.qualification_checkpoint.v1",
                 "explicit qualification checkpoint has the wrong task identity", ValueError)
        completed_candidates = list(saved["completed_candidates"])
        active_alpha_index = int(saved["active_alpha_index"])
        row_start = int(saved["next_row_index"])
        solver_state = saved.get("solver_state")
        _require(0 <= active_alpha_index < len(ALPHAS) and 0 <= row_start <= TRAIN_ROWS,
                 "qualification checkpoint cursor is out of range")

    response = inputs["response_matrix"].to(device=device, dtype=torch.complex128)
    models = {grid: _model_for_grid(inputs, grid, device) for grid in GRIDS}
    candidate_results: list[dict[str, Any]] = completed_candidates
    selected: dict[str, Any] | None = None
    active_resume = resume_checkpoint is not None

    def save_cursor(alpha_index: int, next_row: int, state: Mapping[str, Any] | None) -> None:
        _save_torch(QUALIFICATION_DIR / "qualification_checkpoint.pt", {
            "schema": f"{TASK_ID}.qualification_checkpoint.v1", "task_id": TASK_ID,
            "active_alpha_index": alpha_index, "next_row_index": next_row,
            "solver_state": None if state is None else copy.deepcopy(dict(state)),
            "completed_candidates": copy.deepcopy(candidate_results),
            "seeded_update_count": 0, "mapper_updates": 0, "optical_updates": 0})

    for alpha_index, alpha in enumerate(ALPHAS):
        if alpha_index < active_alpha_index:
            continue
        candidate = _load_candidate_population(alpha)
        semantic_gate = _json(PRIOR_DIR / f"alpha_{alpha:.3f}" / "semantic_gate.json")
        candidate_result: dict[str, Any] = {"alpha": alpha,
                                             "semantic_gate_passed": bool(semantic_gate["passed"]),
                                             "semantic_gate": semantic_gate,
                                             "training_witness_rows": []}
        candidate_dir = QUALIFICATION_DIR / f"alpha_{alpha:.3f}"
        if not semantic_gate["passed"]:
            candidate_result.update({"status": "SEMANTIC_QUALIFICATION_FAILED",
                                     "training_witnesses_passed": 0})
            candidate_results.append(candidate_result)
            save_cursor(alpha_index + 1, 0, None)
            continue
        candidate_dir.mkdir(exist_ok=True)
        train = candidate["train"]
        mean_amplitudes = prior_payload["mean_amplitudes"]
        failed_row: dict[str, Any] | None = None
        row_start = row_start if active_resume and alpha_index == active_alpha_index else 0
        if active_resume and alpha_index == active_alpha_index and row_start:
            for previous_index in range(row_start):
                previous_path = candidate_dir / "witness_rows" / f"row_{previous_index:04d}.pt"
                _require(previous_path.is_file(),
                         f"qualification checkpoint prefix lacks committed row {previous_index}",
                         FileNotFoundError)
                previous = torch.load(previous_path, map_location="cpu", weights_only=False)
                candidate_result["training_witness_rows"].append({
                    key: value for key, value in previous.items()
                    if key not in {"target_coefficients", "amplitude_command", "outputs", "schema", "task_id"}})
        for index in range(row_start, TRAIN_ROWS):
            target = train["coefficients"][index].to(device=device, dtype=torch.complex128)
            label = int(train["labels"][index])
            init_amp = mean_amplitudes[label].to(device=device, dtype=torch.float64)
            fixed_initial = solver_state if active_resume and alpha_index == active_alpha_index and index == row_start else None
            if fixed_initial is None:
                fixed_initial = _initial_fixed_phase_state(response, target, init_amp)

            def fista_checkpoint(state: Mapping[str, Any]) -> None:
                save_cursor(alpha_index, index, state)

            fixed = exposure._fista_fixed_phase_solve(
                response, target, 0.0, max_steps=5000, initial_state=fixed_initial,
                checkpoint=fista_checkpoint, checkpoint_interval=500)
            fixed_command = fixed["amplitude"].to(torch.float32).reshape(1, 1, 200, 200)
            fixed_metrics, fixed_outputs = _measure_command(
                inputs, fixed_command, target, models)
            fixed_pass = all(float(fixed_metrics[key]) <= GATE_LIMIT for key in METRIC_KEYS)
            attempts = [{"solver_source": "fixed_physical_phase_zero",
                         "solver_relative_error": float(fixed["relative_error"]),
                         "solver_iterations": int(fixed["iterations"]),
                         "phase_rad": 0.0, "metrics": fixed_metrics,
                         "all_six_metrics_passed": fixed_pass}]
            chosen_command = fixed_command.detach().cpu()
            chosen_metrics, chosen_outputs = fixed_metrics, fixed_outputs
            chosen_solver = attempts[0]
            if not fixed_pass:
                save_cursor(alpha_index, index, None)
                free = reach_result = None
                from scripts.research.lightgen import diagnose_lightgen_frozen_slm_ols_reachability_v1 as reach
                reach_result = reach._bounded_phase_solve(
                    response, target, starts=8, max_steps=5000, stop_error=1.0e-6)
                free_command = reach_result["amplitude"].to(torch.float32).reshape(1, 1, 200, 200)
                free_metrics, free_outputs = _measure_command(
                    inputs, free_command, target, models)
                free_pass = all(float(free_metrics[key]) <= GATE_LIMIT for key in METRIC_KEYS)
                free = {"solver_source": "existing_eight_start_free_public_phase",
                        "solver_relative_error": float(reach_result["relative_error"]),
                        "solver_iterations": int(reach_result["iterations"]),
                        "phase_rad": float(reach_result["phase_rad"]),
                        "metrics": free_metrics, "all_six_metrics_passed": free_pass}
                attempts.append(free)
                if max(free_metrics[key] for key in METRIC_KEYS) < max(fixed_metrics[key] for key in METRIC_KEYS):
                    chosen_command, chosen_metrics, chosen_outputs, chosen_solver = (
                        free_command.detach().cpu(), free_metrics, free_outputs, free)
            passed = all(float(chosen_metrics[key]) <= GATE_LIMIT for key in METRIC_KEYS)
            witness = {"index": index, "label": label,
                       "proposal_id": int(train["proposal_ids"][index]),
                       "target_coefficients": train["coefficients"][index].clone(),
                       "amplitude_command": chosen_command.clone(),
                       "solver_source": chosen_solver["solver_source"],
                       "solver_relative_error": chosen_solver["solver_relative_error"],
                       "solver_iterations": chosen_solver["solver_iterations"],
                       "phase_rad": chosen_solver["phase_rad"],
                       "metrics": chosen_metrics, "attempts": attempts,
                       "all_six_metrics_passed": passed}
            _save_qualification_row(candidate_dir, index, witness, chosen_outputs)
            candidate_result["training_witness_rows"].append({key: value for key, value in witness.items()
                                                               if key not in {"target_coefficients",
                                                                              "amplitude_command"}})
            active_resume = False
            solver_state = None
            if not passed:
                failed_row = witness
                break
            save_cursor(alpha_index, index + 1, None)
        candidate_result["training_witnesses_passed"] = len(candidate_result["training_witness_rows"])
        if failed_row is not None:
            candidate_result.update({"status": "TRAINING_REFERENCE_INSUFFICIENT",
                                     "failed_row": {key: value for key, value in failed_row.items()
                                                    if key not in {"target_coefficients", "amplitude_command"}}})
            candidate_results.append(candidate_result)
            save_cursor(alpha_index + 1, 0, None)
            row_start = 0
            continue
        candidate_result["status"] = "ALL_TRAINING_WITNESSES_PASSED"
        candidate_results.append(candidate_result)
        selected = {"alpha": alpha, "population": candidate,
                    "records": candidate_result["training_witness_rows"],
                    "candidate_dir": candidate_dir}
        break

    if selected is None:
        terminal = {"task_id": TASK_ID, "status": "PRIOR_QUALIFICATION_INSUFFICIENT",
                    "candidate_results": candidate_results,
                    "training_ready": False, "scientific_updates": 0,
                    "conclusion_scope": "registered five covariance contractions and fixed training population only"}
        _write_json(summary_path, terminal)
        _write_json(QUALIFICATION_DIR / "prior_qualification_failure.json", terminal)
        return terminal

    selected_population = selected["population"]
    holdout, holdout_accounting = _cohort(
        prior, inputs["scale"], alpha=selected["alpha"], role="new_final_holdout2000",
        count=HOLDOUT_PER_CLASS, seed_base=int(config["population"]["holdout_seed_base"]),
        config=config)
    _require(not torch.equal(holdout["coefficients"], selected_population["train"]["coefficients"])
             and not torch.equal(holdout["coefficients"], selected_population["development"]["coefficients"]),
             "final holdout is not independently sampled from the registered seed stream")
    _save_torch(QUALIFICATION_DIR / "selected_population.pt", {
        "schema": f"{TASK_ID}.selected_population.v1", "task_id": TASK_ID,
        "selected_alpha": selected["alpha"],
        "train": selected_population["train"],
        "development": selected_population["development"],
        "holdout": holdout,
        "holdout_sampling_accounting": holdout_accounting,
        "holdout_not_measured_during_training_or_alpha_selection": True})
    witness_paths = sorted(selected["candidate_dir"].glob("witness_rows/row_*.pt"))
    records: list[dict[str, Any]] = []
    for path in witness_paths:
        row = torch.load(path, map_location="cpu", weights_only=False)
        records.append({key: row[key] for key in (
            "index", "label", "proposal_id", "target_coefficients", "amplitude_command",
            "solver_source", "solver_relative_error", "solver_iterations", "phase_rad",
            "metrics", "all_six_metrics_passed")})
    _verify_qualification_output(selected_population, records)
    final = {"task_id": TASK_ID, "status": "PRIOR_CANDIDATE_QUALIFIED",
             "selected_alpha": selected["alpha"], "selected_candidate": f"alpha_{selected['alpha']:.3f}",
             "candidate_results": candidate_results,
             "training_witnesses": _verify_qualification_output(selected_population, records),
             "holdout_count": HOLDOUT_ROWS, "holdout_seed_base": config["population"]["holdout_seed_base"],
             "holdout_outputs_consumed_during_selection": False,
             "training_ready": False, "scientific_updates": 0,
             "mapper_updates": 0, "optical_updates": 0}
    _write_json(summary_path, final)
    _write_json(QUALIFICATION_DIR / "run_config.json", config)
    return final


def _recovery_worker(config: Mapping[str, Any], checkpoint_path: Path, output_path: Path) -> None:
    _require(torch.cuda.is_available(), "independent recovery worker requires CUDA")
    _set_numerics()
    device = torch.device("cuda:0")
    inputs = _load_inputs(config)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    population = torch.load(PREFLIGHT_DIR / "selected_population.pt", map_location="cpu",
                            weights_only=False)
    schedule = _schedule(config)
    _require(saved.get("task_id") == TASK_ID and saved.get("completed_updates") == 3
             and saved.get("schedule_cursor") == 3
             and len(saved.get("history", [])) == 3
             and torch.equal(saved.get("batch"), schedule[2]),
             "recovery worker checkpoint is not the exact three-step fixed schedule state")
    mapper = _new_mapper(inputs, device)
    optimizer = _optimizer(mapper, config)
    mapper.load_state_dict(saved["mapper_state"], strict=True)
    optimizer.load_state_dict(saved["optimizer_state"])
    torch.set_rng_state(saved["rng_state"]["cpu"])
    if "cuda" in saved["rng_state"]:
        torch.cuda.set_rng_state_all(saved["rng_state"]["cuda"])
    batch = schedule[3].clone()
    update = exposure._update(mapper, optimizer, None, inputs,
                              population["train"]["coefficients"][batch], device,
                              "response_matrix")
    result = {"task_id": TASK_ID, "completed_updates": 4, "batch": batch,
              "history": [*saved["history"], {"step": 4, "batch": batch.tolist(), **update}],
              "loss": update["loss"], "mapper_state": exposure._copy_state(mapper.state_dict()),
              "optimizer_state": copy.deepcopy(optimizer.state_dict()), "rng_state": _rng_snapshot()}
    before = _rng_snapshot()
    _save_torch(output_path, result)
    _require(_rng_equal(before, _rng_snapshot()), "worker serialization changed RNG")


def _load_selected_population(qualification_dir: Path = QUALIFICATION_DIR) -> dict[str, Any]:
    path = qualification_dir / "selected_population.pt"
    summary = _json(qualification_dir / "qualification_summary.json")
    _require(path.is_file() and summary.get("task_id") == TASK_ID
             and summary.get("status") == "PRIOR_CANDIDATE_QUALIFIED"
             and summary.get("training_ready") is False,
             "complete candidate qualification is required before preflight", FileNotFoundError)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    alpha = float(summary["selected_alpha"])
    _require(payload.get("schema") == f"{TASK_ID}.selected_population.v1"
             and payload.get("task_id") == TASK_ID and float(payload.get("selected_alpha")) == alpha
             and tuple(payload["train"]["coefficients"].shape) == (TRAIN_ROWS, PORT_COUNT)
             and tuple(payload["development"]["coefficients"].shape) == (DEV_ROWS, PORT_COUNT)
             and tuple(payload["holdout"]["coefficients"].shape) == (HOLDOUT_ROWS, PORT_COUNT)
             and torch.equal(payload["train"]["labels"], torch.arange(10).repeat_interleave(20))
             and torch.equal(payload["development"]["labels"], torch.arange(10).repeat_interleave(20))
             and torch.equal(payload["holdout"]["labels"], torch.arange(10).repeat_interleave(200)),
             "selected candidate population is incomplete or has changed identity")
    return payload


def _validate_recovery_evidence(preflight_dir: Path, recovery: Mapping[str, Any],
                                schedule: torch.Tensor) -> dict[str, float]:
    third = torch.load(preflight_dir / "recovery_step_00003.pt",
                       map_location="cpu", weights_only=False)
    direct = recovery.get("direct_step4", {})
    replay = recovery.get("replayed_step4", {})
    _require(third.get("task_id") == TASK_ID and third.get("completed_updates") == 3
             and third.get("schedule_cursor") == 3 and len(third.get("history", [])) == 3
             and torch.equal(third.get("batch"), schedule[2])
             and _adam_steps(third.get("optimizer_state", {})) == [3, 3, 3, 3],
             "saved third-step recovery checkpoint is not bound to the fixed schedule")
    _require(direct.get("task_id") == TASK_ID and replay.get("task_id") == TASK_ID
             and direct.get("completed_updates") == 4 and replay.get("completed_updates") == 4
             and torch.equal(direct.get("batch"), schedule[3])
             and torch.equal(replay.get("batch"), schedule[3])
             and len(direct.get("history", [])) == 4 and len(replay.get("history", [])) == 4
             and direct["history"][:3] == third["history"]
             and replay["history"][:3] == third["history"]
             and direct.get("history") == replay.get("history")
             and _rng_equal(direct.get("rng_state", {}), replay.get("rng_state", {}))
             and recovery.get("rng_equal") is True
             and recovery.get("history_and_batch_equal") is True
             and _adam_steps(direct.get("optimizer_state", {})) == [4, 4, 4, 4]
             and _adam_steps(replay.get("optimizer_state", {})) == [4, 4, 4, 4],
             "direct and independently restored fourth-step history, RNG, batch, or Adam state differs")
    model_delta = _compare_states(direct.get("mapper_state", {}), replay.get("mapper_state", {}))
    _, optimizer_delta = fit20._tree_max_delta(direct.get("optimizer_state", {}),
                                               replay.get("optimizer_state", {}))
    loss_delta = abs(float(direct.get("loss", math.inf)) - float(replay.get("loss", math.inf)))
    _require(recovery.get("engineering_adam_calls") == 5
             and recovery.get("serialization_preserved_rng") is True
             and model_delta <= RECOVERY_TOLERANCE
             and optimizer_delta <= RECOVERY_TOLERANCE
             and loss_delta <= RECOVERY_TOLERANCE
             and abs(float(recovery.get("model_max_abs_delta", math.inf)) - model_delta) <= 1.0e-14
             and abs(float(recovery.get("optimizer_max_abs_delta", math.inf)) - optimizer_delta) <= 1.0e-14
             and abs(float(recovery.get("loss_abs_delta", math.inf)) - loss_delta) <= 1.0e-14,
             "third-to-fourth recovery evidence exceeds tolerance or its summary is inconsistent")
    return {"model_max_abs_delta": model_delta,
            "optimizer_max_abs_delta": optimizer_delta,
            "loss_abs_delta": loss_delta}


def _ready_receipt_checks_pass(receipt: PreflightReceipt) -> bool:
    checks = receipt.checks
    return (receipt.status == "READY" and receipt.authorization_id == AUTHORITY
            and receipt.selected_engine == "response_matrix"
            and all(checks.get(key, {}).get("passed") is True for key in
                    ("CORRECTNESS_INVARIANTS", "RESOURCE_ENVELOPE", "IMPLEMENTATION_SELECTION"))
            and checks.get("CORRECTNESS_INVARIANTS", {}).get(
                "frozen_optics_projection_gain_prior_and_evaluators_unchanged") is True
            and checks.get("CORRECTNESS_INVARIANTS", {}).get(
                "optimizer_contains_only_mapper_parameters") is True
            and checks.get("IMPLEMENTATION_SELECTION", {}).get("frozen_inputs_unchanged") is True)


def _load_qualification_witnesses(
        population: Mapping[str, Any], alpha: float,
        qualification_dir: Path = QUALIFICATION_DIR) -> list[dict[str, Any]]:
    candidate_dir = qualification_dir / f"alpha_{alpha:.3f}"
    result = []
    for index in range(TRAIN_ROWS):
        path = candidate_dir / "witness_rows" / f"row_{index:04d}.pt"
        _require(path.is_file(), f"candidate qualification witness is missing: {path}", FileNotFoundError)
        item = torch.load(path, map_location="cpu", weights_only=False)
        target = population["train"]["coefficients"][index]
        _require(item.get("task_id") == TASK_ID and item.get("index") == index
                 and item.get("label") == int(population["train"]["labels"][index])
                 and item.get("proposal_id") == int(population["train"]["proposal_ids"][index])
                 and torch.equal(item.get("target_coefficients"), target)
                 and tuple(item.get("amplitude_command", torch.empty(0)).shape) == (1, 1, 200, 200)
                 and item.get("all_six_metrics_passed") is True
                 and all(float(item["metrics"][key]) <= GATE_LIMIT for key in METRIC_KEYS),
                 f"qualification witness {index} is not bound to a valid fixed target")
        command = item["amplitude_command"]
        _require(torch.isfinite(command).all().item() and float(command.min()) >= 0.0
                 and float(command.max()) <= 1.0,
                 f"qualification witness {index} has an invalid amplitude command")
        result.append(item)
    return result


def _preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "fixed-frame training preflight requires CUDA")
    _require(not PREFLIGHT_DIR.exists(), f"preflight output already exists: {PREFLIGHT_DIR}", FileExistsError)
    _set_numerics()
    device = torch.device("cuda:0")
    inputs = _load_inputs(config)
    camera_area = _camera_area(inputs, device)
    _require(abs(camera_area - 6.25e-12) <= 1.0e-22,
             f"frozen camera pixel area differs from 6.25e-12 m^2: {camera_area:.17g}")
    selected = _load_selected_population()
    alpha = float(selected["selected_alpha"])
    witnesses = _load_qualification_witnesses(selected, alpha)
    schedule = _schedule(config)
    box_check = _check_box_loss(device)

    # Re-run ten full-optics positive controls from the qualified fixed population.
    indices = torch.arange(0, TRAIN_ROWS, TRAIN_PER_CLASS)
    commands = torch.cat([witnesses[int(index)]["amplitude_command"] for index in indices.tolist()])
    targets = selected["train"]["coefficients"][indices]
    models = {grid: exposure._make_runtime_model(inputs, grid, device) for grid in GRIDS}
    actual: dict[int, list[dict[str, torch.Tensor]]] = {grid: [] for grid in GRIDS}
    references: dict[int, list[dict[str, torch.Tensor]]] = {grid: [] for grid in GRIDS}
    with torch.no_grad():
        for row_index in range(len(indices)):
            command = commands[row_index:row_index + 1].to(device=device, dtype=torch.float32)
            target = targets[row_index:row_index + 1].to(device=device, dtype=torch.complex128)
            for grid in GRIDS:
                actual[grid].append(projected._forward_amplitude(models[grid], command))
                references[grid].append(projected._forward_target(models[grid], target))
    positive_rows = []
    for row_index in range(len(indices)):
        one = {grid: actual[grid][row_index] for grid in GRIDS}
        direct = {grid: references[grid][row_index]["raw_cmos"] for grid in GRIDS}
        target = targets[row_index:row_index + 1].to(device=device, dtype=torch.complex128)
        metrics = _metric_row(one, target, direct)
        _require(all(float(metrics[key]) <= GATE_LIMIT for key in METRIC_KEYS),
                 f"full-optics positive control failed on fixed training row {row_index}")
        matrix_prediction = exposure._forward_matrix(
            commands[row_index:row_index + 1].to(device), inputs["response_matrix"].to(device))
        optical_coeff = actual[2048][row_index]["coefficients"]
        response_error = float(torch.linalg.vector_norm(matrix_prediction - optical_coeff)
                               / torch.linalg.vector_norm(optical_coeff).clamp_min(1.0e-30))
        _require(response_error <= 1.0e-6,
                 f"selected response matrix differs from full optics on row {row_index}")
        positive_rows.append({"index": int(indices[row_index]), "label": row_index,
                              **metrics, "matrix_response_relative_error": response_error})

    # Negative controls traverse the real optical producer and six-metric consumer.
    valid_command = commands[0:1].to(device=device, dtype=torch.float32)
    valid_target = targets[0:1].to(device=device, dtype=torch.complex128)
    wrong_target = targets[-1:].to(device=device, dtype=torch.complex128)
    wrong_target_metrics, _ = _measure_command(inputs, valid_command, wrong_target, models)
    wrong_encoding_metrics, _ = _measure_command(
        inputs, valid_command.square(), valid_target, models)
    _require(any(float(wrong_target_metrics[key]) > GATE_LIMIT for key in METRIC_KEYS)
             and any(float(wrong_encoding_metrics[key]) > GATE_LIMIT for key in METRIC_KEYS),
             "production negative control accepted a wrong target or amplitude/intensity encoding")

    run_id = uuid.uuid4().hex
    identity = _identity(config, run_id)
    control = ResearchRunControl(identity=identity, authorization_id=AUTHORITY)
    PREFLIGHT_DIR.mkdir(parents=True)
    _write_json(PREFLIGHT_DIR / "preflight_identity.json", {
        "task_id": TASK_ID, "run_id": run_id, "run_identity": identity.to_dict()})
    _save_torch(PREFLIGHT_DIR / "selected_population.pt", selected)
    _save_torch(PREFLIGHT_DIR / "schedule.pt", schedule)
    _save_torch(PREFLIGHT_DIR / "qualification_witnesses.pt", {
        "task_id": TASK_ID, "selected_alpha": alpha,
        "rows": [{key: value for key, value in row.items()
                  if key not in {"outputs"}} for row in witnesses]})
    _save_torch(PREFLIGHT_DIR / "positive_controls.pt", {
        "task_id": TASK_ID, "indices": indices, "commands": commands,
        "targets": targets, "metrics": positive_rows})

    mapper = _new_mapper(inputs, device)
    optimizer = _optimizer(mapper, config)
    frozen_snapshot = _frozen_input_snapshot(inputs)
    mapper_parameters = list(mapper.parameters())
    optimizer_parameters = [parameter for group in optimizer.param_groups
                            for parameter in group["params"]]
    _require(not optimizer.state and _compare_states(mapper.state_dict(),
              inputs["source_mapper"]["mapper_state"]) == 0.0,
             "preflight did not start from the explicit V2 endpoint and empty Adam")
    _require(len(optimizer_parameters) == len(mapper_parameters)
             and {id(parameter) for parameter in optimizer_parameters}
             == {id(parameter) for parameter in mapper_parameters},
             "engineering preflight optimizer contains a non-mapper or missing parameter")
    torch.manual_seed(int(config["training"]["rng_seed"]))
    torch.cuda.manual_seed_all(int(config["training"]["rng_seed"]))
    history = []
    for step in range(3):
        batch = schedule[step].clone()
        update = exposure._update(mapper, optimizer, None, inputs,
                                  selected["train"]["coefficients"][batch], device,
                                  "response_matrix", audit=step == 0)
        history.append({"step": step + 1, "batch": batch.tolist(), **update})
    third = {"task_id": TASK_ID, "completed_updates": 3, "schedule_cursor": 3,
             "history": copy.deepcopy(history), "batch": schedule[2].clone(),
             "mapper_state": exposure._copy_state(mapper.state_dict()),
             "optimizer_state": copy.deepcopy(optimizer.state_dict()),
             "rng_state": _rng_snapshot()}
    third_path = PREFLIGHT_DIR / "recovery_step_00003.pt"
    _save_torch(third_path, third)
    _require(_compare_states(torch.load(third_path, map_location="cpu", weights_only=False)["mapper_state"],
                             mapper.state_dict()) == 0.0,
             "serialized third-step mapper changed")
    batch4 = schedule[3].clone()
    direct_update = exposure._update(mapper, optimizer, None, inputs,
                                     selected["train"]["coefficients"][batch4], device,
                                     "response_matrix")
    _assert_frozen_inputs(inputs, frozen_snapshot)
    direct4 = {"task_id": TASK_ID, "completed_updates": 4, "batch": batch4,
               "history": [*history, {"step": 4, "batch": batch4.tolist(), **direct_update}],
               "loss": direct_update["loss"],
               "mapper_state": exposure._copy_state(mapper.state_dict()),
               "optimizer_state": copy.deepcopy(optimizer.state_dict()),
               "rng_state": _rng_snapshot()}
    direct_path = PREFLIGHT_DIR / "recovery_direct_step_00004.pt"
    rng_before = _rng_snapshot()
    _save_torch(direct_path, direct4)
    _require(_rng_equal(rng_before, _rng_snapshot()), "saving direct fourth-step state changed RNG")
    replay_path = PREFLIGHT_DIR / "recovery_replayed_step_00004.pt"
    worker = [sys.executable, "-m",
              "scripts.research.lightgen.execute_lightgen_g1_fixed_frame_semantic_v1",
              "--mode", "recovery-worker", "--config", str(DEFAULT_CONFIG), "--checkpoint", str(third_path),
              "--output", str(replay_path)]
    completed = subprocess.run(worker, cwd=ROOT, capture_output=True, text=True, check=False)
    _require(completed.returncode == 0 and replay_path.is_file(),
             "independent recovery worker failed: " + completed.stderr[-4000:])
    replay4 = torch.load(replay_path, map_location="cpu", weights_only=False)
    _, optimizer_delta = fit20._tree_max_delta(direct4["optimizer_state"], replay4["optimizer_state"])
    model_delta = _compare_states(direct4["mapper_state"], replay4["mapper_state"])
    loss_delta = abs(float(direct4["loss"]) - float(replay4["loss"]))
    _require(model_delta <= RECOVERY_TOLERANCE and optimizer_delta <= RECOVERY_TOLERANCE
             and loss_delta <= RECOVERY_TOLERANCE
             and _rng_equal(direct4["rng_state"], replay4["rng_state"])
             and torch.equal(direct4["batch"], replay4["batch"])
             and direct4["history"] == replay4["history"]
             and _adam_steps(direct4["optimizer_state"]) == [4, 4, 4, 4],
             "in-process and restored fourth updates do not agree")
    recovery = {"task_id": TASK_ID, "engineering_adam_calls": 5,
                "direct_step4": direct4, "replayed_step4": replay4,
                "model_max_abs_delta": model_delta, "optimizer_max_abs_delta": optimizer_delta,
                "loss_abs_delta": loss_delta, "rng_equal": True,
                "history_and_batch_equal": True, "serialization_preserved_rng": True}
    _save_torch(PREFLIGHT_DIR / "engineering_recovery.pt", recovery)

    resource = {"passed": True, "status": "PASS", "cuda_device": torch.cuda.get_device_name(device),
                "free_memory_bytes_observation": int(torch.cuda.mem_get_info(device)[0]),
                "fixed_free_memory_quota": False}
    correctness = {"passed": True, "status": "PASS", "selected_alpha": alpha,
                   "training_witness_count": len(witnesses), "all_six_training_witnesses_valid": True,
                   "full_optics_positive_control_count": len(positive_rows),
                   "box_loss_value_gradient_and_direction_checked": True,
                   "wrong_target_and_encoding_rejected": True,
                   "v2_terminal_start_and_empty_adam_verified": True,
                   "frozen_optics_projection_gain_prior_and_evaluators_unchanged": True,
                   "optimizer_contains_only_mapper_parameters": True,
                   "training_schedule_exact_exposures": True}
    implementation = {"passed": True, "status": "PASS", "selected_engine": "response_matrix",
                      "matrix_response_max_relative_error": max(
                          row["matrix_response_relative_error"] for row in positive_rows),
                      "independent_recovery_verified": True, "engineering_adam_calls": 5,
                      "frozen_inputs_unchanged": True,
                      "model_optimizer_loss_tolerance": RECOVERY_TOLERANCE}
    receipt = control.make_ready_receipt(correctness=correctness, resource=resource,
                                         implementation=implementation,
                                         selected_engine="response_matrix",
                                         resource_envelope=resource)
    control.mark_preflight_ready(receipt)
    _write_json(PREFLIGHT_DIR / "preflight_receipt.json", receipt.to_dict())
    summary = {"task_id": TASK_ID, "run_id": run_id, "status": "READY",
               "selected_alpha": alpha, "engineering_adam_calls": 5,
               "positive_control_rows": positive_rows,
               "negative_controls": {"wrong_target_rejected": True,
                                     "amplitude_intensity_confusion_rejected": True},
               "box_loss_qualification": box_check,
               "recovery": {key: recovery[key] for key in
                            ("model_max_abs_delta", "optimizer_max_abs_delta", "loss_abs_delta",
                             "rng_equal", "history_and_batch_equal", "serialization_preserved_rng")},
               "mapper_updates_in_preflight": 4, "scientific_updates": 0,
               "optical_updates": 0, "selected_engine": "response_matrix"}
    _write_json(PREFLIGHT_DIR / "preflight_summary.json", summary)
    _write_json(PREFLIGHT_DIR / "run_config.json", config)
    return summary


def _load_ready(config: Mapping[str, Any], population: Mapping[str, Any]
                ) -> tuple[dict[str, Any], PreflightReceipt]:
    required = (PREFLIGHT_DIR / "preflight_summary.json",
                PREFLIGHT_DIR / "preflight_receipt.json",
                PREFLIGHT_DIR / "preflight_identity.json",
                PREFLIGHT_DIR / "qualification_witnesses.pt",
                PREFLIGHT_DIR / "engineering_recovery.pt")
    _require(all(path.is_file() for path in required),
             "the new task's complete executed READY preflight is required", FileNotFoundError)
    summary = _json(required[0])
    receipt = PreflightReceipt.from_mapping(_json(required[1]))
    identity_payload = _json(required[2])
    identity = ResearchRunIdentity.from_mapping(identity_payload["run_identity"])
    witnesses = torch.load(required[3], map_location="cpu", weights_only=False)
    recovery = torch.load(required[4], map_location="cpu", weights_only=False)
    recovery_deltas = _validate_recovery_evidence(PREFLIGHT_DIR, recovery, _schedule(config))
    _require(summary.get("task_id") == TASK_ID and summary.get("status") == "READY"
             and identity_payload.get("task_id") == TASK_ID and identity.task_id == TASK_ID
             and identity.run_id == summary.get("run_id")
             and _ready_receipt_checks_pass(receipt)
             and receipt.source_semantics_id == identity.execution_source_semantics_id
             and witnesses.get("task_id") == TASK_ID and len(witnesses.get("rows", [])) == TRAIN_ROWS
             and float(summary.get("selected_alpha", -1.0)) == float(population["selected_alpha"])
             and recovery_deltas["loss_abs_delta"] <= RECOVERY_TOLERANCE,
             "READY receipt, fixed candidate witnesses, or independent recovery did not verify")
    for index, row in enumerate(witnesses["rows"]):
        _require(int(row["index"]) == index
                 and int(row["proposal_id"]) == int(population["train"]["proposal_ids"][index])
                 and torch.equal(row["target_coefficients"], population["train"]["coefficients"][index])
                 and row["all_six_metrics_passed"] is True
                 and all(float(row["metrics"][key]) <= GATE_LIMIT for key in METRIC_KEYS),
                 f"READY witness identity or measurement failed at training row {index}")
    return {"identity": identity.to_dict(), "summary": summary}, receipt


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _checkpoint_payload(mapper: torch.nn.Module, optimizer: torch.optim.Optimizer,
                        schedule: torch.Tensor, history: Sequence[Mapping[str, Any]],
                        exposure_counts: torch.Tensor, completed: int, run_id: str,
                        alpha: float) -> dict[str, Any]:
    return {"schema": f"{TASK_ID}.checkpoint.v1", "task_id": TASK_ID, "run_id": run_id,
            "completed_updates": completed, "checkpoint_step": completed,
            "source_mapper_updates": SOURCE_MAPPER_UPDATES,
            "lineage_mapper_updates": SOURCE_MAPPER_UPDATES + completed,
            "selected_alpha": alpha, "mapper_state": exposure._copy_state(mapper.state_dict()),
            "optimizer_state": copy.deepcopy(optimizer.state_dict()), "rng_state": _rng_snapshot(),
            "schedule": schedule.clone(), "schedule_cursor": completed,
            "history": copy.deepcopy(list(history)), "history_count": len(history),
            "exposure_counts": exposure_counts.clone(), "engine": "response_matrix",
            "optimizer_state_deepcopied": True}


def _save_training_checkpoint(path: Path, mapper: torch.nn.Module,
                              optimizer: torch.optim.Optimizer, schedule: torch.Tensor,
                              history: Sequence[Mapping[str, Any]], exposure_counts: torch.Tensor,
                              completed: int, run_id: str, alpha: float) -> dict[str, Any]:
    rng_before = _rng_snapshot()
    payload = _checkpoint_payload(mapper, optimizer, schedule, history,
                                  exposure_counts, completed, run_id, alpha)
    _require(_rng_equal(rng_before, _rng_snapshot()), "checkpoint serialization changed RNG")
    _save_torch(path, payload)
    reopened = torch.load(path, map_location="cpu", weights_only=False)
    _require(reopened.get("task_id") == TASK_ID and reopened.get("completed_updates") == completed
             and reopened.get("history_count") == completed and len(reopened.get("history", [])) == completed
             and torch.equal(reopened.get("schedule"), schedule)
             and torch.equal(reopened.get("exposure_counts"), exposure_counts.cpu())
             and _compare_states(reopened["mapper_state"], mapper.state_dict()) == 0.0,
             "fixed-frame checkpoint failed reload verification")
    return reopened


def _restore_training_checkpoint(path: Path, mapper: torch.nn.Module,
                                 optimizer: torch.optim.Optimizer, schedule: torch.Tensor,
                                 alpha: float) -> tuple[int, list[dict[str, Any]], torch.Tensor,
                                                        dict[str, Any]]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    completed = int(saved.get("completed_updates", -1))
    _require(saved.get("schema") == f"{TASK_ID}.checkpoint.v1" and saved.get("task_id") == TASK_ID
             and 0 <= completed < UPDATE_BUDGET and int(saved.get("schedule_cursor", -1)) == completed
             and saved.get("history_count") == completed and len(saved.get("history", [])) == completed
             and float(saved.get("selected_alpha", -1.0)) == alpha
             and torch.equal(saved.get("schedule"), schedule)
             and saved.get("source_mapper_updates") == SOURCE_MAPPER_UPDATES
             and saved.get("lineage_mapper_updates") == SOURCE_MAPPER_UPDATES + completed,
             "explicit resume checkpoint differs from the fixed task or its history prefix")
    mapper.load_state_dict(saved["mapper_state"], strict=True)
    optimizer.load_state_dict(saved["optimizer_state"])
    counts = torch.bincount(schedule[:completed].reshape(-1), minlength=TRAIN_ROWS)
    _require(torch.equal(saved.get("exposure_counts"), counts),
             "checkpoint exposure counts differ from the fixed schedule prefix")
    if "cuda" in saved["rng_state"]:
        torch.cuda.set_rng_state_all(saved["rng_state"]["cuda"])
    torch.set_rng_state(saved["rng_state"]["cpu"])
    return completed, list(saved["history"]), counts, saved


def _observe_2048(inputs: Mapping[str, Any], population: Mapping[str, Any],
                  mapper: torch.nn.Module, device: torch.device, step: int) -> dict[str, Any]:
    model = exposure._make_runtime_model(inputs, 2048, device)
    coefficients, commands = [], []
    with torch.no_grad():
        for start in range(0, len(population["labels"]), 10):
            stop = min(start + 10, len(population["labels"]))
            target_cpu = population["coefficients"][start:stop]
            target = target_cpu.to(device=device, dtype=torch.complex128)
            features = projected._normalized_features(target_cpu, inputs["scale"], device)
            mapped = [projected._map_amplitude(
                mapper, features[row:row + 1], target[row:row + 1],
                inputs["matrix"].to(device), inputs["inverse"].to(device),
                inputs["scale"].to(device))[0] for row in range(stop - start)]
            command = torch.cat(mapped)
            output = projected._forward_amplitude(model, command)["coefficients"]
            commands.append(command.cpu())
            coefficients.append(output.cpu())
    actual = torch.cat(coefficients).to(torch.complex128)
    targets = population["coefficients"].to(torch.complex128)
    errors = torch.tensor([exposure._phase_invariant_relative_error(actual[index], targets[index])
                           for index in range(len(actual))], dtype=torch.float64)
    summary = {"task_id": TASK_ID, "step": step, "role": population["role"],
               "count": len(actual), "coefficient_2048_mean": float(errors.mean()),
               "coefficient_2048_rms": float(errors.square().mean().sqrt()),
               "coefficient_2048_median": float(errors.median()),
               "coefficient_2048_max": float(errors.max()),
               "full_optics_2048_observation": True, "six_metric_gate": False}
    destination = ATTEMPT_DIR / "observations"
    _write_json(destination / f"step_{step:05d}_{population['role']}.json", summary)
    _save_torch(destination / f"step_{step:05d}_{population['role']}.pt", {
        **summary, "labels": population["labels"].clone(),
        "proposal_ids": population["proposal_ids"].clone(),
        "target_coefficients": population["coefficients"].clone(),
        "commands": torch.cat(commands), "actual_coefficients_2048": actual})
    return summary


def _train(config: Mapping[str, Any], *, checkpoint_path: Path | None = None) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "fixed-frame scientific training requires AutoDL CUDA")
    _set_numerics()
    device = torch.device("cuda:0")
    inputs = _load_inputs(config)
    camera_area = _camera_area(inputs, torch.device("cuda:0"))
    _require(abs(camera_area - 6.25e-12) <= 1.0e-22,
             f"frozen camera pixel area differs from 6.25e-12 m^2: {camera_area:.17g}")
    population = _load_selected_population()
    identity_payload, _ = _load_ready(config, population)
    run_id = str(identity_payload["identity"]["run_id"])
    alpha = float(population["selected_alpha"])
    schedule = _schedule(config)

    if checkpoint_path is None:
        _require(not ATTEMPT_DIR.exists(), f"attempt already exists: {ATTEMPT_DIR}", FileExistsError)
        mapper = _new_mapper(inputs, device)
        optimizer = _optimizer(mapper, config)
        _require(not optimizer.state and _compare_states(mapper.state_dict(),
                  inputs["source_mapper"]["mapper_state"]) == 0.0,
                 "training did not start from the V2 terminal weights and empty Adam")
        receipt = PreflightReceipt.from_mapping(_json(PREFLIGHT_DIR / "preflight_receipt.json"))
        identity = ResearchRunIdentity.from_mapping(identity_payload["identity"])
        control = ResearchRunControl(identity=identity, authorization_id=AUTHORITY)
        control.mark_preflight_ready(receipt)
        attempt_identity = control.open_scientific_attempt(
            ATTEMPT_DIR, f"{run_id}:attempt_01", receipt)
        (ATTEMPT_DIR / "checkpoints").mkdir()
        shutil.copytree(PREFLIGHT_DIR, ATTEMPT_DIR / "preflight_evidence")
        shutil.copytree(PRIOR_DIR, ATTEMPT_DIR / "prior_evidence")
        shutil.copytree(QUALIFICATION_DIR, ATTEMPT_DIR / "qualification_evidence")
        _save_torch(ATTEMPT_DIR / "selected_population.pt", population)
        _save_torch(ATTEMPT_DIR / "mapper_initialization.pt", {
            "task_id": TASK_ID, "source_checkpoint": str(inputs["source_mapper_path"]),
            "mapper_state": exposure._copy_state(mapper.state_dict()), "new_adam_state": {},
            "lineage_mapper_updates_before_task": SOURCE_MAPPER_UPDATES})
        shutil.copy2(DEFAULT_CONFIG, ATTEMPT_DIR / "run_config.yaml")
        shutil.copy2(Path(__file__).resolve(), ATTEMPT_DIR / "runner_source.py")
        shutil.copy2(PREFLIGHT_DIR / "preflight_receipt.json", ATTEMPT_DIR / "preflight_receipt.json")
        _write_json(ATTEMPT_DIR / "scientific_attempt_identity.json", {
            "task_id": TASK_ID, "run_id": run_id, "run_identity": attempt_identity.to_dict()})
        torch.manual_seed(int(config["training"]["rng_seed"]))
        torch.cuda.manual_seed_all(int(config["training"]["rng_seed"]))
        history: list[dict[str, Any]] = []
        counts = torch.zeros(TRAIN_ROWS, dtype=torch.long)
        _save_training_checkpoint(ATTEMPT_DIR / "checkpoints" / "checkpoint_step_00000.pt",
                                  mapper, optimizer, schedule, history, counts, 0, run_id, alpha)
        _write_json(ATTEMPT_DIR / "task_state.json", {"task_id": TASK_ID, "run_id": run_id,
                    "state": "SCIENTIFIC_ACTIVE", "completed_updates": 0,
                    "authorized_updates": UPDATE_BUDGET})
        for cohort in (population["train"], population["development"]):
            _observe_2048(inputs, cohort, mapper, device, 0)
        start = 0
    else:
        checkpoint_path = checkpoint_path.resolve()
        _require(ATTEMPT_DIR.is_dir() and checkpoint_path.is_file()
                 and checkpoint_path.is_relative_to((ATTEMPT_DIR / "checkpoints").resolve()),
                 "resume requires an explicit checkpoint inside this fixed attempt", ValueError)
        task_state = _json(ATTEMPT_DIR / "task_state.json")
        _require(task_state.get("state") == "SCIENTIFIC_ACTIVE"
                 and not (ATTEMPT_DIR / "final_decision.json").exists(),
                 "closed or terminal attempts cannot be resumed")
        receipt = PreflightReceipt.from_mapping(_json(PREFLIGHT_DIR / "preflight_receipt.json"))
        identity = ResearchRunIdentity.from_mapping(identity_payload["identity"])
        control = ResearchRunControl(identity=identity, authorization_id=AUTHORITY)
        control.mark_preflight_ready(receipt)
        attempt_identity = control.open_scientific_attempt(
            ATTEMPT_DIR, f"{run_id}:attempt_01", receipt, resume=True)
        saved_attempt_identity = _json(ATTEMPT_DIR / "scientific_attempt_identity.json")
        _require(saved_attempt_identity.get("run_id") == run_id
                 and ResearchRunIdentity.from_mapping(saved_attempt_identity["run_identity"])
                 == attempt_identity,
                 "resume attempt identity differs from its READY-authorized scientific attempt")
        mapper = _new_mapper(inputs, device)
        optimizer = _optimizer(mapper, config)
        start, history, counts, saved = _restore_training_checkpoint(
            checkpoint_path, mapper, optimizer, schedule, alpha)
        _require(saved.get("run_id") == run_id,
                 "resume checkpoint run id differs from the consumed READY identity")
        history_path = ATTEMPT_DIR / f"training_history_checkpoint_prefix_{start:05d}.csv"
        _atomic_csv(history_path, history)

    frozen_snapshot = _frozen_input_snapshot(inputs)
    mapper_parameters = list(mapper.parameters())
    optimizer_parameters = [parameter for group in optimizer.param_groups
                            for parameter in group["params"]]
    _require(len(optimizer_parameters) == len(mapper_parameters)
             and {id(parameter) for parameter in optimizer_parameters}
             == {id(parameter) for parameter in mapper_parameters},
             "formal optimizer contains a non-mapper or missing parameter")
    latest = ATTEMPT_DIR / "checkpoints" / "latest.pt"
    fixed_points = {5000, 10000, UPDATE_BUDGET}
    history_flushed = start
    if checkpoint_path is None:
        history_path = ATTEMPT_DIR / "training_history.csv"
    for step in range(start, UPDATE_BUDGET):
        indices = schedule[step].clone()
        update = exposure._update(mapper, optimizer, None, inputs,
                                  population["train"]["coefficients"][indices], device,
                                  "response_matrix", audit=step == start)
        counts.index_add_(0, indices, torch.ones_like(indices))
        history.append({"completed_update": step + 1, "sample_indices": indices.tolist(),
                        "labels": list(range(10)), "loss_is_pre_update": True,
                        "target_exposure_counts_after_update": counts[indices].tolist(),
                        "coefficient_loss": update["coefficient_loss"], "box_loss": update["box_loss"],
                        "loss": update["loss"],
                        "coefficient_relative_error_mean": update["coefficient_relative_error_mean"],
                        "clipped_fraction": update["clipped_fraction"],
                        "gradient_norms": update["gradient_norms"]})
        completed = step + 1
        if completed % int(config["training"]["progress_interval_steps"]) == 0:
            _write_json(ATTEMPT_DIR / "progress.json", {
                "task_id": TASK_ID, "run_id": run_id, "completed_updates": completed,
                "authorized_updates": UPDATE_BUDGET, "engine": "response_matrix",
                "updated_at_unix": time.time()})
            exposure._append_csv(history_path, history[history_flushed:])
            history_flushed = len(history)
        if completed % int(config["training"]["checkpoint_interval_steps"]) == 0:
            if latest.is_file():
                shutil.copy2(latest, ATTEMPT_DIR / "checkpoints" / "previous.pt")
            _save_training_checkpoint(latest, mapper, optimizer, schedule, history,
                                      counts, completed, run_id, alpha)
            if completed in fixed_points:
                shutil.copy2(latest, ATTEMPT_DIR / "checkpoints" /
                             f"checkpoint_step_{completed:05d}.pt")
                _write_json(ATTEMPT_DIR / "task_state.json", {
                    "task_id": TASK_ID, "run_id": run_id, "state": "SCIENTIFIC_ACTIVE",
                    "completed_updates": completed, "authorized_updates": UPDATE_BUDGET,
                    "adam_steps": completed, "lineage_mapper_updates": SOURCE_MAPPER_UPDATES + completed,
                    "selected_alpha": alpha, "engine": "response_matrix"})
                if completed in {5000, 10000}:
                    for cohort in (population["train"], population["development"]):
                        _observe_2048(inputs, cohort, mapper, device, completed)
    _require(torch.equal(counts, torch.full((TRAIN_ROWS,), EXPOSURES)),
             "training ended with an incorrect per-target exposure count")
    _assert_frozen_inputs(inputs, frozen_snapshot)
    terminal = ATTEMPT_DIR / "checkpoints" / "terminal_checkpoint.pt"
    _save_training_checkpoint(terminal, mapper, optimizer, schedule, history,
                              counts, UPDATE_BUDGET, run_id, alpha)
    shutil.copy2(terminal, ATTEMPT_DIR / "checkpoints" / "checkpoint_step_15000.pt")
    for cohort in (population["train"], population["development"]):
        _observe_2048(inputs, cohort, mapper, device, UPDATE_BUDGET)
    _write_json(ATTEMPT_DIR / "task_state.json", {
        "task_id": TASK_ID, "run_id": run_id, "state": "TERMINAL_MEASUREMENT_PENDING",
        "completed_updates": UPDATE_BUDGET, "authorized_updates": UPDATE_BUDGET,
        "adam_steps": UPDATE_BUDGET, "lineage_mapper_updates": SOURCE_MAPPER_UPDATES + UPDATE_BUDGET,
        "selected_alpha": alpha, "engine": "response_matrix"})
    return {"task_id": TASK_ID, "run_id": run_id, "completed_updates": UPDATE_BUDGET,
            "lineage_mapper_updates": SOURCE_MAPPER_UPDATES + UPDATE_BUDGET,
            "attempt_dir": str(ATTEMPT_DIR)}


def _metric_row(outputs: Mapping[int, Mapping[str, torch.Tensor]], target: torch.Tensor,
                direct: Mapping[int, torch.Tensor]) -> dict[str, Any]:
    """Measure one saved row; source-population indices never index a singleton batch."""
    _require(target.shape[0] == 1
             and all(outputs[grid]["coefficients"].shape[0] == 1
                     and outputs[grid]["raw_cmos"].shape[0] == 1
                     and direct[grid].shape[0] == 1 for grid in GRIDS),
             "single-row metric consumer requires one production sample")
    return projected._error_row(outputs, target, direct, 0)


def _measure_one_cohort(inputs: Mapping[str, Any], population: Mapping[str, Any],
                        mapper: torch.nn.Module, device: torch.device, *, role: str,
                        old_g1_population: Mapping[str, Any] | None = None,
                        frozen_mapper: torch.nn.Module | None = None) -> list[dict[str, Any]]:
    directory = ATTEMPT_DIR / "measurements" / role
    directory.mkdir(parents=True, exist_ok=True)
    expected_shard_count = math.ceil(len(population["labels"]) / 10)
    existing_paths = sorted(directory.glob("shard_*.pt"))
    existing_by_index: dict[int, Path] = {}
    for path in existing_paths:
        try:
            shard_index = int(path.stem.removeprefix("shard_"))
        except ValueError as error:
            raise RuntimeError(f"invalid measurement shard name: {path.name}") from error
        _require(0 <= shard_index < expected_shard_count and shard_index not in existing_by_index,
                 f"unexpected or duplicate measurement shard for {role}: {path.name}")
        existing_by_index[shard_index] = path
    models = {grid: exposure._make_runtime_model(inputs, grid, device) for grid in GRIDS}
    matrix, inverse, scale = (inputs["matrix"].to(device), inputs["inverse"].to(device),
                              inputs["scale"].to(device))
    mapper.eval()
    if frozen_mapper is not None:
        frozen_mapper.eval()
    rows: list[dict[str, Any]] = []
    for shard_no, start in enumerate(range(0, len(population["labels"]), 10)):
        stop = min(start + 10, len(population["labels"]))
        if shard_no in existing_by_index:
            saved = torch.load(existing_by_index[shard_no], map_location="cpu", weights_only=False)
            expected_targets = population["coefficients"][start:stop].to(torch.complex128)
            expected_indices = population["source_indices"][start:stop]
            expected_labels = population["labels"][start:stop]
            expected_proposals = population["proposal_ids"][start:stop]
            _require(saved.get("schema") == f"{TASK_ID}.measurement_shard.v1"
                     and saved.get("task_id") == TASK_ID and saved.get("cohort_role") == role
                     and saved.get("start") == start and saved.get("stop") == stop
                     and isinstance(saved.get("rows"), list)
                     and len(saved["rows"]) == stop - start
                     and torch.equal(saved.get("indices"), expected_indices)
                     and torch.equal(saved.get("labels"), expected_labels)
                     and torch.equal(saved.get("proposal_ids"), expected_proposals)
                     and torch.equal(saved.get("target_coefficients"), expected_targets)
                     and tuple(saved.get("commands", torch.empty(0)).shape) == (stop - start, 1, 200, 200)
                     and torch.isfinite(saved["commands"]).all().item()
                     and float(saved["commands"].min()) >= 0.0
                     and float(saved["commands"].max()) <= 1.0
                     and set(saved.get("candidate_outputs", {})) == set(GRIDS)
                     and set(saved.get("direct_raw_cmos", {})) == set(GRIDS),
                     f"existing {role} measurement shard {shard_no} has a different fixed identity")
            for grid in GRIDS:
                output = saved["candidate_outputs"][grid]
                _require(tuple(output["coefficients"].shape) == (stop - start, PORT_COUNT)
                         and tuple(output["raw_cmos"].shape[0:1]) == (stop - start,)
                         and tuple(saved["direct_raw_cmos"][grid].shape[0:1]) == (stop - start,)
                         and torch.isfinite(output["coefficients"].real).all().item()
                         and torch.isfinite(output["coefficients"].imag).all().item()
                         and torch.isfinite(output["raw_cmos"]).all().item()
                         and torch.isfinite(saved["direct_raw_cmos"][grid]).all().item(),
                         f"existing {role} measurement shard {shard_no} has invalid {grid} output")
            for local in range(stop - start):
                direct = {grid: saved["direct_raw_cmos"][grid][local:local + 1]
                          for grid in GRIDS}
                outputs = {grid: {key: saved["candidate_outputs"][grid][key][local:local + 1]
                                  for key in ("coefficients", "raw_cmos")} for grid in GRIDS}
                metrics = _metric_row(outputs, expected_targets[local:local + 1], direct)
                row = {"index": start + local, "label": int(expected_labels[local]),
                       "proposal_id": int(expected_proposals[local]),
                       **{key: float(metrics[key]) for key in METRIC_KEYS},
                       "row_passed": all(float(metrics[key]) <= GATE_LIMIT for key in METRIC_KEYS)}
                _json_values_equal(row, saved["rows"][local], f"{role}.saved_shard{shard_no}.row{local}")
                rows.append(row)
            if old_g1_population is not None:
                _require(torch.equal(saved.get("old_g1_labels"), old_g1_population["labels"][start:stop])
                         and torch.equal(saved.get("old_g1_proposal_ids"),
                                         old_g1_population["proposal_ids"][start:stop])
                         and tuple(saved.get("old_g1_direct_raw_cmos_2048", torch.empty(0)).shape[0:1])
                         == (stop - start,)
                         and set(saved.get("frozen_v2_outputs", {})) == set(GRIDS)
                         and tuple(saved.get("frozen_v2_commands", torch.empty(0)).shape)
                         == (stop - start, 1, 200, 200),
                         f"existing {role} shard lost its frozen V2 or old G1 comparison")
            continue
        target_cpu = population["coefficients"][start:stop].to(torch.complex128)
        target = target_cpu.to(device=device, dtype=torch.complex128)
        features = projected._normalized_features(target_cpu, scale, device)
        commands = []
        frozen_commands = []
        with torch.no_grad():
            for local in range(stop - start):
                command, _ = projected._map_amplitude(
                    mapper, features[local:local + 1], target[local:local + 1],
                    matrix, inverse, scale)
                commands.append(command.detach())
                if frozen_mapper is not None:
                    baseline, _ = projected._map_amplitude(
                        frozen_mapper, features[local:local + 1], target[local:local + 1],
                        matrix, inverse, scale)
                    frozen_commands.append(baseline.detach())
            command_batch = torch.cat(commands, dim=0).to(torch.float32)
            candidate_outputs = {grid: projected._forward_amplitude(models[grid], command_batch)
                                 for grid in GRIDS}
            direct_outputs = {grid: projected._forward_target(models[grid], target)
                              for grid in GRIDS}
            frozen_outputs = ({grid: projected._forward_amplitude(
                models[grid], torch.cat(frozen_commands, dim=0).to(torch.float32))
                              for grid in GRIDS} if frozen_mapper is not None else None)
            old_direct_raw = None
            old_ids = old_labels = None
            if old_g1_population is not None:
                old_coeff = old_g1_population["coefficients"][start:stop].to(
                    device=device, dtype=torch.complex128)
                old_direct_raw = projected._forward_target(models[2048], old_coeff)["raw_cmos"]
                old_ids = old_g1_population["proposal_ids"][start:stop].clone()
                old_labels = old_g1_population["labels"][start:stop].clone()

        shard_rows = []
        for local in range(stop - start):
            index = start + local
            one = {grid: {"coefficients": candidate_outputs[grid]["coefficients"][local:local + 1],
                          "raw_cmos": candidate_outputs[grid]["raw_cmos"][local:local + 1]}
                   for grid in GRIDS}
            direct = {grid: direct_outputs[grid]["raw_cmos"][local:local + 1] for grid in GRIDS}
            metrics = _metric_row(one, target[local:local + 1], direct)
            row = {"index": index, "label": int(population["labels"][index]),
                   "proposal_id": int(population["proposal_ids"][index]),
                   **{key: float(metrics[key]) for key in METRIC_KEYS},
                   "row_passed": all(float(metrics[key]) <= GATE_LIMIT for key in METRIC_KEYS)}
            shard_rows.append(row)
            rows.append(row)

        payload: dict[str, Any] = {
            "schema": f"{TASK_ID}.measurement_shard.v1", "task_id": TASK_ID,
            "cohort_role": role, "start": start, "stop": stop,
            "indices": population["source_indices"][start:stop].clone(),
            "labels": population["labels"][start:stop].clone(),
            "proposal_ids": population["proposal_ids"][start:stop].clone(),
            "target_coefficients": target_cpu.clone(),
            "commands": command_batch.detach().cpu().clone(),
            "candidate_outputs": {grid: {key: value.detach().cpu().clone()
                                           for key, value in candidate_outputs[grid].items()
                                           if key in {"coefficients", "raw_cmos"}}
                                  for grid in GRIDS},
            "direct_raw_cmos": {grid: direct_outputs[grid]["raw_cmos"].detach().cpu().clone()
                                for grid in GRIDS},
            "rows": shard_rows,
        }
        if frozen_outputs is not None:
            payload["frozen_v2_commands"] = torch.cat(frozen_commands).detach().cpu().clone()
            payload["frozen_v2_outputs"] = {grid: {key: value.detach().cpu().clone()
                                                    for key, value in frozen_outputs[grid].items()
                                                    if key in {"coefficients", "raw_cmos"}}
                                             for grid in GRIDS}
        if old_direct_raw is not None:
            payload["old_g1_labels"] = old_labels
            payload["old_g1_proposal_ids"] = old_ids
            payload["old_g1_direct_raw_cmos_2048"] = old_direct_raw.detach().cpu().clone()
        _save_torch(directory / f"shard_{shard_no:04d}.pt", payload)
        _write_json(directory / f"progress_{shard_no:04d}.json", {
            "task_id": TASK_ID, "population_role": role, "completed_rows": stop,
            "total_rows": len(population["labels"])})
    _require(len(list(directory.glob("shard_*.pt"))) == expected_shard_count,
             f"{role} measurement did not produce every fixed shard")
    rows = []
    for shard_no in range(expected_shard_count):
        shard = torch.load(directory / f"shard_{shard_no:04d}.pt", map_location="cpu",
                           weights_only=False)
        rows.extend(shard["rows"])
    _atomic_csv(directory / "rows.csv", rows)
    _write_json(directory / "summary.json", _summarize_rows(rows))
    return rows


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def stats(items: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
        values = np.asarray([float(row[key]) for row in items], dtype=np.float64)
        return {"mean": float(values.mean()), "rms": float(np.sqrt(np.mean(np.square(values)))),
                "median": float(np.median(values)), "max": float(values.max())}

    per_class = {}
    for label in range(10):
        selected = [row for row in rows if int(row["label"]) == label]
        per_class[str(label)] = {key: stats(selected, key) for key in METRIC_KEYS}
    return {"count": len(rows), "passed_count": sum(bool(row["row_passed"]) for row in rows),
            "all_rows_passed": bool(rows) and all(bool(row["row_passed"]) for row in rows),
            "metrics": {key: stats(rows, key) for key in METRIC_KEYS}, "per_class": per_class}


def _semantic_final_gate(summary: Mapping[str, Any], config: Mapping[str, Any]
                         ) -> dict[str, Any]:
    conditions = summary["conditions"]
    reference = conditions["old_g1_direct_decoder"]
    candidate = conditions["fixed_frame_mapped"]
    max_drop = float(config["measurement"]["semantic_accuracy_drop_max_absolute"])
    min_diversity = float(config["measurement"]["class_diversity_retention_min"])
    accuracy = {}
    for family in ("ED", "EC"):
        ref_value = float(reference["evaluators"][family]["accuracy"])
        candidate_value = float(candidate["evaluators"][family]["accuracy"])
        accuracy[family] = {"old_g1_reference": ref_value, "fixed_frame_candidate": candidate_value,
                           "drop": ref_value - candidate_value,
                           "permitted_drop": max_drop,
                           "old_g1_per_class_recall": reference["evaluators"][family]["per_class_recall"],
                           "fixed_frame_per_class_recall": candidate["evaluators"][family]["per_class_recall"]}
    diversity = {}
    for label in range(CLASS_COUNT):
        ref_value = float(summary["same_class_diversity"]["old_g1_direct_decoder"]
                          ["by_class"][str(label)]["mean_pairwise_normalized_l2"])
        candidate_value = float(summary["same_class_diversity"]["fixed_frame_mapped"]
                                ["by_class"][str(label)]["mean_pairwise_normalized_l2"])
        diversity[str(label)] = {"old_g1_reference": ref_value, "fixed_frame_candidate": candidate_value,
                                 "retention_ratio": candidate_value / max(ref_value, 1.0e-30),
                                 "minimum_retention": min_diversity}
    passed = (all(item["drop"] <= max_drop for item in accuracy.values())
              and all(item["retention_ratio"] >= min_diversity for item in diversity.values()))
    return {"passed": passed, "reference_condition": "old_g1_direct_decoder",
            "candidate_condition": "fixed_frame_mapped", "accuracy": accuracy,
            "class_diversity": diversity,
            "human_readability_measured": False, "evaluator_is_proxy": True}


def _load_shard_outputs(directory: Path, *, condition: str = "candidate_outputs",
                        grid: int = 2048) -> torch.Tensor:
    paths = sorted(directory.glob("shard_*.pt"))
    values = []
    for expected, path in enumerate(paths):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        source = payload[condition]
        if condition == "candidate_outputs" or condition == "frozen_v2_outputs":
            values.append(source[grid]["raw_cmos"])
        elif condition == "direct_raw_cmos":
            values.append(source[grid])
        elif condition == "old_g1_direct_raw_cmos_2048":
            values.append(source)
        else:
            raise ValueError(f"unknown saved measurement condition: {condition}")
        _require(payload.get("start") == expected * 10,
                 f"measurement shard ordering changed in {path.name}")
    return torch.cat(values)


def _draw_contact_sheet(path: Path, readouts: Mapping[str, torch.Tensor],
                        labels: torch.Tensor) -> None:
    conditions = tuple(readouts)
    scale = 6
    label_width = 150
    canvas = Image.new("RGB", (len(conditions) * 28 * scale + label_width,
                                CLASS_COUNT * 2 * 28 * scale), "white")
    draw = ImageDraw.Draw(canvas)
    for column, condition in enumerate(conditions):
        draw.text((column * 28 * scale + 2, 2), condition, fill="black")
    for label in range(CLASS_COUNT):
        indices = [i for i, value in enumerate(labels.tolist()) if int(value) == label][:2]
        for offset, index in enumerate(indices):
            row = label * 2 + offset
            for column, condition in enumerate(conditions):
                image = torch.round(readouts[condition][index, 0].clamp(0, 1) * 255).byte().numpy()
                rendered = Image.fromarray(image).resize((28 * scale, 28 * scale), Image.Resampling.NEAREST)
                canvas.paste(rendered.convert("RGB"), (column * 28 * scale, row * 28 * scale))
            draw.text((len(conditions) * 28 * scale + 5, row * 28 * scale + 4),
                      f"class {label}, sample {offset + 1}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _write_artifact_manifest(evidence_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(item for item in evidence_dir.rglob("*") if item.is_file()
                       and item.name != "artifact_manifest.json"):
        files.append({"path": path.relative_to(evidence_dir).as_posix(),
                      "bytes": path.stat().st_size})
    manifest = {"schema": f"{TASK_ID}.artifact_manifest.v1", "task_id": TASK_ID,
                "files": files, "total_bytes": sum(row["bytes"] for row in files)}
    _write_json(evidence_dir / "artifact_manifest.json", manifest)
    return manifest


def _measure(config: Mapping[str, Any]) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "terminal measurements require AutoDL CUDA")
    _require(ATTEMPT_DIR.is_dir() and not (ATTEMPT_DIR / "final_decision.json").is_file(),
             "measure requires an open attempt and cannot reopen a terminal result")
    _set_numerics()
    device = torch.device("cuda:0")
    inputs = _load_inputs(config)
    camera_area = _camera_area(inputs, device)
    _require(abs(camera_area - 6.25e-12) <= 1.0e-22,
             f"frozen camera pixel area differs from 6.25e-12 m^2: {camera_area:.17g}")
    population = _load_selected_population()
    identity_payload, _ = _load_ready(config, population)
    terminal_path = ATTEMPT_DIR / "checkpoints" / "terminal_checkpoint.pt"
    _require(terminal_path.is_file(), "exact 15,000-update terminal checkpoint is required", FileNotFoundError)
    checkpoint = torch.load(terminal_path, map_location="cpu", weights_only=False)
    _require(checkpoint.get("task_id") == TASK_ID and checkpoint.get("completed_updates") == UPDATE_BUDGET
             and checkpoint.get("run_id") == identity_payload["identity"]["run_id"]
             and checkpoint.get("history_count") == UPDATE_BUDGET
             and checkpoint.get("lineage_mapper_updates") == SOURCE_MAPPER_UPDATES + UPDATE_BUDGET,
             "measurement checkpoint is not the authorized terminal state")
    mapper = _new_mapper(inputs, device, checkpoint["mapper_state"])
    mapper.eval()
    binding_path = ATTEMPT_DIR / "measurement_mapper_binding.pt"
    new_binding = {
        "schema": f"{TASK_ID}.mapper_binding.v1", "task_id": TASK_ID,
        "run_id": checkpoint["run_id"], "checkpoint_step": UPDATE_BUDGET,
        "loaded_mapper_state": exposure._copy_state(mapper.state_dict()),
        "selected_alpha": checkpoint["selected_alpha"], "engine": "response_matrix",
        "single_sample_command_generation": True}
    if binding_path.exists():
        saved_binding = torch.load(binding_path, map_location="cpu", weights_only=False)
        _require(saved_binding.get("schema") == new_binding["schema"]
                 and saved_binding.get("run_id") == new_binding["run_id"]
                 and saved_binding.get("checkpoint_step") == UPDATE_BUDGET
                 and _compare_states(saved_binding.get("loaded_mapper_state", {}),
                                    new_binding["loaded_mapper_state"]) == 0.0,
                 "existing terminal mapper binding differs from the explicit checkpoint")
    else:
        _save_torch(binding_path, new_binding)
    source_mapper = _new_mapper(inputs, device)
    source_mapper.eval()

    old_holdout = _old_g1_cohort(
        inputs, inputs["scale"], count=HOLDOUT_PER_CLASS,
        seed_base=int(config["population"]["holdout_seed_base"]),
        role="old_g1_direct_decoder_reference_holdout2000", config=config)
    old_accounting = old_holdout["proposal_accounting"]
    old_population_path = ATTEMPT_DIR / "old_g1_semantic_reference_population.pt"
    if old_population_path.exists():
        saved_old = torch.load(old_population_path, map_location="cpu", weights_only=False)
        _require(saved_old.get("schema") == f"{TASK_ID}.old_g1_holdout.v1"
                 and torch.equal(saved_old["population"]["coefficients"], old_holdout["coefficients"])
                 and torch.equal(saved_old["population"]["proposal_ids"], old_holdout["proposal_ids"]),
                 "existing old G1 semantic reference population differs from its registered seed stream")
    else:
        _save_torch(old_population_path, {
            "schema": f"{TASK_ID}.old_g1_holdout.v1", "task_id": TASK_ID,
            "population": old_holdout, "sampling_accounting": old_accounting,
            "comparison_role": "old_g1_direct_decoder_semantic_reference"})

    cohort_rows = {}
    for role in ("train", "development", "holdout"):
        cohort = population[role]
        cohort_rows[role] = _measure_one_cohort(
            inputs, cohort, mapper, device, role=role,
            old_g1_population=old_holdout if role == "holdout" else None,
            frozen_mapper=source_mapper if role == "holdout" else None)

    holdout_dir = ATTEMPT_DIR / "measurements" / "holdout"
    semantic_raw = {
        "old_g1_direct_decoder": _load_shard_outputs(holdout_dir,
                                                      condition="old_g1_direct_raw_cmos_2048"),
        "new_prior_direct_decoder": _load_shard_outputs(holdout_dir, condition="direct_raw_cmos"),
        "frozen_v2_mapped": _load_shard_outputs(holdout_dir, condition="frozen_v2_outputs"),
        "fixed_frame_mapped": _load_shard_outputs(holdout_dir, condition="candidate_outputs"),
    }
    semantic_population = {"labels": population["holdout"]["labels"],
                           "proposal_ids": population["holdout"]["proposal_ids"]}
    semantic, readouts = exposure._semantic_summary(inputs, semantic_population, semantic_raw)
    semantic["evidence_role"] = "new_fixed_holdout_2000_semantic_proxy"
    semantic_gate = _semantic_final_gate(semantic, config)
    _write_json(ATTEMPT_DIR / "holdout_semantic_summary.json", semantic)
    _write_json(ATTEMPT_DIR / "holdout_semantic_gate.json", semantic_gate)
    _save_torch(ATTEMPT_DIR / "holdout_semantic_readouts.pt", {
        "task_id": TASK_ID, "labels": semantic_population["labels"],
        "proposal_ids": semantic_population["proposal_ids"], "readouts": readouts})
    _draw_contact_sheet(ATTEMPT_DIR / "fixed_holdout_examples.png", readouts,
                        semantic_population["labels"])

    summaries = {role: _summarize_rows(rows) for role, rows in cohort_rows.items()}
    numeric_pass = bool(summaries["holdout"]["all_rows_passed"])
    final_pass = numeric_pass and bool(semantic_gate["passed"])
    state = "VALID_COMPLETED" if final_pass else "VALID_FAILED"
    decision = {"task_id": TASK_ID, "run_id": checkpoint["run_id"], "state": state,
                "evidence_tier": "diagnostic_only", "selected_alpha": checkpoint["selected_alpha"],
                "new_scientific_updates": UPDATE_BUDGET,
                "source_mapper_updates": SOURCE_MAPPER_UPDATES,
                "lineage_mapper_updates": SOURCE_MAPPER_UPDATES + UPDATE_BUDGET,
                "adam_steps": UPDATE_BUDGET, "selected_engine": "response_matrix",
                "numerical_final_gate_population": "new_final_holdout2000",
                "holdout_all_six_metrics_passed": numeric_pass,
                "holdout_numeric_summary": summaries["holdout"],
                "training_and_development_are_report_only": True,
                "training_summary": summaries["train"],
                "development_summary": summaries["development"],
                "semantic_proxy_gate": semantic_gate,
                "human_readability_measured": False, "physical_hardware_measured": False,
                "final_gate_passed": final_pass}
    _write_json(ATTEMPT_DIR / "final_decision.json", decision)
    _write_json(ATTEMPT_DIR / "task_state.json", {
        "task_id": TASK_ID, "run_id": checkpoint["run_id"],
        "state": "COMPLETED" if final_pass else "VALID_FAILED", "phase": "CLOSED",
        "completed_updates": UPDATE_BUDGET, "authorized_updates": UPDATE_BUDGET,
        "terminal_measurement_complete": True})
    report = _write_final_report(decision, summaries, semantic, semantic_gate)
    _write_json(ATTEMPT_DIR / "final_report.json", report)
    _write_artifact_manifest(ATTEMPT_DIR)
    return decision


def _write_final_report(decision: Mapping[str, Any], summaries: Mapping[str, Any],
                        semantic: Mapping[str, Any], semantic_gate: Mapping[str, Any]) -> dict[str, Any]:
    report = {
        "结果与状态": {"state": decision["state"], "task_id": TASK_ID,
                       "selected_alpha": decision["selected_alpha"],
                       "new_updates": UPDATE_BUDGET, "lineage_mapper_updates": SOURCE_MAPPER_UPDATES + UPDATE_BUDGET},
        "科学认识": {"raw_physical_phase_prior": True,
                     "per_sample_gauge_rotation_before_fit": False,
                     "regularized_covariance_scaled_with_global_alpha": True,
                     "mean_amplitude_replay_not_mean_intensity_sqrt": True,
                     "semantic_proxy": semantic, "semantic_gate": semantic_gate},
        "目标缺口": {"numerical_gate_population": "new_final_holdout2000",
                     "threshold": GATE_LIMIT, "all_six_metrics_required": list(METRIC_KEYS),
                     "training": summaries["train"], "historical_development": summaries["development"],
                     "final_holdout": summaries["holdout"]},
        "工程完成与验证": {"preflight": _json(PREFLIGHT_DIR / "preflight_summary.json"),
                            "terminal_checkpoint": str(ATTEMPT_DIR / "checkpoints" / "terminal_checkpoint.pt"),
                            "single_sample_instruction_generation": True,
                            "measurement_replay_binding": str(ATTEMPT_DIR / "measurement_mapper_binding.pt")},
        "决策影响与下一步": {"final_gate_passed": decision["final_gate_passed"],
                             "boundary": "diagnostic-only; no human readability or hardware claim",
                             "stop_point": "no automatic extra updates or successor experiment"},
    }
    return report


def _json_values_equal(expected: Any, actual: Any, location: str, tolerance: float = 1.0e-14) -> None:
    if isinstance(expected, Mapping):
        _require(isinstance(actual, Mapping) and set(expected) == set(actual),
                 f"saved summary keys differ at {location}")
        for key, value in expected.items():
            _json_values_equal(value, actual[key], f"{location}.{key}", tolerance)
    elif isinstance(expected, list):
        _require(isinstance(actual, list) and len(expected) == len(actual),
                 f"saved summary list differs at {location}")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _json_values_equal(left, right, f"{location}[{index}]", tolerance)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        _require(isinstance(actual, (int, float)) and abs(float(expected) - float(actual)) <= tolerance,
                 f"saved numeric summary differs at {location}")
    else:
        _require(expected == actual, f"saved summary value differs at {location}")


def _verify_artifact_manifest(evidence_dir: Path, *, require_transfer: bool) -> dict[str, Any]:
    path = evidence_dir / "artifact_manifest.json"
    _require(path.is_file(), "artifact byte manifest is missing", FileNotFoundError)
    manifest = _json(path)
    _require(manifest.get("schema") == f"{TASK_ID}.artifact_manifest.v1"
             and manifest.get("task_id") == TASK_ID and isinstance(manifest.get("files"), list),
             "artifact byte manifest identity is invalid")
    seen = set()
    for row in manifest["files"]:
        relative = str(row["path"])
        _require(relative and not relative.startswith("/") and ".." not in Path(relative).parts
                 and relative not in seen, "artifact manifest contains an unsafe or duplicate path")
        seen.add(relative)
        target = (evidence_dir / relative).resolve()
        _require(target.is_relative_to(evidence_dir.resolve()) and target.is_file()
                 and target.stat().st_size == int(row["bytes"]),
                 f"artifact actual byte size differs from manifest: {relative}")
    _require(sum(int(row["bytes"]) for row in manifest["files"]) == int(manifest["total_bytes"]),
             "artifact manifest total byte count is inconsistent")
    if require_transfer:
        receipt_path = evidence_dir / "transfer_receipt.json"
        _require(receipt_path.is_file(), "Windows transfer receipt is required for backup verification",
                 FileNotFoundError)
        receipt = _json(receipt_path)
        _require(receipt.get("status") == "COMPLETE" and isinstance(receipt.get("files"), list),
                 "transfer receipt is incomplete")
        receipt_files = receipt["files"]
        paths = [str(row.get("path", "")) for row in receipt_files]
        _require(len(paths) == len(set(paths)) and set(paths) == seen,
                 "transfer receipt file set differs from the experiment artifact manifest")
        for row in receipt_files:
            relative = str(row["path"])
            actual_size = (evidence_dir / relative).stat().st_size
            _require(row.get("status") == "COPIED"
                     and int(row.get("remote_size_bytes", -1)) == actual_size
                     and int(row.get("local_size_bytes", -1)) == actual_size
                     and int(row.get("scp_exit_code", -1)) == 0,
                     f"transfer receipt size/status does not match local file: {relative}")
    return manifest


def _verify_saved(config: Mapping[str, Any], evidence_dir: Path,
                  report_dir: Path | None = None) -> dict[str, Any]:
    evidence_dir = evidence_dir.resolve()
    attempt = ATTEMPT_DIR.resolve()
    backup = _resolve(config["output"]["windows_backup_root"]).resolve()
    _require(evidence_dir in {attempt, backup},
             "saved-evidence consumer requires the explicit attempt or its Windows backup", ValueError)
    require_transfer = evidence_dir == backup
    manifest = _verify_artifact_manifest(evidence_dir, require_transfer=require_transfer)
    decision = _json(evidence_dir / "final_decision.json")
    state = _json(evidence_dir / "task_state.json")
    checkpoint = torch.load(evidence_dir / "checkpoints" / "terminal_checkpoint.pt",
                            map_location="cpu", weights_only=False)
    initialization = torch.load(evidence_dir / "mapper_initialization.pt",
                                map_location="cpu", weights_only=False)
    scientific_attempt_identity = _json(evidence_dir / "scientific_attempt_identity.json")
    binding = torch.load(evidence_dir / "measurement_mapper_binding.pt",
                         map_location="cpu", weights_only=False)
    local_population_payload = torch.load(evidence_dir / "selected_population.pt",
                                          map_location="cpu", weights_only=False)
    preflight_dir = evidence_dir / "preflight_evidence"
    qualification_dir = evidence_dir / "qualification_evidence"
    prior_dir = evidence_dir / "prior_evidence"
    receipt = PreflightReceipt.from_mapping(_json(preflight_dir / "preflight_receipt.json"))
    preflight_summary = _json(preflight_dir / "preflight_summary.json")
    preflight_identity_payload = _json(preflight_dir / "preflight_identity.json")
    preflight_identity = ResearchRunIdentity.from_mapping(preflight_identity_payload["run_identity"])
    recovery = torch.load(preflight_dir / "engineering_recovery.pt", map_location="cpu",
                          weights_only=False)
    schedule = _schedule(config)
    _validate_recovery_evidence(preflight_dir, recovery, schedule)
    history = checkpoint.get("history", [])
    optimizer_state = checkpoint.get("optimizer_state", {})
    expected_counts = torch.bincount(schedule.flatten(), minlength=TRAIN_ROWS)
    adam = _adam_steps(optimizer_state)
    _require(local_population_payload.get("task_id") == TASK_ID
             and checkpoint.get("schema") == f"{TASK_ID}.checkpoint.v1"
             and checkpoint.get("task_id") == TASK_ID
             and checkpoint.get("completed_updates") == UPDATE_BUDGET
             and checkpoint.get("history_count") == UPDATE_BUDGET
             and len(history) == UPDATE_BUDGET
             and checkpoint.get("lineage_mapper_updates") == SOURCE_MAPPER_UPDATES + UPDATE_BUDGET
             and checkpoint.get("source_mapper_updates") == SOURCE_MAPPER_UPDATES
             and float(checkpoint.get("selected_alpha", -1.0)) == float(local_population_payload["selected_alpha"])
             and torch.equal(checkpoint.get("schedule"), schedule)
             and checkpoint.get("schedule_cursor") == UPDATE_BUDGET
             and torch.equal(checkpoint.get("exposure_counts"), expected_counts)
             and adam == [UPDATE_BUDGET] * 4
             and decision.get("task_id") == TASK_ID and decision.get("run_id") == checkpoint.get("run_id")
             and decision.get("new_scientific_updates") == UPDATE_BUDGET
             and decision.get("final_gate_passed") is (decision.get("state") == "VALID_COMPLETED")
             and state.get("phase") == "CLOSED" and state.get("completed_updates") == UPDATE_BUDGET
             and _ready_receipt_checks_pass(receipt)
             and receipt.source_semantics_id == preflight_identity.execution_source_semantics_id
             and preflight_identity.task_id == TASK_ID
             and preflight_identity.run_id == checkpoint.get("run_id")
             and preflight_summary.get("status") == "READY"
             and preflight_summary.get("engineering_adam_calls") == 5
             and preflight_summary.get("selected_alpha") == local_population_payload.get("selected_alpha")
             and recovery.get("engineering_adam_calls") == 5
             and initialization.get("task_id") == TASK_ID
             and _registered_source_path_matches(
                 initialization.get("source_checkpoint", ""),
                 _resolve(config["source"]["v2_terminal_checkpoint"]))
             and initialization.get("new_adam_state") == {}
             and _compare_states(initialization["mapper_state"],
                                 torch.load(_resolve(config["source"]["v2_terminal_checkpoint"]),
                                            map_location="cpu", weights_only=False)["mapper_state"]) == 0.0
             and binding.get("schema") == f"{TASK_ID}.mapper_binding.v1"
             and binding.get("task_id") == TASK_ID
             and binding.get("run_id") == checkpoint.get("run_id")
             and binding.get("checkpoint_step") == UPDATE_BUDGET
             and _compare_states(binding["loaded_mapper_state"], checkpoint["mapper_state"]) == 0.0,
             "terminal checkpoint, training history, initialization, READY, or mapper binding is invalid")
    attempt_identity = ResearchRunIdentity.from_mapping(scientific_attempt_identity["run_identity"])
    _require(scientific_attempt_identity.get("task_id") == TASK_ID
             and scientific_attempt_identity.get("run_id") == checkpoint.get("run_id")
             and attempt_identity.run_id == checkpoint.get("run_id")
             and attempt_identity.scientific_attempt_id == f"{checkpoint['run_id']}:attempt_01",
             "saved scientific attempt identity is not the READY-bound attempt")

    # Verify the fixed exposure schedule and every consumed sample row.
    counts = torch.zeros(TRAIN_ROWS, dtype=torch.long)
    for step, row in enumerate(history):
        batch = schedule[step]
        counts.index_add_(0, batch, torch.ones_like(batch))
        _require(row.get("completed_update") == step + 1
                 and row.get("sample_indices") == batch.tolist()
                 and row.get("labels") == list(range(10))
                 and row.get("loss_is_pre_update") is True
                 and row.get("target_exposure_counts_after_update") == counts[batch].tolist()
                 and all(math.isfinite(float(row[key])) for key in
                         ("loss", "coefficient_loss", "box_loss", "clipped_fraction")),
                 f"training history schedule or loss record is invalid at update {step + 1}")
    _require(torch.equal(counts, torch.full((TRAIN_ROWS,), EXPOSURES)),
             "terminal schedule does not give each fixed training target 750 exposures")

    inputs = _load_inputs(config)
    camera_area = _camera_area(inputs, torch.device("cuda:0") if torch.cuda.is_available()
                               else torch.device("cpu"))
    _require(abs(camera_area - 6.25e-12) <= 1.0e-22,
             f"frozen camera pixel area differs from 6.25e-12 m^2: {camera_area:.17g}")
    selected = _load_selected_population(qualification_dir)
    alpha = float(selected["selected_alpha"])
    for role in ("train", "development", "holdout"):
        saved_role = local_population_payload[role]
        source_role = selected[role]
        _require(torch.equal(saved_role["coefficients"], source_role["coefficients"])
                 and torch.equal(saved_role["labels"], source_role["labels"])
                 and torch.equal(saved_role["proposal_ids"], source_role["proposal_ids"]),
                 f"saved {role} population differs from the fixed qualified candidate")
    fit, prior_payload = _load_prior_artifacts(prior_dir)
    qualification_summary = _json(qualification_dir / "qualification_summary.json")
    candidate_results = qualification_summary.get("candidate_results", [])
    _require(qualification_summary.get("status") == "PRIOR_CANDIDATE_QUALIFIED"
             and qualification_summary.get("training_ready") is False
             and qualification_summary.get("training_witnesses", {}).get("count") == TRAIN_ROWS
             and qualification_summary.get("training_witnesses", {}).get("all_six_full_optics_metrics_passed") is True,
             "candidate qualification summary does not establish all fixed training witnesses")
    candidate_alphas = [float(row["alpha"]) for row in candidate_results]
    _require(candidate_alphas == list(ALPHAS[:len(candidate_alphas)])
             and len(candidate_alphas) <= len(ALPHAS),
             "candidate qualification did not follow the fixed descending alpha order")
    first_qualified_alpha = next((float(row["alpha"]) for row in candidate_results
                                  if row.get("status") == "ALL_TRAINING_WITNESSES_PASSED"
                                  and row.get("semantic_gate_passed") is True), None)
    _require(first_qualified_alpha == alpha
             and float(qualification_summary.get("selected_alpha", -1.0)) == alpha,
             "selected alpha is not the first candidate passing semantic and training-witness rules")
    for candidate_result in candidate_results:
        candidate_alpha = float(candidate_result["alpha"])
        _require(candidate_alpha in ALPHAS,
                 "qualification summary contains an unregistered covariance contraction")
        if candidate_alpha == alpha:
            _require(candidate_result.get("status") == "ALL_TRAINING_WITNESSES_PASSED"
                     and candidate_result.get("semantic_gate_passed") is True
                     and candidate_result.get("training_witnesses_passed") == TRAIN_ROWS,
                     "selected candidate is not fully qualified")
            break
        _require(candidate_alpha > alpha and candidate_result.get("status") in {
            "SEMANTIC_QUALIFICATION_FAILED", "TRAINING_REFERENCE_INSUFFICIENT"},
            "an earlier candidate was neither rejected nor selected")
    old_dev_payload = torch.load(prior_dir / "old_g1_semantic_reference_dev200.pt",
                                 map_location="cpu", weights_only=False)
    regenerated_old_dev = _old_g1_cohort(
        inputs, inputs["scale"], count=DEV_PER_CLASS,
        seed_base=int(config["population"]["development_seed_base"]),
        role="old_g1_direct_decoder_reference_dev200", config=config)
    _require(torch.equal(old_dev_payload["population"]["coefficients"],
                         regenerated_old_dev["coefficients"])
             and torch.equal(old_dev_payload["population"]["proposal_ids"],
                             regenerated_old_dev["proposal_ids"]),
             "old G1 development semantic reference does not match its fixed seed stream")
    baseline_semantic = _candidate_semantics_from_readouts(
        inputs, old_dev_payload["population"], old_dev_payload["readouts"])
    for candidate_result in candidate_results:
        candidate_alpha = float(candidate_result["alpha"])
        candidate_dir = prior_dir / f"alpha_{candidate_alpha:.3f}"
        candidate_population_payload = torch.load(candidate_dir / "candidate_population.pt",
                                                  map_location="cpu", weights_only=False)
        candidate_semantic_payload = torch.load(candidate_dir / "development_semantic_outputs.pt",
                                                map_location="cpu", weights_only=False)
        expected_train, _ = _cohort(
            fit, inputs["scale"], alpha=candidate_alpha, role="fixed_train200",
            count=TRAIN_PER_CLASS, seed_base=int(config["population"]["train_seed_base"]),
            config=config)
        expected_development, _ = _cohort(
            fit, inputs["scale"], alpha=candidate_alpha, role="fixed_development200",
            count=DEV_PER_CLASS,
            seed_base=int(config["population"]["development_seed_base"]), config=config)
        _require(torch.equal(candidate_population_payload["train"]["coefficients"],
                             expected_train["coefficients"])
                 and torch.equal(candidate_population_payload["train"]["proposal_ids"],
                                 expected_train["proposal_ids"])
                 and torch.equal(candidate_population_payload["development"]["coefficients"],
                                 expected_development["coefficients"])
                 and torch.equal(candidate_population_payload["development"]["proposal_ids"],
                                 expected_development["proposal_ids"])
                 and torch.equal(candidate_semantic_payload["population"]["coefficients"],
                                 expected_development["coefficients"])
                 and torch.equal(candidate_semantic_payload["population"]["proposal_ids"],
                                 expected_development["proposal_ids"]),
                 f"alpha {candidate_alpha:.3f} candidate population is not fixed-seed reproducible")
        recomputed_semantic = _candidate_semantics_from_readouts(
            inputs, candidate_semantic_payload["population"], candidate_semantic_payload["readouts"])
        saved_semantic = candidate_semantic_payload["semantic_summary"]
        saved_semantic_gate_inputs = {
            "conditions": {"direct_decoder": {
                "evaluators": saved_semantic["conditions"]["direct_decoder"]["evaluators"],
                "ED_EC_prediction_agreement": saved_semantic["conditions"]["direct_decoder"][
                    "ED_EC_prediction_agreement"]}},
            "same_class_diversity": saved_semantic["same_class_diversity"]}
        _json_values_equal(recomputed_semantic, saved_semantic_gate_inputs,
                           f"alpha_{candidate_alpha:.3f}.development_semantics")
        recomputed_semantic_gate_passed, recomputed_semantic_gate = _semantic_candidate_passes(
            recomputed_semantic, baseline_semantic, config)
        _json_values_equal(recomputed_semantic_gate, _json(candidate_dir / "semantic_gate.json"),
                           f"alpha_{candidate_alpha:.3f}.semantic_gate")
        _require(bool(candidate_result["semantic_gate_passed"]) is recomputed_semantic_gate_passed
                 and bool(candidate_result["semantic_gate"]["passed"]) is recomputed_semantic_gate_passed,
                 f"alpha {candidate_alpha:.3f} semantic candidate decision does not recompute")
        if not recomputed_semantic_gate_passed:
            _require(candidate_result.get("status") == "SEMANTIC_QUALIFICATION_FAILED"
                     and candidate_result.get("training_witnesses_passed") == 0,
                     f"alpha {candidate_alpha:.3f} semantic failure was not reflected in candidate order")
        elif candidate_alpha < alpha:
            _require(candidate_result.get("status") == "TRAINING_REFERENCE_INSUFFICIENT"
                     and candidate_result.get("training_witnesses_passed", 0) < TRAIN_ROWS,
                     f"alpha {candidate_alpha:.3f} training reference failure was not reflected")
        else:
            _require(candidate_alpha == alpha
                     and candidate_result.get("status") == "ALL_TRAINING_WITNESSES_PASSED"
                     and candidate_result.get("training_witnesses_passed") == TRAIN_ROWS,
                     "selected candidate did not pass both semantic and full-optics witness checks")
            break
    selected_witnesses = _load_qualification_witnesses(selected, alpha, qualification_dir)
    preflight_witnesses = torch.load(preflight_dir / "qualification_witnesses.pt",
                                     map_location="cpu", weights_only=False)
    _require(preflight_witnesses.get("task_id") == TASK_ID
             and float(preflight_witnesses.get("selected_alpha", -1.0)) == alpha
             and len(preflight_witnesses.get("rows", [])) == TRAIN_ROWS,
             "READY preflight does not contain the complete selected candidate qualification")
    for index, row in enumerate(preflight_witnesses["rows"]):
        witness = selected_witnesses[index]
        _require(int(row["index"]) == index
                 and int(row["proposal_id"]) == int(witness["proposal_id"])
                 and torch.equal(row["target_coefficients"], witness["target_coefficients"])
                 and torch.equal(row["amplitude_command"], witness["amplitude_command"])
                 and row["metrics"] == witness["metrics"]
                 and row["all_six_metrics_passed"] is True,
                 f"READY preflight witness differs from full candidate qualification at row {index}")
    witness_summary = _verify_qualification_output(
        selected, [{key: value for key, value in row.items()
                    if key != "outputs"} for row in selected_witnesses])
    _require(witness_summary["count"] == TRAIN_ROWS,
             "selected candidate does not contain all fixed full-optics witnesses")
    for index, row in enumerate(selected_witnesses):
        target = selected["train"]["coefficients"][index:index + 1].to(torch.complex128)
        outputs = row.get("outputs", {})
        _require(set(outputs) == set(GRIDS),
                 f"qualification witness {index} is missing a grid output")
        predicted = exposure._forward_matrix(
            row["amplitude_command"].reshape(1, -1).double(), inputs["response_matrix"].cpu())
        actual_2048 = outputs[2048]["coefficients"].to(torch.complex128)
        response_error = float(torch.linalg.vector_norm(predicted - actual_2048)
                               / torch.linalg.vector_norm(actual_2048).clamp_min(1.0e-30))
        direct = {grid: outputs[grid]["direct_raw_cmos"] for grid in GRIDS}
        measured = {grid: {key: outputs[grid][key]
                           for key in ("coefficients", "raw_cmos")} for grid in GRIDS}
        recomputed_metrics = _metric_row(measured, target, direct)
        _json_values_equal({key: float(recomputed_metrics[key]) for key in METRIC_KEYS},
                           {key: float(row["metrics"][key]) for key in METRIC_KEYS},
                           f"qualification_witness_{index}.metrics")
        _require(response_error <= 1.0e-6
                 and all(float(recomputed_metrics[key]) <= GATE_LIMIT for key in METRIC_KEYS),
                 f"qualification witness {index} fails its independently consumed production output")
    for role, count, seed_key in (("train", TRAIN_PER_CLASS, "train_seed_base"),
                                  ("development", DEV_PER_CLASS, "development_seed_base"),
                                  ("holdout", HOLDOUT_PER_CLASS, "holdout_seed_base")):
        regenerated, _ = _cohort(fit, inputs["scale"], alpha=alpha, role=role,
                                  count=count, seed_base=int(config["population"][seed_key]),
                                  config=config)
        _require(torch.equal(regenerated["coefficients"], selected[role]["coefficients"])
                 and torch.equal(regenerated["proposal_ids"], selected[role]["proposal_ids"]),
                 f"{role} fixed coefficients/proposal identities do not reproduce from the registered seed")

    raw_coeff_path = prior_dir / "raw_physical_phase_train_coefficients.pt"
    prior_path = prior_dir / "physical_phase_gaussian_prior.pt"
    raw_train = torch.load(raw_coeff_path, map_location="cpu", weights_only=False)
    saved_prior = torch.load(prior_path, map_location="cpu", weights_only=False)
    _require(raw_train.get("task_id") == TASK_ID and raw_train.get("rows").numel() == 55_000
             and raw_train.get("labels").numel() == 55_000
             and tuple(raw_train.get("coefficients").shape) == (55_000, PORT_COUNT)
             and saved_prior.get("task_id") == TASK_ID,
             "raw-phase 55,000-row prior source or coefficient population is invalid")
    source_replay_path = prior_dir / "source_matrix_replay_100.pt"
    source_replay = torch.load(source_replay_path, map_location="cpu", weights_only=False)
    source_rows = source_replay.get("source_train_rows")
    source_commands = source_replay.get("amplitude_commands")
    production_coefficients = source_replay.get("production_optical_coefficients")
    matrix_coefficients = source_replay.get("frozen_matrix_coefficients")
    per_row_response_error = source_replay.get("relative_error_per_row")
    expected_source_rows = torch.cat([
        raw_train["rows"][raw_train["labels"] == label][:10]
        for label in range(CLASS_COUNT)])
    expected_source_labels = torch.arange(CLASS_COUNT).repeat_interleave(10)
    _require(source_replay.get("schema") == f"{TASK_ID}.source_matrix_replay_100.v1"
             and source_replay.get("task_id") == TASK_ID
             and torch.equal(source_rows, expected_source_rows)
             and tuple(source_rows.shape) == (100,)
             and tuple(source_commands.shape) == (100, 1, 200, 200)
             and tuple(production_coefficients.shape) == (100, PORT_COUNT)
             and tuple(matrix_coefficients.shape) == (100, PORT_COUNT)
             and tuple(per_row_response_error.shape) == (100,)
             and torch.equal(source_replay.get("labels"), expected_source_labels)
             and torch.isfinite(source_commands).all().item()
             and float(source_commands.min()) >= 0.0 and float(source_commands.max()) <= 1.0,
             "100-row production matrix replay evidence has invalid identity or dimensions")
    predicted_source_coefficients = exposure._forward_matrix(
        source_commands.reshape(100, -1).double(), inputs["response_matrix"].cpu())
    independent_source_error = torch.linalg.vector_norm(
        predicted_source_coefficients - production_coefficients.to(torch.complex128), dim=1
    ) / torch.linalg.vector_norm(production_coefficients.to(torch.complex128), dim=1).clamp_min(1.0e-30)
    _require(torch.allclose(matrix_coefficients.to(torch.complex128), predicted_source_coefficients,
                            atol=1.0e-12, rtol=1.0e-12)
             and torch.allclose(per_row_response_error.double(), independent_source_error,
                                atol=1.0e-14, rtol=1.0e-12)
             and float(independent_source_error.max()) <= 1.0e-6
             and abs(float(source_replay.get("maximum_relative_error", math.inf))
                     - float(independent_source_error.max())) <= 1.0e-14,
             "independent 100-row response replay differs from recorded production coefficients")
    fitted_again = _fit_raw_gaussians(raw_train["coefficients"], raw_train["labels"], inputs["scale"])
    for label in range(CLASS_COUNT):
        for key in ("mean", "covariance", "ridge", "cholesky"):
            delta = float((fitted_again[label][key] - saved_prior["fit"][label][key]).abs().max())
            tolerance = 3.0e-13 if key == "cholesky" else 1.0e-14
            _require(delta <= tolerance, f"physical-phase prior {key} differs for class {label}")

    cpu_mapper = projected._new_mapper(torch.device("cpu"), checkpoint["mapper_state"]).eval()
    frozen_mapper = projected._new_mapper(torch.device("cpu"),
                                          inputs["source_mapper"]["mapper_state"]).eval()
    cpu_matrix, cpu_inverse = inputs["matrix"].cpu().double(), inputs["inverse"].cpu().double()
    cpu_scale = inputs["scale"].cpu().double()
    recomputed_rows: dict[str, list[dict[str, Any]]] = {}
    measured_semantic_raw: dict[str, list[torch.Tensor]] = {
        "old_g1_direct_decoder": [], "new_prior_direct_decoder": [],
        "frozen_v2_mapped": [], "fixed_frame_mapped": []}
    terminal_command_binding_max_abs_error = 0.0
    for role in ("train", "development", "holdout"):
        directory = evidence_dir / "measurements" / role
        shards = sorted(directory.glob("shard_*.pt"))
        base = local_population_payload[role]
        expected_shards = len(base["labels"]) // 10
        _require(len(shards) == expected_shards,
                 f"{role} raw production measurement shards are incomplete")
        rows: list[dict[str, Any]] = []
        for shard_index, path in enumerate(shards):
            shard = torch.load(path, map_location="cpu", weights_only=False)
            start, stop = shard_index * 10, shard_index * 10 + 10
            _require(shard.get("task_id") == TASK_ID and shard.get("cohort_role") == role
                     and shard.get("start") == start and shard.get("stop") == stop
                     and torch.equal(shard.get("indices"), base["source_indices"][start:stop])
                     and torch.equal(shard.get("labels"), base["labels"][start:stop])
                     and torch.equal(shard.get("proposal_ids"), base["proposal_ids"][start:stop])
                     and torch.equal(shard.get("target_coefficients"), base["coefficients"][start:stop]),
                     f"{role} shard {shard_index} does not bind the fixed target identities")
            targets = shard["target_coefficients"].to(torch.complex128)
            features = projected._normalized_features(targets, cpu_scale, torch.device("cpu"))
            replayed = []
            replayed_v2 = []
            with torch.no_grad():
                for local in range(10):
                    output, _ = projected._map_amplitude(
                        cpu_mapper, features[local:local + 1], targets[local:local + 1],
                        cpu_matrix, cpu_inverse, cpu_scale)
                    replayed.append(output.cpu())
                    if role == "holdout":
                        old_output, _ = projected._map_amplitude(
                            frozen_mapper, features[local:local + 1], targets[local:local + 1],
                            cpu_matrix, cpu_inverse, cpu_scale)
                        replayed_v2.append(old_output.cpu())
            replayed_commands = torch.cat(replayed)
            command_delta = float((replayed_commands - shard["commands"]).abs().max())
            terminal_command_binding_max_abs_error = max(
                terminal_command_binding_max_abs_error, command_delta)
            _require(command_delta <= 1.0e-5,
                     f"saved {role} SLM command is not produced by the bound terminal mapper")
            if role == "holdout":
                _require(float((torch.cat(replayed_v2) - shard["frozen_v2_commands"]).abs().max()) <= 1.0e-5,
                         "saved frozen V2 baseline command is not produced by its bound source mapper")
            candidate = shard["candidate_outputs"]
            for grid in GRIDS:
                _require(tuple(candidate[grid]["coefficients"].shape) == (10, PORT_COUNT)
                         and torch.isfinite(candidate[grid]["coefficients"].real).all().item()
                         and torch.isfinite(candidate[grid]["coefficients"].imag).all().item()
                         and torch.isfinite(candidate[grid]["raw_cmos"]).all().item()
                         and torch.isfinite(shard["direct_raw_cmos"][grid]).all().item(),
                         f"{role} {grid} production outputs contain missing or nonfinite values")
            predicted_matrix = exposure._forward_matrix(
                shard["commands"], inputs["response_matrix"].cpu())
            actual_coeff = candidate[2048]["coefficients"].to(torch.complex128)
            matrix_error = torch.linalg.vector_norm(predicted_matrix - actual_coeff, dim=1) / torch.linalg.vector_norm(
                actual_coeff, dim=1).clamp_min(1.0e-30)
            _require(float(matrix_error.max()) <= 1.0e-6,
                     f"{role} actual 2048 production coefficient differs from frozen response matrix")
            for local in range(10):
                outputs = {grid: {"coefficients": candidate[grid]["coefficients"][local:local + 1],
                                  "raw_cmos": candidate[grid]["raw_cmos"][local:local + 1]}
                           for grid in GRIDS}
                direct = {grid: shard["direct_raw_cmos"][grid][local:local + 1] for grid in GRIDS}
                metrics = _metric_row(outputs, targets[local:local + 1], direct)
                saved_row = shard["rows"][local]
                row = {"index": start + local, "label": int(base["labels"][start + local]),
                       "proposal_id": int(base["proposal_ids"][start + local]),
                       **{key: float(metrics[key]) for key in METRIC_KEYS},
                       "row_passed": all(float(metrics[key]) <= GATE_LIMIT for key in METRIC_KEYS)}
                _json_values_equal({key: row[key] for key in METRIC_KEYS},
                                   {key: saved_row[key] for key in METRIC_KEYS},
                                   f"{role}.row{start + local}")
                rows.append(row)
            recomputed_rows[role] = rows
            if role == "holdout":
                for key, tensor in (
                    ("new_prior_direct_decoder", shard["direct_raw_cmos"][2048]),
                    ("fixed_frame_mapped", candidate[2048]["raw_cmos"]),
                    ("frozen_v2_mapped", shard["frozen_v2_outputs"][2048]["raw_cmos"]),
                    ("old_g1_direct_decoder", shard["old_g1_direct_raw_cmos_2048"]),
                ):
                    measured_semantic_raw[key].append(tensor)
        recomputed_summary = _summarize_rows(rows)
        saved_summary = _json(directory / "summary.json")
        _json_values_equal(recomputed_summary, saved_summary, f"{role}.summary")

    old_population = torch.load(evidence_dir / "old_g1_semantic_reference_population.pt",
                                map_location="cpu", weights_only=False)
    old_regenerated = _old_g1_cohort(
        inputs, inputs["scale"], count=HOLDOUT_PER_CLASS,
        seed_base=int(config["population"]["holdout_seed_base"]),
        role="old_g1_direct_decoder_reference_holdout2000", config=config)
    _require(old_population.get("task_id") == TASK_ID
             and torch.equal(old_population["population"]["coefficients"], old_regenerated["coefficients"])
             and torch.equal(old_population["population"]["proposal_ids"], old_regenerated["proposal_ids"]),
             "old G1 semantic reference holdout identities do not match their fixed seed stream")
    raw = {key: torch.cat(values) for key, values in measured_semantic_raw.items()}
    semantic_population = {"labels": local_population_payload["holdout"]["labels"],
                           "proposal_ids": local_population_payload["holdout"]["proposal_ids"]}
    semantic, readouts = exposure._semantic_summary(inputs, semantic_population, raw)
    semantic["evidence_role"] = "new_fixed_holdout_2000_semantic_proxy"
    semantic_gate = _semantic_final_gate(semantic, config)
    saved_semantic = _json(evidence_dir / "holdout_semantic_summary.json")
    saved_gate = _json(evidence_dir / "holdout_semantic_gate.json")
    _json_values_equal(semantic, saved_semantic, "holdout_semantic_summary")
    _json_values_equal(semantic_gate, saved_gate, "holdout_semantic_gate")
    summaries = {role: _summarize_rows(rows) for role, rows in recomputed_rows.items()}
    numeric_pass = bool(summaries["holdout"]["all_rows_passed"])
    final_pass = numeric_pass and bool(semantic_gate["passed"])
    _require(decision.get("holdout_all_six_metrics_passed") is numeric_pass
             and decision.get("final_gate_passed") is final_pass
             and decision.get("state") == ("VALID_COMPLETED" if final_pass else "VALID_FAILED"),
             "saved final decision differs from the independently recomputed holdout and semantic gate")

    verification = {"task_id": TASK_ID, "status": "SAVED_EVIDENCE_VERIFIED",
                    "evidence_dir": str(evidence_dir), "transfer_receipt_required": require_transfer,
                    "actual_file_count": len(manifest["files"]), "actual_total_bytes": manifest["total_bytes"],
                    "selected_alpha": alpha, "training_updates": UPDATE_BUDGET,
                    "exposure_count_each_target": EXPOSURES,
                    "terminal_command_binding_max_abs_error_cpu": terminal_command_binding_max_abs_error,
                    "holdout_all_six_metrics_passed": numeric_pass,
                    "semantic_proxy_gate_passed": bool(semantic_gate["passed"]),
                    "final_state": decision["state"], "scientific_update_count": UPDATE_BUDGET}
    destination = (evidence_dir / "verify_saved_01" if report_dir is None
                   else report_dir.resolve())
    _require(not destination.exists() and destination.is_relative_to(RUNS_DIR.resolve()),
             "verification report path must be new and stay inside this study runs directory",
             FileExistsError)
    destination.mkdir(parents=True)
    _write_json(destination / "verification_summary.json", verification)
    _write_json(destination / "recomputed_final_decision.json", {
        "holdout_summary": summaries["holdout"], "semantic_gate": semantic_gate,
        "final_gate_passed": final_pass, "state": decision["state"]})
    return verification


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=(
        "prepare-prior", "qualify", "preflight", "execute", "resume", "measure",
        "verify-saved", "recovery-worker"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _cli().parse_args()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    _require(config_path == DEFAULT_CONFIG.resolve(), "fixed-frame run requires the registered config", ValueError)
    if args.mode == "prepare-prior":
        result = _prepare_prior(config)
    elif args.mode == "qualify":
        result = _qualify(config, resume_checkpoint=args.checkpoint)
    elif args.mode == "preflight":
        result = _preflight(config)
    elif args.mode == "execute":
        _require(args.checkpoint is None, "execute starts only from the fixed zero-update initialization", ValueError)
        result = _train(config)
    elif args.mode == "resume":
        _require(args.checkpoint is not None, "resume requires an explicit checkpoint", ValueError)
        result = _train(config, checkpoint_path=args.checkpoint)
    elif args.mode == "measure":
        result = _measure(config)
    elif args.mode == "verify-saved":
        _require(args.evidence_dir is not None,
                 "verify-saved requires an explicit attempt or Windows backup evidence directory", ValueError)
        result = _verify_saved(config, args.evidence_dir, args.report_dir)
    else:
        _require(args.checkpoint is not None and args.output is not None,
                 "recovery worker requires explicit checkpoint and output", ValueError)
        _recovery_worker(config, args.checkpoint.resolve(), args.output.resolve())
        result = {"status": "RECOVERY_STEP4_SAVED", "output": str(args.output.resolve())}
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


