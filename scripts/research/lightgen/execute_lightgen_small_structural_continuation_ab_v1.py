"""Run the fixed 5,000-update Gaussian structural-loss continuation comparison."""

from __future__ import annotations

import argparse
import copy
import html
import json
import math
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

import scripts.research.lightgen.execute_lightgen_matched_architecture_v1 as base
from ldm_project.v2.evidence.io import cpu_copy
from ldm_project.v2.application.research.lightgen.native_image_inputs import image_batch
from ldm_project.v2.application.research.lightgen.image_readout import (
    structural_loss_from_raw as _structural_loss_from_raw,
)
from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
)
from ldm_project.v2.evidence import atomic_torch_save, atomic_torch_save_new, atomic_write_json, atomic_write_new_json


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_SMALL_GAUSSIAN_STRUCTURAL_CONTINUATION_AB_V1"
AUTHORITY = "USER_APPROVED_SMALL_GAUSSIAN_STRUCTURAL_CONTINUATION_AB_20261020"
SOURCE_TASK_ID = "LIGHTGEN_SMALL_GAUSSIAN_SYSTEM_BASELINE_V1"
SOURCE_UPDATES = 30_000
CONTINUATION_UPDATES = 5_000
TOTAL_UPDATES = SOURCE_UPDATES + CONTINUATION_UPDATES
BATCH_SIZE = 10
SEED = 2026101501
CHECKPOINT_INTERVAL = 500
PROGRESS_INTERVAL = 100
RECOVERY_RTOL = 2.0e-6
RECOVERY_ATOL = 5.0e-8
WINDOW_TOLERANCE = 1.0e-4
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 2026102004
BRANCHES = ("control", "treatment")


def _require(condition: bool, message: str, error_type: type[Exception] = RuntimeError) -> None:
    if not condition:
        raise error_type(message)


def _read_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "continuation config must be a mapping")
    _require(value.get("schema") == "lightgen_small_gaussian_structural_continuation_ab.v1",
             "continuation config schema changed")
    _require(value.get("task_id") == TASK_ID and value.get("semantic_authority_id") == AUTHORITY,
             "continuation task authorization identity changed")
    training = value["training"]
    _require(training["seed"] == SEED and training["source_completed_updates"] == SOURCE_UPDATES
             and training["continuation_updates"] == CONTINUATION_UPDATES,
             "continuation seed or update budget changed")
    _require(training["batch_size"] == BATCH_SIZE and training["samples_per_class_per_batch"] == 1,
             "continuation batch population changed")
    _require(training["optimizer"] == "Adam" and training["learning_rate"] == 1.0e-3
             and training["betas"] == [0.9, 0.999] and training["epsilon"] == 1.0e-8
             and training["weight_decay"] == 0.0,
             "continuation optimizer changed")
    _require(training["branch_order"] == list(BRANCHES)
             and training["checkpoint_interval"] == CHECKPOINT_INTERVAL
             and training["progress_interval"] == PROGRESS_INTERVAL
             and training["checkpoints_at"] == [0, 2500, 5000],
             "continuation save schedule changed")
    _require(value["data"]["train_count"] == 55_000 and value["data"]["dev_count"] == 5_000
             and value["data"]["dev_per_class"] == 500 and value["data"]["official_test_reads"] == 0,
             "continuation data roles changed")
    small = value["optics"]["small"]
    _require(small["ols_basis"] == "gaussian_s12_v1" and small["phase_parameters"] == 40_000
             and small["production_propagation_grid"] == 2048
             and small["propagation_grid_candidates"] == [2048, 4096]
             and value["optics"]["window_complex_relative_error_max"] == WINDOW_TOLERANCE,
             "Gaussian model geometry or numerical tolerance changed")
    loss = value["loss"]
    _require(loss["structural_definition"] ==
             "mean_1_minus_ssim_after_global_gain_clamp_0_1_then_adaptive_pool_28"
             and loss["ssim_window"] == 7 and loss["ssim_sigma"] == 1.5
             and loss["data_range"] == 1.0 and loss["padding"] == 3
             and loss["lambda_multiplier"] == 1.0
             and loss["freeze_lambda_after_calibration"] is True
             and loss["freeze_source_camera_gain_and_bogt_weights"] is True,
             "structural loss semantics changed")
    _require(value["evaluation"]["paired_bootstrap_resamples"] == BOOTSTRAP_RESAMPLES
             and value["evaluation"]["paired_bootstrap_seed"] == BOOTSTRAP_SEED
             and value["evaluation"]["evaluator_role"] == "frozen_confirmation_only"
             and value["output"]["formal_result_budget_bytes"] == 536_870_912,
             "evaluation contract or result budget changed")
    return value


def _continued_schedule(labels: torch.Tensor, train_indices: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    by_class: list[torch.Tensor] = []
    for label in range(10):
        candidates = train_indices[labels[train_indices] == label]
        _require(candidates.numel() > 0, f"TRAIN class {label} is empty")
        by_class.append(candidates[torch.randperm(candidates.numel(), generator=generator)])
    rows = torch.empty((TOTAL_UPDATES, BATCH_SIZE), dtype=torch.long)
    cursors = [0] * 10
    for step in range(TOTAL_UPDATES):
        for label in range(10):
            if cursors[label] == by_class[label].numel():
                cursors[label] = 0
            rows[step, label] = by_class[label][cursors[label]]
            cursors[label] += 1
    _require(torch.equal(labels[rows[0]], torch.arange(10)), "continuation first batch is not class balanced")
    return rows


def _validate_source_metadata(
    checkpoint: dict[str, Any], config: dict[str, Any], source_config: dict[str, Any]
) -> None:
    _require(checkpoint.get("schema") == "lightgen_matched_architecture_checkpoint.v1",
             "source checkpoint schema changed")
    _require(checkpoint.get("task_id") == SOURCE_TASK_ID and checkpoint.get("arm") == "small"
             and checkpoint.get("seed") == SEED and checkpoint.get("completed_updates") == SOURCE_UPDATES,
             "source checkpoint task, arm, seed, or step is not the specified Gaussian endpoint")
    _require(checkpoint.get("config_identity") == source_config,
             "source checkpoint configuration differs from the explicit Gaussian baseline config")
    expected_identity = base._model_identity(source_config, "small")
    _require(expected_identity == config["source"]["model_identity"]
             and checkpoint.get("model_identity") == expected_identity
             and expected_identity.get("ols") == "gaussian_s12_v1",
             "source checkpoint model identity or OLS basis changed")
    _require(checkpoint.get("evaluator_role") == "frozen_confirmation_only",
             "source checkpoint evaluator role changed")
    optimizer = checkpoint.get("optimizer")
    _require(isinstance(optimizer, dict) and len(optimizer.get("param_groups", [])) == 1,
             "source checkpoint is missing its single Adam parameter group")
    group = optimizer["param_groups"][0]
    _require(group.get("lr") == 1.0e-3 and tuple(group.get("betas", ())) == (0.9, 0.999)
             and group.get("eps") == 1.0e-8 and group.get("weight_decay") == 0.0,
             "source Adam configuration differs from the authorized continuation")


def _load_source(
    source_path: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _require(source_path.is_file(), f"explicit source checkpoint is missing: {source_path}")
    source_config_path = (ROOT / config["source"]["baseline_config"]).resolve()
    source_config = base._read_config(source_config_path)
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, dict), "source checkpoint payload must be a mapping")
    _validate_source_metadata(checkpoint, config, source_config)
    source_run_dir = source_path.parent
    references_path = source_run_dir / "initial_references.pt"
    report_path = source_run_dir / "report.json"
    _require(references_path.is_file() and report_path.is_file(),
             "source baseline references or complete report are missing")
    references = torch.load(references_path, map_location="cpu", weights_only=False)
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(source_report.get("task_id") == SOURCE_TASK_ID
             and source_report.get("run_id") == checkpoint.get("run_id")
             and source_report.get("measurement_complete") is True
             and source_report.get("training", {}).get("updates") == SOURCE_UPDATES,
             "source report is incomplete or does not match the explicit endpoint")
    _require(len(source_report.get("reconstruction", [])) > 0
             and int(source_report["reconstruction"][-1].get("step", -1)) == SOURCE_UPDATES
             and source_report.get("generation", {}).get("population") == 10_000,
             "source final reconstruction or generation measurements are incomplete")
    for name in ("scale", "power_reference", "camera_gain"):
        base._assert_close(checkpoint[name], references[name], f"source_{name}")
    _require(checkpoint["weights"] == references["weights"], "source frozen BOGT coefficients changed")
    rows = checkpoint.get("schedule")
    _require(isinstance(rows, torch.Tensor) and tuple(rows.shape) == (SOURCE_UPDATES, BATCH_SIZE),
             "source checkpoint schedule is incomplete")
    return checkpoint, references, source_report


def _make_identity(config: dict[str, Any], source: dict[str, Any], run_id: str) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=run_id,
        trajectory_id=f"{TASK_ID}:seed:{SEED}:source_step:{SOURCE_UPDATES}",
        semantic_authority_id=AUTHORITY,
        data_role="MNIST_TRAIN_55000_DEV_5000_OFFICIAL_TEST_UNREAD",
        config_identity=config,
        model_identity=source["model_identity"],
        optimizer_identity={"name": "Adam", "lr": 1.0e-3, "betas": [0.9, 0.999],
                            "epsilon": 1.0e-8, "weight_decay": 0.0, "source_step": SOURCE_UPDATES},
        sampler_identity={"seed": SEED, "logical_batch_size": BATCH_SIZE,
                          "start_after_source_updates": SOURCE_UPDATES,
                          "continuation_updates": CONTINUATION_UPDATES},
        execution_source_semantics_id=config["execution_source_semantics_id"],
    )


def _new_branch(
    source: dict[str, Any], config: dict[str, Any], device: torch.device, grid: int
) -> tuple[torch.nn.Module, torch.optim.Adam]:
    model = base._model("small", grid, device, config)
    model.load_state_dict(source["model"], strict=True)
    groups = base._phase_groups(model)
    phase_ids = {id(parameter) for parameter in groups}
    _require(all(parameter.requires_grad == (id(parameter) in phase_ids) for parameter in model.parameters()),
             "source model trainable or frozen parameter scope changed")
    optimizer = torch.optim.Adam(
        groups, lr=1.0e-3, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0,
    )
    # Optimizer state_dict loading can retain the CPU ``step`` tensor by
    # reference when moments are cast onto the target device.  Each preflight
    # branch must therefore receive its own state tree so engineering updates
    # cannot advance the shared, frozen source checkpoint state.
    optimizer.load_state_dict(copy.deepcopy(source["optimizer"]))
    _require(base._optimizer_update_count(optimizer, model) == SOURCE_UPDATES,
             "source Adam state does not contain exactly 30,000 updates")
    return model, optimizer


def _target28(pixels: torch.Tensor, rows: torch.Tensor, device: torch.device) -> torch.Tensor:
    selected = pixels.index_select(0, rows.detach().cpu()).unsqueeze(1).to(device=device, dtype=torch.float32) / 255.0
    _require(tuple(selected.shape) == (BATCH_SIZE, 1, 28, 28), "original 28x28 target rows changed")
    return selected


def _loss_terms(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    rows: torch.Tensor,
    scale: torch.Tensor,
    power_reference: torch.Tensor,
    gain: torch.Tensor,
    branch: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None, dict[str, Any]]:
    target_native = image_batch(pixels, rows, "small", next(model.parameters()).device)
    parts, result = base._loss_components(model, target_native, scale, power_reference, gain)
    structure = None
    if branch == "treatment":
        structure = _structural_loss_from_raw(result["physical_w_per_pixel"], gain,
                                              _target28(pixels, rows, target_native.device))
    return parts, structure, result


def _total_loss(parts: dict[str, torch.Tensor], structure: torch.Tensor | None,
                weights: dict[str, float], structural_weight: float | None) -> torch.Tensor:
    total = sum(weights[name] * parts[name] for name in ("dist", "rec", "op"))
    if structural_weight is not None:
        _require(structure is not None and math.isfinite(structural_weight) and structural_weight > 0.0,
                 "treatment structural coefficient or term is invalid")
        total = total + structural_weight * structure
    return total


def _gradient_norm(gradients: tuple[torch.Tensor | None, ...], loss: torch.Tensor) -> float:
    squared = torch.zeros((), dtype=torch.float64, device=loss.device)
    reached = False
    for gradient in gradients:
        if gradient is None:
            continue
        _require(torch.isfinite(gradient).all().item(), "loss gradient contains a nonfinite value")
        squared += gradient.detach().double().square().sum()
        reached = True
    value = float(squared.sqrt().cpu())
    _require(reached and math.isfinite(value) and value > 0.0, "loss has a zero or disconnected phase gradient")
    return value


def _calibrate_structural_weight(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    rows: torch.Tensor,
    scale: torch.Tensor,
    power_reference: torch.Tensor,
    gain: torch.Tensor,
    weights: dict[str, float],
    config: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    parts, structure, _ = _loss_terms(model, pixels, rows, scale, power_reference, gain, "treatment")
    assert structure is not None
    groups = base._phase_groups(model)
    rec_gradients = torch.autograd.grad(weights["rec"] * parts["rec"], groups, retain_graph=True)
    structure_gradients = torch.autograd.grad(structure, groups)
    rec_norm = _gradient_norm(rec_gradients, parts["rec"])
    structure_norm = _gradient_norm(structure_gradients, structure)
    multiplier = float(config["loss"]["lambda_multiplier"])
    structural_weight = multiplier * rec_norm / structure_norm
    _require(math.isfinite(structural_weight) and structural_weight > 0.0,
             "frozen SSIM structural coefficient is invalid")
    return structural_weight, {
        "weighted_rec_gradient_norm": rec_norm,
        "unweighted_ssim_gradient_norm": structure_norm,
        "lambda_multiplier": multiplier,
        "lambda_ssim": structural_weight,
    }


def _direction_check(
    model: torch.nn.Module,
    objective: Callable[[], torch.Tensor],
    name: str,
    epsilon: float = 1.0e-2,
) -> dict[str, dict[str, float]]:
    groups = base._phase_groups(model)
    value = objective()
    gradients = torch.autograd.grad(value, groups)
    del value
    diagnostics: dict[str, dict[str, float]] = {}
    for group_name, parameter, gradient in zip(("encoder", "decoder"), groups, gradients, strict=True):
        gradient64 = gradient.detach().double()
        norm = torch.linalg.vector_norm(gradient64)
        _require(torch.isfinite(norm).item() and float(norm) > 0.0,
                 f"{name} {group_name} gradient is invalid")
        direction = (gradient64 / norm).to(parameter.dtype)
        analytic = float((gradient64 * direction.double()).sum())
        original = parameter.detach().clone()
        with torch.no_grad():
            plus_state = original + direction * epsilon
            minus_state = original - direction * epsilon
            plus_delta = plus_state.double() - original.double()
            minus_delta = original.double() - minus_state.double()
            actual_step = float(((plus_delta + minus_delta) * direction.double()).sum() * 0.5)
            parameter.copy_(plus_state)
        try:
            plus = float(objective().detach())
            with torch.no_grad():
                parameter.copy_(minus_state)
            minus = float(objective().detach())
        finally:
            with torch.no_grad():
                parameter.copy_(original)
        _require(math.isfinite(actual_step) and actual_step > 0.0,
                 f"{name} {group_name} perturbation is not representable")
        finite = (plus - minus) / (2.0 * actual_step)
        relative = _validate_direction_values(name, group_name, analytic, finite)
        diagnostics[group_name] = {
            "autograd": analytic,
            "central_difference": finite,
            "relative_error": relative,
            "epsilon_rad": epsilon,
            "representable_half_step_rad": actual_step,
        }
        del original, plus_state, minus_state, direction
    return diagnostics


def _validate_direction_values(name: str, group_name: str, analytic: float, finite: float) -> float:
    relative = abs(analytic - finite) / max(abs(analytic), abs(finite), 1.0e-7)
    _require(math.isfinite(analytic) and math.isfinite(finite) and analytic * finite > 0.0
             and relative <= 0.05,
             f"{name} {group_name} finite difference failed: autograd={analytic:.9g}, "
             f"central={finite:.9g}, relative={relative:.6g}")
    return relative


def _train_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Adam,
    pixels: torch.Tensor,
    rows: torch.Tensor,
    device: torch.device,
    scale: torch.Tensor,
    power_reference: torch.Tensor,
    gain: torch.Tensor,
    weights: dict[str, float],
    branch: str,
    structural_weight: float | None,
) -> dict[str, float | None]:
    model.train()
    model_device = next(model.parameters()).device
    expected_device = (torch.device(device.type, torch.cuda.current_device())
                       if device.type == "cuda" else device)
    _require(rows.shape == (BATCH_SIZE,) and model_device == expected_device,
             "logical batch or model device changed")
    optimizer.zero_grad(set_to_none=True)
    parts, structure, result = _loss_terms(model, pixels, rows, scale, power_reference, gain, branch)
    total = _total_loss(parts, structure, weights, structural_weight)
    _require(torch.isfinite(total).item(), "continuation objective is nonfinite")
    total.backward()
    parameters = base._phase_groups(model)
    before = [parameter.detach().clone() for parameter in parameters]
    norms: list[float] = []
    for group_name, parameter in zip(("encoder", "decoder"), parameters, strict=True):
        gradient = parameter.grad
        _require(gradient is not None and torch.isfinite(gradient).all().item()
                 and bool((gradient.abs().sum() > 0)),
                 f"{branch} {group_name} phase gradient is invalid or zero")
        norms.append(float(torch.linalg.vector_norm(gradient.detach().double()).cpu()))
    optimizer.step()
    for group_name, old, parameter in zip(("encoder", "decoder"), before, parameters, strict=True):
        _require(not torch.equal(old, parameter), f"{branch} {group_name} phase group did not update")
    return {
        "dist": float(parts["dist"].detach()),
        "rec": float(parts["rec"].detach()),
        "op": float(parts["op"].detach()),
        "ssim_loss": None if structure is None else float(structure.detach()),
        "total": float(total.detach()),
        "encoder_gradient_norm": norms[0],
        "decoder_gradient_norm": norms[1],
        "mean_plane_power_w": float(result["plane_powers_w"].detach().double().mean()),
        "adam_update_count": base._optimizer_update_count(optimizer, model),
    }


def _worker(staging_dir: Path, source_path: Path, config_path: Path) -> None:
    config = _read_config(config_path)
    input_path = staging_dir / "preflight_step_000003.pt"
    _require(input_path.is_file(), "preflight worker input is missing")
    payload = torch.load(input_path, map_location="cpu", weights_only=False)
    source, _, _ = _load_source(source_path, config)
    _require(payload.get("task_id") == TASK_ID and payload.get("source_run_id") == source["run_id"]
             and payload.get("config_identity") == config,
             "preflight worker identity differs from the source task")
    branch = payload["branch"]
    _require(branch in BRANCHES and payload.get("completed_segment_updates") == 3,
             "preflight worker branch or saved step is invalid")
    device = torch.device("cuda")
    _require(torch.cuda.is_available(), "independent recovery worker requires CUDA")
    model = base._model("small", int(payload["grid"]), device, config)
    model.load_state_dict(payload["model"], strict=True)
    optimizer = torch.optim.Adam(base._phase_groups(model), lr=1.0e-3, betas=(0.9, 0.999),
                                 eps=1.0e-8, weight_decay=0.0)
    optimizer.load_state_dict(payload["optimizer"])
    torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])
    torch.random.set_rng_state(payload["cpu_rng"])
    pixels, _, _, _ = base._load_data(ROOT / config["data"]["training_file"])
    loss = _train_update(model, optimizer, pixels, payload["rows"][3], device,
                         payload["scale"].to(device), payload["power_reference"].to(device),
                         payload["camera_gain"].to(device), payload["weights"], branch,
                         payload["structural_weight"])
    atomic_torch_save_new(staging_dir / "preflight_fourth_update_independent_process.pt", {
        "task_id": TASK_ID,
        "source_run_id": source["run_id"],
        "branch": branch,
        "completed_segment_updates": 4,
        "model": base._state_snapshot(model),
        "optimizer": cpu_copy(optimizer.state_dict()),
        "loss": loss,
    })
    print(json.dumps({"status": "PASSED", "branch": branch}), flush=True)


def _preflight_branch(
    branch: str,
    source: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
    source_path: Path,
    pixels: torch.Tensor,
    schedule: torch.Tensor,
    device: torch.device,
    scale: torch.Tensor,
    power_reference: torch.Tensor,
    gain: torch.Tensor,
    weights: dict[str, float],
    structural_weight: float | None,
    staging_dir: Path,
    grid: int,
) -> dict[str, Any]:
    model, optimizer = _new_branch(source, config, device, grid)
    torch.cuda.set_rng_state_all([state.cpu() for state in source["cuda_rng"]])
    torch.random.set_rng_state(source["cpu_rng"])
    model_initial = base._state_snapshot(model)
    optimizer_initial = copy.deepcopy(optimizer.state_dict())
    cuda_initial = torch.cuda.get_rng_state_all()
    cpu_initial = torch.random.get_rng_state()
    rows = schedule[SOURCE_UPDATES : SOURCE_UPDATES + 4]
    trial_losses = [
        _train_update(model, optimizer, pixels, rows[index], device, scale, power_reference,
                      gain, weights, branch, structural_weight)
        for index in range(3)
    ]
    recovery_input = {
        "task_id": TASK_ID,
        "source_run_id": source["run_id"],
        "config_identity": config,
        "branch": branch,
        "completed_segment_updates": 3,
        "grid": grid,
        "model": base._state_snapshot(model),
        "optimizer": cpu_copy(optimizer.state_dict()),
        "rows": rows.detach().cpu(),
        "scale": scale.detach().cpu(),
        "power_reference": power_reference.detach().cpu(),
        "camera_gain": gain.detach().cpu(),
        "weights": weights,
        "structural_weight": structural_weight,
        "cuda_rng": [state.cpu() for state in torch.cuda.get_rng_state_all()],
        "cpu_rng": torch.random.get_rng_state().cpu(),
    }
    atomic_torch_save_new(staging_dir / "preflight_step_000003.pt", recovery_input)
    direct_loss = _train_update(model, optimizer, pixels, rows[3], device, scale,
                                power_reference, gain, weights, branch, structural_weight)
    direct_model = base._state_snapshot(model)
    direct_optimizer = cpu_copy(optimizer.state_dict())
    worker = subprocess.run(
        [sys.executable, "-m", "scripts.research.lightgen.execute_lightgen_small_structural_continuation_ab_v1",
         "--preflight-worker-dir", str(staging_dir), "--source-checkpoint", str(source_path),
         "--config", str(config_path), "--run-dir", str(staging_dir)],
        capture_output=True, text=True, check=False, cwd=ROOT,
    )
    _require(worker.returncode == 0,
             "independent-process recovery worker failed: " + (worker.stderr[-3000:] or worker.stdout[-3000:]))
    recovered = torch.load(staging_dir / "preflight_fourth_update_independent_process.pt",
                           map_location="cpu", weights_only=False)
    _require(recovered.get("task_id") == TASK_ID and recovered.get("branch") == branch
             and recovered.get("source_run_id") == source["run_id"]
             and recovered.get("completed_segment_updates") == 4,
             "independent recovery output identity changed")
    model_error = base._assert_close(direct_model, recovered["model"], f"{branch}_recovery_model",
                                     )
    optimizer_error = base._assert_close(direct_optimizer, recovered["optimizer"],
                                         f"{branch}_recovery_optimizer")
    base._assert_close(direct_loss, recovered["loss"], f"{branch}_recovery_loss",
                       )
    model.load_state_dict(model_initial, strict=True)
    optimizer.load_state_dict(optimizer_initial)
    torch.cuda.set_rng_state_all(cuda_initial)
    torch.random.set_rng_state(cpu_initial)
    _require(base._assert_close(model_initial, base._state_snapshot(model), f"{branch}_preflight_rollback") == 0.0,
             "preflight model changes were not rolled back")
    base._assert_close(optimizer_initial, optimizer.state_dict(), f"{branch}_optimizer_rollback")
    base._assert_close(cuda_initial, torch.cuda.get_rng_state_all(), f"{branch}_cuda_rng_rollback")
    base._assert_close(cpu_initial, torch.random.get_rng_state(), f"{branch}_cpu_rng_rollback")
    return {
        "status": "PASSED",
        "trial_scientific_updates": 3,
        "engineering_adam_calls": 5,
        "trial_losses": trial_losses,
        "direct_fourth_update_loss": direct_loss,
        "independent_process_reload_and_next_update": True,
        "max_model_recovery_error": model_error,
        "max_optimizer_recovery_error": optimizer_error,
        "preflight_state_rolled_back": True,
    }


def _make_receipt(control: ResearchRunControl, preflight: dict[str, Any], device: torch.device) -> PreflightReceipt:
    window = preflight["source_window"]
    return control.make_ready_receipt(
        correctness={
            "passed": True,
            "status": "PASSED",
            "checks": ["source_checkpoint_identity_and_adam_step",
                       "fixed_30000_to_35000_class_balanced_schedule",
                       "source_gain_and_bogt_reference_values_frozen",
                       "2048_to_4096_seven_plane_window_check",
                       "independent_numpy_ols_and_camera_readout",
                       "structural_production_view_numpy_reference",
                       "control_and_treatment_phase_gradient_finite_difference",
                       "three_real_updates_and_independent_process_recovery_per_branch",
                       "preflight_model_optimizer_rng_rollback",
                       "evaluator_frozen_and_excluded_from_loss"],
            "source_window_max_relative_error": window["complex_relative_error_max"],
            "independent_process_recovery_passed": True,
        },
        resource={"passed": True, "status": "PASSED", "device": torch.cuda.get_device_name(device),
                  "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device)},
        implementation={"passed": True, "status": "PASSED", "engine": "cuda_gaussian_s12_grid_2048",
                        "selected_engine": "cuda_gaussian_s12_grid_2048"},
        selected_engine="cuda_gaussian_s12_grid_2048",
        resource_envelope={"device": torch.cuda.get_device_name(device),
                           "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device)},
    )


def _load_ready_context(
    run_dir: Path, config: dict[str, Any], source: dict[str, Any]
) -> tuple[ResearchRunControl, ResearchRunIdentity, PreflightReceipt]:
    identity_record = json.loads((run_dir / "run_identity.json").read_text(encoding="utf-8"))
    identity = ResearchRunIdentity.from_mapping(identity_record["identity"])
    authorization_id = f"{TASK_ID}:seed:{SEED}:preflight_and_execution"
    _require(identity.task_id == TASK_ID and identity.run_id == identity_record.get("run_id")
             and identity.config_identity == config and identity.model_identity == source["model_identity"]
             and identity_record.get("authorization_id") == authorization_id,
             "saved run identity does not match the explicit continuation inputs")
    receipt = PreflightReceipt.from_mapping(json.loads((run_dir / "preflight_ready.json").read_text(encoding="utf-8")))
    control = ResearchRunControl(identity, authorization_id=authorization_id)
    control.mark_preflight_ready(receipt)
    return control, identity, receipt


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Adam,
    branch: str,
    segment_updates: int,
    run_id: str,
    source: dict[str, Any],
    config: dict[str, Any],
    schedule: torch.Tensor,
    scale: torch.Tensor,
    power_reference: torch.Tensor,
    gain: torch.Tensor,
    weights: dict[str, float],
    structural_weight: float | None,
    history: list[dict[str, Any]],
    elapsed: float,
) -> dict[str, Any]:
    return {
        "schema": "lightgen_small_gaussian_structural_continuation_checkpoint.v1",
        "task_id": TASK_ID,
        "run_id": run_id,
        "source_run_id": source["run_id"],
        "source_checkpoint_step": SOURCE_UPDATES,
        "branch": branch,
        "seed": SEED,
        "model_identity": source["model_identity"],
        "config_identity": config,
        "propagation_grid_size": 2048,
        "completed_updates": segment_updates,
        "completed_global_step": SOURCE_UPDATES + segment_updates,
        "model": base._state_snapshot(model),
        "optimizer": cpu_copy(optimizer.state_dict()),
        "schedule": schedule,
        "schedule_start_global_step": SOURCE_UPDATES,
        "scale": scale.detach().cpu(),
        "power_reference": power_reference.detach().cpu(),
        "camera_gain": gain.detach().cpu(),
        "weights": weights,
        "structural_weight": structural_weight,
        "history": history,
        "cuda_rng": [state.cpu() for state in torch.cuda.get_rng_state_all()],
        "cpu_rng": torch.random.get_rng_state().cpu(),
        "evaluator_role": "frozen_confirmation_only",
        "elapsed_seconds_continuation": elapsed,
    }


def _validate_continuation_checkpoint(
    checkpoint: dict[str, Any], branch: str, run_id: str, source: dict[str, Any],
    config: dict[str, Any], schedule: torch.Tensor, references: dict[str, Any],
) -> int:
    _require(checkpoint.get("schema") == "lightgen_small_gaussian_structural_continuation_checkpoint.v1"
             and checkpoint.get("task_id") == TASK_ID and checkpoint.get("run_id") == run_id
             and checkpoint.get("source_run_id") == source["run_id"]
             and checkpoint.get("source_checkpoint_step") == SOURCE_UPDATES
             and checkpoint.get("branch") == branch and checkpoint.get("seed") == SEED,
             "continuation checkpoint task or branch identity changed")
    _require(checkpoint.get("model_identity") == source["model_identity"]
             and checkpoint.get("config_identity") == config
             and checkpoint.get("propagation_grid_size") == 2048,
             "continuation checkpoint model or config identity changed")
    step = checkpoint.get("completed_updates")
    _require(isinstance(step, int) and 0 <= step <= CONTINUATION_UPDATES
             and checkpoint.get("completed_global_step") == SOURCE_UPDATES + step,
             "continuation checkpoint step is invalid")
    _require(torch.equal(checkpoint.get("schedule"), schedule)
             and checkpoint.get("schedule_start_global_step") == SOURCE_UPDATES,
             "continuation checkpoint sample rows or schedule cursor changed")
    for name, field in (("scale", "scale"), ("power_reference", "power_reference"),
                        ("camera_gain", "camera_gain"), ("weights", "weights")):
        base._assert_close(checkpoint[field], references[name], f"resume_{branch}_{name}")
    expected_lambda = references.get("structural_weight") if branch == "treatment" else None
    _require(checkpoint.get("structural_weight") == expected_lambda,
             "continuation checkpoint frozen SSIM coefficient changed")
    return step


def _save_branch_checkpoint(branch_dir: Path, payload: dict[str, Any], segment_updates: int) -> None:
    latest_path = branch_dir / "checkpoint_latest.pt"
    previous_path = branch_dir / "checkpoint_previous.pt"
    if latest_path.is_file():
        old = torch.load(latest_path, map_location="cpu", weights_only=False)
        atomic_torch_save(previous_path, old)
    atomic_torch_save(latest_path, payload)
    reloaded = torch.load(latest_path, map_location="cpu", weights_only=False)
    _require(reloaded.get("run_id") == payload["run_id"]
             and reloaded.get("branch") == payload["branch"]
             and reloaded.get("completed_updates") == segment_updates,
             "saved continuation checkpoint did not pass file read-back")
    if segment_updates in (0, 2500, 5000):
        endpoint_path = branch_dir / f"checkpoint_step_{segment_updates:06d}.pt"
        atomic_torch_save(endpoint_path, payload)
        endpoint_reloaded = torch.load(endpoint_path, map_location="cpu", weights_only=False)
        _require(endpoint_reloaded.get("completed_updates") == segment_updates
                 and endpoint_reloaded.get("branch") == payload["branch"],
                 "saved registered continuation checkpoint failed read-back")


def _restore_branch(
    checkpoint: dict[str, Any], config: dict[str, Any], device: torch.device
) -> tuple[torch.nn.Module, torch.optim.Adam]:
    model = base._model("small", int(checkpoint["propagation_grid_size"]), device, config)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer = torch.optim.Adam(base._phase_groups(model), lr=1.0e-3, betas=(0.9, 0.999),
                                 eps=1.0e-8, weight_decay=0.0)
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
    torch.random.set_rng_state(checkpoint["cpu_rng"])
    _require(base._optimizer_update_count(optimizer, model) == checkpoint["completed_global_step"],
             "continuation Adam state does not match the global checkpoint step")
    return model, optimizer


def _stratified_paired_bootstrap(
    control: list[dict[str, Any]], treatment: list[dict[str, Any]], metric: str
) -> dict[str, Any]:
    control_by_row = {int(row["row"]): row for row in control}
    treatment_by_row = {int(row["row"]): row for row in treatment}
    _require(control_by_row.keys() == treatment_by_row.keys() and len(control_by_row) == 5_000,
             "A/B reconstruction rows do not form a complete paired DEV population")
    by_class: list[np.ndarray] = []
    for label in range(10):
        rows = sorted(row for row, item in control_by_row.items() if int(item["label"]) == label)
        _require(len(rows) == 500 and all(int(treatment_by_row[row]["label"]) == label for row in rows),
                 f"paired DEV class {label} is incomplete or mislabelled")
        by_class.append(np.asarray([float(treatment_by_row[row][metric]) - float(control_by_row[row][metric])
                                    for row in rows], dtype=np.float64))
    observed = float(np.mean([values.mean() for values in by_class]))
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        samples[index] = np.mean([
            values[generator.integers(0, values.size, size=values.size)].mean() for values in by_class
        ])
    return {
        "estimand": f"equal-class paired mean treatment-minus-control {metric}",
        "observed_difference": observed,
        "ci_95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "strata": "10 classes, 500 paired DEV rows sampled with replacement within class",
    }


def _evaluate_branch(
    branch: str,
    run_id: str,
    branch_dir: Path,
    model: torch.nn.Module,
    evaluator: torch.nn.Module,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    dev_indices: torch.Tensor,
    device: torch.device,
    gain: torch.Tensor,
    scale: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, Any]:
    evaluator_state = base._state_snapshot(evaluator)
    model_state = base._state_snapshot(model)
    reconstruction, examples = base._evaluate_reconstruction(
        model, evaluator, pixels, labels, dev_indices, "small", device, gain, collect_per_image=True,
    )
    reconstruction["step"] = TOTAL_UPDATES
    _require(len(reconstruction.get("per_image", [])) == 5_000,
             f"{branch} endpoint DEV reconstruction population is incomplete")
    comparisons, comparison_references = base._evaluate_reconstruction_comparisons(
        model, evaluator, pixels, labels, train_indices, dev_indices, device, gain,
        reconstruction["per_image"],
    )
    atomic_torch_save(branch_dir / "dev_examples_step_035000.pt", examples)
    base._write_reconstruction_csv(
        branch_dir / "reconstruction_metrics_dev.csv", reconstruction["per_image"],
        comparisons["global_mean_image_metrics"]["per_image"],
        comparisons["same_class_mismatched_input_metrics"]["per_image"],
    )
    base._write_contact_sheet(branch_dir / "dev_preview.png", examples["images"], examples["labels"], 2)
    base._write_contact_sheet(branch_dir / "dev_target_preview.png", examples["targets"], examples["labels"], 2)
    prior = base._fit_gaussians(model, pixels, labels, train_indices, "small", scale, device)
    samples, requested, proposal_ids, sampling = base._sample_gaussians(
        prior, scale.detach().cpu(),
        max_proposals_per_class=int(config["evaluation"]["max_proposals_per_class"]),
    )
    generation, generated_examples = base._evaluate_generation(
        model, evaluator, samples, requested, proposal_ids, labels, pixels,
        train_indices, dev_indices, "small", device, gain,
    )
    generation["proposal_rejection"] = sampling
    generation["status"] = "COMPLETE"
    generated_examples.update({"task_id": TASK_ID, "run_id": run_id, "branch": branch,
                               "global_step": TOTAL_UPDATES})
    atomic_torch_save(branch_dir / "fitted_gaussian_prior.pt", prior)
    atomic_torch_save(branch_dir / "generated_cohort.pt", generated_examples)
    base._write_contact_sheet(branch_dir / "generation_preview.png", generated_examples["images"],
                              generated_examples["requested_labels"], 10)
    quality = base._quality_decision(config, reconstruction, generation)
    # Recheck the actual saved generation population and the two frozen objects.
    _require(generated_examples["images"].shape == (10_000, 1, 28, 28)
             and torch.equal(torch.bincount(generated_examples["requested_labels"], minlength=10),
                             torch.full((10,), 1000)),
             f"{branch} endpoint generation cohort is incomplete")
    _require(base._assert_close(model_state, base._state_snapshot(model), f"{branch}_model_after_evaluation") == 0.0,
             f"{branch} model changed during endpoint evaluation")
    _require(base._assert_close(evaluator_state, base._state_snapshot(evaluator),
                                f"{branch}_evaluator_after_evaluation") == 0.0,
             f"{branch} frozen evaluator changed during endpoint evaluation")
    return {
        "branch": branch,
        "global_step": TOTAL_UPDATES,
        "reconstruction": reconstruction,
        "reconstruction_comparisons": comparisons,
        "generation": generation,
        "quality_gate": quality,
        "status": quality["status"],
        "measurement_complete": True,
    }


def _write_comparison_html(path: Path, report: dict[str, Any]) -> None:
    def metric_row(label: str, record: dict[str, Any]) -> str:
        recon = record["reconstruction"]
        gen = record["generation"]
        return (f"<tr><td>{html.escape(label)}</td><td>{recon['psnr_db_mean']:.4f}</td>"
                f"<td>{recon['ssim_mean']:.5f}</td><td>{gen['macro_recall']:.5f}</td>"
                f"<td>{min(gen['recall_by_class']):.5f}</td>"
                f"<td>{max(gen['proxy_clipping_fraction_by_class']):.5f}</td></tr>")

    rows = []
    for key, label in (("source", "30,000步源端点"), ("control", "A：原BOGT续训"),
                       ("treatment", "B：BOGT＋SSIM续训")):
        if key in report["endpoints"]:
            rows.append(metric_row(label, report["endpoints"][key]))
    paired = report.get("paired_comparison", {})
    lines = [
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Gaussian小架构结构损失续训比较</title>",
        "<style>body{font:16px system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#17202a}"
        "table{border-collapse:collapse}td,th{padding:.45rem .65rem;border:1px solid #ccd1d1;text-align:left}"
        "img{max-width:100%;height:auto;border:1px solid #ccd1d1}.status{font-weight:bold}</style>",
        f"<h1>Gaussian小架构结构损失续训</h1><p class='status'>状态：{html.escape(report['status'])}</p>",
        "<p>三组图像使用相同灰度范围[0,1]；每类固定显示前两张DEV及前十张生成样本。</p>",
        "<table><tr><th>端点</th><th>DEV PSNR dB</th><th>DEV SSIM</th><th>生成E_D宏召回</th>"
        "<th>最低类召回</th><th>最高类代理截断</th></tr>" + "".join(rows) + "</table>",
        "<h2>A/B配对差值（B−A）</h2><pre>" + html.escape(json.dumps(paired, ensure_ascii=False, indent=2)) + "</pre>",
    ]
    for label, folder in (("30,000步源端点", "source"), ("A：原BOGT续训", "control"),
                          ("B：BOGT＋SSIM续训", "treatment")):
        if folder == "source":
            dev_file, target_file, gen_file = ("source_dev_preview.png", "source_dev_target_preview.png",
                                               "source_generation_preview.png")
        else:
            dev_file, target_file, gen_file = (f"{folder}/dev_preview.png", f"{folder}/dev_target_preview.png",
                                               f"{folder}/generation_preview.png")
        lines.extend([f"<h2>{label}</h2><h3>DEV目标</h3><img src='{target_file}'>",
                      f"<h3>DEV重构</h3><img src='{dev_file}'>",
                      f"<h3>条件生成</h3><img src='{gen_file}'>"])
    lines.append("</html>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_preflight(
    source: dict[str, Any], references: dict[str, Any], config: dict[str, Any],
    config_path: Path, source_path: Path, pixels: torch.Tensor, labels: torch.Tensor,
    train_indices: torch.Tensor, schedule: torch.Tensor, device: torch.device,
    evaluator: torch.nn.Module, staging_parent: Path,
) -> tuple[dict[str, Any], float]:
    grid = int(config["optics"]["small"]["production_propagation_grid"])
    scale = source["scale"].to(device)
    power_reference = source["power_reference"].to(device)
    gain = source["camera_gain"].to(device)
    weights = source["weights"]
    model, _ = _new_branch(source, config, device, grid)
    torch.cuda.set_rng_state_all([state.cpu() for state in source["cuda_rng"]])
    torch.random.set_rng_state(source["cpu_rng"])
    started = time.perf_counter()
    selected_grid, source_window = base._window_check(
        "small", pixels, labels, schedule, config, device, model=model,
    )
    _require(selected_grid == grid and source_window["complex_relative_error_max"] <= WINDOW_TOLERANCE,
             "source endpoint failed its fixed 2048/4096 seven-plane window check")
    initial_batch = image_batch(pixels, schedule[0], "small", device)
    independent_readout = base._small_producer_consumer_check(model, initial_batch[:1])
    _require(independent_readout.get("status") == "PASSED",
             "source OLS coefficient or full-camera W/pixel readout check failed")
    treatment_model, _ = _new_branch(source, config, device, grid)
    torch.cuda.set_rng_state_all([state.cpu() for state in source["cuda_rng"]])
    torch.random.set_rng_state(source["cpu_rng"])
    structural_weight, calibration = _calibrate_structural_weight(
        treatment_model, pixels, schedule[SOURCE_UPDATES], scale, power_reference, gain,
        weights, config,
    )
    source_control, _ = _new_branch(source, config, device, grid)
    torch.cuda.set_rng_state_all([state.cpu() for state in source["cuda_rng"]])
    torch.random.set_rng_state(source["cpu_rng"])
    control_rows = schedule[SOURCE_UPDATES]
    source_objectives: dict[str, Callable[[], torch.Tensor]] = {
        "control_total": lambda: _total_loss(
            _loss_terms(source_control, pixels, control_rows, scale, power_reference, gain, "control")[0],
            None, weights, None),
    }
    treatment_rows = schedule[SOURCE_UPDATES]
    treatment_objectives: dict[str, Callable[[], torch.Tensor]] = {
        "ssim_term": lambda: _loss_terms(treatment_model, pixels, treatment_rows, scale,
                                          power_reference, gain, "treatment")[1],
        "treatment_total": lambda: _total_loss(
            *_loss_terms(treatment_model, pixels, treatment_rows, scale, power_reference, gain, "treatment")[:2],
            weights, structural_weight),
    }
    finite_differences = {
        "control_total": _direction_check(source_control, source_objectives["control_total"], "control_total"),
        "treatment_ssim_term": _direction_check(treatment_model, treatment_objectives["ssim_term"], "treatment_ssim"),
        "treatment_total": _direction_check(treatment_model, treatment_objectives["treatment_total"], "treatment_total"),
    }
    evaluator_state = base._state_snapshot(evaluator)
    branch_preflights: dict[str, Any] = {}
    for branch in BRANCHES:
        staging = staging_parent / f"{branch}_{uuid.uuid4().hex[:8]}"
        staging.mkdir(parents=False, exist_ok=False)
        branch_preflights[branch] = _preflight_branch(
            branch, source, config, config_path, source_path, pixels, schedule, device,
            scale, power_reference, gain, weights,
            structural_weight if branch == "treatment" else None,
            staging, grid,
        )
        _require(base._assert_close(evaluator_state, base._state_snapshot(evaluator),
                                    f"{branch}_evaluator_after_preflight") == 0.0,
                 "frozen E_D changed during preflight")
    torch.cuda.set_rng_state_all([state.cpu() for state in source["cuda_rng"]])
    torch.random.set_rng_state(source["cpu_rng"])
    return {
        "status": "READY",
        "source_window": source_window,
        "independent_readout": independent_readout,
        "structural_calibration": calibration,
        "frozen_loss_weights": weights,
        "source_adam_updates": SOURCE_UPDATES,
        "continuation_schedule_rows": schedule[SOURCE_UPDATES:].tolist(),
        "finite_difference": finite_differences,
        "branch_preflight": branch_preflights,
        "evaluator_frozen_and_excluded_from_loss": True,
        "official_test_reads": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
        "scientific_updates_before_execution": 0,
    }, structural_weight


def _run_branch(
    branch: str, run_dir: Path, run_id: str, source: dict[str, Any], config: dict[str, Any],
    pixels: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, dev_indices: torch.Tensor,
    schedule: torch.Tensor, references: dict[str, Any], evaluator: torch.nn.Module,
    device: torch.device, resume_checkpoint: Path | None, elapsed_before: float,
) -> tuple[dict[str, Any], float]:
    branch_dir = run_dir / branch
    references_for_branch = {
        "scale": source["scale"], "power_reference": source["power_reference"],
        "camera_gain": source["camera_gain"], "weights": source["weights"],
        "structural_weight": references.get("structural_weight") if branch == "treatment" else None,
    }
    _require(branch_dir.is_dir(), f"branch output directory is missing: {branch_dir}")
    completed_summary = branch_dir / "endpoint_summary.json"
    if completed_summary.is_file():
        summary = json.loads(completed_summary.read_text(encoding="utf-8"))
        _require(summary.get("branch") == branch and summary.get("measurement_complete") is True,
                 f"saved {branch} endpoint summary is incomplete")
        endpoint = torch.load(branch_dir / "checkpoint_step_005000.pt", map_location="cpu", weights_only=False)
        _validate_continuation_checkpoint(endpoint, branch, run_id, source, config,
                                          schedule[SOURCE_UPDATES:], references_for_branch)
        return summary, elapsed_before

    grid = int(config["optics"]["small"]["production_propagation_grid"])
    if resume_checkpoint is not None and resume_checkpoint.parent.name == branch:
        checkpoint = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        start = _validate_continuation_checkpoint(checkpoint, branch, run_id, source, config,
                                                  schedule[SOURCE_UPDATES:], references_for_branch)
        model, optimizer = _restore_branch(checkpoint, config, device)
        history = checkpoint.get("history")
        _require(isinstance(history, list), "resume checkpoint training history is missing")
        saved_elapsed = checkpoint.get("elapsed_seconds_continuation")
        _require(isinstance(saved_elapsed, (float, int)) and math.isfinite(float(saved_elapsed))
                 and float(saved_elapsed) >= 0.0, "resume elapsed time is invalid")
        elapsed_before = float(saved_elapsed)
    else:
        model, optimizer = _new_branch(source, config, device, grid)
        torch.cuda.set_rng_state_all([state.cpu() for state in source["cuda_rng"]])
        torch.random.set_rng_state(source["cpu_rng"])
        history = []
        start = 0
        initial = _checkpoint_payload(
            model, optimizer, branch, 0, run_id, source, config, schedule[SOURCE_UPDATES:],
            source["scale"], source["power_reference"], source["camera_gain"], source["weights"],
            references_for_branch["structural_weight"], history, elapsed_before,
        )
        _save_branch_checkpoint(branch_dir, initial, 0)

    density_start_step = SOURCE_UPDATES + start
    base._record_gaussian_plane_density_probe(
        model, pixels, labels, dev_indices, run_dir, branch=branch,
        global_step=density_start_step, grid_size=grid,
    )
    evaluator_state = base._state_snapshot(evaluator)
    model_static_state = {key: value for key, value in base._state_snapshot(model).items()
                          if key not in {"runtime.encoder.phase", "runtime.generator.phase"}}
    started = time.perf_counter()
    for cursor in range(start, CONTINUATION_UPDATES):
        segment_step = cursor + 1
        global_step = SOURCE_UPDATES + segment_step
        loss = _train_update(
            model, optimizer, pixels, schedule[SOURCE_UPDATES + cursor], device,
            source["scale"].to(device), source["power_reference"].to(device), source["camera_gain"].to(device),
            source["weights"], branch,
            references_for_branch["structural_weight"],
        )
        if segment_step % PROGRESS_INTERVAL == 0:
            history.append({"segment_step": segment_step, "global_step": global_step, **loss})
            atomic_write_json(branch_dir / "training_history_live.json", history)
            atomic_write_json(run_dir / "progress_summary.json", {
                "branch": branch,
                "completed_updates_in_continuation": segment_step,
                "total_continuation_updates": CONTINUATION_UPDATES,
                "global_step": global_step,
                "latest_loss": loss,
                "elapsed_seconds_this_process": time.perf_counter() - started,
                "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
            })
        if segment_step % CHECKPOINT_INTERVAL == 0:
            payload = _checkpoint_payload(
                model, optimizer, branch, segment_step, run_id, source, config,
                schedule[SOURCE_UPDATES:], source["scale"], source["power_reference"],
                source["camera_gain"], source["weights"], references_for_branch["structural_weight"],
                history, elapsed_before + time.perf_counter() - started,
            )
            _save_branch_checkpoint(branch_dir, payload, segment_step)
            base._record_gaussian_plane_density_probe(
                model, pixels, labels, dev_indices, run_dir, branch=branch,
                global_step=global_step, grid_size=grid,
            )
        if segment_step in (2500, CONTINUATION_UPDATES):
            payload = _checkpoint_payload(
                model, optimizer, branch, segment_step, run_id, source, config,
                schedule[SOURCE_UPDATES:], source["scale"], source["power_reference"],
                source["camera_gain"], source["weights"], references_for_branch["structural_weight"],
                history, elapsed_before + time.perf_counter() - started,
            )
            _save_branch_checkpoint(branch_dir, payload, segment_step)
    _require(base._optimizer_update_count(optimizer, model) == TOTAL_UPDATES,
             f"{branch} Adam did not finish at global step 35,000")
    _require(base._assert_close(evaluator_state, base._state_snapshot(evaluator),
                                f"{branch}_evaluator_after_training") == 0.0,
             f"{branch} frozen E_D changed during training")
    _require(base._assert_close(model_static_state,
                                {key: value for key, value in base._state_snapshot(model).items()
                                 if key not in {"runtime.encoder.phase", "runtime.generator.phase"}},
                                f"{branch}_frozen_model_state_after_training") == 0.0,
             f"{branch} frozen optical state changed during training")
    endpoint_model_state = base._state_snapshot(model)
    endpoint_batch = image_batch(pixels, schedule[0], "small", device)[:1]
    endpoint_readout = base._small_producer_consumer_check(model, endpoint_batch)
    endpoint_grid, endpoint_window = base._window_check(
        "small", pixels, labels, schedule, config, device, model=model,
    )
    _require(endpoint_grid == grid and endpoint_window["complex_relative_error_max"] <= WINDOW_TOLERANCE,
             f"{branch} endpoint failed its fixed production-grid complex-window check")
    _require(endpoint_window["ols_coefficient_relative_error_by_grid"][grid] <= WINDOW_TOLERANCE
             and endpoint_window["camera_w_per_pixel_relative_error_by_grid"][grid] <= WINDOW_TOLERANCE,
             f"{branch} endpoint OLS coefficients or camera readout exceed the fixed tolerance")
    endpoint = _checkpoint_payload(
        model, optimizer, branch, CONTINUATION_UPDATES, run_id, source, config,
        schedule[SOURCE_UPDATES:], source["scale"], source["power_reference"],
        source["camera_gain"], source["weights"], references_for_branch["structural_weight"],
        history, elapsed_before + time.perf_counter() - started,
    )
    _save_branch_checkpoint(branch_dir, endpoint, CONTINUATION_UPDATES)
    atomic_torch_save(branch_dir / "terminal_model.pt", {
        "schema": "lightgen_small_gaussian_structural_continuation_terminal.v1",
        "task_id": TASK_ID,
        "run_id": run_id,
        "branch": branch,
        "global_step": TOTAL_UPDATES,
        "model_identity": source["model_identity"],
        "model": endpoint_model_state,
    })
    summary = _evaluate_branch(
        branch, run_id, branch_dir, model, evaluator, pixels, labels, train_indices, dev_indices,
        device, source["camera_gain"].to(device), source["scale"].to(device), config,
    )
    summary["endpoint_window"] = {
        "status": "PASSED",
        "selected_production_grid": grid,
        "selected_again_at_endpoint": endpoint_grid,
        "complex_relative_error_max": endpoint_window["complex_relative_error_max"],
        "ols_coefficient_relative_error": endpoint_window["ols_coefficient_relative_error_by_grid"][grid],
        "camera_w_per_pixel_relative_error": endpoint_window["camera_w_per_pixel_relative_error_by_grid"][grid],
    }
    summary["independent_readout"] = endpoint_readout
    density_steps = sorted({density_start_step} | {
        global_step for global_step in range(density_start_step + 1, TOTAL_UPDATES + 1)
        if global_step % CHECKPOINT_INTERVAL == 0
    })
    plane_density_logging = base._validate_gaussian_plane_density_coverage(
        run_dir, branch, density_steps, grid,
    )
    summary["training"] = {
        "source_global_step": SOURCE_UPDATES,
        "continuation_updates": CONTINUATION_UPDATES,
        "endpoint_global_step": TOTAL_UPDATES,
        "adam_update_count": base._optimizer_update_count(optimizer, model),
        "history": history,
        "elapsed_seconds_this_process": time.perf_counter() - started,
        "seconds_per_scientific_update": (time.perf_counter() - started) / CONTINUATION_UPDATES,
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
        "device": torch.cuda.get_device_name(device),
        "plane_power_density_logging": plane_density_logging,
    }
    atomic_write_json(branch_dir / "endpoint_summary.json", summary)
    atomic_write_json(branch_dir / "training_history.json", history)
    return summary, elapsed_before + time.perf_counter() - started


def _aggregate_report(
    run_id: str, source_report: dict[str, Any], source: dict[str, Any], config: dict[str, Any],
    preflight: dict[str, Any], control: dict[str, Any], treatment: dict[str, Any],
    elapsed_seconds: float, run_dir: Path,
) -> dict[str, Any]:
    control_rows = control["reconstruction"]["per_image"]
    treatment_rows = treatment["reconstruction"]["per_image"]
    bootstrap = {metric: _stratified_paired_bootstrap(control_rows, treatment_rows, metric)
                 for metric in ("mse", "psnr_db", "ssim")}
    ssim_ci = bootstrap["ssim"]["ci_95"]
    if ssim_ci[0] > 0.0:
        interpretation = "STRUCTURAL_IMPROVEMENT_SUPPORTED_ON_FIXED_DEV"
    elif ssim_ci[1] < 0.0:
        interpretation = "STRUCTURAL_TERM_WORSENED_FIXED_DEV_SSIM"
    else:
        interpretation = "STRUCTURAL_IMPROVEMENT_UNCLEAR_ON_FIXED_DEV"
    source_endpoint = source_report["reconstruction"][-1]
    source_generation = source_report["generation"]
    endpoints = {
        "source": {
            "branch": "source",
            "global_step": SOURCE_UPDATES,
            "reconstruction": source_endpoint,
            "generation": source_generation,
            "status": source_report["status"],
            "measurement_complete": source_report["measurement_complete"],
        },
        "control": control,
        "treatment": treatment,
    }
    source_metrics = source_endpoint
    arm_delta = {}
    for name, record in (("control", control), ("treatment", treatment)):
        recon, generated = record["reconstruction"], record["generation"]
        arm_delta[name] = {
            "psnr_db_from_source": recon["psnr_db_mean"] - source_metrics["psnr_db_mean"],
            "ssim_from_source": recon["ssim_mean"] - source_metrics["ssim_mean"],
            "generation_macro_recall_from_source": generated["macro_recall"] - source_generation["macro_recall"],
            "generation_recall_by_class_from_source": [
                generated["recall_by_class"][i] - source_generation["recall_by_class"][i] for i in range(10)
            ],
        }
    treatment_gate = treatment["quality_gate"]
    complete = all(record.get("measurement_complete") is True for record in (control, treatment))
    status = treatment_gate["status"] if complete else "MEASUREMENT_INCOMPLETE"
    output_bytes = base._directory_bytes(run_dir)
    report = {
        "schema": "lightgen_small_gaussian_structural_continuation_ab_report.v1",
        "task_id": TASK_ID,
        "status": status,
        "measurement_complete": complete,
        "evidence_tier": "diagnostic_only",
        "run_id": run_id,
        "source": {
            "task_id": SOURCE_TASK_ID,
            "run_id": source["run_id"],
            "checkpoint_step": SOURCE_UPDATES,
            "source_checkpoint": "explicit user-specified 30,000-step checkpoint",
            "model_identity": source["model_identity"],
        },
        "scientific_question": "Does one frozen SSIM structural term improve the fixed 5,000-row DEV reconstruction under equal 5,000-update continuation budgets while retaining conditional-generation quality?",
        "training": {
            "branches": {
                "control": {"objective": "source BOGT", "updates": CONTINUATION_UPDATES},
                "treatment": {"objective": "source BOGT + frozen lambda_ssim * mean(1-SSIM)",
                              "updates": CONTINUATION_UPDATES,
                              "lambda_ssim": preflight["structural_calibration"]["lambda_ssim"]},
            },
            "source_global_step": SOURCE_UPDATES,
            "endpoint_global_step_each_branch": TOTAL_UPDATES,
            "branch_order": list(BRANCHES),
            "logical_batch_size": BATCH_SIZE,
            "paired_schedule_seed": SEED,
            "same_post_source_schedule_rows": True,
            "camera_gain_and_original_bogt_weights_frozen": True,
            "optimizer_source_state_preserved": True,
            "evaluator_frozen_and_excluded_from_loss": True,
            "optical_geometry_status": "PROJECT_GEOMETRY_UNMEASURED",
        },
        "preflight": preflight,
        "endpoints": endpoints,
        "change_from_source": arm_delta,
        "paired_comparison": {
            "treatment_minus_control_reconstruction": bootstrap,
            "ssim_interpretation": interpretation,
            "conditional_generation": {
                "macro_recall_difference": treatment["generation"]["macro_recall"]
                - control["generation"]["macro_recall"],
                "recall_by_class_difference": [treatment["generation"]["recall_by_class"][i]
                                                - control["generation"]["recall_by_class"][i]
                                                for i in range(10)],
                "proxy_clipping_by_class_difference": [
                    treatment["generation"]["proxy_clipping_fraction_by_class"][i]
                    - control["generation"]["proxy_clipping_fraction_by_class"][i] for i in range(10)
                ],
                "proposal_streams_shared_but_post_rejection_samples_not_strictly_paired": True,
            },
        },
        "decision": {
            "outer_gate": "complete effective treatment endpoint satisfies every inherited Gaussian system quality threshold",
            "status": status,
            "treatment_quality_passed": treatment_gate["passed"] if complete else False,
            "ssim_effect_interpretation": interpretation,
            "no_model_adoption_or_hardware_claim": True,
        },
        "elapsed_seconds_total": elapsed_seconds,
        "device": control["training"]["device"],
        "output": {
            "formal_result_budget_bytes": config["output"]["formal_result_budget_bytes"],
            "output_bytes_before_report_finalization": output_bytes,
            "within_budget": output_bytes <= int(config["output"]["formal_result_budget_bytes"]),
        },
        "artifacts": {
            "control_endpoint": "control/checkpoint_step_005000.pt",
            "treatment_endpoint": "treatment/checkpoint_step_005000.pt",
            "paired_metrics": "paired_comparison.json",
            "html_report": "results.html",
        },
    }
    if output_bytes > int(config["output"]["formal_result_budget_bytes"]):
        report["status"] = "MEASUREMENT_INCOMPLETE"
        report["measurement_complete"] = False
    return report


def _run(args: argparse.Namespace) -> None:
    if args.preflight_worker_dir is not None:
        _worker(args.preflight_worker_dir.resolve(), args.source_checkpoint.resolve(), args.config.resolve())
        return
    _require(torch.cuda.is_available(), "CUDA is required for the authorized Gaussian continuation")
    config_path = args.config.resolve()
    source_path = args.source_checkpoint.resolve()
    run_dir = args.run_dir.resolve()
    config = _read_config(config_path)
    _require(run_dir == (ROOT / config["output"]["experiment_root"]).resolve(),
             "run directory differs from the explicitly registered attempt path")
    resuming = args.resume_checkpoint is not None
    if resuming:
        _require(run_dir.is_dir() and not (run_dir / "report.json").is_file(),
                 "resume requires an existing incomplete attempt")
        resume_path = args.resume_checkpoint.resolve()
        _require(resume_path.is_file(), "explicit continuation resume checkpoint is missing")
        try:
            resume_path.relative_to(run_dir)
        except ValueError as error:
            raise RuntimeError("resume checkpoint must be inside the explicit continuation attempt") from error
    else:
        _require(not run_dir.exists(), f"refusing to overwrite existing attempt: {run_dir}")
    source, references, source_report = _load_source(source_path, config)
    pixels, labels, train_indices, dev_indices = base._load_data(ROOT / config["data"]["training_file"])
    schedule = _continued_schedule(labels, train_indices, SEED)
    _require(torch.equal(schedule[:SOURCE_UPDATES], source["schedule"]),
             "continued schedule prefix differs from the source 30,000-row schedule")
    _require(torch.equal(labels[schedule].cpu(), torch.arange(10).expand(TOTAL_UPDATES, 10)),
             "continued TRAIN schedule contains a wrong class row")
    evaluator_path = ROOT / config["evaluation"]["evaluator_checkpoint"]
    evaluator = base._load_evaluator(evaluator_path, torch.device("cuda"))
    evaluator_state = base._state_snapshot(evaluator)
    device = torch.device("cuda")
    run_id = uuid.uuid4().hex
    identity = _make_identity(config, source, run_id)
    authorization_id = f"{TASK_ID}:seed:{SEED}:preflight_and_execution"
    control = ResearchRunControl(identity, authorization_id=authorization_id)
    if resuming:
        saved_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        _require(saved_config == config, "resume config differs from the active attempt")
        control, identity, receipt = _load_ready_context(run_dir, config, source)
        run_id = identity.run_id
        saved_report = run_dir / "preflight.json"
        preflight = json.loads(saved_report.read_text(encoding="utf-8"))
        structural_weight = preflight["structural_calibration"]["lambda_ssim"]
        _require(receipt.status == "READY" and preflight.get("status") == "READY",
                 "resume attempt has no complete READY receipt")
        control.resume_scientific_attempt(run_dir, attempt_id="continuation_ab_seed_2026101501", receipt=receipt)
    else:
        stage_parent = run_dir.parent / f"preflight_structural_ab_{uuid.uuid4().hex[:8]}"
        stage_parent.mkdir(parents=True, exist_ok=False)
        try:
            preflight, structural_weight = _run_preflight(
                source, references, config, config_path, source_path, pixels, labels, train_indices,
                schedule, device, evaluator, stage_parent,
            )
            receipt = _make_receipt(control, preflight, device)
            control.mark_preflight_ready(receipt)
            if args.preflight_only:
                preflight_dir = run_dir.parent / f"{run_dir.name}_preflight_{run_id[:8]}"
                preflight_dir.mkdir(parents=False, exist_ok=False)
                atomic_write_new_json(preflight_dir / "config.json", config)
                atomic_write_new_json(preflight_dir / "preflight.json", preflight)
                atomic_write_new_json(preflight_dir / "preflight_ready.json", receipt.to_dict())
                atomic_write_new_json(preflight_dir / "run_identity.json", {
                    "identity": identity.to_dict(), "run_id": run_id, "authorization_id": authorization_id,
                })
                print(json.dumps({"status": "PREFLIGHT_READY", "directory": str(preflight_dir)}), flush=True)
                return
            run_dir.parent.mkdir(parents=True, exist_ok=True)
            control.open_scientific_attempt(run_dir, attempt_id="continuation_ab_seed_2026101501", receipt=receipt)
            for branch in BRANCHES:
                (run_dir / branch).mkdir(parents=False, exist_ok=False)
            atomic_write_new_json(run_dir / "config.json", config)
            atomic_write_new_json(run_dir / "preflight.json", preflight)
            atomic_write_new_json(run_dir / "preflight_ready.json", receipt.to_dict())
            atomic_write_new_json(run_dir / "run_identity.json", {
                "identity": identity.to_dict(), "run_id": run_id, "authorization_id": authorization_id,
                "source_run_id": source["run_id"],
            })
            atomic_torch_save_new(run_dir / "source_references.pt", {
                "scale": source["scale"], "power_reference": source["power_reference"],
                "camera_gain": source["camera_gain"], "weights": source["weights"],
                "source_checkpoint_step": SOURCE_UPDATES,
                "structural_weight": structural_weight,
            })
            source_run = source_path.parent
            for src_name, dest_name in (("dev_preview.png", "source_dev_preview.png"),
                                        ("dev_target_preview.png", "source_dev_target_preview.png"),
                                        ("generation_preview.png", "source_generation_preview.png")):
                source_image = source_run / src_name
                _require(source_image.is_file(), f"source fixed-sample preview is missing: {source_image}")
                (run_dir / dest_name).write_bytes(source_image.read_bytes())
        finally:
            for path in stage_parent.iterdir():
                if path.is_file():
                    path.unlink()
                else:
                    import shutil
                    shutil.rmtree(path)
            stage_parent.rmdir()

    _require(base._assert_close(evaluator_state, base._state_snapshot(evaluator), "evaluator_before_science") == 0.0,
             "frozen confirmation evaluator changed before continuation")
    start_resume = args.resume_checkpoint.resolve() if resuming else None
    elapsed_total = 0.0
    endpoint_summaries: dict[str, dict[str, Any]] = {}
    for branch in BRANCHES:
        if resuming and start_resume is not None:
            checkpoint = torch.load(start_resume, map_location="cpu", weights_only=False)
            resume_branch = checkpoint.get("branch")
            if resume_branch == branch:
                branch_resume = start_resume
            elif resume_branch == "control" and branch == "treatment":
                branch_resume = None
            else:
                branch_resume = None
        else:
            branch_resume = None
        if branch == "treatment" and resuming:
            _require((run_dir / "control" / "endpoint_summary.json").is_file(),
                     "treatment cannot resume before the full control endpoint is evaluated")
        if branch_resume is not None and branch_resume.parent.name != branch:
            branch_resume = None
        summary, elapsed_total = _run_branch(
            branch, run_dir, run_id, source, config, pixels, labels, train_indices, dev_indices,
            schedule, {**references, "structural_weight": structural_weight}, evaluator, device,
            branch_resume, elapsed_total,
        )
        endpoint_summaries[branch] = summary
        atomic_write_json(run_dir / "progress_summary.json", {
            "branch": branch, "branch_endpoint_complete": True,
            "completed_global_step": TOTAL_UPDATES, "status": summary["status"],
        })

    _require(set(endpoint_summaries) == set(BRANCHES), "one or more continuation endpoints are missing")
    _require(base._assert_close(evaluator_state, base._state_snapshot(evaluator), "evaluator_after_science") == 0.0,
             "frozen confirmation evaluator changed during the complete experiment")
    report = _aggregate_report(run_id, source_report, source, config, preflight,
                               endpoint_summaries["control"], endpoint_summaries["treatment"],
                               elapsed_total, run_dir)
    paired_path = run_dir / "paired_comparison.json"
    atomic_write_json(paired_path, report["paired_comparison"])
    _write_comparison_html(run_dir / "results.html", report)
    report["output"]["output_bytes_before_report_finalization"] = base._directory_bytes(run_dir)
    report["output"]["within_budget"] = report["output"]["output_bytes_before_report_finalization"] <= int(
        config["output"]["formal_result_budget_bytes"])
    if not report["output"]["within_budget"]:
        report["status"] = "MEASUREMENT_INCOMPLETE"
        report["measurement_complete"] = False
    atomic_write_json(run_dir / "report.json", report)
    if report["status"] == "VALID_FAILED":
        control.valid_failed()
    elif report["status"] == "VALID_COMPLETED_QUALITY_PASSED":
        control.complete()
    else:
        control.invalid_harness()
    atomic_write_json(run_dir / "run_state.json", {
        "status": control.state,
        "scientific_status": report["status"],
        "completed_global_step_each_branch": TOTAL_UPDATES,
        "measurement_complete": report["measurement_complete"],
    })
    print(json.dumps({"status": report["status"], "run_control_state": control.state,
                      "report": str(run_dir / "report.json")}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--preflight-worker-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.preflight_only and args.resume_checkpoint is not None:
        raise SystemExit("--preflight-only cannot be combined with --resume-checkpoint")
    _run(args)


if __name__ == "__main__":
    main()
