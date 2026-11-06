"""Reproduce two Gaussian training schedules and confirm fixed G1/G4 priors."""

from __future__ import annotations

import argparse
import copy
import csv
import html
import json
import math
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

import scripts.research.lightgen.execute_lightgen_matched_architecture_v1 as base
import scripts.research.lightgen.execute_lightgen_small_conditional_mixture_prior_ab_v1 as prior
import scripts.research.lightgen.execute_lightgen_small_structural_continuation_ab_v1 as structural
from ldm_project.v2.application.research.lightgen.confirmation_evaluators import ThirdConfirmationEvaluator
from ldm_project.v2.application.research.lightgen.native_image_inputs import image_batch
from ldm_project.v2.application.research.lightgen.image_readout import resize_view, ssim_batch
from ldm_project.v2.evidence import atomic_torch_save, atomic_write_json
from ldm_project.v2.evidence.io import cpu_copy


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_SMALL_GAUSSIAN_REPLICATION_PRIOR_CONFIRMATION_V1"
AUTHORITY = "USER_APPROVED_SMALL_GAUSSIAN_REPLICATION_PRIOR_CONFIRMATION_20261021"
SEEDS = (2026102151, 2026102152)
REFERENCE_SEED = 2026101501
TRAIN_COUNT = 55_000
DEV_COUNT = 5_000
UPDATES = 45_000
BOGT_UPDATES = 30_000
STRUCTURAL_UPDATES = 15_000
BATCH_SIZE = 10
CHECKPOINT_INTERVAL = 500
PROGRESS_INTERVAL = 100
EVALUATION_STEPS = (30_000, 35_000, 45_000)
WINDOW_TOLERANCE = 1e-4
REPLAY_RTOL = 2e-6
REPLAY_ATOL = 5e-8
BOOTSTRAP_RESAMPLES = 2_000
G1_NORMAL_SEED_BASE = 2026102200
G4_NORMAL_SEED_BASE = 2026102210
G4_COMPONENT_SEED_BASE = 2026102220
BOOTSTRAP_SEED_BASE = 2026102261
SHAPE_PAIR_SEED_BASE = 2026102501
SAMPLE_COUNT = 1_000
MAX_PROPOSALS = 1_000_000
TRAIN_LABEL_COUNTS = (5423, 6242, 5458, 5631, 5342, 4921, 5418, 5765, 5351, 5449)
QUALITY_GATE = {
    "reconstruction_psnr_db_min": 20.0,
    "reconstruction_ssim_min": 0.80,
    "generation_macro_recall_min": 0.80,
    "generation_recall_per_class_min": 0.60,
    "proxy_clipping_fraction_max": 0.01,
    "generated_to_dev_tv_ratio_min": 0.25,
}


def _require(condition: bool, message: str, error_type: type[Exception] = RuntimeError) -> None:
    if not condition:
        raise error_type(message)


def _resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _read_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(config, dict)
             and config.get("schema") == "lightgen_small_gaussian_replication_prior_confirmation.v1"
             and config.get("task_id") == TASK_ID and config.get("semantic_authority_id") == AUTHORITY,
             "replication configuration identity changed", ValueError)
    reference = config["reference"]
    _require(reference.get("source_task_id") == "LIGHTGEN_SMALL_GAUSSIAN_FOREGROUND_BALANCE_AB_V1"
             and reference.get("source_branch") == "C"
             and reference.get("source_cumulative_step") == UPDATES
             and reference.get("checkpoint") == (
                 "outputs/20261021_lightgen_small_gaussian_foreground_balance_ab_v1/runs/"
                 "autodl_backup_attempt_01/C/checkpoint_step_010000.pt"
             )
             and reference.get("baseline_evidence_dir") == (
                 "outputs/20261021_lightgen_small_gaussian_foreground_balance_ab_v1/runs/"
                 "autodl_backup_attempt_01"
             )
             and reference.get("prior_evidence_dir") == (
                 "outputs/20261021_lightgen_small_conditional_mixture_prior_ab_v1/runs/"
                 "autodl_backup_attempt_01"
             )
             and reference.get("source_30000_35000_checkpoint") == (
                 "outputs/20261020_lightgen_small_gaussian_structural_continuation_ab_v1/"
                 "runs/autodl_backup_attempt_01/treatment/checkpoint_step_005000.pt"
             ),
             "explicit source endpoint and reference evidence changed", ValueError)
    training = config["training"]
    _require(tuple(training.get("seeds", ())) == SEEDS and training.get("updates") == UPDATES
             and training.get("bogt_updates") == BOGT_UPDATES
             and training.get("structural_updates") == STRUCTURAL_UPDATES
             and training.get("logical_batch_size") == BATCH_SIZE
             and training.get("initialization") == "zero_phase"
             and training.get("optimizer") == "Adam"
             and training.get("learning_rate") == 1e-3
             and training.get("betas") == [0.9, 0.999]
             and training.get("epsilon") == 1e-8 and training.get("weight_decay") == 0.0,
             "fixed training schedule or optimizer changed", ValueError)
    data = config["data"]
    _require(data.get("training_file") == "data/MNIST/processed/training.pt"
             and data.get("split_seed") == 1234 and data.get("train_count") == TRAIN_COUNT
             and data.get("train_label_counts") == list(TRAIN_LABEL_COUNTS)
             and data.get("dev_count") == DEV_COUNT and data.get("dev_per_class") == 500
             and data.get("official_test_reads") == 0,
             "fixed MNIST roles or population changed", ValueError)
    optics = config["optics"]
    _require(optics.get("encoder_layers") == 2 and optics.get("decoder_layers") == 2
             and optics.get("phase_shape") == [100, 100]
             and optics.get("trainable_phase_parameters") == 40_000
             and optics.get("slm_shape") == [200, 200]
             and optics.get("basis") == "gaussian_s12_v1"
             and optics.get("ols_ports") == 100
             and optics.get("wavelength_m") == 5.32e-7
             and optics.get("input_power_w") == 2.493e-4
             and optics.get("layer_power_transmission") == 0.889
             and optics.get("phase_pixel_pitch_m") == 2.0e-6
             and optics.get("layer_spacing_m") == 1.5e-3
             and optics.get("port_pitch_m") == 2.0e-5
             and optics.get("field_1e_radius_m") == 3.5e-6
             and optics.get("transport") == "identity"
             and optics.get("relay") == "existing_4f_input_and_b0_output"
             and optics.get("production_grid") == 2048 and optics.get("reference_grid") == 4096
             and optics.get("geometry_status") == "PROJECT_GEOMETRY_UNMEASURED",
             "fixed Gaussian geometry or qualification changed", ValueError)
    evaluation = config["evaluation"]
    _require(evaluation.get("sampling_streams") == 3
             and evaluation.get("generated_samples_per_class_per_stream") == SAMPLE_COUNT
             and evaluation.get("maximum_proposals_per_class") == MAX_PROPOSALS
             and evaluation.get("bootstrap_resamples") == BOOTSTRAP_RESAMPLES
             and evaluation.get("g1_normal_seed_base") == G1_NORMAL_SEED_BASE
             and evaluation.get("g4_normal_seed_base") == G4_NORMAL_SEED_BASE
             and evaluation.get("g4_component_seed_base") == G4_COMPONENT_SEED_BASE
             and evaluation.get("bootstrap_seed_base") == BOOTSTRAP_SEED_BASE
             and evaluation.get("shape_pair_count") == 10_000
             and evaluation.get("shape_pair_seed_base") == SHAPE_PAIR_SEED_BASE
             and evaluation.get("window_error_max") == WINDOW_TOLERANCE
             and evaluation.get("replay_rtol") == REPLAY_RTOL
             and evaluation.get("replay_atol") == REPLAY_ATOL
             and evaluation.get("classifier_based_sample_selection") is False
             and evaluation.get("per_image_normalization") is False
             and evaluation.get("official_test_reads") == 0,
             "fixed evaluation population or thresholds changed", ValueError)
    _require(evaluation.get("ed_checkpoint") == (
                 "outputs/20260920_sic_independent_confirmation_evaluator_v1/runs/attempt_01/"
                 "final_epoch_candidate.pt"
             )
             and evaluation.get("ec_checkpoint") == (
                 "outputs/20260919_third_confirmation_evaluator_v1/runs/attempt_03/"
                 "evaluator_checkpoint.pt"
             ), "frozen E_D/E_C checkpoint path changed", ValueError)
    loss = config["loss"]
    _require(loss.get("family") == "truncated_gaussian_bogt"
             and loss.get("reconstruction") == "fixed_camera_gain_unclipped_native_physical_w_per_pixel_mse"
             and loss.get("distribution") == "truncated_positive_gaussian_kl_mean_1_variance_0_25"
             and loss.get("plane_power") == "relative_log_power_squared_mean_over_model_planes"
             and loss.get("initial_weight_calibration") == "inverse_component_gradient_norm_once_first_balanced_train_batch"
             and loss.get("weights_frozen_after_calibration") is True
             and loss.get("camera_gain_frozen_after_calibration") is True
             and loss.get("ssim", {}).get("definition") == "gaussian_7x7_sigma_1_5_data_range_1_zero_pad_3"
             and loss.get("ssim", {}).get("calibrate_once_at_global_step") == BOGT_UPDATES
             and loss.get("ssim", {}).get("freeze_after_calibration") is True
             and loss.get("ssim", {}).get("multiplier") == 1.0
             and loss.get("ssim", {}).get("evaluator_in_loss") is False,
             "BOGT loss or one-time SSIM calibration changed", ValueError)
    _require(config["output"].get("formal_result_budget_bytes") == 1_073_741_824
             and config["output"].get("primary_stream") == 0
             and config["output"].get("experiment_root") == (
                 "outputs/20261021_lightgen_small_gaussian_replication_prior_confirmation_v1/"
                 "runs/attempt_01"
             ),
             "formal output budget or primary sample stream changed", ValueError)
    runtime_path = _resolve(Path(config["runtime_config"]))
    runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    _require(runtime.get("schema") == "lightgen_matched_small_vs_chen400.v1"
             and runtime["optics"]["small"].get("ols_basis") == "gaussian_s12_v1"
             and runtime["optics"]["small"].get("production_propagation_grid") == 2048
             and runtime["optics"]["small"].get("geometry_status") == "PROJECT_GEOMETRY_UNMEASURED"
             and runtime["data"]["training_file"] == data["training_file"]
             and runtime["data"]["split_seed"] == data["split_seed"],
             "runtime model factory configuration differs from the registered geometry", ValueError)
    return config, runtime


def _full_schedule(labels: torch.Tensor, train_rows: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    by_class = []
    for label in range(10):
        rows = train_rows[labels[train_rows] == label]
        _require(rows.numel() == TRAIN_LABEL_COUNTS[label], f"TRAIN class {label} population changed")
        by_class.append(rows[torch.randperm(rows.numel(), generator=generator)])
    schedule = torch.empty((UPDATES, BATCH_SIZE), dtype=torch.long)
    cursors = [0] * 10
    for step in range(UPDATES):
        for label in range(10):
            if cursors[label] == by_class[label].numel():
                cursors[label] = 0
            schedule[step, label] = by_class[label][cursors[label]]
            cursors[label] += 1
    _require(torch.equal(labels[schedule], torch.arange(10).expand(UPDATES, BATCH_SIZE)),
             "45,000-step schedule is not class-balanced")
    train_membership = torch.zeros(labels.shape[0], dtype=torch.bool)
    train_membership[train_rows] = True
    _require(bool(train_membership[schedule].all()), "schedule includes a row outside TRAIN")
    return schedule


def _load_evidence(
    checkpoint_path: Path, baseline_dir: Path, prior_dir: Path, runtime: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    _require(checkpoint_path.is_file(), f"explicit C checkpoint is missing: {checkpoint_path}", FileNotFoundError)
    _require(baseline_dir.is_dir() and prior_dir.is_dir(), "explicit reference evidence directory is missing", FileNotFoundError)
    old_config_path = prior_dir / "config.json"
    _require(old_config_path.is_file(), "historical G1/G4 configuration is missing", FileNotFoundError)
    old_config = json.loads(old_config_path.read_text(encoding="utf-8"))
    baseline_context = prior._load_context(checkpoint_path, baseline_dir, old_config)
    checkpoint = baseline_context["checkpoint"]
    prior_report_path = prior_dir / "report.json"
    _require(prior_report_path.is_file(), "historical G1/G4 report is missing", FileNotFoundError)
    prior_report = json.loads(prior_report_path.read_text(encoding="utf-8"))
    _require(prior_report.get("task_id") == prior.TASK_ID
             and prior_report.get("status") == "COMPLETE_DIAGNOSTIC"
             and prior_report.get("measurement_complete") is True
             and prior_report.get("source", {}).get("branch") == "C"
             and prior_report.get("source", {}).get("cumulative_step") == UPDATES
             and prior_report.get("source", {}).get("run_id") == baseline_context["baseline_report"].get("run_id")
             and prior_report.get("source", {}).get("historical_status_preserved") == "VALID_FAILED",
             "historical G1/G4 result is incomplete or not tied to the fixed C endpoint")
    _require(checkpoint["model_identity"] == {
        "encoder": [2, 100, 100], "ols": "gaussian_s12_v1",
        "ols_geometry": {"port_pitch_m": 2e-5, "field_1e_radius_m": 3.5e-6, "transport": "identity"},
        "decoder": [2, 100, 100],
    }, "C endpoint model identity changed")
    priors = torch.load(prior_dir / "fitted_priors_G1_G4.pt", map_location="cpu", weights_only=False)
    cache = torch.load(prior_dir / "train_coefficient_cache.pt", map_location="cpu", weights_only=False)
    _require(priors.get("task_id") == prior.TASK_ID
             and set(priors.get("G1", {})) == set(range(10))
             and set(priors.get("G4", {})) == set(range(10)), "historical G1/G4 priors are incomplete")
    _require(cache.get("task_id") == prior.TASK_ID
             and cache.get("source_run_id") == baseline_context["baseline_report"].get("run_id"),
             "historical TRAIN feature cache belongs to another task or source endpoint")
    _require(tuple(cache.get("rows", torch.empty(0)).shape) == (TRAIN_COUNT,)
             and tuple(cache.get("labels", torch.empty(0)).shape) == (TRAIN_COUNT,)
             and tuple(cache.get("features_200", torch.empty(0)).shape) == (TRAIN_COUNT, 200)
             and torch.isfinite(cache["features_200"]).all().item(),
             "historical 55,000-row coefficient cache is incomplete")
    schedule_checkpoint_path = _resolve(Path(config["reference"]["source_30000_35000_checkpoint"]))
    _require(schedule_checkpoint_path.is_file(), f"30,000-35,000 schedule evidence is missing: {schedule_checkpoint_path}", FileNotFoundError)
    schedule_checkpoint = torch.load(schedule_checkpoint_path, map_location="cpu", weights_only=False)
    _require(schedule_checkpoint.get("seed") == REFERENCE_SEED
             and schedule_checkpoint.get("completed_global_step") == 35_000
             and tuple(schedule_checkpoint.get("schedule", torch.empty(0)).shape) == (5_000, BATCH_SIZE),
             "historical 30,000-35,000 schedule segment is incomplete or misidentified")
    ed_path = _resolve(Path(config["evaluation"]["ed_checkpoint"]))
    ec_path = _resolve(Path(config["evaluation"]["ec_checkpoint"]))
    return {
        "checkpoint": checkpoint, "references": baseline_context["references"],
        "baseline_report": baseline_context["baseline_report"],
        "priors": priors, "cache": cache, "source_schedule_30000_35000": schedule_checkpoint["schedule"],
        "ed_path": ed_path, "ec_path": ec_path, "old_config": old_config,
        "prior_report": prior_report, "runtime_config": runtime,
    }


def _model_identity() -> dict[str, Any]:
    return {
        "encoder": [2, 100, 100], "ols": "gaussian_s12_v1",
        "ols_geometry": {"port_pitch_m": 2e-5, "field_1e_radius_m": 3.5e-6, "transport": "identity"},
        "decoder": [2, 100, 100], "phase_shape": [2, 1, 100, 100],
        "trainable_phase_parameters": 40_000,
        "geometry_status": "PROJECT_GEOMETRY_UNMEASURED",
    }


def _rng_state() -> dict[str, Any]:
    return {"cpu_rng": torch.random.get_rng_state(),
            "cuda_rng": [state.cpu() for state in torch.cuda.get_rng_state_all()]}


def _set_rng_state(state: dict[str, Any]) -> None:
    torch.random.set_rng_state(state["cpu_rng"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])


def _train_one(
    model: torch.nn.Module, optimizer: torch.optim.Adam, target_native: torch.Tensor,
    target28: torch.Tensor, scale: torch.Tensor, power_reference: torch.Tensor,
    gain: torch.Tensor, weights: dict[str, float], lambda_ssim: float | None,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    parts, result = base._loss_components(model, target_native, scale, power_reference, gain)
    total = sum(weights[name] * parts[name] for name in ("dist", "rec", "op"))
    ssim_loss: torch.Tensor | None = None
    if lambda_ssim is not None:
        ssim_loss = structural._structural_loss_from_raw(result["physical_w_per_pixel"], gain, target28)
        total = total + lambda_ssim * ssim_loss
    _require(torch.isfinite(total).item(), "training objective is nonfinite")
    total.backward()
    groups = base._phase_groups(model)
    old = [parameter.detach().clone() for parameter in groups]
    gradient_norms = []
    for name, parameter in zip(("encoder", "decoder"), groups, strict=True):
        grad = parameter.grad
        _require(grad is not None and torch.isfinite(grad).all().item() and bool(grad.abs().sum() > 0),
                 f"{name} phase gradient is missing, nonfinite, or zero")
        gradient_norms.append(float(torch.linalg.vector_norm(grad.detach().double()).cpu()))
    optimizer.step()
    for name, previous, parameter in zip(("encoder", "decoder"), old, groups, strict=True):
        _require(not torch.equal(previous, parameter), f"{name} phase group did not update")
    return {
        **{name: float(value.detach()) for name, value in parts.items()},
        "ssim": 0.0 if ssim_loss is None else float(ssim_loss.detach()),
        "total": float(total.detach()),
        "encoder_gradient_norm": gradient_norms[0], "decoder_gradient_norm": gradient_norms[1],
        "mean_plane_power_w": float(result["plane_powers_w"].detach().double().mean()),
    }


def _update_targets(pixels: torch.Tensor, rows: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    native = image_batch(pixels, rows, "small", device)
    target28 = pixels.index_select(0, rows.detach().cpu()).unsqueeze(1).to(device=device, dtype=torch.float32) / 255.0
    _require(tuple(native.shape) == (BATCH_SIZE, 1, 200, 200)
             and tuple(target28.shape) == (BATCH_SIZE, 1, 28, 28), "TRAIN image view geometry changed")
    return native, target28


def _tree_max_error(left: Any, right: Any, name: str) -> float:
    return base._assert_close(left, right, name)


def _recovery_worker(directory: Path, config_path: Path) -> None:
    payload_path = directory / "recovery_input.pt"
    _require(payload_path.is_file(), "recovery worker input is missing")
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    _, runtime = _read_config(config_path)
    device = torch.device("cuda")
    seed = int(payload["seed"])
    model = base._model("small", 2048, device, runtime)
    model.load_state_dict(payload["model"], strict=True)
    optimizer = torch.optim.Adam(base._phase_groups(model), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    optimizer.load_state_dict(payload["optimizer"])
    _set_rng_state(payload["rng"])
    metrics = _train_one(model, optimizer, payload["next_native"].to(device),
                         payload["next_target28"].to(device), payload["scale"].to(device),
                         payload["power_reference"].to(device), payload["camera_gain"].to(device),
                         payload["weights"], payload["lambda_ssim"])
    result = {"model": base._state_snapshot(model), "optimizer": cpu_copy(optimizer.state_dict()),
              "metrics": metrics, "rng": _rng_state(), "seed": seed, "stage": payload["stage"]}
    atomic_torch_save(directory / "recovery_output.pt", result)


def _preflight_recovery(
    *, model: torch.nn.Module, optimizer: torch.optim.Adam, pixels: torch.Tensor,
    schedule: torch.Tensor, start_step: int, seed: int, stage: str,
    scale: torch.Tensor, power_reference: torch.Tensor, gain: torch.Tensor,
    weights: dict[str, float], lambda_ssim: float | None, config_path: Path,
    directory: Path,
) -> dict[str, Any]:
    _require(stage in {"bogt", "bogt_ssim"}, "unknown recovery stage")
    expected_lambda = None if stage == "bogt" else lambda_ssim
    _require((stage == "bogt" and expected_lambda is None)
             or (stage == "bogt_ssim" and isinstance(expected_lambda, float) and expected_lambda > 0),
             "recovery stage has the wrong frozen objective")
    saved_rng = _rng_state()
    clone = base._model("small", 2048, next(model.parameters()).device, _read_config(config_path)[1])
    clone.load_state_dict(base._state_snapshot(model), strict=True)
    clone_optimizer = torch.optim.Adam(base._phase_groups(clone), lr=1e-3, betas=(0.9, 0.999),
                                       eps=1e-8, weight_decay=0.0)
    clone_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    _set_rng_state(saved_rng)
    updates: list[dict[str, float]] = []
    for offset in range(3):
        native, target28 = _update_targets(pixels, schedule[start_step + offset], next(clone.parameters()).device)
        updates.append(_train_one(clone, clone_optimizer, native, target28, scale, power_reference,
                                  gain, weights, expected_lambda))
    after_three = {
        "model": base._state_snapshot(clone), "optimizer": cpu_copy(clone_optimizer.state_dict()),
        "rng": _rng_state(), "seed": seed, "stage": stage,
    }
    directory.mkdir(parents=True, exist_ok=False)
    native_next, target_next = _update_targets(pixels, schedule[start_step + 3], next(clone.parameters()).device)
    worker_payload = {
        **after_three, "next_native": native_next.cpu(), "next_target28": target_next.cpu(),
        "scale": scale.detach().cpu(), "power_reference": power_reference.detach().cpu(),
        "camera_gain": gain.detach().cpu(), "weights": weights, "lambda_ssim": expected_lambda,
    }
    atomic_torch_save(directory / "recovery_input.pt", worker_payload)
    continuous_model = base._model("small", 2048, next(model.parameters()).device, _read_config(config_path)[1])
    continuous_model.load_state_dict(after_three["model"], strict=True)
    continuous_optimizer = torch.optim.Adam(base._phase_groups(continuous_model), lr=1e-3,
                                            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    continuous_optimizer.load_state_dict(copy.deepcopy(after_three["optimizer"]))
    _set_rng_state(after_three["rng"])
    continuous_metrics = _train_one(continuous_model, continuous_optimizer, native_next, target_next,
                                    scale, power_reference, gain, weights, expected_lambda)
    continuous = {"model": base._state_snapshot(continuous_model),
                  "optimizer": cpu_copy(continuous_optimizer.state_dict()),
                  "metrics": continuous_metrics}
    command = [sys.executable, "-m", "scripts.research.lightgen.execute_lightgen_small_gaussian_replication_prior_confirmation_v1",
               "--recovery-worker-dir", str(directory), "--config", str(config_path)]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    _require(completed.returncode == 0, f"independent recovery worker failed: {completed.stderr[-2000:]}")
    restored = torch.load(directory / "recovery_output.pt", map_location="cpu", weights_only=False)
    model_error = _tree_max_error(continuous["model"], restored["model"], f"{stage}_recovery_model")
    optimizer_error = _tree_max_error(continuous["optimizer"], restored["optimizer"], f"{stage}_recovery_optimizer")
    metric_error = _tree_max_error(continuous_metrics, restored["metrics"], f"{stage}_recovery_metrics")
    _require(model_error <= REPLAY_ATOL and optimizer_error <= REPLAY_ATOL and metric_error <= REPLAY_ATOL,
             f"{stage} continuous and independent-process fourth updates differ")
    _set_rng_state(saved_rng)
    return {"stage": stage, "seed": seed, "real_updates_in_control_segment": 3,
            "continuous_fourth_update": True, "independent_process_fourth_update": True,
            "model_max_abs_error": model_error, "optimizer_max_abs_error": optimizer_error,
            "metric_max_abs_error": metric_error, "engine_adam_calls": 5,
            "all_engine_updates_rolled_back": True, "passed": True}


def _initial_model(seed: int, runtime: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, torch.optim.Adam]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    model = base._model("small", 2048, device, runtime)
    encoder, decoder = base._phase_groups(model)
    _require(sum(p.numel() for p in (encoder, decoder)) == 40_000
             and bool(torch.count_nonzero(encoder) == 0) and bool(torch.count_nonzero(decoder) == 0),
             "new trajectory must begin from four zero phase planes")
    optimizer = torch.optim.Adam((encoder, decoder), lr=1e-3, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0)
    _require(not optimizer.state, "new trajectory Adam must be fresh")
    return model, optimizer


def _save_checkpoint(run_dir: Path, payload: dict[str, Any], step: int, milestone: bool) -> None:
    rolling_latest = run_dir / "checkpoint_latest.pt"
    rolling_previous = run_dir / "checkpoint_previous.pt"
    if rolling_latest.is_file():
        previous = torch.load(rolling_latest, map_location="cpu", weights_only=False)
        _require(previous.get("task_id") == TASK_ID and previous.get("seed") == payload["seed"],
                 "rolling checkpoint identity changed before rotation")
        atomic_torch_save(rolling_previous, previous)
    atomic_torch_save(rolling_latest, payload)
    checked = torch.load(rolling_latest, map_location="cpu", weights_only=False)
    _require(checked.get("completed_global_step") == step
             and checked.get("seed") == payload["seed"]
             and checked.get("optimizer_step_count") == step,
             "rolling checkpoint failed read-back")
    if milestone:
        atomic_torch_save(run_dir / f"checkpoint_step_{step:06d}.pt", payload)
        named = torch.load(run_dir / f"checkpoint_step_{step:06d}.pt", map_location="cpu", weights_only=False)
        _require(named.get("completed_global_step") == step, "named milestone checkpoint failed read-back")


def _checkpoint_payload(
    model: torch.nn.Module, optimizer: torch.optim.Adam, seed: int, run_id: str,
    schedule: torch.Tensor, scale: torch.Tensor, power_reference: torch.Tensor,
    gain: torch.Tensor, weights: dict[str, float], lambda_ssim: float | None,
    step: int, history: list[dict[str, Any]], elapsed: float,
) -> dict[str, Any]:
    return {
        "schema": "lightgen_small_gaussian_replication_checkpoint.v1", "task_id": TASK_ID,
        "run_id": run_id, "seed": seed, "model_identity": _model_identity(),
        "completed_global_step": step, "completed_updates": step,
        "sample_cursor": step,
        "next_sample_rows": schedule[step].clone() if step < UPDATES else torch.empty((0, BATCH_SIZE), dtype=torch.long),
        "optimizer_step_count": _phase_optimizer_update_count(optimizer, model),
        "phase": "bogt" if step <= BOGT_UPDATES else "bogt_plus_frozen_ssim",
        "schedule": schedule, "model": cpu_copy(model.state_dict()),
        "optimizer": cpu_copy(optimizer.state_dict()), "scale": scale.detach().cpu(),
        "power_reference": power_reference.detach().cpu(), "camera_gain": gain.detach().cpu(),
        "weights": dict(weights), "lambda_ssim": lambda_ssim,
        **_rng_state(), "history": history, "elapsed_seconds": elapsed,
        "evaluator_role": "frozen_confirmation_only", "official_test_reads": 0,
    }


def _phase_optimizer_update_count(optimizer: torch.optim.Adam, model: torch.nn.Module) -> int:
    parameters = base._phase_groups(model)
    states = [optimizer.state.get(parameter) for parameter in parameters]
    if all(state is None for state in states):
        return 0
    _require(all(isinstance(state, dict) for state in states),
             "Adam phase groups are only partially initialized")
    steps: list[int] = []
    for state in states:
        _require("step" in state, "Adam checkpoint is missing step for a trainable phase group")
        value = state["step"]
        steps.append(int(value.item()) if isinstance(value, torch.Tensor) else int(value))
    _require(len(set(steps)) == 1, "Adam phase groups have different update counts")
    return steps[0]


def _restore_checkpoint(
    checkpoint_path: Path, run_dir: Path, seed: int, config: dict[str, Any], runtime: dict[str, Any],
    schedule: torch.Tensor, device: torch.device,
) -> tuple[dict[str, Any], torch.nn.Module, torch.optim.Adam]:
    _require(checkpoint_path.is_file(), f"explicit resume checkpoint is missing: {checkpoint_path}", FileNotFoundError)
    config_path = run_dir.parent / "config.json"
    _require(config_path.is_file()
             and json.loads(config_path.read_text(encoding="utf-8")) == config,
             "resume configuration differs from the recorded run")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    step = checkpoint.get("completed_global_step")
    _require(checkpoint.get("schema") == "lightgen_small_gaussian_replication_checkpoint.v1"
             and checkpoint.get("task_id") == TASK_ID and checkpoint.get("seed") == seed
             and checkpoint.get("model_identity") == _model_identity()
             and isinstance(step, int) and 0 <= step <= UPDATES
             and checkpoint.get("completed_updates") == step
             and checkpoint.get("sample_cursor") == step
             and torch.equal(checkpoint.get("next_sample_rows"),
                             schedule[step] if step < UPDATES else torch.empty((0, BATCH_SIZE), dtype=torch.long))
             and checkpoint.get("optimizer_step_count") == step
             and torch.equal(checkpoint.get("schedule"), schedule)
             and checkpoint.get("official_test_reads") == 0,
             "resume checkpoint identity, step, or schedule is invalid")
    model = base._model("small", 2048, device, runtime)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer = torch.optim.Adam(base._phase_groups(model), lr=1e-3, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0.0)
    optimizer.load_state_dict(checkpoint["optimizer"])
    _require(_phase_optimizer_update_count(optimizer, model) == step,
             "resumed Adam moments or step count do not match the endpoint")
    references_path = run_dir / "initial_references.pt"
    _require(references_path.is_file(), "resume initial frozen references are missing")
    references = torch.load(references_path, map_location="cpu", weights_only=False)
    _require(torch.equal(references.get("schedule"), schedule), "saved full training schedule changed")
    for name in ("scale", "power_reference", "camera_gain"):
        _require(torch.allclose(torch.as_tensor(references[name]), torch.as_tensor(checkpoint[name]),
                                rtol=0.0, atol=0.0), f"resume frozen reference {name} changed")
    _require(references.get("weights") == checkpoint.get("weights"), "resume BOGT coefficients changed")
    _set_rng_state(checkpoint)
    return checkpoint, model, optimizer


def _read_resume_state(path: Path, run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    _require(path.is_file(), f"explicit resume-state file is missing: {path}", FileNotFoundError)
    state = json.loads(path.read_text(encoding="utf-8"))
    _require(state.get("task_id") == TASK_ID and state.get("config_identity") == config
             and Path(state.get("output_dir", "")).resolve() == run_dir.resolve(),
             "resume-state task, output path, or configuration identity changed")
    _require(not (run_dir / "report.json").is_file(),
             "completed task reports cannot be resumed")
    return state


def _write_progress(path: Path, output_dir: Path, config: dict[str, Any], state: dict[str, Any]) -> None:
    payload = {"task_id": TASK_ID, "config_identity": config, "output_dir": str(output_dir),
               "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **state}
    atomic_write_json(path, payload)


def _merge_progress(path: Path, output_dir: Path, config: dict[str, Any], updates: dict[str, Any]) -> None:
    """Update live progress without dropping completed seed/checkpoint identities."""
    state: dict[str, Any] = {}
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
        _require(current.get("task_id") == TASK_ID and current.get("config_identity") == config
                 and Path(current.get("output_dir", "")).resolve() == output_dir.resolve(),
                 "live progress identity changed")
        state = {key: value for key, value in current.items()
                 if key not in {"task_id", "config_identity", "output_dir", "updated_at"}}
    state.update(updates)
    _write_progress(path, output_dir, config, state)


def _phase_nontrainable_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state = base._state_snapshot(model)
    return {name: value for name, value in state.items()
            if name not in {"runtime.encoder.phase", "runtime.generator.phase"}}


def _calibrate_structural(
    model: torch.nn.Module, pixels: torch.Tensor, rows: torch.Tensor, scale: torch.Tensor,
    power_reference: torch.Tensor, gain: torch.Tensor, weights: dict[str, float],
    config: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    multiplier = float(config["loss"]["ssim"]["multiplier"])
    _require(multiplier == 1.0, "registered SSIM calibration multiplier changed")
    calibration_config = {"loss": {"lambda_multiplier": multiplier}}
    value, record = structural._calibrate_structural_weight(
        model, pixels, rows, scale, power_reference, gain, weights, calibration_config,
    )
    _require(math.isfinite(value) and value > 0.0, "SSIM calibration is zero or nonfinite")
    return value, record


def _direction_and_negative_control(
    model: torch.nn.Module, pixels: torch.Tensor, rows: torch.Tensor, scale: torch.Tensor,
    power_reference: torch.Tensor, gain: torch.Tensor, weights: dict[str, float],
    lambda_ssim: float,
) -> dict[str, Any]:
    def full_objective() -> torch.Tensor:
        parts, structure, _ = structural._loss_terms(
            model, pixels, rows, scale, power_reference, gain, "treatment",
        )
        return structural._total_loss(parts, structure, weights, lambda_ssim)

    diagnostics = structural._direction_check(model, full_objective, "bogt_plus_ssim")
    wrong_sign_rejected: dict[str, bool] = {}
    for group, values in diagnostics.items():
        rejected = False
        try:
            structural._validate_direction_values(
                "wrong_sign_negative_control", group,
                values["autograd"], -values["central_difference"],
            )
        except RuntimeError:
            rejected = True
        _require(rejected, f"wrong-sign derivative negative control was not rejected for {group}")
        wrong_sign_rejected[group] = rejected
    return {"full_objective": diagnostics, "wrong_sign_negative_control_rejected": wrong_sign_rejected}


def _preflight_initial(
    seed: int, model: torch.nn.Module, optimizer: torch.optim.Adam,
    pixels: torch.Tensor, schedule: torch.Tensor, scale: torch.Tensor,
    power_reference: torch.Tensor, gain: torch.Tensor, weights: dict[str, float],
    config_path: Path, evaluators: tuple[torch.nn.Module, torch.nn.Module],
    run_dir: Path,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    target, _ = _update_targets(pixels, schedule[0], device)
    finite_difference = base._finite_difference_check(model, target, scale, power_reference, gain, weights)
    wrong_sign_rejected = False
    for group, values in finite_difference.items():
        analytic = float(values["autograd"])
        numerical = float(values["central_difference"])
        _require(analytic * numerical > 0.0, f"BOGT direction sign failed in {group}")
        rejected = False
        try:
            structural._validate_direction_values("BOGT_wrong_sign_negative_control", group,
                                                  analytic, -numerical)
        except RuntimeError:
            rejected = True
        _require(rejected, f"BOGT wrong-sign negative control was not rejected for {group}")
        wrong_sign_rejected = True
    _require(wrong_sign_rejected, "BOGT wrong-sign negative control was not effective")
    schedule_control = base._assert_row_negative_control(schedule[0])
    preflight_dir = run_dir / "engineering_preflight" / "bogt"
    _require(not preflight_dir.exists(), "BOGT preflight evidence already exists")
    evaluator_states_before = tuple(base._state_snapshot(evaluator) for evaluator in evaluators)
    saved_rng = _rng_state()
    recovery = _preflight_recovery(
        model=model, optimizer=optimizer, pixels=pixels, schedule=schedule,
        start_step=0, seed=seed, stage="bogt", scale=scale, power_reference=power_reference,
        gain=gain, weights=weights, lambda_ssim=None, config_path=config_path,
        directory=preflight_dir,
    )
    _set_rng_state(saved_rng)
    _require(base._assert_close(evaluator_states_before[0], base._state_snapshot(evaluators[0]), "E_D_initial_preflight") == 0.0
             and base._assert_close(evaluator_states_before[1], base._state_snapshot(evaluators[1]), "E_C_initial_preflight") == 0.0,
             "frozen evaluator changed during BOGT preflight")
    return {"seed": seed, "status": "READY", "bogt_direction": finite_difference,
            "wrong_sign_negative_control_rejected": True, "balanced_schedule_negative_control": schedule_control,
            "three_step_update_and_independent_recovery": recovery,
            "frozen_evaluators_unchanged": True, "science_updates": 0,
            "engineering_adam_calls": 5, "rollback_verified": True}


def _evaluate_dev(
    model: torch.nn.Module, evaluator_d: torch.nn.Module, evaluator_c: ThirdConfirmationEvaluator,
    pixels: torch.Tensor, labels: torch.Tensor, dev_rows: torch.Tensor, device: torch.device,
    gain: torch.Tensor, seed: int, step: int, output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows_out: list[dict[str, Any]] = []
    example_rows = torch.stack([dev_rows[labels[dev_rows] == label][:2] for label in range(10)]).reshape(-1)
    example_map: dict[int, dict[str, torch.Tensor]] = {}
    model.eval()
    evaluator_d.eval()
    evaluator_c.eval()
    with torch.inference_mode():
        for batch_rows in dev_rows.split(BATCH_SIZE):
            target_native = image_batch(pixels, batch_rows, "small", device)
            result = base._forward(model, target_native)
            gained = result["physical_w_per_pixel"].float() * gain.to(device=device, dtype=torch.float32)
            view = resize_view(gained)
            target28 = pixels.index_select(0, batch_rows.cpu()).unsqueeze(1).to(device=device, dtype=torch.float32) / 255.0
            error2 = (gained - target_native).square()
            foreground_mass = target_native.sum((1, 2, 3)).clamp_min(1e-12)
            background = 1.0 - target_native
            background_mass = background.sum((1, 2, 3)).clamp_min(1e-12)
            foreground_error = (target_native * error2).sum((1, 2, 3)) / foreground_mass
            background_error = (background * error2).sum((1, 2, 3)) / background_mass
            mse = (view - target28).square().mean((1, 2, 3)).clamp_min(1e-12)
            psnr = -10.0 * torch.log10(mse)
            ssim = ssim_batch(view, target28)
            logits_d = evaluator_d(view)
            logits_c = prior._ec_logits(evaluator_c, view)
            clipping = ((gained < 0.0) | (gained > 1.0)).float().mean((1, 2, 3))
            excess = gained.sub(1.0).clamp_min(0.0).sum((1, 2, 3)) / gained.double().sum((1, 2, 3)).clamp_min(1e-30)
            for index, row in enumerate(batch_rows.tolist()):
                label = int(labels[row])
                rows_out.append({
                    "model_seed": seed, "step": step, "row": int(row), "label": label,
                    "mse": float(mse[index]), "psnr_db": float(psnr[index]), "ssim": float(ssim[index]),
                    "ed_prediction": int(logits_d[index].argmax()), "ec_prediction": int(logits_c[index].argmax()),
                    "ed_correct": int(int(logits_d[index].argmax()) == label),
                    "ec_correct": int(int(logits_c[index].argmax()) == label),
                    "stroke_weighted_unclipped_mse": float(foreground_error[index]),
                    "background_weighted_unclipped_mse": float(background_error[index]),
                    "proxy_clipping_area_fraction": float(clipping[index]),
                    "proxy_clipped_excess_fraction": float(excess[index]),
                    "mean_native_camera_output": float(gained[index].mean()),
                    "peak_native_camera_output": float(gained[index].max()),
                })
                if row in example_rows.tolist():
                    example_map[int(row)] = {
                        "row": torch.tensor(int(row)), "label": torch.tensor(label),
                        "target28": target28[index].detach().cpu(), "view28": view[index].detach().cpu(),
                        "native200": gained[index].clamp(0.0, 1.0).detach().cpu(),
                    }
    _require(len(rows_out) == DEV_COUNT and len(example_map) == 20,
             f"seed {seed} step {step} DEV evaluation is incomplete")
    per_class = []
    for label in range(10):
        selected = [row for row in rows_out if row["label"] == label]
        _require(len(selected) == 500, f"DEV class {label} population changed")
        per_class.append({
            "label": label, "count": len(selected),
            "mse": float(np.mean([row["mse"] for row in selected])),
            "psnr_db": float(np.mean([row["psnr_db"] for row in selected])),
            "ssim": float(np.mean([row["ssim"] for row in selected])),
            "ed_recall": float(np.mean([row["ed_correct"] for row in selected])),
            "ec_recall": float(np.mean([row["ec_correct"] for row in selected])),
            "stroke_weighted_unclipped_mse": float(np.mean([row["stroke_weighted_unclipped_mse"] for row in selected])),
            "background_weighted_unclipped_mse": float(np.mean([row["background_weighted_unclipped_mse"] for row in selected])),
            "proxy_clipping_area_fraction": float(np.mean([row["proxy_clipping_area_fraction"] for row in selected])),
        })
    ed_confusion = np.zeros((10, 10), dtype=np.int64)
    ec_confusion = np.zeros((10, 10), dtype=np.int64)
    for row in rows_out:
        ed_confusion[row["label"], row["ed_prediction"]] += 1
        ec_confusion[row["label"], row["ec_prediction"]] += 1
    summary = {
        "seed": seed, "step": step, "population": len(rows_out),
        "psnr_db_mean": float(np.mean([row["psnr_db"] for row in rows_out])),
        "ssim_mean": float(np.mean([row["ssim"] for row in rows_out])),
        "mse_mean": float(np.mean([row["mse"] for row in rows_out])),
        "E_D_macro_recall": float(np.mean([item["ed_recall"] for item in per_class])),
        "E_C_macro_recall": float(np.mean([item["ec_recall"] for item in per_class])),
        "E_D_confusion_matrix": ed_confusion.tolist(), "E_C_confusion_matrix": ec_confusion.tolist(),
        "proxy_clipping_area_macro": float(np.mean([item["proxy_clipping_area_fraction"] for item in per_class])),
        "per_class": per_class,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"dev_metrics_step_{step:06d}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    examples = [example_map[int(row)] for row in example_rows.tolist()]
    atomic_torch_save(output_dir / f"dev_examples_step_{step:06d}.pt", {
        "task_id": TASK_ID, "seed": seed, "step": step,
        "rows": torch.stack([item["row"] for item in examples]),
        "labels": torch.stack([item["label"] for item in examples]),
        "targets28": torch.stack([item["target28"] for item in examples]),
        "views28": torch.stack([item["view28"] for item in examples]),
        "native200": torch.stack([item["native200"] for item in examples]),
    })
    return summary, rows_out


def _window_check(
    model: torch.nn.Module, pixels: torch.Tensor, labels: torch.Tensor, schedule: torch.Tensor,
    runtime: dict[str, Any], device: torch.device, step: int,
) -> dict[str, Any]:
    grid, diagnostics = base._window_check("small", pixels, labels, schedule, runtime, device, model=model)
    _require(grid == 2048 and diagnostics["complex_relative_error_max"] <= WINDOW_TOLERANCE,
             f"seed window check failed at step {step}")
    return {"step": step, "selected_grid": grid, **base._json_window_diagnostics(diagnostics)}


def _train_seed(
    *, seed: int, config: dict[str, Any], runtime: dict[str, Any], config_path: Path,
    output_dir: Path, pixels: torch.Tensor, labels: torch.Tensor, train_rows: torch.Tensor,
    dev_rows: torch.Tensor, evaluator_d: torch.nn.Module,
    evaluator_c: ThirdConfirmationEvaluator, resume_checkpoint: Path | None,
    resume_state: dict[str, Any] | None, preflight_only: bool = False,
) -> dict[str, Any]:
    seed_dir = output_dir / f"seed_{seed}"
    device = torch.device("cuda")
    schedule = _full_schedule(labels, train_rows, seed)
    evaluators_before = (base._state_snapshot(evaluator_d), base._state_snapshot(evaluator_c))
    start = 0
    initial_payload: dict[str, Any] | None = None
    if resume_checkpoint is None:
        _require(not seed_dir.exists(), f"refusing to overwrite existing seed directory: {seed_dir}")
        seed_dir.mkdir(parents=True, exist_ok=False)
        run_id = uuid.uuid4().hex
        model, optimizer = _initial_model(seed, runtime, device)
        selected_grid, window_start = base._window_check("small", pixels, labels, schedule, runtime, device, model=model)
        _require(selected_grid == 2048 and window_start["complex_relative_error_max"] <= WINDOW_TOLERANCE,
                 "step-zero production window qualification failed")
        first_native, _ = _update_targets(pixels, schedule[0], device)
        readout = base._small_producer_consumer_check(model, first_native[:1])
        _require(readout.get("status") == "PASSED", "step-zero OLS or camera readout check failed")
        scale, power_reference, gain, gain_fit = base._calibrate(model, pixels, train_rows, "small", device)
        norms = base._gradient_norms(model, lambda: base._loss_components(
            model, first_native, scale, power_reference, gain,
        )[0])
        weights = base._weights(norms)
        bogt_direction = base._finite_difference_check(model, first_native, scale, power_reference, gain, weights)
        preflight = _preflight_initial(
            seed, model, optimizer, pixels, schedule, scale, power_reference, gain, weights,
            config_path, (evaluator_d, evaluator_c), seed_dir,
        )
        _require(base._assert_close(evaluators_before[0], base._state_snapshot(evaluator_d), "E_D_initial_preflight") == 0.0
                 and base._assert_close(evaluators_before[1], base._state_snapshot(evaluator_c), "E_C_initial_preflight") == 0.0,
                 "frozen evaluator changed during initial preflight")
        atomic_write_json(seed_dir / "preflight.json", {
            "status": "READY", "seed": seed, "step_zero_window": base._json_window_diagnostics(window_start),
            "independent_readout": readout, "initial_camera_gain_fit": gain_fit,
            "initial_modal_scale": scale.tolist(), "initial_power_reference": power_reference.tolist(),
            "component_gradient_norms": norms, "frozen_weights": weights,
            "BOGT_direction": bogt_direction, "preflight": preflight,
            "official_test_reads": 0, "device": torch.cuda.get_device_name(device),
        })
        atomic_torch_save(seed_dir / "initial_references.pt", {
            "scale": scale.cpu(), "power_reference": power_reference.cpu(), "camera_gain": gain.cpu(),
            "weights": weights, "schedule": schedule,
        })
        history: list[dict[str, Any]] = []
        lambda_ssim: float | None = None
        elapsed_before = 0.0
        initial_payload = _checkpoint_payload(model, optimizer, seed, run_id, schedule, scale,
                                              power_reference, gain, weights, lambda_ssim, 0, history, 0.0)
        _save_checkpoint(seed_dir, initial_payload, 0, milestone=True)
    else:
        checkpoint, model, optimizer = _restore_checkpoint(
            resume_checkpoint, seed_dir, seed, config, runtime, schedule, device,
        )
        start = int(checkpoint["completed_global_step"])
        run_id = checkpoint["run_id"]
        scale, power_reference, gain = (checkpoint[name].to(device) for name in ("scale", "power_reference", "camera_gain"))
        weights = checkpoint["weights"]
        lambda_ssim = checkpoint.get("lambda_ssim")
        history = checkpoint.get("history")
        elapsed_before = float(checkpoint.get("elapsed_seconds", 0.0))
        _require(isinstance(history, list) and math.isfinite(elapsed_before) and elapsed_before >= 0.0,
                 "resume checkpoint history or elapsed time is invalid")
        if start == BOGT_UPDATES and lambda_ssim is None:
            record_path = seed_dir / "frozen_ssim_calibration.json"
            if record_path.is_file():
                record = json.loads(record_path.read_text(encoding="utf-8"))
                lambda_ssim = float(record["lambda_ssim"])

    run_static_state = _phase_nontrainable_snapshot(model)
    phase_ids = {id(parameter) for parameter in base._phase_groups(model)}
    _require(all(parameter.requires_grad == (id(parameter) in phase_ids) for parameter in model.parameters()),
             "only the four declared phase planes may be trainable")
    if resume_state is not None:
        preflight_path = seed_dir / "preflight.json"
        _require(preflight_path.is_file() and json.loads(preflight_path.read_text(encoding="utf-8")).get("status") == "READY",
                 "resume has no completed initial READY preflight")
    if start == 0 and resume_checkpoint is not None:
        _require((seed_dir / "engineering_preflight" / "bogt" / "recovery_output.pt").is_file(),
                 "step-zero resume checkpoint has no verified recovery evidence")
    if start > BOGT_UPDATES:
        structural_receipt = seed_dir / "structural_preflight.json"
        _require(structural_receipt.is_file() and lambda_ssim is not None,
                 "post-30,000 resume is missing its frozen SSIM calibration or stage preflight")
        receipt = json.loads(structural_receipt.read_text(encoding="utf-8"))
        _require(receipt.get("status") == "READY" and receipt.get("seed") == seed
                 and math.isclose(float(receipt.get("calibration", {}).get("lambda_ssim", math.nan)),
                                  float(lambda_ssim), rel_tol=0.0, abs_tol=0.0),
                 "saved SSIM stage preflight is incomplete or belongs to another seed")
    if preflight_only:
        _require(start == 0 and initial_payload is not None,
                 "preflight-only mode requires a new, zero-update trajectory")
        return {"summary": {"task_id": TASK_ID, "seed": seed, "run_id": run_id,
                             "measurement_complete": False, "global_step": 0,
                             "status": "PREFLIGHT_READY"},
                "checkpoint": initial_payload, "model": model, "scale": scale,
                "gain": gain, "power_reference": power_reference, "weights": weights,
                "schedule": schedule, "reconstruction": [], "window_checks": [],
                "run_dir": seed_dir, "preflight_only": True}

    window_records: list[dict[str, Any]] = []
    reconstruction_records: list[dict[str, Any]] = []
    window_path = seed_dir / "window_checks.json"
    if window_path.is_file():
        saved_windows = json.loads(window_path.read_text(encoding="utf-8"))
        _require(isinstance(saved_windows, list)
                 and all(item.get("step") in EVALUATION_STEPS and item.get("step") <= start
                         for item in saved_windows),
                 f"seed {seed} saved propagation-window checks are malformed")
        window_records = saved_windows
    for step in EVALUATION_STEPS:
        metric_path = seed_dir / f"dev_metrics_step_{step:06d}.json"
        if metric_path.is_file():
            saved = json.loads(metric_path.read_text(encoding="utf-8"))
            _require(saved.get("task_id") == TASK_ID and saved.get("seed") == seed
                     and saved.get("step") == step and saved.get("population") == DEV_COUNT
                     and step <= start,
                     f"saved DEV block at step {step} is incomplete or misidentified")
            csv_path = seed_dir / f"dev_metrics_step_{step:06d}.csv"
            example_path = seed_dir / f"dev_examples_step_{step:06d}.pt"
            _require(csv_path.is_file() and example_path.is_file(),
                     f"saved DEV block at step {step} is missing a required consumer input")
            with csv_path.open("r", encoding="utf-8", newline="") as stream:
                csv_rows = list(csv.DictReader(stream))
            example_payload = torch.load(example_path, map_location="cpu", weights_only=False)
            _require(len(csv_rows) == DEV_COUNT and example_payload.get("task_id") == TASK_ID
                     and example_payload.get("seed") == seed and example_payload.get("step") == step
                     and tuple(example_payload.get("rows", torch.empty(0)).shape) == (20,)
                     and tuple(example_payload.get("views28", torch.empty(0)).shape) == (20, 1, 28, 28),
                     f"saved DEV block at step {step} failed its result consumer")
            reconstruction_records.append(saved)
    for missing_step in EVALUATION_STEPS:
        metric_path = seed_dir / f"dev_metrics_step_{missing_step:06d}.json"
        if missing_step <= start and not metric_path.is_file():
            checkpoint_path = seed_dir / f"checkpoint_step_{missing_step:06d}.pt"
            _require(checkpoint_path.is_file(),
                     f"cannot recover missing DEV block at step {missing_step} without its named checkpoint")
            saved_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            _require(saved_checkpoint.get("task_id") == TASK_ID and saved_checkpoint.get("seed") == seed
                     and saved_checkpoint.get("completed_global_step") == missing_step,
                     f"missing DEV block checkpoint at step {missing_step} is misidentified")
            evaluation_model = base._model("small", 2048, device, runtime)
            evaluation_model.load_state_dict(saved_checkpoint["model"], strict=True)
            evaluation_gain = saved_checkpoint["camera_gain"].to(device)
            if not any(int(item.get("step", -1)) == missing_step for item in window_records):
                window_records.append(_window_check(evaluation_model, pixels, labels, schedule,
                                                    runtime, device, missing_step))
                atomic_write_json(window_path, window_records)
            recon, _ = _evaluate_dev(evaluation_model, evaluator_d, evaluator_c, pixels, labels,
                                    dev_rows, device, evaluation_gain, seed, missing_step, seed_dir)
            reconstruction_records.append(recon)
            atomic_write_json(metric_path, {"task_id": TASK_ID, **recon})
            atomic_write_json(seed_dir / "dev_metrics_index.json", {
                "task_id": TASK_ID, "seed": seed,
                "blocks": sorted(reconstruction_records, key=lambda item: int(item["step"])),
            })
            del evaluation_model, saved_checkpoint
            torch.cuda.empty_cache()
    density_branch = f"seed_{seed}"
    base._record_gaussian_plane_density_probe(
        model, pixels, labels, dev_rows, output_dir, branch=density_branch,
        global_step=start, grid_size=2048,
    )
    started = time.perf_counter()
    for step in range(start + 1, UPDATES + 1):
        if step == BOGT_UPDATES + 1:
            lambda_path = seed_dir / "frozen_ssim_calibration.json"
            if lambda_ssim is None:
                lambda_ssim, calibration = _calibrate_structural(
                    model, pixels, schedule[BOGT_UPDATES], scale, power_reference, gain, weights, config,
                )
                atomic_write_json(lambda_path, {
                    "task_id": TASK_ID, "seed": seed, "calibrated_at_global_step": BOGT_UPDATES,
                    "sample_rows": schedule[BOGT_UPDATES].tolist(), "lambda_ssim": lambda_ssim,
                    "frozen": True, **calibration,
                })
            else:
                record = json.loads(lambda_path.read_text(encoding="utf-8"))
                _require(record.get("task_id") == TASK_ID and record.get("seed") == seed
                         and record.get("calibrated_at_global_step") == BOGT_UPDATES
                         and record.get("sample_rows") == schedule[BOGT_UPDATES].tolist()
                         and math.isclose(float(record["lambda_ssim"]), lambda_ssim, rel_tol=0.0, abs_tol=0.0),
                         "saved frozen SSIM calibration does not match this 30,000-step branch")
            _require(lambda_ssim is not None and math.isfinite(lambda_ssim) and lambda_ssim > 0.0,
                     "SSIM coefficient is zero or nonfinite")
            direction = _direction_and_negative_control(
                model, pixels, schedule[BOGT_UPDATES], scale, power_reference, gain, weights, lambda_ssim,
            )
            stage2_dir = seed_dir / "engineering_preflight" / "bogt_ssim"
            receipt_path = seed_dir / "structural_preflight.json"
            if receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                _require(receipt.get("status") == "READY" and receipt.get("seed") == seed
                         and receipt.get("calibration", {}).get("lambda_ssim") == lambda_ssim,
                         "saved structural-stage preflight belongs to another branch or coefficient")
            else:
                if stage2_dir.exists():
                    stage2_dir = stage2_dir.with_name(f"bogt_ssim_retry_{uuid.uuid4().hex[:8]}")
                recovery = _preflight_recovery(
                    model=model, optimizer=optimizer, pixels=pixels, schedule=schedule,
                    start_step=BOGT_UPDATES, seed=seed, stage="bogt_ssim", scale=scale,
                    power_reference=power_reference, gain=gain, weights=weights, lambda_ssim=lambda_ssim,
                    config_path=config_path, directory=stage2_dir,
                )
                atomic_write_json(receipt_path, {
                    "status": "READY", "seed": seed, "calibration": json.loads(lambda_path.read_text(encoding="utf-8")),
                    "direction": direction, "three_step_update_and_independent_recovery": recovery,
                    "frozen_evaluators_unchanged": True,
                })
            _require(base._assert_close(evaluators_before[0], base._state_snapshot(evaluator_d), "E_D_structural_preflight") == 0.0
                     and base._assert_close(evaluators_before[1], base._state_snapshot(evaluator_c), "E_C_structural_preflight") == 0.0,
                     "frozen evaluator changed during structural preflight")
        row = schedule[step - 1]
        native, target28 = _update_targets(pixels, row, device)
        current_lambda = None if step <= BOGT_UPDATES else lambda_ssim
        loss = _train_one(model, optimizer, native, target28, scale, power_reference, gain, weights, current_lambda)
        _require(_phase_optimizer_update_count(optimizer, model) == step,
                 f"seed {seed} Adam step count diverged at update {step}")
        if step % PROGRESS_INTERVAL == 0:
            history.append({"step": step, "stage": "BOGT" if step <= BOGT_UPDATES else "BOGT+SSIM", **loss})
            atomic_write_json(seed_dir / "training_history_live.json", history)
            progress_path = output_dir / "progress_state.json"
            prior_progress = json.loads(progress_path.read_text(encoding="utf-8"))
            seed_checkpoints = dict(prior_progress.get("seed_checkpoints", {}))
            checkpoint_steps = dict(prior_progress.get("checkpoint_global_steps", {}))
            seed_checkpoints[str(seed)] = str(seed_dir / "checkpoint_latest.pt")
            checkpoint_steps[str(seed)] = (step // CHECKPOINT_INTERVAL) * CHECKPOINT_INTERVAL
            _merge_progress(progress_path, output_dir, config, {
                "status": "RUNNING", "current_seed": seed, "current_step": step,
                "checkpoint": str(seed_dir / "checkpoint_latest.pt"),
                "checkpoint_global_step": checkpoint_steps[str(seed)],
                "seed_checkpoints": seed_checkpoints,
                "checkpoint_global_steps": checkpoint_steps,
                "elapsed_seconds_this_process": time.perf_counter() - started,
            })
        if step % CHECKPOINT_INTERVAL == 0:
            payload = _checkpoint_payload(model, optimizer, seed, run_id, schedule, scale,
                                          power_reference, gain, weights, lambda_ssim, step,
                                          history, elapsed_before + time.perf_counter() - started)
            _save_checkpoint(seed_dir, payload, step, milestone=step in {5_000, 10_000, 20_000, 30_000, 35_000, 40_000, 45_000})
            base._record_gaussian_plane_density_probe(
                model, pixels, labels, dev_rows, output_dir, branch=density_branch,
                global_step=step, grid_size=2048,
            )
            progress_path = output_dir / "progress_state.json"
            prior_progress = json.loads(progress_path.read_text(encoding="utf-8"))
            seed_checkpoints = dict(prior_progress.get("seed_checkpoints", {}))
            checkpoint_steps = dict(prior_progress.get("checkpoint_global_steps", {}))
            seed_checkpoints[str(seed)] = str(seed_dir / "checkpoint_latest.pt")
            checkpoint_steps[str(seed)] = step
            _merge_progress(progress_path, output_dir, config, {
                "status": "RUNNING", "current_seed": seed, "current_step": step,
                "checkpoint": str(seed_dir / "checkpoint_latest.pt"),
                "checkpoint_global_step": step,
                "seed_checkpoints": seed_checkpoints,
                "checkpoint_global_steps": checkpoint_steps,
                "elapsed_seconds_this_process": time.perf_counter() - started,
            })
        if step in EVALUATION_STEPS:
            if not any(int(item.get("step", -1)) == step for item in window_records):
                window = _window_check(model, pixels, labels, schedule, runtime, device, step)
                window_records.append(window)
                atomic_write_json(seed_dir / "window_checks.json", window_records)
            recon, _ = _evaluate_dev(model, evaluator_d, evaluator_c, pixels, labels, dev_rows,
                                     device, gain, seed, step, seed_dir)
            reconstruction_records = [item for item in reconstruction_records if int(item["step"]) != step]
            reconstruction_records.append(recon)
            atomic_write_json(seed_dir / f"dev_metrics_step_{step:06d}.json", {
                "task_id": TASK_ID, **recon,
            })
            atomic_write_json(seed_dir / "dev_metrics_index.json", {
                "task_id": TASK_ID, "seed": seed,
                "blocks": sorted(reconstruction_records, key=lambda item: int(item["step"])),
            })
        if step in {0, 5_000, 10_000, 20_000, 30_000, 35_000, 40_000, 45_000} and step % CHECKPOINT_INTERVAL:
            _require(False, "registered milestone is not on the checkpoint interval")
    _require(base._assert_close(run_static_state, _phase_nontrainable_snapshot(model), "frozen_optics_after_training") == 0.0,
             "non-phase optical parameters or buffers changed during training")
    _require(base._assert_close(evaluators_before[0], base._state_snapshot(evaluator_d), "E_D_after_training") == 0.0
             and base._assert_close(evaluators_before[1], base._state_snapshot(evaluator_c), "E_C_after_training") == 0.0,
             "confirmation evaluator changed during training")
    _require(_phase_optimizer_update_count(optimizer, model) == UPDATES,
             f"seed {seed} did not complete exactly 45,000 scientific updates")
    final_payload = torch.load(seed_dir / "checkpoint_step_045000.pt", map_location="cpu", weights_only=False)
    _require(final_payload.get("lambda_ssim") == lambda_ssim and final_payload.get("completed_global_step") == UPDATES,
             "45,000-step endpoint did not preserve its frozen structural coefficient")
    density_steps = sorted({start} | {
        global_step for global_step in range(start + 1, UPDATES + 1)
        if global_step % CHECKPOINT_INTERVAL == 0
    })
    plane_density_logging = base._validate_gaussian_plane_density_coverage(
        output_dir, density_branch, density_steps, 2048,
    )
    summary = {
        "task_id": TASK_ID, "seed": seed, "run_id": run_id, "measurement_complete": True,
        "global_step": UPDATES, "bogt_updates": BOGT_UPDATES, "structural_updates": STRUCTURAL_UPDATES,
        "lambda_ssim": lambda_ssim, "elapsed_seconds_this_process": time.perf_counter() - started,
        "training_history": len(history), "evaluation_steps": [int(item["step"]) for item in reconstruction_records],
        "checkpoint": "checkpoint_step_045000.pt", "status": "TRAINING_COMPLETE",
        "plane_power_density_logging": plane_density_logging,
    }
    atomic_write_json(seed_dir / "training_summary.json", summary)
    return {"summary": summary, "checkpoint": final_payload, "model": model,
            "scale": scale, "gain": gain, "power_reference": power_reference,
            "weights": weights, "schedule": schedule, "reconstruction": reconstruction_records,
            "window_checks": window_records, "run_dir": seed_dir}


def _encode_features(
    model: torch.nn.Module, pixels: torch.Tensor, train_rows: torch.Tensor,
    labels: torch.Tensor, scale: torch.Tensor, device: torch.device,
    output_path: Path, seed: int,
) -> tuple[torch.Tensor, float]:
    started = time.perf_counter()
    _require(train_rows.dtype == torch.long and tuple(train_rows.shape) == (TRAIN_COUNT,)
             and labels.dtype == torch.long and tuple(labels.shape) == (TRAIN_COUNT,),
             "TRAIN feature rows and their aligned labels must be complete")
    features = torch.empty((TRAIN_COUNT, 200), dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, TRAIN_COUNT, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, TRAIN_COUNT)
            rows = train_rows[start:stop]
            output = base._forward(model, image_batch(pixels, rows, "small", device))
            coefficients = output["ols_coefficients"].to(torch.complex128)
            gauged = base._gauge_and_scale(coefficients, scale.to(device))
            features[start:stop] = torch.view_as_real(gauged.cpu()).reshape(stop - start, 200)
            if stop % 5000 == 0 or stop == TRAIN_COUNT:
                torch.cuda.synchronize(device)
                print(json.dumps({"phase": "TRAIN_FEATURE_ENCODING", "seed": seed,
                                  "rows": stop, "total_rows": TRAIN_COUNT,
                                  "elapsed_seconds": time.perf_counter() - started}), flush=True)
    _require(torch.isfinite(features).all().item(), "new TRAIN coefficient features are nonfinite")
    atomic_torch_save(output_path, {
        "schema": "lightgen_small_replication_train_features.v1", "task_id": TASK_ID,
        "seed": seed, "source_checkpoint_step": UPDATES,
        "rows": train_rows.clone(), "labels": labels.clone(),
        "features_200": features, "encoder_forward_images": TRAIN_COUNT,
        "elapsed_seconds": time.perf_counter() - started,
    })
    return features, time.perf_counter() - started


def _fit_new_priors(features: torch.Tensor, labels: torch.Tensor, seed: int, run_dir: Path) -> dict[str, Any]:
    g1: dict[int, dict[str, Any]] = {}
    g4: dict[int, dict[str, Any]] = {}
    assignments = torch.full((labels.numel(),), -1, dtype=torch.long)
    for label in range(10):
        selected = labels == label
        class_features = features[selected]
        _require(class_features.shape == (TRAIN_LABEL_COUNTS[label], 200),
                 f"seed {seed} class {label} prior fitting population changed")
        fit1 = prior._fit_single_class(class_features)
        fit4 = prior._fit_mixture_class(class_features, 4, prior.KMEANS_SEED_BASE + label,
                                        fit1["ridge"], prior.KMEANS_MAX_ITERATIONS)
        assignments[selected] = fit4.pop("assignments")
        fit4["train_rows"] = int(selected.sum())
        fit4["seed"] = prior.KMEANS_SEED_BASE + label
        g1[label] = fit1
        g4[label] = fit4
        _require(bool((assignments[selected] >= 0).all()), f"seed {seed} class {label} G4 assignment is incomplete")
    _require(bool((assignments >= 0).all()), f"seed {seed} G4 assignments omit TRAIN rows")
    path = run_dir / "fitted_priors_G1_G4.pt"
    atomic_torch_save(path, {
        "schema": "lightgen_small_conditional_mixture_priors.v1", "task_id": TASK_ID,
        "seed": seed, "G1": g1, "G4": g4,
        "G4_train_assignments_in_cache_order": assignments,
    })
    return {"G1": g1, "G4": g4, "assignments": assignments}


def _sample_g1_class(
    fit: dict[str, Any], scale: torch.Tensor, label: int, stream: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    generator = torch.Generator(device="cpu").manual_seed(G1_NORMAL_SEED_BASE + 100 * stream + label)
    coefficients: list[torch.Tensor] = []
    proposals: list[torch.Tensor] = []
    evaluated = 0
    retained = 0
    rejected = 0
    while retained < SAMPLE_COUNT:
        count = min(256, MAX_PROPOSALS - evaluated)
        _require(count > 0, f"G1 class {label} stream {stream} exceeded its fixed proposal cap")
        noise = torch.randn((count, 200), generator=generator, dtype=torch.float64)
        draws = fit["mean"] + noise @ fit["cholesky"].T
        normalized = torch.view_as_complex(draws.reshape(count, 100, 2).contiguous())
        values = normalized * scale.detach().cpu().double().sqrt().unsqueeze(0)
        valid = values.abs().square().sum(1) <= base.MAX_MODAL_POWER_W
        positions = torch.nonzero(valid, as_tuple=False).flatten()
        take = positions[: SAMPLE_COUNT - retained]
        if take.numel():
            coefficients.append(values[take].to(torch.complex64))
            proposals.append(take + evaluated)
            retained += int(take.numel())
        rejected += int((~valid).sum())
        evaluated += count
    values_out = torch.cat(coefficients)
    proposal_out = torch.cat(proposals).long()
    _require(tuple(values_out.shape) == (SAMPLE_COUNT, 100)
             and torch.unique(proposal_out).numel() == SAMPLE_COUNT
             and bool((values_out.abs().square().sum(1) <= base.MAX_MODAL_POWER_W).all()),
             "G1 samples failed power bound or proposal identity validation")
    return values_out, torch.zeros(SAMPLE_COUNT, dtype=torch.long), proposal_out, {
        "label": label, "stream": stream, "normal_seed": G1_NORMAL_SEED_BASE + 100 * stream + label,
        "proposal_limit": MAX_PROPOSALS, "proposals_evaluated": evaluated,
        "retained_samples": retained, "power_rejected_candidates": rejected,
        "rejection_fraction": rejected / evaluated, "power_bound_w": base.MAX_MODAL_POWER_W,
    }


def _sample_population(
    method: str, priors: dict[str, Any], scale: torch.Tensor, stream: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    values: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    components: list[torch.Tensor] = []
    proposals: list[torch.Tensor] = []
    accounting: dict[str, Any] = {}
    for label in range(10):
        if method == "G1":
            sample, comp, ids, record = _sample_g1_class(priors["G1"][label], scale, label, stream)
        elif method == "G4":
            old_normal, old_component = prior.SAMPLE_BASE_SEED, prior.COMPONENT_BASE_SEED
            try:
                prior.SAMPLE_BASE_SEED = G4_NORMAL_SEED_BASE + 100 * stream
                prior.COMPONENT_BASE_SEED = G4_COMPONENT_SEED_BASE + 100 * stream
                sample, comp, ids, record = prior._sample_mixture_class(
                    priors["G4"][label], scale, label, SAMPLE_COUNT, MAX_PROPOSALS,
                )
            finally:
                prior.SAMPLE_BASE_SEED, prior.COMPONENT_BASE_SEED = old_normal, old_component
        else:
            raise ValueError(f"unsupported registered prior: {method}")
        values.append(sample)
        labels.append(torch.full((SAMPLE_COUNT,), label, dtype=torch.long))
        components.append(comp)
        proposals.append(ids)
        accounting[str(label)] = record
    coefficients = torch.cat(values)
    requested = torch.cat(labels)
    component_ids = torch.cat(components)
    proposal_ids = torch.cat(proposals)
    _require(coefficients.shape == (10_000, 100)
             and torch.equal(torch.bincount(requested, minlength=10), torch.full((10,), SAMPLE_COUNT))
             and all(torch.unique(proposal_ids[requested == label]).numel() == SAMPLE_COUNT for label in range(10)),
             f"{method} stream {stream} population is incomplete or duplicated")
    return coefficients, requested, component_ids, proposal_ids, accounting


def _cohort_rows(
    model_name: str, stream: int, method: str, cohort: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = cohort["requested_labels"].long()
    for index, label_tensor in enumerate(labels):
        label = int(label_tensor)
        rows.append({
            "model": model_name, "stream": stream, "method": method,
            "requested_label": label, "component_id": int(cohort["component_ids"][index]),
            "proposal_id": int(cohort["proposal_ids"][index]),
            "E_D_prediction": int(cohort["ed_predictions"][index]),
            "E_C_prediction": int(cohort["ec_predictions"][index]),
            "E_D_correct": int(int(cohort["ed_predictions"][index]) == label),
            "E_C_correct": int(int(cohort["ec_predictions"][index]) == label),
            "E_D_requested_margin": float(cohort["ed_requested_margin"][index]),
            "E_C_requested_margin": float(cohort["ec_requested_margin"][index]),
            "E_D_top_margin": float(cohort["ed_top_margin"][index]),
            "E_C_top_margin": float(cohort["ec_top_margin"][index]),
            "capture_power_w": float(cohort["capture_power_w"][index]),
            "raw_mean_w_per_pixel": float(cohort["raw_mean_w_per_pixel"][index]),
            "raw_peak_w_per_pixel": float(cohort["raw_peak_w_per_pixel"][index]),
            "view_mean_intensity": float(cohort["view_mean_intensity"][index]),
            "proxy_clipping_area_fraction": float(cohort["proxy_clipping_area_fraction"][index]),
            "proxy_clipped_excess_fraction": float(cohort["proxy_clipped_excess_fraction"][index]),
        })
    return rows


def _cohort_summary(rows: list[dict[str, Any]], samples_per_class: int = SAMPLE_COUNT) -> dict[str, Any]:
    _require(len(rows) == 10 * samples_per_class, "generated metric consumer received an incomplete population")
    per_class = []
    for label in range(10):
        selected = [row for row in rows if row["requested_label"] == label]
        _require(len(selected) == samples_per_class, f"generated class {label} metric population is incomplete")
        per_class.append({
            "label": label, "count": len(selected),
            "E_D_recall": float(np.mean([row["E_D_correct"] for row in selected])),
            "E_C_recall": float(np.mean([row["E_C_correct"] for row in selected])),
            "proxy_clipping_area_fraction": float(np.mean([row["proxy_clipping_area_fraction"] for row in selected])),
            "proxy_clipped_excess_fraction": float(np.mean([row["proxy_clipped_excess_fraction"] for row in selected])),
            "mean_capture_power_w": float(np.mean([row["capture_power_w"] for row in selected])),
            "mean_view_intensity": float(np.mean([row["view_mean_intensity"] for row in selected])),
            "generated_to_dev_tv_ratio": None,
        })
    return {
        "population": len(rows), "samples_per_class": samples_per_class,
        "E_D_macro_recall": float(np.mean([item["E_D_recall"] for item in per_class])),
        "E_D_recall_by_class": [item["E_D_recall"] for item in per_class],
        "E_C_macro_recall": float(np.mean([item["E_C_recall"] for item in per_class])),
        "E_C_recall_by_class": [item["E_C_recall"] for item in per_class],
        "macro_proxy_clipping_area_fraction": float(np.mean([item["proxy_clipping_area_fraction"] for item in per_class])),
        "macro_proxy_clipped_excess_fraction": float(np.mean([item["proxy_clipped_excess_fraction"] for item in per_class])),
        "mean_capture_power_w": float(np.mean([item["mean_capture_power_w"] for item in per_class])),
        "mean_view_intensity": float(np.mean([item["mean_view_intensity"] for item in per_class])),
        "per_class": per_class,
    }


def _global_phase_check(model: torch.nn.Module, coefficients: torch.Tensor, device: torch.device) -> dict[str, float]:
    _require(coefficients.shape == (10, 100), "global-phase control requires ten saved coefficient rows")
    phase = torch.polar(torch.ones((), device=device), torch.tensor(0.731, device=device))
    model.eval()
    with torch.inference_mode():
        original = model.forward_from_ols(coefficients.to(device))["physical_w_per_pixel"].float().cpu()
        rotated = model.forward_from_ols((coefficients * phase.cpu()).to(device))["physical_w_per_pixel"].float().cpu()
    difference = (original - rotated).abs()
    relative = difference / original.abs().clamp_min(1e-12)
    _require(float(relative.max()) <= 1e-4, "global phase rotation changed the production camera output")
    return {"rotation_rad": 0.731, "max_absolute_error_w_per_pixel": float(difference.max()),
            "max_relative_error": float(relative.max()), "passed": True}


def _generated_window_check(
    model: torch.nn.Module, coefficients: torch.Tensor, runtime: dict[str, Any], device: torch.device,
) -> dict[str, Any]:
    """Compare the production generated-output path with the fixed 4096 reference grid."""
    _require(tuple(coefficients.shape) == (10, 100) and torch.is_complex(coefficients)
             and bool(torch.isfinite(coefficients.real).all() and torch.isfinite(coefficients.imag).all()),
             "generated window check requires the first finite sample for every class")
    reference = base._model("small", 4096, device, runtime)
    reference.load_state_dict(base._state_snapshot(model), strict=True)
    model.eval()
    reference.eval()
    with torch.inference_mode():
        production = model.forward_from_ols(coefficients.to(device))["physical_w_per_pixel"].double()
        reference_output = reference.forward_from_ols(coefficients.to(device))["physical_w_per_pixel"].double()
    _require(production.shape == reference_output.shape == (10, 1, 200, 200)
             and torch.isfinite(production).all().item() and torch.isfinite(reference_output).all().item(),
             "generated production/reference camera outputs are malformed")
    relative_l2 = torch.linalg.vector_norm(production - reference_output) / torch.linalg.vector_norm(
        reference_output
    ).clamp_min(1e-30)
    prod_power = production.sum((1, 2, 3))
    ref_power = reference_output.sum((1, 2, 3))
    per_sample_power_error = ((prod_power - ref_power).abs() / ref_power.abs().clamp_min(1e-30))
    maximum_power_error = float(per_sample_power_error.max())
    maximum_relative_l2 = float(relative_l2)
    _require(maximum_relative_l2 <= WINDOW_TOLERANCE and maximum_power_error <= WINDOW_TOLERANCE,
             "generated-output 2048/4096 reference check failed")
    return {"sample_count": 10, "requested_labels": list(range(10)),
            "production_grid": 2048, "reference_grid": 4096,
            "camera_w_per_pixel_relative_l2_error": maximum_relative_l2,
            "camera_total_power_relative_error_max": maximum_power_error,
            "passed": True}


def _normalized_mutual_information(assignments: torch.Tensor, pivots: torch.Tensor) -> float:
    a = assignments.detach().cpu().numpy().astype(np.int64)
    b = pivots.detach().cpu().numpy().astype(np.int64)
    _require(a.shape == b.shape and a.size > 0, "cluster/pivot association arrays are misaligned")
    table = np.zeros((max(a.max(), 0) + 1, 100), dtype=np.int64)
    np.add.at(table, (a, b), 1)
    pxy = table / table.sum()
    px = pxy.sum(1)
    py = pxy.sum(0)
    nz = pxy > 0
    expected = px[:, None] * py[None, :]
    mi = float(np.sum(pxy[nz] * np.log(pxy[nz] / expected[nz])))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
    return 0.0 if hx + hy == 0 else 2.0 * mi / (hx + hy)


def _cluster_pivot_summary(
    features: torch.Tensor, labels: torch.Tensor, assignments: torch.Tensor,
) -> dict[str, Any]:
    complex_features = torch.view_as_complex(features.reshape(-1, 100, 2).contiguous())
    pivots = complex_features.abs().argmax(1)
    by_class = []
    for label in range(10):
        mask = labels == label
        count = int(mask.sum())
        by_class.append({"label": label, "train_rows": count,
                         "cluster_pivot_nmi": _normalized_mutual_information(assignments[mask], pivots[mask])})
    _require(sum(item["train_rows"] for item in by_class) == features.shape[0],
             "cluster/pivot summary lost TRAIN rows")
    return {"interpretation": "descriptive association only; global phase gauge and pivot chart can influence grouping",
            "per_class": by_class}


def _bootstrap_multistream(
    rows_by_method_stream: dict[str, dict[int, list[dict[str, Any]]]], seed: int,
) -> dict[str, Any]:
    metrics = ("E_D_correct", "E_C_correct", "proxy_clipping_area_fraction",
               "proxy_clipped_excess_fraction", "capture_power_w", "view_mean_intensity",
               "raw_peak_w_per_pixel", "E_D_requested_margin", "E_C_requested_margin")
    methods = ("G1", "G4")
    cell_values = np.empty((2, 3, 10, SAMPLE_COUNT, len(metrics)), dtype=np.float64)
    for method_index, method in enumerate(methods):
        for stream in range(3):
            records = rows_by_method_stream[method][stream]
            for label in range(10):
                selected = [row for row in records if row["requested_label"] == label]
                _require(len(selected) == SAMPLE_COUNT, f"bootstrap {method}/r{stream}/class{label} is incomplete")
                for metric_index, metric in enumerate(metrics):
                    values = np.asarray([row[metric] for row in selected], dtype=np.float64)
                    _require(np.isfinite(values).all(),
                             f"bootstrap {method}/r{stream}/class{label}/{metric} is nonfinite")
                    cell_values[method_index, stream, label, :, metric_index] = values
    rng = np.random.default_rng(seed)
    draws = np.empty((BOOTSTRAP_RESAMPLES, len(metrics)), dtype=np.float64)
    class_metric_indices = (0, 1, 2)
    class_draws = np.empty((BOOTSTRAP_RESAMPLES, 10, len(class_metric_indices)), dtype=np.float64)
    observed_by_method = cell_values.mean(axis=3)
    observed = observed_by_method[1].mean(axis=(0, 1)) - observed_by_method[0].mean(axis=(0, 1))
    observed_class = observed_by_method[1].mean(axis=0) - observed_by_method[0].mean(axis=0)
    for repeat in range(BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, SAMPLE_COUNT, size=(2, 3, 10, SAMPLE_COUNT), dtype=np.int32)
        broadcast_indices = np.broadcast_to(indices[..., None], cell_values.shape)
        resampled = np.take_along_axis(cell_values, broadcast_indices, axis=3).mean(axis=3)
        method_means = resampled.mean(axis=(1, 2))
        draws[repeat] = method_means[1] - method_means[0]
        class_means = resampled.mean(axis=1)
        class_draws[repeat] = class_means[1][:, class_metric_indices] - class_means[0][:, class_metric_indices]
    differences = {}
    for metric_index, key in enumerate(metrics):
        differences[key] = {"observed_difference": float(observed[metric_index]),
                            "ci_95": [float(x) for x in np.quantile(draws[:, metric_index], [0.025, 0.975])]}
    per_class = {}
    for class_metric_index, key in enumerate(("E_D_correct", "E_C_correct", "proxy_clipping_area_fraction")):
        per_class[key] = [{"label": label,
                           "observed_difference": float(observed_class[label, class_metric_index]),
                           "ci_95": [float(x) for x in np.quantile(
                               class_draws[:, label, class_metric_index], [0.025, 0.975]
                           )]}
                          for label in range(10)]
    return {"seed": seed, "resamples": BOOTSTRAP_RESAMPLES,
            "resampling": "independent within each method, stream and requested class; same draw indices shared by all metrics",
            "stream_weighting": "equal across three streams; classes equal-weighted",
            "differences": differences, "per_class_differences": per_class}


def _shape_ratios(
    method_views: dict[str, dict[int, list[torch.Tensor]]],
    dev_pixels: torch.Tensor, dev_labels: torch.Tensor,
    dev_tv_reference: list[float],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for method in ("G1", "G4"):
        by_class = []
        for label in range(10):
            generated = torch.cat(method_views[method][label], dim=0)
            real = dev_pixels[dev_labels == label]
            _require(generated.shape == (3_000, 1, 28, 28) and real.shape == (500, 1, 28, 28),
                     f"{method} shape population is incomplete for class {label}")
            gen_mean = float(base._pair_distances(generated, SHAPE_PAIR_SEED_BASE + label).mean())
            ratio = gen_mean / max(dev_tv_reference[label], 1e-12)
            by_class.append({"label": label, "generated_tv_mean": gen_mean,
                             "dev_tv_reference": dev_tv_reference[label], "ratio": ratio})
        output[method] = {"per_class": by_class, "minimum_ratio": min(item["ratio"] for item in by_class)}
    return output


def _validate_c_prior_cache(
    cache: dict[str, Any], priors: dict[str, Any], labels: torch.Tensor,
) -> dict[str, Any]:
    features = cache["features_200"].double()
    assignments = priors["G4_train_assignments_in_cache_order"].long()
    _require(tuple(assignments.shape) == (TRAIN_COUNT,) and bool((assignments >= 0).all()),
             "source C G4 training assignments are incomplete")
    checks = []
    for label in range(10):
        mask = labels == label
        fit1 = prior._fit_single_class(features[mask])
        stored1 = priors["G1"][label]
        for key in ("mean", "covariance", "ridge", "cholesky"):
            expected = torch.as_tensor(stored1[key]).double()
            _require(torch.allclose(fit1[key], expected, rtol=REPLAY_RTOL, atol=REPLAY_ATOL),
                     f"source C G1 refit differs for class {label} field {key}")
        fit4 = prior._fit_mixture_class(features[mask], 4, prior.KMEANS_SEED_BASE + label,
                                        fit1["ridge"], prior.KMEANS_MAX_ITERATIONS)
        stored4 = priors["G4"][label]
        rows = torch.nonzero(mask, as_tuple=False).flatten()
        _require(torch.equal(fit4["assignments"], assignments[rows]),
                 f"source C G4 TRAIN cluster assignments changed for class {label}")
        for key in ("weights", "means", "covariances", "regularized_covariances", "choleskies"):
            _require(torch.allclose(fit4[key], torch.as_tensor(stored4[key]).double(),
                                    rtol=REPLAY_RTOL, atol=REPLAY_ATOL),
                     f"source C G4 refit differs for class {label} field {key}")
        checks.append({"label": label, "train_rows": int(mask.sum()),
                       "g1_refit_passed": True, "g4_assignments_and_parameters_passed": True})
    return {"passed": True, "classes": checks}


def _freeze_for_inference(model: torch.nn.Module) -> None:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def _replay_source_cohorts(
    model: torch.nn.Module, source_checkpoint: dict[str, Any], prior_dir: Path,
    evaluator_d: torch.nn.Module, evaluator_c: ThirdConfirmationEvaluator,
    device: torch.device, output_dir: Path, expected_prior_run_id: str, expected_source_run_id: str,
) -> dict[str, Any]:
    gain = source_checkpoint["camera_gain"].to(device)
    replay: dict[str, Any] = {"task_id": TASK_ID, "source_step": UPDATES, "methods": {}}
    for method in ("G1", "G4"):
        path = prior_dir / f"generated_{method}.pt"
        _require(path.is_file(), f"original source {method} full cohort is missing: {path}", FileNotFoundError)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        prior._validate_saved_cohort(
            saved, method, expected_prior_run_id, expected_source_run_id,
        )
        _require(saved.get("run_id") == expected_prior_run_id
                 and saved.get("source_run_id") == expected_source_run_id,
                 f"original {method} cohort belongs to another G1/G4 task or source endpoint")
        decoded = prior._decode_cohort(
            method, model, evaluator_d, evaluator_c, saved["ols_coefficients"],
            saved["requested_labels"], saved["component_ids"], saved["proposal_ids"],
            gain, device, capture_count=10,
        )
        method_result = {
            "view": prior._compare_replay(decoded["images"], saved["images"], f"{method}_full_view"),
            "E_D_logits": prior._compare_replay(decoded["ed_logits"], saved["ed_logits"], f"{method}_full_E_D_logits"),
            "E_C_logits": prior._compare_replay(decoded["ec_logits"], saved["ec_logits"], f"{method}_full_E_C_logits"),
            "E_D_predictions_equal": bool(torch.equal(decoded["ed_predictions"], saved["ed_predictions"])),
            "E_C_predictions_equal": bool(torch.equal(decoded["ec_predictions"], saved["ec_predictions"])),
            "independent_numpy_readout": prior._validate_readout_numpy(decoded),
            "population": int(saved["requested_labels"].numel()),
        }
        _require(method_result["E_D_predictions_equal"] and method_result["E_C_predictions_equal"],
                 f"source C {method} full replay predictions changed")
        replay["methods"][method] = method_result
        del decoded, saved
    replay["complete_population"] = 20_000
    replay["passed"] = True
    atomic_write_json(output_dir / "source_C_full_G1_G4_replay.json", replay)
    return replay


def _validate_reference_schedule(
    schedule: torch.Tensor, source_checkpoint: dict[str, Any],
    source_schedule_30000_35000: torch.Tensor,
) -> dict[str, Any]:
    _require(torch.equal(schedule[30_000:35_000], source_schedule_30000_35000),
             "seed 2026101501 30,000-35,000 rows differ from the saved structural continuation segment")
    _require(torch.equal(schedule[35_000:45_000], source_checkpoint["schedule"]),
             "seed 2026101501 35,000-45,000 rows differ from the saved C endpoint segment")
    return {
        "seed": REFERENCE_SEED,
        "segment_30000_35000_rows": 5_000,
        "segment_35000_45000_rows": 10_000,
        "both_saved_segments_replayed_rowwise": True,
    }


def _reference_dev_examples(
    model: torch.nn.Module, saved_examples_path: Path, pixels: torch.Tensor,
    labels: torch.Tensor, gain: torch.Tensor, device: torch.device,
) -> dict[str, Any]:
    _require(saved_examples_path.is_file(), "fixed C source DEV examples are missing", FileNotFoundError)
    saved = torch.load(saved_examples_path, map_location="cpu", weights_only=False)
    rows = saved.get("rows")
    _require(isinstance(rows, torch.Tensor) and tuple(rows.shape) == (20,),
             "fixed C source display must contain 20 first-per-class DEV examples")
    _require(torch.equal(labels[rows], saved["labels"])
             and torch.equal(torch.bincount(saved["labels"], minlength=10), torch.full((10,), 2)),
             "fixed C display row labels are misaligned")
    targets = pixels.index_select(0, rows).unsqueeze(1).to(device=device, dtype=torch.float32) / 255.0
    view_parts: list[torch.Tensor] = []
    native_parts: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for selected in rows.split(BATCH_SIZE):
            raw = base._forward(model, image_batch(pixels, selected, "small", device))["physical_w_per_pixel"]
            gained = raw.float() * gain.to(device=device, dtype=torch.float32)
            native_parts.append(gained.clamp(0.0, 1.0).cpu())
            view_parts.append(resize_view(gained).cpu())
    views = torch.cat(view_parts)
    _require(torch.allclose(views, saved["images"].float(), rtol=REPLAY_RTOL, atol=REPLAY_ATOL),
             "C fixed DEV display replay differs from its saved production views")
    return {"rows": rows, "labels": saved["labels"], "targets28": targets.cpu(),
            "views28": views, "native200": torch.cat(native_parts)}


def _write_gallery_from_examples(
    directory: Path, methods: dict[str, dict[str, Any]], examples: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for label in range(10):
        indices = torch.nonzero(examples["labels"] == label, as_tuple=False).flatten()
        _require(indices.numel() >= 2, f"gallery DEV examples are missing for class {label}")
    for name, cohort in methods.items():
        prior._write_gallery(directory / name, {"G1": cohort["G1"], "G4": cohort["G4"]})
    def save_gray(tensor: torch.Tensor, path: Path) -> None:
        value = tensor.detach().cpu().float().squeeze()
        _require(value.ndim == 2 and torch.isfinite(value).all().item()
                 and float(value.min()) >= 0.0 and float(value.max()) <= 1.0,
                 "fixed gallery view is invalid")
        Image.fromarray(np.rint(value.numpy() * 255).clip(0, 255).astype(np.uint8), mode="L").save(path)
    dev_dir = directory / "DEV"
    dev_dir.mkdir(parents=True, exist_ok=True)
    lines = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>三个训练端点对照</title>",
             "<style>body{font:15px system-ui,sans-serif;margin:1rem}img{image-rendering:pixelated}</style>",
             "<h1>固定展示：每类前两张DEV</h1><p>图像统一灰度范围[0,1]。</p>"]
    for label in range(10):
        lines.append(f"<h2>类别{label}</h2><table><tr><th>行号</th><th>目标</th><th>原生200×200</th><th>生产28×28</th></tr>")
        indices = torch.nonzero(examples["labels"] == label, as_tuple=False).flatten()
        for rank, index_tensor in enumerate(indices):
            index = int(index_tensor)
            native = dev_dir / f"class_{label}_row_{int(examples['rows'][index])}_native.png"
            view = dev_dir / f"class_{label}_row_{int(examples['rows'][index])}_view28.png"
            target = dev_dir / f"class_{label}_row_{int(examples['rows'][index])}_target28.png"
            save_gray(examples["native200"][index], native)
            save_gray(examples["views28"][index], view)
            save_gray(examples["targets28"][index], target)
            lines.append(f"<tr><td>{int(examples['rows'][index])}</td><td><img width='112' height='112' src='{target.relative_to(directory).as_posix()}'></td>"
                         f"<td><img width='200' height='200' src='{native.relative_to(directory).as_posix()}'></td>"
                         f"<td><img width='112' height='112' src='{view.relative_to(directory).as_posix()}'></td></tr>")
        lines.append("</table>")
    lines.append("</html>")
    (directory / "dev_examples.html").write_text("\n".join(lines), encoding="utf-8")


def _write_per_image_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _require(bool(rows), "formal per-image table is empty")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report_html(path: Path, report: dict[str, Any]) -> None:
    rows = []
    for key, endpoint in report["endpoints"].items():
        gen = endpoint["generation"]
        for method in ("G1", "G4"):
            summary = gen[method]["three_stream_equal_weight"]
            rows.append(f"<tr><td>{html.escape(key)}</td><td>{method}</td><td>{summary['E_D_macro_recall']:.4f}</td>"
                        f"<td>{summary['E_C_macro_recall']:.4f}</td><td>{summary['macro_proxy_clipping_area_fraction']:.5f}</td>"
                        f"<td>{gen['shape_diversity'][method]['minimum_ratio']:.4f}</td></tr>")
    lines = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>Gaussian小架构训练轨迹与G1/G4确认</title>",
             "<style>body{font:16px system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#17202a}"
             "table{border-collapse:collapse;margin:1rem 0}td,th{padding:.45rem .65rem;border:1px solid #ccd1d1;text-align:left}"
             "iframe{width:100%;height:80vh;border:1px solid #ccd1d1}.status{font-weight:bold}</style>",
             f"<h1>Gaussian小架构三条训练轨迹G1/G4生成确认</h1><p class='status'>状态：{html.escape(report['status'])}</p>",
             f"<p>{html.escape(report['decision']['result'])}</p>",
             "<table><tr><th>端点</th><th>先验</th><th>E_D宏召回</th><th>E_C宏召回</th><th>代理截断</th><th>最低形态比</th></tr>"
             + "".join(rows) + "</table>",
             "<h2>原系统标准（历史口径）</h2><pre>" + html.escape(json.dumps(report["historical_system_quality_standard"], ensure_ascii=False, indent=2)) + "</pre>",
             "<h2>机器报告</h2><pre>" + html.escape(json.dumps(report["decision"], ensure_ascii=False, indent=2)) + "</pre>",
             "<h2>人工评阅</h2><p>本轮未建立盲评包；本结果限于冻结分类器与代理指标。</p>"]
    for endpoint in report["endpoints"]:
        lines.append(f"<h2>{html.escape(endpoint)} 固定图册</h2>")
        generation_folder = f"generation/{endpoint}/gallery"
        dev_folder = "reference_C/gallery" if endpoint == "C_seed_2026101501" else generation_folder
        lines.append(
            f"<p><a href='{generation_folder}/gallery.html'>G1/G4生成图册</a>　"
            f"<a href='{dev_folder}/dev_examples.html'>DEV示例</a></p>"
        )
    lines.append("</html>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _historical_system_quality(reconstruction: dict[str, Any], generation: dict[str, Any]) -> dict[str, Any]:
    g4 = generation["G4"]["three_stream_equal_weight"]
    rec_class_clip = [float(item.get("proxy_clipping_area_fraction", 0.0))
                      for item in reconstruction.get("per_class", [])]
    gen_class_clip = [float(item["proxy_clipping_area_fraction"])
                      for item in g4.get("per_class", [])]
    values = {
        "reconstruction_psnr_db": float(reconstruction.get("psnr_db_mean", math.nan)),
        "reconstruction_ssim": float(reconstruction.get("ssim_mean", math.nan)),
        "generation_E_D_macro_recall": float(g4["E_D_macro_recall"]),
        "generation_E_D_min_class_recall": min(g4["E_D_recall_by_class"]),
        "reconstruction_max_class_proxy_clipping": max(rec_class_clip, default=math.nan),
        "generation_max_class_proxy_clipping": max(gen_class_clip, default=math.nan),
        "generation_min_class_shape_ratio": float(generation["shape_diversity"]["G4"]["minimum_ratio"]),
    }
    checks = {
        "reconstruction_psnr_db": values["reconstruction_psnr_db"] >= 20.0,
        "reconstruction_ssim": values["reconstruction_ssim"] >= 0.80,
        "generation_E_D_macro_recall": values["generation_E_D_macro_recall"] >= 0.80,
        "generation_each_class_recall": values["generation_E_D_min_class_recall"] >= 0.60,
        "reconstruction_and_generation_clipping": (
            values["reconstruction_max_class_proxy_clipping"] <= 0.01
            and values["generation_max_class_proxy_clipping"] <= 0.01
        ),
        "generation_each_class_shape_ratio": values["generation_min_class_shape_ratio"] >= 0.25,
    }
    return {"status": "MEASURED_DIAGNOSTIC_ONLY", "values": values, "checks": checks,
            "all_historical_checks_pass": all(checks.values()),
            "thresholds": QUALITY_GATE}


def _run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    config_path = _resolve(args.config)
    config, runtime = _read_config(config_path)
    output_dir = _resolve(args.output_dir)
    expected_output = _resolve(Path(config["output"]["experiment_root"]))
    _require(output_dir == expected_output, "output path differs from the explicitly registered new attempt", ValueError)
    reference_checkpoint = _resolve(args.reference_checkpoint)
    reference_baseline_dir = _resolve(args.reference_baseline_dir)
    reference_prior_dir = _resolve(args.reference_prior_dir)
    _require(reference_checkpoint == _resolve(Path(config["reference"]["checkpoint"]))
             and reference_baseline_dir == _resolve(Path(config["reference"]["baseline_evidence_dir"]))
             and reference_prior_dir == _resolve(Path(config["reference"]["prior_evidence_dir"])),
             "CLI reference inputs differ from the explicit config", ValueError)
    if args.preflight_only:
        _require(args.resume_state is None, "preflight-only mode cannot consume resume-state")
        _require(not output_dir.exists(), f"preflight output already exists: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=False)
    elif args.resume_state is None:
        _require(not output_dir.exists(), f"refusing to overwrite existing output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=False)
    else:
        _require(output_dir.is_dir(), f"resume output directory is missing: {output_dir}")
    resume_state = _read_resume_state(_resolve(args.resume_state), output_dir, config) if args.resume_state else None
    atomic_write_json(output_dir / "config.json", config)
    if resume_state is None:
        atomic_write_json(output_dir / "run_identity.json", {
            "task_id": TASK_ID, "run_id": uuid.uuid4().hex,
            "semantic_authority_id": AUTHORITY, "model_identity": _model_identity(),
            "input_paths": {"source_checkpoint": str(reference_checkpoint),
                            "baseline_evidence_dir": str(reference_baseline_dir),
                            "prior_evidence_dir": str(reference_prior_dir),
                            "config": str(config_path)},
            "official_test_reads": 0,
        })
        progress: dict[str, Any] = {"status": "PREFLIGHT", "completed_seeds": [], "seed_checkpoints": {}}
    else:
        progress = {key: value for key, value in resume_state.items()
                    if key not in {"task_id", "config_identity", "output_dir", "updated_at"}}
    _write_progress(output_dir / "progress_state.json", output_dir, config, progress)
    _require(torch.cuda.is_available(), "AutoDL CUDA is required for training and optical confirmation")
    device = torch.device("cuda")
    pixels, labels, train_rows, dev_rows = base._load_data(_resolve(Path(config["data"]["training_file"])))
    train_labels = labels[train_rows].clone()
    _require(torch.equal(torch.bincount(train_labels, minlength=10), torch.tensor(TRAIN_LABEL_COUNTS))
             and torch.equal(torch.bincount(labels[dev_rows], minlength=10), torch.full((10,), 500))
             and not bool(torch.isin(train_rows, dev_rows).any()),
             "MNIST TRAIN/DEV roles or fixed class populations changed")
    context = _load_evidence(reference_checkpoint, reference_baseline_dir, reference_prior_dir, runtime, config)
    source_checkpoint = context["checkpoint"]
    source_schedule = _full_schedule(labels, train_rows, REFERENCE_SEED)
    schedule_receipt = _validate_reference_schedule(
        source_schedule, source_checkpoint, context["source_schedule_30000_35000"],
    )
    cache = context["cache"]
    _require(torch.equal(cache["rows"], train_rows)
             and torch.equal(cache["labels"], train_labels),
             "source C feature cache rows or labels differ from the fixed TRAIN population")
    prior_reference_checks = _validate_c_prior_cache(cache, context["priors"], train_labels)
    evaluator_d = base._load_evaluator(context["ed_path"], device)
    evaluator_c, ec_qualification = prior._load_ec(context["ec_path"], device)
    evaluator_d.eval()
    evaluator_c.eval()
    for evaluator in (evaluator_d, evaluator_c):
        for parameter in evaluator.parameters():
            parameter.requires_grad_(False)
    evaluator_states = (base._state_snapshot(evaluator_d), base._state_snapshot(evaluator_c))

    source_model = base._model("small", 2048, device, runtime)
    source_model.load_state_dict(source_checkpoint["model"], strict=True)
    _freeze_for_inference(source_model)
    source_model_state = base._state_snapshot(source_model)
    source_gain = source_checkpoint["camera_gain"].to(device)
    source_scale = source_checkpoint["scale"].detach().cpu()
    first_native, _ = _update_targets(pixels, source_schedule[0], device)
    source_readout = base._small_producer_consumer_check(source_model, first_native[:1])
    _require(source_readout.get("status") == "PASSED", "source C production OLS/readout check failed")
    source_window = _window_check(source_model, pixels, labels, source_schedule, runtime, device, UPDATES)
    replay_receipt = _replay_source_cohorts(
        source_model, source_checkpoint, reference_prior_dir, evaluator_d, evaluator_c,
        device, output_dir, context["prior_report"]["run_id"],
        context["baseline_report"]["run_id"],
    )
    c_examples = _reference_dev_examples(
        source_model, reference_baseline_dir / "C" / "dev_examples_step_045000.pt",
        pixels, labels, source_gain, device,
    )
    source_c_dir = output_dir / "reference_C"
    source_c_dir.mkdir(exist_ok=True)
    atomic_torch_save(source_c_dir / "dev_examples_step_045000.pt", {
        "task_id": TASK_ID, "seed": REFERENCE_SEED, "step": UPDATES, **c_examples,
    })
    c_gallery = source_c_dir / "gallery"
    _write_gallery_from_examples(c_gallery, {}, c_examples)
    c_cache = {
        "G1": context["priors"]["G1"], "G4": context["priors"]["G4"],
        "assignments": context["priors"]["G4_train_assignments_in_cache_order"],
    }
    source_endpoint = json.loads((reference_baseline_dir / "C" / "endpoint_summary.json").read_text(encoding="utf-8"))
    c_generation = _run_generation_endpoint(
        model_name="C_seed_2026101501", seed=REFERENCE_SEED, model=source_model,
        priors=c_cache, scale=source_scale, gain=source_gain, evaluator_d=evaluator_d,
        evaluator_c=evaluator_c, device=device, pixels=pixels, dev_rows=dev_rows,
        all_labels=labels, output_dir=output_dir, bootstrap_seed=BOOTSTRAP_SEED_BASE,
        assignments=c_cache["assignments"], train_features=cache["features_200"],
        train_labels=train_labels, config=config, runtime=runtime,
    )
    c_summary_reconstruction = source_endpoint.get("reconstruction", {})
    c_generation_metrics = c_generation["metrics"]
    source_model_final_state = base._state_snapshot(source_model)
    source_model_final_error = base._assert_close(source_model_state, source_model_final_state, "source_C_after_all_measurements")
    _require(source_model_final_error == 0.0, "source C endpoint changed during replay and generation evaluation")
    c_system_quality = source_endpoint.get("quality_gate", {
        "status": source_endpoint.get("status", "VALID_FAILED"),
        "historical_endpoint_status": "retained from the original C endpoint evidence",
    })
    endpoints: dict[str, Any] = {
        "C_seed_2026101501": {
            "training_updates_this_task": 0, "cumulative_step": UPDATES,
            "source_status": source_endpoint.get("status", "VALID_FAILED"),
            "reconstruction": c_summary_reconstruction,
            "generation": c_generation_metrics,
            "existing_system_quality": c_system_quality,
        }
    }
    all_generation_rows = list(c_generation["rows"])
    del c_generation
    del source_model
    torch.cuda.empty_cache()

    if args.preflight_only:
        preflight_results = []
        for seed in SEEDS:
            run_dir = output_dir / f"seed_{seed}"
            result = _train_seed(
                seed=seed, config=config, runtime=runtime, config_path=config_path,
                output_dir=output_dir, pixels=pixels, labels=labels, train_rows=train_rows,
                dev_rows=dev_rows, evaluator_d=evaluator_d, evaluator_c=evaluator_c,
                resume_checkpoint=None, resume_state=None, preflight_only=True,
            )
            preflight_results.append(result["summary"])
            progress["seed_checkpoints"][str(seed)] = str(run_dir / "checkpoint_latest.pt")
            progress.setdefault("checkpoint_global_steps", {})[str(seed)] = 0
        receipt = {
            "task_id": TASK_ID, "status": "PREFLIGHT_READY", "source_schedule": schedule_receipt,
            "source_C_full_replay": replay_receipt, "source_C_window": source_window,
            "source_C_readout": source_readout, "source_C_prior_refit": prior_reference_checks,
            "new_seed_preflights": preflight_results, "official_test_reads": 0,
            "scientific_updates": 0,
        }
        atomic_write_json(output_dir / "preflight_only.json", receipt)
        _write_progress(output_dir / "progress_state.json", output_dir, config, {
            "status": "PREFLIGHT_READY", "completed_seeds": [],
            "seed_checkpoints": progress["seed_checkpoints"],
        })
        return

    for seed_index, seed in enumerate(SEEDS):
        seed_dir = output_dir / f"seed_{seed}"
        resume_checkpoint = None
        if args.resume_state is not None:
            if seed_dir.exists():
                registered = progress.get("seed_checkpoints", {}).get(str(seed))
                _require(isinstance(registered, str) and registered,
                         f"resume-state does not name an explicit checkpoint for seed {seed}")
                resume_checkpoint = Path(registered).resolve()
                _require(resume_checkpoint.is_file()
                         and resume_checkpoint.parent.resolve() == seed_dir.resolve(),
                         f"resume-state checkpoint for seed {seed} is missing or outside its run directory")
                checkpoint_identity = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
                _require(checkpoint_identity.get("seed") == seed
                         and checkpoint_identity.get("task_id") == TASK_ID,
                         f"resume-state checkpoint identity for seed {seed} is invalid")
                expected_step = progress.get("checkpoint_global_steps", {}).get(str(seed))
                if expected_step is not None:
                    _require(checkpoint_identity.get("completed_global_step") == expected_step,
                             f"resume-state checkpoint step for seed {seed} changed")
            else:
                _require(str(seed) not in progress.get("seed_checkpoints", {}),
                         f"resume state names a missing seed {seed} checkpoint")
        trained = _train_seed(
            seed=seed, config=config, runtime=runtime, config_path=config_path,
            output_dir=output_dir, pixels=pixels, labels=labels, train_rows=train_rows,
            dev_rows=dev_rows, evaluator_d=evaluator_d, evaluator_c=evaluator_c,
            resume_checkpoint=resume_checkpoint, resume_state=progress if args.resume_state else None,
        )
        model = trained["model"]
        scale = trained["scale"]
        gain = trained["gain"]
        feature_path = seed_dir / "train_coefficient_cache.pt"
        if feature_path.is_file():
            feature_payload = torch.load(feature_path, map_location="cpu", weights_only=False)
            _require(feature_payload.get("task_id") == TASK_ID and feature_payload.get("seed") == seed
                     and feature_payload.get("source_checkpoint_step") == UPDATES
                     and torch.equal(feature_payload.get("rows"), train_rows)
                     and torch.equal(feature_payload.get("labels"), train_labels)
                     and tuple(feature_payload.get("features_200", torch.empty(0)).shape) == (TRAIN_COUNT, 200)
                     and bool(torch.isfinite(feature_payload["features_200"]).all()),
                     f"seed {seed} cached TRAIN features are incomplete or misaligned")
            features = feature_payload["features_200"].double()
            feature_seconds = float(feature_payload.get("elapsed_seconds", 0.0))
        else:
            features, feature_seconds = _encode_features(
                model, pixels, train_rows, train_labels, scale, device, feature_path, seed,
            )
        fitted = _fit_new_priors(features, train_labels, seed, seed_dir)
        prior_path = seed_dir / "fitted_priors_G1_G4.pt"
        prior_readback = torch.load(prior_path, map_location="cpu", weights_only=False)
        _require(prior_readback.get("task_id") == TASK_ID and prior_readback.get("seed") == seed
                 and bool((prior_readback["G4_train_assignments_in_cache_order"] >= 0).all()),
                 f"seed {seed} fitted prior failed read-back")
        endpoint_name = f"seed_{seed}"
        generated = _run_generation_endpoint(
            model_name=endpoint_name, seed=seed, model=model, priors=fitted,
            scale=scale, gain=gain, evaluator_d=evaluator_d, evaluator_c=evaluator_c,
            device=device, pixels=pixels, dev_rows=dev_rows, all_labels=labels,
            output_dir=output_dir, bootstrap_seed=BOOTSTRAP_SEED_BASE + seed_index + 1,
            assignments=fitted["assignments"], train_features=features,
            train_labels=train_labels, config=config, runtime=runtime,
        )
        all_generation_rows.extend(generated["rows"])
        dev_final_path = seed_dir / "dev_metrics_step_045000.json"
        _require(dev_final_path.is_file(), f"seed {seed} final DEV evaluation is missing")
        reconstruction = json.loads(dev_final_path.read_text(encoding="utf-8"))
        historical_quality = _historical_system_quality(reconstruction, generated["metrics"])
        endpoints[endpoint_name] = {
            "training": trained["summary"], "reconstruction": reconstruction,
            "generation": generated["metrics"], "existing_system_quality": historical_quality,
            "feature_encoding_seconds": feature_seconds,
            "cumulative_step": UPDATES,
        }
        fixed_examples = torch.load(seed_dir / "dev_examples_step_045000.pt", map_location="cpu", weights_only=False)
        _write_gallery_from_examples(generated["endpoint_dir"] / "gallery", {}, fixed_examples)
        progress.setdefault("completed_seeds", []).append(seed)
        progress_path = output_dir / "progress_state.json"
        current_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        completed_seeds = list(current_progress.get("completed_seeds", []))
        if seed not in completed_seeds:
            completed_seeds.append(seed)
        seed_checkpoints = dict(current_progress.get("seed_checkpoints", {}))
        checkpoint_steps = dict(current_progress.get("checkpoint_global_steps", {}))
        seed_checkpoints[str(seed)] = str(seed_dir / "checkpoint_step_045000.pt")
        checkpoint_steps[str(seed)] = UPDATES
        completed_blocks = list(current_progress.get("completed_generation_blocks", []))
        completed_blocks.extend(
            block for block in (f"{endpoint_name}_{method}_r{stream}"
                                 for method in ("G1", "G4") for stream in range(3))
            if block not in completed_blocks
        )
        _merge_progress(progress_path, output_dir, config, {
            "status": "RUNNING", "current_seed": seed, "current_step": UPDATES,
            "completed_seeds": completed_seeds, "seed_checkpoints": seed_checkpoints,
            "checkpoint_global_steps": checkpoint_steps,
            "completed_generation_blocks": completed_blocks,
        })
        del model, features, fitted, prior_readback, generated
        torch.cuda.empty_cache()

    _require(len(all_generation_rows) == 180_000,
             "all three endpoints must contribute exactly 180,000 generated metric rows")
    _write_per_image_csv(output_dir / "generation_per_image_metrics.csv", all_generation_rows)
    repeat_names = ["C_seed_2026101501", *(f"seed_{seed}" for seed in SEEDS)]
    repeat_results = {name: endpoints[name]["generation"]["gate"] for name in repeat_names}
    repeated = all(item["passed"] for item in repeat_results.values())
    decision = {
        "question": "Does fixed G4 proxy gain reproduce across three training-order trajectories and new sample streams?",
        "result": "G4_PROXY_GAIN_REPEATED_ON_THREE_TRAINING_SCHEDULES" if repeated
                  else "G4_PROXY_GAIN_REPLICATION_NOT_ESTABLISHED",
        "complete_valid_measurement": True,
        "per_endpoint_gates": repeat_results,
        "all_three_endpoints_passed": repeated,
        "limits": ["three sampling streams are not three training repetitions",
                   "single results per training endpoint do not establish population-wide superiority",
                   "not human readability, independent TEST, hardware readiness, or model adoption"],
    }
    report = {
        "task_id": TASK_ID, "status": "COMPLETE_DIAGNOSTIC",
        "scientific_question": decision["question"], "decision": decision,
        "training": {"new_seed_updates_each": UPDATES, "new_scientific_updates_total": 2 * UPDATES,
                     "training_seeds": list(SEEDS), "reused_C_endpoint_updates": 0,
                     "C_endpoint_reused": True, "official_test_reads": 0},
        "sampling": {"methods": ["G1", "G4"], "streams_per_method_endpoint": 3,
                     "samples_per_class_per_stream": SAMPLE_COUNT,
                     "formal_generation_population": 180_000,
                     "sample_streams_are_not_training_repetitions": True},
        "schedule_reproduction": schedule_receipt,
        "source_C_replay": replay_receipt,
        "source_C_prior_refit": prior_reference_checks,
        "evaluator_E_C_qualification": ec_qualification,
        "historical_system_quality_standard": QUALITY_GATE,
        "endpoints": endpoints,
        "validation": {"window_error_max": WINDOW_TOLERANCE,
                       "replay_rtol": REPLAY_RTOL, "replay_atol": REPLAY_ATOL,
                       "evaluator_frozen": True, "phase_only_training": True,
                       "adam_updates": {str(seed): UPDATES for seed in SEEDS},
                       "engineering_adam_calls": 20, "official_test_reads": 0},
        "elapsed_seconds_total": time.perf_counter() - started,
        "official_test_reads": 0,
    }
    atomic_write_json(output_dir / "report.json", report)
    atomic_write_json(output_dir / "validation_receipt.json", {
        "task_id": TASK_ID, "status": "VALIDATION_COMPLETE",
        "checks": {"source_C_schedule_30k_35k_and_35k_45k": schedule_receipt,
                   "source_C_full_G1_G4_replay": replay_receipt,
                   "source_C_G1_G4_prior_refit": prior_reference_checks,
                   "training_seeds": {str(seed): json.loads((output_dir / f"seed_{seed}" / "preflight.json").read_text(encoding="utf-8"))
                                      for seed in SEEDS},
                   "all_generated_row_count": len(all_generation_rows),
                   "all_three_endpoint_gates": repeat_results},
        "official_test_reads": 0,
    })
    _write_report_html(output_dir / "report.html", report)
    total_bytes = sum(p.stat().st_size for p in output_dir.rglob("*") if p.is_file())
    _require(total_bytes <= int(config["output"]["formal_result_budget_bytes"]),
             f"formal output exceeded 1 GiB budget: {total_bytes} bytes")
    _require(base._assert_close(evaluator_states[0], base._state_snapshot(evaluator_d), "E_D_final") == 0.0
             and base._assert_close(evaluator_states[1], base._state_snapshot(evaluator_c), "E_C_final") == 0.0,
             "confirmation evaluator changed during complete task")
    _require(source_model_final_error == 0.0,
             "source C endpoint changed during replay and generation evaluation")
    _write_progress(output_dir / "progress_state.json", output_dir, config, {
        "status": report["status"], "result": decision["result"],
        "completed_seeds": list(SEEDS),
        "seed_checkpoints": {str(seed): str(output_dir / f"seed_{seed}" / "checkpoint_step_045000.pt")
                             for seed in SEEDS},
        "completed_generation_blocks": [f"{model}_{method}_r{stream}"
                                        for model in repeat_names for method in ("G1", "G4")
                                        for stream in range(3)],
        "formal_result_bytes": total_bytes,
    })
    print(json.dumps({"status": report["status"], "decision": decision,
                      "formal_result_bytes": total_bytes,
                      "elapsed_seconds_total": report["elapsed_seconds_total"]}, ensure_ascii=False), flush=True)


def main() -> None:
    if "--recovery-worker-dir" in sys.argv:
        worker_parser = argparse.ArgumentParser(add_help=False)
        worker_parser.add_argument("--recovery-worker-dir", type=Path, required=True)
        worker_parser.add_argument("--config", type=Path, required=True)
        worker_args = worker_parser.parse_args()
        _recovery_worker(worker_args.recovery_worker_dir.resolve(), _resolve(worker_args.config))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--reference-baseline-dir", type=Path, required=True)
    parser.add_argument("--reference-prior-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-state", type=Path)
    args = parser.parse_args()
    try:
        _run(args)
    except Exception as error:
        output_dir = _resolve(args.output_dir)
        if output_dir.is_dir() and not (output_dir / "report.json").is_file():
            atomic_write_json(output_dir / "engineering_error.json", {
                "task_id": TASK_ID, "status": "ENGINEERING_INVALID",
                "error_type": type(error).__name__, "error": str(error),
                "scientific_updates": sum(
                    int(torch.load(path, map_location="cpu", weights_only=False).get("completed_global_step", 0))
                    for path in output_dir.glob("seed_*/checkpoint_latest.pt")
                ) if output_dir.is_dir() else 0,
            })
        raise


def _phase_gauge_association(
    priors: dict[str, Any], features: torch.Tensor, labels: torch.Tensor,
) -> dict[str, Any]:
    assignments = priors["assignments"]
    return _cluster_pivot_summary(features, labels, assignments)


def _run_generation_endpoint(
    *, model_name: str, seed: int, model: torch.nn.Module, priors: dict[str, Any],
    scale: torch.Tensor, gain: torch.Tensor, evaluator_d: torch.nn.Module,
    evaluator_c: ThirdConfirmationEvaluator, device: torch.device,
    pixels: torch.Tensor, dev_rows: torch.Tensor, all_labels: torch.Tensor,
    output_dir: Path, bootstrap_seed: int, assignments: torch.Tensor,
    train_features: torch.Tensor, train_labels: torch.Tensor,
    config: dict[str, Any], runtime: dict[str, Any],
) -> dict[str, Any]:
    endpoint_started = time.perf_counter()
    endpoint_dir = output_dir / "generation" / model_name
    endpoint_dir.mkdir(parents=True, exist_ok=True)
    method_rows: dict[str, dict[int, list[dict[str, Any]]]] = {"G1": {}, "G4": {}}
    method_views: dict[str, dict[int, list[torch.Tensor]]] = {
        method: {label: [] for label in range(10)} for method in ("G1", "G4")
    }
    method_summaries: dict[str, dict[str, Any]] = {"G1": {}, "G4": {}}
    readout_checks: dict[str, Any] = {}
    sampling_records: dict[str, dict[str, Any]] = {"G1": {}, "G4": {}}
    generated_window_checks: dict[str, Any] = {}
    completed_blocks: list[str] = []
    primary_cohorts: dict[str, dict[str, Any]] = {}
    phase_check: dict[str, Any] | None = None
    model_state_before = base._state_snapshot(model)
    _freeze_for_inference(model)
    for method in ("G1", "G4"):
        for stream in range(3):
            block_started = time.perf_counter()
            coefficients, requested, components, proposal_ids, sampling = _sample_population(
                method, priors, scale, stream,
            )
            expected_labels = torch.arange(10).repeat_interleave(SAMPLE_COUNT)
            _require(torch.equal(requested, expected_labels),
                     f"{model_name} {method}/r{stream} class-major population rows are misaligned")
            first_by_class = torch.arange(10) * SAMPLE_COUNT
            if stream == 0:
                generated_window_checks[method] = _generated_window_check(
                    model, coefficients[first_by_class], runtime, device,
                )
            cohort = prior._decode_cohort(
                method, model, evaluator_d, evaluator_c, coefficients, requested, components,
                proposal_ids, gain, device, capture_count=10 if stream == 0 else 1,
            )
            cohort.update({"task_id": TASK_ID, "model": model_name, "model_seed": seed,
                           "stream": stream, "sample_identity": {
                               "method": method, "stream": stream, "seed_base":
                               G1_NORMAL_SEED_BASE if method == "G1" else G4_NORMAL_SEED_BASE,
                           }, "sampling_accounting": sampling})
            sampling_records[method][f"stream_{stream}"] = sampling
            _require(tuple(cohort["images"].shape) == (10 * SAMPLE_COUNT, 1, 28, 28)
                     and bool(torch.isfinite(cohort["images"]).all()),
                     f"{model_name} {method} stream {stream} decoded views are incomplete")
            if stream == 0:
                phase_check = _global_phase_check(model, coefficients[first_by_class], device)
                primary_cohorts[method] = cohort
            readout_checks[f"{method}_stream_{stream}"] = prior._validate_readout_numpy(cohort)
            rows = _cohort_rows(model_name, stream, method, cohort)
            method_rows[method][stream] = rows
            method_summaries[method][f"stream_{stream}"] = _cohort_summary(rows)
            for label in range(10):
                method_views[method][label].append(cohort["images"][cohort["requested_labels"] == label].cpu())
            cohort_path = endpoint_dir / f"{method}_stream_{stream}.pt"
            save_cohort = dict(cohort)
            if stream != 0:
                save_cohort["images"] = torch.empty((0, 1, 28, 28), dtype=torch.float32)
            atomic_torch_save(cohort_path, save_cohort)
            reread = torch.load(cohort_path, map_location="cpu", weights_only=False)
            expected_population = 10 * SAMPLE_COUNT
            expected_image_rows = expected_population if stream == 0 else 0
            _require(reread.get("task_id") == TASK_ID and reread.get("model") == model_name
                     and reread.get("stream") == stream
                     and tuple(reread.get("images", torch.empty(0)).shape) == (expected_image_rows, 1, 28, 28)
                     and tuple(reread.get("ols_coefficients", torch.empty(0)).shape) == (expected_population, 100)
                     and tuple(reread.get("ed_logits", torch.empty(0)).shape) == (expected_population, 10)
                     and tuple(reread.get("ec_logits", torch.empty(0)).shape) == (expected_population, 10),
                     f"{model_name} {method} stream {stream} saved block failed consumer validation")
            elapsed_block = time.perf_counter() - block_started
            sampling_records[method][f"stream_{stream}"]["elapsed_seconds_including_decode_and_save"] = elapsed_block
            completed_blocks.append(f"{model_name}_{method}_r{stream}")
            _merge_progress(output_dir / "progress_state.json", output_dir, config, {
                "status": "GENERATING", "current_model": model_name,
                "current_method": method, "current_stream": stream,
                "completed_generation_blocks": completed_blocks.copy(),
                "last_generation_block_seconds": elapsed_block,
                "elapsed_seconds_total_so_far": time.perf_counter() - endpoint_started,
            })
            print(json.dumps({"phase": "GENERATION_BLOCK_COMPLETE", "model": model_name,
                              "method": method, "stream": stream, "elapsed_seconds": elapsed_block,
                              "population": len(rows)}, ensure_ascii=False), flush=True)
            del cohort, save_cohort, reread, coefficients, requested, components, proposal_ids
            torch.cuda.empty_cache()
    dev_views = pixels.index_select(0, dev_rows).unsqueeze(1).float() / 255.0
    dev_labels = all_labels[dev_rows]
    dev_tv = [float(base._pair_distances(dev_views[dev_labels == label], 2026101703 + label).mean())
              for label in range(10)]
    shape = _shape_ratios(method_views, dev_views, dev_labels, dev_tv)
    for method in ("G1", "G4"):
        combined_rows = [row for stream in range(3) for row in method_rows[method][stream]]
        method_summaries[method]["three_stream_equal_weight"] = _cohort_summary(combined_rows, 3 * SAMPLE_COUNT)
        for item in shape[method]["per_class"]:
            method_summaries[method]["three_stream_equal_weight"]["per_class"][item["label"]][
                "generated_to_dev_tv_ratio"] = item["ratio"]
    bootstrap = _bootstrap_multistream(method_rows, bootstrap_seed)
    point_ed = bootstrap["differences"]["E_D_correct"]
    point_ec = bootstrap["differences"]["E_C_correct"]
    clip_ci = bootstrap["differences"]["proxy_clipping_area_fraction"]["ci_95"]
    shape_passed = shape["G4"]["minimum_ratio"] >= 0.25
    supported = (point_ed["ci_95"][0] > 0.0 and point_ec["ci_95"][0] > 0.0
                 and clip_ci[1] <= 0.0 and shape_passed)
    metrics = {
        "model": model_name, "seed": seed, "bootstrap_seed": bootstrap_seed,
        "G1": method_summaries["G1"], "G4": method_summaries["G4"],
        "shape_diversity": shape, "bootstrap": bootstrap,
        "gate": {
            "E_D_macro_recall_ci_lower_gt_zero": point_ed["ci_95"][0] > 0.0,
            "E_C_macro_recall_ci_lower_gt_zero": point_ec["ci_95"][0] > 0.0,
            "macro_proxy_clipping_ci_upper_le_zero": clip_ci[1] <= 0.0,
            "all_G4_class_shape_ratios_at_least_0_25": shape_passed,
            "passed": supported,
        },
        "g1_g4_proxy_gain_supported": supported,
        "sampling_streams": 3,
        "formal_generation_population": 60_000,
        "readout_numpy_checks": readout_checks,
        "global_phase_invariance": phase_check,
        "generated_output_window_checks": generated_window_checks,
        "cluster_pivot_association": _phase_gauge_association(priors, train_features, train_labels),
        "classifier_sample_selection": False, "per_image_normalization": False,
        "official_test_reads": 0,
        "elapsed_seconds_generation_endpoint": time.perf_counter() - endpoint_started,
    }
    atomic_write_json(endpoint_dir / "generation_metrics.json", metrics)
    atomic_write_json(endpoint_dir / "sampling_rejections.json", sampling_records)
    gallery_path = endpoint_dir / "gallery"
    prior._write_gallery(gallery_path, primary_cohorts)
    _require(base._assert_close(model_state_before, base._state_snapshot(model), "frozen_model_after_generation") == 0.0,
             f"{model_name} frozen optical model changed during generation evaluation")
    all_rows = [row for method in ("G1", "G4") for stream in range(3) for row in method_rows[method][stream]]
    return {"metrics": metrics, "rows": all_rows, "method_views": method_views,
            "endpoint_dir": endpoint_dir, "model_state_before": model_state_before}





if __name__ == "__main__":
    main()

