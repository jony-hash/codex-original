"""Run a fixed 10,000-update Gaussian stroke/background continuation comparison."""

from __future__ import annotations

import argparse
import copy
import csv
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from PIL import Image

import scripts.research.lightgen.execute_lightgen_matched_architecture_v1 as base
from ldm_project.v2.evidence.io import cpu_copy
from ldm_project.v2.application.research.lightgen.native_image_inputs import image_batch
from ldm_project.v2.application.research.lightgen.image_readout import (
    resize_view,
    ssim_batch,
    structural_loss_from_raw as _structural_loss_from_raw,
)
from ldm_project.v2.application.research.lightgen.confirmation_evaluators import (
    EC_CHECKPOINT_SCHEMA,
    ThirdConfirmationEvaluator,
)
from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
)
from ldm_project.v2.evidence import atomic_torch_save, atomic_torch_save_new, atomic_write_json, atomic_write_new_json


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_SMALL_GAUSSIAN_FOREGROUND_BALANCE_AB_V1"
AUTHORITY = "USER_APPROVED_SMALL_GAUSSIAN_FOREGROUND_BALANCE_AB_20261021"
SOURCE_TASK_ID = "LIGHTGEN_SMALL_GAUSSIAN_STRUCTURAL_CONTINUATION_AB_V1"
SOURCE_UPDATES = 35_000
SOURCE_SCHEDULE_START = 30_000
CONTINUATION_UPDATES = 10_000
TOTAL_UPDATES = SOURCE_UPDATES + CONTINUATION_UPDATES
BATCH_SIZE = 10
SEED = 2026101501
CHECKPOINT_INTERVAL = 500
PROGRESS_INTERVAL = 100
RECOVERY_RTOL = 2.0e-6
RECOVERY_ATOL = 5.0e-8
WINDOW_TOLERANCE = 1.0e-4
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 2026102111
HUMAN_BOOTSTRAP_SEED = 2026102115
BRANCHES = ("C", "V")
REVIEW_SCHEMA = "lightgen_foreground_balance_blind_review.v1"
RATINGS_SCHEMA = "lightgen_foreground_balance_blind_ratings.v1"
EC_ARCHITECTURE = {
    "schema": "third_confirmation_evaluator_architecture.v1",
    "architecture_kind": "third_confirmation_canonical_cnn_v1",
    "input": {"shape": [1, 28, 28], "range": "[0,1]"},
    "layers": [
        {"type": "Conv2d", "in_channels": 1, "out_channels": 32, "kernel": 5, "padding": 2},
        {"type": "ReLU"}, {"type": "MaxPool2d", "kernel": 2},
        {"type": "Conv2d", "in_channels": 32, "out_channels": 64, "kernel": 5, "padding": 2},
        {"type": "ReLU"}, {"type": "MaxPool2d", "kernel": 2}, {"type": "Flatten"},
        {"type": "Linear", "in_features": 3136, "out_features": 128}, {"type": "ReLU"},
        {"type": "Linear", "in_features": 128, "out_features": 10},
    ],
    "forbidden": ["dropout", "batch_norm", "residual_blocks", "pretrained_weights"],
    "class_order": list(range(10)), "prediction_rule": "argmax(logits,dim=1)",
}
EC_PREPROCESSING = {
    "schema": "third_confirmation_evaluator_preprocessing.v1",
    "input_domain": "canonical MNIST train images",
    "value_contract": "float32 values in [0,1] from uint8/255",
    "shape": [1, 28, 28], "normalization": "none beyond canonical uint8/255 conversion",
    "augmentation": "none", "synthetic_cmos": False, "optical_model": False,
}


def _require(condition: bool, message: str, error_type: type[Exception] = RuntimeError) -> None:
    if not condition:
        raise error_type(message)


def _read_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "foreground-balance config must be a mapping")
    _require(value.get("schema") == "lightgen_small_gaussian_foreground_balance_ab.v1",
             "foreground-balance config schema changed")
    _require(value.get("task_id") == TASK_ID and value.get("semantic_authority_id") == AUTHORITY,
             "continuation task authorization identity changed")
    training = value["training"]
    _require(training["seed"] == SEED and training["source_completed_updates"] == SOURCE_UPDATES
             and training["continuation_updates"] == CONTINUATION_UPDATES,
             "source seed or approved continuation budget changed")
    _require(training["batch_size"] == BATCH_SIZE and training["samples_per_class_per_batch"] == 1,
             "continuation batch population changed")
    _require(training["optimizer"] == "Adam" and training["learning_rate"] == 1.0e-3
             and training["betas"] == [0.9, 0.999] and training["epsilon"] == 1.0e-8
             and training["weight_decay"] == 0.0,
             "continuation optimizer changed")
    _require(training["branch_order"] == list(BRANCHES)
             and training["checkpoint_interval"] == CHECKPOINT_INTERVAL
             and training["progress_interval"] == PROGRESS_INTERVAL
             and training["checkpoints_at"] == [0, 5000, 10000]
             and training["schedule_source_verification"] == [SOURCE_SCHEDULE_START, SOURCE_UPDATES],
             "continuation save or source-schedule verification changed")
    _require(value["source"]["task_id"] == SOURCE_TASK_ID
             and value["source"]["checkpoint_step"] == SOURCE_UPDATES
             and value["source"]["branch"] == "treatment"
             and value["source"]["model_identity"]["ols"] == "gaussian_s12_v1",
             "explicit source B endpoint or Gaussian model identity changed")
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
             and loss["structural_weight_fixed"] == 0.3148024967699311
             and loss["ssim_window"] == 7 and loss["ssim_sigma"] == 1.5
             and loss["data_range"] == 1.0 and loss["padding"] == 3
             and loss["balanced_target"] == "bilinear_resize_original_train_to_200x200_align_corners_false"
             and loss["balanced_error_input"] == "fixed_global_gain_times_unclipped_physical_w_per_pixel"
             and loss["balanced_weighting"] == "target_gray_and_one_minus_target_soft_weights"
             and loss["balanced_definition"] ==
             "half_sum_of_target_weighted_and_background_weighted_native_200x200_squared_error_means"
             and loss["kappa_calibration"] ==
             "match_weighted_native_rec_gradient_norm_to_weighted_balanced_gradient_norm_once"
             and loss["freeze_kappa_after_calibration"] is True
             and loss["source_structural_weight_and_bogt_weights_frozen"] is True
             and loss["adam_state_preserved"] is True,
             "foreground-balance or inherited SSIM loss semantics changed")
    evaluation = value["evaluation"]
    review = value["human_review"]
    _require(evaluation["paired_bootstrap_resamples"] == BOOTSTRAP_RESAMPLES
             and evaluation["paired_bootstrap_seed"] == BOOTSTRAP_SEED
             and evaluation["evaluator_role"] == "frozen_confirmation_only"
             and evaluation["generated_samples_per_class"] == 1000
             and review["unique_generated_cards_per_branch"] == 100
             and review["generated_cards_per_class"] == 10
             and review["control_selection_seed"] == 2026102112
             and review["treatment_selection_seed"] == 2026102113
             and review["hidden_duplicates"] == 20 and review["real_anchors"] == 20
             and review["duplicates_per_branch_per_class"] == 1
             and review["anchors_per_class"] == 2
             and review["display_native_pixels"] == 200 and review["display_css_pixels"] == 400
             and review["shuffle_seed"] == 2026102114
             and review["bootstrap_resamples"] == BOOTSTRAP_RESAMPLES
             and review["bootstrap_seed"] == HUMAN_BOOTSTRAP_SEED
             and review["duplicate_consistency_min"] == 18
             and review["anchor_label_match_min"] == 18
             and value["output"]["formal_result_budget_bytes"] == 536_870_912,
             "evaluation, human review, or result budget changed")
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


def _validate_schedule_rows(labels: torch.Tensor, train_indices: torch.Tensor,
                            schedule: torch.Tensor, expected_steps: int) -> None:
    _require(schedule.dtype == torch.long and tuple(schedule.shape) == (expected_steps, BATCH_SIZE),
             "continued sample schedule shape or dtype changed")
    _require(torch.equal(labels[schedule].cpu(), torch.arange(10).expand(expected_steps, BATCH_SIZE)),
             "continued TRAIN schedule contains a wrong class row")
    train_membership = torch.zeros(labels.shape[0], dtype=torch.bool)
    train_membership[train_indices.detach().cpu()] = True
    _require(bool(train_membership[schedule].all()), "continuation schedule contains a row outside TRAIN")


def _validate_source_metadata(
    checkpoint: dict[str, Any], config: dict[str, Any], source_config: dict[str, Any]
) -> None:
    _require(checkpoint.get("schema") == "lightgen_small_gaussian_structural_continuation_checkpoint.v1",
             "source B checkpoint schema changed")
    _require(checkpoint.get("task_id") == SOURCE_TASK_ID and checkpoint.get("branch") == "treatment"
             and checkpoint.get("seed") == SEED and checkpoint.get("completed_global_step") == SOURCE_UPDATES
             and checkpoint.get("completed_updates") == 5_000
             and checkpoint.get("source_checkpoint_step") == SOURCE_SCHEDULE_START
             and checkpoint.get("schedule_start_global_step") == SOURCE_SCHEDULE_START,
             "source checkpoint is not the explicit B branch at cumulative step 35,000")
    _require(checkpoint.get("config_identity") == source_config
             and source_config.get("task_id") == SOURCE_TASK_ID
             and source_config.get("schema") == "lightgen_small_gaussian_structural_continuation_ab.v1",
             "source checkpoint does not match its saved structural-continuation configuration")
    expected_identity = config["source"]["model_identity"]
    _require(checkpoint.get("model_identity") == expected_identity
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
    source_run_dir = source_path.parent.parent
    source_config_path = source_run_dir / "config.json"
    _require(source_config_path.is_file(), "source B run configuration is missing")
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    _require(isinstance(checkpoint, dict), "source checkpoint payload must be a mapping")
    _validate_source_metadata(checkpoint, config, source_config)
    references_path = source_run_dir / "source_references.pt"
    report_path = source_run_dir / "report.json"
    _require(references_path.is_file() and report_path.is_file(),
             "source baseline references or complete report are missing")
    references = torch.load(references_path, map_location="cpu", weights_only=False)
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    source_endpoint = source_report.get("endpoints", {}).get("treatment", {})
    _require(source_report.get("task_id") == SOURCE_TASK_ID
             and source_report.get("run_id") == checkpoint.get("run_id")
             and source_report.get("measurement_complete") is True
             and source_report.get("training", {}).get("endpoint_global_step_each_branch") == SOURCE_UPDATES
             and source_endpoint.get("global_step") == SOURCE_UPDATES
             and source_endpoint.get("measurement_complete") is True
             and source_endpoint.get("branch") == "treatment",
             "source B report is incomplete or does not match the explicit endpoint")
    _require(source_report.get("endpoints", {}).get("source", {}).get("global_step") == SOURCE_SCHEDULE_START
             and source_endpoint.get("generation", {}).get("population") == 10_000,
             "source B reconstruction or generation measurements are incomplete")
    for name in ("scale", "power_reference", "camera_gain", "structural_weight"):
        base._assert_close(checkpoint[name], references[name], f"source_{name}")
    _require(references.get("source_checkpoint_step") == SOURCE_SCHEDULE_START,
             "source frozen reference bundle no longer points to the 30,000-step origin")
    _require(checkpoint["weights"] == references["weights"], "source frozen BOGT coefficients changed")
    for name, shape in (("scale", (100,)), ("power_reference", (5,))):
        value = references[name]
        _require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape
                 and torch.isfinite(value).all().item(), f"source {name} has invalid shape or values")
    _require(isinstance(references["camera_gain"], torch.Tensor)
             and references["camera_gain"].numel() == 1
             and torch.isfinite(references["camera_gain"]).all().item()
             and references["camera_gain"].item() > 0.0
             and references["structural_weight"] == 0.3148024967699311,
             "source camera gain or frozen SSIM coefficient changed")
    rows = checkpoint.get("schedule")
    _require(isinstance(rows, torch.Tensor) and tuple(rows.shape) ==
             (SOURCE_UPDATES - SOURCE_SCHEDULE_START, BATCH_SIZE),
             "source checkpoint 30,000--35,000 schedule segment is incomplete")
    generated_path = source_path.parent / "generated_cohort.pt"
    _require(generated_path.is_file(), "source B fixed generated cohort is missing")
    generated = torch.load(generated_path, map_location="cpu", weights_only=False)
    _require(generated.get("task_id") == SOURCE_TASK_ID and generated.get("run_id") == checkpoint.get("run_id")
             and generated.get("branch") == "treatment" and generated.get("global_step") == SOURCE_UPDATES
             and tuple(generated.get("images", torch.empty(0)).shape) == (10_000, 1, 28, 28)
             and tuple(generated.get("ols_coefficients", torch.empty(0)).shape) == (10_000, 100),
             "source B fixed generated cohort does not match its checkpoint")
    requested = generated.get("requested_labels")
    proposals = generated.get("proposal_ids")
    coefficients = generated.get("ols_coefficients")
    images = generated.get("images")
    _require(isinstance(requested, torch.Tensor) and tuple(requested.shape) == (10_000,)
             and torch.equal(torch.bincount(requested.long(), minlength=10), torch.full((10,), 1000))
             and isinstance(proposals, torch.Tensor) and tuple(proposals.shape) == (10_000,)
             and isinstance(coefficients, torch.Tensor) and torch.isfinite(coefficients).all().item()
             and isinstance(images, torch.Tensor) and torch.isfinite(images).all().item(),
             "source B generated cohort class counts, coefficients, or views are incomplete")
    rejection = source_endpoint["generation"].get("proposal_rejection", {})
    _require(set(rejection) == {str(label) for label in range(10)},
             "source B proposal-rejection report does not contain all ten classes")
    for label in range(10):
        mask = requested == label
        record = rejection[str(label)]
        _require(record.get("noise_seed") == 2026101601 + label
                 and record.get("rejection_fraction") == 0.0
                 and record.get("retained_samples") == 1000
                 and proposals[mask].unique().numel() == 1000
                 and torch.equal(proposals[mask].sort().values, torch.tensor(record["proposal_ids"])),
                 f"source B class {label} proposal identity or rejection record changed")
    checkpoint["_source_generated"] = generated
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
             "source Adam state does not contain exactly 35,000 updates")
    return model, optimizer


def _target28(pixels: torch.Tensor, rows: torch.Tensor, device: torch.device) -> torch.Tensor:
    selected = pixels.index_select(0, rows.detach().cpu()).unsqueeze(1).to(device=device, dtype=torch.float32) / 255.0
    _require(tuple(selected.shape) == (BATCH_SIZE, 1, 28, 28), "original 28x28 target rows changed")
    return selected


def _balanced_reconstruction_parts(
    target: torch.Tensor, gained_unclipped: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-image stroke and background weighted native squared errors."""
    _require(target.ndim == 4 and target.shape == gained_unclipped.shape
             and target.shape[1] == 1 and tuple(target.shape[-2:]) == (200, 200),
             "balanced reconstruction inputs must be aligned native 200x200 grayscale batches")
    _require(torch.isfinite(target).all().item() and torch.isfinite(gained_unclipped).all().item()
             and float(target.min()) >= 0.0 and float(target.max()) <= 1.0,
             "balanced reconstruction inputs must be finite and targets must be in [0,1]")
    foreground_mass = target.sum(dim=(1, 2, 3))
    background_weight = 1.0 - target
    background_mass = background_weight.sum(dim=(1, 2, 3))
    _require(torch.isfinite(foreground_mass).all().item() and torch.isfinite(background_mass).all().item()
             and bool((foreground_mass > 0.0).all()) and bool((background_mass > 0.0).all()),
             "balanced reconstruction target has a zero or nonfinite stroke/background denominator")
    error_squared = (gained_unclipped - target).square()
    stroke = (target * error_squared).sum(dim=(1, 2, 3)) / foreground_mass
    background = (background_weight * error_squared).sum(dim=(1, 2, 3)) / background_mass
    _require(torch.isfinite(stroke).all().item() and torch.isfinite(background).all().item(),
             "balanced reconstruction components are nonfinite")
    return stroke, background


def _balanced_reconstruction_loss(target: torch.Tensor, gained_unclipped: torch.Tensor) -> torch.Tensor:
    stroke, background = _balanced_reconstruction_parts(target, gained_unclipped)
    value = 0.5 * (stroke + background).mean()
    _require(torch.isfinite(value).item(), "balanced reconstruction loss is nonfinite")
    return value


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
    structure = _structural_loss_from_raw(result["physical_w_per_pixel"], gain,
                                          _target28(pixels, rows, target_native.device))
    gained_unclipped = result["physical_w_per_pixel"] * gain.to(
        device=target_native.device, dtype=result["physical_w_per_pixel"].dtype,
    )
    parts["balanced_rec"] = _balanced_reconstruction_loss(target_native, gained_unclipped)
    return parts, structure, result


def _total_loss(parts: dict[str, torch.Tensor], structure: torch.Tensor | None,
                weights: dict[str, float], structural_weight: float,
                branch: str, kappa: float | None) -> torch.Tensor:
    _require(branch in BRANCHES and structure is not None
             and math.isfinite(structural_weight) and structural_weight > 0.0,
             "branch or frozen source SSIM coefficient is invalid")
    if branch == "V":
        _require(kappa is not None and math.isfinite(kappa) and kappa > 0.0,
                 "treatment balanced-loss coefficient is invalid")
        reconstruction = kappa * parts["balanced_rec"]
    else:
        _require(kappa is None, "control branch must not use a balanced-loss coefficient")
        reconstruction = parts["rec"]
    return (weights["dist"] * parts["dist"] + weights["rec"] * reconstruction
            + weights["op"] * parts["op"] + structural_weight * structure)


def _objective_for(
    model: torch.nn.Module, pixels: torch.Tensor, rows: torch.Tensor,
    scale: torch.Tensor, power_reference: torch.Tensor, gain: torch.Tensor,
    weights: dict[str, float], structural_weight: float, branch: str, kappa: float | None,
) -> torch.Tensor:
    parts, structure, _ = _loss_terms(model, pixels, rows, scale, power_reference, gain, branch)
    return _total_loss(parts, structure, weights, structural_weight, branch, kappa)


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


def _calibrate_kappa(
    model: torch.nn.Module,
    pixels: torch.Tensor,
    rows: torch.Tensor,
    scale: torch.Tensor,
    power_reference: torch.Tensor,
    gain: torch.Tensor,
    weights: dict[str, float],
    structural_weight: float,
) -> tuple[float, dict[str, float]]:
    parts, structure, _ = _loss_terms(model, pixels, rows, scale, power_reference, gain, "V")
    _require(structure is not None and structural_weight > 0.0,
             "source SSIM component is missing during balanced-loss calibration")
    groups = base._phase_groups(model)
    rec_gradients = torch.autograd.grad(weights["rec"] * parts["rec"], groups, retain_graph=True)
    balanced_gradients = torch.autograd.grad(weights["rec"] * parts["balanced_rec"], groups)
    rec_norm = _gradient_norm(rec_gradients, parts["rec"])
    balanced_norm = _gradient_norm(balanced_gradients, parts["balanced_rec"])
    kappa = rec_norm / balanced_norm
    _require(math.isfinite(kappa) and kappa > 0.0, "frozen foreground-balance coefficient is invalid")
    return kappa, {
        "weighted_rec_gradient_norm": rec_norm,
        "weighted_balanced_rec_gradient_norm_before_kappa": balanced_norm,
        "kappa": kappa,
        "source_lambda_ssim": structural_weight,
        "calibrated_on_global_step": SOURCE_UPDATES + 1,
        "frozen_after_calibration": True,
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
    structural_weight: float,
    kappa: float | None,
) -> dict[str, float | None]:
    model.train()
    model_device = next(model.parameters()).device
    expected_device = (torch.device(device.type, torch.cuda.current_device())
                       if device.type == "cuda" else device)
    _require(rows.shape == (BATCH_SIZE,) and model_device == expected_device,
             "logical batch or model device changed")
    optimizer.zero_grad(set_to_none=True)
    parts, structure, result = _loss_terms(model, pixels, rows, scale, power_reference, gain, branch)
    total = _total_loss(parts, structure, weights, structural_weight, branch, kappa)
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
        "balanced_rec": float(parts["balanced_rec"].detach()),
        "op": float(parts["op"].detach()),
        "ssim_loss": None if structure is None else float(structure.detach()),
        "kappa": kappa,
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
                         payload["structural_weight"], payload.get("kappa"))
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
    structural_weight: float,
    kappa: float | None,
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
                      gain, weights, branch, structural_weight, kappa)
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
        "kappa": kappa,
        "cuda_rng": [state.cpu() for state in torch.cuda.get_rng_state_all()],
        "cpu_rng": torch.random.get_rng_state().cpu(),
    }
    atomic_torch_save_new(staging_dir / "preflight_step_000003.pt", recovery_input)
    direct_loss = _train_update(model, optimizer, pixels, rows[3], device, scale,
                                power_reference, gain, weights, branch, structural_weight, kappa)
    direct_model = base._state_snapshot(model)
    direct_optimizer = cpu_copy(optimizer.state_dict())
    worker = subprocess.run(
        [sys.executable, "-m", "scripts.research.lightgen.execute_lightgen_small_foreground_balance_ab_v1",
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
            "checks": ["source_B_checkpoint_identity_and_adam_step",
                       "source_30000_to_35000_rows_match_and_future_35000_to_45000_schedule_is_fixed",
                       "source_gain_bogt_weights_and_ssim_coefficient_frozen",
                       "2048_to_4096_seven_plane_window_check",
                       "independent_numpy_ols_and_camera_readout",
                       "unclipped_balanced_loss_numpy_reference_and_analytic_derivative",
                       "C_and_V_phase_gradient_finite_difference",
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
    structural_weight: float,
    kappa: float | None,
    history: list[dict[str, Any]],
    elapsed: float,
    training_elapsed: float,
) -> dict[str, Any]:
    return {
        "schema": "lightgen_small_gaussian_foreground_balance_checkpoint.v1",
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
        "kappa": kappa,
        "history": history,
        "cuda_rng": [state.cpu() for state in torch.cuda.get_rng_state_all()],
        "cpu_rng": torch.random.get_rng_state().cpu(),
        "evaluator_role": "frozen_confirmation_only",
        "elapsed_seconds_continuation": elapsed,
        "training_elapsed_seconds_continuation": training_elapsed,
    }


def _validate_continuation_checkpoint(
    checkpoint: dict[str, Any], branch: str, run_id: str, source: dict[str, Any],
    config: dict[str, Any], schedule: torch.Tensor, references: dict[str, Any],
) -> int:
    _require(checkpoint.get("schema") == "lightgen_small_gaussian_foreground_balance_checkpoint.v1"
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
    expected_lambda = references.get("structural_weight")
    _require(checkpoint.get("structural_weight") == expected_lambda,
             "continuation checkpoint frozen SSIM coefficient changed")
    expected_kappa = references.get("kappa") if branch == "V" else None
    _require(checkpoint.get("kappa") == expected_kappa,
             "continuation checkpoint frozen balance coefficient changed")
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
    if segment_updates in (0, 5000, 10000):
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
    branch_c: list[dict[str, Any]], branch_v: list[dict[str, Any]], metric: str
) -> dict[str, Any]:
    c_by_row = {int(row["row"]): row for row in branch_c}
    v_by_row = {int(row["row"]): row for row in branch_v}
    _require(c_by_row.keys() == v_by_row.keys() and len(c_by_row) == 5_000,
             "A/B reconstruction rows do not form a complete paired DEV population")
    by_class: list[np.ndarray] = []
    for label in range(10):
        rows = sorted(row for row, item in c_by_row.items() if int(item["label"]) == label)
        _require(len(rows) == 500 and all(int(v_by_row[row]["label"]) == label for row in rows),
                 f"paired DEV class {label} is incomplete or mislabelled")
        by_class.append(np.asarray([float(v_by_row[row][metric]) - float(c_by_row[row][metric])
                                    for row in rows], dtype=np.float64))
    observed = float(np.mean([values.mean() for values in by_class]))
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        samples[index] = np.mean([
            values[generator.integers(0, values.size, size=values.size)].mean() for values in by_class
        ])
    return {
        "estimand": f"equal-class paired mean V-minus-C {metric}",
        "observed_difference": observed,
        "ci_95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "strata": "10 classes, 500 paired DEV rows sampled with replacement within class",
    }


def _load_ec_evaluator(path: Path, device: torch.device) -> tuple[ThirdConfirmationEvaluator, dict[str, Any]]:
    _require(path.is_file(), f"E_C checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(payload, dict) and payload.get("schema") == EC_CHECKPOINT_SCHEMA
             and payload.get("architecture_kind") == ThirdConfirmationEvaluator.architecture_kind
             and payload.get("architecture") == EC_ARCHITECTURE
             and payload.get("preprocessing") == EC_PREPROCESSING
             and payload.get("class_order") == list(range(10))
             and payload.get("prediction_rule") == "argmax(logits,dim=1)",
             "E_C architecture, class order, preprocessing, or prediction rule changed")
    qualification = payload.get("qualification", {})
    recalls = qualification.get("per_class_recall")
    policy = qualification.get("policy", {})
    _require(qualification.get("schema") == "third_confirmation_evaluator_qualification.v1"
             and qualification.get("split_role") == "E_C_QUALIFICATION"
             and qualification.get("evaluation_calls") == 1 and qualification.get("count") == 5500
             and qualification.get("passed") is True and isinstance(recalls, list) and len(recalls) == 10
             and float(qualification.get("accuracy", 0.0)) >= float(policy.get("min_accuracy", math.inf))
             and float(qualification.get("macro_recall", 0.0)) >= float(policy.get("min_macro_recall", math.inf))
             and min(map(float, recalls)) >= float(policy.get("min_class_recall", math.inf)),
             "E_C canonical-MNIST historical qualification is missing or failed")
    independence = payload.get("independence", {})
    _require(independence.get("role") == "INDEPENDENT_CONFIRMATION_ONLY"
             and independence.get("used_in_generator_loss") is False
             and independence.get("used_in_gradient_attribution") is False
             and independence.get("used_in_hyperparameter_selection") is False,
             "E_C is not an isolated confirmation evaluator")
    evaluator = ThirdConfirmationEvaluator().to(device)
    evaluator.load_state_dict(payload["state_dict"], strict=True)
    evaluator.freeze()
    _require(not evaluator.training and all(not parameter.requires_grad for parameter in evaluator.parameters()),
             "E_C must be frozen in evaluation mode")
    return evaluator, {
        "schema": payload["schema"], "architecture_kind": payload["architecture_kind"],
        "qualification_count": qualification["count"], "accuracy": qualification["accuracy"],
        "macro_recall": qualification["macro_recall"],
        "minimum_class_recall": min(map(float, recalls)),
        "role": independence["role"],
        "boundary": "canonical MNIST qualification; not CMOS or human-readability qualification",
    }


def _ec_logits(evaluator: ThirdConfirmationEvaluator, images: torch.Tensor) -> torch.Tensor:
    output = evaluator(images)
    _require(isinstance(output, tuple) and len(output) == 2, "E_C must return features and logits")
    features, logits = output
    _require(tuple(features.shape) == (images.shape[0], 128)
             and tuple(logits.shape) == (images.shape[0], 10)
             and torch.isfinite(features).all().item() and torch.isfinite(logits).all().item(),
             "E_C returned malformed or nonfinite output")
    return logits


def _logit_fields(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = logits.argmax(1)
    top2 = logits.topk(2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    rows = torch.arange(logits.shape[0], device=logits.device)
    target_values = logits[rows, targets.to(device=logits.device, dtype=torch.long)]
    competitors = logits.clone()
    competitors[rows, targets.to(device=logits.device, dtype=torch.long)] = -torch.inf
    target_margin = target_values - competitors.max(1).values
    return prediction, margin, target_margin


def _evaluate_reconstruction_dual(
    model: torch.nn.Module,
    evaluator: torch.nn.Module,
    ec_evaluator: ThirdConfirmationEvaluator,
    pixels: torch.Tensor,
    labels: torch.Tensor,
    dev_indices: torch.Tensor,
    device: torch.device,
    gain: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    batches: dict[str, list[torch.Tensor]] = {
        key: [] for key in ("mse", "psnr", "ssim", "clipping", "native_mse", "stroke_mse",
                            "background_mse", "rows", "labels", "ed_logits", "ec_logits")
    }
    fixed_rows = torch.stack([dev_indices[labels[dev_indices] == label][:2] for label in range(10)]).reshape(-1)
    examples: dict[str, list[torch.Tensor]] = {
        key: [] for key in ("rows", "labels", "targets", "images", "native_targets", "native_images")
    }
    model.eval()
    evaluator.eval()
    ec_evaluator.eval()
    with torch.inference_mode():
        for rows in dev_indices.split(BATCH_SIZE):
            rows_cpu = rows.detach().cpu().long()
            target_native = image_batch(pixels, rows_cpu, "small", device)
            result = base._forward(model, target_native)
            raw = result["physical_w_per_pixel"]
            gained = raw * gain.to(device=device, dtype=raw.dtype)
            view = resize_view(gained)
            target28 = pixels.index_select(0, rows_cpu).unsqueeze(1).to(device=device, dtype=torch.float32) / 255.0
            _require(view.shape == target28.shape, "DEV view differs from its fixed MNIST target")
            mse = (view - target28).square().mean((1, 2, 3)).clamp_min(1.0e-12)
            psnr = -10.0 * torch.log10(mse)
            ssim = ssim_batch(view, target28)
            stroke, background = _balanced_reconstruction_parts(target_native, gained)
            native_mse = (gained - target_native).square().mean((1, 2, 3))
            clipping = (((gained < 0.0) | (gained > 1.0)).double().mean((1, 2, 3)))
            target_labels = labels.index_select(0, rows_cpu).to(device=device, dtype=torch.long)
            ed_logits = evaluator(view)
            ec_logits_value = _ec_logits(ec_evaluator, view)
            _require(tuple(ed_logits.shape) == (rows_cpu.numel(), 10)
                     and torch.isfinite(ed_logits).all().item(), "E_D returned invalid DEV logits")
            for key, value in (("mse", mse), ("psnr", psnr), ("ssim", ssim), ("clipping", clipping),
                               ("native_mse", native_mse), ("stroke_mse", stroke),
                               ("background_mse", background), ("rows", rows_cpu),
                               ("labels", target_labels.cpu()), ("ed_logits", ed_logits.cpu()),
                               ("ec_logits", ec_logits_value.cpu())):
                batches[key].append(value.detach().cpu())
            selected = torch.isin(rows_cpu, fixed_rows)
            if bool(selected.any()):
                selected_device = selected.to(device=device)
                examples["rows"].append(rows_cpu[selected])
                examples["labels"].append(target_labels[selected_device].cpu())
                examples["targets"].append(target28[selected_device].cpu())
                examples["images"].append(view[selected_device].cpu())
                examples["native_targets"].append(target_native[selected_device].cpu())
                examples["native_images"].append(gained[selected_device].clamp(0.0, 1.0).cpu())
    values = {key: torch.cat(parts) for key, parts in batches.items()}
    ed_prediction, ed_margin, ed_target_margin = _logit_fields(values["ed_logits"], values["labels"])
    ec_prediction, ec_margin, ec_target_margin = _logit_fields(values["ec_logits"], values["labels"])
    ed_summary = base._classification(values["labels"], ed_prediction)
    ec_summary = base._classification(values["labels"], ec_prediction)
    _require(len(values["rows"]) == 5000 and torch.equal(torch.sort(values["rows"]).values,
                                                         torch.sort(dev_indices.cpu()).values),
             "DEV reconstruction rows are incomplete or duplicated")
    example_rows = torch.cat(examples["rows"])
    example_lookup = {int(row): index for index, row in enumerate(example_rows.tolist())}
    example_order = torch.tensor([example_lookup[int(row)] for row in fixed_rows.tolist()], dtype=torch.long)
    saved_examples = {key: torch.cat(value).index_select(0, example_order)
                      for key, value in examples.items()}
    independent_psnr = base._independent_psnr(saved_examples["images"], saved_examples["targets"])
    row_lookup = {int(row): index for index, row in enumerate(values["rows"].tolist())}
    fixed_positions = torch.tensor([row_lookup[int(row)] for row in fixed_rows.tolist()], dtype=torch.long)
    fixed_psnr = float(values["psnr"].index_select(0, fixed_positions).mean())
    _require(math.isclose(independent_psnr, fixed_psnr, rel_tol=1e-6, abs_tol=1e-6),
             "independent NumPy reference did not consume the production DEV view and target")
    per_image: list[dict[str, Any]] = []
    for index in range(len(values["rows"])):
        label = int(values["labels"][index])
        per_image.append({
            "row": int(values["rows"][index]), "label": label,
            "mse": float(values["mse"][index]), "psnr_db": float(values["psnr"][index]),
            "ssim": float(values["ssim"][index]), "native_mse_unclipped": float(values["native_mse"][index]),
            "native_stroke_weighted_mse": float(values["stroke_mse"][index]),
            "native_background_weighted_mse": float(values["background_mse"][index]),
            "proxy_clipping_fraction": float(values["clipping"][index]),
            "ed_prediction": int(ed_prediction[index]), "ed_margin": float(ed_margin[index]),
            "ed_target_margin": float(ed_target_margin[index]),
            "ec_prediction": int(ec_prediction[index]), "ec_margin": float(ec_margin[index]),
            "ec_target_margin": float(ec_target_margin[index]),
        })
    per_class: list[dict[str, Any]] = []
    for label in range(10):
        mask = values["labels"] == label
        _require(int(mask.sum()) == 500, f"DEV class {label} count is not 500")
        per_class.append({
            "label": label, "count": int(mask.sum()),
            "mse_mean": float(values["mse"][mask].mean()),
            "psnr_db_mean": float(values["psnr"][mask].mean()),
            "ssim_mean": float(values["ssim"][mask].mean()),
            "native_stroke_weighted_mse_mean": float(values["stroke_mse"][mask].mean()),
            "native_background_weighted_mse_mean": float(values["background_mse"][mask].mean()),
            "proxy_clipping_fraction": float(values["clipping"][mask].mean()),
            "ed_recall": float((ed_prediction[mask] == label).double().mean()),
            "ec_recall": float((ec_prediction[mask] == label).double().mean()),
        })
    summary = {
        "population": len(per_image), "step": TOTAL_UPDATES,
        "mse_mean": float(values["mse"].mean()), "psnr_db_mean": float(values["psnr"].mean()),
        "ssim_mean": float(values["ssim"].mean()),
        "native_mse_unclipped_mean": float(values["native_mse"].mean()),
        "native_stroke_weighted_mse_mean": float(values["stroke_mse"].mean()),
        "native_background_weighted_mse_mean": float(values["background_mse"].mean()),
        "proxy_clipping_fraction": float(values["clipping"].mean()),
        "saturation_fraction": float(values["clipping"].mean()),
        "ed_macro_recall": ed_summary["macro_recall"],
        "ed_recall_by_class": ed_summary["recall_by_class"].tolist(),
        "ed_confusion_matrix": ed_summary["confusion_matrix"].tolist(),
        "ec_macro_recall": ec_summary["macro_recall"],
        "ec_recall_by_class": ec_summary["recall_by_class"].tolist(),
        "ec_confusion_matrix": ec_summary["confusion_matrix"].tolist(),
        "per_class": per_class, "per_image": per_image,
        "common_view": "fixed gain then clamp to [0,1] then adaptive average pool 28x28",
        "native_weighted_error": "unclipped fixed-gain W/pixel compared with bilinear 200x200 target",
        "ed_ec_logits_artifact": "dev_evaluator_logits.npz",
        "independent_numpy_psnr_fixed_examples": independent_psnr,
    }
    saved_examples["ed_logits"] = values["ed_logits"].index_select(0, fixed_positions)
    saved_examples["ec_logits"] = values["ec_logits"].index_select(0, fixed_positions)
    saved_examples["per_image_ed_logits"] = values["ed_logits"]
    saved_examples["per_image_ec_logits"] = values["ec_logits"]
    saved_examples["per_image_rows"] = values["rows"]
    return summary, saved_examples


def _select_review_samples(requested: torch.Tensor, proposal_ids: torch.Tensor, branch: str,
                           seed: int) -> tuple[list[dict[str, Any]], list[int]]:
    requested_np = requested.detach().cpu().numpy().astype(np.int64)
    proposal_np = proposal_ids.detach().cpu().numpy().astype(np.int64)
    rng = np.random.default_rng(seed)
    selected_records: list[dict[str, Any]] = []
    capture_indices: set[int] = set()
    for label in range(10):
        candidates = np.flatnonzero(requested_np == label)
        _require(candidates.size == 1000, f"generation class {label} population is not 1,000")
        chosen = rng.choice(candidates, size=10, replace=False)
        for index in chosen.tolist():
            selected_records.append({"branch": branch, "requested_label": label,
                                     "sample_index": int(index), "proposal_id": int(proposal_np[index])})
            capture_indices.add(int(index))
        capture_indices.update(map(int, candidates[:10].tolist()))
    _require(len(selected_records) == 100 and len({(row["requested_label"], row["proposal_id"])
                                                   for row in selected_records}) == 100,
             f"{branch} human-review sample selection is incomplete or duplicated")
    return selected_records, sorted(capture_indices)


def _validate_sampled_cohort(
    samples: torch.Tensor, requested: torch.Tensor, proposal_ids: torch.Tensor,
    sampling: dict[str, Any], proposal_limit: int,
) -> None:
    _require(tuple(samples.shape) == (10_000, 100) and samples.is_complex()
             and torch.isfinite(samples).all().item()
             and tuple(requested.shape) == (10_000,) and requested.dtype == torch.long
             and tuple(proposal_ids.shape) == (10_000,),
             "conditioned Gaussian sampler returned malformed or nonfinite samples")
    _require(torch.equal(torch.bincount(requested, minlength=10), torch.full((10,), 1000)),
             "conditioned Gaussian sampler did not retain exactly 1,000 samples per class")
    _require(set(sampling) == {str(label) for label in range(10)},
             "conditioned Gaussian sampler accounting is missing a class")
    modal_power = samples.abs().square().sum(dim=1)
    _require(bool((modal_power <= base.MAX_MODAL_POWER_W * (1.0 + 1e-6)).all()),
             "conditioned Gaussian sampler retained a sample above the fixed physical modal-power bound")
    for label in range(10):
        mask = requested == label
        ids = proposal_ids[mask]
        record = sampling[str(label)]
        _require(ids.unique().numel() == 1000 and torch.equal(ids, torch.tensor(record["proposal_ids"]))
                 and record.get("noise_seed") == 2026101601 + label
                 and record.get("retained_samples") == 1000
                 and record.get("proposal_limit") == proposal_limit
                 and isinstance(record.get("trials"), int) and 1000 <= record["trials"] <= proposal_limit
                 and math.isfinite(float(record.get("rejection_fraction", math.nan)))
                 and 0.0 <= float(record["rejection_fraction"]) <= 1.0,
                 f"conditioned Gaussian sampler class {label} proposal accounting is invalid")


def _evaluate_generation_dual(
    branch: str, model: torch.nn.Module, evaluator: torch.nn.Module,
    ec_evaluator: ThirdConfirmationEvaluator, samples: torch.Tensor, requested: torch.Tensor,
    proposal_ids: torch.Tensor, labels: torch.Tensor, pixels: torch.Tensor,
    dev_indices: torch.Tensor, device: torch.device, gain: torch.Tensor,
    max_proposals_per_class: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], list[dict[str, Any]]]:
    selection_seed = 2026102112 if branch == "C" else 2026102113
    selected_records, capture_indices = _select_review_samples(requested, proposal_ids, branch, selection_seed)
    images: list[torch.Tensor] = []
    ed_logits_parts: list[torch.Tensor] = []
    ec_logits_parts: list[torch.Tensor] = []
    clipping_parts: list[torch.Tensor] = []
    native_capture: dict[int, torch.Tensor] = {}
    model.eval()
    evaluator.eval()
    ec_evaluator.eval()
    with torch.inference_mode():
        for start in range(0, samples.shape[0], BATCH_SIZE):
            end = min(start + BATCH_SIZE, samples.shape[0])
            result = model.forward_from_ols(samples[start:end].to(device))
            raw = result["physical_w_per_pixel"]
            gained = raw * gain.to(device=device, dtype=raw.dtype)
            view = resize_view(gained)
            ed_logits = evaluator(view)
            ec_logits_value = _ec_logits(ec_evaluator, view)
            _require(tuple(ed_logits.shape) == (end - start, 10) and torch.isfinite(ed_logits).all().item(),
                     "E_D returned malformed generated logits")
            images.append(view.cpu())
            ed_logits_parts.append(ed_logits.cpu())
            ec_logits_parts.append(ec_logits_value.cpu())
            clipping_parts.append((((gained < 0.0) | (gained > 1.0)).double().mean((1, 2, 3))).cpu())
            for index in capture_indices:
                if start <= index < end:
                    native_capture[index] = gained[index - start].clamp(0.0, 1.0).cpu()
    generated_images = torch.cat(images)
    ed_logits_all = torch.cat(ed_logits_parts)
    ec_logits_all = torch.cat(ec_logits_parts)
    clipping_fraction = torch.cat(clipping_parts)
    requested_cpu = requested.detach().cpu().long()
    proposal_cpu = proposal_ids.detach().cpu().long()
    ed_prediction, ed_margin, ed_target_margin = _logit_fields(ed_logits_all, requested_cpu)
    ec_prediction, ec_margin, ec_target_margin = _logit_fields(ec_logits_all, requested_cpu)
    ed_classification = base._classification(requested_cpu, ed_prediction)
    ec_classification = base._classification(requested_cpu, ec_prediction)
    _require(len(native_capture) == len(capture_indices) and generated_images.shape == (10_000, 1, 28, 28),
             f"{branch} generation native gallery capture is incomplete")
    dev_views = pixels.index_select(0, dev_indices.cpu()).unsqueeze(1).float() / 255.0
    dev_labels = labels.index_select(0, dev_indices.cpu()).long()
    generated_tv: list[float] = []
    dev_tv: list[float] = []
    ratios: list[float] = []
    for label in range(10):
        generated_class = generated_images[requested_cpu == label]
        dev_class = dev_views[dev_labels == label]
        generated_mean = float(base._pair_distances(generated_class, 2026101603 + label).mean())
        dev_mean = float(base._pair_distances(dev_class, 2026101703 + label).mean())
        generated_tv.append(generated_mean)
        dev_tv.append(dev_mean)
        ratios.append(generated_mean / max(dev_mean, 1e-12))
    capture_tensor = torch.tensor(capture_indices, dtype=torch.long)
    native_images = torch.stack([native_capture[index] for index in capture_indices])
    examples = {
        "images": generated_images, "requested_labels": requested_cpu,
        "ed_predictions": ed_prediction, "ec_predictions": ec_prediction,
        "ed_logits": ed_logits_all, "ec_logits": ec_logits_all,
        "ed_margin": ed_margin, "ec_margin": ec_margin,
        "ed_requested_margin": ed_target_margin, "ec_requested_margin": ec_target_margin,
        "proposal_ids": proposal_cpu, "ols_coefficients": samples.detach().cpu(),
        "proxy_clipping_fraction": clipping_fraction,
        "native_capture_indices": capture_tensor, "native_capture_images": native_images,
    }
    per_image: list[dict[str, Any]] = []
    for index in range(10_000):
        per_image.append({
            "sample_index": index, "requested_label": int(requested_cpu[index]),
            "proposal_id": int(proposal_cpu[index]),
            "ed_prediction": int(ed_prediction[index]), "ed_margin": float(ed_margin[index]),
            "ed_requested_margin": float(ed_target_margin[index]),
            "ec_prediction": int(ec_prediction[index]), "ec_margin": float(ec_margin[index]),
            "ec_requested_margin": float(ec_target_margin[index]),
            "proxy_clipping_fraction": float(clipping_fraction[index]),
        })
    summary = {
        "population": 10_000, "samples_per_class": 1000,
        "macro_recall": ed_classification["macro_recall"],
        "recall_by_class": ed_classification["recall_by_class"].tolist(),
        "ed_macro_recall": ed_classification["macro_recall"],
        "ed_recall_by_class": ed_classification["recall_by_class"].tolist(),
        "ed_confusion_matrix": ed_classification["confusion_matrix"].tolist(),
        "ec_macro_recall": ec_classification["macro_recall"],
        "ec_recall_by_class": ec_classification["recall_by_class"].tolist(),
        "ec_confusion_matrix": ec_classification["confusion_matrix"].tolist(),
        "proxy_clipping_fraction_by_class": [float(clipping_fraction[requested_cpu == label].mean())
                                              for label in range(10)],
        "proxy_clipping_fraction_max_class": max(
            float(clipping_fraction[requested_cpu == label].mean()) for label in range(10)),
        "generated_within_class_tv_mean": generated_tv,
        "dev_within_class_tv_mean": dev_tv,
        "generated_to_dev_tv_ratio": ratios,
        "generated_shape_pairs_per_class": 10000, "dev_shape_pairs_per_class": 10000,
        "sampling_power_bound_w": base.MAX_MODAL_POWER_W,
        "max_proposals_per_class": max_proposals_per_class,
        "classifier_based_sample_selection": False,
        "proposal_streams_shared_seed_but_rejection_retained_cohorts_not_assumed_strictly_paired": True,
        "native_capture_count_for_review_and_gallery": len(capture_indices),
        "per_image": per_image,
    }
    return summary, examples, selected_records


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _require(bool(rows) and all(row.keys() == rows[0].keys() for row in rows),
             f"CSV rows are empty or have inconsistent fields: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_branch(
    branch: str,
    run_id: str,
    branch_dir: Path,
    model: torch.nn.Module,
    evaluator: torch.nn.Module,
    ec_evaluator: ThirdConfirmationEvaluator,
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
    ec_state = base._state_snapshot(ec_evaluator)
    model_state = base._state_snapshot(model)
    reconstruction, examples = _evaluate_reconstruction_dual(
        model, evaluator, ec_evaluator, pixels, labels, dev_indices, device, gain,
    )
    _require(len(reconstruction.get("per_image", [])) == 5_000,
             f"{branch} endpoint DEV reconstruction population is incomplete")
    examples.update({"task_id": TASK_ID, "run_id": run_id, "branch": branch,
                     "global_step": TOTAL_UPDATES})
    atomic_torch_save(branch_dir / "dev_examples_step_045000.pt", examples)
    saved_dev = torch.load(branch_dir / "dev_examples_step_045000.pt", map_location="cpu", weights_only=False)
    _require(saved_dev.get("task_id") == TASK_ID and saved_dev.get("run_id") == run_id
             and saved_dev.get("branch") == branch and saved_dev.get("global_step") == TOTAL_UPDATES
             and tuple(saved_dev["per_image_rows"].shape) == (5_000,)
             and tuple(saved_dev["per_image_ed_logits"].shape) == (5_000, 10)
             and tuple(saved_dev["per_image_ec_logits"].shape) == (5_000, 10),
             f"{branch} saved DEV readback is incomplete or misidentified")
    np.savez_compressed(branch_dir / "dev_evaluator_logits.npz",
                        rows=examples["per_image_rows"].numpy(),
                        ed_logits=examples["per_image_ed_logits"].numpy(),
                        ec_logits=examples["per_image_ec_logits"].numpy())
    _write_rows_csv(branch_dir / "reconstruction_metrics_dev.csv", reconstruction["per_image"])
    base._write_contact_sheet(branch_dir / "dev_preview.png", examples["images"], examples["labels"], 2)
    base._write_contact_sheet(branch_dir / "dev_target_preview.png", examples["targets"], examples["labels"], 2)
    prior = base._fit_gaussians(model, pixels, labels, train_indices, "small", scale, device)
    samples, requested, proposal_ids, sampling = base._sample_gaussians(
        prior, scale.detach().cpu(),
        max_proposals_per_class=int(config["evaluation"]["max_proposals_per_class"]),
    )
    _validate_sampled_cohort(samples, requested, proposal_ids, sampling,
                             int(config["evaluation"]["max_proposals_per_class"]))
    generation, generated_examples, review_records = _evaluate_generation_dual(
        branch, model, evaluator, ec_evaluator, samples, requested, proposal_ids,
        labels, pixels, dev_indices, device, gain,
        int(config["evaluation"]["max_proposals_per_class"]),
    )
    generation["proposal_rejection"] = sampling
    generation["status"] = "COMPLETE"
    generated_examples.update({"task_id": TASK_ID, "run_id": run_id, "branch": branch,
                               "global_step": TOTAL_UPDATES})
    atomic_torch_save(branch_dir / "fitted_gaussian_prior.pt", prior)
    atomic_torch_save(branch_dir / "generated_cohort.pt", generated_examples)
    saved_generation = torch.load(branch_dir / "generated_cohort.pt", map_location="cpu", weights_only=False)
    _require(saved_generation.get("task_id") == TASK_ID and saved_generation.get("run_id") == run_id
             and saved_generation.get("branch") == branch and saved_generation.get("global_step") == TOTAL_UPDATES
             and tuple(saved_generation["images"].shape) == (10_000, 1, 28, 28)
             and tuple(saved_generation["ols_coefficients"].shape) == (10_000, 100)
             and tuple(saved_generation["ed_logits"].shape) == (10_000, 10)
             and tuple(saved_generation["ec_logits"].shape) == (10_000, 10)
             and torch.equal(saved_generation["proposal_ids"], proposal_ids.detach().cpu())
             and torch.equal(saved_generation["requested_labels"], requested.detach().cpu()),
             f"{branch} saved generation cohort readback is incomplete or misidentified")
    _write_rows_csv(branch_dir / "generation_metrics_per_image.csv", generation["per_image"])
    base._write_contact_sheet(branch_dir / "generation_preview.png", generated_examples["images"],
                              generated_examples["requested_labels"], 10)
    quality = base._quality_decision(config, reconstruction, generation)
    # Recheck the actual saved generation population and the two frozen objects.
    _require(generated_examples["images"].shape == (10_000, 1, 28, 28)
             and torch.equal(torch.bincount(generated_examples["requested_labels"], minlength=10),
                             torch.full((10,), 1000)),
             f"{branch} endpoint generation cohort is incomplete")
    for csv_name, expected_rows in (("reconstruction_metrics_dev.csv", 5_000),
                                    ("generation_metrics_per_image.csv", 10_000)):
        csv_path = branch_dir / csv_name
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            actual_rows = sum(1 for _ in csv.DictReader(stream))
        _require(actual_rows == expected_rows, f"{branch} saved {csv_name} row count changed")
    _require(base._assert_close(model_state, base._state_snapshot(model), f"{branch}_model_after_evaluation") == 0.0,
             f"{branch} model changed during endpoint evaluation")
    _require(base._assert_close(evaluator_state, base._state_snapshot(evaluator),
                                f"{branch}_evaluator_after_evaluation") == 0.0,
             f"{branch} frozen E_D changed during endpoint evaluation")
    _require(base._assert_close(ec_state, base._state_snapshot(ec_evaluator),
                                f"{branch}_EC_after_evaluation") == 0.0,
             f"{branch} frozen E_C changed during endpoint evaluation")
    return {
        "branch": branch,
        "global_step": TOTAL_UPDATES,
        "reconstruction": reconstruction,
        "generation": generation,
        "review_selection": review_records,
        "quality_gate": quality,
        "status": quality["status"],
        "measurement_complete": True,
    }


def _save_gray_png(value: torch.Tensor, path: Path) -> None:
    image = value.detach().cpu().float().squeeze()
    _require(image.ndim == 2 and torch.isfinite(image).all().item()
             and float(image.min()) >= 0.0 and float(image.max()) <= 1.0,
             "review and gallery images must be finite grayscale values in [0,1]")
    pixels_8bit = np.rint(image.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(pixels_8bit, mode="L").save(path, format="PNG", optimize=True)


def _render_blind_review_page(review_id: str, exposed_cards: list[dict[str, str]]) -> str:
    cards_json = json.dumps(exposed_cards, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>图像盲评</title>
<style>body{{font:18px system-ui,sans-serif;max-width:900px;margin:1rem auto;padding:0 1rem;color:#18212b}}main{{text-align:center}}img{{width:400px;height:400px;object-fit:contain;image-rendering:pixelated;border:1px solid #999;background:#fff}}.choices{{display:grid;grid-template-columns:repeat(6,1fr);gap:.4rem;margin:1rem 0}}label{{padding:.5rem;border:1px solid #ccd1d1;border-radius:.4rem}}button{{font-size:1rem;padding:.6rem 1rem;margin:.3rem}}#id{{color:#666;font-size:.8rem}}</style>
<h1>图像评阅</h1><p>请根据图像本身选择一个数字；如果无法辨认，选择“不可辨认”。图像按统一固定灰度显示，不做逐图增强。</p>
<main><p id='progress'></p><img id='image' alt='当前评阅图'><p id='id'></p><div id='choices' class='choices'></div>
<button id='prev'>上一张</button><button id='next'>下一张</button><button id='save'>导出全部评分</button><p id='save_status'></p></main>
<script>
const TASK_ID={json.dumps(TASK_ID)}, REVIEW_ID={json.dumps(review_id)}, CARDS={cards_json};
const STORE='lightgen-review-'+REVIEW_ID;let state={{}};let position=0;
try{{state=JSON.parse(localStorage.getItem(STORE)||'{{}}')}}catch{{state={{}}}}
function persist(){{localStorage.setItem(STORE,JSON.stringify(state));document.getElementById('progress').textContent=`第 ${{position+1}} / ${{CARDS.length}} 张 · 已完成 ${{Object.keys(state).length}} 张`;}}
function render(){{const c=CARDS[position];document.getElementById('image').src=c.image;document.getElementById('id').textContent='';document.getElementById('choices').innerHTML='';document.getElementById('save_status').textContent='';
const opts=[...Array(10)].map((_,i)=>[String(i),String(i)]).concat([['unreadable','不可辨认']]);
for(const [value,text] of opts){{const label=document.createElement('label');const radio=document.createElement('input');radio.type='radio';radio.name='rating';radio.value=value;radio.checked=state[c.card_id]===value;radio.onchange=()=>{{state[c.card_id]=value;persist();render();}};label.append(radio,document.createTextNode(' '+text));document.getElementById('choices').append(label);}}
persist();}}
document.getElementById('prev').onclick=()=>{{position=Math.max(0,position-1);render();}};
document.getElementById('next').onclick=()=>{{position=Math.min(CARDS.length-1,position+1);render();}};
document.getElementById('save').onclick=()=>{{if(Object.keys(state).length!==CARDS.length){{document.getElementById('save_status').textContent='请先完成全部240张评分。';return;}}
const cards=CARDS.map(c=>{{const v=state[c.card_id];return {{card_id:c.card_id,rating:v==='unreadable'?'unreadable':Number(v)}};}});
const payload={{schema:'{RATINGS_SCHEMA}',task_id:TASK_ID,review_id:REVIEW_ID,cards}};
const blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='lightgen_foreground_balance_ratings_'+REVIEW_ID+'.json';a.click();URL.revokeObjectURL(a.href);document.getElementById('save_status').textContent='评分文件已导出。';}};
render();</script></html>"""


def _make_source_gallery(
    source: dict[str, Any], source_checkpoint: Path, config: dict[str, Any],
    pixels: torch.Tensor, device: torch.device,
) -> tuple[dict[str, torch.Tensor], int]:
    grid = int(config["optics"]["small"]["production_propagation_grid"])
    gain = source["camera_gain"].to(device)
    model = base._model("small", grid, device, config)
    model.load_state_dict(source["model"], strict=True)
    model.eval()
    source_run = source_checkpoint.parent
    dev_path = source_run / "dev_examples_step_035000.pt"
    _require(dev_path.is_file(), "source B fixed DEV examples are missing")
    dev_examples = torch.load(dev_path, map_location="cpu", weights_only=False)
    dev_rows = dev_examples["rows"].long()
    dev_labels = dev_examples["labels"].long()
    _require(tuple(dev_rows.shape) == (20,) and tuple(dev_examples["images"].shape) == (20, 1, 28, 28),
             "source B fixed DEV display population changed")
    dev_native_targets = image_batch(pixels, dev_rows, "small", device)
    with torch.inference_mode():
        dev_result = base._forward(model, dev_native_targets)
        dev_gained = dev_result["physical_w_per_pixel"] * gain.to(device=device, dtype=torch.float32)
        dev_native_images = dev_gained.clamp(0.0, 1.0).cpu()
        dev_28_images = resize_view(dev_gained).cpu()
    saved_dev_images = dev_examples["images"].float().cpu()
    _require(torch.allclose(dev_28_images, saved_dev_images, rtol=RECOVERY_RTOL, atol=RECOVERY_ATOL),
             "source B fixed DEV view does not replay from its saved checkpoint")
    generated = source["_source_generated"]
    generated_indices = torch.cat([
        torch.nonzero(generated["requested_labels"] == label, as_tuple=False).flatten()[:10]
        for label in range(10)
    ]).long()
    _require(generated_indices.numel() == 100, "source B fixed generation gallery rows are incomplete")
    source_samples = generated["ols_coefficients"].index_select(0, generated_indices)
    with torch.inference_mode():
        gen_result = model.forward_from_ols(source_samples.to(device))
        gen_gained = gen_result["physical_w_per_pixel"] * gain.to(device=device, dtype=torch.float32)
        gen_native_images = gen_gained.clamp(0.0, 1.0).cpu()
        gen_28_images = resize_view(gen_gained).cpu()
    saved_gen_images = generated["images"].index_select(0, generated_indices).float().cpu()
    _require(torch.allclose(gen_28_images, saved_gen_images, rtol=RECOVERY_RTOL, atol=RECOVERY_ATOL),
             "source B first-ten-per-class generation views fail checkpoint replay")
    return {
        "dev_rows": dev_rows, "dev_labels": dev_labels,
        "dev_targets_28": dev_examples["targets"].float().cpu(),
        "dev_views_28": saved_dev_images,
        "dev_targets_native": dev_native_targets.cpu(),
        "dev_views_native": dev_native_images,
        "generation_indices": generated_indices,
        "generation_labels": generated["requested_labels"].index_select(0, generated_indices).long(),
        "generation_proposal_ids": generated["proposal_ids"].index_select(0, generated_indices).long(),
        "generation_views_28": saved_gen_images,
        "generation_views_native": gen_native_images,
    }, 120


def _write_blind_review(
    run_dir: Path, config: dict[str, Any], source: dict[str, Any], source_checkpoint: Path,
    pixels: torch.Tensor, labels: torch.Tensor, dev_indices: torch.Tensor,
    branch_summaries: dict[str, dict[str, Any]], device: torch.device,
) -> dict[str, Any]:
    key_path = run_dir / "blind_review_key.json"
    page_path = run_dir / "blind_review.html"
    if key_path.is_file() and page_path.is_file():
        key = json.loads(key_path.read_text(encoding="utf-8"))
        _require(key.get("schema") == REVIEW_SCHEMA and len(key.get("cards", [])) == 240
                 and key.get("task_id") == TASK_ID,
                 "saved blind-review key is incomplete")
        return {"review_id": key["review_id"], "cards": len(key["cards"]), "reused_complete_manifest": True}

    started = time.perf_counter()
    source_gallery, source_optical_forwards = _make_source_gallery(source, source_checkpoint, config, pixels, device)
    atomic_torch_save(run_dir / "source_B_display_examples.pt", source_gallery)
    review_cards: list[dict[str, Any]] = []
    branch_cohorts: dict[str, dict[str, Any]] = {}
    for branch in BRANCHES:
        cohort_path = run_dir / branch / "generated_cohort.pt"
        _require(cohort_path.is_file(), f"{branch} generated cohort is missing for blind review")
        cohort = torch.load(cohort_path, map_location="cpu", weights_only=False)
        _require(cohort.get("task_id") == TASK_ID and cohort.get("branch") == branch
                 and cohort.get("global_step") == TOTAL_UPDATES,
                 f"{branch} generated cohort identity does not match the final endpoint")
        branch_cohorts[branch] = cohort
        index_to_image = {int(index): image for index, image in zip(
            cohort["native_capture_indices"].tolist(), cohort["native_capture_images"], strict=True
        )}
        for selected in branch_summaries[branch]["review_selection"]:
            index = int(selected["sample_index"])
            _require(index in index_to_image, f"{branch} native review image was not saved")
            review_cards.append({
                "kind": "generated", "branch": branch, "requested_label": int(selected["requested_label"]),
                "sample_index": index, "proposal_id": int(selected["proposal_id"]),
                "image": index_to_image[index], "duplicate_group": None,
            })
    _require(len(review_cards) == 200, "blind review must contain 100 unique generated images per branch")

    for branch in BRANCHES:
        for label in range(10):
            match = next(card for card in review_cards if card["kind"] == "generated"
                         and card["branch"] == branch and card["requested_label"] == label)
            duplicate_group = f"duplicate_{branch}_{label}"
            match["duplicate_group"] = duplicate_group
            review_cards.append({**match, "kind": "duplicate",
                                 "duplicate_group": duplicate_group})
    fixed_rows = torch.stack([dev_indices[labels[dev_indices] == label][:2] for label in range(10)]).reshape(-1)
    _require(fixed_rows.numel() == 20, "fixed real-reference anchors are incomplete")
    native_targets = image_batch(pixels, fixed_rows, "small", device).cpu()
    for index, row in enumerate(fixed_rows.tolist()):
        label = int(labels[row])
        review_cards.append({"kind": "real_anchor", "row": int(row), "target_label": label,
                             "image": native_targets[index], "branch": None,
                             "requested_label": None, "sample_index": None,
                             "proposal_id": None, "duplicate_group": None})
    _require(len(review_cards) == 240, "blind review card count must be exactly 240")
    permutation = np.random.default_rng(int(config["human_review"]["shuffle_seed"])).permutation(240).tolist()
    shuffled = [review_cards[index] for index in permutation]
    review_id = uuid.uuid4().hex
    cards_dir = run_dir / "review_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    exposed_cards: list[dict[str, str]] = []
    hidden_cards: list[dict[str, Any]] = []
    for card_index, source_card in enumerate(shuffled, start=1):
        card_id = f"R{card_index:04d}"
        image_name = f"card_{card_index:04d}.png"
        _save_gray_png(source_card["image"], cards_dir / image_name)
        exposed_cards.append({"card_id": card_id, "image": f"review_cards/{image_name}"})
        hidden_cards.append({key: value for key, value in source_card.items() if key != "image"} |
                            {"card_id": card_id, "image": f"review_cards/{image_name}"})
    key = {
        "schema": REVIEW_SCHEMA, "task_id": TASK_ID, "review_id": review_id,
        "registered_card_count": 240, "generated_unique_card_count": 200,
        "hidden_duplicate_count": 20, "real_anchor_count": 20,
        "shuffle_seed": int(config["human_review"]["shuffle_seed"]),
        "display": {"source_image_pixels": 200, "css_display_pixels": 400,
                    "range": "fixed gain then clamp [0,1] then uniform nearest-neighbor display"},
        "cards": hidden_cards,
    }
    atomic_write_json(key_path, key)
    atomic_write_json(run_dir / "blind_review_public_manifest.json", {
        "schema": REVIEW_SCHEMA, "task_id": TASK_ID, "review_id": review_id,
        "card_count": 240, "rating_values": list(range(10)) + ["unreadable"],
        "display_source_pixels": 200, "display_css_pixels": 400,
    })
    page = _render_blind_review_page(review_id, exposed_cards)
    page_path.write_text(page, encoding="utf-8")
    _require(sum(path.stat().st_size for path in cards_dir.glob("*.png")) > 0
             and len(list(cards_dir.glob("*.png"))) == 240,
             "blind-review card image export is incomplete")
    saved_page = page_path.read_text(encoding="utf-8")
    _require(all(field not in saved_page for field in ("requested_label", "branch", "proposal_id", "target_label"))
             and 'width:400px;height:400px' in saved_page and "image-rendering:pixelated" in saved_page,
             "blind-review page leaked hidden identity fields or changed fixed display sizing")
    return {"review_id": review_id, "cards": 240, "generated_unique": 200, "duplicates": 20,
            "anchors": 20, "source_gallery_optical_forwards": source_optical_forwards,
            "blind_page": "blind_review.html", "hidden_key": "blind_review_key.json",
            "elapsed_seconds": time.perf_counter() - started}


def _write_comparison_html(path: Path, report: dict[str, Any]) -> None:
    """Write a metrics-only page before human labels exist; never expose a branch gallery."""
    rows = []
    for key, label in (("source_B", "源B 35,000步"), ("C", "C：原目标"), ("V", "V：笔画/背景均衡")):
        record = report["endpoints"].get(key)
        if not record:
            continue
        recon, gen = record["reconstruction"], record["generation"]
        rows.append(f"<tr><td>{html.escape(label)}</td><td>{recon['psnr_db_mean']:.4f}</td>"
                    f"<td>{recon['ssim_mean']:.5f}</td><td>{gen.get('ed_macro_recall', gen.get('macro_recall', float('nan'))):.5f}</td>"
                    f"<td>{gen.get('ec_macro_recall', float('nan')):.5f}</td>"
                    f"<td>{max(gen['proxy_clipping_fraction_by_class']):.5f}</td></tr>")
    lines = [
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Gaussian均衡重构训练自动结果</title>",
        "<style>body{font:16px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#18212b}"
        "table{border-collapse:collapse}td,th{padding:.5rem .7rem;border:1px solid #ccd1d1;text-align:left}</style>",
        f"<h1>自动评价</h1><p>当前状态：{html.escape(report['status'])}</p>",
        "<p>人眼盲评尚未收尾。此页只列自动指标，不包含带分支标签的生成图册。</p>",
        "<table><tr><th>端点</th><th>DEV PSNR dB</th><th>DEV SSIM</th><th>E_D生成宏召回</th>"
        "<th>E_C生成宏召回</th><th>生成最大类代理截断率</th></tr>" + "".join(rows) + "</table>",
        "<h2>配对数值</h2><pre>" + html.escape(json.dumps(report.get("paired_comparison", {}), ensure_ascii=False, indent=2)) + "</pre>",
        "<p><a href='blind_review.html'>打开240张盲评页面</a> · 评分完成前不提供带分支标记的图册。</p></html>",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_preflight(
    source: dict[str, Any], references: dict[str, Any], config: dict[str, Any],
    config_path: Path, source_path: Path, pixels: torch.Tensor, labels: torch.Tensor,
    train_indices: torch.Tensor, schedule: torch.Tensor, device: torch.device,
    evaluator: torch.nn.Module, ec_evaluator: torch.nn.Module, staging_parent: Path,
) -> tuple[dict[str, Any], float, float]:
    grid = int(config["optics"]["small"]["production_propagation_grid"])
    scale = source["scale"].to(device)
    power_reference = source["power_reference"].to(device)
    gain = source["camera_gain"].to(device)
    weights = source["weights"]
    model, _ = _new_branch(source, config, device, grid)
    torch.cuda.set_rng_state_all([state.cpu() for state in source["cuda_rng"]])
    torch.random.set_rng_state(source["cpu_rng"])
    started = time.perf_counter()
    future_schedule = schedule[SOURCE_UPDATES:]
    selected_grid, source_window = base._window_check(
        "small", pixels, labels, future_schedule, config, device, model=model,
    )
    _require(selected_grid == grid and source_window["complex_relative_error_max"] <= WINDOW_TOLERANCE,
             "source endpoint failed its fixed 2048/4096 seven-plane window check")
    initial_batch = image_batch(pixels, schedule[SOURCE_UPDATES], "small", device)
    independent_readout = base._small_producer_consumer_check(model, initial_batch[:1])
    _require(independent_readout.get("status") == "PASSED",
             "source OLS coefficient or full-camera W/pixel readout check failed")
    structural_weight = float(source["structural_weight"])
    treatment_model, _ = _new_branch(source, config, device, grid)
    torch.cuda.set_rng_state_all([state.cpu() for state in source["cuda_rng"]])
    torch.random.set_rng_state(source["cpu_rng"])
    kappa, calibration = _calibrate_kappa(
        treatment_model, pixels, schedule[SOURCE_UPDATES], scale, power_reference, gain,
        weights, structural_weight,
    )
    source_control, _ = _new_branch(source, config, device, grid)
    torch.cuda.set_rng_state_all([state.cpu() for state in source["cuda_rng"]])
    torch.random.set_rng_state(source["cpu_rng"])
    control_rows = schedule[SOURCE_UPDATES]
    treatment_rows = schedule[SOURCE_UPDATES]
    source_objectives: dict[str, Callable[[], torch.Tensor]] = {
        "C_total": lambda: _objective_for(source_control, pixels, control_rows, scale, power_reference,
                                           gain, weights, structural_weight, "C", None),
    }
    treatment_objectives: dict[str, Callable[[], torch.Tensor]] = {
        "V_balanced_rec": lambda: weights["rec"] * kappa * _loss_terms(
            treatment_model, pixels, treatment_rows, scale, power_reference, gain, "V")[0]["balanced_rec"],
        "V_total": lambda: _objective_for(treatment_model, pixels, treatment_rows, scale, power_reference,
                                            gain, weights, structural_weight, "V", kappa),
    }
    finite_differences = {
        "C_total": _direction_check(source_control, source_objectives["C_total"], "C_total"),
        "V_balanced_rec": _direction_check(treatment_model, treatment_objectives["V_balanced_rec"],
                                           "V_balanced_rec"),
        "V_total": _direction_check(treatment_model, treatment_objectives["V_total"], "V_total"),
    }
    evaluator_state = base._state_snapshot(evaluator)
    ec_state = base._state_snapshot(ec_evaluator)
    branch_preflights: dict[str, Any] = {}
    for branch in BRANCHES:
        staging = staging_parent / f"{branch}_{uuid.uuid4().hex[:8]}"
        staging.mkdir(parents=False, exist_ok=False)
        branch_preflights[branch] = _preflight_branch(
            branch, source, config, config_path, source_path, pixels, schedule, device,
            scale, power_reference, gain, weights,
            structural_weight, kappa if branch == "V" else None,
            staging, grid,
        )
        _require(base._assert_close(evaluator_state, base._state_snapshot(evaluator),
                                    f"{branch}_ED_after_preflight") == 0.0
                 and base._assert_close(ec_state, base._state_snapshot(ec_evaluator),
                                        f"{branch}_EC_after_preflight") == 0.0,
                 "frozen evaluator changed during preflight")
    torch.cuda.set_rng_state_all([state.cpu() for state in source["cuda_rng"]])
    torch.random.set_rng_state(source["cpu_rng"])
    return {
        "status": "READY",
        "source_window": source_window,
        "independent_readout": independent_readout,
        "balance_calibration": calibration,
        "fixed_lambda_ssim": structural_weight,
        "fixed_kappa": kappa,
        "frozen_loss_weights": weights,
        "source_adam_updates": SOURCE_UPDATES,
        "source_schedule_verification_range": [SOURCE_SCHEDULE_START, SOURCE_UPDATES],
        "verified_source_schedule_rows": int(source["schedule"].shape[0]),
        "continuation_schedule_rows": int(schedule[SOURCE_UPDATES:].shape[0]),
        "finite_difference": finite_differences,
        "branch_preflight": branch_preflights,
        "evaluator_frozen_and_excluded_from_loss": True,
        "official_test_reads": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
        "scientific_updates_before_execution": 0,
    }, structural_weight, kappa


def _run_branch(
    branch: str, run_dir: Path, run_id: str, source: dict[str, Any], config: dict[str, Any],
    pixels: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, dev_indices: torch.Tensor,
    schedule: torch.Tensor, references: dict[str, Any], evaluator: torch.nn.Module,
    ec_evaluator: ThirdConfirmationEvaluator,
    device: torch.device, resume_checkpoint: Path | None, elapsed_before: float,
) -> tuple[dict[str, Any], float]:
    branch_dir = run_dir / branch
    references_for_branch = {
        "scale": source["scale"], "power_reference": source["power_reference"],
        "camera_gain": source["camera_gain"], "weights": source["weights"],
        "structural_weight": references["structural_weight"],
        "kappa": references["kappa"] if branch == "V" else None,
    }
    _require(branch_dir.is_dir(), f"branch output directory is missing: {branch_dir}")
    completed_summary = branch_dir / "endpoint_summary.json"
    if completed_summary.is_file():
        summary = json.loads(completed_summary.read_text(encoding="utf-8"))
        _require(summary.get("branch") == branch and summary.get("measurement_complete") is True,
                 f"saved {branch} endpoint summary is incomplete")
        endpoint = torch.load(branch_dir / "checkpoint_step_010000.pt", map_location="cpu", weights_only=False)
        _validate_continuation_checkpoint(endpoint, branch, run_id, source, config,
                                          schedule[SOURCE_UPDATES:], references_for_branch)
        branch_elapsed = float(summary.get("training", {}).get("branch_elapsed_seconds", 0.0))
        _require(math.isfinite(branch_elapsed) and branch_elapsed >= 0.0,
                 f"saved {branch} branch duration is invalid")
        return summary, elapsed_before + branch_elapsed

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
        branch_elapsed_before = float(saved_elapsed)
        saved_training_elapsed = checkpoint.get("training_elapsed_seconds_continuation")
        _require(isinstance(saved_training_elapsed, (float, int))
                 and math.isfinite(float(saved_training_elapsed)) and float(saved_training_elapsed) >= 0.0,
                 "resume training elapsed time is invalid")
        training_elapsed_before = float(saved_training_elapsed)
    else:
        branch_elapsed_before = 0.0
        training_elapsed_before = 0.0
        model, optimizer = _new_branch(source, config, device, grid)
        torch.cuda.set_rng_state_all([state.cpu() for state in source["cuda_rng"]])
        torch.random.set_rng_state(source["cpu_rng"])
        history = []
        start = 0
        initial = _checkpoint_payload(
            model, optimizer, branch, 0, run_id, source, config, schedule[SOURCE_UPDATES:],
            source["scale"], source["power_reference"], source["camera_gain"], source["weights"],
            references_for_branch["structural_weight"], references_for_branch["kappa"],
            history, branch_elapsed_before, training_elapsed_before,
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
    branch_started = time.perf_counter()
    training_started = time.perf_counter()
    for cursor in range(start, CONTINUATION_UPDATES):
        segment_step = cursor + 1
        global_step = SOURCE_UPDATES + segment_step
        loss = _train_update(
            model, optimizer, pixels, schedule[SOURCE_UPDATES + cursor], device,
            source["scale"].to(device), source["power_reference"].to(device), source["camera_gain"].to(device),
            source["weights"], branch,
            references_for_branch["structural_weight"], references_for_branch["kappa"],
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
                "elapsed_seconds_this_process": time.perf_counter() - training_started,
                "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
            })
        if segment_step % CHECKPOINT_INTERVAL == 0:
            payload = _checkpoint_payload(
                model, optimizer, branch, segment_step, run_id, source, config,
                schedule[SOURCE_UPDATES:], source["scale"], source["power_reference"],
                source["camera_gain"], source["weights"], references_for_branch["structural_weight"],
                references_for_branch["kappa"], history, branch_elapsed_before + time.perf_counter() - branch_started,
                training_elapsed_before + time.perf_counter() - training_started,
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
                references_for_branch["kappa"], history, branch_elapsed_before + time.perf_counter() - branch_started,
                training_elapsed_before + time.perf_counter() - training_started,
            )
            _save_branch_checkpoint(branch_dir, payload, segment_step)
    _require(base._optimizer_update_count(optimizer, model) == TOTAL_UPDATES,
             f"{branch} Adam did not finish at global step 45,000")
    training_elapsed_this_process = time.perf_counter() - training_started
    _require(base._assert_close(evaluator_state, base._state_snapshot(evaluator),
                                f"{branch}_evaluator_after_training") == 0.0,
             f"{branch} frozen E_D changed during training")
    _require(base._assert_close(model_static_state,
                                {key: value for key, value in base._state_snapshot(model).items()
                                 if key not in {"runtime.encoder.phase", "runtime.generator.phase"}},
                                f"{branch}_frozen_model_state_after_training") == 0.0,
             f"{branch} frozen optical state changed during training")
    endpoint_model_state = base._state_snapshot(model)
    endpoint_batch = image_batch(pixels, schedule[SOURCE_UPDATES], "small", device)[:1]
    endpoint_readout = base._small_producer_consumer_check(model, endpoint_batch)
    future_schedule = schedule[SOURCE_UPDATES:]
    endpoint_grid, endpoint_window = base._window_check(
        "small", pixels, labels, future_schedule, config, device, model=model,
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
        references_for_branch["kappa"], history, branch_elapsed_before + time.perf_counter() - branch_started,
        training_elapsed_before + training_elapsed_this_process,
    )
    _save_branch_checkpoint(branch_dir, endpoint, CONTINUATION_UPDATES)
    atomic_torch_save(branch_dir / "terminal_model.pt", {
        "schema": "lightgen_small_gaussian_foreground_balance_terminal.v1",
        "task_id": TASK_ID,
        "run_id": run_id,
        "branch": branch,
        "global_step": TOTAL_UPDATES,
        "model_identity": source["model_identity"],
        "model": endpoint_model_state,
    })
    evaluation_started = time.perf_counter()
    summary = _evaluate_branch(
        branch, run_id, branch_dir, model, evaluator, ec_evaluator, pixels, labels, train_indices, dev_indices,
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
    evaluation_elapsed = time.perf_counter() - evaluation_started
    branch_elapsed = branch_elapsed_before + time.perf_counter() - branch_started
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
        "branch": branch,
        "structural_weight": references_for_branch["structural_weight"],
        "kappa": references_for_branch["kappa"],
        "adam_update_count": base._optimizer_update_count(optimizer, model),
        "history": history,
        "elapsed_seconds_training_this_process": training_elapsed_this_process,
        "elapsed_seconds_training_cumulative": training_elapsed_before + training_elapsed_this_process,
        "evaluation_elapsed_seconds": evaluation_elapsed,
        "branch_elapsed_seconds": branch_elapsed,
        "cumulative_task_elapsed_seconds": elapsed_before + branch_elapsed,
        "seconds_per_scientific_update": training_elapsed_this_process / CONTINUATION_UPDATES,
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
        "device": torch.cuda.get_device_name(device),
        "plane_power_density_logging": plane_density_logging,
    }
    atomic_write_json(branch_dir / "endpoint_summary.json", summary)
    atomic_write_json(branch_dir / "training_history.json", history)
    return summary, elapsed_before + branch_elapsed


def _aggregate_report(
    run_id: str, source_report: dict[str, Any], source: dict[str, Any], config: dict[str, Any],
    preflight: dict[str, Any], branch_c: dict[str, Any], branch_v: dict[str, Any],
    elapsed_seconds: float, run_dir: Path,
) -> dict[str, Any]:
    c_rows = branch_c["reconstruction"]["per_image"]
    v_rows = branch_v["reconstruction"]["per_image"]
    metrics = ("mse", "psnr_db", "ssim", "native_mse_unclipped",
               "native_stroke_weighted_mse", "native_background_weighted_mse",
               "proxy_clipping_fraction")
    bootstrap = {metric: _stratified_paired_bootstrap(c_rows, v_rows, metric) for metric in metrics}
    ssim_ci = bootstrap["ssim"]["ci_95"]
    human_status = "AWAITING_HUMAN_RATINGS"
    if ssim_ci[0] > 0.0:
        ssim_interpretation = "V_SSIM_IMPROVEMENT_SUPPORTED_ON_FIXED_DEV"
    elif ssim_ci[1] < 0.0:
        ssim_interpretation = "V_SSIM_WORSE_ON_FIXED_DEV"
    else:
        ssim_interpretation = "V_SSIM_DIFFERENCE_UNCLEAR_ON_FIXED_DEV"
    source_endpoint = source_report["endpoints"]["treatment"]
    source_reconstruction = source_endpoint["reconstruction"]
    source_generation = source_endpoint["generation"]
    _require(source_endpoint.get("global_step") == SOURCE_UPDATES
             and source_endpoint.get("measurement_complete") is True
             and source_report.get("status") == "VALID_FAILED",
             "the source B historical failed baseline identity or status changed")
    endpoints = {
        "source_B": {
            "branch": "source_B",
            "global_step": SOURCE_UPDATES,
            "reconstruction": source_reconstruction,
            "generation": source_generation,
            "status": source_report["status"],
            "measurement_complete": source_endpoint["measurement_complete"],
        },
        "C": branch_c,
        "V": branch_v,
    }
    arm_delta: dict[str, Any] = {}
    for name, record in (("C", branch_c), ("V", branch_v)):
        recon, generated = record["reconstruction"], record["generation"]
        arm_delta[name] = {
            "psnr_db_from_source_B": recon["psnr_db_mean"] - source_reconstruction["psnr_db_mean"],
            "ssim_from_source_B": recon["ssim_mean"] - source_reconstruction["ssim_mean"],
            "generation_ed_macro_recall_from_source_B": generated["ed_macro_recall"] - source_generation["macro_recall"],
            "generation_recall_by_class_from_source": [
                generated["ed_recall_by_class"][i] - source_generation["recall_by_class"][i] for i in range(10)
            ],
        }
    complete = all(record.get("measurement_complete") is True for record in (branch_c, branch_v))
    output_bytes = base._directory_bytes(run_dir)
    report = {
        "schema": "lightgen_small_gaussian_foreground_balance_ab_report.v1",
        "task_id": TASK_ID,
        "status": human_status if complete else "MEASUREMENT_INCOMPLETE",
        "measurement_complete": complete,
        "evidence_tier": "diagnostic_only",
        "run_id": run_id,
        "source": {
            "task_id": SOURCE_TASK_ID,
            "run_id": source["run_id"],
            "checkpoint_step": SOURCE_UPDATES,
            "branch": "treatment (old B)",
            "historical_status_preserved": source_report["status"],
            "model_identity": source["model_identity"],
        },
        "scientific_question": "With the inherited SSIM structural term retained, does equalizing target-weighted stroke and background pixel error improve human readability of native 200x200 simulated CMOS generation views?",
        "training": {
            "branches": {
                "C": {"objective": "inherited old-B BOGT plus fixed SSIM", "updates": CONTINUATION_UPDATES},
                "V": {"objective": "C objective with weighted native reconstruction replaced by frozen kappa times balanced stroke/background error",
                      "updates": CONTINUATION_UPDATES,
                      "kappa": preflight["fixed_kappa"]},
            },
            "source_global_step": SOURCE_UPDATES,
            "endpoint_global_step_each_branch": TOTAL_UPDATES,
            "branch_order": list(BRANCHES),
            "logical_batch_size": BATCH_SIZE,
            "paired_schedule_seed": SEED,
            "source_schedule_rows_30000_to_35000_verified": True,
            "same_post_source_schedule_rows_35000_to_45000": True,
            "source_lambda_ssim": preflight["fixed_lambda_ssim"],
            "camera_gain_bogt_weights_modal_scale_and_power_reference_frozen": True,
            "optimizer_source_state_preserved": True,
            "evaluator_frozen_and_excluded_from_loss": {"E_D": True, "E_C": True},
            "optical_geometry_status": "PROJECT_GEOMETRY_UNMEASURED",
        },
        "preflight": preflight,
        "endpoints": endpoints,
        "change_from_source": arm_delta,
        "paired_comparison": {
            "V_minus_C_reconstruction": bootstrap,
            "ssim_interpretation": ssim_interpretation,
            "generation_classifier_and_distribution_differences": {
                "E_D_macro_recall": branch_v["generation"]["ed_macro_recall"] - branch_c["generation"]["ed_macro_recall"],
                "E_C_macro_recall": branch_v["generation"]["ec_macro_recall"] - branch_c["generation"]["ec_macro_recall"],
                "E_D_recall_by_class": [branch_v["generation"]["ed_recall_by_class"][i]
                                         - branch_c["generation"]["ed_recall_by_class"][i] for i in range(10)],
                "E_C_recall_by_class": [branch_v["generation"]["ec_recall_by_class"][i]
                                         - branch_c["generation"]["ec_recall_by_class"][i] for i in range(10)],
                "proxy_clipping_fraction_by_class": [
                    branch_v["generation"]["proxy_clipping_fraction_by_class"][i]
                    - branch_c["generation"]["proxy_clipping_fraction_by_class"][i] for i in range(10)],
                "generated_to_dev_shape_ratio_by_class": [
                    branch_v["generation"]["generated_to_dev_tv_ratio"][i]
                    - branch_c["generation"]["generated_to_dev_tv_ratio"][i] for i in range(10)],
                "cohort_alignment_note": "same sampling seeds; proposals retained after power rejection are reported by identity and are not treated as strict pairs",
            },
        },
        "decision": {
            "primary_endpoint": "human requested-class readability on 200 independent generated native 200x200 cards after review-quality checks",
            "human_review_status": human_status if complete else "BLOCKED_INCOMPLETE_AUTOMATIC_MEASUREMENTS",
            "legacy_quality_thresholds": {"C": branch_c["quality_gate"], "V": branch_v["quality_gate"]},
            "human_readability_conclusion": "PENDING_USER_RATINGS",
            "no_model_adoption_or_hardware_claim": True,
        },
        "elapsed_seconds_total": elapsed_seconds,
        "device": branch_c["training"]["device"],
        "output": {
            "formal_result_budget_bytes": config["output"]["formal_result_budget_bytes"],
            "output_bytes_before_report_finalization": output_bytes,
            "within_budget": output_bytes <= int(config["output"]["formal_result_budget_bytes"]),
        },
        "artifacts": {
            "C_endpoint": "C/checkpoint_step_010000.pt",
            "V_endpoint": "V/checkpoint_step_010000.pt",
            "paired_metrics": "paired_comparison.json",
            "blind_review": "blind_review.html",
            "machine_results_html": "machine_results.html",
        },
    }
    if output_bytes > int(config["output"]["formal_result_budget_bytes"]):
        report["status"] = "MEASUREMENT_INCOMPLETE"
        report["measurement_complete"] = False
    return report


def _rating_value(value: Any, card_id: str) -> int | str:
    _require(value == "unreadable" or (isinstance(value, int) and not isinstance(value, bool)
                                       and 0 <= value <= 9),
             f"rating for {card_id} must be an integer 0-9 or 'unreadable'", ValueError)
    return value


def _validate_review_inputs(run_dir: Path, ratings_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    machine_path = run_dir / "machine_report.json"
    key_path = run_dir / "blind_review_key.json"
    manifest_path = run_dir / "blind_review_public_manifest.json"
    page_path = run_dir / "blind_review.html"
    for path in (machine_path, key_path, manifest_path, page_path, ratings_path):
        _require(path.is_file(), f"human-review finalization input is missing: {path}", FileNotFoundError)
    machine = json.loads(machine_path.read_text(encoding="utf-8"))
    key = json.loads(key_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ratings = json.loads(ratings_path.read_text(encoding="utf-8"))
    _require(machine.get("task_id") == TASK_ID and machine.get("status") == "AWAITING_HUMAN_RATINGS"
             and machine.get("measurement_complete") is True
             and all((run_dir / branch / "endpoint_summary.json").is_file() for branch in BRANCHES),
             "automatic branch measurements are incomplete or already finalized")
    review_id = key.get("review_id")
    _require(key.get("schema") == REVIEW_SCHEMA and key.get("task_id") == TASK_ID
             and key.get("registered_card_count") == 240 and isinstance(review_id, str)
             and manifest.get("schema") == REVIEW_SCHEMA and manifest.get("task_id") == TASK_ID
             and manifest.get("review_id") == review_id and manifest.get("card_count") == 240,
             "blind-review key and public manifest identity or card count changed")
    _require(ratings.get("schema") == RATINGS_SCHEMA and ratings.get("task_id") == TASK_ID
             and ratings.get("review_id") == review_id and isinstance(ratings.get("cards"), list),
             "submitted ratings belong to another task or review")
    cards = key.get("cards")
    _require(isinstance(cards, list) and len(cards) == 240, "hidden review key is incomplete")
    expected_ids = [f"R{index:04d}" for index in range(1, 241)]
    card_ids = [card.get("card_id") for card in cards]
    _require(card_ids == expected_ids, "review card IDs are missing, duplicated, or out of order")
    rating_cards = ratings["cards"]
    _require(all(isinstance(card, dict) and isinstance(card.get("card_id"), str)
                 for card in rating_cards), "ratings contain a malformed card record", ValueError)
    rating_ids = [card.get("card_id") for card in rating_cards if isinstance(card, dict)]
    _require(len(rating_cards) == 240 and len(set(rating_ids)) == 240
             and set(rating_ids) == set(expected_ids),
             "ratings must contain every blind card exactly once", ValueError)
    submitted_by_id = {card["card_id"]: card for card in rating_cards}
    rating_by_id = {card_id: _rating_value(submitted_by_id[card_id].get("rating"), card_id)
                    for card_id in expected_ids}
    kinds = {kind: [card for card in cards if card.get("kind") == kind]
             for kind in ("generated", "duplicate", "real_anchor")}
    _require([len(kinds[kind]) for kind in ("generated", "duplicate", "real_anchor")] == [200, 20, 20],
             "review card source counts differ from the registered design")
    generated_keys = [(card.get("branch"), card.get("requested_label"), card.get("sample_index"),
                       card.get("proposal_id")) for card in kinds["generated"]]
    _require(len(set(generated_keys)) == 200 and all(branch in BRANCHES and isinstance(label, int)
             and 0 <= label <= 9 for branch, label, _, _ in generated_keys),
             "unique generated card identity or requested class is duplicated or malformed")
    for branch in BRANCHES:
        for label in range(10):
            subset = [card for card in kinds["generated"]
                      if card.get("branch") == branch and card.get("requested_label") == label]
            _require(len(subset) == 10, f"blind generated cards for {branch} class {label} are incomplete")
    card_by_id = {card["card_id"]: card for card in cards}
    group_map: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        group = card.get("duplicate_group")
        if group is not None:
            group_map.setdefault(group, []).append(card)
    _require(len(group_map) == 20 and all(len(group) == 2 for group in group_map.values()),
             "hidden duplicate groups are incomplete")
    duplicate_consistency = 0
    duplicate_pairs: list[dict[str, Any]] = []
    for group_name, group in sorted(group_map.items()):
        originals = [card for card in group if card.get("kind") == "generated"]
        copies = [card for card in group if card.get("kind") == "duplicate"]
        _require(len(originals) == 1 and len(copies) == 1,
                 f"hidden duplicate group {group_name} is malformed")
        original, duplicate = originals[0], copies[0]
        identity_fields = ("branch", "requested_label", "sample_index", "proposal_id")
        _require(all(original.get(field) == duplicate.get(field) for field in identity_fields),
                 f"hidden duplicate group {group_name} does not point to the same image")
        original_rating = rating_by_id[original["card_id"]]
        duplicate_rating = rating_by_id[duplicate["card_id"]]
        same = original_rating == duplicate_rating
        duplicate_consistency += int(same)
        duplicate_pairs.append({"duplicate_group": group_name, "original_card_id": original["card_id"],
                                "duplicate_card_id": duplicate["card_id"], "consistent": same})
    anchors = kinds["real_anchor"]
    anchor_correct = sum(int(rating_by_id[card["card_id"]] == card.get("target_label"))
                         for card in anchors)
    anchor_counts = [sum(card.get("target_label") == label for card in anchors) for label in range(10)]
    _require(anchor_counts == [2] * 10, "real anchor labels are not two per class")
    card_root = (run_dir / "review_cards").resolve()
    page_text = page_path.read_text(encoding="utf-8")
    declarations = re.search(
        r'const TASK_ID=("(?:\\.|[^"\\])*")\s*,\s*REVIEW_ID=("(?:\\.|[^"\\])*")'
        r'\s*,\s*CARDS=',
        page_text,
    )
    _require(declarations is not None and "image-rendering:pixelated" in page_text,
             "blind-review page task identity or fixed pixel display is missing")
    page_task_id = json.loads(declarations.group(1))
    page_review_id = json.loads(declarations.group(2))
    page_cards, _ = json.JSONDecoder().raw_decode(page_text, declarations.end())
    expected_page_cards = [{"card_id": card["card_id"], "image": card["image"]} for card in cards]
    _require(page_task_id == TASK_ID and page_review_id == review_id,
             "blind-review page task or review identity differs from the registered manifest")
    _require(page_cards == expected_page_cards,
             "blind-review page image mapping differs from the hidden key")
    for card in cards:
        relative = card.get("image")
        _require(isinstance(relative, str) and relative.startswith("review_cards/"),
                 f"blind image path for {card['card_id']} is malformed")
        image_path = (run_dir / relative).resolve()
        _require(image_path.is_relative_to(card_root) and image_path.is_file(),
                 f"blind image for {card['card_id']} is missing or escapes the review directory")
        with Image.open(image_path) as image:
            _require(image.mode == "L" and image.size == (200, 200),
                     f"blind image {card['card_id']} changed from native 200x200 grayscale")
    for group in group_map.values():
        first = (run_dir / group[0]["image"]).read_bytes()
        second = (run_dir / group[1]["image"]).read_bytes()
        _require(first == second, "hidden duplicate PNGs are not pixel-identical")
    return machine, key, {
        "ratings": ratings,
        "cards": cards,
        "card_by_id": card_by_id,
        "duplicate_consistency_count": duplicate_consistency,
        "duplicate_pairs": duplicate_pairs,
        "anchor_correct_count": anchor_correct,
        "anchor_count": len(anchors),
        "rating_by_id": rating_by_id,
    }


def _human_class_metrics(cards: list[dict[str, Any]], ratings: dict[str, int | str], branch: str) -> dict[str, Any]:
    by_class: list[dict[str, Any]] = []
    success_arrays: list[np.ndarray] = []
    unreadable_arrays: list[np.ndarray] = []
    other_digit_arrays: list[np.ndarray] = []
    readable_arrays: list[np.ndarray] = []
    for label in range(10):
        selected = [card for card in cards if card.get("kind") == "generated"
                    and card.get("branch") == branch and card.get("requested_label") == label]
        _require(len(selected) == 10, f"human sample {branch} class {label} is incomplete")
        values = [ratings[card["card_id"]] for card in selected]
        success = np.asarray([value == label for value in values], dtype=np.float64)
        unreadable = np.asarray([value == "unreadable" for value in values], dtype=np.float64)
        other = np.asarray([isinstance(value, int) and value != label for value in values], dtype=np.float64)
        readable = 1.0 - unreadable
        numeric_correct = float(success.sum())
        readable_count = float(readable.sum())
        by_class.append({
            "requested_label": label, "count": 10, "ratings": values,
            "matches_requested_fraction": float(success.mean()),
            "readable_fraction": float(readable.mean()),
            "other_digit_fraction": float(other.mean()),
            "matches_requested_among_readable": None if readable_count == 0.0
            else numeric_correct / readable_count,
        })
        success_arrays.append(success)
        unreadable_arrays.append(unreadable)
        other_digit_arrays.append(other)
        readable_arrays.append(readable)
    return {
        "branch": branch, "unique_generated_population": 100,
        "matches_requested_class_rate_equal_class": float(np.mean([values.mean() for values in success_arrays])),
        "readable_rate_equal_class": float(np.mean([values.mean() for values in readable_arrays])),
        "other_digit_rate_equal_class": float(np.mean([values.mean() for values in other_digit_arrays])),
        "matches_requested_among_readable_rate_equal_class": float(np.mean([
            row["matches_requested_among_readable"] for row in by_class
            if row["matches_requested_among_readable"] is not None
        ])) if any(row["matches_requested_among_readable"] is not None for row in by_class) else None,
        "per_class": by_class,
        "_success_arrays": success_arrays,
        "_unreadable_arrays": unreadable_arrays,
        "_other_digit_arrays": other_digit_arrays,
    }


def _human_effect_bootstrap(
    c_metrics: dict[str, Any], v_metrics: dict[str, Any], seed: int,
    resamples: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    deltas = np.empty(resamples, dtype=np.float64)
    c_arrays = c_metrics["_success_arrays"]
    v_arrays = v_metrics["_success_arrays"]
    for index in range(resamples):
        c_values = [values[rng.integers(0, values.size, size=values.size)].mean() for values in c_arrays]
        v_values = [values[rng.integers(0, values.size, size=values.size)].mean() for values in v_arrays]
        deltas[index] = float(np.mean(v_values) - np.mean(c_values))
    observed = (v_metrics["matches_requested_class_rate_equal_class"]
                - c_metrics["matches_requested_class_rate_equal_class"])
    return {
        "estimand": "V minus C equal-class mean fraction of generated images rated as the requested class",
        "observed_difference": observed,
        "ci_95": [float(value) for value in np.quantile(deltas, [0.025, 0.975])],
        "resamples": resamples,
        "seed": seed,
        "resampling": "independent bootstrap within branch and requested class; 10 cards per class per branch",
    }


def _save_raw_json(path: Path, source_path: Path) -> None:
    payload = source_path.read_bytes()
    temporary = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _write_revealed_gallery(run_dir: Path) -> int:
    gallery_dir = run_dir / "revealed_gallery"
    staging_dir = run_dir / f".revealed_gallery_staging_{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=False, exist_ok=False)
    gallery_rows: list[dict[str, Any]] = []
    source = torch.load(run_dir / "source_B_display_examples.pt", map_location="cpu", weights_only=False)
    sources: dict[str, dict[str, Any]] = {"source_B": {"data": source}}
    for branch in BRANCHES:
        dev = torch.load(run_dir / branch / "dev_examples_step_045000.pt", map_location="cpu", weights_only=False)
        generation = torch.load(run_dir / branch / "generated_cohort.pt", map_location="cpu", weights_only=False)
        sources[branch] = {"dev": dev, "generation": generation}

    for branch, bundle in sources.items():
        if branch == "source_B":
            data = bundle["data"]
            dev_records = []
            for index in range(20):
                dev_records.append({
                    "label": int(data["dev_labels"][index]), "row": int(data["dev_rows"][index]),
                    "target": data["dev_targets_native"][index], "native": data["dev_views_native"][index],
                    "view": data["dev_views_28"][index],
                })
            gen_records = []
            for index in range(100):
                gen_records.append({
                    "label": int(data["generation_labels"][index]),
                    "proposal_id": int(data["generation_proposal_ids"][index]),
                    "native": data["generation_views_native"][index],
                    "view": data["generation_views_28"][index],
                })
        else:
            dev = bundle["dev"]
            generation = bundle["generation"]
            dev_records = [{
                "label": int(dev["labels"][index]), "row": int(dev["rows"][index]),
                "target": dev["native_targets"][index], "native": dev["native_images"][index],
                "view": dev["images"][index],
            } for index in range(20)]
            requested = generation["requested_labels"]
            native_lookup = {int(index): image for index, image in zip(
                generation["native_capture_indices"].tolist(),
                generation["native_capture_images"], strict=True,
            )}
            selected = [int(torch.nonzero(requested == label, as_tuple=False).flatten()[offset])
                        for label in range(10) for offset in range(10)]
            gen_records = []
            for index in selected:
                _require(index in native_lookup, f"{branch} revealed gallery native image {index} is missing")
                gen_records.append({
                    "label": int(requested[index]), "proposal_id": int(generation["proposal_ids"][index]),
                    "native": native_lookup[index], "view": generation["images"][index],
                })
        for kind, records in (("dev", dev_records), ("generation", gen_records)):
            expected = 20 if kind == "dev" else 100
            _require(len(records) == expected, f"{branch} {kind} revealed gallery is incomplete")
            for index, record in enumerate(records):
                prefix = f"{branch}_{kind}_{index:03d}"
                native_name, view_name = prefix + "_native.png", prefix + "_28x28.png"
                _save_gray_png(record["native"], staging_dir / native_name)
                _save_gray_png(record["view"], staging_dir / view_name)
                target_name = None
                if kind == "dev":
                    target_name = prefix + "_target_native.png"
                    _save_gray_png(record["target"], staging_dir / target_name)
                gallery_rows.append({
                    "branch": branch, "kind": kind, "index": index, "label": record["label"],
                    "mnist_row": record.get("row"), "proposal_id": record.get("proposal_id"),
                    "native": f"revealed_gallery/{native_name}", "view": f"revealed_gallery/{view_name}",
                    "target": None if target_name is None else f"revealed_gallery/{target_name}",
                })
    data_json = json.dumps(gallery_rows, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Gaussian小架构训练后图像对照</title><style>body{{font:15px system-ui,sans-serif;max-width:1200px;margin:1rem auto;padding:0 1rem;color:#18212b}}.row{{display:flex;gap:1rem;align-items:flex-start;border-bottom:1px solid #ddd;padding:.7rem 0}}img{{width:200px;height:200px;object-fit:contain;background:#fff;border:1px solid #aaa;image-rendering:pixelated}}.images{{display:flex;gap:.4rem}}.item{{text-align:center}}select,button{{font:inherit;padding:.4rem;margin-right:.4rem}}small{{display:block;color:#555}}</style>
<h1>训练后图像对照（揭盲）</h1><p>原生图固定为200×200灰度；28×28视图以像素化方式显示在同尺寸画框中。所有图像使用同一[0,1]读出范围，未逐图拉伸。</p>
<label>分支 <select id='branch'><option value='all'>全部</option><option>source_B</option><option>C</option><option>V</option></select></label>
<label>类型 <select id='kind'><option value='all'>全部</option><option>dev</option><option>generation</option></select></label>
<label>类别 <select id='label'><option value='all'>全部</option>{''.join(f'<option>{i}</option>' for i in range(10))}</select></label><div id='gallery'></div>
<script>const DATA={data_json};function render(){{const b=document.getElementById('branch').value,k=document.getElementById('kind').value,l=document.getElementById('label').value;const root=document.getElementById('gallery');root.innerHTML='';for(const r of DATA){{if(b!=='all'&&r.branch!==b)continue;if(k!=='all'&&r.kind!==k)continue;if(l!=='all'&&String(r.label)!==l)continue;const row=document.createElement('div');row.className='row';const title=document.createElement('div');title.textContent=`${{r.branch}} · ${{r.kind==='dev'?'DEV目标':'生成目标'}} 类别 ${{r.label}} · ${{r.mnist_row??('proposal '+r.proposal_id)}}`;row.append(title);const imgs=document.createElement('div');imgs.className='images';for(const [name,src] of [['目标',r.target],['原生200×200',r.native],['池化28×28',r.view]]){{if(!src)continue;const item=document.createElement('div');item.className='item';const im=document.createElement('img');im.src=src;const cap=document.createElement('small');cap.textContent=name;item.append(im,cap);imgs.append(item);}}row.append(imgs);root.append(row);}}}}for(const id of ['branch','kind','label'])document.getElementById(id).onchange=render;render();</script></html>"""
    if gallery_dir.exists():
        _require(gallery_dir.resolve().parent == run_dir.resolve(),
                 "existing revealed gallery escaped its task output directory")
        shutil.rmtree(gallery_dir)
    os.replace(staging_dir, gallery_dir)
    (run_dir / "results.html").write_text(page, encoding="utf-8")
    return len(gallery_rows)


def _finalize_ratings(run_dir: Path, ratings_path: Path, config: dict[str, Any]) -> None:
    started = time.perf_counter()
    run_dir = run_dir.resolve()
    ratings_path = ratings_path.resolve()
    _require(run_dir.is_dir(), f"explicit attempt directory is missing: {run_dir}", FileNotFoundError)
    _require(not (run_dir / "final_report.json").exists(), "this review has already been finalized")
    machine, key, validated = _validate_review_inputs(run_dir, ratings_path)
    identity_record = json.loads((run_dir / "run_identity.json").read_text(encoding="utf-8"))
    identity = ResearchRunIdentity.from_mapping(identity_record["identity"])
    authorization_id = f"{TASK_ID}:seed:{SEED}:preflight_and_execution"
    _require(identity.task_id == TASK_ID and identity.config_identity == config
             and identity_record.get("run_id") == machine.get("run_id"),
             "human finalization run identity differs from the automatic measurement")
    receipt = PreflightReceipt.from_mapping(
        json.loads((run_dir / "preflight_ready.json").read_text(encoding="utf-8")))
    control = ResearchRunControl(identity, authorization_id=authorization_id)
    control.mark_preflight_ready(receipt)
    control.resume_scientific_attempt(run_dir, "foreground_balance_ab_seed_2026101501", receipt)
    cards = validated["cards"]
    ratings = validated["rating_by_id"]
    c_metrics = _human_class_metrics(cards, ratings, "C")
    v_metrics = _human_class_metrics(cards, ratings, "V")
    review_cfg = config["human_review"]
    effect = _human_effect_bootstrap(c_metrics, v_metrics, int(review_cfg["bootstrap_seed"]),
                                     int(review_cfg["bootstrap_resamples"]))
    duplicate_min = int(review_cfg["duplicate_consistency_min"])
    anchor_min = int(review_cfg["anchor_label_match_min"])
    duplicate_ok = validated["duplicate_consistency_count"] >= duplicate_min
    anchors_ok = validated["anchor_correct_count"] >= anchor_min
    qc_passed = duplicate_ok and anchors_ok
    low, high = effect["ci_95"]
    if not qc_passed:
        conclusion = "REVIEW_INCONCLUSIVE"
        result_status = "REVIEW_INCONCLUSIVE"
    else:
        if low > 0.0:
            conclusion = "HUMAN_READABILITY_IMPROVEMENT_SUPPORTED"
        elif high < 0.0:
            conclusion = "HUMAN_READABILITY_WORSE"
        else:
            conclusion = "HUMAN_READABILITY_INDEPENDENT_EFFECT_UNCLEAR"
        result_status = "COMPLETE_DIAGNOSTIC"

    human_review = {
        "status": "COMPLETE_DIAGNOSTIC" if qc_passed else "REVIEW_INCONCLUSIVE",
        "review_id": key["review_id"], "rating_count": len(validated["cards"]),
        "unique_generated_count": 200, "duplicate_count": 20, "real_anchor_count": 20,
        "quality_checks": {
            "duplicate_consistency": {
                "count": validated["duplicate_consistency_count"], "total": 20,
                "minimum": duplicate_min, "passed": duplicate_ok,
                "pairs": validated["duplicate_pairs"],
            },
            "real_anchor_label_match": {
                "count": validated["anchor_correct_count"], "total": 20,
                "minimum": anchor_min, "passed": anchors_ok,
            },
        },
        "primary_endpoint": "fraction rated as the requested class, counting unreadable as not matching",
        "C": {key: value for key, value in c_metrics.items() if not key.startswith("_")},
        "V": {key: value for key, value in v_metrics.items() if not key.startswith("_")},
        "V_minus_C_primary_effect": effect,
        "interpretation": conclusion,
        "limitations": [
            "the fixed sample contains 10 generated images per class per branch",
            "class 9 remains a small subgroup and is descriptive only",
            "the review estimates this saved cohort and does not establish physical CMOS readability",
        ],
        "elapsed_seconds_finalization": time.perf_counter() - started,
    }
    rating_rows: list[dict[str, Any]] = []
    card_rating = validated["rating_by_id"]
    for card in cards:
        rating = card_rating[card["card_id"]]
        requested = card.get("requested_label")
        rating_rows.append({
            "card_id": card["card_id"], "kind": card["kind"], "branch": card.get("branch"),
            "requested_label": requested, "human_rating": rating,
            "matches_requested": None if card["kind"] == "real_anchor" else
            (rating == requested if rating != "unreadable" else False),
            "readable": rating != "unreadable", "sample_index": card.get("sample_index"),
            "proposal_id": card.get("proposal_id"), "mnist_row": card.get("row"),
            "target_label": card.get("target_label"), "duplicate_group": card.get("duplicate_group"),
        })
    _write_rows_csv(run_dir / "human_ratings_unblinded.csv", rating_rows)
    raw_destination = run_dir / "human_ratings_submitted.json"
    _save_raw_json(raw_destination, ratings_path)
    machine["status"] = result_status
    machine["final_status"] = result_status
    machine["human_review_status"] = human_review["status"]
    machine["decision"]["human_review_status"] = human_review["status"]
    machine["decision"]["human_readability_conclusion"] = conclusion
    machine["human_review"] = human_review
    machine["artifacts"].update({
        "human_ratings": "human_ratings_submitted.json",
        "human_ratings_analysis": "human_ratings_unblinded.csv",
        "revealed_results_html": "results.html",
    })
    gallery_rows = _write_revealed_gallery(run_dir)
    machine["human_review"]["revealed_gallery_rows"] = gallery_rows
    output_bytes = base._directory_bytes(run_dir)
    machine["output"]["output_bytes_before_final_report"] = output_bytes
    machine["output"]["within_budget"] = output_bytes <= int(config["output"]["formal_result_budget_bytes"])
    if not machine["output"]["within_budget"]:
        machine["status"] = "MEASUREMENT_INCOMPLETE"
        machine["final_status"] = "MEASUREMENT_INCOMPLETE"
        machine["measurement_complete"] = False
        control.invalid_harness()
    else:
        control.complete()
    atomic_write_json(run_dir / "final_report.json", machine)
    atomic_write_json(run_dir / "report.json", machine)
    atomic_write_json(run_dir / "run_state.json", {
        "status": machine["status"], "scientific_status": machine["status"],
        "research_run_control_state": control.state,
        "human_review_conclusion": conclusion,
        "human_review_quality_checks_passed": qc_passed,
        "completed_global_step_each_branch": TOTAL_UPDATES,
        "measurement_complete": machine["measurement_complete"],
    })
    print(json.dumps({"status": machine["status"], "conclusion": conclusion,
                      "report": str(run_dir / "final_report.json")}, ensure_ascii=False), flush=True)


def _run(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    config = _read_config(config_path)
    if args.preflight_worker_dir is not None:
        _worker(args.preflight_worker_dir.resolve(), args.source_checkpoint.resolve(), config_path)
        return
    run_dir = args.run_dir.resolve()
    expected_run_dir = (ROOT / config["output"]["experiment_root"]).resolve()
    if args.ratings_file is None:
        _require(run_dir == expected_run_dir,
                 "run directory differs from the explicitly registered attempt path")
    else:
        _require(run_dir == expected_run_dir or (run_dir / "machine_report.json").is_file(),
                 "ratings finalization requires the registered attempt or an explicit complete result package")
    if args.ratings_file is not None:
        _require(not args.preflight_only and args.resume_checkpoint is None,
                 "ratings finalization cannot be combined with preflight or checkpoint resume")
        _finalize_ratings(run_dir, args.ratings_file.resolve(), config)
        return

    source_path = args.source_checkpoint.resolve()
    _require(source_path.is_file(), f"explicit source checkpoint is missing: {source_path}", FileNotFoundError)
    _require(torch.cuda.is_available(), "CUDA is required for the authorized Gaussian continuation")
    process_started = time.perf_counter()
    resuming = args.resume_checkpoint is not None
    if resuming:
        _require(run_dir.is_dir() and not (run_dir / "final_report.json").is_file(),
                 "resume requires an existing unfinalized attempt")
        resume_path = args.resume_checkpoint.resolve()
        _require(resume_path.is_file(), "explicit continuation resume checkpoint is missing", FileNotFoundError)
        try:
            resume_path.relative_to(run_dir)
        except ValueError as error:
            raise RuntimeError("resume checkpoint must be inside the explicit continuation attempt") from error
    else:
        _require(not run_dir.exists(), f"refusing to overwrite existing attempt: {run_dir}")

    source, references, source_report = _load_source(source_path, config)
    pixels, labels, train_indices, dev_indices = base._load_data(ROOT / config["data"]["training_file"])
    schedule = _continued_schedule(labels, train_indices, SEED)
    _require(torch.equal(schedule[SOURCE_SCHEDULE_START:SOURCE_UPDATES], source["schedule"]),
             "reconstructed 30,000--35,000 TRAIN schedule differs from the source checkpoint segment")
    _validate_schedule_rows(labels, train_indices, schedule, TOTAL_UPDATES)

    device = torch.device("cuda")
    evaluator = base._load_evaluator(ROOT / config["evaluation"]["evaluator_checkpoint"], device)
    ec_evaluator, ec_qualification = _load_ec_evaluator(ROOT / config["evaluation"]["ec_checkpoint"], device)
    _require(not evaluator.training and all(not parameter.requires_grad for parameter in evaluator.parameters()),
             "E_D is not frozen in eval mode")
    evaluator_state = base._state_snapshot(evaluator)
    ec_state = base._state_snapshot(ec_evaluator)
    run_id = uuid.uuid4().hex
    identity = _make_identity(config, source, run_id)
    authorization_id = f"{TASK_ID}:seed:{SEED}:preflight_and_execution"
    control = ResearchRunControl(identity, authorization_id=authorization_id)

    if resuming:
        saved_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        _require(saved_config == config, "resume config differs from the active attempt")
        control, identity, receipt = _load_ready_context(run_dir, config, source)
        run_id = identity.run_id
        preflight = json.loads((run_dir / "preflight.json").read_text(encoding="utf-8"))
        structural_weight = float(preflight["fixed_lambda_ssim"])
        kappa = float(preflight["fixed_kappa"])
        _require(receipt.status == "READY" and preflight.get("status") == "READY",
                 "resume attempt has no complete READY receipt")
        control.resume_scientific_attempt(run_dir, attempt_id="foreground_balance_ab_seed_2026101501",
                                          receipt=receipt)
    else:
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        stage_parent = run_dir.parent / f"preflight_foreground_balance_{uuid.uuid4().hex[:8]}"
        stage_parent.mkdir(parents=False, exist_ok=False)
        try:
            preflight, structural_weight, kappa = _run_preflight(
                source, references, config, config_path, source_path, pixels, labels, train_indices,
                schedule, device, evaluator, ec_evaluator, stage_parent,
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
            control.open_scientific_attempt(run_dir, attempt_id="foreground_balance_ab_seed_2026101501",
                                            receipt=receipt)
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
                "structural_weight": structural_weight, "kappa": kappa,
            })
            source_branch_dir = source_path.parent
            for src_name, dest_name in (("dev_preview.png", "source_dev_preview.png"),
                                        ("dev_target_preview.png", "source_dev_target_preview.png"),
                                        ("generation_preview.png", "source_generation_preview.png")):
                source_image = source_branch_dir / src_name
                _require(source_image.is_file(), f"source fixed-sample preview is missing: {source_image}")
                (run_dir / dest_name).write_bytes(source_image.read_bytes())
        finally:
            if stage_parent.exists():
                import shutil
                shutil.rmtree(stage_parent)

    _require(base._assert_close(evaluator_state, base._state_snapshot(evaluator), "E_D_before_science") == 0.0
             and base._assert_close(ec_state, base._state_snapshot(ec_evaluator), "E_C_before_science") == 0.0,
             "a frozen evaluator changed before science")
    start_resume = args.resume_checkpoint.resolve() if resuming else None
    if start_resume is not None:
        resume_payload = torch.load(start_resume, map_location="cpu", weights_only=False)
        _require(resume_payload.get("branch") in BRANCHES, "resume checkpoint branch is invalid")
        if resume_payload["branch"] == "V":
            c_summary_path = run_dir / "C" / "endpoint_summary.json"
            _require(c_summary_path.is_file(), "V resume requires a complete C branch endpoint first")
            c_summary = json.loads(c_summary_path.read_text(encoding="utf-8"))
            _require(c_summary.get("branch") == "C" and c_summary.get("measurement_complete") is True,
                     "V resume cannot consume an incomplete C endpoint")

    elapsed_total = 0.0
    endpoint_summaries: dict[str, dict[str, Any]] = {}
    references_for_run = {**references, "structural_weight": structural_weight, "kappa": kappa}
    for branch in BRANCHES:
        if branch == "V":
            _require((run_dir / "C" / "endpoint_summary.json").is_file(),
                     "V cannot start before C is fully evaluated")
        branch_resume = start_resume if start_resume is not None and start_resume.parent.name == branch else None
        summary, elapsed_total = _run_branch(
            branch, run_dir, run_id, source, config, pixels, labels, train_indices, dev_indices,
            schedule, references_for_run, evaluator, ec_evaluator, device,
            branch_resume, elapsed_total,
        )
        endpoint_summaries[branch] = summary
        atomic_write_json(run_dir / "progress_summary.json", {
            "branch": branch, "branch_endpoint_complete": True,
            "completed_global_step": TOTAL_UPDATES, "status": summary["status"],
        })

    _require(set(endpoint_summaries) == set(BRANCHES), "one or more continuation endpoints are missing")
    _require(base._assert_close(evaluator_state, base._state_snapshot(evaluator), "E_D_after_science") == 0.0
             and base._assert_close(ec_state, base._state_snapshot(ec_evaluator), "E_C_after_science") == 0.0,
             "a frozen evaluator changed during training or endpoint evaluation")
    source_gallery, source_gallery_samples = _make_source_gallery(source, source_path, config, pixels, device)
    atomic_torch_save(run_dir / "source_B_display_examples.pt", source_gallery)
    review_summary = _write_blind_review(run_dir, config, source, source_path, pixels, labels,
                                         dev_indices, endpoint_summaries, device)
    report = _aggregate_report(run_id, source_report, source, config, preflight,
                               endpoint_summaries["C"], endpoint_summaries["V"],
                               time.perf_counter() - process_started, run_dir)
    report["evaluation"] = {
        "E_D": {"role": "frozen confirmation only", "logits_saved": True},
        "E_C": ec_qualification, "E_C_logits_consumed": True,
        "training_update_sample_reads": {"C": CONTINUATION_UPDATES * BATCH_SIZE,
                                          "V": CONTINUATION_UPDATES * BATCH_SIZE,
                                          "DEV": 0, "official_TEST": 0},
        "prior_fit_train_rows_read_per_branch": 55_000,
        "dev_evaluation_rows_per_branch": 5_000,
        "official_test_rows_read": 0,
        "source_B_gallery_samples_replayed": source_gallery_samples,
    }
    report["blind_review"] = review_summary
    report["elapsed_seconds"] = {
        "process_wall_seconds_to_machine_stage": time.perf_counter() - process_started,
        "branch_C_training_seconds": endpoint_summaries["C"]["training"]["elapsed_seconds_training_cumulative"],
        "branch_C_evaluation_seconds": endpoint_summaries["C"]["training"]["evaluation_elapsed_seconds"],
        "branch_V_training_seconds": endpoint_summaries["V"]["training"]["elapsed_seconds_training_cumulative"],
        "branch_V_evaluation_seconds": endpoint_summaries["V"]["training"]["evaluation_elapsed_seconds"],
        "blind_review_preparation_seconds": review_summary["elapsed_seconds"],
        "human_wait_seconds": None,
    }
    atomic_write_json(run_dir / "paired_comparison.json", report["paired_comparison"])
    _write_comparison_html(run_dir / "machine_results.html", report)
    report["output"]["output_bytes_before_report_finalization"] = base._directory_bytes(run_dir)
    report["output"]["within_budget"] = report["output"]["output_bytes_before_report_finalization"] <= int(
        config["output"]["formal_result_budget_bytes"])
    if not report["output"]["within_budget"]:
        report["status"] = "MEASUREMENT_INCOMPLETE"
        report["measurement_complete"] = False
    atomic_write_json(run_dir / "machine_report.json", report)
    atomic_write_json(run_dir / "report.json", report)
    atomic_write_json(run_dir / "run_state.json", {
        "status": control.state, "task_status": report["status"],
        "scientific_status": "PENDING_HUMAN_PRIMARY_ENDPOINT" if report["measurement_complete"]
        else "MEASUREMENT_INCOMPLETE",
        "research_run_control_state": control.state,
        "completed_global_step_each_branch": TOTAL_UPDATES,
        "measurement_complete": report["measurement_complete"],
        "human_ratings_pending": report["status"] == "AWAITING_HUMAN_RATINGS",
    })
    print(json.dumps({"status": report["status"], "run_control_state": control.state,
                      "report": str(run_dir / "machine_report.json"),
                      "blind_review": str(run_dir / "blind_review.html")}, ensure_ascii=False), flush=True)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--ratings-file", type=Path)
    parser.add_argument("--preflight-worker-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.preflight_only and args.resume_checkpoint is not None:
        raise SystemExit("--preflight-only cannot be combined with --resume-checkpoint")
    if args.ratings_file is not None and (args.preflight_only or args.resume_checkpoint is not None):
        raise SystemExit("--ratings-file cannot be combined with preflight or checkpoint resume")
    _run(args)


if __name__ == "__main__":
    main()
