"""Run a fixed 5,000-update Decoder-only native-gradient comparison."""

from __future__ import annotations

import argparse
import copy
import csv
import html
import json
import math
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
from ldm_project.v2.application.research.lightgen.native_image_inputs import image_batch
from ldm_project.v2.application.research.lightgen.image_readout import resize_view, ssim_batch
from ldm_project.v2.application.research.lightgen.confirmation_evaluators import (
    EC_CHECKPOINT_SCHEMA,
    ThirdConfirmationEvaluator,
)
from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
)
from ldm_project.v2.evidence import (
    atomic_torch_save,
    atomic_torch_save_new,
    atomic_write_json,
    atomic_write_new_json,
)
from ldm_project.v2.evidence.io import cpu_copy


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_SMALL_DECODER_NATIVE_GRADIENT_AB_V1"
AUTHORITY = "USER_APPROVED_SMALL_DECODER_NATIVE_GRADIENT_AB_20261022"
SOURCE_TASK_ID = "LIGHTGEN_SMALL_GAUSSIAN_FOREGROUND_BALANCE_AB_V1"
GENERATION_TASK_ID = "LIGHTGEN_SMALL_GAUSSIAN_REPLICATION_PRIOR_CONFIRMATION_V1"
SOURCE_STEP = 45_000
CONTINUATION_UPDATES = 5_000
ENDPOINT_STEP = SOURCE_STEP + CONTINUATION_UPDATES
BATCH_SIZE = 10
SEED = 2026101501
CHECKPOINT_INTERVAL = 500
PROGRESS_INTERVAL = 100
RECOVERY_RTOL = 2.0e-6
RECOVERY_ATOL = 5.0e-8
WINDOW_TOLERANCE = 1.0e-4
BOOTSTRAP_RESAMPLES = 2_000
RECON_BOOTSTRAP_SEED = 2026102291
HUMAN_BOOTSTRAP_SEED = 2026102294
BRANCHES = ("D0", "Dgrad")
REVIEW_SCHEMA = "lightgen_small_decoder_native_gradient_blind_review.v1"
RATINGS_SCHEMA = "lightgen_small_decoder_native_gradient_ratings.v1"
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


def _select_review_cohort(requested: torch.Tensor, proposal_ids: torch.Tensor,
                          reviewed_proposal_ids: dict[int, set[int]], seed: int
                          ) -> list[dict[str, int]]:
    _require(isinstance(requested, torch.Tensor) and requested.shape == (10_000,)
             and isinstance(proposal_ids, torch.Tensor) and proposal_ids.shape == (10_000,)
             and torch.equal(torch.bincount(requested.long(), minlength=10), torch.full((10,), 1000)),
             "G4 human-review selection population is malformed")
    rng = np.random.default_rng(seed)
    selected: list[dict[str, int]] = []
    for label in range(10):
        candidates = torch.nonzero(requested == label, as_tuple=False).flatten().numpy()
        keep = np.asarray([int(index) for index in candidates
                           if int(proposal_ids[index]) not in reviewed_proposal_ids[label]], dtype=np.int64)
        _require(keep.size == 990, f"G4 class {label} does not have 990 unrated proposals")
        chosen = rng.choice(keep, size=10, replace=False)
        for index in chosen.tolist():
            selected.append({"cohort_row": int(index), "requested_label": label,
                             "proposal_id": int(proposal_ids[index]), "stream": 0})
    _require(len(selected) == 100 and len({(item["requested_label"], item["proposal_id"])
                                           for item in selected}) == 100,
             "fixed G4 human sample selection is incomplete or duplicated")
    return selected


def _require(condition: bool, message: str, error_type: type[Exception] = RuntimeError) -> None:
    if not condition:
        raise error_type(message)


def _read_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict) and value.get("schema") ==
             "lightgen_small_decoder_native_gradient_ab.v1", "decoder-gradient config schema changed")
    _require(value.get("task_id") == TASK_ID and value.get("semantic_authority_id") == AUTHORITY,
             "decoder-gradient task authorization identity changed")
    training = value["training"]
    _require(training["seed"] == SEED and training["source_completed_updates"] == SOURCE_STEP
             and training["continuation_updates"] == CONTINUATION_UPDATES
             and training["endpoint_global_step"] == ENDPOINT_STEP
             and training["branch_order"] == list(BRANCHES)
             and training["trainable_group"] == "decoder_phase_only",
             "decoder-gradient source step, branches, or update budget changed")
    _require(training["batch_size"] == BATCH_SIZE and training["samples_per_class_per_batch"] == 1
             and training["optimizer"] == "Adam" and training["learning_rate"] == 1.0e-3
             and training["betas"] == [0.9, 0.999] and training["epsilon"] == 1.0e-8
             and training["weight_decay"] == 0.0 and training["checkpoint_interval"] == CHECKPOINT_INTERVAL
             and training["progress_interval"] == PROGRESS_INTERVAL
             and training["checkpoints_at"] == [0, 2500, CONTINUATION_UPDATES],
             "decoder-gradient optimizer or checkpoint schedule changed")
    _require(value["data"]["train_count"] == 55_000 and value["data"]["dev_count"] == 5_000
             and value["data"]["dev_per_class"] == 500 and value["data"]["official_test_reads"] == 0,
             "decoder-gradient data roles changed")
    small = value["optics"]["small"]
    _require(small["ols_basis"] == "gaussian_s12_v1" and small["phase_parameters"] == 40_000
             and small["production_propagation_grid"] == 2048
             and small["propagation_grid_candidates"] == [2048, 4096]
             and value["optics"]["window_complex_relative_error_max"] == WINDOW_TOLERANCE,
             "Gaussian geometry or numerical tolerance changed")
    loss = value["loss"]
    _require(loss["structural_weight"] == 0.3148024967699311
             and loss["native_gradient_definition"] ==
             "half_sum_of_mean_squared_horizontal_and_vertical_forward_difference_residuals"
             and loss["native_gradient_boundary"] == "no_padding_adjacent_pixels_only"
             and loss["gradient_weight_calibration"] ==
             "decoder_gradient_norm_weighted_rec_over_unweighted_native_gradient_once"
             and loss["freeze_gradient_weight_after_calibration"] is True,
             "native-gradient loss semantics changed")
    review = value["human_review"]
    _require(review["generated_samples_per_class"] == 10 and review["unique_generated_per_branch"] == 100
             and review["hidden_duplicates"] == 20 and review["real_anchors"] == 20
             and review["shuffle_seed"] == 2026102293 and review["bootstrap_seed"] == HUMAN_BOOTSTRAP_SEED
             and review["duplicate_consistency_min"] == 18 and review["anchor_label_match_min"] == 18,
             "human-review population or decision rule changed")
    _require(value["evaluation"]["paired_bootstrap_resamples"] == BOOTSTRAP_RESAMPLES
             and value["evaluation"]["reconstruction_bootstrap_seed"] == RECON_BOOTSTRAP_SEED
             and value["evaluation"]["evaluator_role"] == "frozen_confirmation_only"
             and value["output"]["formal_result_budget_bytes"] == 536_870_912,
             "evaluation or output contract changed")
    return value


def _schedule(labels: torch.Tensor, train_indices: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    by_class: list[torch.Tensor] = []
    for label in range(10):
        candidates = train_indices[labels[train_indices] == label]
        _require(candidates.numel() > 0, f"TRAIN class {label} is empty")
        by_class.append(candidates[torch.randperm(candidates.numel(), generator=generator)])
    rows = torch.empty((ENDPOINT_STEP, BATCH_SIZE), dtype=torch.long)
    cursors = [0] * 10
    for step in range(ENDPOINT_STEP):
        for label in range(10):
            if cursors[label] == by_class[label].numel():
                cursors[label] = 0
            rows[step, label] = by_class[label][cursors[label]]
            cursors[label] += 1
    _require(torch.equal(labels[rows].cpu(), torch.arange(10).expand(ENDPOINT_STEP, 10)),
             "reconstructed training schedule is not class balanced")
    return rows


def _load_source(
    source_path: Path, cohort_path: Path, previous_review_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    _require(source_path.is_file(), f"explicit C source checkpoint is missing: {source_path}", FileNotFoundError)
    _require(cohort_path.is_file(), f"explicit fixed G4 cohort is missing: {cohort_path}", FileNotFoundError)
    cp = torch.load(source_path, map_location="cpu", weights_only=False)
    _require(isinstance(cp, dict) and cp.get("schema") == "lightgen_small_gaussian_foreground_balance_checkpoint.v1"
             and cp.get("task_id") == SOURCE_TASK_ID and cp.get("branch") == "C"
             and cp.get("seed") == SEED and cp.get("completed_global_step") == SOURCE_STEP
             and cp.get("completed_updates") == 10_000 and cp.get("source_checkpoint_step") == 35_000
             and cp.get("evaluator_role") == "frozen_confirmation_only",
             "loaded source is not the registered C 45,000-step endpoint")
    identity = cp.get("model_identity", {})
    _require(identity == config["source"]["model_identity"]
             and tuple(cp["model"]["runtime.encoder.phase"].shape) == (2, 1, 100, 100)
             and tuple(cp["model"]["runtime.generator.phase"].shape) == (2, 1, 100, 100),
             "source Gaussian model identity or phase geometry changed")
    _require(float(cp.get("structural_weight", math.nan)) == config["loss"]["structural_weight"],
             "source C SSIM coefficient differs from the fixed value")
    optimizer = cp.get("optimizer", {})
    groups = optimizer.get("param_groups", [])
    states = optimizer.get("state", {})
    _require(len(groups) == 1 and len(groups[0].get("params", [])) == 2
             and groups[0].get("lr") == 1.0e-3 and tuple(groups[0].get("betas", ())) == (0.9, 0.999)
             and groups[0].get("eps") == 1.0e-8 and groups[0].get("weight_decay") == 0.0
             and len(states) == 2 and all(int(state["step"].item()) == SOURCE_STEP for state in states.values()),
             "source Adam groups or phase step counters do not match the C endpoint")
    for name in ("scale", "power_reference", "camera_gain"):
        _require(torch.isfinite(torch.as_tensor(cp[name])).all().item(), f"source {name} is nonfinite")
    _require(isinstance(cp.get("weights"), dict)
             and all(math.isfinite(float(value)) for value in cp["weights"].values()),
             "source BOGT loss weights are malformed")
    source_dir = source_path.parent
    endpoint_path = source_dir / "endpoint_summary.json"
    report_path = source_dir.parent / "final_report.json"
    _require(endpoint_path.is_file() and report_path.is_file(),
             "C endpoint summary or parent task report is missing", FileNotFoundError)
    endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(endpoint.get("branch") == "C" and endpoint.get("global_step") == SOURCE_STEP
             and endpoint.get("measurement_complete") is True
             and endpoint.get("training", {}).get("endpoint_global_step") == SOURCE_STEP
             and report.get("task_id") == SOURCE_TASK_ID and report.get("measurement_complete") is True
             and report.get("run_id") == cp.get("run_id"),
             "C endpoint report is incomplete or does not identify the loaded checkpoint")
    baseline_config_path = (ROOT / config["source"]["baseline_config"]).resolve()
    baseline_config = yaml.safe_load(baseline_config_path.read_text(encoding="utf-8"))
    _require(cp.get("config_identity") == baseline_config,
             "source checkpoint configuration differs from the explicit source config")
    terminal_path = source_dir / "terminal_model.pt"
    if terminal_path.is_file():
        terminal = torch.load(terminal_path, map_location="cpu", weights_only=False)
        _require(terminal.get("branch") == "C" and terminal.get("global_step") == SOURCE_STEP
                 and base._assert_close(cp["model"], terminal.get("model", {}), "source_terminal_model") == 0.0,
                 "source checkpoint differs from its saved C terminal model")

    cohort = torch.load(cohort_path, map_location="cpu", weights_only=False)
    _require(cohort.get("schema") == "lightgen_small_conditional_mixture_generated_cohort.v1"
             and cohort.get("task_id") == GENERATION_TASK_ID and cohort.get("method") == "G4"
             and cohort.get("model") == "C_seed_2026101501" and cohort.get("model_seed") == SEED
             and cohort.get("stream") == 0 and cohort.get("global_step") == SOURCE_STEP
             and cohort.get("classifier_based_sample_selection") is False
             and cohort.get("per_image_normalization") is False,
             "fixed G4 cohort task, endpoint, or generation policy changed")
    coefficients = cohort.get("ols_coefficients")
    requested = cohort.get("requested_labels")
    component_ids = cohort.get("component_ids")
    proposal_ids = cohort.get("proposal_ids")
    _require(isinstance(coefficients, torch.Tensor) and tuple(coefficients.shape) == (10_000, 100)
             and coefficients.is_complex() and torch.isfinite(coefficients).all().item()
             and isinstance(requested, torch.Tensor) and tuple(requested.shape) == (10_000,)
             and torch.equal(torch.bincount(requested.long(), minlength=10), torch.full((10,), 1000))
             and isinstance(component_ids, torch.Tensor) and component_ids.shape == (10_000,)
             and bool(((component_ids >= 0) & (component_ids < 4)).all())
             and isinstance(proposal_ids, torch.Tensor) and tuple(proposal_ids.shape) == (10_000,),
             "fixed G4 coefficients or class/proposal population changed")
    for label in range(10):
        mask = requested == label
        _require(proposal_ids[mask].unique().numel() == 1000,
                 f"fixed G4 class {label} has duplicate proposal IDs")
    _require(tuple(cohort.get("images", torch.empty(0)).shape) == (10_000, 1, 28, 28),
             "fixed G4 source views are incomplete")

    previous_review_dir = previous_review_dir.resolve()
    key_path = previous_review_dir / "blind_review_key.json"
    previous_report_path = previous_review_dir / "final_report.json"
    _require(key_path.is_file() and previous_report_path.is_file(),
             "previous completed readability review is missing", FileNotFoundError)
    previous_key = json.loads(key_path.read_text(encoding="utf-8"))
    previous_report = json.loads(previous_report_path.read_text(encoding="utf-8"))
    _require(previous_key.get("task_id") == "LIGHTGEN_SMALL_G1_G4_NATIVE_READABILITY_V1"
             and previous_report.get("human_review", {}).get("review_id") == previous_key.get("review_id")
             and previous_report.get("human_review", {}).get("quality_checks", {}).get(
                 "duplicate_consistency", {}).get("count", 0) >= 18,
             "previous G4 review key is not a completed, quality-controlled source")
    reviewed: dict[int, set[int]] = {label: set() for label in range(10)}
    for card in previous_key.get("cards", []):
        if card.get("kind") == "generated" and card.get("method") == "G4" and card.get("stream") == 0:
            reviewed[int(card["requested_label"])].add(int(card["proposal_id"]))
    _require(all(len(reviewed[label]) == 10 for label in range(10)),
             "previous G4 review does not identify ten rated samples per class")
    selected = _select_review_cohort(
        requested, proposal_ids, reviewed, int(config["human_review"]["sample_selection_seed"]))
    return {
        "checkpoint": cp, "endpoint_summary": endpoint, "task_report": report,
        "cohort": cohort, "reviewed_proposal_ids": reviewed,
        "review_selection": selected, "source_config": baseline_config,
        "source_dir": source_dir,
    }


def _new_model_optimizer(source: dict[str, Any], config: dict[str, Any], device: torch.device
                         ) -> tuple[torch.nn.Module, torch.optim.Adam]:
    model = base._model("small", 2048, device, config)
    model.load_state_dict(source["checkpoint"]["model"], strict=True)
    encoder, decoder = base._phase_groups(model)
    for parameter in model.parameters():
        parameter.requires_grad_(parameter is decoder)
    _require(not encoder.requires_grad and decoder.requires_grad
             and all(not parameter.requires_grad for parameter in model.parameters()
                     if parameter is not encoder and parameter is not decoder),
             "only the Decoder phase group may be trainable")
    optimizer = torch.optim.Adam((encoder, decoder), lr=1.0e-3, betas=(0.9, 0.999),
                                 eps=1.0e-8, weight_decay=0.0)
    optimizer.load_state_dict(copy.deepcopy(source["checkpoint"]["optimizer"]))
    _require(len(optimizer.param_groups) == 1 and len(optimizer.param_groups[0]["params"]) == 2
             and optimizer.param_groups[0]["params"][0] is encoder
             and optimizer.param_groups[0]["params"][1] is decoder,
             "source Adam phase-to-parameter mapping changed")
    _require(_optimizer_phase_steps(optimizer, encoder, decoder) == (SOURCE_STEP, SOURCE_STEP),
             "source per-phase Adam step counters are not 45,000")
    return model, optimizer


def _optimizer_phase_steps(optimizer: torch.optim.Adam, encoder: torch.nn.Parameter,
                           decoder: torch.nn.Parameter) -> tuple[int, int]:
    values = []
    for parameter in (encoder, decoder):
        state = optimizer.state.get(parameter)
        _require(isinstance(state, dict) and "step" in state,
                 "optimizer phase state is missing its step counter")
        values.append(int(torch.as_tensor(state["step"]).item()))
    return values[0], values[1]


def _native_gradient_loss(gained_unclipped: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _require(gained_unclipped.shape == target.shape and gained_unclipped.ndim == 4
             and gained_unclipped.shape[1] == 1,
             "native-gradient prediction and target shapes differ")
    _require(torch.isfinite(gained_unclipped).all().item() and torch.isfinite(target).all().item(),
             "native-gradient inputs are nonfinite")
    dx = (gained_unclipped[..., 1:] - gained_unclipped[..., :-1]) - (target[..., 1:] - target[..., :-1])
    dy = (gained_unclipped[..., 1:, :] - gained_unclipped[..., :-1, :]) - (
        target[..., 1:, :] - target[..., :-1, :])
    value = 0.5 * (dx.square().mean() + dy.square().mean())
    _require(torch.isfinite(value).item(), "native-gradient loss is nonfinite")
    return value


def _native_gradient_loss_numpy(gained_unclipped: np.ndarray, target: np.ndarray) -> float:
    predicted = np.asarray(gained_unclipped, dtype=np.float64)
    expected = np.asarray(target, dtype=np.float64)
    if predicted.shape != expected.shape or predicted.ndim != 4 or predicted.shape[1] != 1:
        raise ValueError("native-gradient NumPy inputs have invalid shape")
    if not np.isfinite(predicted).all() or not np.isfinite(expected).all():
        raise ValueError("native-gradient NumPy inputs are nonfinite")
    dx = np.diff(predicted, axis=3) - np.diff(expected, axis=3)
    dy = np.diff(predicted, axis=2) - np.diff(expected, axis=2)
    return float(0.5 * (np.mean(dx * dx) + np.mean(dy * dy)))


def _native_gradient_per_image(gained: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dx = (gained[..., 1:] - gained[..., :-1]) - (target[..., 1:] - target[..., :-1])
    dy = (gained[..., 1:, :] - gained[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])
    return 0.5 * (dx.square().mean((1, 2, 3)) + dy.square().mean((1, 2, 3)))


def _loss_parts(
    model: torch.nn.Module, pixels: torch.Tensor, rows: torch.Tensor, device: torch.device,
    scale: torch.Tensor, power_reference: torch.Tensor, gain: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, Any]]:
    target_native = image_batch(pixels, rows, "small", device)
    parts, result = base._loss_components(model, target_native, scale, power_reference, gain)
    target28 = pixels.index_select(0, rows.detach().cpu()).unsqueeze(1).to(
        device=device, dtype=torch.float32) / 255.0
    gained = result["physical_w_per_pixel"] * gain.to(device=device, dtype=torch.float32)
    view = resize_view(gained)
    ssim = (1.0 - ssim_batch(view, target28)).mean()
    gradient = _native_gradient_loss(gained, target_native)
    _require(torch.isfinite(ssim).item(), "source SSIM loss is nonfinite")
    return parts, ssim, gradient, result


def _objective(parts: dict[str, torch.Tensor], ssim: torch.Tensor,
               gradient: torch.Tensor | None, weights: dict[str, float],
               structural_weight: float, gradient_weight: float | None) -> torch.Tensor:
    total = sum(float(weights[name]) * parts[name] for name in ("dist", "rec", "op"))
    total = total + structural_weight * ssim
    if gradient_weight is not None:
        _require(gradient is not None and math.isfinite(gradient_weight) and gradient_weight > 0.0,
                 "native-gradient coefficient or term is invalid")
        total = total + gradient_weight * gradient
    return total


def _gradient_norm(gradient: torch.Tensor | None, name: str) -> float:
    _require(gradient is not None and torch.isfinite(gradient).all().item(),
             f"{name} phase gradient is missing or nonfinite")
    norm = float(torch.linalg.vector_norm(gradient.detach().double()).cpu())
    _require(math.isfinite(norm) and norm > 0.0, f"{name} phase gradient is zero")
    return norm


def _calibrate_gradient_weight(
    model: torch.nn.Module, pixels: torch.Tensor, rows: torch.Tensor, device: torch.device,
    source: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    parts, _ssim, gradient, _result = _loss_parts(
        model, pixels, rows, device, source["checkpoint"]["scale"].to(device),
        source["checkpoint"]["power_reference"].to(device),
        source["checkpoint"]["camera_gain"].to(device),
    )
    decoder = base._phase_groups(model)[1]
    rec_gradient = torch.autograd.grad(float(source["checkpoint"]["weights"]["rec"]) * parts["rec"],
                                       decoder, retain_graph=True)[0]
    native_gradient = torch.autograd.grad(gradient, decoder)[0]
    rec_norm = _gradient_norm(rec_gradient, "weighted reconstruction")
    native_norm = _gradient_norm(native_gradient, "native gradient")
    coefficient = rec_norm / native_norm
    _require(math.isfinite(coefficient) and coefficient > 0.0,
             "native-gradient calibration is zero or nonfinite")
    return coefficient, {
        "weighted_rec_gradient_norm_decoder": rec_norm,
        "unweighted_native_gradient_norm_decoder": native_norm,
        "lambda_native_gradient": coefficient,
        "calibration_global_step": SOURCE_STEP,
    }


def _total_for_branch(model: torch.nn.Module, pixels: torch.Tensor, rows: torch.Tensor,
                      device: torch.device, source: dict[str, Any], branch: str,
                      gradient_weight: float | None) -> tuple[torch.Tensor, torch.Tensor]:
    cp = source["checkpoint"]
    parts, ssim, gradient, _ = _loss_parts(model, pixels, rows, device, cp["scale"].to(device),
                                            cp["power_reference"].to(device), cp["camera_gain"].to(device))
    total = _objective(parts, ssim, gradient if branch == "Dgrad" else None,
                       cp["weights"], float(cp["structural_weight"]), gradient_weight)
    return total, gradient


def _direction_check(model: torch.nn.Module, objective: Callable[[], torch.Tensor],
                     name: str, epsilon: float = 1.0e-2) -> dict[str, float]:
    decoder = base._phase_groups(model)[1]
    value = objective()
    gradient = torch.autograd.grad(value, decoder)[0]
    norm = torch.linalg.vector_norm(gradient.detach().double())
    _require(torch.isfinite(norm).item() and float(norm) > 0.0,
             f"{name} decoder direction is invalid")
    direction = (gradient.detach().double() / norm).to(decoder.dtype)
    original = decoder.detach().clone()
    plus_state = original + epsilon * direction
    minus_state = original - epsilon * direction
    representable = float((((plus_state.double() - original.double()) +
                            (original.double() - minus_state.double())) * direction.double()).sum() * 0.5)
    try:
        with torch.no_grad():
            decoder.copy_(plus_state)
        plus = float(objective().detach())
        with torch.no_grad():
            decoder.copy_(minus_state)
        minus = float(objective().detach())
    finally:
        with torch.no_grad():
            decoder.copy_(original)
    analytic = float((gradient.detach().double() * direction.double()).sum())
    _require(math.isfinite(representable) and representable > 0.0,
             f"{name} perturbation is not representable")
    finite = (plus - minus) / (2.0 * representable)
    relative = abs(analytic - finite) / max(abs(analytic), abs(finite), 1.0e-7)
    _require(math.isfinite(analytic) and math.isfinite(finite) and analytic * finite > 0.0
             and relative <= 0.05,
             f"{name} finite difference failed: autograd={analytic:.9g}, central={finite:.9g}, rel={relative:.6g}")
    return {"autograd": analytic, "central_difference": finite, "relative_error": relative,
            "epsilon_rad": epsilon, "representable_half_step_rad": representable}


def _train_update(model: torch.nn.Module, optimizer: torch.optim.Adam, pixels: torch.Tensor,
                  rows: torch.Tensor, device: torch.device, source: dict[str, Any],
                  branch: str, gradient_weight: float | None) -> dict[str, Any]:
    model.train()
    encoder, decoder = base._phase_groups(model)
    _require(not encoder.requires_grad and decoder.requires_grad and rows.shape == (BATCH_SIZE,),
             "Decoder-only trainable scope or logical batch changed")
    encoder_before = encoder.detach().clone()
    decoder_before = decoder.detach().clone()
    encoder_state_before = copy.deepcopy(optimizer.state[encoder])
    encoder_step_before, decoder_step_before = _optimizer_phase_steps(optimizer, encoder, decoder)
    optimizer.zero_grad(set_to_none=True)
    cp = source["checkpoint"]
    parts, ssim, gradient_value, result = _loss_parts(
        model, pixels, rows, device, cp["scale"].to(device),
        cp["power_reference"].to(device), cp["camera_gain"].to(device))
    total = _objective(parts, ssim, gradient_value if branch == "Dgrad" else None,
                       cp["weights"], float(cp["structural_weight"]), gradient_weight)
    _require(torch.isfinite(total).item(), "Decoder continuation objective is nonfinite")
    values_before_update = {
        "dist": float(parts["dist"].detach()), "rec": float(parts["rec"].detach()),
        "op": float(parts["op"].detach()), "ssim_loss": float(ssim.detach()),
        "native_gradient_loss": float(gradient_value.detach()), "total": float(total.detach()),
        "mean_plane_power_w": float(result["plane_powers_w"].detach().double().mean()),
    }
    total.backward()
    _require(encoder.grad is None, "frozen Encoder received a gradient")
    _require(decoder.grad is not None and torch.isfinite(decoder.grad).all().item()
             and bool((decoder.grad.abs().sum() > 0)), "Decoder gradient is invalid or zero")
    grad_norm = float(torch.linalg.vector_norm(decoder.grad.detach().double()).cpu())
    optimizer.step()
    _require(torch.equal(encoder_before, encoder) and not torch.equal(decoder_before, decoder),
             "Decoder-only optimizer update changed the wrong parameter group")
    _require(base._assert_close(encoder_state_before, optimizer.state[encoder], "encoder_adam_state") == 0.0,
             "frozen Encoder Adam state changed")
    steps = _optimizer_phase_steps(optimizer, encoder, decoder)
    _require(steps == (encoder_step_before, decoder_step_before + 1),
             "Encoder/Decoder Adam step counters advanced unexpectedly")
    return {
        "global_step": steps[1], **values_before_update, "decoder_gradient_norm": grad_norm,
        "encoder_adam_step": steps[0], "decoder_adam_step": steps[1],
    }


def _restore_training_state(payload: dict[str, Any], config: dict[str, Any],
                            device: torch.device) -> tuple[torch.nn.Module, torch.optim.Adam]:
    model = base._model("small", 2048, device, config)
    model.load_state_dict(payload["model"], strict=True)
    encoder, decoder = base._phase_groups(model)
    for parameter in model.parameters():
        parameter.requires_grad_(parameter is decoder)
    optimizer = torch.optim.Adam((encoder, decoder), lr=1.0e-3, betas=(0.9, 0.999),
                                 eps=1.0e-8, weight_decay=0.0)
    optimizer.load_state_dict(payload["optimizer"])
    expected = payload.get("optimizer_steps")
    if isinstance(expected, dict):
        expected_steps = (expected.get("encoder_adam_step"), expected.get("decoder_adam_step"))
    else:
        expected_steps = tuple(expected or ())
    _require(_optimizer_phase_steps(optimizer, encoder, decoder) == expected_steps,
             "resumed per-phase Adam state or parameter mapping changed")
    return model, optimizer


def _recovery_worker(staging_dir: Path, config_path: Path) -> None:
    config = _read_config(config_path)
    payload_path = staging_dir / "recovery_input.pt"
    _require(payload_path.is_file(), "independent recovery worker input is missing")
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    _require(payload.get("task_id") == TASK_ID and payload.get("config_identity") == config
             and payload.get("completed_segment_updates") == 3
             and payload.get("branch") in BRANCHES,
             "independent recovery worker identity or step is invalid")
    _require(torch.cuda.is_available(), "independent recovery worker requires CUDA")
    device = torch.device("cuda")
    model, optimizer = _restore_training_state(payload, config, device)
    torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])
    torch.random.set_rng_state(payload["cpu_rng"])
    pixels, _, _, _ = base._load_data(ROOT / config["data"]["training_file"])
    source = {"checkpoint": payload["frozen_source"]}
    loss = _train_update(model, optimizer, pixels, payload["rows"][3], device, source,
                         payload["branch"], payload["gradient_weight"])
    encoder, decoder = base._phase_groups(model)
    atomic_torch_save_new(staging_dir / "recovery_fourth_update.pt", {
        "task_id": TASK_ID, "branch": payload["branch"], "run_id": payload["run_id"],
        "completed_segment_updates": 4,
        "model": base._state_snapshot(model),
        "optimizer": cpu_copy(optimizer.state_dict()),
        "optimizer_steps": _optimizer_phase_steps(optimizer, encoder, decoder),
        "loss": loss,
    })
    print(json.dumps({"status": "PASSED", "branch": payload["branch"]}), flush=True)


def _preflight_branch(
    branch: str, source: dict[str, Any], config: dict[str, Any], config_path: Path,
    pixels: torch.Tensor, rows: torch.Tensor, device: torch.device,
    gradient_weight: float, staging_dir: Path,
) -> dict[str, Any]:
    model, optimizer = _new_model_optimizer(source, config, device)
    source_cp = source["checkpoint"]
    torch.cuda.set_rng_state_all([state.cpu() for state in source_cp["cuda_rng"]])
    torch.random.set_rng_state(source_cp["cpu_rng"])
    encoder, decoder = base._phase_groups(model)
    model_initial = base._state_snapshot(model)
    optimizer_initial = copy.deepcopy(optimizer.state_dict())
    cuda_initial = torch.cuda.get_rng_state_all()
    cpu_initial = torch.random.get_rng_state()
    trials = [
        _train_update(model, optimizer, pixels, rows[index], device, source, branch,
                      gradient_weight if branch == "Dgrad" else None)
        for index in range(3)
    ]
    phase_steps = _optimizer_phase_steps(optimizer, encoder, decoder)
    recovery_payload = {
        "task_id": TASK_ID, "branch": branch, "run_id": uuid.uuid4().hex,
        "config_identity": config, "completed_segment_updates": 3,
        "model": base._state_snapshot(model),
        "optimizer": cpu_copy(optimizer.state_dict()),
        "optimizer_steps": phase_steps,
        "frozen_source": {key: source_cp[key] for key in
                          ("scale", "power_reference", "camera_gain", "weights", "structural_weight")},
        "gradient_weight": gradient_weight if branch == "Dgrad" else None,
        "rows": rows.detach().cpu(),
        "cuda_rng": [state.cpu() for state in torch.cuda.get_rng_state_all()],
        "cpu_rng": torch.random.get_rng_state().cpu(),
    }
    atomic_torch_save_new(staging_dir / "recovery_input.pt", recovery_payload)
    direct_loss = _train_update(model, optimizer, pixels, rows[3], device, source, branch,
                                gradient_weight if branch == "Dgrad" else None)
    direct_model = base._state_snapshot(model)
    direct_optimizer = cpu_copy(optimizer.state_dict())
    direct_steps = _optimizer_phase_steps(optimizer, encoder, decoder)
    worker = subprocess.run(
        [sys.executable, "-m", "scripts.research.lightgen.execute_lightgen_small_decoder_native_gradient_ab_v1",
         "--recovery-worker-dir", str(staging_dir), "--config", str(config_path),
         "--output-dir", str(ROOT / config["output"]["experiment_root"])],
        capture_output=True, text=True, check=False, cwd=ROOT,
    )
    _require(worker.returncode == 0,
             "independent-process recovery failed: " + (worker.stderr[-2500:] or worker.stdout[-2500:]))
    recovered = torch.load(staging_dir / "recovery_fourth_update.pt", map_location="cpu", weights_only=False)
    _require(recovered.get("task_id") == TASK_ID and recovered.get("branch") == branch
             and recovered.get("completed_segment_updates") == 4,
             "independent recovery output identity changed")
    model_error = base._assert_close(direct_model, recovered["model"], f"{branch}_recovery_model")
    optimizer_error = base._assert_close(direct_optimizer, recovered["optimizer"], f"{branch}_recovery_optimizer")
    base._assert_close(direct_loss, recovered["loss"], f"{branch}_recovery_loss")
    _require(direct_steps == tuple(recovered["optimizer_steps"]),
             "independent recovery phase Adam steps differ")
    _require(model_error <= RECOVERY_ATOL and optimizer_error <= RECOVERY_ATOL,
             f"{branch} recovery exceeded the registered state tolerance")
    model.load_state_dict(model_initial, strict=True)
    optimizer.load_state_dict(optimizer_initial)
    torch.cuda.set_rng_state_all(cuda_initial)
    torch.random.set_rng_state(cpu_initial)
    _require(base._assert_close(model_initial, base._state_snapshot(model), f"{branch}_rollback") == 0.0,
             "preflight model was not rolled back")
    base._assert_close(optimizer_initial, optimizer.state_dict(), f"{branch}_optimizer_rollback")
    return {
        "status": "PASSED", "engineering_adam_calls": 5, "trial_updates": 3,
        "trial_losses": trials, "direct_fourth_update": direct_loss,
        "independent_process_recovery": True, "encoder_adam_step_after_trial": direct_steps[0],
        "decoder_adam_step_after_trial": direct_steps[1],
        "max_model_recovery_error": model_error, "max_optimizer_recovery_error": optimizer_error,
        "preflight_state_rolled_back": True,
    }


def _load_ec_evaluator(path: Path, device: torch.device) -> tuple[ThirdConfirmationEvaluator, dict[str, Any]]:
    _require(path.is_file(), f"explicit E_C checkpoint is missing: {path}", FileNotFoundError)
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
             and qualification.get("accuracy", 0.0) >= policy.get("min_accuracy", math.inf)
             and qualification.get("macro_recall", 0.0) >= policy.get("min_macro_recall", math.inf)
             and min(map(float, recalls)) >= policy.get("min_class_recall", math.inf),
             "E_C historical canonical-MNIST qualification is missing or failed")
    independence = payload.get("independence", {})
    _require(independence.get("role") == "INDEPENDENT_CONFIRMATION_ONLY"
             and independence.get("used_in_generator_loss") is False
             and independence.get("used_in_gradient_attribution") is False
             and independence.get("used_in_hyperparameter_selection") is False,
             "E_C is not isolated from training")
    evaluator = ThirdConfirmationEvaluator().to(device)
    evaluator.load_state_dict(payload["state_dict"], strict=True)
    evaluator.freeze()
    _require(not evaluator.training and all(not parameter.requires_grad for parameter in evaluator.parameters()),
             "E_C must be frozen and in evaluation mode")
    return evaluator, {"schema": payload["schema"], "role": independence["role"],
                       "qualification_count": qualification["count"],
                       "accuracy": qualification["accuracy"],
                       "macro_recall": qualification["macro_recall"],
                       "minimum_class_recall": min(map(float, recalls))}


def _ec_logits(evaluator: ThirdConfirmationEvaluator, images: torch.Tensor) -> torch.Tensor:
    output = evaluator(images)
    _require(isinstance(output, tuple) and len(output) == 2,
             "E_C must return features and logits")
    features, logits = output
    _require(tuple(features.shape) == (images.shape[0], 128)
             and tuple(logits.shape) == (images.shape[0], 10)
             and torch.isfinite(features).all().item() and torch.isfinite(logits).all().item(),
             "E_C returned malformed or nonfinite logits")
    return logits


def _logit_summary(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                                       torch.Tensor, torch.Tensor]:
    _require(logits.ndim == 2 and logits.shape == (targets.numel(), 10)
             and torch.isfinite(logits).all().item(), "evaluator logits are malformed or nonfinite")
    predictions = logits.argmax(1)
    masked = logits.clone()
    masked.scatter_(1, targets.to(logits.device).reshape(-1, 1), -torch.inf)
    target_logits = logits.gather(1, targets.to(logits.device).reshape(-1, 1)).squeeze(1)
    margin = target_logits - masked.max(1).values
    top_values = logits.topk(2, dim=1).values
    top_margin = top_values[:, 0] - top_values[:, 1]
    return predictions.cpu(), margin.cpu(), top_margin.cpu(), logits.cpu()


def _make_preflight(
    source: dict[str, Any], cohort: dict[str, Any], config: dict[str, Any], config_path: Path,
    source_path: Path, pixels: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor,
    schedule: torch.Tensor, device: torch.device, evaluator: torch.nn.Module,
    ec_evaluator: ThirdConfirmationEvaluator, staging_parent: Path,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    cp = source["checkpoint"]
    model, _ = _new_model_optimizer(source, config, device)
    encoder, decoder = base._phase_groups(model)
    encoder_state = encoder.detach().clone()
    buffers_before = {name: value.detach().clone() for name, value in model.named_buffers()}
    ed_state = base._state_snapshot(evaluator)
    ec_state = base._state_snapshot(ec_evaluator)
    selected_grid, source_window = base._window_check(
        "small", pixels, labels, schedule, config, device, model=model)
    relative_error = float(source_window.get("complex_relative_error_max", math.inf))
    _require(selected_grid == 2048 and relative_error <= WINDOW_TOLERANCE,
             "common C endpoint failed its 2048/4096 seven-plane complex-window check")
    batch = image_batch(pixels, schedule[SOURCE_STEP], "small", device)
    readout = base._small_producer_consumer_check(model, batch[:1])
    _require(readout.get("status") == "PASSED", "source OLS/camera production consumer check failed")

    requested = cohort["requested_labels"].long()
    first_indices = torch.stack([torch.nonzero(requested == label, as_tuple=False).flatten()[0]
                                 for label in range(10)])
    coefficients = cohort["ols_coefficients"].index_select(0, first_indices)
    saved_views = cohort["images"].index_select(0, first_indices).float().cpu()
    saved_ed_logits = cohort["ed_logits"].index_select(0, first_indices).float().cpu()
    saved_ec_logits = cohort["ec_logits"].index_select(0, first_indices).float().cpu()
    model.eval()
    evaluator.eval()
    ec_evaluator.eval()
    with torch.inference_mode():
        raw = model.forward_from_ols(coefficients.to(device))["physical_w_per_pixel"]
        gained = raw * cp["camera_gain"].to(device=device, dtype=torch.float32)
        replay_views = resize_view(gained).cpu()
        replay_ed = evaluator(replay_views.to(device)).cpu()
        replay_ec = _ec_logits(ec_evaluator, replay_views.to(device)).cpu()
    _require(torch.allclose(replay_views, saved_views, rtol=RECOVERY_RTOL, atol=RECOVERY_ATOL)
             and torch.allclose(replay_ed, saved_ed_logits, rtol=RECOVERY_RTOL, atol=RECOVERY_ATOL)
             and torch.allclose(replay_ec, saved_ec_logits, rtol=RECOVERY_RTOL, atol=RECOVERY_ATOL)
             and torch.equal(replay_ed.argmax(1), saved_ed_logits.argmax(1))
             and torch.equal(replay_ec.argmax(1), saved_ec_logits.argmax(1)),
             "fixed source C first-per-class G4 views or evaluator logits do not replay")

    rows = schedule[SOURCE_STEP:SOURCE_STEP + 4]
    lambda_model, _ = _new_model_optimizer(source, config, device)
    torch.cuda.set_rng_state_all([state.cpu() for state in cp["cuda_rng"]])
    torch.random.set_rng_state(cp["cpu_rng"])
    gradient_weight, calibration = _calibrate_gradient_weight(
        lambda_model, pixels, rows[0], device, source)
    frozen_source_state = base._state_snapshot(lambda_model)
    fd_records = {}
    for branch in BRANCHES:
        check_model, _ = _new_model_optimizer(source, config, device)
        fd_records[f"{branch}_total"] = _direction_check(
            check_model,
            lambda branch=branch: _total_for_branch(check_model, pixels, rows[0], device, source,
                                                      branch, gradient_weight if branch == "Dgrad" else None)[0],
            f"{branch}_total",
        )
        if branch == "Dgrad":
            fd_records["Dgrad_native_gradient_term"] = _direction_check(
                check_model,
                lambda: _loss_parts(check_model, pixels, rows[0], device,
                                    cp["scale"].to(device), cp["power_reference"].to(device),
                                    cp["camera_gain"].to(device))[2],
                "Dgrad_native_gradient_term",
            )
    _require(base._assert_close(frozen_source_state, base._state_snapshot(lambda_model),
                                "gradient_calibration_model_state") == 0.0,
             "gradient coefficient calibration changed source model parameters")
    branch_trials: dict[str, Any] = {}
    for branch in BRANCHES:
        staging_dir = staging_parent / branch
        staging_dir.mkdir(parents=False, exist_ok=False)
        branch_trials[branch] = _preflight_branch(
            branch, source, config, config_path, pixels, rows, device, gradient_weight, staging_dir)
    _require(torch.equal(encoder_state, base._phase_groups(model)[0])
             and base._assert_close(buffers_before, dict(model.named_buffers()), "source_buffers") == 0.0
             and base._assert_close(ed_state, base._state_snapshot(evaluator), "E_D_preflight") == 0.0
             and base._assert_close(ec_state, base._state_snapshot(ec_evaluator), "E_C_preflight") == 0.0,
             "preflight changed frozen source/evaluator state")
    torch.cuda.set_rng_state_all([state.cpu() for state in cp["cuda_rng"]])
    torch.random.set_rng_state(cp["cpu_rng"])
    return {
        "status": "READY", "task_id": TASK_ID, "source_checkpoint_step": SOURCE_STEP,
        "production_grid": 2048, "source_window": source_window, "independent_readout": readout,
        "g4_source_replay": {"samples": 10, "maximum_view_error": float((replay_views - saved_views).abs().max()),
                             "maximum_E_D_logit_error": float((replay_ed - saved_ed_logits).abs().max()),
                             "maximum_E_C_logit_error": float((replay_ec - saved_ec_logits).abs().max()),
                             "prediction_identical": True},
        "gradient_calibration": calibration, "finite_difference": fd_records,
        "branch_recovery_preflight": branch_trials,
        "frozen_encoder_and_evaluators_unchanged": True,
        "official_test_reads": 0, "scientific_updates_before_execution": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
    }, gradient_weight


def _make_receipt(control: ResearchRunControl, preflight: dict[str, Any],
                  device: torch.device) -> PreflightReceipt:
    return control.make_ready_receipt(
        correctness={
            "passed": True, "status": "PASSED",
            "checks": ["source_C_45000_and_per_phase_Adam_state",
                       "continuous_45000_to_50000_class_balanced_schedule",
                       "fixed_G4_source_replay_views_and_logits",
                       "2048_to_4096_seven_plane_window_and_independent_camera_readout",
                       "native_gradient_value_derivative_and_decoder_direction_checks",
                       "three_real_decoder_updates_and_independent_process_recovery_per_branch",
                       "encoder_parameter_buffer_and_adam_state_frozen",
                       "E_D_E_C_frozen_and_excluded_from_training"],
            "maximum_window_relative_error": preflight["source_window"]["complex_relative_error_max"],
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


def _new_identity(config: dict[str, Any], source: dict[str, Any], run_id: str) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID, run_id=run_id,
        trajectory_id=f"{TASK_ID}:seed:{SEED}:source_step:{SOURCE_STEP}",
        semantic_authority_id=AUTHORITY,
        data_role="MNIST_TRAIN_55000_DEV_5000_OFFICIAL_TEST_UNREAD",
        config_identity=config, model_identity=source["checkpoint"]["model_identity"],
        optimizer_identity={"name": "Adam", "lr": 1.0e-3, "betas": [0.9, 0.999],
                            "epsilon": 1.0e-8, "weight_decay": 0.0,
                            "source_encoder_step": SOURCE_STEP, "source_decoder_step": SOURCE_STEP},
        sampler_identity={"seed": SEED, "logical_batch_size": BATCH_SIZE,
                          "start_after_source_updates": SOURCE_STEP,
                          "continuation_updates": CONTINUATION_UPDATES},
        execution_source_semantics_id=config["execution_source_semantics_id"],
    )


def _write_progress(run_dir: Path, state: dict[str, Any]) -> None:
    state["last_heartbeat_unix"] = time.time()
    state["last_complete_save_unix"] = state.get("last_complete_save_unix", time.time())
    atomic_write_json(run_dir / "progress.json", state)


def _optimizer_summary(optimizer: torch.optim.Adam, model: torch.nn.Module) -> dict[str, int]:
    encoder, decoder = base._phase_groups(model)
    encoder_step, decoder_step = _optimizer_phase_steps(optimizer, encoder, decoder)
    return {"encoder_adam_step": encoder_step, "decoder_adam_step": decoder_step}


def _checkpoint_payload(
    model: torch.nn.Module, optimizer: torch.optim.Adam, branch: str, completed: int,
    run_id: str, source: dict[str, Any], config: dict[str, Any], schedule: torch.Tensor,
    gradient_weight: float, history: list[dict[str, Any]], elapsed: float,
) -> dict[str, Any]:
    return {
        "schema": "lightgen_small_decoder_native_gradient_checkpoint.v1", "task_id": TASK_ID,
        "run_id": run_id, "source_run_id": source["checkpoint"]["run_id"],
        "source_checkpoint_step": SOURCE_STEP, "branch": branch, "seed": SEED,
        "model_identity": source["checkpoint"]["model_identity"], "config_identity": config,
        "propagation_grid_size": 2048, "completed_updates": completed,
        "completed_global_step": SOURCE_STEP + completed,
        "model": base._state_snapshot(model), "optimizer": cpu_copy(optimizer.state_dict()),
        "optimizer_steps": _optimizer_summary(optimizer, model),
        "schedule": schedule.detach().cpu(), "schedule_start_global_step": SOURCE_STEP,
        "scale": source["checkpoint"]["scale"].detach().cpu(),
        "power_reference": source["checkpoint"]["power_reference"].detach().cpu(),
        "camera_gain": source["checkpoint"]["camera_gain"].detach().cpu(),
        "weights": source["checkpoint"]["weights"],
        "structural_weight": source["checkpoint"]["structural_weight"],
        "native_gradient_weight": gradient_weight if branch == "Dgrad" else None,
        "encoder_frozen_at_step": SOURCE_STEP, "history": history,
        "cuda_rng": [state.cpu() for state in torch.cuda.get_rng_state_all()],
        "cpu_rng": torch.random.get_rng_state().cpu(),
        "evaluator_role": "frozen_confirmation_only", "elapsed_seconds_continuation": elapsed,
    }


def _validate_checkpoint(checkpoint: dict[str, Any], branch: str, run_id: str,
                         source: dict[str, Any], config: dict[str, Any], schedule: torch.Tensor,
                         gradient_weight: float) -> int:
    _require(checkpoint.get("schema") == "lightgen_small_decoder_native_gradient_checkpoint.v1"
             and checkpoint.get("task_id") == TASK_ID and checkpoint.get("run_id") == run_id
             and checkpoint.get("source_run_id") == source["checkpoint"]["run_id"]
             and checkpoint.get("source_checkpoint_step") == SOURCE_STEP
             and checkpoint.get("branch") == branch and checkpoint.get("seed") == SEED,
             "resume checkpoint task, source, branch, or seed identity changed")
    _require(checkpoint.get("model_identity") == source["checkpoint"]["model_identity"]
             and checkpoint.get("config_identity") == config
             and checkpoint.get("propagation_grid_size") == 2048
             and torch.equal(checkpoint.get("schedule"), schedule)
             and checkpoint.get("schedule_start_global_step") == SOURCE_STEP,
             "resume checkpoint geometry, config, or training sample rows changed")
    completed = checkpoint.get("completed_updates")
    _require(isinstance(completed, int) and 0 <= completed <= CONTINUATION_UPDATES
             and checkpoint.get("completed_global_step") == SOURCE_STEP + completed,
             "resume checkpoint step is invalid")
    _require(checkpoint.get("encoder_frozen_at_step") == SOURCE_STEP
             and checkpoint.get("native_gradient_weight") ==
             (gradient_weight if branch == "Dgrad" else None),
             "resume checkpoint trainable scope or frozen loss coefficient changed")
    _require(checkpoint.get("structural_weight") == source["checkpoint"]["structural_weight"]
             and checkpoint.get("weights") == source["checkpoint"]["weights"],
             "resume checkpoint source BOGT or SSIM coefficients changed")
    _require(checkpoint.get("optimizer_steps") == {
        "encoder_adam_step": SOURCE_STEP,
        "decoder_adam_step": SOURCE_STEP + completed,
    }, "resume checkpoint per-phase Adam step counters are inconsistent")
    return completed


def _require_progress_matches_checkpoint(progress: dict[str, Any], branch: str,
                                          checkpoint_updates: int) -> None:
    registered = progress.get("branch_completed_updates", {}).get(branch)
    _require(isinstance(registered, int) and registered == checkpoint_updates,
             f"{branch} progress claims {registered} updates but its explicit checkpoint has "
             f"{checkpoint_updates}; reconcile from a preserved complete checkpoint before resume")


def _save_checkpoint(branch_dir: Path, payload: dict[str, Any], completed: int) -> None:
    latest = branch_dir / "checkpoint_latest.pt"
    previous = branch_dir / "checkpoint_previous.pt"
    if latest.is_file():
        old = torch.load(latest, map_location="cpu", weights_only=False)
        atomic_torch_save(previous, old)
    atomic_torch_save(latest, payload)
    check = torch.load(latest, map_location="cpu", weights_only=False)
    _require(check.get("branch") == payload["branch"] and check.get("completed_updates") == completed
             and check.get("optimizer_steps") == payload["optimizer_steps"],
             "rolling checkpoint did not pass read-back validation")
    if completed in (0, 2500, CONTINUATION_UPDATES):
        endpoint = branch_dir / f"checkpoint_step_{completed:06d}.pt"
        atomic_torch_save(endpoint, payload)
        reread = torch.load(endpoint, map_location="cpu", weights_only=False)
        _require(reread.get("completed_updates") == completed and reread.get("branch") == payload["branch"],
                 "registered checkpoint did not pass read-back validation")


def _restore_checkpoint(checkpoint: dict[str, Any], config: dict[str, Any],
                        device: torch.device) -> tuple[torch.nn.Module, torch.optim.Adam]:
    model, optimizer = _restore_training_state(checkpoint, config, device)
    torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
    torch.random.set_rng_state(checkpoint["cpu_rng"])
    return model, optimizer


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _require(bool(rows) and all(list(row) == list(rows[0]) for row in rows),
             f"CSV rows are empty or have inconsistent fields: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_reconstruction(
    model: torch.nn.Module, evaluator: torch.nn.Module,
    ec_evaluator: ThirdConfirmationEvaluator, pixels: torch.Tensor, labels: torch.Tensor,
    dev_indices: torch.Tensor, device: torch.device, source: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    cp = source["checkpoint"]
    gain = cp["camera_gain"].to(device)
    rows_out: list[dict[str, Any]] = []
    fixed_rows = torch.stack([dev_indices[labels[dev_indices] == label][:2]
                              for label in range(10)]).reshape(-1)
    fixed_lookup = set(map(int, fixed_rows.tolist()))
    display: dict[str, list[torch.Tensor]] = {"rows": [], "labels": [], "target28": [],
                                               "view28": [], "target_native": [],
                                               "view_native": []}
    ed_state = base._state_snapshot(evaluator)
    ec_state = base._state_snapshot(ec_evaluator)
    model.eval()
    evaluator.eval()
    ec_evaluator.eval()
    with torch.inference_mode():
        for rows in dev_indices.split(BATCH_SIZE):
            target_native = image_batch(pixels, rows, "small", device)
            result = base._forward(model, target_native)
            gained = result["physical_w_per_pixel"] * gain.to(dtype=torch.float32)
            view28 = resize_view(gained)
            target28 = pixels.index_select(0, rows.detach().cpu()).unsqueeze(1).to(
                device=device, dtype=torch.float32) / 255.0
            _require(view28.shape == target28.shape and gained.shape == target_native.shape,
                     "DEV production view or native reconstruction shape changed")
            mse = (view28 - target28).square().mean((1, 2, 3)).clamp_min(1.0e-12)
            psnr = -10.0 * torch.log10(mse)
            ssim = ssim_batch(view28, target28)
            gradient_error = _native_gradient_per_image(gained, target_native)
            squared = (gained - target_native).square()
            foreground = (target_native * squared).sum((1, 2, 3)) / target_native.sum((1, 2, 3)).clamp_min(1.0e-12)
            background_weights = 1.0 - target_native
            background = (background_weights * squared).sum((1, 2, 3)) / background_weights.sum(
                (1, 2, 3)).clamp_min(1.0e-12)
            clipping = ((gained < 0.0) | (gained > 1.0)).float().mean((1, 2, 3))
            ed_logits = evaluator(view28)
            ec_logits = _ec_logits(ec_evaluator, view28)
            batch_labels = labels.index_select(0, rows.detach().cpu()).long()
            ed_pred, ed_margin, ed_top, _ = _logit_summary(ed_logits, batch_labels.to(device))
            ec_pred, ec_margin, ec_top, _ = _logit_summary(ec_logits, batch_labels.to(device))
            batch_cpu = rows.detach().cpu().long()
            target_cpu = target28.cpu()
            view_cpu = view28.cpu()
            target_native_cpu = target_native.cpu()
            view_native_cpu = gained.clamp(0.0, 1.0).cpu()
            for i, row in enumerate(batch_cpu.tolist()):
                label = int(batch_labels[i])
                rows_out.append({
                    "row": int(row), "label": label, "mse": float(mse[i]),
                    "psnr_db": float(psnr[i]), "ssim": float(ssim[i]),
                    "native_gradient_mse": float(gradient_error[i]),
                    "stroke_weighted_native_mse": float(foreground[i]),
                    "background_weighted_native_mse": float(background[i]),
                    "ed_prediction": int(ed_pred[i]), "ed_requested_margin": float(ed_margin[i]),
                    "ed_top_margin": float(ed_top[i]), "ec_prediction": int(ec_pred[i]),
                    "ec_requested_margin": float(ec_margin[i]), "ec_top_margin": float(ec_top[i]),
                    "proxy_clipping_area_fraction": float(clipping[i]),
                    "ed_correct": bool(int(ed_pred[i]) == label), "ec_correct": bool(int(ec_pred[i]) == label),
                })
                if row in fixed_lookup:
                    display["rows"].append(batch_cpu[i:i + 1])
                    display["labels"].append(batch_labels[i:i + 1].cpu())
                    display["target28"].append(target_cpu[i:i + 1])
                    display["view28"].append(view_cpu[i:i + 1])
                    display["target_native"].append(target_native_cpu[i:i + 1])
                    display["view_native"].append(view_native_cpu[i:i + 1])
    _require(len(rows_out) == 5_000 and len({row["row"] for row in rows_out}) == 5_000,
             "DEV reconstruction is missing or duplicates rows")
    _require(base._assert_close(ed_state, base._state_snapshot(evaluator), "E_D_reconstruction") == 0.0
             and base._assert_close(ec_state, base._state_snapshot(ec_evaluator), "E_C_reconstruction") == 0.0,
             "frozen evaluator state changed during DEV reconstruction")
    labels_np = np.asarray([row["label"] for row in rows_out], dtype=np.int64)
    metrics: dict[str, Any] = {"population": len(rows_out), "per_image": rows_out}
    for key in ("mse", "psnr_db", "ssim", "native_gradient_mse", "stroke_weighted_native_mse",
                "background_weighted_native_mse", "proxy_clipping_area_fraction"):
        values = np.asarray([float(row[key]) for row in rows_out], dtype=np.float64)
        by_class = [float(values[labels_np == label].mean()) for label in range(10)]
        metrics[f"{key}_mean"] = float(values.mean())
        metrics[f"{key}_by_class"] = by_class
    for evaluator_name in ("ed", "ec"):
        predictions = np.asarray([int(row[f"{evaluator_name}_prediction"]) for row in rows_out])
        recalls = [float((predictions[labels_np == label] == label).mean()) for label in range(10)]
        metrics[f"{evaluator_name}_macro_recall"] = float(np.mean(recalls))
        metrics[f"{evaluator_name}_recall_by_class"] = recalls
        metrics[f"{evaluator_name}_confusion_matrix"] = base._classification(
            torch.from_numpy(labels_np), torch.from_numpy(predictions))["confusion_matrix"].tolist()
    display_tensor = {name: torch.cat(parts) for name, parts in display.items()}
    _require(tuple(display_tensor["rows"].shape) == (20,)
             and tuple(display_tensor["view_native"].shape) == (20, 1, 200, 200),
             "fixed two-per-class DEV display population is incomplete")
    return metrics, display_tensor


def _evaluate_generation(
    model: torch.nn.Module, evaluator: torch.nn.Module,
    ec_evaluator: ThirdConfirmationEvaluator, source: dict[str, Any], cohort: dict[str, Any],
    selection: list[dict[str, int]], labels: torch.Tensor, pixels: torch.Tensor,
    dev_indices: torch.Tensor, device: torch.device,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    cp = source["checkpoint"]
    coefficients = cohort["ols_coefficients"]
    requested = cohort["requested_labels"].long()
    proposals = cohort["proposal_ids"].long()
    components = cohort["component_ids"].long()
    selected_by_index = {int(item["cohort_row"]): item for item in selection}
    _require(len(selected_by_index) == 100, "fixed human proposal selection is incomplete")
    gain = cp["camera_gain"].to(device)
    image_parts: list[torch.Tensor] = []
    ed_logit_parts: list[torch.Tensor] = []
    ec_logit_parts: list[torch.Tensor] = []
    native_selected: dict[int, torch.Tensor] = {}
    per_image: list[dict[str, Any]] = []
    ed_state = base._state_snapshot(evaluator)
    ec_state = base._state_snapshot(ec_evaluator)
    model_state = base._state_snapshot(model)
    model.eval()
    evaluator.eval()
    ec_evaluator.eval()
    with torch.inference_mode():
        for start in range(0, 10_000, BATCH_SIZE):
            end = start + BATCH_SIZE
            result = model.forward_from_ols(coefficients[start:end].to(device))
            gained = result["physical_w_per_pixel"] * gain.to(dtype=torch.float32)
            view = resize_view(gained)
            ed_logits = evaluator(view)
            ec_logits = _ec_logits(ec_evaluator, view)
            rows = requested[start:end]
            ed_pred, ed_margin, ed_top, ed_cpu = _logit_summary(ed_logits, rows.to(device))
            ec_pred, ec_margin, ec_top, ec_cpu = _logit_summary(ec_logits, rows.to(device))
            image_parts.append(view.cpu())
            ed_logit_parts.append(ed_cpu)
            ec_logit_parts.append(ec_cpu)
            power = result["physical_w_per_pixel"].sum((1, 2, 3))
            mean_raw = result["physical_w_per_pixel"].mean((1, 2, 3))
            peak_raw = result["physical_w_per_pixel"].amax((1, 2, 3))
            mean_intensity = view.mean((1, 2, 3))
            clipping = ((gained < 0.0) | (gained > 1.0)).float().mean((1, 2, 3))
            for local, index in enumerate(range(start, end)):
                per_image.append({
                    "cohort_row": index, "requested_label": int(requested[index]),
                    "component_id": int(components[index]), "proposal_id": int(proposals[index]),
                    "ed_prediction": int(ed_pred[local]), "ed_requested_margin": float(ed_margin[local]),
                    "ed_top_margin": float(ed_top[local]), "ec_prediction": int(ec_pred[local]),
                    "ec_requested_margin": float(ec_margin[local]), "ec_top_margin": float(ec_top[local]),
                    "capture_power_w": float(power[local]), "raw_mean_w_per_pixel": float(mean_raw[local]),
                    "raw_peak_w_per_pixel": float(peak_raw[local]),
                    "view_mean_intensity": float(mean_intensity[local]),
                    "proxy_clipping_area_fraction": float(clipping[local]),
                })
                if index in selected_by_index:
                    native_selected[index] = gained[local].clamp(0.0, 1.0).cpu()
    images = torch.cat(image_parts)
    ed_logits_all = torch.cat(ed_logit_parts)
    ec_logits_all = torch.cat(ec_logit_parts)
    _require(tuple(images.shape) == (10_000, 1, 28, 28)
             and tuple(ed_logits_all.shape) == (10_000, 10) and tuple(ec_logits_all.shape) == (10_000, 10)
             and len(per_image) == 10_000 and len(native_selected) == 100,
             "fixed G4 generation evaluation is incomplete")
    dev_views = pixels.index_select(0, dev_indices.cpu()).unsqueeze(1).float() / 255.0
    dev_labels = labels.index_select(0, dev_indices.cpu()).long()
    shape_ratios = []
    for label in range(10):
        generated_class = images[requested == label]
        dev_class = dev_views[dev_labels == label]
        generated_mean = float(base._pair_distances(generated_class, 2026101603 + label).mean())
        dev_mean = float(base._pair_distances(dev_class, 2026101703 + label).mean())
        shape_ratios.append(generated_mean / max(dev_mean, 1.0e-12))
    ed_pred, ed_margin, ed_top, _ = _logit_summary(ed_logits_all, requested)
    ec_pred, ec_margin, ec_top, _ = _logit_summary(ec_logits_all, requested)
    summary: dict[str, Any] = {
        "population": 10_000, "samples_per_class": 1000,
        "proposal_identity": "same saved C G4 stream 0 OLS coefficients and proposal IDs for D0 and Dgrad",
        "ed_macro_recall": float(np.mean([np.mean(ed_pred[requested == c].numpy() == c) for c in range(10)])),
        "ed_recall_by_class": [float(np.mean(ed_pred[requested == c].numpy() == c)) for c in range(10)],
        "ec_macro_recall": float(np.mean([np.mean(ec_pred[requested == c].numpy() == c) for c in range(10)])),
        "ec_recall_by_class": [float(np.mean(ec_pred[requested == c].numpy() == c)) for c in range(10)],
        "ed_confusion_matrix": base._classification(requested, ed_pred)["confusion_matrix"].tolist(),
        "ec_confusion_matrix": base._classification(requested, ec_pred)["confusion_matrix"].tolist(),
        "proxy_clipping_area_fraction_macro": float(np.mean([np.mean(
            [row["proxy_clipping_area_fraction"] for row in per_image if row["requested_label"] == label])
            for label in range(10)])),
        "proxy_clipping_area_fraction_by_class": [float(np.mean(
            [row["proxy_clipping_area_fraction"] for row in per_image if row["requested_label"] == label]))
            for label in range(10)],
        "generated_to_dev_tv_ratio_by_class": shape_ratios,
        "generated_to_dev_tv_ratio_min": min(shape_ratios),
        "capture_power_w_mean": float(np.mean([row["capture_power_w"] for row in per_image])),
        "view_mean_intensity_mean": float(np.mean([row["view_mean_intensity"] for row in per_image])),
        "human_population_selection": selection,
        "per_image": per_image,
    }
    generation = {
        "images": images, "requested_labels": requested, "component_ids": components,
        "proposal_ids": proposals, "ols_coefficients": coefficients,
        "ed_logits": ed_logits_all, "ec_logits": ec_logits_all,
        "ed_predictions": ed_pred, "ec_predictions": ec_pred,
        "ed_requested_margin": ed_margin, "ec_requested_margin": ec_margin,
        "ed_top_margin": ed_top, "ec_top_margin": ec_top,
        "native_capture_indices": torch.tensor(sorted(native_selected), dtype=torch.long),
        "native_capture_images": torch.stack([native_selected[index] for index in sorted(native_selected)]),
    }
    _require(base._assert_close(model_state, base._state_snapshot(model), "model_generation_evaluation") == 0.0
             and base._assert_close(ed_state, base._state_snapshot(evaluator), "E_D_generation_evaluation") == 0.0
             and base._assert_close(ec_state, base._state_snapshot(ec_evaluator), "E_C_generation_evaluation") == 0.0,
             "frozen model or evaluator changed during fixed-cohort generation evaluation")
    return summary, generation


def _paired_bootstrap_rows(
    left: list[dict[str, Any]], right: list[dict[str, Any]], metric: str,
    seed: int = RECON_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    a = {int(row["row"]): row for row in left}
    b = {int(row["row"]): row for row in right}
    _require(a.keys() == b.keys() and len(a) == 5000, "paired DEV rows are incomplete or misaligned")
    strata = []
    for label in range(10):
        rows = sorted(row_id for row_id, row in a.items() if int(row["label"]) == label)
        _require(len(rows) == 500 and all(int(b[row]["label"]) == label for row in rows),
                 f"paired DEV class {label} is incomplete")
        strata.append(np.asarray([float(b[row][metric]) - float(a[row][metric]) for row in rows]))
    observed = float(np.mean([values.mean() for values in strata]))
    rng = np.random.default_rng(seed)
    samples = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        samples[index] = np.mean([values[rng.integers(0, values.size, values.size)].mean()
                                  for values in strata])
    return {"estimand": f"equal-class paired Dgrad-minus-D0 mean {metric}",
            "observed_difference": observed,
            "ci_95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
            "resamples": BOOTSTRAP_RESAMPLES, "seed": seed,
            "strata": "10 classes, 500 fixed paired DEV rows resampled within each class"}


def _evaluate_branch(
    branch: str, run_id: str, branch_dir: Path, model: torch.nn.Module,
    evaluator: torch.nn.Module, ec_evaluator: ThirdConfirmationEvaluator,
    source: dict[str, Any], cohort: dict[str, Any], selection: list[dict[str, int]],
    pixels: torch.Tensor, labels: torch.Tensor, dev_indices: torch.Tensor,
    device: torch.device, config: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    reconstruction, dev_display = _evaluate_reconstruction(
        model, evaluator, ec_evaluator, pixels, labels, dev_indices, device, source)
    dev_display.update({"task_id": TASK_ID, "run_id": run_id, "branch": branch,
                        "global_step": ENDPOINT_STEP})
    atomic_torch_save(branch_dir / "dev_display_examples.pt", dev_display)
    _write_rows_csv(branch_dir / "reconstruction_metrics_dev.csv", reconstruction["per_image"])
    base._write_contact_sheet(branch_dir / "dev_preview.png", dev_display["view28"],
                              dev_display["labels"], 2)
    base._write_contact_sheet(branch_dir / "dev_target_preview.png", dev_display["target28"],
                              dev_display["labels"], 2)

    generation, generated = _evaluate_generation(
        model, evaluator, ec_evaluator, source, cohort, selection, labels, pixels,
        dev_indices, device)
    generated.update({"task_id": TASK_ID, "run_id": run_id, "branch": branch,
                      "global_step": ENDPOINT_STEP})
    atomic_torch_save(branch_dir / "generated_cohort.pt", generated)
    saved = torch.load(branch_dir / "generated_cohort.pt", map_location="cpu", weights_only=False)
    _require(saved.get("task_id") == TASK_ID and saved.get("branch") == branch
             and saved.get("global_step") == ENDPOINT_STEP
             and tuple(saved["images"].shape) == (10_000, 1, 28, 28)
             and tuple(saved["ed_logits"].shape) == (10_000, 10)
             and tuple(saved["ec_logits"].shape) == (10_000, 10)
             and torch.equal(saved["proposal_ids"], cohort["proposal_ids"]),
             f"{branch} saved fixed G4 generation consumer failed readback")
    _write_rows_csv(branch_dir / "generation_metrics_per_image.csv", generation["per_image"])
    base._write_contact_sheet(branch_dir / "generation_preview.png", saved["images"],
                              saved["requested_labels"], 10)
    result = {
        "branch": branch, "global_step": ENDPOINT_STEP,
        "reconstruction": {key: value for key, value in reconstruction.items() if key != "per_image"},
        "generation": {key: value for key, value in generation.items() if key != "per_image"},
        "measurement_complete": len(reconstruction["per_image"]) == 5000
            and len(generation["per_image"]) == 10_000,
        "elapsed_seconds_evaluation": time.perf_counter() - started,
        "training_trace": json.loads((branch_dir / "training_history.json").read_text(encoding="utf-8")),
    }
    _require(result["measurement_complete"], f"{branch} endpoint measurement is incomplete")
    return result


def _run_branch(
    branch: str, run_dir: Path, run_id: str, source: dict[str, Any], config: dict[str, Any],
    schedule: torch.Tensor, gradient_weight: float, pixels: torch.Tensor, labels: torch.Tensor,
    train_indices: torch.Tensor, dev_indices: torch.Tensor, cohort: dict[str, Any],
    selection: list[dict[str, int]], evaluator: torch.nn.Module,
    ec_evaluator: ThirdConfirmationEvaluator, device: torch.device,
    resume_checkpoint: Path | None, progress: dict[str, Any],
) -> dict[str, Any]:
    branch_dir = run_dir / branch
    _require(branch_dir.is_dir(), f"registered branch directory is missing: {branch_dir}")
    summary_path = branch_dir / "endpoint_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        endpoint_path = branch_dir / "checkpoint_step_005000.pt"
        _require(summary.get("branch") == branch and summary.get("measurement_complete") is True
                 and endpoint_path.is_file() and "training" in summary
                 and "endpoint_window" in summary and "independent_readout" in summary
                 and (branch_dir / "reconstruction_metrics_dev.csv").is_file()
                 and (branch_dir / "generation_metrics_per_image.csv").is_file()
                 and (branch_dir / "generated_cohort.pt").is_file(),
                 f"{branch} completed endpoint summary is partial or invalid")
        endpoint = torch.load(endpoint_path, map_location="cpu", weights_only=False)
        _validate_checkpoint(endpoint, branch, run_id, source, config,
                             schedule[SOURCE_STEP:], gradient_weight)
        expected_density_steps = list(range(SOURCE_STEP, ENDPOINT_STEP + 1, CHECKPOINT_INTERVAL))
        density_record = base._validate_gaussian_plane_density_coverage(
            run_dir, branch, expected_density_steps, 2048,
        )
        _require(summary.get("training", {}).get("plane_power_density_logging") == density_record,
                 f"{branch} completed summary density coverage differs from its saved records")
        return summary

    started = time.perf_counter()
    history: list[dict[str, Any]]
    elapsed_before = 0.0
    if resume_checkpoint is not None:
        resume_checkpoint = resume_checkpoint.resolve()
        _require(resume_checkpoint.is_file() and resume_checkpoint.parent.resolve() == branch_dir.resolve(),
                 "explicit progress checkpoint is missing or outside its branch")
        checkpoint = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        start = _validate_checkpoint(checkpoint, branch, run_id, source, config,
                                     schedule[SOURCE_STEP:], gradient_weight)
        _require_progress_matches_checkpoint(progress, branch, start)
        model, optimizer = _restore_checkpoint(checkpoint, config, device)
        history = checkpoint.get("history")
        _require(isinstance(history, list), "resume checkpoint training history is missing")
        elapsed_before = float(checkpoint.get("elapsed_seconds_continuation", 0.0))
        _require(math.isfinite(elapsed_before) and elapsed_before >= 0.0,
                 "resume checkpoint elapsed time is invalid")
    else:
        _require(not (branch_dir / "checkpoint_latest.pt").exists(),
                 f"{branch} has an unregistered checkpoint; explicit resume is required")
        model, optimizer = _new_model_optimizer(source, config, device)
        source_cp = source["checkpoint"]
        torch.cuda.set_rng_state_all([state.cpu() for state in source_cp["cuda_rng"]])
        torch.random.set_rng_state(source_cp["cpu_rng"])
        history = []
        start = 0
        initial = _checkpoint_payload(model, optimizer, branch, 0, run_id, source, config,
                                      schedule[SOURCE_STEP:], gradient_weight, history, 0.0)
        _save_checkpoint(branch_dir, initial, 0)
        progress.setdefault("branch_checkpoints", {})[branch] = str(branch_dir / "checkpoint_latest.pt")
        progress.setdefault("branch_completed_updates", {})[branch] = 0
        progress["current_branch"] = branch
        progress["current_stage"] = "TRAINING"
        _write_progress(run_dir, progress)

    density_start_step = SOURCE_STEP
    base._record_gaussian_plane_density_probe(
        model, pixels, labels, dev_indices, run_dir, branch=branch,
        global_step=SOURCE_STEP + start, grid_size=2048,
    )

    encoder, decoder = base._phase_groups(model)
    encoder_state = encoder.detach().clone()
    encoder_adam_state = copy.deepcopy(optimizer.state[encoder])
    buffers_state = {name: value.detach().clone() for name, value in model.named_buffers()}
    evaluator_state = base._state_snapshot(evaluator)
    ec_state = base._state_snapshot(ec_evaluator)
    for cursor in range(start, CONTINUATION_UPDATES):
        segment_step = cursor + 1
        loss = _train_update(model, optimizer, pixels, schedule[SOURCE_STEP + cursor], device,
                             source, branch, gradient_weight if branch == "Dgrad" else None)
        if segment_step % PROGRESS_INTERVAL == 0:
            history.append({"segment_step": segment_step, **loss})
            atomic_write_json(branch_dir / "training_history_live.json", history)
            progress.update({
                "current_branch": branch, "current_stage": "TRAINING",
                "branch_completed_updates": {**progress.get("branch_completed_updates", {}), branch: segment_step},
                "branch_checkpoints": {**progress.get("branch_checkpoints", {}),
                                       branch: str(branch_dir / "checkpoint_latest.pt")},
                "latest_loss": loss, "latest_global_step": SOURCE_STEP + segment_step,
                "training_elapsed_seconds_this_process": time.perf_counter() - started,
                "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
            })
            _write_progress(run_dir, progress)
        if segment_step % CHECKPOINT_INTERVAL == 0 or segment_step in (2500, CONTINUATION_UPDATES):
            payload = _checkpoint_payload(
                model, optimizer, branch, segment_step, run_id, source, config,
                schedule[SOURCE_STEP:], gradient_weight, history,
                elapsed_before + time.perf_counter() - started,
            )
            _save_checkpoint(branch_dir, payload, segment_step)
            base._record_gaussian_plane_density_probe(
                model, pixels, labels, dev_indices, run_dir, branch=branch,
                global_step=SOURCE_STEP + segment_step, grid_size=2048,
            )
            progress.update({
                "current_branch": branch, "current_stage": "TRAINING",
                "branch_completed_updates": {**progress.get("branch_completed_updates", {}), branch: segment_step},
                "branch_checkpoints": {**progress.get("branch_checkpoints", {}),
                                       branch: str(branch_dir / "checkpoint_latest.pt")},
                "last_complete_save_unix": time.time(),
            })
            _write_progress(run_dir, progress)

    _require(torch.equal(encoder_state, encoder) and base._assert_close(
        encoder_adam_state, optimizer.state[encoder], f"{branch}_frozen_encoder_adam") == 0.0
        and base._assert_close(buffers_state, dict(model.named_buffers()), f"{branch}_frozen_buffers") == 0.0,
        f"{branch} training changed frozen Encoder, Adam, or buffers")
    _require(base._assert_close(evaluator_state, base._state_snapshot(evaluator), f"{branch}_E_D_train") == 0.0
             and base._assert_close(ec_state, base._state_snapshot(ec_evaluator), f"{branch}_E_C_train") == 0.0,
             f"{branch} training changed a frozen evaluator")
    steps = _optimizer_summary(optimizer, model)
    _require(steps == {"encoder_adam_step": SOURCE_STEP, "decoder_adam_step": ENDPOINT_STEP},
             f"{branch} per-phase Adam steps do not match 45,000/50,000")

    endpoint_grid, endpoint_window = base._window_check(
        "small", pixels, labels, schedule, config, device, model=model)
    _require(endpoint_grid == 2048 and endpoint_window["complex_relative_error_max"] <= WINDOW_TOLERANCE
             and endpoint_window["ols_coefficient_relative_error_by_grid"][2048] <= WINDOW_TOLERANCE
             and endpoint_window["camera_w_per_pixel_relative_error_by_grid"][2048] <= WINDOW_TOLERANCE,
             f"{branch} endpoint failed the registered 2048/4096 field, OLS, or camera check")
    first_batch = image_batch(pixels, schedule[SOURCE_STEP], "small", device)
    readout = base._small_producer_consumer_check(model, first_batch[:1])
    _require(readout.get("status") == "PASSED", f"{branch} endpoint independent OLS/readout check failed")
    endpoint_path = branch_dir / "checkpoint_step_005000.pt"
    if start < CONTINUATION_UPDATES:
        endpoint = _checkpoint_payload(
            model, optimizer, branch, CONTINUATION_UPDATES, run_id, source, config,
            schedule[SOURCE_STEP:], gradient_weight, history,
            elapsed_before + time.perf_counter() - started)
        _save_checkpoint(branch_dir, endpoint, CONTINUATION_UPDATES)
    else:
        _require(endpoint_path.is_file(), f"{branch} endpoint checkpoint is missing during evaluation resume")
        endpoint = torch.load(endpoint_path, map_location="cpu", weights_only=False)
        _validate_checkpoint(endpoint, branch, run_id, source, config,
                             schedule[SOURCE_STEP:], gradient_weight)
        _require(base._assert_close(endpoint["model"], base._state_snapshot(model),
                                    f"{branch}_endpoint_resume_model") == 0.0
                 and base._assert_close(endpoint["optimizer"], cpu_copy(optimizer.state_dict()),
                                        f"{branch}_endpoint_resume_optimizer") == 0.0,
                 f"{branch} resumed model or optimizer differs from its saved endpoint")
    history_path = branch_dir / "training_history.json"
    if history_path.is_file():
        _require(json.loads(history_path.read_text(encoding="utf-8")) == history,
                 f"{branch} saved training history differs from the endpoint checkpoint")
    else:
        atomic_write_json(history_path, history)
    terminal_path = branch_dir / "terminal_model.pt"
    if terminal_path.is_file():
        terminal = torch.load(terminal_path, map_location="cpu", weights_only=False)
        _require(terminal.get("task_id") == TASK_ID and terminal.get("run_id") == run_id
                 and terminal.get("branch") == branch and terminal.get("global_step") == ENDPOINT_STEP
                 and base._assert_close(terminal.get("model", {}), endpoint["model"],
                                        f"{branch}_terminal_model_resume") == 0.0,
                 f"{branch} existing terminal model does not match its endpoint")
    else:
        atomic_torch_save(terminal_path, {
            "schema": "lightgen_small_decoder_native_gradient_terminal.v1", "task_id": TASK_ID,
            "run_id": run_id, "branch": branch, "global_step": ENDPOINT_STEP,
            "model_identity": source["checkpoint"]["model_identity"], "model": endpoint["model"],
            "optimizer_steps": steps,
        })
    result = _evaluate_branch(
        branch, run_id, branch_dir, model, evaluator, ec_evaluator, source, cohort, selection,
        pixels, labels, dev_indices, device, config)
    result["endpoint_window"] = {
        "status": "PASSED", "selected_production_grid": 2048,
        "complex_relative_error_max": endpoint_window["complex_relative_error_max"],
        "ols_coefficient_relative_error": endpoint_window["ols_coefficient_relative_error_by_grid"][2048],
        "camera_w_per_pixel_relative_error": endpoint_window["camera_w_per_pixel_relative_error_by_grid"][2048],
    }
    result["independent_readout"] = readout
    result["optimizer_steps"] = steps
    density_steps = list(range(density_start_step, ENDPOINT_STEP + 1, CHECKPOINT_INTERVAL))
    plane_density_logging = base._validate_gaussian_plane_density_coverage(
        run_dir, branch, density_steps, 2048,
    )
    result["training"] = {
        "source_global_step": SOURCE_STEP, "new_scientific_updates": CONTINUATION_UPDATES,
        "endpoint_global_step": ENDPOINT_STEP, "optimizer_steps": steps,
        "history": history,
        "elapsed_seconds_training": elapsed_before + time.perf_counter() - started,
        "seconds_per_scientific_update": (elapsed_before + time.perf_counter() - started) / CONTINUATION_UPDATES,
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
        "device": torch.cuda.get_device_name(device),
        "plane_power_density_logging": plane_density_logging,
    }
    atomic_write_json(summary_path, result)
    progress.update({
        "current_branch": branch, "current_stage": "BRANCH_COMPLETE",
        "branch_completed_updates": {**progress.get("branch_completed_updates", {}), branch: CONTINUATION_UPDATES},
        "completed_branches": list(dict.fromkeys(progress.get("completed_branches", []) + [branch])),
        "branch_checkpoints": {**progress.get("branch_checkpoints", {}),
                               branch: str(branch_dir / "checkpoint_step_005000.pt")},
        "last_complete_save_unix": time.time(),
    })
    _write_progress(run_dir, progress)
    return result


def _save_gray_png(value: torch.Tensor, path: Path) -> None:
    image = value.detach().cpu().float().squeeze()
    _require(image.ndim == 2 and image.shape == (200, 200) and torch.isfinite(image).all().item()
             and float(image.min()) >= 0.0 and float(image.max()) <= 1.0,
             "blind-review images must be native 200x200 grayscale in [0,1]")
    pixels = np.rint(image.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(path, format="PNG", optimize=True)


def _render_blind_page(review_id: str, exposed: list[dict[str, str]]) -> str:
    cards_json = json.dumps(exposed, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>LightGen盲评</title>
<style>body{{font:18px system-ui,sans-serif;max-width:900px;margin:1rem auto;padding:0 1rem;color:#18212b}}main{{text-align:center}}img{{width:400px;height:400px;object-fit:contain;image-rendering:pixelated;border:1px solid #999;background:#fff}}.choices{{display:grid;grid-template-columns:repeat(6,1fr);gap:.4rem;margin:1rem 0}}label{{padding:.5rem;border:1px solid #ccd1d1;border-radius:.4rem}}button{{font-size:1rem;padding:.6rem 1rem;margin:.3rem}}#section_break{{display:none;padding:3rem;background:#f3f6f8}}.hidden{{display:none}}</style>
<h1>图像评阅</h1><p>按图像本身选择数字；无法辨认时选“不可辨认”。灰度固定，不逐图增强。每20张可休息。</p>
<main><div id='section_break'><h2>本段完成</h2><p>可以休息，准备好后继续下一段。</p><button id='continue'>继续</button></div>
<div id='card'><p id='progress'></p><img id='image' alt='当前图像'><div id='choices' class='choices'></div>
<button id='prev'>上一张</button><button id='next'>下一张</button><button id='save'>导出全部评分</button></div><p id='save_status'></p></main>
<script>
const TASK_ID={json.dumps(TASK_ID)},REVIEW_ID={json.dumps(review_id)},CARDS={cards_json};
const STORE='lightgen-native-gradient-'+REVIEW_ID;let state={{}};let position=0;let lastRestBoundary=-1;
try{{state=JSON.parse(localStorage.getItem(STORE)||'{{}}')}}catch{{state={{}}}}
function persist(){{localStorage.setItem(STORE,JSON.stringify(state));document.getElementById('progress').textContent=`第 ${{position+1}} / ${{CARDS.length}} 张 · 已完成 ${{Object.keys(state).length}} 张 · 第 ${{Math.floor(position/20)+1}} / 12 段`;}}
function render(){{const rest=position>0&&position%20===0&&state[CARDS[position-1].card_id]!==undefined&&lastRestBoundary!==position;
document.getElementById('section_break').style.display=rest?'block':'none';document.getElementById('card').style.display=rest?'none':'block';if(rest){{lastRestBoundary=position;return;}}
const c=CARDS[position];document.getElementById('image').src=c.image;document.getElementById('choices').innerHTML='';document.getElementById('save_status').textContent='';
const opts=[...Array(10)].map((_,i)=>[String(i),String(i)]).concat([['unreadable','不可辨认']]);
for(const [value,labelText] of opts){{const label=document.createElement('label');const radio=document.createElement('input');radio.type='radio';radio.name='rating';radio.value=value;radio.checked=state[c.card_id]===value;radio.onchange=()=>{{state[c.card_id]=value;persist();render();}};label.append(radio,document.createTextNode(' '+labelText));document.getElementById('choices').append(label);}}persist();}}
document.getElementById('prev').onclick=()=>{{position=Math.max(0,position-1);render();}};
document.getElementById('next').onclick=()=>{{if(state[CARDS[position].card_id]===undefined){{document.getElementById('save_status').textContent='请先为当前图像选择结果。';return;}}position=Math.min(CARDS.length-1,position+1);render();}};
document.getElementById('continue').onclick=()=>{{render();}};
document.getElementById('save').onclick=()=>{{if(Object.keys(state).length!==CARDS.length){{document.getElementById('save_status').textContent='请完成全部240张评分。';return;}}
const cards=CARDS.map(c=>{{const v=state[c.card_id];return {{card_id:c.card_id,rating:v==='unreadable'?'unreadable':Number(v)}};}});
const payload={{schema:'{RATINGS_SCHEMA}',task_id:TASK_ID,review_id:REVIEW_ID,cards}};const blob=new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='lightgen_decoder_gradient_ratings_'+REVIEW_ID+'.json';a.click();URL.revokeObjectURL(a.href);document.getElementById('save_status').textContent='评分文件已导出。';}};render();</script></html>"""


def _shuffle_review_cards(cards: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], int]:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for index, card in enumerate(cards):
        if card.get("duplicate_group"):
            groups.setdefault(str(card["duplicate_group"]), []).append(index)
    _require(len(cards) == 240 and len(groups) == 20
             and all(len(group) == 2 for group in groups.values()),
             "review population or duplicate groups are incomplete")
    for attempt in range(1, 1_000_001):
        order = rng.permutation(len(cards)).tolist()
        position = {original: shuffled for shuffled, original in enumerate(order)}
        minimum = min(abs(position[a] - position[b]) for a, b in groups.values())
        if minimum >= 40:
            return [cards[index] for index in order], attempt
    raise RuntimeError("registered deterministic review shuffle did not satisfy duplicate spacing")


def _write_blind_review(
    run_dir: Path, config: dict[str, Any], source: dict[str, Any],
    selection: list[dict[str, int]], pixels: torch.Tensor, labels: torch.Tensor,
    dev_indices: torch.Tensor,
) -> dict[str, Any]:
    key_path = run_dir / "blind_review_key.json"
    page_path = run_dir / "blind_review.html"
    manifest_path = run_dir / "blind_review_public_manifest.json"
    if key_path.exists() or page_path.exists() or manifest_path.exists():
        _require(key_path.is_file() and page_path.is_file() and manifest_path.is_file(),
                 "partial blind-review artifacts cannot be resumed")
        key = json.loads(key_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _require(key.get("schema") == REVIEW_SCHEMA and key.get("task_id") == TASK_ID
                 and len(key.get("cards", [])) == 240
                 and manifest.get("schema") == REVIEW_SCHEMA and manifest.get("task_id") == TASK_ID
                 and manifest.get("review_id") == key.get("review_id") and manifest.get("card_count") == 240,
                 "saved blind-review key is incomplete")
        page = page_path.read_text(encoding="utf-8")
        match = re.search(r"const TASK_ID=(\"(?:\\.|[^\"\\])*\"),REVIEW_ID=(\"(?:\\.|[^\"\\])*\"),CARDS=", page)
        _require(match is not None, "saved blind-review page identity is malformed")
        page_task_id, page_review_id = json.loads(match.group(1)), json.loads(match.group(2))
        page_cards, _ = json.JSONDecoder().raw_decode(page, match.end())
        expected_public = [{"card_id": card["card_id"], "image": card["image"]} for card in key["cards"]]
        card_root = (run_dir / "review_cards").resolve()
        _require(page_task_id == TASK_ID and page_review_id == key["review_id"]
                 and page_cards == expected_public and len(list(card_root.glob("*.png"))) == 240,
                 "saved blind-review page mapping or image population is incomplete")
        for card in key["cards"]:
            image_path = (run_dir / card["image"]).resolve()
            _require(image_path.is_relative_to(card_root) and image_path.is_file(),
                     f"saved blind-review card is missing: {card['card_id']}")
            with Image.open(image_path) as image:
                _require(image.mode == "L" and image.size == (200, 200),
                         f"saved blind-review card format changed: {card['card_id']}")
        duplicate_groups: dict[str, list[dict[str, Any]]] = {}
        for card in key["cards"]:
            if card.get("duplicate_group"):
                duplicate_groups.setdefault(str(card["duplicate_group"]), []).append(card)
        _require(len(duplicate_groups) == 20 and all(len(group) == 2 for group in duplicate_groups.values()),
                 "saved blind-review duplicate mapping is incomplete")
        _require(key.get("minimum_duplicate_card_distance", 0) >= 40,
                 "saved blind-review duplicate spacing is below the registered minimum")
        for group in duplicate_groups.values():
            _require((run_dir / group[0]["image"]).read_bytes() == (run_dir / group[1]["image"]).read_bytes(),
                     "saved blind-review duplicate image pair changed")
        return {"review_id": key["review_id"], "cards": 240, "reused_complete_manifest": True}
    cards: list[dict[str, Any]] = []
    for branch in BRANCHES:
        path = run_dir / branch / "generated_cohort.pt"
        _require(path.is_file(), f"{branch} generated fixed G4 output is missing")
        generated = torch.load(path, map_location="cpu", weights_only=False)
        _require(generated.get("branch") == branch and generated.get("global_step") == ENDPOINT_STEP,
                 f"{branch} G4 output endpoint identity changed")
        indices = generated["native_capture_indices"].long()
        images = generated["native_capture_images"]
        image_by_index = {int(index): image for index, image in zip(indices.tolist(), images, strict=True)}
        _require(len(image_by_index) == 100, f"{branch} native human sample views are incomplete")
        for selected in selection:
            index = int(selected["cohort_row"])
            _require(index in image_by_index and
                     int(generated["proposal_ids"][index]) == selected["proposal_id"] and
                     int(generated["requested_labels"][index]) == selected["requested_label"],
                     f"{branch} selected G4 image identity changed")
            cards.append({
                "kind": "generated", "branch": branch,
                "requested_label": int(selected["requested_label"]),
                "cohort_row": index, "proposal_id": int(selected["proposal_id"]),
                "stream": 0, "duplicate_group": None, "image_tensor": image_by_index[index],
            })
    _require(len(cards) == 200, "blind review needs 100 same-coefficient G4 images per branch")
    by_branch_class = {(card["branch"], card["requested_label"]): card for card in cards}
    for branch in BRANCHES:
        for label in range(10):
            original = by_branch_class[(branch, label)]
            duplicate_group = f"duplicate_{branch}_{label}"
            original["duplicate_group"] = duplicate_group
            cards.append({**original, "kind": "duplicate", "image_tensor": original["image_tensor"].clone()})

    anchor_rows: list[int] = []
    for label in range(10):
        rows = dev_indices[labels[dev_indices] == label]
        _require(rows.numel() == 500, f"DEV class {label} anchor population changed")
        anchor_rows.extend(map(int, rows[2:4].tolist()))
    anchor_tensor = torch.tensor(anchor_rows, dtype=torch.long)
    anchor_images = image_batch(pixels, anchor_tensor, "small", torch.device("cpu"))
    for image, row in zip(anchor_images, anchor_rows, strict=True):
        cards.append({"kind": "real_anchor", "row": row, "target_label": int(labels[row]),
                      "branch": None, "requested_label": None, "cohort_row": None,
                      "proposal_id": None, "stream": None, "duplicate_group": None,
                      "image_tensor": image})
    shuffled, shuffle_attempt = _shuffle_review_cards(cards, int(config["human_review"]["shuffle_seed"]))
    review_id = uuid.uuid4().hex
    cards_dir = run_dir / "review_cards"
    cards_dir.mkdir(parents=True, exist_ok=False)
    public_cards: list[dict[str, str]] = []
    hidden_cards: list[dict[str, Any]] = []
    for index, card in enumerate(shuffled, start=1):
        card_id = f"R{index:04d}"
        relative_image = f"review_cards/card_{index:04d}.png"
        _save_gray_png(card["image_tensor"], run_dir / relative_image)
        public_cards.append({"card_id": card_id, "image": relative_image})
        hidden_cards.append({key: value for key, value in card.items() if key != "image_tensor"} |
                            {"card_id": card_id, "image": relative_image})
    duplicate_positions: dict[str, list[int]] = {}
    for index, card in enumerate(hidden_cards):
        if card.get("duplicate_group"):
            duplicate_positions.setdefault(card["duplicate_group"], []).append(index)
    min_distance = min(abs(duplicate_positions[key][0] - duplicate_positions[key][1])
                       for key in duplicate_positions)
    key = {
        "schema": REVIEW_SCHEMA, "task_id": TASK_ID, "review_id": review_id,
        "run_id": json.loads((run_dir / "run_identity.json").read_text(encoding="utf-8"))["run_id"],
        "registered_card_count": 240, "generated_unique_card_count": 200,
        "hidden_duplicate_count": 20, "real_anchor_count": 20,
        "shuffle_seed": int(config["human_review"]["shuffle_seed"]),
        "shuffle_attempt": shuffle_attempt, "minimum_duplicate_card_distance": min_distance,
        "anchor_source": "DEV target rows 3 and 4 per class, bilinear 28-to-200 view; not an optical output",
        "display": {"source_image_pixels": 200, "css_display_pixels": 400,
                    "range": "fixed gain then clamp [0,1]; no per-image stretch"},
        "cards": hidden_cards,
    }
    atomic_write_json(key_path, key)
    atomic_write_json(run_dir / "blind_review_public_manifest.json", {
        "schema": REVIEW_SCHEMA, "task_id": TASK_ID, "review_id": review_id,
        "card_count": 240, "rating_values": list(range(10)) + ["unreadable"],
        "display_source_pixels": 200, "display_css_pixels": 400,
        "segments": 12, "cards_per_segment": 20,
    })
    page_path.write_text(_render_blind_page(review_id, public_cards), encoding="utf-8")
    page = page_path.read_text(encoding="utf-8")
    _require(len(list(cards_dir.glob("*.png"))) == 240 and
             all(field not in page for field in ("requested_label", "branch", "proposal_id", "target_label"))
             and "width:400px;height:400px" in page and "image-rendering:pixelated" in page,
             "blind page export leaked identity or changed fixed image presentation")
    return {"review_id": review_id, "cards": 240, "generated_unique": 200,
            "duplicates": 20, "anchors": 20, "minimum_duplicate_card_distance": min_distance,
            "shuffle_attempt": shuffle_attempt, "blind_page": "blind_review.html",
            "hidden_key": "blind_review_key.json"}


def _paired_generation_summary(
    d0: list[dict[str, Any]], dgrad: list[dict[str, Any]],
) -> dict[str, Any]:
    a = d0
    b = dgrad
    _require(len(a) == len(b) == 10_000 and
             all((x["requested_label"], x["proposal_id"], x["component_id"])
                 == (y["requested_label"], y["proposal_id"], y["component_id"])
                 for x, y in zip(a, b, strict=True)),
             "fixed G4 paired generation identities differ across branches")
    return {
        "same_ols_coefficients_and_proposal_ids": True,
        "ed_correctness_rate_difference_Dgrad_minus_D0": float(np.mean([
            int(y["ed_prediction"] == y["requested_label"]) - int(x["ed_prediction"] == x["requested_label"])
            for x, y in zip(a, b, strict=True)])),
        "ec_correctness_rate_difference_Dgrad_minus_D0": float(np.mean([
            int(y["ec_prediction"] == y["requested_label"]) - int(x["ec_prediction"] == x["requested_label"])
            for x, y in zip(a, b, strict=True)])),
        "proxy_clipping_area_fraction_difference_Dgrad_minus_D0": float(np.mean([
            float(y["proxy_clipping_area_fraction"]) - float(x["proxy_clipping_area_fraction"])
            for x, y in zip(a, b, strict=True)])),
        "view_mean_intensity_difference_Dgrad_minus_D0": float(np.mean([
            float(y["view_mean_intensity"]) - float(x["view_mean_intensity"])
            for x, y in zip(a, b, strict=True)])),
    }


def _write_machine_html(run_dir: Path, report: dict[str, Any]) -> None:
    rows = []
    for branch in BRANCHES:
        result = report["endpoints"][branch]
        recon, gen = result["reconstruction"], result["generation"]
        rows.append(f"<tr><td>{branch}</td><td>{recon['psnr_db_mean']:.4f}</td>"
                    f"<td>{recon['ssim_mean']:.5f}</td><td>{gen['ed_macro_recall']:.5f}</td>"
                    f"<td>{gen['ec_macro_recall']:.5f}</td><td>{gen['proxy_clipping_area_fraction_macro']:.4%}</td></tr>")
    ci = report["primary_effect"]["ci_95"]
    body = (
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Gaussian Decoder原生梯度对照</title><style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}"
        "table{border-collapse:collapse}td,th{padding:.5rem;border:1px solid #ccd1d1}img{max-width:100%}</style>"
        f"<h1>冻结Encoder与G4的Decoder梯度项对照</h1><p>状态：{html.escape(report['status'])}</p>"
        f"<p>主终点SSIM差（Dgrad−D0）：{report['primary_effect']['observed_difference']:.6f}; 95%区间 [{ci[0]:.6f}, {ci[1]:.6f}]。</p>"
        "<p>生成是同一批保存G4系数的冻结Decoder重放；DEV与生成指标使用固定灰度，不逐图拉伸。</p>"
        "<table><tr><th>分支</th><th>DEV PSNR(dB)</th><th>DEV SSIM</th><th>G4 E_D宏召回</th><th>G4 E_C宏召回</th><th>G4代理截断</th></tr>"
        + "".join(rows) + "</table>"
    )
    for branch in BRANCHES:
        body += f"<h2>{branch}</h2><h3>DEV目标</h3><img src='{branch}/dev_target_preview.png'>"
        body += f"<h3>DEV重构</h3><img src='{branch}/dev_preview.png'>"
        body += f"<h3>固定G4生成</h3><img src='{branch}/generation_preview.png'>"
    body += "<p>固定盲评卡片见 blind_review.html；揭盲在正式评分导入后生成。</p></html>"
    (run_dir / "report.html").write_text(body, encoding="utf-8")


def _elapsed_seconds_summary(
    preflight: dict[str, Any], endpoint_summaries: dict[str, dict[str, Any]],
    final_process_seconds: float, recovery_events: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    components = {
        "preflight": float(preflight["elapsed_seconds"]),
        "branch_training": sum(float(endpoint_summaries[branch]["training"]["elapsed_seconds_training"])
                                for branch in BRANCHES),
        "branch_evaluation": sum(float(endpoint_summaries[branch]["elapsed_seconds_evaluation"])
                                  for branch in BRANCHES),
        "final_process": float(final_process_seconds),
    }
    _require(all(math.isfinite(value) and value >= 0.0 for value in components.values()),
             "elapsed-time components must be finite and non-negative")
    if recovery_events:
        total = sum(components.values())
        scope = ("sum of saved successful preflight, branch training/evaluation, and final resume process; "
                 "excludes failed uncheckpointed work and elapsed repair gaps")
    else:
        total = components["final_process"]
        scope = "single uninterrupted entrypoint process wall time through report creation"
    return {"elapsed_seconds_total": total, "elapsed_seconds_components": components,
            "elapsed_seconds_total_scope": scope}


def _aggregate_machine_report(
    run_dir: Path, run_id: str, config: dict[str, Any], source: dict[str, Any],
    preflight: dict[str, Any], gradient_weight: float,
    endpoint_summaries: dict[str, dict[str, Any]], review: dict[str, Any],
    elapsed_total: float, recovery_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    with (run_dir / "D0" / "reconstruction_metrics_dev.csv").open("r", encoding="utf-8", newline="") as stream:
        d0_per = list(csv.DictReader(stream))
    with (run_dir / "Dgrad" / "reconstruction_metrics_dev.csv").open("r", encoding="utf-8", newline="") as stream:
        dg_per = list(csv.DictReader(stream))
    primary = _paired_bootstrap_rows(d0_per, dg_per, "ssim")
    low, high = primary["ci_95"]
    if low > 0.0:
        structural_conclusion = "NATIVE_GRADIENT_STRUCTURAL_IMPROVEMENT_SUPPORTED"
    elif high < 0.0:
        structural_conclusion = "NATIVE_GRADIENT_STRUCTURAL_EFFECT_WORSE"
    else:
        structural_conclusion = "NATIVE_GRADIENT_STRUCTURAL_EFFECT_UNCLEAR"
    with (run_dir / "D0" / "generation_metrics_per_image.csv").open("r", encoding="utf-8", newline="") as stream:
        d0_generation = list(csv.DictReader(stream))
    with (run_dir / "Dgrad" / "generation_metrics_per_image.csv").open("r", encoding="utf-8", newline="") as stream:
        dgrad_generation = list(csv.DictReader(stream))
    generation_paired = _paired_generation_summary(d0_generation, dgrad_generation)
    elapsed = _elapsed_seconds_summary(preflight, endpoint_summaries, elapsed_total, recovery_events)
    source_recon = source["endpoint_summary"].get("reconstruction", {})
    report = {
        "schema": "lightgen_small_decoder_native_gradient_report.v1",
        "task_id": TASK_ID, "run_id": run_id, "status": "AWAITING_HUMAN_RATINGS",
        "measurement_complete": True,
        "scientific_question": "With the C Encoder and fixed G4 coefficients frozen, does one calibrated native 200x200 gradient term improve Decoder reconstruction?",
        "decision": {"structural_conclusion": structural_conclusion,
                     "human_readability_conclusion": "PENDING_HUMAN_RATINGS",
                     "historical_system_quality_upgraded": False},
        "source": {
            "task_id": SOURCE_TASK_ID, "branch": "C", "run_id": source["checkpoint"]["run_id"],
            "global_step": SOURCE_STEP, "seed": SEED,
            "model_identity": source["checkpoint"]["model_identity"],
            "historical_status_preserved": "VALID_FAILED",
            "reported_reconstruction": {key: source_recon.get(key) for key in
                                         ("mse_mean", "psnr_db_mean", "ssim_mean", "saturation_fraction")},
            "source_C_quality_summary": {key: source["endpoint_summary"].get("quality_gate", {}).get(key)
                                         for key in ("status", "thresholds")},
        },
        "fixed_generation": {"task_id": GENERATION_TASK_ID, "model": "C_seed_2026101501",
                             "method": "G4", "stream": 0, "population_per_branch": 10_000,
                             "same_saved_ols_coefficients": True, "new_sampling": 0,
                             "prior_refitting": 0, "train_encoder_forward_for_prior": 0},
        "preflight": preflight,
        "engineering_recovery_history": recovery_events or [],
        "gradient_calibration": {"lambda_native_gradient": gradient_weight,
                                 "frozen_after_first_shared_continuation_batch": True},
        "training": {
            "branches": {branch: endpoint_summaries[branch]["training"] for branch in BRANCHES},
            "science_updates_each": CONTINUATION_UPDATES, "total_science_updates": 2 * CONTINUATION_UPDATES,
            "encoder_science_updates": 0, "decoder_science_updates_each": CONTINUATION_UPDATES,
            "decoder_adam_final_step_each": ENDPOINT_STEP, "encoder_adam_frozen_step_each": SOURCE_STEP,
            **elapsed,
        },
        "endpoints": endpoint_summaries,
        "primary_effect": primary,
        "paired_generation_effects": generation_paired,
        "quality_gate": {
            "historical_system_status": "VALID_FAILED", "status_changed": False,
            "thresholds": config.get("quality_gate", {
                "reconstruction_psnr_db_min": 20.0, "reconstruction_ssim_min": 0.80,
                "generation_macro_recall_min": 0.80, "generation_recall_per_class_min": 0.60,
                "proxy_clipping_fraction_max": 0.01, "generated_to_dev_tv_ratio_min": 0.25,
            }),
            "geometry_status": "PROJECT_GEOMETRY_UNMEASURED",
        },
        "human_review": {"status": "AWAITING_HUMAN_RATINGS", **review,
                         "primary_endpoint": "equal-class requested-label match on 200 paired G4 generated cards",
                         "quality_checks": {"duplicate_consistency_min": 18, "anchor_label_match_min": 18}},
        "artifacts": {
            "preflight": "preflight.json", "preflight_receipt": "preflight_ready.json",
            "progress": "progress.json", "selection": "g4_review_selection.json",
            "blind_page": "blind_review.html", "blind_key": "blind_review_key.json",
            "public_manifest": "blind_review_public_manifest.json", "report_html": "report.html",
        },
        "output": {"directory": str(run_dir), "bytes_before_final_report": base._directory_bytes(run_dir),
                   "budget_bytes": int(config["output"]["formal_result_budget_bytes"])},
        "limitations": [
            "the scientific inference is limited to Decoder-only continuation from this single C endpoint",
            "E_D and E_C are auxiliary frozen classifiers and are not human or physical CMOS qualification",
            "the fixed Gaussian geometry remains PROJECT_GEOMETRY_UNMEASURED",
            "the historical system quality state remains VALID_FAILED until its complete criteria are met",
        ],
    }
    if report["output"]["bytes_before_final_report"] > report["output"]["budget_bytes"]:
        report["status"] = "MEASUREMENT_INCOMPLETE"
        report["measurement_complete"] = False
    atomic_write_json(run_dir / "machine_report.json", report)
    atomic_write_json(run_dir / "report.json", report)
    _write_machine_html(run_dir, report)
    return report


def _rating_value(value: Any, card_id: str) -> int | str:
    _require(value == "unreadable" or (isinstance(value, int) and not isinstance(value, bool)
                                       and 0 <= value <= 9),
             f"rating for {card_id} must be 0-9 or 'unreadable'", ValueError)
    return value


def _validate_review_inputs(run_dir: Path, ratings_path: Path
                            ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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
             "machine endpoints are incomplete or the ratings have already been finalized")
    review_id = key.get("review_id")
    _require(key.get("schema") == REVIEW_SCHEMA and key.get("task_id") == TASK_ID
             and key.get("registered_card_count") == 240 and isinstance(review_id, str)
             and manifest.get("schema") == REVIEW_SCHEMA and manifest.get("task_id") == TASK_ID
             and manifest.get("review_id") == review_id and manifest.get("card_count") == 240,
             "hidden review key and public manifest identity or card count changed")
    _require(ratings.get("schema") == RATINGS_SCHEMA and ratings.get("task_id") == TASK_ID
             and ratings.get("review_id") == review_id and isinstance(ratings.get("cards"), list),
             "submitted ratings belong to another task or review")
    cards = key.get("cards")
    expected_ids = [f"R{index:04d}" for index in range(1, 241)]
    _require(isinstance(cards, list) and len(cards) == 240
             and [card.get("card_id") for card in cards] == expected_ids,
             "review key cards are missing, duplicated, or out of order")
    rating_cards = ratings["cards"]
    _require(all(isinstance(card, dict) and isinstance(card.get("card_id"), str)
                 for card in rating_cards), "ratings contain malformed card records", ValueError)
    rating_ids = [card["card_id"] for card in rating_cards]
    _require(len(rating_cards) == 240 and len(set(rating_ids)) == 240
             and set(rating_ids) == set(expected_ids),
             "ratings must contain every registered card exactly once", ValueError)
    rating_by_id = {card["card_id"]: _rating_value(card.get("rating"), card["card_id"])
                    for card in rating_cards}
    kinds = {kind: [card for card in cards if card.get("kind") == kind]
             for kind in ("generated", "duplicate", "real_anchor")}
    _require([len(kinds[name]) for name in ("generated", "duplicate", "real_anchor")] == [200, 20, 20],
             "blind review source counts differ from the registered design")
    generated_keys = [(card.get("branch"), card.get("requested_label"), card.get("cohort_row"),
                       card.get("proposal_id")) for card in kinds["generated"]]
    _require(len(set(generated_keys)) == 200 and all(branch in BRANCHES
             and isinstance(label, int) and 0 <= label <= 9 for branch, label, _, _ in generated_keys),
             "generated card identities or requested classes are malformed")
    for label in range(10):
        left = [card for card in kinds["generated"] if card.get("branch") == "D0"
                and card.get("requested_label") == label]
        right = [card for card in kinds["generated"] if card.get("branch") == "Dgrad"
                 and card.get("requested_label") == label]
        _require(len(left) == len(right) == 10
                 and {(card["cohort_row"], card["proposal_id"]) for card in left}
                 == {(card["cohort_row"], card["proposal_id"]) for card in right},
                 f"paired G4 review identities for class {label} are incomplete")
    duplicate_groups: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        if card.get("duplicate_group"):
            duplicate_groups.setdefault(str(card["duplicate_group"]), []).append(card)
    _require(len(duplicate_groups) == 20 and all(len(group) == 2 for group in duplicate_groups.values()),
             "hidden duplicate groups are incomplete")
    duplicate_pairs: list[dict[str, Any]] = []
    duplicate_consistency = 0
    card_by_id = {card["card_id"]: card for card in cards}
    for group_name, group in sorted(duplicate_groups.items()):
        originals = [card for card in group if card.get("kind") == "generated"]
        copies = [card for card in group if card.get("kind") == "duplicate"]
        _require(len(originals) == len(copies) == 1,
                 f"hidden duplicate group {group_name} is malformed")
        original, duplicate = originals[0], copies[0]
        fields = ("branch", "requested_label", "cohort_row", "proposal_id")
        _require(all(original.get(field) == duplicate.get(field) for field in fields),
                 f"duplicate group {group_name} references a different generated input")
        consistent = rating_by_id[original["card_id"]] == rating_by_id[duplicate["card_id"]]
        duplicate_consistency += int(consistent)
        duplicate_pairs.append({"group": group_name, "original_card_id": original["card_id"],
                                "duplicate_card_id": duplicate["card_id"], "consistent": consistent})
    anchors = kinds["real_anchor"]
    _require([sum(card.get("target_label") == label for card in anchors) for label in range(10)] == [2] * 10,
             "real target references are not exactly two per class")
    anchor_correct = sum(rating_by_id[card["card_id"]] == card.get("target_label") for card in anchors)

    page_text = page_path.read_text(encoding="utf-8")
    match = re.search(r"const TASK_ID=(\"(?:\\.|[^\"\\])*\"),REVIEW_ID=(\"(?:\\.|[^\"\\])*\"),CARDS=",
                      page_text)
    _require(match is not None and "image-rendering:pixelated" in page_text,
             "blind-review page task identity or fixed pixel display is missing")
    page_task_id, page_review_id = json.loads(match.group(1)), json.loads(match.group(2))
    page_cards, _ = json.JSONDecoder().raw_decode(page_text, match.end())
    expected_page = [{"card_id": card["card_id"], "image": card["image"]} for card in cards]
    _require(page_task_id == TASK_ID and page_review_id == review_id and page_cards == expected_page,
             "blind page task, review, or image mapping differs from its hidden key")
    card_root = (run_dir / "review_cards").resolve()
    for card in cards:
        image_path = (run_dir / card["image"]).resolve()
        _require(image_path.is_relative_to(card_root) and image_path.is_file(),
                 f"review image missing or outside card directory: {card['card_id']}")
        with Image.open(image_path) as image:
            _require(image.mode == "L" and image.size == (200, 200),
                     f"review card {card['card_id']} changed size or pixel format")
    for group in duplicate_groups.values():
        a, b = (run_dir / group[0]["image"]).read_bytes(), (run_dir / group[1]["image"]).read_bytes()
        _require(a == b, "duplicate card PNGs are not byte-identical")
    _require(key.get("minimum_duplicate_card_distance", 0) >= 40,
             "hidden duplicate separation is below the registered distance")
    return machine, key, {
        "ratings": ratings, "cards": cards, "card_by_id": card_by_id,
        "rating_by_id": rating_by_id, "duplicate_consistency_count": duplicate_consistency,
        "duplicate_pairs": duplicate_pairs, "anchor_correct_count": int(anchor_correct),
        "anchor_count": 20,
    }


def _human_branch_metrics(cards: list[dict[str, Any]], ratings: dict[str, int | str],
                         branch: str) -> dict[str, Any]:
    per_class = []
    for label in range(10):
        subset = [card for card in cards if card.get("kind") == "generated"
                  and card.get("branch") == branch and card.get("requested_label") == label]
        _require(len(subset) == 10, f"{branch} human class {label} sample is incomplete")
        values = [ratings[card["card_id"]] for card in subset]
        per_class.append({"requested_label": label, "count": len(values),
                          "requested_matches": sum(value == label for value in values),
                          "unreadable": sum(value == "unreadable" for value in values),
                          "other_digit": sum(isinstance(value, int) and value != label for value in values),
                          "requested_match_rate": float(np.mean([value == label for value in values]))})
    return {"branch": branch, "generated_count": 100,
            "equal_class_requested_match_rate": float(np.mean(
                [row["requested_match_rate"] for row in per_class])),
            "unreadable_count": sum(row["unreadable"] for row in per_class),
            "other_digit_count": sum(row["other_digit"] for row in per_class),
            "per_class": per_class}


def _human_paired_bootstrap(cards: list[dict[str, Any]], ratings: dict[str, int | str]
                            ) -> dict[str, Any]:
    by_branch_id: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = {}
    for card in cards:
        if card.get("kind") != "generated":
            continue
        identity = (int(card["requested_label"]), int(card["cohort_row"]), int(card["proposal_id"]))
        by_branch_id.setdefault(identity, {})[str(card["branch"])] = card
    by_class: list[np.ndarray] = []
    for label in range(10):
        pairs = []
        for (requested, _row, _proposal), branches in by_branch_id.items():
            if requested != label:
                continue
            _require(set(branches) == set(BRANCHES),
                     f"paired human coefficient identity is missing a branch in class {label}")
            a = ratings[branches["D0"]["card_id"]] == label
            b = ratings[branches["Dgrad"]["card_id"]] == label
            pairs.append(float(b) - float(a))
        _require(len(pairs) == 10, f"class {label} does not have ten paired human coefficient identities")
        by_class.append(np.asarray(pairs, dtype=np.float64))
    observed = float(np.mean([values.mean() for values in by_class]))
    rng = np.random.default_rng(HUMAN_BOOTSTRAP_SEED)
    samples = np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for index in range(BOOTSTRAP_RESAMPLES):
        samples[index] = np.mean([values[rng.integers(0, values.size, size=values.size)].mean()
                                  for values in by_class])
    return {"estimand": "equal-class paired Dgrad-minus-D0 requested-label match rate",
            "observed_difference": observed,
            "ci_95": [float(value) for value in np.quantile(samples, [0.025, 0.975])],
            "resamples": BOOTSTRAP_RESAMPLES, "seed": HUMAN_BOOTSTRAP_SEED,
            "pairing": "same saved G4 coefficient identity within each class"}


def _write_unblinded_gallery(run_dir: Path, cards: list[dict[str, Any]],
                             ratings: dict[str, int | str]) -> int:
    rows = []
    for card in cards:
        branch = card.get("branch") or "真实DEV参照"
        requested = card.get("requested_label")
        rating = ratings[card["card_id"]]
        metric = "" if card["kind"] == "real_anchor" else (
            "符合请求类别" if rating == requested else "不可辨认" if rating == "unreadable" else "可辨为其他数字")
        rows.append(
            f"<div class='item' data-kind='{html.escape(str(card['kind']))}' "
            f"data-branch='{html.escape(str(branch))}' data-label='{html.escape(str(requested))}'>"
            f"<h3>{card['card_id']} · {html.escape(str(card['kind']))} · {html.escape(str(branch))} · "
            f"请求 {html.escape(str(requested))} · 人工 {html.escape(str(rating))} {html.escape(metric)}</h3>"
            f"<img src='{html.escape(card['image'])}'></div>")
    page = ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>盲评揭盲</title>"
            "<style>body{font:16px system-ui;max-width:1300px;margin:2rem auto}.grid{display:grid;"
            "grid-template-columns:repeat(3,1fr);gap:1rem}.item{border:1px solid #ddd;padding:.5rem}"
            "img{width:200px;height:200px;image-rendering:pixelated}</style>"
            "<h1>240张盲评揭盲</h1><p>原生200×200，灰度固定[0,1]。</p><div class='grid'>"
            + "".join(rows) + "</div></html>")
    (run_dir / "unblinded_results.html").write_text(page, encoding="utf-8")
    return len(rows)


def _finalize_ratings(run_dir: Path, ratings_path: Path, config: dict[str, Any]) -> None:
    started = time.perf_counter()
    _require(not (run_dir / "final_report.json").exists(), "ratings have already been finalized")
    machine, key, validated = _validate_review_inputs(run_dir, ratings_path)
    identity_record = json.loads((run_dir / "run_identity.json").read_text(encoding="utf-8"))
    identity = ResearchRunIdentity.from_mapping(identity_record["identity"])
    authorization_id = f"{TASK_ID}:seed:{SEED}:preflight_and_execution"
    _require(identity.task_id == TASK_ID and identity.config_identity == config
             and identity_record.get("run_id") == machine.get("run_id"),
             "final rating run identity differs from the machine measurement")
    receipt = PreflightReceipt.from_mapping(
        json.loads((run_dir / "preflight_ready.json").read_text(encoding="utf-8")))
    control = ResearchRunControl(identity, authorization_id=authorization_id)
    control.mark_preflight_ready(receipt)
    control.resume_scientific_attempt(run_dir, f"decoder_native_gradient_seed_{SEED}", receipt)
    cards = validated["cards"]
    ratings = validated["rating_by_id"]
    d0 = _human_branch_metrics(cards, ratings, "D0")
    dgrad = _human_branch_metrics(cards, ratings, "Dgrad")
    effect = _human_paired_bootstrap(cards, ratings)
    review_cfg = config["human_review"]
    dup_pass = validated["duplicate_consistency_count"] >= int(review_cfg["duplicate_consistency_min"])
    anchor_pass = validated["anchor_correct_count"] >= int(review_cfg["anchor_label_match_min"])
    quality_ok = dup_pass and anchor_pass
    low, high = effect["ci_95"]
    if not quality_ok:
        human_conclusion = "REVIEW_INCONCLUSIVE"
    elif low > 0.0:
        human_conclusion = "HUMAN_REQUESTED_IDENTITY_IMPROVEMENT_SUPPORTED"
    elif high < 0.0:
        human_conclusion = "HUMAN_REQUESTED_IDENTITY_WORSE"
    else:
        human_conclusion = "HUMAN_INDEPENDENT_EFFECT_UNCLEAR"
    human_review = {
        "status": "COMPLETE_DIAGNOSTIC" if quality_ok else "REVIEW_INCONCLUSIVE",
        "review_id": key["review_id"], "rating_count": len(cards),
        "unique_generated_count": 200, "hidden_duplicate_count": 20, "real_anchor_count": 20,
        "quality_checks": {
            "duplicate_consistency": {"count": validated["duplicate_consistency_count"], "total": 20,
                                       "minimum": int(review_cfg["duplicate_consistency_min"]),
                                       "passed": dup_pass, "pairs": validated["duplicate_pairs"]},
            "real_anchor_label_match": {"count": validated["anchor_correct_count"], "total": 20,
                                         "minimum": int(review_cfg["anchor_label_match_min"]),
                                         "passed": anchor_pass},
        },
        "primary_endpoint": "equal-class requested-label match rate on 100 paired G4 inputs",
        "D0": d0, "Dgrad": dgrad, "Dgrad_minus_D0": effect,
        "conclusion": human_conclusion,
        "limitations": ["single reviewer and fixed 100-coefficient G4 population",
                        "the review does not qualify physical CMOS readability",
                        "class-level results are descriptive with ten images per class"],
        "elapsed_seconds_finalization": time.perf_counter() - started,
    }
    rating_rows: list[dict[str, Any]] = []
    for card in cards:
        value = ratings[card["card_id"]]
        requested = card.get("requested_label")
        rating_rows.append({
            "card_id": card["card_id"], "kind": card["kind"], "branch": card.get("branch"),
            "requested_label": requested, "human_rating": value,
            "matches_requested": None if card["kind"] == "real_anchor" else
                (value == requested if value != "unreadable" else False),
            "readable": value != "unreadable", "cohort_row": card.get("cohort_row"),
            "proposal_id": card.get("proposal_id"), "dev_row": card.get("row"),
            "target_label": card.get("target_label"), "duplicate_group": card.get("duplicate_group"),
        })
    _write_rows_csv(run_dir / "human_ratings_unblinded.csv", rating_rows)
    shutil.copyfile(ratings_path, run_dir / "human_ratings_submitted.json")
    machine["status"] = "COMPLETE_DIAGNOSTIC"
    machine["final_status"] = "COMPLETE_DIAGNOSTIC"
    machine["human_review_status"] = human_review["status"]
    machine["decision"]["human_readability_conclusion"] = human_conclusion
    machine["human_review"] = human_review
    machine["artifacts"].update({
        "human_ratings": "human_ratings_submitted.json",
        "human_ratings_unblinded": "human_ratings_unblinded.csv",
        "unblinded_results_html": "unblinded_results.html",
    })
    gallery_rows = _write_unblinded_gallery(run_dir, cards, ratings)
    machine["human_review"]["unblinded_gallery_rows"] = gallery_rows
    output_bytes = base._directory_bytes(run_dir)
    machine["output"]["bytes_before_final_report"] = output_bytes
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
        "status": control.state, "scientific_status": machine["status"],
        "human_review_conclusion": human_conclusion,
        "duplicate_consistency_passed": dup_pass, "anchor_label_match_passed": anchor_pass,
        "completed_global_step_each_branch": ENDPOINT_STEP,
        "measurement_complete": machine["measurement_complete"],
    })
    atomic_write_json(run_dir / "progress.json", {
        "schema": "lightgen_small_decoder_native_gradient_progress.v1",
        "task_id": TASK_ID, "run_id": machine["run_id"], "status": machine["status"],
        "current_stage": "FINALIZED", "human_review_status": human_review["status"],
        "completed_branches": list(BRANCHES), "last_heartbeat_unix": time.time(),
        "last_complete_save_unix": time.time(),
    })
    print(json.dumps({"status": machine["status"], "human_conclusion": human_conclusion,
                      "report": str(run_dir / "final_report.json")}, ensure_ascii=False), flush=True)


def _run(args: argparse.Namespace) -> None:
    config_path = (args.config if args.config.is_absolute() else ROOT / args.config).resolve()
    config = _read_config(config_path)
    run_dir = (args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir).resolve()
    expected_run_dir = (ROOT / config["output"]["experiment_root"]).resolve()
    _require(run_dir == expected_run_dir,
             "output directory differs from the explicitly registered attempt path")
    if args.recovery_worker_dir is not None:
        _recovery_worker(args.recovery_worker_dir.resolve(), config_path)
        return
    if args.ratings_file is not None:
        _require(not args.preflight_only and args.resume_state is None,
                 "rating finalization cannot be combined with preflight or resume")
        _finalize_ratings(run_dir, args.ratings_file.resolve(), config)
        return
    _require(torch.cuda.is_available(), "CUDA is required for the authorized Gaussian continuation")
    _require(args.source_checkpoint is not None and args.g4_cohort is not None
             and args.previous_review_dir is not None,
             "source checkpoint, fixed G4 cohort, and previous review directory are required")
    source_path = (args.source_checkpoint if args.source_checkpoint.is_absolute()
                   else ROOT / args.source_checkpoint).resolve()
    cohort_path = (args.g4_cohort if args.g4_cohort.is_absolute() else ROOT / args.g4_cohort).resolve()
    previous_review_dir = (args.previous_review_dir if args.previous_review_dir.is_absolute()
                           else ROOT / args.previous_review_dir).resolve()
    resuming = args.resume_state is not None
    if args.preflight_only:
        _require(not resuming, "preflight-only cannot be combined with resume")
        _require(not run_dir.exists(), f"refusing to run preflight over an existing attempt: {run_dir}")
    elif resuming:
        resume_state_path = args.resume_state.resolve()
        _require(run_dir.is_dir() and resume_state_path == (run_dir / "progress.json").resolve()
                 and not (run_dir / "machine_report.json").is_file(),
                 "resume requires the registered incomplete attempt and its explicit progress.json")
        saved_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        _require(saved_config == config, "resume config differs from the registered task")
    else:
        _require(not run_dir.exists(), f"refusing to overwrite existing attempt: {run_dir}")

    overall_started = time.perf_counter()
    source = _load_source(source_path, cohort_path, previous_review_dir, config)
    cohort = source["cohort"]
    pixels, labels, train_indices, dev_indices = base._load_data(ROOT / config["data"]["training_file"])
    _require(train_indices.numel() == 55_000 and dev_indices.numel() == 5_000
             and int(config["data"]["official_test_reads"]) == 0,
             "registered TRAIN/DEV roles or official TEST boundary changed")
    schedule = _schedule(labels, train_indices)
    source_schedule = source["checkpoint"].get("schedule")
    _require(isinstance(source_schedule, torch.Tensor) and tuple(source_schedule.shape) == (10_000, 10)
             and source["checkpoint"].get("schedule_start_global_step") == 35_000
             and torch.equal(schedule[35_000:45_000], source_schedule),
             "reconstructed 35,000-45,000 sample schedule differs from the source checkpoint")
    device = torch.device("cuda")
    evaluator_path = (ROOT / config["evaluation"]["evaluator_checkpoint"]).resolve()
    ec_path = (ROOT / config["evaluation"]["ec_checkpoint"]).resolve()
    evaluator = base._load_evaluator(evaluator_path, device)
    ec_evaluator, ec_qualification = _load_ec_evaluator(ec_path, device)
    ed_state = base._state_snapshot(evaluator)
    ec_state = base._state_snapshot(ec_evaluator)
    _require(not evaluator.training and all(not parameter.requires_grad for parameter in evaluator.parameters())
             and not ec_evaluator.training,
             "E_D/E_C must be eval-only frozen confirmation networks")
    _require(not (args.preflight_only and args.ratings_file is not None),
             "preflight-only cannot be combined with rating finalization")

    if resuming:
        progress = json.loads(args.resume_state.resolve().read_text(encoding="utf-8"))
        _require(progress.get("schema") == "lightgen_small_decoder_native_gradient_progress.v1"
                 and progress.get("task_id") == TASK_ID
                 and progress.get("source_run_id") == source["checkpoint"]["run_id"]
                 and progress.get("config_identity") == config
                 and progress.get("status") in {"RUNNING", "INTERRUPTED"},
                 "progress file task, source, config, or status is not resumable")
        identity_record = json.loads((run_dir / "run_identity.json").read_text(encoding="utf-8"))
        identity = ResearchRunIdentity.from_mapping(identity_record["identity"])
        run_id = str(identity_record["run_id"])
        _require(identity.task_id == TASK_ID and identity.config_identity == config
                 and run_id == progress.get("run_id"), "resume run identity changed")
        preflight = json.loads((run_dir / "preflight.json").read_text(encoding="utf-8"))
        receipt = PreflightReceipt.from_mapping(
            json.loads((run_dir / "preflight_ready.json").read_text(encoding="utf-8")))
        authorization_id = f"{TASK_ID}:seed:{SEED}:preflight_and_execution"
        control = ResearchRunControl(identity, authorization_id=authorization_id)
        control.mark_preflight_ready(receipt)
        control.resume_scientific_attempt(run_dir, f"decoder_native_gradient_seed_{SEED}", receipt)
        gradient_weight = float(preflight["gradient_calibration"]["lambda_native_gradient"])
        saved_selection = json.loads((run_dir / "g4_review_selection.json").read_text(encoding="utf-8"))
        _require(saved_selection.get("selection") == source["review_selection"]
                 and saved_selection.get("task_id") == TASK_ID,
                 "saved G4 review selection does not match the registered source identities")
        selection = source["review_selection"]
        _require(progress.get("official_test_reads") == 0,
                 "resume state indicates an official TEST read")
    else:
        run_id = uuid.uuid4().hex
        identity = _new_identity(config, source, run_id)
        authorization_id = f"{TASK_ID}:seed:{SEED}:preflight_and_execution"
        control = ResearchRunControl(identity, authorization_id=authorization_id)
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        stage_parent = run_dir.parent / f"{run_dir.name}_preflight_{run_id[:8]}"
        stage_parent.mkdir(parents=False, exist_ok=False)
        preserve_failed_preflight = False
        try:
            preflight, gradient_weight = _make_preflight(
                source, cohort, config, config_path, source_path, pixels, labels, train_indices,
                schedule, device, evaluator, ec_evaluator, stage_parent,
            )
            receipt = _make_receipt(control, preflight, device)
            control.mark_preflight_ready(receipt)
            _require(base._assert_close(ed_state, base._state_snapshot(evaluator), "E_D_after_preflight") == 0.0
                     and base._assert_close(ec_state, base._state_snapshot(ec_evaluator), "E_C_after_preflight") == 0.0,
                     "preflight changed a frozen evaluator")
            if args.preflight_only:
                preflight_dir = run_dir.parent / f"{run_dir.name}_preflight_receipt_{run_id[:8]}"
                preflight_dir.mkdir(parents=False, exist_ok=False)
                atomic_write_new_json(preflight_dir / "config.json", config)
                atomic_write_new_json(preflight_dir / "preflight.json", preflight)
                atomic_write_new_json(preflight_dir / "preflight_ready.json", receipt.to_dict())
                atomic_write_new_json(preflight_dir / "run_identity.json", {
                    "identity": identity.to_dict(), "run_id": run_id,
                    "authorization_id": authorization_id,
                })
                print(json.dumps({"status": "PREFLIGHT_READY", "directory": str(preflight_dir)},
                                 ensure_ascii=False), flush=True)
                return
            control.open_scientific_attempt(
                run_dir, f"decoder_native_gradient_seed_{SEED}", receipt)
            for branch in BRANCHES:
                (run_dir / branch).mkdir(parents=False, exist_ok=False)
            atomic_write_new_json(run_dir / "config.json", config)
            atomic_write_new_json(run_dir / "preflight.json", preflight)
            atomic_write_new_json(run_dir / "preflight_ready.json", receipt.to_dict())
            atomic_write_new_json(run_dir / "run_identity.json", {
                "identity": control.identity.to_dict(), "run_id": run_id,
                "authorization_id": authorization_id, "source_run_id": source["checkpoint"]["run_id"],
            })
            atomic_torch_save_new(run_dir / "source_references.pt", {
                "scale": source["checkpoint"]["scale"],
                "power_reference": source["checkpoint"]["power_reference"],
                "camera_gain": source["checkpoint"]["camera_gain"],
                "weights": source["checkpoint"]["weights"],
                "structural_weight": source["checkpoint"]["structural_weight"],
                "source_checkpoint_step": SOURCE_STEP,
            })
            atomic_torch_save_new(run_dir / "g4_fixed_inputs.pt", {
                "schema": "lightgen_small_decoder_native_gradient_fixed_g4_inputs.v1",
                "task_id": TASK_ID, "source_task_id": GENERATION_TASK_ID,
                "model": "C_seed_2026101501", "method": "G4", "stream": 0,
                "global_step": SOURCE_STEP,
                "ols_coefficients": cohort["ols_coefficients"].cpu(),
                "requested_labels": cohort["requested_labels"].cpu(),
                "component_ids": cohort["component_ids"].cpu(),
                "proposal_ids": cohort["proposal_ids"].cpu(),
            })
            atomic_write_new_json(run_dir / "g4_review_selection.json", {
                "task_id": TASK_ID, "source_stream": 0,
                "previous_review_excluded": {str(label): sorted(source["reviewed_proposal_ids"][label])
                                              for label in range(10)},
                "selection_seed": int(config["human_review"]["sample_selection_seed"]),
                "selection": source["review_selection"],
            })
            selection = source["review_selection"]
            progress = {
                "schema": "lightgen_small_decoder_native_gradient_progress.v1",
                "task_id": TASK_ID, "run_id": run_id, "source_run_id": source["checkpoint"]["run_id"],
                "config_identity": config, "status": "RUNNING", "current_stage": "PREFLIGHT_COMPLETE",
                "current_branch": None, "branch_completed_updates": {}, "branch_checkpoints": {},
                "completed_branches": [], "official_test_reads": 0,
                "last_complete_save_unix": time.time(), "last_heartbeat_unix": time.time(),
                "monitoring": {"cloud_resource_telemetry_interval_seconds": 60,
                               "active_session_check_interval_seconds": 300},
            }
            _write_progress(run_dir, progress)
        except Exception as error:
            preserve_failed_preflight = True
            atomic_write_json(stage_parent / "preflight_failure.json", {
                "task_id": TASK_ID, "run_id": run_id, "error_type": type(error).__name__,
                "error": str(error), "elapsed_seconds": time.perf_counter() - overall_started,
            })
            raise
        finally:
            if not preserve_failed_preflight:
                shutil.rmtree(stage_parent, ignore_errors=True)

    _require(base._assert_close(ed_state, base._state_snapshot(evaluator), "E_D_before_training") == 0.0
             and base._assert_close(ec_state, base._state_snapshot(ec_evaluator), "E_C_before_training") == 0.0,
             "confirmation evaluator changed before the scientific update stage")
    if resuming:
        selection = source["review_selection"]
    endpoint_summaries: dict[str, dict[str, Any]] = {}
    for branch in BRANCHES:
        branch_resume: Path | None = None
        if resuming and branch not in progress.get("completed_branches", []):
            registered = progress.get("branch_checkpoints", {}).get(branch)
            if registered is not None:
                branch_resume = Path(registered).resolve()
                _require(branch_resume.is_file() and branch_resume.parent.resolve() ==
                         (run_dir / branch).resolve(),
                         f"explicit progress checkpoint for {branch} is missing or misplaced")
        result = _run_branch(
            branch, run_dir, run_id, source, config, schedule, gradient_weight, pixels,
            labels, train_indices, dev_indices, cohort, selection, evaluator, ec_evaluator,
            device, branch_resume, progress,
        )
        endpoint_summaries[branch] = result
        progress["branch_completed_updates"] = {
            **progress.get("branch_completed_updates", {}), branch: CONTINUATION_UPDATES}
        progress["completed_branches"] = list(dict.fromkeys(progress.get("completed_branches", []) + [branch]))
        progress["last_complete_save_unix"] = time.time()
        _write_progress(run_dir, progress)
    _require(set(endpoint_summaries) == set(BRANCHES)
             and base._assert_close(ed_state, base._state_snapshot(evaluator), "E_D_after_training") == 0.0
             and base._assert_close(ec_state, base._state_snapshot(ec_evaluator), "E_C_after_training") == 0.0,
             "one branch is missing or a confirmation evaluator changed")
    review = _write_blind_review(run_dir, config, source, selection, pixels, labels, dev_indices)
    report = _aggregate_machine_report(
        run_dir, run_id, config, source, preflight, gradient_weight, endpoint_summaries,
        review, time.perf_counter() - overall_started, progress.get("recovery_events", []),
    )
    if report["status"] == "MEASUREMENT_INCOMPLETE":
        control.invalid_harness()
        progress["status"] = "INVALID_HARNESS"
        progress["current_stage"] = "OUTPUT_BUDGET_FAILED"
    else:
        progress["status"] = "AWAITING_HUMAN_RATINGS"
        progress["current_stage"] = "AWAITING_HUMAN_RATINGS"
    progress.update({"completed_branches": list(BRANCHES),
                     "current_branch": None, "last_complete_save_unix": time.time(),
                     "elapsed_seconds_total": time.perf_counter() - overall_started})
    _write_progress(run_dir, progress)
    atomic_write_json(run_dir / "run_state.json", {
        "status": control.state, "scientific_status": report["status"],
        "completed_global_step_each_branch": ENDPOINT_STEP,
        "measurement_complete": report["measurement_complete"],
        "human_review_status": "AWAITING_HUMAN_RATINGS" if report["measurement_complete"] else "NOT_AVAILABLE",
    })
    print(json.dumps({"status": report["status"], "run_control_state": control.state,
                      "report": str(run_dir / "machine_report.json"),
                      "blind_review": str(run_dir / "blind_review.html")}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--g4-cohort", type=Path)
    parser.add_argument("--previous-review-dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--ratings-file", type=Path)
    parser.add_argument("--recovery-worker-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()






