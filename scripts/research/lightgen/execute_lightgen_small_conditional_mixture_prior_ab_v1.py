"""Compare a frozen class-conditional Gaussian prior with a fixed four-component mixture."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

import scripts.research.lightgen.execute_lightgen_matched_architecture_v1 as base
from ldm_project.v2.application.research.lightgen.native_image_inputs import image_batch
from ldm_project.v2.application.research.lightgen.image_readout import resize_view
from ldm_project.v2.application.research.lightgen.confirmation_evaluators import (
    EC_CHECKPOINT_SCHEMA,
    ThirdConfirmationEvaluator,
)
from ldm_project.v2.evidence import atomic_torch_save, atomic_write_json


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_SMALL_CONDITIONAL_MIXTURE_PRIOR_AB_V1"
SOURCE_TASK_ID = "LIGHTGEN_SMALL_GAUSSIAN_FOREGROUND_BALANCE_AB_V1"
SOURCE_STEP = 45_000
SEED = 2026101501
RESAMPLE_SEED = 2026102141
REPLAY_RTOL = 2e-6
REPLAY_ATOL = 5e-8
PRIOR_RTOL = 2e-6
PRIOR_ATOL = 5e-8
POWER_RIDGE = 1e-6
SAMPLE_BASE_SEED = 2026101601
COMPONENT_BASE_SEED = 2026102121
KMEANS_SEED_BASE = 2026102131
SAMPLE_COUNT = 1_000
TRAIN_LABEL_COUNTS = (5_423, 6_242, 5_458, 5_631, 5_342, 4_921, 5_418, 5_765, 5_351, 5_449)
MAX_PROPOSALS = 1_000_000
KMEANS_MAX_ITERATIONS = 200
BATCH_SIZE = 10
BOOTSTRAP_RESAMPLES = 2_000
SHAPE_PAIR_COUNT = 10_000
METRIC_KEYS = (
    "ed_correct", "ec_correct", "proxy_clipping_area_fraction",
    "proxy_clipped_excess_fraction", "capture_power_w", "view_mean_intensity",
    "raw_peak_w_per_pixel", "ed_requested_margin", "ec_requested_margin",
    "ed_top_margin", "ec_top_margin",
)
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
    _require(path.is_file(), f"explicit mixture-prior config is missing: {path}", FileNotFoundError)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(config, dict) and config.get("schema") == "lightgen_small_conditional_mixture_prior_ab.v1"
             and config.get("task_id") == TASK_ID,
             "mixture-prior config identity changed", ValueError)
    _require(config.get("source", {}).get("task_id") == SOURCE_TASK_ID
             and config["source"].get("branch") == "C"
             and config["source"].get("cumulative_step") == SOURCE_STEP,
             "mixture-prior source definition changed", ValueError)
    _require(config["data"].get("train_count") == 55_000
             and config["data"].get("dev_count") == 5_000
             and config["data"].get("train_label_counts") == list(TRAIN_LABEL_COUNTS)
             and config["data"].get("dev_per_class") == 500
             and config["data"].get("official_test_reads") == 0,
             "mixture-prior data roles or counts changed", ValueError)
    _require(config["optics"].get("propagation_grid") == 2048
             and config["prior"].get("g4_components_per_class") == 4
             and config["prior"].get("maximum_lloyd_iterations") == KMEANS_MAX_ITERATIONS
             and config["sampling"].get("samples_per_class") == SAMPLE_COUNT
             and config["sampling"].get("maximum_proposals_per_class") == MAX_PROPOSALS,
             "fixed geometry, mixture size, or sampling population changed", ValueError)
    _require(config["evaluation"].get("bootstrap_resamples") == BOOTSTRAP_RESAMPLES
             and config["evaluation"].get("bootstrap_seed") == RESAMPLE_SEED,
             "registered bootstrap changed", ValueError)
    _require(config["output"].get("formal_result_budget_bytes") == 512 * 1024 * 1024,
             "formal result budget changed", ValueError)
    return config


def _class_record(records: dict[Any, Any], label: int) -> dict[str, torch.Tensor]:
    value = records.get(label, records.get(str(label)))
    _require(isinstance(value, dict), f"class {label} prior is missing")
    return value


def _load_ec(path: Path, device: torch.device) -> tuple[ThirdConfirmationEvaluator, dict[str, Any]]:
    _require(path.is_file(), f"qualified E_C checkpoint is missing: {path}", FileNotFoundError)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(payload, dict) and payload.get("schema") == EC_CHECKPOINT_SCHEMA
             and payload.get("architecture_kind") == ThirdConfirmationEvaluator.architecture_kind
             and payload.get("architecture") == EC_ARCHITECTURE
             and payload.get("preprocessing") == EC_PREPROCESSING
             and payload.get("class_order") == list(range(10))
             and payload.get("prediction_rule") == "argmax(logits,dim=1)",
             "E_C architecture, output classes, or prediction rule changed")
    qualification = payload.get("qualification", {})
    recalls = qualification.get("per_class_recall")
    policy = qualification.get("policy", {})
    _require(qualification.get("schema") == "third_confirmation_evaluator_qualification.v1"
             and qualification.get("split_role") == "E_C_QUALIFICATION"
             and qualification.get("evaluation_calls") == 1 and qualification.get("count") == 5_500
             and qualification.get("passed") is True and isinstance(recalls, list) and len(recalls) == 10
             and float(qualification.get("accuracy", 0.0)) >= float(policy.get("min_accuracy", math.inf))
             and float(qualification.get("macro_recall", 0.0)) >= float(policy.get("min_macro_recall", math.inf))
             and min(map(float, recalls)) >= float(policy.get("min_class_recall", math.inf)),
             "E_C historical MNIST qualification is missing or failed")
    independence = payload.get("independence", {})
    _require(independence.get("role") == "INDEPENDENT_CONFIRMATION_ONLY"
             and independence.get("used_in_generator_loss") is False
             and independence.get("used_in_gradient_attribution") is False
             and independence.get("used_in_hyperparameter_selection") is False,
             "E_C is not isolated from generator training and selection")
    evaluator = ThirdConfirmationEvaluator().to(device)
    evaluator.load_state_dict(payload["state_dict"], strict=True)
    evaluator.freeze()
    _require(not evaluator.training and all(not parameter.requires_grad for parameter in evaluator.parameters()),
             "E_C must remain frozen in eval mode")
    return evaluator, {
        "checkpoint": str(path), "schema": payload["schema"],
        "qualification_count": qualification["count"],
        "qualification_macro_recall": qualification["macro_recall"],
        "role": independence["role"],
        "boundary": "MNIST qualification only; not CMOS or human-readability qualification",
    }


def _load_context(
    source_checkpoint_path: Path, baseline_dir: Path, config: dict[str, Any],
) -> dict[str, Any]:
    expected_checkpoint = (baseline_dir / "C" / "checkpoint_step_010000.pt").resolve()
    _require(source_checkpoint_path.resolve() == expected_checkpoint,
             "source must be the explicit frozen C 45,000-step endpoint")
    for relative in (
        "final_report.json", "machine_report.json", "config.json", "source_references.pt",
        "C/endpoint_summary.json", "C/fitted_gaussian_prior.pt", "C/generated_cohort.pt",
        "C/terminal_model.pt", "C/dev_examples_step_045000.pt", "C/checkpoint_step_010000.pt",
    ):
        _require((baseline_dir / relative).is_file(), f"baseline evidence is missing: {relative}", FileNotFoundError)
    baseline_report = json.loads((baseline_dir / "machine_report.json").read_text(encoding="utf-8"))
    final_report = json.loads((baseline_dir / "final_report.json").read_text(encoding="utf-8"))
    baseline_config = json.loads((baseline_dir / "config.json").read_text(encoding="utf-8"))
    endpoint = json.loads((baseline_dir / "C" / "endpoint_summary.json").read_text(encoding="utf-8"))
    _require(baseline_report.get("task_id") == SOURCE_TASK_ID
             and baseline_report.get("measurement_complete") is True
             and final_report.get("status") == "REVIEW_INCONCLUSIVE"
             and final_report.get("human_review", {}).get("quality_checks", {}).get(
                 "duplicate_consistency", {}).get("count") == 15
             and final_report.get("human_review", {}).get("quality_checks", {}).get(
                 "real_anchor_label_match", {}).get("count") == 19,
             "source review has not been formally closed with the registered inconclusive result")
    _require(endpoint.get("branch") == "C" and endpoint.get("global_step") == SOURCE_STEP
             and endpoint.get("status") == "VALID_FAILED" and endpoint.get("measurement_complete") is True,
             "source C endpoint identity or historical quality decision changed")
    _require(final_report.get("measurement_complete") is True
             and final_report.get("human_review_status") == "REVIEW_INCONCLUSIVE",
             "source machine measurement or human-review closeout is incomplete")
    checkpoint = torch.load(source_checkpoint_path, map_location="cpu", weights_only=False)
    _require(checkpoint.get("task_id") == SOURCE_TASK_ID and checkpoint.get("branch") == "C"
             and checkpoint.get("completed_updates") == 10_000
             and checkpoint.get("completed_global_step") == SOURCE_STEP
             and checkpoint.get("seed") == SEED
             and checkpoint.get("config_identity") == baseline_config
             and checkpoint.get("model_identity") == config["source"]["model_identity"],
             "source checkpoint step, branch, seed, config, or Gaussian model identity changed")
    optimizer_state = checkpoint.get("optimizer", {}).get("state", {})
    adam_steps = [int(torch.as_tensor(state["step"]).item()) for state in optimizer_state.values()
                  if isinstance(state, dict) and "step" in state]
    _require(len(adam_steps) == 2 and all(step == SOURCE_STEP for step in adam_steps),
             "source checkpoint Adam state does not match the 45,000-step endpoint")
    _require(tuple(checkpoint["scale"].shape) == (100,)
             and tuple(checkpoint["model"]["runtime.encoder.phase"].shape) == (2, 1, 100, 100)
             and tuple(checkpoint["model"]["runtime.generator.phase"].shape) == (2, 1, 100, 100),
             "source phase or modal-scale geometry changed")
    _require(checkpoint.get("evaluator_role") == "frozen_confirmation_only",
             "source checkpoint evaluator role changed")
    small = baseline_config["optics"]["small"]
    _require(small.get("ols_basis") == "gaussian_s12_v1"
             and small.get("production_propagation_grid") == config["optics"]["propagation_grid"]
             and small.get("phase_pixel_pitch_m") == config["optics"]["phase_pixel_pitch_m"]
             and small.get("port_pitch_m") == config["optics"]["port_pitch_m"]
             and small.get("field_1e_radius_m") == config["optics"]["gaussian_field_1e_radius_m"]
             and small.get("ols_transport") == config["optics"]["ols_transport"]
             and baseline_config["optics"]["wavelength_m"] == config["optics"]["wavelength_m"]
             and baseline_config["optics"]["input_power_w"] == config["optics"]["input_power_w"]
             and baseline_config["optics"]["layer_power_transmission"] == config["optics"]["layer_power_transmission"]
             and baseline_config["data"]["training_file"] == config["data"]["training_file"],
             "mixture-prior config differs from the frozen Gaussian geometry or data source")
    _require(config["evaluation"]["ed_checkpoint"] == baseline_config["evaluation"]["evaluator_checkpoint"]
             and config["evaluation"]["ec_checkpoint"] == baseline_config["evaluation"]["ec_checkpoint"],
             "mixture-prior evaluators differ from the frozen C evaluation configuration")
    references = torch.load(baseline_dir / "source_references.pt", map_location="cpu", weights_only=False)
    for field in ("scale", "power_reference", "camera_gain"):
        _require(torch.equal(checkpoint[field].detach().cpu(), references[field].detach().cpu()),
                 f"frozen source reference {field} differs between checkpoint and evidence")
    _require(checkpoint["weights"] == references["weights"],
             "frozen BOGT loss weights differ between checkpoint and evidence")
    terminal_model = torch.load(baseline_dir / "C" / "terminal_model.pt", map_location="cpu", weights_only=False)
    _require(terminal_model.get("task_id") == SOURCE_TASK_ID
             and terminal_model.get("branch") == "C"
             and terminal_model.get("global_step") == SOURCE_STEP
             and terminal_model.get("run_id") == baseline_report.get("run_id")
             and base._assert_close(checkpoint["model"], terminal_model.get("model", {}),
                                    "C_checkpoint_vs_terminal_model") == 0.0,
             "explicit C checkpoint differs from its saved terminal model")
    prior_payload = torch.load(baseline_dir / "C" / "fitted_gaussian_prior.pt",
                               map_location="cpu", weights_only=False)
    existing_cohort = torch.load(baseline_dir / "C" / "generated_cohort.pt",
                                 map_location="cpu", weights_only=False)
    _require(existing_cohort.get("task_id") == SOURCE_TASK_ID
             and existing_cohort.get("run_id") == baseline_report.get("run_id")
             and existing_cohort.get("branch") == "C"
             and existing_cohort.get("global_step") == SOURCE_STEP
             and tuple(existing_cohort["ols_coefficients"].shape) == (10_000, 100)
             and tuple(existing_cohort["images"].shape) == (10_000, 1, 28, 28),
             "saved C generated cohort identity or population changed")
    dev_metrics = endpoint["generation"]
    _require(len(dev_metrics.get("dev_within_class_tv_mean", [])) == 10
             and len(dev_metrics.get("generated_within_class_tv_mean", [])) == 10,
             "fixed DEV shape reference is incomplete")
    _require(config["source"].get("model_identity") == checkpoint["model_identity"],
             "configuration does not match loaded endpoint identity")
    return {
        "checkpoint": checkpoint, "baseline_report": baseline_report, "final_report": final_report,
        "baseline_config": baseline_config, "endpoint": endpoint, "references": references,
        "prior_payload": prior_payload, "existing_cohort": existing_cohort,
        "source_run_id": baseline_report["run_id"],
        "source_status": endpoint["status"], "reference_generation": dev_metrics,
    }


def _make_run_identity(
    source_checkpoint: Path, baseline_dir: Path, config_path: Path,
    config: dict[str, Any], context: dict[str, Any], run_id: str,
) -> dict[str, Any]:
    return {
        "schema": "lightgen_small_conditional_mixture_prior_run_identity.v1",
        "task_id": TASK_ID, "run_id": run_id,
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_step": SOURCE_STEP,
        "source_run_id": context["source_run_id"],
        "baseline_evidence_dir": str(baseline_dir.resolve()),
        "config_path": str(config_path.resolve()),
        "config": config,
        "model_identity": context["checkpoint"]["model_identity"],
        "data_roles": config["data"],
    }


def _validate_resume(run_dir: Path, identity: dict[str, Any]) -> None:
    _require(run_dir.is_dir() and (run_dir / "run_identity.json").is_file()
             and (run_dir / "preflight.json").is_file(),
             "resume requires a saved run identity and completed representative preflight")
    saved_identity = json.loads((run_dir / "run_identity.json").read_text(encoding="utf-8"))
    preflight = json.loads((run_dir / "preflight.json").read_text(encoding="utf-8"))
    _require(saved_identity == identity and preflight.get("status") == "READY"
             and preflight.get("run_id") == identity.get("run_id"),
             "resume identity differs from the explicit source, config, or preflight")
    expected_roles = {
        "train_label_counts": identity["config"]["data"]["train_label_counts"],
        "dev_label_counts": [identity["config"]["data"]["dev_per_class"]] * 10,
    }
    _require(preflight.get("data_roles") == expected_roles,
             "resume preflight is missing the fixed TRAIN/DEV role validation")
    _require(not (run_dir / "report.json").is_file(), "completed mixture-prior run cannot be resumed")


def _fit_single_class(features: torch.Tensor, relative_ridge: float = POWER_RIDGE) -> dict[str, torch.Tensor]:
    _require(features.ndim == 2 and features.shape[0] >= 2 and torch.isfinite(features).all().item(),
             "Gaussian fit requires at least two finite feature rows")
    values = features.to(dtype=torch.float64, device="cpu")
    mean = values.mean(0)
    centered = values - mean
    covariance = centered.T @ centered / float(values.shape[0] - 1)
    ridge = covariance.diag().mean() * relative_ridge
    _require(torch.isfinite(ridge).item() and float(ridge) > 0.0,
             "Gaussian covariance ridge must be finite and positive")
    regularized = covariance + ridge * torch.eye(values.shape[1], dtype=torch.float64)
    cholesky, info = torch.linalg.cholesky_ex(regularized)
    _require(int(info) == 0 and torch.isfinite(cholesky).all().item(),
             "Gaussian covariance Cholesky factorization failed")
    return {"mean": mean, "covariance": covariance, "ridge": ridge,
            "regularized_covariance": regularized, "cholesky": cholesky}


def _squared_distances(values: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return (values.square().sum(1, keepdim=True) + centers.square().sum(1)[None, :]
            - 2.0 * (values @ centers.T)).clamp_min_(0.0)


def _kmeans_labels(
    features: torch.Tensor, components: int, seed: int, maximum_iterations: int = KMEANS_MAX_ITERATIONS,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    values = features.to(dtype=torch.float64, device="cpu")
    _require(values.ndim == 2 and torch.isfinite(values).all().item()
             and components >= 1 and values.shape[0] >= components,
             "K-means inputs or fixed component count are invalid")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    centers = [values[torch.randint(values.shape[0], (1,), generator=generator).item()]]
    nearest = _squared_distances(values, torch.stack(centers)).squeeze(1)
    for _ in range(1, components):
        mass = nearest.sum()
        _require(torch.isfinite(mass).item() and float(mass) > 0.0,
                 "K-means++ cannot select a distinct center from zero remaining distance")
        selected = int(torch.multinomial(nearest, 1, replacement=True, generator=generator).item())
        centers.append(values[selected])
        nearest = torch.minimum(nearest, _squared_distances(values, values[selected:selected + 1]).squeeze(1))
    center_tensor = torch.stack(centers)
    previous: torch.Tensor | None = None
    converged = False
    labels = torch.empty(values.shape[0], dtype=torch.long)
    iteration_count = 0
    for iteration in range(1, maximum_iterations + 1):
        labels = _squared_distances(values, center_tensor).argmin(1)
        counts = torch.bincount(labels, minlength=components)
        _require(bool((counts > 0).all()), "K-means produced an empty cluster")
        new_centers = torch.stack([values[labels == component].mean(0) for component in range(components)])
        iteration_count = iteration
        if previous is not None and torch.equal(labels, previous):
            center_tensor = new_centers
            converged = True
            break
        previous = labels.clone()
        center_tensor = new_centers
    _require(converged, f"K-means failed to converge within {maximum_iterations} iterations")
    final_labels = _squared_distances(values, center_tensor).argmin(1)
    _require(torch.equal(final_labels, labels), "K-means final assignment changed after convergence")
    return final_labels, center_tensor, iteration_count


def _fit_mixture_class(
    features: torch.Tensor, components: int, seed: int, ridge: torch.Tensor,
    maximum_iterations: int = KMEANS_MAX_ITERATIONS,
) -> dict[str, Any]:
    values = features.to(dtype=torch.float64, device="cpu")
    labels, means, iterations = _kmeans_labels(values, components, seed, maximum_iterations)
    counts = torch.bincount(labels, minlength=components)
    _require(bool((counts >= 2).all()), "mixture component has fewer than two TRAIN rows")
    covariances: list[torch.Tensor] = []
    choleskies: list[torch.Tensor] = []
    regularized_covariances: list[torch.Tensor] = []
    identity = torch.eye(values.shape[1], dtype=torch.float64)
    for component in range(components):
        class_rows = values[labels == component]
        centered = class_rows - means[component]
        covariance = centered.T @ centered / float(class_rows.shape[0] - 1)
        regularized = covariance + ridge.to(dtype=torch.float64, device="cpu") * identity
        cholesky, info = torch.linalg.cholesky_ex(regularized)
        _require(int(info) == 0 and torch.isfinite(covariance).all().item()
                 and torch.isfinite(cholesky).all().item(),
                 f"mixture component {component} covariance Cholesky factorization failed")
        covariances.append(covariance)
        regularized_covariances.append(regularized)
        choleskies.append(cholesky)
    weights = counts.to(torch.float64) / float(values.shape[0])
    _require(torch.isfinite(weights).all().item() and torch.isclose(weights.sum(), torch.tensor(1.0, dtype=torch.float64)),
             "mixture weights are invalid")
    return {
        "weights": weights, "means": means, "covariances": torch.stack(covariances),
        "regularized_covariances": torch.stack(regularized_covariances),
        "choleskies": torch.stack(choleskies), "ridge": ridge.detach().cpu().double(),
        "counts": counts, "assignments": labels, "iterations": iterations,
    }


def _validate_feature_cache(
    payload: dict[str, Any], train_rows: torch.Tensor, train_labels: torch.Tensor,
    source_run_id: str,
) -> torch.Tensor:
    _require(isinstance(payload, dict) and payload.get("task_id") == TASK_ID
             and payload.get("source_run_id") == source_run_id
             and isinstance(payload.get("rows"), torch.Tensor)
             and isinstance(payload.get("labels"), torch.Tensor)
             and torch.equal(payload["rows"], train_rows)
             and torch.equal(payload["labels"], train_labels),
             "TRAIN feature cache has wrong source identity, rows, or labels")
    features = payload.get("features_200")
    _require(isinstance(features, torch.Tensor) and tuple(features.shape) == (55_000, 200)
             and features.dtype == torch.float64 and torch.isfinite(features).all().item(),
             "TRAIN feature cache is incomplete, malformed, or nonfinite")
    return features


def _validate_data_roles(
    train_rows: torch.Tensor, dev_rows: torch.Tensor,
    train_labels: torch.Tensor, dev_labels: torch.Tensor,
    data_contract: dict[str, Any],
) -> dict[str, list[int]]:
    _require(train_rows.ndim == dev_rows.ndim == train_labels.ndim == dev_labels.ndim == 1
             and train_rows.numel() == data_contract["train_count"]
             and dev_rows.numel() == data_contract["dev_count"]
             and train_rows.numel() == train_labels.numel()
             and dev_rows.numel() == dev_labels.numel(),
             "fixed TRAIN/DEV row counts or label alignment changed")
    _require(torch.unique(train_rows).numel() == train_rows.numel()
             and torch.unique(dev_rows).numel() == dev_rows.numel()
             and not bool(torch.isin(train_rows, dev_rows).any()),
             "TRAIN/DEV rows are duplicated or overlap")
    _require(bool(((train_labels >= 0) & (train_labels < 10)).all())
             and bool(((dev_labels >= 0) & (dev_labels < 10)).all()),
             "TRAIN/DEV contains labels outside the fixed ten classes")
    train_counts = torch.bincount(train_labels, minlength=10)
    dev_counts = torch.bincount(dev_labels, minlength=10)
    expected_train_counts = torch.tensor(data_contract["train_label_counts"], dtype=torch.long)
    expected_dev_count = int(data_contract["dev_per_class"])
    _require(expected_train_counts.shape == (10,) and int(expected_train_counts.sum()) == train_rows.numel()
             and torch.equal(train_counts, expected_train_counts)
             and torch.equal(dev_counts, torch.full((10,), expected_dev_count, dtype=torch.long)),
             "TRAIN/DEV class populations differ from the fixed source split")
    return {"train_label_counts": train_counts.tolist(), "dev_label_counts": dev_counts.tolist()}


def _validate_saved_cohort(
    payload: dict[str, Any], method: str, run_id: str, source_run_id: str,
) -> None:
    _require(isinstance(payload, dict) and payload.get("task_id") == TASK_ID
             and payload.get("method") == method and payload.get("run_id") == run_id
             and payload.get("source_run_id") == source_run_id
             and payload.get("global_step") == SOURCE_STEP,
             f"saved {method} cohort identity is incomplete or mismatched")
    for key, shape in (
        ("images", (10_000, 1, 28, 28)), ("requested_labels", (10_000,)),
        ("component_ids", (10_000,)), ("proposal_ids", (10_000,)),
        ("ols_coefficients", (10_000, 100)), ("ed_logits", (10_000, 10)),
        ("ec_logits", (10_000, 10)), ("capture_power_w", (10_000,)),
        ("proxy_clipping_area_fraction", (10_000,)),
    ):
        value = payload.get(key)
        _require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape,
                 f"saved {method} cohort field {key} is missing or malformed")
    _require(torch.equal(torch.bincount(payload["requested_labels"], minlength=10),
                         torch.full((10,), SAMPLE_COUNT))
             and torch.isfinite(payload["images"]).all().item()
             and torch.isfinite(payload["ed_logits"]).all().item()
             and torch.isfinite(payload["ec_logits"]).all().item()
             and torch.isfinite(payload["ols_coefficients"].real).all().item()
             and torch.isfinite(payload["ols_coefficients"].imag).all().item()
             and bool((payload["ols_coefficients"].abs().square().sum(1) <= base.MAX_MODAL_POWER_W).all()),
             f"saved {method} cohort contains incomplete, nonfinite, or over-power rows")
    _require(tuple(payload.get("native_capture_images", torch.empty(0)).shape) == (100, 1, 200, 200)
             and tuple(payload.get("native_capture_indices", torch.empty(0)).shape) == (100,),
             f"saved {method} native display subset is incomplete")


def _encode_train_features(
    model: torch.nn.Module, pixels: torch.Tensor, train_rows: torch.Tensor,
    train_labels: torch.Tensor, scale: torch.Tensor, device: torch.device,
    output_path: Path, source_run_id: str,
) -> tuple[torch.Tensor, float, int]:
    started = time.perf_counter()
    features = torch.empty((train_rows.numel(), 200), dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for start in range(0, train_rows.numel(), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, train_rows.numel())
            rows = train_rows[start:stop]
            target = image_batch(pixels, rows, "small", device)
            coefficients = base._forward(model, target)["ols_coefficients"].to(torch.complex128)
            gauged = base._gauge_and_scale(coefficients, scale.to(device))
            features[start:stop] = torch.view_as_real(gauged.cpu()).reshape(stop - start, 200)
            if stop % 5_000 == 0 or stop == train_rows.numel():
                torch.cuda.synchronize(device)
                print(json.dumps({"phase": "TRAIN_FEATURE_ENCODING", "rows": int(stop),
                                  "total_rows": int(train_rows.numel()),
                                  "elapsed_seconds": time.perf_counter() - started}), flush=True)
    _require(torch.isfinite(features).all().item(), "TRAIN coefficient encoding produced nonfinite features")
    payload = {
        "schema": "lightgen_small_conditional_mixture_train_features.v1",
        "task_id": TASK_ID, "source_run_id": source_run_id,
        "source_checkpoint_step": SOURCE_STEP, "rows": train_rows.clone(),
        "labels": train_labels.clone(), "features_200": features,
        "encoder_forward_images": int(train_rows.numel()),
        "encoder_forward_batches": math.ceil(train_rows.numel() / BATCH_SIZE),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_torch_save(output_path, payload)
    return features, time.perf_counter() - started, math.ceil(train_rows.numel() / BATCH_SIZE)


def _fit_priors(
    features: torch.Tensor, train_labels: torch.Tensor, existing_prior: dict[Any, Any],
    run_dir: Path,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], torch.Tensor, dict[str, Any]]:
    g1: dict[int, dict[str, Any]] = {}
    g4: dict[int, dict[str, Any]] = {}
    assignments_all = torch.full((train_labels.numel(),), -1, dtype=torch.int64)
    reference_checks: dict[str, Any] = {}
    identity_cache = torch.eye(200, dtype=torch.float64)
    for label in range(10):
        mask = train_labels == label
        class_features = features[mask]
        class_count = int(mask.sum())
        _require(class_count >= 4 and class_features.shape == (class_count, 200),
                 f"TRAIN class {label} prior population is incomplete for fixed K=4")
        fit = _fit_single_class(class_features)
        old = _class_record(existing_prior, label)
        old_mean = old["mean"].detach().cpu().double()
        old_cholesky = old["cholesky"].detach().cpu().double()
        old_ridge = torch.as_tensor(old["ridge"]).detach().cpu().double()
        old_covariance = old.get("covariance")
        if not isinstance(old_covariance, torch.Tensor):
            old_covariance = old_cholesky @ old_cholesky.T - old_ridge * identity_cache
        else:
            old_covariance = old_covariance.detach().cpu().double()
        _require(torch.allclose(fit["mean"], old_mean, rtol=PRIOR_RTOL, atol=PRIOR_ATOL)
                 and torch.allclose(fit["ridge"], old_ridge, rtol=PRIOR_RTOL, atol=PRIOR_ATOL)
                 and torch.allclose(fit["covariance"], old_covariance, rtol=PRIOR_RTOL, atol=PRIOR_ATOL)
                 and torch.allclose(fit["cholesky"], old_cholesky, rtol=PRIOR_RTOL, atol=PRIOR_ATOL),
                 f"G1 refit from fixed TRAIN cache does not reproduce saved class {label} prior")
        g1[label] = fit
        mixture = _fit_mixture_class(class_features, 4, KMEANS_SEED_BASE + label,
                                     fit["ridge"], KMEANS_MAX_ITERATIONS)
        assignments_all[mask] = mixture.pop("assignments")
        mixture["train_rows"] = class_count
        mixture["seed"] = KMEANS_SEED_BASE + label
        g4[label] = mixture
        reference_checks[str(label)] = {
            "train_rows": int(class_features.shape[0]), "g1_mean_max_abs_error": float((fit["mean"] - old_mean).abs().max()),
            "g1_covariance_max_abs_error": float((fit["covariance"] - old_covariance).abs().max()),
            "g1_cholesky_max_abs_error": float((fit["cholesky"] - old_cholesky).abs().max()),
            "g1_ridge_abs_error": float((fit["ridge"] - old_ridge).abs()),
            "g1_covariance_source": "saved covariance" if isinstance(old.get("covariance"), torch.Tensor)
            else "reconstructed from saved Cholesky and ridge",
            "g4_cluster_counts": g4[label]["counts"].tolist(),
            "g4_weights": g4[label]["weights"].tolist(),
            "g4_lloyd_iterations": g4[label]["iterations"],
        }
        _require(bool((assignments_all[mask] >= 0).all()), f"G4 class {label} assignment is incomplete")
    _require(bool((assignments_all >= 0).all()), "G4 TRAIN cluster assignments omit one or more rows")
    atomic_torch_save(run_dir / "fitted_priors_G1_G4.pt", {
        "schema": "lightgen_small_conditional_mixture_priors.v1", "task_id": TASK_ID,
        "G1": g1, "G4": g4, "G4_train_assignments_in_cache_order": assignments_all,
    })
    atomic_write_json(run_dir / "g1_reference_reconstruction.json", reference_checks)
    return g1, g4, assignments_all, reference_checks


def _sample_mixture_class(
    prior: dict[str, Any], scale: torch.Tensor, label: int, sample_count: int = SAMPLE_COUNT,
    proposal_limit: int = MAX_PROPOSALS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    normal_generator = torch.Generator(device="cpu").manual_seed(SAMPLE_BASE_SEED + label)
    component_generator = torch.Generator(device="cpu").manual_seed(COMPONENT_BASE_SEED + label)
    accepted_coefficients: list[torch.Tensor] = []
    accepted_components: list[torch.Tensor] = []
    accepted_proposals: list[torch.Tensor] = []
    proposed_by_component = torch.zeros(4, dtype=torch.int64)
    rejected_by_component = torch.zeros(4, dtype=torch.int64)
    accepted_by_component = torch.zeros(4, dtype=torch.int64)
    evaluated = 0
    retained = 0
    while retained < sample_count:
        remaining_budget = proposal_limit - evaluated
        _require(remaining_budget > 0,
                 f"G4 class {label} sampling incomplete: retained {retained}/{sample_count} within proposal limit")
        count = min(256, remaining_budget)
        noise = torch.randn((count, prior["means"].shape[1]), generator=normal_generator, dtype=torch.float64)
        components = torch.multinomial(prior["weights"], count, replacement=True, generator=component_generator)
        draws = torch.empty_like(noise)
        for component in range(4):
            mask = components == component
            if bool(mask.any()):
                draws[mask] = prior["means"][component] + noise[mask] @ prior["choleskies"][component].T
        normalized = torch.view_as_complex(draws.reshape(count, 100, 2).contiguous())
        coefficients = normalized * scale.detach().cpu().double().sqrt().unsqueeze(0)
        powers = coefficients.abs().square().sum(1)
        valid = powers <= base.MAX_MODAL_POWER_W
        valid_positions = torch.nonzero(valid, as_tuple=False).flatten()
        for component in range(4):
            component_mask = components == component
            proposed_by_component[component] += int(component_mask.sum())
            rejected_by_component[component] += int((component_mask & ~valid).sum())
        need = sample_count - retained
        chosen = valid_positions[:need]
        if chosen.numel():
            chosen_components = components[chosen]
            accepted_coefficients.append(coefficients[chosen].to(torch.complex64))
            accepted_components.append(chosen_components)
            accepted_proposals.append(chosen + evaluated)
            accepted_by_component += torch.bincount(chosen_components, minlength=4)
            retained += int(chosen.numel())
        evaluated += count
    coefficients_out = torch.cat(accepted_coefficients)
    component_out = torch.cat(accepted_components).long()
    proposal_out = torch.cat(accepted_proposals).long()
    _require(tuple(coefficients_out.shape) == (sample_count, 100)
             and torch.unique(proposal_out).numel() == sample_count
             and bool((coefficients_out.abs().square().sum(1) <= base.MAX_MODAL_POWER_W).all()),
             f"G4 class {label} cohort failed its registered power or identity check")
    return coefficients_out, component_out, proposal_out, {
        "requested_samples": sample_count, "retained_samples": retained,
        "proposals_evaluated": evaluated, "proposal_limit": proposal_limit,
        "power_rejected_candidates": int(rejected_by_component.sum()),
        "rejection_fraction": float(rejected_by_component.sum() / evaluated),
        "proposed_by_component": proposed_by_component.tolist(),
        "power_rejected_by_component": rejected_by_component.tolist(),
        "retained_by_component": accepted_by_component.tolist(),
        "normal_seed": SAMPLE_BASE_SEED + label,
        "component_seed": COMPONENT_BASE_SEED + label,
        "power_bound_w": base.MAX_MODAL_POWER_W,
    }


def _sample_g4(
    priors: dict[int, dict[str, Any]], scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    coefficients: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    components: list[torch.Tensor] = []
    proposals: list[torch.Tensor] = []
    records: dict[str, Any] = {}
    for label in range(10):
        class_coefficients, class_components, class_proposals, accounting = _sample_mixture_class(
            priors[label], scale, label,
        )
        coefficients.append(class_coefficients)
        labels.append(torch.full((SAMPLE_COUNT,), label, dtype=torch.long))
        components.append(class_components)
        proposals.append(class_proposals)
        records[str(label)] = accounting
    return (torch.cat(coefficients), torch.cat(labels), torch.cat(components),
            torch.cat(proposals), records)


def _ec_logits(evaluator: ThirdConfirmationEvaluator, images: torch.Tensor) -> torch.Tensor:
    output = evaluator(images)
    _require(isinstance(output, tuple) and len(output) == 2, "E_C must return features and logits")
    features, logits = output
    _require(tuple(features.shape) == (images.shape[0], 128)
             and tuple(logits.shape) == (images.shape[0], 10)
             and torch.isfinite(features).all().item() and torch.isfinite(logits).all().item(),
             "E_C returned malformed or nonfinite values")
    return logits


def _logit_fields(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prediction = logits.argmax(1)
    margin = logits.topk(2, dim=1).values.diff(dim=1).abs().squeeze(1)
    rows = torch.arange(logits.shape[0], device=logits.device)
    target = labels.to(device=logits.device, dtype=torch.long)
    requested_margin = logits[rows, target].clone()
    competitors = logits.clone()
    competitors[rows, target] = -torch.inf
    requested_margin -= competitors.max(1).values
    return prediction, margin, requested_margin


def _decode_cohort(
    method: str, model: torch.nn.Module, evaluator_d: torch.nn.Module,
    evaluator_c: ThirdConfirmationEvaluator, coefficients: torch.Tensor, labels: torch.Tensor,
    components: torch.Tensor, proposals: torch.Tensor, gain: torch.Tensor,
    device: torch.device, capture_count: int, expected_per_class: int | None = SAMPLE_COUNT,
) -> dict[str, Any]:
    count = int(labels.numel())
    _require(tuple(coefficients.shape) == (count, 100) and tuple(components.shape) == (count,)
             and tuple(proposals.shape) == (count,), f"{method} cohort arrays are misaligned")
    _require(bool(torch.isfinite(coefficients.real).all() and torch.isfinite(coefficients.imag).all()),
             f"{method} OLS coefficients are nonfinite")
    class_counts = torch.bincount(labels, minlength=10)
    _require(bool((class_counts == class_counts[0]).all()) and int(class_counts[0]) > 0
             and (expected_per_class is None or bool((class_counts == expected_per_class).all())),
             f"{method} cohort class population is incomplete or unbalanced")
    _require(bool((coefficients.abs().square().sum(1) <= base.MAX_MODAL_POWER_W).all()),
             f"{method} cohort contains coefficients over the fixed physical modal power bound")
    capture_ids = torch.cat([
        torch.nonzero(labels == label, as_tuple=False).flatten()[:capture_count] for label in range(10)
    ]).long()
    capture_set = set(map(int, capture_ids.tolist()))
    captured_native: dict[int, torch.Tensor] = {}
    image_parts: list[torch.Tensor] = []
    ed_logit_parts: list[torch.Tensor] = []
    ec_logit_parts: list[torch.Tensor] = []
    capture_power_parts: list[torch.Tensor] = []
    raw_mean_parts: list[torch.Tensor] = []
    raw_peak_parts: list[torch.Tensor] = []
    view_mean_parts: list[torch.Tensor] = []
    clip_area_parts: list[torch.Tensor] = []
    clipped_excess_parts: list[torch.Tensor] = []
    model.eval()
    evaluator_d.eval()
    evaluator_c.eval()
    with torch.inference_mode():
        for start in range(0, count, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, count)
            output = model.forward_from_ols(coefficients[start:stop].to(device))
            raw = output["physical_w_per_pixel"].float()
            scaled = raw * gain.to(device=device, dtype=raw.dtype)
            view = resize_view(scaled)
            ed_logits = evaluator_d(view)
            ec_logits = _ec_logits(evaluator_c, view)
            _require(tuple(ed_logits.shape) == tuple(ec_logits.shape) == (stop - start, 10)
                     and torch.isfinite(ed_logits).all().item() and torch.isfinite(ec_logits).all().item(),
                     f"{method} evaluator logits are malformed or nonfinite")
            image_parts.append(view.cpu())
            ed_logit_parts.append(ed_logits.float().cpu())
            ec_logit_parts.append(ec_logits.float().cpu())
            capture_power_parts.append(raw.double().sum((1, 2, 3)).cpu())
            raw_mean_parts.append(raw.double().mean((1, 2, 3)).cpu())
            raw_peak_parts.append(raw.double().amax((1, 2, 3)).cpu())
            view_mean_parts.append(view.double().mean((1, 2, 3)).cpu())
            clip_area_parts.append((scaled > 1.0).double().mean((1, 2, 3)).cpu())
            total_scaled = scaled.double().sum((1, 2, 3)).clamp_min(1e-30)
            clipped_excess_parts.append((scaled.double().sub(1.0).clamp_min(0.0).sum((1, 2, 3))
                                         / total_scaled).cpu())
            for local_index, sample_index in enumerate(range(start, stop)):
                if sample_index in capture_set:
                    captured_native[sample_index] = scaled[local_index].clamp(0.0, 1.0).cpu()
    images = torch.cat(image_parts)
    ed_logits_all = torch.cat(ed_logit_parts)
    ec_logits_all = torch.cat(ec_logit_parts)
    ed_prediction, ed_margin, ed_requested_margin = _logit_fields(ed_logits_all, labels)
    ec_prediction, ec_margin, ec_requested_margin = _logit_fields(ec_logits_all, labels)
    _require(set(captured_native) == capture_set, f"{method} native display captures are incomplete")
    capture_indices = torch.tensor(sorted(capture_set), dtype=torch.long)
    native_images = torch.stack([captured_native[int(index)] for index in capture_indices])
    return {
        "schema": "lightgen_small_conditional_mixture_generated_cohort.v1",
        "task_id": TASK_ID, "method": method, "global_step": SOURCE_STEP,
        "images": images, "requested_labels": labels.cpu().long(),
        "component_ids": components.cpu().long(), "proposal_ids": proposals.cpu().long(),
        "ols_coefficients": coefficients.cpu(),
        "ed_logits": ed_logits_all, "ec_logits": ec_logits_all,
        "ed_predictions": ed_prediction.cpu(), "ec_predictions": ec_prediction.cpu(),
        "ed_top_margin": ed_margin.cpu(), "ec_top_margin": ec_margin.cpu(),
        "ed_requested_margin": ed_requested_margin.cpu(),
        "ec_requested_margin": ec_requested_margin.cpu(),
        "capture_power_w": torch.cat(capture_power_parts),
        "raw_mean_w_per_pixel": torch.cat(raw_mean_parts),
        "raw_peak_w_per_pixel": torch.cat(raw_peak_parts),
        "view_mean_intensity": torch.cat(view_mean_parts),
        "proxy_clipping_area_fraction": torch.cat(clip_area_parts),
        "proxy_clipped_excess_fraction": torch.cat(clipped_excess_parts),
        "native_capture_indices": capture_indices,
        "native_capture_images": native_images,
        "classifier_based_sample_selection": False,
        "per_image_normalization": False,
        "power_bound_w": base.MAX_MODAL_POWER_W,
    }


def _compare_replay(actual: torch.Tensor, expected: torch.Tensor, field: str) -> dict[str, float]:
    left = actual.detach().cpu().float()
    right = expected.detach().cpu().float()
    _require(left.shape == right.shape and torch.isfinite(left).all().item()
             and torch.isfinite(right).all().item(), f"G1 replay {field} shape or finiteness changed")
    _require(torch.allclose(left, right, rtol=REPLAY_RTOL, atol=REPLAY_ATOL),
             f"G1 replay {field} exceeds rtol={REPLAY_RTOL}, atol={REPLAY_ATOL}")
    difference = (left - right).abs()
    return {"max_absolute_error": float(difference.max()),
            "max_relative_error": float((difference / right.abs().clamp_min(1e-12)).max()),
            "passed": True}


def _preflight_replay(
    model: torch.nn.Module, cohort: dict[str, Any], evaluator_d: torch.nn.Module,
    evaluator_c: ThirdConfirmationEvaluator, gain: torch.Tensor, device: torch.device,
) -> dict[str, Any]:
    first_by_class = torch.tensor([
        int(torch.nonzero(cohort["requested_labels"] == label, as_tuple=False).flatten()[0])
        for label in range(10)
    ], dtype=torch.long)
    _require(torch.equal(cohort["requested_labels"][first_by_class], torch.arange(10)),
             "G1 representative rows are not the first saved sample of each class")
    decoded = _decode_cohort("G1_PRECHECK", model, evaluator_d, evaluator_c,
                             cohort["ols_coefficients"][first_by_class],
                             cohort["requested_labels"][first_by_class],
                             torch.zeros(10, dtype=torch.long), cohort["proposal_ids"][first_by_class],
                             gain, device, capture_count=1, expected_per_class=1)
    return {
        "rows_per_class": 1,
        "view": _compare_replay(decoded["images"], cohort["images"][first_by_class], "preflight_view"),
        "E_D_logits": _compare_replay(decoded["ed_logits"], cohort["ed_logits"][first_by_class], "preflight_E_D_logits"),
        "E_C_logits": _compare_replay(decoded["ec_logits"], cohort["ec_logits"][first_by_class], "preflight_E_C_logits"),
        "E_D_predictions_equal": bool(torch.equal(decoded["ed_predictions"], cohort["ed_predictions"][first_by_class])),
        "E_C_predictions_equal": bool(torch.equal(decoded["ec_predictions"], cohort["ec_predictions"][first_by_class])),
        "encoder_forwards": 0, "optimizer_calls": 0, "phase_updates": 0,
    }


def _validate_readout_numpy(cohort: dict[str, Any]) -> dict[str, float]:
    selected = torch.cat([
        torch.nonzero(cohort["requested_labels"] == label, as_tuple=False).flatten()[:1]
        for label in range(10)
    ]).long()
    native = cohort["native_capture_images"]
    capture_lookup = {int(index): image.numpy()[0] for index, image in zip(
        cohort["native_capture_indices"], native, strict=True
    )}
    maximum_absolute_error = 0.0
    for index in selected.tolist():
        source = capture_lookup[index].astype(np.float64)
        height, width = source.shape
        pooled = np.empty((28, 28), dtype=np.float64)
        for row in range(28):
            y0 = math.floor(row * height / 28)
            y1 = math.ceil((row + 1) * height / 28)
            for column in range(28):
                x0 = math.floor(column * width / 28)
                x1 = math.ceil((column + 1) * width / 28)
                pooled[row, column] = source[y0:y1, x0:x1].mean()
        expected = cohort["images"][index, 0].double().numpy()
        maximum_absolute_error = max(maximum_absolute_error, float(np.abs(pooled - expected).max()))
    _require(maximum_absolute_error <= 1e-6,
             "independent NumPy raw-clamp-pool readout differs from production views")
    return {"sample_count": int(selected.numel()), "maximum_absolute_error": maximum_absolute_error,
            "operation_order": "gain then clamp [0,1] then adaptive average pool 28x28"}


def _cohort_metrics(
    cohort: dict[str, Any], reference_generation: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = cohort["requested_labels"].long()
    ed = base._classification(labels, cohort["ed_predictions"])
    ec = base._classification(labels, cohort["ec_predictions"])
    per_class: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    generated_tv: list[float] = []
    dev_tv = list(map(float, reference_generation["dev_within_class_tv_mean"]))
    for label in range(10):
        mask = labels == label
        image_class = cohort["images"][mask]
        generated_distance = float(base._pair_distances(image_class, 2026101603 + label).mean())
        ratio = generated_distance / max(dev_tv[label], 1e-12)
        generated_tv.append(generated_distance)
        per_class.append({
            "label": label, "count": int(mask.sum()),
            "E_D_recall": float(ed["recall_by_class"][label]),
            "E_C_recall": float(ec["recall_by_class"][label]),
            "E_D_confusion_row": ed["confusion_matrix"][label].tolist(),
            "E_C_confusion_row": ec["confusion_matrix"][label].tolist(),
            "mean_capture_power_w": float(cohort["capture_power_w"][mask].mean()),
            "mean_raw_w_per_pixel": float(cohort["raw_mean_w_per_pixel"][mask].mean()),
            "mean_view_intensity": float(cohort["view_mean_intensity"][mask].mean()),
            "mean_raw_peak_w_per_pixel": float(cohort["raw_peak_w_per_pixel"][mask].mean()),
            "proxy_clipping_area_fraction": float(cohort["proxy_clipping_area_fraction"][mask].mean()),
            "proxy_clipped_excess_fraction": float(cohort["proxy_clipped_excess_fraction"][mask].mean()),
            "generated_within_class_tv_mean": generated_distance,
            "fixed_dev_within_class_tv_mean": dev_tv[label],
            "generated_to_dev_tv_ratio": ratio,
        })
        for index in torch.nonzero(mask, as_tuple=False).flatten().tolist():
            rows.append({
                "method": cohort["method"], "requested_label": label,
                "component_id": int(cohort["component_ids"][index]),
                "proposal_id": int(cohort["proposal_ids"][index]),
                "E_D_prediction": int(cohort["ed_predictions"][index]),
                "E_C_prediction": int(cohort["ec_predictions"][index]),
                "E_D_correct": int(cohort["ed_predictions"][index] == label),
                "E_C_correct": int(cohort["ec_predictions"][index] == label),
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
    _require(all(row["count"] == SAMPLE_COUNT for row in per_class),
             f"{cohort['method']} class evaluation population is incomplete")
    metrics = {
        "method": cohort["method"], "population": int(labels.numel()),
        "samples_per_class": SAMPLE_COUNT,
        "E_D_macro_recall": float(ed["macro_recall"]),
        "E_D_recall_by_class": [row["E_D_recall"] for row in per_class],
        "E_D_confusion_matrix": ed["confusion_matrix"].tolist(),
        "E_C_macro_recall": float(ec["macro_recall"]),
        "E_C_recall_by_class": [row["E_C_recall"] for row in per_class],
        "E_C_confusion_matrix": ec["confusion_matrix"].tolist(),
        "macro_proxy_clipping_area_fraction": float(np.mean(
            [row["proxy_clipping_area_fraction"] for row in per_class])),
        "proxy_clipping_area_by_class": [row["proxy_clipping_area_fraction"] for row in per_class],
        "macro_proxy_clipped_excess_fraction": float(np.mean(
            [row["proxy_clipped_excess_fraction"] for row in per_class])),
        "mean_capture_power_w": float(np.mean([row["mean_capture_power_w"] for row in per_class])),
        "mean_view_intensity": float(np.mean([row["mean_view_intensity"] for row in per_class])),
        "generated_within_class_tv_mean": generated_tv,
        "fixed_dev_within_class_tv_mean": dev_tv,
        "generated_to_dev_tv_ratio": [row["generated_to_dev_tv_ratio"] for row in per_class],
        "per_class": per_class,
        "sampling_power_bound_w": base.MAX_MODAL_POWER_W,
        "classifier_based_sample_selection": False,
    }
    return metrics, rows


def _row_metric_arrays(cohort: dict[str, Any]) -> dict[str, np.ndarray]:
    labels = cohort["requested_labels"].long()
    return {
        "ed_correct": (cohort["ed_predictions"] == labels).numpy().astype(np.float64),
        "ec_correct": (cohort["ec_predictions"] == labels).numpy().astype(np.float64),
        "proxy_clipping_area_fraction": cohort["proxy_clipping_area_fraction"].numpy().astype(np.float64),
        "proxy_clipped_excess_fraction": cohort["proxy_clipped_excess_fraction"].numpy().astype(np.float64),
        "capture_power_w": cohort["capture_power_w"].numpy().astype(np.float64),
        "view_mean_intensity": cohort["view_mean_intensity"].numpy().astype(np.float64),
        "raw_peak_w_per_pixel": cohort["raw_peak_w_per_pixel"].numpy().astype(np.float64),
        "ed_requested_margin": cohort["ed_requested_margin"].numpy().astype(np.float64),
        "ec_requested_margin": cohort["ec_requested_margin"].numpy().astype(np.float64),
        "ed_top_margin": cohort["ed_top_margin"].numpy().astype(np.float64),
        "ec_top_margin": cohort["ec_top_margin"].numpy().astype(np.float64),
    }


def _bootstrap_differences(
    g1: dict[str, Any], g4: dict[str, Any], seed: int, resamples: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    arrays = {"G1": _row_metric_arrays(g1), "G4": _row_metric_arrays(g4)}
    draws = {key: np.empty(resamples, dtype=np.float64) for key in METRIC_KEYS}
    per_class = {key: np.empty((resamples, 10), dtype=np.float64)
                 for key in ("ed_correct", "ec_correct", "proxy_clipping_area_fraction")}
    requested = {method: g1["requested_labels"].numpy() if method == "G1"
                 else g4["requested_labels"].numpy() for method in ("G1", "G4")}
    for repeat in range(resamples):
        method_means: dict[str, dict[str, list[float]]] = {
            method: {key: [] for key in METRIC_KEYS} for method in ("G1", "G4")
        }
        for method, cohort_arrays in arrays.items():
            for label in range(10):
                indices = np.flatnonzero(requested[method] == label)
                _require(indices.size == SAMPLE_COUNT, f"{method} bootstrap class {label} is incomplete")
                sampled = indices[rng.integers(0, indices.size, size=indices.size)]
                for key in METRIC_KEYS:
                    mean = float(cohort_arrays[key][sampled].mean())
                    method_means[method][key].append(mean)
                    if key in per_class:
                        per_class[key][repeat, label] = mean
        for key in METRIC_KEYS:
            draws[key][repeat] = float(np.mean(method_means["G4"][key]) - np.mean(method_means["G1"][key]))
    summary: dict[str, Any] = {
        "seed": seed, "resamples": resamples,
        "resampling": "independent samples within each method and requested class; each draw shared across all metrics",
        "differences": {}, "per_class_metric_differences": {},
    }
    for key in METRIC_KEYS:
        observed = float(np.mean([
            arrays["G4"][key][requested["G4"] == label].mean()
            - arrays["G1"][key][requested["G1"] == label].mean()
            for label in range(10)
        ]))
        summary["differences"][key] = {
            "estimand": f"G4 minus G1 equal-class mean {key}",
            "observed_difference": observed,
            "ci_95": [float(value) for value in np.quantile(draws[key], [0.025, 0.975])],
        }
    for key in per_class:
        summary["per_class_metric_differences"][key] = [
            {"label": label,
             "observed_difference": float(arrays["G4"][key][requested["G4"] == label].mean()
                                          - arrays["G1"][key][requested["G1"] == label].mean()),
             "ci_95": [float(value) for value in np.quantile(per_class[key][:, label], [0.025, 0.975])]}
            for label in range(10)
        ]
    return summary, {**{f"delta_{key}": value for key, value in draws.items()},
                     **{f"per_class_delta_{key}": value for key, value in per_class.items()}}


def _write_per_image_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _require(bool(rows), "per-image metric table is empty")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_gray(value: torch.Tensor, path: Path) -> None:
    image = value.detach().cpu().float().squeeze()
    _require(image.ndim == 2 and torch.isfinite(image).all().item()
             and float(image.min()) >= 0.0 and float(image.max()) <= 1.0,
             "gallery images must be finite grayscale values in [0,1]")
    array = np.rint(image.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(array, mode="L").save(path, format="PNG", optimize=True)


def _write_gallery(run_dir: Path, cohorts: dict[str, dict[str, Any]]) -> int:
    gallery_root = run_dir / "gallery"
    gallery_root.mkdir(parents=True, exist_ok=True)
    blocks = ["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>G1/G4固定先验图册</title>",
              "<style>body{font:15px system-ui,sans-serif;margin:1rem;color:#17202a}section{margin:2rem 0}" 
              "table{border-collapse:collapse}td,th{padding:.3rem;border:1px solid #bbb;text-align:center}" 
              "img{image-rendering:pixelated;object-fit:contain;background:#fff}</style>",
              "<h1>冻结C端点条件先验对照</h1><p>灰度固定为[0,1]；原生200×200与生产28×28并列。每类按接受顺序展示前10条。</p>"]
    exported = 0
    for method, cohort in cohorts.items():
        capture_lookup = {int(index): image for index, image in zip(
            cohort["native_capture_indices"], cohort["native_capture_images"], strict=True
        )}
        blocks.append(f"<h2>{html.escape(method)}</h2>")
        for label in range(10):
            indices = torch.nonzero(cohort["requested_labels"] == label, as_tuple=False).flatten()[:10]
            blocks.append(f"<section><h3>类别 {label}</h3><table><tr><th>接受序号</th><th>分量</th>"
                          "<th>proposal</th><th>原生200×200</th><th>生产28×28</th></tr>")
            for rank, index_tensor in enumerate(indices):
                index = int(index_tensor)
                _require(index in capture_lookup, f"{method} class {label} native display sample is missing")
                native_relative = f"gallery/{method}/class_{label}/sample_{rank:02d}_native.png"
                view_relative = f"gallery/{method}/class_{label}/sample_{rank:02d}_view28.png"
                native_path = run_dir / native_relative
                view_path = run_dir / view_relative
                native_path.parent.mkdir(parents=True, exist_ok=True)
                _save_gray(capture_lookup[index], native_path)
                _save_gray(cohort["images"][index], view_path)
                component = int(cohort["component_ids"][index])
                proposal = int(cohort["proposal_ids"][index])
                blocks.append(
                    f"<tr><td>{rank}</td><td>{component}</td><td>{proposal}</td>"
                    f"<td><img width='200' height='200' src='{html.escape(native_relative)}'></td>"
                    f"<td><img width='224' height='224' src='{html.escape(view_relative)}'></td></tr>"
                )
                exported += 2
            blocks.append("</table></section>")
    blocks.append("</html>")
    (run_dir / "gallery.html").write_text("\n".join(blocks), encoding="utf-8")
    return exported


def _validate_numpy_readout_independent(g1: dict[str, Any], g4: dict[str, Any]) -> dict[str, Any]:
    return {method: _validate_readout_numpy(cohort) for method, cohort in (("G1", g1), ("G4", g4))}


def _run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    config_path = args.config.resolve()
    source_checkpoint = args.source_checkpoint.resolve()
    baseline_dir = args.baseline_evidence_dir.resolve()
    run_dir = args.output_dir.resolve()
    config = _read_config(config_path)
    expected_output = (ROOT / config["output"]["experiment_root"]).resolve()
    _require(run_dir == expected_output, "output directory differs from the explicitly registered new attempt")
    _require(torch.cuda.is_available(), "the authorized AutoDL task requires CUDA")
    context = _load_context(source_checkpoint, baseline_dir, config)
    pixels, all_labels, train_rows, dev_rows = base._load_data(ROOT / config["data"]["training_file"])
    train_labels = all_labels[train_rows].clone()
    dev_labels = all_labels[dev_rows]
    data_roles = _validate_data_roles(train_rows, dev_rows, train_labels, dev_labels, config["data"])
    run_identity_path = run_dir / "run_identity.json"
    if args.preflight_only:
        if run_dir.exists():
            _require(args.resume and not (run_dir / "report.json").is_file(),
                     f"existing preflight directory requires explicit --resume: {run_dir}")
            existing_identity = None
            if run_identity_path.is_file():
                existing_identity = json.loads(run_identity_path.read_text(encoding="utf-8"))
                run_id = existing_identity.get("run_id")
                _require(isinstance(run_id, str) and len(run_id) == 32,
                         "existing preflight run identity is malformed")
            else:
                allowed = {"engineering_error.json", "run_state.json"}
                _require({item.name for item in run_dir.iterdir()} <= allowed,
                         "refusing to reuse a preflight directory with unrecognized partial artifacts")
                run_id = uuid.uuid4().hex
        else:
            _require(not args.resume, "preflight resume requires an existing explicit output directory")
            run_dir.mkdir(parents=True, exist_ok=False)
            run_id = uuid.uuid4().hex
    else:
        _require(args.resume, "full measurement requires the completed representative preflight via --resume")
        run_id = ""
    identity = _make_run_identity(source_checkpoint, baseline_dir, config_path, config, context, run_id)

    device = torch.device("cuda")
    grid = int(config["optics"]["propagation_grid"])
    baseline_config = context["baseline_config"]
    model = base._model("small", grid, device, baseline_config)
    model.load_state_dict(context["checkpoint"]["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model_state = base._state_snapshot(model)
    evaluator_d = base._load_evaluator(ROOT / baseline_config["evaluation"]["evaluator_checkpoint"], device)
    evaluator_c, ec_qualification = _load_ec(ROOT / baseline_config["evaluation"]["ec_checkpoint"], device)
    ed_state = base._state_snapshot(evaluator_d)
    ec_state = base._state_snapshot(evaluator_c)
    _require(not evaluator_d.training and not evaluator_c.training,
             "confirmation evaluators must be in eval mode")
    gain = context["checkpoint"]["camera_gain"].to(device=device)
    cohort_source = context["existing_cohort"]
    preflight_replay = _preflight_replay(model, cohort_source, evaluator_d, evaluator_c, gain, device)
    _require(preflight_replay["E_D_predictions_equal"] and preflight_replay["E_C_predictions_equal"],
             "G1 representative prediction replay failed")
    if args.preflight_only:
        torch.cuda.synchronize(device)
        preflight = {
            "schema": "lightgen_small_conditional_mixture_preflight.v1", "task_id": TASK_ID,
            "status": "READY", "run_id": run_id,
            "source_checkpoint": str(source_checkpoint), "source_run_id": context["source_run_id"],
            "source_checkpoint_step": SOURCE_STEP, "source_branch": "C",
            "model_identity": context["checkpoint"]["model_identity"],
            "representative_G1_replay": preflight_replay,
            "E_C_qualification": ec_qualification,
            "data_roles": data_roles,
            "train_rows": int(train_rows.numel()), "dev_rows": int(dev_rows.numel()),
            "official_TEST_reads": 0, "encoder_forwards": 0, "decoder_samples": 10,
            "scientific_updates": 0, "Adam_calls": 0, "phase_updates": 0,
            "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)),
            "device": torch.cuda.get_device_name(device),
            "elapsed_seconds": time.perf_counter() - started,
            "G1_G4_preflight_decision": "G1 replay matches fixed C baseline before fitting or generation",
        }
        atomic_write_json(run_dir / "config.json", config)
        atomic_write_json(run_identity_path, identity)
        atomic_write_json(run_dir / "preflight.json", preflight)
        atomic_write_json(run_dir / "run_state.json", {"status": "READY", "phase": "PREFLIGHT_COMPLETE",
                                                         "scientific_updates": 0})
        _require(base._assert_close(model_state, base._state_snapshot(model), "model_after_preflight") == 0.0
                 and base._assert_close(ed_state, base._state_snapshot(evaluator_d), "E_D_after_preflight") == 0.0
                 and base._assert_close(ec_state, base._state_snapshot(evaluator_c), "E_C_after_preflight") == 0.0,
                 "model or evaluator changed during representative preflight")
        print(json.dumps({"status": "READY", "preflight": str(run_dir / "preflight.json"),
                          "run_id": run_id}), flush=True)
        return

    saved_identity = json.loads(run_identity_path.read_text(encoding="utf-8"))
    identity["run_id"] = saved_identity.get("run_id")
    identity["config"] = config
    _validate_resume(run_dir, identity)
    run_id = identity["run_id"]
    preflight = json.loads((run_dir / "preflight.json").read_text(encoding="utf-8"))
    _require(preflight.get("data_roles") == data_roles,
             "current fixed TRAIN/DEV population differs from the READY preflight")
    cache_path = run_dir / "train_coefficient_cache.pt"
    encode_seconds = 0.0
    encoder_forward_batches = math.ceil(train_rows.numel() / BATCH_SIZE)
    if cache_path.is_file():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        features = _validate_feature_cache(cache, train_rows, train_labels, context["source_run_id"])
        encoding_seconds_total = float(cache.get("elapsed_seconds", math.nan))
        _require(math.isfinite(encoding_seconds_total) and encoding_seconds_total >= 0.0
                 and cache.get("encoder_forward_batches") == encoder_forward_batches
                 and cache.get("encoder_forward_images") == train_rows.numel(),
                 "TRAIN feature cache is missing its completed encoding record")
    else:
        atomic_write_json(run_dir / "run_state.json", {"status": "SCIENTIFIC_ACTIVE",
                                                         "phase": "TRAIN_FEATURE_ENCODING",
                                                         "scientific_updates": 0})
        features, encode_seconds, encoder_forward_batches = _encode_train_features(
            model, pixels, train_rows, train_labels, context["checkpoint"]["scale"], device,
            cache_path, context["source_run_id"],
        )
        encoding_seconds_total = encode_seconds
        _validate_feature_cache(torch.load(cache_path, map_location="cpu", weights_only=False),
                                train_rows, train_labels, context["source_run_id"])
    atomic_write_json(run_dir / "run_state.json", {"status": "SCIENTIFIC_ACTIVE",
                                                     "phase": "FIT_G1_AND_G4_PRIORS",
                                                     "scientific_updates": 0})
    g1_priors, g4_priors, assignments, g1_checks = _fit_priors(
        features, train_labels, context["prior_payload"], run_dir,
    )
    _require(torch.equal(train_rows, train_rows.sort().values), "TRAIN cache rows are not in the registered order")

    if (run_dir / "generated_G1.pt").is_file() and (run_dir / "g1_full_replay.json").is_file():
        g1_cohort = torch.load(run_dir / "generated_G1.pt", map_location="cpu", weights_only=False)
        _validate_saved_cohort(g1_cohort, "G1", run_id, context["source_run_id"])
        replay_receipt = json.loads((run_dir / "g1_full_replay.json").read_text(encoding="utf-8"))
        _require(replay_receipt.get("task_id") == TASK_ID and replay_receipt.get("run_id") == run_id
                 and all(replay_receipt.get("checks", {}).get(key, {}).get("passed") is True
                         for key in ("view", "E_D_logits", "E_C_logits"))
                 and replay_receipt.get("checks", {}).get("E_D_predictions_equal") is True
                 and replay_receipt.get("checks", {}).get("E_C_predictions_equal") is True,
                 "saved G1 replay receipt is incomplete or failed")
        g1_replay_seconds = float(replay_receipt.get("elapsed_seconds", math.nan))
        _require(math.isfinite(g1_replay_seconds) and g1_replay_seconds >= 0.0,
                 "saved G1 replay duration is invalid")
    else:
        atomic_write_json(run_dir / "run_state.json", {"status": "SCIENTIFIC_ACTIVE",
                                                         "phase": "G1_FULL_BASELINE_REPLAY",
                                                         "scientific_updates": 0})
        g1_started = time.perf_counter()
        labels_g1 = cohort_source["requested_labels"].cpu().long()
        g1_cohort = _decode_cohort("G1", model, evaluator_d, evaluator_c,
                                   cohort_source["ols_coefficients"].cpu(), labels_g1,
                                   torch.zeros(10_000, dtype=torch.long), cohort_source["proposal_ids"].cpu().long(),
                                   gain, device, capture_count=10)
        g1_cohort["source_run_id"] = context["source_run_id"]
        g1_cohort["run_id"] = run_id
        _require(torch.equal(g1_cohort["proposal_ids"], cohort_source["proposal_ids"].cpu())
                 and torch.equal(g1_cohort["requested_labels"], labels_g1),
                 "G1 replay changed saved class/proposal identities")
        replay_checks = {
            "view": _compare_replay(g1_cohort["images"], cohort_source["images"], "full_view"),
            "E_D_logits": _compare_replay(g1_cohort["ed_logits"], cohort_source["ed_logits"], "full_E_D_logits"),
            "E_C_logits": _compare_replay(g1_cohort["ec_logits"], cohort_source["ec_logits"], "full_E_C_logits"),
            "E_D_predictions_equal": bool(torch.equal(g1_cohort["ed_predictions"], cohort_source["ed_predictions"])),
            "E_C_predictions_equal": bool(torch.equal(g1_cohort["ec_predictions"], cohort_source["ec_predictions"])),
        }
        _require(replay_checks["E_D_predictions_equal"] and replay_checks["E_C_predictions_equal"],
                 "G1 full saved-cohort predictions differ from original C results")
        atomic_torch_save(run_dir / "generated_G1.pt", g1_cohort)
        g1_replay_seconds = time.perf_counter() - g1_started
        atomic_write_json(run_dir / "g1_full_replay.json", {"task_id": TASK_ID, "run_id": run_id,
                                                             "checks": replay_checks,
                                                             "elapsed_seconds": g1_replay_seconds})
    g1_replay_seconds = locals().get("g1_replay_seconds", 0.0)

    if (run_dir / "generated_G4.pt").is_file() and (run_dir / "sampling_rejections_G4.json").is_file():
        g4_cohort = torch.load(run_dir / "generated_G4.pt", map_location="cpu", weights_only=False)
        _validate_saved_cohort(g4_cohort, "G4", run_id, context["source_run_id"])
        sampling_payload = json.loads((run_dir / "sampling_rejections_G4.json").read_text(encoding="utf-8"))
        _require(sampling_payload.get("task_id") == TASK_ID and sampling_payload.get("run_id") == run_id
                 and isinstance(sampling_payload.get("per_class"), dict)
                 and all(str(label) in sampling_payload["per_class"] for label in range(10)),
                 "saved G4 rejection record is incomplete or has another identity")
        sampling_records = sampling_payload["per_class"]
        sampling_seconds = float(sampling_payload.get("elapsed_seconds", math.nan))
        _require(math.isfinite(sampling_seconds) and sampling_seconds >= 0.0,
                 "saved G4 sampling duration is invalid")
    else:
        atomic_write_json(run_dir / "run_state.json", {"status": "SCIENTIFIC_ACTIVE",
                                                         "phase": "G4_POWER_REJECTED_SAMPLING_AND_DECODE",
                                                         "scientific_updates": 0})
        sampling_started = time.perf_counter()
        coefficients_g4, labels_g4, components_g4, proposals_g4, sampling_records = _sample_g4(
            g4_priors, context["checkpoint"]["scale"],
        )
        sampling_seconds = time.perf_counter() - sampling_started
        atomic_write_json(run_dir / "sampling_rejections_G4.json", {
            "task_id": TASK_ID, "run_id": run_id, "elapsed_seconds": sampling_seconds,
            "per_class": sampling_records,
        })
        g4_cohort = _decode_cohort("G4", model, evaluator_d, evaluator_c,
                                   coefficients_g4, labels_g4, components_g4, proposals_g4,
                                   gain, device, capture_count=10)
        g4_cohort["source_run_id"] = context["source_run_id"]
        g4_cohort["run_id"] = run_id
        for label in range(10):
            mask = g4_cohort["requested_labels"] == label
            _require(int(mask.sum()) == SAMPLE_COUNT and int(g4_cohort["component_ids"][mask].max()) < 4,
                     f"G4 class {label} population or component identity is invalid")
        atomic_torch_save(run_dir / "generated_G4.pt", g4_cohort)
        _validate_saved_cohort(g4_cohort, "G4", run_id, context["source_run_id"])
    atomic_write_json(run_dir / "run_state.json", {"status": "SCIENTIFIC_ACTIVE",
                                                     "phase": "REPORT_AND_VALIDATION",
                                                     "scientific_updates": 0})

    readout_checks = _validate_numpy_readout_independent(g1_cohort, g4_cohort)
    g1_metrics, g1_rows = _cohort_metrics(g1_cohort, context["reference_generation"])
    g4_metrics, g4_rows = _cohort_metrics(g4_cohort, context["reference_generation"])
    source_generation = context["endpoint"]["generation"]
    _require(abs(g1_metrics["E_D_macro_recall"] - float(source_generation["ed_macro_recall"])) <= 1e-12
             and abs(g1_metrics["E_C_macro_recall"] - float(source_generation["ec_macro_recall"])) <= 1e-12,
             "G1 full replay metrics differ from the saved C endpoint metrics")
    _require(all(abs(g1_metrics["generated_within_class_tv_mean"][label]
                     - float(source_generation["generated_within_class_tv_mean"][label])) <= 1e-6
                 for label in range(10)),
             "G1 replay does not reproduce the saved C shape-diversity references")
    bootstrap, bootstrap_arrays = _bootstrap_differences(
        g1_cohort, g4_cohort, RESAMPLE_SEED, BOOTSTRAP_RESAMPLES,
    )
    atomic_write_json(run_dir / "generation_metrics_G1.json", g1_metrics)
    atomic_write_json(run_dir / "generation_metrics_G4.json", g4_metrics)
    atomic_write_json(run_dir / "bootstrap_differences.json", bootstrap)
    np.savez_compressed(run_dir / "bootstrap_samples.npz", **bootstrap_arrays)
    _write_per_image_csv(run_dir / "per_image_metrics.csv", g1_rows + g4_rows)
    gallery_count = _write_gallery(run_dir, {"G1": g1_cohort, "G4": g4_cohort})

    ed_ci = bootstrap["differences"]["ed_correct"]["ci_95"]
    ec_ci = bootstrap["differences"]["ec_correct"]["ci_95"]
    clipping_ci = bootstrap["differences"]["proxy_clipping_area_fraction"]["ci_95"]
    shape_passed = all(value >= float(config["evaluation"]["minimum_generation_to_dev_tv_ratio"])
                       for value in g4_metrics["generated_to_dev_tv_ratio"])
    proxy_gate = (ed_ci[0] > 0.0 and ec_ci[0] > 0.0 and clipping_ci[1] <= 0.0 and shape_passed)
    if proxy_gate:
        decision = "FIXED_G4_PROXY_IMPROVEMENT_SUPPORTED"
        interpretation = ("The fixed four-component prior improved both frozen classifier macro-recall proxies, "
                          "did not increase macro proxy clipping, and kept every class shape ratio above threshold.")
    elif (ed_ci[0] <= 0.0 <= ed_ci[1] or ec_ci[0] <= 0.0 <= ec_ci[1]
          or clipping_ci[0] <= 0.0 <= clipping_ci[1] or not shape_passed):
        decision = "MIXED_OR_UNSUPPORTED_TRADEOFF"
        interpretation = "The registered proxy criteria are mixed or their intervals do not support replacing G1."
    else:
        decision = "G4_CANDIDATE_NOT_SUPPORTED"
        interpretation = "The fixed G4 candidate failed the registered proxy improvement rule."
    report = {
        "schema": "lightgen_small_conditional_mixture_prior_ab_report.v1",
        "task_id": TASK_ID, "run_id": run_id, "status": "COMPLETE_DIAGNOSTIC",
        "measurement_complete": True, "evidence_tier": "diagnostic_only",
        "scientific_question": "At the frozen C 45,000-step optical endpoint, does a fixed four-component class prior improve generation proxy metrics while controlling clipping and class shape distance?",
        "decision": {"result": decision, "interpretation": interpretation,
                     "gate": {"E_D_macro_recall_delta_ci_lower_gt_zero": ed_ci[0] > 0.0,
                              "E_C_macro_recall_delta_ci_lower_gt_zero": ec_ci[0] > 0.0,
                              "macro_clipping_delta_ci_upper_le_zero": clipping_ci[1] <= 0.0,
                              "all_G4_class_shape_ratios_at_least_0_25": shape_passed,
                              "E_D_delta_ci_95": ed_ci, "E_C_delta_ci_95": ec_ci,
                              "macro_clipping_delta_ci_95": clipping_ci}},
        "source": {"task_id": SOURCE_TASK_ID, "run_id": context["source_run_id"],
                   "branch": "C", "cumulative_step": SOURCE_STEP,
                   "historical_status_preserved": context["source_status"],
                   "human_review_status": context["final_report"]["human_review_status"],
                   "human_readability_inference": "not made"},
        "methods": {
            "G1": {"prior": "existing single full-covariance Gaussian per class, refit from the fixed TRAIN cache and matched against saved prior",
                   "metrics": g1_metrics},
            "G4": {"prior": "four full-covariance Gaussians per class, fixed K-means++ seed, one initialization, power rejection only",
                   "metrics": g4_metrics},
        },
        "bootstrap": bootstrap,
        "fitting": {
            "train_rows": int(train_rows.numel()), "official_test_reads": 0,
            **data_roles,
            "encoder_forward_images": int(train_rows.numel()),
            "encoder_forward_batches": int(encoder_forward_batches),
            "encoder_forward_images_this_process": 0 if (cache_path.is_file() and encode_seconds == 0.0)
            else int(train_rows.numel()),
            "encoder_forward_batches_this_process": 0 if (cache_path.is_file() and encode_seconds == 0.0)
            else int(encoder_forward_batches),
            "encoding_seconds_total_from_cache": encoding_seconds_total,
            "G1_cache_reference_check": g1_checks,
            "G4_components_per_class": 4,
            "G4_cluster_assignments_saved_in_cache_order": True,
            "covariance_ridge": "reuses original G1 class ridge = 1e-6 times mean diagonal covariance",
        },
        "replay": {
            "G1_rows": 10_000, "G1_view_and_logits_replay_checks": "g1_full_replay.json",
            "replay_rtol": REPLAY_RTOL, "replay_atol": REPLAY_ATOL,
            "G1_prediction_match": True,
            "independent_numpy_readout": readout_checks,
        },
        "sampling": {"G4_samples_per_class": SAMPLE_COUNT, "G4_population": 10_000,
                     "max_proposals_per_class": MAX_PROPOSALS,
                     "rejection_records": sampling_records,
                     "classifier_based_sample_selection": False,
                     "per_image_normalization": False},
        "updates": {"scientific_updates": 0, "adam_calls": 0, "phase_updates": 0,
                    "gradients": 0, "official_test_reads": 0},
        "immutability": {"model_unchanged": True, "E_D_unchanged": True, "E_C_unchanged": True},
        "artifacts": {
            "train_coefficient_cache": "train_coefficient_cache.pt",
            "priors": "fitted_priors_G1_G4.pt",
            "G1_cohort": "generated_G1.pt", "G4_cohort": "generated_G4.pt",
            "per_image_metrics": "per_image_metrics.csv", "bootstrap": "bootstrap_differences.json",
            "gallery": "gallery.html", "gallery_png_count": gallery_count,
        },
        "elapsed_seconds": {
            "total_wall_seconds": time.perf_counter() - started,
            "train_encoding_seconds_this_process": encode_seconds,
            "G1_replay_seconds_this_process": g1_replay_seconds,
            "G4_sampling_seconds_this_process": sampling_seconds,
        },
    }
    _require(base._assert_close(model_state, base._state_snapshot(model), "frozen_model_after_measurement") == 0.0
             and base._assert_close(ed_state, base._state_snapshot(evaluator_d), "frozen_E_D_after_measurement") == 0.0
             and base._assert_close(ec_state, base._state_snapshot(evaluator_c), "frozen_E_C_after_measurement") == 0.0,
             "model or frozen evaluator changed during prior fitting or evaluation")
    output_bytes = base._directory_bytes(run_dir)
    report["output_bytes_before_report"] = output_bytes
    report["output_budget_bytes"] = int(config["output"]["formal_result_budget_bytes"])
    report["within_output_budget"] = output_bytes <= report["output_budget_bytes"]
    if not report["within_output_budget"]:
        report["measurement_complete"] = False
        report["status"] = "MEASUREMENT_INCOMPLETE"
    validation = {
        "status": "PASSED" if report["measurement_complete"] else "OUTPUT_BUDGET_EXCEEDED",
        "checks": {
            "source_checkpoint_identity_and_adam_state": "PASSED",
            "source_review_formally_closed": "PASSED",
            "train_dev_roles_and_population": "PASSED",
            "G1_prior_refit_matches_saved_mean_covariance_cholesky": "PASSED",
            "G1_full_view_and_logits_replay": "PASSED",
            "G4_fixed_component_fit_and_power_rejection": "PASSED",
            "independent_numpy_readout_consumer": "PASSED",
            "model_and_evaluator_state_unchanged": "PASSED",
            "zero_optimizer_gradient_and_phase_updates": "PASSED",
            "official_test_reads_zero": "PASSED",
            "output_budget": "PASSED" if report["within_output_budget"] else "FAILED",
        },
        "run_id": run_id, "scientific_updates": 0,
        "optimizer_calls": 0, "gradient_calls": 0, "official_test_reads": 0,
    }
    atomic_write_json(run_dir / "validation_receipt.json", validation)
    atomic_write_json(run_dir / "report.json", report)
    atomic_write_json(run_dir / "run_state.json", {
        "status": report["status"], "decision": decision,
        "measurement_complete": report["measurement_complete"],
        "scientific_updates": 0, "official_test_reads": 0,
    })
    _require(report["measurement_complete"], "formal result package exceeded the fixed 512 MiB budget")
    print(json.dumps({"status": report["status"], "decision": decision,
                      "report": str(run_dir / "report.json"),
                      "output_bytes": output_bytes,
                      "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-evidence-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        _run(args)
    except BaseException as error:
        run_dir = args.output_dir.resolve()
        if run_dir.is_dir() and not (run_dir / "report.json").is_file():
            atomic_write_json(run_dir / "engineering_error.json", {
                "task_id": TASK_ID, "status": "ENGINEERING_INVALID",
                "error_type": type(error).__name__, "error": str(error),
            })
            atomic_write_json(run_dir / "run_state.json", {
                "status": "ENGINEERING_INVALID", "error_type": type(error).__name__,
                "scientific_updates": 0,
            })
        raise


if __name__ == "__main__":
    main()
