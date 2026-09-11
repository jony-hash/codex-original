"""Diagnose the frozen LightGen real-image control failure without retraining.

This consumer is deliberately bounded to non-protected train/development/
calibration roles and the already-materialized Stage-1 control summary.  It
does not run a protected forward pass, create an optimizer, update parameters,
generate from the prior, tune an evaluator, or reuse consumed sealed values.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ldm_project.v2.application.research.lightgen_2d_semantic_complex_ols_training import (
    ALPHA_LATENT,
    BETA_IMAGE,
    GAMMA_OPTICAL,
    GAUGE_EPSILON,
    LightGen2DComplexOLSTrainingConsumer,
    OLS_PORT_COUNT,
    PRIOR_NAMESPACE,
    PRIOR_SIGMA,
    PRODUCTION_PROPAGATION_GRID,
    derive_prior_seed,
    gauge_invariant_ols_representation,
    load_frozen_training_contract,
)
from ldm_project.v2.evidence import (
    atomic_write_new_json,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ldm_project.v2.optics.propagation import clear_production_transfer_cache
from scripts.research import (
    execute_lightgen_2d_semantic_complex_ols_post_training_evaluation_v1 as evaluation,
)


TASK_ID = "DIAGNOSE_LIGHTGEN_2D_SEMANTIC_REAL_PATH_CONTROL_FAILURE_V1"
GATE = TASK_ID
OUTPUT_DIR = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_real_path_control_failure_diagnosis_v1_attempt_01"
)
REPORT_NAME = "lightgen_2d_semantic_real_path_control_failure_diagnosis_report.v1.json"
LINEAGE_NAME = "lightgen_2d_semantic_real_path_control_failure_diagnosis_lineage.v1.json"
REPORT_SCHEMA = "lightgen_2d_semantic_real_path_control_failure_diagnosis_report.v1"
LINEAGE_SCHEMA = "lightgen_2d_semantic_real_path_control_failure_diagnosis_lineage.v1"

POST_EVALUATION_REPORT_PATH = evaluation.OUTPUT_DIR / evaluation.REPORT_NAME
POST_EVALUATION_REPORT_SHA256 = "4048de57a35a1899240a231cfbeec70ed5950275b2014fbcb1dfc96c664c4996"
POST_EVALUATION_LINEAGE_PATH = evaluation.OUTPUT_DIR / evaluation.LINEAGE_NAME
POST_EVALUATION_LINEAGE_SHA256 = "9f1a89bafa8f57f2d6cd4242de2153a2cd5d4c02ac2d2ad82a652680a204f19a"
POST_EVALUATION_CONTROL_PATH = evaluation.OUTPUT_DIR / evaluation.CONTROL_ARTIFACT_NAME
POST_EVALUATION_CONTROL_SHA256 = "c5b2cf0bd2d98b5eceb4a53881312c1397bda91e68a87def788364d72244a7a3"

DIAGNOSTIC_PANEL_PER_CLASS = 100
TRAIN_PANEL_SEED = 20260913
PAIR_DISTANCE_SEED = 20260914
GRADIENT_GLOBAL_STEP = 99
GRADIENT_BATCH_SIZE = 10
NUMERICAL_COLLAPSE_TOLERANCE = 1.0e-12
POWER_COLLAPSE_RATIO = 1.0e-2

PRIMARY_CLASSIFICATIONS = {
    "A": "REAL_PATH_FAILURE_EVALUATOR_DOMAIN_MISMATCH_SUPPORTED",
    "B": "REAL_PATH_FAILURE_RECONSTRUCTION_COLLAPSE_SUPPORTED",
    "C": "REAL_PATH_FAILURE_ENCODER_OLS_SEMANTIC_COLLAPSE_SUPPORTED",
    "D": "REAL_PATH_FAILURE_GENERATOR_DECODE_SUPPORTED",
    "E": "REAL_PATH_FAILURE_LATENT_REGULARIZATION_CONFLICT_SUPPORTED",
    "F": "REAL_PATH_FAILURE_OPTICAL_REGULARIZATION_CONFLICT_SUPPORTED",
    "G": "REAL_PATH_FAILURE_CHECKPOINT_OBJECTIVE_MISMATCH_SUPPORTED",
    "H": "REAL_PATH_FAILURE_INSUFFICIENT_TRAINING_PROGRESS_SUPPORTED",
    "I": "REAL_PATH_FAILURE_GENERALIZATION_SHIFT_SUPPORTED",
}
MIXED_CLASSIFICATION = "REAL_PATH_FAILURE_MIXED"
UNRESOLVED_CLASSIFICATION = "REAL_PATH_FAILURE_UNRESOLVED"


class BlockedDiagnosis(RuntimeError):
    """A frozen authority or data boundary cannot be established."""


class InvalidDiagnosis(RuntimeError):
    """The diagnostic harness is invalid, not a scientific result."""


def _require(condition: bool, message: str, *, blocked: bool = False) -> None:
    if condition:
        return
    error = BlockedDiagnosis if blocked else InvalidDiagnosis
    raise error(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _git_state(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    result = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    status = result.stdout.splitlines()
    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "dirty_paths": [line[3:] if len(line) >= 4 else line for line in status],
        "worktree_dirty": bool(status),
    }


def _ref(repo_root: Path, path: Path, role: str, expected_sha256: str) -> dict[str, Any]:
    target = repo_root / path
    _require(target.is_file(), f"{role} is missing: {path.as_posix()}", blocked=True)
    actual = file_sha256(target)
    _require(actual == expected_sha256, f"{role} SHA256 changed", blocked=True)
    return {"path": _relative(repo_root, target), "role": role, "sha256": actual}


def _read_json(repo_root: Path, path: Path, role: str, expected_sha256: str) -> dict[str, Any]:
    target = repo_root / path
    _ref(repo_root, path, role, expected_sha256)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise BlockedDiagnosis(f"{role} cannot be read") from exc
    _require(isinstance(value, dict), f"{role} must be a JSON object", blocked=True)
    return value


def _summary(values: torch.Tensor) -> dict[str, Any]:
    values = values.detach().cpu().to(torch.float64).flatten()
    _require(values.numel() > 0 and torch.isfinite(values).all().item(), "summary values are invalid")
    quantiles = torch.quantile(values, torch.tensor([0.01, 0.50, 0.99], dtype=torch.float64))
    return {
        "count": int(values.numel()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(quantiles[1]),
        "q01": float(quantiles[0]),
        "q99": float(quantiles[2]),
    }


def _balanced_panel(
    role_indices: torch.Tensor,
    labels: torch.Tensor,
    *,
    per_class: int,
    seed: int,
) -> torch.Tensor:
    _require(role_indices.dtype == torch.long and labels.dtype == torch.long, "diagnostic role indices/labels must be int64")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected: list[torch.Tensor] = []
    role_labels = labels.index_select(0, role_indices)
    for digit in range(evaluation.CLASS_COUNT):
        matches = role_indices[role_labels == digit]
        _require(matches.numel() >= per_class, f"diagnostic role class {digit} is too small")
        order = torch.randperm(matches.numel(), generator=generator)
        selected.append(matches.index_select(0, order[:per_class]))
    panel = torch.cat(selected).contiguous()
    _require(panel.numel() == per_class * evaluation.CLASS_COUNT, "diagnostic panel count changed")
    _require(torch.unique(panel).numel() == panel.numel(), "diagnostic panel contains duplicate indices")
    counts = torch.bincount(labels.index_select(0, panel), minlength=evaluation.CLASS_COUNT)
    _require(torch.equal(counts, torch.full((evaluation.CLASS_COUNT,), per_class)), "diagnostic panel is not class balanced")
    return panel


class ImageDomainAccumulator:
    """Streaming descriptive statistics before evaluator preprocessing."""

    def __init__(self) -> None:
        self.count = 0
        self.pixel_count = 0
        self.sum = 0.0
        self.sum_square = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.zero_count = 0
        self.saturation_count = 0
        self.per_image: dict[str, list[torch.Tensor]] = {
            "minimum": [],
            "maximum": [],
            "dynamic_range": [],
            "zero_fraction": [],
            "saturation_fraction": [],
            "spatial_variance": [],
        }

    def update(self, images: torch.Tensor) -> None:
        _require(torch.is_tensor(images) and images.ndim == 4, "domain-statistics input is invalid")
        values = images.detach().cpu().to(torch.float64)
        _require(torch.isfinite(values).all().item(), "domain-statistics input is nonfinite")
        flat = values.flatten(start_dim=1)
        self.count += int(flat.shape[0])
        self.pixel_count += int(flat.numel())
        self.sum += float(flat.sum())
        self.sum_square += float(flat.square().sum())
        self.minimum = min(self.minimum, float(flat.min()))
        self.maximum = max(self.maximum, float(flat.max()))
        self.zero_count += int((flat == 0.0).sum())
        self.saturation_count += int((flat >= 1.0).sum())
        self.per_image["minimum"].append(flat.min(dim=1).values)
        self.per_image["maximum"].append(flat.max(dim=1).values)
        self.per_image["dynamic_range"].append(flat.max(dim=1).values - flat.min(dim=1).values)
        self.per_image["zero_fraction"].append((flat == 0.0).to(torch.float64).mean(dim=1))
        self.per_image["saturation_fraction"].append((flat >= 1.0).to(torch.float64).mean(dim=1))
        self.per_image["spatial_variance"].append(flat.var(dim=1, unbiased=False))

    def finalize(self, *, preprocessing: str) -> dict[str, Any]:
        _require(self.count > 0 and self.pixel_count > 0, "domain-statistics accumulator is empty")
        mean = self.sum / self.pixel_count
        variance = max(self.sum_square / self.pixel_count - mean * mean, 0.0)
        per_image = {
            name: _summary(torch.cat(values))
            for name, values in self.per_image.items()
        }
        return {
            "count": self.count,
            "pixel_count": self.pixel_count,
            "min": self.minimum,
            "max": self.maximum,
            "dynamic_range": self.maximum - self.minimum,
            "mean": mean,
            "std_population": math.sqrt(variance),
            "zero_fraction": self.zero_count / self.pixel_count,
            "saturation_fraction_ge_1": self.saturation_count / self.pixel_count,
            "per_image": per_image,
            "preprocessing_before_this_measurement": preprocessing,
        }


class EvaluatorAccumulator:
    """Frozen evaluator summary for one diagnostic role and path."""

    def __init__(self) -> None:
        self.count = 0
        self.confidences: list[torch.Tensor] = []
        self.confusion = torch.zeros((evaluation.CLASS_COUNT, evaluation.CLASS_COUNT), dtype=torch.long)

    def update(self, labels: torch.Tensor, predictions: torch.Tensor, confidence: torch.Tensor) -> None:
        labels = labels.detach().cpu().to(torch.long)
        predictions = predictions.detach().cpu().to(torch.long)
        confidence = confidence.detach().cpu().to(torch.float64)
        _require(labels.ndim == predictions.ndim == 1 and labels.shape == predictions.shape, "evaluator labels/predictions are invalid")
        _require(confidence.shape == labels.shape and torch.isfinite(confidence).all().item(), "evaluator confidence is invalid")
        self.count += int(labels.numel())
        self.confidences.append(confidence)
        self.confusion.index_put_((labels, predictions), torch.ones_like(labels), accumulate=True)

    def finalize(self, labels: torch.Tensor) -> dict[str, Any]:
        labels = labels.detach().cpu().to(torch.long)
        _require(labels.numel() == self.count, "evaluator label count differs from measurement")
        label_counts = torch.bincount(labels, minlength=evaluation.CLASS_COUNT)
        correct_by_class = self.confusion.diag()
        recall = correct_by_class.to(torch.float64) / label_counts.to(torch.float64)
        correct = int(correct_by_class.sum())
        confidences = torch.cat(self.confidences)
        _require(torch.isfinite(recall).all().item(), "evaluator recall is nonfinite")
        return {
            "count": self.count,
            "accuracy": correct / self.count,
            "macro_recall": float(recall.mean()),
            "label_counts": label_counts.tolist(),
            "correct_by_class": correct_by_class.tolist(),
            "per_class_recall": recall.tolist(),
            "confusion_matrix_actual_rows_predicted_columns": self.confusion.tolist(),
            "confidence": _summary(confidences),
            "preprocessing": "frozen evaluator path; no alternate normalization/threshold/clipping",
        }


@dataclass
class RoleMeasurement:
    role: str
    indices: torch.Tensor
    labels: torch.Tensor
    z: torch.Tensor
    original_evaluator: dict[str, Any]
    real_path_evaluator: dict[str, Any]
    original_domain: dict[str, Any]
    raw_domain: dict[str, Any]
    target_domain: dict[str, Any]
    reconstruction: dict[str, Any]
    path_proof: dict[str, Any]

    def json_payload(self) -> dict[str, Any]:
        return {
            "count": int(self.indices.numel()),
            "source_indices_sha256": tensor_sha256(self.indices),
            "label_counts": torch.bincount(self.labels, minlength=evaluation.CLASS_COUNT).tolist(),
            "original_evaluator": self.original_evaluator,
            "real_path_evaluator": self.real_path_evaluator,
            "original_domain": self.original_domain,
            "raw_cmos_domain": self.raw_domain,
            "target_slm_domain": self.target_domain,
            "reconstruction": self.reconstruction,
            "path_proof": self.path_proof,
        }


class RoleAccumulator:
    def __init__(self, role: str, indices: torch.Tensor, labels: torch.Tensor) -> None:
        self.role = role
        self.indices = indices.cpu().contiguous()
        self.labels = labels.index_select(0, self.indices).cpu().contiguous()
        self.original_domain = ImageDomainAccumulator()
        self.raw_domain = ImageDomainAccumulator()
        self.target_domain = ImageDomainAccumulator()
        self.original_evaluator = EvaluatorAccumulator()
        self.real_path_evaluator = EvaluatorAccumulator()
        self.z_parts: list[torch.Tensor] = []
        self.mse_parts: list[torch.Tensor] = []
        self.contrast_parts: list[torch.Tensor] = []
        self.inter_sample_parts: list[torch.Tensor] = []
        self.raw_sum = torch.zeros((evaluation.RAW_OUTPUT_SHAPE[1] * evaluation.RAW_OUTPUT_SHAPE[2],), dtype=torch.float64)
        self.raw_square_sum = torch.zeros_like(self.raw_sum)
        self.last_raw: torch.Tensor | None = None
        self.path_proof: dict[str, Any] | None = None

    def update(
        self,
        *,
        images: torch.Tensor,
        labels: torch.Tensor,
        raw_cmos: torch.Tensor,
        target_slm: torch.Tensor,
        z: torch.Tensor,
        original_predictions: torch.Tensor,
        original_confidence: torch.Tensor,
        real_predictions: torch.Tensor,
        real_confidence: torch.Tensor,
        path_proof: dict[str, Any],
    ) -> None:
        images_cpu = images.detach().cpu()
        raw_cpu = raw_cmos.detach().cpu()
        target_cpu = target_slm.detach().cpu()
        _require(tuple(raw_cpu.shape[1:]) == evaluation.RAW_OUTPUT_SHAPE, "raw CMOS shape changed during diagnosis")
        _require(tuple(target_cpu.shape[1:]) == evaluation.RAW_OUTPUT_SHAPE, "aligned target shape changed during diagnosis")
        self.original_domain.update(images_cpu)
        self.raw_domain.update(raw_cpu)
        self.target_domain.update(target_cpu)
        self.original_evaluator.update(labels, original_predictions, original_confidence)
        self.real_path_evaluator.update(labels, real_predictions, real_confidence)
        self.z_parts.append(z.detach().cpu().to(torch.complex64).contiguous())

        raw_flat = raw_cpu.to(torch.float64).flatten(start_dim=1)
        target_flat = target_cpu.to(torch.float64).flatten(start_dim=1)
        self.raw_sum.add_(raw_flat.sum(dim=0))
        self.raw_square_sum.add_(raw_flat.square().sum(dim=0))
        self.mse_parts.append((raw_flat - target_flat).square().mean(dim=1))
        foreground = target_flat > 0.0
        foreground_count = foreground.sum(dim=1)
        background_count = (~foreground).sum(dim=1)
        valid = (foreground_count > 0) & (background_count > 0)
        if valid.any():
            foreground_mean = (raw_flat * foreground).sum(dim=1) / foreground_count.clamp_min(1)
            background_mean = (raw_flat * (~foreground)).sum(dim=1) / background_count.clamp_min(1)
            self.contrast_parts.append((foreground_mean - background_mean)[valid])

        distances: list[torch.Tensor] = []
        if self.last_raw is not None:
            distances.append((raw_flat[0] - self.last_raw).square().mean().sqrt().reshape(1))
        if raw_flat.shape[0] > 1:
            distances.append((raw_flat[1:] - raw_flat[:-1]).square().mean(dim=1).sqrt())
        if distances:
            self.inter_sample_parts.append(torch.cat(distances))
        self.last_raw = raw_flat[-1].clone()
        if self.path_proof is None:
            self.path_proof = dict(path_proof)

    def finalize(self) -> RoleMeasurement:
        z = torch.cat(self.z_parts).contiguous()
        mse = torch.cat(self.mse_parts)
        inter_sample = torch.cat(self.inter_sample_parts)
        contrast = torch.cat(self.contrast_parts) if self.contrast_parts else torch.zeros((1,), dtype=torch.float64)
        count = float(self.indices.numel())
        prototype_variance = (self.raw_square_sum / count - (self.raw_sum / count).square()).clamp_min(0.0)
        raw_domain = self.raw_domain.finalize(preprocessing="none; raw CMOS measured before clamp/pool evaluator view")
        reconstruction = {
            "mse": _summary(mse),
            "normalized_mse": {
                "status": "UNRESOLVED",
                "reason": "no normalized-MSE convention is established by the frozen authority",
            },
            "psnr": {"status": "UNRESOLVED", "reason": "not an existing frozen diagnostic definition"},
            "ssim": {"status": "UNRESOLVED", "reason": "not an existing frozen diagnostic definition"},
            "foreground_background_contrast": {
                "metric": "mean(raw CMOS over target>0) - mean(raw CMOS over target==0)",
                "target_mask_rule": "exact target_slm_intensity > 0; descriptive only",
                "summary": _summary(contrast),
            },
            "per_image_output_variance": raw_domain["per_image"]["spatial_variance"],
            "inter_sample_output_distance": {
                "metric": "adjacent/boundary raw-CMOS RMS pixel distance",
                "summary": _summary(inter_sample),
            },
            "mean_image_prototype_distance": {
                "metric": "RMS pixel distance to the role mean raw-CMOS image",
                "value": float(prototype_variance.mean().sqrt()),
            },
        }
        return RoleMeasurement(
            role=self.role,
            indices=self.indices,
            labels=self.labels,
            z=z,
            original_evaluator=self.original_evaluator.finalize(self.labels),
            real_path_evaluator=self.real_path_evaluator.finalize(self.labels),
            original_domain=self.original_domain.finalize(preprocessing="none; source is already normalized 28x28 evaluator input"),
            raw_domain=raw_domain,
            target_domain=self.target_domain.finalize(preprocessing="none; aligned SLM intensity target"),
            reconstruction=reconstruction,
            path_proof=self.path_proof or {},
        )


def _load_published_evaluation(repo_root: Path) -> dict[str, Any]:
    report = _read_json(
        repo_root,
        POST_EVALUATION_REPORT_PATH,
        "published post-training evaluation report",
        POST_EVALUATION_REPORT_SHA256,
    )
    lineage = _read_json(
        repo_root,
        POST_EVALUATION_LINEAGE_PATH,
        "published post-training evaluation lineage",
        POST_EVALUATION_LINEAGE_SHA256,
    )
    control_path = repo_root / POST_EVALUATION_CONTROL_PATH
    _ref(repo_root, POST_EVALUATION_CONTROL_PATH, "published Stage-1 control artifact", POST_EVALUATION_CONTROL_SHA256)
    control = safe_torch_load(control_path, expected_sha256=POST_EVALUATION_CONTROL_SHA256, map_location="cpu")
    _require(isinstance(control, Mapping), "published control artifact is not a mapping", blocked=True)
    _require(control.get("schema") == evaluation.CONTROL_ARTIFACT_SCHEMA, "published control artifact schema changed", blocked=True)
    _require(report.get("schema") == evaluation.REPORT_SCHEMA, "published evaluation report schema changed", blocked=True)
    _require(report.get("result") == "VALID_COMPLETED", "published evaluation result changed", blocked=True)
    _require(
        report.get("classification") == evaluation.CONTROL_FAILURE_CLASSIFICATION,
        "published evaluation classification changed",
        blocked=True,
    )
    control_summary = _mapping(report.get("control"), "published control summary")
    _require(control_summary.get("status") == "FAIL" and control_summary.get("pass") is False, "published control status changed", blocked=True)
    _require(
        float(control_summary.get("protected_macro_recall")) == 0.16177222311773765,
        "published protected macro recall changed",
        blocked=True,
    )
    _require(float(control_summary.get("threshold")) == 0.937, "published control threshold changed", blocked=True)
    _require(lineage.get("schema") == evaluation.LINEAGE_SCHEMA, "published evaluation lineage schema changed", blocked=True)
    _require(lineage.get("result") == "VALID_COMPLETED", "published evaluation lineage result changed", blocked=True)
    firewall = _mapping(lineage.get("data_firewall"), "published evaluation data firewall")
    _require(firewall.get("sealed_values_loaded") is False and firewall.get("sealed_values_reused") is False, "published sealed boundary changed", blocked=True)
    _require(_mapping(firewall.get("protected_source"), "published protected source").get("value_load_count") == 1, "published protected load count changed", blocked=True)
    return {
        "report": report,
        "lineage": lineage,
        "control": control,
        "report_ref": _ref(repo_root, POST_EVALUATION_REPORT_PATH, "published evaluation report", POST_EVALUATION_REPORT_SHA256),
        "lineage_ref": _ref(repo_root, POST_EVALUATION_LINEAGE_PATH, "published evaluation lineage", POST_EVALUATION_LINEAGE_SHA256),
        "control_ref": _ref(repo_root, POST_EVALUATION_CONTROL_PATH, "published Stage-1 control artifact", POST_EVALUATION_CONTROL_SHA256),
    }


def _preflight(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    try:
        evaluation_preflight = evaluation._preflight(repo_root)
    except (evaluation.BlockedExecution, evaluation.InvalidHarnessError) as exc:
        raise BlockedDiagnosis(str(exc)) from exc
    published = _load_published_evaluation(repo_root)
    _require(torch.cuda.is_available(), "CUDA is required for the authoritative 2048-grid diagnostic", blocked=True)
    _require(torch.cuda.device_count() >= 1, "cuda:0 is unavailable", blocked=True)
    git_before = evaluation_preflight["git_before"]
    _require(git_before["branch"] == evaluation.EXPECTED_BRANCH, "diagnostic branch changed", blocked=True)
    _require(git_before["head"] == evaluation.EXPECTED_STARTING_HEAD, "diagnostic starting HEAD changed", blocked=True)
    return {
        "evaluation_preflight": evaluation_preflight,
        "published": published,
        "git_before": git_before,
        "protected_value_load_count_before": evaluation_preflight["protected_value_load_count"],
    }


def _load_training_data(repo_root: Path, preflight: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    try:
        roles = evaluation._derive_training_roles(repo_root)
    except (evaluation.BlockedExecution, evaluation.InvalidHarnessError) as exc:
        raise BlockedDiagnosis(str(exc)) from exc
    _require(preflight["protected_value_load_count_before"] == 0, "protected values were loaded during preflight", blocked=True)
    images = roles["images"]
    labels = roles["labels"]
    train_panel = _balanced_panel(
        roles["train_indices"],
        labels,
        per_class=DIAGNOSTIC_PANEL_PER_CLASS,
        seed=TRAIN_PANEL_SEED,
    )
    development_panel = roles["selection_indices"].clone().contiguous()
    calibration_panel = roles["calibration_indices"].clone().contiguous()
    panels = {"train": train_panel, "development": development_panel, "calibration": calibration_panel}
    _require(torch.unique(torch.cat(tuple(panels.values()))).numel() == sum(panel.numel() for panel in panels.values()), "diagnostic panels overlap")
    return {"images": images, "labels": labels, "panels": panels, **roles}


def _load_consumer(repo_root: Path, preflight: Mapping[str, Any], device: torch.device) -> LightGen2DComplexOLSTrainingConsumer:
    training_contract = load_frozen_training_contract(
        repo_root / evaluation.TRAINING_CONTRACT_PATH,
        repo_root=repo_root,
    )
    checkpoint = _mapping(preflight["evaluation_preflight"]["checkpoint"], "selected checkpoint")
    consumer = LightGen2DComplexOLSTrainingConsumer(
        training_contract,
        device=device,
        propagation_grid_size=PRODUCTION_PROPAGATION_GRID,
    )
    consumer.load_trainable_state_dict(_mapping(checkpoint.get("trainable_parameter_state"), "checkpoint trainable state"))
    consumer.runtime.eval()
    consumer._enforce_parameter_firewall()
    _require(all(parameter.requires_grad for parameter in consumer.runtime.parameters()), "diagnostic gradient parameters were unexpectedly frozen")
    return consumer


def _run_role(
    role: str,
    consumer: LightGen2DComplexOLSTrainingConsumer,
    evaluator: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> RoleMeasurement:
    accumulator = RoleAccumulator(indices=indices, labels=labels, role=role)
    first = True
    with torch.inference_mode():
        for start in range(0, indices.numel(), evaluation.BATCH_SIZE):
            stop = min(start + evaluation.BATCH_SIZE, indices.numel())
            source_indices = indices[start:stop]
            image_batch = images.index_select(0, source_indices).to(device)
            label_batch = labels.index_select(0, source_indices)
            result = consumer.runtime.forward_from_image(image_batch)
            evaluation._validate_real_path_result(result, first=first)
            first = False
            _require(result.image_input is not None, "real-path image input is missing")
            original_features, _, _, original_probabilities, original_predictions = evaluation._evaluate_views(evaluator, image_batch)
            del original_features
            real_view = evaluation._evaluator_view(result.raw_intensity)
            real_features, _, _, real_probabilities, real_predictions = evaluation._evaluate_views(evaluator, real_view)
            del real_features
            accumulator.update(
                images=image_batch,
                labels=label_batch,
                raw_cmos=result.raw_intensity,
                target_slm=result.image_input.slm_intensity,
                z=result.z_ols.coefficients,
                original_predictions=original_predictions,
                original_confidence=original_probabilities.max(dim=1).values,
                real_predictions=real_predictions,
                real_confidence=real_probabilities.max(dim=1).values,
                path_proof={
                    "image_adapter": True,
                    "input_relay": True,
                    "trained_encoder": True,
                    "mandatory_modal_projection": result.ols_input_field is not None,
                    "coefficient_space_transport": result.generator_input_field.provenance.parameters.get("transport"),
                    "shared_w_ols": True,
                    "shared_generator": True,
                    "raw_cmos_output": True,
                },
            )
            if stop % 1000 == 0 or stop == indices.numel():
                print(f"diagnostic real path: {stop}/{indices.numel()}", flush=True)
    measurement = accumulator.finalize()
    return measurement


def _run_role_named(
    role: str,
    consumer: LightGen2DComplexOLSTrainingConsumer,
    evaluator: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> RoleMeasurement:
    return _run_role(role, consumer, evaluator, images, labels, indices, device=device)


def _latent_pair_distance(z: torch.Tensor, representation: torch.Tensor) -> dict[str, Any]:
    count = z.shape[0]
    pair_count = min(1024, count // 2)
    generator = torch.Generator(device="cpu").manual_seed(PAIR_DISTANCE_SEED)
    permutation = torch.randperm(count, generator=generator)
    left = permutation[:pair_count]
    right = permutation[pair_count : 2 * pair_count]
    representation_distance = (representation[left] - representation[right]).square().mean(dim=1).sqrt()
    complex_distance = (z[left] - z[right]).abs().square().mean(dim=1).sqrt()
    return {
        "pair_count": pair_count,
        "seed": PAIR_DISTANCE_SEED,
        "gauge_invariant_representation_rms": _summary(representation_distance),
        "complex_coefficient_rms": _summary(complex_distance),
    }


def _effective_rank(eigenvalues: torch.Tensor) -> float:
    positive = eigenvalues.to(torch.float64).clamp_min(0.0)
    numerator = positive.sum().square()
    denominator = positive.square().sum()
    return float(numerator / denominator) if float(denominator) > 0.0 else 0.0


def _class_separation(representation: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    centroids = torch.zeros((evaluation.CLASS_COUNT, representation.shape[1]), dtype=torch.float64)
    within_values: list[torch.Tensor] = []
    counts: list[int] = []
    for digit in range(evaluation.CLASS_COUNT):
        values = representation[labels == digit]
        _require(values.shape[0] >= 2, f"latent class {digit} has too few diagnostic samples")
        centroid = values.mean(dim=0)
        centroids[digit] = centroid
        counts.append(int(values.shape[0]))
        within_values.append((values - centroid).square().mean(dim=1).sqrt())
    centroid_distances = (centroids[:, None, :] - centroids[None, :, :]).square().mean(dim=2).sqrt()
    between = centroid_distances[torch.triu(torch.ones_like(centroid_distances, dtype=torch.bool), diagonal=1)]
    within = torch.cat(within_values)
    within_mean = float(within.mean())
    between_mean = float(between.mean())
    return {
        "class_counts": counts,
        "within_class_distance_to_centroid": _summary(within),
        "between_class_centroid_distance": _summary(between),
        "between_to_within_mean_ratio": between_mean / within_mean if within_mean > 0.0 else None,
        "centroid_distance_matrix": centroid_distances.tolist(),
        "distance_representation": "gauge_invariant full-complex representation, RMS over 200 coordinates",
    }


def _latent_diagnostics(measurements: Sequence[RoleMeasurement]) -> dict[str, Any]:
    z = torch.cat([measurement.z for measurement in measurements]).to(torch.complex128).contiguous()
    labels = torch.cat([measurement.labels for measurement in measurements]).to(torch.long).contiguous()
    representation = gauge_invariant_ols_representation(
        z.to(torch.complex64),
        sigma=PRIOR_SIGMA,
        epsilon=GAUGE_EPSILON,
    ).to(torch.float64)
    norm = z.abs().square().sum(dim=1).sqrt()
    power = z.abs().square().mean(dim=1)
    centered_representation = representation - representation.mean(dim=0, keepdim=True)
    covariance = centered_representation.transpose(0, 1).mm(centered_representation) / float(z.shape[0] - 1)
    covariance_eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    centered_z = z - z.mean(dim=0, keepdim=True)
    complex_covariance = centered_z.conj().transpose(0, 1).mm(centered_z) / float(z.shape[0] - 1)
    complex_covariance_eigenvalues = torch.linalg.eigvalsh(complex_covariance).real.clamp_min(0.0)
    anchor = z.sum(dim=1, keepdim=True) / math.sqrt(float(OLS_PORT_COUNT))
    regularized_factor = anchor.conj() / (anchor.abs().square() + GAUGE_EPSILON**2).sqrt()
    relative = z * regularized_factor
    unit = relative / relative.abs().clamp_min(torch.finfo(torch.float64).tiny)
    phase_concentration = unit.mean(dim=0).abs().clamp(0.0, 1.0)
    per_port_real_variance = z.real.var(dim=0, unbiased=False)
    per_port_imag_variance = z.imag.var(dim=0, unbiased=False)
    per_port_power = z.abs().square().mean(dim=0)
    role_payload = {}
    for measurement in measurements:
        role_z = measurement.z.to(torch.complex128)
        role_payload[measurement.role] = {
            "norm": _summary(role_z.abs().square().sum(dim=1).sqrt()),
            "power": _summary(role_z.abs().square().mean(dim=1)),
            "mean_complex": {
                "real": float(role_z.real.mean()),
                "imag": float(role_z.imag.mean()),
            },
        }
    return {
        "sample_count": int(z.shape[0]),
        "authoritative_latent": "z_real in C^100; no q substitution",
        "norm": _summary(norm),
        "power": _summary(power),
        "per_port_variance": {
            "real": per_port_real_variance.tolist(),
            "imag": per_port_imag_variance.tolist(),
            "power": per_port_power.tolist(),
        },
        "relative_phase": {
            "definition": "phase after the frozen global gauge anchor used by gauge_invariant_ols_representation",
            "per_port_circular_concentration": phase_concentration.tolist(),
            "per_port_circular_variation": (1.0 - phase_concentration).tolist(),
            "concentration_summary": _summary(phase_concentration),
            "variation_summary": _summary(1.0 - phase_concentration),
        },
        "covariance_spectrum": {
            "gauge_invariant_real_200d": covariance_eigenvalues.tolist(),
            "gauge_invariant_effective_rank": _effective_rank(covariance_eigenvalues),
            "raw_complex_100d": complex_covariance_eigenvalues.tolist(),
            "raw_complex_effective_rank": _effective_rank(complex_covariance_eigenvalues),
        },
        "pairwise_sample_distance": _latent_pair_distance(z, representation),
        "class_centroid_separation": _class_separation(representation, labels),
        "by_role": role_payload,
    }


def _same_w_ols_diagnostic(
    consumer: LightGen2DComplexOLSTrainingConsumer,
    images: torch.Tensor,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    batch = images.index_select(0, indices[:GRADIENT_BATCH_SIZE]).to(device)
    with torch.inference_mode():
        image_result = consumer.runtime.forward_from_image(batch)
        direct_result = consumer.runtime.forward_from_ols(image_result.z_ols.coefficients)
    input_delta = (image_result.generator_input_field.tensor - direct_result.generator_input_field.tensor).abs()
    output_delta = (image_result.generator_output_field.tensor - direct_result.generator_output_field.tensor).abs()
    raw_delta = (image_result.raw_intensity - direct_result.raw_intensity).abs()
    _require(torch.isfinite(input_delta).all().item(), "same-W_OLS input comparison is nonfinite")
    _require(torch.isfinite(raw_delta).all().item(), "same-W_OLS raw comparison is nonfinite")
    return {
        "batch_count": int(batch.shape[0]),
        "real_source": image_result.z_ols.source,
        "direct_source": direct_result.z_ols.source,
        "same_writer_operation": image_result.generator_input_field.provenance.operation
        == direct_result.generator_input_field.provenance.operation
        == "lightgen_ols_coefficient_transmission_and_modal_reemission",
        "same_transport": image_result.generator_input_field.provenance.parameters.get("transport")
        == direct_result.generator_input_field.provenance.parameters.get("transport")
        == "T=I_100_coefficient_space",
        "post_ols_generator_input_max_abs_error": float(input_delta.max()),
        "generator_output_max_abs_error": float(output_delta.max()),
        "raw_cmos_max_abs_error": float(raw_delta.max()),
        "post_ols_and_generator_numerically_equivalent": bool(torch.equal(image_result.generator_input_field.tensor, direct_result.generator_input_field.tensor)),
        "raw_cmos_numerically_equivalent": bool(torch.equal(image_result.raw_intensity, direct_result.raw_intensity)),
        "structural_conclusion": "same W_OLS and downstream generator route confirmed on an allowed diagnostic batch",
    }


def _gradient_norm(value: torch.Tensor) -> float:
    return float(value.detach().to(torch.float64).square().sum().sqrt().cpu())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.flatten().to(torch.float64)
    right = right.flatten().to(torch.float64)
    denominator = left.norm() * right.norm()
    if float(denominator) == 0.0:
        return None
    return float(torch.dot(left, right) / denominator)


def _gradient_component(
    value: torch.Tensor,
    parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> dict[str, torch.Tensor]:
    for _, parameter in parameters:
        parameter.grad = None
    value.backward(retain_graph=True)
    gradients: dict[str, torch.Tensor] = {}
    for name, parameter in parameters:
        if parameter.grad is None:
            gradients[name] = torch.zeros_like(parameter).detach().cpu()
        else:
            _require(torch.isfinite(parameter.grad).all().item(), f"diagnostic gradient is nonfinite: {name}")
            gradients[name] = parameter.grad.detach().cpu().clone()
    for _, parameter in parameters:
        parameter.grad = None
    return gradients


def _gradient_record_for_batch(
    consumer: LightGen2DComplexOLSTrainingConsumer,
    images: torch.Tensor,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    parameters = tuple(consumer.runtime.named_parameters())
    batch_indices = indices[:GRADIENT_BATCH_SIZE].contiguous()
    batch = images.index_select(0, batch_indices).to(device)
    breakdown = consumer.forward_and_loss(batch, global_step=GRADIENT_GLOBAL_STEP)
    image_gradients = _gradient_component(BETA_IMAGE * breakdown.image, parameters)
    latent_gradients = _gradient_component(ALPHA_LATENT * breakdown.latent, parameters)
    names = [name for name, _ in parameters]
    image_vector = torch.cat([image_gradients[name].flatten() for name in names]).to(torch.float64)
    latent_vector = torch.cat([latent_gradients[name].flatten() for name in names]).to(torch.float64)
    total_vector = image_vector + latent_vector
    parameter_records: dict[str, Any] = {}
    for name in names:
        parameter_records[name] = {
            "image_weighted_norm": _gradient_norm(image_gradients[name]),
            "latent_weighted_norm": _gradient_norm(latent_gradients[name]),
            "total_weighted_norm": _gradient_norm(image_gradients[name] + latent_gradients[name]),
            "image_gradient_present": bool(image_gradients[name].abs().any()),
            "latent_gradient_present": bool(latent_gradients[name].abs().any()),
            "image_latent_cosine": _cosine(image_gradients[name], latent_gradients[name]),
            "latent_to_image_norm_ratio": (
                _gradient_norm(latent_gradients[name]) / _gradient_norm(image_gradients[name])
                if _gradient_norm(image_gradients[name]) > 0.0
                else None
            ),
        }
    optical_applicable = GAMMA_OPTICAL != 0.0 or breakdown.optical.requires_grad
    record = {
        "global_step": GRADIENT_GLOBAL_STEP,
        "batch_indices_sha256": tensor_sha256(batch_indices),
        "batch_size": int(batch.shape[0]),
        "loss": breakdown.metrics(),
        "weights": {"beta_image": BETA_IMAGE, "alpha_latent": ALPHA_LATENT, "gamma_optical": GAMMA_OPTICAL},
        "parameters": parameter_records,
        "combined": {
            "image_weighted_norm": _gradient_norm(image_vector),
            "latent_weighted_norm": _gradient_norm(latent_vector),
            "total_weighted_norm": _gradient_norm(total_vector),
            "image_latent_cosine": _cosine(image_vector, latent_vector),
            "latent_to_image_norm_ratio": _gradient_norm(latent_vector) / _gradient_norm(image_vector)
            if _gradient_norm(image_vector) > 0.0
            else None,
        },
        "optical": {
            "applicable": optical_applicable,
            "weighted_norm": None if not optical_applicable else "not separately materialized",
            "reason": "frozen contract has gamma_optical=0 and L_optical is identically zero"
            if not optical_applicable
            else "optical component would require a separately defined gradient path",
        },
        "optimizer_created": False,
        "optimizer_step": False,
    }
    del breakdown
    return record


def _gradient_diagnostics(
    consumer: LightGen2DComplexOLSTrainingConsumer,
    images: torch.Tensor,
    panels: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, Any]:
    records = {
        role: _gradient_record_for_batch(consumer, images, indices, device=device)
        for role, indices in panels.items()
    }
    parameter_names = tuple(name for name, _ in consumer.runtime.named_parameters())
    aggregate: dict[str, Any] = {"parameters": {}, "combined": {}}
    for name in parameter_names:
        for key in ("image_weighted_norm", "latent_weighted_norm", "total_weighted_norm", "image_latent_cosine", "latent_to_image_norm_ratio"):
            values = [records[role]["parameters"][name][key] for role in records if records[role]["parameters"][name][key] is not None]
            if not values:
                aggregate.setdefault("parameters", {}).setdefault(name, {})[key] = None
            elif key.endswith("cosine") or key.endswith("ratio"):
                aggregate.setdefault("parameters", {}).setdefault(name, {})[key] = _summary(torch.tensor(values, dtype=torch.float64))
            else:
                aggregate.setdefault("parameters", {}).setdefault(name, {})[key] = _summary(torch.tensor(values, dtype=torch.float64))
    for key in ("image_weighted_norm", "latent_weighted_norm", "total_weighted_norm", "image_latent_cosine", "latent_to_image_norm_ratio"):
        values = [records[role]["combined"][key] for role in records if records[role]["combined"][key] is not None]
        aggregate["combined"][key] = _summary(torch.tensor(values, dtype=torch.float64)) if values else None
    return {
        "component_definition": {
            "g_image": "grad(beta * L_image)",
            "g_latent": "grad(alpha * L_latent)",
            "g_optical": "not applicable under frozen gamma_optical=0, L_optical=0 contract",
            "total_gradient": "g_image + g_latent because frozen gamma_optical=0",
        },
        "by_role": records,
        "aggregate": aggregate,
        "parameter_state_mutation": "none; gradients cleared after each component and no optimizer exists",
    }


def _trajectory_diagnostics(training_report: Mapping[str, Any]) -> dict[str, Any]:
    trace = training_report.get("training_trace")
    _require(isinstance(trace, list) and len(trace) == 100, "training trace is not the frozen 100-step trace", blocked=True)
    result: dict[str, Any] = {"step_count": len(trace), "components": {}, "gradient_and_update_summaries": {}}
    for component in ("L_image", "L_latent", "L_optical", "L_total"):
        values = torch.tensor([float(_mapping(item.get("loss"), "training trace loss")[component]) for item in trace], dtype=torch.float64)
        differences = values[1:] - values[:-1]
        minimum_index = int(values.argmin())
        result["components"][component] = {
            "initial": float(values[0]),
            "final": float(values[-1]),
            "delta_final_minus_initial": float(values[-1] - values[0]),
            "minimum": float(values.min()),
            "minimum_global_step": int(trace[minimum_index]["global_step"]),
            "maximum": float(values.max()),
            "monotonic_nonincreasing": bool(torch.all(differences <= 0.0)),
            "decreasing_transition_count": int((differences < 0.0).sum()),
            "increasing_transition_count": int((differences > 0.0).sum()),
            "last_transition": float(differences[-1]),
        }
    for parameter_name in ("encoder.phase", "generator.phase"):
        gradient_values = torch.tensor(
            [float(_mapping(_mapping(item.get("autograd"), "training autograd").get("gradient_norms"), "gradient norms")[parameter_name]) for item in trace],
            dtype=torch.float64,
        )
        update_values = torch.tensor(
            [float(_mapping(item.get("parameter_update_norms"), "parameter update norms")[parameter_name]) for item in trace],
            dtype=torch.float64,
        )
        result["gradient_and_update_summaries"][parameter_name] = {
            "gradient_norm": _summary(gradient_values),
            "update_norm": _summary(update_values),
            "final_gradient_norm": float(gradient_values[-1]),
            "final_update_norm": float(update_values[-1]),
        }
    development = training_report.get("loss", {}).get("checkpoint_development", [])
    _require(isinstance(development, list) and len(development) == 2, "training checkpoint development records changed", blocked=True)
    by_step = {int(item["checkpoint_step"]): item for item in development}
    _require(set(by_step) == {50, 100}, "training checkpoint development steps changed", blocked=True)
    result["checkpoint_objective"] = {
        "selected_checkpoint_step": int(training_report["selected_checkpoint"]["completed_steps"]),
        "selected_by_minimum_recorded_development_L_total": training_report["selected_checkpoint"].get("selected") is True,
        "development_step_50": {key: by_step[50][key] for key in ("L_image", "L_latent", "L_optical", "L_total")},
        "development_step_100": {key: by_step[100][key] for key in ("L_image", "L_latent", "L_optical", "L_total")},
        "step_100_minus_step_50": {
            key: float(by_step[100][key] - by_step[50][key]) for key in ("L_image", "L_latent", "L_optical", "L_total")
        },
        "image_loss_improved_with_selected_step": bool(by_step[100]["L_image"] < by_step[50]["L_image"]),
        "total_loss_improved_with_selected_step": bool(by_step[100]["L_total"] < by_step[50]["L_total"]),
        "only_two_recorded_checkpoints": True,
    }
    initial = result["components"]
    image_delta = abs(float(initial["L_image"]["delta_final_minus_initial"]))
    latent_delta = abs(float(initial["L_latent"]["delta_final_minus_initial"]))
    result["weighted_contributions"] = {
        "weights": {"beta_image": BETA_IMAGE, "alpha_latent": ALPHA_LATENT, "gamma_optical": GAMMA_OPTICAL},
        "initial": {
            "weighted_image": BETA_IMAGE * float(training_report["loss"]["initial"]["L_image"]),
            "weighted_latent": ALPHA_LATENT * float(training_report["loss"]["initial"]["L_latent"]),
            "weighted_optical": GAMMA_OPTICAL * float(training_report["loss"]["initial"]["L_optical"]),
        },
        "final": {
            "weighted_image": BETA_IMAGE * float(training_report["loss"]["final"]["L_image"]),
            "weighted_latent": ALPHA_LATENT * float(training_report["loss"]["final"]["L_latent"]),
            "weighted_optical": GAMMA_OPTICAL * float(training_report["loss"]["final"]["L_optical"]),
        },
        "endpoint_absolute_improvement": {"image": image_delta, "latent": latent_delta, "optical": 0.0},
        "endpoint_largest_improvement": "image" if image_delta > latent_delta else "latent" if latent_delta > image_delta else "tie",
    }
    result["interpretation"] = {
        "image_loss_endpoint_decreased": result["components"]["L_image"]["delta_final_minus_initial"] < 0.0,
        "image_loss_still_monotonic_at_step_100": result["components"]["L_image"]["last_transition"] <= 0.0,
        "nonzero_final_updates": all(
            result["gradient_and_update_summaries"][name]["final_update_norm"] > 0.0
            for name in ("encoder.phase", "generator.phase")
        ),
        "insufficient_progress_is_a_bounded_hypothesis_only": True,
        "checkpoint_objective_mismatch_is_not_identified_by_two_dev_records": True,
    }
    return result


def _prior_semantic_diagnostic(
    calibration_z: torch.Tensor,
    calibration_measurement: RoleMeasurement,
    *,
    device: torch.device,
) -> dict[str, Any]:
    sample_count = int(calibration_z.shape[0])
    seed = derive_prior_seed(
        namespace=PRIOR_NAMESPACE,
        role="diagnostic_prior_mmd",
        global_step=GRADIENT_GLOBAL_STEP,
        sample_count=sample_count,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    real = torch.randn((sample_count, OLS_PORT_COUNT), generator=generator, dtype=torch.float32)
    imag = torch.randn((sample_count, OLS_PORT_COUNT), generator=generator, dtype=torch.float32)
    prior = torch.complex(real, imag) * (PRIOR_SIGMA / math.sqrt(2.0))
    _require(torch.isfinite(prior).all().item(), "diagnostic prior is nonfinite")
    mmd = evaluation._full_complex_mmd(calibration_z, prior, device)
    real_power = float(calibration_z.abs().square().mean())
    prior_power = float(prior.abs().square().mean())
    return {
        "role": calibration_measurement.role,
        "sample_count": sample_count,
        "global_step": GRADIENT_GLOBAL_STEP,
        "prior_namespace": PRIOR_NAMESPACE,
        "prior_role": "diagnostic_prior_mmd",
        "prior_seed": seed,
        "prior_sample_order": "real quadratures then imaginary quadratures; CPU torch.Generator",
        "real_mean_power": real_power,
        "prior_mean_power": prior_power,
        "real_to_prior_mean_power_ratio": real_power / prior_power if prior_power > 0.0 else None,
        "full_complex_gauge_invariant_mmd": float(mmd.detach().cpu()),
        "semantic_recall": calibration_measurement.real_path_evaluator["macro_recall"],
        "reconstruction_mse": calibration_measurement.reconstruction["mse"],
        "interpretation_boundary": "descriptive prior-vs-semantic comparison only; no prior-to-generator samples were evaluated",
    }


def _generalization_diagnostic(
    measurements: Mapping[str, RoleMeasurement],
    published_control: Mapping[str, Any],
) -> dict[str, Any]:
    protected_macro = float(_mapping(published_control.get("control"), "published control").get("protected_macro_recall"))
    threshold = float(evaluation.EVALUATOR_QUALIFICATION_THRESHOLD)
    role_performance = {
        role: {
            "original_macro_recall": measurement.original_evaluator["macro_recall"],
            "real_path_macro_recall": measurement.real_path_evaluator["macro_recall"],
            "real_path_accuracy": measurement.real_path_evaluator["accuracy"],
        }
        for role, measurement in measurements.items()
    }
    output_values = [entry["real_path_macro_recall"] for entry in role_performance.values()]
    original_values = [entry["original_macro_recall"] for entry in role_performance.values()]
    if all(value >= threshold for value in original_values) and all(value < threshold for value in output_values):
        interpretation = "train/dev/calibration real-path performance is below the frozen evaluator qualification reference; protected shift is not the primary supported explanation"
        status = "not_supported"
    elif all(value >= threshold for value in output_values) and protected_macro < min(output_values):
        interpretation = "permitted real-path roles are strong while the published protected aggregate is poor; domain/generalization shift is supported, but no protected rerun is allowed"
        status = "supported"
    else:
        interpretation = "permitted-role and protected aggregate comparison is mixed; no protected rerun is allowed"
        status = "unresolved"
    return {
        "diagnostic_roles": role_performance,
        "published_protected_macro_recall": protected_macro,
        "frozen_evaluator_qualification_reference": threshold,
        "status": status,
        "interpretation": interpretation,
        "protected_rerun": False,
    }


def _mechanism_attribution(
    measurements: Mapping[str, RoleMeasurement],
    latent: Mapping[str, Any],
    same_route: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    gradients: Mapping[str, Any],
    generalization: Mapping[str, Any],
) -> dict[str, Any]:
    original_good = all(
        measurement.original_evaluator["macro_recall"] >= evaluation.EVALUATOR_QUALIFICATION_THRESHOLD
        for measurement in measurements.values()
    )
    real_path_poor = all(
        measurement.real_path_evaluator["macro_recall"] < evaluation.EVALUATOR_QUALIFICATION_THRESHOLD
        for measurement in measurements.values()
    )
    power_ratio = float(latent["power_vs_prior"]["real_to_prior_mean_power_ratio"])
    separation_ratio = latent["class_centroid_separation"].get("between_to_within_mean_ratio")
    latent_supported = real_path_poor and power_ratio < POWER_COLLAPSE_RATIO and separation_ratio is not None and separation_ratio < 1.0
    image_output_nonconstant = all(
        measurement.reconstruction["per_image_output_variance"]["mean"] > NUMERICAL_COLLAPSE_TOLERANCE
        and measurement.reconstruction["inter_sample_output_distance"]["summary"]["mean"] > NUMERICAL_COLLAPSE_TOLERANCE
        for measurement in measurements.values()
    )
    gradient_combined = _mapping(_mapping(gradients.get("aggregate"), "gradient aggregate").get("combined"), "combined gradient aggregate")
    gradient_cosine = gradient_combined.get("image_latent_cosine", {}).get("mean") if isinstance(gradient_combined.get("image_latent_cosine"), Mapping) else None
    gradient_ratio = gradient_combined.get("latent_to_image_norm_ratio", {}).get("mean") if isinstance(gradient_combined.get("latent_to_image_norm_ratio"), Mapping) else None
    regularization_conflict = gradient_cosine is not None and gradient_ratio is not None and gradient_cosine < 0.0 and gradient_ratio > 1.0
    image_endpoint_decreased = bool(trajectory["interpretation"]["image_loss_endpoint_decreased"])
    nonzero_updates = bool(trajectory["interpretation"]["nonzero_final_updates"])
    mechanisms: dict[str, Any] = {
        "A_evaluator_domain_interface_mismatch": {
            "status": "not_supported" if original_good else "unresolved",
            "classification_if_primary": PRIMARY_CLASSIFICATIONS["A"],
            "evidence": {
                "all_original_roles_meet_qualification_reference": original_good,
                "role_original_macro_recall": {role: value.original_evaluator["macro_recall"] for role, value in measurements.items()},
                "role_real_path_macro_recall": {role: value.real_path_evaluator["macro_recall"] for role, value in measurements.items()},
            },
            "reason": "the same frozen evaluator performs well on original images but poorly after the optical path" if original_good else "original-image evaluator behavior is not uniformly qualified on permitted roles",
        },
        "B_image_reconstruction_collapse": {
            "status": "not_supported" if image_output_nonconstant else "unresolved",
            "classification_if_primary": PRIMARY_CLASSIFICATIONS["B"],
            "evidence": {"numerical_output_nonconstant": image_output_nonconstant},
            "reason": "raw outputs retain nonzero spatial and inter-sample dispersion; this excludes exact numerical collapse but not general reconstruction degradation"
            if image_output_nonconstant
            else "numerical collapse cannot be excluded by the bounded statistics",
        },
        "C_encoder_complex_OLS_semantic_collapse": {
            "status": "supported" if latent_supported else "unresolved",
            "classification_if_primary": PRIMARY_CLASSIFICATIONS["C"],
            "evidence": {
                "real_to_prior_mean_power_ratio": power_ratio,
                "power_ratio_diagnostic_rule": f"< {POWER_COLLAPSE_RATIO}",
                "between_to_within_mean_ratio": separation_ratio,
                "semantic_failure_on_permitted_roles": real_path_poor,
            },
            "reason": "real z power is orders below the frozen prior and class-centroid separation is below within-class spread while all permitted real paths fail"
            if latent_supported
            else "the bounded latent evidence does not satisfy the registered collapse pattern",
        },
        "D_generator_side_decode_failure": {
            "status": "unresolved",
            "classification_if_primary": PRIMARY_CLASSIFICATIONS["D"],
            "evidence": {"same_W_OLS_equivalence": same_route["raw_cmos_numerically_equivalent"]},
            "reason": "same-route equivalence excludes route divergence, but no alternate latent/generator probe is allowed to isolate decoder capacity",
        },
        "E_latent_vs_image_loss_conflict": {
            "status": "supported" if regularization_conflict else "not_supported" if gradient_cosine is not None and gradient_cosine >= 0.0 else "unresolved",
            "classification_if_primary": PRIMARY_CLASSIFICATIONS["E"],
            "evidence": {"encoder_image_latent_cosine_mean": gradient_cosine, "encoder_latent_to_image_norm_ratio_mean": gradient_ratio},
            "reason": "weighted encoder gradients oppose and latent magnitude exceeds image magnitude" if regularization_conflict else "no opposing weighted encoder-gradient relation was established" if gradient_cosine is not None and gradient_cosine >= 0.0 else "gradient relation was not fully resolved",
        },
        "F_optical_vs_image_loss_conflict": {
            "status": "not_supported" if GAMMA_OPTICAL == 0.0 else "unresolved",
            "classification_if_primary": PRIMARY_CLASSIFICATIONS["F"],
            "evidence": {"gamma_optical": GAMMA_OPTICAL, "L_optical": 0.0, "separate_optical_gradient": False},
            "reason": "the frozen training contract omits optical loss, so it cannot account for this run's failure as an active competing objective",
        },
        "G_checkpoint_selection_objective_mismatch": {
            "status": "unresolved",
            "classification_if_primary": PRIMARY_CLASSIFICATIONS["G"],
            "evidence": trajectory["checkpoint_objective"],
            "reason": "step 100 improved both recorded development total and image losses, but only two development checkpoints exist and protected comparison cannot be repeated",
        },
        "H_insufficient_bounded_training_progress": {
            "status": "unresolved" if image_endpoint_decreased and nonzero_updates else "not_supported",
            "classification_if_primary": PRIMARY_CLASSIFICATIONS["H"],
            "evidence": {
                "image_loss_endpoint_decreased": image_endpoint_decreased,
                "nonzero_final_updates": nonzero_updates,
                "final_gradient_norms": trajectory["gradient_and_update_summaries"],
            },
            "reason": "continued loss/update signal is compatible with insufficient bounded progress, but no extension/retraining is allowed to establish causality"
            if image_endpoint_decreased and nonzero_updates
            else "the frozen trajectory does not show a remaining update signal",
        },
        "I_train_dev_protected_generalization_shift": {
            "status": generalization["status"],
            "classification_if_primary": PRIMARY_CLASSIFICATIONS["I"],
            "evidence": generalization,
            "reason": generalization["interpretation"],
        },
        "J_mixed_or_unresolved": {
            "status": "mixed" if latent_supported and regularization_conflict else "unresolved",
            "reason": "multiple supported mechanisms would require a mixed primary classification" if latent_supported and regularization_conflict else "the bounded evidence has one supported latent mechanism and several unresolved alternatives",
        },
    }
    return mechanisms


def _primary_classification(mechanisms: Mapping[str, Any]) -> str:
    supported = [key for key, value in mechanisms.items() if key != "J_mixed_or_unresolved" and value.get("status") == "supported"]
    if len(supported) > 1:
        return MIXED_CLASSIFICATION
    if len(supported) == 1:
        return mechanisms[supported[0]]["classification_if_primary"]
    return UNRESOLVED_CLASSIFICATION


def _build_lineage(
    repo_root: Path,
    *,
    created_at: str,
    git_before: Mapping[str, Any],
    git_after: Mapping[str, Any],
    preflight: Mapping[str, Any],
    published: Mapping[str, Any],
    panel_info: Mapping[str, Any],
    result: Mapping[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    return {
        "schema": LINEAGE_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "valid_completed",
        "result": "VALID_COMPLETED",
        "classification": result["primary_diagnosis"],
        "created_at_utc": created_at,
        "authorities": {
            "checkpoint": _ref(repo_root, evaluation.CHECKPOINT_PATH, "selected step-100 checkpoint", evaluation.CHECKPOINT_SHA256),
            "training_report": _ref(repo_root, evaluation.TRAINING_REPORT_PATH, "training report", evaluation.TRAINING_REPORT_SHA256),
            "training_contract": _ref(repo_root, evaluation.TRAINING_CONTRACT_PATH, "training contract", evaluation.TRAINING_CONTRACT_SHA256),
            "aligned_runtime": _ref(repo_root, evaluation.ALIGNMENT_RUNTIME_PATH, "aligned runtime", evaluation.ALIGNMENT_RUNTIME_SHA256),
            "training_consumer": _ref(repo_root, evaluation.TRAINING_CONSUMER_PATH, "training consumer", evaluation.TRAINING_CONSUMER_SHA256),
            "published_post_training_evaluation": published["report_ref"],
            "published_post_training_lineage": published["lineage_ref"],
            "published_stage_1_control": published["control_ref"],
            "semantic_evaluator": _ref(repo_root, evaluation.EVALUATOR_PATH, "frozen semantic evaluator", evaluation.EVALUATOR_SHA256),
        },
        "data_firewall": {
            "training_source": {
                "path": _relative(repo_root, repo_root / evaluation.TRAINING_SOURCE_PATH),
                "sha256": evaluation.TRAINING_SOURCE_SHA256,
                "value_use": "train/development/calibration diagnostic roles only",
            },
            "diagnostic_panels": panel_info,
            "protected_source": {
                "path": _relative(repo_root, repo_root / evaluation.PROTECTED_TEST_PATH),
                "sha256": evaluation.PROTECTED_TEST_SHA256,
                "value_load_count_during_diagnosis": 0,
                "new_forward_or_evaluation": False,
                "published_aggregate_read_only": True,
            },
            "test_values_loaded": False,
            "sealed_values_loaded": False,
            "sealed_values_reused": False,
            "roles_moved": False,
        },
        "execution": {
            "device": "cuda:0",
            "gpu": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "propagation_grid_size": PRODUCTION_PROPAGATION_GRID,
            "batch_size": evaluation.BATCH_SIZE,
            "diagnostic_gradient_batch_size": GRADIENT_BATCH_SIZE,
            "autocast": False,
            "optimizer_created": False,
            "optimizer_step": False,
            "backward_called_for_diagnosis": True,
            "parameter_values_updated": False,
            "checkpoint_search": False,
            "threshold_tuning": False,
            "evaluator_modified": False,
            "prior_generation_forward": False,
            "protected_forward": False,
        },
        "same_route": result["runtime"],
        "git": {
            "branch": git_before["branch"],
            "head_before": git_before["head"],
            "head_after": git_after["head"],
            "worktree_before": dict(git_before),
            "worktree_after": dict(git_after),
            "commit_status": "not_committed",
        },
        "published_evaluation_control": {
            "classification": preflight["published"]["report"]["classification"],
            "protected_macro_recall": preflight["published"]["report"]["control"]["protected_macro_recall"],
            "threshold": preflight["published"]["report"]["control"]["threshold"],
        },
        "report": {"path": _relative(repo_root, report_path), "sha256": "not_embedded_to_avoid_cycle"},
        "next_action": "STOP_AFTER_DIAGNOSIS",
    }


def execute(repo_root: Path, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve() if output_dir.is_absolute() else (repo_root / output_dir).resolve()
    _require(not output_dir.exists(), f"refusing to reuse write-once diagnosis output: {output_dir}", blocked=True)
    preflight = _preflight(repo_root)
    published = preflight["published"]
    device = torch.device("cuda:0")
    data = _load_training_data(repo_root, preflight)
    consumer = _load_consumer(repo_root, preflight, device)
    evaluator, evaluator_payload = evaluation._load_evaluator_authority(repo_root, device)
    _require(float(evaluator_payload["validation_accuracy"]) >= evaluation.EVALUATOR_QUALIFICATION_THRESHOLD, "frozen evaluator qualification changed", blocked=True)
    measurements: dict[str, RoleMeasurement] = {}
    for role, indices in data["panels"].items():
        measurements[role] = _run_role_named(
            role,
            consumer,
            evaluator,
            data["images"],
            data["labels"],
            indices,
            device=device,
        )
    latent_base = _latent_diagnostics(tuple(measurements.values()))
    calibration_measurement = measurements["calibration"]
    prior_semantic = _prior_semantic_diagnostic(calibration_measurement.z, calibration_measurement, device=device)
    latent = {
        **latent_base,
        "power_vs_prior": {
            "real_mean_power": prior_semantic["real_mean_power"],
            "prior_mean_power": prior_semantic["prior_mean_power"],
            "real_to_prior_mean_power_ratio": prior_semantic["real_to_prior_mean_power_ratio"],
        },
    }
    same_route = _same_w_ols_diagnostic(
        consumer,
        data["images"],
        data["panels"]["train"],
        device=device,
    )
    trajectory = _trajectory_diagnostics(json_load_training_report(repo_root))
    # Propagation caches populated by inference_mode may contain inference
    # tensors that autograd cannot save.  A fresh consumer keeps the exact
    # checkpoint/runtime identity while making the diagnostic backward path
    # valid, without creating an optimizer or mutating the evaluated model.
    clear_production_transfer_cache()
    gradient_consumer = _load_consumer(repo_root, preflight, device)
    gradient_parameter_state_before = {
        name: value.detach().cpu().clone() for name, value in gradient_consumer.runtime.state_dict().items()
    }
    gradients = _gradient_diagnostics(gradient_consumer, data["images"], data["panels"], device=device)
    parameter_state_after = {name: value.detach().cpu() for name, value in gradient_consumer.runtime.state_dict().items()}
    _require(
        all(torch.equal(gradient_parameter_state_before[name], parameter_state_after[name]) for name in gradient_parameter_state_before),
        "diagnostic changed a checkpoint parameter",
    )
    generalization = _generalization_diagnostic(measurements, published["report"])
    mechanisms = _mechanism_attribution(measurements, latent, same_route, trajectory, gradients, generalization)
    primary = _primary_classification(mechanisms)
    secondary = {
        key: {"status": value["status"], "reason": value["reason"]}
        for key, value in mechanisms.items()
        if key != "J_mixed_or_unresolved"
    }
    git_before = preflight["git_before"]
    git_after = _git_state(repo_root)
    _require(git_after["branch"] == git_before["branch"] and git_after["head"] == git_before["head"], "branch or HEAD changed during diagnosis", blocked=True)
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    panel_info = {
        role: {
            "count": int(indices.numel()),
            "per_class": int(torch.bincount(data["labels"].index_select(0, indices), minlength=evaluation.CLASS_COUNT).min()),
            "indices_sha256": tensor_sha256(indices),
            "source_role": "train" if role == "train" else "development_checkpoint_selection" if role == "development" else "development_calibration",
            "selection": "fixed deterministic balanced diagnostic panel; no scientific acceptance threshold",
        }
        for role, indices in data["panels"].items()
    }
    report = {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "valid_completed",
        "result": "VALID_COMPLETED",
        "primary_diagnosis": primary,
        "created_at_utc": created_at,
        "scope": {
            "diagnosis_only": True,
            "protected_forward_or_rerun": False,
            "test_values_loaded": False,
            "sealed_values_loaded_or_reused": False,
            "new_prior_to_generator_samples": False,
            "training_or_optimizer_update": False,
        },
        "authorities": {
            "checkpoint": _ref(repo_root, evaluation.CHECKPOINT_PATH, "selected step-100 checkpoint", evaluation.CHECKPOINT_SHA256),
            "training_report": _ref(repo_root, evaluation.TRAINING_REPORT_PATH, "training report", evaluation.TRAINING_REPORT_SHA256),
            "training_contract": _ref(repo_root, evaluation.TRAINING_CONTRACT_PATH, "training contract", evaluation.TRAINING_CONTRACT_SHA256),
            "aligned_runtime": _ref(repo_root, evaluation.ALIGNMENT_RUNTIME_PATH, "aligned runtime", evaluation.ALIGNMENT_RUNTIME_SHA256),
            "training_consumer": _ref(repo_root, evaluation.TRAINING_CONSUMER_PATH, "training consumer", evaluation.TRAINING_CONSUMER_SHA256),
            "published_post_training_evaluation": published["report_ref"],
            "published_post_training_lineage": published["lineage_ref"],
            "published_stage_1_control": published["control_ref"],
            "semantic_evaluator": _ref(repo_root, evaluation.EVALUATOR_PATH, "frozen semantic evaluator", evaluation.EVALUATOR_SHA256),
            "diagnosis_consumer": {"path": _relative(repo_root, Path(__file__)), "sha256": file_sha256(Path(__file__))},
        },
        "evaluator": {
            "validation_accuracy": float(evaluator_payload["validation_accuracy"]),
            "qualification_threshold": evaluation.EVALUATOR_QUALIFICATION_THRESHOLD,
            "preprocessing": "original image directly; raw CMOS clamp[0,1] then adaptive_avg_pool2d(28,28)",
            "by_role": {role: {"original": measurement.original_evaluator, "real_path": measurement.real_path_evaluator} for role, measurement in measurements.items()},
            "domain_statistics_by_role": {
                role: {"original": measurement.original_domain, "raw_cmos": measurement.raw_domain}
                for role, measurement in measurements.items()
            },
        },
        "reconstruction": {"by_role": {role: measurement.reconstruction for role, measurement in measurements.items()}},
        "complex_ols": latent,
        "runtime": same_route,
        "training_trajectory": trajectory,
        "gradient_attribution": gradients,
        "prior_vs_semantic": prior_semantic,
        "generalization": generalization,
        "mechanism_attribution": mechanisms,
        "secondary_findings": secondary,
        "diagnostic_panels": panel_info,
        "boundaries": {
            "training": False,
            "optimizer_created": False,
            "optimizer_step": False,
            "parameter_values_updated": False,
            "checkpoint_search": False,
            "evaluator_modified": False,
            "threshold_tuned": False,
            "prior_generation_forward": False,
            "protected_forward": False,
            "test_values_loaded": False,
            "sealed_values_loaded": False,
            "sealed_values_reused": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
        "claim_boundary": {
            "fact": "the selected step-100 checkpoint failed the published protected real-path semantic control",
            "diagnosis_scope": "bounded descriptive attribution on non-protected roles plus read-only published protected aggregate",
            "model_assumptions": [
                "fixed diagnostic panels are representative only for this bounded diagnosis, not a protected claim",
                "the explicit power/separation pattern is a diagnostic interpretation rule, not a new scientific acceptance threshold",
            ],
            "not_concluded": [
                "LightGen-aligned OLS architecture is invalid",
                "full-complex Gaussian prior is invalid",
                "gauge-invariant MMD is inherently wrong",
                "100-port single-mode OLS cannot carry semantics",
                "semantic generation is impossible",
            ],
        },
        "provenance": {
            "branch": git_before["branch"],
            "head": git_before["head"],
            "protected_value_load_count_during_diagnosis": 0,
            "published_protected_control_read_only": True,
            "panel_indices": panel_info,
        },
        "next_action": "STOP_AFTER_DIAGNOSIS",
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        report_path = temporary / REPORT_NAME
        lineage_path = temporary / LINEAGE_NAME
        lineage = _build_lineage(
            repo_root,
            created_at=created_at,
            git_before=git_before,
            git_after=git_after,
            preflight=preflight,
            published=published,
            panel_info=panel_info,
            result={"primary_diagnosis": primary, "runtime": same_route},
            report_path=output_dir / REPORT_NAME,
        )
        atomic_write_new_json(lineage_path, lineage, artifact_kind="LightGen real-path diagnosis lineage")
        atomic_write_new_json(report_path, report, artifact_kind="LightGen real-path diagnosis report")
        actual = {path.name for path in temporary.iterdir() if path.is_file()}
        _require(actual == {REPORT_NAME, LINEAGE_NAME}, f"diagnosis output tree is not exact: {sorted(actual)}")
        _require(not output_dir.exists(), "diagnosis output appeared during execution", blocked=True)
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    report_sha = file_sha256(output_dir / REPORT_NAME)
    lineage_sha = file_sha256(output_dir / LINEAGE_NAME)
    return {
        "result": "VALID_COMPLETED",
        "primary_diagnosis": primary,
        "output_directory": _relative(repo_root, output_dir),
        "report": _relative(repo_root, output_dir / REPORT_NAME),
        "report_sha256": report_sha,
        "lineage": _relative(repo_root, output_dir / LINEAGE_NAME),
        "lineage_sha256": lineage_sha,
        "protected_value_load_count": 0,
        "training_steps": 0,
        "optimizer_created": False,
        "optimizer_step": False,
        "parameter_values_updated": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "next_action": "STOP_AFTER_DIAGNOSIS",
    }


def json_load_training_report(repo_root: Path) -> dict[str, Any]:
    """Load the already-bound training report without selecting a new artifact."""

    return _read_json(repo_root, evaluation.TRAINING_REPORT_PATH, "training report", evaluation.TRAINING_REPORT_SHA256)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    try:
        result = execute(args.repo_root, args.output_dir)
    except BlockedDiagnosis as exc:
        print(json.dumps({"result": "BLOCKED", "reason": str(exc)}, sort_keys=True))
        return 3
    except InvalidDiagnosis as exc:
        print(json.dumps({"result": "INVALID", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
