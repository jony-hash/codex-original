#!/usr/bin/env python3
"""v61 held-out linear decodability upper bound for the v60 physical student.

This is deliberately not a new student-training stage.  It freezes the v60
student, forwards the readable train/validation/test splits through the
unchanged formal optical path, and fits *offline* linear ridge probes.  The
intensity probes use detector intensities; the complex probes are an explicit
software oracle over the recorded real/imaginary field.  Neither is evidence
that a hardware camera has a corresponding electronic decoder.
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ldm_project import (
    OneShotOpticalGenerator,
    load_one_shot_checkpoint,
    load_training_config,
    validate_physical_forward_contract,
)
from ldm_project.optics.fields import validate_optical_field_chain
from ldm_project.optics.propagation import normalize_real_intensity


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = (
    REPO_ROOT
    / "outputs/20260514_unversioned/runs/"
    "20260610_151747_768931_train_teacher_distill_15k_acceptance_matched_"
    "diffusion_e40_gs40_readable_split_from_full15k_cuda/artifacts"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "outputs/20260715_v60/runs/"
    "20260715_010531_532021_train_v60_coherent_relay_full_fov_from_scratch/"
    "checkpoints/one_shot_distill_best.pt"
)
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "configs/train_v56_coherent_relay_from_scratch.yaml"
DEFAULT_REFERENCE_REPORT = REPO_ROOT / "outputs/reports/v58_3_b0_relay_cuda_preflight.json"
DEFAULT_DATASETS = {
    "train": DATASET_ROOT / "teacher_distill_dataset_readable_train.pt",
    "val": DATASET_ROOT / "teacher_distill_dataset_readable_val.pt",
    "test": DATASET_ROOT / "teacher_distill_dataset_readable_test.pt",
}
DEFAULT_OUTPUT = REPO_ROOT / "outputs/reports/v61_offline_decodability_upper_bound.json"

SPLIT_COUNTS = {"train": 800, "val": 100, "test": 100}
SPLIT_CLASS_COUNTS = {"train": 80, "val": 10, "test": 10}
MODALITIES = (
    "chip_intensity",
    "chip_complex_oracle",
    "camera_intensity",
    "camera_complex_oracle",
)
RIDGE_PENALTIES = (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0)
FOREIGN_OPERATION_NAMES = {
    "generic_resize",
    "sample_center_aperture",
    "physical_aperture_sampling",
}
REQUIRED_V58_3_SECTIONS = {
    "formal_full_fov_validation",
    "independent_analytic_validation",
    "measured_pupil_psf_validation",
    "propagation_component_diagnostic",
    "sampling_and_independence_audit",
}


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _invalid_report(*, inputs: Mapping[str, Path], reasons: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "v61-offline-linear-decodability-upper-bound",
        "status": "harness-invalid",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "inputs": {name: str(path) for name, path in inputs.items()},
        "reasons": reasons,
    }


def _load_tensor_bundle(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"expected dictionary tensor bundle: {path}")
    return payload


def _validate_readable_split(payload: Mapping[str, Any], split: str) -> list[str]:
    reasons: list[str] = []
    codes = payload.get("codes")
    labels = payload.get("class_labels")
    target = payload.get("teacher_camera")
    expected_count = SPLIT_COUNTS[split]
    expected_per_class = SPLIT_CLASS_COUNTS[split]
    if not torch.is_tensor(codes) or tuple(codes.shape) != (expected_count, 128):
        reasons.append(
            f"{split}: codes must have shape [{expected_count},128], got "
            f"{tuple(codes.shape) if torch.is_tensor(codes) else None}"
        )
    if not torch.is_tensor(labels) or tuple(labels.shape) != (expected_count,):
        reasons.append(
            f"{split}: class_labels must have shape [{expected_count}], got "
            f"{tuple(labels.shape) if torch.is_tensor(labels) else None}"
        )
    if not torch.is_tensor(target) or tuple(target.shape) != (expected_count, 1, 200, 200):
        reasons.append(
            f"{split}: teacher_camera must have shape [{expected_count},1,200,200], got "
            f"{tuple(target.shape) if torch.is_tensor(target) else None}"
        )
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().long()
        counts = {label: int((labels == label).sum()) for label in range(10)}
        if counts != {label: expected_per_class for label in range(10)}:
            reasons.append(f"{split}: expected balanced class counts {expected_per_class}, got {counts}")
    if torch.is_tensor(codes) and not bool(torch.isfinite(codes).all()):
        reasons.append(f"{split}: codes are non-finite")
    if torch.is_tensor(target) and not bool(torch.isfinite(target).all()):
        reasons.append(f"{split}: teacher_camera is non-finite")
    if payload.get("readability_split") != split:
        reasons.append(f"{split}: readability_split is {payload.get('readability_split')!r}")
    config = payload.get("config")
    filter_config = config.get("filter", {}) if isinstance(config, dict) else {}
    if filter_config.get("teacher_filter") != "readability_strict_split":
        reasons.append(f"{split}: dataset was not produced by readability_strict_split")
    nearest = payload.get("label_audit_nearest_label")
    margin = payload.get("label_audit_margin")
    if not torch.is_tensor(nearest) or not torch.is_tensor(labels) or not torch.equal(nearest.long(), labels):
        reasons.append(f"{split}: readable label audit does not match requested labels")
    if not torch.is_tensor(margin) or not bool(torch.isfinite(margin).all()) or bool((margin <= 0).any()):
        reasons.append(f"{split}: label audit margins are missing, non-finite, or non-positive")
    return reasons


def _validate_v58_3_report(path: Path) -> list[str]:
    if not path.is_file():
        return [f"v58.3 report does not exist: {path}"]
    report = json.loads(path.read_text(encoding="utf-8"))
    reasons: list[str] = []
    if report.get("status") != "physical-forward-model-valid":
        reasons.append("v58.3 report is not physical-forward-model-valid")
    if report.get("validation_revision") != "v58.3-full-fov-b0-relay-acceptance":
        reasons.append("v58.3 report has an unexpected validation revision")
    if not REQUIRED_V58_3_SECTIONS.issubset(report):
        reasons.append("v58.3 report is missing required acceptance sections")
    if report.get("diagnostic_only") is not True or report.get("accepted_summary_written") is not False:
        reasons.append("v58.3 report has invalid diagnostic acceptance metadata")
    return reasons


def _validate_contract(config) -> list[str]:
    reasons: list[str] = []
    try:
        validate_physical_forward_contract(config, config.physics)
    except ValueError as exc:
        return [str(exc)]
    one_shot = config.one_shot
    if one_shot.target_code_diagnostic_mode != "none" or one_shot.target_code_bridge_source != "none":
        reasons.append("target-code intervention is enabled")
    if one_shot.ols_transfer_calibration != "none" or one_shot.ols_kind != "identity":
        reasons.append("OLS is not uncalibrated identity")
    if one_shot.decoder_diagnostic_mode != "normal":
        reasons.append("decoder diagnostic mode is not normal")
    if one_shot.decoder_diagnostic_modulation_override != "none":
        reasons.append("decoder modulation override is enabled")
    return reasons


def _sampling_matches(record: Mapping[str, Any], shape: tuple[int, int], pixel: float, fov: float) -> bool:
    sampling = record.get("sampling")
    if not isinstance(sampling, Mapping):
        return False
    return (
        tuple(sampling.get("shape", ())) == shape
        and math.isclose(float(sampling.get("pixel_size_x", 0.0)), pixel, rel_tol=0.0, abs_tol=1.0e-15)
        and math.isclose(float(sampling.get("pixel_size_y", 0.0)), pixel, rel_tol=0.0, abs_tol=1.0e-15)
        and math.isclose(float(sampling.get("fov_x", 0.0)), fov, rel_tol=0.0, abs_tol=1.0e-15)
        and math.isclose(float(sampling.get("fov_y", 0.0)), fov, rel_tol=0.0, abs_tol=1.0e-15)
    )


def _field_from_record(fields: Mapping[str, Mapping[str, Any]], name: str) -> torch.Tensor:
    record = fields.get(name)
    if not isinstance(record, Mapping):
        raise ValueError(f"strict optical field is missing {name!r}")
    real, imag = record.get("real"), record.get("imag")
    if not torch.is_tensor(real) or not torch.is_tensor(imag) or real.shape != imag.shape:
        raise ValueError(f"strict optical field {name!r} lacks matching real/imag")
    if not bool(torch.isfinite(real).all()) or not bool(torch.isfinite(imag).all()):
        raise ValueError(f"strict optical field {name!r} is non-finite")
    return torch.complex(real, imag)


def _validate_forward(
    outputs: Mapping[str, Any],
    *,
    config,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    fields = outputs.get("optical_fields")
    if not isinstance(fields, Mapping):
        raise ValueError("model did not return strict optical fields")
    target_active = bool(outputs["target_code_diagnostic_active"].detach().item())
    bridge_active = bool(outputs["target_code_bridge_active"].detach().item())
    chain = validate_optical_field_chain(
        fields,
        front_path_model=config.physics.front_path_model,
        target_code_active=target_active,
        bridge_active=bridge_active,
    )
    if chain["status"] != "valid":
        raise ValueError(f"strict field chain is {chain['status']}: {chain['reasons'] or chain['errors']}")
    if "lightgen_input" in fields:
        raise ValueError("legacy lightgen_input plane is present")
    for name, record in fields.items():
        if record.get("operation") in FOREIGN_OPERATION_NAMES:
            raise ValueError(f"{name} uses forbidden operation {record.get('operation')!r}")
    if not _sampling_matches(fields["chip"], (100, 100), 2.0e-6, 200.0e-6):
        raise ValueError("chip sampling is not 100x100 at 2 um / 200 um FOV")
    if not _sampling_matches(fields["camera_field"], (200, 200), 2.5e-6, 500.0e-6):
        raise ValueError("camera sampling is not 200x200 at 2.5 um / 500 um FOV")
    if not torch.equal(fields["ols_input"]["real"], fields["ols_output"]["real"]) or not torch.equal(
        fields["ols_input"]["imag"], fields["ols_output"]["imag"]
    ):
        raise ValueError("OLS field is not elementwise identity")
    chip = _field_from_record(fields, "chip")
    camera = _field_from_record(fields, "camera_field")
    raw_intensity = outputs.get("physical_camera_intensity_raw")
    detector_intensity = outputs.get("physical_camera_plane")
    if not torch.is_tensor(raw_intensity) or not torch.is_tensor(detector_intensity):
        raise ValueError("model output lacks raw or detector-normalized camera intensity")
    field_intensity = camera.abs().square()
    if not torch.allclose(raw_intensity, field_intensity, rtol=1.0e-5, atol=1.0e-6):
        difference = float((raw_intensity - field_intensity).abs().amax().detach().cpu())
        raise ValueError(f"raw camera intensity differs from |E_camera|^2 (max error {difference})")
    expected_detector = normalize_real_intensity(
        raw_intensity,
        config.physics.readout.normalize_percentile,
        clamp=False,
    ).clamp(0.0, 1.0)
    if not torch.allclose(detector_intensity, expected_detector, rtol=1.0e-5, atol=1.0e-6):
        difference = float((detector_intensity - expected_detector).abs().amax().detach().cpu())
        raise ValueError(f"physical_camera_plane does not match detector normalization (max error {difference})")
    return {
        "chip_intensity": chip.abs().square(),
        "chip_complex_oracle": torch.cat((chip.real, chip.imag), dim=1),
        "camera_intensity": detector_intensity,
        "camera_complex_oracle": torch.cat((camera.real, camera.imag), dim=1),
    }, {
        "raw_camera_intensity_field_max_error": float((raw_intensity - field_intensity).abs().amax().detach().cpu()),
        "detector_normalization_max_error": float((detector_intensity - expected_detector).abs().amax().detach().cpu()),
    }


def _state_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _state_changes(model: torch.nn.Module, before: Mapping[str, torch.Tensor]) -> list[str]:
    return [
        name
        for name, value in model.state_dict().items()
        if name not in before or not torch.equal(before[name], value.detach().cpu())
    ]


def _extract_split(
    model: torch.nn.Module,
    payload: Mapping[str, Any],
    *,
    config,
    device: torch.device,
    batch_size: int,
    feature_sinks: Mapping[str, np.memmap],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    codes = payload["codes"].detach().cpu().float()
    labels = payload["class_labels"].detach().cpu().long()
    target = payload["teacher_camera"].detach().cpu().float()
    max_errors = {
        "raw_camera_intensity_field_max_error": 0.0,
        "detector_normalization_max_error": 0.0,
    }
    with torch.no_grad():
        for start in range(0, int(codes.shape[0]), batch_size):
            end = min(start + batch_size, int(codes.shape[0]))
            outputs = model(
                codes[start:end].to(device),
                labels[start:end].to(device),
                return_optical_fields=True,
            )
            batch_features, errors = _validate_forward(outputs, config=config)
            for name, values in batch_features.items():
                feature_sinks[name][start:end] = values.detach().flatten(1).cpu().float().numpy()
            for name, value in errors.items():
                max_errors[name] = max(max_errors[name], value)
            del outputs, batch_features
            torch.cuda.empty_cache()
    for sink in feature_sinks.values():
        sink.flush()
    return labels, target, max_errors


def _feature_dimensions() -> dict[str, int]:
    return {
        "chip_intensity": 100 * 100,
        "chip_complex_oracle": 2 * 100 * 100,
        "camera_intensity": 200 * 200,
        "camera_complex_oracle": 2 * 200 * 200,
    }


def _create_feature_memmaps(root: Path) -> dict[str, dict[str, np.memmap]]:
    dimensions = _feature_dimensions()
    return {
        modality: {
            split: np.memmap(
                root / f"{modality}_{split}.f32",
                mode="w+",
                dtype=np.float32,
                shape=(SPLIT_COUNTS[split], dimensions[modality]),
            )
            for split in SPLIT_COUNTS
        }
        for modality in MODALITIES
    }


def _ridge_fit(features: torch.Tensor, targets: torch.Tensor, penalty: float) -> dict[str, torch.Tensor | float]:
    if features.ndim != 2 or targets.ndim != 2 or features.shape[0] != targets.shape[0]:
        raise ValueError("ridge features and targets must be aligned rank-2 tensors")
    feature_mean = features.mean(dim=0)
    centered = features - feature_mean
    feature_scale = centered.square().mean().sqrt().clamp_min(1.0e-8)
    normalized = centered / feature_scale
    feature_count = float(normalized.shape[1])
    kernel = normalized @ normalized.T / feature_count
    target_mean = targets.mean(dim=0)
    target_centered = targets - target_mean
    system = kernel + float(penalty) * torch.eye(kernel.shape[0], device=kernel.device, dtype=kernel.dtype)
    chol = torch.linalg.cholesky(system)
    alpha = torch.cholesky_solve(target_centered, chol)
    return {
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "normalized_train": normalized,
        "target_mean": target_mean,
        "alpha": alpha,
        "feature_count": feature_count,
    }


def _ridge_predict(model: Mapping[str, torch.Tensor | float], features: torch.Tensor) -> torch.Tensor:
    feature_mean = model["feature_mean"]
    feature_scale = model["feature_scale"]
    normalized_train = model["normalized_train"]
    target_mean = model["target_mean"]
    alpha = model["alpha"]
    feature_count = float(model["feature_count"])
    assert all(torch.is_tensor(value) for value in (feature_mean, feature_scale, normalized_train, target_mean, alpha))
    normalized = (features - feature_mean) / feature_scale
    kernel = normalized @ normalized_train.T / feature_count
    return kernel @ alpha + target_mean


def _correlation_per_sample(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    left = prediction.flatten(1)
    right = target.flatten(1)
    left = left - left.mean(dim=1, keepdim=True)
    right = right - right.mean(dim=1, keepdim=True)
    return (left * right).mean(dim=1) / (
        left.square().mean(dim=1).sqrt() * right.square().mean(dim=1).sqrt()
    ).clamp_min(1.0e-8)


def _dice_per_sample(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.2) -> torch.Tensor:
    left = prediction.clamp(0.0, 1.0) > threshold
    right = target.clamp(0.0, 1.0) > threshold
    dims = tuple(range(1, left.ndim))
    intersection = (left & right).float().sum(dim=dims)
    denominator = left.float().sum(dim=dims) + right.float().sum(dim=dims)
    return 2.0 * intersection / denominator.clamp_min(1.0)


def _prototype_scores(samples: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    sample = samples.flatten(1)
    prototype = prototypes.flatten(1)
    sample = sample - sample.mean(dim=1, keepdim=True)
    prototype = prototype - prototype.mean(dim=1, keepdim=True)
    return (sample / sample.norm(dim=1, keepdim=True).clamp_min(1.0e-8)) @ (
        prototype / prototype.norm(dim=1, keepdim=True).clamp_min(1.0e-8)
    ).T


def _teacher_prototypes(target: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    values = []
    for label in range(10):
        selected = target[labels == label]
        if int(selected.shape[0]) == 0:
            raise ValueError(f"teacher prototype split lacks class {label}")
        values.append(selected.mean(dim=0))
    return torch.stack(values)


def _summarize_reconstruction(
    prediction: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
) -> dict[str, Any]:
    raw = prediction.float()
    target = target.float()
    corr = _correlation_per_sample(raw, target)
    dice = _dice_per_sample(raw, target)
    mse = (raw - target).square().flatten(1).mean(dim=1)
    scores = _prototype_scores(raw.clamp(0.0, 1.0), prototypes)
    requested = scores.gather(1, labels[:, None]).squeeze(1)
    predicted = scores.argmax(dim=1)
    other = scores.clone()
    other[torch.arange(labels.shape[0]), labels] = -torch.inf
    margin = requested - other.max(dim=1).values
    per_class: dict[str, Any] = {}
    for label in range(10):
        indices = labels == label
        per_class[str(label)] = {
            "count": int(indices.sum()),
            "mse_mean": float(mse[indices].mean()),
            "corr_mean": float(corr[indices].mean()),
            "dice_mean": float(dice[indices].mean()),
            "prototype_label_accuracy": float((predicted[indices] == labels[indices]).float().mean()),
            "prototype_label_margin_mean": float(margin[indices].mean()),
        }
    return {
        "sample_count": int(labels.shape[0]),
        "raw_mse_mean": float(mse.mean()),
        "target_corr_mean": float(corr.mean()),
        "thresholded_dice_mean": float(dice.mean()),
        "prototype_label_accuracy": float((predicted == labels).float().mean()),
        "prototype_label_margin_mean": float(margin.mean()),
        "per_class": per_class,
    }


def _is_decodable(metrics: Mapping[str, Any]) -> bool:
    return (
        float(metrics["target_corr_mean"]) >= 0.78
        and float(metrics["thresholded_dice_mean"]) >= 0.72
        and float(metrics["prototype_label_accuracy"]) >= 0.70
    )


def _decode_modality(
    feature_splits: Mapping[str, torch.Tensor],
    target_splits: Mapping[str, torch.Tensor],
    labels: Mapping[str, torch.Tensor],
    *,
    compute_device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    train_features = feature_splits["train"].to(compute_device)
    val_features = feature_splits["val"].to(compute_device)
    train_target = target_splits["train"].flatten(1).to(compute_device)
    val_target = target_splits["val"].flatten(1).to(compute_device)
    validation: list[dict[str, float]] = []
    for penalty in RIDGE_PENALTIES:
        model = _ridge_fit(train_features, train_target, penalty)
        prediction = _ridge_predict(model, val_features)
        validation.append({"penalty": penalty, "raw_mse_mean": float((prediction - val_target).square().mean())})
        del model, prediction
    selected = min(validation, key=lambda item: item["raw_mse_mean"])
    all_features = torch.cat((feature_splits["train"], feature_splits["val"]), dim=0).to(compute_device)
    all_targets = torch.cat((target_splits["train"], target_splits["val"]), dim=0).flatten(1).to(compute_device)
    model = _ridge_fit(all_features, all_targets, float(selected["penalty"]))
    test_prediction = _ridge_predict(model, feature_splits["test"].to(compute_device)).reshape(-1, 1, 200, 200)
    test_prediction = test_prediction.detach().cpu().float()
    prototypes = _teacher_prototypes(
        torch.cat((target_splits["train"], target_splits["val"]), dim=0),
        torch.cat((labels["train"], labels["val"]), dim=0),
    )
    test_metrics = _summarize_reconstruction(test_prediction, target_splits["test"], labels["test"], prototypes)
    test_metrics["decodable_at_preregistered_thresholds"] = _is_decodable(test_metrics)
    return {
        "probe": {
            "kind": "held_out_linear_kernel_ridge_dual",
            "feature_standardization": "train-split feature mean and scalar RMS only",
            "target": "teacher_camera at native 200x200 sampling",
            "penalties": list(RIDGE_PENALTIES),
            "selection_split": "val",
            "selection_metric": "raw_mse_mean",
            "validation": validation,
            "selected_penalty": selected["penalty"],
            "test_split": "test",
        },
        "test": test_metrics,
    }, test_prediction


def _read_feature_parts(
    parts: tuple[np.memmap, ...],
    start: int,
    end: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Read a native-sampling feature block without materializing a full split on CUDA."""
    return torch.cat(
        [torch.from_numpy(np.asarray(part[:, start:end])).to(device=device, dtype=torch.float32) for part in parts],
        dim=0,
    )


def _stream_feature_statistics(
    parts: tuple[np.memmap, ...],
    *,
    device: torch.device,
    feature_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact train-split per-feature mean and scalar RMS in column blocks."""
    sample_count = sum(int(part.shape[0]) for part in parts)
    feature_count = int(parts[0].shape[1])
    mean = torch.empty(feature_count, dtype=torch.float32)
    total_squared = torch.zeros((), device=device, dtype=torch.float64)
    for start in range(0, feature_count, feature_block):
        end = min(start + feature_block, feature_count)
        block = _read_feature_parts(parts, start, end, device=device)
        block_mean = block.mean(dim=0)
        mean[start:end] = block_mean.cpu()
        total_squared += (block - block_mean).square().sum(dtype=torch.float64)
        del block, block_mean
    scale = (total_squared / float(sample_count * feature_count)).sqrt().clamp_min(1.0e-8).float().cpu()
    return mean, scale


def _stream_normalized_kernel(
    left: tuple[np.memmap, ...],
    right: tuple[np.memmap, ...],
    *,
    feature_mean: torch.Tensor,
    feature_scale: torch.Tensor,
    device: torch.device,
    feature_block: int,
) -> torch.Tensor:
    """Exact dual ridge kernel, accumulated over native Fourier-plane samples."""
    left_count = sum(int(part.shape[0]) for part in left)
    right_count = sum(int(part.shape[0]) for part in right)
    feature_count = int(left[0].shape[1])
    if feature_count != int(right[0].shape[1]):
        raise ValueError("ridge kernel feature dimensions differ")
    kernel = torch.zeros((left_count, right_count), device=device, dtype=torch.float32)
    for start in range(0, feature_count, feature_block):
        end = min(start + feature_block, feature_count)
        center = feature_mean[start:end].to(device)
        left_block = (_read_feature_parts(left, start, end, device=device) - center) / feature_scale.to(device)
        right_block = (_read_feature_parts(right, start, end, device=device) - center) / feature_scale.to(device)
        kernel.addmm_(left_block, right_block.T, beta=1.0, alpha=1.0 / feature_count)
        del center, left_block, right_block
    return kernel


def _stream_ridge_prediction(
    *,
    fit_parts: tuple[np.memmap, ...],
    predict_parts: tuple[np.memmap, ...],
    fit_targets: torch.Tensor,
    penalty: float,
    device: torch.device,
    feature_block: int,
) -> torch.Tensor:
    feature_mean, feature_scale = _stream_feature_statistics(
        fit_parts,
        device=device,
        feature_block=feature_block,
    )
    train_kernel = _stream_normalized_kernel(
        fit_parts,
        fit_parts,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        device=device,
        feature_block=feature_block,
    )
    target = fit_targets.flatten(1).to(device)
    target_mean = target.mean(dim=0)
    system = train_kernel + float(penalty) * torch.eye(train_kernel.shape[0], device=device)
    alpha = torch.linalg.solve(system, target - target_mean)
    prediction_kernel = _stream_normalized_kernel(
        predict_parts,
        fit_parts,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        device=device,
        feature_block=feature_block,
    )
    prediction = prediction_kernel @ alpha + target_mean
    return prediction.detach().cpu().reshape(-1, 1, 200, 200)


def _decode_native_memmap_modality(
    feature_splits: Mapping[str, np.memmap],
    target_splits: Mapping[str, torch.Tensor],
    labels: Mapping[str, torch.Tensor],
    *,
    compute_device: torch.device,
    feature_block: int = 8_192,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Held-out ridge with disk-backed native fields, not a resized or learned probe."""
    train_part = (feature_splits["train"],)
    val_part = (feature_splits["val"],)
    train_and_val = (feature_splits["train"], feature_splits["val"])
    validation: list[dict[str, float]] = []
    for penalty in RIDGE_PENALTIES:
        prediction = _stream_ridge_prediction(
            fit_parts=train_part,
            predict_parts=val_part,
            fit_targets=target_splits["train"],
            penalty=penalty,
            device=compute_device,
            feature_block=feature_block,
        )
        validation.append(
            {
                "penalty": penalty,
                "raw_mse_mean": float((prediction - target_splits["val"]).square().mean()),
            }
        )
        del prediction
        torch.cuda.empty_cache()
    selected = min(validation, key=lambda item: item["raw_mse_mean"])
    test_prediction = _stream_ridge_prediction(
        fit_parts=train_and_val,
        predict_parts=(feature_splits["test"],),
        fit_targets=torch.cat((target_splits["train"], target_splits["val"]), dim=0),
        penalty=float(selected["penalty"]),
        device=compute_device,
        feature_block=feature_block,
    )
    prototypes = _teacher_prototypes(
        torch.cat((target_splits["train"], target_splits["val"]), dim=0),
        torch.cat((labels["train"], labels["val"]), dim=0),
    )
    test_metrics = _summarize_reconstruction(test_prediction, target_splits["test"], labels["test"], prototypes)
    test_metrics["decodable_at_preregistered_thresholds"] = _is_decodable(test_metrics)
    return {
        "probe": {
            "kind": "held_out_linear_kernel_ridge_dual",
            "feature_standardization": "train-split feature mean and scalar RMS only",
            "feature_storage": "temporary native float32 disk map; deleted after report",
            "target": "teacher_camera at native 200x200 sampling",
            "penalties": list(RIDGE_PENALTIES),
            "selection_split": "val",
            "selection_metric": "raw_mse_mean",
            "validation": validation,
            "selected_penalty": selected["penalty"],
            "test_split": "test",
        },
        "test": test_metrics,
    }, test_prediction


def _decision(modalities: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    decoded = {name: bool(payload["test"]["decodable_at_preregistered_thresholds"]) for name, payload in modalities.items()}
    class_only = any(float(payload["test"]["prototype_label_accuracy"]) >= 0.70 for payload in modalities.values())
    if decoded["camera_intensity"]:
        return {
            "status": "camera-intensity-decodable",
            "next_gate": "explain the discrepancy with true CMOS imagery; do not promote the software probe",
        }
    if decoded["camera_complex_oracle"]:
        return {
            "status": "camera-phase-to-intensity-representation-mismatch",
            "next_gate": "isolate a physically valid phase-to-intensity conversion; do not tune the student",
        }
    if decoded["chip_intensity"]:
        return {
            "status": "backend-relay-intensity-mismatch-candidate",
            "next_gate": "independently validate chip-to-CMOS B=0 intensity mapping",
        }
    if decoded["chip_complex_oracle"]:
        return {
            "status": "backend-relay-information-mismatch-candidate",
            "next_gate": "independently validate chip-to-CMOS B=0 complex-field mapping",
        }
    if class_only:
        return {
            "status": "class-only-information-insufficient-for-image-generation",
            "next_gate": "audit generator/target representation; do not tune decoder or losses",
        }
    return {
        "status": "objective-representation-mismatch-confirmed",
        "next_gate": "audit upstream generator/target representation; training remains stopped",
    }


def _render_preview(
    output: Path,
    *,
    target: torch.Tensor,
    labels: torch.Tensor,
    predictions: Mapping[str, torch.Tensor],
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to render the v61 preview") from exc
    indices = [int(torch.nonzero(labels == label, as_tuple=False)[0]) for label in range(10)]
    columns = (("teacher target", target),) + tuple(
        (f"offline {name}", predictions[name]) for name in MODALITIES
    )
    figure, axes = plt.subplots(10, len(columns), figsize=(13, 24), squeeze=False)
    for row, index in enumerate(indices):
        for column, (title, tensor) in enumerate(columns):
            axes[row, column].imshow(tensor[index, 0].detach().cpu().clamp(0.0, 1.0), cmap="gray", interpolation="nearest")
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(title)
        axes[row, 0].set_ylabel(f"class {int(labels[index])}")
    figure.suptitle("v61 offline linear decodability diagnostic — not physical CMOS output", y=0.999)
    figure.tight_layout()
    path = output.with_name(f"{output.stem}_preview.png")
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def run(
    *,
    train_config: str | Path = DEFAULT_TRAIN_CONFIG,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    reference_report: str | Path = DEFAULT_REFERENCE_REPORT,
    train_dataset: str | Path = DEFAULT_DATASETS["train"],
    val_dataset: str | Path = DEFAULT_DATASETS["val"],
    test_dataset: str | Path = DEFAULT_DATASETS["test"],
    output: str | Path = DEFAULT_OUTPUT,
    device: str = "cuda",
    batch_size: int = 4,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    paths = {
        "train_config": Path(train_config),
        "checkpoint": Path(checkpoint),
        "reference_report": Path(reference_report),
        "train_dataset": Path(train_dataset),
        "val_dataset": Path(val_dataset),
        "test_dataset": Path(test_dataset),
    }
    output_path = Path(output)
    reasons = [f"missing input: {name}={path}" for name, path in paths.items() if not path.is_file()]
    if Path(checkpoint).resolve() != DEFAULT_CHECKPOINT.resolve():
        reasons.append("v61 accepts only the pre-registered v60 epoch-5 checkpoint")
    if device != "cuda":
        reasons.append("v61 full-geometry diagnostic requires --device cuda")
    if not torch.cuda.is_available():
        reasons.append("CUDA is unavailable")
    if reasons:
        report = _invalid_report(inputs=paths, reasons=reasons)
        _json_dump(output_path, report)
        return report
    config = load_training_config(paths["train_config"])
    reasons.extend(_validate_contract(config))
    reasons.extend(_validate_v58_3_report(paths["reference_report"]))
    datasets = {split: _load_tensor_bundle(paths[f"{split}_dataset"]) for split in SPLIT_COUNTS}
    for split, payload in datasets.items():
        reasons.extend(_validate_readable_split(payload, split))
    if reasons:
        report = _invalid_report(inputs=paths, reasons=reasons)
        _json_dump(output_path, report)
        return report
    resolved_device = torch.device("cuda")
    model = OneShotOpticalGenerator(config.physics, config.one_shot, image_size=config.first_stage.image_size).to(resolved_device)
    load_one_shot_checkpoint(model, torch.load(paths["checkpoint"], map_location=resolved_device, weights_only=False))
    model.eval()
    before = _state_snapshot(model)
    report: dict[str, Any] = {
        "schema_version": "v61-offline-linear-decodability-upper-bound",
        "status": "manual-review-required",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "inputs": {name: str(path) for name, path in paths.items()},
        "invariants": {
            "student_training": False,
            "optimizer_created": False,
            "parameter_update_count": 0,
            "target_code_or_bridge": "disabled",
            "temporary_feature_storage": "native float32 memory maps; deleted after diagnostic",
            "native_feature_sampling": {
                "chip": "100x100 at 2 um",
                "camera": "200x200 at 2.5 um",
                "teacher_target": "200x200",
            },
        },
        "camera_intensity_semantics": {
            "raw_source": "physical_camera_intensity_raw = abs(camera_field)^2",
            "probe_feature_source": "physical_camera_plane = detector-normalized raw intensity",
        },
        "progress": {"completed_splits": [], "completed_modalities": []},
        "reasons": [],
    }
    _json_dump(output_path, report)
    label_splits: dict[str, torch.Tensor] = {}
    target_splits: dict[str, torch.Tensor] = {}
    semantic_errors: dict[str, float] = {
        "raw_camera_intensity_field_max_error": 0.0,
        "detector_normalization_max_error": 0.0,
    }
    try:
        with tempfile.TemporaryDirectory(prefix="v61_native_features_") as temp_dir:
            feature_splits = _create_feature_memmaps(Path(temp_dir))
            for split in ("train", "val", "test"):
                labels, target, errors = _extract_split(
                    model,
                    datasets[split],
                    config=config,
                    device=resolved_device,
                    batch_size=batch_size,
                    feature_sinks={name: feature_splits[name][split] for name in MODALITIES},
                )
                label_splits[split] = labels
                target_splits[split] = target
                for name, value in errors.items():
                    semantic_errors[name] = max(semantic_errors[name], value)
                report["progress"]["completed_splits"].append(split)
                _json_dump(output_path, report)
            changed = _state_changes(model, before)
            report["invariants"]["parameter_update_count"] = len(changed)
            if changed:
                report["status"] = "harness-invalid"
                report["reasons"].append(f"model parameters changed during diagnostic: {changed[:5]}")
                _json_dump(output_path, report)
                return report
            report["field_semantic_checks"] = semantic_errors
            # The optical forward is complete.  Release propagation-grid buffers before
            # the software-only ridge fits so the camera-sized oracle cannot compete
            # with the frozen D2NN for CUDA memory.
            model.cpu()
            del model
            torch.cuda.empty_cache()
            modality_reports: dict[str, Any] = {}
            predictions: dict[str, torch.Tensor] = {}
            for name in MODALITIES:
                modality_report, prediction = _decode_native_memmap_modality(
                    feature_splits[name],
                    target_splits,
                    label_splits,
                    compute_device=resolved_device,
                )
                modality_reports[name] = modality_report
                predictions[name] = prediction
                report["progress"]["completed_modalities"].append(name)
                report["modalities"] = modality_reports
                _json_dump(output_path, report)
                torch.cuda.empty_cache()
            report["modalities"] = modality_reports
            report["decision"] = _decision(modality_reports)
            preview = _render_preview(
                output_path,
                target=target_splits["test"],
                labels=label_splits["test"],
                predictions=predictions,
            )
            tensor_path = output_path.with_suffix(".pt")
            torch.save(
                {
                    "schema_version": report["schema_version"],
                    "diagnostic_only": True,
                    "accepted_summary_written": False,
                    "class_labels": label_splits["test"],
                    "teacher_target": target_splits["test"],
                    "decoded_targets": predictions,
                },
                tensor_path,
            )
            report["artifacts"] = {
                "test_predictions": str(tensor_path),
                "preview": str(preview),
            }
            report.pop("progress", None)
    except (RuntimeError, ValueError) as exc:
        report["status"] = "harness-invalid"
        report["reasons"].append(str(exc))
    finally:
        _json_dump(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--reference-report", type=Path, default=DEFAULT_REFERENCE_REPORT)
    parser.add_argument("--train-dataset", type=Path, default=DEFAULT_DATASETS["train"])
    parser.add_argument("--val-dataset", type=Path, default=DEFAULT_DATASETS["val"])
    parser.add_argument("--test-dataset", type=Path, default=DEFAULT_DATASETS["test"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    result = run(**vars(args))
    print(json.dumps({"status": result["status"], "decision": result.get("decision"), "reasons": result.get("reasons", [])}, sort_keys=True))
    raise SystemExit(0 if result["status"] != "harness-invalid" else 1)


if __name__ == "__main__":
    main()
