#!/usr/bin/env python3
"""Zero-training native conditional-control audit for the v60 student.

The v61 held-out ridge upper bound established that the frozen final camera
intensity can be decoded in software.  This audit does not refit that readout:
it asks whether its already-held-out test predictions contain paired,
within-class residual information or only category templates.  It also forwards
two native conditional controls through the unchanged optical model:

* label-only: one fixed readable-train code with each requested label;
* code-only: each requested code with one fixed label.

The optical student remains frozen.  The controls, the within-class target
derangement, and the residual analysis are all diagnostic-only; none is an
inference-time readout or a promotion path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

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
DEFAULT_UPPER_REPORT = REPO_ROOT / "outputs/reports/v61_offline_decodability_upper_bound.json"
DEFAULT_UPPER_TENSORS = REPO_ROOT / "outputs/reports/v61_offline_decodability_upper_bound.pt"
DEFAULT_DATASETS = {
    "train": DATASET_ROOT / "teacher_distill_dataset_readable_train.pt",
    "val": DATASET_ROOT / "teacher_distill_dataset_readable_val.pt",
    "test": DATASET_ROOT / "teacher_distill_dataset_readable_test.pt",
}
DEFAULT_OUTPUT = REPO_ROOT / "outputs/reports/v62_native_conditional_control_audit.json"

SPLIT_COUNTS = {"train": 800, "val": 100, "test": 100}
SPLIT_CLASS_COUNTS = {"train": 80, "val": 10, "test": 10}
REQUIRED_V58_3_SECTIONS = {
    "formal_full_fov_validation",
    "independent_analytic_validation",
    "measured_pupil_psf_validation",
    "propagation_component_diagnostic",
    "sampling_and_independence_audit",
}
FOREIGN_OPERATION_NAMES = {
    "generic_resize",
    "sample_center_aperture",
    "physical_aperture_sampling",
}
BOOTSTRAP_ITERATIONS = 2_000
PERMUTATION_COUNT = 128
INSTANCE_RESIDUAL_CORR_MIN = 0.10
CONDITIONAL_RESPONSE_MIN = 0.02


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _invalid_report(*, inputs: Mapping[str, Path], reasons: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "v62-native-conditional-control-audit",
        "status": "harness-invalid",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "inputs": {name: str(path) for name, path in inputs.items()},
        "reasons": reasons,
    }


def _load_bundle(path: Path) -> dict[str, Any]:
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
        reasons.append(f"{split}: codes must have shape [{expected_count},128]")
    if not torch.is_tensor(labels) or tuple(labels.shape) != (expected_count,):
        reasons.append(f"{split}: class_labels must have shape [{expected_count}]")
    if not torch.is_tensor(target) or tuple(target.shape) != (expected_count, 1, 200, 200):
        reasons.append(f"{split}: teacher_camera must have shape [{expected_count},1,200,200]")
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


def _validate_forward(outputs: Mapping[str, Any], *, config) -> tuple[torch.Tensor, dict[str, float]]:
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
        raw_intensity, config.physics.readout.normalize_percentile, clamp=False
    ).clamp(0.0, 1.0)
    if not torch.allclose(detector_intensity, expected_detector, rtol=1.0e-5, atol=1.0e-6):
        difference = float((detector_intensity - expected_detector).abs().amax().detach().cpu())
        raise ValueError(f"physical_camera_plane does not match detector normalization (max error {difference})")
    return detector_intensity.detach().cpu().float(), {
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


def deterministic_within_class_target_rotation(target: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Rotate target assignments inside each label, with no fixed point."""
    if target.shape[0] != labels.shape[0]:
        raise ValueError("target and labels must have aligned leading dimensions")
    rotated = target.detach().clone()
    for label in torch.unique(labels.detach().cpu().long(), sorted=True):
        indices = torch.nonzero(labels.detach().cpu().long() == label, as_tuple=False).flatten()
        if indices.numel() < 2:
            raise ValueError(f"label {int(label)} has fewer than two samples for target rotation")
        rotated.index_copy_(0, indices, target.index_select(0, indices.roll(1)))
    return rotated


def _code_digest(code: torch.Tensor) -> str:
    return hashlib.sha256(code.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _forward_camera(
    model: torch.nn.Module,
    *,
    codes: torch.Tensor,
    labels: torch.Tensor,
    config,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    collected: list[torch.Tensor] = []
    errors = {
        "raw_camera_intensity_field_max_error": 0.0,
        "detector_normalization_max_error": 0.0,
    }
    with torch.no_grad():
        for start in range(0, int(codes.shape[0]), batch_size):
            end = min(start + batch_size, int(codes.shape[0]))
            outputs = model(codes[start:end].to(device), labels[start:end].to(device), return_optical_fields=True)
            camera, batch_errors = _validate_forward(outputs, config=config)
            collected.append(camera)
            for name, value in batch_errors.items():
                errors[name] = max(errors[name], value)
            del outputs, camera
            torch.cuda.empty_cache()
    return torch.cat(collected, dim=0), errors


def _normalized_features(values: torch.Tensor) -> torch.Tensor:
    features = values.detach().cpu().float().flatten(1)
    features = features - features.mean(dim=1, keepdim=True)
    return features / features.norm(dim=1, keepdim=True).clamp_min(1.0e-12)


def _fold_ids(labels: torch.Tensor, folds: int = 5) -> torch.Tensor:
    result = torch.empty_like(labels)
    for label in torch.unique(labels, sorted=True):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        result[indices] = torch.arange(indices.numel()) % folds
    return result


def prototype_cv_accuracy(values: torch.Tensor, labels: torch.Tensor, *, folds: int = 5) -> float:
    features = _normalized_features(values)
    fold_ids = _fold_ids(labels, folds)
    classes = torch.unique(labels, sorted=True)
    correct, total = 0, 0
    for fold in range(folds):
        test = fold_ids == fold
        train = ~test
        centroids = []
        for label in classes:
            centroid = features[train & (labels == label)].mean(dim=0)
            centroids.append(centroid / centroid.norm().clamp_min(1.0e-12))
        prediction = classes[(features[test] @ torch.stack(centroids).T).argmax(dim=1)]
        correct += int((prediction == labels[test]).sum())
        total += int(test.sum())
    return float(correct / max(total, 1))


def permutation_null(values: torch.Tensor, labels: torch.Tensor, *, seed: int) -> dict[str, float | int]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    observed = prototype_cv_accuracy(values, labels)
    null = torch.tensor(
        [
            prototype_cv_accuracy(values, labels[torch.randperm(labels.numel(), generator=generator)])
            for _ in range(PERMUTATION_COUNT)
        ]
    )
    return {
        "observed_accuracy": observed,
        "permutation_count": PERMUTATION_COUNT,
        "seed": seed,
        "null_mean": float(null.mean()),
        "null_p95": float(torch.quantile(null, 0.95)),
        "p_value": float(((null >= observed).sum() + 1) / (PERMUTATION_COUNT + 1)),
    }


def _per_sample_correlation(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    a, b = left.float().flatten(1), right.float().flatten(1)
    a, b = a - a.mean(dim=1, keepdim=True), b - b.mean(dim=1, keepdim=True)
    return (a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1.0e-12)


def _image_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    corr = _per_sample_correlation(prediction, target)
    mse = (prediction.float() - target.float()).square().flatten(1).mean(dim=1)
    return {
        "sample_count": int(prediction.shape[0]),
        "corr_mean": float(corr.mean()),
        "mse_mean": float(mse.mean()),
        "per_sample_corr": corr.tolist(),
        "per_sample_mse": mse.tolist(),
    }


def leave_one_out_class_residual(values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Remove each sample's class template without including that sample in it."""
    result = torch.empty_like(values)
    for label in torch.unique(labels, sorted=True):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        if indices.numel() < 2:
            raise ValueError(f"label {int(label)} has fewer than two samples for residual analysis")
        group = values.index_select(0, indices)
        template = (group.sum(dim=0, keepdim=True) - group) / float(indices.numel() - 1)
        result.index_copy_(0, indices, group - template)
    return result


def _residual_metrics(prediction: torch.Tensor, target: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    prediction_residual = leave_one_out_class_residual(prediction, labels)
    target_residual = leave_one_out_class_residual(target, labels)
    metrics = _image_metrics(prediction_residual, target_residual)
    target_energy = target_residual.square().flatten(1).mean(dim=1).clamp_min(1.0e-12)
    metrics["residual_explained_variance_mean"] = float(
        1.0 - (prediction_residual - target_residual).square().flatten(1).mean(dim=1).div(target_energy).mean()
    )
    metrics["prediction_residual_energy_mean"] = float(prediction_residual.square().flatten(1).mean())
    metrics["target_residual_energy_mean"] = float(target_residual.square().flatten(1).mean())
    return metrics


def paired_bootstrap_delta(
    matched: list[float],
    control: list[float],
    *,
    seed: int,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, float | int]:
    left, right = torch.tensor(matched), torch.tensor(control)
    if left.shape != right.shape or left.numel() == 0:
        raise ValueError("paired bootstrap values must be non-empty and aligned")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(left.numel(), (iterations, left.numel()), generator=generator)
    draws = (left[indices] - right[indices]).mean(dim=1)
    return {
        "mean": float((left - right).mean()),
        "ci_low": float(torch.quantile(draws, 0.025)),
        "ci_high": float(torch.quantile(draws, 0.975)),
        "iterations": iterations,
        "seed": seed,
    }


def _relative_response(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    delta = (candidate.float() - reference.float()).flatten(1).norm(dim=1)
    denom = reference.float().flatten(1).norm(dim=1).clamp_min(1.0e-12)
    values = delta / denom
    return {"mean": float(values.mean()), "median": float(values.median()), "per_sample": values.tolist()}


def _within_class_residual_energy(values: torch.Tensor, labels: torch.Tensor) -> float:
    residual = leave_one_out_class_residual(values, labels)
    full_energy = values.float().square().flatten(1).mean(dim=1).mean().clamp_min(1.0e-12)
    return float(residual.float().square().flatten(1).mean() / full_energy)


def _validate_upper_bound(
    report_path: Path,
    tensor_path: Path,
    *,
    checkpoint: Path,
    test_payload: Mapping[str, Any],
) -> tuple[list[str], torch.Tensor]:
    reasons: list[str] = []
    decoded = torch.empty(0)
    if not report_path.is_file() or not tensor_path.is_file():
        missing = [str(path) for path in (report_path, tensor_path) if not path.is_file()]
        return [f"v61 native upper-bound artifact missing: {', '.join(missing)}"], decoded
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "v61-offline-linear-decodability-upper-bound":
        reasons.append("v61 upper-bound report has an unexpected schema")
    if report.get("decision", {}).get("status") != "camera-intensity-decodable":
        reasons.append("v61 upper-bound decision is not camera-intensity-decodable")
    if report.get("diagnostic_only") is not True or report.get("accepted_summary_written") is not False:
        reasons.append("v61 upper-bound acceptance metadata is invalid")
    report_checkpoint = Path(str(report.get("inputs", {}).get("checkpoint", "")))
    if report_checkpoint != checkpoint:
        reasons.append(f"v61 upper-bound checkpoint mismatch: {report_checkpoint} != {checkpoint}")
    bundle = _load_bundle(tensor_path)
    labels = bundle.get("class_labels")
    target = bundle.get("teacher_target")
    predictions = bundle.get("decoded_targets")
    expected_labels = test_payload.get("class_labels")
    expected_target = test_payload.get("teacher_camera")
    if bundle.get("schema_version") != "v61-offline-linear-decodability-upper-bound":
        reasons.append("v61 upper-bound tensor has an unexpected schema")
    if bundle.get("diagnostic_only") is not True or bundle.get("accepted_summary_written") is not False:
        reasons.append("v61 upper-bound tensor acceptance metadata is invalid")
    if not torch.is_tensor(labels) or not torch.is_tensor(expected_labels) or not torch.equal(labels.long(), expected_labels.long()):
        reasons.append("v61 upper-bound tensor labels do not match the readable test split")
    if not torch.is_tensor(target) or not torch.is_tensor(expected_target) or not torch.equal(target.float(), expected_target.float()):
        reasons.append("v61 upper-bound tensor teacher targets do not match the readable test split")
    if not isinstance(predictions, Mapping) or not torch.is_tensor(predictions.get("camera_intensity")):
        reasons.append("v61 upper-bound tensor lacks camera_intensity predictions")
    else:
        decoded = predictions["camera_intensity"].detach().cpu().float()
        if tuple(decoded.shape) != (SPLIT_COUNTS["test"], 1, 200, 200) or not bool(torch.isfinite(decoded).all()):
            reasons.append("v61 camera_intensity predictions are not finite native 100x1x200x200 tensors")
    return reasons, decoded


def _decision(controls: Mapping[str, Any], pairing: Mapping[str, Any]) -> dict[str, str]:
    label_signal = controls["label_only"]["prototype_cv"]["p_value"] < 0.01
    code_response = (
        controls["baseline_vs_label_only"]["mean"] >= CONDITIONAL_RESPONSE_MIN
        and controls["code_only"]["within_class_residual_energy"] >= CONDITIONAL_RESPONSE_MIN
    )
    residual_delta = pairing["residual_corr_vs_within_class_target_rotation"]
    instance_signal = (
        pairing["matched_residual"]["corr_mean"] >= INSTANCE_RESIDUAL_CORR_MIN
        and residual_delta["ci_low"] > 0.0
    )
    if instance_signal and code_response:
        return {
            "status": "camera-intensity-contains-code-conditioned-instance-residual",
            "next_gate": "isolate why the retained instance signal is visually unreadable; do not train or promote",
        }
    if instance_signal:
        return {
            "status": "paired-instance-residual-with-weak-explicit-code-control",
            "next_gate": "audit the code-conditioning intervention before any representation change; do not train",
        }
    if label_signal and not code_response:
        return {
            "status": "class-template-only-camera-decodability-candidate",
            "next_gate": "audit the teacher/code conditional target contract; no decoder or loss tuning",
        }
    if label_signal:
        return {
            "status": "class-template-dominated-no-verified-instance-residual",
            "next_gate": "audit the teacher/code conditional target contract; no decoder or loss tuning",
        }
    return {
        "status": "conditional-control-inconclusive",
        "next_gate": "inspect harness and conditional source provenance before any training",
    }


def _render_preview(
    output: Path,
    *,
    target: torch.Tensor,
    labels: torch.Tensor,
    baseline: torch.Tensor,
    label_only: torch.Tensor,
    code_only: torch.Tensor,
    decoded: torch.Tensor,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to render the v62 preview") from exc
    indices = [int(torch.nonzero(labels == label, as_tuple=False)[0]) for label in range(10)]
    columns = (
        ("teacher target", target),
        ("native baseline CMOS", baseline),
        ("native label-only CMOS", label_only),
        ("native code-only CMOS", code_only),
        ("v61 offline decoded target", decoded),
    )
    figure, axes = plt.subplots(10, len(columns), figsize=(13, 24), squeeze=False)
    for row, index in enumerate(indices):
        for column, (title, tensor) in enumerate(columns):
            image = tensor[index, 0].detach().cpu().float()
            scale = image.quantile(0.995).clamp_min(1.0e-12)
            axes[row, column].imshow((image / scale).clamp(0.0, 1.0), cmap="gray", interpolation="nearest")
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(title)
        axes[row, 0].set_ylabel(f"class {int(labels[index])}")
    figure.suptitle("v62 conditional controls — diagnostic only, not a physical readout", y=0.999)
    figure.tight_layout()
    preview = output.with_name(f"{output.stem}_preview.png")
    figure.savefig(preview, dpi=160)
    plt.close(figure)
    return preview


def run(
    *,
    train_config: str | Path = DEFAULT_TRAIN_CONFIG,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    reference_report: str | Path = DEFAULT_REFERENCE_REPORT,
    upper_bound_report: str | Path = DEFAULT_UPPER_REPORT,
    upper_bound_tensors: str | Path = DEFAULT_UPPER_TENSORS,
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
        "upper_bound_report": Path(upper_bound_report),
        "upper_bound_tensors": Path(upper_bound_tensors),
        "train_dataset": Path(train_dataset),
        "val_dataset": Path(val_dataset),
        "test_dataset": Path(test_dataset),
    }
    output_path = Path(output)
    reasons = [f"missing input: {name}={path}" for name, path in paths.items() if not path.is_file()]
    if paths["checkpoint"].resolve() != DEFAULT_CHECKPOINT.resolve():
        reasons.append("v62 accepts only the pre-registered v60 epoch-5 checkpoint")
    if device != "cuda":
        reasons.append("v62 native conditional-control diagnostic requires --device cuda")
    if not torch.cuda.is_available():
        reasons.append("CUDA is unavailable")
    if reasons:
        report = _invalid_report(inputs=paths, reasons=reasons)
        _json_dump(output_path, report)
        return report
    config = load_training_config(paths["train_config"])
    reasons.extend(_validate_contract(config))
    reasons.extend(_validate_v58_3_report(paths["reference_report"]))
    datasets = {split: _load_bundle(paths[f"{split}_dataset"]) for split in SPLIT_COUNTS}
    for split, payload in datasets.items():
        reasons.extend(_validate_readable_split(payload, split))
    upper_reasons, decoded = _validate_upper_bound(
        paths["upper_bound_report"],
        paths["upper_bound_tensors"],
        checkpoint=paths["checkpoint"],
        test_payload=datasets["test"],
    )
    reasons.extend(upper_reasons)
    if reasons:
        report = _invalid_report(inputs=paths, reasons=reasons)
        _json_dump(output_path, report)
        return report
    train_codes = datasets["train"]["codes"].detach().cpu().float()
    test_codes = datasets["test"]["codes"].detach().cpu().float()
    test_labels = datasets["test"]["class_labels"].detach().cpu().long()
    test_target = datasets["test"]["teacher_camera"].detach().cpu().float()
    fixed_code = train_codes[0].clone()
    label_only_codes = fixed_code.unsqueeze(0).expand_as(test_codes).clone()
    code_only_labels = torch.zeros_like(test_labels)
    rotated_target = deterministic_within_class_target_rotation(test_target, test_labels)
    resolved_device = torch.device("cuda")
    model = OneShotOpticalGenerator(config.physics, config.one_shot, image_size=config.first_stage.image_size).to(resolved_device)
    load_one_shot_checkpoint(model, torch.load(paths["checkpoint"], map_location=resolved_device, weights_only=False))
    model.eval()
    before = _state_snapshot(model)
    report: dict[str, Any] = {
        "schema_version": "v62-native-conditional-control-audit",
        "status": "manual-review-required",
        "diagnostic_only": True,
        "accepted_summary_written": False,
        "inputs": {name: str(path) for name, path in paths.items()},
        "invariants": {
            "student_training": False,
            "optimizer_created": False,
            "parameter_update_count": 0,
            "formal_contract_changed": False,
            "target_code_or_bridge": "disabled",
            "camera_sampling": "native 200x200 at 2.5 um",
            "preexisting_v61_readout_refit": False,
        },
        "pre_registered_controls": {
            "label_only": "one fixed readable-train code; requested test labels unchanged",
            "code_only": "requested test codes; all labels fixed to 0",
            "within_class_target_rotation": "one-position deterministic rotation within each test class",
            "residual": "leave-one-out class templates on the held-out test split",
            "instance_residual_corr_min": INSTANCE_RESIDUAL_CORR_MIN,
            "conditional_response_min": CONDITIONAL_RESPONSE_MIN,
            "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
            "permutation_count": PERMUTATION_COUNT,
            "fixed_train_code_sha256": _code_digest(fixed_code),
        },
        "progress": [],
        "reasons": [],
    }
    _json_dump(output_path, report)
    try:
        baseline, baseline_errors = _forward_camera(
            model,
            codes=test_codes,
            labels=test_labels,
            config=config,
            device=resolved_device,
            batch_size=batch_size,
        )
        report["progress"].append("baseline")
        _json_dump(output_path, report)
        label_only, label_errors = _forward_camera(
            model,
            codes=label_only_codes,
            labels=test_labels,
            config=config,
            device=resolved_device,
            batch_size=batch_size,
        )
        report["progress"].append("label_only")
        _json_dump(output_path, report)
        code_only, code_errors = _forward_camera(
            model,
            codes=test_codes,
            labels=code_only_labels,
            config=config,
            device=resolved_device,
            batch_size=batch_size,
        )
        report["progress"].append("code_only")
        _json_dump(output_path, report)
        changed = _state_changes(model, before)
        report["invariants"]["parameter_update_count"] = len(changed)
        if changed:
            report["status"] = "harness-invalid"
            report["reasons"].append(f"model parameters changed during diagnostic: {changed[:5]}")
            return report
        report["field_semantic_checks"] = {
            name: max(baseline_errors[name], label_errors[name], code_errors[name])
            for name in baseline_errors
        }
        matched_full = _image_metrics(decoded, test_target)
        rotated_full = _image_metrics(decoded, rotated_target)
        matched_residual = _residual_metrics(decoded, test_target, test_labels)
        rotated_residual = _residual_metrics(decoded, rotated_target, test_labels)
        controls = {
            "label_only": {
                "prototype_cv": permutation_null(label_only, test_labels, seed=6201),
                "within_class_residual_energy": _within_class_residual_energy(label_only, test_labels),
            },
            "code_only": {
                "prototype_cv_against_original_labels": permutation_null(code_only, test_labels, seed=6202),
                "within_class_residual_energy": _within_class_residual_energy(code_only, test_labels),
            },
            "baseline": {
                "prototype_cv": permutation_null(baseline, test_labels, seed=6203),
                "within_class_residual_energy": _within_class_residual_energy(baseline, test_labels),
            },
            "baseline_vs_label_only": _relative_response(baseline, label_only),
            "baseline_vs_code_only": _relative_response(baseline, code_only),
        }
        pairing = {
            "matched_full": matched_full,
            "within_class_rotated_full": rotated_full,
            "matched_residual": matched_residual,
            "within_class_rotated_residual": rotated_residual,
            "full_corr_vs_within_class_target_rotation": paired_bootstrap_delta(
                matched_full["per_sample_corr"], rotated_full["per_sample_corr"], seed=6204
            ),
            "residual_corr_vs_within_class_target_rotation": paired_bootstrap_delta(
                matched_residual["per_sample_corr"], rotated_residual["per_sample_corr"], seed=6205
            ),
        }
        report["controls"] = controls
        report["pairing"] = pairing
        report["decision"] = _decision(controls, pairing)
        preview = _render_preview(
            output_path,
            target=test_target,
            labels=test_labels,
            baseline=baseline,
            label_only=label_only,
            code_only=code_only,
            decoded=decoded,
        )
        tensor_path = output_path.with_suffix(".pt")
        torch.save(
            {
                "schema_version": report["schema_version"],
                "diagnostic_only": True,
                "accepted_summary_written": False,
                "class_labels": test_labels,
                "teacher_target": test_target,
                "within_class_rotated_target": rotated_target,
                "baseline_camera": baseline,
                "label_only_camera": label_only,
                "code_only_camera": code_only,
                "v61_camera_intensity_decoded_target": decoded,
            },
            tensor_path,
        )
        report["artifacts"] = {"tensors": str(tensor_path), "preview": str(preview)}
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
    parser.add_argument("--upper-bound-report", type=Path, default=DEFAULT_UPPER_REPORT)
    parser.add_argument("--upper-bound-tensors", type=Path, default=DEFAULT_UPPER_TENSORS)
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
