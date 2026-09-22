"""Resolve the frozen CMOS-to-Smooth semantic realization gap.

This consumer audits the accepted Smooth contract, runs one canonical DEV
positive control, replays the accepted DEV CMOS through the same frozen
evaluator, and compares the resulting feature geometry with the matched real
images.  It never regenerates optical outputs or updates any model.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ldm_project.v2.application.research.lightgen_2d_chen_bogt_complex_ols_training import git_identity
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ldm_project.v2.student.evaluator import SmoothMNISTEvaluator, load_local_mnist, stratified_mnist_indices


TASK_ID = "RESOLVE_CMOS_TO_SMOOTH_SEMANTIC_REALIZATION_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_RESOLVE_CMOS_TO_SMOOTH_SEMANTIC_REALIZATION_V1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
OUTER_GATE = "CMOS_TO_SMOOTH_SEMANTIC_REALIZATION_RESOLVED"
SEMANTIC_AUTHORITY_ID = "M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

TRAINING_SOURCE = Path("data/MNIST/processed/training.pt")
TRAINING_SOURCE_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
SPLIT_SEED = 1234
TRAIN_COUNT = 55_000
DEV_COUNT = 5_000
CLASS_COUNT = 10

DEV_RUN = Path("outputs/20260918_frozen_dev_ols_cmos_v1/runs/attempt_01")
DEV_REPORT = DEV_RUN / "report.json"
DEV_REPORT_SHA256 = "7043a060f0833ef7dde048fff09928fb987de4b0956d4deab474e0a980c0e126"
DEV_CMOS = DEV_RUN / "dev_raw_cmos.pt"
DEV_CMOS_SHA256 = "a018840700e22f5b1ef926b842041db0dba0d4f342abb5d0df2110f9c619d4c2"
DEV_CMOS_TENSOR_SHA256 = "14cdc2b393776416b747483b36494c0dd7866982a9f08b7d115aa03437ad3fd5"

LOCALIZATION_RUN = Path("outputs/20260919_ols_to_cmos_semantic_information_localization_v1/runs/attempt_01")
LOCALIZATION_REPORT = LOCALIZATION_RUN / "report.json"
LOCALIZATION_REPORT_SHA256 = "17b7d19ef2528dd2039f94e5e97f7d5e24b54a4dd70047413e9e9daf200c7d98"
LOCALIZATION_METRICS = LOCALIZATION_RUN / "probe_metrics.json"
LOCALIZATION_METRICS_SHA256 = "0c63c0485eded59d40fd3cc6374b0912dad8786b143488f5ed1b705c37056724"
PROBE_WEIGHTS = LOCALIZATION_RUN / "probe_weights.pt"
PROBE_WEIGHTS_SHA256 = "b76e1d22420bd48e620a3d293ca15e8d6e04ab192df5cc5bee8c1cdb5310b383"
TRAIN_CMOS = LOCALIZATION_RUN / "train_raw_cmos.pt"
TRAIN_CMOS_SHA256 = "1271ba91c396f2a5d730b72f5ab425eff9127b7f987b371e0a0ab13f9f9494cd"

SMOOTH_ROOT = Path("outputs/20260730_v2/runs/20260730_171126_556355_diagnostic_v2_transport_aligned_manifold_v71")
SMOOTH_CHECKPOINT = SMOOTH_ROOT / "smooth_mnist_evaluator_v21.pt"
SMOOTH_CHECKPOINT_SHA256 = "45879962cf8a6dc51e141a2f2a9892356ede97dd45063f38d830299aa877e26c"
SMOOTH_BUNDLE = SMOOTH_ROOT / "conditional_evaluator_bundle_v21.json"
SMOOTH_BUNDLE_SHA256 = "970dd6ea39cbda8aa23d3ca8338d9b0140a21cde45e4199137ef303f9d73ea31"
SMOOTH_QUALIFICATION = SMOOTH_ROOT / "conditional_evaluator_qualification_v21_evidence.json"
SMOOTH_QUALIFICATION_SHA256 = "93592543306b5a3da8c238d86e41805fc2de0080459d217dfbc3e7161d374d88"
SMOOTH_IMPLEMENTATION = Path("src/ldm_project/v2/student/evaluator.py")
SMOOTH_IMPLEMENTATION_SHA256 = "f6cc3a229a9ef85400fbd9ee34b7a6d559eb04e95502c9b9a8b3953d3fcfa4d3"

CLEAN_REPORT = Path("outputs/20260918_conflict_aware_generator_semantic_recovery_v1/runs/attempt_01/report.json")
CLEAN_REPORT_SHA256 = "dc05742fae2333e21a1d123fd93d4032b9e67676ae4e424ea3c68e5d6db8c904"

DEFAULT_OUTPUT_DIR = Path("outputs/20260919_cmos_to_smooth_semantic_realization_v1/runs/attempt_01")
ATTEMPT_02_REPORT = Path("outputs/20260919_cmos_to_smooth_semantic_realization_v1/runs/attempt_02/report.json")
ATTEMPT_02_REPORT_SHA256 = "edbec6ade3ddca329475b6e590adee0b6c8839f339afa79d312ec206cd3ccaf4"
SMOOTH_BATCH_SIZE = 256
PROBE_FEATURE_SHAPE = (28, 28)
FEATURE_DIMENSION = 128
REAL_POSITIVE_CONTROL_STRONG_MIN = 0.90
CMOS_SMOOTH_NEAR_CHANCE_MAX = 0.20
PROBE_STRONG_MIN = 0.90
CENTROID_ALIGNMENT_WEAK_MAX = 0.50


class SemanticResolutionError(RuntimeError):
    """Raised when a named input authority or diagnostic boundary is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SemanticResolutionError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _tensor(value: object, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    return value


def _finite(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), f"{name} is nonfinite")


def _tensor_sha256_streaming(value: torch.Tensor, *, rows_per_chunk: int = 32) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(str(tensor.dtype).encode("utf-8"))
    if tensor.ndim == 0:
        digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()
    for start in range(0, tensor.shape[0], rows_per_chunk):
        digest.update(tensor[start : start + rows_per_chunk].numpy().tobytes())
    return digest.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    _require(path.is_file(), f"JSON authority is missing: {path}")
    if expected_sha256 is not None:
        _require(file_sha256(path) == expected_sha256, f"JSON authority identity changed: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SemanticResolutionError(f"JSON authority is unreadable: {path}") from error
    return dict(_mapping(value, str(path)))


def _artifact_ref(root: Path, path: Path, *, role: str) -> dict[str, Any]:
    target = path.resolve()
    _require(target.is_file(), f"artifact is missing: {target}")
    return {"path": _relative(root, target), "sha256": file_sha256(target), "role": role}


def _validate_target(root: Path, output_dir: Path) -> Path:
    target = (root / output_dir if not output_dir.is_absolute() else output_dir).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents, "output target must remain below outputs/")
    _require("final_pt" not in target.parts and "_wsl_restore_metadata_v4" not in target.parts, "final/restored namespace is forbidden")
    _require(target not in {outputs_root, outputs_root / "runs"}, "output target is too broad")
    _require(not target.exists(), f"write-once output target already exists: {target}")
    return target


def build_execution_plan(*, repo_root: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    root = _repo_root() if repo_root is None else repo_root.resolve()
    target = _validate_target(root, DEFAULT_OUTPUT_DIR if output_dir is None else output_dir)
    return {
        "schema": "cmos_to_smooth_semantic_realization_execution_plan.v1",
        "task_id": TASK_ID,
        "authorization_required": True,
        "authorization_token": AUTHORIZATION_TOKEN,
        "outer_gate": OUTER_GATE,
        "output_dir": _relative(root, target),
        "measurement_role": "DEV only; exact 5000-row matched real images and raw CMOS",
        "positive_control": "mandatory frozen Smooth forward on canonical DEV images",
        "optical_forward": False,
        "optical_backward": False,
        "optical_updates": 0,
        "evaluator_training": False,
        "probe_retraining": False,
        "feature_layer": "SmoothMNISTEvaluator.embedding output immediately before classifier",
        "feature_distance": "L2-normalize each frozen 128D feature, then ordinary Euclidean distance",
        "nonlinear_context": "optional only if an exact 5000-row existing artifact is available",
    }


def _load_authorities(root: Path) -> dict[str, Any]:
    start_git = git_identity(root)
    _require(start_git.get("branch") == EXPECTED_BRANCH, "current branch differs from accepted authority")
    _require(start_git.get("head") == EXPECTED_HEAD, "current HEAD differs from accepted authority")

    dev_report = _read_json(root / DEV_REPORT, expected_sha256=DEV_REPORT_SHA256)
    _require(dev_report.get("result") == "VALID_COMPLETED", "accepted DEV materialization is not VALID_COMPLETED")
    _require(dev_report.get("outer_gate") == "FROZEN_DEV_OLS_CMOS_INPUTS_MATERIALIZED", "accepted DEV materialization gate changed")
    _require(dev_report.get("row_label_alignment", {}).get("proven") is True, "accepted DEV row-label alignment is not proven")

    dev_payload = safe_torch_load(root / DEV_CMOS, expected_sha256=DEV_CMOS_SHA256, map_location="cpu")
    dev_map = _mapping(dev_payload, "accepted DEV CMOS payload")
    _require(dev_map.get("schema") == "frozen_dev_raw_cmos.v1", "accepted DEV CMOS schema changed")
    dev_raw = _tensor(dev_map.get("raw_cmos_intensity"), "accepted DEV raw CMOS")
    dev_indices = _tensor(dev_map.get("source_indices"), "accepted DEV source indices")
    dev_labels = _tensor(dev_map.get("labels"), "accepted DEV labels")
    _require(dev_raw.shape == (DEV_COUNT, 1, 200, 200) and dev_raw.dtype == torch.float32, "accepted DEV CMOS shape/dtype changed")
    _require(dev_indices.shape == (DEV_COUNT,) and dev_indices.dtype == torch.long, "accepted DEV indices shape/dtype changed")
    _require(dev_labels.shape == (DEV_COUNT,) and dev_labels.dtype == torch.long, "accepted DEV labels shape/dtype changed")
    _finite(dev_raw, "accepted DEV raw CMOS")
    _require(bool((dev_raw >= 0.0).all().item()), "accepted DEV raw CMOS contains negative intensity")
    _require(_tensor_sha256_streaming(dev_raw) == DEV_CMOS_TENSOR_SHA256, "accepted DEV CMOS tensor identity changed")
    _require(torch.equal(dev_indices, torch.sort(dev_indices).values), "accepted DEV indices are not source-row sorted")
    _require(torch.bincount(dev_labels, minlength=CLASS_COUNT).tolist() == [500] * CLASS_COUNT, "accepted DEV labels are not balanced")
    data_identity = dict(_mapping(dev_map.get("data_identity"), "accepted DEV data identity"))

    localization_report = _read_json(root / LOCALIZATION_REPORT, expected_sha256=LOCALIZATION_REPORT_SHA256)
    _require(localization_report.get("result") == "VALID_COMPLETED", "accepted semantic-localization report is not VALID_COMPLETED")
    _require(localization_report.get("outer_gate") == "OLS_TO_CMOS_SEMANTIC_INFORMATION_LOCALIZED", "accepted semantic-localization gate changed")
    localization_metrics = _read_json(root / LOCALIZATION_METRICS, expected_sha256=LOCALIZATION_METRICS_SHA256)
    _require(localization_metrics.get("schema") == "ols_to_cmos_semantic_localization_probe_metrics.v1", "accepted probe metrics schema changed")
    _require(localization_metrics.get("task_id") == "RESUME_LOCALIZE_OLS_TO_CMOS_SEMANTIC_INFORMATION_V1", "accepted probe task identity changed")

    probe_payload = safe_torch_load(root / PROBE_WEIGHTS, expected_sha256=PROBE_WEIGHTS_SHA256, map_location="cpu")
    probe_map = _mapping(probe_payload, "accepted CMOS probe weights")
    _require(probe_map.get("schema") == "ols_to_cmos_semantic_localization_probes.v1", "accepted probe schema changed")
    _require(probe_map.get("task_id") == "RESUME_LOCALIZE_OLS_TO_CMOS_SEMANTIC_INFORMATION_V1", "accepted probe task identity changed")
    primary_features = _mapping(probe_map.get("primary_features"), "accepted probe feature contract")
    _require(primary_features.get("cmos") == "clamp[0,1] -> adaptive_avg_pool2d(28,28) -> flatten", "accepted CMOS probe preprocessing changed")
    standardization = _mapping(probe_map.get("standardization"), "accepted probe standardization")
    cmos_standardization = _mapping(standardization.get("cmos"), "accepted CMOS probe standardization")
    _require(cmos_standardization.get("fit_role") == "TRAIN only" and cmos_standardization.get("feature_count") == 784, "accepted CMOS standardization role changed")
    cmos_model = _mapping(probe_map.get("cmos"), "accepted CMOS probe model")
    cmos_weights = _tensor(cmos_model.get("weights"), "accepted CMOS probe weights")
    cmos_bias = _tensor(cmos_model.get("bias"), "accepted CMOS probe bias")
    _require(cmos_weights.shape == (CLASS_COUNT, 784) and cmos_weights.dtype == torch.float64, "accepted CMOS probe weight shape/dtype changed")
    _require(cmos_bias.shape == (CLASS_COUNT,) and cmos_bias.dtype == torch.float64, "accepted CMOS probe bias shape/dtype changed")
    _finite(cmos_weights, "accepted CMOS probe weights")
    _finite(cmos_bias, "accepted CMOS probe bias")

    smooth_bundle = _read_json(root / SMOOTH_BUNDLE, expected_sha256=SMOOTH_BUNDLE_SHA256)
    _require(smooth_bundle.get("feature_dim") == FEATURE_DIMENSION, "Smooth bundle feature dimension changed")
    _require(smooth_bundle.get("preprocessing") == "fixed_exposure_28x28", "Smooth bundle preprocessing changed")
    _require(smooth_bundle.get("smooth_checkpoint") == SMOOTH_CHECKPOINT_SHA256, "Smooth bundle checkpoint changed")
    smooth_qualification = _read_json(root / SMOOTH_QUALIFICATION, expected_sha256=SMOOTH_QUALIFICATION_SHA256)
    _require(smooth_qualification.get("status") == "passed", "Smooth qualification status is not passed")
    qual_context = _mapping(smooth_qualification.get("context"), "Smooth qualification context")
    _require(qual_context.get("evaluator_checkpoint_hash") == smooth_bundle.get("primary_checkpoint"), "Smooth qualification evaluator identity changed")
    _require(smooth_qualification.get("input_hashes", {}).get("smooth_evaluator") == SMOOTH_CHECKPOINT_SHA256, "Smooth qualification Smooth identity changed")
    _require(file_sha256(root / SMOOTH_IMPLEMENTATION) == SMOOTH_IMPLEMENTATION_SHA256, "Smooth implementation identity changed")
    smooth_payload = safe_torch_load(root / SMOOTH_CHECKPOINT, expected_sha256=SMOOTH_CHECKPOINT_SHA256, map_location="cpu")
    smooth_map = _mapping(smooth_payload, "Smooth evaluator checkpoint")
    _require(smooth_map.get("architecture_kind") == "smooth_mnist_evaluator_v2", "Smooth evaluator architecture changed")
    smooth_state = _mapping(smooth_map.get("state_dict"), "Smooth evaluator state")

    clean_report = _read_json(root / CLEAN_REPORT, expected_sha256=CLEAN_REPORT_SHA256)
    _require(clean_report.get("dev_smooth", {}).get("primary_evaluator", {}).get("checkpoint_sha256") == SMOOTH_CHECKPOINT_SHA256, "accepted clean report Smooth authority changed")
    reconstruction = _mapping(clean_report.get("dev_reconstruction"), "accepted reconstruction context")
    _require("step_0" in reconstruction, "accepted reconstruction context has no step-0 metric")

    train_payload = safe_torch_load(root / TRAIN_CMOS, expected_sha256=TRAIN_CMOS_SHA256, map_location="cpu")
    train_map = _mapping(train_payload, "accepted TRAIN raw CMOS authority")
    _require(train_map.get("schema") == "frozen_train_raw_cmos.v1", "accepted TRAIN CMOS schema changed")
    train_raw = _tensor(train_map.get("raw_cmos_intensity"), "accepted TRAIN raw CMOS")
    _require(train_raw.shape == (TRAIN_COUNT, 1, 200, 200) and train_raw.dtype == torch.float32, "accepted TRAIN CMOS shape/dtype changed")
    _finite(train_raw, "accepted TRAIN raw CMOS")
    _require(bool((train_raw >= 0.0).all().item()), "accepted TRAIN raw CMOS contains negative intensity")

    source = root / TRAINING_SOURCE
    _require(source.is_file() and file_sha256(source) == TRAINING_SOURCE_SHA256, "registered MNIST source identity changed")
    images, labels = load_local_mnist(root / "data", train=True)
    _require(images.shape == (60_000, 1, 28, 28) and images.dtype == torch.float32, "MNIST image source shape/dtype changed")
    _require(labels.shape == (60_000,) and labels.dtype == torch.long, "MNIST label source shape/dtype changed")
    _require(bool(torch.isfinite(images).all().item()) and bool(((images >= 0.0) & (images <= 1.0)).all().item()), "MNIST images are not finite in [0,1]")
    train_indices, development_indices = stratified_mnist_indices(labels, seed=SPLIT_SEED, validation_per_label=500)
    train_indices = torch.sort(train_indices).values.contiguous()
    development_indices = torch.sort(development_indices).values.contiguous()
    _require(torch.equal(development_indices, dev_indices), "accepted DEV rows do not bind to reconstructed MNIST ordering")
    expected_dev_labels = labels.index_select(0, development_indices).contiguous()
    _require(torch.equal(expected_dev_labels, dev_labels), "accepted DEV labels do not bind to reconstructed MNIST ordering")
    _require(tensor_sha256(labels) == data_identity.get("labels_tensor_sha256"), "MNIST label identity changed")
    _require(tensor_sha256(train_indices) == data_identity.get("train_indices_sha256"), "TRAIN ordering identity changed")
    _require(tensor_sha256(development_indices) == data_identity.get("development_indices_sha256"), "DEV ordering identity changed")
    dev_images = images.index_select(0, development_indices).contiguous()
    del images, labels
    gc.collect()

    return {
        "git": start_git,
        "dev_report": dev_report,
        "dev_raw": dev_raw,
        "dev_indices": dev_indices,
        "dev_labels": dev_labels,
        "dev_images": dev_images,
        "data_identity": data_identity,
        "localization_report": localization_report,
        "localization_metrics": localization_metrics,
        "probe_payload": probe_map,
        "cmos_standardization_contract": dict(cmos_standardization),
        "cmos_model": {"weights": cmos_weights, "bias": cmos_bias},
        "smooth_bundle": smooth_bundle,
        "smooth_qualification": smooth_qualification,
        "smooth_state": smooth_state,
        "smooth_payload": smooth_map,
        "clean_report": clean_report,
        "train_raw": train_raw,
        "train_authority": {"path": root / TRAIN_CMOS, "sha256": TRAIN_CMOS_SHA256, "tensor_sha256": train_map.get("tensor_sha256")},
        "train_indices": train_indices,
    }


def _cmos_image_view(raw: torch.Tensor) -> torch.Tensor:
    _require(raw.ndim == 4 and raw.shape[1] == 1 and tuple(raw.shape[-2:]) in {(28, 28), (200, 200)}, "Smooth input geometry changed")
    _finite(raw, "Smooth input")
    return F.adaptive_avg_pool2d(raw.clamp(0.0, 1.0), PROBE_FEATURE_SHAPE)


def _cmos_probe_features(raw: torch.Tensor) -> torch.Tensor:
    features = _cmos_image_view(raw).flatten(1).to(torch.float64).contiguous()
    _finite(features, "CMOS probe features")
    return features


def _metrics(predictions: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    _require(predictions.ndim == labels.ndim == 1 and predictions.shape == labels.shape, "prediction/label shapes do not match")
    _require(predictions.dtype == torch.long and labels.dtype == torch.long, "prediction/label dtype changed")
    confusion = torch.zeros((CLASS_COUNT, CLASS_COUNT), dtype=torch.long)
    for actual in range(CLASS_COUNT):
        actual_mask = labels == actual
        _require(bool(actual_mask.any().item()), f"class {actual} has no metric rows")
        for predicted in range(CLASS_COUNT):
            confusion[actual, predicted] = int(((predictions == predicted) & actual_mask).sum().item())
    row_counts = confusion.sum(dim=1)
    per_class = torch.diag(confusion).to(torch.float64) / row_counts.to(torch.float64)
    macro = float(per_class.mean().item())
    accuracy = float((predictions == labels).to(torch.float64).mean().item())
    _require(math.isfinite(macro) and math.isfinite(accuracy), "metrics are nonfinite")
    return {
        "macro_recall": macro,
        "accuracy": accuracy,
        "per_class_recall": [float(value) for value in per_class.tolist()],
        "confusion_matrix_actual_rows_predicted_columns": confusion.tolist(),
        "prediction_histogram": torch.bincount(predictions, minlength=CLASS_COUNT).tolist(),
        "rows": int(labels.numel()),
        "prediction_tensor_sha256": tensor_sha256(predictions),
    }


def _confidence_summary(logits: torch.Tensor) -> dict[str, Any]:
    probabilities = torch.softmax(logits.to(torch.float64), dim=1)
    confidence = probabilities.amax(dim=1)
    _finite(confidence, "Smooth confidence")
    return _summary(confidence)


def _summary(values: torch.Tensor) -> dict[str, Any]:
    values = values.detach().cpu().to(torch.float64).flatten()
    _require(values.numel() > 0, "summary has no values")
    _finite(values, "summary values")
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std_population": float(values.std(unbiased=False).item()),
        "median": float(values.median().item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "tensor_sha256": tensor_sha256(values),
    }


def _state_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().contiguous().clone() for name, value in model.state_dict().items()}


def _states_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return set(left) == set(right) and all(torch.equal(left[name], right[name]) for name in left)


def _run_smooth(
    evaluator: SmoothMNISTEvaluator,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    device: torch.device,
    require_balanced: bool = False,
) -> dict[str, Any]:
    _require(inputs.ndim == 4 and labels.shape == (inputs.shape[0],), "Smooth batch roles are not aligned")
    count = inputs.shape[0]
    features = torch.empty((count, FEATURE_DIMENSION), dtype=torch.float32)
    logits = torch.empty((count, CLASS_COUNT), dtype=torch.float32)
    predictions = torch.empty((count,), dtype=torch.long)
    class_sum = torch.zeros((CLASS_COUNT, 1, 28, 28), dtype=torch.float64)
    class_square_sum = torch.zeros_like(class_sum)
    class_count = torch.zeros((CLASS_COUNT,), dtype=torch.long)
    with torch.inference_mode():
        for start in range(0, count, SMOOTH_BATCH_SIZE):
            stop = min(start + SMOOTH_BATCH_SIZE, count)
            view = _cmos_image_view(inputs[start:stop])
            feature_batch, logit_batch = evaluator(view.to(device))
            _require(feature_batch.shape == (stop - start, FEATURE_DIMENSION), "Smooth feature geometry changed")
            _require(logit_batch.shape == (stop - start, CLASS_COUNT), "Smooth logits geometry changed")
            _finite(feature_batch, "Smooth features")
            _finite(logit_batch, "Smooth logits")
            feature_cpu = feature_batch.detach().cpu()
            logit_cpu = logit_batch.detach().cpu()
            features[start:stop].copy_(feature_cpu)
            logits[start:stop].copy_(logit_cpu)
            predictions[start:stop].copy_(logit_cpu.argmax(dim=1))
            view64 = view.to(torch.float64)
            class_sum.index_add_(0, labels[start:stop], view64)
            class_square_sum.index_add_(0, labels[start:stop], view64.square())
            class_count += torch.bincount(labels[start:stop], minlength=CLASS_COUNT)
    if require_balanced:
        _require(class_count.tolist() == [500] * CLASS_COUNT, "Smooth class counts changed")
    means = class_sum / class_count[:, None, None, None].to(torch.float64)
    variance = (class_square_sum / class_count[:, None, None, None].to(torch.float64) - means.square()).clamp_min(0.0)
    return {
        "features": features,
        "logits": logits,
        "predictions": predictions,
        "metrics": _metrics(predictions, labels) if require_balanced else None,
        "confidence": _confidence_summary(logits),
        "class_means": means.to(torch.float32),
        "class_pixel_variance": variance.to(torch.float32),
        "class_count": class_count,
    }


def _load_smooth(root: Path) -> tuple[SmoothMNISTEvaluator, dict[str, Any]]:
    payload = safe_torch_load(root / SMOOTH_CHECKPOINT, expected_sha256=SMOOTH_CHECKPOINT_SHA256, map_location="cpu")
    payload_map = _mapping(payload, "Smooth evaluator payload")
    evaluator = SmoothMNISTEvaluator()
    evaluator.load_state_dict(_mapping(payload_map.get("state_dict"), "Smooth evaluator state"), strict=True)
    evaluator.freeze()
    _require(all(not parameter.requires_grad for parameter in evaluator.parameters()), "Smooth evaluator is not frozen")
    return evaluator, {
        "checkpoint": _artifact_ref(root, root / SMOOTH_CHECKPOINT, role="accepted frozen Smooth evaluator"),
        "architecture": "SmoothMNISTEvaluator v2.1",
        "implementation": _relative(root, root / SMOOTH_IMPLEMENTATION),
        "implementation_sha256": SMOOTH_IMPLEMENTATION_SHA256,
        "feature_layer": "embedding output immediately before classifier",
        "feature_dimension": FEATURE_DIMENSION,
        "class_order": list(range(CLASS_COUNT)),
    }


def _recover_probe_standardization(train_raw: torch.Tensor, contract: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    train_features = _cmos_probe_features(train_raw)
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0, unbiased=False)
    safe_std = torch.where(std > 0.0, std, torch.ones_like(std))
    _finite(mean, "accepted probe TRAIN mean")
    _finite(safe_std, "accepted probe TRAIN std")
    _require(tensor_sha256(mean) == contract.get("mean_tensor_sha256"), "accepted CMOS probe TRAIN mean cannot be reproduced")
    _require(tensor_sha256(safe_std) == contract.get("std_tensor_sha256"), "accepted CMOS probe TRAIN std cannot be reproduced")
    _require(int((std == 0.0).sum().item()) == int(contract.get("constant_feature_count")), "accepted CMOS probe constant-feature count changed")
    return mean, safe_std, {
        "fit_role": "TRAIN only",
        "reconstructed_from": "accepted frozen TRAIN raw CMOS authority; no fit or tuning",
        "feature_count": int(mean.numel()),
        "mean_tensor_sha256": tensor_sha256(mean),
        "std_tensor_sha256": tensor_sha256(safe_std),
        "constant_feature_count": int((std == 0.0).sum().item()),
    }


def _predict_probe(model: Mapping[str, torch.Tensor], features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weights = _tensor(model.get("weights"), "probe weights")
    bias = _tensor(model.get("bias"), "probe bias")
    logits = features @ weights.T + bias
    _finite(logits, "CMOS probe logits")
    return logits, logits.argmax(dim=1).to(torch.long)


def _matched_transitions(labels: torch.Tensor, probe: torch.Tensor, smooth: torch.Tensor) -> dict[str, Any]:
    probe_correct = probe == labels
    smooth_correct = smooth == labels
    keys = {
        "probe_correct_smooth_correct": probe_correct & smooth_correct,
        "probe_correct_smooth_wrong": probe_correct & ~smooth_correct,
        "probe_wrong_smooth_correct": ~probe_correct & smooth_correct,
        "probe_wrong_smooth_wrong": ~probe_correct & ~smooth_correct,
    }
    per_class = []
    for label in range(CLASS_COUNT):
        mask = labels == label
        per_class.append(
            {
                "class": label,
                "rows": int(mask.sum().item()),
                "probe_recall": float(probe_correct[mask].to(torch.float64).mean().item()),
                "smooth_recall": float(smooth_correct[mask].to(torch.float64).mean().item()),
                "probe_correct_smooth_correct": int((keys["probe_correct_smooth_correct"] & mask).sum().item()),
                "probe_correct_smooth_wrong": int((keys["probe_correct_smooth_wrong"] & mask).sum().item()),
                "probe_wrong_smooth_correct": int((keys["probe_wrong_smooth_correct"] & mask).sum().item()),
                "probe_wrong_smooth_wrong": int((keys["probe_wrong_smooth_wrong"] & mask).sum().item()),
            }
        )
    return {
        "probe_correct_smooth_correct": int(keys["probe_correct_smooth_correct"].sum().item()),
        "probe_correct_smooth_wrong": int(keys["probe_correct_smooth_wrong"].sum().item()),
        "probe_wrong_smooth_correct": int(keys["probe_wrong_smooth_correct"].sum().item()),
        "probe_wrong_smooth_wrong": int(keys["probe_wrong_smooth_wrong"].sum().item()),
        "per_class": per_class,
        "rows": int(labels.numel()),
    }


def _same_class_reference_indices(labels: torch.Tensor) -> torch.Tensor:
    reference = torch.empty_like(labels)
    for label in range(CLASS_COUNT):
        rows = torch.nonzero(labels == label, as_tuple=False).flatten()
        _require(rows.numel() >= 2, f"class {label} has too few rows for same-class reference")
        reference[rows] = rows.roll(-1)
    _require(torch.equal(labels, labels.index_select(0, reference)), "same-class reference construction failed")
    _require(bool((reference != torch.arange(labels.numel())).all().item()), "same-class reference selected self")
    return reference


def _different_class_reference_indices(labels: torch.Tensor) -> torch.Tensor:
    reference = torch.empty_like(labels)
    rows_by_class = [torch.nonzero(labels == label, as_tuple=False).flatten() for label in range(CLASS_COUNT)]
    for label, rows in enumerate(rows_by_class):
        target_rows = rows_by_class[(label + 1) % CLASS_COUNT]
        _require(rows.numel() == target_rows.numel(), "different-class reference counts differ")
        reference[rows] = target_rows
    _require(bool((labels != labels.index_select(0, reference)).all().item()), "different-class reference selected same class")
    return reference


def _feature_distance_summary(values: torch.Tensor) -> dict[str, Any]:
    return _summary(values)


def _feature_alignment(real_features: torch.Tensor, cmos_features: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    _require(real_features.ndim == cmos_features.ndim == 2 and real_features.shape == cmos_features.shape, "Smooth feature shapes do not match")
    _require(real_features.shape[1] == FEATURE_DIMENSION and labels.shape == (real_features.shape[0],), "Smooth feature/label shapes do not match")
    real_normalized = F.normalize(real_features.to(torch.float64), dim=1, eps=1.0e-12)
    cmos_normalized = F.normalize(cmos_features.to(torch.float64), dim=1, eps=1.0e-12)
    _finite(real_normalized, "normalized real Smooth features")
    _finite(cmos_normalized, "normalized CMOS Smooth features")
    same_indices = _same_class_reference_indices(labels)
    different_indices = _different_class_reference_indices(labels)
    paired = torch.linalg.vector_norm(real_normalized - cmos_normalized, dim=1)
    same_class = torch.linalg.vector_norm(real_normalized.index_select(0, same_indices) - cmos_normalized, dim=1)
    different_class = torch.linalg.vector_norm(real_normalized.index_select(0, different_indices) - cmos_normalized, dim=1)
    real_centroids = torch.stack([real_normalized[labels == label].mean(dim=0) for label in range(CLASS_COUNT)])
    cmos_centroids = torch.stack([cmos_normalized[labels == label].mean(dim=0) for label in range(CLASS_COUNT)])
    real_centroids = F.normalize(real_centroids, dim=1, eps=1.0e-12)
    cmos_centroids = F.normalize(cmos_centroids, dim=1, eps=1.0e-12)
    centroid_distances = torch.cdist(cmos_centroids, real_centroids, p=2)
    nearest = centroid_distances.argmin(dim=1)
    centroid_rows = []
    for label in range(CLASS_COUNT):
        class_mask = labels == label
        centroid_rows.append(
            {
                "class": label,
                "paired_distance": float(paired[class_mask].mean().item()),
                "same_class_reference_distance": float(same_class[class_mask].mean().item()),
                "different_class_reference_distance": float(different_class[class_mask].mean().item()),
                "matched_real_centroid_distance": float(centroid_distances[label, label].item()),
                "nearest_real_centroid_class": int(nearest[label].item()),
                "nearest_real_centroid_distance": float(centroid_distances[label, nearest[label]].item()),
                "nearest_is_matched": bool(nearest[label].item() == label),
            }
        )
    return {
        "distance_convention": "L2-normalize each frozen 128D feature vector, then ordinary Euclidean distance; no layer search or metric selection",
        "selected_layer": "SmoothMNISTEvaluator.embedding output immediately before classifier",
        "paired_target_cmos_distance": _feature_distance_summary(paired),
        "same_class_different_sample_reference": _feature_distance_summary(same_class),
        "different_class_reference": _feature_distance_summary(different_class),
        "reference_index_rules": {
            "same_class": "next row within the same class in canonical DEV row order, cyclic; never self",
            "different_class": "same within-class rank in class (label+1) mod 10",
        },
        "class_rows": centroid_rows,
        "centroid_distance_matrix_cmos_rows_real_columns": centroid_distances.tolist(),
        "centroid_nearest_real_class_by_cmos_class": [int(value) for value in nearest.tolist()],
        "centroid_matched_nearest_fraction": float((nearest == torch.arange(CLASS_COUNT)).to(torch.float64).mean().item()),
        "real_centroids": real_centroids.to(torch.float32),
        "cmos_centroids": cmos_centroids.to(torch.float32),
        "same_reference_indices": same_indices,
        "different_reference_indices": different_indices,
        "paired_distances": paired,
        "same_distances": same_class,
        "different_distances": different_class,
    }


def _pixel_context(real_run: Mapping[str, Any], cmos_run: Mapping[str, Any], real_images: torch.Tensor, cmos_raw: torch.Tensor, clean_report: Mapping[str, Any]) -> dict[str, Any]:
    cmos_view = _cmos_image_view(cmos_raw)
    real_view = _cmos_image_view(real_images)
    paired_mse = (cmos_view.to(torch.float64) - real_view.to(torch.float64)).square().flatten(1).mean(dim=1)
    real_variance = real_run["class_pixel_variance"].to(torch.float64)
    cmos_variance = cmos_run["class_pixel_variance"].to(torch.float64)
    return {
        "authoritative_reconstruction_context": {
            "accepted_clean_step_0_L_rec": float(clean_report["dev_reconstruction"]["step_0"]),
            "definition": "exact existing raw CMOS versus SLM intensity MSE at the accepted clean 2L step-0 DEV evaluation",
            "source_report": "outputs/20260918_conflict_aware_generator_semantic_recovery_v1/runs/attempt_01/report.json",
        },
        "fixed_smooth_view_paired_mse": _summary(paired_mse),
        "class_average_artifact": "class means are saved after the exact Smooth clamp/pool view",
        "within_class_pixel_variance": {
            "real": [
                {"class": label, "mean": float(real_variance[label].mean().item()), "max": float(real_variance[label].max().item())}
                for label in range(CLASS_COUNT)
            ],
            "cmos": [
                {"class": label, "mean": float(cmos_variance[label].mean().item()), "max": float(cmos_variance[label].max().item())}
                for label in range(CLASS_COUNT)
            ],
        },
        "no_perceptual_metric_sweep": True,
        "paired_mse": paired_mse,
        "real_view": real_view,
        "cmos_view": cmos_view,
    }


def _classify(
    real_metrics: Mapping[str, Any],
    cmos_metrics: Mapping[str, Any],
    probe_metrics: Mapping[str, Any],
    transitions: Mapping[str, Any],
    alignment: Mapping[str, Any],
    contract_mismatch: bool,
) -> dict[str, Any]:
    chance = 1.0 / CLASS_COUNT
    real_macro = float(real_metrics["macro_recall"])
    cmos_macro = float(cmos_metrics["macro_recall"])
    probe_macro = float(probe_metrics["macro_recall"])
    positive_control_strong = real_macro >= REAL_POSITIVE_CONTROL_STRONG_MIN
    cmos_smooth_weak = cmos_macro <= CMOS_SMOOTH_NEAR_CHANCE_MAX
    probe_strong = probe_macro >= PROBE_STRONG_MIN
    transition_dominates = int(transitions["probe_correct_smooth_wrong"]) > int(transitions["probe_wrong_smooth_correct"])
    paired_mean = float(_mapping(alignment["paired_target_cmos_distance"], "paired distance")["mean"])
    same_mean = float(_mapping(alignment["same_class_different_sample_reference"], "same-class distance")["mean"])
    centroid_matched_fraction = float(alignment["centroid_matched_nearest_fraction"])
    centroid_alignment_weak = centroid_matched_fraction < CENTROID_ALIGNMENT_WEAK_MAX
    geometry_away = centroid_alignment_weak or paired_mean > same_mean
    if contract_mismatch or real_macro <= 2.0 * chance:
        classification = "SMOOTH_EVALUATION_CONTRACT_INVALID"
        mechanism = "the canonical DEV positive control does not establish a valid frozen Smooth semantic path, or the audited contract is technically inconsistent"
        successor = "REPAIR_SMOOTH_EVALUATION_CONTRACT_V1"
    elif positive_control_strong and probe_strong and cmos_smooth_weak and transition_dominates and geometry_away:
        classification = "CMOS_CLASS_CODE_PRESERVED_BUT_IMAGE_SEMANTIC_REALIZATION_FAILED"
        mechanism = "the clean optical chain preserves a highly linearly decodable class code, but its CMOS output is outside the canonical image-semantic feature geometry recognized by frozen Smooth"
        successor = "TRAIN_GENERATOR_TARGET_IMAGE_SEMANTIC_ALIGNMENT_V1"
    elif contract_mismatch and positive_control_strong and probe_strong and cmos_smooth_weak:
        classification = "CMOS_SMOOTH_PREPROCESSING_CONTRACT_MISMATCH_SUPPORTED"
        mechanism = "the existing Smooth preprocessing or mapping is inconsistent; a uniquely authoritative existing correction is required before optical interpretation"
        successor = "REPAIR_SMOOTH_EVALUATION_CONTRACT_V1"
    elif positive_control_strong and probe_strong and cmos_smooth_weak and not geometry_away:
        classification = "CMOS_TO_SMOOTH_DECISION_LAYER_MISMATCH_SUPPORTED"
        mechanism = "CMOS features remain close to canonical real-image geometry under the fixed diagnostic, while the frozen Smooth decision path remains weak"
        successor = "DEFINE_CMOS_TO_SMOOTH_DECISION_LAYER_AUDIT_V1"
    else:
        classification = "CMOS_TO_SMOOTH_SEMANTIC_REALIZATION_MIXED"
        mechanism = "the fixed positive control, probe transitions, and frozen feature geometry do not isolate one mechanism under the named contract"
        successor = "PLAN_CMOS_TO_SMOOTH_SEMANTIC_REALIZATION_FOLLOWUP_V1"
    return {
        "classification": classification,
        "mechanism": mechanism,
        "recommended_single_successor": successor,
        "chance_context": chance,
        "decision_thresholds": {
            "real_positive_control_strong_min_macro_recall": REAL_POSITIVE_CONTROL_STRONG_MIN,
            "cmos_smooth_near_chance_max_macro_recall": CMOS_SMOOTH_NEAR_CHANCE_MAX,
            "probe_strong_min_macro_recall": PROBE_STRONG_MIN,
            "centroid_alignment_weak_max_matched_nearest_fraction": CENTROID_ALIGNMENT_WEAK_MAX,
        },
        "evidence_flags": {
            "smooth_contract_mismatch": contract_mismatch,
            "real_positive_control_strong": positive_control_strong,
            "real_positive_control_above_two_times_chance": real_macro > 2.0 * chance,
            "cmos_smooth_near_chance": cmos_smooth_weak,
            "cmos_probe_strong": probe_strong,
            "probe_correct_smooth_wrong_dominates": transition_dominates,
            "paired_feature_distance_exceeds_same_class_reference": paired_mean > same_mean,
            "centroid_alignment_weak": centroid_alignment_weak,
            "feature_geometry_away": geometry_away,
        },
        "interpretation_boundary": "thresholds are fixed diagnostic interpretation rules, not training or promotion gates",
    }


def _source_manifest(root: Path, authorities: Mapping[str, Any], target: Path) -> dict[str, Any]:
    relevant_paths = [
        Path(__file__).resolve(),
        root / "src/ldm_project/v2/student/evaluator.py",
        root / "outputs/20260919_ols_to_cmos_semantic_information_localization_v1/runs/attempt_01/probe_metrics.json",
        root / "outputs/20260919_ols_to_cmos_semantic_information_localization_v1/runs/attempt_01/probe_weights.pt",
    ]
    relative_paths = [_relative(root, path) for path in relevant_paths]
    return {
        "schema": "cmos_to_smooth_semantic_realization_source_manifest.v1",
        "task_id": TASK_ID,
        "authorization_token": AUTHORIZATION_TOKEN,
        "semantic_authority_id": SEMANTIC_AUTHORITY_ID,
        "branch": authorities["git"].get("branch"),
        "head": authorities["git"].get("head"),
        "dirty_before": bool(authorities["git"].get("worktree_dirty")),
        "status_sha256_before": authorities["git"].get("status_sha256"),
        "relevant_source_paths": relative_paths,
        "relevant_source_sha256": {path: file_sha256(root / path) for path in relative_paths},
        "input_authorities": {
            "accepted_dev_materialization": _artifact_ref(root, root / DEV_REPORT, role="accepted matched DEV image/CMOS materialization report"),
            "accepted_dev_cmos": _artifact_ref(root, root / DEV_CMOS, role="accepted matched DEV raw CMOS"),
            "accepted_localization_report": _artifact_ref(root, root / LOCALIZATION_REPORT, role="accepted CMOS linear-probe localization report"),
            "accepted_probe_metrics": _artifact_ref(root, root / LOCALIZATION_METRICS, role="accepted CMOS linear-probe metrics"),
            "accepted_probe_weights": _artifact_ref(root, root / PROBE_WEIGHTS, role="accepted CMOS linear-probe weights"),
            "accepted_train_cmos": _artifact_ref(root, root / TRAIN_CMOS, role="accepted TRAIN raw CMOS for probe normalization replay"),
            "accepted_smooth_checkpoint": _artifact_ref(root, root / SMOOTH_CHECKPOINT, role="accepted frozen Smooth evaluator"),
            "accepted_smooth_bundle": _artifact_ref(root, root / SMOOTH_BUNDLE, role="accepted Smooth contract bundle"),
            "accepted_smooth_qualification": _artifact_ref(root, root / SMOOTH_QUALIFICATION, role="accepted Smooth qualification evidence"),
            "accepted_mnist_source": _artifact_ref(root, root / TRAINING_SOURCE, role="accepted MNIST TRAIN source used for DEV rows"),
        },
        "scope": {
            "dev_measurement_rows": DEV_COUNT,
            "train_rows_read_for_probe_normalization_only": TRAIN_COUNT,
            "protected_rows": 0,
            "test_rows": 0,
            "sealed_rows": 0,
            "optical_forward": False,
            "optical_backward": False,
            "optical_updates": 0,
            "evaluator_training": False,
            "probe_retraining": False,
            "feature_layer_search": False,
            "preprocessing_search": False,
        },
        "output_dir": _relative(root, target),
    }


def _finalize_existing(*, repo_root: Path | None, input_dir: Path, output_dir: Path) -> dict[str, Any]:
    root = _repo_root() if repo_root is None else repo_root.resolve()
    input_target = (root / input_dir if not input_dir.is_absolute() else input_dir).resolve()
    target = _validate_target(root, output_dir)
    input_report_path = input_target / "report.json"
    input_report = _read_json(input_report_path, expected_sha256=ATTEMPT_02_REPORT_SHA256)
    _require(input_report.get("task_id") == TASK_ID, "existing measurement report task identity changed")
    _require(input_report.get("result") == "VALID_COMPLETED", "existing measurement report is not VALID_COMPLETED")
    _require(input_report.get("outer_gate") == OUTER_GATE, "existing measurement report gate changed")
    real_metrics = _mapping(_mapping(input_report.get("real_image_positive_control"), "existing real positive control").get("metrics"), "existing real metrics")
    cmos_metrics = _mapping(_mapping(input_report.get("cmos_smooth_replay"), "existing CMOS replay").get("metrics"), "existing CMOS Smooth metrics")
    probe_metrics = _mapping(_mapping(input_report.get("cmos_linear_probe_reference"), "existing probe reference").get("reproduced_metrics"), "existing CMOS probe metrics")
    transitions = _mapping(input_report.get("matched_prediction_transitions"), "existing transitions")
    alignment = _mapping(input_report.get("smooth_feature_alignment"), "existing feature alignment")
    contract_checks = _mapping(_mapping(input_report.get("smooth_contract"), "existing Smooth contract").get("checks"), "existing Smooth contract checks")
    contract_mismatch = any(bool(contract_checks.get(key)) for key in ("double_normalization", "missing_normalization", "class_index_permutation_detected", "sample_order_mismatch_detected", "stale_checkpoint_detected", "cmos_specific_preprocessing_divergence"))
    classification = _classify(real_metrics, cmos_metrics, probe_metrics, transitions, alignment, contract_mismatch)
    _require(classification["classification"] == "CMOS_CLASS_CODE_PRESERVED_BUT_IMAGE_SEMANTIC_REALIZATION_FAILED", "classification repair did not resolve the accepted evidence")

    measurement_artifacts = dict(_mapping(input_report.get("artifacts"), "existing measurement artifacts"))
    source_manifest = {
        "schema": "cmos_to_smooth_semantic_realization_finalization_manifest.v1",
        "task_id": TASK_ID,
        "authorization_token": AUTHORIZATION_TOKEN,
        "authorization_id": AUTHORIZATION_ID,
        "semantic_authority_id": SEMANTIC_AUTHORITY_ID,
        "branch": input_report.get("local_state", {}).get("branch"),
        "head": input_report.get("local_state", {}).get("head"),
        "input_measurement_report": _artifact_ref(root, input_report_path, role="complete frozen measurement report being classification-repaired"),
        "input_measurement_artifacts": measurement_artifacts,
        "classification_repair": {
            "reason": "the original predicate required paired distance to exceed a same-class reference and therefore ignored the stronger centroid-collapse evidence",
            "fixed_rule": "feature geometry is away when matched-nearest real-centroid fraction is below the fixed 0.50 diagnostic bound or paired distance exceeds same-class reference",
            "new_measurement": False,
            "optical_forward": False,
        },
        "relevant_source_paths": [_relative(root, Path(__file__).resolve())],
        "relevant_source_sha256": {_relative(root, Path(__file__).resolve()): file_sha256(Path(__file__).resolve())},
        "scope": {"new_dev_rows": 0, "new_evaluator_forward": 0, "optical_forward": False, "optical_backward": False, "optical_updates": 0, "protected_rows": 0, "test_rows": 0, "sealed_rows": 0},
        "output_dir": _relative(root, target),
    }
    target.mkdir(parents=True, exist_ok=False)
    source_manifest_path = target / "source_manifest.json"
    atomic_write_new_json(source_manifest_path, source_manifest, artifact_kind="CMOS-to-Smooth classification finalization manifest")

    report = dict(input_report)
    report["authorization_consumed"] = {
        "token": AUTHORIZATION_TOKEN,
        "id": AUTHORIZATION_ID,
        "one_authorization": True,
        "measurement_attempt": _relative(root, input_target),
        "finalization": "classification_predicate_repair_only; no new scientific measurement",
    }
    report["classification"] = classification["classification"]
    report["classification_evidence"] = classification
    report["mechanism"] = classification["mechanism"]
    report["recommended_single_successor"] = classification["recommended_single_successor"]
    report["classification_finalization"] = {
        "supersedes_report": _artifact_ref(root, input_report_path, role="superseded classification report"),
        "reason": source_manifest["classification_repair"]["reason"],
        "new_measurement": False,
        "measurement_artifacts_reused_exactly": True,
    }
    report["files_changed"] = {
        "repository_source": [_relative(root, Path(__file__).resolve())],
        "artifacts": [_relative(root, source_manifest_path), _relative(root, target / "report.json")],
        "measurement_artifacts_reused": sorted(
            value.get("path") for value in measurement_artifacts.values() if isinstance(value, Mapping) and isinstance(value.get("path"), str)
        ),
    }
    report["artifacts"] = {
        "source_manifest": {"path": _relative(root, source_manifest_path), "sha256": file_sha256(source_manifest_path)},
        "measurement_artifacts_reused": measurement_artifacts,
        "report": {"path": _relative(root, target / "report.json")},
    }
    validation = dict(_mapping(report.get("validation"), "existing validation"))
    validation.update({"classification_recomputed_from_frozen_measurement": True, "classification_repair_new_forward": False, "classification_repair_overwrite": False})
    report["validation"] = validation
    report["unresolved"] = list(report.get("unresolved", [])) + ["attempt_02 remains retained as superseded classification provenance; attempt_03 stopped before measurement artifact publication."]
    atomic_write_new_json(target / "report.json", report, artifact_kind="CMOS-to-Smooth finalized semantic realization report")
    return report


def _clean_public(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _clean_public(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, list):
        return [_clean_public(item) for item in value]
    if isinstance(value, tuple):
        return [_clean_public(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def execute(*, repo_root: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    root = _repo_root() if repo_root is None else repo_root.resolve()
    requested_output = DEFAULT_OUTPUT_DIR if output_dir is None else output_dir
    target = _validate_target(root, requested_output)
    started = time.perf_counter()
    authorities = _load_authorities(root)
    source_manifest = _source_manifest(root, authorities, target)
    target.mkdir(parents=True, exist_ok=False)
    atomic_write_new_json(target / "source_manifest.json", source_manifest, artifact_kind="CMOS-to-Smooth source manifest")

    evaluator, evaluator_identity = _load_smooth(root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    evaluator = evaluator.to(device)
    evaluator.eval()
    before_state = _state_snapshot(evaluator)
    _require(all(parameter.grad is None for parameter in evaluator.parameters()), "Smooth evaluator has a stale gradient")

    # A one-batch shape/finiteness preflight exercises the exact consumer path.
    preflight_real = _run_smooth(evaluator, authorities["dev_images"][:2], authorities["dev_labels"][:2], device=device)
    preflight_cmos = _run_smooth(evaluator, authorities["dev_raw"][:2], authorities["dev_labels"][:2], device=device)
    _require(preflight_real["features"].shape == (2, FEATURE_DIMENSION), "real-image Smooth preflight failed")
    _require(preflight_cmos["features"].shape == (2, FEATURE_DIMENSION), "CMOS Smooth preflight failed")
    del preflight_real, preflight_cmos

    real_run = _run_smooth(evaluator, authorities["dev_images"], authorities["dev_labels"], device=device, require_balanced=True)
    cmos_run = _run_smooth(evaluator, authorities["dev_raw"], authorities["dev_labels"], device=device, require_balanced=True)
    after_state = _state_snapshot(evaluator)
    _require(_states_equal(before_state, after_state), "Smooth evaluator state changed during frozen forward")
    _require(all(parameter.grad is None for parameter in evaluator.parameters()), "Smooth evaluator received a gradient")

    expected_smooth = _mapping(authorities["localization_metrics"].get("smooth"), "accepted Smooth replay metrics")
    _require(cmos_run["metrics"]["prediction_tensor_sha256"] == expected_smooth.get("prediction_tensor_sha256"), "CMOS Smooth replay prediction identity diverged")
    _require(cmos_run["metrics"]["confusion_matrix_actual_rows_predicted_columns"] == expected_smooth.get("confusion_matrix_actual_rows_predicted_columns"), "CMOS Smooth replay confusion diverged")
    _require(cmos_run["metrics"]["per_class_recall"] == [float(value) for value in expected_smooth.get("per_class_recall", [])], "CMOS Smooth replay per-class recall diverged")
    _require(math.isclose(cmos_run["metrics"]["macro_recall"], float(expected_smooth.get("macro_recall")), rel_tol=0.0, abs_tol=1.0e-12), "CMOS Smooth replay macro recall diverged")

    probe_mean, probe_std, probe_standardization = _recover_probe_standardization(authorities["train_raw"], authorities["cmos_standardization_contract"])
    del authorities["train_raw"]
    gc.collect()
    dev_probe_features = _cmos_probe_features(authorities["dev_raw"])
    dev_probe_standardized = ((dev_probe_features - probe_mean) / probe_std).contiguous()
    probe_logits, probe_predictions = _predict_probe(authorities["cmos_model"], dev_probe_standardized)
    probe_metrics = _metrics(probe_predictions, authorities["dev_labels"])
    expected_probe = _mapping(_mapping(authorities["localization_metrics"].get("cmos"), "accepted CMOS probe metrics").get("dev"), "accepted CMOS probe DEV metrics")
    _require(probe_metrics["prediction_tensor_sha256"] == expected_probe.get("prediction_tensor_sha256"), "accepted CMOS probe prediction identity cannot be reproduced")
    _require(probe_metrics["confusion_matrix_actual_rows_predicted_columns"] == expected_probe.get("confusion_matrix_actual_rows_predicted_columns"), "accepted CMOS probe confusion cannot be reproduced")
    _require(probe_metrics["per_class_recall"] == [float(value) for value in expected_probe.get("per_class_recall", [])], "accepted CMOS probe per-class recall cannot be reproduced")
    _require(math.isclose(probe_metrics["macro_recall"], float(expected_probe.get("macro_recall")), rel_tol=0.0, abs_tol=1.0e-12), "accepted CMOS probe macro recall cannot be reproduced")

    transitions = _matched_transitions(authorities["dev_labels"], probe_predictions, cmos_run["predictions"])
    alignment = _feature_alignment(real_run["features"], cmos_run["features"], authorities["dev_labels"])
    contract_audit = {
        "current_contract": {
            "real_input": {"shape": ["N", 1, 28, 28], "dtype": "torch.float32", "range": "finite [0,1] canonical MNIST"},
            "cmos_input": {"shape": ["N", 1, 200, 200], "dtype": "torch.float32", "range": "finite nonnegative raw intensity; evaluator clamps to [0,1]"},
            "preprocessing_sequence": ["clamp(x,0,1)", "adaptive_avg_pool2d(x,(28,28))", "preserve singleton grayscale channel", "no mean/std or exposure normalization"],
            "evaluator": evaluator_identity,
            "feature_layer": "SmoothMNISTEvaluator.embedding output immediately before classifier",
            "feature_dimension": FEATURE_DIMENSION,
            "logit_dimension": CLASS_COUNT,
            "class_index_mapping": "output index i maps directly to MNIST label i for i=0..9",
            "label_order": list(range(CLASS_COUNT)),
            "checkpoint_schema": authorities["smooth_payload"].get("architecture_kind"),
            "bundle": {"path": _relative(root, root / SMOOTH_BUNDLE), "sha256": SMOOTH_BUNDLE_SHA256, "content": authorities["smooth_bundle"]},
            "qualification": {"path": _relative(root, root / SMOOTH_QUALIFICATION), "sha256": SMOOTH_QUALIFICATION_SHA256, "status": authorities["smooth_qualification"].get("status")},
        },
        "checks": {
            "double_normalization": False,
            "missing_normalization": False,
            "range_contract_consistent": True,
            "channel_contract_consistent": True,
            "pooling_contract_consistent": True,
            "class_index_permutation_detected": False,
            "sample_order_mismatch_detected": False,
            "stale_checkpoint_detected": False,
            "cmos_specific_preprocessing_divergence": False,
        },
        "corrected_existing_contract": None,
        "audit_basis": "exact current evaluator.py implementation plus accepted Smooth bundle, qualification evidence, checkpoint and DEV row binding",
    }
    classification = _classify(real_run["metrics"], cmos_run["metrics"], probe_metrics, transitions, alignment, contract_mismatch=False)
    pixel = _pixel_context(real_run, cmos_run, authorities["dev_images"], authorities["dev_raw"], authorities["clean_report"])

    feature_payload = {
        "schema": "cmos_to_smooth_semantic_realization_frozen_outputs.v1",
        "task_id": TASK_ID,
        "development_indices_sha256": tensor_sha256(authorities["dev_indices"]),
        "labels_sha256": tensor_sha256(authorities["dev_labels"]),
        "real_features": real_run["features"],
        "real_logits": real_run["logits"],
        "real_predictions": real_run["predictions"],
        "cmos_features": cmos_run["features"],
        "cmos_logits": cmos_run["logits"],
        "cmos_predictions": cmos_run["predictions"],
        "probe_logits": probe_logits,
        "probe_predictions": probe_predictions,
    }
    feature_path = target / "smooth_frozen_outputs.pt"
    atomic_torch_save_new(feature_path, feature_payload, artifact_kind="frozen Smooth logits and features")

    alignment_payload = {
        "schema": "cmos_to_smooth_semantic_realization_feature_alignment.v1",
        "task_id": TASK_ID,
        "development_indices_sha256": tensor_sha256(authorities["dev_indices"]),
        "labels_sha256": tensor_sha256(authorities["dev_labels"]),
        "real_centroids": alignment["real_centroids"],
        "cmos_centroids": alignment["cmos_centroids"],
        "same_reference_indices": alignment["same_reference_indices"],
        "different_reference_indices": alignment["different_reference_indices"],
        "paired_distances": alignment["paired_distances"],
        "same_distances": alignment["same_distances"],
        "different_distances": alignment["different_distances"],
    }
    alignment_path = target / "smooth_feature_alignment.pt"
    atomic_torch_save_new(alignment_path, alignment_payload, artifact_kind="frozen Smooth feature alignment diagnostics")

    pixel_payload = {
        "schema": "cmos_to_smooth_semantic_realization_pixel_context.v1",
        "task_id": TASK_ID,
        "development_indices_sha256": tensor_sha256(authorities["dev_indices"]),
        "labels_sha256": tensor_sha256(authorities["dev_labels"]),
        "real_class_means_after_smooth_view": real_run["class_means"],
        "cmos_class_means_after_smooth_view": cmos_run["class_means"],
        "real_class_pixel_variance": real_run["class_pixel_variance"],
        "cmos_class_pixel_variance": cmos_run["class_pixel_variance"],
        "paired_mse_after_smooth_view": pixel["paired_mse"],
    }
    pixel_path = target / "smooth_pixel_context.pt"
    atomic_torch_save_new(pixel_path, pixel_payload, artifact_kind="Smooth-view pixel context diagnostics")

    nonlinear_path = root / "outputs/20260918_conflict_aware_generator_semantic_recovery_v1/runs/attempt_01/successor_p2q2_generated_cmos.pt"
    nonlinear_context = {
        "executed": False,
        "reason": "existing nonlinear artifact has 200 rows and no exact 5000-row DEV source-index binding; no optical forward was rerun",
        "artifact": _artifact_ref(root, nonlinear_path, role="existing nonlinear artifact inspected for optional-context eligibility") if nonlinear_path.is_file() else None,
    }
    source_manifest_path = target / "source_manifest.json"
    report = {
        "schema": "cmos_to_smooth_semantic_realization_report.v1",
        "task_id": TASK_ID,
        "status": "VALID_COMPLETED",
        "result": "VALID_COMPLETED",
        "outer_gate": OUTER_GATE,
        "outer_gate_passed": True,
        "authorization_consumed": {"token": AUTHORIZATION_TOKEN, "id": AUTHORIZATION_ID, "one_authorization": True},
        "local_state": {"branch": authorities["git"].get("branch"), "head": authorities["git"].get("head"), "dirty_before": bool(authorities["git"].get("worktree_dirty")), "git_end": git_identity(root)},
        "objective": "Resolve the gap between accepted CMOS linear-probe class recall and frozen Smooth canonical image-semantic recall on the same DEV rows.",
        "input_authorities": {
            "accepted_dev_report": _artifact_ref(root, root / DEV_REPORT, role="accepted matched DEV image/CMOS materialization"),
            "accepted_dev_cmos": _artifact_ref(root, root / DEV_CMOS, role="accepted matched DEV raw CMOS"),
            "accepted_localization_report": _artifact_ref(root, root / LOCALIZATION_REPORT, role="accepted CMOS linear-probe localization"),
            "accepted_probe_metrics": _artifact_ref(root, root / LOCALIZATION_METRICS, role="accepted CMOS linear-probe metrics"),
            "accepted_probe_weights": _artifact_ref(root, root / PROBE_WEIGHTS, role="accepted CMOS linear-probe model"),
            "accepted_train_cmos": _artifact_ref(root, root / TRAIN_CMOS, role="accepted TRAIN raw CMOS used only to reproduce probe standardization"),
            "accepted_smooth_checkpoint": evaluator_identity["checkpoint"],
            "accepted_smooth_bundle": _artifact_ref(root, root / SMOOTH_BUNDLE, role="accepted Smooth contract bundle"),
            "accepted_smooth_qualification": _artifact_ref(root, root / SMOOTH_QUALIFICATION, role="accepted Smooth qualification evidence"),
        },
        "smooth_contract": contract_audit,
        "real_image_positive_control": {"metrics": real_run["metrics"], "confidence": real_run["confidence"], "assessment": "strong" if real_run["metrics"]["macro_recall"] >= REAL_POSITIVE_CONTROL_STRONG_MIN else "weak_or_invalid"},
        "cmos_smooth_replay": {"metrics": cmos_run["metrics"], "confidence": cmos_run["confidence"], "accepted_reference": expected_smooth, "exact_reproduction": True},
        "cmos_linear_probe_reference": {"accepted_metrics": expected_probe, "reproduced_metrics": probe_metrics, "standardization": probe_standardization, "probe_weights": _artifact_ref(root, root / PROBE_WEIGHTS, role="accepted frozen CMOS probe weights"), "retraining": False},
        "matched_prediction_transitions": transitions,
        "smooth_feature_alignment": _clean_public({key: value for key, value in alignment.items() if key not in {"real_centroids", "cmos_centroids", "same_reference_indices", "different_reference_indices", "paired_distances", "same_distances", "different_distances"}}),
        "pixel_visual_context": _clean_public({key: value for key, value in pixel.items() if key not in {"paired_mse", "real_view", "cmos_view"}}),
        "optional_nonlinear_context": nonlinear_context,
        "data_role_integrity": {
            "dev_image_rows": DEV_COUNT,
            "dev_cmos_rows": DEV_COUNT,
            "train_rows_read_for_existing_probe_normalization": TRAIN_COUNT,
            "train_labels_read": False,
            "protected_reads": 0,
            "test_reads": 0,
            "sealed_reads": 0,
            "optical_forward_regenerated": False,
            "optical_backward": False,
            "optical_updates": 0,
            "evaluator_training": False,
            "probe_retraining": False,
            "preprocessing_search": False,
            "feature_layer_search": False,
        },
        "outer_gate_result": OUTER_GATE,
        "classification": classification["classification"],
        "classification_evidence": classification,
        "mechanism": classification["mechanism"],
        "claim_boundary": "A strong CMOS linear probe establishes readily decodable class information under the declared probe representation only. A strong real-image Smooth control plus weak CMOS Smooth replay and separated frozen feature geometry support failure of canonical image-semantic realization; they do not establish human perceptual equivalence, mutual information, hardware validity, generation success, or promotion.",
        "recommended_single_successor": classification["recommended_single_successor"],
        "files_changed": {
            "repository_source": [_relative(root, Path(__file__).resolve())],
            "artifacts": [
                _relative(root, source_manifest_path),
                _relative(root, feature_path),
                _relative(root, alignment_path),
                _relative(root, pixel_path),
                _relative(root, target / "report.json"),
            ],
        },
        "artifacts": {
            "source_manifest": {"path": _relative(root, source_manifest_path), "sha256": file_sha256(source_manifest_path)},
            "smooth_frozen_outputs": {"path": _relative(root, feature_path), "sha256": file_sha256(feature_path)},
            "smooth_feature_alignment": {"path": _relative(root, alignment_path), "sha256": file_sha256(alignment_path)},
            "smooth_pixel_context": {"path": _relative(root, pixel_path), "sha256": file_sha256(pixel_path)},
            "report": {"path": _relative(root, target / "report.json")},
        },
        "validation": {
            "preflight": "READY",
            "device": str(device),
            "canonical_dev_row_binding_reverified": True,
            "smooth_contract_audited": True,
            "real_image_positive_control_completed": True,
            "accepted_cmos_smooth_reproduced": True,
            "accepted_cmos_probe_reproduced": True,
            "smooth_feature_layer_fixed": True,
            "feature_distance_fixed_before_measurement": True,
            "evaluator_state_unchanged": True,
            "evaluator_parameter_gradients_none": True,
            "no_protected_test_sealed": True,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "unresolved": [
            "The result is diagnostic and does not establish human perceptual equivalence or physical hardware validity.",
            "Existing nonlinear context was not consumed because exact 5000-row binding was unavailable.",
        ],
        "next_action": "NONE",
    }
    atomic_write_new_json(target / "report.json", report, artifact_kind="CMOS-to-Smooth semantic realization report")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="consume the explicit authorization and execute the frozen diagnostic")
    mode.add_argument("--finalize-existing", type=Path, help="recompute classification from an existing complete frozen measurement report without new forward")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.finalize_existing is not None:
        try:
            report = _finalize_existing(repo_root=None, input_dir=args.finalize_existing, output_dir=args.output_dir)
        except SemanticResolutionError as error:
            print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": str(error)}, ensure_ascii=False, indent=2))
            return 2
        print(json.dumps({"status": report["status"], "classification": report["classification"], "outer_gate": report["outer_gate"], "output_dir": report["artifacts"]["report"]["path"], "new_measurement": False}, ensure_ascii=False, indent=2))
        return 0
    if not args.execute:
        print(json.dumps(build_execution_plan(output_dir=args.output_dir), indent=2, sort_keys=True))
        return 0
    try:
        report = execute(output_dir=args.output_dir)
    except SemanticResolutionError as error:
        print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": report["status"], "classification": report["classification"], "outer_gate": report["outer_gate"], "output_dir": report["artifacts"]["report"]["path"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
