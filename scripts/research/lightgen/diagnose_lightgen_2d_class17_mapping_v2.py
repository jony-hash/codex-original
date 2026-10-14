"""Frozen step-1200/2200 attribution for the LightGen class-1/class-7 mapping."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from ldm_project.v2.application.research.lightgen import canonical_semantics as canonical
from ldm_project.v2.application.research.lightgen import lightgen_2d_chen_bogt_complex_ols_training as bogt
from ldm_project.v2.application.research.lightgen.confirmation_evaluators import (
    PixelMLPConfirmationEvaluatorV1,
    ThirdConfirmationEvaluator,
)
from ldm_project.v2.student.evaluator import (
    FrozenMNISTEvaluator,
    SmoothMNISTEvaluator,
    stratified_mnist_indices,
)
from scripts.research.lightgen import execute_lightgen_2d_ols_to_cmos_semantic_information_localization_v1 as probes


TASK_ID = "M0_CLASS17_MAPPING_DIAGNOSTIC_V2"
OUTER_GATE = "M0_CLASS17_MAPPING_DIAGNOSTIC_RESOLVED"
REPORT_SCHEMA = "m0_class17_mapping_diagnostic_report.v2"
PREFLIGHT_SCHEMA = "m0_class17_mapping_diagnostic_preflight.v2"
SPLIT_SEED = 1234
LABEL_SHUFFLE_SEED = 2026100201
FIT_ROWS = 200
OBSERVATION_ROWS = 100
LOGICAL_BATCH = 10
FOCUS_CLASSES = (1, 7)
CONTROL_CLASSES = (0, 4, 8)
DISPLAY_CLASSES = (1, 7, 0, 4, 8)
DELTA_SIC = 1.365988374680088e-5
KAPPA = 0.6936758419764932
EXPECTED_ADAM = {
    "name": "Adam",
    "parameter_names": ["encoder.phase", "generator.phase"],
    "learning_rate": 1.0e-3,
    "betas": [0.9, 0.999],
    "eps": 1.0e-8,
    "weight_decay": 0.0,
    "amsgrad": False,
}
DEFAULTS = {
    "step1200": Path("outputs/20260929_m0_bogt_canonical_gradient_augmentation_canary_v1/runs/attempt_02/scientific_step_001200.pt"),
    "step2200": Path("outputs/20260930_m0_bogt_small_bank_joint_fit_canary_v1/runs/attempt_02/scientific_step_002200.pt"),
    "readout1200": Path("outputs/20261001_m0_bogt_class17_joint_mapping_attribution_v1/runs/attempt_01/step1200_measurements.pt"),
    "readout2200": Path("outputs/20261001_m0_bogt_class17_joint_mapping_attribution_v1/runs/attempt_01/step2200_measurements.pt"),
    "training_data": Path("data/MNIST/processed/training.pt"),
    "e_s": Path("outputs/20260730_v2/runs/20260730_171126_556355_diagnostic_v2_transport_aligned_manifold_v71/smooth_mnist_evaluator_v21.pt"),
    "e_i": Path("outputs/20260724_v2/runs/20260724_114819_935841_formal_v2_amplitude_finite_crop_identity/frozen_mnist_evaluator.pt"),
    "e_c": Path("outputs/20260919_third_confirmation_evaluator_v1/runs/attempt_03/evaluator_checkpoint.pt"),
    "e_d": Path("outputs/20260920_sic_independent_confirmation_evaluator_v1/runs/attempt_01/final_epoch_candidate.pt"),
    "output_dir": Path("outputs/20261002_m0_class17_mapping_diagnostic_v2/runs/attempt_01"),
}
ENDPOINTS = (("step1200", 1200), ("step2200", 2200))
PROBE_NAMES = ("ols_raw", "ols_gauge", "cmos_raw_pool", "cmos_evaluator_view", "e_d")


class DiagnosticError(RuntimeError):
    """An input, execution, or frozen-state invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _load_torch(path: Path, *, mmap: bool = False) -> Any:
    _require(path.is_file(), f"required input is missing: {path}")
    return torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)


def _finite_tensor(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), f"{name} contains nonfinite values")


def _state_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().contiguous().clone() for name, value in module.state_dict().items()}


def _state_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return set(left) == set(right) and all(torch.equal(left[name], right[name]) for name in left)


def _optimizer_snapshot(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return copy.deepcopy(optimizer.state_dict())


def _assert_tree_equal(left: Any, right: Any, name: str) -> None:
    if torch.is_tensor(left) or torch.is_tensor(right):
        _require(
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(left.detach().cpu(), right.detach().cpu()),
            f"{name} changed",
        )
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        _require(isinstance(left, Mapping) and isinstance(right, Mapping) and set(left) == set(right), f"{name} keys changed")
        for key in left:
            _assert_tree_equal(left[key], right[key], f"{name}.{key}")
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        _require(isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) and len(left) == len(right), f"{name} length changed")
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_tree_equal(a, b, f"{name}[{index}]")
        return
    _require(left == right, f"{name} changed")


def _checkpoint_step(payload: Mapping[str, Any], expected: int, name: str) -> int:
    if expected == 1200:
        _require(payload.get("schema") == "m0_bogt_canonical_gradient_augmentation_checkpoint.v1", "step1200 checkpoint schema changed")
        _require(payload.get("step") == 1200 and payload.get("completed_updates") == 600, "step1200 checkpoint step/update count mismatch")
        _require(payload.get("optimizer_step_vector") == [1200, 1200], "step1200 Adam step vector mismatch")
        identity = _mapping(payload.get("run_identity"), "step1200 run identity")
        model = _mapping(identity.get("model_identity"), "step1200 model identity")
        _require(model.get("runtime") == "LIGHTGEN_2D_CHEN_BOGT_COMPLEX_OLS_CONSUMER_V1", "step1200 runtime identity mismatch")
        _require(model.get("trainable") == ["encoder.phase", "generator.phase"], "step1200 trainable groups changed")
        _require(payload.get("kappa") == KAPPA, "step1200 kappa changed")
        manifest = _mapping(payload.get("source_manifest"), "step1200 source manifest")
        config = _mapping(manifest.get("config_identity"), "step1200 source configuration")
        _require(config.get("delta_SIC") == DELTA_SIC, "step1200 delta_SIC changed")
    else:
        _require(payload.get("schema") == "m0_bogt_small_bank_joint_fit_recovery.v1", "step2200 checkpoint schema changed")
        _require(payload.get("checkpoint_kind") == "SCIENTIFIC_TRAINING_STATE", "step2200 is not a scientific training state")
        _require(payload.get("phase") == "TRAINING_COMPLETE" and payload.get("parent_global_step") == 2200, "step2200 endpoint mismatch")
        _require(payload.get("cursor") == 1000 and payload.get("optimizer_step_vector") == [2200, 2200], "step2200 recovery cursor/Adam steps mismatch")
        _require(payload.get("row_update_counts") == [50] * 200, "step2200 small-bank row update counts mismatch")
        _require(name == "step2200", "step2200 payload was supplied for another endpoint")
    return expected


def _validate_optimizer_payload(payload: Mapping[str, Any], expected_step: int, name: str) -> None:
    config = _mapping(payload.get("optimizer_configuration"), f"{name} optimizer configuration")
    for key, expected in EXPECTED_ADAM.items():
        _require(config.get(key) == expected, f"{name} Adam configuration {key} mismatch")
    _require(config.get("gradient_accumulation") is False and config.get("autocast") is False and config.get("grad_scaler") is False, f"{name} Adam execution options changed")
    state = _mapping(payload.get("optimizer_state"), f"{name} optimizer state")
    groups = state.get("param_groups")
    entries = state.get("state")
    _require(isinstance(groups, list) and len(groups) == 1 and isinstance(entries, Mapping), f"{name} Adam state structure mismatch")
    group = _mapping(groups[0], f"{name} Adam param group")
    _require(group.get("params") == [0, 1], f"{name} Adam parameter order mismatch")
    for key, expected in (("lr", 1.0e-3), ("betas", (0.9, 0.999)), ("eps", 1.0e-8), ("weight_decay", 0.0), ("amsgrad", False), ("maximize", False)):
        _require(group.get(key) == expected, f"{name} Adam param group {key} mismatch")
    _require(set(entries) == {0, 1}, f"{name} Adam state does not cover both phase groups")
    for index in (0, 1):
        item = _mapping(entries[index], f"{name} Adam state {index}")
        _require(set(item) >= {"step", "exp_avg", "exp_avg_sq"}, f"{name} Adam moments are incomplete")
        step = item["step"]
        _require(torch.is_tensor(step) and float(step.item()) == expected_step, f"{name} Adam internal step mismatch for parameter {index}")
        phase = payload["trainable_parameter_state"][("encoder.phase", "generator.phase")[index]]
        for slot in ("exp_avg", "exp_avg_sq"):
            value = item[slot]
            _require(torch.is_tensor(value) and value.shape == phase.shape and value.dtype == phase.dtype, f"{name} Adam {slot} shape/dtype mismatch")
            _finite_tensor(value, f"{name} Adam {slot}")
        _require(bool((item["exp_avg_sq"] >= 0.0).all().item()), f"{name} Adam second moment is negative")


def _validate_checkpoint_pair(
    payloads: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    first = _mapping(payloads["step1200"], "step1200 checkpoint")
    second = _mapping(payloads["step2200"], "step2200 checkpoint")
    for name, step in ENDPOINTS:
        payload = _mapping(payloads[name], name)
        _checkpoint_step(payload, step, name)
        params = _mapping(payload.get("trainable_parameter_state"), f"{name} parameter state")
        _require(set(params) == {"encoder.phase", "generator.phase"}, f"{name} trainable parameter names mismatch")
        for key in ("encoder.phase", "generator.phase"):
            value = params[key]
            _require(torch.is_tensor(value) and tuple(value.shape) == (2, 1, 100, 100) and value.dtype == torch.float32, f"{name} {key} shape/dtype mismatch")
            _finite_tensor(value, f"{name} {key}")
        _validate_optimizer_payload(payload, step, name)
    data1200 = _mapping(first.get("data_identity"), "step1200 data identity")
    data2200 = _mapping(second.get("data_identity"), "step2200 data identity")
    for field in ("source_path", "seed", "split", "train_count", "development_count", "train_order", "protected_test_read", "sealed_read"):
        _require(data1200.get(field) == data2200.get(field), f"checkpoint data identity differs at {field}")
    _require(data1200.get("seed") == SPLIT_SEED and data1200.get("train_count") == 55_000, "checkpoint TRAIN split identity mismatch")
    refs_payload = _mapping(first.get("references"), "step1200 frozen references")
    weights_payload = _mapping(first.get("weights"), "step1200 BOGT weights")
    _require(second.get("references") is None and second.get("weights") is None, "step2200 unexpectedly supplies a competing shared calibration")
    weights = dict(weights_payload)
    _require(weights["data_role"] == bogt.TRAIN_ROLE and weights["engineering_only"] is False, "step1200 BOGT weights are not scientific TRAIN weights")
    return dict(refs_payload), weights, {"step1200": data1200, "step2200": data2200}


def _read_image_labels(payload: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(payload, (tuple, list)) and len(payload) == 2:
        images, labels = payload
    elif isinstance(payload, Mapping):
        images = payload.get("images", payload.get("data"))
        labels = payload.get("labels", payload.get("targets"))
    else:
        raise DiagnosticError("training.pt must contain the registered image/label pair")
    _require(torch.is_tensor(images) and tuple(images.shape) == (60_000, 28, 28) and images.dtype == torch.uint8, "training image container shape/dtype mismatch")
    _require(torch.is_tensor(labels) and tuple(labels.shape) == (60_000,) and labels.dtype == torch.long, "training labels shape/dtype mismatch")
    _require(bool(((labels >= 0) & (labels <= 9)).all().item()), "training labels are outside 0..9")
    return images, labels


def _validate_population(
    indices1200: torch.Tensor,
    labels1200: torch.Tensor,
    indices2200: torch.Tensor,
    labels2200: torch.Tensor,
    source_labels: torch.Tensor,
) -> dict[str, Any]:
    for name, indices, labels in (
        ("step1200", indices1200, labels1200),
        ("step2200", indices2200, labels2200),
    ):
        _require(indices.shape == (300,) and indices.dtype == torch.long, f"{name} row indices shape/dtype mismatch")
        _require(labels.shape == (300,) and labels.dtype == torch.long, f"{name} row labels shape/dtype mismatch")
        _require(torch.unique(indices).numel() == 300, f"{name} population contains duplicate rows")
    _require(torch.equal(indices1200, indices2200), "endpoint source-row order differs")
    _require(torch.equal(labels1200, labels2200), "endpoint labels differ")
    _require(torch.equal(labels1200, source_labels.index_select(0, indices1200)), "cached row labels differ from canonical TRAIN labels")
    fit_labels = labels1200[:FIT_ROWS]
    observation_labels = labels1200[FIT_ROWS:]
    _require(torch.bincount(fit_labels, minlength=10).tolist() == [20] * 10, "first 200 rows are not 20-per-class fit rows")
    _require(torch.bincount(observation_labels, minlength=10).tolist() == [10] * 10, "last 100 rows are not 10-per-class observation rows")
    _require(not set(indices1200[:FIT_ROWS].tolist()).intersection(indices1200[FIT_ROWS:].tolist()), "fit and observation source rows overlap")
    fit_batches = fit_labels.reshape(20, LOGICAL_BATCH)
    _require(bool((fit_batches.sort(dim=1).values == torch.arange(10).expand(20, 10)).all().item()), "original fit order is not 20 complete class-balanced batches")
    train_indices, _ = stratified_mnist_indices(source_labels, seed=SPLIT_SEED, validation_per_label=500)
    train_membership = torch.zeros((source_labels.numel(),), dtype=torch.bool)
    train_membership[train_indices] = True
    _require(bool(train_membership.index_select(0, indices1200).all().item()), "diagnostic population includes a non-TRAIN row")
    return {
        "count": 300,
        "fit_count": FIT_ROWS,
        "observation_count": OBSERVATION_ROWS,
        "fit_class_counts": torch.bincount(fit_labels, minlength=10).tolist(),
        "observation_class_counts": torch.bincount(observation_labels, minlength=10).tolist(),
        "fit_batch_count": 20,
        "fit_batch_size": LOGICAL_BATCH,
        "fit_observation_disjoint": True,
        "canonical_train_only": True,
        "split_seed": SPLIT_SEED,
        "row_order": "the source order saved in both prior readouts; first 200 fit, last 100 observation",
    }


def load_inputs(root: Path, paths: Mapping[str, Path]) -> dict[str, Any]:
    payloads = {name: _load_torch(paths[name]) for name in ("step1200", "step2200")}
    _validate_checkpoint_pair(payloads)
    old_readouts = {name: _mapping(_load_torch(paths[f"readout{name[-4:]}"]), f"{name} fixed-row readout") for name, _ in ENDPOINTS}
    for name, step in ENDPOINTS:
        _require(old_readouts[name].get("step") == step, f"{name} row readout step mismatch")
        _require(old_readouts[name].get("updates") == 0, f"{name} historical fixed-row readout records updates")
        _require(old_readouts[name].get("parameter_state_unchanged") is True, f"{name} historical readout does not record frozen parameters")
    images, source_labels = _read_image_labels(_load_torch(paths["training_data"], mmap=True))
    rows = _validate_population(
        old_readouts["step1200"]["indices"],
        old_readouts["step1200"]["labels"],
        old_readouts["step2200"]["indices"],
        old_readouts["step2200"]["labels"],
        source_labels,
    )
    selected_indices = old_readouts["step1200"]["indices"].clone().contiguous()
    selected_labels = source_labels.index_select(0, selected_indices).contiguous()
    selected_images = images.index_select(0, selected_indices).unsqueeze(1).to(torch.float32).div_(255.0).contiguous()
    _finite_tensor(selected_images, "selected TRAIN images")
    _require(float(selected_images.min()) >= 0.0 and float(selected_images.max()) <= 1.0, "selected TRAIN images are outside [0,1]")
    reference_payload, weights, checkpoint_data = _validate_checkpoint_pair(payloads)
    return {
        "payloads": payloads,
        "references": reference_payload,
        "weights": weights,
        "checkpoint_data": checkpoint_data,
        "source_indices": selected_indices,
        "labels": selected_labels,
        "images": selected_images,
        "row_contract": rows,
        "historical_readout_provenance": {
            name: {
                "path": paths[f"readout{name[-4:]}"].as_posix(),
                "artifact_kind": old_readouts[name].get("artifact_kind"),
                "used_for": "fixed source-row order and label cross-check only; no scientific predictions or CMOS/OLS measurements reused",
            }
            for name, _ in ENDPOINTS
        },
    }


def _probe_features(z: torch.Tensor, raw: torch.Tensor, references: Any) -> dict[str, torch.Tensor]:
    gauged, _, _ = bogt.max_q_gauge(z, references)
    return {
        "ols_raw": torch.cat((z.real, z.imag), dim=1).to(torch.float64).contiguous(),
        "ols_gauge": torch.cat((gauged.real, gauged.imag), dim=1).to(torch.float64).contiguous(),
        "cmos_raw_pool": F.adaptive_avg_pool2d(raw, (28, 28)).flatten(1).to(torch.float64).contiguous(),
        "cmos_evaluator_view": canonical.cmos_view(raw).flatten(1).to(torch.float64).contiguous(),
    }


def _fit_existing_probe(features: torch.Tensor, fit_labels: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, Any], torch.Tensor, dict[str, Any]]:
    train_z, observation_z, standardization = probes._standardize_train_only(features[:FIT_ROWS], features[FIT_ROWS:])
    model, metadata = probes._fit_logistic(train_z, fit_labels)
    reporting_standardization = {
        key: value
        for key, value in standardization.items()
        if not key.lower().endswith(("_sha256", "_hash", "_digest"))
    }
    return model, metadata, torch.cat((train_z, observation_z), dim=0), reporting_standardization


def _wilson(successes: int, total: int, z_value: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    rate = successes / total
    denominator = 1.0 + z_value * z_value / total
    center = (rate + z_value * z_value / (2.0 * total)) / denominator
    radius = z_value * math.sqrt(rate * (1.0 - rate) / total + z_value * z_value / (4.0 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _metrics_with_intervals(labels: torch.Tensor, predictions: torch.Tensor) -> dict[str, Any]:
    base = probes._metrics(predictions, labels)
    base.pop("prediction_tensor_sha256", None)
    confusion = torch.tensor(base["confusion_matrix_actual_rows_predicted_columns"], dtype=torch.long)
    row_counts = confusion.sum(dim=1)
    correct = confusion.diag()
    base["per_class_correct"] = correct.tolist()
    base["per_class_wilson_95"] = {
        str(class_id): _wilson(int(correct[class_id]), int(row_counts[class_id]))
        for class_id in range(10)
    }
    base["predicted_class_counts"] = torch.bincount(predictions, minlength=10).tolist()
    return base


def _exact_fixed_prediction_pvalue(labels: torch.Tensor, predictions: torch.Tensor, class_id: int) -> float:
    _require(labels.shape == predictions.shape and labels.numel() == OBSERVATION_ROWS, "permutation-test observation shape mismatch")
    total = int(labels.numel())
    class_rows = int((labels == class_id).sum().item())
    predicted_count = int((predictions == class_id).sum().item())
    observed = int(((labels == class_id) & (predictions == class_id)).sum().item())
    denominator = math.comb(total, class_rows)
    numerator = 0
    lower = max(0, class_rows - (total - predicted_count))
    upper = min(class_rows, predicted_count)
    for matches in range(max(observed, lower), upper + 1):
        numerator += math.comb(predicted_count, matches) * math.comb(total - predicted_count, class_rows - matches)
    return min(1.0, numerator / denominator)


def _holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted


def _probe_support(metrics: Mapping[str, Any], negative_metrics: Mapping[str, Any], adjusted_p: float, class_id: int) -> bool:
    correct = int(metrics["per_class_correct"][class_id])
    return (
        correct >= 5
        and adjusted_p <= 0.05
        and float(metrics["macro_recall"]) >= 0.50
        and float(negative_metrics["macro_recall"]) <= 0.20
        and float(metrics["macro_recall"]) - float(negative_metrics["macro_recall"]) >= 0.20
    )


def _ed_support(metrics: Mapping[str, Any], adjusted_p: float, class_id: int) -> bool:
    return int(metrics["per_class_correct"][class_id]) >= 5 and adjusted_p <= 0.05


def _loss_reductions(
    bundle: Mapping[str, Any],
    images: torch.Tensor,
    smooth: Any,
    independent: Any,
    ec: Any,
) -> dict[str, torch.Tensor]:
    raw = bundle["result"].raw_intensity
    raw_view = canonical.cmos_view(raw)
    target_view = canonical.cmos_view(images)
    s_generated, _ = smooth(raw_view)
    i_generated, _ = independent(raw_view)
    c_generated, _ = ec(raw_view)
    with torch.no_grad():
        s_target, _ = smooth(target_view)
        i_target, _ = independent(target_view)
        c_target, _ = ec(target_view)
    return {
        "L_S": canonical.normalized_feature_loss(s_generated, s_target, "L_S"),
        "L_I": canonical.normalized_feature_loss(i_generated, i_target, "L_I"),
        "L_C": canonical.normalized_feature_loss(c_generated, c_target, "L_C"),
    }


def _flatten_group(values: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([values[name].reshape(-1) for name in ("encoder.phase", "generator.phase")])


def _split_group(vector: torch.Tensor, parameter_shapes: Mapping[str, torch.Size]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    offset = 0
    for name in ("encoder.phase", "generator.phase"):
        size = math.prod(parameter_shapes[name])
        result[name] = vector[offset : offset + size].reshape(parameter_shapes[name])
        offset += size
    _require(offset == vector.numel(), "gradient vector parameter geometry mismatch")
    return result


def _adam_candidate(
    gradients: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    parameters: Mapping[str, torch.nn.Parameter],
    *,
    zero_history: bool = False,
) -> dict[str, torch.Tensor]:
    group = optimizer.param_groups[0]
    beta1, beta2 = group["betas"]
    lr = float(group["lr"])
    epsilon = float(group["eps"])
    result: dict[str, torch.Tensor] = {}
    for name in ("encoder.phase", "generator.phase"):
        parameter = parameters[name]
        gradient = gradients[name].detach()
        state = optimizer.state[parameter]
        if zero_history:
            first = torch.zeros_like(gradient)
            second = torch.zeros_like(gradient)
            step = 1
        else:
            first = state["exp_avg"].detach()
            second = state["exp_avg_sq"].detach()
            step = int(float(state["step"].item())) + 1
        first_next = first * beta1 + gradient * (1.0 - beta1)
        second_next = second * beta2 + gradient.square() * (1.0 - beta2)
        bias1 = 1.0 - beta1**step
        bias2 = 1.0 - beta2**step
        denominator = second_next.sqrt() / math.sqrt(bias2)
        update = -(lr / bias1) * first_next / denominator.add(epsilon)
        _finite_tensor(update, f"{name} read-only Adam candidate")
        result[name] = update
    return result


def _direction_record(semantic: torch.Tensor, displacement: torch.Tensor) -> dict[str, Any]:
    semantic_norm = float(semantic.detach().to(torch.float64).norm().cpu())
    displacement_norm = float(displacement.detach().to(torch.float64).norm().cpu())
    if semantic_norm == 0.0 or displacement_norm == 0.0:
        return {"status": "UNDECIDABLE_ZERO_GRADIENT", "dot": None, "cosine": None}
    dot = float((semantic.detach().to(torch.float64) * displacement.detach().to(torch.float64)).sum().cpu())
    cosine = dot / (semantic_norm * displacement_norm)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    if abs(cosine) < 0.05:
        status = "BOUNDARY"
    elif cosine < 0.0:
        status = "BENEFICIAL"
    else:
        status = "HARMFUL"
    return {"status": status, "dot": dot, "cosine": cosine, "angle_degrees": angle}


def _direction_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {key: sum(record["status"] == key for record in records) for key in ("BENEFICIAL", "HARMFUL", "BOUNDARY", "UNDECIDABLE_ZERO_GRADIENT")}
    consistent = max(counts["BENEFICIAL"], counts["HARMFUL"])
    return {**counts, "rows": len(records), "stable_mode": "BENEFICIAL" if counts["BENEFICIAL"] >= 16 else ("HARMFUL" if counts["HARMFUL"] >= 16 else None), "stable": consistent >= 16}


def _resource_snapshot(device: torch.device) -> dict[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allocated = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    return {
        "device": str(device),
        "total_bytes": int(total_bytes),
        "free_bytes": int(free_bytes),
        "global_used_bytes": int(total_bytes - free_bytes),
        "global_used_fraction": float((total_bytes - free_bytes) / total_bytes),
        "pytorch_allocated_bytes": allocated,
        "pytorch_reserved_bytes": reserved,
    }


def _load_evaluator(
    path: Path,
    model: torch.nn.Module,
    expected_architecture: str,
    name: str,
    device: torch.device,
) -> torch.nn.Module:
    payload = _mapping(_load_torch(path), f"{name} checkpoint")
    _require(payload.get("architecture_kind") == expected_architecture, f"{name} evaluator architecture mismatch")
    state = _mapping(payload.get("state_dict"), f"{name} evaluator state")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.freeze()
    model.eval()
    _require(all(not parameter.requires_grad and parameter.grad is None for parameter in model.parameters()), f"{name} evaluator is not frozen")
    return model


def load_evaluators(paths: Mapping[str, Path], device: torch.device) -> dict[str, torch.nn.Module]:
    return {
        "E_S": _load_evaluator(paths["e_s"], SmoothMNISTEvaluator(), SmoothMNISTEvaluator.architecture_kind, "E_S", device),
        "E_I": _load_evaluator(paths["e_i"], FrozenMNISTEvaluator(), "frozen_mnist_evaluator_v1", "E_I", device),
        "E_C": _load_evaluator(paths["e_c"], ThirdConfirmationEvaluator(), ThirdConfirmationEvaluator.architecture_kind, "E_C", device),
        "E_D": _load_evaluator(paths["e_d"], PixelMLPConfirmationEvaluatorV1(), PixelMLPConfirmationEvaluatorV1.architecture_kind, "E_D", device),
    }


def _new_endpoint(
    name: str,
    payload: Mapping[str, Any],
    references_payload: Mapping[str, Any],
    weights_payload: Mapping[str, Any],
    device: torch.device,
) -> tuple[Any, torch.optim.Optimizer, dict[str, torch.nn.Parameter], Any, Any]:
    consumer = bogt.LightGen2DChenBOGTComplexOLSTrainingConsumer(
        device=device,
        propagation_grid_size=bogt.PRODUCTION_PROPAGATION_GRID,
    )
    references = bogt.FrozenBogtReferences.from_payload(references_payload)
    if payload.get("weights") is not None:
        embedded_weights = _mapping(payload.get("weights"), f"{name} BOGT weights")
        for key in ("alpha", "beta", "gamma", "data_role", "engineering_only"):
            _require(embedded_weights.get(key) == weights_payload.get(key), f"{name} BOGT weight {key} differs from the inherited reference")
    weights = bogt.BogtLossWeights.from_payload(weights_payload)
    consumer.bind_references(references, for_scientific=True)
    consumer.bind_weights(weights, for_scientific=True)
    consumer.load_trainable_state_dict(_mapping(payload.get("trainable_parameter_state"), f"{name} phase state"))
    consumer.runtime.eval()
    parameters = dict(consumer.runtime.named_parameters())
    _require(tuple(parameters) == ("encoder.phase", "generator.phase"), f"{name} runtime trainable group names changed")
    optimizer = torch.optim.Adam(
        [parameters["encoder.phase"], parameters["generator.phase"]],
        lr=EXPECTED_ADAM["learning_rate"],
        betas=tuple(EXPECTED_ADAM["betas"]),
        eps=EXPECTED_ADAM["eps"],
        weight_decay=EXPECTED_ADAM["weight_decay"],
        amsgrad=EXPECTED_ADAM["amsgrad"],
    )
    optimizer.load_state_dict(copy.deepcopy(_mapping(payload.get("optimizer_state"), f"{name} optimizer state")))
    consumer.optimizer_manifest(optimizer)
    _require(consumer.runtime_identity().get("production_propagation_grid") == bogt.PRODUCTION_PROPAGATION_GRID, f"{name} runtime is not the production propagation grid")
    _require(consumer.references is not None and consumer.weights is not None, f"{name} did not bind inherited references/weights")
    return consumer, optimizer, parameters, references, weights


def _runtime_description(consumer: Any) -> dict[str, Any]:
    identity = consumer.runtime_identity()
    return {
        key: value
        for key, value in identity.items()
        if not key.lower().endswith(("_sha256", "_hash", "_digest"))
    }


def _output_views(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return F.adaptive_avg_pool2d(raw, (28, 28)), canonical.cmos_view(raw)


def _forward_readouts(
    consumer: Any,
    evaluators: Mapping[str, torch.nn.Module],
    images: torch.Tensor,
    *,
    batch_size: int = 10,
) -> dict[str, torch.Tensor]:
    device = consumer.device
    count = images.shape[0]
    z = torch.empty((count, 100), dtype=torch.complex64)
    raw = torch.empty((count, 1, 200, 200), dtype=torch.float32)
    target_raw = torch.empty((count, 1, 200, 200), dtype=torch.float32)
    powers = torch.empty((count, 5), dtype=torch.float64)
    ed_logits = torch.empty((count, 10), dtype=torch.float32)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        batch = images[start:stop].to(device)
        with torch.no_grad():
            result = consumer.runtime.forward_from_image(batch)
            cmos_raw, cmos_view = _output_views(result.raw_intensity)
            logits = evaluators["E_D"](cmos_view)
            z[start:stop].copy_(result.z_ols.coefficients.detach().cpu())
            raw[start:stop].copy_(result.raw_intensity.detach().cpu())
            powers[start:stop].copy_(bogt.contracted_plane_powers(result).detach().cpu())
            ed_logits[start:stop].copy_(logits.detach().cpu())
            _require(result.image_input is not None, "full optical path omitted the real target SLM input")
            target_raw[start:stop].copy_(result.image_input.slm_intensity.detach().cpu())
        del result, cmos_raw, cmos_view, logits
    for name, model in evaluators.items():
        _require(all(parameter.grad is None for parameter in model.parameters()), f"{name} acquired gradients during forward-only confirmation")
    _finite_tensor(z.real, "fresh OLS real coefficients")
    _finite_tensor(z.imag, "fresh OLS imaginary coefficients")
    _finite_tensor(raw, "fresh raw CMOS")
    _finite_tensor(target_raw, "fresh target SLM intensity")
    _finite_tensor(powers, "fresh plane powers")
    _finite_tensor(ed_logits, "fresh E_D logits")
    return {"z": z, "raw_cmos": raw, "target_slm_intensity": target_raw, "power_by_plane": powers, "e_d_logits": ed_logits}


def _target_e_d_logits(e_d: torch.nn.Module, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    logits: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, images.shape[0], 128):
            logits.append(e_d(images[start : start + 128].to(device)).detach().cpu())
    return torch.cat(logits)


def _row_reconstruction_metrics(
    raw: torch.Tensor,
    target_raw: torch.Tensor,
    images: torch.Tensor,
) -> dict[str, torch.Tensor]:
    target_view = canonical.cmos_view(images)
    generated_raw_view, generated_evaluator_view = _output_views(raw)
    target_eval_view = target_view
    raw_mask = target_raw > 0.1
    eval_mask = target_view > 0.1

    def masked_mse(generated: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expanded = mask.expand_as(generated)
        counts = expanded.flatten(1).sum(dim=1)
        squared = (generated - target).square()
        numerator = (squared * expanded).flatten(1).sum(dim=1)
        values = numerator / counts.clamp_min(1)
        values = torch.where(counts > 0, values, torch.full_like(values, float("nan")))
        return values, counts

    raw_foreground, raw_foreground_pixels = masked_mse(raw, target_raw, raw_mask)
    raw_background, raw_background_pixels = masked_mse(raw, target_raw, ~raw_mask)
    view_foreground, view_foreground_pixels = masked_mse(generated_evaluator_view, target_eval_view, eval_mask)
    view_background, view_background_pixels = masked_mse(generated_evaluator_view, target_eval_view, ~eval_mask)
    flat = raw.flatten(1)
    return {
        "raw_foreground_mse": raw_foreground,
        "raw_background_mse": raw_background,
        "raw_foreground_pixels": raw_foreground_pixels,
        "raw_background_pixels": raw_background_pixels,
        "evaluator_view_foreground_mse": view_foreground,
        "evaluator_view_background_mse": view_background,
        "evaluator_view_foreground_pixels": view_foreground_pixels,
        "evaluator_view_background_pixels": view_background_pixels,
        "raw_zero_image": flat.abs().sum(dim=1) == 0.0,
        "raw_constant_image": flat.std(dim=1, unbiased=False) == 0.0,
        "raw_fraction_above_one": (raw > 1.0).flatten(1).to(torch.float64).mean(dim=1),
        "raw_minimum": raw.flatten(1).amin(dim=1),
        "raw_maximum": raw.flatten(1).amax(dim=1),
        "target_raw_200x200": target_raw,
        "target_evaluator_view_28x28": target_view,
        "generated_raw_pool_28x28": generated_raw_view,
        "generated_evaluator_view_28x28": generated_evaluator_view,
    }


def _fit_probes(
    readouts: Mapping[str, torch.Tensor],
    images: torch.Tensor,
    labels: torch.Tensor,
    references: Any,
) -> dict[str, Any]:
    fit_labels = labels[:FIT_ROWS].cpu()
    observation_labels = labels[FIT_ROWS:].cpu()
    features = _probe_features(readouts["z"], readouts["raw_cmos"], references)
    generator = torch.Generator(device="cpu").manual_seed(LABEL_SHUFFLE_SEED)
    permutation = torch.randperm(FIT_ROWS, generator=generator)
    negative_fit_labels = fit_labels.index_select(0, permutation)
    records: dict[str, Any] = {}
    models: dict[str, Any] = {}
    predictions_for_tests: dict[str, torch.Tensor] = {}
    for name in PROBE_NAMES[:-1]:
        model, metadata, standardized, standardization = _fit_existing_probe(features[name], fit_labels)
        negative_model, negative_metadata = probes._fit_logistic(standardized[:FIT_ROWS], negative_fit_labels)
        train_predictions = probes._predict(model, standardized[:FIT_ROWS])
        observation_predictions = probes._predict(model, standardized[FIT_ROWS:])
        negative_observation_predictions = probes._predict(negative_model, standardized[FIT_ROWS:])
        train_metrics = _metrics_with_intervals(fit_labels, train_predictions)
        observation_metrics = _metrics_with_intervals(observation_labels, observation_predictions)
        negative_metrics = _metrics_with_intervals(observation_labels, negative_observation_predictions)
        records[name] = {
            "fit": train_metrics,
            "observation": observation_metrics,
            "label_shuffle_negative_control": negative_metrics,
            "negative_control": {"seed": LABEL_SHUFFLE_SEED, "fit_labels_shuffled_only": True, "observation_labels_unchanged": True},
            "solver": {**dict(metadata), "negative_control": dict(negative_metadata)},
            "standardization": standardization,
        }
        models[name] = {
            "weights": model["weights"],
            "bias": model["bias"],
            "negative_weights": negative_model["weights"],
            "negative_bias": negative_model["bias"],
        }
        predictions_for_tests[name] = observation_predictions
    target_features = images.flatten(1).to(torch.float64)
    target_model, target_metadata, target_standardized, target_standardization = _fit_existing_probe(
        target_features,
        fit_labels,
    )
    target_predictions = probes._predict(target_model, target_standardized[FIT_ROWS:])
    target_fit_predictions = probes._predict(target_model, target_standardized[:FIT_ROWS])
    records["target_positive_control"] = {
        "fit": _metrics_with_intervals(fit_labels, target_fit_predictions),
        "observation": _metrics_with_intervals(observation_labels, target_predictions),
        "solver": dict(target_metadata),
        "standardization": target_standardization,
        "feature_source": "real fixed TRAIN target images flattened at 28x28",
    }
    models["target_positive_control"] = {"weights": target_model["weights"], "bias": target_model["bias"]}
    predictions_for_tests["e_d"] = readouts["e_d_logits"][FIT_ROWS:].argmax(dim=1).cpu()
    records["e_d"] = {
        "fit": _metrics_with_intervals(fit_labels, readouts["e_d_logits"][:FIT_ROWS].argmax(dim=1).cpu()),
        "observation": _metrics_with_intervals(observation_labels, predictions_for_tests["e_d"]),
        "role": "frozen forward-only confirmation; excluded from autograd",
    }
    return {
        "records": records,
        "models": models,
        "predictions": predictions_for_tests,
        "fit_labels": fit_labels,
        "observation_labels": observation_labels,
    }


def _adjust_endpoint_tests(endpoint_probes: Mapping[str, Any], labels: torch.Tensor) -> dict[str, Any]:
    tests: dict[str, float] = {}
    for probe_name in PROBE_NAMES:
        predictions = endpoint_probes["predictions"][probe_name]
        for class_id in FOCUS_CLASSES:
            name = f"{probe_name}:class_{class_id}"
            tests[name] = _exact_fixed_prediction_pvalue(labels, predictions, class_id)
    return {"raw_p": tests, "holm_adjusted_p": _holm_adjust(tests)}


def _support_records(endpoint_probes: Mapping[str, Any], tests: Mapping[str, Any]) -> dict[str, Any]:
    support: dict[str, dict[str, bool]] = {}
    adjusted = tests["holm_adjusted_p"]
    for probe_name in PROBE_NAMES:
        support[probe_name] = {}
        for class_id in FOCUS_CLASSES:
            pvalue = float(adjusted[f"{probe_name}:class_{class_id}"])
            metrics = endpoint_probes["records"][probe_name]["observation"]
            if probe_name == "e_d":
                support[probe_name][str(class_id)] = _ed_support(metrics, pvalue, class_id)
            else:
                negative = endpoint_probes["records"][probe_name]["label_shuffle_negative_control"]
                support[probe_name][str(class_id)] = _probe_support(metrics, negative, pvalue, class_id)
    return support


def _gradients_for_loss(
    loss: torch.Tensor,
    parameters: Mapping[str, torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> dict[str, torch.Tensor]:
    values = torch.autograd.grad(
        loss,
        tuple(parameters[name] for name in ("encoder.phase", "generator.phase")),
        retain_graph=retain_graph,
        allow_unused=True,
    )
    result: dict[str, torch.Tensor] = {}
    for name, parameter, value in zip(("encoder.phase", "generator.phase"), parameters.values(), values):
        result[name] = torch.zeros_like(parameter) if value is None else value.detach()
    return result


def _safe_canonical_gradient(
    gradients: Mapping[str, Mapping[str, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor] | None, str | None]:
    flattened: list[torch.Tensor] = []
    for evaluator_name in ("L_S", "L_I", "L_C"):
        vector = _flatten_group(gradients[evaluator_name])
        norm = float(vector.to(torch.float64).norm().cpu())
        if not math.isfinite(norm) or norm <= 0.0:
            return None, f"{evaluator_name} gradient is zero or invalid"
        flattened.append(vector / norm)
    joint = flattened[0] + flattened[1] + flattened[2]
    rms = float(joint.to(torch.float64).square().mean().sqrt().cpu())
    if not math.isfinite(rms) or rms <= 0.0:
        return None, "equal-weight evaluator direction is zero or invalid"
    scaled = joint / rms * DELTA_SIC
    return _split_group(scaled, {name: gradients["L_S"][name].shape for name in ("encoder.phase", "generator.phase")}), None


def _batch_direction(
    consumer: Any,
    optimizer: torch.optim.Optimizer,
    parameters: Mapping[str, torch.nn.Parameter],
    references: Any,
    evaluators: Mapping[str, torch.nn.Module],
    images: torch.Tensor,
    batch_index: int,
) -> dict[str, Any]:
    batch = images[batch_index * LOGICAL_BATCH : (batch_index + 1) * LOGICAL_BATCH].to(consumer.device)
    breakdown = consumer.forward_loss(batch, data_role=bogt.TRAIN_ROLE)
    semantic_losses = _loss_reductions(
        {"result": breakdown.result},
        batch,
        evaluators["E_S"],
        evaluators["E_I"],
        evaluators["E_C"],
    )
    losses = {
        "L_dist": breakdown.L_dist,
        "L_rec": breakdown.L_rec,
        "L_op": breakdown.L_op,
        **semantic_losses,
    }
    gradients: dict[str, dict[str, torch.Tensor]] = {}
    names = tuple(losses)
    for index, loss_name in enumerate(names):
        gradients[loss_name] = _gradients_for_loss(
            losses[loss_name],
            parameters,
            retain_graph=index < len(names) - 1,
        )
    shapes = {name: parameters[name].shape for name in ("encoder.phase", "generator.phase")}
    weights = consumer.weights
    _require(weights is not None, "BOGT weights are not bound during gradient attribution")
    grad_bogt = {
        name: (
            weights.alpha * gradients["L_dist"][name]
            + weights.beta * gradients["L_rec"][name]
            + weights.gamma * gradients["L_op"][name]
        )
        for name in shapes
    }
    canonical_gradient, canonical_unavailable = _safe_canonical_gradient(gradients)
    grad_total = None
    adam_saved = None
    adam_zero = None
    projected_adam = None
    projection_record: dict[str, Any] | None = None
    if canonical_gradient is not None:
        grad_total = {
            name: grad_bogt[name] + KAPPA * canonical_gradient[name]
            for name in shapes
        }
        adam_saved = _adam_candidate(grad_total, optimizer, parameters)
        adam_zero = _adam_candidate(grad_total, optimizer, parameters, zero_history=True)
        flat_bogt = _flatten_group(grad_bogt)
        flat_semantic = _flatten_group(canonical_gradient)
        safe_bogt, projection_record = canonical.project_auxiliary_gradient(
            flat_bogt,
            flat_semantic,
            "BOGT auxiliary gradient",
        )
        projected_gradient = _split_group(
            KAPPA * flat_semantic + safe_bogt,
            {name: parameters[name].shape for name in shapes},
        )
        projected_adam = _adam_candidate(projected_gradient, optimizer, parameters)
    _require(all(parameter.grad is None for parameter in parameters.values()), "read-only batched autograd populated an optimizer .grad buffer")
    row_positions = list(range(batch_index * LOGICAL_BATCH, (batch_index + 1) * LOGICAL_BATCH))
    return {
        "batch_index": batch_index,
        "fit_positions": row_positions,
        "source_row_indices": None,
        "labels": None,
        "losses": {name: float(value.detach().cpu()) for name, value in losses.items()},
        "gradient_components": {
            name: {group: value.detach().cpu() for group, value in groups.items()}
            for name, groups in {
                **gradients,
                "G_BOGT": grad_bogt,
                **({"G_canonical": canonical_gradient} if canonical_gradient is not None else {}),
                **({"G_total": grad_total} if grad_total is not None else {}),
            }.items()
        },
        "canonical_unavailable_reason": canonical_unavailable,
        "projection": projection_record,
        "candidate_displacements": {
            name: {group: value.detach().cpu() for group, value in candidate.items()}
            for name, candidate in (
                ("saved_adam", adam_saved),
                ("zero_history_adam", adam_zero),
                ("projected_adam", projected_adam),
            )
            if candidate is not None
        },
        "raw_gradient_descent": None if grad_total is None else {name: (-value).detach().cpu() for name, value in grad_total.items()},
    }


def _single_row_gradients(
    consumer: Any,
    evaluators: Mapping[str, torch.nn.Module],
    image: torch.Tensor,
    parameters: Mapping[str, torch.nn.Parameter],
) -> dict[str, Any]:
    batch = image.unsqueeze(0).to(consumer.device)
    result = consumer.runtime.forward_from_image(batch)
    raw = result.raw_intensity
    target_raw = result.image_input.slm_intensity
    raw_view = canonical.cmos_view(raw)
    target_view = canonical.cmos_view(image.unsqueeze(0).to(consumer.device))
    s_generated, _ = evaluators["E_S"](raw_view)
    i_generated, _ = evaluators["E_I"](raw_view)
    c_generated, _ = evaluators["E_C"](raw_view)
    with torch.no_grad():
        s_target, _ = evaluators["E_S"](target_view)
        i_target, _ = evaluators["E_I"](target_view)
        c_target, _ = evaluators["E_C"](target_view)
    losses = {
        "L_rec": (raw - target_raw).square().mean(),
        "L_S": canonical.normalized_feature_loss(s_generated, s_target, "per-row L_S"),
        "L_I": canonical.normalized_feature_loss(i_generated, i_target, "per-row L_I"),
        "L_C": canonical.normalized_feature_loss(c_generated, c_target, "per-row L_C"),
    }
    names = tuple(losses)
    gradients = {
        name: _gradients_for_loss(loss, parameters, retain_graph=index < len(names) - 1)
        for index, (name, loss) in enumerate(losses.items())
    }
    _require(all(parameter.grad is None for parameter in parameters.values()), "read-only per-row autograd populated an optimizer .grad buffer")
    sem = {
        name: sum(gradients[key][name] for key in ("L_S", "L_I", "L_C")) / 3.0
        for name in ("encoder.phase", "generator.phase")
    }
    return {
        "losses": {name: float(value.detach().cpu()) for name, value in losses.items()},
        "gradients": {
            name: {group: value.detach().cpu() for group, value in values.items()}
            for name, values in gradients.items()
        },
        "semantic_diagnostic_gradient": {name: value.detach().cpu() for name, value in sem.items()},
        "semantic_diagnostic_definition": "equal-weight arithmetic mean of per-row gradients of L_S, L_I, and L_C; diagnostic only, not a canonical training scalar",
    }


def _per_row_direction_records(
    row_gradients: Sequence[Mapping[str, Any]],
    batches: Sequence[Mapping[str, Any]],
    labels: torch.Tensor,
    source_indices: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    flat_rows: list[dict[str, Any]] = []
    for position, gradients in enumerate(row_gradients):
        class_id = int(labels[position].item())
        if class_id not in (*FOCUS_CLASSES, *CONTROL_CLASSES):
            continue
        batch_index = position // LOGICAL_BATCH
        batch = batches[batch_index]
        semantic = _flatten_group(gradients["semantic_diagnostic_gradient"])
        candidates: dict[str, torch.Tensor] = {}
        if batch["raw_gradient_descent"] is not None:
            candidates["raw_gradient_descent"] = _flatten_group(batch["raw_gradient_descent"])
        for name, groups in batch["candidate_displacements"].items():
            candidates[name] = _flatten_group(groups)
        if "G_canonical" in batch["gradient_components"]:
            candidates["canonical_descent"] = -_flatten_group(batch["gradient_components"]["G_canonical"])
        directions = {name: _direction_record(semantic, value) for name, value in candidates.items()}
        row = {
            "fit_position": position,
            "batch_index": batch_index,
            "source_row_index": int(source_indices[position].item()),
            "label": class_id,
            "losses": dict(gradients["losses"]),
            "directions_vs_per_row_semantic_gradient": directions,
        }
        flat_rows.append(row)
        records.setdefault(str(class_id), []).append(row)
    summaries = {
        class_id: {
            direction: _direction_summary(
                [row["directions_vs_per_row_semantic_gradient"][direction] for row in class_rows if direction in row["directions_vs_per_row_semantic_gradient"]]
            )
            for direction in sorted({name for row in class_rows for name in row["directions_vs_per_row_semantic_gradient"]})
        }
        for class_id, class_rows in records.items()
    }
    return summaries, flat_rows


def _aggregate_image_statistics(
    images: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    source_indices: torch.Tensor,
) -> dict[str, Any]:
    per_row = _row_reconstruction_metrics(
        images["raw_cmos"],
        images["target_slm_intensity"],
        images["input_images"],
    )

    def summarize(positions: torch.Tensor) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for key, tensor in per_row.items():
            if tensor.ndim == 1 and tensor.shape[0] == labels.shape[0]:
                selected = tensor.index_select(0, positions)
                if tensor.dtype == torch.bool:
                    values[key] = int(selected.sum().item())
                else:
                    finite = selected[torch.isfinite(selected.to(torch.float64))]
                    values[key] = float(finite.to(torch.float64).mean().item()) if finite.numel() else None
        values["rows"] = int(positions.numel())
        values["source_row_indices"] = source_indices.index_select(0, positions).tolist()
        return values

    by_class = {
        str(class_id): {
            "fit": summarize(torch.nonzero(labels[:FIT_ROWS] == class_id, as_tuple=False).flatten()),
            "observation": summarize(FIT_ROWS + torch.nonzero(labels[FIT_ROWS:] == class_id, as_tuple=False).flatten()),
            "combined": summarize(torch.nonzero(labels == class_id, as_tuple=False).flatten()),
        }
        for class_id in range(10)
    }
    return {
        "overall": summarize(torch.arange(labels.numel())),
        "by_class": by_class,
        "raw_200x200_mse_and_zero_constant_clip_metrics": True,
        "evaluator_28x28_mse_uses_clamp_pool_view": True,
        "foreground_definition": "target SLM intensity > 0.1 at the corresponding resolution",
    }, per_row


def _save_montage(
    path: Path,
    images: torch.Tensor,
    labels: torch.Tensor,
    readouts: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    target = images
    view1200 = canonical.cmos_view(readouts["step1200"]["raw_cmos"])
    view2200 = canonical.cmos_view(readouts["step2200"]["raw_cmos"])
    positions = [
        position
        for class_id in DISPLAY_CLASSES
        for position in torch.nonzero(labels == class_id, as_tuple=False).flatten().tolist()
    ]
    _require(len(positions) == 150, "montage must show all 150 rows from the five declared classes")
    scale = 2
    image_side = 28 * scale
    tile_width = image_side * 3 + 26
    tile_height = image_side + 10
    columns = 10
    rows = math.ceil(len(positions) / columns)
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    def as_image(tensor: torch.Tensor) -> Image.Image:
        pixels = tensor.detach().cpu().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
        return Image.fromarray(pixels, mode="L").resize((image_side, image_side), Image.Resampling.NEAREST).convert("RGB")

    for display_index, source_position in enumerate(positions):
        x = (display_index % columns) * tile_width
        y = (display_index // columns) * tile_height
        class_id = int(labels[source_position].item())
        split_name = "F" if source_position < FIT_ROWS else "O"
        draw.text((x, y), f"{class_id}{split_name}", fill=(0, 0, 0))
        for image_index, view in enumerate((target, view1200, view2200)):
            image = as_image(view[source_position, 0])
            canvas.paste(image, (x + 22 + image_index * image_side, y))
    canvas.save(path, format="PNG", optimize=True)
    return {
        "path": path.name,
        "rows_displayed": 150,
        "class_order": list(DISPLAY_CLASSES),
        "fit_observation_marker": {"F": "fit", "O": "observation"},
        "panel_order": ["target", "step1200", "step2200"],
        "view": "28x28 canonical clamp[0,1] then adaptive average pool",
        "fixed_display_range": [0.0, 1.0],
        "per_image_rescaling": False,
        "source_row_indices_in_display_order": [int(readouts["step1200"]["source_indices"][index]) for index in positions],
    }


def _check_output_budget(target: Path) -> dict[str, Any]:
    existing = target
    while not existing.exists():
        existing = existing.parent
    disk = shutil.disk_usage(existing)
    free_bytes = int(disk.free)
    _require(free_bytes >= 1_500_000_000, "PREFLIGHT_BLOCKED(resource): output volume has less than 1.5 GB free")
    return {"checked_path": existing.as_posix(), "free_bytes": free_bytes, "minimum_free_bytes": 1_500_000_000}


def _check_gpu_budget(device: torch.device) -> dict[str, Any]:
    snapshot = _resource_snapshot(device)
    _require(snapshot["global_used_fraction"] <= 0.90, "PREFLIGHT_BLOCKED(resource): whole-device use exceeds the 90% safety line")
    return snapshot


def sample_resource_telemetry(
    path: Path,
    device: torch.device,
    phase: str,
    *,
    progress: Mapping[str, Any] | None = None,
    started: float | None = None,
) -> dict[str, Any]:
    """Append one whole-device and PyTorch allocator observation."""
    sample = {
        "task_id": TASK_ID,
        "phase": phase,
        "progress": dict(progress or {}),
        "elapsed_seconds_observation_only": None if started is None else time.perf_counter() - started,
        "resource": _resource_snapshot(device),
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
    _require(sample["resource"]["global_used_fraction"] <= 0.90, "PREFLIGHT_BLOCKED(resource): whole-device use exceeded the 90% safety line")
    return sample


def _observed_task_duration(target: Path, started: float, *, recovered: bool) -> float:
    observed = time.perf_counter() - started
    if not recovered:
        return observed
    telemetry_path = target / "resource_telemetry.jsonl"
    if not telemetry_path.is_file():
        return observed
    durations = []
    for line in telemetry_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = _mapping(json.loads(line), "resource telemetry record")
        if record.get("task_id") == TASK_ID:
            durations.append(float(record["elapsed_seconds_observation_only"]))
    return max([observed, *durations])


def _write_json_new(path: Path, payload: Mapping[str, Any], kind: str) -> None:
    from ldm_project.v2.evidence import atomic_write_new_json

    atomic_write_new_json(path, dict(payload), artifact_kind=kind)
    written = json.loads(path.read_text(encoding="utf-8"))
    _require(written == dict(payload), f"write/read lifecycle verification failed for {path.name}")


def _write_torch_new(path: Path, payload: Mapping[str, Any], kind: str) -> None:
    from ldm_project.v2.evidence import atomic_torch_save_new

    atomic_torch_save_new(path, dict(payload), artifact_kind=kind)
    restored = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(restored, Mapping), f"write/read lifecycle payload is invalid for {path.name}")
    _require(set(restored) == set(payload), f"write/read lifecycle keys changed for {path.name}")


def _next_artifact_path(target: Path, stem: str, suffix: str = ".json") -> Path:
    candidate = target / f"{stem}{suffix}"
    revision = 2
    while candidate.exists():
        candidate = target / f"{stem}_recheck_{revision:02d}{suffix}"
        revision += 1
    return candidate


def _preflight(
    inputs: Mapping[str, Any],
    paths: Mapping[str, Path],
    target: Path,
) -> dict[str, Any]:
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "PREFLIGHT_BLOCKED(resource): CUDA device 0 is unavailable")
    device = torch.device("cuda:0")
    output_budget = _check_output_budget(target)
    gpu_before = _check_gpu_budget(device)
    preflight_started = time.perf_counter()
    telemetry_path = target / "resource_telemetry.jsonl"
    sample_resource_telemetry(telemetry_path, device, "preflight_start", progress={"rows": 0}, started=preflight_started)
    evaluators = load_evaluators(paths, device)
    evaluator_states = {name: _state_snapshot(model) for name, model in evaluators.items()}
    preflight_endpoints: dict[str, Any] = {}
    peak_allocated = 0
    peak_reserved = 0
    for name, _step in ENDPOINTS:
        consumer, optimizer, parameters, references, _weights = _new_endpoint(
            name,
            inputs["payloads"][name],
            inputs["references"],
            inputs["weights"],
            device,
        )
        first_batch = inputs["images"][:LOGICAL_BATCH].to(device)
        with torch.no_grad():
            warm_result = consumer.runtime.forward_from_image(first_batch)
            warm_view = canonical.cmos_view(warm_result.raw_intensity)
            _ = evaluators["E_D"](warm_view)
        del warm_result, warm_view
        torch.cuda.synchronize(device)
        before_runtime = _state_snapshot(consumer.runtime)
        before_optimizer = _optimizer_snapshot(optimizer)
        before_references = consumer.references.to_payload()
        before_weights = consumer.weights.to_payload()
        _check_gpu_budget(device)
        direction = _batch_direction(consumer, optimizer, parameters, references, evaluators, inputs["images"][:FIT_ROWS], 0)
        _require(direction["gradient_components"]["L_S"] is not None, "preflight did not exercise E_S autograd")
        _require(direction["gradient_components"]["L_I"] is not None, "preflight did not exercise E_I autograd")
        _require(direction["gradient_components"]["L_C"] is not None, "preflight did not exercise E_C autograd")
        _require(direction["gradient_components"]["L_dist"]["generator.phase"].abs().sum().item() == 0.0, "L_dist-to-Generator structural zero changed")
        _require("saved_adam" in direction["candidate_displacements"], "preflight did not construct read-only saved-Adam candidate")
        _require(all(parameter.grad is None for parameter in parameters.values()), "preflight left optical .grad buffers populated")
        _assert_tree_equal(before_optimizer, _optimizer_snapshot(optimizer), f"{name} preflight Adam state")
        _require(_state_equal(before_runtime, _state_snapshot(consumer.runtime)), f"{name} preflight optical state changed")
        _assert_tree_equal(before_references, consumer.references.to_payload(), f"{name} frozen references")
        _assert_tree_equal(before_weights, consumer.weights.to_payload(), f"{name} BOGT weights")

        # Reapply the named endpoint state to exercise the same explicit restore boundary used by resume.
        consumer.load_trainable_state_dict(inputs["payloads"][name]["trainable_parameter_state"])
        optimizer.load_state_dict(copy.deepcopy(inputs["payloads"][name]["optimizer_state"]))
        _assert_tree_equal(before_optimizer, _optimizer_snapshot(optimizer), f"{name} checkpoint restore")
        _require(_state_equal(before_runtime, _state_snapshot(consumer.runtime)), f"{name} checkpoint restore changed runtime state")
        _require(all(parameter.grad is None for parameter in parameters.values()), "checkpoint restore introduced stale optical gradients")

        resource = _check_gpu_budget(device)
        sample_resource_telemetry(
            telemetry_path,
            device,
            f"preflight_{name}_read_only_autograd",
            progress={"checkpoint_step": _step, "fit_rows": LOGICAL_BATCH, "optimizer_steps": 0},
            started=preflight_started,
        )
        peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated(device)))
        peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved(device)))
        preflight_endpoints[name] = {
            "checkpoint_step": int(_step),
            "full_production_forward": True,
            "target_evaluator_features_in_no_grad": True,
            "three_training_evaluator_autograd": True,
            "e_d_forward_only": True,
            "read_only_adam_candidate": True,
            "optimizer_step_calls": 0,
            "encoder_generator_state_unchanged": True,
            "frozen_runtime_state_unchanged": True,
            "source_adam_state_unchanged": True,
            "restore_boundary_reloaded_exactly": True,
            "resource_after_endpoint": resource,
            "probe_direction": {
                "canonical_available": direction["canonical_unavailable_reason"] is None,
                "canonical_unavailable_reason": direction["canonical_unavailable_reason"],
            },
        }
        del direction, consumer, optimizer, parameters, references
        gc.collect()
        torch.cuda.empty_cache()
    for name, model in evaluators.items():
        _require(_state_equal(evaluator_states[name], _state_snapshot(model)), f"{name} evaluator parameters changed during preflight")
        _require(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()), f"{name} evaluator left the frozen boundary")
    gpu_after = _check_gpu_budget(device)
    sample_resource_telemetry(telemetry_path, device, "preflight_ready", progress={"endpoints": len(preflight_endpoints), "status": "READY"}, started=preflight_started)
    return {
        "schema": PREFLIGHT_SCHEMA,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "status": "READY",
        "device": str(device),
        "scope": "no optical optimizer.step; no OLS coefficient update; fixed TRAIN rows only",
        "output_budget": output_budget,
        "telemetry_artifact": telemetry_path.name,
        "cuda_before": gpu_before,
        "cuda_after": gpu_after,
        "pytorch_peak_allocated_bytes": peak_allocated,
        "pytorch_peak_reserved_bytes": peak_reserved,
        "whole_device_safety_fraction_max": max(gpu_before["global_used_fraction"], gpu_after["global_used_fraction"]),
        "endpoints": preflight_endpoints,
        "evaluator_state_unchanged": True,
        "duration_seconds_observation_only": None,
    }


def _population_record(inputs: Mapping[str, Any], paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "input_paths": {name: paths[name].as_posix() for name in sorted(paths)},
        "row_contract": dict(inputs["row_contract"]),
        "source_row_indices": inputs["source_indices"].tolist(),
        "labels": inputs["labels"].tolist(),
        "historical_readouts": dict(inputs["historical_readout_provenance"]),
        "step1200_reference_and_weight_inheritance": {
            "step1200_has_shared_values": True,
            "step2200_references_and_weights": "explicitly inherited from step1200",
            "source_manifest_ids_are_provenance_only": True,
        },
        "checkpoint_data_identity": {
            name: {
                key: value
                for key, value in identity.items()
                if key in ("source_path", "seed", "split", "train_count", "development_count", "train_order", "protected_test_read", "sealed_read")
            }
            for name, identity in inputs["checkpoint_data"].items()
        },
    }


def _load_stage(target: Path, name: str) -> Any | None:
    path = target / name
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


def _fresh_readout_stage(
    target: Path,
    endpoint_name: str,
    consumer: Any,
    evaluators: Mapping[str, torch.nn.Module],
    inputs: Mapping[str, Any],
    references: Any,
) -> dict[str, torch.Tensor]:
    stage_name = f"readouts_{endpoint_name}.pt"
    saved = _load_stage(target, stage_name)
    if saved is not None:
        saved_map = _mapping(saved, stage_name)
        _require(torch.equal(saved_map.get("source_indices"), inputs["source_indices"]), f"{stage_name} source rows changed")
        _require(torch.equal(saved_map.get("labels"), inputs["labels"]), f"{stage_name} labels changed")
        _require(torch.equal(saved_map.get("input_images"), inputs["images"]), f"{stage_name} selected target images changed")
        return dict(saved_map)
    readouts = _forward_readouts(consumer, evaluators, inputs["images"])
    readouts["source_indices"] = inputs["source_indices"].clone()
    readouts["labels"] = inputs["labels"].clone()
    readouts["input_images"] = inputs["images"].clone()
    _probe_features(readouts["z"], readouts["raw_cmos"], references)
    _write_torch_new(target / stage_name, readouts, f"fresh frozen {endpoint_name} TRAIN readouts")
    return readouts


def _image_metric_stage(
    target: Path,
    endpoint_name: str,
    per_row_metrics: Mapping[str, torch.Tensor],
    inputs: Mapping[str, Any],
) -> str:
    stage_name = f"image_metrics_{endpoint_name}.pt"
    saved = _load_stage(target, stage_name)
    if saved is not None:
        saved_map = _mapping(saved, stage_name)
        _require(torch.equal(saved_map.get("source_indices"), inputs["source_indices"]), f"{stage_name} source rows changed")
        _require(torch.equal(saved_map.get("labels"), inputs["labels"]), f"{stage_name} labels changed")
        return stage_name
    scalar_rows = {
        name: value.detach().cpu()
        for name, value in per_row_metrics.items()
        if value.ndim == 1 and value.shape[0] == OBSERVATION_ROWS + FIT_ROWS
    }
    payload = {
        "endpoint": endpoint_name,
        "source_indices": inputs["source_indices"],
        "labels": inputs["labels"],
        "per_row_metrics": scalar_rows,
        "raw_cmos_coordinate_order": "200x200 row-major; axis -2=row, axis -1=column",
        "evaluator_view": "canonical clamp[0,1] then adaptive average pool to 28x28",
        "target_view": "original selected TRAIN image at 28x28 in [0,1]",
        "foreground_definition": "target intensity > 0.1",
    }
    _write_torch_new(target / stage_name, payload, f"{endpoint_name} per-row raw and view image metrics")
    return stage_name


def _target_e_d_stage(
    target: Path,
    e_d: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    source_indices: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any], str]:
    path = target / "target_e_d_positive_control.pt"
    if path.is_file():
        saved = _mapping(torch.load(path, map_location="cpu", weights_only=True), path.name)
        _require(torch.equal(saved.get("labels"), labels), "target E_D positive-control labels changed")
        _require(torch.equal(saved.get("source_indices"), source_indices), "target E_D positive-control source rows changed")
        _require(torch.equal(saved.get("target_images"), images), "target E_D positive-control images changed")
        logits = saved.get("logits")
        _require(torch.is_tensor(logits) and logits.shape == (300, 10), "saved target E_D logits shape changed")
        metrics = dict(_mapping(saved.get("metrics"), "target E_D metrics"))
        return logits, metrics, path.name
    logits = _target_e_d_logits(e_d, images, device)
    metrics = _metrics_with_intervals(labels, logits.argmax(dim=1))
    _write_torch_new(
        path,
        {
            "source": "the fixed 300 real TRAIN target images",
            "source_indices": source_indices,
            "labels": labels,
            "target_images": images,
            "logits": logits,
            "metrics": metrics,
            "role": "positive-control qualification only; no generated model attribution",
        },
        "target E_D positive-control readout",
    )
    return logits, metrics, path.name


def _probe_stage(
    target: Path,
    endpoint_name: str,
    readouts: Mapping[str, torch.Tensor],
    inputs: Mapping[str, Any],
    references: Any,
) -> dict[str, Any]:
    stage_name = f"probes_{endpoint_name}.pt"
    saved = _load_stage(target, stage_name)
    if saved is not None:
        saved_map = _mapping(saved, stage_name)
        _require(saved_map.get("fit_rows") == FIT_ROWS and saved_map.get("observation_rows") == OBSERVATION_ROWS, f"{stage_name} split contract changed")
        _require(torch.equal(saved_map.get("source_indices"), inputs["source_indices"]), f"{stage_name} source rows changed")
        _require(torch.equal(saved_map.get("labels"), inputs["labels"]), f"{stage_name} labels changed")
        return dict(saved_map)
    probe_values = _fit_probes(readouts, inputs["images"], inputs["labels"], references)
    result = {
        "fit_rows": FIT_ROWS,
        "observation_rows": OBSERVATION_ROWS,
        "source_indices": inputs["source_indices"],
        "labels": inputs["labels"],
        "records": probe_values["records"],
        "models": probe_values["models"],
        "predictions": probe_values["predictions"],
    }
    _write_torch_new(target / stage_name, result, f"{endpoint_name} probes and fixed negative control")
    return result


def _direction_stage(
    target: Path,
    endpoint_name: str,
    consumer: Any,
    optimizer: torch.optim.Optimizer,
    parameters: Mapping[str, torch.nn.Parameter],
    references: Any,
    evaluators: Mapping[str, torch.nn.Module],
    inputs: Mapping[str, Any],
    runtime_snapshot: Mapping[str, torch.Tensor],
    optimizer_snapshot: Mapping[str, Any],
    telemetry_path: Path,
    started: float,
) -> dict[str, Any]:
    stage_name = f"directions_{endpoint_name}.pt"
    saved = _load_stage(target, stage_name)
    if saved is not None:
        saved_map = _mapping(saved, stage_name)
        _require(saved_map.get("fit_rows") == FIT_ROWS and saved_map.get("endpoint") == endpoint_name, f"{stage_name} identity mismatch")
        _require(torch.equal(saved_map.get("source_indices"), inputs["source_indices"]), f"{stage_name} source rows changed")
        _require(torch.equal(saved_map.get("labels"), inputs["labels"]), f"{stage_name} labels changed")
        return dict(saved_map)
    batches: list[dict[str, Any]] = []
    for batch_index in range(FIT_ROWS // LOGICAL_BATCH):
        _check_gpu_budget(consumer.device)
        batch = _batch_direction(
            consumer,
            optimizer,
            parameters,
            references,
            evaluators,
            inputs["images"][:FIT_ROWS],
            batch_index,
        )
        start = batch_index * LOGICAL_BATCH
        batch["source_row_indices"] = inputs["source_indices"][start : start + LOGICAL_BATCH].tolist()
        batch["labels"] = inputs["labels"][start : start + LOGICAL_BATCH].tolist()
        batches.append(batch)
        del batch
        if batch_index % 5 == 4:
            gc.collect()
            torch.cuda.empty_cache()
            _check_gpu_budget(consumer.device)
            sample_resource_telemetry(
                telemetry_path,
                consumer.device,
                f"{endpoint_name}_batch_gradient",
                progress={"completed_batches": batch_index + 1, "target_batches": FIT_ROWS // LOGICAL_BATCH},
                started=started,
            )
    row_gradients: list[dict[str, Any] | None] = [None] * FIT_ROWS
    focus_positions = [
        position
        for position, class_id in enumerate(inputs["labels"][:FIT_ROWS].tolist())
        if class_id in (*FOCUS_CLASSES, *CONTROL_CLASSES)
    ]
    for index, position in enumerate(focus_positions):
        _check_gpu_budget(consumer.device)
        row_gradients[position] = _single_row_gradients(
            consumer,
            evaluators,
            inputs["images"][position],
            parameters,
        )
        if index % 10 == 9:
            gc.collect()
            torch.cuda.empty_cache()
            _check_gpu_budget(consumer.device)
    compact_rows = [row_gradients[position] for position in focus_positions]
    _require(all(value is not None for value in compact_rows), f"{endpoint_name} per-row gradients are incomplete")
    expanded_rows: list[dict[str, Any]] = []
    by_position: dict[int, dict[str, Any]] = dict(zip(focus_positions, compact_rows))  # type: ignore[arg-type]
    for position in range(FIT_ROWS):
        if position in by_position:
            expanded_rows.append(by_position[position])
        else:
            expanded_rows.append({"unused_class": int(inputs["labels"][position].item())})
    summaries, row_records = _per_row_direction_records(
        expanded_rows,
        batches,
        inputs["labels"][:FIT_ROWS],
        inputs["source_indices"][:FIT_ROWS],
    )
    _require(_state_equal(runtime_snapshot, _state_snapshot(consumer.runtime)), f"{endpoint_name} runtime state changed during direction measurement")
    _assert_tree_equal(optimizer_snapshot, _optimizer_snapshot(optimizer), f"{endpoint_name} source Adam state")
    _require(all(parameter.grad is None for parameter in parameters.values()), f"{endpoint_name} direction measurement populated .grad")
    result = {
        "endpoint": endpoint_name,
        "fit_rows": FIT_ROWS,
        "source_indices": inputs["source_indices"],
        "labels": inputs["labels"],
        "balanced_batches": len(batches),
        "batches": batches,
        "per_row_gradients": expanded_rows,
        "per_row_direction_records": row_records,
        "per_class_direction_summary": summaries,
        "L_dist_to_generator": "structural zero: generator phase is absent from the OLS distribution loss graph",
        "semantic_diagnostic": "per-row equal-weight arithmetic mean of L_S, L_I, and L_C gradients; not a scalar training objective",
        "optimizer_steps": 0,
        "state_unchanged": True,
    }
    _write_torch_new(target / stage_name, result, f"{endpoint_name} read-only gradients and candidate directions")
    return result


def _global_probe_tests(
    probes_by_step: Mapping[str, Mapping[str, Any]],
    labels: torch.Tensor,
) -> dict[str, Any]:
    pvalues: dict[str, float] = {}
    for endpoint_name, endpoint_probes in probes_by_step.items():
        for probe_name in PROBE_NAMES:
            predictions = endpoint_probes["predictions"][probe_name]
            for class_id in FOCUS_CLASSES:
                pvalues[f"{endpoint_name}|{probe_name}:class_{class_id}"] = _exact_fixed_prediction_pvalue(
                    labels[FIT_ROWS:],
                    predictions,
                    class_id,
                )
    _require(len(pvalues) == 20, "family-wise test count is not exactly 20")
    return {"family": "fixed predicted-class counts; exact upper-tail label permutation", "test_count": len(pvalues), "raw_p": pvalues, "holm_adjusted_p": _holm_adjust(pvalues)}


def _support_for_endpoint(
    endpoint_name: str,
    endpoint_probes: Mapping[str, Any],
    global_tests: Mapping[str, Any],
) -> dict[str, Any]:
    adjusted = global_tests["holm_adjusted_p"]
    support: dict[str, dict[str, bool]] = {}
    for probe_name in PROBE_NAMES:
        support[probe_name] = {}
        metrics = endpoint_probes["records"][probe_name]["observation"]
        for class_id in FOCUS_CLASSES:
            pvalue = float(adjusted[f"{endpoint_name}|{probe_name}:class_{class_id}"])
            if probe_name == "e_d":
                supported = _ed_support(metrics, pvalue, class_id)
            else:
                negative = endpoint_probes["records"][probe_name]["label_shuffle_negative_control"]
                supported = _probe_support(metrics, negative, pvalue, class_id)
            support[probe_name][str(class_id)] = supported
    return support


def _qualification(
    probes_by_step: Mapping[str, Mapping[str, Any]],
    target_e_d_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    target_probe_macro = min(
        float(endpoint_probes["records"]["target_positive_control"]["observation"]["macro_recall"])
        for endpoint_probes in probes_by_step.values()
    )
    ed_class_recall = {
        str(class_id): float(target_e_d_metrics["per_class_recall"][class_id])
        for class_id in FOCUS_CLASSES
    }
    qualified = (
        target_probe_macro >= 0.50
        and float(target_e_d_metrics["macro_recall"]) >= 0.90
        and all(recall >= 0.80 for recall in ed_class_recall.values())
    )
    return {
        "qualified": qualified,
        "target_probe_macro_recall_minimum": target_probe_macro,
        "target_probe_minimum_required": 0.50,
        "target_e_d": target_e_d_metrics,
        "target_e_d_requirements": {"macro_recall_minimum": 0.90, "class_1_and_7_recall_minimum_each": 0.80},
        "target_e_d_class_1_and_7_recall": ed_class_recall,
        "reason": None if qualified else "real target positive control did not meet its declared threshold; model attribution is withheld",
    }


def _direction_mode(
    direction: Mapping[str, Any],
    class_id: int,
    candidate: str,
) -> str | None:
    return direction.get("per_class_direction_summary", {}).get(str(class_id), {}).get(candidate, {}).get("stable_mode")


def _final_decision(
    probes_by_step: Mapping[str, Mapping[str, Any]],
    support_by_step: Mapping[str, Mapping[str, Any]],
    global_tests: Mapping[str, Any],
    qualification: Mapping[str, Any],
    directions_by_step: Mapping[str, Mapping[str, Any]],
    image_stats: Mapping[str, Any],
) -> dict[str, Any]:
    if not qualification["qualified"]:
        return {
            "classification": "MEASUREMENT_QUALIFICATION_INSUFFICIENT",
            "training_recommendation": "NONE",
            "reason": qualification["reason"],
            "stop": True,
        }
    primary = support_by_step["step2200"]
    same_gauge_support = all(primary["ols_gauge"][str(class_id)] for class_id in FOCUS_CLASSES)
    step2200_direction = directions_by_step["step2200"]

    if same_gauge_support and all(
        _direction_mode(step2200_direction, class_id, "raw_gradient_descent") == "BENEFICIAL"
        and _direction_mode(step2200_direction, class_id, "saved_adam") == "HARMFUL"
        and _direction_mode(step2200_direction, class_id, "zero_history_adam") == "BENEFICIAL"
        for class_id in FOCUS_CLASSES
    ):
        return {
            "classification": "OPTIMIZER_STATE_DIRECTION_CANDIDATE",
            "training_recommendation": "在相同 step2200 模型、目标、数据和更新数下，仅比较保留与重置 Adam 状态。",
            "reason": "两类去全局相位 OLS 探针均获支持，合成梯度下降与零历史 Adam 方向有利，而保存状态 Adam 候选方向有害，且每项达到至少16/20稳定一致。",
            "stop": True,
        }

    controls_clear = all(
        _direction_mode(step2200_direction, class_id, "projected_adam") != "HARMFUL"
        for class_id in CONTROL_CLASSES
    )
    if all(
        _direction_mode(step2200_direction, class_id, "canonical_descent") == "BENEFICIAL"
        and _direction_mode(step2200_direction, class_id, "saved_adam") == "HARMFUL"
        and _direction_mode(step2200_direction, class_id, "projected_adam") == "BENEFICIAL"
        for class_id in FOCUS_CLASSES
    ) and controls_clear:
        return {
            "classification": "TRAINING_DIRECTION_INTERFERENCE_CANDIDATE",
            "training_recommendation": "下一任务只比较当前合成规则与将 BOGT 辅助梯度投影到不反对语义方向的规则。",
            "reason": "两类语义下降方向有利，保存 Adam 候选有害，投影候选恢复有利；比较类别未出现新增稳定有害模式。",
            "stop": True,
        }

    both_views_supported = all(
        all(primary[probe_name][str(class_id)] for class_id in FOCUS_CLASSES)
        for probe_name in ("ols_gauge", "cmos_raw_pool", "cmos_evaluator_view")
    )
    focus_image_stats = image_stats["by_class"]
    presentation_pattern = all(
        float(focus_image_stats[str(class_id)]["combined"]["evaluator_view_foreground_mse"])
        > float(focus_image_stats[str(class_id)]["combined"]["evaluator_view_background_mse"])
        and float(focus_image_stats[str(class_id)]["combined"]["raw_fraction_above_one"]) <= 0.01
        for class_id in FOCUS_CLASSES
    )
    ed_unsupported_both = all(not primary["e_d"][str(class_id)] for class_id in FOCUS_CLASSES)
    if both_views_supported and ed_unsupported_both and presentation_pattern:
        return {
            "classification": "STANDARD_DIGIT_PRESENTATION_CANDIDATE",
            "training_recommendation": "仅提出并待检验：固定 Encoder/OLS，做 Generator-only 对照，将重建目标设为前景、背景各0.5权重。",
            "reason": "两类信息依次通过去相位 OLS、未截断 CMOS 和评价器视图探针；E_D 未支持，且两类评价器视图前景误差高于背景、raw 软件截断不超过1%。",
            "stop": True,
        }

    raw_supported = all(primary["cmos_raw_pool"][str(class_id)] for class_id in FOCUS_CLASSES)
    eval_not_supported = all(not primary["cmos_evaluator_view"][str(class_id)] for class_id in FOCUS_CLASSES)
    clipping_observed = any(
        float(focus_image_stats[str(class_id)]["combined"]["raw_fraction_above_one"]) > 0.0
        for class_id in FOCUS_CLASSES
    )
    if raw_supported and eval_not_supported and clipping_observed:
        return {
            "classification": "VIEW_CLIPPING_CANDIDATE",
            "training_recommendation": "NONE",
            "reason": "未截断 CMOS 探针获得支持而评价器视图探针未支持，并观察到 raw 强度超过软件视图上限；先做一次尺度/软件截断定位，不改变校准或评价器。",
            "stop": True,
        }
    return {
        "classification": "MIXED_OR_INSUFFICIENT",
        "training_recommendation": "NONE",
        "reason": "预声明的保守证据条件没有共同指向同一段或同一干预方向；只报告各段测量和边界结果。",
        "stop": True,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "step1200",
        "step2200",
        "readout1200",
        "readout2200",
        "training_data",
        "e_s",
        "e_i",
        "e_c",
        "e_d",
        "output_dir",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, default=DEFAULTS[name])
    parser.add_argument("--preflight-only", action="store_true", help="load exact named inputs and run the no-update CUDA preflight")
    parser.add_argument("--resume", action="store_true", help="resume this explicit attempt directory after verifying its stored population and READY preflight")
    return parser.parse_args()


def _resolve_paths(root: Path, arguments: argparse.Namespace) -> dict[str, Path]:
    names = tuple(DEFAULTS)
    return {name: _resolve(root, getattr(arguments, name)) for name in names}


def _prepare_target(root: Path, target: Path, *, resume: bool) -> None:
    outputs = (root / "outputs").resolve()
    _require(target != outputs and outputs in target.parents, "output directory must be a named child of repository outputs/")
    if target.exists():
        _require(target.is_dir(), "output target exists and is not a directory")
        _require(resume, f"output directory already exists; use --resume only for this explicit attempt: {target}")
    else:
        _require(not resume, f"--resume target does not exist: {target}")
        target.mkdir(parents=True, exist_ok=False)


def _input_population_identity(inputs: Mapping[str, Any], paths: Mapping[str, Path]) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "input_paths": {name: paths[name].as_posix() for name in sorted(paths)},
        "source_row_indices": inputs["source_indices"].tolist(),
        "labels": inputs["labels"].tolist(),
        "fit_rows": FIT_ROWS,
        "observation_rows": OBSERVATION_ROWS,
        "class_order": list(range(10)),
        "focus_classes": list(FOCUS_CLASSES),
        "control_classes": list(CONTROL_CLASSES),
    }


def _validate_resume(target: Path, expected_population: Mapping[str, Any]) -> dict[str, Any]:
    population_path = target / "input_population.json"
    preflight_path = target / "preflight.json"
    _require(population_path.is_file(), "resume requires the saved input population")
    stored_population = json.loads(population_path.read_text(encoding="utf-8"))
    _require(stored_population == dict(expected_population), "resume input paths, rows, labels, or split contract differ")
    report_path = target / "report.json"
    if report_path.is_file():
        report = _mapping(json.loads(report_path.read_text(encoding="utf-8")), "completed diagnostic report")
        _require(report.get("task_id") == TASK_ID and report.get("status") == "VALID_COMPLETED", "existing report is not a completed diagnostic")
        return {"complete": True, "report": dict(report), "preflight": None, "preflight_artifact": None}
    preflight = None
    preflight_artifact = None
    if preflight_path.is_file():
        stored_preflight = _mapping(json.loads(preflight_path.read_text(encoding="utf-8")), "stored preflight")
        if stored_preflight.get("task_id") == TASK_ID and stored_preflight.get("status") == "READY":
            preflight = dict(stored_preflight)
            preflight_artifact = preflight_path.name
        else:
            preflight_artifact = _next_artifact_path(target, "preflight").name
    else:
        preflight_artifact = _next_artifact_path(target, "preflight").name
    return {"complete": False, "report": None, "preflight": preflight, "preflight_artifact": preflight_artifact}


def _stage_probe_view(probe_payload: Mapping[str, Any]) -> dict[str, Any]:
    predictions = _mapping(probe_payload["predictions"], "probe predictions")
    return {
        "fit_rows": probe_payload["fit_rows"],
        "observation_rows": probe_payload["observation_rows"],
        "records": probe_payload["records"],
        "predictions": {
            name: value.detach().cpu().tolist() if torch.is_tensor(value) else list(value)
            for name, value in predictions.items()
        },
    }


def execute(arguments: argparse.Namespace | None = None) -> dict[str, Any]:
    arguments = arguments or _arguments()
    root = _root()
    paths = _resolve_paths(root, arguments)
    target = paths["output_dir"]
    _prepare_target(root, target, resume=bool(arguments.resume))
    started = time.perf_counter()
    try:
        inputs = load_inputs(root, paths)
        population = _input_population_identity(inputs, paths)
        population_path = target / "input_population.json"
        if arguments.resume:
            resume_state = _validate_resume(target, population)
            if resume_state["complete"]:
                return resume_state["report"]
        else:
            _write_json_new(population_path, population, "fixed TRAIN row population")

        preflight_artifact = "preflight.json"
        preflight_path = target / preflight_artifact
        if arguments.resume:
            preflight = resume_state["preflight"]
            preflight_artifact = resume_state["preflight_artifact"] or "preflight.json"
            preflight_path = target / preflight_artifact
            if preflight is None:
                try:
                    preflight = _preflight(inputs, paths, target)
                except (DiagnosticError, bogt.TrainingContractError, RuntimeError, OSError, torch.cuda.OutOfMemoryError) as error:
                    failure = {
                        "schema": PREFLIGHT_SCHEMA,
                        "task_id": TASK_ID,
                        "outer_gate": OUTER_GATE,
                        "status": "INVALID_HARNESS",
                        "classification": None,
                        "failure_type": type(error).__name__,
                        "failure": str(error),
                        "scientific_measurement_started": False,
                        "elapsed_seconds_observation_only": time.perf_counter() - started,
                    }
                    _write_json_new(_next_artifact_path(target, "preflight_failure"), failure, "invalid diagnostic preflight")
                    raise
                preflight["duration_seconds_observation_only"] = time.perf_counter() - started
                _write_json_new(preflight_path, preflight, "read-only CUDA preflight")
        else:
            try:
                preflight = _preflight(inputs, paths, target)
            except (DiagnosticError, bogt.TrainingContractError, RuntimeError, OSError, torch.cuda.OutOfMemoryError) as error:
                failure = {
                    "schema": PREFLIGHT_SCHEMA,
                    "task_id": TASK_ID,
                    "outer_gate": OUTER_GATE,
                    "status": "INVALID_HARNESS",
                    "classification": None,
                    "failure_type": type(error).__name__,
                    "failure": str(error),
                    "scientific_measurement_started": False,
                    "elapsed_seconds_observation_only": time.perf_counter() - started,
                }
                _write_json_new(_next_artifact_path(target, "preflight_failure"), failure, "invalid diagnostic preflight")
                raise
            preflight["duration_seconds_observation_only"] = time.perf_counter() - started
            _write_json_new(preflight_path, preflight, "read-only CUDA preflight")
        _require(preflight.get("status") == "READY", "preflight did not return READY")
        if arguments.preflight_only:
            return {
                "task_id": TASK_ID,
                "status": "READY",
                "output_dir": target.as_posix(),
                "preflight": preflight,
                "next_action": "resume this explicit attempt with --resume to perform the fixed-row diagnostic",
            }

        measurement_stage_names = [
            *(f"{kind}_{endpoint}.pt" for endpoint, _step in ENDPOINTS for kind in ("readouts", "image_metrics", "probes", "directions")),
            "target_e_d_positive_control.pt",
            "global_inference_tests.json",
            "class17_mapping_montage.png",
        ]
        reused_completed_measurement_stages = bool(arguments.resume) and all(
            (target / name).is_file() for name in measurement_stage_names
        )
        device = torch.device("cuda:0")
        evaluators = load_evaluators(paths, device)
        evaluator_states = {name: _state_snapshot(model) for name, model in evaluators.items()}
        endpoints: dict[str, Any] = {}
        probes_by_step: dict[str, Any] = {}
        directions_by_step: dict[str, Any] = {}
        readouts_by_step: dict[str, Any] = {}
        image_statistics_by_step: dict[str, Any] = {}
        per_row_image_by_step: dict[str, Any] = {}
        peak_allocated = 0
        peak_reserved = 0
        for endpoint_name, checkpoint_step in ENDPOINTS:
            _check_gpu_budget(device)
            consumer, optimizer, parameters, references, weights = _new_endpoint(
                endpoint_name,
                inputs["payloads"][endpoint_name],
                inputs["references"],
                inputs["weights"],
                device,
            )
            warm_batch = inputs["images"][:LOGICAL_BATCH].to(device)
            with torch.no_grad():
                warm_result = consumer.runtime.forward_from_image(warm_batch)
                _ = canonical.cmos_view(warm_result.raw_intensity)
            del warm_batch, warm_result
            torch.cuda.synchronize(device)
            runtime_snapshot = _state_snapshot(consumer.runtime)
            optimizer_snapshot = _optimizer_snapshot(optimizer)
            references_snapshot = consumer.references.to_payload()
            weights_snapshot = consumer.weights.to_payload()
            readouts = _fresh_readout_stage(target, endpoint_name, consumer, evaluators, inputs, references)
            sample_resource_telemetry(
                target / "resource_telemetry.jsonl",
                device,
                f"{endpoint_name}_fresh_forward",
                progress={"rows": int(readouts["z"].shape[0]), "checkpoint_step": checkpoint_step},
                started=started,
            )
            image_statistics, per_row_image = _aggregate_image_statistics(
                readouts,
                inputs["labels"],
                inputs["source_indices"],
            )
            image_metric_artifact = _image_metric_stage(
                target,
                endpoint_name,
                per_row_image,
                inputs,
            )
            probe_payload = _probe_stage(target, endpoint_name, readouts, inputs, references)
            sample_resource_telemetry(
                target / "resource_telemetry.jsonl",
                device,
                f"{endpoint_name}_probe_fit",
                progress={"fit_rows": FIT_ROWS, "observation_rows": OBSERVATION_ROWS},
                started=started,
            )
            direction_payload = _direction_stage(
                target,
                endpoint_name,
                consumer,
                optimizer,
                parameters,
                references,
                evaluators,
                inputs,
                runtime_snapshot,
                optimizer_snapshot,
                target / "resource_telemetry.jsonl",
                started,
            )
            sample_resource_telemetry(
                target / "resource_telemetry.jsonl",
                device,
                f"{endpoint_name}_direction_complete",
                progress={"balanced_batches": FIT_ROWS // LOGICAL_BATCH, "individual_rows": 100},
                started=started,
            )
            _require(_state_equal(runtime_snapshot, _state_snapshot(consumer.runtime)), f"{endpoint_name} optical state changed over its diagnostic stage")
            _assert_tree_equal(optimizer_snapshot, _optimizer_snapshot(optimizer), f"{endpoint_name} source Adam state after diagnostic stage")
            _assert_tree_equal(references_snapshot, consumer.references.to_payload(), f"{endpoint_name} references after diagnostic stage")
            _assert_tree_equal(weights_snapshot, consumer.weights.to_payload(), f"{endpoint_name} BOGT weights after diagnostic stage")
            _require(all(parameter.grad is None for parameter in parameters.values()), f"{endpoint_name} optical .grad buffers are not empty")
            for evaluator_name, model in evaluators.items():
                _require(_state_equal(evaluator_states[evaluator_name], _state_snapshot(model)), f"{evaluator_name} evaluator changed at {endpoint_name}")
                _require(all(parameter.grad is None for parameter in model.parameters()), f"{evaluator_name} retained a gradient at {endpoint_name}")
            readouts["source_indices"] = inputs["source_indices"]
            readouts["labels"] = inputs["labels"]
            readouts_by_step[endpoint_name] = readouts
            probe_view = _stage_probe_view(probe_payload)
            probes_by_step[endpoint_name] = probe_payload
            directions_by_step[endpoint_name] = direction_payload
            image_statistics_by_step[endpoint_name] = image_statistics
            per_row_image_by_step[endpoint_name] = per_row_image
            endpoints[endpoint_name] = {
                "checkpoint_step": checkpoint_step,
                "model_identity": _runtime_description(consumer),
                "optimizer_configuration": dict(EXPECTED_ADAM),
                "optimizer_internal_steps": [checkpoint_step, checkpoint_step],
                "weights": {
                    "alpha": float(weights.alpha),
                    "beta": float(weights.beta),
                    "gamma": float(weights.gamma),
                    "delta_SIC": DELTA_SIC,
                    "kappa": KAPPA,
                    "step2200_shared_values_inherited_from_step1200": endpoint_name == "step2200",
                },
                "fresh_forward_rows": int(readouts["z"].shape[0]),
                "fresh_forward_reconstruction_and_raw_metrics": image_statistics,
                "probe_results": probe_view,
                "direction_summaries": direction_payload["per_class_direction_summary"],
                "gradient_artifact": f"directions_{endpoint_name}.pt",
                "readout_artifact": f"readouts_{endpoint_name}.pt",
                "probe_artifact": f"probes_{endpoint_name}.pt",
                "per_row_image_metrics_artifact": image_metric_artifact,
            }
            peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated(device)))
            peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved(device)))
            del consumer, optimizer, parameters, references, weights, readouts, probe_payload, direction_payload
            gc.collect()
            torch.cuda.empty_cache()

        montage_path = target / "class17_mapping_montage.png"
        if not montage_path.exists():
            montage_info = _save_montage(montage_path, inputs["images"], inputs["labels"], readouts_by_step)
        else:
            montage_info = {
                "path": montage_path.name,
                "rows_displayed": 150,
                "class_order": list(DISPLAY_CLASSES),
                "panel_order": ["target", "step1200", "step2200"],
                "view": "28x28 canonical clamp[0,1] then adaptive average pool",
                "fixed_display_range": [0.0, 1.0],
                "per_image_rescaling": False,
                "resumed_existing_artifact": True,
            }

        global_tests = _global_probe_tests(probes_by_step, inputs["labels"])
        support_by_step = {
            name: _support_for_endpoint(name, probes_by_step[name], global_tests)
            for name, _step in ENDPOINTS
        }
        _target_logits, target_e_d_metrics, target_e_d_artifact = _target_e_d_stage(
            target,
            evaluators["E_D"],
            inputs["images"],
            inputs["labels"],
            inputs["source_indices"],
            device,
        )
        qualification = _qualification(probes_by_step, target_e_d_metrics)
        final_image_stats = image_statistics_by_step["step2200"]
        final_direction = directions_by_step["step2200"]
        decision = _final_decision(
            probes_by_step,
            support_by_step,
            global_tests,
            qualification,
            directions_by_step,
            final_image_stats,
        )
        recovery_artifact = None
        if reused_completed_measurement_stages:
            recovery_artifact = "finalization_recovery.json"
            recovery_record = {
                "task_id": TASK_ID,
                "status": "FINALIZATION_RECOVERED_FROM_SAVED_EVIDENCE",
                "reason": "The initial report serialization stopped after all diagnostic stages were persisted; report finalization now reuses those stage artifacts.",
                "saved_measurement_stages_reused": measurement_stage_names,
                "readout_rows_revalidated": 300,
                "fit_rows_revalidated": FIT_ROWS,
                "observation_rows_revalidated": OBSERVATION_ROWS,
                "optical_parameter_updates_during_recovery": 0,
                "ols_coefficient_updates_during_recovery": 0,
            }
            recovery_path = target / recovery_artifact
            if recovery_path.is_file():
                saved_recovery = _mapping(json.loads(recovery_path.read_text(encoding="utf-8")), "saved finalization recovery record")
                _require(saved_recovery == recovery_record, "saved finalization recovery record differs")
            else:
                _write_json_new(recovery_path, recovery_record, "saved-evidence finalization recovery")
        for name, model in evaluators.items():
            _require(_state_equal(evaluator_states[name], _state_snapshot(model)), f"{name} evaluator state changed before terminal report")
        resources = _check_gpu_budget(device)
        report = {
            "schema": REPORT_SCHEMA,
            "task_id": TASK_ID,
            "outer_gate": OUTER_GATE,
            "result": "VALID_COMPLETED",
            "status": "VALID_COMPLETED",
            "classification": decision["classification"],
            "training_recommendation": decision["training_recommendation"],
            "reason": decision["reason"],
            "authorization_scope": {
                "frozen_steps": [1200, 2200],
                "fixed_population": 300,
                "fit_rows": 200,
                "observation_rows": 100,
                "optical_parameter_updates": 0,
                "ols_coefficient_updates": 0,
                "s1_resampling": False,
                "dev_images_measured": 0,
                "protected_test_sealed_reads": 0,
                "e_d_in_autograd": False,
                "successor_training_authorized": False,
                "stop_after_report": True,
            },
            "qualification": qualification,
            "population": inputs["row_contract"],
            "population_artifact": "input_population.json",
            "source_rows": inputs["source_indices"].tolist(),
            "source_labels": inputs["labels"].tolist(),
            "evaluators": {
                "E_S": {"path": paths["e_s"].as_posix(), "role": "frozen semantic feature loss"},
                "E_I": {"path": paths["e_i"].as_posix(), "role": "frozen semantic feature loss"},
                "E_C": {"path": paths["e_c"].as_posix(), "role": "frozen semantic feature loss"},
                "E_D": {"path": paths["e_d"].as_posix(), "role": "forward-only independent confirmation"},
            },
            "endpoint_results": endpoints,
            "global_inference_tests": global_tests,
            "support_by_endpoint": support_by_step,
            "measurement_recovery": {
                "saved_measurement_stages_reused": reused_completed_measurement_stages,
                "artifact": recovery_artifact,
                "optical_parameter_updates_during_recovery": 0,
                "ols_coefficient_updates_during_recovery": 0,
            },
            "training_direction": {
                "primary_endpoint": "step2200",
                "step1200_context": directions_by_step["step1200"]["per_class_direction_summary"],
                "step2200": final_direction["per_class_direction_summary"],
                "definitions": {
                    "G_BOGT": "alpha*grad(L_dist) + beta*grad(L_rec) + gamma*grad(L_op)",
                    "G_canonical": "delta_SIC * RMSNormalize(sum_k grad(L_k)/||grad(L_k)||_2), k in E_S,E_I,E_C",
                    "G_combined": "G_BOGT + kappa*G_canonical",
                    "saved_Adam_candidate": "one read-only Adam displacement from saved moments and step; no optimizer.step",
                    "zero_history_candidate": "same gradient from empty moments at candidate step 1",
                    "projected_candidate": "Adam candidate from G_canonical plus BOGT gradient projected to not oppose G_canonical",
                    "row_comparator": "per-row equal-weight arithmetic mean of gradients of L_S,L_I,L_C",
                    "boundary": "absolute cosine < 0.05",
                    "stable_direction": "at least 16/20 non-boundary row comparisons share a sign",
                },
                "gradient_artifacts": {name: f"directions_{name}.pt" for name, _step in ENDPOINTS},
            },
            "image_presentation": {
                "step1200": image_statistics_by_step["step1200"],
                "step2200": image_statistics_by_step["step2200"],
                "montage": montage_info,
                "views": {
                    "raw": "unclipped raw CMOS intensity at 200x200",
                    "raw_probe": "adaptive average pool of raw CMOS directly to 28x28, no clamp",
                    "evaluator_view": "canonical clamp[0,1] then adaptive average pool to 28x28",
                    "foreground": "target intensity > 0.1 at the matching resolution",
                },
            },
            "coordinates_and_normalization": {
                "ols_ports": bogt.OLS_PORT_COUNT,
                "ols_interface": endpoints["step2200"]["model_identity"].get("ols_interface"),
                "raw_cmos_plane": "lightgen_raw_cmos, row-major tensor axes [-2]=row and [-1]=column, 200x200",
                "target_inputs": "canonical TRAIN pixel values converted from uint8 to float32 [0,1]",
                "ols_probe": "complex Cartesian Re/Im; gauge probe uses current max_q_gauge; no coefficient normalization update",
                "display": "all panels use the same 28x28 [0,1] intensity scale; no per-image stretch",
            },
            "preflight": {
                "artifact": preflight_artifact,
                "status": preflight["status"],
                "no_optimizer_step": True,
                "full_production_path": True,
            },
            "resources": {
                "whole_device": resources,
                "pytorch_peak_allocated_bytes": peak_allocated,
                "pytorch_peak_reserved_bytes": peak_reserved,
                "duration_seconds_observation_only": _observed_task_duration(
                    target,
                    started,
                    recovered=reused_completed_measurement_stages,
                ),
            },
            "historical_attempt_boundary": {
                "prior_attempt_status": "INVALID_HARNESS",
                "prior_attempt_updates": 0,
                "prior_measurement_readouts_used_as_science": False,
                "prior_attempt_status_modified": False,
            },
            "next_action": "NONE",
            "stop": True,
            "evidence_paths": {
                "report": "report.json",
                "preflight": preflight_artifact,
                "montage": montage_path.name,
                "global_tests": "global_inference_tests.json",
                "finalization_recovery": recovery_artifact,
                "target_e_d_positive_control": target_e_d_artifact,
                "readouts": {name: f"readouts_{name}.pt" for name, _step in ENDPOINTS},
                "per_row_image_metrics": {name: f"image_metrics_{name}.pt" for name, _step in ENDPOINTS},
                "probes": {name: f"probes_{name}.pt" for name, _step in ENDPOINTS},
                "gradients": {name: f"directions_{name}.pt" for name, _step in ENDPOINTS},
                "resource_telemetry": "resource_telemetry.jsonl",
            },
        }
        global_tests_path = target / "global_inference_tests.json"
        if global_tests_path.is_file():
            saved_global_tests = _mapping(json.loads(global_tests_path.read_text(encoding="utf-8")), "saved global inference tests")
            _require(saved_global_tests == global_tests, "saved global inference tests differ from the frozen probe predictions")
        else:
            _write_json_new(global_tests_path, global_tests, "20-test Holm family")
        _write_json_new(target / "report.json", report, "completed fixed-row diagnostic")
        return report
    except (DiagnosticError, bogt.TrainingContractError, RuntimeError, TypeError, OSError, torch.cuda.OutOfMemoryError) as error:
        failure_path = _next_artifact_path(target, "invalid_harness")
        if not (target / "report.json").exists():
            failure = {
                "task_id": TASK_ID,
                "outer_gate": OUTER_GATE,
                "status": "INVALID_HARNESS",
                "classification": None,
                "failure_type": type(error).__name__,
                "failure": str(error),
                "scientific_result": None,
                "elapsed_seconds_observation_only": time.perf_counter() - started,
            }
            _write_json_new(failure_path, failure, "invalid diagnostic harness closeout")
        raise


def main() -> int:
    arguments = _arguments()
    try:
        result = execute(arguments)
    except (DiagnosticError, bogt.TrainingContractError, RuntimeError, TypeError, OSError, torch.cuda.OutOfMemoryError) as error:
        print(json.dumps({
            "task_id": TASK_ID,
            "status": "INVALID_HARNESS",
            "classification": None,
            "failure": str(error),
        }, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({
        "task_id": result.get("task_id"),
        "status": result.get("status"),
        "classification": result.get("classification"),
        "training_recommendation": result.get("training_recommendation"),
        "output_dir": result.get("output_dir", str(_resolve(_root(), arguments.output_dir))),
        "next_action": result.get("next_action"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
